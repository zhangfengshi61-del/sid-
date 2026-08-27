"""L8 RESCORE-DEPTH on an ETEGRec model (the long-code analogue of tiger_rescore_depth.py).
PIVOTAL for ask#4 (training-accel viability): on LONG codes, do the DEEP capacity tokens (5..L)
help HEAD ranking, or is head saturated by ~3-4 tokens?
  - if HEAD R@10 saturates early (deep tokens don't help head) -> deep-token training is droppable
    for the dominant bucket -> training-accel for long codes is viable.
  - if HEAD keeps rising to k=L -> deep tokens are load-bearing -> training-accel via our lens is NOT clean.
Also reports MID/TAIL (do deep tokens demote tail, as on L4?).

Method: for each test user, take teacher full-beam top-N candidate items, teacher-force score them
(perpos_logprob), rerank by first-k cumulative logp for k=1..L, map to items, per popularity bucket.
Interface mirrors code/eteg_certified.py. RUN ON THE CLEAN RK-Means d8 ckpt (forced-util, 256/level),
NOT the collapsed RQVAE d8.  UNTESTED — debug at first run (check m.generate n_return_sequences>=beam).
"""
import os, sys, json, yaml, argparse, math
from collections import defaultdict
import numpy as np, torch
sys.path.insert(0, './repos/ETEGRec')
os.chdir('./repos/ETEGRec')
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from model import Model
from utils import convert_config_dict, safe_load
from data import load_split_data, SequentialSplitDataset, Collator
from torch.utils.data import DataLoader

def build_model(config, num_items):
    mc = T5Config(num_layers=config['encoder_layers'], num_decoder_layers=config['decoder_layers'],
        d_model=config['d_model'], d_ff=config['d_ff'], num_heads=config['num_heads'], d_kv=config['d_kv'],
        dropout_rate=config['dropout_rate'], activation_function=config['activation_function'],
        vocab_size=1, pad_token_id=0, eos_token_id=300, decoder_start_token_id=0,
        feed_forward_proj=config['feed_forward_proj'], n_positions=config['max_length'])
    t5 = T5ForConditionalGeneration(config=mc)
    return Model(config=config, model=t5, n_items=num_items, code_length=config['code_length'], code_number=config['code_num'])

@torch.no_grad()
def perpos_logprob(m, enc_last, attn_row, cand_codes):
    C = cand_codes.size(0)
    enc_rep = BaseModelOutput(last_hidden_state=enc_last.expand(C, -1, -1).contiguous())
    attn_rep = attn_row.expand(C, -1).contiguous()
    out = m(encoder_outputs=enc_rep, attention_mask=attn_rep, labels=cand_codes)
    logp = torch.log_softmax(out.logits, dim=-1)
    return logp.gather(-1, cand_codes.unsqueeze(-1)).squeeze(-1)  # (C,L)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cfg', default=None)
    ap.add_argument('--beam', type=int, default=50)
    ap.add_argument('--topn', type=int, default=50)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_batches', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = torch.device(a.device)
    cfg_name = a.cfg or a.dataset
    config = convert_config_dict(yaml.safe_load(open(f'./config/{cfg_name}.yaml')))
    config['device'] = dev; config['num_beams'] = a.beam
    item2id, num_items, train, valid, test = load_split_data(config)
    m = build_model(config, num_items).to(dev); m.device = dev
    safe_load(m, a.ckpt, verbose=True); m.eval()
    code = json.load(open(a.ckpt[:-3] + '.code.json'))
    N, L = len(code) - 1, config['code_length']
    all_code = torch.tensor(code, dtype=torch.long, device=dev)
    code2item = {tuple(code[i]): i for i in range(1, N + 1)}
    # popularity = item frequency across train sequences
    pop = defaultdict(int)
    for seq in train:
        for it in seq: pop[int(it)] += 1
    cs = sorted(pop.values())
    def bucket(it):
        c = pop.get(it, 0)
        if c == 0: return 'cold'
        n = len(cs); lo = cs[n//3]; hi = cs[2*n//3]
        return 'tail' if c <= lo else ('mid' if c <= hi else 'head')
    KS = list(range(1, L + 1)); BKTS = ['all','head','mid','tail','cold']
    rec = {b:{k:0 for k in KS} for b in BKTS}; ndcg = {b:{k:0.0 for k in KS} for b in BKTS}; cnt = {b:0 for b in BKTS}

    test2 = test[::a.stride]
    test_ds = SequentialSplitDataset(config=config, n_items=num_items, inter_seq=test2)
    collator = Collator(eos_token_id=-1, pad_token_id=0, max_length=config['max_length'])
    dl = DataLoader(test_ds, batch_size=a.bs, shuffle=False, collate_fn=collator)
    print(f'N={N} L={L} test={len(test2)}', flush=True)
    nb = 0
    for batch in dl:
        if a.max_batches and nb >= a.max_batches: break
        nb += 1
        input_ids0 = batch['input_ids'].to(dev); targets = batch['targets'].to(dev)
        B = input_ids0.size(0)
        input_ids = all_code[input_ids0].reshape(B, -1); attn = (input_ids != -1).bool()
        preds = m.generate(input_ids=input_ids.clone(), attention_mask=attn, n_return_sequences=a.topn)  # (B,topn,L)
        emb = m.get_input_embeddings(input_ids.clone(), attn)
        enc_all = m.get_encoder()(inputs_embeds=emb, attention_mask=attn, return_dict=True).last_hidden_state
        predl = preds.tolist(); tgt = targets.tolist()
        for b in range(B):
            gt = int(tgt[b]) if not isinstance(tgt[b], list) else int(tgt[b][0])
            bkt = bucket(gt)
            for t in ('all',bkt): cnt[t]+=1
            cand_codes = preds[b]  # (topn,L)
            lp = perpos_logprob(m, enc_all[b:b+1], attn[b:b+1], cand_codes)  # (topn,L)
            cum = lp.cumsum(dim=1)
            items = [code2item.get(tuple(predl[b][j]), -1) for j in range(a.topn)]
            for k in KS:
                order = torch.argsort(cum[:,k-1], descending=True).tolist()
                seen=set(); ranked=[]
                for j in order:
                    it=items[j]
                    if it==-1 or it in seen: continue
                    seen.add(it); ranked.append(it)
                try: rk=ranked.index(gt)
                except ValueError: rk=-1
                if rk!=-1 and rk<10:
                    for t in ('all',bkt): rec[t][k]+=1; ndcg[t][k]+=1.0/math.log2(rk+2)
        if nb%10==0: print(f'batch {nb}', flush=True)
    out={'dataset':a.dataset,'cfg':cfg_name,'L':L,'topn':a.topn,'rec':{},'ndcg':{},'n':{}}
    for b in BKTS:
        n=max(cnt[b],1); out['n'][b]=cnt[b]
        out['rec'][b]={k:rec[b][k]/n for k in KS}; out['ndcg'][b]={k:ndcg[b][k]/n for k in KS}
    json.dump(out, open(a.out,'w'), indent=2)
    print(f'=== L{L} RESCORE-DEPTH {a.dataset} (R@10 by #tokens k=1..{L}) ===')
    for b in BKTS:
        n=cnt[b]; row=' '.join(f'k{k}={rec[b][k]/max(n,1):.4f}' for k in KS)
        print(f'[{b:4s} n={n:6d}] {row}')
    print('OUT',a.out,flush=True); print('ERDDONE',flush=True)

if __name__ == '__main__':
    main()
