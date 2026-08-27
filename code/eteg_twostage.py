"""Two-stage accelerator on a trained ETEGRec model:
   stage1 = beam-decode shallow prefix H ; stage2 = rescore candidates' FULL code by same-model likelihood.
   Reports recall@5/10 + ndcg@5/10 for full-beam vs two-stage, exact-top10-match, pool size, timing.
"""
import os, sys, json, yaml, argparse, time, math
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
    return Model(config=config, model=t5, n_items=num_items,
                 code_length=config['code_length'], code_number=config['code_num'])


@torch.no_grad()
def full_code_logprob(m, enc_last, attn_row, cand_codes):
    """enc_last (1,S,d), attn_row (1,S), cand_codes (C,L) -> (C,) summed log-prob of each full code."""
    C = cand_codes.size(0)
    enc_rep = BaseModelOutput(last_hidden_state=enc_last.expand(C, -1, -1).contiguous())
    attn_rep = attn_row.expand(C, -1).contiguous()
    out = m(encoder_outputs=enc_rep, attention_mask=attn_rep, labels=cand_codes)
    logp = torch.log_softmax(out.logits, dim=-1)               # (C,L,cn)
    lp = logp.gather(-1, cand_codes.unsqueeze(-1)).squeeze(-1)  # (C,L)
    return lp.sum(-1)


def rank_of(tgt, ordered_list):
    """0-based rank of tgt tuple in ordered_list of tuples, or None."""
    for r, c in enumerate(ordered_list):
        if c == tgt:
            return r
    return None


class Acc:
    def __init__(self): self.r5 = self.r10 = self.n5 = self.n10 = 0.0
    def add(self, rank):
        if rank is None: return
        if rank < 5:  self.r5 += 1; self.n5 += 1.0 / math.log2(rank + 2)
        if rank < 10: self.r10 += 1; self.n10 += 1.0 / math.log2(rank + 2)
    def line(self, nq):
        return f"recall@5={self.r5/nq:.4f} recall@10={self.r10/nq:.4f} ndcg@5={self.n5/nq:.4f} ndcg@10={self.n10/nq:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cfg', default=None, help='config name if != dataset (e.g. scientific_d8)')
    ap.add_argument('--Hs', default='1,2,3')
    ap.add_argument('--beam', type=int, default=20)
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--max_batches', type=int, default=0)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    dev = torch.device(args.device)
    cfg_name = args.cfg or args.dataset
    config = convert_config_dict(yaml.safe_load(open(f'./config/{cfg_name}.yaml')))
    config['device'] = dev
    config['num_beams'] = args.beam

    item2id, num_items, train, valid, test = load_split_data(config)
    m = build_model(config, num_items).to(dev); m.device = dev
    safe_load(m, args.ckpt, verbose=True); m.eval()

    code = json.load(open(args.ckpt[:-3] + '.code.json'))     # [N+1][L]
    all_code = torch.tensor(code, dtype=torch.long, device=dev)
    N, L = len(code) - 1, config['code_length']
    Hs = [int(x) for x in args.Hs.split(',')]

    prefix2items = {}
    for H in Hs:
        d = {}
        for i in range(1, N + 1):
            d.setdefault(tuple(code[i][:H]), []).append(i)
        prefix2items[H] = d

    test = test[::args.stride]
    test_ds = SequentialSplitDataset(config=config, n_items=num_items, inter_seq=test)
    collator = Collator(eos_token_id=-1, pad_token_id=0, max_length=config['max_length'])
    dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False, collate_fn=collator)

    acc_full = Acc()
    acc_ts = {H: Acc() for H in Hs}
    stat = {H: dict(pool=0, exact=0, overlap=0) for H in Hs}
    nq = 0; nb = 0
    t_full = 0.0; t_stage = {H: 0.0 for H in Hs}

    for batch in dl:
        nb += 1
        if args.max_batches and nb > args.max_batches:
            break
        input_ids0 = batch['input_ids'].to(dev)
        targets = batch['targets'].to(dev)
        B = input_ids0.size(0)
        input_ids = all_code[input_ids0].reshape(B, -1)
        attn = (input_ids != -1).bool()
        tgt_code = all_code[targets].reshape(B, L)

        torch.cuda.synchronize(); t0 = time.time()
        preds_full = m.generate(input_ids=input_ids.clone(), attention_mask=attn, n_return_sequences=10)  # (B,10,L) ordered
        torch.cuda.synchronize(); t_full += time.time() - t0
        full_lists = [[tuple(c) for c in preds_full[b].tolist()] for b in range(B)]
        full_sets = [set(fl) for fl in full_lists]
        for b in range(B):
            acc_full.add(rank_of(tuple(tgt_code[b].tolist()), full_lists[b]))

        emb = m.get_input_embeddings(input_ids.clone(), attn)
        enc_all = m.get_encoder()(inputs_embeds=emb, attention_mask=attn, return_dict=True).last_hidden_state

        for H in Hs:
            torch.cuda.synchronize(); t0 = time.time()
            beams = m.my_beam_search(input_ids=input_ids.clone(), attention_mask=attn,
                                     max_length=H + 1, num_beams=args.beam, num_return_sequences=args.beam)
            beams = beams[:, 1:].reshape(B, args.beam, H)
            for b in range(B):
                prefs = set(tuple(p) for p in beams[b].tolist())
                cands = set()
                for p in prefs:
                    cands.update(prefix2items[H].get(p, []))
                cands = list(cands)
                stat[H]['pool'] += len(cands)
                tgt = tuple(tgt_code[b].tolist())
                if not cands:
                    continue
                cand_codes = all_code[torch.tensor(cands, device=dev)]
                lp = full_code_logprob(m, enc_all[b:b + 1], attn[b:b + 1], cand_codes)
                order = torch.argsort(lp, descending=True)[:10].tolist()
                ts_list = [tuple(cand_codes[j].tolist()) for j in order]
                ts_set = set(ts_list)
                if ts_set == full_sets[b]:
                    stat[H]['exact'] += 1
                stat[H]['overlap'] += len(ts_set & full_sets[b])
                acc_ts[H].add(rank_of(tgt, ts_list))
            torch.cuda.synchronize(); t_stage[H] += time.time() - t0
        nq += B

    print(f"\n=== dataset={args.dataset} cfg={cfg_name} ckpt={os.path.basename(args.ckpt)} N={N} L={L} beam={args.beam} queries={nq} ===")
    print(f"FULL-beam : {acc_full.line(nq)}   |  {t_full*1000/nb:.1f} ms/batch")
    for H in Hs:
        s = stat[H]
        print(f"TWO-STAGE H={H}: {acc_ts[H].line(nq)}  | pool={s['pool']/nq:.1f} "
              f"exact_top10={s['exact']/nq*100:.1f}% overlap={s['overlap']/nq:.2f}/10 "
              f"| {t_stage[H]*1000/nb:.1f} ms/batch(unvec)")


if __name__ == '__main__':
    main()
