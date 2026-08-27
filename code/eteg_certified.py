"""Certified parallel continuation: prove two-stage == full beam BY CONSTRUCTION.

Stage1: constrained beam-decode to horizon H (top-B H-prefixes) -- same first H steps as full beam.
Stage2: ONE parallel teacher-forced pass over ALL catalog items under those H-prefixes -> per-position
        logp; then EMULATE the remaining beam steps H+1..L with top-B pruning using those logp,
        WITH NO EXTRA MODEL CALLS. Output is identical to full beam by induction.

Reports, vs full-beam (m.generate) reference, for each H:
  certified : ordered-top10 identity + set identity  (expect ~100% by construction; fixed-length => no length-penalty issue)
  rerank    : the old "rank all candidates by summed full-code logprob" (a SUPERSET of beam survivors)
  rerank_vs_certified mismatches = the hidden-weakness probe (where exhaustive rerank diverges from beam)
"""
import os, sys, json, yaml, argparse, time, math
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
    return Model(config=config, model=t5, n_items=num_items,
                 code_length=config['code_length'], code_number=config['code_num'])


@torch.no_grad()
def perpos_logprob(m, enc_last, attn_row, cand_codes):
    """(C,L) per-position logp of each candidate's true token under the frozen model."""
    C = cand_codes.size(0)
    enc_rep = BaseModelOutput(last_hidden_state=enc_last.expand(C, -1, -1).contiguous())
    attn_rep = attn_row.expand(C, -1).contiguous()
    out = m(encoder_outputs=enc_rep, attention_mask=attn_rep, labels=cand_codes)
    logp = torch.log_softmax(out.logits, dim=-1)
    return logp.gather(-1, cand_codes.unsqueeze(-1)).squeeze(-1)  # (C,L)


def certified_continuation(codes, cum, H, beam, K):
    """codes: list of C code-tuples (len L). cum: (C,L) cumulative logp. Emulate beam H..L w/ top-B
       pruning. Returns ordered list of top-K code-tuples == full beam by construction."""
    L = len(codes[0])
    groups = defaultdict(list)
    for c in range(len(codes)):
        groups[codes[c][:H]].append(c)
    beamset = [(pref, cum[idxs[0], H - 1].item(), idxs) for pref, idxs in groups.items()]
    beamset.sort(key=lambda x: -x[1]); beamset = beamset[:beam]
    for t in range(H, L):
        nxt = []
        for pref, score, idxs in beamset:
            sub = defaultdict(list)
            for c in idxs:
                sub[codes[c][t]].append(c)
            for tok, sidxs in sub.items():
                nxt.append((pref + (tok,), cum[sidxs[0], t].item(), sidxs))
        nxt.sort(key=lambda x: -x[1]); beamset = nxt[:beam]
    return [pref for pref, score, idxs in beamset[:K]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cfg', default=None)
    ap.add_argument('--Hs', default='2,3')
    ap.add_argument('--beam', type=int, default=20)
    ap.add_argument('--stride', type=int, default=20)
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

    code = json.load(open(args.ckpt[:-3] + '.code.json'))
    N, L = len(code) - 1, config['code_length']
    Hs = [int(x) for x in args.Hs.split(',')]
    all_code = torch.tensor(code, dtype=torch.long, device=dev)

    prefix2items = {}
    for H in Hs:
        d = defaultdict(list)
        for i in range(1, N + 1):
            d[tuple(code[i][:H])].append(i)
        prefix2items[H] = d

    test = test[::args.stride]
    test_ds = SequentialSplitDataset(config=config, n_items=num_items, inter_seq=test)
    collator = Collator(eos_token_id=-1, pad_token_id=0, max_length=config['max_length'])
    dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False, collate_fn=collator)

    nq = 0
    stat = {H: dict(cert_set=0, cert_ord=0, rerank_set=0, rr_vs_cert=0, pool=0) for H in Hs}

    for batch in dl:
        input_ids0 = batch['input_ids'].to(dev)
        targets = batch['targets'].to(dev)
        B = input_ids0.size(0)
        input_ids = all_code[input_ids0].reshape(B, -1)
        attn = (input_ids != -1).bool()

        preds_full = m.generate(input_ids=input_ids.clone(), attention_mask=attn, n_return_sequences=10)
        full_lists = [[tuple(c) for c in preds_full[b].tolist()] for b in range(B)]
        full_sets = [set(fl) for fl in full_lists]

        emb = m.get_input_embeddings(input_ids.clone(), attn)
        enc_all = m.get_encoder()(inputs_embeds=emb, attention_mask=attn, return_dict=True).last_hidden_state

        for H in Hs:
            beams = m.my_beam_search(input_ids=input_ids.clone(), attention_mask=attn,
                                     max_length=H + 1, num_beams=args.beam, num_return_sequences=args.beam)
            beams = beams[:, 1:].reshape(B, args.beam, H)
            for b in range(B):
                prefs = set(tuple(p) for p in beams[b].tolist())
                cands = []
                for p in prefs:
                    cands.extend(prefix2items[H].get(p, []))
                cands = list(set(cands))
                stat[H]['pool'] += len(cands)
                if not cands:
                    continue
                cand_codes = all_code[torch.tensor(cands, device=dev)]
                lp = perpos_logprob(m, enc_all[b:b + 1], attn[b:b + 1], cand_codes)   # (C,L)
                cum = lp.cumsum(dim=1)
                code_tuples = [tuple(cc) for cc in cand_codes.tolist()]
                # certified continuation (emulate beam, == full beam by construction)
                cert = certified_continuation(code_tuples, cum, H, args.beam, 10)
                cert_set = set(cert)
                # superset rerank (old): rank ALL cands by summed logprob
                tot = cum[:, -1]
                order = torch.argsort(tot, descending=True)[:10].tolist()
                rerank = [code_tuples[j] for j in order]
                rerank_set = set(rerank)
                # agreement vs full beam
                if cert_set == full_sets[b]: stat[H]['cert_set'] += 1
                if cert == full_lists[b]:    stat[H]['cert_ord'] += 1
                if rerank_set == full_sets[b]: stat[H]['rerank_set'] += 1
                if cert_set != rerank_set:   stat[H]['rr_vs_cert'] += 1
            nq_inc = B
        nq += B
        if args.max_batches and nq // args.bs >= args.max_batches:
            break

    print(f"\n=== certified continuation: dataset={args.dataset} cfg={cfg_name} N={N} L={L} beam={args.beam} queries={nq} ===")
    for H in Hs:
        s = stat[H]
        print(f"H={H}: pool={s['pool']/nq:.1f} | "
              f"CERT vs full: set={s['cert_set']/nq*100:.3f}%  ordered={s['cert_ord']/nq*100:.3f}%  || "
              f"RERANK vs full: set={s['rerank_set']/nq*100:.3f}%  | rerank!=cert: {s['rr_vs_cert']}/{nq}")


if __name__ == '__main__':
    main()
