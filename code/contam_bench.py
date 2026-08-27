#!/usr/bin/env python
"""Batch-contamination curve (the decisive 'max not mean' test, real TIGER).
Form bs=64 decode batches with EXACTLY k deep queries + (64-k) shallow queries; time each.
If batched latency ~ batch-MAX depth: dt jumps from t(shallow) to t(deep) as soon as k>=1, then is
~FLAT in k. If latency tracked the MEAN (avg-length proxy), dt would rise linearly in k. Per-query
depth taken from saved beams.json (record order == test_data order, eval used shuffle=False)."""
import argparse, json, time
import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import *
from collator import TestCollator
from generation_trie import Trie

def main(args):
    set_seed(args.seed)
    dev = torch.device("cuda", args.gpu_id)
    tok = T5Tokenizer.from_pretrained("./ckpt/TIGER", model_max_length=512)
    train_data, _ = load_datasets(args)
    tok.add_tokens(train_data.datasets[0].get_new_tokens())
    model = T5ForConditionalGeneration.from_pretrained(args.ckpt_path, low_cpu_mem_usage=True,
                                                       device_map={"": args.gpu_id}).eval()
    test_data = load_test_dataset(args); test_data.set_prompt(0)
    coll = TestCollator(args, tok)
    trie = Trie([[0] + tok.encode(c) for c in test_data.get_all_items()])
    eos = tok.eos_token_id
    pfx = lambda bid, s: (trie.get(s.tolist()) or [eos])
    # per-query depth from beams.json (same policy, same order)
    recs = json.load(open(args.beams))
    depth = np.array([max((b.count("<") for b in r["beams"]), default=0) + 1 for r in recs])
    shallow = np.where(depth <= args.shallow)[0]
    deep = np.where(depth >= args.deep)[0]
    print(f"depths: n={len(depth)} mean={depth.mean():.2f} | shallow(<= {args.shallow})={len(shallow)} "
          f"deep(>= {args.deep})={len(deep)}", flush=True)
    rng = np.random.RandomState(0); B = 64

    def make_batch(idxs):
        b = coll([test_data[int(j)] for j in idxs])
        return b[0]["input_ids"].to(dev), b[0]["attention_mask"].to(dev)

    def decode(ids, attn):
        return model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=12,
                              prefix_allowed_tokens_fn=pfx, num_beams=args.num_beams,
                              num_return_sequences=args.num_beams, return_dict_in_generate=True,
                              early_stopping=True)

    # warmup
    with torch.no_grad():
        ids, attn = make_batch(rng.choice(shallow, B)); 
        for _ in range(3): decode(ids, attn)
        torch.cuda.synchronize(dev)
        out = {}
        for k in [0, 1, 2, 4, 8, 16, 32, 64]:
            dts, deps = [], []
            for _ in range(args.reps):
                nd = min(k, len(deep)); ns = B - nd
                idxs = np.concatenate([rng.choice(deep, nd, replace=True) if nd else np.array([], int),
                                       rng.choice(shallow, ns, replace=True)])
                rng.shuffle(idxs)
                ids, attn = make_batch(idxs)
                torch.cuda.synchronize(dev); t0 = time.perf_counter()
                o = decode(ids, attn)
                torch.cuda.synchronize(dev); dts.append(time.perf_counter() - t0)
                deps.append(o["sequences"].shape[1] - 1)
            out[k] = dict(dt_med=float(np.median(dts)), depth=float(np.median(deps)))
            print(f"k={k:>3} deep/64 | dt={out[k]['dt_med']:.4f}s  batch_depth={out[k]['depth']:.0f}  "
                  f"mean_target_len~={(k*args.deep+(B-k)*args.shallow)/B:.1f}", flush=True)
    json.dump(out, open(args.out, "w"))
    d0, d1, d64 = out[0]['dt_med'], out[1]['dt_med'], out[64]['dt_med']
    print(f"\nVERDICT: dt(k=0)={d0:.3f} dt(k=1)={d1:.3f} dt(k=64)={d64:.3f}", flush=True)
    print(f"  one deep query (k=0->1) raises dt by {100*(d1-d0)/d0:.0f}%; k=1->64 by {100*(d64-d1)/d1:.0f}%", flush=True)
    print(f"  if MEAN ruled, dt(k=1)~dt(k=0); if MAX rules, dt(k=1)~dt(k=64). -> {'MAX' if abs(d1-d64)<abs(d1-d0) else 'MEAN'} governs", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--beams", required=True); p.add_argument("--out", required=True)
    p.add_argument("--shallow", type=int, default=3); p.add_argument("--deep", type=int, default=10)
    p.add_argument("--reps", type=int, default=12)
    main(p.parse_args())
