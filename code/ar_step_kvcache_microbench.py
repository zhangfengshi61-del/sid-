#!/usr/bin/env python
"""
KV-cached per-step AR-decode forward cost f_kv(M) vs concurrent beams M — rec-sid-eff Exp4b.

Refines Exp4 (which used recompute over (M,55), an upper bound). REAL serving KV-caches the
history+prefix, so each decode step forwards only ONE new token over M beams attending to the
cached context. Cheaper per step => GPU saturates at LARGER M => the amortization knee moves right.
This pins whether batched genrec AR serving (M = queries x beam) is in the amortizing or linear
regime, which decides whether variable-length's step-saving / padding-waste actually bites.

Method: per timed iter, (untimed) prefill ctx (M, S_ctx) with use_cache -> past; time ONE step
forward (M,1) with past_key_values=past. Re-prefilling each iter avoids cache-mutation pitfalls.
"""
import argparse, json, statistics
import torch
from transformers import GPT2Config, GPT2Model


def time_step(model, ctx_in, step_in, warmup, iters):
    times = []
    for i in range(warmup + iters):
        with torch.no_grad():
            out = model(inputs_embeds=ctx_in, use_cache=True)
            past = out.past_key_values
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            model(inputs_embeds=step_in, past_key_values=past, use_cache=True)
            e.record(); torch.cuda.synchronize()
        if i >= warmup:
            times.append(s.elapsed_time(e))
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M_list", type=str, default="20,50,100,200,500,1000,2000,5000,10000,20000")
    ap.add_argument("--ctx_len", type=int, default=54)   # history 50 + ~4 SID prefix (cached)
    ap.add_argument("--n_embd", type=int, default=448)
    ap.add_argument("--n_layer", type=int, default=2)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_inner", type=int, default=1024)
    ap.add_argument("--codebook_size", type=int, default=256)
    ap.add_argument("--sid_len", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    dev = "cuda"
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    torch.manual_seed(0)
    E = args.n_embd
    vocab = args.sid_len * args.codebook_size + 2
    cfg = GPT2Config(vocab_size=vocab, n_positions=args.ctx_len + 8, n_embd=E, n_layer=args.n_layer,
                     n_head=args.n_head, n_inner=args.n_inner, activation_function="gelu_new",
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = GPT2Model(cfg).to(dev).to(dt).eval()
    print(f"[ar-step-kv] device={torch.cuda.get_device_name(0)} dtype={args.dtype} ctx={args.ctx_len}", flush=True)
    res = {"meta": {"device": torch.cuda.get_device_name(0), "args": vars(args)}, "runs": []}
    for M in [int(x) for x in args.M_list.split(",")]:
        try:
            ctx = torch.randn(M, args.ctx_len, E, device=dev, dtype=dt)
            step = torch.randn(M, 1, E, device=dev, dtype=dt)
            t = time_step(model, ctx, step, args.warmup, args.iters)
            med = statistics.median(t)
            res["runs"].append({"M": M, "f_kv_ms": med, "per_beam_us": 1000 * med / M})
            print(f"[M={M:>6}] f_kv={med:8.4f}ms  per_beam={1000*med/M:7.3f}us", flush=True)
        except Exception as ex:
            print(f"[M={M}] {type(ex).__name__}: {ex}", flush=True)
            res["runs"].append({"M": M, "error": f"{type(ex).__name__}: {ex}"})
            torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[ar-step-kv] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
