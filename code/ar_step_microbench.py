#!/usr/bin/env python
"""
Per-step AR-decode forward cost f(M) vs active-beam count M — rec-sid-eff Exp4 (decisive part).

The variable-length-SID efficiency claim (CapsID/VarLenRec) = shorter avg decode length L̄ -> fewer
AR steps. Whether this survives BATCHED serving turns on one unknown: does the AR decoder forward
AMORTIZE with the number of concurrent beams M (= queries x beam)? Same GPT2 backbone as RPG/Exp2.

  - If f(M) is ~flat up to large M: batched serving is forward-amortized -> reducing steps (varlen)
    barely helps throughput, and padding-to-Lmax barely hurts (extra steps process many beams ~free).
  - If f(M) grows ~linearly early: steps dominate -> varlen helps, but naive batched padding-to-Lmax
    erases the avg-L saving (continuous-batching/early-exit needed to recover it).

Recompute forward over (M, S) (consistent with Exp2; KV-cache would lower absolute per-step cost
but NOT the relative step-count padding-waste, which is distribution-driven). Reports f(M) and the
per-beam cost f(M)/M (flat per-beam => linear regime; falling per-beam => amortization).
"""
import argparse, json, statistics
import torch
from transformers import GPT2Config, GPT2Model


def cuda_time(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M_list", type=str, default="20,50,100,200,500,1000,2000,5000,10000")
    ap.add_argument("--seq_len", type=int, default=55)   # history 50 + ~5 SID prefix
    ap.add_argument("--n_embd", type=int, default=448)
    ap.add_argument("--n_layer", type=int, default=2)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_inner", type=int, default=1024)
    ap.add_argument("--codebook_size", type=int, default=256)
    ap.add_argument("--sid_len", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    dev = "cuda"
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    torch.manual_seed(0)
    E, S = args.n_embd, args.seq_len
    vocab = args.sid_len * args.codebook_size + 2
    cfg = GPT2Config(vocab_size=vocab, n_positions=S + 4, n_embd=E, n_layer=args.n_layer,
                     n_head=args.n_head, n_inner=args.n_inner, activation_function="gelu_new",
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = GPT2Model(cfg).to(dev).to(dt).eval()
    print(f"[ar-step] device={torch.cuda.get_device_name(0)} dtype={args.dtype} S={S}", flush=True)
    res = {"meta": {"device": torch.cuda.get_device_name(0), "args": vars(args)}, "runs": []}
    for M in [int(x) for x in args.M_list.split(",")]:
        try:
            inp = torch.randn(M, S, E, device=dev, dtype=dt)
            with torch.no_grad():
                fn = lambda: model(inputs_embeds=inp).last_hidden_state
                t = cuda_time(fn, args.warmup, args.iters)
            med = statistics.median(t)
            res["runs"].append({"M": M, "f_ms": med, "per_beam_us": 1000 * med / M,
                                "peak_mem_MB": torch.cuda.max_memory_allocated() / 1e6})
            print(f"[M={M:>6}] f={med:8.4f}ms  per_beam={1000*med/M:7.3f}us  "
                  f"peakmem={torch.cuda.max_memory_allocated()/1e6:.0f}MB", flush=True)
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError as ex:
            print(f"[M={M}] RuntimeError: {ex}", flush=True)
            res["runs"].append({"M": M, "error": str(ex)})
            torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[ar-step] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
