#!/usr/bin/env python
"""
Autoregressive (TIGER-style) decode microbench — rec-sid-eff, experiment 2.

Tests the contribution-defining question: RPG's abstract claims autoregressive (AR) decode is
"inefficient due to resource-intensive beam search and sequential decoding," motivating its
parallel long-SID design. But Exp1 showed RPG's parallel decode pays an O(N) catalog scan that
dominates at N>~1-3M. AR beam search over a prefix tree is SUBLINEAR in N (no full scan). So:
does AR actually BEAT parallel-RPG at deployment-scale N, despite AR's L sequential forwards?

Fair design: SAME GPT2 backbone as RPG (n_embd=448, n_layer=2, n_head=4) -> isolates decode
STRATEGY, not model size. AR decode = L sequential forwards over a beam of B, with per-step
expansion over a codebook of size C (TIGER-style). NO O(N) scan at runtime (candidate set is
the beam; item resolution is a trie-leaf lookup, O(B)). Runtime is N-INDEPENDENT by construction
-> we sweep L (and B) to map AR latency, then overlay with Exp1's RPG(N) curve to find the
crossover N where AR wins.

This is the recompute (no-KV-cache) variant => an UPPER BOUND on AR cost (real TIGER caches the
encoder + decoder KV). If even this upper bound beats RPG at scale, the conclusion holds.
"""
import argparse, json, statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def stats(ms):
    ms = sorted(ms); n = len(ms)
    return {"median": statistics.median(ms), "mean": statistics.fmean(ms),
            "p10": ms[max(0, int(0.10 * n))], "p90": ms[min(n - 1, int(0.90 * n))],
            "min": ms[0], "max": ms[-1], "n": n}


@torch.no_grad()
def run_cfg(args, L, B, dev, dt):
    """AR beam decode with SID length L, beam width B, batch=1 query. Same backbone as RPG."""
    C, E = args.codebook_size, args.n_embd
    H = args.seq_len  # user-history length
    vocab = L * C + 2
    cfg = GPT2Config(vocab_size=vocab, n_positions=H + L + 2, n_embd=E, n_layer=args.n_layer,
                     n_head=args.n_head, n_inner=args.n_inner, activation_function="gelu_new",
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = GPT2Model(cfg).to(dev).to(dt).eval()
    heads = nn.ModuleList([nn.Linear(E, C).to(dev).to(dt) for _ in range(L)]).eval()
    wte = model.get_input_embeddings()
    # history context embedding (B beams share the same query's history)
    hist_ids = torch.randint(0, vocab, (1, H), device=dev)
    hist_emb = wte(hist_ids).expand(B, H, E)  # (B, H, E)

    def encoder_forward():
        # process the user history once (shared by beams); cost ~ RPG encoder
        return model(inputs_embeds=hist_emb[:1]).last_hidden_state

    def ar_decode():
        # beams start empty; recompute over [history; sid_prefix] each step (upper bound, no KV cache)
        beam_scores = torch.zeros(B, device=dev, dtype=torch.float32)
        prefix_emb = torch.zeros(B, 0, E, device=dev, dtype=dt)
        active = 1  # number of live beams (1 at step 0, then B)
        for t in range(L):
            inp = torch.cat([hist_emb[:active], prefix_emb[:active]], dim=1)  # (active, H+t, E)
            hid = model(inputs_embeds=inp).last_hidden_state[:, -1, :]        # (active, E)
            logits = F.log_softmax(heads[t](hid), dim=-1)                     # (active, C)
            cand = beam_scores[:active].unsqueeze(1) + logits                 # (active, C)
            flat = cand.reshape(-1)
            nb = min(B, flat.numel())
            top = torch.topk(flat, nb)
            beam_scores = top.values
            parent = top.indices // C
            tok = top.indices % C
            # gather new prefix embeddings: parent's prefix + new token embedding
            tok_emb = wte(tok + t * C + 1)                                    # (nb, E) codebook-t token ids
            new_prefix = torch.cat([prefix_emb[parent], tok_emb.unsqueeze(1)], dim=1)  # (nb, t+1, E)
            prefix_emb = new_prefix
            active = nb
        # item resolution: trie-leaf lookup over B complete SIDs is O(B); negligible (omitted/timed as ~0)
        return beam_scores

    res = {"L": L, "B": B, "batch_size": 1, "dtype": args.dtype,
           "encoder_forward": stats(cuda_time(encoder_forward, args.warmup, args.iters)),
           "ar_decode_total": stats(cuda_time(ar_decode, args.warmup, args.iters))}
    del model, heads
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L_list", type=str, default="4,8,16,32")   # SID length (TIGER=4, RPG=32)
    ap.add_argument("--B_list", type=str, default="20,50")        # beam width
    ap.add_argument("--codebook_size", type=int, default=256)
    ap.add_argument("--n_embd", type=int, default=448)
    ap.add_argument("--n_layer", type=int, default=2)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_inner", type=int, default=1024)
    ap.add_argument("--seq_len", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    dev = "cuda"
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    torch.manual_seed(0)
    print(f"[ar-microbench] device={torch.cuda.get_device_name(0)} dtype={args.dtype}", flush=True)
    all_res = {"meta": {"device": torch.cuda.get_device_name(0), "args": vars(args),
                        "torch": torch.__version__}, "runs": []}
    for B in [int(x) for x in args.B_list.split(",")]:
        for L in [int(x) for x in args.L_list.split(",")]:
            try:
                r = run_cfg(args, L, B, dev, dt)
                all_res["runs"].append(r)
                ef, ad = r["encoder_forward"]["median"], r["ar_decode_total"]["median"]
                print(f"[L={L:>2} B={B:>3}] encoder={ef:7.3f}ms  ar_decode={ad:8.3f}ms  "
                      f"total={ef+ad:8.3f}ms", flush=True)
            except RuntimeError as ex:
                print(f"[L={L} B={B}] RuntimeError: {ex}", flush=True)
                all_res["runs"].append({"L": L, "B": B, "error": str(ex)})
                torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_res, f, indent=2)
        print(f"[ar-microbench] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
