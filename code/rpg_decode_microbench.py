#!/usr/bin/env python
"""
RPG decode cost-breakdown microbench — rec-sid-eff, experiment 1.

Question: RPG ("Generating Long Semantic IDs in Parallel", KDD'25) sells its decode as
efficient because it predicts all 32 SID tokens in ONE parallel forward (vs autoregressive
L sequential forwards). But genrec/models/RPG/model.py::generate() then scores candidates
either by (a) a BRUTE-FORCE scan over ALL items [default], or (b) an O(N^2)-build graph.
So the marketed saving (the forward) may be the cheap part, and the ignored part (the
O(N) catalog scan) may dominate and scale with N.

This microbench faithfully replicates the decode arithmetic of generate() with random
weights (timing is shape-driven, so no training/checkpoint needed) and times each component
separately, swept over catalog size N. Single-GPU, cuda-event timed, warmup+repeats.

Components timed (per query, batch_size configurable):
  (1) encoder_forward    : GPT2 over user history  -> last_hidden            [the seq model]
  (2) parallel_decode    : 32 pred-heads + per-digit logits (the "parallel" win RPG markets)
  (3) bruteforce_scan    : expand+gather over ALL N items, mean, topk  [the ignored O(N) cost]
  (3g) graph_build+prop   : build_ii_sim_mat (O(N^2)) + graph_propagation  [optional, small N]

Faithful to genrec/models/RPG/model.py @ vendor/RPG (Meta, CC-BY-NC). Config defaults match
genrec/models/RPG/config.yaml (n_codebook=32, codebook_size=256, n_embd=448, n_layer=2,
n_head=4, n_inner=1024, max_item_seq_len=50, num_beams=50, n_edges=50, propagation_steps=3).
"""
import argparse, json, statistics, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model


class ResBlock(nn.Module):
    """Verbatim from genrec/models/RPG/model.py."""
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()
    def forward(self, x):
        return x + self.act(self.linear(x))


def cuda_time(fn, warmup, iters):
    """Return list of per-call ms via cuda events (sync each iter)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return out


def stats(ms):
    ms = sorted(ms)
    n = len(ms)
    return {
        "median": statistics.median(ms),
        "mean": statistics.fmean(ms),
        "p10": ms[max(0, int(0.10 * n) - 0)],
        "p90": ms[min(n - 1, int(0.90 * n))],
        "min": ms[0], "max": ms[-1], "n": n,
    }


def build_pieces(args, N, dev, dt):
    D, C, E = args.n_digit, args.codebook_size, args.n_embd
    vocab = D * C + 2  # RPG uses wte.weight[1:-1] == D*C rows
    cfg = GPT2Config(vocab_size=vocab, n_positions=args.seq_len, n_embd=E,
                     n_layer=args.n_layer, n_head=args.n_head, n_inner=args.n_inner,
                     activation_function="gelu_new", resid_pdrop=0.0, embd_pdrop=0.0,
                     attn_pdrop=0.0)
    gpt2 = GPT2Model(cfg).to(dev).to(dt).eval()
    pred_heads = nn.ModuleList([ResBlock(E).to(dev).to(dt) for _ in range(D)]).eval()
    # item_id2tokens: (N, D); digit k token id in [k*C+1, k*C+C]
    item_id2tokens = torch.stack(
        [torch.randint(k * C + 1, k * C + C + 1, (N,), device=dev) for k in range(D)], dim=1)
    wte = gpt2.get_input_embeddings()
    B = args.batch_size
    input_ids = torch.randint(0, N, (B, args.seq_len), device=dev)
    attn = torch.ones(B, args.seq_len, device=dev, dtype=dt)
    seq_lens = torch.full((B,), args.seq_len, device=dev, dtype=torch.long)
    return gpt2, pred_heads, item_id2tokens, wte, input_ids, attn, seq_lens


@torch.no_grad()
def run_N(args, N, dev, dt):
    D, C, E = args.n_digit, args.codebook_size, args.n_embd
    B = args.batch_size
    gpt2, pred_heads, item_id2tokens, wte, input_ids, attn, seq_lens = build_pieces(args, N, dev, dt)

    def encoder_forward():
        input_tokens = item_id2tokens[input_ids]            # (B, seq, D)
        input_embs = wte(input_tokens).mean(dim=-2)         # (B, seq, E)
        out = gpt2(inputs_embeds=input_embs, attention_mask=attn)
        return out.last_hidden_state                        # (B, seq, E)

    last_hidden = encoder_forward()

    def parallel_decode():
        final = torch.cat([pred_heads[i](last_hidden).unsqueeze(-2) for i in range(D)], dim=-2)  # (B,seq,D,E)
        states = final.gather(1, (seq_lens - 1).view(-1, 1, 1, 1).expand(-1, 1, D, E))            # (B,1,D,E)
        states = F.normalize(states, dim=-1)
        token_emb = F.normalize(wte.weight[1:-1], dim=-1)
        token_embs = torch.chunk(token_emb, D, dim=0)
        logits = [torch.matmul(states[:, 0, i, :], token_embs[i].T) for i in range(D)]            # (B,C)
        logits = [F.log_softmax(l, dim=-1) for l in logits]
        return torch.cat(logits, dim=-1)                                                          # (B, D*C)

    token_logits = parallel_decode()

    def bruteforce_scan():
        item_logits = torch.gather(
            input=token_logits.unsqueeze(-2).expand(-1, N - 1, -1),                                # view
            dim=-1,
            index=(item_id2tokens[1:, :] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1),
        ).mean(dim=-1)                                                                             # (B, N-1)
        return item_logits.topk(args.topk, dim=-1).indices + 1

    res = {"N": N, "batch_size": B, "dtype": args.dtype,
           "encoder_forward": stats(cuda_time(encoder_forward, args.warmup, args.iters)),
           "parallel_decode": stats(cuda_time(parallel_decode, args.warmup, args.iters)),
           "bruteforce_scan": stats(cuda_time(bruteforce_scan, args.warmup, args.iters))}
    res["peak_mem_MB"] = torch.cuda.max_memory_allocated() / 1e6

    # optional graph path (only feasible at small N: builds N x N matrix)
    if args.graph and (N * N * 4) <= args.graph_mem_cap_bytes:
        chunk = args.chunk_size
        def build_ii_sim_mat():
            token_embs = F.normalize(wte.weight[1:-1].view(D, C, -1), dim=-1)
            token_sims01 = 0.5 * (torch.bmm(token_embs, token_embs.transpose(1, 2)) + 1.0)        # (D,C,C)
            iim = torch.zeros((N, N), device=dev, dtype=torch.float32)
            for i0 in range(1, N, chunk):
                i1 = min(i0 + chunk, N); ti = item_id2tokens[i0:i1]
                for j0 in range(1, N, chunk):
                    j1 = min(j0 + chunk, N); tj = item_id2tokens[j0:j1]
                    sb = torch.zeros((i1 - i0, j1 - j0), device=dev, dtype=torch.float32)
                    for k in range(D):
                        ri = ti[:, k] - k * C - 1; ci = tj[:, k] - k * C - 1
                        sb += token_sims01[k].index_select(0, ri).index_select(1, ci)
                    iim[i0:i1, j0:j1] = sb / D
            return iim
        # graph build is expensive; time with fewer iters
        gi = max(1, args.iters // 6)
        res["graph_build"] = stats(cuda_time(build_ii_sim_mat, 1, gi))
        adjacency = torch.topk(build_ii_sim_mat(), k=min(args.n_edges, N - 1), dim=-1).indices

        def graph_prop():
            bs = token_logits.shape[0]
            topk_nodes = torch.randint(1, N, (bs, args.num_beams), dtype=torch.long, device=dev)
            for _ in range(args.propagation_steps):
                alln = adjacency[topk_nodes].view(bs, -1)
                nxt = []
                for b in range(bs):
                    nb = torch.unique(alln[b])
                    sc = torch.gather(token_logits[b].unsqueeze(0).expand(nb.shape[0], -1),
                                      -1, (item_id2tokens[nb] - 1)).mean(-1)
                    idxs = torch.topk(sc, min(args.num_beams, nb.shape[0])).indices
                    nxt.append(nb[idxs])
                topk_nodes = torch.stack(nxt, dim=0)
            return topk_nodes
        res["graph_prop"] = stats(cuda_time(graph_prop, 2, args.iters))
    elif args.graph:
        res["graph_build"] = "SKIPPED_OOM_BY_DESIGN (N^2 too large)"
    del gpt2, pred_heads, item_id2tokens
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_items_list", type=str, default="12101,100000,1000000,10000000")
    ap.add_argument("--n_digit", type=int, default=32)
    ap.add_argument("--codebook_size", type=int, default=256)
    ap.add_argument("--n_embd", type=int, default=448)
    ap.add_argument("--n_layer", type=int, default=2)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_inner", type=int, default=1024)
    ap.add_argument("--seq_len", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_beams", type=int, default=50)
    ap.add_argument("--n_edges", type=int, default=50)
    ap.add_argument("--propagation_steps", type=int, default=3)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--graph_mem_cap_bytes", type=int, default=8_000_000_000)
    ap.add_argument("--chunk_size", type=int, default=1024)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    dev = "cuda"
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    torch.manual_seed(0)
    print(f"[microbench] device={torch.cuda.get_device_name(0)} dtype={args.dtype} "
          f"batch={args.batch_size} iters={args.iters}", flush=True)
    all_res = {"meta": {"device": torch.cuda.get_device_name(0), "args": vars(args),
                        "torch": torch.__version__}, "runs": []}
    for N in [int(x) for x in args.n_items_list.split(",")]:
        try:
            r = run_N(args, N, dev, dt)
            all_res["runs"].append(r)
            ef, pd, bf = r["encoder_forward"]["median"], r["parallel_decode"]["median"], r["bruteforce_scan"]["median"]
            print(f"[N={N:>9}] encoder={ef:8.3f}ms  parallel_decode={pd:8.3f}ms  "
                  f"bruteforce_scan={bf:9.3f}ms  scan/decode={bf/max(pd,1e-9):7.1f}x  "
                  f"peakmem={r['peak_mem_MB']:.0f}MB", flush=True)
        except RuntimeError as ex:
            print(f"[N={N}] RuntimeError: {ex}", flush=True)
            all_res["runs"].append({"N": N, "error": str(ex)})
            torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_res, f, indent=2)
        print(f"[microbench] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
