#!/usr/bin/env python
"""
Faithfulness anchor: time the REAL genrec/models/RPG/model.py::generate() and compare to the
microbench. Loads the real RPG class with lightweight stubs for the genrec abstract bases (no
framework deps), random weights (timing shape-driven). generate() internally = forward (encoder +
parallel pred-heads) + brute-force scan + topk == microbench's (encoder+parallel_decode+scan) sum.
Override item_id2tokens with a tensor for large N (skip the slow N-entry dict build).

Compare to microbench clean sums (exp1, same hardware): 12K->2.97ms, 1M->3.65ms, 10M->9.69ms.
"""
import sys, types, argparse, json, statistics, importlib.util
import torch
import torch.nn as nn

# --- stub genrec base modules so the REAL RPG model.py imports without framework deps ---
sys.modules['genrec'] = types.ModuleType('genrec')
gd = types.ModuleType('genrec.dataset')
class AbstractDataset:
    def __init__(self, *a, **k): pass
gd.AbstractDataset = AbstractDataset; sys.modules['genrec.dataset'] = gd
gt = types.ModuleType('genrec.tokenizer')
class AbstractTokenizer:
    def __init__(self, *a, **k): pass
gt.AbstractTokenizer = AbstractTokenizer; sys.modules['genrec.tokenizer'] = gt
gm = types.ModuleType('genrec.model')
class AbstractModel(nn.Module):
    def __init__(self, config, dataset, tokenizer):
        super().__init__(); self.config = config; self.dataset = dataset; self.tokenizer = tokenizer
gm.AbstractModel = AbstractModel; sys.modules['genrec.model'] = gm

MODEL_PY = './code/RPG/genrec/models/RPG/model.py'


def load_real_rpg_class(path):
    spec = importlib.util.spec_from_file_location('rpg_real_model', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RPG


class DS:
    def __init__(self, N):
        self.n_items = N
        self.item2id = {i: i for i in range(N)}


class TOK:
    def __init__(self, N, D, C, build_dict=True):
        self.n_digit = D; self.codebook_size = C; self.vocab_size = D * C + 2
        self.max_token_seq_len = 50; self.eos_token = D * C + 1; self.ignored_label = -100
        if build_dict:
            self.item2tokens = {i: [k * C + 1 + (i % C) for k in range(D)] for i in range(1, N)}
        else:
            self.item2tokens = {}
    def log(self, *a, **k): pass


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
    ap.add_argument("--n_items_list", type=str, default="12101,1000000,10000000")
    ap.add_argument("--n_digit", type=int, default=32)
    ap.add_argument("--codebook_size", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=50)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    dev = "cuda"
    D, C = args.n_digit, args.codebook_size
    RPG = load_real_rpg_class(MODEL_PY)
    torch.manual_seed(0)
    cfg = {'device': dev, 'n_embd': 448, 'n_layer': 2, 'n_head': 4, 'n_inner': 1024,
           'activation_function': 'gelu_new', 'resid_pdrop': 0.0, 'embd_pdrop': 0.5, 'attn_pdrop': 0.5,
           'layer_norm_epsilon': 1e-12, 'initializer_range': 0.02, 'temperature': 0.07,
           'codebook_size': C, 'chunk_size': 1024, 'num_beams': 50, 'n_edges': 50, 'propagation_steps': 3}
    print(f"[faith] real generate() from {MODEL_PY}", flush=True)
    res = {"meta": {"device": torch.cuda.get_device_name(0), "src": MODEL_PY}, "runs": []}
    micro = {12101: 2.97, 1000000: 3.65, 10000000: 9.69}  # exp1 clean microbench sums
    SMALL = 12101
    for N in [int(x) for x in args.n_items_list.split(",")]:
        # build a real RPG at small N (fast dict), then override for large N
        tok = TOK(SMALL, D, C, build_dict=True)
        ds = DS(SMALL)
        model = RPG(cfg, ds, tok).to(dev).eval()
        if N != SMALL:
            model.item_id2tokens = torch.stack(
                [torch.randint(k * C + 1, k * C + C + 1, (N,), device=dev) for k in range(D)], dim=1)
            model.dataset.n_items = N
        B = 1
        batch = {
            'input_ids': torch.randint(0, min(N, SMALL), (B, args.seq_len), device=dev),
            'attention_mask': torch.ones(B, args.seq_len, device=dev),
            'seq_lens': torch.full((B,), args.seq_len, device=dev, dtype=torch.long),
        }
        with torch.no_grad():
            t = cuda_time(lambda: model.generate(batch, n_return_sequences=args.topk), args.warmup, args.iters)
        st = {"median": statistics.median(t), "p10": sorted(t)[int(0.1*len(t))], "p90": sorted(t)[int(0.9*len(t))]}
        res["runs"].append({"N": N, "real_generate_ms": st, "microbench_sum_ms": micro.get(N)})
        mb = micro.get(N)
        rel = f"{100*(st['median']-mb)/mb:+.0f}%" if mb else "n/a"
        print(f"[N={N:>9}] real_generate={st['median']:7.3f}ms  microbench_sum={mb}  diff={rel}", flush=True)
        del model
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[faith] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
