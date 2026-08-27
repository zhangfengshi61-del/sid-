"""s39: EARN baseline (register-token encoder compression) faithful to SID-MLP App-D D.3.

Recipe (guo-mlps-2026 App-D D.3): adapt EARN to the TIGER encoder. Place one PREFIX register at the
first valid encoder position and one SUFFIX register at the last valid encoder position. After
encoder layer k, prune ALL non-register states; the remaining encoder layers AND the decoder
cross-attention use ONLY the two register states. Jointly fine-tune the resulting T5 + register
embeddings from the matched TIGER checkpoint. Sweep k in {1,2,3}, select by validation NDCG@10.
EARN keeps the FULL autoregressive T5 decoder (only the encoder states are compressed) => modest
speedup (paper 1.94x), and TIGER's bidirectional encoder lacks EARN's head/tail attention-sink, so
quality drops. Eval = constrained valid-SID beam-50 (HF generate + a valid-SID prefix Trie).

Standalone (does not modify genrec/sid_mlp). fp32 for robustness (TIGER is 4.59M params). Modes:
train / eval. Env: sidmlp2 (torch 2.7.1, transformers 4.53).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

R = os.environ.get("SERVESID_ROOT", ".")
sys.path.insert(0, f"{R}/repos/SID-MLP")
sys.path.insert(0, f"{R}/repos")
sys.path.insert(0, f"{R}/code")
os.chdir(f"{R}/repos")
for k in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[k] = "1"
from torch.utils.data import DataLoader
from transformers.modeling_outputs import BaseModelOutput
from sid_mlp.constants import DIGIT_OFFSETS
from sid_mlp.beam import build_valid_tensors
from sid_mlp.inference import build_tiger_pipeline, configure_precision, find_valid_set

DEV = torch.device("cuda")


class EARN(nn.Module):
    """TIGER T5 with a register-compressed encoder (prefix+suffix register, prune after layer k)."""

    def __init__(self, t5, k):
        super().__init__()
        self.t5 = t5
        self.k = int(k)
        self.N = t5.config.num_layers  # encoder layers
        d = t5.config.d_model
        self.prefix_reg = nn.Parameter(torch.randn(d) * 0.02)
        self.suffix_reg = nn.Parameter(torch.randn(d) * 0.02)

    def encode(self, input_ids, attn):
        """Register-augment + prune after layer k. Returns (reg (B,2,d), reg_mask (B,2))."""
        enc = self.t5.encoder
        B, L = input_ids.shape
        dev = input_ids.device
        emb = enc.embed_tokens(input_ids)                       # (B,L,d)
        dt = emb.dtype
        d = emb.shape[-1]
        vlen = attn.sum(1).long()                               # (B,)
        Lp = L + 2
        h = emb.new_zeros(B, Lp, d)
        m = attn.new_zeros(B, Lp)
        ar = torch.arange(B, device=dev)
        # prefix register at index 0
        h[:, 0] = self.prefix_reg.to(dt); m[:, 0] = 1
        # original content shifted to indices 1..L
        h[:, 1:L + 1] = emb; m[:, 1:L + 1] = attn
        # suffix register right after the last valid content token (index vlen+1, a former pad slot)
        idx = vlen + 1                                           # (B,)
        h[ar, idx] = self.suffix_reg.to(dt); m[ar, idx] = 1
        # extended (bidirectional, padding-only) mask folded into the position bias
        ext = (1.0 - m[:, None, None, :].to(dt)) * torch.finfo(dt).min      # (B,1,1,Lp)
        sa0 = enc.block[0].layer[0].SelfAttention
        pb_full = sa0.compute_bias(Lp, Lp, device=dev) + ext               # (B,h,Lp,Lp)
        hid = enc.dropout(h)
        for i in range(self.k):                                             # blocks 0..k-1 on full seq
            # mask is folded into pb_full; T5 ignores attention_mask when position_bias is provided
            hid = enc.block[i](hid, attention_mask=None, position_bias=pb_full)[0]
        # prune: keep only the 2 register states
        reg = torch.stack([hid[:, 0], hid[ar, idx]], dim=1)                 # (B,2,d)
        pb2 = sa0.compute_bias(2, 2, device=dev)                           # (1,h,2,2), all-valid => no mask
        for i in range(self.k, self.N):                                     # remaining blocks on 2 registers
            reg = enc.block[i](reg, attention_mask=None, position_bias=pb2)[0]
        reg = enc.dropout(enc.final_layer_norm(reg))
        reg_mask = m.new_ones(B, 2)
        return reg, reg_mask

    def forward(self, input_ids, attn, labels):
        reg, reg_mask = self.encode(input_ids, attn)
        out = self.t5(attention_mask=reg_mask, encoder_outputs=BaseModelOutput(last_hidden_state=reg),
                      labels=labels)
        return out.loss

    @torch.inference_mode()
    def generate(self, input_ids, attn, beam, n_digit, fn, eos_tok, pad_tok):
        reg, reg_mask = self.encode(input_ids, attn)
        gen = self.t5.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=reg), attention_mask=reg_mask,
            max_new_tokens=n_digit + 1, num_beams=beam, num_return_sequences=beam,
            prefix_allowed_tokens_fn=fn, use_cache=True, decoder_start_token_id=0,
            eos_token_id=eos_tok, pad_token_id=pad_tok)
        B = input_ids.shape[0]
        gen = gen.reshape(B, beam, -1)
        return gen[:, :, 1:1 + n_digit]                                    # (B,beam,n_digit)


def build_trie(vi4):
    """Nested valid-SID Trie over offset tokens: d1 -> d2 -> d3 -> set(d4)."""
    trie = {}
    for tup in vi4.keys():
        a, b, c, d = (int(x) for x in tup)
        trie.setdefault(a, {}).setdefault(b, {}).setdefault(c, set()).add(d)
    return trie, sorted(trie.keys())


def make_prefix_fn(trie, d1_list, eos_tok, pad_tok):
    def fn(batch_id, ids):
        seq = ids.tolist()
        n = len(seq) - 1                                # digits generated (seq[0]=decoder_start)
        if n <= 0:
            return d1_list
        if n == 1:
            nxt = trie.get(seq[1]); return list(nxt.keys()) if nxt else [eos_tok]
        if n == 2:
            nxt = trie.get(seq[1], {}).get(seq[2]); return list(nxt.keys()) if nxt else [eos_tok]
        if n == 3:
            nxt = trie.get(seq[1], {}).get(seq[2], {}).get(seq[3]); return list(nxt) if nxt else [eos_tok]
        if n == 4:
            return [eos_tok]
        return [pad_tok]
    return fn


# --------------------------------------------------------------------------- train
def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.fp32 = True; configure_precision(args, DEV)
    pipe = build_tiger_pipeline(args)
    t5 = pipe.model.to(DEV).t5.float()
    for p in t5.parameters():
        p.requires_grad_(True)                                  # joint fine-tune the whole T5
    model = EARN(t5, args.k).to(DEV)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tr = DataLoader(pipe.tokenized_datasets["train"], batch_size=args.batch_size, shuffle=True,
                    collate_fn=pipe.tokenizer.collate_fn["train"], num_workers=2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"[earn] k={args.k}/{model.N} trainable={n_train} train={len(pipe.tokenized_datasets['train'])}",
          flush=True)
    from tqdm import tqdm
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for bi, b in enumerate(tqdm(tr, desc=f"EARN k{args.k} {epoch}/{args.epochs}", ncols=90)):
            if args.test and bi >= 5:
                break
            ii = b["input_ids"].to(DEV); am = b["attention_mask"].to(DEV); lab = b["labels"].to(DEV)
            opt.zero_grad(set_to_none=True)
            loss = model(ii, am, lab)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); losses.append(float(loss))
        sch.step()
        print(f"epoch={epoch} loss={np.mean(losses):.4f}", flush=True)
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "cfg": {"k": args.k}}, args.out_ckpt)
    print(f"[earn] saved {args.out_ckpt}", flush=True)


# --------------------------------------------------------------------------- eval
def build_from_ckpt(args, pipe):
    t5 = pipe.model.to(DEV).t5.float()
    ck = torch.load(args.lens_path, map_location="cpu")
    m = EARN(t5, ck["cfg"]["k"])
    m.load_state_dict(ck["model_state_dict"])
    return m.to(DEV).eval()


def evaluate(args):
    args.fp32 = True; configure_precision(args, DEV)
    pipe = build_tiger_pipeline(args)
    model = build_from_ckpt(args, pipe)
    n_digit = pipe.tokenizer.n_digit
    eos_tok = int(pipe.tokenizer.eos_token); pad_tok = int(pipe.tokenizer.padding_token)
    vs = find_valid_set(args.dataset_tag, explicit_path=None, lens_path=args.lens_path,
                        graph_dir=args.graph_dir)
    (m12, m123, d4k, d4m, k4, v4, _im, vi4) = build_valid_tensors(vs, DEV)
    trie, d1_list = build_trie(vi4)
    fn = make_prefix_fn(trie, d1_list, eos_tok, pad_tok)

    split = args.split if args.split in pipe.tokenized_datasets else "test"
    ds = pipe.tokenized_datasets[split]
    cf = pipe.tokenizer.collate_fn[split if split in pipe.tokenizer.collate_fn else "test"]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=cf)
    rec = defaultdict(int); ndcg = defaultdict(float); tot = 0
    gen_t = 0.0; gen_n = 0; torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for bi, b in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        ii = b["input_ids"].to(DEV); am = b["attention_mask"].to(DEV)
        torch.cuda.synchronize(); _ts = time.time()
        quad = model.generate(ii, am, args.beam_size, n_digit, fn, eos_tok, pad_tok)
        torch.cuda.synchronize()
        if bi >= 3:
            gen_t += time.time() - _ts; gen_n += ii.shape[0]
        codes = quad.detach().cpu().tolist(); labs = b["labels"].tolist()
        for r in range(ii.shape[0]):
            gt = tuple(int(x) for x in labs[r][:n_digit]); gt_it = vi4.get(gt, -1)
            if gt_it == -1:
                continue
            tot += 1
            seen = set(); items = []
            for c in codes[r]:
                it = vi4.get(tuple(int(x) for x in c), -1)
                if it == -1 or it in seen:
                    continue
                seen.add(it); items.append(it)
            try:
                rk = items[:args.final_top_n].index(gt_it)
            except ValueError:
                rk = -1
            for K in (5, 10):
                if 0 <= rk < K:
                    rec[K] += 1; ndcg[K] += 1.0 / math.log2(rk + 2)
        if bi % 20 == 0:
            print(f"batch {bi} n={tot} elapsed={time.time()-t0:.0f}s", flush=True)
    out = {"tag": args.tag, "category": args.category, "k": model.k, "split": split, "n": tot,
           "recall5": rec[5] / max(tot, 1), "ndcg5": ndcg[5] / max(tot, 1),
           "recall10": rec[10] / max(tot, 1), "ndcg10": ndcg[10] / max(tot, 1),
           "samples_per_s": (gen_n / gen_t) if gen_t > 0 else 0.0,
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("EARN_EVAL", json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--category", required=True)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--dataset_tag", required=True)
        p.add_argument("--graph_dir", required=True)
        p.add_argument("--dataset", default="AmazonReviews2023")
        p.add_argument("--sem_ids", required=True)
        p.add_argument("--tag", default="")
        if name == "train":
            p.add_argument("--k", type=int, required=True)
            p.add_argument("--epochs", type=int, default=40)
            p.add_argument("--batch_size", type=int, default=128)
            p.add_argument("--lr", type=float, default=1e-4)
            p.add_argument("--weight_decay", type=float, default=1e-4)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--out_ckpt", required=True)
            p.add_argument("--test", action="store_true")
        else:
            p.add_argument("--lens_path", required=True)
            p.add_argument("--split", default="test")
            p.add_argument("--batch_size", type=int, default=32)
            p.add_argument("--beam_size", type=int, default=50)
            p.add_argument("--final_top_n", type=int, default=50)
            p.add_argument("--max_batches", type=int, default=0)
            p.add_argument("--out", required=True)
    args = ap.parse_args()
    train(args) if args.mode == "train" else evaluate(args)


if __name__ == "__main__":
    main()
