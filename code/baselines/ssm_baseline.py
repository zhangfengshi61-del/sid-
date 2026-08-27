"""s38: SSM decoder baselines (GatedDeltaNet / Mamba2) faithful to SID-MLP App-D D.3.

Recipe (guo-mlps-2026 App-D D.3): keep the TIGER encoder FROZEN, feed the encoder hidden
states + a learned bridge token + prefix embeddings (from the frozen TIGER token embedding)
into a CAUSAL SSM stack; the SSM outputs 256-way logits for each SID digit; train only the
SSM-side modules + bridge/output head with the SAME KL+CE objective as SID-MLP (digit-level
combined loss, no list/prefix-KL). Valid-SID constraints applied during beam search.

Faithful decode layout: per example the sequence fed to the causal SSM is
    [ enc_hidden(valid_len) , BRIDGE , e(d1) , e(d2) , e(d3) , <pad> ]
so the 4 decode positions sit immediately AFTER that example's valid encoder tokens (padding is
at the tail only -> the decode positions' causal history never contains padding). Outputs at the
[BRIDGE, e1, e2, e3] positions predict [d1, d2, d3, d4].

This is a STANDALONE script (does not modify the shared sid_mlp package): it reuses
sid_mlp.dataset / .losses / .beam / .inference utilities. Modes: `train`, `eval`.

Env: sidmlp2 (torch 2.7.1+cu126) with fla (flash-linear-attention git 0.5.2, Triton backend).
"""
import argparse, datetime as dt, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

R = os.environ.get("SERVESID_ROOT", ".")
sys.path.insert(0, f"{R}/repos/SID-MLP")
sys.path.insert(0, f"{R}/repos")
os.chdir(f"{R}/repos")
for k in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[k] = "1"

from torch.utils.data import DataLoader
from sid_mlp.constants import CODEBOOK_SIZE, DIGIT_OFFSETS, set_seed, triplet_key_int, quad_key_int
from sid_mlp.losses import loss_fn, d4_loss_fn, topk_correct
from sid_mlp.dataset import prepare_common_data, collate_fn_gpu
from sid_mlp.beam import build_valid_tensors, gather_d4_mask, lookup_item_idx
from sid_mlp.inference import (build_tiger_pipeline, configure_precision, _torch_load,
                               find_valid_set)

DEV = torch.device("cuda")


# --------------------------------------------------------------------------- model
class MLP(nn.Module):
    def __init__(self, d, mult=2, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, mult * d)
        self.fc2 = nn.Linear(mult * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class SSMBlock(nn.Module):
    """Pre-norm residual block: x + SSM(norm(x)) ; x + MLP(norm(x))."""

    def __init__(self, d_model, ssm_type, num_heads, dropout=0.1):
        super().__init__()
        from fla.layers import GatedDeltaNet, Mamba2
        self.norm1 = nn.LayerNorm(d_model)
        if ssm_type == "gdn":
            self.mixer = GatedDeltaNet(hidden_size=d_model, num_heads=num_heads,
                                       head_dim=d_model // num_heads, expand_v=1,
                                       use_short_conv=True, mode="chunk")
        elif ssm_type == "mamba2":
            self.mixer = Mamba2(hidden_size=d_model, head_dim=d_model // num_heads,
                                expand=2, n_groups=1)
        else:
            raise ValueError(ssm_type)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mult=2, dropout=dropout)

    def forward(self, x, attn_mask):
        out = self.mixer(self.norm1(x), attention_mask=attn_mask)
        out = out[0] if isinstance(out, tuple) else out
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x


class SIDSSM(nn.Module):
    def __init__(self, embedding_weight, ssm_type="mamba2", d_model=128, num_heads=4,
                 ssm_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.ssm_type = ssm_type
        embed_dim = embedding_weight.shape[1]
        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=True)
        self.enc_proj = nn.Linear(embed_dim, d_model)
        self.pref_proj = nn.Linear(embed_dim, d_model)
        self.bridge = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.bridge, std=0.02)
        self.blocks = nn.ModuleList(
            [SSMBlock(d_model, ssm_type, num_heads, dropout) for _ in range(ssm_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, CODEBOOK_SIZE)

    def _run(self, enc_seq, enc_mask, prefix_tokens):
        """Run the causal SSM over [enc(valid) , BRIDGE , e(prefix...) , pad] and return the
        hidden states at the Ldec = 1+len(prefix) decode positions. prefix_tokens: (B, k) offset
        token-ids (k in 0..3). Returns (B, Ldec, 256) head logits at the decode positions."""
        B, L, _ = enc_seq.shape
        k = prefix_tokens.shape[1] if prefix_tokens is not None else 0
        Ldec = 1 + k
        valid = (~enc_mask)                                   # (B,L) True=valid
        valid_len = valid.sum(1)                              # (B,)
        bridge = self.bridge.view(1, 1, -1).expand(B, 1, -1)
        blocks = [bridge]
        if k > 0:
            e = self.embedding(prefix_tokens.reshape(-1)).view(B, k, -1)
            blocks.append(self.pref_proj(e))
        dec_blk = torch.cat(blocks, dim=1)                    # (B, Ldec, d_model)
        W = L + Ldec
        W = ((W + 31) // 32) * 32   # s38: bucket seqlen -> <=6 triton shapes (was ~27s/step recompile churn)
        x = enc_seq.new_zeros(B, W, self.d_model)
        x[:, :L] = self.enc_proj(enc_seq) * valid.unsqueeze(-1).to(enc_seq.dtype)
        ar = torch.arange(Ldec, device=enc_seq.device)
        pos = valid_len.view(B, 1) + ar.view(1, -1)           # (B, Ldec) decode positions
        x.scatter_(1, pos.unsqueeze(-1).expand(-1, -1, self.d_model), dec_blk.to(x.dtype))
        m = (torch.arange(W, device=enc_seq.device).view(1, -1)
             < (valid_len + Ldec).view(B, 1)).long()          # (B,W) 1=keep
        for blk in self.blocks:
            x = blk(x, m)
        h = x.gather(1, pos.unsqueeze(-1).expand(-1, -1, self.d_model))  # (B, Ldec, d_model)
        return self.head(self.out_norm(h))                    # (B, Ldec, 256)

    def forward(self, enc_seq, enc_mask, d1_ids=None, d2_ids=None, d3_ids=None):
        """Teacher-forced: given the GT prefix (d1,d2,d3 offset token-ids), one SSM pass over
        [enc, bridge, e1, e2, e3] -> logits at [bridge,e1,e2,e3] = (logits1..4)."""
        if d1_ids is None:  # self-decode fallback (not used in training)
            return self._self_decode(enc_seq, enc_mask)
        prefix = torch.stack([d1_ids, d2_ids, d3_ids], dim=1)  # (B,3) offset token-ids
        logits = self._run(enc_seq, enc_mask, prefix).float()  # (B,4,256)
        return logits[:, 0], logits[:, 1], logits[:, 2], logits[:, 3]

    def next_logits(self, enc_seq, enc_mask, prefix_tokens):
        """Beam step: logits (B,256) for the digit after `prefix_tokens` (B,k), k in 0..3."""
        return self._run(enc_seq, enc_mask, prefix_tokens)[:, -1].float()

    def _self_decode(self, enc_seq, enc_mask):
        l1 = self.next_logits(enc_seq, enc_mask, enc_seq.new_zeros(enc_seq.shape[0], 0, dtype=torch.long))
        d1 = l1.argmax(-1) + DIGIT_OFFSETS[0]
        l2 = self.next_logits(enc_seq, enc_mask, d1.view(-1, 1))
        d2 = l2.argmax(-1) + DIGIT_OFFSETS[1]
        l3 = self.next_logits(enc_seq, enc_mask, torch.stack([d1, d2], 1))
        d3 = l3.argmax(-1) + DIGIT_OFFSETS[2]
        l4 = self.next_logits(enc_seq, enc_mask, torch.stack([d1, d2, d3], 1))
        return l1, l2, l3, l4


# --------------------------------------------------------------------------- train
def train(args):
    set_seed(args.seed)
    tk = _torch_load(args.checkpoint, weights_only=False)
    emb = tk["t5.shared.weight"]; del tk
    train_ds = prepare_common_data(args.graph_dir, args.dataset_tag, DEV, torch.float32)
    val_ds = prepare_common_data(args.graph_dir, args.val_dataset_tag, DEV, torch.float32)
    mk = lambda ds: (lambda idx: collate_fn_gpu(ds, idx))
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=mk(train_ds))
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=mk(val_ds))
    model = SIDSSM(emb, ssm_type=args.ssm_type, d_model=128, num_heads=args.num_heads,
                   ssm_layers=args.ssm_layers, dropout=args.dropout).to(DEV)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_args = argparse.Namespace(loss="combined", temperature=args.temperature, alpha=args.alpha,
                                   d4_teacher="combined")

    def run_batch(batch):
        enc, em, _hk, d1, d2, d3, _d4, tl1, tl2, tl3, tl4, gd1, gd2, gd3, gd4 = batch[:15]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l1, l2, l3, l4 = model(enc, em, d1_ids=d1, d2_ids=d2, d3_ids=d3)
        loss = (loss_fn(loss_args, l1, tl1, gd1) + loss_fn(loss_args, l2, tl2, gd2)
                + loss_fn(loss_args, l3, tl3, gd3) + d4_loss_fn(loss_args, l4, tl4, gd4))
        return loss, (l1, l2, l3, l4), (gd1, gd2, gd3, gd4)

    print(f"[ssm] type={args.ssm_type} layers={args.ssm_layers} heads={args.num_heads} "
          f"trainable={n_train} train={len(train_ds)} val={len(val_ds)}", flush=True)
    from tqdm import tqdm
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for bi, batch in enumerate(tqdm(tl, desc=f"SSM {epoch}/{args.epochs}", ncols=90)):
            if args.test and bi >= 5:
                break
            opt.zero_grad(set_to_none=True)
            loss, _, _ = run_batch(batch)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); losses.append(float(loss))
        sch.step()
        model.eval(); c1 = np.zeros(4); tot = 0
        with torch.no_grad():
            for bi, batch in enumerate(vl):
                if args.test and bi >= 5:
                    break
                _l, logits, labels = run_batch(batch)
                tot += labels[-1].shape[0]
                for i, (lg, lb) in enumerate(zip(logits, labels)):
                    cc, _ = topk_correct(lg, lb, k=5); c1[i] += cc
        print(f"epoch={epoch} train_loss={np.mean(losses):.5f} "
              f"d1@1={c1[0]/max(tot,1):.4f} d2@1={c1[1]/max(tot,1):.4f} "
              f"d3@1={c1[2]/max(tot,1):.4f} d4@1={c1[3]/max(tot,1):.4f}", flush=True)
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "cfg": dict(ssm_type=args.ssm_type, d_model=128, num_heads=args.num_heads,
                            ssm_layers=args.ssm_layers)},
               args.out_ckpt)
    print(f"[ssm] saved {args.out_ckpt}", flush=True)


# --------------------------------------------------------------------------- eval (beam R/N)
def build_model_from_ckpt(path, emb):
    ck = torch.load(path, map_location="cpu")
    cfg = ck["cfg"]
    m = SIDSSM(emb, ssm_type=cfg["ssm_type"], d_model=cfg["d_model"], num_heads=cfg["num_heads"],
               ssm_layers=cfg["ssm_layers"])
    m.load_state_dict(ck["model_state_dict"]); return m.to(DEV).eval()


@torch.inference_mode()
def ssm_beam(model, enc, pad, beam, masks):
    """4-digit valid-SID masked beam using the SSM. Returns quad token-ids (B, beam, 4) +
    scores (B, beam) score-desc."""
    mask_d1_d2, mask_d1d2_d3, d4k, d4m = masks
    B = enc.shape[0]
    W = min(beam, CODEBOOK_SIZE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l1 = model.next_logits(enc, pad, enc.new_zeros(B, 0, dtype=torch.long))
    d1_lp = F.log_softmax(l1, -1)
    d1s, d1i = torch.topk(d1_lp, W, 1)
    d1t = d1i + DIGIT_OFFSETS[0]
    encW = enc.unsqueeze(1).expand(-1, W, -1, -1).reshape(B * W, enc.shape[1], enc.shape[2])
    padW = pad.unsqueeze(1).expand(-1, W, -1).reshape(B * W, pad.shape[1])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l2 = model.next_logits(encW, padW, d1t.reshape(B * W, 1))
    d1cb = (d1t.reshape(-1) - DIGIT_OFFSETS[0]).clamp(0, CODEBOOK_SIZE - 1)
    l2 = l2.masked_fill(~mask_d1_d2[d1cb], float("-inf"))
    d2_lp = F.log_softmax(l2, 1).view(B, W, CODEBOOK_SIZE)
    sc2 = d1s.unsqueeze(-1) + d2_lp
    b2s, b2i = torch.topk(sc2.masked_fill(torch.isnan(sc2), float("-inf")).view(B, -1), beam, 1)
    W2 = beam
    encW = enc.unsqueeze(1).expand(-1, W2, -1, -1).reshape(B * W2, enc.shape[1], enc.shape[2])
    padW = pad.unsqueeze(1).expand(-1, W2, -1).reshape(B * W2, pad.shape[1])
    sb2 = b2i // CODEBOOK_SIZE; sd2 = b2i % CODEBOOK_SIZE
    d1t = torch.gather(d1t, 1, sb2); d2t = sd2 + DIGIT_OFFSETS[1]; psc = b2s
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l3 = model.next_logits(encW, padW, torch.stack([d1t.reshape(-1), d2t.reshape(-1)], 1))
    d1cb = (d1t.reshape(-1) - DIGIT_OFFSETS[0]).clamp(0, CODEBOOK_SIZE - 1)
    d2cb = (d2t.reshape(-1) - DIGIT_OFFSETS[1]).clamp(0, CODEBOOK_SIZE - 1)
    l3 = l3.masked_fill(~mask_d1d2_d3[d1cb, d2cb], float("-inf"))
    d3_lp = F.log_softmax(l3, 1).view(B, W2, CODEBOOK_SIZE)
    sc3 = psc.unsqueeze(-1) + d3_lp
    b3s, b3i = torch.topk(sc3.masked_fill(torch.isnan(sc3), float("-inf")).view(B, -1), beam, 1)
    sb3 = b3i // CODEBOOK_SIZE; sd3 = b3i % CODEBOOK_SIZE
    d1t = torch.gather(d1t, 1, sb3); d2t = torch.gather(d2t, 1, sb3); d3t = sd3 + DIGIT_OFFSETS[2]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l4 = model.next_logits(encW, padW, torch.stack([d1t.reshape(-1), d2t.reshape(-1), d3t.reshape(-1)], 1))
    key3 = triplet_key_int(d1t.reshape(-1) - DIGIT_OFFSETS[0], d2t.reshape(-1) - DIGIT_OFFSETS[1],
                           d3t.reshape(-1) - DIGIT_OFFSETS[2])
    l4 = l4.masked_fill(~gather_d4_mask(d4k, d4m, key3), float("-inf"))
    d4_lp = F.log_softmax(l4, 1)
    t4v, t4i = torch.topk(d4_lp, CODEBOOK_SIZE, 1)
    d4t = t4i + DIGIT_OFFSETS[3]
    quad = b3s.unsqueeze(-1) + t4v.view(B, beam, CODEBOOK_SIZE)
    tqs, tqi = torch.topk(quad.masked_fill(torch.isnan(quad), float("-inf")).view(B, -1), beam, 1)
    bi = tqi // CODEBOOK_SIZE
    d1f = torch.gather(d1t, 1, bi); d2f = torch.gather(d2t, 1, bi)
    d3f = torch.gather(d3t, 1, bi); d4f = torch.gather(d4t.view(B, -1), 1, tqi)
    return torch.stack([d1f, d2f, d3f, d4f], -1), tqs


def evaluate(args):
    args.encoder_ckpt = None; args.fp32 = False
    configure_precision(args, DEV)
    pipe = build_tiger_pipeline(args)
    t5 = pipe.model.to(DEV).eval().t5
    t5.bfloat16()
    for p in t5.parameters():
        p.requires_grad_(False)
    tk = _torch_load(args.checkpoint, weights_only=False)
    model = build_model_from_ckpt(args.lens_path, tk["t5.shared.weight"]); del tk
    vs = find_valid_set(args.dataset_tag, explicit_path=None, lens_path=args.lens_path, graph_dir=args.graph_dir)
    (m12, m123, d4k, d4m, k4, v4, _im, vi4) = build_valid_tensors(vs, DEV)
    masks = (m12, m123, d4k, d4m)
    ds = pipe.tokenized_datasets["test"]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=pipe.tokenizer.collate_fn["test"])
    rec = defaultdict(int); ndcg = defaultdict(float); tot = 0
    t0 = time.time()
    beam_t = 0.0; beam_n = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            if args.max_batches and bi >= args.max_batches:
                break
            ii = batch["input_ids"].to(DEV); am = batch["attention_mask"].to(DEV)
            torch.cuda.synchronize(); _ts = time.time()  # s38: end-to-end (encoder+beam) throughput
            enc = t5.encoder(input_ids=ii, attention_mask=am, return_dict=True).last_hidden_state
            pad = ~am.bool()
            quad, _ = ssm_beam(model, enc, pad, args.beam_size, masks)
            torch.cuda.synchronize()
            if bi >= 3:  # skip warmup batches (triton compile)
                beam_t += time.time() - _ts; beam_n += ii.shape[0]
            codes = quad.detach().cpu().tolist()
            labs = batch["labels"].tolist()
            for r in range(ii.shape[0]):
                gt = tuple(int(x) for x in labs[r][:4]); gt_it = vi4.get(gt, -1)
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
    out = {"tag": args.tag, "category": args.category, "n": tot,
           "recall5": rec[5] / max(tot, 1), "ndcg5": ndcg[5] / max(tot, 1),
           "recall10": rec[10] / max(tot, 1), "ndcg10": ndcg[10] / max(tot, 1),
           "samples_per_s": (beam_n / beam_t) if beam_t > 0 else 0.0,
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "timed_n": beam_n}
    import json
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("SSM_EVAL", json.dumps(out), flush=True)


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--category", required=True)
        p.add_argument("--checkpoint", required=True)   # TIGER teacher .pth (embedding + encoder)
        p.add_argument("--dataset_tag", required=True)
        p.add_argument("--graph_dir", required=True)
        p.add_argument("--dataset", default="AmazonReviews2023")
        p.add_argument("--sem_ids", default="")
        p.add_argument("--ssm_type", default="mamba2", choices=["mamba2", "gdn"])
        p.add_argument("--ssm_layers", type=int, default=2)
        p.add_argument("--num_heads", type=int, default=4)
        p.add_argument("--tag", default="")
        if name == "train":
            p.add_argument("--val_dataset_tag", required=True)
            p.add_argument("--epochs", type=int, default=200)
            p.add_argument("--batch_size", type=int, default=512)
            p.add_argument("--lr", type=float, default=5e-4)
            p.add_argument("--weight_decay", type=float, default=1e-4)
            p.add_argument("--dropout", type=float, default=0.1)
            p.add_argument("--alpha", type=float, default=0.7)
            p.add_argument("--temperature", type=float, default=1.0)
            p.add_argument("--grad_clip", type=float, default=1.0)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--out_ckpt", required=True)
            p.add_argument("--test", action="store_true")
        else:
            p.add_argument("--lens_path", required=True)   # SSM ckpt
            p.add_argument("--batch_size", type=int, default=32)
            p.add_argument("--beam_size", type=int, default=50)
            p.add_argument("--final_top_n", type=int, default=50)
            p.add_argument("--max_batches", type=int, default=0)
            p.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
