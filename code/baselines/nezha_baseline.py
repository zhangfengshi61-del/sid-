"""s38: NEZHA baseline (self-drafting MTP head) faithful to SID-MLP App-D D.4.

Recipe (guo-mlps-2026 App-D D.4): keep NEZHA's hidden-state contract on the TIGER encoder —
input [BOS, q_user, x1..xt, SP1..SP4] (4 SID placeholders appended to the encoder input); the
final context hidden before SP1 = h0, the 4 placeholder hidden states = h1..h4; the T5 DECODER is
bypassed; an MTP head predicts the 4 SID digits with per-position heads + a recurrent state update;
the T5 encoder + MTP head are JOINTLY fine-tuned (from the matched TIGER checkpoint) with hard SID
labels. Inference = self-drafting: one encoder pass then cheap recurrent MTP steps (no AR decoder)
with valid-SID beam masking ⇒ the fast tier (paper 7.27x, but -15% NDCG@10).

Standalone (does not modify the shared genrec/sid_mlp packages). Modes: train / eval.
Env: sidmlp2 (torch 2.7.1). Data via the genrec pipeline (live encoder needs raw input_ids).
"""
import argparse, math, os, sys, time
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
from sid_mlp.constants import CODEBOOK_SIZE, DIGIT_OFFSETS, set_seed, triplet_key_int
from sid_mlp.beam import build_valid_tensors, gather_d4_mask
from sid_mlp.inference import (build_tiger_pipeline, configure_precision, _torch_load,
                               find_valid_set)

DEV = torch.device("cuda")
OFF = DIGIT_OFFSETS  # [1,257,513,769]


def make_head(in_dim, hidden=256, dropout=0.1):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(hidden, CODEBOOK_SIZE))


class NEZHA(nn.Module):
    def __init__(self, t5, d_model=128, state_dim=128, head_hidden=256, dropout=0.1):
        super().__init__()
        self.t5 = t5                                   # trainable encoder + shared embedding
        self.d_model = d_model
        self.placeholders = nn.Parameter(torch.randn(4, d_model) * 0.02)
        self.state_proj = nn.Linear(d_model, state_dim)
        self.gru = nn.GRUCell(d_model, state_dim)      # in = chosen-digit token embed; hidden = state
        self.heads = nn.ModuleList([make_head(d_model + state_dim, head_hidden, dropout) for _ in range(4)])

    def encode(self, input_ids, attn):
        """One encoder pass with the 4 SID placeholders appended. Returns h0 (B,d) = last valid
        content hidden, hph (B,4,d) = the 4 placeholder hidden states."""
        B, L = input_ids.shape
        emb = self.t5.shared(input_ids)                                     # (B,L,d)
        ph = self.placeholders.unsqueeze(0).expand(B, -1, -1).to(emb.dtype)  # (B,4,d)
        x = torch.cat([emb, ph], dim=1)                                     # (B,L+4,d)
        m = torch.cat([attn, attn.new_ones(B, 4)], dim=1)
        enc = self.t5.encoder(inputs_embeds=x, attention_mask=m).last_hidden_state  # (B,L+4,d)
        vlen = attn.sum(1).clamp(min=1)                                     # last valid content pos = vlen-1
        h0 = enc[torch.arange(B, device=enc.device), vlen - 1]             # (B,d)
        hph = enc[:, L:L + 4]                                               # (B,4,d)
        return h0, hph

    def forward(self, input_ids, attn, label_tokens):
        """Teacher-forced: label_tokens (B,4) offset SID token-ids. Returns 4x (B,256) logits."""
        h0, hph = self.encode(input_ids, attn)
        state = self.state_proj(h0.float())
        logits = []
        for i in range(4):
            logits.append(self.heads[i](torch.cat([hph[:, i].float(), state], dim=-1)))
            e = self.t5.shared(label_tokens[:, i]).float()                  # teacher-forced digit embed
            state = self.gru(e, state)
        return logits


# --------------------------------------------------------------------------- train
def _codes_from_labels(label_tokens):
    off = torch.as_tensor(OFF, device=label_tokens.device)
    return (label_tokens[:, :4] - off.view(1, 4)).clamp(0, CODEBOOK_SIZE - 1)


def train(args):
    pipe = build_tiger_pipeline(args)
    # s48 SEED BUG FIX: genrec's Pipeline.__init__ calls init_seed(config["rand_seed"]) (a FIXED
    # config value), which re-seeds random/numpy/torch/cuda/accelerate. Seeding BEFORE the pipeline
    # build was therefore a no-op -> every --seed produced bit-identical weights (verified: seeds
    # 43 and 44 had identical 40-epoch loss traces and identical model_state_dict hashes).
    # ssm_baseline.py was always correct (it seeds after the build); this restores the same order.
    set_seed(args.seed)
    t5 = pipe.model.to(DEV).t5
    for p in t5.parameters():
        p.requires_grad_(True)                          # joint fine-tune the encoder
    model = NEZHA(t5, d_model=t5.config.d_model, state_dim=args.state_dim,
                  head_hidden=args.head_hidden, dropout=args.dropout).to(DEV)
    # only encoder + shared + nezha modules train; the decoder is unused (freeze to save memory)
    for p in t5.decoder.parameters():
        p.requires_grad_(False)
    if t5.lm_head is not None:
        for p in t5.lm_head.parameters():
            p.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tr = DataLoader(pipe.tokenized_datasets["train"], batch_size=args.batch_size, shuffle=True,
                    collate_fn=pipe.tokenizer.collate_fn["train"], num_workers=2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"[nezha] trainable={n_train} train={len(pipe.tokenized_datasets['train'])} "
          f"state={args.state_dim}", flush=True)
    from tqdm import tqdm
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []; c1 = np.zeros(4); tot = 0
        for bi, b in enumerate(tqdm(tr, desc=f"NEZHA {epoch}/{args.epochs}", ncols=90)):
            if args.test and bi >= 5:
                break
            ii = b["input_ids"].to(DEV); am = b["attention_mask"].to(DEV)
            lab = b["labels"].to(DEV)
            codes = _codes_from_labels(lab)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ii, am, lab[:, :4])
            loss = sum(F.cross_entropy(logits[i].float(), codes[:, i]) for i in range(4))
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); losses.append(float(loss))
            tot += ii.shape[0]
            for i in range(4):
                c1[i] += (logits[i].argmax(-1) == codes[:, i]).sum().item()
        sch.step()
        print(f"epoch={epoch} loss={np.mean(losses):.4f} d1@1={c1[0]/max(tot,1):.4f} "
              f"d2@1={c1[1]/max(tot,1):.4f} d3@1={c1[2]/max(tot,1):.4f} d4@1={c1[3]/max(tot,1):.4f}",
              flush=True)
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "cfg": dict(state_dim=args.state_dim, head_hidden=args.head_hidden,
                            d_model=t5.config.d_model)}, args.out_ckpt)
    print(f"[nezha] saved {args.out_ckpt}", flush=True)


# --------------------------------------------------------------------------- eval (MTP beam)
@torch.inference_mode()
def nezha_beam(model, input_ids, attn, beam, masks):
    mask_d1_d2, mask_d1d2_d3, d4k, d4m = masks
    B = input_ids.shape[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h0, hph = model.encode(input_ids, attn)
    hph = hph.float()
    state = model.state_proj(h0.float())                                   # (B, s)
    W = min(beam, CODEBOOK_SIZE)
    l1 = model.heads[0](torch.cat([hph[:, 0], state], -1))                  # (B,256)
    d1_lp = F.log_softmax(l1, -1)
    d1s, d1i = torch.topk(d1_lp, W, 1)                                      # (B,W)
    d1t = d1i + OFF[0]
    # expand to beam (W == beam for beam<=256); step the recurrent state with the chosen d1
    state = state.unsqueeze(1).expand(-1, W, -1).reshape(B * W, -1)
    e = model.t5.shared(d1t.reshape(-1)).float()
    state = model.gru(e, state)                                            # (B*W, s)
    # digit 2
    def mask2(lg):
        d1cb = (d1t.reshape(-1) - OFF[0]).clamp(0, CODEBOOK_SIZE - 1)
        return lg.reshape(B * W, CODEBOOK_SIZE).masked_fill(~mask_d1_d2[d1cb], float("-inf")).view(B, W, CODEBOOK_SIZE)
    h2 = hph.unsqueeze(1).expand(-1, W, -1, -1)[:, :, 1].reshape(B * W, -1)
    l2 = model.heads[1](torch.cat([h2, state], -1)).view(B, W, CODEBOOK_SIZE)
    l2 = mask2(l2)
    d2lp = F.log_softmax(l2, -1)
    sc2 = d1s.unsqueeze(-1) + d2lp
    b2s, b2i = torch.topk(sc2.masked_fill(torch.isnan(sc2), float("-inf")).view(B, -1), beam, 1)
    sb2 = b2i // CODEBOOK_SIZE; sd2 = b2i % CODEBOOK_SIZE
    d1t = torch.gather(d1t, 1, sb2); d2t = sd2 + OFF[1]
    state = state.view(B, W, -1).gather(1, sb2.unsqueeze(-1).expand(-1, -1, state.shape[-1])).reshape(B * beam, -1)
    e = model.t5.shared(d2t.reshape(-1)).float(); state = model.gru(e, state)
    hphB = hph.unsqueeze(1).expand(-1, beam, -1, -1)
    # digit 3
    h3 = hphB[:, :, 2].reshape(B * beam, -1)
    l3 = model.heads[2](torch.cat([h3, state], -1)).view(B, beam, CODEBOOK_SIZE)
    d1cb = (d1t.reshape(-1) - OFF[0]).clamp(0, CODEBOOK_SIZE - 1)
    d2cb = (d2t.reshape(-1) - OFF[1]).clamp(0, CODEBOOK_SIZE - 1)
    l3 = l3.reshape(B * beam, CODEBOOK_SIZE).masked_fill(~mask_d1d2_d3[d1cb, d2cb], float("-inf")).view(B, beam, CODEBOOK_SIZE)
    d3lp = F.log_softmax(l3, -1)
    sc3 = b2s.unsqueeze(-1) + d3lp
    b3s, b3i = torch.topk(sc3.masked_fill(torch.isnan(sc3), float("-inf")).view(B, -1), beam, 1)
    sb3 = b3i // CODEBOOK_SIZE; sd3 = b3i % CODEBOOK_SIZE
    d1t = torch.gather(d1t, 1, sb3); d2t = torch.gather(d2t, 1, sb3); d3t = sd3 + OFF[2]
    state = state.view(B, beam, -1).gather(1, sb3.unsqueeze(-1).expand(-1, -1, state.shape[-1])).reshape(B * beam, -1)
    e = model.t5.shared(d3t.reshape(-1)).float(); state = model.gru(e, state)
    # digit 4
    h4 = hphB[:, :, 3].reshape(B * beam, -1)
    l4 = model.heads[3](torch.cat([h4, state], -1)).view(B * beam, CODEBOOK_SIZE)
    key3 = triplet_key_int(d1t.reshape(-1) - OFF[0], d2t.reshape(-1) - OFF[1], d3t.reshape(-1) - OFF[2])
    l4 = l4.masked_fill(~gather_d4_mask(d4k, d4m, key3), float("-inf"))
    d4lp = F.log_softmax(l4, -1)
    t4v, t4i = torch.topk(d4lp, CODEBOOK_SIZE, 1)
    d4t = t4i + OFF[3]
    quad = b3s.unsqueeze(-1) + t4v.view(B, beam, CODEBOOK_SIZE)
    tqs, tqi = torch.topk(quad.masked_fill(torch.isnan(quad), float("-inf")).view(B, -1), beam, 1)
    bi = tqi // CODEBOOK_SIZE
    d1f = torch.gather(d1t, 1, bi); d2f = torch.gather(d2t, 1, bi)
    d3f = torch.gather(d3t, 1, bi); d4f = torch.gather(d4t.view(B, -1), 1, tqi)
    return torch.stack([d1f, d2f, d3f, d4f], -1)


def build_from_ckpt(args, pipe):
    t5 = pipe.model.to(DEV).t5
    ck = torch.load(args.lens_path, map_location="cpu")
    cfg = ck["cfg"]
    m = NEZHA(t5, d_model=cfg["d_model"], state_dim=cfg["state_dim"], head_hidden=cfg["head_hidden"])
    m.load_state_dict(ck["model_state_dict"]); return m.to(DEV).eval()


def evaluate(args):
    args.encoder_ckpt = None; args.fp32 = False
    configure_precision(args, DEV)
    pipe = build_tiger_pipeline(args)
    model = build_from_ckpt(args, pipe)
    vs = find_valid_set(args.dataset_tag, explicit_path=None, lens_path=args.lens_path, graph_dir=args.graph_dir)
    (m12, m123, d4k, d4m, k4, v4, _im, vi4) = build_valid_tensors(vs, DEV)
    masks = (m12, m123, d4k, d4m)
    ds = pipe.tokenized_datasets["test"]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=pipe.tokenizer.collate_fn["test"])
    rec = defaultdict(int); ndcg = defaultdict(float); tot = 0
    beam_t = 0.0; beam_n = 0; torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    with torch.inference_mode():
        for bi, b in enumerate(loader):
            if args.max_batches and bi >= args.max_batches:
                break
            ii = b["input_ids"].to(DEV); am = b["attention_mask"].to(DEV)
            torch.cuda.synchronize(); _ts = time.time()
            quad = nezha_beam(model, ii, am, args.beam_size, masks)
            torch.cuda.synchronize()
            if bi >= 3:
                beam_t += time.time() - _ts; beam_n += ii.shape[0]
            codes = quad.detach().cpu().tolist()
            labs = b["labels"].tolist()
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
    import json
    out = {"tag": args.tag, "category": args.category, "n": tot,
           "recall5": rec[5]/max(tot, 1), "ndcg5": ndcg[5]/max(tot, 1),
           "recall10": rec[10]/max(tot, 1), "ndcg10": ndcg[10]/max(tot, 1),
           "samples_per_s": (beam_n/beam_t) if beam_t > 0 else 0.0,
           "peak_mem_gb": torch.cuda.max_memory_allocated()/1e9}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("NEZHA_EVAL", json.dumps(out), flush=True)


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
        p.add_argument("--state_dim", type=int, default=128)
        p.add_argument("--head_hidden", type=int, default=256)
        p.add_argument("--tag", default="")
        if name == "train":
            p.add_argument("--epochs", type=int, default=60)
            p.add_argument("--batch_size", type=int, default=256)
            p.add_argument("--lr", type=float, default=3e-4)
            p.add_argument("--weight_decay", type=float, default=1e-4)
            p.add_argument("--dropout", type=float, default=0.1)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--out_ckpt", required=True)
            p.add_argument("--test", action="store_true")
        else:
            p.add_argument("--lens_path", required=True)
            p.add_argument("--batch_size", type=int, default=32)
            p.add_argument("--beam_size", type=int, default=50)
            p.add_argument("--final_top_n", type=int, default=50)
            p.add_argument("--max_batches", type=int, default=0)
            p.add_argument("--out", required=True)
    args = ap.parse_args()
    train(args) if args.mode == "train" else evaluate(args)


if __name__ == "__main__":
    main()
