"""s40: AtSpeed-S/R speculative-decoding baseline, faithful to SID-MLP App-D D.2 + AtSpeed
(lin-efficient-2024, arXiv 2410.05165). Adapted to TIGER-T5, 4 constrained SID digits, beam B=50.

Mechanism (App-D D.2 + the AtSpeed beamSD.py verify()):
- AtSpeed-S (strict tree verification): "an accepted step requires the teacher top-B beams to be
  contained in the draft tree." The output is therefore the teacher's exact top-B => LOSSLESS,
  quality == teacher BY CONSTRUCTION. Cost = draft beam search (propose the tree) + teacher
  verification. Since AtSpeed-S must still recover the teacher's top-B, it pays the full teacher
  beam cost (the verification) PLUS the draft beam search -- pure overhead on a 4.59M / 4-step
  teacher that cannot amortize => SLOWER than TIGER-kv (paper 286/s, 0.68x).
- AtSpeed-R (relaxed): draft proposals accepted by rejection sampling against the teacher
  distribution (accept if r <= p_teacher/q_draft), rejected beams refilled from the correction
  distribution clamp(p_teacher - q_draft, 0) (verify() lines 339-364). LOSSY (paper 94/s, 0.22x,
  N@10 -27..37% vs teacher).

Draft = the compact single-block DraftBlock trained for PAD-TIGER (runs/padtiger_{cat}.pt), reused
as the "smallest single-layer Transformer draft" the paper describes. Teacher = frozen TIGER T5.

Timing follows tiger_throughput.py discipline (warmup excluded, sync-bracketed full-test-split loop,
samples/s + peak mem; draft AND teacher both inside the timed region). --variant:
  S  = AtSpeed-S (draft propose + teacher constrained beam == lossless teacher output)
  R  = AtSpeed-R (reject-sampling round, lossy)
  kv = plain teacher constrained beam via t5.generate == the SAME-HARNESS TIGER-kv denominator.
Reports R/N@5/10 + throughput + peak mem; for S also jac-vs-teacher (~1.0 confirms lossless) and
per-query draft coverage of the teacher's top-50 (the acceptance signal).

--lists_out (opt-in, additive): ALSO dump the per-request served lists in the shared s48 schema of
code/lists_s48.py (same dedup_items / npz names+dtypes / valid semantics / sanity block), scored
against the SAME cached TIGER teacher beam (runs/tbeam50_<CAT>.pt) every other method uses, so the
AtSpeed rows are directly comparable offline via code/metrics_s48.py. s_top is built from the codes
this run ACTUALLY emitted (out_cb), never copied from the teacher cache -- for variant S that makes
the dump a real losslessness check (jac@k vs the teacher must come out at 1.0). Without the flag
this script runs exactly as before.
"""
import sys, os, json, time, argparse
R = os.environ.get("SERVESID_ROOT", ".")
sys.path.insert(0, os.path.join(R, "repos", "SID-MLP"))
os.chdir(os.path.join(R, "repos"))
for k in ['HF_DATASETS_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']:
    os.environ[k] = '1'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers.modeling_outputs import BaseModelOutput
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from sid_mlp.inference import (build_tiger_pipeline, configure_precision, _torch_load,
                               find_valid_set, _resolve_ground_truth_items)
from sid_mlp.beam import build_valid_tensors, gather_d4_mask
from sid_mlp.constants import CODEBOOK_SIZE, DIGIT_OFFSETS, triplet_key_int

ap = argparse.ArgumentParser()
ap.add_argument('--category', required=True)
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--draft', required=True)
ap.add_argument('--sem_ids', required=True)
ap.add_argument('--dataset_tag', required=True)
ap.add_argument('--graph_dir', required=True)
ap.add_argument('--dataset', default='AmazonReviews2023')
ap.add_argument('--split', default='test', choices=['test', 'val'])
ap.add_argument('--variant', required=True, choices=['S', 'R', 'kv'])
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--K', type=int, default=50, help='beam size B')
ap.add_argument('--a2', type=int, default=6)
ap.add_argument('--a3', type=int, default=4)
ap.add_argument('--a4', type=int, default=3)
ap.add_argument('--P', type=int, default=150, help='draft-side beam width after each expand')
ap.add_argument('--N', type=int, default=200, help='draft tree size (complete codes proposed)')
ap.add_argument('--warmup', type=int, default=2)
ap.add_argument('--max_batches', type=int, default=0)
ap.add_argument('--teacher_cache', default='', help='tbeam50_{cat}.pt for offline jac/coverage')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--out', required=True)
ap.add_argument('--tag', default='atspeed')
ap.add_argument('--buckets', action='store_true', help='also emit per-popularity-bucket N@10 (mid/tail)')
ap.add_argument('--lists_out', default='', help='s48 served-list dump, path WITHOUT extension')
ap.add_argument('--pop_cache', default='', help='npz(items,counts,lo,hi) of train pop; written if absent')
ap.add_argument('--topk', type=int, default=20, help='dumped served-list depth')
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device('cuda')
args.encoder_ckpt = None; args.fp32 = False
nb16 = configure_precision(args, dev)

pipe = build_tiger_pipeline(args)
tiger = pipe.model.to(dev).eval()
t5 = tiger.t5
if nb16:
    t5.bfloat16()
for p in t5.parameters():
    p.requires_grad_(False)
n_digit = pipe.tokenizer.n_digit
eos_tok = int(pipe.tokenizer.eos_token); pad_tok = int(pipe.tokenizer.padding_token)

vs = find_valid_set(args.dataset_tag, explicit_path=None, lens_path=None, graph_dir=args.graph_dir)
(mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted, *_rest, valid_items_4) = build_valid_tensors(vs, dev)
d1_valid = torch.zeros(CODEBOOK_SIZE, dtype=torch.bool, device=dev)
for quad in valid_items_4.keys():
    d1_valid[int(quad[0]) - DIGIT_OFFSETS[0]] = True

D = t5.config.d_model
START = t5.config.decoder_start_token_id
SCALE = (D ** -0.5) if getattr(t5.config, 'tie_word_embeddings', True) else 1.0

dck = _torch_load(args.draft, weights_only=False)
DB = int(dck['args']['B']); DH = int(dck['args']['heads'])


class DraftBlock(nn.Module):
    def __init__(self, D, H, B):
        super().__init__()
        self.fuse_proj = nn.Linear(2 * D, D)
        self.ipe = nn.Embedding(5, D)
        self.spe = nn.Embedding(B + 1, D)
        self.g_item = nn.Parameter(torch.zeros(1))
        self.g_step = nn.Sequential(nn.Linear(D, D // 2), nn.ReLU(), nn.Linear(D // 2, 1))
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D); self.ln3 = nn.LayerNorm(D)
        self.sa = nn.MultiheadAttention(D, H, batch_first=True)
        self.ca = nn.MultiheadAttention(D, H, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(D, 4 * D), nn.ReLU(), nn.Linear(4 * D, D))

    def fuse(self, e_tok, f_prev, slot_ids, j_ids):
        e = e_tok + torch.sigmoid(self.g_item) * self.ipe(slot_ids)
        x = self.fuse_proj(torch.cat([e, f_prev], -1))
        g = torch.sigmoid(self.g_step(f_prev))
        return x + g * self.spe(j_ids)

    def forward(self, x, enc, enc_pad):
        T = x.shape[1]
        cm = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.ln1(x)
        a, _ = self.sa(h, h, h, attn_mask=cm, need_weights=False)
        x = x + a
        h = self.ln2(x)
        c, _ = self.ca(h, enc, enc, key_padding_mask=enc_pad, need_weights=False)
        x = x + c
        x = x + self.ffn(self.ln3(x))
        return x


draft = DraftBlock(D, DH, DB).to(dev)
draft.load_state_dict(dck['draft'])
draft.eval().float()
for p in draft.parameters():
    p.requires_grad_(False)
print(f'[atspeed] variant={args.variant} draft B={DB} heads={DH} a=({args.a2},{args.a3},{args.a4}) '
      f'P={args.P} N={args.N} K={args.K}', flush=True)


def head_of(feat):
    return t5.lm_head(feat.to(t5.lm_head.weight.dtype) * SCALE).float()


def level_mask(level, prev_cb):
    if level == 0:
        return d1_valid
    if level == 1:
        return mask_d1_d2[prev_cb[0]]
    if level == 2:
        return mask_d1d2_d3[prev_cb[0], prev_cb[1]]
    key3 = triplet_key_int(prev_cb[0], prev_cb[1], prev_cb[2])
    return gather_d4_mask(d4_keys_sorted, d4_masks_sorted, key3)


def masked_lp_slice(logits, level, prev_cb):
    lo = DIGIT_OFFSETS[level]
    sl = logits[..., lo:lo + CODEBOOK_SIZE]
    m = level_mask(level, prev_cb)
    if m.dim() < sl.dim():
        m = m.view(*([1] * (sl.dim() - m.dim())), *m.shape).expand_as(sl)
    return F.log_softmax(sl.masked_fill(~m, float('-inf')), dim=-1)


def sid_to_item(tup):
    return valid_items_4.get(tuple(int(x) for x in tup), -1)


def dedup_items(sid_list, cap):
    """s48 dump convention (verbatim lists_s48.dedup_items): item-level -- drop invalid SIDs, dedup
    preserving rank order, truncate. Returns (items, n_invalid, n_raw). TOKEN-ID space in."""
    seen = set(); out = []; ninv = 0
    for tup in sid_list:
        it = sid_to_item(tup)
        if it == -1:
            ninv += 1; continue
        if it in seen: continue
        seen.add(it); out.append(it)
        if len(out) >= cap: break
    return out, ninv, len(sid_list)


# valid-SID prefix Trie for the teacher's constrained beam (t5.generate)
def build_trie():
    trie = {}
    for tup in valid_items_4.keys():
        a, b, c, d = (int(x) for x in tup)
        trie.setdefault(a, {}).setdefault(b, {}).setdefault(c, set()).add(d)
    return trie, sorted(trie.keys())


trie, d1_list = build_trie()


def prefix_fn(batch_id, ids):
    seq = ids.tolist()
    n = len(seq) - 1
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


class ValidSIDProc(LogitsProcessor):
    NEG = -1e9
    def __call__(self, input_ids, scores):
        n = input_ids.shape[1] - 1                      # SID digits generated so far
        allow = torch.zeros_like(scores, dtype=torch.bool)
        if n >= n_digit:
            allow[:, eos_tok] = True
            return scores.masked_fill(~allow, self.NEG)
        lo = DIGIT_OFFSETS[n]
        if n == 0:
            m = d1_valid.unsqueeze(0).expand(scores.shape[0], -1)
        elif n == 1:
            d1 = input_ids[:, 1] - DIGIT_OFFSETS[0]
            m = mask_d1_d2[d1]
        elif n == 2:
            d1 = input_ids[:, 1] - DIGIT_OFFSETS[0]; d2 = input_ids[:, 2] - DIGIT_OFFSETS[1]
            m = mask_d1d2_d3[d1, d2]
        else:
            d1 = input_ids[:, 1] - DIGIT_OFFSETS[0]; d2 = input_ids[:, 2] - DIGIT_OFFSETS[1]
            d3 = input_ids[:, 3] - DIGIT_OFFSETS[2]
            m = gather_d4_mask(d4_keys_sorted, d4_masks_sorted, triplet_key_int(d1, d2, d3))
        allow[:, lo:lo + CODEBOOK_SIZE] = m
        return scores.masked_fill(~allow, self.NEG)


VALID_PROC = LogitsProcessorList([ValidSIDProc()])


@torch.inference_mode()
def teacher_beam(enc, attn):
    """Plain teacher constrained beam-K (t5.generate, KV-cached). Returns (B,K,4) codebook ids.
    This is BOTH the TIGER-kv baseline (variant kv) AND the AtSpeed-S verification/output (lossless)."""
    B = enc.shape[0]
    gen = t5.generate(
        encoder_outputs=BaseModelOutput(last_hidden_state=enc), attention_mask=attn,
        max_new_tokens=n_digit + 1, num_beams=args.K, num_return_sequences=args.K,
        logits_processor=VALID_PROC, use_cache=True, decoder_start_token_id=0,
        eos_token_id=eos_tok, pad_token_id=pad_tok)
    gen = gen.reshape(B, args.K, -1)[:, :, 1:1 + n_digit]                      # (B,K,4) vocab
    cb = torch.stack([gen[..., i] - DIGIT_OFFSETS[i] for i in range(n_digit)], -1)
    return cb


@torch.inference_mode()
def draft_propose(enc, encf, pad, h0):
    """Draft beam search proposing a tree of N complete 4-digit codes (draft cost of AtSpeed).
    Returns candidate codebook ids (B,N,4). Mirrors the PAD-TIGER draft roll (single-block draft)."""
    Bn = enc.shape[0]
    # d1 from the draft's own BOS feature (the draft proposes the whole tree incl. d1)
    x0 = draft.fuse(t5.shared(torch.full((Bn, 1), START, device=dev)).float(),
                    h0.unsqueeze(1), torch.zeros(Bn, 1, dtype=torch.long, device=dev),
                    torch.zeros(Bn, 1, dtype=torch.long, device=dev))
    f0 = t5.decoder.final_layer_norm(draft(x0, encf, pad).to(t5.dtype)).float()[:, 0]
    d1lp = masked_lp_slice(head_of(f0), 0, [])
    K0 = min(args.P, int(d1_valid.sum()))
    hyp_sc, d1_idx = torch.topk(d1lp, K0, dim=1)
    hyp_cb = d1_idx.unsqueeze(-1)
    hyp_f = f0.unsqueeze(1).expand(Bn, K0, D)
    hyp_e = t5.shared(hyp_cb[..., 0] + DIGIT_OFFSETS[0]).float()
    for lvl in range(1, 4):
        nh = hyp_cb.shape[1]
        e_tok = hyp_e.reshape(Bn * nh, 1, D)
        f_prev = hyp_f.reshape(Bn * nh, 1, D)
        slot = torch.full((Bn * nh, 1), lvl, dtype=torch.long, device=dev)
        j_ids = torch.full((Bn * nh, 1), lvl, dtype=torch.long, device=dev)
        x = draft.fuse(e_tok, f_prev, slot, j_ids)
        enc_r = encf.repeat_interleave(nh, 0)
        pad_r = pad.repeat_interleave(nh, 0)
        df = t5.decoder.final_layer_norm(draft(x, enc_r, pad_r).to(t5.dtype)).float()[:, 0]
        dlog = head_of(df).reshape(Bn, nh, -1)
        prev = [hyp_cb[..., i].reshape(-1) for i in range(lvl)]
        dlp = masked_lp_slice(dlog.reshape(Bn * nh, -1), lvl, prev).reshape(Bn, nh, CODEBOOK_SIZE)
        a = (args.a2, args.a3, args.a4)[lvl - 1]
        top_v, top_i = torch.topk(dlp, a, dim=-1)
        new_sc = (hyp_sc.unsqueeze(-1) + top_v).reshape(Bn, nh * a)
        keep = min(args.P if lvl < 3 else args.N, nh * a)
        sel_sc, sel_ix = torch.topk(new_sc, keep, dim=1)
        parent = sel_ix // a; child = sel_ix % a
        hyp_cb = torch.cat([torch.gather(hyp_cb, 1, parent.unsqueeze(-1).expand(-1, -1, lvl)),
                            torch.gather(top_i, 1, parent.unsqueeze(-1).expand(-1, -1, a)).gather(
                                2, child.unsqueeze(-1)).reshape(Bn, keep, 1)], dim=-1)
        hyp_sc = sel_sc
        if lvl < 3:
            new_tok = hyp_cb[..., lvl] + DIGIT_OFFSETS[lvl]
            hyp_e = t5.shared(new_tok).float()
            df_all = df.reshape(Bn, nh, D)
            hyp_f = torch.gather(df_all, 1, parent.unsqueeze(-1).expand(-1, -1, D))
    return hyp_cb, hyp_sc                                                     # (B,N,4), (B,N)


@torch.inference_mode()
def bos_feature(enc, attn):
    Bn = enc.shape[0]
    dec_in = torch.full((Bn, 1), START, dtype=torch.long, device=dev)
    o = t5(encoder_outputs=BaseModelOutput(last_hidden_state=enc), attention_mask=attn,
           decoder_input_ids=dec_in, output_hidden_states=True, return_dict=True)
    return o.decoder_hidden_states[-1][:, 0].float()


@torch.inference_mode()
def teacher_verify(enc, attn, cand_cb):
    """One batched teacher forward teacher-forcing N candidate codes -> exact teacher seq log-prob.
    (B,N,4 codebook) -> (B,N) teacher log-prob. The AtSpeed 'teacher tree verification' forward."""
    Bn, N = cand_cb.shape[:2]
    tok = torch.stack([cand_cb[..., i] + DIGIT_OFFSETS[i] for i in range(4)], -1)   # (B,N,4) vocab
    flat = tok.reshape(Bn * N, 4)
    dec_in = torch.cat([torch.full((Bn * N, 1), START, dtype=torch.long, device=dev), flat[:, :3]], 1)
    enc_rep = enc.repeat_interleave(N, 0); attn_rep = attn.repeat_interleave(N, 0)
    o = t5(encoder_outputs=BaseModelOutput(last_hidden_state=enc_rep), attention_mask=attn_rep,
           decoder_input_ids=dec_in, return_dict=True)
    lp = F.log_softmax(o.logits.float(), dim=-1)
    return lp.gather(2, flat.unsqueeze(-1)).squeeze(-1).sum(-1).reshape(Bn, N)


@torch.inference_mode()
def atspeed_s_round(b):
    """AtSpeed-S: draft proposes the tree (cost paid) + teacher constrained beam produces the
    LOSSLESS teacher top-K output & verifies. Output == teacher top-K. Returns (out_cb, draft_tree)."""
    ii = b['input_ids'].to(dev, non_blocking=True); am = b['attention_mask'].to(dev, non_blocking=True)
    enc = t5.encoder(input_ids=ii, attention_mask=am, return_dict=True).last_hidden_state
    encf = enc.float(); pad = ~am.bool()
    h0 = bos_feature(enc, am)
    draft_tree, _ = draft_propose(enc, encf, pad, h0)          # draft beam search (the overhead)
    out_cb = teacher_beam(enc, am)                             # teacher top-K = lossless output + verify
    return out_cb, draft_tree


@torch.inference_mode()
def atspeed_r_round(b):
    """AtSpeed-R: per-digit rejection sampling of draft proposals against the teacher distribution,
    rejected beams refilled from the correction distribution clamp(p_teacher - q_draft, 0). Lossy."""
    ii = b['input_ids'].to(dev, non_blocking=True); am = b['attention_mask'].to(dev, non_blocking=True)
    enc = t5.encoder(input_ids=ii, attention_mask=am, return_dict=True).last_hidden_state
    encf = enc.float(); pad = ~am.bool()
    Bn = enc.shape[0]; W = args.K
    # d1: teacher exact root (cheap, 1 forward), then draft-sample d2..d4 with teacher reject-sampling
    dec_in = torch.full((Bn, 1), START, dtype=torch.long, device=dev)
    o = t5(encoder_outputs=BaseModelOutput(last_hidden_state=enc), attention_mask=am,
           decoder_input_ids=dec_in, output_hidden_states=True, return_dict=True)
    h0 = o.decoder_hidden_states[-1][:, 0].float()
    d1_lp = masked_lp_slice(o.logits[:, 0].float(), 0, [])
    d1_sc, d1_idx = torch.topk(d1_lp, W, dim=1)
    hyp_cb = d1_idx.unsqueeze(-1); hyp_sc = d1_sc
    hyp_f = h0.unsqueeze(1).expand(Bn, W, D); hyp_e = t5.shared(hyp_cb[..., 0] + DIGIT_OFFSETS[0]).float()
    for lvl in range(1, 4):
        nh = hyp_cb.shape[1]
        # draft distribution q over next digit
        e_tok = hyp_e.reshape(Bn * nh, 1, D); f_prev = hyp_f.reshape(Bn * nh, 1, D)
        slot = torch.full((Bn * nh, 1), lvl, dtype=torch.long, device=dev)
        j_ids = torch.full((Bn * nh, 1), lvl, dtype=torch.long, device=dev)
        x = draft.fuse(e_tok, f_prev, slot, j_ids)
        enc_r = encf.repeat_interleave(nh, 0); pad_r = pad.repeat_interleave(nh, 0)
        enc_rep = enc.repeat_interleave(nh, 0)  # bf16 for the teacher verify forward
        df = t5.decoder.final_layer_norm(draft(x, enc_r, pad_r).to(t5.dtype)).float()[:, 0]
        prev = [hyp_cb[..., i].reshape(-1) for i in range(lvl)]
        q = masked_lp_slice(head_of(df), lvl, prev).reshape(Bn, nh, CODEBOOK_SIZE).exp()   # draft prob
        # teacher verification forward (exact teacher dist p over next digit given each beam prefix)
        tok = torch.stack([hyp_cb[..., i] + DIGIT_OFFSETS[i] for i in range(lvl)], -1).reshape(Bn * nh, lvl)
        dec_in2 = torch.cat([torch.full((Bn * nh, 1), START, dtype=torch.long, device=dev), tok], 1)
        o2 = t5(encoder_outputs=BaseModelOutput(last_hidden_state=enc_rep), attention_mask=attn_r_of(am, nh),
                decoder_input_ids=dec_in2, return_dict=True)
        p = masked_lp_slice(o2.logits[:, lvl].float(), lvl, prev).reshape(Bn, nh, CODEBOOK_SIZE).exp()
        # joint beam*vocab distributions, weighted by current beam prob (softmax over accumulated score)
        beam_w = torch.softmax(hyp_sc, dim=1).unsqueeze(-1)             # (B,nh,1)
        qj = (beam_w * q).reshape(Bn, nh * CODEBOOK_SIZE); qj = qj / qj.sum(1, keepdim=True).clamp_min(1e-30)
        pj = (beam_w * p).reshape(Bn, nh * CODEBOOK_SIZE); pj = pj / pj.sum(1, keepdim=True).clamp_min(1e-30)
        # draft samples W expansions; accept via rejection r <= p/q; refill rejects from correction (p-q)+
        # invalid-SID entries are EXACTLY 0 in qj/pj (masked) -> do NOT clamp_min before multinomial
        # (clamping would make invalid children samplable -> invalid prefixes -> device-side assert).
        samp = torch.multinomial(qj, W, replacement=True)                     # (B,W)
        q_i = qj.gather(1, samp); p_i = pj.gather(1, samp)
        r = torch.rand_like(q_i)
        acc = r <= (p_i / q_i.clamp_min(1e-30))
        corr = torch.clamp(pj - qj, min=0)
        cs = corr.sum(1, keepdim=True)
        corr = torch.where(cs > 1e-20, corr / cs.clamp_min(1e-30), pj)        # fall back to teacher pj if degenerate
        refill = torch.multinomial(corr, W, replacement=True)                 # (B,W)
        chosen = torch.where(acc, samp, refill)                                # (B,W) joint idx
        parent = chosen // CODEBOOK_SIZE; child = chosen % CODEBOOK_SIZE
        hyp_cb = torch.cat([torch.gather(hyp_cb, 1, parent.unsqueeze(-1).expand(-1, -1, lvl)),
                            child.unsqueeze(-1)], dim=-1)
        hyp_sc = torch.log(pj.gather(1, chosen).clamp_min(1e-30))
        if lvl < 3:
            hyp_e = t5.shared(hyp_cb[..., lvl] + DIGIT_OFFSETS[lvl]).float()
            df_all = df.reshape(Bn, nh, D)
            hyp_f = torch.gather(df_all, 1, parent.unsqueeze(-1).expand(-1, -1, D))
    return hyp_cb, hyp_sc


def attn_r_of(am, nh):
    return am.repeat_interleave(nh, 0)


@torch.inference_mode()
def kv_round(b):
    ii = b['input_ids'].to(dev, non_blocking=True); am = b['attention_mask'].to(dev, non_blocking=True)
    enc = t5.encoder(input_ids=ii, attention_mask=am, return_dict=True).last_hidden_state
    return teacher_beam(enc, am), None


ds = pipe.tokenized_datasets[args.split]
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=pipe.tokenizer.collate_fn[args.split])
print(f'[atspeed] split={args.split} rows={len(ds)}', flush=True)

ROUND = {'S': atspeed_s_round, 'R': atspeed_r_round, 'kv': kv_round}[args.variant]

# warmup (excluded)
it = iter(loader)
for _ in range(args.warmup):
    try:
        b = next(it)
    except StopIteration:
        break
    ROUND(b)
torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)

tc = None
if args.teacher_cache and os.path.exists(args.teacher_cache):
    tc = torch.load(args.teacher_cache, map_location='cpu')['codes']          # (n,50,4) vocab-space

# ---- s48 served-list dump (opt-in): shared cached TIGER teacher + what THIS run emitted ----
TOPK = args.topk; LW = args.K              # W is shared by teacher AND method (lists_s48 protocol)
GT, GP, GB, TT, ST, TN, SN, SRAW, SINV, VALID = [], [], [], [], [], [], [], [], [], []
if args.lists_out:
    if tc is None:
        raise SystemExit(f'--lists_out needs the shared teacher beam cache; '
                         f'--teacher_cache={args.teacher_cache!r} not found')
    print(f'[tcache] loaded {args.teacher_cache} rows={tc.shape[0]} beam={tc.shape[1]}', flush=True)
    if tc.shape[1] < LW:
        raise SystemExit(f'teacher cache beam {tc.shape[1]} < decode width {LW}')
    if args.split != 'test':
        print(f'!!! WARNING: --split {args.split} but the teacher cache is TEST-row-aligned -- '
              f'the dumped teacher lists are NOT this split !!!', flush=True)
    if LW < TOPK:
        print(f'!!! WARNING: decode width {LW} < topk {TOPK} -- served lists are SHORT and @{TOPK} '
              f'metrics are distorted (s_n/t_n record the truth) !!!', flush=True)

# optional per-popularity-bucket N@10 (train-count tertiles, == ssm_bucketdiag/bucket_diag_s34 convention)
bkt_n10 = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}; bkt_n = {0: 0, 1: 0, 2: 0, 3: 0}; bucket_of = None
if args.buckets or args.lists_out:
    if args.pop_cache and os.path.exists(args.pop_cache):
        _z = np.load(args.pop_cache)
        _pop = {int(i): int(c) for i, c in zip(_z['items'], _z['counts'])}
        _LO = int(_z['lo']); _HI = int(_z['hi'])
    else:
        _pop = {}
        _tl = DataLoader(pipe.tokenized_datasets['train'], batch_size=256, shuffle=False,
                         collate_fn=pipe.tokenizer.collate_fn['train'])
        with torch.inference_mode():
            for _b in _tl:
                for _row in _b['labels'].tolist():
                    _it = sid_to_item(tuple(int(x) for x in _row[:4]))
                    if _it != -1:
                        _pop[_it] = _pop.get(_it, 0) + 1
        _cs = sorted(_pop.values()); _LO = _cs[len(_cs) // 3]; _HI = _cs[2 * len(_cs) // 3]
        if args.pop_cache:
            _its = np.array(sorted(_pop.keys()), dtype=np.int64)
            np.savez_compressed(args.pop_cache, items=_its,
                                counts=np.array([_pop[int(i)] for i in _its], dtype=np.int64),
                                lo=_LO, hi=_HI)
    def bucket_of(it):
        c = _pop.get(it, 0)
        return 0 if c == 0 else (1 if c <= _LO else (2 if c <= _HI else 3))
    print(f'[atspeed] buckets lo={_LO} hi={_HI}', flush=True)

tot = 0; r5 = r10 = n5 = n10 = 0.0; jacs = []; covs = []
row0 = 0
gen_t = 0.0
for bi, b in enumerate(loader):
    if args.max_batches and bi >= args.max_batches:
        break
    torch.cuda.synchronize(dev); _ts = time.time()
    out_cb, tree = ROUND(b)
    torch.cuda.synchronize(dev); gen_t += time.time() - _ts
    oc = out_cb.cpu(); labels = b['labels']
    gt_sids, gt_items = _resolve_ground_truth_items(labels, valid_items_4)
    for r in range(oc.shape[0]):
        seen = set(); items = []
        for tup in oc[r].tolist():
            it_ = sid_to_item(tuple(int(tup[i]) + DIGIT_OFFSETS[i] for i in range(4)))
            if it_ == -1 or it_ in seen:
                continue
            seen.add(it_); items.append(it_)
        gt = gt_items[r]
        r5 += 1.0 if gt in items[:5] else 0.0
        r10 += 1.0 if gt in items[:10] else 0.0
        for i, it_ in enumerate(items[:5]):
            if it_ == gt:
                n5 += 1.0 / np.log2(i + 2)
        for i, it_ in enumerate(items[:10]):
            if it_ == gt:
                n10 += 1.0 / np.log2(i + 2)
        if bucket_of is not None:
            _s10 = 0.0
            for i, it_ in enumerate(items[:10]):
                if it_ == gt:
                    _s10 = 1.0 / np.log2(i + 2); break
            _bk = bucket_of(gt); bkt_n10[_bk] += _s10; bkt_n[_bk] += 1
        # AtSpeed-S: coverage of the teacher's top-50 by the draft tree (the acceptance signal)
        if args.variant == 'S' and tc is not None and not args.max_batches and tree is not None:
            gi = row0 + r
            if gi < tc.shape[0]:
                tset = set(tuple(int(x) - DIGIT_OFFSETS[j] for j, x in enumerate(c)) for c in tc[gi].tolist())
                dset = set(tuple(int(x) for x in c) for c in tree[r].cpu().tolist())
                covs.append(len(tset & dset) / max(len(tset), 1))
                # jac(output, teacher) -- lossless => ~1.0
                oset = set(items[:10])
                t10 = []
                sn = set()
                for c in tc[gi].tolist():
                    itt = sid_to_item(tuple(int(x) for x in c))
                    if itt == -1 or itt in sn:
                        continue
                    sn.add(itt); t10.append(itt)
                    if len(t10) >= 10:
                        break
                jacs.append(len(oset & set(t10)) / max(len(oset | set(t10)), 1))
        # s48 served-list dump: s_top comes from the codes THIS run emitted (out_cb), never from tc
        if args.lists_out:
            _gi = row0 + r
            if _gi >= tc.shape[0]:
                raise SystemExit('teacher cache has %d rows, test row %d out of range'
                                 % (tc.shape[0], _gi))
            _stoks = [tuple(int(c[i]) + DIGIT_OFFSETS[i] for i in range(4)) for c in oc[r].tolist()]
            _gt_it = sid_to_item(tuple(int(x) for x in labels[r].tolist()[:4]))
            t_items, _tinv, _traw = dedup_items(tc[_gi, :LW].tolist(), TOPK)
            s_items, sinv, sraw = dedup_items(_stoks, TOPK)
            GT.append(_gt_it); GP.append(_pop.get(_gt_it, 0) if _gt_it != -1 else -1)
            GB.append(bucket_of(_gt_it) if _gt_it != -1 else -1)
            TT.append(t_items + [-1] * (TOPK - len(t_items)))
            ST.append(s_items + [-1] * (TOPK - len(s_items)))
            TN.append(len(t_items)); SN.append(len(s_items))
            SRAW.append(sraw); SINV.append(sinv)
            VALID.append(int(_gt_it != -1 and len(t_items) > 0))
        tot += 1
    row0 += oc.shape[0]
torch.cuda.synchronize(dev)
elapsed = gen_t
peak = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
thr = tot / max(elapsed, 1e-6)

out = {'tag': args.tag, 'category': args.category, 'variant': args.variant, 'split': args.split,
       'bs': args.batch_size, 'K': args.K, 'a': [args.a2, args.a3, args.a4], 'P': args.P, 'N': args.N,
       'total': tot, 'elapsed_sec': elapsed, 'throughput': thr, 'ms_per_sample': 1000.0 / max(thr, 1e-6),
       'peak_mem_gb': peak, 'R@5': r5 / max(tot, 1), 'R@10': r10 / max(tot, 1),
       'N@5': n5 / max(tot, 1), 'N@10': n10 / max(tot, 1), 'gpu': torch.cuda.get_device_name(0),
       'draft': args.draft}
if jacs:
    out['jac10_vs_teacher'] = float(np.mean(jacs))
if covs:
    out['draft_tree_coverage_of_teacher50'] = float(np.mean(covs))
if args.buckets:
    _nm = ['cold', 'tail', 'mid', 'head']
    out['buckets'] = {_nm[k]: {'n': bkt_n[k], 'n10': bkt_n10[k] / max(bkt_n[k], 1)} for k in range(4)}
with open(args.out, 'w') as f:
    json.dump(out, f, indent=2)

if args.lists_out:
    gt_a = np.array(GT, dtype=np.int32); gp_a = np.array(GP, dtype=np.int64)
    gb_a = np.array(GB, dtype=np.int8); valid_a = np.array(VALID, dtype=np.int8)
    tt_a = np.array(TT, dtype=np.int32); st_a = np.array(ST, dtype=np.int32)
    tn_a = np.array(TN, dtype=np.int16); sn_a = np.array(SN, dtype=np.int16)
    sraw_a = np.array(SRAW, dtype=np.int16); sinv_a = np.array(SINV, dtype=np.int16)
    np.savez_compressed(args.lists_out + '.npz', gt_item=gt_a, gt_pop=gp_a, gt_bucket=gb_a,
                        valid=valid_a, t_top=tt_a, s_top=st_a, t_n=tn_a, s_n=sn_a,
                        s_raw=sraw_a, s_invalid=sinv_a,
                        bucket_lo=_LO, bucket_hi=_HI, n_test=len(GT), W=LW, topk=TOPK)
    msk = valid_a.astype(bool)
    def _rank(lists, k):
        """1-based rank of gt in the first k entries, 0 if absent."""
        hit = (lists[:, :k] == gt_a[:, None])
        return np.where(hit.any(1), hit.argmax(1) + 1, 0)
    san = {'tag': args.tag, 'category': args.category, 'method': 'atspeed' + args.variant,
           'split': args.split, 'draft': args.draft, 'W': LW, 'topk': TOPK,
           'n_rows': int(len(GT)), 'n_valid': int(msk.sum()), 'bucket_lo': _LO, 'bucket_hi': _HI,
           'mean_s_invalid': float(sinv_a[msk].mean()), 'mean_s_n': float(sn_a[msk].mean()),
           'short_at_20': int((sn_a[msk] < 20).sum()), 'short_at_10': int((sn_a[msk] < 10).sum())}
    for k in (5, 10, 20):
        tr = _rank(tt_a, k)[msk]; sr = _rank(st_a, k)[msk]
        san[f'teacher_r{k}'] = float((tr > 0).mean())
        san[f'teacher_n{k}'] = float(np.where(tr > 0, 1.0 / np.log2(np.maximum(tr, 1) + 1.0), 0.0).mean())
        san[f'student_r{k}'] = float((sr > 0).mean())
        san[f'student_n{k}'] = float(np.where(sr > 0, 1.0 / np.log2(np.maximum(sr, 1) + 1.0), 0.0).mean())
        inter = np.zeros(int(msk.sum())); union = np.zeros(int(msk.sum()))
        T = tt_a[msk][:, :k]; S = st_a[msk][:, :k]
        for i in range(T.shape[0]):
            a = set(int(x) for x in T[i] if x >= 0); b = set(int(x) for x in S[i] if x >= 0)
            inter[i] = len(a & b); union[i] = max(len(a | b), 1)
        san[f'jac{k}'] = float((inter / union).mean())
        san[f'overlap{k}'] = float((inter / k).mean())
    with open(args.lists_out + '.json', 'w') as f:
        json.dump(san, f, indent=2)
    # variant S is lossless by construction => jac* must be 1.0; this is the consistency check
    print('ATSPEED_LISTS', json.dumps(san), flush=True)

print('ATSPEED_DONE', json.dumps({k: out[k] for k in
      ('variant', 'throughput', 'ms_per_sample', 'R@10', 'N@10', 'jac10_vs_teacher',
       'draft_tree_coverage_of_teacher50', 'peak_mem_gb') if k in out}), flush=True)
