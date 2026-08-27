# s41 CDRS: augment an existing teacher beam dump with PER-STEP teacher log-probs.
# Reuses the dump's already-sorted candidate codes (so codes/scores/rank identity are preserved
# EXACTLY, delivery §5.2) and re-runs ONLY the teacher T5 rescore, keeping the per-digit log-probs
# (the original dump summed them away). Precision matches the original dump: bf16 T5 forward +
# fp32 log_softmax (so per-step logp is on the same footing as the stored served-list scores).
# Adds: step_logp[N,W,4] (per-digit logp), eos_logp[N,W], score_scale[4] (std of teacher active-pair
# cumulative gaps per depth, delivery §3.3). Runs the §5 data-validation checks.
import sys, os, time, argparse
import torch, torch.nn.functional as F
R = os.environ.get("SERVESID_ROOT", ".")
sys.path.insert(0, os.path.join(R, "repos", "SID-MLP"))
os.chdir(os.path.join(R, "repos"))
for k in ['HF_DATASETS_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']:
    os.environ[k] = '1'
from transformers.modeling_outputs import BaseModelOutput
from sid_mlp.inference import build_tiger_pipeline, configure_precision
from sid_mlp.constants import CODEBOOK_SIZE, DIGIT_OFFSETS

ap = argparse.ArgumentParser()
ap.add_argument('--category', default='Industrial_and_Scientific')
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--sem_ids', required=True)
ap.add_argument('--dataset_tag', required=True)
ap.add_argument('--graph_dir', required=True)
ap.add_argument('--dataset', default='AmazonReviews2023')
ap.add_argument('--in_dump', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--batch_size', type=int, default=256)
ap.add_argument('--chunk', type=int, default=8192)
ap.add_argument('--n_pos', type=int, default=10)
ap.add_argument('--n_neg_hi', type=int, default=40)
args = ap.parse_args()
args.encoder_ckpt = None; args.fp32 = False; args.lens_path = None
dev = torch.device('cuda')
nb16 = configure_precision(args, dev)   # bf16 (matches original dump meta bf16=True)

pipe = build_tiger_pipeline(args)
t5 = pipe.model.t5.to(dev).eval()
if nb16: t5.bfloat16()
for p in t5.parameters(): p.requires_grad_(False)
EOS = t5.config.eos_token_id; START = t5.config.decoder_start_token_id
print(f'[aug] t5 ready bf16={nb16} EOS={EOS} START={START}', flush=True)

D = torch.load(args.in_dump, map_location='cpu', weights_only=False)
codes = D['codes']       # (N,W,4) uint8 codebook-space, teacher-score-sorted
scores = D['scores']     # (N,W) fp16 exact total (sum digit logp + eos)
nvalid = D['nvalid']     # (N,)
N, W, _ = codes.shape
print(f'[aug] dump N={N} W={W} nvalid mean={float(nvalid.float().mean()):.1f} min={int(nvalid.min())}', flush=True)

cache_file = os.path.join(args.graph_dir, 'hidden_states', args.dataset_tag, 'packed_cache', 'k0_fp32.pt')
assert os.path.exists(cache_file), cache_file
cached = torch.load(cache_file, map_location='cpu', weights_only=False, mmap=True)
enc_all = cached['enc_seq']      # (N,S,128) cpu fp32
msk_all = cached['enc_mask']     # (N,S) bool True=PAD
assert enc_all.shape[0] == N, (enc_all.shape[0], N)

OFF = torch.tensor(DIGIT_OFFSETS, dtype=torch.long, device=dev)
step_out = torch.zeros(N, W, 4, dtype=torch.float32)   # fp32 in-memory for tight checksum; stored as fp16
eos_out = torch.zeros(N, W, dtype=torch.float32)

@torch.inference_mode()
def teacher_steplp(enc_b, attn, cand_user, cand_tokens, chunk):
    Nc = cand_tokens.shape[0]
    step = torch.empty(Nc, 4, dtype=torch.float32, device=enc_b.device)
    eos = torch.empty(Nc, dtype=torch.float32, device=enc_b.device)
    dec_in = torch.cat([torch.full((Nc, 1), START, dtype=torch.long, device=enc_b.device), cand_tokens], 1)
    for lo in range(0, Nc, chunk):
        hi = min(Nc, lo + chunk); rows = cand_user[lo:hi]
        eo = BaseModelOutput(last_hidden_state=enc_b.index_select(0, rows))
        am = attn.index_select(0, rows)
        out = t5(encoder_outputs=eo, attention_mask=am, decoder_input_ids=dec_in[lo:hi], return_dict=True)
        lp = F.log_softmax(out.logits.float(), dim=-1)                 # fp32 log_softmax (matches dump)
        step[lo:hi] = lp[:, :4, :].gather(2, cand_tokens[lo:hi].unsqueeze(-1)).squeeze(-1)
        eos[lo:hi] = lp[:, 4, EOS]
    return step, eos

bs = args.batch_size; t0 = time.time()
for blo in range(0, N, bs):
    bhi = min(N, blo + bs)
    enc = enc_all[blo:bhi].to(dev, non_blocking=True)
    pad = msk_all[blo:bhi].to(dev, non_blocking=True)
    attn = (~pad).long()
    enc_b = enc.bfloat16() if nb16 else enc
    B = enc.shape[0]
    cb = codes[blo:bhi].to(dev).long()                # (B,W,4) codebook 0..255
    tok = cb + OFF.view(1, 1, 4)                       # vocab-space
    cand_tokens = tok.reshape(B * W, 4)
    cand_user = torch.arange(B, device=dev).repeat_interleave(W)
    step, eos = teacher_steplp(enc_b, attn, cand_user, cand_tokens, args.chunk)
    step_out[blo:bhi] = step.view(B, W, 4).cpu()
    eos_out[blo:bhi] = eos.view(B, W).cpu()
    if (blo // bs) % 20 == 0:
        el = time.time() - t0
        print(f'[aug] {bhi}/{N} el {el:.0f}s eta {el / max(bhi,1) * (N - bhi):.0f}s', flush=True)

# ---------------- validation (delivery §5) ----------------
val = torch.arange(W).view(1, W) < nvalid.view(N, 1)          # (N,W) valid mask
recon = step_out.sum(-1) + eos_out                            # 4-digit sum + eos (== dump total defn)
err = (recon - scores.float()).abs()[val]
print(f'[VAL 5.1] recon(step.sum+eos) vs stored total: max {err.max():.2e} mean {err.mean():.2e} '
      f'(stored is fp16; expect ~1e-3)', flush=True)
eosv = eos_out[val].abs()
print(f'[VAL eos] |eos_logp| mean {eosv.mean():.4f} max {eosv.max():.4f} '
      f'(gap between 4-digit cumsum and full total; CDRS uses the 4-digit cumsum, no eos)', flush=True)

# score_scale (delivery §3.3): std of teacher active-pair cumulative gaps per depth (pos 0..n_pos x
# neg n_pos..n_neg_hi, active AFTER first divergence). Also spot-checks §5.4 pre-divergence gap == 0.
np_, nn_ = args.n_pos, args.n_neg_hi
tcum = step_out.cumsum(-1)                                    # (N,W,4)
depth = torch.arange(4).view(1, 1, 1, 4)
scale_sum = torch.zeros(4); scale_sq = torch.zeros(4); scale_cnt = torch.zeros(4)
prediv_max = 0.0
CH = 4096
for lo in range(0, N, CH):
    hi = min(N, lo + CH)
    tp = tcum[lo:hi, :np_]; tn = tcum[lo:hi, np_:nn_]         # (b,np,4),(b,neg,4)
    gap = tp.unsqueeze(2) - tn.unsqueeze(1)                   # (b,np,neg,4)
    ci = codes[lo:hi, :np_].unsqueeze(2); cj = codes[lo:hi, np_:nn_].unsqueeze(1)
    diff = (ci != cj)                                         # (b,np,neg,4)
    hasdiv = diff.any(-1)
    first = diff.to(torch.float32).argmax(-1)                 # (b,np,neg) first divergence depth
    vp = val[lo:hi, :np_].unsqueeze(2); vn = val[lo:hi, np_:nn_].unsqueeze(1)
    pv = (vp & vn)
    active = (depth >= first.unsqueeze(-1)) & hasdiv.unsqueeze(-1) & pv.unsqueeze(-1)
    prediv = (depth < first.unsqueeze(-1)) & hasdiv.unsqueeze(-1) & pv.unsqueeze(-1)
    if prediv.any():
        prediv_max = max(prediv_max, gap[prediv].abs().max().item())
    for d in range(4):
        m = active[..., d]
        if m.any():
            g = gap[..., d][m]
            scale_sum[d] += g.sum(); scale_sq[d] += (g * g).sum(); scale_cnt[d] += m.sum()
mean = scale_sum / scale_cnt.clamp_min(1)
var = (scale_sq / scale_cnt.clamp_min(1)) - mean * mean
score_scale = var.clamp_min(0).sqrt().clamp_min(1e-4)
print(f'[VAL 5.4] max pre-divergence |gap| (should be ~0 by construction): {prediv_max:.2e}', flush=True)
print(f'[score_scale] per depth d1..d4: {[round(x,4) for x in score_scale.tolist()]} '
      f'(active-pair counts {[int(x) for x in scale_cnt.tolist()]})', flush=True)

D['step_logp'] = step_out.half()
D['eos_logp'] = eos_out.half()
D['score_scale'] = score_scale.float()
D['cdrs_meta'] = {'n_pos': np_, 'n_neg_hi': nn_, 'src_dump': args.in_dump, 'bf16': bool(nb16),
                  'val_5_1_maxerr': float(err.max()), 'eos_abs_mean': float(eosv.mean()),
                  'prediv_max': float(prediv_max)}
torch.save(D, args.out)
print(f'[aug] SAVED {args.out}  (+step_logp +eos_logp +score_scale)', flush=True)
