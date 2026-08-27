# s43e: backfill step_scale (per-depth teacher INCREMENT-gap std) into an augmented CDRS dump.
# The s41 dump stored score_scale = std of CUMULATIVE gaps (for s41's cumulative CDRS); DA-CDRS
# uses INCREMENTAL gaps, so it needs the increment-gap std. step_logp is already in the dump ->
# pure CPU tensor op, NO teacher forward. Mirrors augment_dump_cdrs's active-after-first-divergence
# masking so the scale is over the same pairs the loss actually supervises.
import argparse, torch
ap = argparse.ArgumentParser()
ap.add_argument('--dump', required=True)
ap.add_argument('--n_pos', type=int, default=10)
ap.add_argument('--n_neg_hi', type=int, default=40)
a = ap.parse_args()

D = torch.load(a.dump, map_location='cpu', weights_only=False)
codes = D['codes']; nvalid = D['nvalid']; step = D['step_logp'].float()   # (N,W,4) per-digit logp
N, W, _ = codes.shape
val = torch.arange(W).view(1, W) < nvalid.view(N, 1)
np_, nn_ = a.n_pos, a.n_neg_hi
depth = torch.arange(4).view(1, 1, 1, 4)
ssum = torch.zeros(4); ssq = torch.zeros(4); scnt = torch.zeros(4)
CH = 4096
for lo in range(0, N, CH):
    hi = min(N, lo + CH)
    sp = step[lo:hi, :np_]; sn = step[lo:hi, np_:nn_]     # (b,np,4),(b,neg,4) INCREMENTS (per-digit)
    gap = sp.unsqueeze(2) - sn.unsqueeze(1)               # (b,np,neg,4) incremental gap
    ci = codes[lo:hi, :np_].unsqueeze(2); cj = codes[lo:hi, np_:nn_].unsqueeze(1)
    diff = (ci != cj); hasdiv = diff.any(-1)
    first = diff.to(torch.float32).argmax(-1)             # first divergence depth
    vp = val[lo:hi, :np_].unsqueeze(2); vn = val[lo:hi, np_:nn_].unsqueeze(1)
    pv = (vp & vn)
    active = (depth >= first.unsqueeze(-1)) & hasdiv.unsqueeze(-1) & pv.unsqueeze(-1)
    for d in range(4):
        m = active[..., d]
        if m.any():
            g = gap[..., d][m]
            ssum[d] += g.sum(); ssq[d] += (g * g).sum(); scnt[d] += m.sum()
mean = ssum / scnt.clamp_min(1)
var = (ssq / scnt.clamp_min(1)) - mean * mean
step_scale = var.clamp_min(0).sqrt().clamp_min(1e-4)
print('[step_scale] incremental-gap std per depth d1..d4:', [round(x, 4) for x in step_scale.tolist()],
      '| active-pair counts', [int(x) for x in scnt.tolist()])
D['step_scale'] = step_scale.float()
torch.save(D, a.dump)
print('[saved]', a.dump)
