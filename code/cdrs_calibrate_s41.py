# s41 CDRS §7 gradient calibration (train-only, never test-tuned). On ~12 train mini-batches at
# init, measure ||grad(L_CDRS_raw)|| vs ||grad(L_base+list)|| over the shared params, then pick a
# single CDRS_W so ||CDRS_W * grad(L_CDRS)|| / ||grad(L_base/list)|| lands in [0.3, 1.0] (target 0.5).
# The chosen weight is fixed for MI/VG and all seeds (delivery §7.4).
import sys, os, argparse
from types import SimpleNamespace
R = os.environ.get("SERVESID_ROOT", ".")
sys.path.insert(0, os.path.join(R, "repos", "SID-MLP"))
os.chdir(os.path.join(R, "repos"))
for k in ['HF_DATASETS_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']:
    os.environ[k] = '1'
import torch
from torch.utils.data import DataLoader
from sid_mlp.dataset import prepare_common_data, attach_beam_dump, collate_fn_gpu
from sid_mlp.beam import build_valid_tensors
from sid_mlp.inference import find_valid_set
from sid_mlp.models import SIDMLP
from sid_mlp.constants import set_seed
from sid_mlp.losses import (loss_fn, d4_loss_fn, list_kl_loss,
                            student_list_scores_steps, cross_depth_relative_score_loss)

ap = argparse.ArgumentParser()
ap.add_argument('--category', default='Industrial_and_Scientific')
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--dataset_tag', required=True)
ap.add_argument('--graph_dir', required=True)
ap.add_argument('--beam_dump', required=True)
ap.add_argument('--list_kl', type=float, default=4.0)
ap.add_argument('--batches', type=int, default=12)
ap.add_argument('--batch_size', type=int, default=512)
ap.add_argument('--target', type=float, default=0.5)
a = ap.parse_args()

args = SimpleNamespace(loss='combined', d4_teacher='combined', alpha=0.7, temperature=1.0,
                       list_temp=1.0, cdrs_tau=0.0, cdrs_pos_k=10, cdrs_neg_end=40, variant='base')
set_seed(42)
dev = torch.device('cuda')
tk = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
emb = tk['t5.shared.weight']; del tk
train_ds = prepare_common_data(a.graph_dir, a.dataset_tag, dev, torch.float32)
attach_beam_dump(train_ds, a.beam_dump, dev, topm=0)
scale = train_ds.bm_score_scale
assert scale is not None, 'augmented dump missing score_scale'
print('[calib] score_scale', [round(x, 4) for x in scale.tolist()], flush=True)
vs = find_valid_set(a.dataset_tag, explicit_path=None, graph_dir=a.graph_dir)
vt = build_valid_tensors(vs, dev)
list_masks = tuple(vt[i].clone() for i in range(4))
model = SIDMLP(emb, embed_dim=emb.shape[1], d_model=128, num_heads=4, ffn_dim=1024, dropout=0.2,
               head_hidden=512, head_layers=1, attn_dim=384).to(dev)
loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=0,
                    collate_fn=lambda ii: collate_fn_gpu(train_ds, ii))

def gnorm():
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += float((p.grad.detach() ** 2).sum())
    return s ** 0.5

r_bl, r_cd, ratios = [], [], []
for bi, batch in enumerate(loader):
    if bi >= a.batches:
        break
    enc_seq, enc_mask, _hk, d1, d2, d3, _d4, tl1, tl2, tl3, tl4, gd1, gd2, gd3, gd4 = batch[:15]
    bm_cb, bm_sc, bm_nv, bm_slp = batch[15], batch[16], batch[17], batch[18]
    M = bm_cb.shape[1]
    valid = torch.arange(M, device=bm_cb.device).unsqueeze(0) < bm_nv.unsqueeze(1)
    # ---- L_base+list = digit CE/KD + list_kl*list ----
    logits1, logits2, logits3, logits4 = model(enc_seq, enc_mask, d1_ids=d1, d2_ids=d2, d3_ids=d3)
    digit = (loss_fn(args, logits1, tl1, gd1) + loss_fn(args, logits2, tl2, gd2)
             + loss_fn(args, logits3, tl3, gd3) + d4_loss_fn(args, logits4, tl4, gd4))
    slp = student_list_scores_steps(model, enc_seq, enc_mask, bm_cb, list_masks)
    l_list = list_kl_loss(slp.sum(-1), bm_sc, valid, temperature=1.0)
    L_bl = digit + a.list_kl * l_list
    model.zero_grad(set_to_none=True); L_bl.backward(); g_bl = gnorm()
    # ---- L_CDRS raw (weight 1) ----
    slp2 = student_list_scores_steps(model, enc_seq, enc_mask, bm_cb, list_masks)
    l_cdrs = cross_depth_relative_score_loss(bm_slp, slp2, bm_cb, valid, scale, tau=1.0,
                                             n_pos=10, n_neg_hi=40)
    model.zero_grad(set_to_none=True); l_cdrs.backward(); g_cd = gnorm()
    r_bl.append(g_bl); r_cd.append(g_cd); ratios.append(g_cd / max(g_bl, 1e-12))
    print(f'[calib] b{bi} L_bl={float(L_bl):.4f} g_bl={g_bl:.3f} | L_cdrs={float(l_cdrs):.4f} '
          f'g_cdrs={g_cd:.3f} | ratio(cdrs/bl)={ratios[-1]:.4f}', flush=True)

import statistics as st
mr = st.median(ratios)
cdrs_w = a.target / max(mr, 1e-9)
print(f'\n[calib] median grad-ratio (cdrs/base+list) over {len(ratios)} batches = {mr:.4f}', flush=True)
print(f'[calib] mean g_bl={st.mean(r_bl):.3f} mean g_cdrs={st.mean(r_cd):.3f}', flush=True)
print(f'[calib] -> CDRS_W = target({a.target}) / ratio = {cdrs_w:.4f}  '
      f'(so ||W*grad_cdrs||/||grad_bl|| ~= {a.target})', flush=True)
# round to a clean value in-band [0.3,1.0]*grad-parity
print(f'[calib] RECOMMEND CDRS_W ~= {round(cdrs_w, 2)}', flush=True)
