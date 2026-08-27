# s43e DA-CDRS gradient calibration (train-only, never test-tuned). On ~12 train mini-batches at
# init, measure ||grad(L_da_correction_raw)|| vs ||grad(L_base + 4*list + 1*pair)|| over the shared
# params, then pick a single DA_W so ||DA_W * grad(da)|| / ||grad(base)|| lands ~ target (0.5). The
# da correction is added ON TOP of digit-CE + list-KL + the pair anchor, so calibrate against that
# full base. Chosen weight is fixed for all seeds/datasets/modes (mirrors the s41 §7 protocol).
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
from sid_mlp.losses import (loss_fn, d4_loss_fn, list_kl_loss, boundary_pair_loss,
                            student_list_scores_steps, da_cdrs_loss)

ap = argparse.ArgumentParser()
ap.add_argument('--category', default='Industrial_and_Scientific')
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--dataset_tag', required=True)
ap.add_argument('--graph_dir', required=True)
ap.add_argument('--beam_dump', required=True)
ap.add_argument('--list_kl', type=float, default=4.0)
ap.add_argument('--pair_w', type=float, default=1.0)
ap.add_argument('--pair_gamma', type=float, default=0.05)
ap.add_argument('--batches', type=int, default=12)
ap.add_argument('--batch_size', type=int, default=512)
ap.add_argument('--target', type=float, default=0.5)
a = ap.parse_args()

set_seed(42)
dev = torch.device('cuda')
tk = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
emb = tk['t5.shared.weight']; del tk
args = SimpleNamespace(loss='combined', d4_teacher='combined', alpha=0.7, temperature=1.0,
                       list_temp=1.0, variant='base')
train_ds = prepare_common_data(a.graph_dir, a.dataset_tag, dev, torch.float32)
attach_beam_dump(train_ds, a.beam_dump, dev, topm=0)
scale = getattr(train_ds, 'bm_step_scale', None)
assert scale is not None, 'augmented dump missing step_scale (run add_step_scale_s43e first)'
print('[calib] step_scale', [round(x, 4) for x in scale.tolist()], flush=True)
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


import statistics as st
ratios = []
for bi, batch in enumerate(loader):
    if bi >= a.batches:
        break
    enc_seq, enc_mask, _hk, d1, d2, d3, _d4, tl1, tl2, tl3, tl4, gd1, gd2, gd3, gd4 = batch[:15]
    bm_cb, bm_sc, bm_nv, bm_slp = batch[15], batch[16], batch[17], batch[18]
    M = bm_cb.shape[1]
    valid = torch.arange(M, device=bm_cb.device).unsqueeze(0) < bm_nv.unsqueeze(1)
    # ---- base = digit CE/KD + list_kl*list + pair_w*pair ----
    logits1, logits2, logits3, logits4 = model(enc_seq, enc_mask, d1_ids=d1, d2_ids=d2, d3_ids=d3)
    digit = (loss_fn(args, logits1, tl1, gd1) + loss_fn(args, logits2, tl2, gd2)
             + loss_fn(args, logits3, tl3, gd3) + d4_loss_fn(args, logits4, tl4, gd4))
    slp = student_list_scores_steps(model, enc_seq, enc_mask, bm_cb, list_masks)
    s_S = slp.sum(-1)
    l_list = list_kl_loss(s_S, bm_sc, valid, temperature=1.0)
    l_pair = boundary_pair_loss(s_S, valid, gamma=a.pair_gamma)
    L_base = digit + a.list_kl * l_list + a.pair_w * l_pair
    model.zero_grad(set_to_none=True); L_base.backward(); g_base = gnorm()
    # ---- da correction raw (counterfactual, weight 1) ----
    slp2 = student_list_scores_steps(model, enc_seq, enc_mask, bm_cb, list_masks)
    l_da = da_cdrs_loss(bm_slp, slp2, bm_cb, valid, scale, gamma=a.pair_gamma, huber_delta=1.0,
                        counterfactual=True, n_pos=10, n_neg_hi=40)
    model.zero_grad(set_to_none=True); l_da.backward(); g_da = gnorm()
    ratios.append(g_da / max(g_base, 1e-12))
    print(f'[calib] b{bi} L_base={float(L_base):.4f} g_base={g_base:.3f} | L_da={float(l_da):.4f} '
          f'g_da={g_da:.3f} | ratio(da/base)={ratios[-1]:.4f}', flush=True)

mr = st.median(ratios)
da_w = a.target / max(mr, 1e-9)
print(f'\n[calib] median grad-ratio (da/base) over {len(ratios)} batches = {mr:.4f}', flush=True)
print(f'[calib] -> DA_W = target({a.target}) / ratio = {da_w:.4f}  (||DA_W*grad_da||/||grad_base|| ~= {a.target})', flush=True)
print(f'[calib] RECOMMEND DA_W ~= {round(da_w, 2)}', flush=True)
