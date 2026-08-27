"""Training loop for the public SID-MLP decoder."""

import datetime as dt
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .beam import build_valid_tensors
from .constants import CODEBOOK_SIZE, DIGIT_OFFSETS, set_seed
from .dataset import attach_beam_dump, collate_fn_gpu, prepare_common_data
from .inference import (
    build_tiger_pipeline,
    evaluate_loader,
    find_valid_set,
    replace_t5_encoder_with_sidmlp_pp,
)
from .losses import (
    boundary_neg_indices,
    boundary_pair_loss,
    d4_loss_fn,
    list_kl_loss,
    list_kl_group_loss,
    loss_fn,
    prefix_kl_loss,
    student_list_scores,
    student_list_scores_steps,
    cross_depth_relative_score_loss,
    serve_kl_loss,
    da_cdrs_loss,
    student_list_scores_jp,
    student_list_scores_pd,
    topk_correct,
)
from .models import SIDMLP, SIDMLPJP, SIDMLPPD  # s23/s24

import torch.nn.functional as F

BEAM_VAL_INTERVAL = 5


def _fmt_float(value):
    text = f"{value:e}"
    mantissa, exponent = text.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exp = int(exponent)
    return f"{mantissa}e{exp}" if exp >= 0 else f"{mantissa}e-{abs(exp)}"


def _save_prefix(args):
    attn = f"_ad{args.attn_dim}" if args.attn_dim > 0 else ""
    variant = "sidmlpp" if getattr(args, "encoder_ckpt", None) else "sidmlp"
    base = (
        f"{args.category}_{variant}_k0_xattn"
        f"_h{args.num_heads}_f{args.ffn_dim}{attn}"
        f"_hhid{args.head_hidden}_hl{args.head_layers}"
        f"_lr{_fmt_float(args.lr)}"
    )
    if args.run_tag:
        base += f"_{args.run_tag}"
    return base


def _checkpoint_args(args):
    return {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_") and not callable(value)
    }


def _batch_loss_jp(args, model, batch, list_masks=None):
    """s24 arm C: joint-prefix-head variant. CE on the occupied (d1,d2) class + the usual
    suffix digit losses + the s24 list losses via the JP scorer."""
    enc_seq, enc_mask, _hk, d1, d2, d3, _d4, tl1, tl2, tl3, tl4, gd1, gd2, gd3, gd4 = batch[:15]
    extras = batch[15:] if len(batch) > 15 else None
    comps = {}
    logitsJP, logits3, logits4 = model(enc_seq, enc_mask, d1_ids=d1, d2_ids=d2, d3_ids=d3)
    jp_cls = model.jp_inv[gd1 * CODEBOOK_SIZE + gd2]
    digit_loss = (
        F.cross_entropy(logitsJP, jp_cls.clamp_min(0))
        + loss_fn(args, logits3, tl3, gd3)
        + d4_loss_fn(args, logits4, tl4, gd4)
    )
    comps["digit"] = float(digit_loss.item())
    loss = digit_loss
    lam_list = float(getattr(args, "list_kl", 0.0) or 0.0)
    lam_pf = float(getattr(args, "prefix_kl", 0.0) or 0.0)
    lam_pair = float(getattr(args, "pair_w", 0.0) or 0.0)
    if extras is not None and (lam_list > 0 or lam_pf > 0 or lam_pair > 0):
        bm_cb, bm_sc, bm_nv = extras
        M = bm_cb.shape[1]
        valid = torch.arange(M, device=bm_cb.device).unsqueeze(0) < bm_nv.unsqueeze(1)
        s_S = student_list_scores_jp(model, enc_seq, enc_mask, bm_cb, list_masks)
        if lam_list > 0:
            l_list = list_kl_loss(s_S, bm_sc, valid, temperature=args.list_temp)
            loss = loss + lam_list * l_list
            comps["list"] = float(l_list.item())
        if lam_pf > 0:
            l_pf = prefix_kl_loss(s_S, bm_sc, bm_cb, valid, temperature=args.list_temp, w2=args.prefix_w2, w3=args.prefix_w3)  # s36
            loss = loss + lam_pf * l_pf
            comps["pf"] = float(l_pf.item())
        if lam_pair > 0:
            l_pair = boundary_pair_loss(s_S, valid, gamma=float(getattr(args, "pair_gamma", 0.05)),
                                        n_pos=int(getattr(args, "pair_k", 10)),
                                        n_neg_hi=int(getattr(args, "pair_b", 40)),
                                        n_neg_sample=int(getattr(args, "pair_negs", 0)))
            loss = loss + lam_pair * l_pair
            comps["pair"] = float(l_pair.item())
        with torch.no_grad():
            comps["ltop1"] = float((s_S.masked_fill(~valid, -1e9).argmax(1) == 0).float().mean().item())
    return loss, (logitsJP, logits3, logits4), (jp_cls, gd3, gd4), comps


def _batch_loss(args, model, batch, list_masks=None):
    if getattr(args, "variant", "base") == "jp":  # s24
        return _batch_loss_jp(args, model, batch, list_masks)
    """Digit losses (optionally seq-KD-swapped to the teacher's exact top-1 code) plus the
    s24 search-distribution losses (list-KL over teacher beam candidates, prefix-frontier KL)."""
    enc_seq, enc_mask, _hk, d1, d2, d3, _d4, tl1, tl2, tl3, tl4, gd1, gd2, gd3, gd4 = batch[:15]
    extras = batch[15:] if len(batch) > 15 else None
    comps = {}

    seqkd = bool(getattr(args, "seqkd", False)) and extras is not None
    if seqkd:
        bm_cb = extras[0]
        top1 = bm_cb[:, 0, :]  # dump rows are sorted by exact teacher score desc
        d1 = top1[:, 0] + DIGIT_OFFSETS[0]
        d2 = top1[:, 1] + DIGIT_OFFSETS[1]
        d3 = top1[:, 2] + DIGIT_OFFSETS[2]
        gd1, gd2, gd3, gd4 = top1[:, 0], top1[:, 1], top1[:, 2], top1[:, 3]

    logits1, logits2, logits3, logits4 = model(enc_seq, enc_mask, d1_ids=d1, d2_ids=d2, d3_ids=d3)
    if seqkd:
        # teacher-forced on the teacher-top-1 prefix -> cached GT-prefix teacher logits are
        # mismatched; the clean NAT-style objective is plain CE to the teacher's code.
        digit_loss = (
            F.cross_entropy(logits1, gd1)
            + F.cross_entropy(logits2, gd2)
            + F.cross_entropy(logits3, gd3)
            + F.cross_entropy(logits4, gd4)
        )
    else:
        digit_loss = (
            loss_fn(args, logits1, tl1, gd1)
            + loss_fn(args, logits2, tl2, gd2)
            + loss_fn(args, logits3, tl3, gd3)
            + d4_loss_fn(args, logits4, tl4, gd4)
        )
    comps["digit"] = float(digit_loss.item())
    loss = digit_loss

    lam_list = float(getattr(args, "list_kl", 0.0) or 0.0)
    lam_pf = float(getattr(args, "prefix_kl", 0.0) or 0.0)
    lam_pair = float(getattr(args, "pair_w", 0.0) or 0.0)
    lam_cdrs = float(getattr(args, "cdrs_w", 0.0) or 0.0)  # s41 CDRS
    lam_serve = float(getattr(args, "serve_kl", 0.0) or 0.0)  # s43e E1
    lam_da = float(getattr(args, "da_w", 0.0) or 0.0)         # s43e DA-CDRS
    if extras is not None and (lam_list > 0 or lam_pf > 0 or lam_pair > 0 or lam_cdrs > 0 or lam_serve > 0 or lam_da > 0):
        bm_cb, bm_sc, bm_nv = extras[0], extras[1], extras[2]
        bm_slp = extras[3] if len(extras) > 3 else None  # s41 CDRS: teacher per-digit logp
        M = bm_cb.shape[1]
        valid = torch.arange(M, device=bm_cb.device).unsqueeze(0) < bm_nv.unsqueeze(1)
        _slp = None
        if getattr(args, "variant", "base") == "pd":
            s_S = student_list_scores_pd(model, enc_seq, enc_mask, bm_cb, list_masks,
                                         pd_depth=int(getattr(args, "pd_depth", 4)))
        elif lam_cdrs > 0 or lam_da > 0:  # s41/s43e: per-digit scorer; sum == student_list_scores
            _slp = student_list_scores_steps(model, enc_seq, enc_mask, bm_cb, list_masks)
            s_S = _slp.sum(-1)
        else:
            s_S = student_list_scores(model, enc_seq, enc_mask, bm_cb, list_masks)
        pair_negs = int(getattr(args, "pair_negs", 0))
        da_negs = int(getattr(args, "da_negs", 0))
        shared_neg_idx = None
        if lam_da > 0 and da_negs > 0:
            pair_k = int(getattr(args, "pair_k", 10))
            pair_b = int(getattr(args, "pair_b", 40))
            da_k = int(getattr(args, "cdrs_pos_k", 10))
            da_b = int(getattr(args, "cdrs_neg_end", 40))
            if lam_pair <= 0 or pair_negs != da_negs or pair_k != da_k or pair_b != da_b:
                raise SystemExit(
                    "--da_negs requires pair_w>0, pair_negs==da_negs, "
                    "pair_k==cdrs_pos_k, and pair_b==cdrs_neg_end so pair+DA use identical pairs"
                )
            shared_neg_idx = boundary_neg_indices(
                bm_cb, valid, n_pos=pair_k, n_neg_hi=pair_b,
                n_neg_sample=da_negs, seed=int(getattr(args, "seed", 0)))
            if shared_neg_idx is None:
                raise SystemExit("--da_negs must be smaller than the configured outsider range")
            with torch.no_grad():
                outsider_valid = valid[:, pair_k:pair_b]
                selected_valid = outsider_valid.gather(1, shared_neg_idx)
                comps["da_neg_valid_mean"] = float(outsider_valid.sum(1).float().mean().item())
                comps["da_neg_sample_mean"] = float(selected_valid.sum(1).float().mean().item())
                comps["da_neg_short_frac"] = float((outsider_valid.sum(1) < da_negs).float().mean().item())
        if lam_list > 0:
            _lut = getattr(args, "_strat_lut", None)  # s34
            if _lut is not None:
                _key = (((bm_cb[:, :, 0].long() * 256 + bm_cb[:, :, 1].long()) * 256
                         + bm_cb[:, :, 2].long()) * 256 + bm_cb[:, :, 3].long())
                _idx = torch.searchsorted(_lut[0], _key).clamp(max=_lut[0].numel() - 1)
                _grp = torch.where(_lut[0][_idx] == _key, _lut[1][_idx],
                                   torch.zeros_like(_key))
                l_list = list_kl_group_loss(s_S, bm_sc, valid, _grp,
                                            temperature=args.list_temp)
            else:
                l_list = list_kl_loss(s_S, bm_sc, valid, temperature=args.list_temp)
            loss = loss + lam_list * l_list
            comps["list"] = float(l_list.item())
        if lam_pf > 0:
            l_pf = prefix_kl_loss(s_S, bm_sc, bm_cb, valid, temperature=args.list_temp, w2=args.prefix_w2, w3=args.prefix_w3)  # s36
            loss = loss + lam_pf * l_pf
            comps["pf"] = float(l_pf.item())
        if lam_pair > 0:
            l_pair = boundary_pair_loss(s_S, valid, gamma=float(getattr(args, "pair_gamma", 0.05)),
                                        n_pos=int(getattr(args, "pair_k", 10)),
                                        n_neg_hi=int(getattr(args, "pair_b", 40)),
                                        n_neg_sample=0 if shared_neg_idx is not None else pair_negs,
                                        neg_idx=shared_neg_idx)
            loss = loss + lam_pair * l_pair
            comps["pair"] = float(l_pair.item())
        if lam_cdrs > 0:  # s41 CDRS
            _scale = getattr(args, "_cdrs_scale", None)
            if _slp is None or bm_slp is None or _scale is None:
                raise SystemExit("--cdrs_w requires variant=base + a beam_dump augmented with step_logp+score_scale")
            _tau = float(getattr(args, "cdrs_tau", 0.0) or 0.0) or float(getattr(args, "list_temp", 1.0))
            l_cdrs = cross_depth_relative_score_loss(
                bm_slp, _slp, bm_cb, valid, _scale, tau=_tau,
                n_pos=int(getattr(args, "cdrs_pos_k", 10)), n_neg_hi=int(getattr(args, "cdrs_neg_end", 40)))
            loss = loss + lam_cdrs * l_cdrs
            comps["cdrs"] = float(l_cdrs.item())
        if lam_serve > 0:  # s43e E1 unified serving-KL
            l_serve = serve_kl_loss(s_S, bm_sc, bm_cb, valid, temperature=args.list_temp,
                                    w1=float(getattr(args, "serve_w1", 1.0)), w2=float(getattr(args, "serve_w2", 1.0)),
                                    w3=float(getattr(args, "serve_w3", 1.0)), w4=float(getattr(args, "serve_w4", 1.0)))
            loss = loss + lam_serve * l_serve
            comps["serve"] = float(l_serve.item())
        if lam_da > 0:  # s43e DA-CDRS correction (pair anchor added separately via pair_w)
            _dascale = getattr(args, "_da_scale", None)
            if _slp is None or bm_slp is None or _dascale is None:
                raise SystemExit("--da_w requires an augmented dump with step_logp + step_scale (run add_step_scale_s43e)")
            l_da = da_cdrs_loss(bm_slp, _slp, bm_cb, valid, _dascale,
                                gamma=float(getattr(args, "da_gamma", 0.05)), huber_delta=float(getattr(args, "da_huber", 1.0)),
                                counterfactual=(str(getattr(args, "da_mode", "cf")) != "uniform"),
                                n_pos=int(getattr(args, "cdrs_pos_k", 10)), n_neg_hi=int(getattr(args, "cdrs_neg_end", 40)),
                                neg_idx=shared_neg_idx)
            loss = loss + lam_da * l_da
            comps["da"] = float(l_da.item())
        with torch.no_grad():
            comps["ltop1"] = float((s_S.masked_fill(~valid, -1e9).argmax(1) == 0).float().mean().item())

    return loss, (logits1, logits2, logits3, logits4), (gd1, gd2, gd3, gd4), comps


@torch.no_grad()
def _validate_digits(args, model, loader, device):
    model.eval()
    losses = []
    correct_at1 = np.zeros(4, dtype=np.int64)
    correct_at5 = np.zeros(4, dtype=np.int64)
    total = 0

    for batch in tqdm(loader, desc="Val digits", ncols=100):
        loss, logits, labels, _comps = _batch_loss(args, model, batch)
        losses.append(float(loss.item()))
        total += labels[-1].shape[0]
        for idx, (logit, label) in enumerate(zip(logits, labels)):
            c1, c5 = topk_correct(logit, label, k=5)
            correct_at1[idx] += c1
            correct_at5[idx] += c5

    return {
        "loss": float(np.mean(losses)) if losses else float("inf"),
        "acc1": correct_at1 / max(total, 1),
        "acc5": correct_at5 / max(total, 1),
    }


def _setup_beam_validation(args, device):
    pipeline = build_tiger_pipeline(args)
    t5 = pipeline.model.t5.to(device).eval()
    for param in t5.parameters():
        param.requires_grad_(False)

    if args.encoder_ckpt:
        replace_t5_encoder_with_sidmlp_pp(
            t5,
            args.encoder_ckpt,
            device,
        )
        if not getattr(t5.encoder, "sidmlp_pp_patched", False):
            raise RuntimeError("SID-MLP++ validation failed to patch the TIGER teacher encoder.")
        print(f"Patched TIGER teacher encoder with SID-MLP++ encoder for validation: {args.encoder_ckpt}", flush=True)

    split = "val" if "val" in pipeline.tokenized_datasets else "test"
    loader = DataLoader(
        pipeline.tokenized_datasets[split],
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=pipeline.tokenizer.collate_fn[split],
        num_workers=0,
        pin_memory=False,
    )
    valid_set = find_valid_set(
        args.dataset_tag,
        explicit_path=args.valid_set,
        graph_dir=args.graph_dir,
    )
    if valid_set is None:
        raise FileNotFoundError(
            f"Cannot find valid_item_set for dataset_tag={args.dataset_tag}. "
            "Run build_valid_set.py or pass --valid_set."
        )
    return t5, loader, build_valid_tensors(valid_set, device)


def _check_sidmlp_pp_training_tags(args):
    if not getattr(args, "encoder_ckpt", None):
        return
    base_tag = f"{args.dataset}_{args.category}"
    if args.dataset_tag == base_tag or args.val_dataset_tag == f"{base_tag}_val":
        raise SystemExit(
            "SID-MLP++ stage2 must train on transformed encoder features, not the base TIGER "
            "hidden-state tags. Run scripts/extract.sh sidmlp-pp first and use the transformed dataset_tag."
        )


def train_sidmlp(args):
    _check_sidmlp_pp_training_tags(args)
    if not getattr(args, "graph_dir", None):
        raise SystemExit("--graph_dir is required. Set it in the launcher script or pass it on the CLI.")
    graph_dir = os.path.abspath(args.graph_dir)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = os.path.abspath(args.ckpt_dir or os.path.join(graph_dir, "SID-MLP", "ckpt"))
    log_dir = os.path.abspath(args.log_dir or os.path.join(graph_dir, "SID-MLP", "logs"))
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    save_prefix = _save_prefix(args)
    log_path = os.path.join(log_dir, f"{save_prefix}_{dt.datetime.now():%Y%m%d_%H%M%S}.log")

    def log(message):
        print(message, flush=True)
        with open(log_path, "a") as handle:
            print(message, file=handle)

    teacher_ckpt = torch.load(os.path.abspath(args.checkpoint), map_location="cpu", weights_only=False)
    embedding_weight = teacher_ckpt["t5.shared.weight"]
    del teacher_ckpt

    storage_dtype = torch.float16 if args.storage_fp16 else torch.float32
    train_ds = prepare_common_data(graph_dir, args.dataset_tag, device, storage_dtype)
    val_ds = prepare_common_data(graph_dir, args.val_dataset_tag, device, storage_dtype)

    # ---- s24: search-distribution distillation wiring ----
    use_s24 = bool(getattr(args, "beam_dump", None))
    variant = getattr(args, "variant", "base")
    list_masks = None
    jp_pairs = None
    if use_s24:
        attach_beam_dump(train_ds, args.beam_dump, device, topm=int(getattr(args, 'list_topm', 0) or 0))  # s36
        if float(getattr(args, 'cdrs_w', 0.0) or 0.0) > 0:  # s41 CDRS
            args._cdrs_scale = getattr(train_ds, 'bm_score_scale', None)
            if args._cdrs_scale is None:
                raise SystemExit('--cdrs_w requires an augmented beam_dump (step_logp+score_scale)')
        if float(getattr(args, 'da_w', 0.0) or 0.0) > 0:  # s43e DA-CDRS
            args._da_scale = getattr(train_ds, 'bm_step_scale', None)
            if args._da_scale is None:
                raise SystemExit('--da_w requires an augmented dump with step_scale (run add_step_scale_s43e)')
        if variant not in ("base", "jp", "pd"):
            raise SystemExit("s24 list/seqkd losses are implemented for base/pd/jp variants only.")
        if variant == "pd" and getattr(args, "seqkd", False):
            raise SystemExit("seqkd is not supported for the pd variant")
        if variant == "jp" and getattr(args, "seqkd", False):
            raise SystemExit("seqkd is not supported for the jp variant")
    elif (getattr(args, "seqkd", False)
          or float(getattr(args, "list_kl", 0.0) or 0.0) > 0
          or float(getattr(args, "prefix_kl", 0.0) or 0.0) > 0
          or float(getattr(args, "pair_w", 0.0) or 0.0) > 0):
        raise SystemExit("--seqkd/--list_kl/--prefix_kl/--pair_w require --beam_dump")
    need_masks = variant == "jp" or (use_s24 and (
        float(getattr(args, "list_kl", 0.0) or 0.0) > 0
        or float(getattr(args, "prefix_kl", 0.0) or 0.0) > 0
        or float(getattr(args, "pair_w", 0.0) or 0.0) > 0
        or float(getattr(args, "cdrs_w", 0.0) or 0.0) > 0
        or float(getattr(args, "serve_kl", 0.0) or 0.0) > 0
        or float(getattr(args, "da_w", 0.0) or 0.0) > 0))  # s43e
    if need_masks:
        vs_path = find_valid_set(args.dataset_tag, explicit_path=args.valid_set, graph_dir=args.graph_dir)
        if vs_path is None:
            raise FileNotFoundError("valid_item_set required for list scoring; run build_valid_set.py")
        vt = build_valid_tensors(vs_path, device)
        # clone out of inference mode: build_valid_tensors outputs cannot be saved for backward
        list_masks = (vt[0].clone(), vt[1].clone(), vt[2].clone(), vt[3].clone())
        if variant == "jp":
            jp_pairs = list_masks[0].nonzero().cpu()  # occupied (d1,d2) codebook pairs

    make_collate = lambda dataset: (lambda indices: collate_fn_gpu(dataset, indices))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=make_collate(train_ds),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=make_collate(val_ds),
    )

    if int(getattr(args, "strat_kl", 0) or 0):  # s34: popularity-stratified list-KL control
        print("[s34 strat] counting GT code frequencies over train set...", flush=True)
        _cnt = {}
        for _b in train_loader:
            _ks = (((_b[11].long() * 256 + _b[12].long()) * 256 + _b[13].long()) * 256
                   + _b[14].long()).tolist()
            for _kk in _ks:
                _cnt[_kk] = _cnt.get(_kk, 0) + 1
        _vals = sorted(_cnt.values())
        _lo = _vals[len(_vals) // 3]; _hi = _vals[2 * len(_vals) // 3]
        _dev = torch.device("cuda")
        _keys = torch.tensor(sorted(_cnt.keys()), dtype=torch.long, device=_dev)
        _grps = torch.tensor([0 if _cnt[int(k)] <= _lo else (1 if _cnt[int(k)] <= _hi else 2)
                              for k in _keys.tolist()], dtype=torch.long, device=_dev)
        args._strat_lut = (_keys, _grps)
        print(f"[s34 strat] codes={_keys.numel()} lo={_lo} hi={_hi}", flush=True)

    _model_kwargs = dict(
        embed_dim=embedding_weight.shape[1],
        d_model=128,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        head_hidden=args.head_hidden,
        head_layers=args.head_layers,
        attn_dim=args.attn_dim,
    )
    if variant == "jp":  # s24
        model = SIDMLPJP(embedding_weight, jp_pairs=jp_pairs, **_model_kwargs).to(device)
        print(f"SIDMLPJP: {jp_pairs.shape[0]} occupied (d1,d2) prefixes", flush=True)
    else:
        model_cls = SIDMLPPD if variant != "base" else SIDMLP  # s23
        if variant != "base":
            _model_kwargs["pd_depth"] = int(getattr(args, "pd_depth", 4))
            _model_kwargs["pd_mode"] = str(getattr(args, "pd_mode", "full") or "full")
        model = model_cls(embedding_weight, **_model_kwargs).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    beam_t5 = beam_loader = beam_valid_tensors = None
    if args.epochs >= BEAM_VAL_INTERVAL and variant != "jp":  # s24: JP evals via frontier only
        beam_t5, beam_loader, beam_valid_tensors = _setup_beam_validation(args, device)

    log(f"Training {'SID-MLP++' if args.encoder_ckpt else 'SID-MLP'}")
    log(f"train={len(train_ds)} val={len(val_ds)} checkpoint_dir={ckpt_dir}")
    if use_s24:
        log(f"s24: beam_dump={args.beam_dump} seqkd={getattr(args, 'seqkd', False)} "
            f"list_kl={getattr(args, 'list_kl', 0.0)} prefix_kl={getattr(args, 'prefix_kl', 0.0)} "
            f"list_temp={getattr(args, 'list_temp', 1.0)} pd_depth={getattr(args, 'pd_depth', 4)} "
            f"pd_mode={getattr(args, 'pd_mode', 'full')}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        comp_sums, comp_counts = {}, {}
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", ncols=100)):
            if args.test and batch_idx >= 5:
                break
            optimizer.zero_grad(set_to_none=True)
            loss, _logits, _labels, comps = _batch_loss(args, model, batch, list_masks)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))
            for k, v in comps.items():
                comp_sums[k] = comp_sums.get(k, 0.0) + v
                comp_counts[k] = comp_counts.get(k, 0) + 1

        scheduler.step()
        digit_metrics = _validate_digits(args, model, val_loader, device)
        beam_metrics = {}
        if beam_t5 is not None and epoch % BEAM_VAL_INTERVAL == 0:
            beam_metrics = evaluate_loader(
                model,
                beam_t5,
                beam_loader,
                beam_valid_tensors,
                device,
                beam_size=args.beam_size,
                final_top_n=args.final_top_n,
                max_batches=5 if args.test else None,
            )

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        comp_str = " ".join(
            f"{k}={comp_sums[k] / max(comp_counts[k], 1):.4f}" for k in sorted(comp_sums)
        )
        log(
            f"epoch={epoch} train_loss={train_loss:.6f} val_loss={digit_metrics['loss']:.6f} "
            f"d1@1={digit_metrics['acc1'][0]:.4f} d2@1={digit_metrics['acc1'][1]:.4f} "
            f"d3@1={digit_metrics['acc1'][2]:.4f} d4@1={digit_metrics['acc1'][3]:.4f} "
            f"ndcg10={beam_metrics.get('ndcg10', 0.0):.6f}"
            + (f" {comp_str}" if comp_str else "")
        )

    final_path = os.path.join(ckpt_dir, f"{save_prefix}_final.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "args": _checkpoint_args(args),
        "model_type": "SID-MLP++" if args.encoder_ckpt else "SID-MLP",
    }, final_path)
    log(f"Saved final checkpoint: {final_path}")
