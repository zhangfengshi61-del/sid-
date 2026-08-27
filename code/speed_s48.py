"""s48 §1.1: ONE end-to-end speed harness that times every method with the SAME bracket.

WHY THIS EXISTS
---------------
The per-method harnesses time DIFFERENT regions, so their numbers are not comparable:
  - `code/tiger_throughput.py` (TIGER-kv): whole-loop wall clock around the full test loader
    (dataloader + H2D + the entire `generate()`), 2 warmup batches excluded.
  - SID-MLP / ServeSID student (`repos/SID-MLP/sid_mlp/inference.py::run_inference`): the ENTIRE
    `evaluate_loader` call -- H2D + encoder + beam + item lookup + d2h + the full Python per-row
    METRIC loop, and NO warmup at all.
  - `code/nezha_baseline.py::evaluate`, `code/ssm_baseline.py::evaluate`, `code/earn_baseline.py`:
    ONLY the GPU beam, per-batch `cuda.synchronize()` bracket, first 3 batches skipped; H2D, d2h
    and the metric loop are OUTSIDE.
Consequence: NEZHA self-reported ~6600-6800 samples/s and looked FASTER than the SID-MLP student
(~3300/s), inverting the SID-MLP paper's ordering (NEZHA 7.27x < SID-MLP 8.74x). That inversion is
a cross-harness artifact, not a real speed difference. This script replaces all of them.

THE TIMED REGION (identical for every method -- also written into every emitted JSON)
------------------------------------------------------------------------------------
The clock is started ONCE after `torch.cuda.synchronize()` and stopped ONCE after another
`torch.cuda.synchronize()`. Between them, for every method, exactly:

  INSIDE   (1) iterating the test DataLoader (CPU-side collation via the pipeline's own
               `collate_fn['test']` -- byte-identical work for every method),
           (2) H2D transfer of the WHOLE collated batch dict (identical tensors for every
               method, so the transfer cost is matched even though only TIGER's generate
               consumes the extra keys),
           (3) the method's REAL deployed decode -- the same function its quality eval calls --
               i.e. encoder forward + constrained/valid-SID decoding, ending at the decoded
               SID tensor still ON GPU.

  OUTSIDE  model loading, model init, `build_tiger_pipeline`, valid-set / mask / Trie
           construction, popularity stats, the `--warmup` batches, d2h (`.cpu()/.tolist()`),
           SID->item lookup, metric computation, JSON writing.

DELIBERATE PROTOCOL CHOICE -- WE STOP AT THE DECODED SID TENSOR.
No `.cpu()`, no SID->item lookup, no dedup, no metrics. That is precisely what makes the region
matched: the methods disagree about where the item lookup lives (TIGER does part of it inside
`generate`, the students do it in a Python loop after d2h), so the only region every method shares
is "bytes in -> decoded SID quads on GPU". The lookup/metric tail is real serving work, but it is
method-independent Python and including it is what produced the artifact this script fixes.

KNOWN, DOCUMENTED ASYMMETRIES (report them with the numbers)
------------------------------------------------------------
  * TIGER-kv's `tiger.generate()` is a black box: its internal valid-SID masking (and whatever
    lookup genrec does inside) is inside the bracket because it cannot be split out without
    changing the deployed decode. It returns SID token sequences, so the bracket still ends at
    a decoded SID tensor.
  * EARN runs in **fp32** with the **constrained** valid-SID Trie callback, because that is its
    deployed decode (`earn_baseline.evaluate` / `earn_bucketdiag.py`). Note `code/earn_throughput.py`
    measured something else: bf16 + UNCONSTRAINED `generate` (no `prefix_allowed_tokens_fn`) as an
    explicit stand-in, so this harness will report a much lower EARN throughput than
    `runs/earntput_*.json`. Different region AND different decode -- do not compare them.
  * Every other method runs bf16 exactly as its own quality eval configures it (see `_build_*`).
  * `loader_sec` is reported per rep as a diagnostic only. GPU work is asynchronous, so loader time
    OVERLAPS device execution -- it must NOT be subtracted from `elapsed_sec`.

Usage (one invocation = one dataset; the dataset dimension is separate invocations):
  PY=$R/envs/sidmlp2/bin/python
  $PY $R/code/speed_s48.py --category Industrial_and_Scientific \
    --checkpoint $R/repos/SID-MLP-assets/TIGER-AmazonReviews2023-category_Industrial_and_Scientific.pth \
    --sem_ids $R/repos/SID-MLP-assets/semantic_ids/AmazonReviews2023-Industrial_and_Scientific_sentence-t5-base_256,256,256,256.sem_ids \
    --dataset_tag AmazonReviews2023_Industrial_and_Scientific --graph_dir $R/sidmlp-graph \
    --methods tigerkv,sidmlp,nezha,gdn,mamba2,earn \
    --sidmlp_ckpt $R/sidmlp-graph/SID-MLP/ckpt/..._serveU4pair_final.pth \
    --nezha_ckpt  $R/runs/nezha_Industrial_and_Scientific_s42.pth \
    --gdn_ckpt    $R/runs/ssm_gdn_Industrial_and_Scientific_s42.pth \
    --mamba2_ckpt $R/runs/ssm_mamba2_Industrial_and_Scientific_s42.pth \
    --earn_ckpt   $R/runs/earn_Industrial_and_Scientific_k2.pth \
    --batch_size 32 --beam 50 --warmup 2 --reps 3 --out $R/runs/s48speed
  # bs=1 online latency point (p50/p95/p99/mean ms per request), same bracket per request:
  ... --batch_size 1 --latency --max_batches 2000 --out $R/runs/s48speed
"""
import argparse, gc, hashlib, json, os, sys, time
import numpy as np
import torch

R = os.environ.get("SERVESID_ROOT", ".")
sys.path.insert(0, f"{R}/repos/SID-MLP")
sys.path.insert(0, f"{R}/repos")
sys.path.insert(0, f"{R}/code")
os.chdir(f"{R}/repos")
for k in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[k] = "1"
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sid_mlp.constants import CODEBOOK_SIZE, DIGIT_OFFSETS, triplet_key_int
from sid_mlp.beam import build_valid_tensors, gather_d4_mask
from sid_mlp.inference import (build_tiger_pipeline, build_sidmlp_from_checkpoint,
                               configure_precision, _torch_load, find_valid_set)
# model builders + deployed decodes -- imported, never reimplemented
from nezha_baseline import build_from_ckpt as nezha_build_from_ckpt, nezha_beam
from ssm_baseline import build_model_from_ckpt as ssm_build_from_ckpt, ssm_beam
from earn_baseline import (build_from_ckpt as earn_build_from_ckpt, build_trie, make_prefix_fn)

DEV = torch.device("cuda")
METHODS = ("tigerkv", "sidmlp", "nezha", "gdn", "mamba2", "earn")
TIMED_REGION = (
    "single sync-bracketed wall clock over the full test split: [cuda.synchronize -> clock start] "
    "{DataLoader iteration (collate_fn['test']) + H2D of the whole collated batch dict + the "
    "method's deployed decode (encoder forward + constrained/valid-SID decoding), ending at the "
    "decoded SID tensor on GPU} [cuda.synchronize -> clock stop]. EXCLUDED: model load/init, "
    "pipeline+valid-set/mask/Trie construction, the --warmup batches, d2h, SID->item lookup, "
    "metric computation."
)

ap = argparse.ArgumentParser()
ap.add_argument("--category", required=True)
ap.add_argument("--checkpoint", required=True)      # TIGER teacher .pth (arch + shared embedding)
ap.add_argument("--sem_ids", required=True)
ap.add_argument("--dataset_tag", required=True)
ap.add_argument("--graph_dir", required=True)
ap.add_argument("--dataset", default="AmazonReviews2023")
ap.add_argument("--methods", required=True,
                help="comma-separated subset of " + ",".join(METHODS))
ap.add_argument("--sidmlp_ckpt", default="")        # ServeSID / SID-MLP student ckpt
ap.add_argument("--nezha_ckpt", default="")
ap.add_argument("--gdn_ckpt", default="")
ap.add_argument("--mamba2_ckpt", default="")
ap.add_argument("--earn_ckpt", default="")
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--beam", type=int, default=50)
ap.add_argument("--warmup", type=int, default=2)    # batches run+discarded before every rep
ap.add_argument("--reps", type=int, default=3)      # isolated repetitions; NO averaging inside one
ap.add_argument("--max_batches", type=int, default=0)   # 0 = full split; >0 = smoke / bs1 cap
ap.add_argument("--latency", action="store_true")   # per-request sync bracket -> p50/p95/p99/mean
ap.add_argument("--out", required=True)             # OUTPUT DIRECTORY (one json per method)
ap.add_argument("--tag", default="")
args = ap.parse_args()
args.encoder_ckpt = None

sel = [m.strip() for m in args.methods.split(",") if m.strip()]
bad = [m for m in sel if m not in METHODS]
if bad:
    raise SystemExit(f"unknown --methods {bad}; choose from {METHODS}")
CKPT = {"tigerkv": args.checkpoint, "sidmlp": args.sidmlp_ckpt, "nezha": args.nezha_ckpt,
        "gdn": args.gdn_ckpt, "mamba2": args.mamba2_ckpt, "earn": args.earn_ckpt}
for m in sel:                                        # fail before any GPU work
    if not CKPT[m]:
        raise SystemExit(f"--methods includes {m} but --{m}_ckpt is empty")
    if not os.path.exists(CKPT[m]):
        raise SystemExit(f"{m}: checkpoint not found: {CKPT[m]}")
if args.latency and args.batch_size != 1:
    print(f"!!! WARNING: --latency with batch_size={args.batch_size}: the reported percentiles are "
          f"per BATCH, not per request !!!", flush=True)
os.makedirs(args.out, exist_ok=True)


# --------------------------------------------------------------------------- per-method builders
# Each returns (step, pipe, dtype_str); `step(d)` runs the deployed decode on an already-on-device
# batch dict and closes over everything it needs (dropping `step` is what frees the method).
def _build_tigerkv():
    """code/tiger_throughput.py: pipe.model (genrec TIGER) + tiger.generate(batch_dict, beam)."""
    args.fp32 = False
    nb16 = configure_precision(args, DEV)
    pipe = build_tiger_pipeline(args)
    tiger = pipe.model.to(DEV).eval()
    if nb16:
        tiger.bfloat16()

    def step(d):
        tiger.generate(d, n_return_sequences=args.beam)
    return step, pipe, ("bf16" if nb16 else "fp32")


def _build_sidmlp():
    """code/lists_s48.py --method sidmlp: frozen TIGER encoder + build_sidmlp_from_checkpoint,
    decoded with the deployed masked-valid factorized search (student_beam, copied verbatim from
    lists_s48.student_beam == bucket_diag_s34.student_beam_frontiers geometry)."""
    args.fp32 = False
    nb16 = configure_precision(args, DEV)
    pipe = build_tiger_pipeline(args)
    t5 = pipe.model.to(DEV).eval().t5
    if nb16:
        t5.bfloat16()
    for p in t5.parameters():
        p.requires_grad_(False)
    tk = _torch_load(args.checkpoint, weights_only=False)
    lens, _ck = build_sidmlp_from_checkpoint(args.sidmlp_ckpt, tk["t5.shared.weight"], DEV)
    if nb16:
        lens.bfloat16()
    del tk
    vs = find_valid_set(args.dataset_tag, explicit_path=None, lens_path=args.sidmlp_ckpt,
                        graph_dir=args.graph_dir)
    (mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted, _k4, _v4, _im,
     _vi4) = build_valid_tensors(vs, DEV)

    def student_beam(enc, pad, W):
        """VERBATIM code/lists_s48.py::student_beam -- deployed masked-valid factorized search,
        width W -> (B, W, 4) token-id quads, score-ordered."""
        B = enc.shape[0]
        ctx = lens.encode(enc, pad)
        d1_lp = F.log_softmax(lens.head1(ctx).float(), dim=-1)
        W1 = min(W, CODEBOOK_SIZE)
        d1_scores, d1_idx = torch.topk(d1_lp, W1, dim=1)
        d1_tokens = d1_idx + DIGIT_OFFSETS[0]
        ctx_flat = ctx.unsqueeze(1).expand(-1, W1, -1).reshape(B * W1, -1)
        d1_flat = d1_tokens.reshape(B * W1)
        logits2 = lens.head2(torch.cat([ctx_flat, lens.embedding(d1_flat)], 1)).float()
        d1_cb = (d1_flat - DIGIT_OFFSETS[0]).clamp(0, CODEBOOK_SIZE - 1)
        logits2 = logits2.masked_fill(~mask_d1_d2[d1_cb], float('-inf'))
        d2_lp = F.log_softmax(logits2, 1).view(B, W1, CODEBOOK_SIZE)
        score2 = d1_scores.unsqueeze(-1) + d2_lp
        best2_score, best2_idx = torch.topk(
            score2.masked_fill(torch.isnan(score2), float('-inf')).view(B, -1), W, 1)
        sel_beam2 = best2_idx // CODEBOOK_SIZE; sel_d2 = best2_idx % CODEBOOK_SIZE
        d1_tokens = torch.gather(d1_tokens, 1, sel_beam2); d2_tokens = sel_d2 + DIGIT_OFFSETS[1]
        prefix_score = best2_score
        ctx_flat = ctx.unsqueeze(1).expand(-1, W, -1).reshape(B * W, -1)
        d1_flat = d1_tokens.reshape(-1); d2_flat = d2_tokens.reshape(-1)
        logits3 = lens.head3(torch.cat([ctx_flat, lens.embedding(d1_flat),
                                        lens.embedding(d2_flat)], 1)).float()
        d1_cb = (d1_flat - DIGIT_OFFSETS[0]).clamp(0, CODEBOOK_SIZE - 1)
        d2_cb = (d2_flat - DIGIT_OFFSETS[1]).clamp(0, CODEBOOK_SIZE - 1)
        logits3 = logits3.masked_fill(~mask_d1d2_d3[d1_cb, d2_cb], float('-inf'))
        d3_lp = F.log_softmax(logits3, 1).view(B, W, CODEBOOK_SIZE)
        score3 = prefix_score.unsqueeze(-1) + d3_lp
        best3_score, best3_idx = torch.topk(
            score3.masked_fill(torch.isnan(score3), float('-inf')).view(B, -1), W, 1)
        sel_beam3 = best3_idx // CODEBOOK_SIZE; sel_d3 = best3_idx % CODEBOOK_SIZE
        d1_tokens = torch.gather(d1_tokens, 1, sel_beam3)
        d2_tokens = torch.gather(d2_tokens, 1, sel_beam3)
        d3_tokens = sel_d3 + DIGIT_OFFSETS[2]
        d1_flat = d1_tokens.reshape(-1); d2_flat = d2_tokens.reshape(-1)
        d3_flat = d3_tokens.reshape(-1)
        logits4 = lens.head4(torch.cat([ctx_flat, lens.embedding(d1_flat), lens.embedding(d2_flat),
                                        lens.embedding(d3_flat)], 1)).float()
        key3 = triplet_key_int(d1_flat - DIGIT_OFFSETS[0], d2_flat - DIGIT_OFFSETS[1],
                               d3_flat - DIGIT_OFFSETS[2])
        logits4 = logits4.masked_fill(~gather_d4_mask(d4_keys_sorted, d4_masks_sorted, key3),
                                      float('-inf'))
        d4_lp = F.log_softmax(logits4, 1)
        top_d4_val, top_d4_idx = torch.topk(d4_lp, CODEBOOK_SIZE, 1)
        d4_tokens = top_d4_idx + DIGIT_OFFSETS[3]
        quad = best3_score.unsqueeze(-1) + top_d4_val.view(B, W, CODEBOOK_SIZE)
        topq_score, topq_idx = torch.topk(
            quad.masked_fill(torch.isnan(quad), float('-inf')).view(B, -1), W, 1)
        beam_idx = topq_idx // CODEBOOK_SIZE
        d1f = torch.gather(d1_tokens, 1, beam_idx); d2f = torch.gather(d2_tokens, 1, beam_idx)
        d3f = torch.gather(d3_tokens, 1, beam_idx)
        d4f = torch.gather(d4_tokens.view(B, -1), 1, topq_idx)
        return torch.stack([d1f, d2f, d3f, d4f], -1)

    def step(d):
        am = d["attention_mask"]
        enc = t5.encoder(input_ids=d["input_ids"], attention_mask=am,
                         return_dict=True).last_hidden_state
        student_beam(enc, ~am.bool(), args.beam)
    return step, pipe, ("bf16" if nb16 else "fp32")


def _build_nezha():
    """code/nezha_bucketdiag.py: nezha_baseline.build_from_ckpt + nezha_beam (self-drafting MTP,
    encoder pass is INSIDE nezha_beam)."""
    args.fp32 = False
    configure_precision(args, DEV)
    pipe = build_tiger_pipeline(args)
    args.lens_path = args.nezha_ckpt                  # nezha_build_from_ckpt reads args.lens_path
    model = nezha_build_from_ckpt(args, pipe)
    vs = find_valid_set(args.dataset_tag, explicit_path=None, lens_path=args.nezha_ckpt,
                        graph_dir=args.graph_dir)
    (m12, m123, d4k, d4m, _k4, _v4, _im, _vi4) = build_valid_tensors(vs, DEV)
    masks = (m12, m123, d4k, d4m)

    def step(d):
        nezha_beam(model, d["input_ids"], d["attention_mask"], args.beam, masks)
    return step, pipe, "bf16-autocast"


def _build_ssm(kind):
    """code/ssm_bucketdiag.py: frozen bf16 TIGER encoder + ssm_baseline.build_model_from_ckpt,
    decoded with ssm_beam. `kind` in {gdn, mamba2}; the ckpt cfg is authoritative and is asserted."""
    args.fp32 = False
    configure_precision(args, DEV)
    pipe = build_tiger_pipeline(args)
    t5 = pipe.model.to(DEV).eval().t5
    t5.bfloat16()
    for p in t5.parameters():
        p.requires_grad_(False)
    tk = _torch_load(args.checkpoint, weights_only=False)
    ckpt = args.gdn_ckpt if kind == "gdn" else args.mamba2_ckpt
    model = ssm_build_from_ckpt(ckpt, tk["t5.shared.weight"]); del tk
    if model.ssm_type != kind:
        raise SystemExit(f"{kind}: ckpt {ckpt} has ssm_type={model.ssm_type!r}")
    vs = find_valid_set(args.dataset_tag, explicit_path=None, lens_path=ckpt,
                        graph_dir=args.graph_dir)
    (m12, m123, d4k, d4m, _k4, _v4, _im, _vi4) = build_valid_tensors(vs, DEV)
    masks = (m12, m123, d4k, d4m)

    def step(d):
        am = d["attention_mask"]
        enc = t5.encoder(input_ids=d["input_ids"], attention_mask=am,
                         return_dict=True).last_hidden_state
        ssm_beam(model, enc, ~am.bool(), args.beam, masks)
    return step, pipe, "bf16-autocast"


def _build_earn():
    """code/earn_bucketdiag.py: earn_baseline.build_from_ckpt (fp32) + EARN.generate with the
    CONSTRAINED valid-SID Trie callback -- the deployed decode. NOT code/earn_throughput.py, which
    timed a bf16 UNCONSTRAINED stand-in."""
    args.fp32 = True
    configure_precision(args, DEV)
    pipe = build_tiger_pipeline(args)
    args.lens_path = args.earn_ckpt                   # earn_build_from_ckpt reads args.lens_path
    model = earn_build_from_ckpt(args, pipe)
    n_digit = pipe.tokenizer.n_digit
    eos_tok = int(pipe.tokenizer.eos_token); pad_tok = int(pipe.tokenizer.padding_token)
    vs = find_valid_set(args.dataset_tag, explicit_path=None, lens_path=args.earn_ckpt,
                        graph_dir=args.graph_dir)
    (_m12, _m123, _d4k, _d4m, _k4, _v4, _im, vi4) = build_valid_tensors(vs, DEV)
    trie, d1_list = build_trie(vi4)
    fn = make_prefix_fn(trie, d1_list, eos_tok, pad_tok)

    def step(d):
        model.generate(d["input_ids"], d["attention_mask"], args.beam, n_digit, fn,
                       eos_tok, pad_tok)
    return step, pipe, "fp32"


BUILDERS = {"tigerkv": _build_tigerkv, "sidmlp": _build_sidmlp, "nezha": _build_nezha,
            "gdn": lambda: _build_ssm("gdn"), "mamba2": lambda: _build_ssm("mamba2"),
            "earn": _build_earn}


# --------------------------------------------------------------------------- timing
def to_dev(batch):
    """H2D of the WHOLE collated batch dict -- identical bytes moved for every method."""
    return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_loader(pipe):
    """Fixed, identical test batch order for every method and every rep (shuffle=False)."""
    return DataLoader(pipe.tokenized_datasets["test"], batch_size=args.batch_size, shuffle=False,
                      collate_fn=pipe.tokenizer.collate_fn["test"])


def fingerprint(pipe):
    """Cheap batch-order identity check across methods: hash of the first collated batch +
    split size. Identical across methods <=> they saw the same data in the same order."""
    b = next(iter(make_loader(pipe)))
    h = hashlib.sha256()
    for k in sorted(b.keys()):
        if torch.is_tensor(b[k]):
            raw = b[k].cpu().contiguous().flatten().view(torch.uint8)   # dtype-agnostic bytes
            h.update(k.encode()); h.update(raw.numpy().tobytes())
    h.update(str(len(pipe.tokenized_datasets["test"])).encode())
    return h.hexdigest()[:16]


def one_rep(step, pipe):
    """One ISOLATED repetition: fresh warmup, fresh DataLoader, one sync-bracketed pass over the
    full split. Returns the raw rep record -- nothing is averaged inside a rep."""
    it = iter(make_loader(pipe))                                  # warmup: excluded from the clock
    with torch.inference_mode():
        for _ in range(args.warmup):
            try:
                b = next(it)
            except StopIteration:
                break
            step(to_dev(b))
    del it
    torch.cuda.synchronize(DEV)
    torch.cuda.reset_peak_memory_stats(DEV)
    base_mem = torch.cuda.memory_allocated(DEV) / (1024 ** 3)

    loader = make_loader(pipe)
    it = iter(loader)
    total = 0; nb = 0; loader_sec = 0.0; lat = []
    torch.cuda.synchronize(DEV)
    t0 = time.perf_counter()
    # ------------------------------- TIMED REGION START -------------------------------
    with torch.inference_mode():
        while True:
            tl = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                break
            loader_sec += time.perf_counter() - tl
            d = to_dev(batch)                                     # H2D
            B = d["input_ids"].shape[0]
            if args.latency:
                torch.cuda.synchronize(DEV)
                tb = time.perf_counter()
                step(d)
                torch.cuda.synchronize(DEV)
                lat.append((time.perf_counter() - tb) * 1000.0)
            else:
                step(d)
            total += B; nb += 1
            if args.max_batches and nb >= args.max_batches:
                break
    torch.cuda.synchronize(DEV)
    # -------------------------------- TIMED REGION END --------------------------------
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(DEV) / (1024 ** 3)
    thr = total / max(elapsed, 1e-9)
    rec = {"n_samples": total, "n_batches": nb, "elapsed_sec": elapsed,
           "samples_per_s": thr, "ms_per_sample": 1000.0 / max(thr, 1e-9),
           "peak_mem_gb": peak, "mem_baseline_gb": base_mem, "loader_sec": loader_sec}
    if args.latency and lat:
        a = np.array(lat)
        rec.update({"n_requests": int(a.size), "lat_p50_ms": float(np.percentile(a, 50)),
                    "lat_p95_ms": float(np.percentile(a, 95)),
                    "lat_p99_ms": float(np.percentile(a, 99)),
                    "lat_mean_ms": float(a.mean())})
    return rec


def mean_std(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


# --------------------------------------------------------------------------- driver
env = {"gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
       "cuda_version": torch.version.cuda}
print(f"[env] gpu={env['gpu']} torch={env['torch_version']} cuda={env['cuda_version']} "
      f"bs={args.batch_size} beam={args.beam} warmup={args.warmup} reps={args.reps} "
      f"latency={args.latency} methods={sel}", flush=True)

rows = []
for meth in sel:
    print(f"\n=== {meth} :: building (EXCLUDED from the timed region) @ {time.strftime('%T')} ===",
          flush=True)
    tb = time.perf_counter()
    step, pipe, dtype = BUILDERS[meth]()
    fp = fingerprint(pipe)
    print(f"[{meth}] built in {time.perf_counter()-tb:.1f}s dtype={dtype} ckpt={CKPT[meth]} "
          f"n_test={len(pipe.tokenized_datasets['test'])} order_fingerprint={fp}", flush=True)

    reps = []
    for r in range(args.reps):
        rec = one_rep(step, pipe); rec["rep"] = r + 1
        reps.append(rec)
        msg = (f"[{meth}] rep {r+1}/{args.reps}: {rec['samples_per_s']:.1f} samples/s "
               f"({rec['ms_per_sample']:.4f} ms/sample) n={rec['n_samples']} "
               f"elapsed={rec['elapsed_sec']:.2f}s peak={rec['peak_mem_gb']:.3f}GB "
               f"loader={rec['loader_sec']:.2f}s")
        if args.latency and "lat_p50_ms" in rec:
            msg += (f" | p50={rec['lat_p50_ms']:.3f} p95={rec['lat_p95_ms']:.3f} "
                    f"p99={rec['lat_p99_ms']:.3f} mean={rec['lat_mean_ms']:.3f} ms/req")
        print(msg, flush=True)

    out = {"script": "speed_s48.py", "tag": args.tag, "method": meth, "category": args.category,
           "dataset": args.dataset, "dataset_tag": args.dataset_tag, "split": "test",
           "checkpoint": CKPT[meth], "teacher_checkpoint": args.checkpoint,
           "batch_size": args.batch_size, "beam": args.beam, "warmup": args.warmup,
           "reps": args.reps, "max_batches": args.max_batches, "latency": bool(args.latency),
           "dtype": dtype, "timed_region": TIMED_REGION,
           "n_samples": reps[0]["n_samples"], "n_batches": reps[0]["n_batches"],
           "order_fingerprint": fp, **env, "reps_detail": reps}
    for key in ("samples_per_s", "ms_per_sample", "peak_mem_gb", "elapsed_sec"):
        mu, sd = mean_std([x[key] for x in reps])
        out[f"{key}_mean"] = mu; out[f"{key}_std"] = sd
    if args.latency and "lat_p50_ms" in reps[0]:
        for key in ("lat_p50_ms", "lat_p95_ms", "lat_p99_ms", "lat_mean_ms"):
            mu, sd = mean_std([x[key] for x in reps])
            out[f"{key}_mean"] = mu; out[f"{key}_std"] = sd
        out["note"] = ("latency mode: per-request cuda.synchronize brackets perturb the aggregate "
                       "wall clock -- take samples_per_s only from a non-latency run")
    name = (f"speed_s48_{args.category}_{meth}" + (f"_{args.tag}" if args.tag else "")
            + (".bs1" if args.latency else "") + ".json")
    path = os.path.join(args.out, name)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[wrote] {path}", flush=True)
    rows.append(out)

    del step, pipe
    gc.collect(); torch.cuda.empty_cache()

# ---- table ----
print(f"\n=== S48 SPEED {args.category} (bs={args.batch_size} beam={args.beam} "
      f"reps={args.reps} warmup={args.warmup}{' LATENCY' if args.latency else ''}) ===")
hdr = f"{'method':<9} {'dtype':<14} {'samples/s (mean+-std)':>24} {'ms/sample':>12} {'peak GB':>9}"
if args.latency:
    hdr += f" {'p50':>8} {'p95':>8} {'p99':>8} {'mean ms':>8}"
hdr += "   per-rep samples/s"
print(hdr)
for o in rows:
    line = (f"{o['method']:<9} {o['dtype']:<14} "
            f"{o['samples_per_s_mean']:>15.1f} +- {o['samples_per_s_std']:>5.1f} "
            f"{o['ms_per_sample_mean']:>12.4f} {o['peak_mem_gb_mean']:>9.3f}")
    if args.latency:
        line += (f" {o.get('lat_p50_ms_mean', float('nan')):>8.3f} "
                 f"{o.get('lat_p95_ms_mean', float('nan')):>8.3f} "
                 f"{o.get('lat_p99_ms_mean', float('nan')):>8.3f} "
                 f"{o.get('lat_mean_ms_mean', float('nan')):>8.3f}")
    line += "   " + ", ".join(f"{x['samples_per_s']:.1f}" for x in o["reps_detail"])
    print(line)
fps = {o["method"]: o["order_fingerprint"] for o in rows}
print(f"order_fingerprints={fps} "
      f"({'IDENTICAL' if len(set(fps.values())) <= 1 else '!!! MISMATCH -- batch order differs !!!'})")
print("S48_SPEED_DONE", flush=True)
