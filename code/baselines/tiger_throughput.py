"""s35 (2026-07-11): TIGER-kv teacher throughput denominator, matched to the SID-MLP
inference.py timed-region discipline (sync-bracketed full-test-split loop; model init +
data load excluded). Also a --latency mode for the bs1 online operating point (p50/p95/p99).

Timed region = tiger.generate() over the full test split (forward + AR beam decode +
valid-SID mask + item lookup, all internal to genrec's TIGER generate). Metric computation
is omitted from the timed region (it is <1% for both teacher and student, and omitting it
here only makes the teacher look marginally faster => our speedup is conservative).

Usage:
  PYTHONPATH=$SID_MLP_DIR:$PROJECT_DIR python tiger_throughput.py \
    --category Musical_Instruments --checkpoint <TIGER.pth> --sem_ids <..sem_ids> \
    --dataset_tag AmazonReviews2023_Musical_Instruments --batch_size 32 --teacher_beam 50 \
    [--latency --batch_size 1 --teacher_beam 10] [--max_batches 5] --out runs/tput_TIGERkv_<cat>.json
"""
import argparse, os, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sid_mlp.inference import build_tiger_pipeline, configure_precision

ap = argparse.ArgumentParser()
ap.add_argument('--category', required=True)
ap.add_argument('--checkpoint', required=True)          # TIGER teacher .pth
ap.add_argument('--dataset', default='AmazonReviews2023')
ap.add_argument('--dataset_tag', default=None)
ap.add_argument('--sem_ids', required=True)
ap.add_argument('--graph_dir', default=os.environ.get('SID_MLP_GRAPH_DIR'))
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--teacher_beam', type=int, default=50)
ap.add_argument('--warmup', type=int, default=2)        # batches excluded from timing
ap.add_argument('--max_batches', type=int, default=0)   # 0 = full split; >0 = smoke
ap.add_argument('--latency', action='store_true')       # per-call synced latency percentiles (online point)
ap.add_argument('--fp32', action='store_true')
ap.add_argument('--tag', default='TIGERkv')
ap.add_argument('--out', default='')
args = ap.parse_args()

dev = torch.device('cuda')
native_bf16 = configure_precision(args, dev)
pipe = build_tiger_pipeline(args)
tiger = pipe.model.to(dev).eval()
if native_bf16:
    tiger.bfloat16()
ds = pipe.tokenized_datasets['test']
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=pipe.tokenizer.collate_fn['test'])
n_batches = len(loader)

def to_dev(batch):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}

# ---- warmup (excluded from timing) ----
it = iter(loader)
with torch.inference_mode():
    for _ in range(args.warmup):
        try:
            b = next(it)
        except StopIteration:
            break
        tiger.generate(to_dev(b), n_return_sequences=args.teacher_beam)
if dev.type == 'cuda':
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)

total = 0
lat = []  # per-call synced latency (ms), only populated in --latency mode
t0 = time.time()
with torch.inference_mode():
    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        d = to_dev(batch)
        B = d['input_ids'].shape[0]
        if args.latency:
            torch.cuda.synchronize(dev)
            tb = time.time()
            tiger.generate(d, n_return_sequences=args.teacher_beam)
            torch.cuda.synchronize(dev)
            lat.append((time.time() - tb) * 1000.0 / max(B, 1))  # ms/sample
        else:
            tiger.generate(d, n_return_sequences=args.teacher_beam)
        total += B
if dev.type == 'cuda':
    torch.cuda.synchronize(dev)
peak = torch.cuda.max_memory_allocated(dev) / (1024 ** 3) if dev.type == 'cuda' else 0.0
elapsed = time.time() - t0
thr = total / max(elapsed, 1e-6)
out = {
    'tag': args.tag, 'category': args.category, 'split': 'test',
    'batch_size': args.batch_size, 'teacher_beam': args.teacher_beam,
    'total': total, 'elapsed_sec': elapsed, 'throughput': thr,
    'ms_per_sample': 1000.0 / max(thr, 1e-6), 'peak_mem_gb': peak,
    'dtype': 'native-bfloat16' if native_bf16 else 'fp32', 'warmup': args.warmup,
}
if args.latency and lat:
    a = np.array(lat)
    out.update({'lat_p50_ms': float(np.percentile(a, 50)),
                'lat_p95_ms': float(np.percentile(a, 95)),
                'lat_p99_ms': float(np.percentile(a, 99)),
                'lat_mean_ms': float(a.mean())})
print(f"{args.tag}: cat={args.category} bs={args.batch_size} beam={args.teacher_beam} "
      f"total={total} throughput={thr:.1f}/s ms/sample={out['ms_per_sample']:.3f} "
      f"peak_mem={peak:.3f}GB dtype={out['dtype']}", flush=True)
if args.latency and lat:
    print(f"  latency ms/sample p50={out['lat_p50_ms']:.2f} p95={out['lat_p95_ms']:.2f} "
          f"p99={out['lat_p99_ms']:.2f} mean={out['lat_mean_ms']:.2f}", flush=True)
if args.out:
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[wrote] {args.out}", flush=True)
