"""s39: EARN throughput (samples/s + peak mem), matched to code/tiger_throughput.py discipline.

Plain HF beam generate (UNCONSTRAINED == the genrec TIGER generate used for the s35 TIGER-kv
denominator), bf16, sync-bracketed full-test timed region (model init + data load excluded).
EARN's speedup vs TIGER-kv comes from the 2-register compressed encoder memory (cheaper decoder
cross-attention over 2 keys instead of ~80); the valid-SID mask is a cheap per-step logit op that
BOTH the teacher and EARN pay equally, so the plain-generate ratio is the faithful speed comparison
(and avoids the Python prefix_allowed_tokens_fn callback that would dominate a constrained run).
Compare the reported throughput to the s35 TIGER-kv 570/s (H20, bf16, same harness discipline).
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.modeling_outputs import BaseModelOutput

R = os.environ.get("SERVESID_ROOT", ".")
sys.path.insert(0, f"{R}/repos/SID-MLP")
sys.path.insert(0, f"{R}/repos")
sys.path.insert(0, f"{R}/code")
os.chdir(f"{R}/repos")
for k in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[k] = "1"
from sid_mlp.inference import build_tiger_pipeline, configure_precision
from earn_baseline import EARN

DEV = torch.device("cuda")

ap = argparse.ArgumentParser()
ap.add_argument("--category", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--lens_path", required=True)
ap.add_argument("--sem_ids", required=True)
ap.add_argument("--dataset_tag", required=True)
ap.add_argument("--graph_dir", required=True)
ap.add_argument("--dataset", default="AmazonReviews2023")
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--beam_size", type=int, default=50)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--max_batches", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="earn")
args = ap.parse_args()
args.fp32 = False
native_bf16 = configure_precision(args, DEV)                 # bf16 to match tiger_throughput.py
pipe = build_tiger_pipeline(args)
t5 = pipe.model.to(DEV).t5
ck = torch.load(args.lens_path, map_location="cpu")
model = EARN(t5, ck["cfg"]["k"]); model.load_state_dict(ck["model_state_dict"])
model = model.to(DEV).eval()
if native_bf16:
    model = model.bfloat16()
n_digit = pipe.tokenizer.n_digit
eos_tok = int(pipe.tokenizer.eos_token); pad_tok = int(pipe.tokenizer.padding_token)
ds = pipe.tokenized_datasets["test"]
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=pipe.tokenizer.collate_fn["test"])


@torch.inference_mode()
def gen(ii, am):
    reg, reg_mask = model.encode(ii, am)
    return model.t5.generate(
        encoder_outputs=BaseModelOutput(last_hidden_state=reg), attention_mask=reg_mask,
        max_new_tokens=n_digit + 1, num_beams=args.beam_size, num_return_sequences=args.beam_size,
        use_cache=True, decoder_start_token_id=0, eos_token_id=eos_tok, pad_token_id=pad_tok)


it = iter(loader)
with torch.inference_mode():
    for _ in range(args.warmup):
        try:
            b = next(it)
        except StopIteration:
            break
        gen(b["input_ids"].to(DEV), b["attention_mask"].to(DEV))
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
total = 0; t0 = time.time()
with torch.inference_mode():
    for bi, b in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        ii = b["input_ids"].to(DEV); am = b["attention_mask"].to(DEV)
        gen(ii, am); total += ii.shape[0]
torch.cuda.synchronize()
el = time.time() - t0; thr = total / max(el, 1e-6); peak = torch.cuda.max_memory_allocated() / 1e9
out = {"tag": args.tag, "category": args.category, "k": model.k, "batch_size": args.batch_size,
       "beam": args.beam_size, "total": total, "elapsed_sec": el, "throughput": thr,
       "ms_per_sample": 1000.0 / max(thr, 1e-6), "peak_mem_gb": peak,
       "dtype": "bf16" if native_bf16 else "fp32"}
json.dump(out, open(args.out, "w"), indent=2)
print("EARN_TPUT", json.dumps(out), flush=True)
