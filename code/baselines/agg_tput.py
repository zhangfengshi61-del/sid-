#!/usr/bin/env python3
"""Aggregate the 3-rep isolated throughput sweep into runs/tput_summary.json + a printed table.
Reads runs/tput_<cat>_<arm>.rep<N>.json (students) + runs/tput_TIGERkv_<cat>.rep<N>.json (teacher).
Speedup = student_throughput / TIGERkv_throughput (same cat), computed per-rep then averaged."""
import glob, json, os, re, statistics as st

RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")  # fallback; overridden below
RUNS = "runs"  # run from repo root

def load(pat):
    out = {}
    for f in glob.glob(os.path.join(RUNS, pat)):
        d = json.load(open(f))
        out.setdefault((d.get("category"), d.get("tag")), {})[d.get("rep")] = d
    return out

# students: tput_<cat>_<arm>.rep<N>.json (tag=arm). teacher: tput_TIGERkv_<cat>.rep<N>.json (tag=TIGERkv)
cells = {}  # (cat, arm) -> {rep: {throughput, peak_mem_gb, n10, r10}}
for f in glob.glob(os.path.join(RUNS, "tput_*.rep*.json")):
    d = json.load(open(f))
    cat, arm = d.get("category"), d.get("tag")
    rep = d.get("rep")
    if rep is None:  # tiger JSONs lack the rep field — parse it from the filename
        m = re.search(r"\.rep(\d+)\.json$", f)
        rep = int(m.group(1)) if m else f
    cells.setdefault((cat, arm), {})[rep] = d

def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return (None, None)
    m = sum(vals)/len(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return (m, s)

# teacher throughput per cat (mean over reps) for speedup denominator
tiger_thr = {}
for (cat, arm), reps in cells.items():
    if arm == "TIGERkv":
        tiger_thr[cat] = agg([r.get("throughput") for r in reps.values()])[0]

summary = {}
for (cat, arm), reps in sorted(cells.items()):
    thr = [r.get("throughput") for r in reps.values()]
    mem = [r.get("peak_mem_gb") for r in reps.values()]
    n10 = [r.get("n10") for r in reps.values() if r.get("n10") is not None]
    tm, ts = agg(thr)
    mm, ms = agg(mem)
    # per-rep speedup then average (student rep vs teacher-mean; teacher speedup=1)
    denom = tiger_thr.get(cat)
    spd = (tm/denom) if (tm and denom) else None
    summary.setdefault(cat, {})[arm] = {"n_reps": len(reps),
                    "thr_mean": round(tm, 1) if tm else None, "thr_std": round(ts, 1) if ts is not None else None,
                    "thr_reps": [round(x, 1) for x in thr if x], "peakmem_mean": round(mm, 3) if mm else None,
                    "ms_per_sample_mean": round(1000.0/tm, 3) if tm else None,
                    "speedup_vs_tigerkv": round(spd, 2) if spd else None,
                    "n10_mean": round(agg(n10)[0], 4) if n10 else None}

json.dump(summary, open(os.path.join(RUNS, "tput_summary.json"), "w"), indent=2)

# printed table
print(f"{'cat':<26}{'arm':<10}{'thr(/s)':>14}{'±std':>8}{'speedup':>9}{'ms/smp':>8}{'peakGB':>8}{'N@10':>8}  reps")
for cat in sorted(summary):
  for arm in sorted(summary[cat]):
    v = dict(summary[cat][arm], category=cat, arm=arm)
    thr = f"{v['thr_mean']:.1f}" if v['thr_mean'] else "NA"
    std = f"{v['thr_std']:.1f}" if v['thr_std'] is not None else ""
    spd = f"{v['speedup_vs_tigerkv']:.2f}x" if v['speedup_vs_tigerkv'] else "1.00x"
    ms = f"{v['ms_per_sample_mean']:.3f}" if v['ms_per_sample_mean'] else "NA"
    mem = f"{v['peakmem_mean']:.3f}" if v['peakmem_mean'] else "NA"
    n10 = f"{v['n10_mean']:.4f}" if v['n10_mean'] else ""
    print(f"{v['category']:<26}{v['arm']:<10}{thr:>14}{std:>8}{spd:>9}{ms:>8}{mem:>8}{n10:>8}  {v['n_reps']}")
print("\nwrote runs/tput_summary.json")
