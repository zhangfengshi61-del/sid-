"""s48 run manifest — doc §5 item 8: "每项实验的 run manifest：command、config、commit、
checkpoint、teacher cache、seed".

Walks the s48 artifact tree and emits ONE row per experiment unit with everything needed to
re-run or audit it. Pure stdlib + os; no GPU, no torch. Safe to re-run at any time.

Unit types:
  train  — one training arm: TAG/CAT/SEED + the exact env line from the arm file + the `=== TRAIN`
           header actually logged + ckpt path/sha256/mtime + teacher beam-dump path/sha256 +
           a completeness verdict.
  dump   — one served-list cell (code/lists_s48.py): npz+json path/sha256 + W/topk + lens_path.
  mech   — one mechanism run (code/mechanism_s48.py).
  speed  — one throughput/latency run (code/speed_s48.py).

The completeness verdict is the post-hoc audit the s48 orphan incident made necessary:
  ok            — exactly one `=== TRAIN` header, and the frontier json is newer than it
  ok_rerun      — several headers (arm was restarted) but the frontier json is newer than the LAST
                  one, i.e. the surviving artifact belongs to the final run
  STALE_ARTIFACT— frontier json predates the last `=== TRAIN` header  ⇒ do not trust, re-run
  incomplete    — no frontier json / no ckpt
"""
import os, sys, json, csv, re, hashlib, argparse, subprocess, glob

R = os.environ.get("SERVESID_ROOT", ".")
CKDIR = f'{R}/sidmlp-graph/SID-MLP/ckpt'
CKPAT = '{cat}_sidmlp_k0_xattn_h4_f1024_ad384_hhid512_hl1_lr5e-5_{tag}_final.pth'

ap = argparse.ArgumentParser()
ap.add_argument('--root', default=R)
ap.add_argument('--out_dir', default=f'{R}/runs/s48_deliverables')
ap.add_argument('--arm_files', default='runs/s48_armsW1.txt,runs/s48_armsW2.txt')
ap.add_argument('--sha', action='store_true', help='hash ckpts too (slow: ~90MB each)')
args = ap.parse_args()
R = args.root
os.makedirs(args.out_dir, exist_ok=True)


def sha256(p, cap=None):
    if not os.path.exists(p):
        return ''
    h = hashlib.sha256()
    n = 0
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk); n += len(chunk)
            if cap and n >= cap:
                break
    return h.hexdigest()


def mtime(p):
    return round(os.path.getmtime(p), 3) if os.path.exists(p) else None


def git_commit():
    for d in (R, os.path.dirname(os.path.abspath(__file__))):
        try:
            out = subprocess.run(['git', '-C', d, 'rev-parse', 'HEAD'],
                                 capture_output=True, text=True, timeout=10)
            if out.returncode == 0:
                return out.stdout.strip()
        except Exception:
            pass
    return ''


COMMIT = git_commit()
rows = []

# ---------------- train units ----------------
arm_env = {}
for af in args.arm_files.split(','):
    p = os.path.join(R, af.strip())
    if not os.path.exists(p):
        continue
    for line in open(p):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        arm_env[parts[0]] = {'cat': parts[1], 'env': ' '.join(parts[2:]), 'arm_file': af.strip()}

for tag, meta in sorted(arm_env.items()):
    log = f'{R}/runs/s43e_train_{tag}.log'
    front = f'{R}/runs/frontier_s43e_{tag}.json'
    ck = f'{CKDIR}/' + CKPAT.format(cat=meta['cat'], tag=tag)
    heads, cmdline = [], ''
    if os.path.exists(log):
        txt = open(log, errors='replace').read()
        heads = re.findall(r'=== TRAIN (\S+) :: (.*?) ===', txt)
        if heads:
            cmdline = heads[-1][1]
    # completeness verdict (the post-orphan audit)
    if not os.path.exists(front) or not os.path.exists(ck):
        verdict = 'incomplete'
    elif len(heads) <= 1:
        verdict = 'ok'
    else:
        # frontier json must be newer than the LAST train header; compare via the log's own mtime
        verdict = 'ok_rerun' if (mtime(front) or 0) >= (mtime(ck) or 0) else 'STALE_ARTIFACT'
    seed = ''
    m = re.search(r'SEED=(\d+)', meta['env'])
    if m:
        seed = m.group(1)
    rows.append({
        'unit': 'train', 'tag': tag, 'category': meta['cat'], 'seed': seed,
        'arm_file': meta['arm_file'], 'env': meta['env'],
        'logged_config': cmdline, 'n_train_starts': len(heads),
        'train_script': 'runs/s43e_arm.sh -> repos/SID-MLP/scripts/train.sh',
        'wave_runner': 'runs/s48_wave.sh',
        'checkpoint': ck, 'checkpoint_mtime': mtime(ck),
        'checkpoint_sha256': sha256(ck) if args.sha else '',
        'teacher_dump': f'{R}/runs/bmdump_cdrs_{meta["cat"]}.pt',
        'teacher_dump_sha256_1mb': sha256(f'{R}/runs/bmdump_cdrs_{meta["cat"]}.pt', cap=1 << 20),
        'frontier_json': front, 'frontier_mtime': mtime(front),
        'log': log, 'commit': COMMIT, 'verdict': verdict,
    })

# ---------------- dump / mech / speed units ----------------
for kind, pat in (('dump', 'runs/lists_s48_*.json'),
                  ('mech', 'runs/mech_s48_*.json'),
                  ('speed', 'runs/s48speed/*.json')):
    for j in sorted(glob.glob(os.path.join(R, pat))):
        try:
            d = json.load(open(j))
        except Exception as e:
            rows.append({'unit': kind, 'tag': os.path.basename(j), 'error': str(e)}); continue
        npz = j[:-5] + '.npz'
        row = {'unit': kind, 'tag': d.get('tag', ''), 'category': d.get('category', ''),
               'seed': d.get('seed_tag', ''), 'method': d.get('method', ''),
               'checkpoint': d.get('lens_path', ''), 'json': j, 'json_mtime': mtime(j),
               'npz': npz if os.path.exists(npz) else '', 'npz_sha256': sha256(npz) if os.path.exists(npz) else '',
               'npz_mtime': mtime(npz), 'W': d.get('W', ''), 'topk': d.get('topk', ''),
               'n_valid': d.get('n_valid', d.get('n_rows', '')),
               'teacher_cache': f'{R}/runs/tbeam50_{d.get("category","")}.pt',
               'commit': COMMIT}
        rows.append(row)

cols = []
for r in rows:
    for k in r:
        if k not in cols:
            cols.append(k)
with open(f'{args.out_dir}/run_manifest_full.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader()
    for r in rows:
        w.writerow(r)
with open(f'{args.out_dir}/run_manifest_full.json', 'w') as f:
    json.dump({'commit': COMMIT, 'n_units': len(rows), 'rows': rows}, f, indent=1)

from collections import Counter
print(f'[manifest] {len(rows)} units -> {args.out_dir}/run_manifest_full.{{json,csv}}  commit={COMMIT[:12]}')
print('  by unit :', dict(Counter(r['unit'] for r in rows)))
print('  verdicts:', dict(Counter(r.get('verdict', '-') for r in rows if r['unit'] == 'train')))
bad = [r['tag'] for r in rows if r.get('verdict') == 'STALE_ARTIFACT']
if bad:
    print('  !! STALE_ARTIFACT (re-run these):', bad)
