# s43e harvest + gate analysis. Reads frontier_s43e_<tag>.json (student_jac10 = the headline that
# matches the reference ladder r0base .539/list .656/paironly .667/clean-list+pair .675/lkpf .661).
# Prints the arm table + the DA-CDRS/E3 gate verdicts + the E1 unified-serving-KL comparison.
import json, os, sys
R = os.environ.get("SERVESID_ROOT", ".")
suffix = sys.argv[1] if len(sys.argv) > 1 else ''   # e.g. '' (Sci s42), '_MI', '_s43' ...
arms = ['lk', 'lkpair', 'uinc', 'dauni', 'dacdrs', 'lkpf', 'serveW', 'serveU4']
K = ['student_jac10', 'student_r10', 'student_ndcg10', 'student_r5', 'student_ndcg5',
     'teacher_r10', 'teacher_ndcg10']
rows = {}
for a in arms:
    p = f'{R}/runs/frontier_s43e_s43e_{a}{suffix}.json'
    rows[a] = {k: json.load(open(p)).get(k) for k in K} if os.path.exists(p) else None


def j(a):
    return rows[a]['student_jac10'] if rows.get(a) else None


print(f"{'arm':10} {'jac10':>8} {'R@10':>8} {'N@10':>8} {'R@5':>8} {'N@5':>8}")
for a in arms:
    r = rows[a]
    if r is None:
        print(f'{a:10}  (missing)'); continue
    print(f"{a:10} {r['student_jac10']:8.4f} {r['student_r10']:8.5f} {r['student_ndcg10']:8.5f} "
          f"{r['student_r5']:8.5f} {r['student_ndcg5']:8.5f}")
r0 = next((rows[a] for a in arms if rows[a]), None)
if r0:
    print(f"{'TEACHER':10} {'-':>8} {r0['teacher_r10']:8.5f} {r0['teacher_ndcg10']:8.5f}")

print('\n=== reference-ladder sanity (should reproduce) ===')
print(f"  list-only(lk)={j('lk')}  ~ref .656   |  clean list+pair(lkpair)={j('lkpair')} ~ref .675  |  lkpf={j('lkpf')} ~ref .661")

print('\n=== E3 / DA-CDRS gate (Sci seed42) ===')
if j('dacdrs') is not None and j('lkpair') is not None:
    d = j('dacdrs') - j('lkpair')
    print(f"  [PRIMARY] dacdrs {j('dacdrs'):.4f} vs clean list+pair {j('lkpair'):.4f}: {d:+.4f}  "
          f"(gate >= +0.005 -> {'PASS' if d >= 0.005 else 'FAIL'})")
if j('dacdrs') is not None and j('dauni') is not None:
    print(f"  [attribution] dacdrs {j('dacdrs'):.4f} vs dauni(pair+uniform) {j('dauni'):.4f}: {j('dacdrs')-j('dauni'):+.4f}")
if j('dauni') is not None and j('lkpair') is not None:
    print(f"  [uniform-corr] dauni {j('dauni'):.4f} vs lkpair {j('lkpair'):.4f}: {j('dauni')-j('lkpair'):+.4f}")
if j('uinc') is not None and j('lk') is not None:
    print(f"  [incr-alone] uinc {j('uinc'):.4f} vs list-only {j('lk'):.4f}: {j('uinc')-j('lk'):+.4f}  (s41 cumulative CDRS: 0.672)")

print('\n=== E1 / unified serving-KL (Sci) ===')
if j('lkpf') is not None:
    for v in ['serveW', 'serveU4']:
        if j(v) is not None:
            d = j(v) - j('lkpf')
            print(f"  {v} {j(v):.4f} vs ServeSID/lkpf {j('lkpf'):.4f}: {d:+.4f}  "
                  f"(bar >= lkpf within seed noise ~ -0.003 -> {'PASS' if d >= -0.003 else 'below'})")
