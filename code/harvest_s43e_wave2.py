# s43e wave-2 harvest: E1-MI + DA-CDRS Sci 3-seed (s42 from wave-1, s43/s44 from wave-2a) + attribution.
import json, os, statistics
R = os.environ.get("SERVESID_ROOT", ".")


def g(tag):
    p = f'{R}/runs/frontier_s43e_s43e_{tag}.json'
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d.get('student_jac10'), d.get('student_r10'), d.get('student_ndcg10')


print('=== E1 unified serving-KL on MI (bar: serve >= lkpf; ref lkpf hist 0.743) ===')
lkpf = g('lkpf_MI')
for v in ['lkpf_MI', 'serveW_MI', 'serveU4_MI']:
    r = g(v)
    if r:
        extra = f'  (vs lkpf {r[0]-lkpf[0]:+.4f})' if lkpf and v != 'lkpf_MI' else ''
        print(f'  {v:12} jac={r[0]:.4f} R@10={r[1]:.5f} N@10={r[2]:.5f}{extra}')

print('\n=== DA-CDRS Sci 3-seed vs clean list+pair (PRIMARY: mean delta stably > 0, ideally >= +0.005) ===')
seeds = {'s42': ('dacdrs', 'lkpair'), 's43': ('dacdrs_s43', 'lkpair_s43'), 's44': ('dacdrs_s44', 'lkpair_s44')}
deltas = []
for s, (dt, lt) in seeds.items():
    dd, ll = g(dt), g(lt)
    if dd and ll:
        delta = dd[0] - ll[0]; deltas.append(delta)
        print(f'  {s}: dacdrs {dd[0]:.4f} vs lkpair {ll[0]:.4f} = {delta:+.4f}  (R@10 {dd[1]:.5f} vs {ll[1]:.5f})')
if deltas:
    print(f'  >>> MEAN dacdrs-lkpair over {len(deltas)} seeds = {statistics.mean(deltas):+.4f} '
          f'(min {min(deltas):+.4f}, max {max(deltas):+.4f})')

print('\n=== attribution robustness (dacdrs vs dauni; s42 was +0.0108) ===')
for s, (dt, ut) in {'s42': ('dacdrs', 'dauni'), 's43': ('dacdrs_s43', 'dauni_s43')}.items():
    dd, uu = g(dt), g(ut)
    if dd and uu:
        print(f'  {s}: dacdrs {dd[0]:.4f} vs dauni {uu[0]:.4f} = {dd[0]-uu[0]:+.4f}')
