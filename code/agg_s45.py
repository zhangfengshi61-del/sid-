# s45 aggregation: re-derive the serveU4+pair headline + DA delta from the raw s45 frontier JSONs into
# one summary artifact, so paper-trail N.* measures point at a re-derivable file (not a typed value).
# Reads runs/frontier_s43e_s45_*.json (all seeds), writes runs/s45_summary.json.
import json, glob, os, collections

D = os.path.join(os.path.dirname(__file__), '..', 'runs')
D = os.path.abspath(D)
DS = {'Sci': 'Industrial_and_Scientific', 'MI': 'Musical_Instruments', 'VG': 'Video_Games'}


def parse(tag):
    x = tag[len('s45_'):]
    ds = 'MI' if '_MI' in x else ('VG' if '_VG' in x else 'Sci')
    if x.startswith('full'):
        a = 'full'
    elif 'dauni' in x:
        a = 'dauni'
    elif 'serveU4pair' in x:
        a = 'serveU4pair'
    elif x.startswith('serveU4'):
        a = 'serveU4'
    else:
        a = None
    return ds, a


vals = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
for f in sorted(glob.glob(os.path.join(D, 'frontier_s43e_s45_*.json'))):
    tag = os.path.basename(f)[len('frontier_s43e_'):-len('.json')]
    if tag == 'smoke':
        continue
    ds, a = parse(tag)
    if a is None:
        continue
    d = json.load(open(f))
    vals[ds][a]['jac'].append(d['student_jac10'])
    vals[ds][a]['r10'].append(d['student_r10'])
    vals[ds][a]['n10'].append(d['student_ndcg10'])

out = {}
for short, full in DS.items():
    if short not in vals:
        continue
    row = {}
    for a, m in vals[short].items():
        row[a] = {k: (sum(v) / len(v)) for k, v in m.items()}
        row[a]['nseed'] = len(m['jac'])
    if 'serveU4pair' in row and 'serveU4' in row:
        row['pair_gain'] = row['serveU4pair']['jac'] - row['serveU4']['jac']
    if 'full' in row and 'serveU4pair' in row:
        row['da_gain'] = row['full']['jac'] - row['serveU4pair']['jac']
    out[full] = row

with open(os.path.join(D, 's45_summary.json'), 'w') as f:
    json.dump(out, f, indent=1)
for full, row in out.items():
    sp = row.get('serveU4pair', {})
    print('%-28s serveU4pair jac=%.4f (n=%d) | pair_gain=%s da_gain=%s' % (
        full, sp.get('jac', 0), sp.get('nseed', 0),
        ('%.4f' % row['pair_gain']) if 'pair_gain' in row else '-',
        ('%.4f' % row['da_gain']) if 'da_gain' in row else '-'))
print('wrote', os.path.join(D, 's45_summary.json'))
