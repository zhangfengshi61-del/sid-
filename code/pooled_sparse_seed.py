"""s35: pooled 3-seed sparse-bin (gt_pop 0-15) paired lkpf-vs-r0base test on the bucketdiag npz.
Per-user N@10 from s_rank (1/log2(rank+2), rank>=0 else 0); hit@10 = s_rank>=0.
Pool paired deltas across seeds -> bootstrap CI; McNemar (exact binom on discordant) on hit@10.
Run: uv run --with numpy,scipy python code/pooled_sparse_seed.py"""
import numpy as np
from scipy.stats import binomtest
rng=np.random.default_rng(42)
def ndcg10(r): return np.where(r>=0, 1.0/np.log2(r.astype(float)+2), 0.0)
def load(cat,arm,seed):
    suf='' if seed=='s42' else f'.{seed}'
    return np.load(f'runs/bucketdiag_pu_{cat}_{arm}{suf}.npz')
def analyze(cat, seeds):
    pooled=[]; b_tot=c_tot=0; n=0; per=[]
    for s in seeds:
        L=load(cat,'lkpf',s); R=load(cat,'r0base',s)
        gp=R['gt_pop']; m=(gp>=0)&(gp<=15)
        d=ndcg10(L['s_rank'])[m]-ndcg10(R['s_rank'])[m]; pooled.append(d)
        hl=(L['s_rank'][m]>=0); hr=(R['s_rank'][m]>=0)
        b=int((hr&~hl).sum()); c=int((hl&~hr).sum()); b_tot+=b; c_tot+=c; n+=int(m.sum())
        p=binomtest(min(b,c),b+c,0.5).pvalue if b+c else 1.0
        per.append(f"{s}: n={int(m.sum())} d={d.mean():+.6f} hit {int(hr.sum())}->{int(hl.sum())} b={b} c={c} McN={p:.4f}")
    D=np.concatenate(pooled)
    bs=np.array([rng.choice(D,len(D),replace=True).mean() for _ in range(20000)])
    ci=(np.percentile(bs,2.5),np.percentile(bs,97.5))
    pM=binomtest(min(b_tot,c_tot),b_tot+c_tot,0.5).pvalue
    print(f"=== {cat} pooled {seeds} ==="); [print("  "+l) for l in per]
    print(f"  POOLED n={n} meanD={D.mean():+.6f} CI[{ci[0]:+.6f},{ci[1]:+.6f}] McNemar b={b_tot} c={c_tot} p={pM:.2e}")
if __name__=='__main__':
    analyze('Musical_Instruments',['s42','s123','s44'])
    analyze('Industrial_and_Scientific',['s42','s123','s44'])
