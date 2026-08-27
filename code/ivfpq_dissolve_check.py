#!/usr/bin/env python
"""
Does standard IVFPQ dissolve the parallel-factorized-SID "O(N) holistic scan"? — rec-sid-eff Exp5.

iter-8 claim (math+literature): RPG/ACERec/MDGR factorized scoring ranks items by
Score(j)=Σ_k log softmax(state_k·cb_k)[c_{j,k}]; the per-digit logZ_k is constant across items, so
ranking = MIPS(state, PQ-reconstruction(item)). The codebooks ARE product quantization (OPQ). So the
"O(N) scan" is un-indexed MIPS over PQ codes; standard FAISS IVFPQ should retrieve it sublinearly.

This MEASURES it (don't conclude 'dead' from math alone). Faithful mapping:
  exact "holistic scan" = IndexPQ (exact ADC over all N, METRIC_INNER_PRODUCT)
  IVFPQ = IVF-pruned ADC.
SID = M=32 subquantizers x 256 codes (matches RPG/ACERec/MDGR 32-digit, 256-codebook SID).
Measure recall@10 (IVFPQ vs exact-ADC top-10) + speedup, sweeping N and nprobe.
CAVEAT: random-Gaussian item/query vectors (neutral test; real learned embeddings usually MORE
clusterable => better ANN recall, so this is conservative-ish). CPU-only.
"""
import argparse, time, json
import numpy as np
import faiss


def recall_at_k(I_test, I_exact, k):
    n = I_test.shape[0]
    hit = 0
    for i in range(n):
        hit += len(set(I_test[i, :k].tolist()) & set(I_exact[i, :k].tolist()))
    return hit / (n * k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=512)         # item embedding dim
    ap.add_argument("--M", type=int, default=32)          # subquantizers = SID digits
    ap.add_argument("--nbits", type=int, default=8)       # 2^8=256 codes/digit
    ap.add_argument("--nq", type=int, default=1000)       # queries (=model states)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--N_list", type=str, default="12101,100000,1000000")
    ap.add_argument("--nprobe_list", type=str, default="1,4,16,64,256")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    np.random.seed(0)
    faiss.omp_set_num_threads(16)
    res = {"args": vars(args), "faiss": faiss.__version__, "runs": []}
    print(f"[ivfpq] faiss={faiss.__version__} D={args.D} M={args.M} codes={2**args.nbits}", flush=True)
    for N in [int(x) for x in args.N_list.split(",")]:
        xb = np.random.randn(N, args.D).astype('float32')
        xq = np.random.randn(args.nq, args.D).astype('float32')
        # exact ADC baseline == genrec "holistic scan" over all N
        ipq = faiss.IndexPQ(args.D, args.M, args.nbits, faiss.METRIC_INNER_PRODUCT)
        ipq.train(xb); ipq.add(xb)
        t0 = time.time(); _, I_exact = ipq.search(xq, args.k); t_exact = (time.time() - t0) / args.nq * 1e3
        # IVFPQ (IVF-pruned ADC)
        nlist = max(64, int(4 * np.sqrt(N)))
        quant = faiss.IndexFlatIP(args.D)
        ivf = faiss.IndexIVFPQ(quant, args.D, nlist, args.M, args.nbits, faiss.METRIC_INNER_PRODUCT)
        ivf.train(xb); ivf.add(xb)
        row = {"N": N, "nlist": nlist, "t_exact_scan_ms_per_q": t_exact, "ivfpq": []}
        print(f"[N={N:>9}] exact-ADC scan: {t_exact:.4f} ms/query (nlist={nlist})", flush=True)
        for nprobe in [int(x) for x in args.nprobe_list.split(",")]:
            if nprobe > nlist:
                continue
            ivf.nprobe = nprobe
            t0 = time.time(); _, I = ivf.search(xq, args.k); t = (time.time() - t0) / args.nq * 1e3
            rec = recall_at_k(I, I_exact, args.k)
            row["ivfpq"].append({"nprobe": nprobe, "recall@10": rec, "ms_per_q": t,
                                 "speedup_vs_scan": t_exact / t if t > 0 else None})
            print(f"   nprobe={nprobe:>4}  recall@10={rec:.3f}  {t:.4f} ms/q  "
                  f"speedup={t_exact/t:6.1f}x", flush=True)
        res["runs"].append(row)
        del xb, xq, ipq, ivf
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[ivfpq] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
