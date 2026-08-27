# baselines — speed-baseline reproductions

Faithful reimplementations of the decoding-acceleration baselines we compare
against, each following the recipe in the SID-MLP paper's Appendix D, adapted to the
TIGER-T5 teacher (4.59M params, 4 constrained SID digits, valid-SID beam-50). These
are standalone scripts; they depend on the public upstream `SID-MLP` / `genrec`
checkout for the teacher pipeline (set `SERVESID_ROOT` to a directory containing
`repos/SID-MLP` and `code/`).

## Baseline implementations
- `atspeed_baseline.py` — AtSpeed-S (strict tree verification, lossless == teacher)
  and AtSpeed-R (relaxed rejection sampling, lossy) speculative decoding
  (AtSpeed, arXiv:2410.05165). Also emits the same-harness TIGER-kv throughput
  denominator.
- `earn_baseline.py` — EARN register-token encoder compression (App-D D.3).
- `nezha_baseline.py` — NEZHA self-drafting MTP head (App-D D.4).
- `ssm_baseline.py` — GatedDeltaNet / Mamba2 SSM decoders (App-D D.3).

## Evaluation harness
- `bucket_diag.py` — shared per-popularity-bucket (head/mid/tail, train-count
  tertiles) + prefix-diagnostic evaluation of a student vs the teacher.
- `*_bucketdiag.py` — the same per-bucket metrics (R/N@5/10, PS@4, PopGap@10)
  driving each baseline's own decoder.
- `tiger_throughput.py`, `*_throughput.py`, `agg_tput.py` — samples/s + peak-memory
  throughput harness and its aggregator.

The files are the final same-harness snapshots used by the paper pipeline.
