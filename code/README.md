# Paper-facing code inventory

The active training path is now `sid-mlp-distill/` configured with
`configs/musid_kd.yaml`. The MuSID-KD losses are isolated in
`sid_mlp/musid_kd.py` and wired in `sid_mlp/train_decoder.py`.

- HCDA uses shared teacher candidate scores and prefix grouping at all four
  depths; RIGD uses top-10 vs ranks 11–40 pairs and single-depth positive
  Softplus replacement gains.
- MLP, GRU, CNN share the same factorized SID token heads and valid-set
  constrained serving path.
- Best checkpoints are tracked by validation NDCG@10, and
  `scripts/run_musid_seeds.sh` launches seeds 42, 43, 44.
- `sid_mlp/dataset.py` ensures beam-dump codes, scores and per-token
  teacher log probabilities are reordered and truncated together.

Other `code/*.py` scripts and the `baselines/` directory are historical
ServeSID research and controls retained unchanged for provenance. They do
not constitute independently validated MuSID-KD evaluation results.
