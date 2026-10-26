# MuSID-KD SID-MLP overlay (WWW 2027)

This directory overlays the public SID-MLP commit
`02d9d8e9e8a8f7c92af63e108215174b5a60b72c`.
Copy `sid_mlp/`, `scripts/`, and `configs/` over the upstream checkout,
and point `SID_MLP_DIR` to it. Configure the same frozen teacher,
semantic-ID assignment, packed history/GT caches and valid-SID trie as the
original evaluation protocol.

## Objective and numerical conventions

`L = CE + 4 * HCDA + 31.42 * RIGD`

- **CE**: teacher-forced cross entropy of *observed next-item* SID digits.
- **HCDA**: temperature 1 item-score softmax; group by each prefix
  (positions 1 through 4); average per-request sum of KL(teacher||student).
- **RIGD**: teacher positions 1–10 vs 11–40 (K=10, B=40). For each pair,
  supervised levels start at first differing token and stop at L-1; last
  digit still affects the *full* student gap. A depth's weight is its
  stop-gradient positive loss improvement
  `relu(softplus(0.05-Gs) - softplus(0.05-(Gs-gs_d+gt_d)))`,
  normalized over eligible depths with floor 1e-9.
  Huber(beta=1) acts on `(gs_d-gt_d)/sigma_d`, with
  `sigma_d = max(training-pair std,1e-4)`.
  Pairs with no positive weight still count as valid pairs in the mean.
- Cached teacher `step_logp` must have shape [N, M, 4] aligned to
  `codes`. EOS is **not** included in SID-path scores. Candidate
  re-ranking and top-M truncation always reorder per-token logps together.
- Distillation objectives are training-only. Online beam search is
  unchanged and uses width 50.

MLP, GRU, and CNN variants use the same four prefix-conditioned MLP heads
and frozen token embedding table, changing only the lightweight history
readout. Exact match to all published parameter counts and reported
accuracy requires the original data/caches/checkpoints; those assets
are not in this repository.

## Run

```bash
export SID_MLP_DIR=/path/to/upstream/SID-MLP
export GRAPH_DIR=/path/to/packed_teacher_cache
export MODEL_DIR=/path/to/model_assets
export BEAM_DUMP=/path/to/aligned_teacher_beam_dump_with_step_logp.pt
export DATASET=AmazonReviews2014
export READOUT=mlp
bash code/sid-mlp-distill/scripts/run_musid_seeds.sh Beauty 0
```

For Toys, change category to Toys. For Video Games use the
corresponding `AmazonReviews2023` dataset/category tags; Steam-100K
requires its own packed cache and SID mapping. The wrapper launches
**seeds 42, 43, 44** with separate checkpoint prefixes. Each run
logs epoch validation metrics and saves the best validation NDCG@10
checkpoint plus a final checkpoint; results must be independently
evaluated and aggregated before claiming reported means/SDs.

CPU-only objective checks (after overlay in a checkout with PyTorch):

```bash
PYTHONPATH=/path/to/upstream/SID-MLP python -m unittest discover \
  -s code/sid-mlp-distill/tests -p 'test_*.py'
```

Legacy ServeSID experiments can explicitly set
`CONFIG=code/sid-mlp-distill/configs/servesid_final.yaml`.
