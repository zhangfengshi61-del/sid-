# MuSID-KD: Structured Distillation for Multi-Token Semantic IDs (WWW 2027)

This repository's **default training overlay is MuSID-KD**. It adapts the
SID-MLP framework to a frozen TIGER teacher and trains lightweight
prefix-conditioned students using:

- **CE:** observed next-item SID-token cross entropy;
- **HCDA:** the equal-weight sum of candidate-prefix marginal KL divergences
  across depths 1–4 (including item-level KL at depth 4);
- **RIGD:** per-depth teacher/student pair-gap Huber alignment, weighted by
  stop-gradient **positive reduction** in margin-Softplus loss when substituting
  only that depth's teacher gap.

The student inference path is unchanged: a frozen TIGER history encoder,
lightweight student context readout and four prefix-conditioned MLP prediction
heads, followed by constrained SID beam search. There is no teacher,
verifier, reranker, or additional KD component at inference.

## Implementation

| Path | Description |
| --- | --- |
| [code/sid-mlp-distill](code/sid-mlp-distill/README.md) | MuSID-KD training and inference overlay |
| [sid_mlp/musid_kd.py](code/sid-mlp-distill/sid_mlp/musid_kd.py) | HCDA / RIGD / training-only scales |
| [configs/musid_kd.yaml](code/sid-mlp-distill/configs/musid_kd.yaml) | Paper coefficients and core hyperparameters |
| [scripts/run_musid_seeds.sh](code/sid-mlp-distill/scripts/run_musid_seeds.sh) | Run independent seeds 42, 43, 44 |
| [tests/test_musid_kd.py](code/sid-mlp-distill/tests/test_musid_kd.py) | CPU numerical checks |

The overlay is applied to the public SID-MLP upstream checkout at commit
`02d9d8e9e8a8f7c92af63e108215174b5a60b72c`; see the detailed README
for setup. A request-aligned teacher beam dump **including** four per-token
log probabilities is mandatory. The training loader ranks cached candidates by
the **sum of SID token log probabilities without EOS**, and reorders codes
and per-token log probabilities together.

Training uses at most 64 teacher candidates and rank 1–10 vs 11–40 pairs.
HCDA temperature is 1, and coefficient 4. RIGD coefficient is 31.42,
margin 0.05 and unit-threshold Huber. Its standard deviations use only
teacher pair gaps from training requests. Online beam width is 50.
Checkpoints are selected by validation NDCG@10, not test data.

The included script launches three seeds (42, 43, 44). **No actual
three-seed training results are included or implied by this code update.**
Per-dataset SID mappings, teacher caches and validation/test partitions
must match those used in the paper to claim reproduction, and the existing
repository does not supply all raw data and teacher checkpoints.

## Historical artifact

Earlier ServeSID code, fixed manifests, AAAI artifact files and analysis
scripts are preserved for provenance, not represented as results for
MuSID-KD. The older configuration remains explicitly selectable at
`code/sid-mlp-distill/configs/servesid_final.yaml`.
