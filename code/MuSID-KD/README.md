# MuSID-KD (WWW 2027) — paper implementation

This is the main **MuSID-KD** implementation. The Python package is now
`musid_kd`, the main command is `musid-kd`, and generated models/logs
are written under `MuSID-KD/`. The old in-repository `sid_mlp/` directory
is no longer the project entrypoint.

## Structure

- `musid_kd/objectives.py`: HCDA, RIGD, train-set-only pair-gap scales.
- `musid_kd/train_decoder.py`: objective wiring and best validation checkpoint.
- `musid_kd/models.py`: MuSIDStudentMLP, MuSIDStudentGRU, MuSIDStudentCNN.
- `musid_kd/dataset.py`: jointly reorder scores, SID codes and teacher step logps.
- `musid_kd/inference.py`: unchanged constrained-SID beam semantics.
- `musid_kd/cli/train.py`: primary `musid-kd` subcommand.
- `configs/musid_kd.yaml`: paper coefficients and hyperparameters.
- `scripts/run_musid_seeds.sh`: seeds 42, 43, 44.
- `tests/test_musid_kd.py`: CPU objective checks.

## Mathematical objective

`L = CE + 4 * HCDA + 31.42 * RIGD`.

HCDA marginalizes candidate-item probabilities to all four prefix depths.
RIGD uses teacher top-10 positives and rank 11–40 negatives, with
counterfactual positive reduction in margin-Softplus pair loss and a scaled,
stop-gradient weighted per-depth Huber term. Only positions after the first
difference and before the final SID digit are corrected. Input teacher
step-log-probabilities must be aligned with candidate SID tokens; item scores
sum only the four SID token log probabilities, not EOS.

## Dependencies and runs

MuSID-KD **builds on** upstream SID-MLP at commit
`02d9d8e9e8a8f7c92af63e108215174b5a60b72c`. The upstream
`sid_mlp.constants`, `sid_mlp.beam`, `sid_mlp.pretrain_encoder` modules
and `genrec` pipeline remain third-party dependencies, not our project name.
Three small bridge modules make those dependencies explicit without renaming
their API. Use the upstream checkout as `SID_MLP_DIR`; **do not copy**
our `musid_kd/` package over the upstream `sid_mlp/` package.

```bash
export SID_MLP_DIR=/path/to/upstream/SID-MLP
export GRAPH_DIR=/path/to/packed_teacher_cache
export MODEL_DIR=/path/to/teacher_and_semantic_id_assets
export BEAM_DUMP=/path/to/aligned_beam_dump_with_step_logp.pt
export DATASET=AmazonReviews2014
export READOUT=mlp
bash code/MuSID-KD/scripts/run_musid_seeds.sh Beauty 0
```

The script launches three independent runs (42, 43, 44) with distinct
checkpoint filenames and saves the best validation NDCG@10 checkpoint for
each run. Repeat for other dataset/student combinations using matching
teacher, SID mapping, data splits and valid SID constraints. Published
scores are not hardcoded, and no three-seed training has been executed
by this repository rename.

Objective tests (requires PyTorch):

```bash
PYTHONPATH="$PWD/code/MuSID-KD" python -m unittest discover \
  -s code/MuSID-KD/tests -p 'test_*.py'
```

Historical baseline configurations are archived in `configs/legacy/`.
For historical ServeSID use `CONFIG=.../configs/legacy/servesid_final.yaml`
and explicitly run the `sidmlp` compatibility subcommand. Baseline
architecture and historical identifiers still retain their original
scientifically accurate names.
