# Data appendix

This directory contains anonymized derived data for reviewer-side verification.
It does not contain review text, user profiles, raw interaction histories,
machine paths, checkpoints, or infrastructure metadata.

## Included data

`derived/s53/` contains 14 complete per-request experiment cells:

- nine K-sensitivity cells: three training cutoffs on each of Scientific,
  Instruments, and Video Games;
- five controlled B-sensitivity cells on Scientific with the pair and
  depth-aware losses sharing exactly ten negatives.

Each public experiment ID has:

- `.npz`: integer item indices, teacher/student ranked lists, validity flags,
  popularity buckets, and list lengths;
- `.json`: aggregate metrics recomputed from the paired `.npz`.

The 14 cells contain no strings in the NPZ arrays and can be loaded with
`allow_pickle=False`. `index.json` records the public IDs and experimental
protocol.

The complete cross-method aggregate tables are under `delivery/2026-07-22/`;
the supplemental aggregate tables and independent verification summary are
under `delivery/2026-07-27/`.

## Verification

```bash
python code/verify_s53.py --data-dir data/derived/s53
```

This executes an independent NumPy rank/set implementation over every request
and checks 18 metrics per cell.

## Raw-data availability

Experiments use the public Amazon Reviews 2023 categories named in the tables.
Raw review text and interaction records are not redistributed in this archive
because they are large and remain subject to the dataset provider's terms.
The included item indices are dataset-local anonymous integers; they are not
Amazon account identifiers.

For full training, obtain Amazon Reviews 2023 through its public distribution
and follow the upstream TIGER/SID-MLP preprocessing pipeline. The exact
scientific settings used here are recorded in the run manifests and
`code/sid-mlp-distill/configs/servesid_final.yaml`.
