# Data and third-party materials

## Data

The experiments use public Amazon Reviews 2023 categories. Raw reviews and
interaction histories are not redistributed. The archive includes:

- complete aggregate result tables for all reported method cells;
- complete de-identified per-request ranked-list arrays for the 14 supplemental
  sensitivity cells;
- independent verification outputs and public run manifests.

The per-request arrays contain dataset-local integer item indices only. They do
not contain account identifiers, review text, timestamps, machine paths, or
checkpoint names. See `data/README.md`.

## Source overlays

The final method is distributed as a source overlay on SID-MLP. Unchanged
upstream source remains governed by the upstream repository's license and is
not copied into this archive. The baseline scripts are standalone adaptations
that still require their cited public dependencies.

## Review-use notice

This anonymous archive is supplied for peer review and reproducibility
assessment. It does not add a license to third-party datasets, models, or
repositories, and it does not broaden their original distribution terms.
