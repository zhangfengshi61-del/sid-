# Environment and compute

## Reviewer-side check

The independent metric check requires Python 3.10+ and NumPy. The exact
reviewer requirements are in `requirements-review.txt`.

## Original training/evaluation environment

The final method and paper pipeline were executed with:

| Package | Version |
|---|---:|
| Python | 3.10 |
| PyTorch | 2.7.1 |
| Transformers | 4.53.0 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| PyYAML | 6.0.2 |
| tqdm | 4.67.1 |
| FAISS GPU | 1.7.2 |
| flash-linear-attention | 0.5.1 |

GPU experiments used NVIDIA H20 accelerators. Efficiency tables carry their
own batch size, beam width, precision, warm-up, and measurement scope in the
delivery metadata; do not compare timings across rows without those qualifiers.

`requirements-full.txt` records pip-installable packages. FAISS GPU should be
installed through the CUDA-compatible conda channel. The SSM baselines alone
require `flash-linear-attention`; the final ServeSID method does not.

## Upstream repositories

- SID-MLP at commit `02d9d8e9e8a8f7c92af63e108215174b5a60b72c`;
- the TIGER/genrec dependencies required by that repository;
- baseline-specific public repositories only when rerunning those baselines.

Third-party source is not duplicated in the anonymous ZIP. The overlay and
standalone scripts under `code/` contain all changes made for this work.
