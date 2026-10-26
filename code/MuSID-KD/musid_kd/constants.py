"""Compatibility bridge to upstream SID-MLP codebook conventions.

These constants are part of the published SID data format. Keeping the import
from the baseline avoids silently changing SID assignments.
"""
from sid_mlp.constants import (CODEBOOK_SIZE, DIGIT_OFFSETS, set_seed,
                               triplet_key_int, quad_key_int, token_to_codebook)
