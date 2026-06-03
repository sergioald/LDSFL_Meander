"""LDSFL-Meander Python package.

Small runtime defaults to avoid thread oversubscription.
These are defaults only (we use setdefault), so users can override them by
setting environment variables before import.
"""

from __future__ import annotations

import os

__version__ = '0.6.3.1'

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
