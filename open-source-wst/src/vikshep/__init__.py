"""
vikshep
=======
Vectorized Invariant Kernels for Scattering & High-performance Extraction
Pipelines (Vikshep).  A Zero-Copy C++/CUDA Wavelet Scattering Transform
engine for non-stationary time-series analysis.

This package is **domain-agnostic** — it processes any 1-D time-series
(EEG, FRB, seismic, financial) through a cascaded scattering transform
with optional PCA dimensionality reduction.

Public API
----------
>>> from vikshep import WaveletScatteringExtractor, ScatteringResult
>>> extractor = WaveletScatteringExtractor(J=8, Q=(8, 1))
>>> result = extractor.transform(signal_array)

High-Performance Native Engine
------------------------------
>>> from vikshep import NativeWSTExtractor
>>> engine = NativeWSTExtractor(J=8, Q=16, depth=2)
>>> coefficients = engine.fingerprint(signal_array)
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("vikshep")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.0.0.dev"

# ── Core imports (no heavy dependencies) ───────────────────────────────────────
from vikshep.core import (
    WaveletScatteringExtractor,
    ScatteringResult,
    NativeWSTExtractor,
)
from vikshep.utils import compute_snr_db, detect_outlier_paths

# ── Native C++/CUDA extension (optional — available after pip install -e .) ────
try:
    from vikshep import _vikshep_core
except ImportError:
    _vikshep_core = None  # type: ignore[assignment]


# ── Lazy imports for heavy optional dependencies ──────────────────────────────
# PCAReducer depends on sklearn which pulls in pyarrow — defer to access time.
def __getattr__(name: str):
    if name == "PCAReducer":
        from vikshep.reduction import PCAReducer
        return PCAReducer
    if name in ("load_npy_directory", "save_arrays"):
        from vikshep import io as _io
        return getattr(_io, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "WaveletScatteringExtractor",
    "ScatteringResult",
    "NativeWSTExtractor",
    "_vikshep_core",
    "PCAReducer",
    "load_npy_directory",
    "save_arrays",
    "compute_snr_db",
    "detect_outlier_paths",
    "__version__",
]