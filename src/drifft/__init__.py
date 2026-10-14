"""Matrix-free Wright–Fisher transitions with reusable Equinox models."""

# Functional Core: public API exports.
from ._operator import Transition, Underdominance
from ._solve import ExpectedVisitsResult
from .backends import FFT, NUFFT, AbstractTransitionBackend

__all__ = [
    "ExpectedVisitsResult",
    "Transition",
    "AbstractTransitionBackend",
    "FFT",
    "NUFFT",
    "Underdominance",
]
