"""Matrix-free Wright–Fisher transitions with reusable Equinox models."""

# pattern: Functional Core
from ._demography import DemographicTransition, Demography
from ._operator import Transition, Underdominance
from ._solve import ExpectedVisitsResult
from .backends import FFT, NUFFT, AbstractTransitionBackend

__all__ = [
    "Demography",
    "DemographicTransition",
    "ExpectedVisitsResult",
    "Transition",
    "AbstractTransitionBackend",
    "FFT",
    "NUFFT",
    "Underdominance",
]
