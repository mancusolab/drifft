"""Matrix-free Wright–Fisher transitions with reusable Equinox models."""

# pattern: Functional Core
from ._demography import DemographicTransition, Demography
from ._frequency import (
    AbstractFrequencyMap,
    Composition,
    FrequencyMap,
    Mutation,
    Neutral,
    Underdominance,
)
from ._operator import Transition
from ._solve import ExpectedVisitsResult
from .backends import FFT, NUFFT, AbstractTransitionBackend

__all__ = [
    "AbstractFrequencyMap",
    "Composition",
    "FrequencyMap",
    "Mutation",
    "Neutral",
    "Demography",
    "DemographicTransition",
    "ExpectedVisitsResult",
    "Transition",
    "AbstractTransitionBackend",
    "FFT",
    "NUFFT",
    "Underdominance",
]
