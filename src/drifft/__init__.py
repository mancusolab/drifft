"""Matrix-free Wright–Fisher transitions with reusable Equinox models."""

# pattern: Functional Core
from ._demography import (
    DemographicResult,
    DemographicTransition,
    Demography,
    IntervalDiagnostics,
)
from ._epochs import AbstractEpochMethod, EpochResult, EvolutionStatus, Sequential
from ._frequency import (
    AbstractFrequencyMap,
    Composition,
    FrequencyMap,
    Mutation,
    Neutral,
    Underdominance,
)
from ._jacobi import Jacobi
from ._krylov import RationalKrylov
from ._operator import Transition
from ._solve import ExpectedVisitsResult
from .backends import FFT, NUFFT, AbstractTransitionBackend

__all__ = [
    "AbstractEpochMethod",
    "AbstractFrequencyMap",
    "AbstractTransitionBackend",
    "Composition",
    "DemographicResult",
    "DemographicTransition",
    "Demography",
    "EpochResult",
    "EvolutionStatus",
    "ExpectedVisitsResult",
    "FFT",
    "FrequencyMap",
    "IntervalDiagnostics",
    "Jacobi",
    "Mutation",
    "NUFFT",
    "Neutral",
    "RationalKrylov",
    "Sequential",
    "Transition",
    "Underdominance",
]
