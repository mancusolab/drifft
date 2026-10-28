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
    AdditiveSelection,
    Composition,
    DiploidSelection,
    DominanceSelection,
    DominantSelection,
    FrequencyMap,
    MultiplicativeSelection,
    Mutation,
    Neutral,
    Overdominance,
    RecessiveSelection,
    Underdominance,
)
from ._jacobi import Jacobi
from ._krylov import PolynomialKrylov, RationalKrylov
from ._operator import Transition
from ._solve import ExpectedVisitsResult
from .backends import FFT, NUFFT, AbstractTransitionBackend

__all__ = [
    "AbstractEpochMethod",
    "AbstractFrequencyMap",
    "AbstractTransitionBackend",
    "AdditiveSelection",
    "Composition",
    "DemographicResult",
    "DemographicTransition",
    "Demography",
    "DiploidSelection",
    "DominanceSelection",
    "DominantSelection",
    "EpochResult",
    "EvolutionStatus",
    "ExpectedVisitsResult",
    "FFT",
    "FrequencyMap",
    "IntervalDiagnostics",
    "Jacobi",
    "MultiplicativeSelection",
    "Mutation",
    "NUFFT",
    "Neutral",
    "Overdominance",
    "PolynomialKrylov",
    "RationalKrylov",
    "RecessiveSelection",
    "Sequential",
    "Transition",
    "Underdominance",
]
