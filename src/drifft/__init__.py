"""Wright–Fisher transitions with reusable Equinox models."""

# pattern: Functional Core
from ._backend import FFT, NUFFT, AbstractTransitionBackend, Dense, Jacobi
from ._demography import (
    DemographicResult,
    DemographicTransition,
    Demography,
    IntervalDiagnostics,
)
from ._evolution.methods import (
    AbstractEpochMethod,
    PolynomialKrylov,
    RationalKrylov,
    Sequential,
)
from ._evolution.problem import EpochResult, EvolutionStatus
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
from ._operator import Transition
from ._solve import ExpectedVisitsResult

__all__ = [
    "AbstractEpochMethod",
    "AbstractFrequencyMap",
    "AbstractTransitionBackend",
    "AdditiveSelection",
    "Composition",
    "DemographicResult",
    "DemographicTransition",
    "Demography",
    "Dense",
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
