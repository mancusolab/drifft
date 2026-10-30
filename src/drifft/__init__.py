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
    prepare_epoch,
)
from ._evolution.problem import EpochProblem, EpochResult, EvolutionStatus
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
from ._solve import ExpectedVisitsResult, ExpectedVisitsStatus

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
    "EpochProblem",
    "EpochResult",
    "EvolutionStatus",
    "ExpectedVisitsResult",
    "ExpectedVisitsStatus",
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
    "prepare_epoch",
]
