"""Matrix-free Wright–Fisher transitions with reusable Equinox models."""

# Functional Core: public API exports.
from ._operator import Transition, Underdominance
from ._plan import TransitionPlan, build_transition_plan
from ._solve import ExpectedVisitsResult

__all__ = [
    "ExpectedVisitsResult",
    "Transition",
    "TransitionPlan",
    "Underdominance",
    "build_transition_plan",
]
