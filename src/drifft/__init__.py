"""Matrix-free Wright–Fisher transition operators."""

# Functional Core: public API exports.
from ._operator import (
    apply_p_banded,
    apply_probabilities,
    expected_next_values,
    expected_segregating_values,
    underdominance,
)
from ._plan import TransitionPlan, build_transition_plan
from ._solve import ExpectedVisitsResult, build_segregating_operator, expected_visits

__all__ = [
    "ExpectedVisitsResult",
    "TransitionPlan",
    "build_segregating_operator",
    "build_transition_plan",
    "expected_next_values",
    "expected_segregating_values",
    "expected_visits",
    "apply_p_banded",
    "apply_probabilities",
    "underdominance",
]
