"""Host-side tolerance validation and explicit backend error accounting."""

# Functional Core: immutable reports and scalar preparation checks, without I/O.
import math
from typing import NamedTuple


class ErrorComponent(NamedTuple):
    """One row-L1 error contribution; None means no established bound.

    Bounds are exact-arithmetic analytic bounds unless the component says
    otherwise. A numerical budget is an allocation, never evidence of a bound.
    Gradient bounds use the coordinate stated in the enclosing report.
    """

    name: str
    bound: float | None
    grad_bound: float | None = None
    budget: float | None = None
    grad_budget: float | None = None


class BackendErrorReport(NamedTuple):
    """Requested operator tolerances and the evidence available for them.

    The value norm is maximum row L1 (equivalently infinity-to-infinity).
    It controls expectation error times the input infinity norm and forward
    mass error times the input L1 norm. Bounds do not promise relative accuracy
    for rare events. Reports describe preparation, not measured runtime errors.
    """

    atol: float
    grad_atol: float | None
    components: tuple[ErrorComponent, ...]
    grad_coordinate: str | None = None

    @property
    def bound(self) -> float | None:
        """Total bound, unavailable if any contribution remains unbounded."""
        if any(part.bound is None for part in self.components):
            return None
        return sum(part.bound for part in self.components if part.bound is not None)

    @property
    def grad_bound(self) -> float | None:
        """Total gradient bound, unavailable without complete accounting."""
        if self.grad_atol is None or any(
            part.grad_bound is None for part in self.components
        ):
            return None
        return sum(
            part.grad_bound for part in self.components if part.grad_bound is not None
        )


def _validate_atol(atol: float, name: str = "atol") -> None:
    """Reject invalid host-side absolute operator tolerances before preparation."""
    if isinstance(atol, bool) or not math.isfinite(atol) or not 0 < atol < 1:
        raise ValueError(f"{name} must be finite and in (0, 1)")


def _validate_gradient(grad_atol: float | None, sensitivity: float | None) -> None:
    """Require an explicit parameter-domain bound for gradient planning."""
    if grad_atol is not None:
        _validate_atol(grad_atol, "grad_atol")
        if sensitivity is None:
            raise ValueError("grad_atol requires an angular_sensitivity bound")
    elif sensitivity is not None:
        raise ValueError("angular_sensitivity requires grad_atol")
    if sensitivity is not None and (
        isinstance(sensitivity, bool)
        or not math.isfinite(sensitivity)
        or sensitivity < 0
    ):
        raise ValueError("angular_sensitivity must be finite and nonnegative")


def _gradient_coordinate(sensitivity: float | None) -> str | None:
    """Describe the caller-specified derivative direction in a report."""
    if sensitivity is None:
        return None
    return f"parameter direction with angular sensitivity <= {sensitivity}"
