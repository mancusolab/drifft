"""Lineax transient operators and occupation-count solves."""

# Functional Core: operator construction and structured numerical results.
import math
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import lineax as lx


class ExpectedVisitsResult(NamedTuple):
    r"""Occupation counts and convergence diagnostics for the approximate operator.

    **Arguments:**

    - `visits`: Candidate float64 expected visits of length `gene_copies-1`,
      including generation zero. Index zero represents one mutant copy.
    - `status`: Unmodified Lineax result code.
    - `stats`: Unmodified Lineax iteration diagnostics.
    - `residual_norm`: Infinity norm of $(I-Q^T)g-e_1$.
    - `converged`: Successful status, finite candidate and acceptable residual.

    **Returns:**

    An immutable PyTree with the five fields above.

    **Failure Modes:**

    Use `visits` only when `converged` is true. Inspect `status`, `stats` and the
    residual otherwise; increase iteration limits or revisit tolerances before
    retrying. Even a small residual is not an exact-model error certificate.
    """

    visits: jax.Array
    status: lx.RESULTS
    stats: dict[str, jax.Array | int | None]
    residual_norm: jax.Array
    converged: jax.Array


def _expected_visits(
    operator: lx.FunctionLinearOperator,
    gene_copies: int,
    *,
    solver: Literal["gmres", "bicgstab"] = "gmres",
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_steps: int = 2000,
    restart: int = 50,
) -> ExpectedVisitsResult:
    r"""Solve $(I-Q^T)g=e_1$ from one mutant using GMRES or BiCGStab.

    **Arguments:**

    - `operator`: Square segregating-state expectation operator.
    - `gene_copies`: Population size in haploid copies, fixed during the solve.
    - `solver`: `"gmres"` or `"bicgstab"`; no symmetric-matrix solver is valid.
    - `rtol`, `atol`: Finite nonnegative solver tolerances, at least one positive.
      These are separate from the plan's approximation tolerance.
    - `max_steps`: Positive integer iteration limit (outer steps for GMRES).
    - `restart`: Positive integer GMRES subspace size; ignored by BiCGStab.

    **Returns:**

    `ExpectedVisitsResult` with candidate `visits` of shape `(gene_copies-1,)`. Index zero is
    state one. `converged` includes an explicit residual check against
    `atol + rtol`, since the right-hand side has infinity norm one.

    **Raises:**

    `ValueError` for invalid static solver configuration, checked during tracing.

    **Failure Modes:**

    Lineax runs with `throw=False`: numerical failure is returned, not raised.
    Under `vmap`, inspect each result's `converged` flag. Static configuration
    errors reject the whole call. Floating-point paths and iteration counts can
    differ across backends. Global differentiability of solves is not certified.
    """
    if not (math.isfinite(rtol) and math.isfinite(atol)):
        raise ValueError("rtol and atol must be finite")
    if any(
        not isinstance(limit, int) or isinstance(limit, bool)
        for limit in (max_steps, restart)
    ):
        raise ValueError("max_steps and restart must be positive integers")
    if not (
        rtol >= 0 and atol >= 0 and rtol + atol > 0 and max_steps > 0 and restart > 0
    ):
        raise ValueError(
            "Require nonnegative tolerances, positive total tolerance and iteration limits"
        )
    if solver == "gmres":
        algorithm = lx.GMRES(
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            restart=min(restart, gene_copies - 1),
        )
    elif solver == "bicgstab":
        algorithm = lx.BiCGStab(rtol=rtol, atol=atol, max_steps=max_steps)
    else:
        raise ValueError("solver must be 'gmres' or 'bicgstab'")
    transpose = operator.T
    spec = jax.ShapeDtypeStruct((gene_copies - 1,), jnp.float64)
    system = lx.FunctionLinearOperator(lambda v: v - transpose.mv(v), spec)
    rhs = jnp.zeros(gene_copies - 1, dtype=jnp.float64).at[0].set(1.0)
    solution = lx.linear_solve(system, rhs, algorithm, throw=False)
    # Recompute the residual rather than trusting the solver's termination code.
    residual = jnp.max(jnp.abs(system.mv(solution.value) - rhs))
    converged = (
        (solution.result == lx.RESULTS.successful)
        & jnp.all(jnp.isfinite(solution.value))
        & (residual <= atol + rtol)
    )
    return ExpectedVisitsResult(
        solution.value, solution.result, solution.stats, residual, converged
    )
