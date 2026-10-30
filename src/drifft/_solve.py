"""Lineax transient operators and occupation-count solves."""

# Functional Core: operator construction and structured numerical results.
import math
from typing import Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx


class ExpectedVisitsStatus(eqx.Enumeration):
    r"""Complete occupation-solve outcomes, including independent residual checks.

    - `successful`: Finite visits and an accepted residual after a successful solve.
    - `nonfinite`: Nonfinite visits or residual.
    - `unconverged`: The independently recomputed residual exceeds tolerance.
    - `linear_solve_failed`: Lineax reported an unsuccessful solve.

    Compare `ExpectedVisitsResult.status` with these members. Solver failure
    takes precedence over nonfinite output, then residual nonconvergence.
    The original Lineax outcome is retained separately as `solver_status`.
    """

    successful = ""
    nonfinite = (
        "Nonfinite occupation output or residual; check the transition and inputs."
    )
    unconverged = "Occupation residual exceeds tolerance; tighten the solve or increase iteration limits."
    linear_solve_failed = (
        "Occupation linear solve failed; inspect solver_status and stats."
    )


class ExpectedVisitsResult(NamedTuple):
    r"""Occupation counts and convergence diagnostics for the approximate operator.

    The candidate occupation vector solves $(I-Q^T)u=e_1$, including generation
    zero. The convergence flag combines solver status, finiteness, and a residual
    check against the approximate operator.

    **Fields:**

    - `visits`: Float64 vector of expected interior visits, including generation
      zero. Entry zero corresponds to one mutant copy.
    - `status`: `ExpectedVisitsStatus` for the complete calculation.
    - `solver_status`, `stats`: Unmodified Lineax result code and iteration diagnostics.
    - `residual_norm`: Scalar infinity norm of $(I-Q^T)u-e_1$.
    - `converged`: Scalar boolean, true exactly when the domain `status` is
      successful. This includes the solver, finiteness and residual checks.
    """

    visits: jax.Array
    status: ExpectedVisitsStatus
    solver_status: lx.RESULTS
    stats: dict[str, jax.Array | int | None]
    residual_norm: jax.Array
    converged: jax.Array


# Name the defining module so runtime documentation can resolve the generated method.
ExpectedVisitsResult.__new__.__module__ = __name__
# Document the generated constructor without replacing its implementation.
ExpectedVisitsResult.__new__.__doc__ = r"""Construct the result container without numerical validation.

**Arguments:**

- `visits`: Candidate float64 expected visits of length `gene_copies-1`,
  including generation zero. Index zero represents one mutant copy.
- `status`: Complete calculation outcome, an `ExpectedVisitsStatus`.
- `solver_status`: Unmodified Lineax result code.
- `stats`: Unmodified Lineax iteration diagnostics.
- `residual_norm`: Infinity norm of $(I-Q^T)u-e_1$.
- `converged`: Successful status, finite candidate and acceptable residual.

**Returns:**

An immutable result with the supplied fields.

!!! warning "Usage preconditions"

    Use `visits` only when `converged` is true. Inspect `status`, `solver_status`,
    `stats` and the residual otherwise; increase iteration limits or revisit
    tolerances before retrying. Even a small residual is not an exact-model error certificate.
"""


def _solve_expected_visits(
    operator: lx.FunctionLinearOperator,
    gene_copies: int,
    *,
    solver: Literal["gmres", "bicgstab"] = "gmres",
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_steps: int = 2000,
    restart: int = 50,
    throw: bool = True,
) -> ExpectedVisitsResult:
    r"""Solve $(I-Q^T)u=e_1$ from one mutant using GMRES or BiCGStab.

    **Arguments:**

    - `operator`: Square segregating-state expectation operator.
    - `gene_copies`: Population size in haploid copies, fixed during the solve.
    - `solver`: `"gmres"` or `"bicgstab"`; no symmetric-matrix solver is valid.
    - `rtol`, `atol`: Finite nonnegative solver tolerances, at least one positive.
      These are separate from the backend's approximation tolerance.
    - `max_steps`: Positive integer iteration limit (outer steps for GMRES).
    - `restart`: Positive integer GMRES subspace size; ignored by BiCGStab.
    - `throw`: Raise on numerical failure by default; false retains the candidate.

    **Returns:**

    `ExpectedVisitsResult` with candidate `visits` of shape `(gene_copies-1,)`. Index zero is
    state one. `converged` includes an explicit residual check against
    `atol + rtol`, since the right-hand side has infinity norm one.

    **Raises:**

    `ValueError` for invalid static solver configuration, checked during tracing.
    Runtime errors for numerical failure when `throw=True`.

    !!! warning "Numerical limitations"

        With `throw=False`, inspect each result's `converged` flag. With true,
        a failed member under `vmap` may fail the whole batch. Static configuration
        errors reject the whole call. Floating-point paths and iteration counts can
        differ across backends. Global differentiability of solves is not certified.
    """
    # Tolerances are static Python configuration, so math validates them at
    # tracing time; operator parameters and solve vectors stay in JAX.
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
    # Summing visits gives u = e_1 + Q.T u. Q acts on observables, so its
    # transpose is required here; using Q would solve a different occupation law.
    spec = jax.ShapeDtypeStruct((gene_copies - 1,), jnp.float64)
    system = lx.FunctionLinearOperator(lambda v: v - transpose.mv(v), spec)
    rhs = jnp.zeros(gene_copies - 1, dtype=jnp.float64).at[0].set(1.0)
    solution = lx.linear_solve(system, rhs, algorithm, throw=False)
    # Recompute the residual rather than trusting the solver's termination code.
    residual = jnp.max(jnp.abs(system.mv(solution.value) - rhs))
    status = ExpectedVisitsStatus.where(
        residual <= atol + rtol,
        ExpectedVisitsStatus.successful,
        ExpectedVisitsStatus.unconverged,
    )
    status = ExpectedVisitsStatus.where(
        jnp.all(jnp.isfinite(solution.value)) & jnp.isfinite(residual),
        status,
        ExpectedVisitsStatus.nonfinite,
    )
    status = ExpectedVisitsStatus.where(
        solution.result == lx.RESULTS.successful,
        status,
        ExpectedVisitsStatus.linear_solve_failed,
    )
    converged = status == ExpectedVisitsStatus.successful
    visits = solution.value
    if throw:
        visits = eqx.error_if(
            visits,
            ~converged,
            "expected visits failed: inspect status, solver_status and residual_norm "
            "with throw=False; check the transition or increase iteration limits",
        )
    return ExpectedVisitsResult(
        visits, status, solution.result, solution.stats, residual, converged
    )
