"""Independent dense binomial transitions for small-population validation."""

# pattern: Functional Core
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp

from ._accuracy import (
    BackendErrorReport,
    ErrorComponent,
    _gradient_coordinate,
    _validate_atol,
    _validate_gradient,
)
from ._base import AbstractTransitionBackend


@final
class Dense(AbstractTransitionBackend):
    r"""Construct binomial transition rows directly as an independent numerical reference:

    $$
    P_{ij}=\binom{M}{j}q_i^j(1-q_i)^{M-j},\qquad j=0,\ldots,M.
    $$

    Here $M=2N$ for diploid destination population size $N$ (`pop_size`),
    and $q_i$ is the offspring allele probability for source row $i$.
    Multiplying these rows by destination values $v_j$ gives the expectations
    $(Pv)_i=\sum_{j=0}^{M}P_{ij}v_j$. Repeated Bernoulli convolution builds
    each row, avoiding factorials, logarithms, and divisions by endpoint
    probabilities. The calculation is polynomial throughout, so ordinary
    JAX differentiation gives finite endpoint derivatives. It shares no
    numerical kernels with [`FFT`][drifft.FFT], [`NUFFT`][drifft.NUFFT], or [`Jacobi`][drifft.Jacobi]. There is no truncation or
    interpolation, but floating-point roundoff remains.
    With $S$ source rows, matrix construction uses $O(SM)$ storage and
    $O(SM^2)$ arithmetic; use this backend for small-population checks.

    !!! note "Theory"

        See the [Wright–Fisher model](../guide/model.md#one-generation) for
        the binomial transition law.
    """

    pop_size: int = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    grad_atol: float | None = eqx.field(static=True)
    angular_sensitivity: float | None = eqx.field(static=True)

    def __init__(
        self,
        pop_size: int,
        *,
        atol: float = 1e-10,
        grad_atol: float | None = None,
        angular_sensitivity: float | None = None,
    ) -> None:
        r"""Configure the destination count grid outside JIT.

        Only static configuration is stored; matrices are formed on application.

        **Arguments:**

        - `pop_size`: Diploid effective population size, a Python integer at least one.
        - `atol`: Requested one-step maximum row L1 error, in `(0, 1)`.
          Dense has no approximation to tune; roundoff is not certified.
        - `grad_atol`, `angular_sensitivity`: Optional derivative target and
          parameter-domain bound, as in [`NUFFT`][drifft.NUFFT]. Analytic
          approximation error is zero; derivative roundoff remains uncertified.

        **Raises:**

        `ValueError` for invalid counts or if JAX x64 is disabled.
        """
        _validate_atol(atol)
        _validate_gradient(grad_atol, angular_sensitivity)
        if isinstance(pop_size, bool) or not isinstance(pop_size, int) or pop_size < 1:
            raise ValueError("pop_size must be a Python integer >= 1")
        if not jax.config.x64_enabled:
            raise ValueError("Dense requires jax_enable_x64=True")
        self.pop_size = pop_size
        self.atol = float(atol)
        self.grad_atol = None if grad_atol is None else float(grad_atol)
        self.angular_sensitivity = (
            None if angular_sensitivity is None else float(angular_sensitivity)
        )

    @property
    def error_report(self) -> BackendErrorReport:
        """Report zero analytic approximation error and unbounded roundoff."""
        return BackendErrorReport(
            self.atol,
            self.grad_atol,
            (
                ErrorComponent("approximation", 0.0, 0.0),
                ErrorComponent("roundoff", None),
            ),
            _gradient_coordinate(self.angular_sensitivity),
        )

    def expect(self, probabilities: jax.Array, values: jax.Array) -> jax.Array:
        r"""Form binomial rows and multiply by destination values.

        **Arguments:**

        - `probabilities`: Vector of finite source probabilities in `[0, 1]`.
        - `values`: Finite real destination vector of length `2 * pop_size + 1`.
          Both inputs are converted to float64.

        **Returns:**

        One expectation per probability, linear in `values`. Signed values,
        transposition, batching, and higher derivatives are supported.

        **Raises:**

        `ValueError` for incompatible vector shapes. Finiteness and probability
        ranges are caller preconditions; outputs are not clipped or normalized.
        """
        probabilities = jnp.asarray(probabilities, dtype=jnp.float64)
        values = jnp.asarray(values, dtype=jnp.float64)
        if probabilities.ndim != 1 or values.shape != (self._gene_copies + 1,):
            raise ValueError(
                "expect requires a probability vector and 2 * pop_size + 1 destination values"
            )
        probability = probabilities[:, None]
        rows = jnp.zeros(
            (probabilities.size, self._gene_copies + 1), dtype=values.dtype
        )
        rows = rows.at[:, 0].set(1)

        def convolve(_, current):
            """Add one Bernoulli trial without dividing by q or 1-q."""
            shifted = jnp.pad(current[:, :-1], ((0, 0), (1, 0)))
            return (1 - probability) * current + probability * shifted

        matrix = jax.lax.fori_loop(0, self._gene_copies, convolve, rows)
        return matrix @ values
