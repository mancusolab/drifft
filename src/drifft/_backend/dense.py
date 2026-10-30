"""Independent dense binomial transitions for small-population validation."""

# pattern: Functional Core
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp

from ._base import AbstractTransitionBackend


@final
class Dense(AbstractTransitionBackend):
    r"""Form binomial transition rows for an independent numerical reference.

    For destination gene-copy count $M$ and source probabilities $q_i$, construct

    $$P_{ij}=\binom{M}{j}q_i^j(1-q_i)^{M-j},\qquad j=0,\ldots,M.$$

    Each row is built by repeated Bernoulli convolution, starting at count zero.
    This avoids factorials, logarithms, and divisions by endpoint probabilities.
    The calculation is polynomial throughout, so ordinary JAX differentiation
    gives finite endpoint derivatives without custom rules. It shares no
    numerical kernels with FFT, NUFFT, or Jacobi.

    `expect` explicitly forms the matrix before multiplying by destination
    values. With $N$ source rows, this uses $O(NM)$ matrix storage and $O(NM^2)$
    arithmetic. Reverse-mode differentiation can retain intermediate rows.
    Use this backend for small-population checks of transitions, demographic
    evolution, and gradients; it is not intended for large populations. There
    is no truncation or interpolation, but floating-point roundoff remains.
    """

    gene_copies: int = eqx.field(static=True)

    def __init__(self, gene_copies: int) -> None:
        r"""Configure the destination count grid outside JIT.

        **Arguments:**

        - `gene_copies`: Python integer at least two, equal to twice diploid Ne.

        **Returns:**

        `None`. Only the static count is stored; matrices are formed on application.

        **Raises:**

        `ValueError` for invalid counts or if JAX x64 is disabled.
        """
        if (
            isinstance(gene_copies, bool)
            or not isinstance(gene_copies, int)
            or gene_copies < 2
        ):
            raise ValueError("gene_copies must be a Python integer >= 2")
        if not jax.config.x64_enabled:
            raise ValueError("Dense requires jax_enable_x64=True")
        self.gene_copies = gene_copies

    def expect(self, probabilities: jax.Array, values: jax.Array) -> jax.Array:
        r"""Form binomial rows and multiply by destination values.

        **Arguments:**

        - `probabilities`: Vector of finite source probabilities in `[0, 1]`.
        - `values`: Finite real destination vector of length `gene_copies + 1`.
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
        if probabilities.ndim != 1 or values.shape != (self.gene_copies + 1,):
            raise ValueError(
                "expect requires a probability vector and gene_copies + 1 destination values"
            )
        probability = probabilities[:, None]
        rows = jnp.zeros((probabilities.size, self.gene_copies + 1), dtype=values.dtype)
        rows = rows.at[:, 0].set(1)

        def convolve(_, current):
            """Add one Bernoulli trial without dividing by q or 1-q."""
            shifted = jnp.pad(current[:, :-1], ((0, 0), (1, 0)))
            return (1 - probability) * current + probability * shifted

        matrix = jax.lax.fori_loop(0, self.gene_copies, convolve, rows)
        return matrix @ values
