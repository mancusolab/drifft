# Functional Core: deterministic numerical contract tests.
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expit, logit
from scipy.stats import binom

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("M", [100, 1000, 10000])
@pytest.mark.parametrize("fraction", [0.1, 0.5, 0.9])
def test_local_identity(M, fraction):
    from drifft._operator import apply_block
    from drifft._plan import make_block

    j0 = int(M * fraction)
    sigma = np.sqrt(j0 * (1 - j0 / M))
    lo, hi = max(0, j0 - int(3 * sigma)), min(M, j0 + int(4 * sigma))
    rlo, rhi = -3, 4
    block = make_block(M, j0, lo, hi, rlo, rhi)
    v = np.random.default_rng(7).normal(size=hi - lo + 1)
    p = expit(logit(j0 / M) + np.arange(rlo, rhi + 1) / sigma**2)
    exact = binom.pmf(np.arange(lo, hi + 1)[None, :], M, p[:, None]) @ v
    actual = jax.jit(apply_block)(block, jnp.asarray(v))
    np.testing.assert_allclose(actual, exact, atol=2e-11, rtol=0)


def test_noncentered_window_and_no_wraparound():
    from drifft._operator import apply_block
    from drifft._plan import make_block

    for hi in [57, 58]:
        block = make_block(100, 50, 47, hi, 2, 8)
        v = jnp.zeros(hi - 46).at[-1].set(1)
        p = expit(np.arange(2, 9) / 25)
        np.testing.assert_allclose(
            apply_block(block, v), binom.pmf(hi, 100, p), atol=1e-14
        )


def test_optional_balancing_preserves_the_operator():
    from drifft._operator import apply_block
    from drifft._plan import make_block

    v = jnp.asarray(np.random.default_rng(8).normal(size=161))
    balanced = make_block(1000, 128, 58, 218, -8, 8, balance=True)
    unbalanced = make_block(1000, 128, 58, 218, -8, 8, balance=False)
    np.testing.assert_allclose(
        apply_block(balanced, v), apply_block(unbalanced, v), atol=2e-13
    )
