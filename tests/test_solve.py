# Functional Core: dense reference and solver failure contracts.
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from scipy.stats import binom

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("M", [16, 64, 256])
@pytest.mark.parametrize("s", [0.0, 0.001, 0.5, 1.0])
def test_occupation(M, s):
    import drifft as wf

    plan = wf.build_transition_plan(M)
    x = np.arange(1, M) / M
    p = (x - s * x * (1 - x)) / (1 - 2 * s * x * (1 - x))
    Q = binom.pmf(np.arange(1, M)[None, :], M, p[:, None])
    rhs = np.eye(1, M - 1).ravel()
    exact = np.linalg.solve(np.eye(M - 1) - Q.T, rhs)
    result = wf.Transition(plan, wf.Underdominance(jnp.asarray(s))).expected_visits(
        rtol=1e-11, atol=1e-12, restart=50
    )
    assert bool(result.converged)
    assert result.status == lx.RESULTS.successful
    assert float(result.residual_norm) < 2e-10
    g = np.asarray(result.visits)
    np.testing.assert_allclose(g, exact, atol=2e-9, rtol=2e-8)
    np.testing.assert_allclose(g.sum(), exact.sum(), atol=1e-8, rtol=2e-8)
    assert np.min(g) >= -2e-10
    assert np.linalg.norm(g - Q.T @ g - rhs, np.inf) < 2e-10
    u = np.random.default_rng(2).normal(size=M - 1)
    np.testing.assert_allclose(
        wf.Transition(plan, wf.Underdominance(jnp.asarray(s)))
        .as_lineax(segregating=True)
        .T.mv(jnp.asarray(u)),
        Q.T @ u,
        atol=2e-11,
    )


def test_nonconvergence_and_bicgstab():
    import drifft as wf

    plan = wf.build_transition_plan(64)
    result = wf.Transition(plan, wf.Underdominance(jnp.asarray(0.0))).expected_visits(
        max_steps=1, restart=2
    )
    assert not bool(result.converged)
    assert result.status != lx.RESULTS.successful
    result = wf.Transition(plan, wf.Underdominance(jnp.asarray(0.5))).expected_visits(
        solver="bicgstab"
    )
    assert bool(result.converged)
