"""Dense occupation references and solver failure diagnostics."""

# Functional Core: dense reference and solver failure contracts.
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from scipy.stats import binom

jax.config.update("jax_enable_x64", True)


# Sweep selection on one small chain; larger chains check size-dependent solves.
# test_large.py separately exercises the genuinely truncated spectral solve.
@pytest.mark.parametrize(
    "M,s", [(16, 0.0), (16, 0.001), (16, 0.5), (16, 1.0), (64, 0.001), (256, 0.5)]
)
@pytest.mark.parametrize("backend_name", ["FFT", "NUFFT"])
def test_occupation(M, s, backend_name):
    """Validate visits, total absorption time, and residuals against a dense chain."""
    import drifft as wf

    backend = getattr(wf, backend_name)(M)
    x = np.arange(1, M) / M
    p = (x - s * x * (1 - x)) / (1 - 2 * s * x * (1 - x))
    Q = binom.pmf(np.arange(1, M)[None, :], M, p[:, None])
    rhs = np.eye(1, M - 1).ravel()
    exact = np.linalg.solve(np.eye(M - 1) - Q.T, rhs)
    result = wf.Transition(backend, wf.Underdominance(jnp.asarray(s))).expected_visits(
        rtol=1e-11, atol=1e-12, restart=50
    )
    assert isinstance(result, wf.ExpectedVisitsResult)
    assert bool(result.converged)
    assert result.status == wf.ExpectedVisitsStatus.successful
    assert result.solver_status == lx.RESULTS.successful
    assert float(result.residual_norm) < 2e-10
    g = np.asarray(result.visits)
    np.testing.assert_allclose(g, exact, atol=2e-9, rtol=2e-8)
    np.testing.assert_allclose(g.sum(), exact.sum(), atol=1e-8, rtol=2e-8)
    assert np.min(g) >= -2e-10
    assert np.linalg.norm(g - Q.T @ g - rhs, np.inf) < 2e-10
    u = np.random.default_rng(2).normal(size=M - 1)
    np.testing.assert_allclose(
        wf.Transition(backend, wf.Underdominance(jnp.asarray(s)))
        .as_lineax(segregating=True)
        .T.mv(jnp.asarray(u)),
        Q.T @ u,
        atol=2e-11,
    )


def test_nonconvergence_and_bicgstab():
    """Expose failed convergence and verify the alternative nonsymmetric solver."""
    import drifft as wf

    backend = wf.NUFFT(64)
    transition = wf.Transition(backend, wf.Underdominance(jnp.asarray(0.0)))
    result = transition.expected_visits(max_steps=1, restart=2, throw=False)
    assert not bool(result.converged)
    assert result.status == wf.ExpectedVisitsStatus.linear_solve_failed
    assert result.solver_status != lx.RESULTS.successful
    with pytest.raises(eqx.EquinoxRuntimeError, match="expected visits failed"):
        transition.expected_visits(max_steps=1, restart=2).visits.block_until_ready()
    result = wf.Transition(
        backend, wf.Underdominance(jnp.asarray(0.5))
    ).expected_visits(solver="bicgstab")
    assert bool(result.converged)


def test_complete_occupation_status_checks_residual_and_finiteness(monkeypatch):
    """Reject bad candidates even when the underlying solver reports success."""
    from types import SimpleNamespace

    import drifft as wf
    from drifft._solve import _solve_expected_visits

    def candidate(system, rhs, algorithm, **kwargs):
        """Report success after one action, irrespective of its true residual."""
        return SimpleNamespace(
            value=system.mv(rhs), result=lx.RESULTS.successful, stats={}
        )

    monkeypatch.setattr(lx, "linear_solve", candidate)

    def solve(scale, *, throw):
        """Control a diagonal system to exercise domain acceptance under vmap."""
        operator = lx.FunctionLinearOperator(
            lambda v: (1 - scale) * v, jax.ShapeDtypeStruct((3,), jnp.float64)
        )
        return _solve_expected_visits(operator, 4, throw=throw)

    # The system is scale*I, and the fake candidate is scale*rhs. Only scale=1
    # is accepted in this batch; zero fails its residual, NaN/inf fail finiteness.
    result = eqx.filter_jit(jax.vmap(lambda s: solve(s, throw=False)))(
        jnp.array([1.0, 0.0, jnp.nan, jnp.inf])
    )
    np.testing.assert_array_equal(result.converged, [True, False, False, False])
    expected = (
        wf.ExpectedVisitsStatus.successful,
        wf.ExpectedVisitsStatus.unconverged,
        wf.ExpectedVisitsStatus.nonfinite,
        wf.ExpectedVisitsStatus.nonfinite,
    )
    for index, status in enumerate(expected):
        assert jax.tree.map(lambda x, i=index: x[i], result.status) == status
    assert jnp.all(result.solver_status == lx.RESULTS.successful)
    with pytest.raises(eqx.EquinoxRuntimeError, match="expected visits failed"):
        eqx.filter_jit(lambda s: solve(s, throw=True))(
            jnp.asarray(0.0)
        ).visits.block_until_ready()
