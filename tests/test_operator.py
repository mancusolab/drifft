# Functional Core: full transition and transformation contracts.
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binom

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("M", [16, 64, 256, 1024])
def test_full_and_transient(M):
    import drifft as wf

    plan = wf.build_transition_plan(M)
    apply = jax.jit(lambda s, v: wf.Transition(plan, wf.Underdominance(s)).expect(v))
    rng = np.random.default_rng(42)
    vectors = [
        np.ones(M + 1),
        np.zeros(M + 1),
        rng.normal(size=M + 1),
        rng.uniform(size=M + 1),
        (-1.0) ** np.arange(M + 1),
    ]
    for j in [0, 1, M // 2, M - 1, M]:
        vectors.append(np.eye(1, M + 1, j).ravel())
    x = np.arange(M + 1) / M
    for s in [0.0, 0.001, 0.05, 0.5, 0.95, 1.0]:
        p = (x - s * x * (1 - x)) / (1 - 2 * s * x * (1 - x))
        exact = binom.pmf(np.arange(M + 1)[None, :], M, p[:, None])
        for v in vectors:
            actual = np.asarray(apply(jnp.asarray(s), jnp.asarray(v)))
            assert np.max(np.abs(actual - exact @ v)) / max(1, np.max(np.abs(v))) < 1e-9
            np.testing.assert_array_equal(actual[[0, -1]], v[[0, -1]])
        transient = wf.Transition(
            plan, wf.Underdominance(jnp.asarray(s))
        ).expect_segregating(jnp.ones(M - 1))
        np.testing.assert_allclose(
            transient, exact[1:-1, 1:-1].sum(axis=1), atol=1e-9, rtol=0
        )


def test_dynamic_selection_linearity_vmap_and_transpose():
    import drifft as wf

    M = 1024
    plan = wf.build_transition_plan(M)
    traces = []

    def apply(s, v):
        traces.append(None)
        return wf.Transition(plan, wf.Underdominance(s)).expect_segregating(v)

    compiled = jax.jit(apply)
    u, v = np.random.default_rng(1).normal(size=(2, M - 1))
    for s in [0.0, 0.5, 1.0]:
        compiled(jnp.asarray(s), jnp.asarray(v)).block_until_ready()
    assert len(traces) == 1
    s = jnp.asarray(0.2)
    np.testing.assert_allclose(
        compiled(s, 2 * u - 3 * v), 2 * compiled(s, u) - 3 * compiled(s, v), atol=2e-13
    )
    qt = wf.Transition(plan, wf.Underdominance(s)).as_lineax(segregating=True).T
    np.testing.assert_allclose(
        u @ compiled(s, v), qt.mv(jnp.asarray(u)) @ v, atol=1e-12
    )
    mapped = jax.jit(
        jax.vmap(
            lambda s: wf.Transition(plan, wf.Underdominance(s)).expect_segregating(
                jnp.asarray(v)
            )
        )
    )(jnp.array([0.0, 0.5, 1.0]))
    for i, s in enumerate([0.0, 0.5, 1.0]):
        np.testing.assert_allclose(mapped[i], compiled(jnp.asarray(s), v), atol=1e-12)


def test_sampled_kernel_coefficients_and_negative_mass():
    import drifft as wf

    M = 4096
    plan = wf.build_transition_plan(M)
    p = jnp.array([0.0, 1e-8, 0.01, 0.25, 0.5, 0.9, 1 - 1e-8, 1.0])
    transition = wf.Transition.from_probabilities(plan, p)
    actual = np.asarray(jax.jit(jax.vmap(transition.advance))(jnp.eye(len(p))))
    exact = binom.pmf(np.arange(M + 1)[None, :], M, np.asarray(p)[:, None])
    error = np.sum(np.abs(actual - exact), axis=1).max()
    negative_mass = np.sum(np.maximum(-actual, 0), axis=1).max()
    print(f"sampled kernel row L1 error={error:.3g}, negative mass={negative_mass:.3g}")
    assert error < 1e-9
    assert negative_mass < 1e-10
