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
    apply = jax.jit(lambda s, v: wf.expected_next_values(plan, s, v))
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
        transient = wf.expected_segregating_values(
            plan, jnp.asarray(s), jnp.ones(M - 1)
        )
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
        return wf.expected_segregating_values(plan, s, v)

    compiled = jax.jit(apply)
    u, v = np.random.default_rng(1).normal(size=(2, M - 1))
    for s in [0.0, 0.5, 1.0]:
        compiled(jnp.asarray(s), jnp.asarray(v)).block_until_ready()
    assert len(traces) == 1
    s = jnp.asarray(0.2)
    np.testing.assert_allclose(
        compiled(s, 2 * u - 3 * v), 2 * compiled(s, u) - 3 * compiled(s, v), atol=2e-13
    )
    qt = wf.build_segregating_operator(plan, s).T
    np.testing.assert_allclose(
        u @ compiled(s, v), qt.mv(jnp.asarray(u)) @ v, atol=1e-12
    )
    mapped = jax.jit(
        jax.vmap(lambda s: wf.expected_segregating_values(plan, s, jnp.asarray(v)))
    )(jnp.array([0.0, 0.5, 1.0]))
    for i, s in enumerate([0.0, 0.5, 1.0]):
        np.testing.assert_allclose(mapped[i], compiled(jnp.asarray(s), v), atol=1e-12)


def test_seams_and_truncation():
    from scipy.special import expit

    import drifft as wf

    plan = wf.build_transition_plan(4096)
    eta = np.asarray(plan.edges)
    p = np.r_[
        expit(eta - 1e-10),
        expit(eta),
        expit(eta + 1e-10),
        128 / 4096 + np.array([-1e-12, 0, 1e-12]),
    ]
    v = np.random.default_rng(2).normal(size=4097)
    actual = jax.jit(wf.apply_probabilities)(plan, jnp.asarray(p), jnp.asarray(v))
    exact = binom.pmf(np.arange(4097)[None, :], 4096, p[:, None]) @ v
    np.testing.assert_allclose(actual, exact, atol=2e-9, rtol=0)
    for bucket in plan.buckets:
        assert np.all(np.isfinite(bucket.a))
        assert np.all(np.isfinite(bucket.b))
    assert float(plan.max_tail_mass) <= plan.tail_tol * 1.01


def test_banded_reference():
    import drifft as wf

    M = 256
    v = np.random.default_rng(3).normal(size=M + 1)
    x = np.arange(M + 1) / M
    exact = binom.pmf(np.arange(M + 1)[None, :], M, x[:, None]) @ v
    np.testing.assert_allclose(
        jax.jit(lambda v: wf.apply_p_banded(M, 0.0, v))(jnp.asarray(v)),
        exact,
        atol=2e-11,
        rtol=0,
    )


def test_sampled_kernel_coefficients_and_negative_mass():
    from scipy.special import expit

    import drifft as wf

    M = 4096
    plan = wf.build_transition_plan(M)
    edges = np.asarray(plan.edges)
    p = jnp.asarray(
        np.r_[
            128 / M - 1e-12,
            128 / M + 1e-12,
            expit(edges[[0, len(edges) // 2, -1]] + 1e-10),
        ]
    )
    transpose = jax.linear_transpose(
        lambda v: wf.apply_probabilities(plan, p, v), jnp.zeros(M + 1)
    )
    actual = np.asarray(jax.jit(jax.vmap(lambda u: transpose(u)[0]))(jnp.eye(len(p))))
    exact = binom.pmf(np.arange(M + 1)[None, :], M, np.asarray(p)[:, None])
    error = np.sum(np.abs(actual - exact), axis=1).max()
    negative_mass = np.sum(np.maximum(-actual, 0), axis=1).max()
    print(f"sampled kernel row L1 error={error:.3g}, negative mass={negative_mass:.3g}")
    assert error < 1e-9
    assert negative_mass < 1e-10
