"""Common transition-backend contracts for Jacobi geometry."""

# pattern: Functional Core
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binom

import drifft

jax.config.update("jax_enable_x64", True)


def test_jacobi_is_a_transition_backend():
    """Expose Jacobi through the same public backend contract and exports."""
    backend = drifft.Jacobi(16)
    assert isinstance(backend, drifft.AbstractTransitionBackend)


def test_jacobi_expect_and_rectangular_transpose_match_binomial():
    """Arbitrary probability rows include endpoints, signed values and derivatives."""
    copies = 16
    backend = drifft.Jacobi(copies)
    q = jnp.array([0.0, 1e-9, 0.1, 0.37, 0.8, 1 - 1e-9, 1.0])
    values = jnp.sin(jnp.arange(copies + 1) * 0.7)
    mass = jnp.cos(jnp.arange(q.size) * 0.3)
    matrix = jnp.asarray(
        binom.pmf(np.arange(copies + 1)[None, :], copies, np.asarray(q)[:, None])
    )
    transition = drifft.Transition(backend, probabilities=q)
    assert jnp.allclose(transition.expect(values), matrix @ values, atol=2e-12)
    assert jnp.allclose(transition.advance(mass), matrix.T @ mass, atol=2e-12)
    assert jnp.allclose(
        mass @ transition.expect(values), transition.advance(mass) @ values, atol=2e-13
    )
    np.testing.assert_array_equal(
        transition.expect(values)[jnp.array([0, -1])], values[jnp.array([0, -1])]
    )
    dmatrix = copies * (
        binom.pmf(
            np.arange(copies + 1)[None, :] - 1, copies - 1, np.asarray(q)[:, None]
        )
        - binom.pmf(np.arange(copies + 1)[None, :], copies - 1, np.asarray(q)[:, None])
    )
    expected = jnp.asarray(dmatrix) @ values
    actual = jax.jit(jax.grad(lambda q: backend.expect(q, values).sum()))(q)
    assert jnp.allclose(actual, expected, atol=2e-9, rtol=2e-10)
    second = (
        copies
        * (copies - 1)
        * (
            binom.pmf(
                np.arange(copies + 1)[None, :] - 2, copies - 2, np.asarray(q)[:, None]
            )
            - 2
            * binom.pmf(
                np.arange(copies + 1)[None, :] - 1, copies - 2, np.asarray(q)[:, None]
            )
            + binom.pmf(
                np.arange(copies + 1)[None, :], copies - 2, np.asarray(q)[:, None]
            )
        )
    )
    hessian = jax.jit(jax.hessian(lambda q: backend.expect(q, values).sum()))(q)
    assert jnp.allclose(
        hessian, jnp.diag(jnp.asarray(second) @ values), atol=2e-8, rtol=2e-10
    )
    batch = jnp.stack((values, -values))
    assert jnp.allclose(
        jax.jit(jax.vmap(backend.expect, in_axes=(None, 0)))(q, batch),
        batch @ matrix.T,
        atol=2e-12,
    )


def test_jacobi_sequential_history_matches_its_actual_transition_transpose():
    """Generic Sequential supports custom maps and default Jacobi boundaries."""

    def backend(copies):
        return drifft.Jacobi(copies, rank=3)

    rule = drifft.FrequencyMap(lambda x: x * x)
    model = drifft.DemographicTransition(
        drifft.Demography([6, 4, 7], [4, 3]),
        backend_factory=backend,
        frequency_map=rule,
    )
    initial = jnp.sin(jnp.arange(13) * 0.3)
    values = jnp.cos(jnp.arange(15) * 0.4)
    expected = initial
    for source, destination, duration in [(12, 8, 4), (8, 14, 3)]:
        hold = drifft.Transition(backend(source), rule)
        for _ in range(duration - 1):
            expected = hold.advance(expected)
        expected = drifft.Transition(
            backend(destination), rule, source_gene_copies=source
        ).advance(expected)
    actual = model.advance(initial).distribution
    assert jnp.allclose(actual, expected, atol=2e-12)
    assert jnp.allclose(actual @ values, initial @ model.expect(values), atol=2e-12)
    assert jnp.allclose(
        jax.grad(lambda p: model.advance(p).distribution @ values)(initial),
        model.expect(values),
        atol=2e-12,
    )


@pytest.mark.parametrize("throw", [False, True])
def test_sequential_jacobi_one_step_does_not_apply_reduced_assembly_domain(throw):
    """Strong selection is valid for arbitrary-probability one-step evaluation."""
    backend = drifft.Jacobi(16)
    rule = drifft.DiploidSelection(0.8, 0.99)
    initial = jnp.eye(17)[5]
    expected = drifft.Transition(drifft.FFT(16), rule).advance(initial)
    actual = drifft.Sequential().advance(
        backend, initial, frequency_map=rule, generations=1, throw=throw
    )
    assert actual.converged
    assert jnp.allclose(actual.distribution, expected, atol=2e-12)


def test_transition_rejects_unsupported_backend_at_construction():
    """Runtime validation must match the annotated backend contract."""

    class InvalidBackend:
        gene_copies = 16

    with pytest.raises(TypeError, match="AbstractTransitionBackend"):
        drifft.Transition(InvalidBackend(), drifft.Neutral())  # ty: ignore[invalid-argument-type]


def test_jacobi_segregating_rectangular_operations():
    """Interior operations discard endpoint hits without changing grid semantics."""
    source, destination = 20, 16
    rule = drifft.Mutation(0.02, 0.03)
    transition = drifft.Transition(
        drifft.Jacobi(destination), rule, source_gene_copies=source
    )
    probabilities = np.asarray(rule(jnp.arange(source + 1) / source))
    matrix = jnp.asarray(
        binom.pmf(
            np.arange(destination + 1)[None, :], destination, probabilities[:, None]
        )
    )[1:-1, 1:-1]
    values = jnp.sin(jnp.arange(destination - 1) * 0.4)
    mass = jnp.cos(jnp.arange(source - 1) * 0.2)
    assert jnp.allclose(
        transition.expect(values, segregating=True), matrix @ values, atol=2e-12
    )
    assert jnp.allclose(
        transition.advance(mass, segregating=True), matrix.T @ mass, atol=2e-12
    )


def test_truncated_jacobi_one_step_values_and_endpoint_derivatives():
    """Exercise the common interface with an actually truncated spectral basis."""
    copies = 512
    backend = drifft.Jacobi(copies)
    assert backend.rank < copies - 1
    q = jnp.array([0.0, 1e-6, 0.005, 0.12, 0.41, 0.9, 1 - 1e-6, 1.0])
    values = jnp.sin(jnp.arange(copies + 1) * 0.13)
    counts = np.arange(copies + 1)[None, :]
    probabilities = np.asarray(q)[:, None]
    matrix = jnp.asarray(binom.pmf(counts, copies, probabilities))
    derivative = jnp.asarray(
        copies
        * (
            binom.pmf(counts - 1, copies - 1, probabilities)
            - binom.pmf(counts, copies - 1, probabilities)
        )
    )
    assert jnp.allclose(
        backend.expect(q, values), matrix @ values, atol=2e-10, rtol=2e-10
    )
    actual = jax.jit(jax.grad(lambda q: backend.expect(q, values).sum()))(q)
    assert jnp.allclose(actual, derivative @ values, atol=2e-7, rtol=2e-9)


def test_jacobi_expected_visits_and_selection_derivative():
    """The common backend supports implicit occupation solves and their gradients."""
    from scipy.special import comb

    copies = 16
    backend = drifft.Jacobi(copies)
    frequencies = jnp.arange(1, copies) / copies
    counts = jnp.arange(1, copies)
    choose = jnp.asarray(comb(copies, np.arange(1, copies)))

    def visits(s):
        result = drifft.Transition(
            backend, drifft.AdditiveSelection(s)
        ).expected_visits(rtol=1e-12, atol=1e-13)
        return result.visits

    def reference(s):
        q = drifft.AdditiveSelection(s)(frequencies)
        matrix = choose * q[:, None] ** counts * (1 - q[:, None]) ** (copies - counts)
        return jnp.linalg.solve(jnp.eye(copies - 1) - matrix.T, jnp.eye(copies - 1)[0])

    strength = jnp.asarray(0.01)
    assert jnp.allclose(visits(strength), reference(strength), atol=2e-10, rtol=2e-10)
    actual = jax.jit(jax.grad(lambda s: visits(s).sum()))(strength)
    expected = jax.grad(lambda s: reference(s).sum())(strength)
    assert jnp.allclose(actual, expected, atol=2e-8, rtol=2e-8)


def test_sequential_jacobi_history_higher_derivatives_match_dense():
    """Keep gradients and Hessians through mutation and default Jacobi boundaries."""
    from scipy.special import comb

    history = drifft.Demography([6, 4, 7], [4, 3])
    model = drifft.DemographicTransition(
        history,
        backend_factory=drifft.Jacobi,
        frequency_map=drifft.Mutation(0.002, 0.008),
    )
    initial = jnp.eye(13)[2]
    values = jnp.cos(jnp.arange(15) * 0.3)
    choose = {m: jnp.asarray(comb(m, np.arange(m + 1))) for m in (8, 12, 14)}

    def objective(rate):
        updated = eqx.tree_at(
            lambda m: (m.frequency_map.forward_rate, m.frequency_map.backward_rate),
            model,
            (0.2 * rate, 0.8 * rate),
        )
        return updated.advance(initial).distribution @ values

    def reference(rate):
        state = initial
        for source, destination, duration in [(12, 8, 4), (8, 14, 3)]:
            for target in [source] * (duration - 1) + [destination]:
                frequencies = jnp.arange(source + 1) / source
                q = 0.2 * rate + (1 - rate) * frequencies
                counts = jnp.arange(target + 1)
                matrix = (
                    choose[target]
                    * q[:, None] ** counts
                    * (1 - q[:, None]) ** (target - counts)
                )
                state = matrix.T @ state
        return state @ values

    rate = jnp.asarray(0.01)
    for order in range(3):
        fn, ref = objective, reference
        for _ in range(order):
            fn, ref = jax.grad(fn), jax.grad(ref)
        assert jnp.allclose(
            eqx.filter_jit(fn)(rate), jax.jit(ref)(rate), atol=2e-8, rtol=2e-9
        )
