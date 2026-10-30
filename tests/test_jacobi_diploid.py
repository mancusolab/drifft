"""Diploid selection and recurrent mutation against independent binomial models."""

# pattern: Functional Core
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import gammaln, xlog1py, xlogy

import drifft

jax.config.update("jax_enable_x64", True)


def dense_diploid(copies, parameters, mutation_first=True, source_copies=None):
    """Form the full binomial matrix without calling a drifft frequency map."""
    het, hom, forward, backward = parameters
    x = jnp.arange((source_copies or copies) + 1) / (source_copies or copies)
    if mutation_first:
        x = forward + (1 - forward - backward) * x
    mean = (1 - x) ** 2 + 2 * (1 - het) * x * (1 - x) + (1 - hom) * x**2
    q = ((1 - het) * x * (1 - x) + (1 - hom) * x**2) / mean
    if not mutation_first:
        q = forward + (1 - forward - backward) * q
    k = jnp.arange(copies + 1, dtype=jnp.float64)
    return jnp.exp(
        gammaln(copies + 1)
        - gammaln(k + 1)
        - gammaln(copies - k + 1)
        + xlogy(k[None, :], q[:, None])
        + xlog1py(copies - k[None, :], -q[:, None])
    )


def composed_map(parameters, mutation_first=True):
    """Construct the public map in the requested biological order."""
    het, hom, forward, backward = parameters
    selection = drifft.DiploidSelection(het, hom)
    mutation = drifft.Mutation(forward, backward)
    return (
        drifft.Composition(mutation, selection)
        if mutation_first
        else drifft.Composition(selection, mutation)
    )


@pytest.mark.parametrize("mutation_first", [True, False])
@pytest.mark.parametrize("segregating", [False, True])
@pytest.mark.parametrize("generations", [1, 9])
def test_composed_epochs_match_dense(mutation_first, segregating, generations):
    """Endpoint return, asymmetric selection, killing and immigration use one sampling step."""
    copies = 16
    parameters = jnp.array([-0.035, 0.08, 0.012, 0.027])
    matrix = dense_diploid(copies, parameters, mutation_first)
    if segregating:
        matrix = matrix[1:-1, 1:-1]
    initial = jnp.arange(1, matrix.shape[0] + 1, dtype=jnp.float64)
    initial /= initial.sum()
    source = jnp.zeros_like(initial).at[1].set(1e-4)
    expected = initial
    for _ in range(generations):
        expected = expected @ matrix + source
    result = drifft.RationalKrylov(dimension=32).advance(
        drifft.Jacobi(copies),
        initial,
        frequency_map=composed_map(parameters, mutation_first),
        generations=generations,
        segregating=segregating,
        immigration=source,
    )
    assert result.converged
    np.testing.assert_allclose(result.distribution, expected, atol=2e-12, rtol=2e-10)


@pytest.mark.parametrize("endpoint", [0, -1])
@pytest.mark.parametrize("rate", [0.0, 1e-8, 0.02])
def test_mutation_leaves_endpoints_without_losing_small_probabilities(endpoint, rate):
    """At realistic mutation rates, compare segregating mass relatively rather than to mass one."""
    copies, generations = 20, 25
    parameters = jnp.array([0.001, 0.002, rate, 2 * rate])
    initial = jnp.eye(copies + 1)[endpoint]
    expected = initial @ jnp.linalg.matrix_power(
        dense_diploid(copies, parameters), generations
    )
    actual = (
        drifft.RationalKrylov(dimension=32)
        .advance(
            drifft.Jacobi(copies),
            initial,
            frequency_map=composed_map(parameters),
            generations=generations,
        )
        .distribution
    )
    np.testing.assert_allclose(actual.sum(), 1, atol=2e-12)
    np.testing.assert_allclose(actual[1:-1], expected[1:-1], atol=2e-16, rtol=2e-7)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-7)


@pytest.mark.parametrize("mutation_first", [True, False])
def test_composed_parameters_have_correct_jvp_vjp_and_batching(mutation_first):
    """Differentiate all four parameters through assembly, shifted solves and decoding."""
    copies, generations = 12, 7
    backend = drifft.Jacobi(copies)
    method = drifft.RationalKrylov(dimension=20)
    initial = jnp.eye(copies + 1)[0]
    values = jnp.sin(jnp.arange(copies + 1) * 0.3)

    @eqx.filter_jit
    def objective(parameters):
        return (
            method.advance(
                backend,
                initial,
                frequency_map=composed_map(parameters, mutation_first),
                generations=generations,
            ).distribution
            @ values
        )

    def reference(parameters):
        return (
            initial
            @ jnp.linalg.matrix_power(
                dense_diploid(copies, parameters, mutation_first), generations
            )
            @ values
        )

    parameters = jnp.array([0.015, -0.025, 0.007, 0.011])
    np.testing.assert_allclose(
        jax.grad(objective)(parameters),
        jax.grad(reference)(parameters),
        atol=2e-9,
        rtol=2e-8,
    )
    direction = jnp.array([0.2, -0.3, 0.1, -0.15])
    tangent = jax.jvp(objective, (parameters,), (direction,))[1]
    finite_difference = (
        reference(parameters + 1e-6 * direction)
        - reference(parameters - 1e-6 * direction)
    ) / 2e-6
    np.testing.assert_allclose(tangent, finite_difference, atol=2e-8, rtol=2e-7)
    batch = jnp.stack((parameters, parameters * 0.7))
    np.testing.assert_allclose(
        jax.vmap(objective)(batch), jax.vmap(reference)(batch), atol=2e-12, rtol=2e-10
    )


def test_composed_demography_matches_changing_size_dense_history():
    """Public demographic integration uses the composed map for holds and boundaries."""
    parameters = jnp.array([0.025, 0.05, 0.001, 0.003])
    history = drifft.Demography([8, 5, 9], [7, 4])
    initial = jnp.eye(17)[0]
    expected = initial
    for source, destination, duration in [(16, 10, 7), (10, 18, 4)]:
        expected = expected @ jnp.linalg.matrix_power(
            dense_diploid(source, parameters), duration - 1
        )
        expected = expected @ dense_diploid(
            destination, parameters, source_copies=source
        )
    model = drifft.DemographicTransition(
        history,
        backend=drifft.Jacobi,
        boundary_backend=drifft.FFT,
        method=drifft.RationalKrylov(dimension=24),
        frequency_map=composed_map(parameters),
    )
    result = model.advance(initial)
    assert result.converged
    np.testing.assert_allclose(result.distribution, expected, atol=3e-12, rtol=2e-9)


@pytest.mark.parametrize(
    "forward,backward", [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.8, 0.7)]
)
def test_mutation_only_including_constant_and_reversing_maps(forward, backward):
    """Mutation need not be invertible or increasing to define a binomial kernel."""
    parameters = jnp.array([0.0, 0.0, forward, backward])
    initial = jnp.arange(1, 10, dtype=jnp.float64) / 45
    expected = initial @ jnp.linalg.matrix_power(dense_diploid(8, parameters), 9)
    actual = (
        drifft.RationalKrylov(dimension=12)
        .advance(
            drifft.Jacobi(8),
            initial,
            frequency_map=drifft.Mutation(forward, backward),
            generations=9,
        )
        .distribution
    )
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-10)


@pytest.mark.parametrize(
    "selection,het,hom",
    [
        (drifft.DiploidSelection(-0.03, 0.04), -0.03, 0.04),
        (drifft.AdditiveSelection(0.02), 0.02, 0.04),
        (drifft.DominanceSelection(0.04, 0.3), 0.012, 0.04),
        (drifft.RecessiveSelection(0.04), 0.0, 0.04),
        (drifft.DominantSelection(0.04), 0.04, 0.04),
        (drifft.MultiplicativeSelection(0.98), 0.02, 1 - 0.98**2),
    ],
)
def test_constrained_selection_rules_match_diploid_reference(selection, het, hom):
    """The named classes select the same rational rule as their diploid constraints."""
    initial = jnp.arange(1, 14, dtype=jnp.float64) / 91
    expected = initial @ jnp.linalg.matrix_power(
        dense_diploid(12, jnp.array([het, hom, 0, 0])), 11
    )
    actual = (
        drifft.RationalKrylov(dimension=16)
        .advance(
            drifft.Jacobi(12),
            initial,
            frequency_map=selection,
            generations=11,
        )
        .distribution
    )
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-10)


@pytest.mark.parametrize("order", ["standalone", "mutation_first", "selection_first"])
@pytest.mark.parametrize("strength", [0.0, 0.03, 0.1])
def test_overdominance_matches_independent_dense_reference(order, strength):
    """Heterozygote advantage uses negative diploid reduction in either order."""
    copies, generations = 12, 7
    selection = drifft.Overdominance(strength)
    forward, backward = (0.0, 0.0) if order == "standalone" else (0.007, 0.011)
    mutation = drifft.Mutation(forward, backward)
    frequency_map = selection
    if order == "mutation_first":
        frequency_map = drifft.Composition(mutation, selection)
    elif order == "selection_first":
        frequency_map = drifft.Composition(selection, mutation)
    initial = jnp.arange(1, copies + 2, dtype=jnp.float64)
    initial /= initial.sum()
    expected = initial @ jnp.linalg.matrix_power(
        dense_diploid(
            copies,
            jnp.array([-strength, 0.0, forward, backward]),
            mutation_first=order != "selection_first",
        ),
        generations,
    )
    result = drifft.RationalKrylov(dimension=16).advance(
        drifft.Jacobi(copies),
        initial,
        frequency_map=frequency_map,
        generations=generations,
    )
    assert result.converged
    np.testing.assert_allclose(result.distribution, expected, atol=3e-12, rtol=3e-10)


@pytest.mark.parametrize("order", ["standalone", "mutation_first", "selection_first"])
@pytest.mark.parametrize("strength", [-0.03, float("nan"), float("inf")])
@pytest.mark.parametrize("generations", [0, 1, 7])
def test_overdominance_domain_checked_including_identity(order, strength, generations):
    """Named-rule constraints remain active before identity and nontrivial actions."""
    selection = drifft.Overdominance(strength)
    mutation = drifft.Mutation(0.01)
    frequency_map = selection
    if order == "mutation_first":
        frequency_map = drifft.Composition(mutation, selection)
    elif order == "selection_first":
        frequency_map = drifft.Composition(selection, mutation)
    with pytest.raises(eqx.EquinoxRuntimeError, match="selection strength"):
        drifft.RationalKrylov().advance(
            drifft.Jacobi(8),
            jnp.eye(9)[0],
            frequency_map=frequency_map,
            generations=generations,
            throw=False,
        ).distribution.block_until_ready()


@pytest.mark.parametrize(
    "parameters,message",
    [
        ([0.11, 0.0, 0.0, 0.0], "selection coefficients"),
        ([0.0, -0.11, 0.0, 0.0], "selection coefficients"),
        ([float("nan"), 0.0, 0.0, 0.0], "selection coefficients"),
        ([0.0, float("inf"), 0.0, 0.0], "selection coefficients"),
        ([0.0, 0.0, -0.01, 0.0], "mutation rates"),
        ([0.0, 0.0, 0.0, 1.01], "mutation rates"),
        ([0.0, 0.0, float("nan"), 0.0], "mutation rates"),
    ],
)
def test_composed_domains_are_checked_even_for_zero_generations(parameters, message):
    """Planning limits and mutation preconditions remain active for identity actions."""
    with pytest.raises(eqx.EquinoxRuntimeError, match=message):
        drifft.RationalKrylov().advance(
            drifft.Jacobi(8),
            jnp.eye(9)[0],
            frequency_map=composed_map(jnp.array(parameters)),
            generations=0,
            throw=False,
        ).distribution.block_until_ready()


@pytest.mark.parametrize("rank", [1, None])
def test_padding_handles_strength_bounds_approaching_one(rank):
    """A pole-bound ellipse below floating-point resolution falls back to full dimension."""
    backend = drifft.Jacobi(8, rank=rank, max_strength=np.nextafter(1.0, 0.0))
    assert backend._diploid_padded_rank == 7


def test_genuinely_truncated_rational_assembly_matches_sampled_reduced_operator():
    """Compare ED independently when both rank and Hahn padding are truncated."""
    from drifft.backends.jacobi import _assemble_diploid

    backend = drifft.Jacobi(512)
    geometry = backend._geometry
    parameters = jnp.array([-0.04, 0.08, 0.002, 0.004])
    frequency_map = composed_map(parameters)
    assert backend.rank < backend._diploid_padded_rank < 511
    decoder = eqx.filter_jit(jax.jacfwd(lambda c: geometry.decode(c, False)))(
        jnp.zeros(backend.rank + 2)
    )
    # Transpose of population-space evaluation D, with one row per count.
    encoder = eqx.filter_jit(
        jax.jacfwd(lambda p: geometry.encode(p, frequency_map, False))
    )(jnp.zeros(513))
    expected = decoder.T @ encoder.T
    actual = eqx.filter_jit(_assemble_diploid)(
        frequency_map, 512, backend.rank, backend._diploid_padded_rank
    )
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-9)


def test_composed_underdominance_keeps_its_nonnegative_domain():
    """Wrapping a constrained model does not broaden that model's parameter domain."""
    frequency_map = drifft.Composition(
        drifft.Mutation(0.01), drifft.Underdominance(-0.03)
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="selection strength"):
        drifft.RationalKrylov().advance(
            drifft.Jacobi(8),
            jnp.eye(9)[0],
            frequency_map=frequency_map,
            generations=0,
            throw=False,
        ).distribution.block_until_ready()


@pytest.mark.parametrize(
    "frequency_map",
    [
        drifft.FrequencyMap(lambda x: x),
        drifft.Composition(drifft.Mutation(0.01), drifft.Mutation(0.02)),
        drifft.Composition(
            drifft.DiploidSelection(0.02, 0.03), drifft.DiploidSelection(0.01, 0.02)
        ),
        drifft.Composition(
            drifft.Mutation(0.01),
            drifft.DiploidSelection(0.02, 0.03),
            drifft.Mutation(0.02),
        ),
    ],
)
def test_unsupported_compositions_are_rejected_explicitly(frequency_map):
    """The quadratic implementation must not silently accept higher-degree maps."""
    with pytest.raises(TypeError, match="diploid selection|one Mutation"):
        drifft.Jacobi(8).validate(frequency_map)


def test_truncated_selected_mutation_epoch_matches_dense_sample_probabilities():
    """A genuine reduction preserves rare segregating mass and sampled likelihoods."""
    from scipy.stats import hypergeom

    copies, generations = 512, 100
    parameters = jnp.array([1e-4, 2e-4, 1e-8, 1e-8])
    backend = drifft.Jacobi(copies)
    initial = jnp.eye(copies + 1)[0]
    matrix = dense_diploid(copies, parameters)
    expected = jax.lax.fori_loop(0, generations, lambda _, p: p @ matrix, initial)
    actual = (
        drifft.RationalKrylov(dimension=256)
        .advance(
            backend,
            initial,
            frequency_map=composed_map(parameters),
            generations=generations,
        )
        .distribution
    )
    assert backend.rank < backend._diploid_padded_rank < copies - 1
    assert float(jnp.sum(jnp.abs(actual[1:-1] - expected[1:-1]))) < 1e-7 * float(
        expected[1:-1].sum()
    )
    weights = hypergeom.pmf(
        np.arange(1, 10)[:, None], copies, np.arange(copies + 1)[None, :], 20
    )
    np.testing.assert_allclose(
        weights @ np.asarray(actual),
        weights @ np.asarray(expected),
        rtol=2e-7,
        atol=1e-16,
    )
    np.testing.assert_allclose(actual.sum(), 1, atol=2e-12)
