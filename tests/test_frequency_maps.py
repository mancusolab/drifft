"""Frequency-map composition, mutation, and differentiation contracts."""

# pattern: Functional Core
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import drifft

jax.config.update("jax_enable_x64", True)


def test_frequency_map_interface_and_composition_order():
    """Expose apply and callable shorthand with explicit left-to-right order."""
    frequencies = jnp.array([0.0, 0.2, 0.7, 1.0])
    selection = drifft.Underdominance(0.3)
    mutation = drifft.Mutation(forward_rate=0.1, backward_rate=0.2)
    custom = drifft.FrequencyMap(lambda x: x**2)
    for model in (drifft.Neutral(), selection, mutation, custom):
        assert isinstance(model, drifft.AbstractFrequencyMap)
        assert jnp.allclose(model.apply(frequencies), model(frequencies))
    assert jnp.array_equal(drifft.Neutral().apply(frequencies), frequencies)
    assert jnp.allclose(custom.apply(frequencies), frequencies**2)
    selected = selection.apply(frequencies)
    expected = selected * 0.8 + (1 - selected) * 0.1
    combined = drifft.Composition(selection, mutation)
    assert jnp.allclose(combined.apply(frequencies), expected)
    assert jnp.allclose(combined(frequencies), expected)
    assert not jnp.allclose(
        drifft.Composition(mutation, selection)(frequencies), expected
    )


@pytest.mark.parametrize("backend", [drifft.FFT, drifft.NUFFT])
def test_mutation_reintroduces_alleles_from_both_endpoints(backend):
    """Forward mutation escapes loss and back mutation escapes fixation."""
    transition = drifft.Transition(backend(16), frequency_map=drifft.Mutation(0.1, 0.2))
    initial = jnp.eye(17)[jnp.array([0, 16])]
    actual = eqx.filter_vmap(transition.advance)(initial)
    assert jnp.allclose(actual.sum(axis=1), 1, atol=1e-11)
    assert jnp.allclose(actual @ jnp.arange(17), jnp.array([1.6, 12.8]), atol=1e-10)
    assert jnp.all(actual[:, 1:-1].sum(axis=1) > 0)


@pytest.mark.parametrize("backend", [drifft.FFT, drifft.NUFFT])
def test_demographic_mutation_gradients(backend):
    """Keep both mutation rates dynamic through constant-size and changing-size steps."""
    model = drifft.DemographicTransition(
        drifft.Demography([8, 5], [3]),
        backend_factory=backend,
        frequency_map=drifft.Composition(drifft.Neutral(), drifft.Mutation(0.1, 0.2)),
    )
    initial = jnp.eye(17)[4]
    frequencies = jnp.arange(11) / 10

    def objective(rates):
        """Reuse geometry while replacing the two mutation-rate leaves."""
        updated = eqx.tree_at(
            lambda m: (
                m.frequency_map.maps[1].forward_rate,
                m.frequency_map.maps[1].backward_rate,
            ),
            model,
            (rates[0], rates[1]),
        )
        return updated.advance(initial).distribution @ frequencies

    def reference(rates):
        """Propagate the exact expected frequency under neutral mutation."""
        frequency = 0.25
        for _ in range(3):
            frequency = frequency * (1 - rates[1]) + (1 - frequency) * rates[0]
        return frequency

    rates = jnp.array([0.1, 0.2])
    assert jnp.allclose(objective(rates), reference(rates), atol=1e-10)
    assert jnp.allclose(
        jax.jit(jax.grad(objective))(rates), jax.grad(reference)(rates), atol=1e-9
    )


def test_frequency_map_configuration_validation():
    """Reject malformed components and nonscalar rate parameters."""
    with pytest.raises(ValueError, match="scalar"):
        drifft.Mutation(jnp.array([0.1, 0.2]), 0.1)
    with pytest.raises(TypeError):
        drifft.FrequencyMap(None)  # ty: ignore[invalid-argument-type] -- invalid wrapper
    with pytest.raises(ValueError):
        drifft.Composition()
    with pytest.raises(TypeError):
        drifft.Composition(lambda x: x)  # ty: ignore[invalid-argument-type] -- unwrapped function
    backend = drifft.FFT(16)
    with pytest.raises(TypeError):
        drifft.Transition(backend, frequency_map=lambda x: x)  # ty: ignore[invalid-argument-type] -- unwrapped function
    with pytest.raises(TypeError):
        drifft.DemographicTransition(
            drifft.Demography([8, 8], [1]),
            backend_factory=drifft.FFT,
            frequency_map=lambda x: x,  # ty: ignore[invalid-argument-type] -- unwrapped function
        )


@pytest.mark.parametrize("s_het,s_hom", [(0.1, 0.3), (-0.2, -0.4), (1.0, 0.2)])
def test_diploid_selection_matches_genotype_survival(s_het, s_hom):
    """Compute the offspring allele fraction from surviving genotype counts."""
    assert hasattr(drifft, "DiploidSelection")
    frequencies = jnp.array([[0.0, 0.2], [0.7, 1.0]])
    model = drifft.DiploidSelection(s_het=s_het, s_hom=s_hom)
    genotypes = jnp.stack(
        ((1 - frequencies) ** 2, 2 * frequencies * (1 - frequencies), frequencies**2),
        axis=-1,
    )
    survivors = genotypes * jnp.array([1.0, 1 - s_het, 1 - s_hom])
    expected = survivors @ jnp.array([0.0, 0.5, 1.0]) / survivors.sum(axis=-1)
    actual = eqx.filter_jit(model)(frequencies)
    assert actual.shape == frequencies.shape
    assert jnp.allclose(actual, expected, atol=1e-15)
    assert jnp.array_equal(
        actual[jnp.array([0, 1]), jnp.array([0, 1])], jnp.array([0, 1])
    )


SELECTION_CASES = [
    ("AdditiveSelection", {"s": 0.3}, (0.15, 0.3)),
    ("DominanceSelection", {"s": 0.4, "h": 0.3}, (0.12, 0.4)),
    ("RecessiveSelection", {"s": 0.3}, (0.0, 0.3)),
    ("DominantSelection", {"s": 0.3}, (0.3, 0.3)),
    ("Overdominance", {"strength": 0.3}, (-0.3, 0.0)),
    ("Underdominance", {"strength": 0.3}, (0.3, 0.0)),
    ("MultiplicativeSelection", {"r": 1.2}, (-0.2, -0.44)),
    ("Neutral", {}, (0.0, 0.0)),
]


@pytest.mark.parametrize("coefficient", [-0.4, 0.0, 0.3])
def test_additive_coefficient_matches_fitnesses_and_half_dominance(coefficient):
    """Use the same homozygote penalty for additive and dominance parameters."""
    frequencies = jnp.array([0.0, 0.2, 0.7, 1.0])
    genotypes = jnp.stack(
        ((1 - frequencies) ** 2, 2 * frequencies * (1 - frequencies), frequencies**2),
        axis=-1,
    )

    def reference(s):
        """Weight genotype counts by their independent viability fitnesses."""
        survivors = genotypes * jnp.array([1.0, 1 - s / 2, 1 - s])
        return survivors @ jnp.array([0.0, 0.5, 1.0]) / survivors.sum(axis=-1)

    def additive(s):
        """Evaluate the parameterization being checked."""
        return drifft.AdditiveSelection(s)(frequencies)

    np.testing.assert_allclose(
        additive(coefficient), reference(coefficient), atol=1e-15
    )
    np.testing.assert_allclose(
        additive(coefficient),
        drifft.DominanceSelection(coefficient, 0.5)(frequencies),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(additive))(coefficient),
        jax.jacfwd(reference)(coefficient),
        atol=2e-15,
    )


@pytest.mark.parametrize("name,parameters,coefficients", SELECTION_CASES)
def test_selection_specializations_preserve_constraints(name, parameters, coefficients):
    """Apply each biological parameterization through the common diploid rule."""
    assert hasattr(drifft, name)
    frequencies = jnp.linspace(0, 1, 21)
    model = getattr(drifft, name)(**parameters)
    assert isinstance(model, drifft.AbstractFrequencyMap)
    expected = drifft.DiploidSelection(*coefficients)(frequencies)
    assert jnp.allclose(eqx.filter_jit(model)(frequencies), expected, atol=1e-15)
    if name == "MultiplicativeSelection":
        ratio = parameters["r"]
        assert jnp.allclose(
            model(frequencies), ratio * frequencies / (1 + (ratio - 1) * frequencies)
        )
    if name == "Neutral":
        assert jnp.array_equal(model(frequencies), frequencies)


@pytest.mark.parametrize(
    "name,parameters,field",
    [
        (name, parameters, field)
        for name, parameters, _ in SELECTION_CASES[:-1]
        + [("DiploidSelection", {"s_het": 0.1, "s_hom": 0.3}, None)]
        for field in parameters
    ],
)
def test_selection_parameters_remain_dynamic_and_differentiable(
    name, parameters, field
):
    """Updating leaves must recompute constraints and retain parameter derivatives."""
    assert hasattr(drifft, name)
    model = getattr(drifft, name)(**parameters)
    original = parameters[field]
    frequencies = jnp.array([0.15, 0.4, 0.85])

    def evaluate(value):
        """Replace a model parameter without reconstructing cached derived fields."""
        updated = eqx.tree_at(lambda m: getattr(m, field), model, value)
        return updated(frequencies)

    value = jnp.asarray(original * 0.8)
    rebuilt = getattr(drifft, name)(**{**parameters, field: value})
    assert jnp.allclose(jax.jit(evaluate)(value), rebuilt(frequencies))
    tangent = jax.jacfwd(evaluate)(value)
    reverse = jax.jacrev(evaluate)(value)
    finite_difference = (evaluate(value + 1e-5) - evaluate(value - 1e-5)) / 2e-5
    assert jnp.allclose(tangent, reverse, atol=1e-12)
    assert jnp.allclose(tangent, finite_difference, atol=1e-9)
    batch = jnp.array([original * 0.8, original * 0.9])
    assert jnp.allclose(
        jax.jit(jax.vmap(evaluate))(batch), jnp.stack([evaluate(v) for v in batch])
    )


@pytest.mark.parametrize(
    "name,parameters",
    [(name, parameters) for name, parameters, _ in SELECTION_CASES]
    + [("DiploidSelection", {"s_het": 0.1, "s_hom": 0.3})],
)
def test_selection_input_derivatives_match_finite_differences(name, parameters):
    """Preserve input derivatives for vector inputs and mapped scalar inputs."""
    model = getattr(drifft, name)(**parameters)
    x = jnp.array([0.15, 0.4, 0.85])
    finite_difference = (model(x + 1e-5) - model(x - 1e-5)) / 2e-5
    forward = jax.jit(jax.jacfwd(model))(x)
    reverse = jax.jit(jax.grad(lambda values: model(values).sum()))(x)
    batched = jax.jit(jax.vmap(jax.grad(model)))(x)
    assert jnp.allclose(forward, jnp.diag(finite_difference), rtol=1e-8, atol=1e-10)
    assert jnp.allclose(reverse, finite_difference, rtol=1e-8, atol=1e-10)
    assert jnp.allclose(batched, finite_difference, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(
    "model_type,parameters",
    [
        (drifft.DiploidSelection, [0.1, 0.3, 0.4]),
        (drifft.MultiplicativeSelection, [1.2, 0.4]),
    ],
)
def test_selection_joint_jvp_is_linear(model_type, parameters):
    """Combine parameter and input tangents linearly with the correct derivative."""

    def evaluate(values):
        """Treat the last coordinate as frequency and the others as parameters."""
        return model_type(*values[:-1])(values[-1])

    point = jnp.array(parameters)
    first = jnp.arange(1, len(parameters) + 1, dtype=jnp.float64) / 10
    second = first[::-1]
    direction = 2 * first - 3 * second
    directional = jax.jit(lambda tangent: jax.jvp(evaluate, (point,), (tangent,))[1])
    finite_difference = (
        evaluate(point + 1e-5 * direction) - evaluate(point - 1e-5 * direction)
    ) / 2e-5
    assert jnp.allclose(
        directional(direction),
        2 * directional(first) - 3 * directional(second),
        rtol=1e-12,
        atol=1e-12,
    )
    assert jnp.allclose(
        directional(direction), finite_difference, rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize("backend", [drifft.FFT, drifft.NUFFT])
def test_diploid_selection_transition_and_gradients(backend):
    """Compare the full transition and parameter gradients to binomial rows."""
    from jax.scipy.stats import binom

    assert hasattr(drifft, "DiploidSelection")
    model = drifft.Transition(backend(16), drifft.DiploidSelection(0.1, 0.3))
    x = jnp.arange(17) / 16
    values = jnp.sin(jnp.arange(17))
    mutation = drifft.Mutation(0.01, 0.02)

    def objective(coefficients):
        """Keep prepared transition geometry while updating selection parameters."""
        updated = eqx.tree_at(
            lambda m: (m.frequency_map.s_het, m.frequency_map.s_hom),
            model,
            (coefficients[0], coefficients[1]),
        )
        return updated.expect(values)[5]

    def reference(coefficients):
        """Use genotype survival and a dense binomial row as an independent oracle."""
        p = x[5]
        counts = jnp.array([(1 - p) ** 2, 2 * p * (1 - p), p * p])
        surviving = counts * jnp.array([1.0, 1 - coefficients[0], 1 - coefficients[1]])
        q = surviving @ jnp.array([0.0, 0.5, 1.0]) / surviving.sum()
        row = jnp.exp(binom.logpmf(jnp.arange(17), 16, q))
        return row @ values

    coefficients = jnp.array([0.1, 0.3])
    assert jnp.allclose(objective(coefficients), reference(coefficients), atol=1e-11)
    assert jnp.allclose(
        jax.jit(jax.grad(objective))(coefficients),
        jax.grad(reference)(coefficients),
        atol=1e-10,
    )
    assert model.frequency_map is not None
    composed = drifft.Composition(mutation, model.frequency_map)
    assert jnp.allclose(composed(x), model.frequency_map(mutation(x)))
    mass = jnp.linspace(0.0, 1.0, 17)
    assert jnp.allclose(
        model.advance(mass) @ values, mass @ model.expect(values), atol=1e-10
    )


@pytest.mark.parametrize(
    "name,parameters",
    [(n, p) for n, p, _ in SELECTION_CASES if p]
    + [("DiploidSelection", {"s_het": 0.1, "s_hom": 0.2})],
)
def test_selection_rejects_nonscalar_parameters(name, parameters):
    """Each parameter is a scalar leaf; batching belongs outside construction."""
    assert hasattr(drifft, name)
    for field in parameters:
        with pytest.raises(ValueError, match="scalar"):
            getattr(drifft, name)(**{**parameters, field: jnp.array([0.1, 0.2])})


def test_jacobi_general_selection_matches_named_symmetric_rule():
    """General diploid and specialized symmetric assembly agree on shared inputs."""
    backend = drifft.Jacobi(8)
    method = drifft.RationalKrylov(dimension=12)
    initial = jnp.arange(1, 10, dtype=jnp.float64) / 45
    general = method.advance(
        backend,
        initial,
        frequency_map=drifft.DiploidSelection(0.1, 0.0),
        generations=7,
    )
    symmetric = method.advance(
        backend,
        initial,
        frequency_map=drifft.Underdominance(0.1),
        generations=7,
    )
    assert general.converged
    assert symmetric.converged
    assert jnp.allclose(
        general.distribution, symmetric.distribution, atol=3e-12, rtol=3e-10
    )


@pytest.mark.parametrize("ratio", [1e-10, 1e-200, 1e200])
def test_multiplicative_selection_retains_extreme_positive_fitness(ratio):
    """Avoid rounding tiny positive fitnesses to zero through 1 - r squared."""
    x = jnp.array([0.0, 0.25, 0.5, 1.0])
    actual = eqx.filter_jit(drifft.MultiplicativeSelection(ratio))(x)
    assert jnp.all(jnp.isfinite(actual))
    assert actual[0] == 0
    assert actual[-1] == 1
    # Selection multiplies allele odds by the per-copy fitness ratio.
    expected = ratio * x[1:3] / ((1 - x[1:3]) + ratio * x[1:3])
    assert jnp.allclose(actual[1:3], expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    "ratio,frequency,expected",
    [
        (1e-200, 0.0, (0.0, 1e-200)),
        (1e-200, 0.25, (1 / 3, 16e-200 / 9)),
        (1e-200, 1.0, (0.0, 1e200)),
        (1e-10, 1.0, (0.0, 1e10)),
        (1e200, 0.0, (0.0, 1e200)),
        (1e200, 1e-200, (2.5e-201, 2.5e199)),
        (1e200, 0.25, (0.0, 1.6e-199)),
        (1e200, 1.0, (0.0, 1e-200)),
    ],
)
@pytest.mark.parametrize("compiled", [False, True])
def test_multiplicative_selection_extreme_fitness_derivatives(
    ratio, frequency, expected, compiled
):
    """Retain representable analytic partials at extreme fitness and fixation."""

    def evaluate(parameters):
        """Differentiate both the fitness ratio and source frequency."""
        return drifft.MultiplicativeSelection(parameters[0])(parameters[1])

    parameters = jnp.array([ratio, frequency])
    for transform in (jax.jacfwd, jax.grad):
        derivative = transform(evaluate)
        if compiled:
            derivative = jax.jit(derivative)
        actual = derivative(parameters)
        assert jnp.allclose(actual, jnp.array(expected), rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    "s_het,s_hom,frequency,expected",
    [
        (
            -1e200,
            -1e200,
            0.25,
            (6.122448979591837e-202, -6.122448979591837e-202, 16 / 49),
        ),
        (-1e200, 0.0, 0.25, (0.0, -8.333333333333333e-202, 4.444444444444444e-200)),
        (0.0, -1e200, 0.25, (3e-200, 0.0, 1.12e-198)),
        (-1e200, 0.0, 0.0, (0.0, 0.0, 1e200)),
        (-1e200, 0.0, 1.0, (0.0, 0.0, 1e200)),
        (1.0, 0.3, 0.0, (0.0, 0.0, 0.0)),
        (1.0, 0.3, 1.0, (0.0, 0.0, 0.0)),
        (0.0, float(jnp.nextafter(1.0, 0.0)), 1.0, (0.0, 0.0, 2**53)),
    ],
)
@pytest.mark.parametrize("compiled", [False, True])
def test_diploid_selection_extreme_fitness_derivatives(
    s_het, s_hom, frequency, expected, compiled
):
    """Avoid spurious quotient overflow and preserve the boundary derivatives."""

    def evaluate(parameters):
        """Differentiate each independent reduction and the source frequency."""
        return drifft.DiploidSelection(*parameters[:2])(parameters[2])

    parameters = jnp.array([s_het, s_hom, frequency])
    for transform in (jax.jacfwd, jax.grad):
        derivative = transform(evaluate)
        if compiled:
            derivative = jax.jit(derivative)
        actual = derivative(parameters)
        assert jnp.allclose(actual, jnp.array(expected), rtol=1e-12, atol=0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_neutral_preserves_input_dtype(dtype):
    """Keep the existing identity map's dtype contract despite shared selection rules."""
    frequencies = jnp.array([0.0, 0.25, 0.8, 1.0], dtype=dtype)
    result = drifft.Neutral()(frequencies)
    assert result.dtype == frequencies.dtype
    assert jnp.array_equal(result, frequencies)
