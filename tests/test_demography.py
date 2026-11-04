"""Demographic timing, dense reference, and transformation contracts."""

# pattern: Functional Core
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from _reference import dense_transition

import drifft

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("duration", [1, 3])
def test_demographic_holds_use_custom_method_advance(duration):
    """Honor custom population-space evolution only when a hold is required."""

    class ScaledAdvance(drifft.AbstractEpochMethod):
        allows_reduced_representation = False

        def validate(self, backend, frequency_map):
            drifft.Sequential().validate(backend, frequency_map)

        def solve(self, problem, *, throw=True):
            raise AssertionError("The custom advance handles the hold")

        def advance(self, backend, distribution, *, throw=True, **kwargs):
            assert not throw
            return drifft.Sequential().advance(
                backend, 2 * distribution, throw=throw, **kwargs
            )

    model = drifft.DemographicTransition(
        drifft.Demography([4, 3], [duration]),
        backend_factory=drifft.Dense,
        frequency_map=drifft.Neutral(),
        method=ScaledAdvance(),
    )
    initial = jnp.eye(9)[2]
    expected = (
        initial
        @ jnp.linalg.matrix_power(dense_transition(4, 4, 0), duration - 1)
        @ dense_transition(4, 3, 0)
    )
    if duration > 1:
        expected *= 2
    assert jnp.allclose(model.advance(initial).distribution, expected, atol=1e-12)


def test_factories_reuse_geometry_for_repeated_sizes_and_shared_boundaries():
    """Prepare each distinct size once when hold and boundary factories agree."""
    calls = []

    def factory(copies):
        """Observe preparation requests in diploid units."""
        calls.append(copies)
        return drifft.Dense(copies)

    drifft.DemographicTransition(
        drifft.Demography([4, 3, 4], [2, 2]),
        backend_factory=factory,
        boundary_backend_factory=factory,
        frequency_map=drifft.Neutral(),
    )
    assert calls == [4, 3]


@pytest.mark.parametrize("backend", [drifft.FFT, drifft.NUFFT])
@pytest.mark.parametrize(
    "sizes,durations", [([8, 5, 11], [3, 2]), ([8, 5], [1]), ([8, 8], [4]), ([8], [])]
)
def test_demographic_evaluation_matches_dense_boundary_history(
    backend, sizes, durations
):
    """Hold the source size and change only on the final interval generation."""
    history = drifft.Demography(pop_sizes=sizes, generations=durations)
    model = drifft.DemographicTransition(
        history, backend_factory=backend, frequency_map=drifft.Underdominance(0.07)
    )
    matrix = jnp.eye(2 * sizes[0] + 1)
    for source, destination, duration in zip(
        sizes[:-1], sizes[1:], durations, strict=True
    ):
        matrix = matrix @ jnp.linalg.matrix_power(
            dense_transition(source, source, 0.07), duration - 1
        )
        matrix = matrix @ dense_transition(source, destination, 0.07)
    distribution = jnp.arange(1, 2 * sizes[0] + 2, dtype=jnp.float64)
    distribution /= distribution.sum()
    values = jnp.sin(jnp.arange(2 * sizes[-1] + 1))
    actual = model.advance(distribution).distribution
    expected = model.expect(values).values
    assert jnp.allclose(actual, distribution @ matrix, atol=2e-10, rtol=0)
    assert jnp.allclose(expected, matrix @ values, atol=2e-10, rtol=0)
    assert jnp.allclose(actual.sum(), 1, atol=2e-10, rtol=0)
    assert jnp.allclose(actual @ values, distribution @ expected, atol=2e-10, rtol=0)


@pytest.mark.parametrize("backend", [drifft.FFT, drifft.NUFFT])
def test_demographic_selection_gradient_and_batching(backend):
    """Differentiate dynamic selection through size changes and fixed-length loops."""
    model = drifft.DemographicTransition(
        drifft.Demography([8, 5, 11], [3, 2]),
        backend_factory=backend,
        frequency_map=drifft.Underdominance(0.07),
    )
    initial = jnp.eye(17)[3]
    values = jnp.sin(jnp.arange(23))

    def objective(strength):
        """Replace selection while reusing the prepared geometry."""
        updated = eqx.tree_at(lambda m: m.frequency_map.strength, model, strength)
        return updated.advance(initial).distribution @ values

    def reference(strength):
        """Evaluate the same five-generation history using dense binomial matrices."""
        first = dense_transition(8, 8, strength)
        last = dense_transition(5, 5, strength)
        return (
            initial
            @ first
            @ first
            @ dense_transition(8, 5, strength)
            @ last
            @ dense_transition(5, 11, strength)
            @ values
        )

    strength = jnp.asarray(0.07)
    assert jnp.allclose(
        jax.grad(objective)(strength),
        jax.grad(reference)(strength),
        atol=2e-8,
        rtol=2e-6,
    )
    epsilon = 1e-5
    difference = (objective(strength + epsilon) - objective(strength - epsilon)) / (
        2 * epsilon
    )
    assert jnp.allclose(jax.grad(objective)(strength), difference, atol=2e-8, rtol=2e-6)
    batch = jnp.stack([initial, jnp.roll(initial, 1)])
    actual = eqx.filter_jit(eqx.filter_vmap(model.advance))(batch).distribution
    assert jnp.allclose(
        actual,
        jnp.stack([model.advance(row).distribution for row in batch]),
        atol=1e-12,
    )


def test_demography_normalizes_array_schedule():
    """Copy concrete integer arrays into immutable host metadata."""
    history = drifft.Demography(jnp.array([8, 5, 11]), jnp.array([3, 2]))
    assert history.pop_sizes == (8, 5, 11)
    assert history.generations == (3, 2)
    assert history.initial_pop_size == 8
    assert history.final_pop_size == 11
    assert history.total_generations == 5


@pytest.mark.parametrize(
    "sizes,durations",
    [
        ([], []),
        ([8, 5], []),
        ([8], [1]),
        ([0, 5], [1]),
        ([8, -1], [1]),
        ([8.5, 5], [1]),
        ([True, 5], [1]),
        ([8, 5], [0]),
        ([8, 5], [-1]),
        ([8, 5], [1.5]),
        ([8, 5], [True]),
        ([[8], [5]], [1]),
    ],
)
def test_demography_rejects_invalid_schedule(sizes, durations):
    """Reject ambiguous lengths, noninteger counts, and nonpositive metadata."""
    with pytest.raises(ValueError):
        drifft.Demography(sizes, durations)


@pytest.mark.parametrize("method,size", [("advance", 10), ("expect", 16)])
def test_demographic_methods_reject_wrong_shape(method, size):
    """Validate source and destination vector lengths at the public boundary."""
    model = drifft.DemographicTransition(
        drifft.Demography([8, 5], [2]),
        backend_factory=drifft.FFT,
        frequency_map=drifft.Underdominance(0.07),
    )
    with pytest.raises(ValueError):
        getattr(model, method)(jnp.ones(size))


@pytest.mark.parametrize("segregating", [False, True])
def test_expect_transposes_custom_sequential_hold_backend(segregating):
    """Transpose the configured hold operator even when boundary geometry differs."""
    model = drifft.DemographicTransition(
        drifft.Demography([200, 200], [5]),
        boundary_backend_factory=drifft.FFT,
        backend_factory=lambda n: drifft.FFT(
            n, interp_order=2, min_mode_count=8, block_spacing=4
        ),
        frequency_map=drifft.Underdominance(0.03),
    )
    size = 399 if segregating else 401
    distribution = jnp.zeros(size).at[170].set(1)
    values = jnp.sin(jnp.arange(size) * 0.13)
    assert jnp.allclose(
        model.advance(distribution, segregating=segregating).distribution @ values,
        distribution @ model.expect(values, segregating=segregating).values,
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "backend", [drifft.Dense, drifft.FFT, drifft.NUFFT, drifft.Jacobi]
)
@pytest.mark.parametrize(
    "sizes,durations", [([4, 3, 5], [3, 2]), ([4, 3], [1]), ([4, 4], [3]), ([4], [])]
)
def test_segregating_affine_expect_matches_dense_history(backend, sizes, durations):
    """Match killed-chain rewards and singleton sources against dense histories."""
    model = drifft.DemographicTransition(
        drifft.Demography(sizes, durations),
        backend_factory=backend,
        frequency_map=drifft.Underdominance(0.03),
    )
    values = jnp.sin(jnp.arange(2 * sizes[-1] - 1) + 0.2)
    matrix = jnp.eye(2 * sizes[0] - 1)
    for source, destination, duration in zip(
        sizes[:-1], sizes[1:], durations, strict=True
    ):
        matrix = (
            matrix
            @ jnp.linalg.matrix_power(
                dense_transition(source, source, 0.03)[1:-1, 1:-1], duration - 1
            )
            @ dense_transition(source, destination, 0.03)[1:-1, 1:-1]
        )
    actual = model.expect(values, segregating=True).values
    assert jnp.allclose(actual, matrix @ values, atol=1e-11, rtol=0)
    rate = 0.007
    result = model.expect(values, segregating=True, singleton_mutation_rate=rate)
    assert isinstance(result, drifft.DemographicExpectation)
    linear, offset = result.values, result.offset
    mass = jnp.linspace(-0.2, 0.4, 2 * sizes[0] - 1)
    expected = mass
    for source, destination, duration in zip(
        sizes[:-1], sizes[1:], durations, strict=True
    ):
        for _ in range(duration - 1):
            expected = (
                (expected @ dense_transition(source, source, 0.03)[1:-1, 1:-1])
                .at[0]
                .add(2 * source * rate)
            )
        expected = (
            (expected @ dense_transition(source, destination, 0.03)[1:-1, 1:-1])
            .at[0]
            .add(2 * destination * rate)
        )
    assert jnp.allclose(linear, actual, atol=1e-12)
    assert jnp.allclose(mass @ linear + offset, expected @ values, atol=1e-11)
    forward = model.advance(mass, segregating=True, singleton_mutation_rate=rate)
    assert jnp.allclose(
        mass @ linear + offset, forward.distribution @ values, atol=1e-11
    )


def test_affine_expect_recurrent_mutation_gradients_and_batching():
    """Match forward rewards and derivatives with recurrent mutation and influx."""
    model = drifft.DemographicTransition(
        drifft.Demography([4, 3, 5], [3, 2]),
        backend_factory=drifft.Jacobi,
        boundary_backend_factory=drifft.FFT,
        frequency_map=drifft.Mutation(0.01, 0.02),
    )
    mass = jnp.linspace(-0.1, 0.3, 7)
    values = jnp.sin(jnp.arange(9) + 0.2)

    def objective(mutation, rate, terminal, *, backward):
        updated = eqx.tree_at(lambda m: m.frequency_map.forward_rate, model, mutation)
        if backward:
            linear, offset = updated.expect(
                terminal,
                segregating=True,
                singleton_mutation_rate=rate,
            )
            return mass @ linear + offset
        return (
            updated.advance(
                mass, segregating=True, singleton_mutation_rate=rate
            ).distribution
            @ terminal
        )

    def backward(*args):
        return objective(*args, backward=True)

    def forward(*args):
        return objective(*args, backward=False)

    inputs = (jnp.asarray(0.01), jnp.asarray(0.007), values)
    assert jnp.allclose(backward(*inputs), forward(*inputs), atol=1e-11)
    for actual, expected in zip(
        jax.grad(backward, argnums=(0, 1, 2))(*inputs),
        jax.grad(forward, argnums=(0, 1, 2))(*inputs),
        strict=True,
    ):
        assert jnp.allclose(actual, expected, atol=1e-10)
    direction = jax.tree.map(jnp.ones_like, inputs)
    assert jnp.allclose(
        jax.jvp(backward, inputs, direction)[1],
        jax.jvp(forward, inputs, direction)[1],
        atol=1e-10,
    )
    batch = (
        jnp.array([0.01, 0.03]),
        jnp.array([0.0, 0.007]),
        jnp.stack([values, 2 * values]),
    )
    assert jnp.allclose(
        jax.jit(jax.vmap(backward))(*batch),
        jax.jit(jax.vmap(forward))(*batch),
        atol=1e-11,
    )


def test_expect_validates_layout_source_and_return_contract():
    """Reject invalid inputs even for identity histories; never drop a source."""
    model = drifft.DemographicTransition(
        drifft.Demography([4], []),
        backend_factory=drifft.Dense,
        frequency_map=drifft.Neutral(),
    )
    for kwargs in (
        {"segregating": 1},
        {"singleton_mutation_rate": 0.01},
        {
            "segregating": True,
            "singleton_mutation_rate": jnp.ones(2),
        },
    ):
        with pytest.raises(ValueError):
            model.expect(jnp.ones(7), **kwargs)  # ty: ignore[invalid-argument-type] -- intentionally invalid flags and source
    for rate in (-0.1, float("nan"), float("inf")):
        with pytest.raises(eqx.EquinoxRuntimeError, match="mutation_rate"):
            model.expect(
                jnp.ones(7),
                segregating=True,
                singleton_mutation_rate=rate,
            )[0].block_until_ready()
    with pytest.raises(ValueError, match="entries"):
        model.expect(jnp.ones(9), segregating=True)
    for method in (drifft.PolynomialKrylov(), drifft.RationalKrylov()):
        updated = eqx.tree_at(lambda m: m.method, model, method)
        with pytest.raises(ValueError, match="Sequential"):
            updated.expect(jnp.ones(7), segregating=True)
    values = jnp.arange(9.0)
    result = model.expect(values)
    assert isinstance(result, drifft.DemographicExpectation)
    assert jnp.array_equal(result.values, values)
    assert result.offset == 0


@pytest.mark.parametrize("component", [0, 1])
@pytest.mark.parametrize("rate", [-0.1, float("nan"), float("inf")])
def test_affine_identity_rate_check_survives_partial_output(component, rate):
    """An outer JIT must retain domain checks for either returned component."""
    model = drifft.DemographicTransition(
        drifft.Demography([4], []),
        backend_factory=drifft.Dense,
        frequency_map=drifft.Neutral(),
    )

    @eqx.filter_jit
    def selected(rate):
        return model.expect(
            jnp.ones(7),
            segregating=True,
            singleton_mutation_rate=rate,
        )[component]

    with pytest.raises(eqx.EquinoxRuntimeError, match="mutation_rate"):
        selected(jnp.asarray(rate)).block_until_ready()
