"""Transition object contracts against independent binomial calculations."""

# Functional Core: state ownership, operator orientation, and transformations.
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binom

import drifft

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("segregating", [False, True])
def test_transition_dispatches_to_backend_advance(segregating):
    """Use specialized forward kernels and restrict rectangular interior states."""

    class ForwardBackend(drifft.AbstractTransitionBackend):
        """Test backend that forbids deriving its forward kernel from expect."""

        pop_size: int
        matrix: jax.Array

        def expect(self, probabilities, values):
            """Fail if Transition bypasses the specialized forward operation."""
            raise AssertionError("use the backend's advance implementation")

        def advance(self, probabilities, distribution):
            """Apply the prepared forward matrix directly."""
            return self.matrix.T @ distribution

    probabilities = jnp.array([0.0, 0.4, 1.0])
    matrix = binom.pmf(np.arange(5)[None, :], 4, np.asarray(probabilities)[:, None])
    backend = ForwardBackend(pop_size=2, matrix=jnp.asarray(matrix))
    transition = drifft.Transition(backend, probabilities=probabilities)
    mass = jnp.array([0.2, 0.5, 0.3])
    if segregating:
        mass = mass[1:-1]
        matrix = matrix[1:-1, 1:-1]
    np.testing.assert_allclose(
        transition.advance(mass, segregating=segregating), matrix.T @ mass, atol=1e-12
    )


def test_selection_model_and_both_transition_directions():
    """Match underdominance expectations and propagation to dense references."""
    backend = drifft.NUFFT(pop_size=8)
    transition = drifft.Transition(
        backend, frequency_map=drifft.Underdominance(strength=0.1)
    )
    x = np.arange(17) / 16
    p = (x - 0.1 * x * (1 - x)) / (1 - 0.2 * x * (1 - x))
    matrix = binom.pmf(np.arange(17)[None, :], 16, p[:, None])
    values = jnp.linspace(-1, 1, 17)
    mass = jnp.arange(1, 18, dtype=jnp.float64) / 153
    np.testing.assert_allclose(
        transition.expect(values=values), matrix @ values, atol=1e-12
    )
    np.testing.assert_allclose(
        transition.advance(distribution=mass), matrix.T @ mass, atol=1e-12
    )
    np.testing.assert_allclose(
        transition.expect(segregating=True, values=values[1:-1]),
        matrix[1:-1, 1:-1] @ values[1:-1],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        transition.advance(segregating=True, distribution=mass[1:-1]),
        matrix[1:-1, 1:-1].T @ mass[1:-1],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        transition.as_lineax().T.mv(mass), matrix.T @ mass, atol=1e-12
    )
    assert float(transition.advance(mass[1:-1], segregating=True).sum()) < float(
        mass[1:-1].sum()
    )


def test_explicit_probabilities_preserve_rectangular_transitions():
    """Preserve rectangular diploid grids and nonabsorbing endpoint probabilities."""
    backend = drifft.NUFFT(4)
    p = jnp.array([0.05, 0.7, 0.4, 0.8, 0.95])
    transition = drifft.Transition(backend, probabilities=p)
    matrix = binom.pmf(np.arange(9)[None, :], 8, np.asarray(p)[:, None])
    values = jnp.arange(9, dtype=jnp.float64)
    mass = jnp.array([0.1, 0.2, 0.3, 0.25, 0.15])
    assert transition.source_pop_size == 2
    np.testing.assert_allclose(transition.expect(values), matrix @ values, atol=1e-12)
    np.testing.assert_allclose(transition.advance(mass), matrix.T @ mass, atol=1e-12)
    assert transition.as_lineax().in_structure().shape == (9,)
    assert transition.as_lineax().out_structure().shape == (5,)
    np.testing.assert_allclose(
        transition.expect(values[1:-1], segregating=True),
        matrix[1:-1, 1:-1] @ values[1:-1],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        transition.advance(mass[1:-1], segregating=True),
        matrix[1:-1, 1:-1].T @ mass[1:-1],
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="square"):
        transition.expected_visits()


def test_custom_selection_and_rectangular_source_size():
    """Apply arbitrary selection callables on the requested source grid."""
    backend = drifft.NUFFT(4)
    transition = drifft.Transition(
        backend,
        frequency_map=drifft.FrequencyMap(lambda x: x**2),
        source_pop_size=2,
    )
    np.testing.assert_allclose(transition.probabilities, (np.arange(5) / 4) ** 2)
    np.testing.assert_allclose(
        transition.expect(jnp.arange(9, dtype=jnp.float64)),
        8 * (np.arange(5) / 4) ** 2,
        atol=1e-12,
    )


def test_explicit_probability_gradients_preserve_population_grid():
    """Differentiate a rectangular transition's offspring mean through explicit rows."""
    backend = drifft.Dense(4)
    mass = jnp.array([0.1, 0.2, 0.3, 0.25, 0.15])

    def mean_count(probabilities):
        """Average the offspring allele count over source states."""
        transition = drifft.Transition(backend, probabilities=probabilities)
        return transition.advance(mass) @ jnp.arange(transition.dest_size())

    probabilities = jnp.array([0.05, 0.7, 0.4, 0.8, 0.95])
    np.testing.assert_allclose(
        jax.jit(jax.grad(mean_count))(probabilities), 8 * mass, atol=1e-12
    )


def test_parameters_are_dynamic_and_not_stale_cached_probabilities():
    """Reuse compiled calls while selection parameters change as dynamic leaves."""
    backend = drifft.NUFFT(256)
    transition = drifft.Transition(backend, frequency_map=drifft.Underdominance(0.1))
    traces = []

    @eqx.filter_jit
    def expect(model, values):
        """Evaluate expectations for the differentiated probability or parameter input."""
        traces.append(1)
        return model.expect(values)

    values = jnp.sin(jnp.arange(513))
    first = expect(transition, values)
    changed = drifft.Transition(backend, frequency_map=drifft.Underdominance(0.2))
    second = expect(changed, values)
    assert len(traces) == 1
    assert not np.allclose(first, second)
    assert changed.backend is transition.backend
    derivative = jax.grad(
        lambda s: drifft.Transition(backend, drifft.Underdominance(s)).expect(values)[
            200
        ]
    )(0.1)
    h = 1e-5

    def evaluate(s):
        """Evaluate the scalar objective used to check dynamic parameters."""
        return drifft.Transition(backend, drifft.Underdominance(s)).expect(values)[200]

    np.testing.assert_allclose(
        derivative,
        (evaluate(0.1 + h) - evaluate(0.1 - h)) / (2 * h),
        rtol=1e-4,
        atol=1e-7,
    )


def test_custom_probabilities_drive_expected_visits():
    """Use caller-supplied probabilities in the occupation system."""
    backend = drifft.NUFFT(8)
    p = jnp.arange(17, dtype=jnp.float64) / 16
    transition = drifft.Transition(backend, probabilities=p)
    matrix = binom.pmf(np.arange(17)[None, :], 16, np.asarray(p)[:, None])
    rhs = np.zeros(15)
    rhs[0] = 1
    exact = np.linalg.solve(np.eye(15) - matrix[1:-1, 1:-1].T, rhs)
    result = transition.expected_visits()
    assert bool(result.converged)
    np.testing.assert_allclose(result.visits, exact, atol=1e-9)


def test_transition_rejects_ambiguous_or_invalid_configuration():
    """Reject competing selection modes and inconsistent source dimensions."""
    backend = drifft.NUFFT(8)
    with pytest.raises(ValueError):
        drifft.Transition(backend)
    with pytest.raises(ValueError):
        drifft.Transition(backend, probabilities=jnp.ones((2, 2)))
    with pytest.raises(ValueError):
        drifft.Transition(backend, probabilities=jnp.array([]))
    with pytest.raises(ValueError):
        drifft.Transition(backend, drifft.Underdominance(0.1), source_pop_size=0)
    with pytest.raises(ValueError):
        drifft.Underdominance(jnp.array([0.1, 0.2]))


def test_custom_model_gradients_through_advance_and_batched_solves():
    """Differentiate custom selection through propagation and batched visit solves."""
    backend = drifft.NUFFT(8)
    mass = jnp.zeros(17).at[7].set(1.0)

    def mean_next_count(bias):
        # Construct the callable inside differentiation so its parameter stays
        # connected to the outer trace rather than a precomputed probability map.
        """Return the propagated mean count for a custom selection model."""
        model = drifft.Transition(
            backend, frequency_map=drifft.FrequencyMap(lambda x: x + bias * x * (1 - x))
        )
        return model.advance(mass) @ jnp.arange(17, dtype=jnp.float64)

    derivative = jax.jit(jax.grad(mean_next_count))(jnp.asarray(0.02))
    np.testing.assert_allclose(derivative, 16 * (7 / 16) * (9 / 16), atol=1e-10)
    results = jax.vmap(
        lambda strength: drifft.Transition(
            backend, drifft.Underdominance(strength)
        ).expected_visits()
    )(jnp.array([0.0, 0.1]))
    assert bool(jnp.all(results.converged))
    assert results.visits.shape == (2, 15)


def test_transition_shape_errors_are_explicit():
    """Report vector shape mismatches before numerical application."""
    backend = drifft.NUFFT(8)
    model = drifft.Transition(backend, drifft.Underdominance(0.0))
    with pytest.raises(ValueError, match="distribution"):
        model.advance(jnp.ones(16))
    with pytest.raises(ValueError, match="interior"):
        model.advance(jnp.ones(16), segregating=True)
    with pytest.raises(ValueError, match="exactly one"):
        drifft.Transition(
            backend, drifft.Underdominance(0.0), probabilities=jnp.linspace(0, 1, 17)
        )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("shape", [(), (4,), (5, 1)])
def test_map_shape_is_checked_at_construction(compiled, shape):
    """Reject invalid map outputs without applying the constructed transition."""
    backend = drifft.Dense(4)

    def construct():
        """Require one map output for each of the five source states."""
        return drifft.Transition(
            backend,
            drifft.FrequencyMap(lambda x: jnp.full(shape, 0.5)),
            source_pop_size=2,
        )

    run_construct = eqx.filter_jit(construct) if compiled else construct
    with pytest.raises(ValueError, match="one probability per source frequency"):
        run_construct()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32, jnp.bool_, jnp.complex128])
def test_map_dtype_is_checked_at_construction(compiled, dtype):
    """Reject incompatible map dtypes instead of silently converting the output."""
    backend = drifft.Dense(2)

    def construct():
        """Return the correct grid shape with an incompatible dtype."""
        return drifft.Transition(
            backend, drifft.FrequencyMap(lambda x: x.astype(dtype))
        )

    run_construct = eqx.filter_jit(construct) if compiled else construct
    with pytest.raises(ValueError, match="float64 probabilities"):
        run_construct()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("kind", ["list", "tuple", "dict", "none"])
def test_map_output_must_be_one_array(compiled, kind):
    """Reject structured outputs even when their leaves have valid dtypes."""
    backend = drifft.Dense(2)

    def mapping(x):
        """Produce a non-array output for constructor validation."""
        if kind == "list":
            return list(x)
        if kind == "tuple":
            return (x,)
        if kind == "dict":
            return {"probabilities": x}
        return None

    def construct():
        """Bind the malformed map without applying the transition."""
        return drifft.Transition(backend, drifft.FrequencyMap(mapping))

    run_construct = eqx.filter_jit(construct) if compiled else construct
    with pytest.raises(ValueError, match="single float64 array"):
        run_construct()


@pytest.mark.parametrize("size", [0, -1, True, 2.5])
def test_invalid_source_pop_size(size):
    """Reject invalid source units before building a frequency grid."""
    with pytest.raises(ValueError, match="source_pop_size"):
        drifft.Transition(drifft.Dense(3), drifft.Neutral(), source_pop_size=size)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("length", [0, 1, 2, 4, 6])
def test_explicit_rows_require_complete_grid_lengths(compiled, length):
    """Reject row counts that cannot describe a positive diploid population."""
    backend = drifft.Dense(pop_size=3)

    def construct(probabilities):
        """Construct from explicit probabilities without applying the operator."""
        return drifft.Transition(backend, probabilities=probabilities)

    run_construct = eqx.filter_jit(construct) if compiled else construct
    with pytest.raises(ValueError, match="odd length at least three"):
        run_construct(jnp.linspace(0, 1, length))


def test_explicit_rows_infer_and_validate_pop_size():
    """Infer the source population or require the supplied size to agree."""
    backend = drifft.Dense(pop_size=3)
    transition = drifft.Transition(backend, probabilities=jnp.linspace(0, 1, 5))
    assert transition.source_pop_size == 2
    assert transition.source_size() == 5
    explicit = drifft.Transition(
        backend, probabilities=jnp.linspace(0, 1, 3), source_pop_size=1
    )
    assert explicit.source_size() == 3
    with pytest.raises(ValueError, match="match the probability vector"):
        drifft.Transition(
            backend, probabilities=jnp.linspace(0, 1, 5), source_pop_size=3
        )
