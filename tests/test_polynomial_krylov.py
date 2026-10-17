"""Polynomial power actions against independent dense and sequential references."""

# pattern: Functional Core
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from test_demography import dense_transition

import drifft
from drifft import _krylov
from drifft._epochs import _PowerProblem

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("steps", [0, 1, 2, 3, 8, 19, 500])
def test_reduced_power_action_and_derivatives(steps):
    """Check signed, nonnormal powers and their derivatives against recurrence."""
    matrix = jnp.array([[0.97, 0.08, -0.03], [0.0, 0.91, 0.05], [0.0, 0.0, 0.83]])
    seed = jnp.array([0.3, -0.5, 0.7])

    def reference(a, x):
        return jax.lax.fori_loop(0, steps, lambda _, v: a @ v, x)

    action = jax.jit(lambda a, x: _krylov._matrix_power_matvec(a, x, steps))
    direction = (jnp.ones_like(matrix) * 0.001, -seed * 0.1)
    actual, tangent = jax.jvp(action, (matrix, seed), direction)
    expected, expected_tangent = jax.jvp(reference, (matrix, seed), direction)
    assert jnp.allclose(actual, expected, rtol=1e-11, atol=1e-13)
    assert jnp.allclose(tangent, expected_tangent, rtol=1e-10, atol=1e-13)
    weights = jnp.array([1.0, -0.7, 0.2])
    actual_grad = jax.grad(lambda a: weights @ action(a, seed))(matrix)
    expected_grad = jax.grad(lambda a: weights @ reference(a, seed))(matrix)
    assert jnp.allclose(actual_grad, expected_grad, rtol=1e-10, atol=1e-13)
    assert jnp.allclose(
        jax.vmap(action, in_axes=(None, 0))(matrix, jnp.stack((seed, -seed))),
        jnp.stack((expected, -expected)),
        rtol=1e-11,
        atol=1e-13,
    )


@pytest.mark.parametrize("backend_type", [drifft.FFT, drifft.NUFFT, drifft.Jacobi])
@pytest.mark.parametrize("segregating", [False, True])
def test_polynomial_matches_dense_with_immigration(backend_type, segregating):
    """Match selected powers with signed mass and a source after every generation."""
    backend = backend_type(12)
    size = 11 if segregating else 13
    mass = jnp.linspace(-0.01, 0.04, size)
    source = jnp.zeros(size).at[1].set(0.002)
    matrix = dense_transition(6, 6, 0.04)
    if segregating:
        matrix = matrix[1:-1, 1:-1]
    expected = mass
    for _ in range(19):
        expected = matrix.T @ expected + source
    result = drifft.PolynomialKrylov(dimension=20).advance(
        backend,
        mass,
        frequency_map=drifft.Underdominance(0.04),
        generations=19,
        segregating=segregating,
        immigration=source,
    )
    assert result.converged
    assert jnp.allclose(result.distribution, expected, atol=2e-10, rtol=2e-9)


@pytest.mark.parametrize("generations", [0, 1, 2, 40])
def test_polynomial_zero_and_invariant_seeds(generations):
    """Zero mass and absorbing states remain finite, including Arnoldi breakdown."""
    method = drifft.PolynomialKrylov(dimension=8)
    backend = drifft.FFT(12)
    for mass in (jnp.zeros(13), jnp.eye(13)[0]):
        result = method.advance(
            backend, mass, frequency_map=drifft.Neutral(), generations=generations
        )
        assert result.converged
        assert jnp.allclose(result.distribution, mass, atol=1e-12)


def test_polynomial_uses_only_operator_applications():
    """Power an implicit shift with a known answer and no matrix construction."""
    operator = lx.FunctionLinearOperator(
        lambda x: jnp.concatenate((jnp.zeros(1), 0.5 * x[:-1])),
        jax.ShapeDtypeStruct((50,), jnp.float64),
    )
    seed = jnp.eye(50)[0]
    problem = _PowerProblem(
        operator, seed, 3, lambda x: x, jnp.zeros(50), False, jnp.asarray(1.0)
    )
    result = drifft.PolynomialKrylov(dimension=4).solve(problem)
    assert result.converged
    assert jnp.allclose(result.distribution, jnp.eye(50)[3] / 8, atol=1e-14)


def test_polynomial_nonconvergence_and_nonfinite_status():
    """An inadequate basis must not silently accept a long power."""
    method = drifft.PolynomialKrylov(dimension=2)
    backend = drifft.FFT(80)
    mass = jnp.eye(79)[0]
    result = method.advance(
        backend,
        mass,
        throw=False,
        frequency_map=drifft.Neutral(),
        generations=100,
        segregating=True,
    )
    assert not result.converged
    assert result.status == drifft.EvolutionStatus.unconverged
    with pytest.raises(eqx.EquinoxRuntimeError, match="epoch failed"):
        method.advance(
            backend,
            mass,
            frequency_map=drifft.Neutral(),
            generations=100,
            segregating=True,
        ).distribution.block_until_ready()
    result = method.advance(
        backend,
        mass.at[0].set(jnp.nan),
        throw=False,
        frequency_map=drifft.Neutral(),
        generations=100,
        segregating=True,
    )
    assert not result.converged
    assert result.status == drifft.EvolutionStatus.nonfinite


@pytest.mark.parametrize("backend_type", [drifft.FFT, drifft.NUFFT, drifft.Jacobi])
def test_polynomial_demography_and_gradients(backend_type):
    """Differentiate selected demographic evolution, mass and immigration."""
    model = drifft.DemographicTransition(
        drifft.Demography([4, 3, 5], [4, 3]),
        backend=backend_type,
        boundary_backend=drifft.FFT,
        method=drifft.PolynomialKrylov(dimension=16),
        frequency_map=drifft.Underdominance(0.03),
    )
    mass = jnp.linspace(0.01, 0.03, 7)
    weights = jnp.sin(jnp.arange(9))

    def objective(s, p, rate):
        updated = eqx.tree_at(lambda m: m.frequency_map.strength, model, s)
        return updated.advance_segregating(p, mutation_rate=rate).distribution @ weights

    def reference(s, p, rate):
        for a, b, g in ((4, 3, 4), (3, 5, 3)):
            for _ in range(g - 1):
                p = (p @ dense_transition(a, a, s)[1:-1, 1:-1]).at[0].add(2 * a * rate)
            p = (p @ dense_transition(a, b, s)[1:-1, 1:-1]).at[0].add(2 * b * rate)
        return p @ weights

    args = (jnp.asarray(0.03), mass, jnp.asarray(1e-5))
    actual = eqx.filter_jit(jax.grad(objective, argnums=(0, 1, 2)))(*args)
    expected = jax.grad(reference, argnums=(0, 1, 2))(*args)
    for a, b in zip(actual, expected, strict=True):
        assert jnp.allclose(a, b, rtol=2e-7, atol=2e-8)
    tangent = jax.tree.map(jnp.ones_like, args)
    assert jnp.allclose(
        jax.jvp(objective, args, tangent)[1],
        jax.jvp(reference, args, tangent)[1],
        rtol=2e-7,
        atol=2e-8,
    )
    batched = eqx.filter_jit(
        jax.vmap(lambda p: model.advance_segregating(p).distribution)
    )(jnp.stack((mass, 2 * mass)))
    assert jnp.allclose(batched[1], 2 * batched[0], atol=1e-10)
    with pytest.raises(ValueError, match="Sequential"):
        model.expect(jnp.ones(11))


@pytest.mark.parametrize("backend_type", [drifft.FFT, drifft.NUFFT])
@pytest.mark.parametrize("copies, generations", [(16, 25), (512, 12)])
def test_polynomial_composed_map_matches_sequential(backend_type, copies, generations):
    """FFT and NUFFT retain general frequency-map support under polynomial powering."""
    backend = backend_type(copies)
    frequency_map = drifft.Composition(
        drifft.Underdominance(0.4), drifft.Mutation(0.01, 0.02)
    )
    mass = jnp.zeros(copies + 1).at[copies // 3].set(1)
    expected = drifft.Sequential().advance(
        backend, mass, frequency_map=frequency_map, generations=generations
    )
    actual = drifft.PolynomialKrylov(dimension=20).advance(
        backend, mass, frequency_map=frequency_map, generations=generations
    )
    assert actual.converged
    assert jnp.allclose(actual.distribution, expected.distribution, atol=1e-10)


@pytest.mark.parametrize(
    "options",
    [{"dimension": 1}, {"dimension": True}, {"rtol": -1}, {"atol": float("nan")}],
)
def test_polynomial_configuration_validation(options):
    """Reject invalid static settings before tracing."""
    with pytest.raises(ValueError):
        drifft.PolynomialKrylov(**options)


@pytest.mark.parametrize("lower, converged", [(0.05, False), (0.7, True)])
def test_truncated_polynomial_power_and_jvp(lower, converged):
    """A truncated basis approximates a long diagonal power and its derivative."""
    diagonal = jnp.linspace(lower, 0.8, 50)
    seed = jnp.linspace(0.1, 1.0, 50)
    method = drifft.PolynomialKrylov(dimension=20, rtol=1e-7)

    def action(scale):
        operator = lx.FunctionLinearOperator(
            lambda x: scale * diagonal * x,
            jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        )
        problem = _PowerProblem(
            operator,
            seed,
            25,
            lambda x: x,
            jnp.zeros_like(seed),
            False,
            jnp.asarray(1.0),
        )
        return method.solve(problem, throw=False)

    result = eqx.filter_jit(action)(jnp.asarray(1.0))
    expected = diagonal**25 * seed
    # Broad spectra can have an accurate fine result while the nested-space
    # comparison conservatively rejects convergence.
    assert bool(result.converged) == converged
    assert jnp.allclose(result.distribution, expected, atol=1e-11, rtol=1e-7)
    tangent = jax.jvp(
        lambda scale: action(scale).distribution,
        (jnp.asarray(1.0),),
        (jnp.asarray(1.0),),
    )[1]
    step = 1e-5
    finite_difference = (
        action(1 + step).distribution - action(1 - step).distribution
    ) / (2 * step)
    assert jnp.allclose(tangent, finite_difference, atol=1e-10, rtol=2e-7)
    assert jnp.allclose(tangent, 25 * expected, atol=1e-10, rtol=2e-7)


@pytest.mark.parametrize("backend_type", [drifft.FFT, drifft.NUFFT, drifft.Jacobi])
@pytest.mark.parametrize("generations", [0, 1, 7])
def test_zero_initial_mass_with_immigration(backend_type, generations):
    """Immigration starts an empty population and is omitted for zero generations."""
    mass = jnp.zeros(11)
    source = mass.at[0].set(1e-3)
    matrix = dense_transition(6, 6, 0)[1:-1, 1:-1]
    expected = mass
    for _ in range(generations):
        expected = matrix.T @ expected + source
    result = drifft.PolynomialKrylov(dimension=16).advance(
        backend_type(12),
        mass,
        frequency_map=drifft.Neutral(),
        generations=generations,
        segregating=True,
        immigration=source,
    )
    assert result.converged
    assert jnp.allclose(result.distribution, expected, atol=1e-12)
