"""Independent sampling references, conservation, and linear transform contracts."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import hijax
from scipy.stats import hypergeom

import drifft
from drifft._sampling import _CountInterpolation

jax.config.update("jax_enable_x64", True)


def _matrix(population, sample):
    """Form independent exact rows for small reference problems."""
    if population <= 32:
        return np.array(
            [
                [
                    math.comb(j, k)
                    * math.comb(population - j, sample - k)
                    / math.comb(population, sample)
                    if k <= j and 0 <= sample - k <= population - j
                    else 0.0
                    for k in range(sample + 1)
                ]
                for j in range(population + 1)
            ]
        )
    return hypergeom.pmf(
        np.arange(sample + 1)[None, :],
        population,
        np.arange(population + 1)[:, None],
        sample,
    )


@pytest.mark.parametrize(
    "population,sample", [(2, 0), (2, 2), (12, 2), (24, 8), (24, 22), (24, 24)]
)
def test_exact_sampling(population, sample):
    """Uncompressed actions agree with exact sampling, including limiting cases."""
    sampling = drifft.HypergeometricSampling(
        pop_size=population // 2, sample_size=sample // 2, accuracy=0
    )
    assert sampling.pop_size == population // 2
    assert sampling.sample_size == sample // 2
    matrix = _matrix(population, sample)
    assert sampling.source_size() == matrix.shape[0]
    assert sampling.dest_size() == matrix.shape[1]
    assert isinstance(sampling.source_size(), int)
    assert isinstance(sampling.dest_size(), int)
    mass = jnp.linspace(-0.3, 1.0, sampling.source_size())
    values = jnp.cos(jnp.arange(sampling.dest_size()))
    np.testing.assert_allclose(
        sampling.advance(mass), matrix.T @ mass, atol=2e-14, rtol=0
    )
    np.testing.assert_allclose(
        sampling.expect(values), matrix @ values, atol=2e-14, rtol=0
    )


@pytest.mark.parametrize("sample", [16, 128, 1900])
def test_compression_accuracy(sample):
    """Tighter grouping resolves singleton, interior, and near-census rows."""
    population = 2048
    matrix = _matrix(population, sample)
    selected = np.array([0, 1, 2, 101, 333, 1024, 1701, 2047, 2048])
    initials = jax.nn.one_hot(selected, population + 1, dtype=jnp.float64)
    errors = []
    for tolerance in [1e-3, 1e-7]:
        sampling = drifft.HypergeometricSampling(
            population // 2, sample // 2, accuracy=tolerance
        )
        actual = jax.vmap(sampling.advance)(initials)
        errors.append(np.max(np.sum(np.abs(actual - matrix[selected]), axis=1) / 2))
        np.testing.assert_allclose(actual.sum(axis=1), 1, atol=2e-14, rtol=0)
        bound = sampling.interpolation_bound + sampling.tail_bound
        assert bound <= tolerance
        assert errors[-1] <= bound + 3e-12
        assert np.min(actual) >= -bound - 3e-12
        np.testing.assert_array_equal(actual[0], matrix[0])
        np.testing.assert_array_equal(actual[-1], matrix[-1])
    assert errors[1] < 0.002
    assert errors[1] <= errors[0] + 1e-14


def test_linearity_mean_and_adjoint():
    """Fixed interpolation preserves untruncated means and paired signed actions."""
    population, sample = 4000, 40
    sampling = drifft.HypergeometricSampling(
        population // 2, sample // 2, accuracy=1e-6, tail_tol=0
    )
    assert sampling.representative_counts.size < population // 10
    np.testing.assert_allclose(
        sampling.expect(jnp.arange(sample + 1)),
        sample * np.arange(population + 1) / population,
        atol=2e-13,
        rtol=0,
    )
    a = jnp.sin(jnp.arange(population + 1)) / population
    b = jnp.cos(jnp.arange(population + 1)) / population
    f = jnp.sin(jnp.arange(sample + 1))
    np.testing.assert_allclose(
        sampling.advance(2 * a - b),
        2 * sampling.advance(a) - sampling.advance(b),
        atol=1e-15,
        rtol=0,
    )
    np.testing.assert_allclose(
        sampling.advance(a) @ f, a @ sampling.expect(f), atol=1e-15, rtol=0
    )
    np.testing.assert_allclose(
        jax.grad(lambda x: sampling.advance(x) @ f)(a),
        sampling.expect(f),
        atol=1e-14,
        rtol=0,
    )
    np.testing.assert_allclose(
        jax.jvp(sampling.advance, (a,), (b,))[1],
        sampling.advance(b),
        atol=1e-15,
        rtol=0,
    )
    np.testing.assert_allclose(
        eqx.filter_jit(lambda op, x: op.expect(x))(sampling, f),
        sampling.expect(f),
        rtol=0,
    )


def test_tail_truncation_budget():
    """With no grouping, the actual row TV error stays within the tail budget."""
    population, sample, eps = 400, 200, 0.01
    sampling = drifft.HypergeometricSampling(
        population // 2, sample // 2, accuracy=2 * eps, interp_order=2, tail_tol=eps
    )
    actual = np.asarray(jax.vmap(sampling.advance)(jnp.eye(population + 1)))
    exact = _matrix(population, sample)
    assert np.max(np.sum(np.abs(actual - exact), axis=1) / 2) <= eps
    assert sampling._weight.size < exact.size / 2


@pytest.mark.parametrize(
    "backend", [drifft.Dense, drifft.FFT, drifft.NUFFT, drifft.Jacobi]
)
def test_demographic_composition(backend):
    """Sampling composes with demographic expectations and source offsets."""
    model = drifft.DemographicTransition(
        drifft.Demography(pop_sizes=[16, 12], generations=[3]).prepare(
            backend_factory=backend
        ),
        method=drifft.Sequential(),
        frequency_map=drifft.Neutral(),
    )
    sampling = drifft.HypergeometricSampling(12, 4)
    initial = jnp.zeros(33).at[1].set(1)
    values = jnp.arange(9) ** 2
    result = model.advance(initial)
    expectation = model.expect(sampling.expect(values))
    np.testing.assert_allclose(
        sampling.advance(result.distribution) @ values,
        initial @ expectation.values,
        atol=1e-12,
        rtol=0,
    )
    # Interior mass is padded explicitly: sampling endpoints remain observable.
    interior = jnp.zeros(31)
    result = model.advance(interior, segregating=True, singleton_mutation_rate=1e-5)
    reward = sampling.expect(values)[1:-1]
    expectation = model.expect(reward, segregating=True, singleton_mutation_rate=1e-5)
    np.testing.assert_allclose(
        sampling.advance(jnp.pad(result.distribution, (1, 1))) @ values,
        interior @ expectation.values + expectation.offset,
        atol=1e-14,
        rtol=0,
    )


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((0, 0), {}),
        ((-1, 0), {}),
        ((1.5, 1), {}),
        ((True, 1), {}),
        ((8, True), {}),
        ((8, 9), {}),
        ((8, -1), {}),
        ((8, 2.0), {}),
        ((8, 2), {"accuracy": float("nan")}),
        ((8, 2), {"accuracy": -1}),
        ((8, 2), {"tail_tol": 1}),
        ((8, 2), {"tail_tol": -0.1}),
        ((8, 2), {"interp_order": 3}),
        ((8, 2), {"batch_size": 0}),
    ],
)
def test_invalid_configuration(args, kwargs):
    """Reject invalid static geometry before preparing rows."""
    with pytest.raises(ValueError):
        drifft.HypergeometricSampling(*args, **kwargs)


def test_vector_contract_and_x64():
    """Reject wrong shapes, complex values, and construction without float64."""
    sampling = drifft.HypergeometricSampling(5, 2)
    for values in [jnp.ones(3), jnp.ones((2, 4)), jnp.ones(5, dtype=complex)]:
        with pytest.raises(ValueError):
            sampling.expect(values)
    with pytest.raises(ValueError):
        sampling.advance(jnp.ones(10))
    with jax.enable_x64(False), pytest.raises(ValueError, match="x64"):
        drifft.HypergeometricSampling(5, 2)


def test_compressed_rows_match_integer_combinatorics():
    """Exercise actual grouping against exact arithmetic, including reflection."""
    population, sample = 5000, 12
    sampling = drifft.HypergeometricSampling(
        population // 2, sample // 2, accuracy=1e-6, tail_tol=0
    )
    selected = np.array([3, 71, 851, 2191])
    selected = np.concatenate((selected, population - selected))
    initial = jax.nn.one_hot(selected, population + 1, dtype=jnp.float64)
    actual = np.asarray(jax.vmap(sampling.advance)(initial))
    denominator = math.comb(population, sample)
    exact = np.array(
        [
            [
                math.comb(int(j), k)
                * math.comb(population - int(j), sample - k)
                / denominator
                if k <= j and sample - k <= population - j
                else 0.0
                for k in range(sample + 1)
            ]
            for j in selected
        ]
    )
    assert sampling.representative_counts.size < population // 10
    assert np.max(np.sum(np.abs(actual - exact), axis=1) / 2) < 0.002
    np.testing.assert_allclose(actual[:4], actual[4:, ::-1], rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        actual @ np.arange(sample + 1),
        sample * selected / population,
        rtol=0,
        atol=1e-13,
    )
    # Cubic and higher stencils preserve the quadratic conditional variance.
    actual_variance = (
        actual @ np.arange(sample + 1) ** 2 - (sample * selected / population) ** 2
    )
    exact_variance = (
        sample
        * (selected / population)
        * (1 - selected / population)
        * (population - sample)
        / (population - 1)
    )
    np.testing.assert_allclose(actual_variance, exact_variance, rtol=0, atol=2e-13)


@pytest.mark.parametrize("sample", [0, 2, 6, 18])
def test_expect_batching_and_reverse_derivative(sample):
    """Expectation batches and pullbacks match independent finite-population rows."""
    population = 18
    sampling = drifft.HypergeometricSampling(population // 2, sample // 2, accuracy=0)
    exact = _matrix(population, sample)
    values = jnp.stack(
        (jnp.sin(jnp.arange(sample + 1)), jnp.cos(jnp.arange(sample + 1)))
    )
    mass = jnp.linspace(-0.5, 0.9, population + 1)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(sampling.expect))(values), values @ exact.T, rtol=0, atol=1e-14
    )
    gradient = jax.grad(lambda f: mass @ sampling.expect(f))(values[0])
    np.testing.assert_allclose(gradient, exact.T @ mass, rtol=0, atol=1e-14)


@pytest.mark.parametrize("order", [1, 2, 4, 8])
def test_hypergeometric_difference_identity(order):
    """Check the planning identity against exact combinatorics on a finite grid."""
    population, sample = 17, 7
    rows = _matrix(population, sample)
    differences = np.diff(rows, n=order, axis=0)
    if order > sample:
        np.testing.assert_allclose(differences, 0, rtol=0, atol=3e-14)
        return
    reduced = _matrix(population - order, sample - order)
    expected = np.zeros_like(differences)
    factor = math.prod((sample - k) / (population - k) for k in range(order))
    for shift in range(order + 1):
        expected[:, shift : shift + sample - order + 1] += (
            factor * (-1) ** (order - shift) * math.comb(order, shift) * reduced
        )
    np.testing.assert_allclose(differences, expected, rtol=0, atol=3e-14)


@pytest.mark.parametrize("order", [2, 4, 8, 12, 16])
@pytest.mark.parametrize("population", [1000, 1002])
def test_planning_bounds_on_every_population_row(order, population):
    """The combined interpolation and tail budget covers all rows, not a few probes."""
    sample, accuracy = 18, 1e-4
    sampling = drifft.HypergeometricSampling(
        population // 2, sample // 2, accuracy=accuracy, interp_order=order
    )
    actual = np.asarray(jax.vmap(sampling.expect)(jnp.eye(sample + 1))).T
    exact = _matrix(population, sample)
    bound = sampling.interpolation_bound + sampling.tail_bound
    assert bound <= accuracy
    assert np.max(np.sum(np.abs(actual - exact), axis=1) / 2) <= bound + 2e-12
    # One-sided order-16 stencils amplify floating-point row-sum errors.
    np.testing.assert_allclose(actual.sum(axis=1), 1, rtol=0, atol=2e-13)
    # Bound actual signed interpolation weights independently of row probabilities.
    from drifft._sampling import _interpolate

    if sampling.representative_counts.size < population + 1:
        size = sampling.representative_counts.size
        interpolation = jax.vmap(
            lambda x: _interpolate(
                sampling.representative_counts,
                sampling._barycentric,
                x,
                population=population,
                batch_size=128,
                transposed=False,
            )
        )(jnp.eye(size))
        assert np.max(np.sum(np.abs(interpolation), axis=0)) <= sampling.lebesgue_bound


@pytest.mark.parametrize("sample", [2, 4, 6])
def test_exact_polynomial_dimension(sample):
    """A sample-size degree polynomial recovers every row without heuristic grouping."""
    population = 1000
    sampling = drifft.HypergeometricSampling(population // 2, sample // 2, tail_tol=0)
    assert sampling.representative_counts.size <= sample + 2
    assert sampling.interpolation_bound == 0
    actual = np.asarray(jax.vmap(sampling.expect)(jnp.eye(sample + 1))).T
    np.testing.assert_allclose(actual, _matrix(population, sample), rtol=0, atol=2e-14)


def test_streaming_ad_and_batch_sizes():
    """Streaming transpose supports nonleading batching and second derivatives."""
    population, sample = 2000, 10
    sampling = drifft.HypergeometricSampling(
        population // 2, sample // 2, batch_size=37
    )
    other = drifft.HypergeometricSampling(population // 2, sample // 2, batch_size=256)
    a = jnp.sin(jnp.arange(population + 1)) / population
    np.testing.assert_allclose(
        sampling.advance(a), other.advance(a), rtol=0, atol=2e-15
    )
    batch = jnp.stack((a, 2 * a), axis=1)
    np.testing.assert_allclose(
        jax.vmap(sampling.advance, in_axes=1, out_axes=1)(batch),
        jnp.stack((sampling.advance(a), 2 * sampling.advance(a)), axis=1),
        rtol=0,
        atol=2e-15,
    )
    f = jnp.sin(jnp.arange(sample + 1))
    gradient = sampling.expect(f)

    # For 1/2 (f^T S^T x)^2, Hessian-vector products are known analytically.
    def fun(x):
        """Compose sampling with a quadratic observable."""
        return 0.5 * (sampling.advance(x) @ f) ** 2

    direction = jnp.cos(jnp.arange(population + 1)) / population
    hvp = jax.jvp(jax.grad(fun), (a,), (direction,))[1]
    np.testing.assert_allclose(
        hvp, gradient * (gradient @ direction), rtol=0, atol=2e-14
    )


@pytest.mark.parametrize("action", ["advance", "expect"])
def test_tangent_only_batching_preserves_shared_primal(action):
    """Batch nonleading tangent axes without broadcasting the fixed primal."""
    population, sample = 60, 4
    sampling = drifft.HypergeometricSampling(population // 2, sample // 2, tail_tol=0)
    assert sampling.representative_counts.size < population + 1
    matrix = jnp.asarray(_matrix(population, sample))
    operator = matrix.T if action == "advance" else matrix
    values = jnp.sin(jnp.arange(operator.shape[1]))
    directions = jnp.stack((jnp.ones_like(values), -values), axis=1)
    fn = getattr(sampling, action)
    primal, tangent = jax.jit(
        jax.vmap(
            lambda d: jax.jvp(fn, (values,), (d,)),
            in_axes=1,
            out_axes=(None, 1),
        )
    )(directions)
    np.testing.assert_allclose(primal, operator @ values, rtol=0, atol=2e-13)
    np.testing.assert_allclose(tangent, operator @ directions, rtol=0, atol=2e-13)


@pytest.mark.parametrize("transposed", [False, True])
def test_interpolation_preserves_symbolic_zeros_and_geometry_guard(transposed):
    """Inactive derivatives avoid dense zeros but still reject trainable geometry."""
    sampling = drifft.HypergeometricSampling(30, 2, tail_tol=0)
    knots, barycentric = sampling.representative_counts, sampling._barycentric
    values = jnp.ones(61 if transposed else knots.size)
    primals = (knots, barycentric, values)
    primitive = _CountInterpolation(
        *map(jax.typeof, primals),
        population=60,
        batch_size=7,
        transposed=transposed,
    )
    zeros = tuple(hijax.Zero(jax.typeof(x).to_tangent_aval()) for x in primals)
    _, tangent = primitive.jvp(primals, zeros)
    assert isinstance(tangent, hijax.Zero)
    assert tangent.aval == primitive.out_aval.to_tangent_aval()
    out_ct = hijax.Zero(primitive.out_aval.to_tangent_aval())
    accum = hijax.ValAccum(jax.typeof(values))
    primitive.transpose(out_ct, knots, barycentric, accum)
    assert isinstance(accum.freeze(), hijax.Zero)
    with pytest.raises(ValueError, match="fixed geometry"):
        primitive.jvp(primals, (zeros[0], jnp.zeros_like(barycentric), zeros[2]))
    with pytest.raises(ValueError, match="fixed geometry"):
        primitive.transpose(
            out_ct, knots, hijax.ValAccum(jax.typeof(barycentric)), values
        )


@pytest.mark.parametrize("transposed", [False, True])
def test_interpolation_pullback_retains_only_geometry(transposed):
    """A fixed linear interpolation pullback must not retain the input vector."""
    sampling = drifft.HypergeometricSampling(30, 2, tail_tol=0)
    knots, barycentric = sampling.representative_counts, sampling._barycentric
    values = jnp.ones(61 if transposed else knots.size)
    primitive = _CountInterpolation(
        *map(jax.typeof, (knots, barycentric, values)),
        population=60,
        batch_size=7,
        transposed=transposed,
    )
    _, pullback = jax.vjp(lambda x: primitive(knots, barycentric, x), values)
    # Bound retained array storage, without depending on residual tuple layout.
    retained_bytes = sum(
        leaf.nbytes for leaf in jax.tree.leaves(pullback) if eqx.is_array(leaf)
    )
    assert retained_bytes <= knots.nbytes + barycentric.nbytes


@pytest.mark.parametrize("action", ["advance", "expect"])
def test_sampling_reverse_mode_compositions(action):
    """Check checkpointing, shared gradients and higher derivatives against exact rows."""
    sampling = drifft.HypergeometricSampling(30, 2, tail_tol=0, batch_size=7)
    matrix = jnp.asarray(_matrix(60, 4))
    operator = matrix.T if action == "advance" else matrix
    fn = getattr(sampling, action)
    values = jnp.sin(jnp.arange(operator.shape[1]))
    weights = jnp.cos(jnp.arange(operator.shape[0]))
    direction = jnp.cos(jnp.arange(values.size))
    coefficient = operator.T @ weights

    def loss(x):
        return 0.5 * (fn(x) @ weights) ** 2

    expected_gradient = coefficient * (coefficient @ values)
    actual = jax.jit(jax.grad(jax.checkpoint(loss)))(values)
    np.testing.assert_allclose(actual, expected_gradient, rtol=2e-12, atol=2e-12)

    def shared_loss(x):
        return jax.vmap(lambda scale: scale * loss(x))(jnp.array([1.0, 2.0, 3.0])).sum()

    np.testing.assert_allclose(
        jax.jit(jax.grad(shared_loss))(values),
        6 * expected_gradient,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda x: loss(x) + 2 * loss(x)))(values),
        3 * expected_gradient,
        rtol=2e-12,
        atol=2e-12,
    )
    expected_hvp = coefficient * (coefficient @ direction)
    forward_over_reverse = jax.jit(
        lambda x: jax.jvp(jax.grad(loss), (x,), (direction,))[1]
    )(values)
    reverse_over_reverse = jax.jit(jax.grad(lambda x: jax.grad(loss)(x) @ direction))(
        values
    )
    for actual in (forward_over_reverse, reverse_over_reverse):
        np.testing.assert_allclose(actual, expected_hvp, rtol=2e-12, atol=2e-12)
