"""Independent sampling references, conservation, and linear transform contracts."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import hypergeom

import drifft

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
    "population,sample", [(1, 0), (1, 1), (11, 1), (24, 7), (24, 23), (24, 24)]
)
def test_exact_sampling(population, sample):
    """Uncompressed actions agree with exact sampling, including limiting cases."""
    sampling = drifft.HypergeometricSampling(population, sample, tv_sd=0, row_eps=0)
    matrix = _matrix(population, sample)
    mass = jnp.linspace(-0.3, 1.0, population + 1)
    values = jnp.cos(jnp.arange(sample + 1))
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
    for spacing in [0.4, 0.1]:
        sampling = drifft.HypergeometricSampling(population, sample, tv_sd=spacing)
        actual = jax.vmap(sampling.advance)(initials)
        errors.append(np.max(np.sum(np.abs(actual - matrix[selected]), axis=1) / 2))
        np.testing.assert_allclose(actual.sum(axis=1), 1, atol=2e-14, rtol=0)
        assert np.min(actual) >= 0
        np.testing.assert_array_equal(actual[0], matrix[0])
        np.testing.assert_array_equal(actual[-1], matrix[-1])
    assert errors[1] < 0.002
    assert errors[1] <= errors[0] + 1e-14


def test_linearity_mean_and_adjoint():
    """Fixed interpolation preserves untruncated means and paired signed actions."""
    population, sample = 4000, 40
    sampling = drifft.HypergeometricSampling(population, sample, tv_sd=0.3, row_eps=0)
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
    sampling = drifft.HypergeometricSampling(population, sample, tv_sd=0, row_eps=eps)
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
        drifft.Demography(ne=[16, 12], generations=[3]),
        backend_factory=backend,
        method=drifft.Sequential(),
        frequency_map=drifft.Neutral(),
    )
    sampling = drifft.HypergeometricSampling(24, 8)
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
        ((True, 1), {}),
        ((8, True), {}),
        ((8, 9), {}),
        ((8, -1), {}),
        ((8, 2.0), {}),
        ((8, 2), {"tv_sd": float("nan")}),
        ((8, 2), {"tv_sd": -1}),
        ((8, 2), {"row_eps": 1}),
        ((8, 2), {"row_eps": -0.1}),
    ],
)
def test_invalid_configuration(args, kwargs):
    """Reject invalid static geometry before preparing rows."""
    with pytest.raises(ValueError):
        drifft.HypergeometricSampling(*args, **kwargs)


def test_vector_contract_and_x64():
    """Reject wrong shapes, complex values, and construction without float64."""
    sampling = drifft.HypergeometricSampling(10, 3)
    for values in [jnp.ones(3), jnp.ones((2, 4)), jnp.ones(4, dtype=complex)]:
        with pytest.raises(ValueError):
            sampling.expect(values)
    with pytest.raises(ValueError):
        sampling.advance(jnp.ones(10))
    with jax.enable_x64(False), pytest.raises(ValueError, match="x64"):
        drifft.HypergeometricSampling(10, 3)


def test_compressed_rows_match_integer_combinatorics():
    """Exercise actual grouping against exact arithmetic, including reflection."""
    population, sample = 5000, 12
    sampling = drifft.HypergeometricSampling(population, sample, tv_sd=0.1, row_eps=0)
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
    # Mixtures of rows preserve means but broaden conditional distributions.
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
    assert np.min(actual_variance - exact_variance) >= -1e-13


@pytest.mark.parametrize("sample", [0, 1, 5, 17])
def test_expect_batching_and_reverse_derivative(sample):
    """Expectation batches and pullbacks match independent finite-population rows."""
    population = 17
    sampling = drifft.HypergeometricSampling(population, sample, tv_sd=0, row_eps=0)
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
