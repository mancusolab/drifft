"""Independent dense references for compact SPD pentadiagonal Lineax solves."""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from drifft._banded import PentadiagonalCholesky, SymmetricPentadiagonalOperator

jax.config.update("jax_enable_x64", True)


def _bands(size, dtype):
    """Build a strictly diagonally dominant real symmetric positive matrix."""
    return (
        jnp.linspace(2.0, 3.0, size, dtype=dtype),
        jnp.full(size - 1, 0.1, dtype=dtype),
        jnp.full(max(size - 2, 0), -0.07, dtype=dtype),
    )


def _dense(diagonal, adjacent, second):
    """Construct an independent dense reference from the stated band convention."""
    matrix = jnp.diag(diagonal)
    if diagonal.size > 1:
        matrix += jnp.diag(adjacent, 1) + jnp.diag(adjacent, -1)
    if diagonal.size > 2:
        matrix += jnp.diag(second, 2) + jnp.diag(second, -2)
    return matrix


@pytest.mark.parametrize("size", [1, 2, 7, 32])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_banded_values_and_reused_state(size, dtype):
    """Match matrix actions and multiple right-hand sides, including short bands."""
    bands = _bands(size, dtype)
    operator = SymmetricPentadiagonalOperator(*bands)
    matrix = _dense(*bands)
    rhs = jnp.stack((jnp.linspace(-0.3, 0.8, size, dtype=dtype), jnp.ones(size, dtype)))
    solver = PentadiagonalCholesky()
    state = solver.init(operator, {})
    tolerance = 2e-6 if dtype == jnp.float32 else 2e-14
    actual = jax.jit(
        jax.vmap(lambda b: lx.linear_solve(operator, b, solver, state=state).value)
    )(rhs)
    expected = jax.vmap(lambda b: jnp.linalg.solve(matrix, b))(rhs)
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(operator.mv(rhs[0]), matrix @ rhs[0], atol=tolerance)
    np.testing.assert_array_equal(operator.as_matrix(), matrix)
    assert actual.dtype == dtype
    assert operator.T is operator
    assert lx.conj(operator) is operator
    assert lx.linearise(operator) is operator
    assert lx.materialise(operator) is operator
    assert lx.is_symmetric(operator)


def test_banded_joint_derivatives_and_hessian():
    """Differentiate all stored bands and the RHS through reused factor state."""
    size = 7
    bands = _bands(size, jnp.float64)
    point = jnp.concatenate((*bands, jnp.linspace(-0.4, 0.8, size)))
    direction = jnp.sin(jnp.arange(point.size)) * 0.05
    reward = jnp.cos(jnp.arange(size) * 0.4)

    def objective(parameters, *, dense=False):
        """Use independent dense or compact implicit solves for one scalar reward."""
        diagonal = parameters[:size]
        adjacent = parameters[size : 2 * size - 1]
        second = parameters[2 * size - 1 : 3 * size - 3]
        rhs = parameters[3 * size - 3 :]
        if dense:
            value = jnp.linalg.solve(_dense(diagonal, adjacent, second), rhs)
        else:
            operator = SymmetricPentadiagonalOperator(diagonal, adjacent, second)
            solver = PentadiagonalCholesky()
            state = solver.init(operator, {})
            value = lx.linear_solve(operator, rhs, solver, state=state).value
        return reward @ value

    def reference(parameters):
        """Differentiate an independent dense solve with the same scalar reward."""
        return objective(parameters, dense=True)

    actual = jax.jit(jax.value_and_grad(objective))(point)
    expected = jax.jit(jax.value_and_grad(reference))(point)
    np.testing.assert_allclose(actual[0], expected[0], atol=2e-14)
    np.testing.assert_allclose(actual[1], expected[1], atol=3e-14)
    actual_jvp = jax.jit(lambda p, d: jax.jvp(objective, (p,), (d,))[1])(
        point, direction
    )
    expected_jvp = jax.jvp(reference, (point,), (direction,))[1]
    np.testing.assert_allclose(actual_jvp, expected_jvp, atol=3e-14)
    np.testing.assert_allclose(
        jax.jit(jax.hessian(objective))(point),
        jax.jit(jax.hessian(reference))(point),
        atol=8e-14,
    )


def test_banded_solver_never_materializes_dense_matrix(monkeypatch):
    """Keep forward and differentiated solves on linear-storage band actions."""

    def forbidden(*args):
        """Fail if a generic fallback attempts dense operator materialization."""
        raise AssertionError("dense materialization is not permitted")

    monkeypatch.setattr(SymmetricPentadiagonalOperator, "as_matrix", forbidden)
    bands = _bands(64, jnp.float64)
    rhs = jnp.sin(jnp.arange(64) * 0.1)

    def solve(diagonal):
        """Exercise both the implicit derivative and the compact primal path."""
        operator = SymmetricPentadiagonalOperator(diagonal, *bands[1:])
        return lx.linear_solve(operator, rhs, PentadiagonalCholesky()).value.sum()

    function = jax.jit(jax.value_and_grad(solve))
    hlo = function.lower(bands[0]).as_text()
    assert "64x64" not in hlo
    result = function(bands[0])
    assert all(jnp.all(jnp.isfinite(x)) for x in result)


@pytest.mark.parametrize("diagonal", [[0.0, 2.0], [-1.0, 2.0], [float("nan"), 2.0]])
def test_banded_invalid_pivots_report_lineax_failure(diagonal):
    """Invalid SPD inputs report status or raise according to Lineax throw."""
    operator = SymmetricPentadiagonalOperator(
        jnp.array(diagonal), jnp.zeros(1), jnp.zeros(0)
    )
    solver = PentadiagonalCholesky()
    result = lx.linear_solve(operator, jnp.ones(2), solver, throw=False)
    assert result.result == lx.RESULTS.singular
    with pytest.raises(eqx.EquinoxRuntimeError):
        eqx.filter_jit(lambda b: lx.linear_solve(operator, b, solver).value)(
            jnp.ones(2)
        ).block_until_ready()


@pytest.mark.parametrize(
    "bands",
    [
        ([], [], []),
        ([1.0, 2.0], [], []),
        ([1.0, 2.0, 3.0], [0.0, 0.0], []),
        ([1, 2], [0], []),
    ],
)
def test_banded_operator_rejects_invalid_structure(bands):
    """Reject unsupported shapes and integer storage before entering a solve."""
    with pytest.raises(ValueError):
        SymmetricPentadiagonalOperator(*bands)


def test_banded_solver_rejects_other_operators():
    """Do not silently densify an unsupported representation."""
    operator = lx.MatrixLinearOperator(jnp.eye(3))
    with pytest.raises(TypeError, match="SymmetricPentadiagonalOperator"):
        PentadiagonalCholesky().init(operator, {})


def test_banded_small_positive_pivot_matches_dense():
    """Retain valid SPD systems whose smallest pivot is far below the others."""
    size = 9
    lower = jnp.diag(jnp.linspace(0.8, 1.2, size).at[0].set(1e-4))
    lower += jnp.diag(jnp.full(size - 1, 0.03), -1)
    lower += jnp.diag(jnp.full(size - 2, -0.02), -2)
    matrix = lower @ lower.T
    operator = SymmetricPentadiagonalOperator(
        jnp.diag(matrix), jnp.diag(matrix, 1), jnp.diag(matrix, 2)
    )
    rhs = jnp.linspace(-0.2, 0.7, size)
    result = lx.linear_solve(operator, rhs, PentadiagonalCholesky())
    assert result.result == lx.RESULTS.successful
    np.testing.assert_allclose(
        result.value, jnp.linalg.solve(matrix, rhs), rtol=2e-10, atol=2e-10
    )
