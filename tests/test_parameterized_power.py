"""Parameter-space power semantics and prepared-operator compatibility."""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import drifft
from drifft._evolution.autodiff import _power
from drifft._evolution.parameterized import _JacobiOperator, _pack

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("augmented", [False, True])
def test_compact_power_matches_repeated_dense_action(transposed, augmented):
    """Check shared-operator accumulation, source gradients, JVP and mapped AD."""
    backend = drifft.Jacobi(4)
    method = drifft.RationalKrylov(dimension=16)
    parameters = jnp.array([0.03, 0.01, 0.02])
    source = jnp.linspace(0.01, 0.03, backend.rank)
    initial = jnp.linspace(-0.2, 0.3, backend.rank + augmented)

    def operator(p, source):
        return _JacobiOperator(
            backend,
            drifft.Composition(
                drifft.AdditiveSelection(p[0]), drifft.Mutation(p[1], p[2])
            ),
            source if augmented else None,
            True,
            transposed,
        )

    def objective(p, source, x, *, native):
        op = operator(p, source)
        matrix = op.as_matrix()
        if native:
            op = _pack(op, jax.lax.stop_gradient(matrix))
            # Reusing the opaque value requires addition in parameter space.
            result = _power(op, x, method, 3) + _power(op, x * 0.7, method, 2)
        else:
            result = matrix @ matrix @ matrix @ x + matrix @ matrix @ (x * 0.7)
        return jnp.cos(jnp.arange(x.size)) @ result

    def actual(p, source, x):
        return objective(p, source, x, native=True)

    def reference(p, source, x):
        return objective(p, source, x, native=False)

    inputs = parameters, source, initial
    directions = parameters * 0.1, source * 0.2, initial * -0.3
    for transform in (
        lambda fn: jax.value_and_grad(fn, argnums=(0, 1, 2)),
        lambda fn: lambda *args: jax.jvp(fn, args, directions),
    ):
        expected = transform(reference)(*inputs)
        result = jax.jit(transform(actual))(*inputs)
        for a, b in zip(
            jax.tree.leaves(result), jax.tree.leaves(expected), strict=True
        ):
            assert jnp.allclose(a, b, rtol=2e-8, atol=2e-10)
    batch = jnp.stack((parameters, parameters * 0.8))
    result = jax.jit(jax.vmap(jax.grad(actual), in_axes=(0, None, None)))(
        batch, source, initial
    )
    expected = jax.vmap(jax.grad(reference), in_axes=(0, None, None))(
        batch, source, initial
    )
    assert jnp.allclose(result, expected, rtol=2e-8, atol=2e-10)
    result = jax.jit(
        jax.grad(
            lambda p: jnp.sum(
                jax.vmap(actual, in_axes=(None, None, 0))(
                    p, source, jnp.stack((initial, -initial * 0.5))
                )
            )
        )
    )(parameters)
    expected = jax.grad(lambda p: reference(p, source, initial * 0.5))(parameters)
    assert jnp.allclose(result, expected, rtol=2e-8, atol=2e-10)


def test_operator_cotangent_is_parameter_vector():
    """A matrix cache must not become a dense matrix cotangent."""
    backend = drifft.Jacobi(4)
    op = _JacobiOperator(backend, drifft.AdditiveSelection(0.02), None, False)
    opaque = _pack(op, op.as_matrix())
    initial = jnp.linspace(0.1, 0.4, backend.rank + 2)
    gradient = jax.jit(
        jax.grad(
            lambda a: jnp.sum(
                _power(a, initial, drifft.RationalKrylov(dimension=16), 3)
            )
        )
    )(opaque)
    assert gradient.shape == (1,)
    expected = jax.grad(
        lambda s: jnp.sum(
            jnp.linalg.matrix_power(
                _JacobiOperator(
                    backend, drifft.AdditiveSelection(s), None, False
                ).as_matrix(),
                3,
            )
            @ initial
        )
    )(jnp.asarray(0.02))
    assert jnp.allclose(gradient[0], expected, rtol=1e-9, atol=1e-11)


def test_prepared_operator_keeps_parameter_derivatives():
    """Custom solvers can differentiate the ordinary Lineax prepared operator."""
    backend = drifft.Jacobi(4)
    mass = jnp.linspace(0.01, 0.03, 9)
    method = drifft.RationalKrylov(dimension=16)

    def prepared(s):
        problem = drifft.prepare_epoch(
            method,
            backend,
            mass,
            frequency_map=drifft.AdditiveSelection(s),
            generations=4,
        )
        return problem.reconstruct(
            problem.operator.mv(
                problem.operator.mv(problem.operator.mv(problem.initial))
            )
        )

    def builtin(s):
        return method.advance(
            backend, mass, frequency_map=drifft.AdditiveSelection(s), generations=4
        ).distribution

    value = jnp.asarray(0.02)
    actual = eqx.filter_jit(jax.jacfwd(prepared))(value)
    expected = eqx.filter_jit(jax.jacfwd(builtin))(value)
    assert jnp.allclose(actual, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("transposed", [False, True])
def test_tangent_only_batch_and_linear_solve(transposed):
    """Share the primal across JVP directions and support Lineax custom solvers."""
    backend = drifft.Jacobi(4)
    method = drifft.RationalKrylov(dimension=16)
    parameters = jnp.array([0.02, 0.01])
    initial = jnp.linspace(0.1, 0.3, backend.rank)

    def operator(p):
        return _JacobiOperator(backend, drifft.Mutation(*p), None, True, transposed)

    def action(p):
        op = operator(p)
        return _power(
            _pack(op, jax.lax.stop_gradient(op.as_matrix())), initial, method, 3
        )

    directions = jnp.eye(2)
    result = jax.jit(jax.vmap(lambda d: jax.jvp(action, (parameters,), (d,))))(
        directions
    )
    expected = jax.vmap(
        lambda d: jax.jvp(
            lambda p: jnp.linalg.matrix_power(operator(p).as_matrix(), 3) @ initial,
            (parameters,),
            (d,),
        )
    )(directions)
    for a, b in zip(result, expected, strict=True):
        assert jnp.allclose(a, b, rtol=1e-8, atol=1e-10)

    def solve(p, *, dense):
        op = operator(p)
        shifted = lx.IdentityLinearOperator(op.in_structure()) * 2 - op
        result = (
            jnp.linalg.solve(shifted.as_matrix(), initial)
            if dense
            else lx.linear_solve(shifted, initial, lx.LU()).value
        )
        return jnp.sum(result)

    actual = jax.jit(jax.value_and_grad(lambda p: solve(p, dense=False)))(parameters)
    expected = jax.value_and_grad(lambda p: solve(p, dense=True))(parameters)
    for a, b in zip(actual, expected, strict=True):
        assert jnp.allclose(a, b, rtol=1e-10, atol=1e-12)


def test_source_only_operator_has_no_process_directions():
    """A neutral map still propagates encoded-source and state cotangents."""
    backend = drifft.Jacobi(4)
    source = jnp.linspace(0.001, 0.003, backend.rank)
    initial = jnp.linspace(0.1, 0.4, backend.rank + 1)
    method = drifft.RationalKrylov(dimension=16)

    def objective(source, *, native):
        op = _JacobiOperator(backend, drifft.Neutral(), source, True)
        matrix = op.as_matrix()
        result = (
            _power(_pack(op, jax.lax.stop_gradient(matrix)), initial, method, 3)
            if native
            else jnp.linalg.matrix_power(matrix, 3) @ initial
        )
        return jnp.cos(jnp.arange(initial.size)) @ result

    actual = jax.jit(jax.grad(lambda x: objective(x, native=True)))(source)
    expected = jax.grad(lambda x: objective(x, native=False))(source)
    assert jnp.allclose(actual, expected, rtol=1e-9, atol=1e-11)


def test_nested_parameter_batches():
    """Parameter grids preserve the prepared power under nested mapping."""
    backend = drifft.Jacobi(3)
    initial = jnp.linspace(0.1, 0.3, backend.rank)
    method = drifft.RationalKrylov(dimension=12)

    def action(rate, *, native):
        op = _JacobiOperator(backend, drifft.Mutation(rate, 0.02), None, True)
        matrix = op.as_matrix()
        return (
            _power(_pack(op, jax.lax.stop_gradient(matrix)), initial, method, 3)
            if native
            else jnp.linalg.matrix_power(matrix, 3) @ initial
        )

    parameters = jnp.array([[0.01, 0.02], [0.03, 0.04]])
    actual = jax.jit(jax.vmap(jax.vmap(lambda p: action(p, native=True))))(parameters)
    expected = jax.vmap(jax.vmap(lambda p: action(p, native=False)))(parameters)
    assert jnp.allclose(actual, expected, rtol=1e-9, atol=1e-11)
    actual_gradient = jax.jit(
        jax.vmap(jax.vmap(jax.grad(lambda p: jnp.sum(action(p, native=True)))))
    )(parameters)
    expected_gradient = jax.vmap(
        jax.vmap(jax.grad(lambda p: jnp.sum(action(p, native=False))))
    )(parameters)
    assert jnp.allclose(actual_gradient, expected_gradient, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("active", [(0,), (1,), (0, 2), (0, 1, 2), (3,), (0, 3)])
@pytest.mark.parametrize("transposed", [False, True])
def test_partial_gradient_assembles_only_active_process(
    monkeypatch, active, transposed
):
    """Execute only requested process directions, including with zero source rate."""
    from drifft._evolution.autodiff import parameterized_power
    from drifft._evolution.parameterized import _Spec

    backend = drifft.Jacobi(3)
    method = drifft.PolynomialKrylov(dimension=8)
    initial = jnp.concatenate((jnp.linspace(0.1, 0.3, backend.rank), jnp.ones(1)))
    visits = []
    assemble = _Spec.assemble

    def counted(spec, parameters, geometry):
        # Count executed directional assemblies, including iterations inside a
        # compiled scan. This checks skipped work rather than a private mask.
        jax.debug.callback(lambda: visits.append(1))
        return assemble(spec, parameters, geometry)

    monkeypatch.setattr(_Spec, "assemble", counted)

    def objective(s, u, v, rate, *, native):
        process = drifft.Composition(drifft.AdditiveSelection(s), drifft.Mutation(u, v))
        source = rate * jnp.linspace(0.1, 0.2, backend.rank)
        op = _JacobiOperator(backend, process, source, True, transposed)
        matrix = op.as_matrix()
        value = matrix @ matrix @ initial
        if native:
            value = parameterized_power(
                method,
                op,
                initial,
                2,
                jax.lax.stop_gradient(value),
                jax.lax.stop_gradient(matrix),
            )
        return jnp.arange(initial.size) @ value

    inputs = tuple(map(jnp.asarray, (0.0, 0.01, 0.02, 0.0)))
    result = jax.jit(
        jax.grad(lambda *args: objective(*args, native=True), argnums=active)
    )(*inputs)
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert len(visits) == sum(i < 3 for i in active)
    expected = jax.grad(lambda *args: objective(*args, native=False), argnums=active)(
        *inputs
    )
    for actual, reference in zip(result, expected, strict=True):
        assert jnp.allclose(actual, reference, rtol=1e-9, atol=1e-11)


def test_partial_parameter_batches_and_zero_jvp():
    """Partial gradients compose with batching and numerical zero JVP directions."""
    from drifft._evolution.autodiff import parameterized_power

    backend = drifft.Jacobi(3)
    method = drifft.PolynomialKrylov(dimension=8)
    initial = jnp.linspace(0.1, 0.3, backend.rank)

    def objective(s, x, *, native):
        process = drifft.Composition(
            drifft.AdditiveSelection(s), drifft.Mutation(0.01, 0.02)
        )
        op = _JacobiOperator(backend, process, None, True)
        matrix = op.as_matrix()
        value = matrix @ matrix @ x
        if native:
            value = parameterized_power(
                method,
                op,
                x,
                2,
                jax.lax.stop_gradient(value),
                jax.lax.stop_gradient(matrix),
            )
        return jnp.arange(initial.size) @ value

    def actual(s, x):
        return objective(s, x, native=True)

    def reference(s, x):
        return objective(s, x, native=False)

    inputs = jnp.array([[0.0, 0.01], [0.02, 0.03]])
    for fn in (actual, reference):
        result = jax.jit(
            jax.vmap(jax.vmap(jax.grad(fn), in_axes=(0, None)), in_axes=(0, None))
        )(inputs, initial)
        if fn is actual:
            observed = result
        else:
            assert jnp.allclose(observed, result, rtol=1e-9, atol=1e-11)
    for direction in (0.0, 0.1):
        observed = jax.jit(
            lambda s, ds: jax.jvp(lambda q: actual(q, initial), (s,), (ds,))
        )(jnp.array(0.0), jnp.array(direction))
        expected = jax.jvp(
            lambda q: reference(q, initial), (jnp.array(0.0),), (jnp.array(direction),)
        )
        for a, b in zip(observed, expected, strict=True):
            assert jnp.allclose(a, b, rtol=1e-9, atol=1e-11)
    # Keeping the opaque primal live also preserves the derivative of a vector
    # pullback with respect to the process; stopping that payload would lose it.
    observed = jax.jit(jax.jacfwd(jax.grad(actual, argnums=1), argnums=0))(
        jnp.array(0.02), initial
    )
    expected = jax.jacfwd(jax.grad(reference, argnums=1), argnums=0)(
        jnp.array(0.02), initial
    )
    assert jnp.allclose(observed, expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize(
    "process,segregating,expected_leaves",
    [
        (drifft.Neutral(), True, 0),
        (drifft.Underdominance(0.02), True, 0),
        (drifft.Mutation(0.01, 0.02), False, 0),
        (drifft.Mutation(0.01, 0.02), True, 1),
    ],
)
def test_operator_keeps_only_assembly_geometry(process, segregating, expected_leaves):
    """Encoding tables must not become residuals of the reduced power itself."""
    backend = drifft.Jacobi(4)
    op = _JacobiOperator(backend, process, None, segregating)
    opaque = _pack(op, op.as_matrix())
    assert len(opaque.geometry) == expected_leaves
    if expected_leaves:
        assert jnp.array_equal(
            opaque.geometry[0], backend._geometry.endpoint_correction
        )


def test_parameter_batch_keeps_geometry_unmapped():
    """A batch of processes returns one shared endpoint correction array."""
    from drifft._evolution.parameterized import _unpack

    backend = drifft.Jacobi(4)

    def geometry(rate):
        op = _JacobiOperator(backend, drifft.Mutation(rate, 0.02), None, True)
        return _unpack(_pack(op, op.as_matrix())).geometry

    actual = jax.jit(jax.vmap(geometry, out_axes=None))(jnp.array([0.01, 0.02, 0.03]))
    assert len(actual) == 1
    assert jnp.array_equal(actual[0], backend._geometry.endpoint_correction)


@pytest.mark.parametrize("geometry_outer", [False, True])
def test_nested_batches_with_shared_and_mapped_geometry(geometry_outer):
    """Nested maps can independently batch processes and fixed geometry."""
    backend = drifft.Jacobi(3)
    method = drifft.PolynomialKrylov(dimension=8)
    initial = jnp.linspace(0.1, 0.3, backend.rank)
    corrections = jnp.stack(
        (
            backend._geometry.endpoint_correction,
            backend._geometry.endpoint_correction * 0.8,
        ),
        axis=1,
    )
    rates = jnp.array([0.01, 0.02, 0.03])

    def objective(rate, correction, x, *, native):
        geometry = eqx.tree_at(
            lambda b: b._geometry.endpoint_correction, backend, correction
        )
        op = _JacobiOperator(geometry, drifft.Mutation(rate, 0.02), None, True, True)
        matrix = op.as_matrix()
        value = (
            _power(_pack(op, jax.lax.stop_gradient(matrix)), x, method, 2)
            if native
            else matrix @ matrix @ x
        )
        return jnp.arange(x.size) @ value

    def evaluate(native):
        fn = jax.value_and_grad(
            lambda p, g, x: objective(p, g, x, native=native), argnums=(0, 2)
        )
        # Use a nonleading input axis for geometry, and exercise both nesting
        # orders: each outer map shifts different axes of the opaque payload.
        axes = ((0, None, None), (None, 1, None))
        inner, outer = axes if geometry_outer else axes[::-1]
        return jax.jit(jax.vmap(jax.vmap(fn, in_axes=inner), in_axes=outer))(
            rates, corrections, initial
        )

    observed, expected = evaluate(True), evaluate(False)
    for a, b in zip(jax.tree.leaves(observed), jax.tree.leaves(expected), strict=True):
        assert jnp.allclose(a, b, rtol=1e-9, atol=1e-11)


def test_retained_unbatched_vjp_with_minimal_geometry():
    """A separately compiled pullback remains callable after its forward pass."""
    backend = drifft.Jacobi(4)
    method = drifft.PolynomialKrylov(dimension=8)
    initial = jnp.linspace(0.1, 0.3, backend.rank)

    def action(rate, *, native):
        op = _JacobiOperator(backend, drifft.Mutation(rate, 0.02), None, True)
        matrix = op.as_matrix()
        return (
            _power(_pack(op, jax.lax.stop_gradient(matrix)), initial, method, 2)
            if native
            else matrix @ matrix @ initial
        )

    rate = jnp.array(0.01)
    value, pullback = jax.jit(lambda p: jax.vjp(lambda q: action(q, native=True), p))(
        rate
    )
    actual = jax.jit(pullback)(jnp.arange(value.size, dtype=value.dtype))[0]
    expected = jax.grad(lambda p: jnp.arange(value.size) @ action(p, native=False))(
        rate
    )
    assert jnp.allclose(actual, expected, rtol=1e-9, atol=1e-11)


def test_batched_scan_with_shared_geometry():
    """Forward scans compose with shared geometry and parameter batches."""
    backend = drifft.Jacobi(3)
    method = drifft.PolynomialKrylov(dimension=8)
    initial = jnp.linspace(0.1, 0.3, backend.rank)

    def objective(rates, *, native):
        def step(x, rate):
            op = _JacobiOperator(backend, drifft.Mutation(rate, 0.02), None, True)
            matrix = op.as_matrix()
            value = (
                _power(_pack(op, jax.lax.stop_gradient(matrix)), x, method, 2)
                if native
                else matrix @ matrix @ x
            )
            return value, None

        final, _ = jax.lax.scan(step, initial, rates)
        return jnp.arange(initial.size) @ final

    rates = jnp.array([[0.01, 0.02], [0.03, 0.04]])
    actual = jax.jit(jax.vmap(lambda p: objective(p, native=True)))(rates)
    expected = jax.vmap(lambda p: objective(p, native=False))(rates)
    assert jnp.allclose(actual, expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("transform", ["jvp", "vjp", "transpose"])
def test_pullback_rejects_unsupported_derivative_transforms(transform):
    """The parameter pullback reports its AD boundary without argument errors."""
    from drifft._evolution.autodiff import _Pullback

    backend = drifft.Jacobi(3)
    op = _JacobiOperator(backend, drifft.Mutation(0.01, 0.02), None, True)
    opaque = _pack(op, op.as_matrix())
    initial = jnp.linspace(0.1, 0.3, backend.rank)
    cotangent = jnp.ones_like(initial)
    args = opaque, initial, cotangent
    primitive = _Pullback(
        jax.tree.map(jax.typeof, args), drifft.PolynomialKrylov(dimension=8), 2
    )

    def action(ct):
        return primitive(opaque, initial, ct)

    with pytest.raises(NotImplementedError, match="Higher-order Krylov.*Sequential"):
        if transform == "jvp":
            jax.jvp(action, (cotangent,), (cotangent,))
        elif transform == "vjp":
            _, pullback = jax.vjp(action, cotangent)
            pullback(jnp.ones_like(opaque.parameters))
        else:
            # Standalone transposition bypasses the JVP rejection and must not
            # inherit the four-input sensitivity rule for this three-input op.
            jax.linear_transpose(action, cotangent)(jnp.ones_like(opaque.parameters))
