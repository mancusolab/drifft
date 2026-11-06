"""Private Jacobi powers with parameter-space tangents and cotangents.

The ordinary Lineax adapter keeps prepared problems compatible with Equinox.
Only the mathematical power boundary uses an opaque HiType: its dense matrix
is numerical state, whereas its tangent is the small process/source vector.
"""

# pattern: Functional Core
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax.experimental import hijax
from jax.flatten_util import ravel_pytree

from .._backend.jacobi import Jacobi
from .._frequency import AbstractFrequencyMap
from .problem import _KrylovConfig
from .sensitivity import _frechet_adjoint_factors, _frechet_power


class _JacobiOperator(lx.AbstractLinearOperator):
    """Lazily assemble a reduced operator, preserving public preparation AD."""

    backend: Jacobi
    frequency_map: AbstractFrequencyMap
    source: jax.Array | None
    segregating: bool = eqx.field(static=True)
    transposed: bool = eqx.field(static=True, default=False)

    def as_matrix(self):
        """Materialize the differentiable mathematical operator for custom solvers."""
        matrix = self.backend._reduced_expectation(
            self.frequency_map, segregating=self.segregating
        ).T
        if self.source is not None:
            n = matrix.shape[0]
            matrix = jnp.pad(matrix, ((0, 1), (0, 1)))
            matrix = matrix.at[:n, n].set(self.source).at[n, n].set(1)
        return matrix.T if self.transposed else matrix

    def mv(self, vector):
        """Apply the represented operator with ordinary JAX differentiation."""
        return self.as_matrix() @ vector

    def transpose(self):
        """Reverse orientation while retaining the same parameterization."""
        return _JacobiOperator(
            self.backend,
            self.frequency_map,
            self.source,
            self.segregating,
            not self.transposed,
        )

    def in_structure(self):
        size = self.backend.rank + (0 if self.segregating else 2)
        size += self.source is not None
        return jax.ShapeDtypeStruct((size,), jnp.float64)

    out_structure = in_structure


@lx.is_diagonal.register(_JacobiOperator)
@lx.is_tridiagonal.register(_JacobiOperator)
@lx.is_lower_triangular.register(_JacobiOperator)
@lx.is_upper_triangular.register(_JacobiOperator)
@lx.is_symmetric.register(_JacobiOperator)
@lx.is_positive_semidefinite.register(_JacobiOperator)
@lx.is_negative_semidefinite.register(_JacobiOperator)
def _no_tags(operator):
    """Make no unchecked structural claims for a general frequency map."""
    return False


@lx.linearise.register(_JacobiOperator)
@lx.materialise.register(_JacobiOperator)
def _materialise(operator):
    return lx.MatrixLinearOperator(operator.as_matrix())


@lx.conj.register(_JacobiOperator)
def _conj(operator):
    return operator


@dataclass(frozen=True)
class _Spec:
    """Static reconstruction recipe; runtime geometry stays in the payload."""

    static: tuple
    unravel: Callable
    geometry_tree: jax.tree_util.PyTreeDef
    process_size: int
    augmented: bool
    transposed: bool

    def assemble(self, parameters, geometry):
        dynamic = self.geometry_tree.unflatten(geometry)
        backend, frequency_map, source, segregating = eqx.combine(
            (dynamic, *self.unravel(parameters)), self.static
        )
        return _JacobiOperator(
            backend, frequency_map, source, segregating, self.transposed
        ).as_matrix()


@dataclass(frozen=True)
class _Operator:
    """Numerical cache whose logical coordinates are process/source parameters."""

    parameters: jax.Array
    geometry: tuple
    matrix: jax.Array
    spec: _Spec


@dataclass(frozen=True)
class _Axis(hijax.MappingSpec):
    """Map the same payload axis, including HiJAX's nested-vmap axis shifts."""

    axis: int = 0

    # VmapOf adjusts mapping positions with comparisons and integer offsets.
    # Preserve the mapping-spec type when it moves an existing mapped axis.
    def __lt__(self, other):
        return self.axis < other.axis

    def __sub__(self, offset):
        return _Axis(self.axis - offset)

    def __add__(self, offset):
        return _Axis(self.axis + offset)


@dataclass(frozen=True)
class _OperatorType(hijax.HiType):
    """Lower the cache but differentiate only its parameter coordinates."""

    parameters: hijax.ShapedArray
    geometry: tuple
    matrix: hijax.ShapedArray
    spec: _Spec

    def lo_ty(self):
        return [self.parameters, *self.geometry, self.matrix]

    def lower_val(self, value):
        return [value.parameters, *value.geometry, value.matrix]

    def raise_val(self, *lo_vals):
        parameters, *rest = lo_vals
        return _Operator(parameters, tuple(rest[:-1]), rest[-1], self.spec)

    def to_tangent_aval(self):
        return self.parameters.to_tangent_aval()

    def inc_rank(self, size, spec):
        if spec is None:
            return self
        assert isinstance(spec, _Axis)
        return _OperatorType(
            self.parameters.inc_rank(size, spec.axis),
            tuple(a.inc_rank(size, spec.axis) for a in self.geometry),
            self.matrix.inc_rank(size, spec.axis),
            self.spec,
        )

    def dec_rank(self, size, spec):
        if spec is None:
            return self
        assert isinstance(spec, _Axis)
        return _OperatorType(
            self.parameters.dec_rank(size, spec.axis),
            tuple(a.dec_rank(size, spec.axis) for a in self.geometry),
            self.matrix.dec_rank(size, spec.axis),
            self.spec,
        )

    def leading_axis_spec(self):
        return _Axis()


hijax.register_hitype(
    _Operator,
    lambda op: _OperatorType(
        jax.typeof(op.parameters),
        tuple(map(jax.typeof, op.geometry)),
        jax.typeof(op.matrix),
        op.spec,
    ),
)


class _Pack(hijax.HiPrim):
    """Bind parameters to an AD-stopped cache assembled by the numerical solve."""

    def __init__(self, avals, spec):
        self.in_avals = avals
        self.out_aval = _OperatorType(avals[0], avals[1], avals[2], spec)
        self.params = dict(spec=spec)
        super().__init__()

    def expand(self, *args):
        parameters, geometry, matrix = args
        return _Operator(parameters, geometry, matrix, cast(_Spec, self.params["spec"]))

    def jvp(self, primals, tangents):
        # Geometry is fixed; matrix is a numerical cache, not an independent
        # mathematical input. Assembly is differentiated only on demand below.
        if any(not isinstance(t, hijax.Zero) for t in jax.tree.leaves(tangents[1])):
            raise TypeError("Jacobi power geometry is nondifferentiable")
        return self(*primals), tangents[0]

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def batch(self, axis_data, args, dims):
        if all(d is None for d in jax.tree.leaves(dims)):
            return self(*args), None

        def leading(x, axis):
            return (
                jnp.broadcast_to(x, (axis_data.size, *x.shape))
                if axis is None
                else jnp.moveaxis(x, axis, 0)
            )

        args = jax.tree.map(leading, args, dims)
        return _Pack(jax.tree.map(jax.typeof, args), self.params["spec"])(
            *args
        ), _Axis()


def _pack(operator, matrix):
    dynamic, static = eqx.partition(
        (
            operator.backend,
            operator.frequency_map,
            operator.source,
            operator.segregating,
        ),
        eqx.is_array,
    )
    parameters, unravel = ravel_pytree(dynamic[1:])
    geometry, geometry_tree = jax.tree.flatten(dynamic[0])
    process_size = sum(x.size for x in jax.tree.leaves(dynamic[1]))
    spec = _Spec(
        static,
        unravel,
        geometry_tree,
        process_size,
        operator.source is not None,
        operator.transposed,
    )
    args = parameters, tuple(geometry), matrix
    return _Pack(jax.tree.map(jax.typeof, args), spec)(*args)


class _Unpack(hijax.HiPrim):
    """Expose numerical payloads only inside mathematical primitive expansions."""

    def __init__(self, aval):
        self.in_avals = (aval,)
        self.out_aval = (aval.parameters, aval.geometry, aval.matrix)
        self.params = {}
        super().__init__()

    def expand(self, *args):
        (operator,) = args
        return operator.parameters, operator.geometry, operator.matrix

    def batch(self, axis_data, args, dims):
        (operator,), (dim,) = args, dims
        result = _Unpack(jax.typeof(operator))(operator)
        return result, jax.tree.map(
            lambda _: None if dim is None else dim.axis, self.out_aval
        )


def _unpack(operator):
    parameters, geometry, matrix = _Unpack(jax.typeof(operator))(operator)
    return _Operator(parameters, geometry, matrix, jax.typeof(operator).spec)


class _Power(hijax.HiPrim):
    """Apply a mathematical power or attach its rule to a cached primal."""

    def __init__(self, avals, method, steps, cached, *, with_parameter_leaves=False):
        self.in_avals = avals
        self.out_aval = avals[1]
        self.params = dict(
            method=method,
            steps=steps,
            cached=cached,
            with_parameter_leaves=with_parameter_leaves,
        )
        super().__init__()

    def expand(self, *args):
        operator, initial, *value = args
        if self.params["with_parameter_leaves"]:
            _, *value = value
        operator = _unpack(operator)
        if self.params["cached"]:
            return value[0]
        return cast(Any, self.params["method"])._power_action(
            lx.MatrixLinearOperator(operator.matrix), initial, self.params["steps"]
        )

    def jvp(self, primals, tangents):
        operator, initial, *_ = primals
        dp, dx, *_ = tangents
        active_process = None
        if self.params["with_parameter_leaves"]:
            # These leaves alias the opaque payload, but retain structural zeros
            # until this boundary. They own the parameter tangent here; using
            # the opaque tangent as well would count that contribution twice.
            leaves, directions = primals[2], tangents[2]
            active = []
            offset = 0
            n = jax.typeof(operator).spec.process_size
            for leaf, direction in zip(leaves, directions, strict=True):
                if not isinstance(direction, hijax.Zero):
                    active.extend(range(offset, min(offset + leaf.size, n)))
                offset += leaf.size
            active_process = tuple(active)
            if all(isinstance(t, hijax.Zero) for t in directions):
                dp = hijax.Zero(jax.typeof(operator).to_tangent_aval())
            else:
                dp, _ = ravel_pytree(tuple(map(hijax.instantiate_zeros, directions)))
        if isinstance(dp, hijax.Zero):
            tangent = _power(operator, hijax.instantiate_zeros(dx), **self._settings())
        else:
            args = operator, initial, dp, hijax.instantiate_zeros(dx)
            tangent = _Sensitivity(
                jax.tree.map(jax.typeof, args),
                active_process=active_process,
                **self._settings(),
            )(*args)
        return self(*primals), tangent

    def _settings(self):
        return {key: self.params[key] for key in ("method", "steps")}

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def transpose(self, out_ct, *args):
        ct = out_ct
        operator, initial, *value = args
        if isinstance(operator, hijax.GradAccum) or value:
            raise ValueError("power is linear only in the initial state")
        if isinstance(initial, hijax.GradAccum):
            initial.accum(
                _power(
                    _transpose(operator),
                    hijax.instantiate_zeros(ct),
                    **self._settings(),
                )
            )

    def batch_dim_rule(self, axis_data, dims):
        return 0 if any(d is not None for d in jax.tree.leaves(dims)) else None


class _Sensitivity(_Power):
    """Linearized power with direct parameter-space adjoint contraction."""

    def __init__(self, avals, method, steps, *, active_process=None):
        self.in_avals = avals
        self.out_aval = avals[1]
        self.params = dict(method=method, steps=steps, active_process=active_process)
        hijax.HiPrim.__init__(self)

    def expand(self, *args):
        operator, initial, dp, dx = args
        operator = _unpack(operator)
        _, perturbation = jax.jvp(
            lambda p: operator.spec.assemble(p, operator.geometry),
            (operator.parameters,),
            (dp,),
        )
        result = _frechet_power(
            cast(_KrylovConfig, self.params["method"]),
            lx.MatrixLinearOperator(operator.matrix),
            lx.MatrixLinearOperator(perturbation),
            initial,
            dx,
            self.params["steps"],
        )
        return eqx.error_if(
            result.distribution,
            ~result.converged,
            "Krylov sensitivity action did not converge; increase dimension or use Sequential",
        )

    def jvp(self, primals, tangents):
        raise NotImplementedError(
            "Higher-order Krylov sensitivities are not implemented; use Sequential"
        )

    def transpose(self, out_ct, *args):
        ct = out_ct
        operator, initial, dp, dx = args
        if isinstance(operator, hijax.GradAccum) or isinstance(
            initial, hijax.GradAccum
        ):
            raise ValueError("sensitivity is linear only in tangent arguments")
        cotangent = hijax.instantiate_zeros(ct)
        if isinstance(dp, hijax.GradAccum):
            args = operator, initial, cotangent
            dp.accum(
                _Pullback(
                    jax.tree.map(jax.typeof, args),
                    active_process=self.params["active_process"],
                    **self._settings(),
                )(*args)
            )
        if isinstance(dx, hijax.GradAccum):
            dx.accum(_power(_transpose(operator), cotangent, **self._settings()))


class _Transpose(hijax.HiPrim):
    """Transpose numerical state without changing parameter coordinates."""

    def __init__(self, aval):
        self.in_avals = (aval,)
        self.out_aval = replace(
            aval, spec=replace(aval.spec, transposed=not aval.spec.transposed)
        )
        self.params = {}
        super().__init__()

    def expand(self, *args):
        (operator,) = args
        op = _unpack(operator)
        args = op.parameters, op.geometry, op.matrix.T
        return _Pack(jax.tree.map(jax.typeof, args), self.out_aval.spec)(*args)

    def jvp(self, primals, tangents):
        return self(*primals), tangents[0]

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def batch_dim_rule(self, axis_data, dims):
        return dims[0]


def _transpose(operator):
    return _Transpose(jax.typeof(operator))(operator)


class _Pullback(_Sensitivity):
    """Contract power-adjoint factors without constructing a matrix cotangent."""

    def __init__(self, avals, method, steps, *, active_process=None):
        self.in_avals = avals
        self.out_aval = avals[0].to_ct_aval()
        self.params = dict(method=method, steps=steps, active_process=active_process)
        hijax.HiPrim.__init__(self)

    def expand(self, *args):
        operator, initial, cotangent = args
        operator = _unpack(operator)
        parameters = operator.parameters
        left, right, _, converged = _frechet_adjoint_factors(
            cast(_KrylovConfig, self.params["method"]),
            lx.MatrixLinearOperator(operator.matrix),
            initial,
            cotangent,
            self.params["steps"],
        )
        left = eqx.error_if(
            left,
            ~converged,
            "Krylov operator pullback did not converge; increase dimension or use Sequential",
        )
        n = operator.spec.process_size

        # Stream the few process directions. Source coordinates admit an exact
        # factor contraction, so their number does not multiply assembly work.
        def contract(index):
            direction = jnp.zeros_like(parameters).at[index].set(1)
            _, derivative = jax.jvp(
                lambda p: operator.spec.assemble(p, operator.geometry),
                (parameters,),
                (direction,),
            )
            return jnp.sum(left * (derivative @ right))

        active = self.params["active_process"]
        indices = tuple(range(n)) if active is None else cast(tuple[int, ...], active)
        if indices:
            selected = jax.lax.map(contract, jnp.asarray(indices))
            if indices == tuple(range(n)):
                process_ct = selected
            else:
                process_ct = (
                    jnp.zeros(n, parameters.dtype)
                    .at[jnp.asarray(indices)]
                    .set(selected)
                )
        else:
            process_ct = jnp.zeros(n, parameters.dtype)
        if operator.spec.augmented:
            left_source, right_source = (
                (right, left) if operator.spec.transposed else (left, right)
            )
            source_ct = left_source[:-1] @ right_source[-1]
            process_ct = jnp.concatenate((process_ct, source_ct))
        return process_ct


def _power(operator, initial, method, steps, *value):
    args = operator, initial, *value
    return _Power(jax.tree.map(jax.typeof, args), method, steps, bool(value))(*args)


def parameterized_power(method, operator, initial, steps, value, matrix):
    """Attach a parameter-space mathematical rule to a stopped numerical cache."""
    leaves = tuple(
        jax.tree.leaves(
            eqx.filter((operator.frequency_map, operator.source), eqx.is_array)
        )
    )
    args = _pack(operator, matrix), initial, leaves, value
    return _Power(
        jax.tree.map(jax.typeof, args), method, steps, True, with_parameter_leaves=True
    )(*args)
