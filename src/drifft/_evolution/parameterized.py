r"""Adapt Jacobi epoch operators to parameter-space differentiation.

For a Jacobi working-coordinate operator $A(\theta)$, the process and encoded
source parameters $\theta$ determine a matrix used repeatedly by numerical
power evaluation. Treating that matrix as the differentiation coordinate can
produce a dense matrix cotangent and retain an assembly reverse tape. Instead,
this module carries the matrix as a numerical cache while representing operator
tangents and cotangents in the coordinates of $\theta$.

An ordinary `_JacobiOperator` implements the Lineax interface for prepared
problems and custom methods. At the power boundary, an opaque HiJAX type carries
the parameters, required fixed geometry and cached matrix. Given adjoint factors
$L,R$, process-coordinate cotangents are evaluated as

$$
\bar\theta_i=\langle L,(\partial A/\partial\theta_i)R\rangle_F.
$$

Only active process directions are assembled, one at a time. The affine source
column permits a direct factor contraction instead of one assembly per source
coordinate. Parameter-space differentiation and state-space reduction are
separate ideas: the former controls derivative storage; this implementation
specifically adapts the latter for Jacobi epochs.

Packing, unpacking, transposition and batching preserve that representation.
Transposition changes the matrix orientation, not the parameter coordinates;
geometry remains shared across batches when it is not mapped. Visible parameter
leaves preserve symbolic activity before flattening, without counting their
tangent again through the opaque value.

Jacobi geometry and assembly formulas live in `_backend/jacobi.py`. This module
owns their evolution adapter and parameter derivatives, while `autodiff.py` owns
power rules and `sensitivity.py` owns numerical sensitivity evaluation. Encoding,
reconstruction and history source placement remain part of the enclosing chain
rule, rather than being absorbed into the cached matrix's derivative.
"""

# pattern: Functional Core
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax.experimental import hijax
from jax.flatten_util import ravel_pytree

from .._backend.jacobi import Jacobi, _assemble_reduced
from .._frequency import AbstractFrequencyMap
from .operators import _numerical_operator


class _JacobiOperator(lx.AbstractLinearOperator):
    r"""Expose the Jacobi working-coordinate action as a differentiable Lineax operator.

    The backend assembles a reduced expectation matrix; its transpose advances
    working mass. An encoded source adds a homogeneous column and scalar row,
    turning repeated affine updates into a linear power. The `transposed` flag
    reverses the complete augmented action while retaining its parameters.

    This ordinary Equinox PyTree is used in prepared problems, so custom methods
    and direct operator differentiation see parameter-dependent assembly. Built-in
    Krylov power bindings separately pack it into `_Operator` to use compact
    cotangents and a stopped numerical matrix. It represents an epoch's working
    action, not the full population-space encoding and reconstruction.
    """

    backend: Jacobi
    frequency_map: AbstractFrequencyMap
    source: jax.Array | None
    segregating: bool = eqx.field(static=True)
    transposed: bool = eqx.field(static=True, default=False)

    def as_matrix(self):
        """Materialize the differentiable mathematical operator for custom solvers."""
        return _forward_matrix(
            self.backend._reduced_expectation(
                self.frequency_map, segregating=self.segregating
            ),
            self.source,
            self.transposed,
        )

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


@_numerical_operator.register(_JacobiOperator)
def _jacobi_numerical_operator(operator):
    """Assemble once for numerical projection, preserving ordinary preparation AD."""
    return lx.MatrixLinearOperator(operator.as_matrix())


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


def _forward_matrix(expectation, source, transposed):
    """Orient the reduced expectation and include a constant source column."""
    matrix = expectation.T
    if source is not None:
        n = matrix.shape[0]
        matrix = jnp.pad(matrix, ((0, 1), (0, 1)))
        matrix = matrix.at[:n, n].set(source).at[n, n].set(1)
    return matrix.T if transposed else matrix


@dataclass(frozen=True)
class _Spec:
    r"""Describe how to reconstruct $A(\theta)$ from parameter coordinates.

    The filtered process/source structure and unravel function recover parameter
    leaves from a flat vector. Static dimensions select the assembly shape; any
    required runtime geometry is supplied separately, never hidden in this
    hashable reconstruction recipe. Orientation and augmentation determine the
    final action represented by the cached matrix.

    `process_size` separates process coordinates, which require directional
    assembly, from source coordinates, whose adjoint contraction is explicit.
    Keeping this recipe separate from numerical payloads lets derivative kernels
    reconstruct a direction without retaining the backend's encoding tables.
    """

    static: tuple
    unravel: Callable
    copies: int
    rank: int
    padded_rank: int
    diploid_padded_rank: int
    segregating: bool
    process_size: int
    augmented: bool
    transposed: bool

    def assemble(self, parameters, geometry):
        frequency_map, source = eqx.combine(self.unravel(parameters), self.static)
        expectation = _assemble_reduced(
            frequency_map,
            self.copies,
            self.rank,
            self.padded_rank,
            self.diploid_padded_rank,
            geometry[0] if geometry else None,
            segregating=self.segregating,
        )
        return _forward_matrix(expectation, source, self.transposed)


@dataclass(frozen=True)
class _Operator:
    r"""Carry a parameterized primal operator with numerical payloads for evaluation.

    `parameters` identifies the logical derivative coordinates; `matrix` caches
    $A(\theta)$ for numerical actions, and `geometry` contains only the fixed
    arrays needed to reconstruct it. `spec` describes that reconstruction.
    The caller must supply a cache consistent with the parameters and orientation.

    The registered `_OperatorType` gives this value a parameter-vector tangent
    and cotangent rather than a tangent for each stored array. Thus the matrix
    can be retained for efficient solves without becoming a dense cotangent.
    Payload access during tracing goes through the representation primitives.
    """

    parameters: jax.Array
    geometry: tuple
    matrix: jax.Array
    spec: _Spec


@dataclass(frozen=True)
class _AxisShift:
    r"""Record component-wise offsets for nested mappings of an operator payload.

    Parameters and the matrix share one batch-axis position; geometry can have
    its own positions or remain unmapped. Removing or inserting an enclosing
    mapped axis therefore need not shift every payload component equally.

    `_Axis` comparisons produce these offsets for HiJAX's nested-map adjustment
    expressions. They are internal mapping arithmetic, not a scalar ordering of
    operators. Carrying geometry offsets separately preserves sharing rather
    than introducing spurious batch dimensions.
    """

    axis: int
    geometry: tuple[int, ...] | None


@dataclass(frozen=True)
class _Axis(hijax.MappingSpec):
    r"""Map operator payloads without requiring geometry to follow parameter batches.

    `axis` maps parameters and the matrix together. An explicit `geometry` tuple
    specifies each geometry leaf's mapped axis, with a None entry denoting a
    shared leaf. A None tuple instead means uniform mapping on `axis`, as needed
    when an entire value is stacked, for example by a scan.

    Nested maps insert and remove dimensions independently in these payloads.
    Component-wise comparisons and shifts preserve their positions through
    HiJAX's mapping adjustments. This is a representation contract: batching
    changes storage axes, not which parameters the mathematical operator depends
    on, and shared geometry must not be broadcast merely to match the matrix.
    """

    axis: int = 0
    geometry: tuple[int | None, ...] | None = None

    def geometry_axes(self, count):
        # None is the uniform leading-axis contract used when scan stacks an
        # entire value. An explicit tuple can leave individual leaves shared.
        return (self.axis,) * count if self.geometry is None else self.geometry

    # VmapOf adjusts mappings with d - (other < d) and d + (other < d).
    # Geometry can have different mapped axes from the parameter payload, so
    # those comparisons and shifts must be component-wise as well.
    def __lt__(self, other):
        if self.geometry is None and other.geometry is None:
            geometry = None
        else:
            count = len(self.geometry if self.geometry is not None else other.geometry)
            geometry = tuple(
                int(a < b) if a is not None and b is not None else 0
                for a, b in zip(
                    self.geometry_axes(count), other.geometry_axes(count), strict=True
                )
            )
        return _AxisShift(int(self.axis < other.axis), geometry)

    def _shift(self, offset, sign):
        if not isinstance(offset, _AxisShift):
            offset = _AxisShift(int(offset), None)
        if self.geometry is None and offset.geometry is None:
            geometry = None
        else:
            reference = self.geometry if self.geometry is not None else offset.geometry
            assert reference is not None
            count = len(reference)
            shifts = (
                (offset.axis,) * count if offset.geometry is None else offset.geometry
            )
            geometry = tuple(
                None if axis is None else axis + sign * shift
                for axis, shift in zip(self.geometry_axes(count), shifts, strict=True)
            )
        return _Axis(self.axis + sign * offset.axis, geometry)

    def __sub__(self, offset):
        return self._shift(offset, -1)

    def __add__(self, offset):
        return self._shift(offset, 1)


@dataclass(frozen=True)
class _OperatorType(hijax.HiType):
    r"""Define an operator whose primal storage and derivative coordinates differ.

    Lowering exposes parameters, required geometry and the numerical matrix;
    raising reconstructs the same `_Operator` value. The tangent type is only
    the parameter-vector type, so cotangent accumulation takes place in those
    coordinates instead of allocating derivatives for fixed geometry or caches.

    Rank changes follow `_Axis`: parameters and matrix are mapped together,
    while geometry may remain shared. Static reconstruction metadata stays in
    the type rather than becoming runtime numerical storage. This HiType exists
    because a conventional PyTree tangent would mirror all primal array leaves
    and would not express the intended compact derivative contract.
    """

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
            tuple(
                a if axis is None else a.inc_rank(size, axis)
                for a, axis in zip(
                    self.geometry, spec.geometry_axes(len(self.geometry)), strict=True
                )
            ),
            self.matrix.inc_rank(size, spec.axis),
            self.spec,
        )

    def dec_rank(self, size, spec):
        if spec is None:
            return self
        assert isinstance(spec, _Axis)
        return _OperatorType(
            self.parameters.dec_rank(size, spec.axis),
            tuple(
                a if axis is None else a.dec_rank(size, axis)
                for a, axis in zip(
                    self.geometry, spec.geometry_axes(len(self.geometry)), strict=True
                )
            ),
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
    r"""Introduce the parameterized operator's logical AD boundary.

    The primal bundles $(\theta,\mathrm{geometry},A_{\mathrm{cache}})$ into an
    opaque operator. Its JVP returns only $\dot\theta$: the cached matrix is
    numerical state, not an independent mathematical argument, and active
    geometry tangents are rejected. Numerical callers provide a matrix consistent
    with the represented operator; this primitive does not reassemble or check it.

    Binding this operation as a primitive prevents later AD from assigning
    matrix-shaped cotangents to the payload. Its batch rule maps parameters and
    matrix while leaving unmapped geometry shared, preserving the same logical
    operator coordinates across batches.
    """

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

        parameters, geometry, matrix = args
        parameter_dim, geometry_dims, matrix_dim = dims
        geometry = tuple(
            leaf if dim is None else jnp.moveaxis(leaf, dim, 0)
            for leaf, dim in zip(geometry, geometry_dims, strict=True)
        )
        args = leading(parameters, parameter_dim), geometry, leading(matrix, matrix_dim)
        mapping = _Axis(0, tuple(None if dim is None else 0 for dim in geometry_dims))
        return _Pack(jax.tree.map(jax.typeof, args), self.params["spec"])(
            *args
        ), mapping


def _pack(operator, matrix):
    dynamic, static = eqx.partition(
        (operator.frequency_map, operator.source), eqx.is_array
    )
    parameters, unravel = ravel_pytree(dynamic)
    backend = operator.backend
    correction = backend._reduced_correction(
        operator.frequency_map, segregating=operator.segregating
    )
    geometry = () if correction is None else (correction,)
    process_size = sum(x.size for x in jax.tree.leaves(dynamic[0]))
    spec = _Spec(
        static,
        unravel,
        backend._gene_copies,
        backend.rank,
        backend.padded_rank,
        backend._diploid_padded_rank,
        operator.segregating,
        process_size,
        operator.source is not None,
        operator.transposed,
    )
    args = parameters, geometry, matrix
    return _Pack(jax.tree.map(jax.typeof, args), spec)(*args)


class _Unpack(hijax.HiPrim):
    r"""Expose an opaque operator's payload inside numerical primitive expansions.

    The output is the stored parameter vector, geometry tuple and matrix cache.
    This is a representation access operation, not the mathematical map
    $\theta\mapsto A(\theta)$: it does not define a JVP that differentiates cache
    contents. Power and sensitivity primitives own the derivative semantics of
    the computations that use these arrays.

    Its batching rule preserves each component's mapping, including geometry
    that remains shared. Keeping access behind a primitive allows traced opaque
    values to be used by numerical kernels without treating their payload as
    independent operator derivative coordinates.
    """

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
        if dim is None:
            return result, jax.tree.map(lambda _: None, self.out_aval)
        return result, (dim.axis, dim.geometry_axes(len(self.out_aval[1])), dim.axis)


def _unpack(operator):
    parameters, geometry, matrix = _Unpack(jax.typeof(operator))(operator)
    return _Operator(parameters, geometry, matrix, jax.typeof(operator).spec)


class _Transpose(hijax.HiPrim):
    r"""Reverse operator orientation while preserving parameter coordinates.

    The primal transposes the cached matrix and flips the reconstruction recipe's
    orientation. A direction in $\theta$ is unchanged: it now describes the
    variation of $A(\theta)^T$ through the transposed recipe, rather than being a
    matrix tangent that must itself be transposed.

    This identity map on parameter tangents supports opposite-orientation state
    pullbacks without losing process dependence in mixed derivatives. Geometry
    and parameter payloads are retained, and the mapping specification is passed
    through so transposition does not replicate shared geometry.
    """

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


def _parameter_pushforward(operator, direction):
    """Differentiate assembly in parameter coordinates with fixed geometry."""
    _, perturbation = jax.jvp(
        lambda p: operator.spec.assemble(p, operator.geometry),
        (operator.parameters,),
        (direction,),
    )
    return lx.MatrixLinearOperator(perturbation)


def _contract_parameter_factors(operator, left, right, active_process):
    """Contract fixed adjoint factors without an assembly tape or matrix cotangent."""
    parameters = operator.parameters
    n = operator.spec.process_size

    # Stream the few process directions. Source coordinates admit an exact
    # factor contraction, so their number does not multiply assembly work.
    def contract(index):
        direction = jnp.zeros_like(parameters).at[index].set(1)
        derivative = _parameter_pushforward(operator, direction).as_matrix()
        return jnp.sum(left * (derivative @ right))

    indices = tuple(range(n)) if active_process is None else active_process
    if indices:
        selected = jax.lax.map(contract, jnp.asarray(indices))
        if indices == tuple(range(n)):
            process_ct = selected
        else:
            process_ct = (
                jnp.zeros(n, parameters.dtype).at[jnp.asarray(indices)].set(selected)
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


def _parameter_direction(leaves, directions, process_size, tangent_aval):
    """Preserve activity before flattening; visible leaves own the tangent once."""
    active = []
    offset = 0
    n = process_size
    for leaf, direction in zip(leaves, directions, strict=True):
        if not isinstance(direction, hijax.Zero):
            active.extend(range(offset, min(offset + leaf.size, n)))
        offset += leaf.size
    if all(isinstance(t, hijax.Zero) for t in directions):
        dp = hijax.Zero(tangent_aval)
    else:
        dp, _ = ravel_pytree(tuple(map(hijax.instantiate_zeros, directions)))
    return dp, tuple(active)
