# Jacobi reduction and Krylov powers

Read [choosing numerical algorithms](../guide/backends.md) for supported
combinations and [accuracy and differentiation](../guide/numerics.md) for
validation. Here we explain the reduced representation and its power action.

Let $P$ be the one-generation expectation matrix on allele counts. The Jacobi
approximation factors it as $P\approx DE$, where $E$ encodes an observable in
polynomial coefficients and $D$ evaluates those coefficients after selection.
The full-state representation includes two endpoint coordinates. The
segregating representation retains only the interior polynomial coefficients.

With $H=ED$, forward evolution through $g\geq1$ generations is

$$p_g\approx E^T(H^T)^{g-1}D^T p_0.$$

For constant immigration $b$, initialize $y_0=D^Tp_0$ and apply

$$
y_{t+1}=H^Ty_t+D^Tb,\qquad
p_g\approx E^Ty_{g-1}+b.
$$

The implementation turns this affine recurrence into a homogeneous augmented
matrix. For its matrix $A$ and exponent $k=g-1$, rational Arnoldi builds an
orthonormal basis from $(I+\tau(I-A))^{-1}$ with $\tau=k/8$ and two rounds of
reorthogonalization. It projects the **original** matrix, $B=V^TAV$, then
evaluates $VB^kV^Ty_0$ by binary powering the small matrix.

Jacobi assembles a dense reduced matrix without constructing the full
population transition. Storage grows quadratically with rank, and the internal
LU factorization grows cubically. RationalKrylov reuses that factorization
throughout basis construction. Increasing rank therefore has a different cost
from increasing Krylov dimension.

For the one-step projection and assembly bounds, see the
[Jacobi analysis note](https://github.com/mancusolab/drifft/blob/main/notes/wf_jacobi_bounds.md).
