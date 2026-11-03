# Accuracy and differentiation

Validate the final statistic and its gradients across the parameter range you
will analyze. Backend approximation, repeated evolution, and linear solves
introduce separate errors.

## Check the quantity you will use

1. Compare the final statistic under tighter backend settings and, separately,
   tighter power or solve settings. Check mass and negative entries; outputs
   are not clipped or renormalized.
2. Compare with `Dense` on a small population or an analytic reference. A mean
   frequency alone is insufficient because low-degree polynomials can be
   reproduced accurately.
3. Compare gradients with the reference and finite differences at several step
   sizes. Forward accuracy does not establish gradient accuracy.

This example checks one-generation loss probability and its selection derivative.
It also compares with the direct binomial formula for this particular event:

```python
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)
M = 256
coarse = drifft.NUFFT(M, accuracy=1e-6)
fine = drifft.NUFFT(M, accuracy=1e-10, derivative_accuracy=True)
loss_indicator = jnp.zeros(M + 1).at[0].set(1.0)

# The allele starts with four copies.
def loss_probability(backend, s):
    transition = drifft.Transition(
        backend, drifft.AdditiveSelection(s=s)
    )
    return transition.expect(loss_indicator)[4]

def exact_loss(s):
    q = drifft.AdditiveSelection(s=s)(jnp.asarray(4 / M))
    return (1 - q) ** M

strength = jnp.asarray(0.02)
exact_value = exact_loss(strength)
exact_gradient = jax.grad(exact_loss)(strength)
for label, backend in [("coarse", coarse), ("fine", fine)]:
    value = loss_probability(backend, strength)
    gradient = jax.grad(lambda s: loss_probability(backend, s))(strength)
    print(f"{label}: value error={abs(value - exact_value):.3e}, "
          f"gradient error={abs(gradient - exact_gradient):.3e}")
```

This reference checks binomial reproduction with the same selection map.
For histories and expected visits, apply the checks to the final quantity,
not only a one-generation transition.

## Which settings control which errors?

| Component | How to tighten the approximation |
| --- | --- |
| FFT | Lower `tail_tol` for omitted binomial mass; increase `interp_order` for interpolation. |
| NUFFT | Lower `accuracy` for polynomial degree, `tail_tol` for sampled tails, and `nufft_tol` for transform evaluation. |
| Jacobi | Increase `rank`; lower `one_step_tolerance` for automatic rank and padding. |
| Krylov | Increase `dimension`; lower `rtol` and `atol`. |
| Expected visits | Lower solver tolerances and increase iteration limits. Check sensitivity to backend settings separately. |

For reduced Jacobi evolution, prepare `max_strength` to cover the absolute
values of both diploid selection coefficients over your parameter range.
See the [Jacobi API](../api/backends.md#drifft.Jacobi) for assembly restrictions.

With `throw=False`, check `converged` before using a result. Krylov's
`error_estimate` assesses the power approximation; expected-visits results
report a solve residual. Neither includes backend error. The
[Krylov chapter](../methods/jacobi_krylov.md#accuracy-and-controls) defines its
acceptance criterion.

NUFFT's `derivative_accuracy=True` increases polynomial degree by 50%, capped
at exact degree; it is a heuristic, not a derivative-error bound. See the
[backend API](../api/backends.md) for defaults and encoding options.
If a Krylov shifted solve fails, increasing the outer `dimension` does not fix
it; see [Krylov controls](../methods/jacobi_krylov.md#accuracy-and-controls).

## Differentiating biological parameters

Selection and mutation parameters, initial mass, and immigration can be
array-valued inputs to a differentiated function. Population sizes, durations,
and numerical dimensions are fixed configuration. Construct backends outside
the function, then create or update the frequency map inside it; the
[tour](../tour.md#differentiate-selection)
shows this pattern with `jax.grad`.

Krylov operator sensitivities support first derivatives; use Sequential for
higher-order calculations. Sensitivity failures raise even with `throw=False`.
See the [autodiff accuracy contract](../methods/autodiff.md#accuracy-and-limitations)
for the derivative targets and limitations.

For a complete finite-time calculation with a derivative plot and
finite-difference checks, see [sensitivity to selection](../examples/selection_sensitivity.md).

## Memory and timing

Differentiation can retain more intermediates than forward evaluation,
especially across long intervals. NUFFT's `batch_size` controls temporary
binomial-sampling storage; native FINUFFT workspace is additional to JAX memory
reports. Krylov basis storage grows with working-state size and `dimension`.

For timing, separate preparation, first-call compilation, and warm execution,
and synchronize JAX outputs. Use [validation and benchmarks](../benchmarks.md)
for the repository's comparison commands and recorded results.
