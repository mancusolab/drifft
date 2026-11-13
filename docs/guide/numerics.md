# Accuracy and differentiation

Validate the final statistic and its gradients across the parameter range you
will analyze. Backend approximation, repeated evolution, and linear solves
introduce separate errors.

## Check the quantity you will use

1. Compare the final statistic under tighter backend settings and, separately,
   tighter power or solve settings. Check mass and negative entries; outputs
   are not clipped or renormalized.
2. Compare with [`Dense`](../api/backends.md#drifft.Dense) on a small population or an analytic reference. A mean
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
pop_size = 128
M = 2 * pop_size  # Number of copies in the exact binomial reference.
coarse = drifft.NUFFT(pop_size=pop_size, atol=1e-6)
# Angular sensitivity <= 1 covers additive selection over |s| <= 0.1.
fine = drifft.NUFFT(
    pop_size=pop_size, atol=1e-10, grad_atol=1e-8, angular_sensitivity=1.0
)


# The allele starts with four copies.
def loss_probability(backend, s):
    transition = drifft.Transition(backend, drifft.AdditiveSelection(s=s))
    loss_indicator = jnp.zeros(transition.dest_size()).at[0].set(1.0)
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
    print(
        f"{label}: value error={abs(value - exact_value):.3e}, "
        f"gradient error={abs(gradient - exact_gradient):.3e}"
    )
```

This reference checks binomial reproduction with the same selection map.
For histories and expected visits, apply the checks to the final quantity,
not only a one-generation transition.

## Which settings control which errors?

| Component | How to tighten the approximation |
| --- | --- |
| FFT | Lower `atol` to tighten the tail budget; inspect `diagnostics()` for analytic interpolation bounds and numerical amplification before changing geometry. |
| NUFFT | Lower `atol` for values and `grad_atol` for parameter derivatives; `tail_tol` and `nufft_tol` can request stricter numerical evaluation. |
| Jacobi | Lower `atol` or `grad_atol` for automatic rank, quadrature, and sampling; explicit `rank` overrides projection planning. |
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

All built-in backends accept `atol`: a requested one-step maximum row L1 error.
For bounded observables, multiply this target by the input infinity norm; for
forward mass, multiply it by the input L1 norm. This is not a relative error
promise for rare events. Inspect `backend.error_report.components`: a `bound`
of `None` means that contribution is not certified. A `budget` is only an
allocation. In particular, transform and roundoff allowances do not establish
a total floating-point error bound.

NUFFT and Jacobi accept `grad_atol` with an explicit `angular_sensitivity` bound
on $|\partial_s q|/\sqrt{q(1-q)}$ over the parameter domain. The coordinate $s$
may be a physical parameter or its logarithm; the bound must match that choice
and the full composed frequency map. For several parameters, bound all unit
directions in the chosen parameter norm. Constant absorbing endpoints contribute
zero; support-changing endpoints need a separate argument. Preparation uses
one fixed geometry for values and derivatives. `grad_atol=None` plans values only.
FFT currently rejects gradient-tolerance requests: its diagnostics bound
branchwise angular derivatives, but branch switches can have value jumps and
there is no automatic gradient planner. Dense has zero analytic approximation error, with
uncertified arithmetic error for both values and derivatives.

`grad_atol` replaces the removed NUFFT `derivative_accuracy` heuristic. NUFFT's
`accuracy` and Jacobi's `one_step_tolerance` are replaced by `atol`. The new
derivative planning is independent of Krylov's `check_sensitivity` flag. Backend
targets concern one generation; Krylov/solver tolerances retain their existing
power-action/residual meanings. They are not end-to-end epoch guarantees. See the
[backend API](../api/backends.md) for defaults and encoding options.
If a Krylov shifted solve fails, increasing the outer `dimension` does not fix
it; see [Krylov controls](../methods/jacobi_krylov.md#accuracy-and-controls).

## Differentiating biological parameters

Selection and mutation parameters, initial mass, and source mass can be
array-valued inputs to a differentiated function. Population sizes, durations,
and numerical dimensions are fixed configuration. Construct backends outside
the function, then create or update the frequency map inside it; the
[tour](../tour.md#differentiate-selection)
shows this pattern with `jax.grad`.

For strictly positive mutation rates and positive selection strengths, prefer
natural-log inputs to the differentiated objective and exponentiate them inside
the frequency map construction. This rescales forward sensitivity directions
and can improve Krylov accuracy. It does not guarantee accuracy or make the
frequency map itself operate in log space. Use ordinary coordinates for zero
rates or selection that can be zero or negative. The
[log-input likelihood example](../methods/autodiff.md#use-log-inputs-for-positive-parameters)
shows the chain rule, sequential JVPs, and finite-difference checks.

Krylov operator sensitivities support first derivatives; use Sequential for
higher-order calculations. Sensitivity inner-solve failures and nonfinite outputs
raise even with `throw=False`. Set `check_sensitivity=True` on a Krylov method
to additionally enforce its heuristic sensitivity-error checks; the default is
`False`. This flag does not change primal convergence checks or guarantee
gradient accuracy.
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
