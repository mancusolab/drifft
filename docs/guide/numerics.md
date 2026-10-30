# Accuracy and differentiation

A calculation combines a transition approximation with a method for repeated
evolution or a linear solve. Tightening one part does not remove errors from
the others. Validate the final statistic and its gradients for your workload.

## Check the quantity you will use

Compare the same calculation with tighter numerical settings. Use a nonlinear
observable that exercises the transition approximation: a mean frequency alone
can agree because low-degree polynomials are reproduced accurately.

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
def loss_probability(backend, s_het):
    transition = drifft.Transition(
        backend, drifft.AdditiveSelection(s_het=s_het)
    )
    return transition.expect(loss_indicator)[4]

def exact_loss(s_het):
    q = drifft.AdditiveSelection(s_het=s_het)(jnp.asarray(4 / M))
    return (1 - q) ** M

strength = jnp.asarray(0.01)
exact_value = exact_loss(strength)
exact_gradient = jax.grad(exact_loss)(strength)
for label, backend in [("coarse", coarse), ("fine", fine)]:
    value = loss_probability(backend, strength)
    gradient = jax.grad(lambda s: loss_probability(backend, s))(strength)
    print(f"{label}: value error={abs(value - exact_value):.3e}, "
          f"gradient error={abs(gradient - exact_gradient):.3e}")
```

Inspect both errors against the precision required by your application.
This reference uses exact binomial reproduction and the same selection map;
it checks the transition approximation, not the selection formula itself.
For general quantities, use [`Dense`](../api/backends.md#drifft.Dense) on a small
population as an independent binomial reference.
For a history or an expected-visits solve, also compare the final statistic
under tighter settings: one-step agreement is not sufficient.

## Which settings control which errors?

| Component | Controls | What to check |
| --- | --- | --- |
| FFT | `tail_tol`, `interp_order` | Omitted binomial tails and local interpolation |
| NUFFT | `accuracy`, `tail_tol`, `nufft_tol`, `encoding` | Polynomial approximation, sampled rows, and transform evaluation |
| Jacobi | `rank`, `one_step_tolerance`, `max_strength` | Polynomial truncation and padded assembly |
| Krylov methods | `dimension`, `rtol`, `atol` | Approximation of repeated evolution |
| Expected visits | Solver tolerances and iteration limits | Linear-solve residual and amplification of transition errors |

Jacobi's `one_step_tolerance` plans ideal polynomial truncation and assembly
padding separately. It is not a total implemented-error guarantee.
`max_strength` bounds the absolute values of both diploid selection
coefficients, including those derived by constrained selection classes. Prepare
geometry with a bound covering the parameters you will evaluate in reduced
Krylov evolution. Jacobi checks these assembly bounds and the mutation-rate
domain at runtime, even for identity Krylov histories. One-generation operations
and Sequential do not use the reduced assembly. See
[supported frequency maps](../api/backends.md#drifft.Jacobi).

Krylov's `error_estimate` combines a smaller-basis comparison with a check of
the reduced recurrence. The default acceptance threshold is
`atol + rtol * sum(abs(distribution))`. With `throw=False`, inspect
`converged` and `status` before using the result. Domain errors still raise;
a failing member under `vmap` can fail the batch when throwing is enabled.

!!! warning "Convergence is not total accuracy"

    Krylov diagnostics assess the power approximation; expected-visits residuals
    assess the linear solve for an approximate transition. Neither certifies
    the exact Wright–Fisher result. Check backend accuracy separately, and
    validate gradients independently of values.

## Tune the representation separately from the method

For FFT, tighten tail truncation and increase interpolation order. For NUFFT,
lower `accuracy` and compare the final quantity again. Setting
`derivative_accuracy=True` increases polynomial degree by 50%, capped at exact
degree; it is a heuristic for derivative accuracy. Filtered encoding can reduce
the sample count needed by its spectral bound, but must still be checked for
your statistic and gradient.

For Jacobi, increase `rank`; for Krylov, increase `dimension` separately. If a
shifted solve fails, increasing the outer Krylov dimension does not fix it.
See the [backend API](../api/backends.md) and [method API](../api/methods.md) for
parameter defaults and [NUFFT transforms](../methods/nufft.md) for encoding bounds.

Check mass, negative entries, and the final observable. The implementations do
not clip or renormalize transition outputs; tiny negative entries can arise
from the numerical approximation.

## Differentiating biological parameters

The [autodiff chapter](../methods/autodiff.md) derives finite-time sensitivities
and explains JAX's differentiation of transitions, Krylov bases, and linear solves.

Frequency-map parameters, input distributions, and immigration can be dynamic
array leaves. Prepared geometry, population sizes, durations, and numerical
dimensions are fixed configuration. Construct geometry outside differentiated
functions, then replace model leaves or construct a frequency map inside the
function being differentiated. See the [population-history example](demography.md#expectations-and-selection-gradients).

NUFFT differentiates its polynomial in the offspring probability rather than
through the singular angle coordinate at endpoints. These are derivatives of
the approximation. Derivatives of its sampling nodes are unsupported.

Krylov derivatives use separate approximations to the mathematical power
sensitivity and its operator pullback. They do not differentiate Arnoldi bases.
Zero and invariant seeds are supported, but a failed sensitivity or pullback
raises even with `throw=False`. Higher-order operator sensitivities currently
raise; use Sequential for those calculations. See the
[autodiff chapter](../methods/autodiff.md#accuracy-and-limitations) for details.

!!! tip "Validate derivatives on a small reference problem"

    Compare automatic derivatives with finite differences at several step sizes
    and with a dense model where feasible. Test the parameter range you will fit,
    including relevant boundaries, rather than only the initial parameter value.

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
