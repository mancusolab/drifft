# Accuracy and differentiation

A calculation combines a transition approximation with a method for repeated
evolution or a linear solve. Tightening one part does not remove errors from
the others. Validate the final statistic and its gradients for your workload.

## Compare a calculation at tighter settings

This example checks an expected offspring frequency at two NUFFT accuracy
settings while differentiating a biological parameter:

```python
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)
coarse = drifft.NUFFT(64, accuracy=1e-8)
fine = drifft.NUFFT(64, accuracy=1e-11, derivative_accuracy=True)
frequencies = jnp.arange(65) / 64

def statistic(backend, strength):
    transition = drifft.Transition(
        backend, drifft.AdditiveSelection(s_het=strength)
    )
    return transition.expect(frequencies)[1]

strength = jnp.asarray(0.01)
value_difference = statistic(coarse, strength) - statistic(fine, strength)
gradient_difference = (
    jax.grad(lambda s: statistic(coarse, s))(strength)
    - jax.grad(lambda s: statistic(fine, s))(strength)
)
```

Inspect differences against the precision required by your application. This
one statistic is an example, not a test of the complete transition: also check
the distribution or other observables you actually use. For small populations,
compare with a dense binomial reference. For long histories and expected visits,
check the final output rather than only one-generation behavior.

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
`max_strength` bounds supported underdominance strength for assembly planning;
prepare geometry with a bound covering the strengths you will evaluate.

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

## NUFFT controls

| Option | Default | Meaning |
| --- | --- | --- |
| `accuracy` | `1e-10` | Nominal value-error budget; `[1e-13, .1)` |
| `encoding` | `"lobatto"` | Ordinary interpolation or `"filtered"` tapered encoding |
| `tail_tol` | Allocated automatically | Optional stricter omitted binomial mass per node |
| `nufft_tol` | `min(1e-13, accuracy/4)` | Requested FINUFFT tolerance; at least `1e-15` |
| `batch_size` | `32` | Sampling nodes processed together |
| `derivative_accuracy` | `False` | Increase degree by 50%, capped at exact degree |
| `dtype` | `jnp.float64` | Float64 only; enable JAX x64 before planning |

The nominal budget includes polynomial approximation, binomial sampling,
transform evaluation, and roundoff. Actual FINUFFT error and arithmetic need
empirical validation. `derivative_accuracy=True` is a larger-degree heuristic,
not a derivative certificate. Details of the two encodings and their bounds
are in [Chebyshev/NUFFT transforms](../methods/nufft.md).

Sampling weights are normalized within retained binomial stencils. This is
part of constructing the approximation; outputs are not clipped or normalized.
Exact endpoint treatment does not guarantee positivity of approximate interior
coefficients. Check mass, negative entries, and your final observables.

## Differentiating biological parameters

Frequency-map parameters, input distributions, and immigration can be dynamic
array leaves. Prepared geometry, population sizes, durations, and numerical
dimensions are fixed configuration. Construct geometry outside differentiated
functions, then replace model leaves or construct a frequency map inside the
function being differentiated. See the [population-history example](demography.md#expectations-and-selection-gradients).

NUFFT differentiates its polynomial in the offspring probability rather than
through the singular angle coordinate at endpoints. These are derivatives of
the approximation. Derivatives of its sampling nodes are unsupported.

JAX differentiates the input-dependent Krylov basis. At zero seeds or changes
in Arnoldi breakdown, derivatives are not guaranteed. Rational Krylov's
FFT/NUFFT shifted solves use implicit differentiation through Lineax; failed
tangent or transpose solves raise even if the primal call used `throw=False`.

!!! tip "Validate derivatives on a small reference problem"

    Compare automatic derivatives with finite differences at several step sizes
    and with a dense model where feasible. Test the parameter range you will fit,
    including relevant boundaries, rather than only the initial parameter value.

## Memory and timing

Differentiation can retain more intermediates than forward evaluation,
especially across long intervals. NUFFT's `batch_size` controls temporary
binomial-sampling storage; native FINUFFT workspace is additional to JAX memory
reports. Krylov basis storage grows with working-state size and `dimension`.

For timing, separate preparation, first-call compilation, and warm execution,
and synchronize JAX outputs. Use [validation and benchmarks](../benchmarks.md)
for the repository's comparison commands and recorded results.
