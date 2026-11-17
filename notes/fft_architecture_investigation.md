# A local FFT architecture with independent convolution and interpolation grids

Research date: 2026-11-13. Baseline: `a1c2af9` on `fft_diagnostic`.

Production integration: 2026-11-17. The two-grid construction is now in
`src/drifft/_backend/fft.py`; the archived measurements below still describe the
original research baseline and prototype.

## Recommendation and scope

Keep the exact binomial diagonal–Toeplitz–diagonal (DTD) factorization, but
separate two choices that the baseline implementation tied together:

1. Choose a **fine convolution spacing** `h` to control diagonal amplification.
2. Choose a **coarser interpolation spacing** `H = k h`, with integer `k`, from
   the existing value and derivative bounds. Retain every `k`th convolution
   output. Avoid differentiating unnecessarily close, independently rounded
   node values.
3. Construct the diagonal factors using centered, cancellation-resistant
   formulas. Changing spacing alone does not fix factor-construction error.

This is a concrete candidate with provable analytic/conditioning properties
and measured improvements, not a proof of universal optimality or a complete
floating-point tolerance certificate. The research driver has been retired;
the original experiment and its compatible baseline are recoverable at commit
`c4e07c7`. Production regressions live in
[`tests/test_fft_numerics.py`](../tests/test_fft_numerics.py), using the independent
60-digit oracle in [`tests/_reference.py`](../tests/_reference.py). They construct
production FFT backends directly and run in normal test discovery and CI shard 2.

The archived prototype reused the then-production `FFT` container, runtime
`expect`, JVP/VJP, `advance`, and cached report. It substituted two host builders
and adjusted convolution gather indices after construction. Those construction
overrides were confined to the research driver; its application used the same
runtime kernels.

## What can be proved

### 1. Convolution spacing is a free parameter

Let `M` be the number of gene copies, `j0` an interior reference count,
`p=j0/M`, `eta0=logit(p)`, and `x=j-j0`. For **any** positive `h`,

\[
q_r=\operatorname{logistic}(\eta_0+hr),\qquad
B_{M,j}(q_r)=a_r e^{-h(r-x)^2/2}b_j,
\]

where

\[
b_j=\frac{B_{M,j}(p)}{B_{M,j_0}(p)}e^{hx^2/2},\qquad
a_r=B_{M,j_0}(p)
 \exp\left(j_0hr-M\log(1+p\operatorname{expm1}(hr))+hr^2/2\right).
\]

Proof: exponential tilting gives the factor `exp(hr*x)`, and
`r*x = (r*r+x*x-(r-x)*(r-x))/2`. The baseline choice
`h = 1/[Mp(1-p)]` is therefore a parameterization, not a requirement for an exact
convolution. Multiplying `h` by a constant preserves the exact binomial kernel.

For `H=kh`, query interpolation nodes at coarse integer index `t` use convolution
index `r=kt`. These are exact members of the same convolution grid. No resampling
approximation is introduced between the two grids. If a block retains `L`
columns and coarse nodes `tlo,...,thi`, the convolution gather indices become

\[
L-1+k\{0,1,\ldots,t_{hi}-t_{lo}\}.
\]

Padding must use the **dense** row span `R=k*(thi-tlo)+1`, even though only
`thi-tlo+1` node values are retained.

### 2. Refinement improves both analytic remainders

The [existing Sobolev analysis](drifft_local_fft_sobolev_analysis.md) already
supplies everything required here. Write
`A_n = sup_halo sqrt(H_n(M,q))`, using the energy polynomial from the original
analysis, and let `vmin = min_halo q(1-q)`. For interpolation step `H`, even
order `m`, and angular coordinate `theta=2 asin(sqrt(q))`, the current bounds are

\[
E_0\le\alpha_m H^m A_m,
\]

\[
E_\theta\le\frac{1}{\sqrt{v_{min}}}
 \left(\gamma_m H^{m-1}A_m+
       \frac{\alpha_m}{m+1}H^m A_{m+1}\right),
\]

with `alpha_m = binom(m,m/2)/4^m` and
`gamma_m = 1/[(m/2) binom(m,m/2)]`.

Hold the physical core and order fixed. Halving `H` gives a nested grid and a
contained stencil halo: `floor(2z)>=2 floor(z)` and
`ceil(2z)<=2 ceil(z)` show this for both endpoints. Thus the energy suprema do
not increase and `vmin` does not decrease. Consequently,

\[
E_0(H/2)\le2^{-m}E_0(H),\qquad
E_\theta(H/2)\le2^{-(m-1)}E_\theta(H)
\]

for these sufficient bounds. One can use the old halo's energy enclosure to
obtain these inequalities directly, irrespective of whether a newly computed
Bernstein enclosure is tighter. This does **not** say that total floating-point
error decreases monotonically with refinement.

### 3. Binomial curvature gives a structural conditioning gate

For an interior count `j`, the exact second difference of `log b` is

\[
\Delta^2\log b_j=h-\kappa_j,\qquad
\kappa_j=\log(1+1/j)+\log(1+1/(M-j)).
\]

Over a retained interval, let `kappa_*` be the minimum of these curvatures. If
`h < kappa_*`, then `b` is strongly log-concave with margin
`mu=kappa_*-h>0`. This prevents the input factor from decreasing and then growing
again in the retained tails, a failure mode allowed by the baseline spacing.

A conservative, exact rational lower bound is particularly cheap:

\[
\kappa_j\ge\frac1{j+1}+\frac1{M-j+1}.
\]

It follows from `log(1+z)>=z/(1+z)`. Its minimum over the integer interval occurs
at the retained count nearest `M/2`. The experiment records an outward-rounded
upper bound for `h/kappa_lower`, computed using `Fraction` and the existing
`_upper` helper. A recorded value below one establishes positive curvature
margin for the **ideal factors at the stored binary64 spacing**; it is not an
enclosure for every rounding error in constructing those factors.

Rescale so that `max b=1`, and let `j*` be a maximizer. Summing the decreasing
first differences gives

\[
b_{j_*\pm d}\le \exp[-\mu d(d-1)/2],\qquad d\ge1.
\]

Hence

\[
\|b\|_2^2\le3+\sqrt{\pi/\mu},\qquad
\|K\|_2^2\le1+\sqrt{\pi/h}
\]

for the Gaussian convolution kernel `K`. These are bounds on the whole retained
tail, not spot checks of its endpoints.

Input curvature alone is insufficient: excessively small `h` can enlarge the
output factor. After this normalization, the exact identity at `j*` gives

\[
a_r=B_{M,j_*}(q_r)\exp[h(r-(j_*-j_0))^2/2].
\]

One useful uniform bound is

\[
\max_j B_{M,j}(q)
 \le\min\left(1,\frac{\sqrt\pi}{2\sqrt{2Mq(1-q)}}\right).
\]

To prove it, Fourier-invert the binomial characteristic function, take absolute
values, use `(1-4v sin^2(t/2))^(M/2)<=exp(-2Mv sin^2(t/2))`, then
`sin(|t|/2)>=|t|/pi` on `[-pi,pi]`, and extend the resulting Gaussian integral to
the real line. Together these inequalities give an explicit bound on both
diagonals and the kernel. A production planner must control the resulting
product, not just test log-concavity or finiteness.

### 4. Why the two grids help derivative conditioning

The central interpolation weights obey `sum |lambda_i|<=2` and
`sum |lambda_i'|<=4` for the supported orders. If each retained node action has
error at most `e ||v||_infinity`, its contribution after interpolation is bounded
by

\[
2e\|v\|_\infty\quad\text{and}\quad
\frac{4e}{H\sqrt{v_{min}}}\|v\|_\infty
\]

for values and angular derivatives respectively. Parameter derivatives multiply
the second expression by the caller's `angular_sensitivity`.

The old architecture uses `H=h`, even when interpolation is substantially more
accurate than required. Reducing `h` helps the DTD factors but worsens this
derivative amplification. With `H=kh`, the explicit inverse-step factor is
smaller by `k`; the halo variance and node-error bound must also be recomputed.
This is why simply choosing smaller and smaller spacing is not the final design.

For fixed `h` and candidate order, the prototype chooses the largest dyadic
integer `k<=64` meeting both interpolation budgets. Its tails use `H` too.
The upper cap is a finite research search limit, not a mathematically preferred
constant. A production cost search should consider more than just the largest
admissible stride, since larger halos can alter factor amplification and padding.

### 5. What the amplification number means

The study reports

\[
\mathcal K=\sqrt{N_{FFT}}\max_r|a_r|\|b\|_2\|K\|_2.
\]

This is a sufficient **roundoff-amplification factor**, not the condition number
of the stochastic transition matrix. Opposite scalar shifts of the two
diagonals leave it unchanged.

For clarity, there is a conditional theorem behind this proxy. Suppose each
forward FFT has normwise relative error at most `delta_F` against the norm of
the exact transform, the inverse has at most `delta_I`, and pointwise complex
multiplication has componentwise relative error at most `delta_mul`. With the
usual unnormalized forward and `1/N` inverse conventions (interpreting real FFTs
as full Hermitian spectra), exact input factors,
and no under/overflow, the convolution action error is at most

\[
\left((1+\delta_F)^2(1+\delta_I)(1+\delta_{mul})-1\right)
\mathcal K\|v\|_\infty.
\]

Proof: use `||F||_2=sqrt(N)`, `||F^-1||_2=1/sqrt(N)`, and
`||u .* w||_2<=||u||_2 ||w||_2` to bound each perturbation and their products.
The preceding interpolation inequalities propagate this nodal bound to values
and derivatives. Factor construction, coordinate evaluation, interpolation
arithmetic, and transpose accumulation require their own additional terms.

No platform-specific constants for the installed NumPy/JAX FFT implementations
were established here. General FFT error results depend on details such as
accurate twiddle factors; the [FFTW authors' accuracy discussion](https://fftw.org/accuracy/comments.html)
explains this dependence. It is therefore incorrect to label `u*K` alone a
certified production error bound. The measured `K` values below use ordinary
floating-point norms; only the curvature-ratio upper bound is outward enclosed.

## Stable factor preparation is essential

The original output log-factor contains

\[
j_0\delta-M\log(1+p\operatorname{expm1}(\delta)),\qquad\delta=hr.
\]

Each term is first order in `delta`, while their difference is second order.
Numerical noise in independently prepared nodes is then amplified by a
derivative divided by `H`. This explains why spacing refinement alone still
missed the tight gradient target in an intermediate experiment.

Define the centered Bernoulli cumulant

\[
C_p(\delta)=\log(1+p\operatorname{expm1}(\delta))-p\delta.
\]

Use `C_p(delta)=C_(1-p)(-delta)` to reflect to `p<=1/2`. Then, with
`t=p*expm1(delta)`, evaluate

\[
C_p(\delta)
=p\,[\operatorname{expm1}(\delta)-\delta]
 +[\log(1+t)-t].
\]

Both bracketed terms begin at second order. The prototype evaluates them by
Horner series in their small-argument ranges, and uses the ordinary functions
outside those ranges. The exponential remainder through degree 27 has tail
at most `e*abs(delta)^28/28!` for `abs(delta)<=1`. The logarithmic remainder
through degree 65 has tail at most `abs(t)^66/[66*(1-abs(t))]` for
`abs(t)<=1/2`. These are analytic truncation bounds, not complete library or
Horner rounding bounds. The implementation is checked against 70-digit Decimal
evaluation, including small `delta` and reflection.

The column recurrence also uses small relative quantities:

\[
\log(b_{j+1}/b_j)=
\log\left(1-\frac{j-j_0}{M-j_0}\right)
-\log\left(1+\frac{j-j_0+1}{j_0}\right)
+h(j-j_0+1/2).
\]

Accumulate outward from `j0`, rather than accumulating from the left edge and
subtracting a large common prefix. Runtime arithmetic remains binary64. Decimal
is used only by the independent validation oracle, not by the new factor builder.

## Evidence and limitations

Results are recorded in [`fft_architecture_results.json`](fft_architecture_results.json).
The final comparison uses identical physical query points for both plans.
Earlier ablations retain their original plan-specific query sets and should not
be treated as paired uniform-domain tests.

The oracle constructs complete binomial rows by an independent normalized
recurrence at 60 decimal digits, with exact conversion of the binary64 query.
It also computes `B_(M,j)(q)*(j-Mq)/sqrt(q(1-q))` at high precision. NumPy
`longdouble` has only binary64 precision on this machine, so it would not have
been an independent higher-precision reference.

Full coefficient and derivative rows are extracted through JAX transposition
in batches of 16 query seeds. Their maximum row L1 errors measure all signed
destination vectors for the extracted linear operator. Because floating-point
execution itself is not exactly linear, the driver separately evaluates actual
forward actions on random signs and signs of the largest observed coefficient
errors. It also checks a signed forward distribution and records its L1 error.
The tests exercise JIT, batching, JVP, VJP, and transposition.

Timing uses the same full population query grid `q=linspace(0,1,M+1)` for every
plan, dynamic geometry arguments, synchronized outputs, and repeated warm runs.
These are local CPU measurements on macOS ARM64, JAX/JAXLIB 0.11.2, Equinox
0.13.8, NumPy 2.5.4, and SciPy 1.18.1. They are not GPU or cross-machine results.
Planner caches are reused across candidates within an invocation. The paired
reproduction commands use separate invocations to compare preparation costs.

### Paired results

Both plans requested `atol=1e-12`, `grad_atol=1e-10`, and
`angular_sensitivity=1`. Each was checked at the same 61 probabilities,
including endpoints, both sides of direct cutoffs, representative block
interfaces, and logit-spaced interiors. Timing uses 25 warm repetitions.

The baseline planner selected order 16, cutoff 128, and `h=H=1/V`. The proposed
candidate used order 12, cutoff 32, `h=1/(4V)`, stable factors, and per-block
strides 1–8 (`M=8192`) or 1–16 (`M=32768`). These are two candidate selections,
not a claim that those constants are optimal for every tolerance or population.

| Gene copies | Plan | Maximum sampled value row L1 error | Maximum sampled angular-gradient row L1 error | Maximum amplification `K` |
| ---: | --- | ---: | ---: | ---: |
| 8,192 | Current | 1.15e-12 | 1.07e-9 | 6.30e5 |
| 8,192 | Proposed | 8.95e-14 | 8.69e-12 | 1.79e2 |
| 32,768 | Current | 2.20e-12 | 6.28e-9 | 1.41e6 |
| 32,768 | Proposed | 2.65e-14 | 2.66e-11 | 2.26e2 |

The proposal reduced these observed gradient errors by 123× and 236×, and the
amplification proxy by 3,511× and 6,241×. Its curvature-ratio **upper bounds**
were 0.730 and 0.741, so every retained input factor has a positive proved
log-concavity margin. The baseline plans' upper ratios round outward to 1.909 and
1.954; these plans do not meet this sufficient gate.

Actual signed forward/JVP actions also improved. At `M=32768`, the largest
tested gradient action error fell from 3.84e-9 to 2.66e-11, and signed
`advance` L1 error fell from 1.22e-13 to 2.85e-15. No clipping, positivity repair,
or mass normalization was applied to the backend output.

| Gene copies | Plan | `expect` ms | value + angular JVP ms | `advance` ms | Stored arrays MB | Preparation s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8,192 | Current | 1.992 | 3.484 | 4.331 | 5.552 | 10.27 |
| 8,192 | Proposed | 1.878 | 2.215 | 3.001 | 5.881 | 10.54 |
| 32,768 | Current | 5.382 | 11.499 | 15.528 | 22.439 | 22.70 |
| 32,768 | Proposed | 5.577 | 6.872 | 10.629 | 23.204 | 25.95 |

At 32,768 copies this is a 1.67× gradient speedup and 1.46× forward-propagation
speedup, with value application 3.6% slower, stored geometry 3.4% larger, and
preparation 14.3% slower. Stored arrays exclude compiler/runtime temporaries.
The dense FFT work proxy actually increased by 5.8%; the reduced order and
direct width, and fewer stored output nodes, explain why application can still
improve. This is a practical accuracy/cost improvement, not an asymptotic gain
or a strict speed win on every operation.

### Ablations and negative results

Earlier `M=8192` runs used 42 plan-specific probes. At cutoff 128:

| Change from baseline plan | Maximum angular-gradient row L1 error |
| --- | ---: |
| Current, order 16 | 7.14e-10 |
| Stable factors only, order 16 | 4.92e-10 |
| Half spacing only, order 12 | 2.22e-10 |
| Half spacing + stable factors, order 12 | 5.25e-11 |
| Quarter spacing + stable factors, order 12 | 5.67e-11 |

Neither factor repair nor spacing refinement alone met this target. Further
refinement was not monotonically better. With stable half spacing, introducing
separate interpolation spacing reduced the sampled **interior** gradient error
from 1.34e-11 to 5.69e-12; direct rows then dominated the total. Reducing the
direct cutoff to 32 produced the final combined candidate.

These larger errors in the baseline backend concern floating-point arithmetic;
they do not refute its documented exact-arithmetic, branchwise component bounds.
Likewise, the successful proposed runs are measurements, not a uniform proof
over all probabilities, vectors, sizes, or machines. Even the reduced conservative
gradient amplification factor is not, on its own, sufficient to certify the
requested `1e-10` target under a general FFT error model. Establishing a complete
arithmetic budget remains separate work. The precise claims established here
are the analytic identities/envelopes above and the observed improvements below
the requested tolerances in these experiments.

### Reproduction and checks

To reproduce the original experiment, use a separate checkout of commit
`c4e07c7` and its locked environment. Run these commands from that checkout,
where the archived driver and its compatible production code are both present:

```sh
uv sync --locked
mkdir -p .cache
uv run python benchmarks/investigate_fft_spacing.py \
  --sizes 8192 32768 --scales 1 --cutoffs 128 \
  --atol 1e-12 --grad-atol 1e-10 --common-queries --repeats 25 \
  --output .cache/fft_spacing_current.json

uv run python benchmarks/investigate_fft_spacing.py \
  --sizes 8192 32768 --scales .25 --cutoffs 32 \
  --atol 1e-12 --grad-atol 1e-10 --common-queries --repeats 25 \
  --stable-factors --separate-grids \
  --output .cache/fft_spacing_proposed.json

uv run pytest -p no:capture benchmarks/test_fft_spacing.py -q
```

The archived CLI saves a JSON report after each candidate and writes summaries
to stdout. Inspect any `error` entries in the report; that historical CLI does
not indicate rejected candidates with a nonzero exit status. The recorded
2026-11-13 measurements retain their original capture format.

Use `--plan-only` to inspect geometry without reference or timing work, and
`--grad-atol none` to disable gradient planning. `--query-count` controls probe
density; interface probes increase the actual reported count. With
`--common-queries`, the default `--query-count 32` produces 61 queries at the
paired study sizes.

The original investigation passed **15 independent checks**, including baseline
equivalence to the baseline backend at power-of-two and non-power-of-two sizes,
high-precision factor identities and centered cumulants, the log-concavity
envelope, the refinement bound, and signed JIT/JVP/VJP/batching/transpose actions
for both single-grid and two-grid plans. Current production regressions retain
the factorization, conditioning, refinement, signed-action, and large-case
accuracy checks, along with direct-only fallback, Decimal context isolation,
probability validation, and nonfinite-factor rejection. Trial-only comparisons
and CLI tests were removed with the driver. Run the retained checks from the
current checkout with:

```sh
uv run pytest -p no:capture tests/test_fft_numerics.py -q
```

This retirement does not regenerate the recorded performance tables.
The aggregate contains 45 candidate runs across 10 experiment batches, including
planning-only and unsuccessful ablations. It does not claim that all 45
candidates improved on baseline.

## Production design

### Implemented construction

The production backend starts at `h=1/(4V)` and `c=32`, tries orders 12/14/16,
and selects dyadic interpolation strides up to 64 from the existing `atol` and
`grad_atol` allocations. Explicit cutoffs remain honored as initial cutoffs.
`inverse_step` stores `1/H`; the misleading `variance` field has been removed.
Strided convolution gathers and dense-span FFT padding are prepared together.
All production helpers remain in the FFT module, with no new public classes.

After tail windows are final, the builder checks the rational curvature lower
bound. If necessary it halves `h` and doubles the stride, preserving every
interpolation node. Centered cumulants and outward `log1p` recurrences construct
the factors. The default balancing now normalizes `max(b)=1`: centering the
entire log range can overflow when a wide window includes very small tails.
The driver at the historical checkpoint uses its original balancing.
Production evaluates large centered-cumulant shifts with `logaddexp`, avoiding
the exponential cancellation in the original helper outside its series range.
Temporary block factors remain in NumPy until bucket packing; runtime geometry
is converted to JAX only after that preparation.

Independent production regressions at the two research sizes, using the same
61 physical probabilities and 60-digit reference, measured:

| Gene copies | Maximum row L1 error | Maximum angular-gradient row L1 error |
| --- | --- | --- |
| 8,192 | `8.95e-14` | `8.69e-12` |
| 32,768 | `2.65e-14` | `2.66e-11` |

Signed value/JVP actions and transpose propagation also passed the requested
`atol=1e-12`, `grad_atol=1e-10` tolerances. These remain sampled CPU regressions.
The tests separately verify factor identities, curvature refinement preserving
nodes, strided padding/gathers, and second-order AD against binomial derivatives.

### Further certified planning

The public interface can remain `FFT(..., atol=..., grad_atol=...,
angular_sensitivity=...)`, with the same report type. This does not require a
new diagnostics class or runtime acceptance test.

The host planner should jointly consider order, cutoff, convolution spacing,
interpolation stride, and tail windows. For each candidate:

1. Bound interpolation using `H`, including its full halo.
2. Choose count windows from the value and derivative tail budgets.
3. Check the exact-rational curvature lower bound on the **final** window.
4. Bound output-factor and convolution amplification as well; input curvature
   alone does not establish numerical stability.
5. Account for factor/FFT/interpolation/coordinate arithmetic under the supported
   implementation's error model. Reject infeasible targets or use a representation
   with a justified arithmetic budget. Do not promise an arbitrarily small
   binary64 gradient error merely by enlarging the direct region.
6. Select among accepted candidates using actual bucket sizes and direct width.

This search runs during construction. All selected geometry stays fixed during
an application and its derivative. The experiments compare particular candidate
plans; they do not implement this entire automatically certified search.

The integration implements the tested construction and final-window curvature
guard, not this full arithmetic-certified or cost-optimized search. The report
continues to leave roundoff and total bounds unknown. Input curvature alone
does not certify output-factor amplification or arbitrary binary64 targets.

For fixed tolerances and bounded refinement factors, local core spacing gives
`O(sqrt(M))` blocks with `O(sqrt(M))` worst-case padded transforms: the same
`O(M log M)` convolution scaling. More precisely, cost is

\[
O\left(\sum_b N_b\log N_b+m^2 Q_I+WQ_D\right).
\]

Here the `m^2` term describes the existing product-form interpolation code,
`W` is direct width, and actual bucket padding matters. Striding adds no FFT
per block; it can enlarge the dense halo, while reducing retained nodes.
No asymptotic speedup is claimed. Preparation gets more expensive when searching
strides; application can get faster because a smaller interpolation order and
direct region suffice.

The existing branchwise qualification remains. Block/direct switches can still
jump, and stencil switches can have derivative kinks. A global classical `C^1`
or weak Sobolev promise would require an additional overlap construction and its
derivative error terms. This study does not implement or validate that change.

## Alternatives evaluated

- **Only rebalance the two diagonals:** useful for exponent range, but leaves
  their product-norm amplification unchanged.
- **Only raise interpolation order or tighten tails:** the baseline planner can
  improve its analytic bounds while making DTD arithmetic worse. The baseline
  tight-tolerance experiments exhibit this problem.
- **Only halve spacing:** a useful first improvement, but dense interpolation
  amplifies node noise in gradients; factor preparation still needs repair.
- **Only improve factor preparation:** removes one error source, but leaves the
  unstable DTD product norm when `h` is too large.
- **Generic chirp-z replacement:** does not establish better conditioning.
  [SciPy's CZT documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.CZT.html)
  itself notes precision limitations. That is context for rejecting an assumed
  improvement, not a theorem about this binomial construction.
- **Smooth block blending:** potentially useful for global gradient regularity,
  but does not fix diagonal amplification or inaccurate prepared node values.

The recommendation is the combined construction above. It has a specific
reason for each component and can use the current backend machinery.
