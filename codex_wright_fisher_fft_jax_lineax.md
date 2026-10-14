# Codex task: JAX/Lineax Wright–Fisher operator using local log-odds FFTs

Implement and validate a **matrix-free, FFT-accelerated Wright–Fisher (WF) transition operator** in **JAX**, and expose the transient fundamental-matrix solve through **Lineax**. The algorithm should exploit an **exact local diagonal–Toeplitz–diagonal factorization of the binomial transition kernel**, with controlled approximation introduced only by finite input windows and interpolation between local log-odds grids.

**Do the implementation, tests, and benchmarks—not just a design sketch.** Inspect the repository and conform to its existing module layout, packaging, typing, and test conventions. If no relevant implementation exists, add a self-contained module with a small documented public API. Do not introduce a dependency on a sparse-matrix package or a custom CUDA kernel. NumPy/SciPy may be used for *offline reference calculations or one-time host-side plan construction*, but all repeated matvecs and iterative solves must execute entirely in JAX/XLA.

## 1. Statistical model and required outputs

Use `M = 2N` haploid gene copies and states `i,j = 0,...,M`.

The one-generation transition probabilities are

\[
P_{ij}(s)=\binom{M}{j}q_s(i/M)^j[1-q_s(i/M)]^{M-j},
\]

with symmetric underdominance

\[
q_s(x)=\frac{x^2+(1-s)x(1-x)}{1-2s x(1-x)}
      =\frac{x-sx(1-x)}{1-2s x(1-x)}.
\]

Initially support `0 <= s <= 1`, including `s=0` (neutrality) and `s=1`. Make it straightforward to substitute another differentiable `q(x, params)` later without rebuilding the FFT geometry. States `0` and `M` are absorbing.

Implement:

- A full operator `P(s) @ v`, accepting and returning vectors of length `M+1` (useful for correctness tests).
- A transient operator `Q(s) @ v`, where `Q = P[1:M, 1:M]`, accepting and returning vectors of length `M-1`. Implement this by padding the input with zero boundary entries, evaluating the full binomial transition on transient output rows, and stripping the output boundaries. **Do not renormalize transient rows.**
- Its transpose `Q(s).T @ v`, preferably obtained by JAX/Lineax operator transposition rather than a separately coded summation.
- A Lineax-based solution for the expected occupation counts from one mutant:

  \[
  g_j(s)=\sum_{t\ge0}(Q_s^t)_{1j},\quad
  (I-Q_s^\mathsf T)g=e_1.
  \]

Here `e_1` denotes state `i=1`, i.e. **index 0** in arrays of length `M-1`. Return solution diagnostics; never silently present an unconverged result as valid.

Provide a baseline, vectorized **band-truncated binomial matvec implemented in JAX** for accuracy/performance comparisons. It can be a test/benchmark implementation rather than the production backend. Do not materialize dense `M x M` matrices except in small tests.

## 2. Exact local FFT identity — implement this, not a diffusion approximation

For any interior success probability `p` define

\[
\eta=\operatorname{logit}(p),\qquad
A(\eta)=\operatorname{softplus}(\eta)=\log(1+e^\eta).
\]

The binomial probabilities can be written

\[
P_{\eta,j}=\binom Mj\exp\{j\eta-MA(\eta)\}.
\]

Select an **integer** reference count `j0` and set

\[
p_0=j_0/M,\quad \eta_0=\operatorname{logit}(p_0),\quad
\sigma_0^2=Mp_0(1-p_0),\quad h=1/\sigma_0^2.
\]

At equally spaced log-odds nodes `eta_r = eta0 + h*r`, with integer `r`, and columns `j = j0 + ell` with integer `ell`, define

\[
B_0=\binom M{j_0}\exp\{j_0\eta_0-MA(\eta_0)\},
\]

\[
\begin{aligned}
a_r&=\exp\left[h r j_0 - M\{A(\eta_0+hr)-A(\eta_0)\}+\tfrac h2r^2\right],\\
b_\ell&=\frac{\binom M{j_0+\ell}}{\binom M{j_0}}\exp\left[\eta_0\ell+\tfrac h2\ell^2\right],\\
k_d&=\exp(-\tfrac h2d^2).
\end{aligned}
\]

Then **exactly**, apart from floating-point rounding,

\[
\boxed{P_{\eta_r,j_0+\ell}=B_0 a_r k_{r-\ell}b_\ell.}
\]

The identity follows from `r*ell = (r^2 + ell^2 - (r-ell)^2)/2`. It is not a Gaussian approximation to the binomial distribution.

For input columns `ell_min:ell_max`, compute simultaneously for all integer output nodes `r_min:r_max`:

\[
F_r=\sum_{\ell=\ell_{\min}}^{\ell_{\max}}
P_{\eta_r,j_0+\ell}\,v_{j_0+\ell}
=B_0 a_r\,[k*(b\odot v)]_r.
\]

Use a **linear FFT convolution**, never a circular convolution with wraparound aliases:

- `L = ell_max - ell_min + 1`, `R = r_max - r_min + 1`.
- Set `u[t] = b_(ell_min+t) * v[j0+ell_min+t]`, `t=0,...,L-1`.
- Set `kmin = r_min - ell_max`; construct `k[q] = exp(-h*(kmin+q)^2/2)`, `q=0,...,R+L-2`.
- The full linear convolution has length `R+2L-2`, so pad both transforms to `nfft >= R+2L-2` (typically next power of two).
- The desired output is

  `F[r_min+a] = B0 * a[r_min+a] * conv[u,k][a+L-1]`, for `a=0,...,R-1`.

Confirm this indexing using a small unit test against a direct binomial matrix **before building the global method**. Precompute FFTs of the kernels once per block (or block batch) because they do not depend on `v` or `s`.

## 3. Piecewise-local approximation across the entire frequency range

A global log-odds factorization is numerically ill-conditioned. **Use small overlapping local blocks** whose widths are proportional to the local binomial standard deviation; do not apply the above diagonal factors to an entire `M`-length grid.

Suggested initial algorithm (improve details if needed, but document changes):

1. **Boundary cutoff.** Choose a configurable `min_mode_count` (start around `128`, tune based on numerical tests). For output rows with `M*p` near zero, or `M*(1-p)` near zero, use a direct **fixed-width, JIT-compatible binomial sum**. Route based on `p=q_s(i/M)`, not on `i`, because selection changes the mean. Include `p=0` and `p=1` exactly. Choose the direct-sum width from a binomial-tail bound so it achieves the requested accuracy throughout the boundary region.
2. **Interior centers.** On the one-time, selection-independent host planning path, choose ascending integer reference counts `j0` in `[min_mode_count, M-min_mode_count]` with successive spacings `O(sqrt(j0*(1-j0/M)))`; a constant times `sigma0` is an appropriate initial spacing. Set each block's *core* log-odds interval using midpoints between adjacent centers in `eta0`, with the extreme cores ending at the boundary thresholds.
3. **Output nodes.** For each block, generate `eta_r = eta0 + h*r` covering its core log-odds interval, plus enough **overlap/halo for high-order interpolation**. Choose integer `r_min`, `r_max` at plan-construction time. The range should remain `O(sigma0)` away from the boundary regime.
4. **Input window.** Determine `[j_min,j_max]` that captures the binomial mass, to a configurable tolerance `tail_tol`, for **every** log-odds node used by that block, including the interpolation halo. A conservative option is Bernstein's binomial tail bound. For `X ~ Bin(M,p)`:

   \[
   \Pr(|X-Mp|>t) \le 2\exp\left(-\frac{t^2}{2(Mp(1-p)+t/3)}\right).
   \]

   Taking `L_eps = log(2/tail_tol)` and

   \[
   t = \sqrt{2\,v_{\max} L_{\epsilon}} + \tfrac23 L_{\epsilon}
   \]

   with `v_max = max_{p in block} M*p*(1-p)` gives a conservative width. Use the extrema of the output-node means to bound `j_min`/`j_max`, clipping to `[0,M]`.
5. **Compute a local FFT** producing `F_r` at those nodes. These `F_r` are the *truncated binomial sums* at the grid points, not Gaussian approximations.
6. **Interpolate** `F(eta)` from its local uniform grid to each actual `eta_i(s) = logit(q_s(i/M))` falling in the corresponding block core. Start with 8-point local polynomial (or barycentric Lagrange) interpolation; make interpolation order configurable. All interpolation weights must depend on `eta_i(s)` and static nodes, **never on `v`**, so the resulting operator stays linear in `v`.
7. Ensure each interior row uses one block's core interval; output-node halos exist only to support interpolation, not to double-count rows. Test continuity and error near every block seam.

The offline plan should cover every interior target frequency, independently of the selected `s`, by constructing the blocks over the **full allowed range of success probabilities**. Dynamic row-to-block routing can use vectorized `jnp.searchsorted` on sorted static boundaries. No Python loops over individual rows in the JIT path.

If an 8-point interpolant introduces small negative kernel coefficients, do **not** silently clamp final `P@v` or `Q@v` values: clipping is nonlinear in `v` and breaks the assumptions of `FunctionLinearOperator` and transpose-based solves. Quantify the error instead; explore alternative linear interpolation schemes if needed.

## 4. Numerical stability — a core requirement

Use **JAX float64** for the initial implementation (`jax_enable_x64=True`) and benchmark float32 only after float64 is reliable. In particular:

- Never form `binom(M,j)` directly. Compute `log(B0)` via log-gamma or an equivalent stable binomial log-PMF.
- Form `log(a_r)` and `log(b_ell)` using stable expressions; avoid exponentiating huge intermediate factors. Scale/recenter the diagonal multipliers *within each block* if needed, without changing the mathematical product.
- One useful stable identity for softplus differences is

  \[
  A(\eta_0+\delta)-A(\eta_0)
  =\log1p\{p_0\operatorname{expm1}(\delta)\}.
  \]

- The input scaling has an especially useful **local recurrence**:

  \[
  \log b_{\ell+1}-\log b_\ell
  =\log(M-j_0-\ell)-\log(j_0+\ell+1)
   +\eta_0+h(\ell+\tfrac12),\qquad b_0=1.
  \]

  Use forward/backward cumulative sums from `ell=0` when this is more accurate than subtracting two large `gammaln` values. Guard endpoints.
- A stable expression for `log(a_r)` can use `delta=hr` and subtract the first two terms of the local softplus Taylor series, taking advantage of `h=1/(Mp0(1-p0))` to cancel the quadratic term analytically. Check whether small-`delta` series evaluation improves accuracy.
- In the exact boundary/reference matvec, compute binomial PMFs in log space using `gammaln`, `xlogy`, or equivalent endpoint-safe functions. Handle `p` equal to zero or one explicitly.
- Check overflow/underflow and cancellation in both positive and **signed** input vectors. A signed vector is essential because Krylov iterates need not be nonnegative.
- Establish numerical error **relative to `||v||_infinity`**, not only relative to a potentially near-zero result component.
- `tail_tol`, interpolation order, block spacing, boundary cutoff, and optional balancing/scaling should be externally configurable with documented defaults. Conservative defaults are preferable until benchmarks justify relaxation.

The two sources of approximation are **truncation of the binomial input support** and **interpolation in log-odds**. Distinguish them in the tests. FFT roundoff is a third, numerical rather than modeling, error source.

## 5. JAX/XLA design and performance expectations

Prioritize a fast repeatedly applied operator, e.g. inside an iterative solver or over many `s` values. The basic architecture should be:

- A **selection-independent plan** built once for a given `M`/precision/tolerances. Store JAX arrays for static counts, offsets, local grid metadata, scaled diagonal multipliers, and precomputed kernel FFTs. A PyTree dataclass or an `equinox.Module` is fine if it fits existing conventions.
- Plan construction may use Python/NumPy/SciPy **outside JIT**, but matvec execution must not use host callbacks, device-to-host transfers, SciPy, Python per-row loops, dynamic-length arrays, or dense `P`/`Q`.
- Since local FFT sizes differ, **bucket blocks by power-of-two FFT length** and `vmap`/batch the RFFTs and IRFFTs within each bucket. Keep padded arrays and masks JIT-static. A small, static Python loop over buckets during tracing is acceptable; a Python loop over all individual blocks or rows is not.
- Precompute/reuse `rfft(k)` for each block. Use `jnp.fft.rfft` and `jnp.fft.irfft` with explicit transform lengths and linear-convolution indexing.
- Store all computed log-odds grid outputs as one flat array (or equivalent efficiently indexable bucket arrays). Route each target `eta_i(s)` to its block and gather its interpolation stencil without triggering recompilation for new `s` values.
- Ensure padded entries **cannot leak into valid outputs** and do not produce NaNs after rescaling.
- In the hot path, treat `M`, plan shapes, interpolation order, and FFT lengths as static; treat `s` and `v` as runtime JAX arguments. Repeated calls with different `s` must not rebuild the FFT kernels or recompile merely because `s` changed.
- Support `jax.jit`; check `jax.vmap` over selection coefficients if reasonable. Use `.block_until_ready()` for honest timing.

The target, for fixed precision and well-behaved block sizes, is **approximately `O(M log M)` work and `O(M)` storage** for the FFT matvec, *not* `O(M^2)` and not a disguised `O(M^(3/2))` batch of direct probability evaluations. Verify empirically; do not assert that FFT wins on a GPU without measuring it. In particular, the baseline banded JAX gather may outperform many small FFTs at moderate `M`.

## 6. Lineax integration and transpose

Represent the transient operator as a `lineax.FunctionLinearOperator`. Use `Q.T` or `jax.linear_transpose` for the transpose; verify this works through **FFT, interpolating gathers, boundary routing, and scatters**. The numerical map must be linear in its input `v` for fixed `s`.

Illustrative shape/API (adapt to the installed Lineax version):

```python
import jax
import jax.numpy as jnp
import lineax as lx

spec = jax.ShapeDtypeStruct((M - 1,), jnp.dtype("float64"))

# plan is constructed once, independently of s.
q_op = lx.FunctionLinearOperator(
    lambda v: apply_q_fft(plan, s, v),
    input_structure=spec,
)
qt_op = q_op.T

a_op = lx.FunctionLinearOperator(
    lambda x: x - qt_op.mv(x),
    input_structure=spec,
)

rhs = jnp.zeros((M - 1,), dtype=jnp.float64).at[0].set(1.0)
solver = lx.GMRES(rtol=1e-9, atol=1e-11, max_steps=..., restart=...)
solution = lx.linear_solve(a_op, rhs, solver=solver, throw=False)
g = solution.value
residual = a_op.mv(g) - rhs
# Inspect solution.result, solution.stats, residual, and numerical accuracy.
```

Also benchmark `lx.BiCGStab` as an alternative. **Do not use CG for the nonsymmetric matrix** `I-Q.T`, and do not tag the operator symmetric or positive definite without a mathematical proof.

The exact transpose relationship is mandatory:

\[
\langle u,Qv\rangle=\langle Q^\mathsf Tu,v\rangle
\]

up to float64 roundoff for the **implemented approximate operator**, with signed random vectors. The transpose of a numerical approximation need not be the exact transpose of the original binomial matrix; compare both to the exact small-`M` matrix separately.

Report `|| (I-Q.T)g-e1 ||`, number of iterations, and the Lineax result status. Solving an approximate operator to a tiny *internal* residual does **not** by itself establish accuracy for the original WF fundamental matrix. The condition number or expected absorption time can amplify one-step operator error, particularly near neutrality.

If derivatives with respect to `s` are straightforward under the resulting JAX design, test them against finite differences as an optional extension. Do not sacrifice correct matvec/transpose/solve behavior to implement derivatives prematurely; piecewise block routing and interpolation can complicate derivatives at grid or block boundaries.

## 7. Tests — implement these as automated tests

### A. Factorization, FFT indexing, and single-block stability

1. For moderate `M` (e.g. `100`, `1000`, `10000`), several `j0` (including asymmetric `p0=0.1` and `p0=0.9`), and random signed `v`, compare the single-block FFT result **at its own log-odds nodes** with a direct binomial sum restricted to **exactly the same input window**. This isolates factorization/FFT error from truncation and interpolation.
2. Include nonzero `ell_min` and nonzero `r_min`, odd/even lengths, and padding to exercise the convolution indexing. Assert no circular wraparound.
3. Check `log(a)`, `log(b)`, `B0` are finite over all constructed blocks. Test the tail/truncation bound or a conservative numerical tail estimate at the extreme grid points.

### B. Complete full and transient matvecs

Compare to `P` explicitly formed using stable log-PMFs for **small** `M`; use a chunked direct reference or independently written JAX banded reference for larger `M`.

Test `M = 16, 64, 256, 1024, 4096, 10000` (adjust slow cases in CI), and `s in {0, 1e-3, 0.05, 0.5, 0.95, 1.0}`. Include:

- `v = ones`, `v = zeros`, boundary-only basis vectors, central and boundary basis vectors, positive random and signed random vectors, and alternating-sign vectors.
- **Row-stochasticity** for full `P`: `P @ ones ~= ones`.
- `P` absorbing endpoints, no accidental renormalization of `Q`; `Q @ ones` equals the probability of staying transient.
- Absolute error scaled by `max(1, ||v||_infinity)` and max normalized error at a specified tolerance. Start with a target around `1e-9` in float64 at moderate `M`, and document actual achievable accuracy rather than hard-coding an unrealistic threshold.
- Accuracy at and immediately across every interior block seam, and at the switch between direct boundary and FFT interpolation.
- Linearity: `T(a*v+b*w) ~= a*T(v)+b*T(w)`.
- Adjoint test for `Q` using dot products, and a direct dense-transpose comparison for small `M`.
- JIT compile/run with `s` as a *dynamic JAX scalar*, and change `s` without recompiling.

For large `M` (e.g. `M=65536` or larger where feasible), avoid the dense reference. Use independently computed exact probabilities for sampled rows, covering the boundaries, the center, strong selection, and block seams.

### C. Fundamental-matrix solve

For small `M` and several `s`, form `Q` explicitly in the **test only**, and compare

\[
g_{\mathrm{reference}} = \operatorname{solve}(I-Q^T,e_1)
\]

against the Lineax solution. Verify: residual against the implemented operator, residual against the **exact** `Q` in small tests, nonnegativity of the occupation counts within numerical tolerances, and `sum(g)` versus the exact expected absorption time.

Exercise near-neutral cases (which may converge slowly) and strong underdominance. Detect/report nonconvergence; do not hide it through a permissive return status. Keep the linear-solve tolerance and the FFT operator-approximation tolerance as separate parameters.

### D. Benchmarks

Measure separately **plan construction**, **JIT compilation**, **steady-state forward matvec**, **steady-state transposed matvec**, memory use (as feasible), and **end-to-end Lineax solve**. Time after warmup with `.block_until_ready()`.

Compare, at minimum:

1. dense matvec (only small `M`),
2. JAX banded/truncated-binomial matvec,
3. local FFT matvec.

Benchmark across several `M` values spanning small to GPU-useful sizes and multiple `s`. Provide an accuracy-versus-runtime comparison as tolerances/interpolation orders change. Clearly state where FFT is **slower**, not just where it wins. If running on CPU only, explicitly say so and leave GPU benchmarks runnable as a script.

## 8. Suggested public API and deliverables

Adapt names to the repository, but aim for an API resembling:

```python
plan = make_wf_fft_plan(
    M=20000,
    tail_tol=1e-12,
    interp_order=8,
    min_mode_count=128,
    block_spacing=...,  # dimensionless multiple of local sigma
    dtype=jnp.float64,
)

Pv = apply_p_fft(plan, s, v_full)          # (M+1,)
Qv = apply_q_fft(plan, s, v_transient)     # (M-1,)
q_op = make_q_lineax_operator(plan, s)
result = solve_wf_occupation(plan, s, solver=..., ...)
# result includes g, solver status, residual, timings/diagnostics if available
```

Deliver:

1. The actual JAX implementation, Lineax wrapper, and a documented configuration object/plan.
2. Automated tests for factorization, truncation/interpolation, transpose, row probabilities, and the solve.
3. A reproducible benchmark script with direct/banded/FFT comparisons.
4. A short README or module docstring explaining limitations, numerical tolerance choices, and when to prefer the banded method.
5. A summary of files changed, commands run, achieved errors and timing results, and remaining failures or limitations.

**Implementation order:** (i) exact single-block FFT test; (ii) full multi-block forward matvec with boundary fallback; (iii) JIT/FFT bucketing and performance; (iv) transpose; (v) Lineax solve; (vi) accuracy/performance sweeps. Run the tests. If numerical error or performance is disappointing, diagnose and document it; don't paper over failures with clipping, renormalization, or unverified claims about asymptotic speed.

## References for API verification

- [Lineax: FunctionLinearOperator and transposes](https://docs.kidger.site/lineax/api/operators/)
- [Lineax: GMRES, BiCGStab, and preconditioner options](https://docs.kidger.site/lineax/api/solvers/)
- [Lineax: linear_solve](https://docs.kidger.site/lineax/api/linear_solve/)
- [JAX: linear_transpose](https://docs.jax.dev/en/latest/_autosummary/jax.linear_transpose.html)
- [JAX: custom_linear_solve](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html) (optional if implicit differentiation is later required)
