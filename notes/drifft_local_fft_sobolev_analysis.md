# Verified-source Sobolev, BV, and multi-step error analysis of `drifft.FFT`

*A technical note, 9 November 2026*

**Scope and source.** This note analyzes the **actual attached** `fft.py` (610 lines, received 9 November 2026), not a hypothesized implementation. All line numbers refer to that file. It also uses the accompanying NUFFT/DCT Sobolev note as a template for the multi-step Fisher-information argument. Analytic estimates are in exact arithmetic unless explicitly stated otherwise. The source audit covers `fft.py`; it does **not** establish the behavior of separately imported `_accuracy.py`, `_base.py`, or a particular installed JAX/XLA FFT implementation. Independent checks ported the documented geometry and exact-node mathematics to NumPy/SciPy; they did not execute the complete Equinox/HiJAX backend in this environment. The tests are *diagnostics*, not interval-arithmetic proofs.

**Executive findings.** The implementation realizes the specified exact diagonal–Toeplitz–diagonal factorization and moving Lagrange interpolation. It has no interpolation-error or roundoff certificate; `grad_atol` raises `NotImplementedError` (`fft.py:230–235`, `372–384`). Fixed stencils admit joint value/derivative Sobolev–Peano bounds. Its stencil switches are value-continuous but generally have derivative kinks; block, direct/interior, and direct-only reflection switches can have **value jumps**. Hence the unmodified continuous-probability operator is generally **BV, not globally** `W^{1,1}` or `C^1`. This does not mean that JAX AD is unusable: it computes derivatives of selected branch extensions almost everywhere. For rigorous global `W^{1,\infty}` control, first eliminate value jumps; smoothing every stencil kink is necessary only if pointwise classical `C^1` regularity is demanded. Independently, admissible small-cutoff configurations can suffer severe FFT cancellation even when all factors are finite.

## 1. Motivation, norm contract, and principal results

For destination copy count $M$, define the binomial row

$$
 b_M(q)=(b_{M,j}(q))_{j=0}^{M},\qquad
 b_{M,j}(q)=\binom Mj q^j(1-q)^{M-j}.
$$

Given source states $p_i$, the exact backward Wright–Fisher operator is

$$
 (P_s f)_i=\sum_{j=0}^{M} b_{M,j}(q_s(p_i))f_j.
$$

The observable $f$ may be any bounded, signed vector. For an approximate row $\widetilde b$, the relevant absolute norm is

$$
 \delta_s=\|P_s-\widetilde P_s\|_{\infty\to\infty}
 =\max_i\sum_j|b_{M,j}(q_s(p_i))-\widetilde b_j(q_s(p_i))|.
 \tag{1.1}
$$

The derivative target is a **separate** absolute operator error,

$$
 \eta_s=\|\partial_sP_s-\partial_s\widetilde P_s\|_{\infty\to\infty}.
 \tag{1.2}
$$

For a rectangular transition this also bounds the transpose forward action in $\ell_1$. If $\widetilde P_s$ has signed rows, its row-$\ell_1$ error is still the correct norm, but it is not automatically twice a total-variation distance.

**Main conclusions.**

1. Each fixed interpolation stencil admits a joint, computable $W^{1,\infty}$ error estimate in terms of a *single* $m$-th derivative bound on the binomial kernel. The derivative estimate need not invoke the $(m+1)$-st derivative.
2. Within-block moving-stencil joins are continuous (although generally not $C^1$), and therefore have a legitimate weak first derivative in $L^\infty$. General hard block/direct switches may have **value jumps**, so the assembled raw approximation is generally **BV but not $W^{1,1}$**, let alone $W^{1,\infty}$.
3. A uniform **branchwise automatic-differentiation (AD)** estimate follows from local Sobolev bounds, but this is not a global classical derivative theorem. Smooth switching gives a genuinely global $C^1$ family, at a quantifiable blending cost.
4. Once a globally differentiable fixed approximate matrix family has one-step errors $(\delta,\eta)$, the *same* pathwise Fisher-information argument as in the attached DCT note gives multi-step value and gradient bounds, even if the approximate matrix is signed.
5. Sobolev regularity alone gives **no numerical-stability certificate** for the diagonal–Toeplitz–diagonal FFT realization. The previous analysis identified potentially severe diagonal amplification. The planner must treat this independently.

### Notation crosswalk

| This note | Role / supplied implementation terminology |
|---|---|
| $M$ | Destination gene copies; row has $M+1$ entries |
| $c$ | `min_mode_count`; boundary/core threshold and first center |
| $\beta$ | `block_spacing`; determines block-center spacing |
| $m=2h$ | Number of moving interpolation nodes, $2\le m\le16$ in `fft.py` |
| $\tau$ | `tail_tol`; nominal omitted binomial mass |
| $j_b,p_b,V_b,\xi_b$ | Block count center, $p_b=j_b/M$, $V_b=Mp_b(1-p_b)$, $\xi_b=\operatorname{logit}p_b$ |
| $r$ | Local log-odds grid coordinate $V_b(\operatorname{logit}q-\xi_b)$ |
| $\vartheta$ | Angular probability coordinate $2\arcsin\sqrt q$ |
| $\Theta_*$ | Uniform bound on $|\partial_s\vartheta|$ across source states and parameter domain |
| $\delta,\eta$ | Absolute **one-step** operator and parameter-derivative errors |
| $K$ | Number of generations in a constant-size epoch |

## 2. The operator implemented in `fft.py`

The attached `fft.py` constructs centers $j_b$ in the interior, beginning at $c$ and ending at $M-c$, with successive integer spacing

$$
 j_{\mathrm{next}}=\min\{M-c,\,j+\max(1,\lfloor\beta\sqrt{j(1-j/M)}\rfloor)\}.
 \tag{2.1}
$$

Adjacent block cores meet at midpoints of their log-odds centers. Fix a block $b$. The exact row in local coordinates is

$$
 g_b(r)=b_M(q_b(r)),\quad
 q_b(r)=\operatorname{logistic}(\xi_b+r/V_b),\quad
 r=V_b(\operatorname{logit}q-\xi_b).
 \tag{2.2}
$$

For even $m=2h$, the moving stencil uses

$$
 a=\lfloor r\rfloor-h+1,\qquad t=r-a,\qquad
 (I_{b,a}g)(r)=\sum_{k=0}^{m-1}\lambda_k(t)g(a+k),
 \tag{2.3}
$$

$$
 \lambda_k(t)=\prod_{\ell\ne k}\frac{t-\ell}{k-\ell}.
$$

Whenever $r$ lies between neighboring integer knots, $t\in[h-1,h]$. This central evaluation range is critical: the $m$-point interpolation is not being extrapolated far outside the stencil.

Each block retains a fixed count window $[\ell_b,u_b]$ covering the halo of nodes. Its precomputed truncated node row is

$$
 y_{b,n}=\big(b_{M,j}(q_b(n))\mathbf1_{\ell_b\le j\le u_b}\big)_{j=0}^{M}.
 \tag{2.4}
$$

The FFT computes the action of these rows on $f$; the interpolation is on the **completed, scaled node expectation**, not on an unscaled Gaussian surrogate. Outside interior cores, the implemented algorithm uses directly truncated binomial rows, reflected in the upper tail.

### 2.1 Actual source mechanics and their mathematical consequences

The following is a source audit, not a list of proposed features.

| Code location | Implemented operation | Consequence for a certificate |
|---|---|---|
| `FFT.__init__`, lines 185–257 | `M=2*pop_size`, `atol`/`tail_tol` are validated; `m` is even, 2–16; `c>=8`, `0<beta<=4`, and JAX x64 required | The admissible planner search is discrete and bounded in interpolation order; x64 is necessary but insufficient for error control |
| Lines 230–235, 372–384 | `grad_atol` raises `NotImplementedError`; `error_report` places `None` for interpolation and roundoff | Neither requested `atol` nor `grad_atol` currently certifies the operator norm; `atol` only constrains the nominal mass budget |
| Lines 258–311 | Centers, log-odds midpoint cores, integer halos, and SciPy `binom.ppf`/`isf` tail windows | Our local-grid and tail theorems match actual host geometry; returned quantiles must still be checked under numerical error |
| `_build_block`, lines 74–119 | Float64 recurrence for the input diagonal, modal Stirling correction, centered log-factor shift, FFT of Gaussian-shaped kernel | Exact DTD identity; numerical factor construction, extreme scaling, and FFT error must be bounded separately |
| Lines 312–341, 464–476 | Bucketed zero-padded real FFTs; padded outputs accumulate at a separate sentinel index | Padding ensures correct convolution indices in exact arithmetic; per-bucket padding affects cost and numerical implementation |
| Lines 342–370, 430–462, 596–610 | Reflected direct binomial tails, a batch-size-64 `lax.map`, and a `HiPrim` binomial-weight evaluator | Boundary rows are direct **truncated**, not normalized; batching is an execution choice, not a mathematical approximation |
| Lines 477–504 | Inactive probabilities clipped for safe interpolation; `searchsorted(..., side="right")`, `floor`, Lagrange product weights, final `where(is_boundary, direct, interpolated)` | Hard switches are real; block ties choose the block on the right, boundary ties choose the direct result; derivative is branchwise |
| `_BinomialWeights.jvp`, lines 548–580 | Analytic identity $b'_{n,j}=n(b_{n-1,j-1}-b_{n-1,j})$, including endpoint rows | The direct branch has a mathematically correct polynomial derivative for fixed prepared coefficients; this does not remove reflection or branch-selection jumps |

For a given `q`, the program computes the direct branch first when needed. Its `safe_probabilities=clip(q,cutoff,1-cutoff)` is used only to avoid invalid **unused** interior coordinates; the final `where` selects the direct value at and beyond the cutoff (`fft.py:477–504`). The `floor`, `searchsorted`, conditional reflection at `q>0.5`, and integer quantile choices are not differentiated as smooth operations. Prepared geometry is independent of the selection parameter.

**Not directly checked here:** the exact contents of `AbstractTransitionBackend` or `BackendErrorReport`; GPU FFT behavior, higher-order JAX transforms of `HiPrim`, and the runtime's complex FFT normalization. The source visibly specifies the intended formula and linearized primitive, but those implementation details require executable integration tests.

### 2.2 Why the FFT factorization is exact in exact arithmetic

Let $h_b=V_b^{-1}$ and write $p=p_b$, $j_c=j_b$. Define

$$
 A_n=\exp\left[j_ch_bn-M\log(1-p+pe^{h_bn})+\frac{h_bn^2}{2}\right],
$$

$$
 D_j=b_{M,j}(p)\exp\left[\frac{h_b(j-j_c)^2}{2}\right],
 \quad G_{n,j}=\exp\left[-\frac{h_b(n-(j-j_c))^2}{2}\right].
$$

Expanding the squares gives

$$
 A_nG_{n,j}D_j
 =b_{M,j}(p)\frac{e^{j h_bn}}{(1-p+pe^{h_bn})^M}
 =b_{M,j}(q_b(n)).\tag{2.5}
$$

Thus, **before floating-point arithmetic**, the local matrix is a diagonal–Toeplitz–diagonal factorization of an exact binomial kernel. This identity is useful for fast action, but it does *not* imply that the factorized evaluation is well-conditioned.

## 3. A binomial-specific Sobolev energy bound

Our Banach target space is $X=\ell_1^{M+1}$. On a stencil hull $I\subset\mathbb R$, define

$$
 \|g\|_{W^{m,\infty}(I;X)}
 =\max_{0\le k\le m}\operatorname*{ess\,sup}_{r\in I}\|g^{(k)}(r)\|_1.
 \tag{3.1}
$$

The derivative relevant for interpolation is with respect to **local $r$**, not $s$. Write $v=q(1-q)$ and $D=v\partial_q=\partial_{\operatorname{logit}q}$. Then

$$
 g_b^{(n)}(r)=V_b^{-n}D^n b_M(q_b(r)).
 \tag{3.2}
$$

### 3.1 Exact orthogonality and the derivative energy $H_n$

Define the coefficient polynomials $C_{n,k}(q)$ recursively by $C_{1,1}=1$, $C_{n,k}=0$ outside $1\le k\le n$, and

$$
 C_{n+1,k}
 =k(1-2q)C_{n,k}+vC_{n,k}'+C_{n,k-1}.
 \tag{3.3}
$$

Induction using the product rule gives

$$
 D^n b_M(q)=\sum_{k=1}^{\min(n,M)}v^k C_{n,k}(q)\,\partial_q^k b_M(q).
 \tag{3.4}
$$

For $q\in(0,1)$, the generating identity

$$
 \sum_{j=0}^{M}\frac{b_{M,j}(q+x)b_{M,j}(q+y)}{b_{M,j}(q)}
 =\left(1+\frac{xy}{v}\right)^M
 \tag{3.5}
$$

follows by the binomial theorem. Equating Taylor coefficients yields

$$
 \sum_{j=0}^{M}\frac{\partial_q^k b_{M,j}(q)\partial_q^\ell b_{M,j}(q)}{b_{M,j}(q)}
 =\mathbf1_{k=\ell}\frac{k!(M)_{\underline k}}{v^k}.
 \tag{3.6}
$$

The convention $(M)_{\underline k}=0$ for $k>M$ handles high orders. Define

$$
 H_n(M,q)=\sum_{k=1}^{\min(n,M)}k!(M)_{\underline k}v^k C_{n,k}(q)^2.
 \tag{3.7}
$$

By (3.4)–(3.6),

$$
 \sum_j\frac{[D^n b_{M,j}(q)]^2}{b_{M,j}(q)}=H_n(M,q).
$$

Cauchy–Schwarz with weights $b_{M,j}$ gives the **whole-row** bound

$$
 \boxed{\|D^n b_M(q)\|_1\le\sqrt{H_n(M,q)}.}\tag{3.8}
$$

This is a true $\ell_1$ statement, applying to all signed bounded observables. It introduces no artificial factor of $M+1$ from summing separate coordinatewise worst cases. For orientation,

$$
 H_1=Mv,\qquad H_2=Mv+2M(M-3)v^2.
 \tag{3.9}
$$

Every $H_n$ is symmetric under $q\mapsto1-q$ and can be represented as a polynomial in $v=q(1-q)$ of degree at most $n$. Thus, for $m\le16$, the required maxima are low-degree univariate polynomial problems. A certified planner may convert the polynomial to the Bernstein basis on a rational interval in $v$ and enclose its range with interval subdivisions and outward rounding; a fine sampling grid alone is not a proof.

For any block halo interval $I_b$ define

$$
 \mathcal D_{n,b}
 =V_b^{-n}\sqrt{\sup_{r\in I_b}H_n(M,q_b(r))}.
 \tag{3.10}
$$

Then $\|g_b^{(n)}\|_{L^\infty(I_b;X)}\le\mathcal D_{n,b}$.

**Important limitation.** The displayed $H_n$ is an upper bound on the $\ell_1$ derivative norm through Cauchy–Schwarz, not an identity for that norm. The Gram identity in the earlier local-FFT analysis can yield tighter certificates, but it must be used with its separate reference-distribution and numerical-evaluation requirements.

## 4. Sobolev–Peano bounds for value *and* derivative

Fix a full $m$-node stencil $a,\ldots,a+m-1$. Define the error functionals

$$
 L_{0,t}(g)=g(a+t)-\sum_k\lambda_k(t)g(a+k),
$$

$$
 L_{1,t}(g)=g'(a+t)-\sum_k\lambda_k'(t)g(a+k).
 \tag{4.1}
$$

Both functionals annihilate every polynomial of degree $<m$. The vector-valued Peano-kernel theorem (obtainable by integrating Taylor's formula with integral remainder and applying the error functional) gives

$$
 L_{\nu,t}(g)=\int_0^{m-1}K_{m,\nu}(t,u)g^{(m)}(a+u)\,du,\quad\nu\in\{0,1\},
 \tag{4.2}
$$

where, for $m\ge2$,

$$
 K_{m,\nu}(t,u)=\frac{
 \partial_t^\nu(t-u)_+^{m-1}
 -\sum_{k=0}^{m-1}\lambda_k^{(\nu)}(t)(k-u)_+^{m-1}}
 {(m-1)!}.
 \tag{4.3}
$$

Values at the isolated knots $t=u$ do not affect the integral. Define the dimensionless finite constants

$$
 C_{m,\nu}
 =\sup_{h-1\le t\le h}\int_0^{m-1}|K_{m,\nu}(t,u)|\,du.
 \tag{4.4}
$$

Taking the Bochner-integral norm in $X$ proves

$$
 \boxed{
 \begin{aligned}
 \|g_b-I_{b,a}g_b\|_{L^\infty([a+h-1,a+h];X)}
 &\le C_{m,0}\mathcal D_{m,b},\\
 \|g_b'-(I_{b,a}g_b)'\|_{L^\infty([a+h-1,a+h];X)}
 &\le C_{m,1}\mathcal D_{m,b}.
 \end{aligned}}\tag{4.5}
$$

These use the **same** $m$-th derivative. In particular, we do **not** obtain the second inequality by differentiating an unspecified scalar Lagrange remainder point.

The usual product remainder, or direct Peano-kernel integration, gives

$$
 C_{m,0}=\frac{1}{m!}\max_{h-1\le t\le h}\left|\prod_{k=0}^{m-1}(t-k)\right|
 =\frac{\binom{2h}{h}}{4^{2h}}.
 \tag{4.6}
$$

For $m=2$, one obtains $C_{2,0}=1/8$ and $C_{2,1}=1/2$.

A useful **rigorous but possibly conservative fallback** for the derivative constant follows by Taylor-expanding each nodal value about the query $t$ to degree $m-1$ and using that $L_{1,t}$ annihilates this polynomial:

$$
 C_{m,1}\le\overline C_{m,1}:=
 \frac{1}{m!}\sup_{h-1\le t\le h}
 \sum_{k=0}^{m-1}|\lambda_k'(t)|\,|k-t|^m.
 \tag{4.6a}
$$

This is a one-dimensional piecewise-polynomial maximization, readily certified by rational polynomial interval arithmetic. It does **not** use $H_{m+1}$. A tighter $C_{m,1}$ results from the exact Peano integral in (4.4); the planner should use the smaller of all **certified** derivative constants. A numerical maximizer alone is not a rigorous certificate.

For larger even $m$, **do not insert an unproved closed-form value for $C_{m,1}$**. Equation (4.4) is a reproducible finite computation: for fixed $t$, split the integral at integer nodes and at $u=t$, integrate the absolute values of piecewise polynomials using verified root isolations, then maximize over $t\in[h-1,h]$ by interval subdivision. Precomputing these constants once for $m\in\{2,4,\ldots,16\}$ is inexpensive relative to an FFT application. The constants depend on $m$, not on $M$ or block geometry.

### 4.1 A useful comparison against the earlier derivative certificate

The alternative, already derived from the differentiated divided-difference integral, is

$$
 \|g_b'-(I_{b,a}g_b)'\|_1
 \le \gamma_m\mathcal D_{m,b}
 +\frac{\alpha_m}{m+1}\mathcal D_{m+1,b},
 \tag{4.7}
$$

where $\alpha_m=C_{m,0}$ and $\gamma_m=1/[h\binom{2h}{h}]$. The first-derivative certificate can therefore take the **minimum** of (4.5) and (4.7), provided both derivative-energy intervals have been certified. The Peano route is especially attractive when $H_{m+1}$ grows rapidly; the two bounds need not have a fixed ranking for every configuration.

### 4.2 Tail truncation and signed interpolation weights

Suppose the truncated node row $y_{b,n}$ satisfies

$$
 \|y_{b,n}-g_b(n)\|_1\le T_b
 \quad\text{for every stencil node in the halo}.
 \tag{4.8}
$$

Define the computable interpolation Lebesgue constants

$$
 \Lambda_m=\sup_{h-1\le t\le h}\sum_{k=0}^{m-1}|\lambda_k(t)|,
 \qquad
 \Lambda_m^{(1)}=\sup_{h-1\le t\le h}\sum_{k=0}^{m-1}|\lambda_k'(t)|.
 \tag{4.9}
$$

The local, exact-arithmetic *value* and local-$r$ *derivative* errors obey

$$
 \boxed{e_{0,b}=C_{m,0}\mathcal D_{m,b}+\Lambda_mT_b,}
 \tag{4.10}
$$

$$
 \boxed{e_{1,b}=C_{m,1}\mathcal D_{m,b}+\Lambda_m^{(1)}T_b.}
 \tag{4.11}
$$

Here ``error'' means a worst-case norm bound. The derivative truncation term is from the **signed derivatives of interpolation weights**; the prepared node values are constant in the differentiation of the selected interpolation formula.

A sufficient, explicit tail-quantile convention chooses $\ell_b,u_b$ so that, at the **leftmost and rightmost halo probabilities** $q_-,q_+$,

$$
 \Pr_{q_-}(J<\ell_b)\le\tau/2,
 \qquad
 \Pr_{q_+}(J>u_b)\le\tau/2.
 \tag{4.12}
$$

Stochastic monotonicity of the binomial family proves that these two inequalities apply to the respective tails at every intermediate $q$; hence $T_b\le\tau$. This is how `fft.py:293–305` chooses and checks them: `lo=binom.ppf(tau/2,M,pmin)`, `hi=binom.isf(tau/2,M,pmax)`, followed by a maximum of the **total** omitted mass at both halo endpoints. The resulting `max_tail_mass` field (`fft.py:360–361`) is diagnostic; it is **not** included as a proved interpolation or roundoff bound in `error_report`. Since the retained set is contiguous, an alternative certificate can check *total* omitted mass at both endpoints: retained mass is unimodal as a function of $q$, so omitted mass has its maximum on the interval at an endpoint. Either convention requires actual tail checks, not just trusting a numerical quantile API's endpoint behavior.

### 4.3 Conversion to a model-parameter derivative

Assume a supplied uniform angular sensitivity

$$
 \Theta_*\ge\sup_{s,i}
 \frac{|\partial_sq_s(p_i)|}{\sqrt{q_s(p_i)(1-q_s(p_i))}},
 \tag{4.13}
$$

on interior rows, with zero sensitivity for parameter-independent absorbing rows. Using

$$
 \partial_s r_b=\frac{V_b}{q(1-q)}\partial_s q,
$$

we get

$$
 G_b:=\sup_{s,i\,\text{assigned to }b}|\partial_s r_b|
 \le \Theta_*\frac{V_b}{\sqrt{\min_{q\in Q_b}q(1-q)}}.
 \tag{4.14}
$$

The set $Q_b$ must include all queried probabilities in the relevant core or blend extension. Thus the branchwise absolute parameter-derivative error is bounded by $G_be_{1,b}$. In the angular coordinate $\vartheta=2\arcsin\sqrt q$, this conversion is equivalently $\partial_\vartheta r_b=V_b/\sqrt{q(1-q)}$. Using the local Sobolev norm without this conversion would not certify (1.2).

## 5. Direct regions and uniform one-step assembly

For $q\le q_c=\min(c/M,1/2)$, the implementation retains $j=0,\ldots,W-1$, with `W=min(M+1,binom.isf(tau,M,q_c)+1)` (`fft.py:342–350`). If

$$
 \Pr_{q_c}\{J\ge W\}\le T_D,
 \tag{5.1}
$$

then monotonicity gives a direct value error at most $T_D$ throughout the lower direct region. Reflection yields the upper-region bound.

To bound derivatives independently of values, let $Z_\vartheta=(J-Mq)/\sqrt{q(1-q)}$. Since $\mathbb E Z_\vartheta^2=M$, Cauchy–Schwarz implies

$$
 \sum_{j\ge W}|\partial_\vartheta b_{M,j}(q)|
 =\mathbb E[\mathbf1_{J\ge W}|Z_\vartheta|]
 \le\sqrt{M\Pr_q(J\ge W)}.
 \tag{5.2}
$$

Therefore

$$
 \boxed{\eta_D\le\Theta_*\sqrt{MT_D}.}\tag{5.3}
$$

A sharper nonnegative-score formula is available if $W\ge\lceil Mq_c\rceil$ (with $1\le W\le M$):

$$
 \sum_{j\ge W}|\partial_q b_{M,j}(q)|
 =M b_{M-1,W-1}(q),
 \quad 0\le q\le q_c.
 \tag{5.4}
$$

Consequently a valid angular bound is

$$
 \eta_D\le\Theta_* M\binom{M-1}{W-1}
 \max_{0\le q\le q_c}q^{W-1/2}(1-q)^{M-W+1/2},
 \tag{5.5}
$$

with the maximizer $q_*=\min(q_c,(W-1/2)/M)$ when $W\ge1$, and zero error for $W=M+1$. At $q=0$ the boundary derivative of the full binomial family is $M(e_1-e_0)$: retaining only the endpoint mass may preserve its **value** but not its derivative. Support-changing parameter endpoints require a separate treatment.

The custom `_BinomialWeights` primitive explicitly implements the polynomial derivative $M(b_{M-1,j-1}-b_{M-1,j})$ rather than differentiating $\log q$ at $q=0$ or $1$ (`fft.py:507–580`). It therefore addresses an **endpoint differentiation** problem that would otherwise be easy to confuse with switching irregularity. In the direct-only case (`M\le2c`), the reflection condition `p>0.5` in `_expect_boundary` selects the lower reflected branch exactly at $q=1/2$ (`fft.py:600–608`); an incomplete window can still cause a value jump there. Since the admissible $c$ is at least eight and the selected width is at least two, the implemented direct window preserves the first probability derivative at $q=0$ and $q=1$ in exact arithmetic, although its derivative elsewhere is truncated.


If the query rule chooses exactly one block or direct branch, the *value* certificate is

$$
 \boxed{
 \delta\le\max\{T_D,\max_b e_{0,b}\}.}
 \tag{5.6}
$$

On parameter intervals where the selected branch is fixed, the derivative bound is

$$
 \boxed{
 \eta_{\rm branch}\le\max\{\eta_D,\max_bG_be_{1,b}\}.}
 \tag{5.7}
$$

Both are exact-arithmetic analytic bounds for the implemented truncated-node operator in exact arithmetic. Neither covers FFT roundoff, and (5.7) must not be read as a global classical derivative theorem at switching points.

## 6. Global regularity: $W^{1,\infty}$ locally, BV globally

This is the principal difference from the smooth global DCT/NUFFT approximation in the attached note.

### 6.1 Within-block moving-stencil switches

At an integer query coordinate, neighboring stencils share the query node and therefore agree in **value**. Each stencil polynomial has uniformly bounded derivative on its central interval, so the assembled within-block interpolant is locally Lipschitz. It belongs to $W^{1,\infty}$ across these knots, although its classical derivative generally jumps. The weak derivative is the piecewise polynomial derivative almost everywhere.

For $m=2h$, the jump in the derivative with respect to $r$ is a constant multiple of the $2h$-th finite difference of the adjacent nodes:

$$
 P_+'(n)-P_-'(n)
 =\frac{(-1)^{h-1}[(h-1)!]^2}{(2h-1)!}\Delta^{2h}y_{n-h}.
 \tag{6.1}
$$

It is generally nonzero. This prohibits a universal $C^1$ statement for the raw moving-stencil scheme but **does not invalidate a weak first-derivative bound**.

### 6.2 Block, direct/interpolated, and reflection switches

Independently computed blocks generally have different interpolating polynomials at their common boundary, even if each is approximating the same true row. Direct truncated and interpolated rows can also have different retained supports. In direct-only reflected evaluation, the two truncated supports can disagree at the reflection midpoint. These can introduce **value jumps**.

A finite jump at an interior point prevents membership in $W^{1,1}$ on an interval containing that point, since one-dimensional $W^{1,1}$ functions have absolutely continuous representatives. In particular, an arbitrarily small but nonzero jump is still an obstruction to a global $W^{1,\infty}$ guarantee. The implemented tie-breaking rule chooses a value at the switch; it does not remove the jump.

### 6.3 The mathematically correct global object is BV

Fix a scalar angular parameter $\vartheta$ and let

$$
 e(\vartheta)=\widetilde b_M(\vartheta)-b_M(\vartheta).
$$

For finitely many switch points $\vartheta_i$, piecewise smoothness yields the vector-valued distributional derivative

$$
 \boxed{
 De=e'_{\rm pw}(\vartheta)\,d\vartheta
 +\sum_iJ_i\,\delta_{\vartheta_i},
 \qquad J_i=e(\vartheta_i^+)-e(\vartheta_i^-).}
 \tag{6.2}
$$

The total variation satisfies

$$
 |De|([0,\pi])
 =\int_0^\pi\|e'_{\rm pw}(\vartheta)\|_1\,d\vartheta
 +\sum_i\|J_i\|_1.
 \tag{6.3}
$$

At a switch from branches with row-value bounds $e_{0,-},e_{0,+}$,

$$
 \|J_i\|_1\le e_{0,-}+e_{0,+}.
 \tag{6.4}
$$

Thus $\|e\|_{L^\infty}+|De|$ gives a **unified, meaningful error certificate for the unsmoothed algorithm**: the absolutely continuous part measures the ordinary derivative error, while the atomic part explicitly records switching defects. Unlike a branchwise gradient norm, this certificate does not hide value discontinuities. Beware that BV in $\vartheta$ does not automatically give the same total variation after composing with a highly oscillatory map $s\mapsto\vartheta(s)$; the number of boundary crossings must also be controlled.

### 6.4 Repairing switches with a $C^1$ blend

Extend two neighboring fixed branch formulas $F_-,F_+$ into a short shared angular overlap $[\vartheta_0-\Delta/2,\vartheta_0+\Delta/2]$, chosen so both extensions are defined and their derivative certificates remain valid. Put $u=(\vartheta-\vartheta_0+\Delta/2)/\Delta$, and set

$$
 w(u)=3u^2-2u^3,\qquad
 F=(1-w)F_-+wF_+.
 \tag{6.5}
$$

Because $w'(0)=w'(1)=0$, the patched approximation joins with a continuous first derivative when it matches the unblended branch at the edges. This includes stencil joins if those joins are blended as well; smoothing only inter-block joins would still leave internal derivative kinks and would prove $W^{1,\infty}$ but not necessarily $C^1$.

Let $E_\pm$ bound the row errors of both extended formulas and let $D_\pm$ bound their angular-derivative errors. Since $0\le w\le1$,

$$
 \|F-b_M\|_1\le\max(E_-,E_+).
 \tag{6.6}
$$

Differentiating (6.5), and using $\max_{u\in[0,1]}|w'(u)|=3/2$, gives

$$
 \boxed{
 \|F'-b_M'\|_1
 \le\max(D_-,D_+)+\frac{3}{2\Delta}
 \sup_{\rm overlap}\|F_+-F_-\|_1.}
 \tag{6.7}
$$

This last term is essential. A narrower overlap may **increase** the derivative bound if the original boundary has a value jump. Conservatively $\sup\|F_+-F_-\|_1\le E_++E_-$; directly certifying the difference can be sharper. After multiplying by $\Theta_*$, this is a global parameter-derivative certificate. A fully patched, $C^1$ fixed operator family supports both classical derivative estimates and the multi-generation theorem below.

**Alternative:** If a weak derivative is sufficient and the operator values are continuous, $W^{1,\infty}$ may suffice without smoothing derivative kinks. Value jumps cannot be handled this way.

### 6.5 A cheaper repair sufficient for Sobolev and almost-everywhere gradient guarantees

If the scientific contract is a **weak** first derivative (or an almost-everywhere AD result), it is unnecessary to smooth the many moving-stencil derivative kinks. One may instead blend only across interfaces that have **value jumps**: block-to-block, direct-to-interpolation, and (when no interior blocks exist and the window is incomplete) reflection at $q=1/2$. Choose disjoint angular overlaps; use the cubic smoothstep in (6.5); and keep the original moving-stencil interpolants outside those overlaps. As a function of $\vartheta$, the resulting row is continuous and piecewise $C^1$ with bounded slopes, hence lies in $W^{1,\infty}([0,\pi];\ell_1^{M+1})$. It still need not be $C^1$ at moving-stencil knots.

The weak derivative obeys (6.7) in the blends and (4.11)/(5.7) elsewhere, hence gives a **global essential-supremum gradient guarantee** for parameter maps that are sufficiently regular under composition. This is a less invasive implementation target than smoothing every stencil knot. Strict pointwise classical differentiability at *every* parameter value requires the full $C^1$ repair.

### 6.6 New: propagating jump errors through powers without pretending they are derivatives

The raw backend can be analyzed **without repairing it** using BV. Let $Q_-$ and $Q_+$ be the exact-arithmetic matrix limits across a single parameter switch $s_0$, and put $J=Q_+-Q_-$. Suppose each side is within $\delta$ of the continuous stochastic $P_{s_0}$. Then $\|J\|\le 2\delta$, and $\|Q_\pm\|\le R=1+\delta$. The noncommutative telescoping identity gives

$$
 Q_+^K-Q_-^K
 =\sum_{t=0}^{K-1}Q_-^t J Q_+^{K-1-t},
$$

so the **jump atom of the $K$-step error** is bounded by

$$
 \boxed{\|J^{[K]}\|:=\|Q_+^K-Q_-^K\|\le K R^{K-1}\|J\|\le 2K R^{K-1}\delta.}
 \tag{6.8}
$$

This holds for signed $Q_\pm$. At every regular parameter point the multi-step gradient bounds in §8 apply to the differentiable branch extension; at switches, the distributional derivative of the error has atomic mass $J^{[K]}\delta_{s_0}$. Summing (6.8) over a **finite** set of crossings gives a global BV estimate, provided the map $s\mapsto Q_s$ has finite variation. If the map crosses a switch arbitrarily many times, a finite uniform value bound alone does **not** bound that sum. These jump atoms are absent from an ordinary AD gradient and cannot be hidden in $\eta$.

This gives a unified *value + smooth-gradient + jump* calculus for the unmodified implementation and an explicit route to the full Sobolev calculus after value-jump repair.

## 7. One-step information geometry and a natural scaled Sobolev norm

For $J\sim\operatorname{Bin}(M,q)$ and $\vartheta=2\arcsin\sqrt q$, the score is

$$
 Z_\vartheta(J)=\partial_\vartheta\log b_{M,J}(q)
 =\frac{J-Mq}{\sqrt{q(1-q)}}.
 \tag{7.1}
$$

It has mean zero and variance $M$. For every $\|f\|_\infty\le1$, Bessel's inequality applied to the orthonormal pair $1,Z_\vartheta/\sqrt M$ proves

$$
 \boxed{
 |b_M(q)f|^2+\frac1M|\partial_\vartheta[b_M(q)f]|^2\le1.}
 \tag{7.2}
$$

Thus $\|\partial_\vartheta P\|\le\sqrt M$ and $\|\partial_sP_s\|\le A_s:=\sqrt M\Theta_s$. It is conceptually natural to define a scaled Sobolev operator norm such as

$$
 \|E\|_{\mathcal W^1_M}
 =\|E\|_{\infty\to\infty}
 +M^{-1/2}\|\partial_\vartheta E\|_{\infty\to\infty}.
 \tag{7.3}
$$

This expresses the intrinsic information scale, but the required planning tolerances still apply to the **unscaled** $\delta$ and $\eta$. The local log-odds grid derivative transforms through (4.14), rather than having an automatically uniform angular mesh.

## 8. Multi-generation bounds for a smooth fixed approximate operator

Now consider a fixed square epoch, $M_s=M$, with a globally differentiable approximate matrix family $Q_s$ (for instance the $C^1$-repaired operator), and assume

$$
 \|Q_s-P_s\|\le\delta,\qquad
 \|Q_s'-P_s'\|\le\eta,\qquad
 \|P_s\|=1,
 \quad\|Q_s\|\le R:=1+\delta.
 \tag{8.1}
$$

All norms in this section are $\infty\to\infty$. $Q_s$ may have negative entries, and the parameter domain and codec layout are fixed. We also require locally fixed support of all exact transition rows; parameter-independent absorbing rows are allowed. These are the same substantive assumptions as in the attached NUFFT note's multi-step theorem.

### 8.1 Value error

A telescoping expansion, or the ordered expansion of $(P_s+(Q_s-P_s))^K$, gives

$$
 \boxed{
 \|P_s^K-Q_s^K\|\le R^K-1.
 }\tag{8.2}
$$

For $K\delta\ll1$, this is approximately $K\delta$. This theorem does not assume the approximate matrix is stochastic, so replacing $R$ by $1$ would require an additional norm or positivity certificate.

**Application to the current hard-switched backend:** On an open parameter interval with fixed active branches and stencils, $Q_s$ is differentiable and the derivative theorems below apply to that branch. At a stencil kink, only the selected branch derivative (or a weak derivative almost everywhere) is available. At a value jump, no global classical derivative exists. If only the value-jump interfaces are repaired as in §6.5, the theorem holds for the weak derivative almost everywhere in $s$. For the raw backend, add the BV jump atoms from §6.6 if global distributional control is desired.


### 8.2 Fisher information of an exact path

Let $X_0,\dots,X_n$ be a path of the **exact** Markov chain from a fixed initial state. Its one-step log-likelihood score increment is

$$
 \xi_t=\partial_s\log P_s(X_t,X_{t+1}).
$$

Conditional on the history through time $t$, it has mean zero and variance

$$
 I_s(X_t)=M(\partial_s\vartheta_s(p_{X_t}))^2\le A_s^2.
$$

The $\xi_t$ are martingale differences, so cross-covariances vanish. For the accumulated path score $S_n=\sum_{t<n}\xi_t$,

$$
 \mathbb E_i S_n^2
 =\sum_{t=0}^{n-1}(P_s^t I_s)_i\le nA_s^2.
 \tag{8.3}
$$

By differentiating terminal expectations and applying Cauchy–Schwarz,

$$
 \boxed{\|(P_s^n)'\|\le A_s\sqrt n.}\tag{8.4}
$$

This is the source of the improved multi-step derivative scaling; it does not rely on mixing, reversibility, stationarity, or a diffusion approximation.

### 8.3 Derivative error theorem

Write $E_s=Q_s-P_s$. Expand $Q_s^K-P_s^K$ as noncommutative words containing at least one $E_s$. When a derivative hits an error factor, summing the contributions gives $K\eta R^{K-1}$. When it hits an exact block of $n$ consecutive $P_s$ factors, use (8.4), not $n\|P_s'\|$. Counting all words gives

$$
 \boxed{
 \|(P_s^K)'-(Q_s^K)'\|
 \le K\eta R^{K-1}+A_sU_K(\delta),}
 \tag{8.5}
$$

where $U_1=0$ and for $K\ge2$,

$$
 \boxed{
 U_K(\delta)=\delta\sum_{n=1}^{K-1}\sqrt n
 \left[2R^{K-n-1}+(K-n-1)\delta R^{K-n-2}\right].}
 \tag{8.6}
$$

For the terminal $n=K-1$, interpret the second summand as zero. For clarity, the combinatorial identity used is

$$
 \sum_{\ell=0}^N(\ell+2)\binom N\ell\delta^\ell
 =2R^N+N\delta R^{N-1}.
$$

This proves (8.5) for **any** fixed differentiable approximate operator obeying (8.1), not just the filtered DCT operator of the attached note. The proof requires genuine differentiability of $Q_s$; it cannot be applied across hard value jumps by interpreting branchwise AD as the derivative of a continuous family.

A convenient, looser bound (obtained by Cauchy–Schwarz over words) is

$$
 \boxed{
 \|(P_s^K)'-(Q_s^K)'\|
 \le K R^{K-1}
 \big[\eta+\sqrt{2(K-1)}A_s\delta\big].}
 \tag{8.7}
$$

For small $K\delta$, the sharper finite sum has leading behavior

$$
 K\eta+\frac43 A_s\delta K^{3/2}
 \quad\text{(up to lower-order terms)}.
 \tag{8.8}
$$

A deterministic product-rule estimate would contribute at order $A_s\delta K^2$. Thus the probabilistic path-score argument supplies a meaningful improvement for long epochs even though the local interpolation approximation itself is not stochastic.

### 8.4 Rectangular demographies and varying transitions

For a compatible sequence of stochastic rectangular kernels $P_t$ and approximations $Q_t$, with errors $\delta_t,\eta_t$ and exact angular information scales $A_t$, define $R_*=\prod_{t=1}^K(1+\delta_t)$. Then

$$
 \left\|\prod_{t=1}^KP_t-\prod_{t=1}^KQ_t\right\|\le R_*-1.
 \tag{8.9}
$$

A valid gradient bound follows by the same path-score and subset expansion as Appendix A of the supplied NUFFT note. Explicitly, with $r_t=1+\delta_t$,

$$
 \left\|\partial_s\prod_tP_t-\partial_s\prod_tQ_t\right\|
 \le \sum_t\eta_t\frac{R_*}{r_t}+\sqrt{(R_*-1)W_*},
 \tag{8.10}
$$

$$
 W_*=\sum_t A_t^2\left\{
 \frac{R_*}{r_t}\left[1+\sum_{\ell\ne t}\frac{\delta_\ell}{r_\ell}\right]-1\right\}.
 \tag{8.11}
$$

This applies to common scalar-parameter derivatives or a fixed direction in parameter space. It does not mean that a rectangular matrix is being raised to a power.

## 9. Numerical FFT error is a separate proof obligation

The analytic inequalities (4.10)–(5.7) concern **exact arithmetic on the specified truncated binomial nodes**. To use them with an FFT, one must bound the node computation errors. Write

$$
 \widehat y_{b,n}=y_{b,n}+z_{b,n},\qquad
 \|z_{b,n}\|_1\le\varepsilon_{\rm node,b}.
$$

Then the analytic error bounds become, at minimum,

$$
 e_{0,b}^{\rm total}
 \le C_{m,0}\mathcal D_{m,b}
 +\Lambda_m(T_b+\varepsilon_{\rm node,b})
 +\varepsilon_{\rm weights,0,b}
 +\varepsilon_{\rm coord,0,b},
 \tag{9.1}
$$

$$
 e_{1,b}^{\rm total}
 \le C_{m,1}\mathcal D_{m,b}
 +\Lambda_m^{(1)}(T_b+\varepsilon_{\rm node,b})
 +\varepsilon_{\rm weights,1,b}
 +\varepsilon_{\rm coord,1,b}.
 \tag{9.2}
$$

Here the last errors must themselves be *proved* for the floating-point implementation; naming them is not a numerical certificate. Equations (9.1)–(9.2) are an **error-accounting template** rather than automatically certified bounds for a JAX backend.

### Numerical realization used by `fft.py`

`_build_block` builds the column factor by a log recurrence, normalizes by a Stirling-corrected modal PMF, and balances the factors by an opposite constant shift (`fft.py:80–103`). The zero-padded kernel FFT is computed with **NumPy** during host preparation (`fft.py:104–119`). Application uses `jnp.fft.rfft`, complex multiplication by the cached spectrum, and `jnp.fft.irfft` (`fft.py:465–476`). Buckets group blocks with identical FFT length and zero-pad their input/output arrays to the maximum length **within each bucket** (`fft.py:312–341`). Actual costs and temporary memory should therefore be computed from the bucket layout, not idealized as one unpadded FFT per block.

The `balance` transformation makes no difference to the exact binomial output and does not change the scale-invariant cancellation measure below. `np.all(np.isfinite(a/b))` detects overflow and NaNs, **not loss of significant digits**. There is no per-block a posteriori residual or alternate direct path when the FFT becomes inaccurate; users must impose their own numerical acceptance tests or change the backend.

### 9.1 Diagonal amplification

For a block with $R_b$ output nodes and $L_b=u_b-\ell_b+1$ retained counts, the exact factorization has a Gaussian Toeplitz kernel and two diagonal scalings. If the zero-padded FFT length is $N_b$, a representative normwise amplification factor is

$$
 \kappa_b=\sqrt{N_b}\,\max_n|A_n|\,\|k_b\|_2\,\|D_b\|_2,
 \tag{9.3}
$$

under the specified normalization of the FFT and the input. A forward/inverse FFT perturbation bound may multiply $\kappa_b$ by a library- and precision-dependent relative error factor. Different FFT normalization conventions change intermediate constants, so the exact bound must match the implemented transforms.

Large $\kappa_b$ can make the algorithm unstable even when the analytic interpolation and tail errors are tiny. Opposite constant rescalings of the two diagonal factors can prevent overflow without reducing their *product norm* in (9.3). The independent NumPy/SciPy experiments in §11 reproduce large binary64 errors for small-center, wide-halo cases; they do not certify the behavior of the actual JAX FFT runtime.

### 9.2 What is needed for an end-to-end certificate

A complete production certificate requires, at least:

- Rounding enclosures for $p_b,V_b,\xi_b,q_{b,n}$, log recurrences, factorial/Stirling or `lgamma` calls, and diagonal exponentials; underflow/overflow must be bounded after **all** rescalings.
- A specified FFT library normalization and a forward/inverse transform error model (e.g. a normwise bound under a declared floating-point model), including complex products and zero padding.
- A bound on the input-output amplification of the factored convolution; use an interval or carefully controlled bound when cancellation is severe.
- Stable interpolation weights, barycentric or equivalent formulas, and error bounds for derivative weights.
- Bounds on logit/angle calculation, $\partial_s r$, and branch decisions when a query is numerically near an interface.
- Separate backward and transpose-scatter arithmetic analyses. Transposition of the *exact* linear codec does not automatically certify floating-point scatter order or adaptive JAX differentiation.

One may instead fall back to a stable direct binomial row action for blocks that fail the numerical gate. This changes the cost, not the analytic target.

## 10. A certificate-driven planner

Given absolute one-step tolerances `atol` and `grad_atol`, a supplied sensitivity bound $\Theta_*$, and a fixed parameter domain, search finite candidates $(c,\beta,m,\tau)$ with even $2\le m\le16$. If the desired guarantee is for a $K$-step epoch, first use (8.2), (8.5) or (8.7) to convert candidate one-step bounds into *final* error bounds, rather than heuristically allocating $\eta=\delta$.

```text
for candidate (c, beta, m, tau) in a bounded, cost-ordered search:
    prepare fixed block centers, cores, halos, count windows
    certify omitted node/direct binomial tails at halo endpoints
    for each block:
        certify max H_m(M,q) on its entire stencil hull
        calculate D_m,b and certified Peano constants C_m,0 / C_m,1
        calculate signed Lebesgue factors Lambda_m / Lambda'_m
        bound local value error and local-r derivative error
        convert derivative to parameter coordinate via the local G_b
        analyze factor construction, FFT rounding and amplification
        reject or replace unstable blocks with direct evaluation
    certify direct-boundary value and derivative tails
    if a classical derivative is required:
        form smooth overlaps at all value-jump and derivative-kink interfaces
        recompute certificates over extended stencil/branch domains
        include the blend term 3 J / (2 Delta)
    else:
        retain branchwise derivative bound and BV jump ledger
    assemble global delta and eta from maximum of row/branch bounds
    if an epoch is requested:
        evaluate multi-step value and gradient bounds
    accept only if both targets and numerical budgets are certified
choose the lowest-cost accepted candidate
if none is certified: use a suitably certified direct binomial action
```

**Nonmonotonicity matters.** Increasing interpolation order expands the halo; changing $\beta$ changes block centers and core widths but does not freely reduce local log-odds grid spacing; tightening $\tau$ may add unstable far-tail factors. Therefore binary search on a single setting is not generally justified. A certificate can also be too conservative even for an accurate approximation, especially when a whole-halo $H_m$ maximum replaces a smaller, location-specific error. A per-query Gram or Peano refinement is a sensible second stage.

### Cost model

Let block $b$ have $R_b$ grid nodes, $L_b$ retained counts, and padded FFT length $N_b\ge R_b+2L_b-2$ (per the verified source). After preparation, one backward application costs schematically

$$
 O\left(\sum_b[N_b\log N_b+R_b+L_b]+mQ_I+WQ_D\right),
 \tag{10.1}
$$

where $Q_I,Q_D$ count interpolated and direct queries and $W$ is the direct retained width (or a bound on the varying widths). Persistent block geometry/storage and FFT temporaries scale with the corresponding sums of halo lengths, retained widths, and padded transforms; the source caches block spectra (`kernel_fft`) during preparation, which affects constants and memory. A smooth blend may require two nearby branch evaluations. The cost of interval certification and numerical amplification testing is *planner/preparation* cost, not included in (10.1).

## 11. Independent source-matched numerical checks

I reconstructed block-center spacing, quantile windows, halo indices, local polynomial weights, and the direct/reflection masks directly from `fft.py:258–370, 477–504, 596–610`, using independent NumPy/SciPy calculations. For each tested switch, left and right *branch formulas were evaluated at the same probability*, avoiding false “jumps” caused by finite-offset differences. Each difference is the full row $\ell_1$ norm; the derivative figures are in the angular coordinate $\vartheta$. Node probabilities and derivatives came from full binomial PMFs, **not from FFT output**, so the switch table isolates mathematical discontinuities from arithmetic cancellation.

The following maxima are computed over the enumerated source-generated switches for each parameter set, **not over all $q\in[0,1]$**:

| $M$ | $c$ | $m$ | $\tau$ | Largest block value jump | Direct/interior or reflection value jump | Largest stencil angular-derivative kink |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 8 | 8 | $10^{-12}$ | $4.57\times10^{-5}$ | $3.89\times10^{-13}$ | $1.28\times10^{-2}$ |
| 256 | 128 | 12 | $10^{-12}$ | — (direct-only) | $1.76\times10^{-12}$ at reflection | — |
| 512 | 128 | 12 | $10^{-12}$ | $2.99\times10^{-13}$ | $8.97\times10^{-13}$ | $1.08\times10^{-9}$ |
| 1024 | 128 | 12 | $10^{-12}$ | $1.27\times10^{-13}$ | $7.49\times10^{-13}$ | $6.99\times10^{-10}$ |
| 2048 | 128 | 12 | $10^{-12}$ | $1.16\times10^{-13}$ | $6.44\times10^{-13}$ | $6.98\times10^{-10}$ |

A separate independent real-FFT calculation used the **same mathematical DTD factors and zero-padding formula**, with `scipy.stats.binom.logpmf` for the modal normalization rather than the code's truncated Stirling evaluator. Three signed/bounded test vectors were used (constant, alternating, and seeded Rademacher signs). Maximum differences between the FFT node expectations and the corresponding exact truncated binomial-node expectations were:

| $M$ | $c$ | $m$ | $\tau$ | Maximum measured node error | Maximum sufficient amplification factor $\kappa_b$ |
|---:|---:|---:|---:|---:|---:|
| 128 | 8 | 8 | $10^{-12}$ | $3.55\times 10^{5}$ | $2.42\times10^{23}$ |
| 512 | 16 | 16 | $10^{-12}$ | $2.08\times10^{4}$ | $1.06\times10^{22}$ |
| 512 | 128 | 12 | $10^{-12}$ | $6.90\times10^{-13}$ | $4.46\times10^2$ |
| 1024 | 128 | 12 | $10^{-12}$ | $1.45\times10^{-12}$ | $3.00\times10^3$ |
| 2048 | 128 | 12 | $10^{-12}$ | $3.81\times10^{-12}$ | $8.89\times10^3$ |
| 2048 | 32 | 16 | $10^{-12}$ | $1.82\times10^{-6}$ | $6.33\times10^{11}$ |

The small-cutoff failures are **observed numerical failures in an independent NumPy real-FFT implementation of the factorization**, not failures proved for a given JAX device. At the recommended/default $c=128,m=12$ in these modest-$M$ checks, the FFT error was much smaller, but still nonzero and potentially relevant when planning near $10^{-12}$. The modal-normalization difference and the absence of direct JAX execution limit what can be concluded about code-specific floating-point behavior. These checks establish neither a certified bound nor the global optimum for any parameter setting. They do establish that `finite(a)` and `finite(b)` are not sufficient safety criteria for all allowed configurations.

To turn these into backend regression tests, evaluate **full** numerical Jacobian rows on every switch and both branch limits, test signed and nonnegative observables, validate the transpose and its AD, compare CPU/GPU FFT results, and report maximum row-$\ell_1$ and angular-derivative errors separately. Recompute the data with the exact `_stirling_correction` code and JAX/XLA runtime before quoting production precision.

## 12. Comparison to the supplied NUFFT Sobolev analysis

| Question | Attached global filtered DCT/NUFFT note | This local log-odds FFT analysis |
|---|---|---|
| Approximation | Fixed sampled cosine multiplier | Piecewise moving Lagrange interpolation of exact binomial node expectations |
| Smoothness source | Analytic periodic binomial kernel, complex strip | Local $W^{m,\infty}(I_b;\ell_1)$ regularity and Peano kernels |
| Value error | Fourier truncation plus aliasing | Peano value remainder, node tail, signed weight amplification |
| Derivative error | Independently bounded Fourier derivative | Peano derivative remainder using the **same $m$-th derivative**, plus derivative-weight tails |
| Global regularity | Globally smooth fixed cosine representation | Raw scheme generally BV because block/direct switches may jump |
| Need for repair | Not inherently for regularity | Yes for global $C^1$ and pointwise classical derivative guarantee |
| Multi-step proof | Signed-operator expansion plus exact-path Fisher information | **Reusable without change** once $Q_s$ is a genuine differentiable fixed family |
| FFT numerical risk | Fourier codec error/FFT accuracy | Additional severe diagonal–Toeplitz–diagonal amplification may occur |

The distinction is important: the complex-strip DCT bound (e.g. its $6E_L$ value and $9\Theta_*LE_L$ derivative sufficient bounds) **cannot** simply be transferred to the local FFT interpolant. Conversely, the path-Fisher multi-step theorem depends only on one-step operator bounds and differentiability, so it transfers directly after the regularity issue is addressed.

## 13. What needs to change in `drifft.FFT`

1. **Add honest certifying semantics to `error_report`** (`fft.py:372–384`). The existing `max_tail_mass` controls a *node mass* term, not the signed interpolation amplification, so it should not be reported as a final row-error certificate. Certify $H_m$, $C_{m,\nu}$, and $\Lambda_m$ over every halo/core before allocating an actual value tolerance. Include a separately proven arithmetic budget or mark the result explicitly uncertified.
2. **Implement `grad_atol` planning** (`fft.py:230–235`). Use the same $H_m$ for both Peano estimates, add the derivative-weight amplification and direct derivative-tail bound, and convert with the supplied $\Theta_*$. Distinguish branchwise gradients from global weak/classical derivatives in the API.
3. **Make switch semantics explicit.** The current branch choices (`searchsorted` right, direct on cutoff, reflected lower at exactly $0.5$) should be in the public mathematical contract. Either (a) expose BV/branchwise semantics and jumps; (b) blend only across *value-jump* interfaces for globally weak $W^{1,\infty}$ gradients; or (c) also smooth stencil joins to guarantee classical $C^1$.
4. **Add a per-block FFT stability gate** based on precomputed diagonal/kernel amplification, direct-check spot tests, and a documented device-specific FFT error assumption. For uncertifiable blocks, enlarge the direct region or select a safer representation. Avoid relying solely on the `balance` flag or a finite-factor test.
5. **Validate the actual JAX path**, not just host mathematical reconstruction: test endpoint JVP/VJP/second derivatives from `HiPrim`; signed random observables; exact block/direct/reflection switches; and the zero sentinel scatter under reverse-mode differentiation.
6. **Make horizon-aware planning optional.** When a requested epoch has $K$ generations, select one-step budgets by the exact value and derivative power inequalities in §8, not by arbitrary division of a gradient tolerance. If retaining raw jumps, propagate their atoms with (6.8) rather than claiming a globally differentiable matrix family.

The current defaults may perform well for ordinary value actions, as the experiments suggest, but the code does **not** yet establish a user-requested absolute row-$\ell_1$ or gradient certificate. The independent diagnostics should be incorporated as regression cases, not substituted for a numerical proof.

## 14. Conclusions

The correct unified mathematical framework for the specified local FFT approach is

$$
 \underbrace{\ell_1\text{-valued binomial derivative energy}}_{H_m(M,q)}
 \;\longrightarrow\;
 \underbrace{W^{m,\infty}\text{ local regularity}}_{\mathcal D_{m,b}}
 \;\longrightarrow\;
 \underbrace{W^{1,\infty}\text{ Peano interpolation bounds}}_{e_{0,b},e_{1,b}}
 \;\longrightarrow\;
 \begin{cases}
 BV & \text{raw hard switching},\\
 C^1\cap W^{1,\infty} & \text{fully smoothed switching}.
 \end{cases}
$$

The notable improvement is that **one certified $m$-th derivative bound controls both interpolation value and first derivative**, avoiding mandatory use of $H_{m+1}$. For raw switches, BV records the difference between ordinary piecewise derivative error and singular jump error without pretending a globally smooth approximation exists. For a truly differentiable repaired codec, the NUFFT note's pathwise Fisher-information argument gives the same signed-operator multi-generation gradient improvement, from $O(K^2)$ deterministic propagation to $O(K^{3/2})$ for the value-error contribution in the small-error regime.

The principal unresolved *certification* issue in the verified implementation is separate: the diagonal FFT realization must pass a numerical amplification/roundoff certificate. Neither Sobolev regularity nor accurate exact factors guarantees stable floating-point evaluation.

## References and provenance

1. **Primary code inspected:** user-provided `fft.py`, 610 lines, `drifft.FFT` and supporting `_FFTBlock`, `_FFTBucket`, `_BinomialWeights`, and `_expect_boundary`; code line references throughout refer to this attachment. It is also the repository path [`mancusolab/drifft/src/drifft/_backend/fft.py`](https://github.com/mancusolab/drifft/blob/main/src/drifft/_backend/fft.py), but the source audit is based on the actual uploaded file, not on a fresh GitHub fetch.
2. **Comparative source:** user-provided `wf_joint_operator_sobolev_bounds_revised.md`, *Joint one-step and multi-step error bounds for filtered DCT approximation of the Wright–Fisher operator* (24 October 2026). The signed-matrix power and path-score arguments in §8 adapt that document's derivation; the cosine approximation guarantees cannot be imported for local interpolation.
3. **Earlier algorithm specification:** user-provided `Pasted text(6).txt`, *Research prompt: error bounds for local FFT binomial expectations*. The present note replaces its unverified implementation assumptions with the actual attached source.
4. C. de Boor (2005), *Divided Differences*, *Surveys in Approximation Theory* 1:46–69, especially §9 on Peano kernels and Genocchi–Hermite: <https://pages.cs.wisc.edu/~deboor/sat/papers/2/2.pdf>.
5. G. Arampatzis, M. A. Katsoulakis, and Y. Pantazis (2015), *Pathwise Sensitivity Analysis in Transient Regimes*, <https://arxiv.org/abs/1502.05430>. General pathwise information framework; the finite-binomial score and perturbation formulas here are derived explicitly.
6. P. Dupuis, M. A. Katsoulakis, Y. Pantazis, and P. Plecháč (2015), *Path-space information bounds for uncertainty quantification and sensitivity analysis of stochastic dynamics*, <https://arxiv.org/abs/1503.05136>.
7. N. J. Higham (2002), *Accuracy and Stability of Numerical Algorithms*, 2nd ed. Standard numerical-error methodology, **not** a verified forward-error constant for the JAX/XLA FFT used here.

**Verification boundary.** The mathematical statements are derived for exact prepared binomial nodes; FFT numeric statements are conditional except for explicitly reported **NumPy/SciPy diagnostic measurements**. This analysis has read the attached `fft.py` but has not executed the full JAX backend, checked all imported collaborators, or produced a device-independent floating-point certificate.
