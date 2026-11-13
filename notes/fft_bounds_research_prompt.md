# Research prompt: error bounds for local FFT binomial expectations

You are analyzing a numerical algorithm, not implementing software. Develop rigorous, computable error bounds for the **local log-odds FFT approximation of binomial expectations** described below. The goal is to support automatic planning from absolute value and gradient tolerances.

Do not assume access to a repository. Use the algorithm specified here as the object of analysis. Clearly distinguish proved bounds, conditional numerical bounds, conjectures, and empirical observations. Search for relevant primary mathematical sources and cite them precisely.

## Mathematical target

For $M$ gene copies, define the binomial row

$$
b_j(q)=\binom Mj q^j(1-q)^{M-j},\qquad j=0,\ldots,M,
$$

and the expectation

$$
B_M[f](q)=\sum_{j=0}^{M}b_j(q)f_j.
$$

The input $f$ is arbitrary and bounded; do not assume smoothness across allele counts.

For an approximation $\widetilde b(q)$, the desired one-step value error is

$$
\delta=\sup_{q\in[0,1]}\sum_{j=0}^{M}
|\widetilde b_j(q)-b_j(q)|.
$$

This is maximum row L1 error, equivalently the infinity-to-infinity operator norm. It controls expectation error by $\delta\|f\|_\infty$ and forward-distribution L1 error by $\delta\|p\|_1$, including signed input mass.

For a parameterized probability $q=q(s,x)$, also seek

$$
\eta=\sup_{s,x}\sum_j
|\partial_s\widetilde b_j(q(s,x))-\partial_s b_j(q(s,x))|.
$$

The parameter coordinate $s$ and its domain are fixed. You may assume a supplied finite angular-sensitivity bound

$$
\Theta_*\ge\sup_{s,x}
\frac{|\partial_s q(s,x)|}{\sqrt{q(s,x)(1-q(s,x))}}.
$$

Constant absorbing rows have zero sensitivity. Handle support-changing endpoints separately.

The planning targets are $\delta\le\texttt{atol}$ and $\eta\le\texttt{grad\_atol}$. These are absolute, unscaled operator errors—not relative errors for rare events.

## Algorithm to analyze

### 1. Block centers and core regions

Let $c=\texttt{min\_mode\_count}$, and let $\beta=\texttt{block\_spacing}$.

If $M>2c$, choose integer block centers starting at $j_0=c$, progressing by

$$
j_{\mathrm{next}}=
\min\!\left(M-c,\,
j+\max\{1,\lfloor\beta\sqrt{j(1-j/M)}\rfloor\}\right),
$$

until reaching $M-c$.

For each center $j_b$, set

$$
p_b=j_b/M,\qquad V_b=Mp_b(1-p_b),\qquad
\eta_b=\log\frac{p_b}{1-p_b}.
$$

Adjacent core regions meet at the midpoint of their centers in log-odds coordinates. The outer core boundaries are $\operatorname{logit}(c/M)$ and $\operatorname{logit}(1-c/M)$.

### 2. Local integer grids and halos

Each block uses integer nodes $r$, corresponding to

$$
q_{b,r}=\operatorname{logistic}(\eta_b+r/V_b).
$$

For an even interpolation stencil size $m$, currently restricted to $2\le m\le16$, the node interval is

$$
r_{\min}=
\left\lfloor V_b(\eta_{\mathrm{left}}-\eta_b)\right\rfloor-m/2,
$$

$$
r_{\max}=
\left\lceil V_b(\eta_{\mathrm{right}}-\eta_b)\right\rceil+m/2.
$$

Thus changing $\beta$ changes block coverage and centers; it does not directly introduce a free multiplier in the local grid spacing $1/V_b$.

### 3. Retained binomial windows

For each block, choose one fixed count window $[\ell_b,u_b]$. Its lower endpoint comes from a lower-tail binomial quantile at $q_{b,r_{\min}}$; its upper endpoint comes from an upper-tail quantile at $q_{b,r_{\max}}$. Each tail receives a nominal allowance $\tau/2$, where $\tau=\texttt{tail\_tol}$.

The intended property is

$$
\sup_{r\in[r_{\min},r_{\max}]}
\Pr_{\mathrm{Bin}(M,q_{b,r})}
\{J\notin[\ell_b,u_b]\}\le\tau.
$$

Establish a sufficient choice of quantile conventions and any necessary endpoint checks. Do not assume that evaluating total omitted mass at two endpoints automatically proves a uniform bound without justification.

Retained rows are not renormalized.

### 4. Exact local convolution identity

Write $h_b=1/V_b$. The ideal factors are

$$
a_r=
\exp\!\left[
j_bh_br
-M\log\!\bigl(1-p_b+p_be^{h_br}\bigr)
+\frac{h_br^2}{2}
\right],
$$

$$
d_j=b_j(p_b)\exp\!\left[\frac{h_b(j-j_b)^2}{2}\right].
$$

Then

$$
b_j(q_{b,r})
=a_r\,e^{-h_b(r-(j-j_b))^2/2}\,d_j.
$$

Verify this identity. It is a diagonal–Toeplitz–diagonal factorization of the binomial kernel, not a normal approximation.

Compute retained node values

$$
y_r=a_r\sum_{j=\ell_b}^{u_b}
e^{-h_b(r-(j-j_b))^2/2}d_jf_j
$$

using zero-padded FFT convolution.

If there are $R$ output nodes and $L$ retained input counts, the kernel has $R+L-1$ entries. The FFT length is a power of two at least $R+2L-2$, preventing circular wraparound.

For numerical construction, $d_j/b_{j_b}(p_b)$ is obtained through a log recurrence:

$$
\log\frac{d_{j+1}}{d_j}
=\log(M-j)-\log(j+1)+\eta_b+h_b(j-j_b+1/2).
$$

The modal probability is incorporated into the output factor. Its normalization uses Stirling corrections through the $n^{-9}$ term for factorial arguments at least 16, and a log-gamma calculation for smaller positive arguments. Opposite constant shifts of the two log-diagonal factors are used to balance their magnitudes.

Separate this numerical factor construction from the ideal exact identity.

### 5. Moving local interpolation

For an interior query $q$, choose its block by log-odds core membership, with ties assigned to the block on the right. Define

$$
r=V_b(\operatorname{logit}(q)-\eta_b),\qquad
a=\lfloor r\rfloor-m/2+1,\qquad t=r-a.
$$

Evaluate

$$
\widetilde B_M[f](q)
=\sum_{k=0}^{m-1}\lambda_k(t)y_{a+k},
$$

where

$$
\lambda_k(t)=
\prod_{\substack{0\le l<m\\l\ne k}}\frac{t-l}{k-l}.
$$

The interpolation is applied to the completed binomial node expectations $y_r$, including their output scaling. The weights may be signed.

### 6. Direct boundary evaluation

Let $q_c=\min(c/M,1/2)$. For $q\le q_c$, evaluate a direct truncated binomial sum over counts $0,\ldots,W-1$, where $W$ is chosen so

$$
\Pr_{\mathrm{Bin}(M,q_c)}(J\ge W)\le\tau.
$$

For $q\ge1-q_c$, use the reflected construction. At a direct/interpolated interface, choose the direct branch.

If $M\le2c$, use direct sums throughout, reflecting the upper half onto the lower half.

Direct rows are not renormalized. Their probability derivatives use

$$
b'_j(q)=M\bigl(b_{M-1,j-1}(q)-b_{M-1,j}(q)\bigr),
$$

with out-of-range counts interpreted as zero.

### 7. Differentiation semantics

All prepared geometry, windows, and integer choices remain fixed across the parameter domain. Automatic differentiation follows the selected branch and differentiates interpolation weights or direct binomial sums. It does not differentiate block indices, `floor`, quantile choices, or other discrete selections.

Forward propagation is the transpose of the same linear-in-$f$ expectation approximation. There is no output clipping or renormalization.

## Required analysis

### A. Value-error decomposition

Derive computable uniform row-L1 bounds for:

1. Interpolating exact, untruncated binomial rows.
2. Amplification of nodewise truncation errors by signed interpolation weights.
3. Direct boundary truncation.
4. The global assembly across blocks.

Consider a decomposition of the form

$$
\delta_{\mathrm{analytic}}
\le
\max_b\{E_{\mathrm{interp},b}
+\Lambda_bE_{\mathrm{node},b}\},
$$

combined with the direct-region bound, but establish the correct form and constants.

Exploit the binomial structure to avoid needlessly pessimistic componentwise bounds. Give finite-size formulas suitable for computation, not only asymptotic rates.

### B. Derivative bounds and switching boundaries

Derive derivative errors independently of value errors, including interpolation-weight derivative amplification and direct derivative-tail bounds. Account for the conversion between local log-odds derivatives and the supplied parameter sensitivity.

Examine:

- Stencil changes when $r$ crosses an integer.
- Changes of selected block.
- Direct/interpolated interfaces.
- Reflection interfaces in the direct-only case.

Determine continuity and differentiability at each type of interface. Distinguish classical derivatives, one-sided derivatives, and derivatives returned by branchwise AD.

If a uniform classical derivative guarantee is impossible for the algorithm as specified, demonstrate the obstruction. State a precise weaker result, if available, and propose a minimal modification that permits a genuine derivative guarantee. Prove bounds for the modification separately.

### C. Numerical stability

Separate analytic approximation error from:

- Factor-construction error, including truncated Stirling corrections.
- Diagonal scaling and cancellation.
- FFT evaluation error.
- Interpolation-weight and accumulation error.
- Numerical derivative evaluation.

State any floating-point or FFT-library assumptions explicitly. If a useful rigorous numerical bound cannot be established, identify the missing assumption or certificate. Do not count an allocated numerical budget as a proved error bound.

### D. A usable planner

Provide pseudocode that selects or constrains $m$, $\beta$, $c$, and $\tau$ from `atol`, optional `grad_atol`, and the parameter-domain sensitivity information.

Address feasibility, nonmonotone bounds, numerical precision floors, direct-evaluation fallback, and the present interpolation-order limit of 16. Explain preparation cost, application cost, and storage implications.

Do not assume that increasing interpolation order always improves numerical accuracy or that reducing block spacing necessarily resolves every error component.

### E. Independent validation

If computational tools are available, implement a small reference experiment using direct binomial rows and their analytic derivatives. Measure full row-L1 errors, not just errors on smooth observables.

Include exact switching points and approaches from both sides, endpoints, near-boundary probabilities, signed observables, and several population sizes and interpolation orders.

Compare observations with the proposed bounds, quantify conservatism, and report counterexamples. Numerical experiments support but do not replace proofs.

## Deliverable

Produce a self-contained technical note with:

1. A clear statement of what can and cannot be guaranteed for the algorithm as specified.
2. Explicit assumptions and notation.
3. Derived value and derivative bounds with proofs or precise source attribution.
4. A separate numerical-error model.
5. Practical planning pseudocode.
6. Validation results, if computed.
7. Remaining gaps and any required algorithm changes.

Prioritize a correct, useful result over a claim of completeness. Do not transfer global Chebyshev-compression bounds to this local interpolation scheme without establishing the connection.
