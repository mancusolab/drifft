# Wright–Fisher Jacobi reduction: row-error bounds and rational assembly

## Motivation and scope

The [FFT/NUFFT note](wf_filtered_chebyshev_bounds.md) studies compression of a Wright–Fisher row by sampled, filtered Chebyshev polynomials. The Jacobi backend solves a related problem: represent a fixed-population transition in a small polynomial space, then apply many generations through a reduced matrix. A useful error analysis must connect the polynomial representation to errors in the original allele-count distribution.

The relevant basis is Jacobi with parameters $(1,1)$, equivalently Gegenbauer with parameter $3/2$. The backend separates the two endpoint values and multiplies its interior polynomials by $1-z^2$. This construction gives exact absorbing endpoints and an explicit connection to discrete Hahn polynomials. It also permits rational selection maps to be assembled using a symmetric tridiagonal matrix.

This note derives the ideal projection bound used by the current rank selector, gives separate bounds for sampled coefficients and padded assembly, and explains their implications for repeated transitions. The resulting sufficient rank is $O(\sqrt{M\log(M/\varepsilon)})$, capped at the exact finite dimension. A bound with a dimension-independent Gaussian prefactor is **not established here**. The larger sufficient rank is a limitation of the certificate, not a proof that the Jacobi approximation needs that many modes.

The implementation discussed is [src/drifft/_jacobi.py](../src/drifft/_jacobi.py), specifically `Jacobi`, `_choose_rank`, `_JacobiGeometry`, and `_assemble`. Its supported frequency maps are neutral evolution and underdominance, with no mutation in the frequency map. Immigration is a separate affine forcing. All mathematical certificates below are in exact arithmetic unless an additional numerical error is stated explicitly.

## 1. Setup and notation

Let $M\ge4$ be the offspring gene-copy count. For destination values $f=(f_0,\ldots,f_M)$, define the binomial expectation

$$
(B_M f)(y)=\sum_{j=0}^{M}\binom Mj
\left(\frac{1+y}{2}\right)^j
\left(\frac{1-y}{2}\right)^{M-j}f_j,
\qquad -1\le y\le1.
$$

Thus $y=2q-1$ is the centered offspring probability. The count grid is $z_j=2j/M-1$. For a centered frequency map $g$, the backward transition on values is $(Pf)_i=(B_Mf)(g(z_i))$; forward mass evolves by $P^T$.

For a linear approximate row family, let

$$
\delta=\sup_{-1\le y\le1}\sum_{j=0}^{M}
|b_{M,j}(y)-\widetilde b_{M,j}(y)|.
$$

By duality between $\ell_1$ and $\ell_\infty$, this equals the operator norm from bounded count-space values to uniformly bounded functions of $y$. Evaluating at any source probabilities in $[0,1]$ therefore gives

$$
\|P-\widetilde P\|_{\infty\to\infty}\le\delta,
\qquad
\|P^T-\widetilde P^T\|_{1\to1}\le\delta.
\tag{1}
$$

This is the same error criterion as in the FFT/NUFFT note. A weighted polynomial norm will be an intermediate tool, not a replacement for (1).

### 1.1 Separating the endpoints

Set $h(z)=1-z^2$, $\gamma=1-1/M$, and write

$$
\ell_f(z)=a_f+b_fz,
\qquad a_f=\frac{f_0+f_M}{2},\qquad b_f=\frac{f_M-f_0}{2},
\qquad
u_j=\frac{f_j-\ell_f(z_j)}{h(z_j)},\quad 1\le j\le M-1.
$$

The denominator is used only at interior counts. Define the shifted binomial operator

$$
(Su)(y)=\sum_{j=1}^{M-1}\binom{M-2}{j-1}
\left(\frac{1+y}{2}\right)^{j-1}
\left(\frac{1-y}{2}\right)^{M-1-j}u_j.
$$

The identity $j(M-j)\binom Mj=M(M-1)\binom{M-2}{j-1}$ gives

$$
\boxed{(B_Mf)(y)=\ell_f(y)+\gamma h(y)(Su)(y).}
\tag{2}
$$

This is an exact finite-binomial identity, with no diffusion approximation. The factor $h$ makes the interior term vanish at both endpoints.

### 1.2 The continuous and discrete orthogonal bases

Use the probability weight $w(z)=3(1-z^2)/4$ on $[-1,1]$. Let $\phi_k$ be the Jacobi polynomial of degree $k$ with positive leading coefficient, normalized in $L^2(w)$:

$$
\phi_k(z)=\sqrt{\frac{(k+2)(2k+3)}{6(k+1)}}\,P_k^{(1,1)}(z),
\qquad
\int_{-1}^{1}\phi_k(z)\phi_l(z)w(z)\,dz=\delta_{kl}.
$$

In particular, $\phi_0=1$ and $\phi_1=\sqrt5\,z$. Standard Jacobi normalization and endpoint bounds [1] give

$$
z\phi_k(z)=\beta_k\phi_{k+1}(z)+\beta_{k-1}\phi_{k-1}(z),
\qquad
\beta_k^2=\frac{(k+1)(k+3)}{(2k+3)(2k+5)},
$$

$$
|\phi_k(z)|\le\phi_k(1),\qquad
\phi_k(1)^2=\frac{(k+1)(k+2)(2k+3)}6.
\tag{3}
$$

The term containing $\beta_{-1}$ is omitted. Mixing the binomial rows of $S$ against $w$ gives the discrete probability weights

$$
\pi_j=\int_{-1}^{1}w(y)\Pr\{1+\operatorname{Bin}(M-2,(1+y)/2)=j\}\,dy
=\frac{6j(M-j)}{(M-1)M(M+1)}.
$$

Write $\|u\|_\pi^2=\sum_{j=1}^{M-1}\pi_j|u_j|^2$. Let $\psi_k$ be the degree-$k$ orthonormal polynomial on the grid $z_j$ for these weights, again with positive leading coefficient. These are normalized Hahn polynomials with parameters $(1,1)$ [2].

## 2. Exact singular values and the retained-mode construction

The beta-binomial conditional-expectation correspondence [3] specializes to

$$
S\psi_k=d_k\phi_k,
\qquad
S^*\phi_k=d_k\psi_k,
\qquad 0\le k\le M-2,
$$

$$
\boxed{d_0=1,\qquad
 d_k^2=\prod_{j=2}^{k+1}\frac{M-j}{M+j}.}
\tag{4}
$$

Here $S^*$ is the adjoint between $L^2(w)$ and $\ell_2(\pi)$, not an unweighted transpose. The numbers $d_k$ are singular values of the shifted binomial operator. They are not the eigenvalues of the selected Wright–Fisher transition.

One way to obtain (4) is to let $Q$ have the beta distribution with parameters $(2,2)$ and, conditional on $Q$, let $J\sim\operatorname{Bin}(M-2,Q)$. Then $J$ has the weights $\pi_{J+1}$, while $Q\mid J$ has parameters $(J+2,M-J)$. Both conditional expectations preserve polynomial degree. Their composition is self-adjoint and preserves each polynomial subspace, so the orthogonal polynomial of each degree is an eigenfunction. Its leading coefficient is multiplied by

$$
\frac{(M-2)(M-3)\cdots(M-k-1)}{(M+2)(M+3)\cdots(M+k+1)},
$$

which is $d_k^2$. Positive leading coefficients fix the signs in the singular relations.

For $1\le r\le M-1$, retain degrees $0,\ldots,r-1$. With $c_k=\langle Su,\phi_k\rangle_w$, define

$$
(\widetilde B_{M,r}f)(y)
=\ell_f(y)+\gamma h(y)\sum_{k=0}^{r-1}c_k\phi_k(y).
\tag{5}
$$

The full representation has $r+2$ coordinates and degree at most $r+1$ in $y$. At $r=M-1$, (5) is exact. The constant and linear functions are reproduced at every rank. After composition with selection, linear reproduction means $Pz=g(z)$, not conservation of the allele-frequency mean.

## 3. A uniform row-error certificate

### 3.1 From singular values to pointwise error

Expand $u=\sum_{k=0}^{M-2}\widehat u_k\psi_k$. Cauchy–Schwarz and (4) give

$$
\left|Su(y)-\sum_{k=0}^{r-1}c_k\phi_k(y)\right|
\le \|u\|_\pi
\left(\sum_{k=r}^{M-2}d_k^2\phi_k(y)^2\right)^{1/2}.
\tag{6}
$$

The exact truncation norm from $\ell_2(\pi)$ to $L^2(w)$ is $d_r$ when $r\le M-2$. Equation (6) is the extra step needed for uniform control.

### 3.2 The cost of subtracting the endpoint interpolant

If $\|f\|_\infty\le1$, then $|f_j-\ell_f(z_j)|\le2$. Direct summation gives

$$
\|u\|_\pi^2\le4\sum_{j=1}^{M-1}\frac{\pi_j}{h(z_j)^2}
=\frac{3M^2}{M^2-1}H_{M-1}=:C_M^2,
\qquad H_n=\sum_{j=1}^{n}\frac1j.
\tag{7}
$$

Thus division by the endpoint envelope costs at most $C_M=O(\sqrt{\log M})$ in this norm. It would cost $O(M)$ in the ordinary maximum norm; keeping the discrete weight is useful.

Combining (2), (6), and (7) proves the envelope-aware certificate

$$
\delta_{M,r}\le\gamma C_M
\sup_{-1\le y\le1}h(y)
\left(\sum_{k=r}^{M-2}d_k^2\phi_k(y)^2\right)^{1/2}.
\tag{8}
$$

A simpler bound uses $h\le1$ and (3). Define, with an empty sum equal to zero,

$$
T_r^2=\sum_{k=r}^{M-2}
 d_k^2\frac{(k+1)(k+2)(2k+3)}6.
$$

Then

$$
\boxed{\delta_{M,r}\le\gamma C_M T_r.}
\tag{9}
$$

This is the ideal one-step bound used by `_choose_rank`. It holds uniformly over offspring probabilities, hence also over the supported selection strengths. It excludes coefficient quadrature, stencil truncation, matrix padding, and arithmetic errors.

### 3.3 Gaussian decay and sufficient rank

Since $\log((1-x)/(1+x))\le-2x$ for $0\le x<1$,

$$
d_k\le\exp\left(-\frac{k(k+3)}{2M}\right).
\tag{10}
$$

For an explicit, deliberately coarse closed-form certificate, each endpoint factor in $T_r^2$ is at most $M^3/3$ and there are fewer than $M$ terms. Therefore

$$
\delta_{M,r}\le
\frac{\gamma C_M M^2}{\sqrt3}
\exp\left(-\frac{r(r+3)}{2M}\right).
\tag{11}
$$

For $0<\varepsilon<1$, a sufficient choice is

$$
r=\min\left\{M-1,
\left\lceil\sqrt{2M\log\left(
\frac{\gamma C_M M^2}{\sqrt3\,\varepsilon}\right)}\right\rceil\right\}.
\tag{12}
$$

The finite sum (9) is preferable in practice and is what the code evaluates. Formula (12) establishes $r=O(\sqrt{M\log(M/\varepsilon)})$ before saturation. It does not establish the $O(\sqrt{M\log(1/\varepsilon)})$ guarantee proved for the tapered Chebyshev operator.

The losses are identifiable: (7) bounds all endpoint residuals by their worst case, while (9) replaces the envelope-weighted polynomial values by endpoint maxima. Improving either estimate may sharpen the certificate. Singular-value decay alone does not remove these losses.

## 4. Gauss–Jacobi sampling and binomial stencils

The implementation uses $N=r+32$ Gauss–Jacobi nodes $x_a$ and positive weights $\omega_a$ normalized for $w$. In exact arithmetic, their sum is one and the quadrature is exact through degree $2N-1$ [4]. It computes

$$
c_k^{(N)}=\sum_{a=1}^{N}\omega_a\phi_k(x_a)(Su)(x_a),
\qquad 0\le k<r.
$$

Symmetry permits storage of half the nodes; it does not change this mathematical operator. The fixed choice $N=r+32$ is an implementation rule, not by itself a universal error certificate.

### 4.1 A quadrature bound

Define the evaluation factor

$$
A_r=\sup_{-1\le y\le1}h(y)
\left(\sum_{k=0}^{r-1}\phi_k(y)^2\right)^{1/2}
\le\left(\sum_{k=0}^{r-1}\phi_k(1)^2\right)^{1/2}.
$$

Let $K=2N-r+1$, and set $T_K=0$ if $K>M-2$. Every product $\phi_k\phi_l$ with $k<r$ and $l<K$ is integrated exactly. Only modes $l\ge K$ can contaminate the retained coefficients.

The sampled columns $\phi_0,\ldots,\phi_{r-1}$ are orthonormal in the quadrature inner product because $N\ge r$. Projection onto these columns has norm one. Apply this observation to the high-degree tail, then use its pointwise bound from (6):

$$
\|c^{(N)}-c\|_2\le T_K\|u\|_\pi,
\qquad
\boxed{\delta_{\mathrm{quad}}\le\gamma C_M A_rT_K.}
\tag{13}
$$

In particular, retained coefficients are exact when $2N-1\ge M+r-3$. Otherwise (13) gives a conservative bound. With $N=r+32$, the first possibly aliased degree is $K=r+65$.

### 4.2 Truncated and renormalized binomial rows

Suppose every sampled row of $S$ is replaced by a probability row with $\ell_1$ error at most $\tau$. Since the smallest interior envelope is $4(M-1)/M^2$,

$$
\|u\|_\infty\le U_M\|f\|_\infty,
\qquad U_M=\frac{M^2}{2(M-1)}.
$$

The error in each sampled expectation is at most $\tau U_M\|f\|_\infty$. The same discrete orthogonal projection argument gives

$$
\boxed{\delta_{\mathrm{stencil}}\le\gamma A_rU_M\tau.}
\tag{14}
$$

If an exact probability row loses mass $\eta$ by truncation and is then renormalized, its $\ell_1$ error is $2\eta$. This statement does not cover errors in evaluating the retained probabilities. The backend uses a fixed binomial stencil radius and floating-point log probabilities; those arithmetic errors require a separate allowance.

For one sampled transition, (9), (13), and (14) combine as

$$
\delta_{\mathrm{one\ step}}
\le\gamma C_MT_r+\gamma C_MA_rT_K
+\gamma A_rU_M\tau+\delta_{\mathrm{arithmetic}}.
\tag{15}
$$

This bound explains why the planning argument `one_step_tolerance` is not a total implemented-error guarantee. It also provides a way to certify a proposed rank and quadrature order independently of observed errors on a few test vectors.

## 5. Rational selection and reduced matrix assembly

For the supported underdominance map, let $0\le s<1$, $a=1-s/2$, and $b=s/2$. Then

$$
g(z)=\frac{z}{a+bz^2},\qquad
m(z)=\frac{a^2-b^2z^2}{(a+bz^2)^2},
\qquad h(g(z))=h(z)m(z).
\tag{16}
$$

On the real interval, $0<m(z)\le1$ and $g([-1,1])=[-1,1]$. Neutral evolution is $s=0$. If $Q$ is the exact interior transition with absorbing states removed, then

$$
\widehat Q=\operatorname{diag}(h)^{-1}Q\operatorname{diag}(h),
\qquad
(\widehat Qu)_i=\gamma m(z_i)(Su)(g(z_i)).
$$

Its row sums are $\gamma m(z_i)$, so $\|\widehat Q^n\|_\infty\le\gamma^n$. This contraction is in the transformed interior norm. It is not a claim that an arbitrary reduced coefficient matrix has the same norm bound.

### 5.1 Why a tridiagonal matrix appears

Let $C$ map interior values to all $M-1$ Jacobi coefficients of $Su$, and put $\Sigma=\operatorname{diag}(d_0,\ldots,d_{M-2})$. Multiplication by the count coordinate in the orthonormal Hahn basis is represented by the symmetric matrix $Z$ with

$$
Z_{k,k}=0,\qquad
Z_{k,k+1}=\beta_k\sqrt{1-\left(\frac{k+2}{M}\right)^2},
\qquad 0\le k\le M-3.
\tag{17}
$$

Its eigenvalues are the interior grid points $z_j$, and its spectral weights at the first coordinate are $\pi_j$. For any function $v$ on that grid,

$$
Cv=\Sigma v(Z)e_0.
\tag{18}
$$

Indeed, the coefficients of $v$ in the Hahn basis are $v(Z)e_0$, and (4) multiplies each by $d_k$. Here $e_0=(1,0,\ldots,0)^T$; matrix indices start at zero. Formula (17) follows from the Hahn recurrence [2], or from the identity

$$
S[zu](y)=\left(1-\frac2M\right)y(Su)(y)
+\frac{1-y^2}{M}(Su)'(y)
$$

and the singular scaling (4).

Let $G$ be the $r\times r$ interior reduced matrix. Its column $k$ is the first $r$ coordinates of

$$
\gamma\Sigma m(Z)\phi_k(g(Z))e_0,
\qquad 0\le k<r.
\tag{19}
$$

The endpoint slope contributes a forcing column $e$, the first $r$ coordinates of

$$
\Sigma b\,g(Z)e_0,
\tag{20}
$$

because $(g(z)-z)/h(z)=b\,g(z)$. These formulas explain the rational assembly in `_assemble`. They do not require a population-sized dense transition matrix.

### 5.2 Padding and a separate assembly certificate

The code replaces $Z$ by its leading $R\times R$ principal block $Z_R$, with $r\le R\le M-1$. It solves systems with $aI+bZ_R^2$ and applies the polynomial recurrence. The eigenvalues of this positive-definite matrix lie in $[a,1]$, so its condition number is at most $1/a<2$. Assembly takes $O(rR)$ arithmetic operations, in addition to storing the dense $r\times r$ output.

For a polynomial $p$ of degree at most $2R-r$, the first $r$ entries of $p(Z)e_0$ and $p(Z_R)e_0$ agree. A tridiagonal path must take at least $R$ steps to reach the first deleted index and another $R-r+1$ steps to return to a retained index. Thus no shorter path detects the truncation.

Suppose each function $F_k(z)=m(z)\phi_k(g(z))$, $k<r$, admits a degree-$(2R-r)$ polynomial approximation with uniform error at most $\epsilon_R$. Symmetric functional calculus and $\|\gamma\Sigma\|_2\le1$ imply

$$
\boxed{\|G-G_R\|_F\le2\sqrt r\,\epsilon_R.}
\tag{21}
$$

The forcing function $b\,g(z)$ must also be bounded. If its corresponding scalar approximation error is $\epsilon_R^{\mathrm{force}}$, then $\|e-e_R\|_2\le2\epsilon_R^{\mathrm{force}}$.

Analyticity makes these estimates practical. On a Bernstein ellipse of radius $\rho>1$ whose boundary and interior contain no poles of $g$, put $B_\rho=\max_{k<r}\max_{z\in E_\rho}|F_k(z)|$. Truncation of the Chebyshev series at degree $d$ gives [5]

$$
\epsilon_R\le\frac{2B_\rho}{\rho-1}\rho^{-d},
\qquad d=2R-r.
\tag{22}
$$

For completeness, explicit conservative constants can be obtained with $\rho=3/2$. Its semiaxes are $c=13/12$ and $d_\rho=5/12$. Since

$$
|a+bz^2|\ge a-bd_\rho^2>0,
\qquad
|g(z)|\le L:=\frac{c}{a-bd_\rho^2},
$$

one may use $\chi=L+\sqrt{L^2+1}$ and

$$
B_\rho\le
\frac{a^2+b^2c^2}{(a-bd_\rho^2)^2}
\phi_{r-1}(1)\chi^{r-1}.
\tag{23}
$$

The disk $|z|\le L$ lies inside the ellipse of radius $\chi$; the standard polynomial growth bound on an ellipse and (3) give (23). The forcing function is bounded there by $bL$. All these constants are uniform for $0\le s\le s_{\max}<1$. Equations (21)–(23) prove sufficient padding $R=O(r+\log(1/\varepsilon))$, with $R=M-1$ as the exact fallback.

The implementation's `_padded_rank` uses a tighter ellipse-image estimate than (23). The conservative derivation here establishes the mechanism and scaling, not an independent validation of every constant in that routine. A Frobenius assembly tolerance is also not yet a population-space row-error tolerance.

## 6. Endpoints, positivity, and repeated generations

Write $E_rf=(a_f,b_f,c_0,\ldots,c_{r-1})$ for the exact coefficient map and let $D_r$ evaluate (5) at $g(z_i)$. Then

$$
\widetilde P=D_rE_r,\qquad
H=E_rD_r=
\begin{pmatrix}
1&0&0\\
0&1&0\\
0&e&G
\end{pmatrix},
\qquad
\widetilde P^{\,n}=D_rH^{n-1}E_r,\quad n\ge1.
\tag{24}
$$

For forward mass, transpose the complete expression. The code calls $D_r^T$ `encode` and the sampled approximation to $E_r^T$ `decode`; those names refer to mass propagation, whereas (24) acts on values.

The full-state construction preserves constants, hence row sums, and leaves the absorbing endpoint rows exact. It need not preserve positivity. A row with $\ell_1$ error at most $\delta$ from a probability row has negative mass at most $\delta/2$. Removing the endpoint outputs gives the corresponding interior approximation; its exact target is substochastic, and full-state mass conservation should not be asserted for it.

### 6.1 Powers of the ideal projected transition

For the fixed matrix $\widetilde P=D_rE_r$ and $\delta=\|P-\widetilde P\|_\infty$, stochasticity gives $\|P\|_\infty=1$ and $\|\widetilde P\|_\infty\le1+\delta$. Telescoping therefore proves

$$
\|P^n-\widetilde P^{\,n}\|_\infty
\le(1+\delta)^n-1.
\tag{25}
$$

Choosing $\delta\le\log(1+\varepsilon)/n$ is sufficient for absolute operator error $\varepsilon$. Substitution in (12) gives the conservative rank scaling $O(\sqrt{M\log(Mn/\varepsilon)})$, capped at $M-1$ interior modes. This is an accuracy allocation, not a complexity bound for computing the power.

### 6.2 The implemented reduced power is a separate approximation

Let $\widehat E_r$ denote sampled coefficients and $\widehat H$ the padded rational assembly. The implemented backward action, before power-algorithm and arithmetic errors, is

$$
D_r\widehat H^{\,n-1}\widehat E_r.
$$

In general $\widehat H\ne\widehat E_rD_r$: assembly uses analytic coefficient identities, while the external transform uses quadrature. Consequently, one cannot simply insert the sampled one-step error (15) into (25) and call it a certificate for the implemented multi-generation action.

A direct bound, with induced maximum norms throughout and $t=n-1$, is

$$
\begin{aligned}
\|D_r\widehat H^t\widehat E_r-D_rH^tE_r\|_\infty
\le{}&\|D_r\|_\infty\|\widehat H^t\|_\infty
       \|\widehat E_r-E_r\|_\infty\\
&+\|D_r\|_\infty\|E_r\|_\infty
\sum_{j=0}^{t-1}\|\widehat H^{t-1-j}\|_\infty
\|\widehat H-H\|_\infty\|H^j\|_\infty.
\end{aligned}
\tag{26}
$$

The sum is empty when $t=0$. This bound can be pessimistic, but exposes the missing factors. For example, an $r\times r$ Frobenius bound converts to a maximum-norm bound with a factor at most $\sqrt r$; the full reduced matrix also includes the forcing column. Neither eigenvalues nor an unweighted coefficient residual alone control these amplification factors.

A rational Krylov approximation introduces an additional power-action error. For forward propagation, that error must be measured after decoding, or bounded using a decoder norm. Immigration introduces a sum of propagated source terms. Neither extension is automatically covered by a one-step rank tolerance. Parameter-derivative and relative rare-event errors likewise need separate analyses.

## 7. Comparison with the filtered Chebyshev result

The two methods share the Gaussian suppression of high polynomial degrees, but the proved certificates differ. Chebyshev filtering in the companion note controls the sampled operator directly in the row norm. The Jacobi argument here passes through weighted Hilbert spaces, endpoint division, and a pointwise polynomial bound.

There is a common polynomial-degree obstruction. If $r+2\le M$, every approximation with output degree at most $r+1$ in the offspring coordinate has uniform row error at least

$$
\lambda_{M,r+2}=\prod_{j=0}^{r+1}\left(1-\frac jM\right).
\tag{27}
$$

To see this, use the bounded input $f_j=T_{r+2}(z_j)$. The Bernstein expectation multiplies its leading coefficient by $\lambda_{M,r+2}$, and subtracting any degree-$(r+1)$ polynomial leaves that leading coefficient unchanged. Chebyshev equioscillation gives (27) [5]. This is a continuum degree lower bound; it is not a lower bound for every factorization on a particular finite set of source probabilities. It says nothing additional at the exact rank $r=M-1$.

A filtered Jacobi construction is a reasonable next object to study, but would change the operator. Uniform stability of filtered Jacobi means is known in suitable weighted spaces [6]; matching those hypotheses to endpoint reconstruction and Gauss–Jacobi sampling is necessary before claiming a dimension-independent row bound. In particular, the Chebyshev taper and its constant cannot simply be transferred to the present sharp Jacobi truncation.

The useful conclusions for the current backend are therefore the exact singular values (4), the finite-sum row certificate (9), the sampled one-step budget (15), and the separate assembly and propagation estimates (21) and (26). They identify which errors are controlled by rank, quadrature order, padding, and the power algorithm, respectively.

## References

1. NIST Digital Library of Mathematical Functions. [§18.3, classical orthogonal-polynomial normalization](https://dlmf.nist.gov/18.3), [§18.9, recurrence relations](https://dlmf.nist.gov/18.9), and [§18.14, inequalities](https://dlmf.nist.gov/18.14). Jacobi/Gegenbauer normalization, recurrence, and endpoint bounds.
2. NIST Digital Library of Mathematical Functions. [§18.19, Hahn-class definitions](https://dlmf.nist.gov/18.19), and [§18.22, recurrence relations](https://dlmf.nist.gov/18.22). The discrete orthogonal basis and its coordinate matrix.
3. Diaconis, P., Khare, K., and Saloff-Coste, L. (2008). *Gibbs Sampling, Exponential Families and Orthogonal Polynomials*. Statistical Science **23**(2), 151–178. [doi:10.1214/07-STS252](https://doi.org/10.1214/07-STS252); [arXiv:0808.3852](https://arxiv.org/abs/0808.3852). Conditional expectation and the beta-binomial/Hahn–Jacobi correspondence.
4. NIST Digital Library of Mathematical Functions. [§3.5(v), Gauss quadrature](https://dlmf.nist.gov/3.5#v). Positive weights and polynomial exactness.
5. Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice*. SIAM. [doi:10.1137/1.9781611972405](https://doi.org/10.1137/1.9781611972405). Polynomial approximation on Bernstein ellipses and Chebyshev equioscillation; see also the author's [Chebfun guide, Chapter 4](https://www.chebfun.org/docs/guide/guide04.html).
6. Themistoclakis, W. (2012). *Uniform approximation on $[-1,1]$ via discrete de la Vallée Poussin means*. Numerical Algorithms **60**, 593–612. [doi:10.1007/s11075-012-9588-4](https://doi.org/10.1007/s11075-012-9588-4); [CNR research record](https://iris.cnr.it/handle/20.500.14243/244101). Discrete filtered approximation using Jacobi nodes and weighted uniform norms.

These references supply the classical ingredients. The explicit constants, norm conversions, and implementation distinctions in this note are derived above; they should not be read as claims that the cited papers analyze this particular backend.
