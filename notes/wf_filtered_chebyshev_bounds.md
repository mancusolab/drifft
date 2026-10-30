# Wright–Fisher FFT/NUFFT compression: sharper row-$\ell_1$ bounds

## Notation crosswalk

This derivation retains its local symbols. The [shared documentation notation](../docs/guide/notation.md)
uses the following corresponding quantities.

| This note | Shared docs convention |
| --- | --- |
| Angular coordinate $\theta$ | $\vartheta$ |
| Retained mode count $L$ | $n_{\mathrm{modes}}$; output degree is $L_{\mathrm{docs}}=L-1$ |
| Filter weights $v_k$ | Local taper weights, not population mass $v_t$ |
| Complex-strip displacement $t$ | Local analytic variable, not a generation index |

## Scope and status

These are derivations for the Bernstein-to-Chebyshev codec, not for rational Jacobi reduction. The explicit WF-specific bounds below are derived here. They do not claim optimal numerical constants or a lower bound for every possible low-rank representation. The reference Python file checks finite sets of rows; those checks are not substitutes for the proof and exclude a production JAX implementation.

The important algorithmic distinction is between a sharp Chebyshev projection, ordinary Chebyshev–Lobatto interpolation, and the tapered sampled Fourier/Chebyshev operator defined below. The dimension-independent Gaussian error bound is for the tapered operator. This operator is linear and preserves constants, but need not be positive.

## 1. Setup

Let $M$ be the offspring gene-copy count and define

$$
b_{M,j}(\theta)=\binom Mj\sin^{2j}(\theta/2)\cos^{2(M-j)}(\theta/2),\quad 0\le j\le M.
$$

Thus $q = \sin^2(\theta/2)$ and a Wright–Fisher row at success probability $q$ is $b_M(\theta)$. Source probabilities may incorporate arbitrary selection and mutation maps, provided they lie in $[0,1]$.

For an approximate row family define

$$
\delta=\sup_{\theta\in[0,\pi]}\sum_{j=0}^{M}|b_{M,j}(\theta)-\widetilde b_{M,j}(\theta)|.
$$

For every matrix obtained by evaluating this family at source probabilities,

$$
\|P-\widetilde P\|_{\infty\to\infty}\le\delta,
\qquad
\|P^T-\widetilde P^T\|_{1\to1}\le\delta.
$$

This is twice the maximum row total-variation distance when both kernels are stochastic. If the approximation is signed, it remains a valid row-$\ell_1$/operator guarantee, but should not be described as a probability distribution.

The key exact identity, valid for every real $\theta$ and real $t$, is

$$
\boxed{\|b_M(\theta+it)\|_1=(\cosh t)^M.}
$$

It follows by summing the absolute values with the binomial theorem and using
$|\sin((\theta+it)/2)|^2 + |\cos((\theta+it)/2)|^2 = \cosh(t)$.
No central-limit or diffusion approximation is involved.

## 2. Construction with $L$ retained modes

$L$ denotes the number of retained cosine modes, $0,\ldots,L-1$, so the output polynomial degree is at most $L-1$. The encoder samples $L+1$ Lobatto points.

Choose an integer taper width $1 \le w \le L$, set $a = L-w$, and define

$$
v_k = \max\left(0,\min\left(1,\frac{L-k}{w}\right)\right),
\qquad k\ge 0.
$$

On the full periodic angular grid $\theta_\ell = \pi\ell/L$, $\ell=0,\ldots,2L-1$, compute
$$
\widehat{b}^{d}_k =
\frac{1}{2L}
\sum_{\ell=0}^{2L-1}
b_M(\theta_\ell)e^{-ik\theta_\ell}.
$$

The approximate family is

$$
\widetilde{b}_{M,L,w}(\theta) =
\sum_{k=-(L-1)}^{L-1}
v_{|k|}\widehat{b}^{d}_k e^{ik\theta}.
$$

Evenness reduces this computation to a DCT-I of the $L+1$ distinct Lobatto samples. The final evaluation can use a cosine transform or a NUFFT. The approximation has rank at most $L$ on any collection of source probabilities. The taper is deterministic, independent of the input vector.

## 3. Explicit sampled-operator bound

For every $t > 0$,

$$
\boxed{
\delta_{M,L,w}\le
\frac{4\cosh^2(wt/2)}{w\sinh t\,[1-e^{-2Lt}]}
\exp\{M\log\cosh t-Lt\}.
}
\tag{1}
$$

This includes sampling aliasing, but excludes binomial-sample, DCT, NUFFT, and floating-point errors.

### 3.1 Continuous filtering error

Write $S_n$ for Fourier projection through frequencies $-n,\ldots,n$. The continuous filtered operator is

$$
V_{L,w}=\frac{1}{w}\sum_{n=a}^{L-1}S_n.
$$

Its positive-frequency residual multiplier $r_k = 1-v_k$ has generating function

$$
\sum_{k\ge1}r_k z^k
=\frac{z^{a+1}(1-z^w)}{w(1-z)^2}.
$$

Set $z = \exp(-t+iu)$. A contour shift expresses the positive-frequency residual as convolution of $b_M(\cdot-it)$ with this kernel; the negative-frequency residual uses $b_M(\cdot+it)$.

Normalize circular $L^1$ integrals by $1/(2\pi)$. Since

$$
\frac1{2\pi}\int_{-\pi}^{\pi}\frac{du}{|1-e^{-t+iu}|^2}
=\frac1{1-e^{-2t}},
$$

we obtain

$$
\|b_M-V_{L,w}b_M\|_{L^\infty(\ell_1)}
\le
\frac{e^{wt}+1}{w\sinh t}\,
\exp\{M\log\cosh t-Lt\}.
\tag{2}
$$

### 3.2 Sampling aliasing

The DFT coefficients satisfy

$$
\widehat b^d_k=\sum_{r\in\mathbb Z}\widehat b_{k+2Lr}.
$$

Let $V(u) = \sum_{|k|<L} v_{|k|}\exp(iku)$. Its Laurent-polynomial representation is

$$
V(z)=\frac{z^a+z^{-a}-z^L-z^{-L}}{w(1-z)(1-z^{-1})}.
$$

Consequently,

$$
\|V(\cdot+it)\|_{L^1}
\le\frac{\cosh(at)+\cosh(Lt)}{w\sinh t}.
$$

Shift the contour separately for positive and negative alias groups and sum their geometric decay. This gives

$$
\|\widetilde b_{M,L,w}-V_{L,w}b_M\|_{L^\infty(\ell_1)}
\le
2(\cosh t)^M\frac{e^{-2Lt}}{1-e^{-2Lt}}
\frac{\cosh(at)+\cosh(Lt)}{w\sinh t}.
\tag{3}
$$

Adding (2) and (3) and simplifying yields (1). In particular, no factor counting the number of discarded Fourier coefficients is introduced.

## 4. A dimension-independent Gaussian certificate

Assume $\lceil\sqrt{M}\rceil \le L < M$. Set

$$
t_*=\operatorname{arctanh}(L/M),\qquad
w=\left\lceil1/t_*\right\rceil.
$$

The condition ensures $w \le L$. Define

$$
\Psi(u)=\frac{1+u}{2}\log(1+u)+\frac{1-u}{2}\log(1-u).
$$

Then $M\log(\cosh(t_*)) - Lt_* = -M\Psi(L/M)$. Also $\Psi(u) \ge u^2/2$.

The prefactor in (1) is below 6 with this choice. To verify this, put $x = wt_*$. For $t_* \le 1$, $1 \le x \le 2$ and

$$
\frac{4\cosh^2(x/2)}{x}
\le 2+2\cosh(1)<5.087,
$$

while $t_*/\sinh(t_*) \le 1$ and $1-\exp(-2Lt_*) \ge 1-\exp(-2)$. Thus the prefactor is less than 5.89. For $t_* > 1$, $w=1$, and the same prefactor is at most $2\coth(1/2)/(1-\exp(-2)) < 5.01$.

Therefore

$$
\boxed{\delta_{M,L,w}\le6e^{-M\Psi(L/M)}\le6e^{-L^2/(2M)}.}
\tag{4}
$$

A sufficient number of modes for row-$\ell_1$ error $\varepsilon \in (0,1)$ is

$$
\boxed{L=\left\lceil\sqrt{2M\log(6/\varepsilon)}\right\rceil.}
$$

If this reaches $M$, use the exact degree-$M$ representation with $M+1$ modes, or treat the limiting top-degree case separately.

Thus the sufficient rank is $O(\sqrt{M\log(1/\varepsilon)})$, capped at $M+1$. At fixed accuracy it is $O(\sqrt{M})$; a dense reduced matrix has $O(M\log(1/\varepsilon))$ entries.

The taper width is approximately $M/L$. This matches the scale over which $\exp(-k^2/(2M))$ changes by a constant factor at $k=L$. A fixed fractional taper is unnecessary and would weaken the exponent relative to the final degree.

For a sharper numerical certificate, optimize the right-hand side of (1) over $t>0$ for each candidate integer $w$. The constant 6 is convenient, not asserted to be minimax-optimal.

## 5. Lower bound on the polynomial degree

Consider approximation of the whole binomial family uniformly in $q$, with outputs restricted to degree at most $L-1$ in $q$. Set

$$
x_j=T_L(1-2j/M),\qquad |x_j|\le1.
$$

The Bernstein expectation of this input has degree $L$, and its highest Chebyshev coefficient is

$$
\lambda_{M,L}=\frac{M(M-1)\cdots(M-L+1)}{M^L}.
$$

This follows because the Bernstein expectation multiplies the leading coefficient of a degree-$L$ polynomial by $(M)_L/M^L$. Any degree-$(L-1)$ approximation leaves this leading coefficient unchanged. Chebyshev equioscillation implies its uniform error is at least $\lambda_{M,L}$, and subtracting the lower-degree terms attains that lower bound for this particular input.

Consequently every approximation operator with this polynomial-degree restriction has uniform row-$\ell_1$/operator error at least

$$
\boxed{\lambda_{M,L}=\prod_{j=0}^{L-1}(1-j/M).}
\tag{5}
$$

For $L=o(M)$,

$$
\log\lambda_{M,L}
=-\frac{L(L-1)}{2M}+O(L^3/M^2).
$$

Thus the leading Gaussian exponent agrees with (4). In the compressed regime the degree scaling $\sqrt{M\log(1/\varepsilon)}$ cannot be improved in order for this representation. This is a degree lower bound for approximation of the full continuum of binomial rows, not a lower bound for every low-rank factorization or every particular selection map.

## 6. What is true for unfiltered methods?

### Continuous sharp projection

For the projection $S_{L-1}$, the positive-frequency tail kernel after contour shifting is

$$
\frac{e^{-Lt+iLu}}{1-e^{-t+iu}}.
$$

Hence

$$
\|b_M-S_{L-1}b_M\|_{L^\infty(\ell_1)}
\le2A(t)\exp\{M\log\cosh t-Lt\},
$$

where

$$
A(t)=\frac1{2\pi}\int_{-\pi}^{\pi}\frac{du}{|1-e^{-t+iu}|}
=O(1+\log(1/t))\quad(t\downarrow0).
$$

At $t=\operatorname{arctanh}(L/M)$ this replaces the old approximately $\sqrt{M}$ prefactor by a logarithmic prefactor. It yields a sufficient degree of order $\sqrt{M[\log(1/\varepsilon)+\log\log M]}$ at fixed-accuracy large-$M$ scaling, rather than $\sqrt{M\log(M/\varepsilon)}$.

A dimension-independent Gaussian bound is actually false for this literal sharp projection. Here is a counterexample rather than just a generic Lebesgue-constant warning.

Take $L/\sqrt{M} \to \alpha > 0$ and evaluate the approximated row at $q=0$. For every fixed $j\ge1$,

$$
(S_{L-1}b_{M,j})(0)\longrightarrow
h_j(\alpha)=\frac1{\pi j}\mathbb E[\sin(2\alpha\sqrt{Y_j})],
\quad Y_j\sim\operatorname{Gamma}(j,1).
$$

To obtain this limit, substitute $u=M\sin^2(\theta/2)$ into the Dirichlet-kernel integral. The binomial mass tends to $\exp(-u)u^j/j!$, and the sine factor tends to $\sin(2\alpha\sqrt{u})$. Equivalently the integral is a beta expectation whose scaled beta variable tends to $\operatorname{Gamma}(j,1)$.

A Taylor expansion of $\sqrt{Y_j}$ around $\sqrt{j}$, together with the exact gamma characteristic function, gives

$$
h_j(\alpha)=\frac{e^{-\alpha^2/2}}{\pi j}
\{\sin(2\alpha\sqrt j)+O_\alpha(j^{-1/2})\}.
$$

Moreover

$$
\sum_{j\le J}\frac{|\sin(2\alpha\sqrt j)|}{j}
=\frac2\pi\log J+O_\alpha(1).
$$

Thus $\sum_j |h_j(\alpha)|$ diverges. Since the true $q=0$ row has zero probability at every $j\ge1$, considering any fixed initial segment of columns and then letting that segment grow proves that the sharp-projection row-$\ell_1$ error is unbounded as $M$ increases with $L/\sqrt{M}\to\alpha$. This disproves a uniform $C\exp(-cL^2/M)$ estimate for that operator. This argument does not by itself give the precise finite-$M$ rate of divergence or a lower bound for ordinary DCT interpolation, which preserves endpoint samples.

### Ordinary Lobatto interpolation

Let $I_L$ be the degree-$L$ interpolant and $\Lambda_L$ its Lebesgue constant. Since it reproduces degree-$L$ polynomials,

$$
\|F-I_LF\|_\infty\le(1+\Lambda_L)E_L(F),\qquad
\Lambda_L\le1+\frac2\pi\log(L+1).
$$

The continuous tapered construction with cutoff $L+1$ gives

$$
E_L(F_x)\le4.2e^{-M\Psi((L+1)/M)}\|x\|_\infty,
$$

when $\sqrt{M} \le L+1 < M$. Therefore the unchanged ordinary DCT interpolation also has an improved $O(\log L\,\exp(-L^2/(2M)))$ certificate. The dimension-independent constant in (4), however, is established for the sampled tapered operator.

## 7. Numerical errors, positivity, endpoints, and powers

### Sample errors

Let $\Lambda_{L,w}$ be the norm mapping errors in the $2L$ periodic samples to the reconstructed polynomial. It is enough to use

$$
\Lambda_{L,w}\le\min\{(2L-w)/w,\;5+\log(L/w)\}.
$$

The first bound follows from writing the filter as a difference of Fejer kernels, whose discrete sums are positive and normalized. For the second, use

$$
V(u)=\frac{\sin((2L-w)u/2)\sin(wu/2)}{w\sin^2(u/2)}
$$

and bound it by $\min(2L, \pi/|u|, \pi^2/(wu^2))$, then sum over angular grid annuli of width $\pi/L$. At most two grid points lie in each annulus. The first annulus contributes at most 2, the harmonic portion at most $1+\log(L/w)$, and the remaining quadratic tail at most 2.

If every sampled binomial row is approximated in $\ell_1$ to $\tau$, its contribution to the final operator error is at most $\Lambda_{L,w}\tau$. Because $L/w$ is approximately $L^2/M$, the amplification does not grow with $M$ at fixed target accuracy.

### Conditional NUFFT error

For input norm $\|x\|_\infty\le1$, exact sampled Bernstein expectations have absolute value at most one. Parseval and Cauchy–Schwarz give, for the complex Fourier coefficients entering the evaluation,

$$
\sum_{k=-(L-1)}^{L-1}
\left|v_{|k|}\widehat{F}^{d}_k\right|
\le
\sqrt{2L-\frac{4w}{3}+\frac{1}{3w}}
\le
\sqrt{2L-1}.
$$
If a NUFFT has output error at most $\eta$ times the coefficient $\ell_1$ norm, the combined certificate is

$$
\delta_{\mathrm{impl}}\le\delta_{M,L,w}
+\Lambda_{L,w}\tau
+\eta\sqrt{2L-1}(1+\tau)
+\delta_{\mathrm{arithmetic}}.
$$

A library tolerance is not automatically a proof of this assumed NUFFT error model.

### Mass and positivity

The construction preserves constants, hence row sums are one. It need not preserve positivity. If the exact row is stochastic and its $\ell_1$ error is at most $\delta$, the approximate row's total negative mass is at most $\delta/2$. Clipping and renormalizing entries is not a free repair: it can destroy the low-rank structure and changes the fixed linear operator.

### Endpoints

For $L\ge2$, exact endpoints can be restored by

$$
\widetilde b^{\mathrm{end}}(q)=\widetilde b(q)
+(1-q)[b(0)-\widetilde b(0)]
+q[b(1)-\widetilde b(1)].
$$

The degree and rank bound do not increase, row sums remain one, and the error bound increases by at most a factor two. This preserves absorbing endpoints where appropriate. Low-degree reproduction requires those degrees to lie in the filter's flat passband.

### Powers

For a fixed square transition with one-step error $\delta$, $\|P\|_\infty=1$ and $\|\widetilde P\|_\infty\le1+\delta$. Thus

$$
\|P^K-\widetilde P^K\|_\infty\le(1+\delta)^K-1.
$$

Taking $\delta \le \log(1+\varepsilon_{\mathrm{total}})/K$ is sufficient. Consequently the conservative rank for $K$ generations scales as $\sqrt{M\log(K/\varepsilon_{\mathrm{total}})}$, with no added $\log M$. This is an accuracy statement, not a fast-powering algorithm or a uniform relative bound on rare probabilities. Value-error bounds do not automatically certify parameter derivatives.

## 8. Scope relative to FFT, Toeplitz, and fastDTWF

The theorem concerns angular Fourier/Chebyshev compression. Uniform versus nonuniform output angles changes the evaluation algorithm, not this approximation bound. The local log-odds diagonal–Toeplitz–diagonal factorization remains exact on its grid; it is not itself a rank reduction, and the theorem does not remove its interpolation or scaling errors.

Count-grid Fourier truncation is different. At $q=0$ the row is a delta mass whose discrete Fourier transform has magnitude one at every frequency. Dropping even one frequency gives row-$\ell_1$ error at least one. Even after handling exact endpoints separately, $q=1/M$ has all characteristic-function magnitudes at least $(1-2/M)^M$, approaching $\exp(-2)$. Thus the angular transform is essential to the global small-rank claim; a naive count-grid Fourier truncation cannot simply inherit it.

Spence et al. prove fixed-tolerance compression with $O(\sqrt{M})$ representative rows and use concentration/sparsity to obtain $O(M)$ matvecs. The filtered codec now matches the fixed-tolerance rank scaling, with logarithmic precision dependence. It is not claimed to be faster, and it does not inherit positivity or the practical moment-matching behavior of fastDTWF.

## References for standard ingredients

- Spence, Zeng, Mostafavi, and Pritchard (2023). Scaling the discrete-time Wright–Fisher model to biobank-scale datasets. Genetics 225, iyad168.
- Occorsio and Themistoclakis (2021). Some remarks on filtered polynomial interpolation at Chebyshev nodes. Dolomites Research Notes on Approximation 14, 68–84; arXiv:2101.04551.
- Trefethen. Approximation Theory and Approximation Practice. Chapters 4, 10, and 15: aliasing, equioscillation, and Lebesgue constants.
- Deregowska and collaborators. On the Norms and Minimal Properties of de la Vallée Poussin's type Operators. Background on Fourier projection norms and averaged projections.

The explicit WF formulas (1)–(5), the stated sampled-codec certificate, and the sharp-cutoff counterexample are supplied as derivations in this note, not attributed to those references.
