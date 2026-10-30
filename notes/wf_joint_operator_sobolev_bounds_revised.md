# Joint one-step and multi-step error bounds for filtered DCT approximation of the Wright–Fisher operator

*A supplemental note — revised 24 October 2026*

## 1. Motivation and scope

The numerical objective is to approximate the finite-population, discrete-generation Wright–Fisher transition matrix and its selection derivative using **the same fixed compressed operator**. For a single generation, the requested guarantees are

\[
\|P_s-\widetilde P_s\|_{\infty\to\infty}\le\delta,
\qquad
\|\partial_sP_s-\partial_s\widetilde P_s\|_{\infty\to\infty}\le\eta.
\tag{1.1}
\]

For an epoch of \(K\) generations, the corresponding targets are

\[
\|P_s^K-\widetilde P_s^K\|_{\infty\to\infty},
\qquad
\|\partial_s(P_s^K)-\partial_s(\widetilde P_s^K)\|_{\infty\to\infty}.
\tag{1.2}
\]

These are **absolute operator errors**, uniform over all bounded input observables. They are not average errors over allele frequencies, errors restricted to smooth inputs, or errors divided by a population-size-dependent sensitivity scale. By transposition, the same bounds control forward distributions and their derivatives in vector \(\ell_1\).

The analysis has two parts. Analytic Sobolev estimates in a variance-stabilized coordinate control the one-step filtered DCT approximation. Then a perturbation expansion controls its powers. For the gradient of a power, we improve the usual product-rule bound using the Fisher information of a Wright–Fisher trajectory: score increments are martingale differences, so their variances add rather than their absolute magnitudes.

The resulting multi-step theorem allows the compressed matrix to have signed entries. It assumes neither a stationary distribution nor reversibility, mixing, a diffusion approximation, or normality of the reduced matrix. Thus it applies to recurrent-mutation models and to selection models with parameter-independent absorbing endpoint rows.

This revision retains the one-step construction in the preceding project note, tightens its convenient angular-derivative constant from 15 to 9, and adds a separate multi-step analysis. The standard harmonic-analysis and path-information ingredients are identified in the references; the combined certificates and their derivations are given here.

## 2. Definitions and assumptions

### 2.1 The transition matrix and the two norms

Let \(M\ge2\) be the number of destination gene copies. Let \(p_i=i/M_s\), \(i=0,\ldots,M_s\), denote the source frequencies. The one-step matrix can be rectangular. Powers require a constant-size epoch with \(M_s=M\).

The full frequency map, including selection, mutation, and their prescribed order, produces a probability \(q_s(p_i)\in[0,1]\). Set

\[
b_{M,j}(q)=\binom Mj q^j(1-q)^{M-j},
\qquad
(P_s)_{ij}=b_{M,j}(q_s(p_i)).
\tag{2.1}
\]

An observable \(x\in\mathbb R^{M+1}\) is a column vector. Its backward expectation is \(P_sx\). Throughout,

\[
\|A\|:=\|A\|_{\infty\to\infty}
=\max_i\sum_j|A_{ij}|,
\qquad
\|x\|_\infty=\max_j|x_j|.
\tag{2.2}
\]

Thus \(\|Ax\|_\infty\le\|A\|\|x\|_\infty\) and \(\|A^T\|_{1\to1}=\|A\|\). The exact matrix is stochastic, so \(\|P_s\|=1\). The approximation will preserve constants but need not be positive. Its row-\(\ell_1\) error should not be called twice a total-variation distance unless both rows are probability distributions.

We treat \(x\), population sizes, the integer \(K\), and all codec dimensions as independent of \(s\). A prime on a matrix denotes differentiation with respect to \(s\); a prime on a function of \(\theta\) denotes angular differentiation. The arguments make the distinction explicit.

### 2.2 Angular coordinates and sensitivity

Use

\[
q=\sin^2(\theta/2),\qquad
\theta_s(p)=2\arcsin\sqrt{q_s(p)},\qquad 0\le\theta\le\pi.
\tag{2.3}
\]

Define the row-valued kernel and scalar expectation

\[
B_M(\theta)=\bigl(b_{M,j}(\sin^2(\theta/2))\bigr)_{j=0}^M,
\qquad F_x(\theta)=B_M(\theta)x.
\tag{2.4}
\]

These extend to even, entire, \(2\pi\)-periodic trigonometric polynomials of degree at most \(M\). Selection changes the evaluation points, not this kernel:

\[
(P_sx)_i=F_x(\theta_s(p_i)).
\]

At interior probabilities,

\[
\dot\theta_s(p_i)=
\frac{\partial_s q_s(p_i)}{\sqrt{q_s(p_i)(1-q_s(p_i))}}.
\]

We assume the angular map is differentiable and

\[
\Theta_s:=\max_i|\dot\theta_s(p_i)|<\infty,
\qquad A_s:=\sqrt M\,\Theta_s.
\tag{2.5}
\]

Constant absorbing endpoint rows have derivative zero; their angular sensitivity is taken to be zero. For the path-score arguments, the support of each exact transition row must be locally fixed in \(s\). This holds when all nonconstant rows have probabilities in \((0,1)\), while any endpoint rows are identically absorbing throughout a parameter neighborhood. A mutation parameter evaluated at a support-changing boundary requires a separate argument and is not covered by silently substituting a singular score.

For a guarantee throughout a parameter interval, use a uniform bound \(\Theta_*\ge\sup_s\Theta_s\). The codec must remain fixed on that interval. Selecting a new integer rank or taper separately at each parameter value changes the differentiated algorithm.

### 2.3 Notation guide

| Symbol | Meaning |
|---|---|
| \(M_s,M\) | Source and destination gene-copy counts; equal within a constant-size epoch |
| \(s\) | Selection parameter; all other parameters may be included in the frequency map |
| \(L\) | Number of retained cosine modes; output degree at most \(L-1\) |
| \(w\) | Number of modes in the linear taper; \(a=L-w\) |
| \(\Theta_s\) | Maximum angular sensitivity of the frequency map |
| \(A_s=\sqrt M\Theta_s\) | Uniform square root of one-generation Fisher information |
| \(\delta,\eta\) | Upper bounds on one-step value and selection-derivative errors |
| \(K\) | Number of generations, held fixed when differentiating |
| \(r=1+\delta\) | A bound on \(\|\widetilde P_s\|\); **not a polynomial rank** |
| \(\mathcal E_K,\mathcal G_K\) | Multi-step value and derivative error bounds |

## 3. The fixed filtered DCT approximation

Fix integers \(1\le w\le L\), and let \(a=L-w\). Define

\[
v_k=\begin{cases}
1,&0\le k\le a,\\
(L-k)/w,&a<k<L,\\
0,&k\ge L.
\end{cases}
\tag{3.1}
\]

Sample the kernel at \(\theta_\ell=\pi\ell/L\), \(\ell=0,\ldots,2L-1\), and form

\[
\widehat B_k^{\,d}=\frac1{2L}\sum_{\ell=0}^{2L-1}
B_M(\theta_\ell)e^{-ik\theta_\ell},
\qquad
\widetilde B(\theta)=\sum_{|k|<L}v_{|k|}\widehat B_k^{\,d}e^{ik\theta}.
\tag{3.2}
\]

Evenness means that only the \(L+1\) distinct Lobatto samples on \([0,\pi]\) are needed: this is a DCT-I, a diagonal taper, and cosine-series evaluation. Define

\[
(\widetilde P_sx)_i=\widetilde B(\theta_s(p_i))x.
\tag{3.3}
\]

This is a fixed linear, rank-at-most-\(L\) approximation. The same encoder produces values and derivatives. In exact arithmetic,

\[
\widetilde P_s\mathbf1=\mathbf1,
\qquad \widetilde P_s'\mathbf1=0,
\]

because the samples of the constant observable are constant and \(v_0=1\). These identities do not imply positivity or exact absorbing endpoint rows.

The analysis initially assumes exact sampling and arithmetic. Section 8 separates spectral error from numerical implementation and reduced-power-action errors.

## 4. One-step operator bounds

### 4.1 Why Sobolev scaling is intrinsic, but not an accuracy substitution

For \(J\sim\operatorname{Bin}(M,q)\), the angular score is

\[
Z_\theta(J)=\partial_\theta\log b_{M,J}(q)
=\frac{J-Mq}{\sqrt{q(1-q)}}.
\]

It has mean zero and variance \(M\). Therefore \(1\) and \(Z_\theta/\sqrt M\) are orthonormal in the binomial \(L^2\) space. Bessel's inequality gives

\[
\boxed{
|F_x(\theta)|^2+\frac1M|F_x'(\theta)|^2
\le\mathbb E_q|x_J|^2\le\|x\|_\infty^2.
}
\tag{4.1}
\]

This explains the information-geometric scaling \(M^{-1/2}\partial_\theta\). In particular,

\[
\|P_s'\|\le A_s.
\tag{4.2}
\]

The rescaled Sobolev norm

\[
\|F\|_{W_M^{1,\infty}}=\|F\|_\infty+M^{-1/2}\|F'\|_\infty
\]

is useful for analysis. Our final tolerances, however, remain the two **unscaled** matrix norms in (1.1).

### 4.2 Complex-strip estimates and analytic Sobolev regularity

For \(z=\theta+it\), the binomial theorem gives

\[
\boxed{\|B_M(\theta+it)\|_1=(\cosh t)^M.}
\tag{4.3}
\]

To bound the derivative, write

\[
a_j(z)=\sqrt{\binom Mj}\sin^j(z/2)\cos^{M-j}(z/2),
\qquad B_{M,j}(z)=a_j(z)^2.
\]

The vector \(a\) is the symmetric tensor power of \((\cos(z/2),\sin(z/2))\). Direct differentiation gives

\[
\|a\|_2^2=(\cosh t)^M,
\quad
\|a'\|_2^2=\frac M4(\cosh t)^M[1+(M-1)\tanh^2t].
\]

Using \(\sum_j|2a_ja_j'|\le2\|a\|_2\|a'\|_2\), we obtain

\[
\boxed{
\|B_M'(\theta+it)\|_1\le A_M(t)(\cosh t)^M,
\qquad A_M(t)=\sqrt{M[1+(M-1)\tanh^2t]}.
}
\tag{4.4}
\]

There is also an exact joint scalar inequality behind these separate bounds. For \(t>0\), let \(\nu_j=|B_{M,j}(z)|/(\cosh t)^M\) and \(S_j=B_{M,j}'(z)/B_{M,j}(z)\). Direct calculation gives

\[
\mathbb E_\nu S=-iM\tanh t,
\qquad
\mathbb E_\nu|S+iM\tanh t|^2=M\operatorname{sech}^2t.
\]

Apply Bessel's inequality to \(x_jB_{M,j}(z)/|B_{M,j}(z)|\) and the centered score. This yields

\[
\boxed{
|F_x(z)|^2+\frac{\cosh^2t}{M}
|F_x'(z)+iM\tanh t\,F_x(z)|^2
\le(\cosh t)^{2M}\|x\|_\infty^2.
}
\tag{4.5}
\]

At \(t=0\), this reduces to (4.1). Maximizing a linear functional over the ellipse in (4.5) recovers (4.4), so the complex derivative scale is also intrinsically derived.

For the conventional analytic Sobolev norm

\[
\|F\|_{\mathcal A^1_{t,M}}^2
=\sum_{k\in\mathbb Z}\left(1+\frac{k^2}{M}\right)e^{2t|k|}|\widehat F_k|^2,
\]

Parseval on the two displaced contours gives the convenient consequence

\[
\|F_x\|_{\mathcal A^1_{t,M}}^2
\le2[2+(M-1)\tanh^2t](\cosh t)^{2M}\|x\|_\infty^2.
\tag{4.6}
\]

These estimates jointly control values and derivatives. To obtain good uniform approximation constants, we next use the cancellation of the filtered Fourier kernel rather than a generic Sobolev embedding of independently bounded coefficients.

### 4.3 The continuous filtering remainder

Let \(V\) be the continuous Fourier multiplier with weights \(v_{|k|}\), and let \(V^d\) denote the sampled version (3.2). The positive-frequency remainder has generating function

\[
\sum_{k\ge1}(1-v_k)z^k
=\frac{z^{a+1}(1-z^w)}{w(1-z)^2}.
\tag{4.7}
\]

For an analytic periodic function \(G\) with values in a Banach space, write

\[
B_G(t)=\max_{\pm}\sup_\theta\|G(\theta\pm it)\|.
\]

Shift positive and negative Fourier contours in opposite directions. The normalized circular integral

\[
\frac1{2\pi}\int_{-\pi}^{\pi}\frac{du}{|1-e^{-t+iu}|^2}
=\frac1{1-e^{-2t}}
\]

and (4.7) imply

\[
\|(I-V)G\|_\infty\le c_c(t,w)e^{-Lt}B_G(t),
\qquad
c_c(t,w)=\frac{e^{wt}+1}{w\sinh t}.
\tag{4.8}
\]

Integrating this kernel as a whole, rather than summing worst-case bounds for each coefficient, avoids the earlier unnecessary factor of order \(M/L\).

### 4.4 Aliasing from the sampled DCT

The sampled coefficients satisfy

\[
\widehat G_k^{\,d}=\sum_{m\in\mathbb Z}\widehat G_{k+2Lm}.
\tag{4.9}
\]

Let \(V(u)=\sum_{|k|<L}v_{|k|}e^{iku}\) also denote the multiplier's kernel. Its Laurent representation is

\[
V(z)=\frac{z^a+z^{-a}-z^L-z^{-L}}{w(1-z)(1-z^{-1})}.
\]

Consequently, on a displaced contour,

\[
\|V(\cdot+it)\|_{L^1}
\le\frac{\cosh(at)+\cosh(Lt)}{w\sinh t},
\]

where circular integrals are normalized by \(1/(2\pi)\). Shifting the contours for the two signs of alias index and summing their geometric factors gives

\[
\|(V^d-V)G\|_\infty
\le c_a(t,w)e^{-Lt}B_G(t),
\tag{4.10}
\]

with

\[
c_a(t,w)=
\frac{2e^{-Lt}[\cosh((L-w)t)+\cosh(Lt)]}
{w\sinh t[1-e^{-2Lt}]}.
\tag{4.11}
\]

Both (4.8) and (4.10) apply to vector-valued kernels in \(\ell_1\). No Gaussian approximation to a binomial distribution has been used.

### 4.5 Finite-parameter value and derivative certificates

Define

\[
R_M(L,t)=\exp\{M\log\cosh t-Lt\}.
\]

Combining (4.3), (4.8), and (4.10) yields

\[
\boxed{
\delta_0(M,L,w;t)
=[c_c(t,w)+c_a(t,w)]R_M(L,t).
}
\tag{4.12}
\]

The sum simplifies to

\[
c_c+c_a=\frac{4\cosh^2(wt/2)}{w\sinh t[1-e^{-2Lt}]}.
\]

For derivatives, split

\[
(B_M-V^dB_M)'=(I-V)B_M'+\partial_\theta(V-V^d)B_M.
\]

The continuous multiplier commutes with differentiation. The alias error is a trigonometric polynomial of degree at most \(L-1\). Bernstein's inequality, applied by scalar duality to this Banach-valued polynomial, gives

\[
\|\partial_\theta(V-V^d)B_M\|_\infty
\le(L-1)\|(V-V^d)B_M\|_\infty.
\]

The scalar Bernstein inequality and its integral representations are reviewed in [1]; scalar duality gives the vector-valued version used here. Therefore,

\[
\boxed{
\delta_1(M,L,w;t)
=[c_c(t,w)A_M(t)+(L-1)c_a(t,w)]R_M(L,t).
}
\tag{4.13}
\]

The resulting **two one-step operator bounds** are

\[
\boxed{
\|P_s-\widetilde P_s\|\le\delta_0(M,L,w;t),
\qquad
\|P_s'-\widetilde P_s'\|\le\Theta_s\delta_1(M,L,w;t).
}
\tag{4.14}
\]

These hold for any admissible fixed \(L,w\) and every \(t>0\). The strip height \(t\) is only a proof parameter. For a fixed codec, the value and derivative bounds can be minimized over \(t\) separately without changing the approximation. Changing \(w\), in contrast, changes the codec and must be done consistently for both targets.

The derivative estimate is independently proved; it is **not** obtained by differentiating a bound on the value error.

### 4.6 A simple rank rule and a sharper derivative constant

Define

\[
\Psi(u)=\frac{1+u}{2}\log(1+u)+\frac{1-u}{2}\log(1-u).
\]

Since \(\Psi''(u)=1/(1-u^2)\ge1\) and \(\Psi(0)=\Psi'(0)=0\),

\[
\Psi(u)\ge\frac{u^2}{2}.
\]

For \(\lceil\sqrt M\rceil\le L<M\), choose

\[
t_* =\operatorname{arctanh}(L/M),
\qquad w=\lceil1/t_*\rceil,
\qquad E_L=e^{-M\Psi(L/M)}.
\tag{4.15}
\]

Here \(w\le L\), because \(Lt_*\ge L^2/M\ge1\). Also,

\[
R_M(L,t_*)=E_L\le e^{-L^2/(2M)}.
\]

**Theorem 1 — joint one-step accuracy.** With this fixed sampled filter,

\[
\boxed{
\|P_s-\widetilde P_s\|\le6E_L,
\qquad
\|P_s'-\widetilde P_s'\|\le9\Theta_sLE_L.
}
\tag{4.16}
\]

**Proof of the constants.** First suppose \(t_*\le1\), and put \(b=wt_*\in[1,2]\). The value prefactor is at most

\[
\frac{1}{1-e^{-2}}
\max_{1\le b\le2}\frac{2+2\cosh b}{b}
=\frac{2+2\cosh1}{1-e^{-2}}<6.
\]

For the derivative, do not discard the factor \(b\) in the denominator. The function \((e^b+1)/b\) has only one interior critical point, a minimum, so

\[
c_c\le\max_{1\le b\le2}\frac{e^b+1}{b}
=\frac{e^2+1}{2}.
\]

Moreover,

\[
A_M(t_*)\le\sqrt{M+L^2}\le\sqrt2L,
\qquad
c_a\le\frac{2\coth(Lt_*)}{w\sinh t_*}\le2\coth1.
\]

Thus the derivative prefactor is bounded by

\[
\left[\frac{\sqrt2(e^2+1)}2+2\coth1\right]L<9L.
\]

If \(t_*>1\), then \(w=1\). In that case

\[
c_c=\frac2{1-e^{-t_*}},\qquad
c_a\le\frac{2\coth(Lt_*)}{\sinh t_*}.
\]

The same estimates give a derivative constant below 7 and a value constant below 6. This proves (4.16). The improvement from 15 to 9 changes a convenient sufficient constant, not the exponential approximation rate. □

Equivalently, the joint rescaled Sobolev operator error obeys

\[
\|\mathcal B_M-\widetilde{\mathcal B}_{M,L}\|
_{\ell_\infty\to W_M^{1,\infty}}
\le(6+9L/\sqrt M)E_L.
\tag{4.17}
\]

Equation (4.16), rather than its rescaled summary, remains the absolute-error contract.

### 4.7 Selection maps, endpoints, and parameter directions

For symmetric underdominance, write \(z=2p-1\) and

\[
g_s(z)=\frac{z}{D_s(z)},\quad
D_s(z)=1-\frac s2(1-z^2),\quad
q_s(p)=\frac{1+g_s(2p-1)}2.
\]

For \(0\le s<1\), direct differentiation gives

\[
|\dot\theta_s(z)|=
\frac{|z|\sqrt{1-z^2}}
{2D_s(z)\sqrt{1-s+(s^2/4)(1-z^2)}}.
\tag{4.18}
\]

The endpoint limits are zero. At neutrality, the continuous-frequency maximum is \(1/4\), so the source-grid maximum satisfies \(\Theta_0\le1/4\). A convenient bound away from neutrality is

\[
\Theta_s\le\frac1{4(1-s/2)\sqrt{1-s}}.
\]

For mutation–selection compositions, compute \(\Theta_s\) for the full map in the specified order. The one-step approximation theorem needs only valid probabilities and bounded angular sensitivity; it does not require weak selection.

For \(s\in\mathbb R^d\), define the gradient norm by

\[
\|\nabla_s A\|_{2\to\mathrm{op}}
:=\sup_{\|v\|_2=1}\left\|\sum_{a=1}^dv_a\partial_{s_a}A\right\|.
\tag{4.19}
\]

All derivative bounds hold in this directional-Jacobian norm with
\(\Theta_s=\max_i\|\nabla_s\theta_s(p_i)\|_2\). This is not a sum of entrywise gradient norms. No extra factor equal to the number of parameters is needed for the norm in (4.19).

## 5. Multi-step bounds for the same compressed operator

We now assume a fixed square epoch, \(M_s=M\), and abbreviate

\[
P=P_s,\quad Q=\widetilde P_s,\quad E=Q-P,
\quad \|E\|\le\delta,\quad\|E'\|\le\eta,
\quad r=1+\delta.
\tag{5.1}
\]

The quantity \(\eta\) is a bound on \(E'\), not the derivative of the scalar bound \(\delta\). We have \(\|P\|=1\) and \(\|Q\|\le r\). The approximation \(Q\) is allowed to be signed.

### 5.1 Value error: a general finite-horizon bound

Expand \((P+E)^K\) as an ordered sum of noncommutative words. Words with exactly \(m\) factors \(E\) have norm at most \(\delta^m\), and there are \(\binom Km\) of them. The word with no \(E\) is \(P^K\). Consequently,

\[
\boxed{
\|P^K-Q^K\|\le\mathcal E_K(\delta):=(1+\delta)^K-1.
}
\tag{5.2}
\]

This holds for \(K\ge1\); \(\mathcal E_0=0\). It implies

\[
\mathcal E_K\le K\delta e^{K\delta}.
\]

It is possible to improve (5.2) using an independently established norm bound \(\|Q\|\le b\): telescoping gives \(\delta\sum_{j=0}^{K-1}b^j\). In particular a genuinely stochastic approximation gives \(K\delta\). We do not assume that additional property of the DCT approximation.

### 5.2 A baseline gradient bound

The same word expansion can be differentiated. Terms in which differentiation hits an error factor contribute at most

\[
K\eta r^{K-1}.
\]

Using only \(\|P'\|\le A_s\), the terms in which it hits an exact factor give

\[
KA_s(r^{K-1}-1).
\]

Thus a valid baseline is

\[
\boxed{
\|\partial_s(P^K)-\partial_s(Q^K)\|
\le K\eta r^{K-1}+KA_s(r^{K-1}-1).
}
\tag{5.3}
\]

For \(K\delta\ll1\), this behaves as
\(K\eta+A_sK(K-1)\delta\). The quadratic factor comes from bounding exact block sensitivities by the deterministic product rule. Wright–Fisher information geometry permits a better estimate.

### 5.3 Pathwise Fisher information bounds exact block sensitivities

Consider the exact chain \((X_0,\ldots,X_n)\), initialized at a fixed state \(i\) independently of \(s\). Its one-step score is

\[
\xi_t=\partial_s\log P_s(X_t,X_{t+1})
=\dot\theta_s(p_{X_t})
\frac{X_{t+1}-Mq_s(p_{X_t})}
{\sqrt{q_s(p_{X_t})(1-q_s(p_{X_t}))}}.
\tag{5.4}
\]

Set it to zero on parameter-independent absorbing rows. Conditional on \(X_0,\ldots,X_t\),

\[
\mathbb E[\xi_t\mid X_0,\ldots,X_t]=0,
\qquad
\mathbb E[\xi_t^2\mid X_0,\ldots,X_t]
=I_s(X_t),
\]

where

\[
I_s(i)=M\dot\theta_s(p_i)^2\le A_s^2.
\tag{5.5}
\]

Thus the score increments are orthogonal martingale differences. For the total path score \(S_n=\sum_{t=0}^{n-1}\xi_t\),

\[
\mathbb E_i S_n^2
=\sum_{t=0}^{n-1}(P^tI_s)_i\le nA_s^2.
\tag{5.6}
\]

Additivity of Markov path Fisher information and its use in observable-sensitivity inequalities are standard [2]. Equations (5.4)–(5.6) provide the specific finite-WF calculation, including the absorbing-row convention.

Differentiating a terminal expectation gives

\[
\partial_s(P^nx)_i=\mathbb E_i[x_{X_n}S_n].
\]

Cauchy–Schwarz therefore implies

\[
\boxed{
\|\partial_s(P^n)\|\le a_n
:=\left[\max_i\sum_{t=0}^{n-1}(P^tI_s)_i\right]^{1/2}
\le A_s\sqrt n.
}
\tag{5.7}
\]

The same orthogonality gives a joint pointwise energy inequality:

\[
|(P^nx)_i|^2+
\frac{|\partial_s(P^nx)_i|^2}{\mathbb E_i S_n^2}
\le(P^n|x|^2)_i,
\tag{5.8}
\]

whenever the denominator is positive; if it is zero, the derivative is zero. This is the multi-step counterpart of the one-step Bessel inequality (4.1).

The estimate \(A_s\sqrt n\) requires no mixing and no stationarity. The more detailed \(a_n\) can be smaller when the chain spends time in weak-sensitivity states, but computing it may itself require propagation. The simple theorem below does not require doing that computation.

### 5.4 A sharper signed-operator gradient-error theorem

**Theorem 2 — multi-step value and derivative errors.** Under the assumptions above, (5.2) holds and

\[
\boxed{
\|\partial_s(P^K)-\partial_s(Q^K)\|
\le K r^{K-1}\eta+A_sU_K(\delta),
}
\tag{5.9}
\]

where \(U_1=0\) and, for \(K\ge2\),

\[
\boxed{
U_K(\delta)=\delta\sum_{n=1}^{K-1}\sqrt n
\left[2r^{K-n-1}+(K-n-1)\delta r^{K-n-2}\right].
}
\tag{5.10}
\]

When \(K-n-1=0\), its second term is zero. If tighter exact-block sensitivity bounds \(a_n\) are available, replace \(A_s\sqrt n\) in (5.10) by \(a_n\).

**Proof.** Each error word can be grouped into exact blocks:

\[
P^{n_0}EP^{n_1}E\cdots EP^{n_m},
\qquad n_0+\cdots+n_m=K-m,\quad m\ge1.
\tag{5.11}
\]

If the derivative hits an \(E\), summing the \(m\) possible locations and all words gives

\[
\sum_{m=1}^K\binom Km m\eta\delta^{m-1}
=K\eta r^{K-1}.
\]

If it hits an exact block, (5.7) bounds that word's contribution by

\[
A_s\delta^m\sum_{j=0}^m\sqrt{n_j}.
\]

Fix one block length \(n\ge1\). There are \(m+1\) choices of that block, and
\(\binom{K-n-1}{m-1}\) choices for the remaining block lengths. Hence the total exact-block contribution is bounded by

\[
A_s\sum_{n=1}^{K-1}\sqrt n
\sum_{m=1}^{K-n}(m+1)\delta^m\binom{K-n-1}{m-1}.
\]

For \(N=K-n-1\), the inner sum is

\[
\delta\sum_{\ell=0}^{N}(\ell+2)\binom N\ell\delta^\ell
=\delta[2r^N+N\delta r^{N-1}],
\]

which proves (5.9)–(5.10). The proof uses only positivity of the **exact** Markov kernel to establish (5.7); no positivity of \(Q\) is needed. □

### 5.5 Closed forms and the improvement over the baseline

For a word with \(m\) error factors, Cauchy–Schwarz gives

\[
\sum_{j=0}^m\sqrt{n_j}\le\sqrt{(m+1)(K-m)}.
\]

Applying Cauchy–Schwarz again over the words yields

\[
U_K(\delta)\le
\sqrt{
K(r^K-1)
\left[r^{K-1}-1+(K-1)\delta r^{K-2}\right]
}.
\tag{5.12}
\]

Here we used

\[
\sum_{m=1}^K\binom Km\delta^m(m+1)(K-m)
=K\left[r^{K-1}-1+(K-1)\delta r^{K-2}\right].
\]

For a still simpler expression, use
\(r^K-1\le K\delta r^{K-1}\) and
\(r^{K-1}-1\le(K-1)\delta r^{K-2}\). Then

\[
\boxed{
\|\partial_s(P^K)-\partial_s(Q^K)\|
\le K r^{K-1}
\left[\eta+\sqrt{2(K-1)}\,A_s\delta\right].
}
\tag{5.13}
\]

For \(K=1\), the second term vanishes and the bound is \(\eta\). For \(K=0\), both value and derivative errors are zero.

When \(K\delta\) is small,

\[
U_K(\delta)
=2\delta\sum_{n=1}^{K-1}\sqrt n
+O(\delta^2K^{5/2}e^{K\delta}).
\]

Because \(\sum_{n=1}^{K-1}\sqrt n=\frac23K^{3/2}+O(\sqrt K)\), the derivative estimate behaves as

\[
\boxed{
\|\partial_s(P^K)-\partial_s(Q^K)\|
\lesssim K\eta+\frac43A_s\delta K^{3/2}.
}
\tag{5.14}
\]

This improves the value-error contribution from \(O(A_s\delta K^2)\) to \(O(A_s\delta K^{3/2})\). It does **not** remove accumulation altogether, and the factor \(r^{K-1}\) matters outside the small-\(K\delta\) regime. We do not claim that the power \(K^{3/2}\) is optimal for this particular filtered codec.

At very small \(K\), or when a much smaller direct bound on \(\|P'\|\) is known, the baseline (5.3) can be preferable. Both are valid; taking their minimum is legitimate.

### 5.6 The requested bounds after inserting the DCT error

For the filter in Theorem 1, take

\[
\delta_L=6E_L,
\qquad \eta_L=9\Theta_sLE_L,
\qquad A_s=\sqrt M\Theta_s.
\]

Equations (5.2) and (5.13) give the explicit pair

\[
\boxed{
\|P_s^K-\widetilde P_s^K\|
\le(1+6E_L)^K-1,
}
\tag{5.15}
\]

\[
\boxed{
\|\partial_s(P_s^K)-\partial_s(\widetilde P_s^K)\|
\le K\Theta_s(1+6E_L)^{K-1}
\left[9L+6\sqrt{2M(K-1)}\right]E_L.
}
\tag{5.16}
\]

The finite-strip certificates (4.12)–(4.14) can replace \(\delta_L,\eta_L\) for tighter numerical planning. Likewise, (5.9) is tighter than the convenient closed form (5.16).

Equations (4.16), (5.15), and (5.16) concern the **same sampled, filtered matrix**. They do not replace it by an exact polynomial restriction, an unfiltered Jacobi projection, or a diffusion semigroup.

## 6. Choosing one rank for value and gradient accuracy

Let \(\varepsilon_P\) and \(\varepsilon_s\) be the desired **absolute errors after \(K\) steps**. Choose a bound \(\Theta_*\) for the entire parameter range on which the codec will be differentiated. For each candidate \(L\), fix its taper, calculate one-step certificates \(\delta_L,\eta_L\), and require

\[
\boxed{
(1+\delta_L)^K-1\le\varepsilon_P,
\qquad
K(1+\delta_L)^{K-1}\eta_L+A_*U_K(\delta_L)
\le\varepsilon_s,
}
\tag{6.1}
\]

where \(A_*=\sqrt M\Theta_*\). The closed form (5.13) is an inexpensive sufficient replacement for the second condition. At \(K=1\), (6.1) reduces to the original separate one-step targets.

The value condition is equivalent to

\[
\delta_L\le\exp\!\left(\frac{\log(1+\varepsilon_P)}K\right)-1.
\tag{6.2}
\]

Use `log1p` and `expm1` to evaluate this threshold and the power-error bounds when errors are small. There is no need to allocate a derivative tolerance by guessing that value accuracy implies derivative accuracy.

Using the simple one-step certificates, a sufficient explicit pair of tests is

\[
K\log(1+6E_L)\le\log(1+\varepsilon_P),
\]

\[
K\Theta_*(1+6E_L)^{K-1}
[9L+6\sqrt{2M(K-1)}]E_L\le\varepsilon_s.
\tag{6.3}
\]

For fixed \(\Theta_*\), these convenient bounds decrease with \(L\) in the stated compression regime, so a scalar rank search is straightforward. A proof based on individually optimized \(t,w\) should evaluate its actual certificates rather than assume the same monotonicity across arbitrary taper changes.

If no compressed candidate meets the tolerances, use the exact degree-\(M\) representation with all \(M+1\) modes and no taper. That eliminates spectral truncation and sampling alias error in exact arithmetic, not floating-point error.

### 6.1 What improves numerically

The following sufficient mode counts were calculated from the sharper exponent \(M\Psi(L/M)\), with \(\Theta_*=1/4\) and both final tolerances equal to \(10^{-6}\). They describe a derivative certificate at neutrality; a broader selection interval needs its own \(\Theta_*\).

Both columns use the **same** one-step constants \((6,9)\). The difference is only whether the multi-step derivative bound uses the deterministic estimate (5.3) or the path-information estimate (5.9).

| Gene copies \(M\) | Generations \(K\) | Deterministic product-rule bound | Path-information bound |
|---:|---:|---:|---:|
| 6,000 | 100 | 578 modes | 562 modes |
| 6,000 | 10,000 | 666 modes | 627 modes |
| 64,000 | 100 | 1,929 modes | 1,878 modes |
| 64,000 | 10,000 | 2,211 modes | 2,085 modes |

These are sufficient ranks from the stated formulas, evaluated in ordinary floating-point arithmetic. They are not empirically minimal ranks, interval-arithmetic certificates, or runtime measurements. The reduction in rank is more modest than the reduction in the \(K\)-dependence of the error prefactor, because the approximation error decreases exponentially in \(L^2/M\).

For small total value error, the factor \((1+\delta_L)^{K-1}\) is close to one. The derivative target then has two interpretable contributions:

\[
K\eta_L
\quad\text{and}\quad
O(\Theta_*\sqrt M\,\delta_L K^{3/2}).
\]

The first is repeated insertion of one-step derivative error. The second is propagation of ordinary transition error through selection-sensitive exact dynamics. Both must be budgeted.

## 7. Reduced matrices and derivatives of the power action

The preceding bounds concern the exact powers of the compressed matrix. They are independent of the algorithm chosen to evaluate those powers.

Write the codec as

\[
Q_s=D_sE,
\qquad H_s=ED_s,
\]

where \(E\) is the selection-independent filtered DCT encoder and \(D_s\) evaluates its \(L\) cosine coefficients. For \(K\ge1\),

\[
Q_s^K=D_sH_s^{K-1}E.
\tag{7.1}
\]

Because \(E'=0\),

\[
H_s'=ED_s',
\]

and

\[
\partial_s(Q_s^K)
=D_s'H_s^{K-1}E
+D_s\left[\sum_{j=0}^{K-2}H_s^jH_s'H_s^{K-2-j}\right]E.
\tag{7.2}
\]

The sum is empty for \(K=1\). Differentiating only the final decoder omits the second term and is incorrect for multiple generations.

Equation (7.2) can be evaluated through a tangent recurrence, a block power, or a differentiated reduced power action. The operator-error theorem does not certify any particular Krylov stopping rule or finite-precision matrix-power routine.

For example, let

\[
z_s=H_s^{K-1}Ex,
\]

and suppose a solver produces \(\widehat z_s\) and \(\widehat z_s'\) with coefficient-\(\ell_1\) errors at most \(\zeta_0\) and \(\zeta_1\). Since each cosine basis function has supremum at most one,

\[
\|D_s\|_{1\to\infty}\le1,
\qquad
\|D_s'\|_{1\to\infty}\le(L-1)\Theta_s.
\]

The resulting full-state bounds are

\[
\|P_s^Kx-D_s\widehat z_s\|_\infty
\le\mathcal E_K\|x\|_\infty+\zeta_0,
\tag{7.3}
\]

\[
\|\partial_s(P_s^Kx)-(D_s'\widehat z_s+D_s\widehat z_s')\|_\infty
\le\mathcal G_K\|x\|_\infty
+(L-1)\Theta_s\zeta_0+\zeta_1.
\tag{7.4}
\]

Thus reduced value-action error also enters the decoded gradient. If the solver's \(\zeta\)-bounds are only for a particular input, these are action-specific guarantees. Uniform matrix-norm guarantees require uniform action bounds or a frozen linear approximation; an input-adaptive Krylov calculation is not automatically one fixed matrix.

If the input itself depends on selection, an additional term appears:

\[
\partial_s[(P_s^K-Q_s^K)x(s)]
=\partial_s(P_s^K-Q_s^K)x(s)+(P_s^K-Q_s^K)x'(s).
\]

Accordingly, add \(\mathcal E_K\|x'(s)\|_\infty\) to the derivative certificate.

## 8. Numerical error, optional boundary correction, and validation

### 8.1 Numerical error is separate from analytic compression error

Let \(\widehat Q_s\) be a finite-precision or approximate-evaluation implementation of the fixed codec. Suppose separate bounds establish

\[
\|Q_s-\widehat Q_s\|\le\delta_{\rm num},
\qquad
\|Q_s'-\widehat Q_s'\|\le\eta_{\rm num}.
\]

Then the multi-step theorems apply to \(\widehat Q_s\) with

\[
\delta=\delta_{\rm spectral}+\delta_{\rm num},
\qquad
\eta=\eta_{\rm spectral}+\eta_{\rm num}.
\tag{8.1}
\]

This assertion presumes an implementation that defines a fixed linear matrix family and the stated derivative. Rounding in a complete adaptive procedure may instead be analyzed as action defects.

The derivative numerical budget must account for multiplication of Fourier mode \(k\) by \(k\), errors in both \(\theta_s\) and \(\dot\theta_s\), and the accuracy of the differentiated evaluation routine. A library's value-tolerance setting alone is not a derivative-accuracy certificate. Likewise, an integer rank/taper planner should not be differentiated as though it were a smooth parameter-dependent operation.

### 8.2 An optional endpoint-exact variant

The raw filtered codec need not reproduce absorbing endpoint rows. If exact endpoints are required, define the row residual \(R(q)=b_M(q)-\widetilde b_M(q)\) and use

\[
\widetilde b_M^{\rm end}(q)
=\widetilde b_M(q)+(1-q)R(0)+qR(1).
\tag{8.2}
\]

For \(L\ge2\), this does not increase polynomial degree or rank. If the angular residual and its derivative are bounded by \(\delta_0\) and \(\delta_1\), respectively, then

\[
\sup_q\|b_M(q)-\widetilde b_M^{\rm end}(q)\|_1\le2\delta_0,
\]

\[
\|P_s'-(\widetilde P_s^{\rm end})'\|
\le\Theta_s(\delta_1+\delta_0).
\tag{8.3}
\]

The derivative bound follows from
\(|\partial_s q_s|\le\Theta_s/2\) and
\(\|R(1)-R(0)\|_1\le2\delta_0\).
The same multi-step results then apply with these adjusted one-step bounds. This is an explicitly different, boundary-corrected codec; it must not be substituted into the main theorem without updating the constants. It still need not be positive.

### 8.3 Reference checks

A small dense NumPy/SciPy reference calculation checked the exact and sampled-filtered matrices and their analytical selection derivatives at:

- \(M=32,L=18,s=0\), without mutation;
- \(M=64,L=28,s=0.2\), without mutation;
- \(M=128,L=42,s=0.05\), followed by asymmetric mutation \(u=0.001,v=0.002\).

For each case, \(K\in\{1,2,10,100\}\) was tested. The measured one-step errors were below the finite-strip certificates. The measured multi-step errors were below (5.2) and (5.9), and the exact sensitivities were below \(A_s\sqrt K\). Separate complex-angle tests verified (4.5) for 180 bounded random inputs across several strip widths and angles.

These checks are not proofs, optimized benchmarks, or finite-precision certification. Some intentionally modest-rank examples give very conservative long-horizon bounds; the purpose was to test the identities and inequalities, not to claim tight observed error.

The companion script and machine-readable results record the test parameters and rank calculations.

## 9. Conclusions and limits

For the fixed sampled DCT approximation, the one-step results are

\[
\|P_s-\widetilde P_s\|\le6E_L,
\qquad
\|P_s'-\widetilde P_s'\|\le9\Theta_sLE_L,
\qquad
E_L=e^{-M\Psi(L/M)}.
\]

For the same approximation after \(K\) generations, a convenient explicit pair is

\[
\|P_s^K-\widetilde P_s^K\|\le(1+6E_L)^K-1,
\]

\[
\|\partial_s(P_s^K)-\partial_s(\widetilde P_s^K)\|
\le K\Theta_s(1+6E_L)^{K-1}
[9L+6\sqrt{2M(K-1)}]E_L.
\]

The finite-strip and finite-sum forms are sharper and are preferable when selecting a practical rank. The proof keeps all derivatives in absolute units. Information geometry explains the derivative scales and improves the multi-step argument; it does not change the requested tolerances.

No claim is made that these bounds are optimal in \(K\), that errors saturate under arbitrary selection, or that a signed approximation is a stochastic kernel. Relative accuracy in rare probabilities or log-likelihood gradients needs additional lower bounds on the relevant probabilities. Numerical sampling, transform evaluation, and reduced-power computation still require separate error budgets.

## Appendix A. Changing population sizes and nonidentical transitions

The same idea extends to a compatible sequence of possibly rectangular matrices. Let

\[
\mathcal P=P_1P_2\cdots P_K,
\qquad
\mathcal Q=Q_1Q_2\cdots Q_K,
\]

where each exact \(P_j\) is stochastic and may have its own destination count \(M_j\) and frequency map. Assume

\[
\|Q_j-P_j\|\le\delta_j,
\quad\|Q_j'-P_j'\|\le\eta_j,
\quad A_j=\sqrt{M_j}\Theta_j.
\]

Set

\[
r_j=1+\delta_j,
\qquad R=\prod_{j=1}^Kr_j.
\]

Then

\[
\boxed{\|\mathcal P-\mathcal Q\|\le R-1.}
\tag{A.1}
\]

For the derivative, one useful closed bound is

\[
\boxed{
\|\mathcal P'-\mathcal Q'\|
\le\sum_{j=1}^K\eta_j\frac R{r_j}
+\sqrt{(R-1)W},
}
\tag{A.2}
\]

where

\[
W=\sum_{j=1}^KA_j^2
\left\{
\frac R{r_j}
\left[1+\sum_{\ell\ne j}\frac{\delta_\ell}{r_\ell}\right]-1
\right\}.
\tag{A.3}
\]

**Derivation.** Expand the product by the subset \(S\) of positions at which an error matrix is used. A word with \(|S|=m\) has at most \(m+1\) exact Markov blocks. The path-score argument bounds the derivative norm of an exact block by the square root of the sum of its one-step Fisher-information bounds. Thus, differentiating exact blocks contributes at most

\[
\left(\prod_{j\in S}\delta_j\right)
\sqrt{(m+1)\sum_{j\notin S}A_j^2}.
\]

Sum over nonempty \(S\) and apply Cauchy–Schwarz. The first squared factor is \(R-1\); the second is \(W\), obtained by collecting the terms for which a particular position \(j\) is not an error position. Differentiated error factors give the first term of (A.2).

For identical \(\delta_j=\delta\) and \(A_j=A_s\), this reduces to (5.12) and the \(K\eta r^{K-1}\) term. No rectangular matrix is raised to a power. The parameter derivative in this appendix is for one common parameter or one fixed direction in a parameter vector.

## References and provenance

1. H. Queffélec and R. Zarouf (2019). *On Bernstein's inequality for polynomials*. arXiv:1903.10801. Used for the classical trigonometric Bernstein inequality; the Banach-valued form follows by scalar duality.
2. G. Arampatzis, M. A. Katsoulakis, and Y. Pantazis (2015). *Pathwise Sensitivity Analysis in Transient Regimes*. arXiv:1502.05430. Theorem 3.1 decomposes Markov path Fisher information into one-step contributions; its sensitivity discussion gives the corresponding information-based observable bound. The finite-WF specialization and signed-approximation error expansion are derived in this note.
3. Project note, *Joint transition and selection-derivative error bounds for a filtered DCT approximation of the discrete Wright–Fisher operator* (24 October 2026). The present revision retains its sampled filter and complex-kernel construction, improves the convenient derivative constant, and adds multi-step guarantees for that same approximation.

All finite-WF constants and combined approximation certificates displayed here are supported by the derivations in this note. They are not attributed to a diffusion log-Sobolev theorem or to a published theorem about this specific codec.
