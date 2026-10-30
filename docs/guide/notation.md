# Mathematical notation

These conventions connect the probability model, numerical methods, and
sensitivity calculations. Chapters define their local quantities where used;
API parameter names and Python examples retain their code spelling. Use this
page as a reference: the [model](model.md) introduces the biological process,
then [computational tasks](../methods/overview.md) explains why the numerical
representations are needed.

## Population model

These symbols describe allele counts, biological parameters, and one-generation
transitions. The matrix orientation fixes how propagation and expectation relate.

| Concept | Symbol |
| --- | --- |
| Diploid effective population size | $N_e$ |
| Population gene-copy count | $M=2N_e$; $M_s,M_d$ for source and destination |
| Sample gene-copy count | $n$ |
| Source and destination allele counts | $i,j$ |
| Allele frequency | $x$, with $x_i=i/M_s$ |
| Offspring frequency after the full frequency map | $q_\theta(x)$; omit the subscript when parameters are implicit |
| Model parameter collection and its dimension | $\theta,n_\theta$ |
| Selection coefficients and dominance | $s_{\mathrm{het}},s_{\mathrm{hom}},h$ |
| Directional selection coefficient | $s=s_{\mathrm{hom}}$, with $s_{\mathrm{het}}=hs$; additive selection fixes $h=1/2$ |
| Multiplicative fitness ratio | $\rho$ (API parameter `r`) |
| Directional mutation rates | $\mu_{0\to1},\mu_{1\to0}$ |
| One-generation transition and its numerical approximation | $P,\widetilde P$ |
| Population mass column vector | $v_t$ |
| Observable or destination-value vector | $f$ |
| Generation index and interval duration | $t,g$ |
| Exact and represented interior transitions | $Q,\widetilde Q$ |
| Expected-visits vector | $u$ |
| Immigration/source vector | $b$ |

Rows of $P$ index source states; columns index destination states. Vectors
are columns unless transposed. The [model chapter](model.md) develops the
transition, propagation, and expected-visits equations.

The [fastDTWF paper](https://doi.org/10.1093/genetics/iyad168) uses the same
column-distribution and transpose convention, but names its transition matrix
$\mathbf M$ and its haploid population size $N$. Here those correspond to $P$
and the gene-copy count $M$, respectively. We retain $N_e$ for diploid size.

## Numerical representations

These symbols distinguish computational dimensions from biological sizes and
durations. The [computational overview](../methods/overview.md) introduces
the representations; each algorithm chapter defines its construction.

| Concept | Symbol |
| --- | --- |
| Prepared forward operator; input and output working states | $A;\ z,y$ |
| Represented forward population operator | $\widetilde P^T$ |
| Prepared exponent; reconstruction map and offset | $k;\ R,c$ |
| Working dimension and Krylov basis dimension | $d,m$ |
| Krylov basis and projected operator | $V_m$, $H_m$ |
| Jacobi coefficient transform and evaluation | $E,D$ |
| Reduced Jacobi expectation operator | $H_{\mathrm J}$ |
| Retained interior Jacobi dimension and assembly padding | $r,r_{\mathrm{pad}}$ |
| Polynomial degree and coefficient count | $L,n_{\mathrm{modes}}=L+1$ |
| Quadrature, sampling, and evaluation counts | $n_{\mathrm{quad}},n_{\mathrm{nodes}},n_{\mathrm{eval}}$ |
| Retained binomial stencil width | $W$ |
| Angular coordinate | $\vartheta$ |
| Centered Jacobi coordinate | $\xi=2q-1$ |
| FFT log-odds spacing | $\Delta_{\mathrm{logit}}$ |

Use $k$ for the prepared exponent and $g$ for physical generations. A tilde
marks a numerical approximation. $P$ and $\widetilde P$ act on population
observables; $\widetilde P^T$ propagates population mass. $A$ is the operator
in the coordinates used for the power calculation: $\widetilde P^T$ in
population coordinates, or $H_{\mathrm J}^T$ in reduced Jacobi coordinates.

Polynomial families retain the standard notation $T_j,U_j$ for Chebyshev,
$P_j^{(\alpha,\beta)}$ for Jacobi, and $C_j^{(\lambda)}$ for Gegenbauer;
$\phi_j$ denotes a normalized basis. These conventions follow
[NIST DLMF](https://dlmf.nist.gov/18.3). A polynomial approximation may be called $p_L$; it is not a population mass.

## Derivatives and local notation

Derivative notation distinguishes perturbations of a calculation from
sensitivities propagated backward from its output.

A dot denotes a directional derivative; a bar denotes the cotangent of the
same named quantity. Subscripts identify coordinates or projection spaces;
they do not change the underlying symbol's role.

| Concept | Symbol |
| --- | --- |
| Scalar objective | $\ell$ |
| Population, working-state, and operator cotangents | $\bar v_t,\bar z,\bar y,\bar A$ |
| Coupled tangent/state operator and vectors | $A_{\mathrm{blk}},z_{\mathrm{blk}},y_{\mathrm{blk}}$ |
| Coupled Krylov basis and projected operator (at most $2m$ columns) | $V_{2m,\mathrm{blk}},H_{2m,\mathrm{blk}}$ |
| Backward/forward adjoint bases (at most $m$ columns each) | $V_{m,\mathrm L},V_{m,\mathrm R}$ |
| Their projected operators and seeds | $H_{m,\mathrm L},H_{m,\mathrm R};\ \bar y_{m,\mathrm L},z_{m,\mathrm R}$ |
| Projected operator-cotangent core and lifted left factor | $C,F$ |
| Rational Krylov shift and shifted operator | $\tau,S=I+\tau(I-A)$ |
| Expected-visits incoming cotangent and adjoint solve variable | $\bar u,\eta_u$ |

The [autodiff chapter](../methods/autodiff.md) defines the labelled constructions.
It uses $\bar\theta_A$ for the contribution to the parameter cotangent through
$A$ alone; other parameter-dependent inputs contribute separately.

Write $\mathcal L_\varphi(A,\Delta A)$ for the Fréchet derivative of a matrix function $\varphi$,
and $\mathcal L_k$ for the power function $\varphi(A)=A^k$. The matrix-function
literature commonly uses $E$ for the perturbation; we use $\Delta A$ to avoid
confusion with the Jacobi transform. See
[Kandolf et al.](https://arxiv.org/abs/2008.12926).

Local summation indices and auxiliary functions may vary, but must be defined.
In particular, sample allele counts may use $k$ locally without referring to a
prepared power. Research notes retain their derivation notation and include
explicit crosswalks to these conventions.
