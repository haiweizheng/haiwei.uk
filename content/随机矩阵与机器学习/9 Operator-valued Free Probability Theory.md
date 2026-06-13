---
source: "[[High-Dimensional Analysis- Random Matrices and Machine Learning.pdf]]"
tags:
  - 随机矩阵与机器学习_random-matrices-and-machine-learning
  - 自由概率论_free-probability-theory
  - 算子值_operator-valued
  - 自由累积量_free-cumulant
---

We have seen that the calculation of the eigenvalue distribution or the Stieltjes transform of polynomials or even rational functions in several random matrices is relevant. *Free probability theory*, which was introduced by Dan Voiculescu in the 1980’s, provides powerful tools for dealing with this. In the following we will give an appetizer for this; for more details on those topics, see [MS17].

## 9.1. Free cumulants and freeness

In Section 7 we have seen that our matrices often have some special structure (at least asymptotically) for the cumulants of their entries. In order not to have to bother with the transpose for general rectangular matrices, we consider now symmetric square matrices:

$$
X = (x_{ij})_{i,j=1}^n, \quad X = X^T \quad (\text{i.e. } x_{ij} = x_{ji}).
$$

For those, typically in leading order only cumulants with cyclic index structure survive:

$$
\kappa_\ell(x_{i(1)i(2)}, x_{i(2)i(3)}, \dots, x_{i(\ell)i(1)}) \sim n^{-(\ell-1)}.
$$

Their value is independent of $i(1), \dots, i(\ell)$ for distinct $i(1), \dots, i(\ell)$. So let us put

$$
r_\ell := \lim_{n \to \infty} n^{\ell-1} \kappa_\ell(x_{i(1)i(2)}, x_{i(2)i(3)}, \dots, x_{i(\ell)i(1)}).
$$

Note that the order $n^{-(\ell-1)}$ is the right one for a $\ell$-th cumulant to make a contribution in the calculation of

$$
\begin{aligned}
E[\operatorname{tr}(X^\ell)]
&= E\left[\frac{1}{n}\operatorname{Tr}(X^\ell)\right] \\
&= \frac{1}{n} \sum_{i(1),\dots,i(\ell)=1}^n E[x_{i(1)i(2)} x_{i(2)i(3)} \cdots x_{i(n)i(1)}] \\
&= \frac{1}{n} \sum_{i(1),\dots,i(\ell)=1}^n \sum_{\pi \in \mathcal{P}(\ell)}
\kappa_\pi(x_{i(1)i(2)}, x_{i(2)i(3)}, \dots, x_{i(n)i(1)}).
\end{aligned}
$$

Since $\kappa_\pi(x_{i(1)i(2)}, \dots, x_{i(n)i(1)})$ is at most of order $n^{-(\ell-1)}$ and the sum over the $i(j)$ has about $n^\ell$ terms, we get $E[\operatorname{tr}(X^\ell)] \sim 1$.

If one collects the leading order contributions in this, one gets a “non-commutative” version of a moment-cumulant relation between the
$$
m_\ell := \lim_{n \to \infty} \operatorname{tr}(X^\ell)
$$
and the $r_\ell$.


**Example.** If $\ell = 1$, then

$$
E[\operatorname{tr}(X)] = \sum_i \frac{1}{n} E[x_{ii}]
= \sum_i \frac{1}{n} \kappa_1(x_{ii}) \to r_1.
$$

and

$$
E[\operatorname{tr}(X^2)] = \sum_{i,j} \frac{1}{n} E[x_{ij} x_{ji}]
= \sum_{i,j} \frac{1}{n} \big( \kappa_1(x_{ij})\kappa_1(x_{ij}) + \kappa_2(x_{ij}, x_{ji}) \big)
\to r_1 r_1 + r_2.
$$

this looks like the normal moment-cumulant relation for the usual cumulants. But consider now $\ell = 4$, and assume that odd cumulants don’t contribute, i.e. $r_1 = r_3 = 0$. Then

$$
E[\operatorname{tr}(X^4)] = \frac{1}{n} \sum_{i(1),\dots,i(4)=1}^n
x_{i(1)i(2)} x_{i(2)i(3)} x_{i(3)i(4)} x_{i(4)i(1)}.
$$

Looking at the summands in detail, we have

$$
\kappa_{\text{(one block)}} \sim r_4 n^{-3} \leadsto r_4,
$$

$$
\kappa_{\text{(pairing)}} = \kappa_2(x_{i(1)i(2)}, x_{i(2)i(3)}) \cdot
\kappa_2(x_{i(3)i(4)}, x_{i(4)i(1)})
$$

$$
\sim \delta_{i(1)i(3)} r_2 n^{-1} \cdot \delta_{i(3)i(1)} r_2 n^{-1}
\sim r_2 \cdot r_2,
$$

$$
\kappa_{\text{(other pairing)}} \leadsto r_2 \cdot r_2,
$$

$$
\kappa_{\text{(crossing)}} \sim r_2 \cdot r_2 n^{-2} \cdot \delta(\text{at least two indices equal})
\leadsto 0.
$$

Thus crossing partitions do not contribute asymptotically; this is true in general, and we have the “free” moment-cumulant relation:

$$
m_\ell = \sum_{\pi \in NC(\ell)} r_\pi,
$$

where $NC(\ell) \subset \mathcal{P}(\ell)$ is the set of all non-crossing partitions. The $r_\ell$ are called **free cumulants**.

Note that one also has a multivariate version of this; then the vanishing of classical mixed cumulants of entries of two independent matrices implies vanishing of the corresponding mixed free cumulants, which gives a notion of “free independence” or “freeness”, hence the name free probability theory.

## 9.2 Linearization of non-linear problems

Vanishing of mixed cumulants implies additivity of free cumulants for the sum, and allows sums of “asymptotically free” random matrices to be treated. This looks nice, but how about polynomials or rational functions, as they showed up in Section 7? Those can also be addressed via the following linearization trick. The idea is to reformulate a polynomial (non-linear) problem into a linear one with matrix coefficients, which makes it an operator-valued linear problem.

We will give the idea of this only via a concrete example.


**Example.** Consider the polynomial
$$
P = p(X,Y) = XY + YX + X^2.
$$

We want its Stieltjes transform
$$
S_p(z) = E\left[\operatorname{tr}\big((P - z\cdot 1)^{-1}\big)\right].
$$

How do we deal with the inverse of $P - z\cdot 1$? For this, we embed the problem in matrices:
$$
\begin{pmatrix}
XY + YX + X^2 - z\cdot 1 & 0 & 0 \\
0 & 0 & -1 \\
0 & -1 & 0
\end{pmatrix}
=
\begin{pmatrix}
1 & Y + \tfrac{X}{2} & X \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
-z & X & Y + \tfrac{X}{2} \\
X & 0 & -1 \\
Y + \tfrac{X}{2} & -1 & 0
\end{pmatrix}
\begin{pmatrix}
1 & 0 & 0 \\
Y + \tfrac{X}{2} & 1 & 0 \\
X & 0 & 1
\end{pmatrix}.
$$

Since the triangular matrices are always invertible, we have
$$
\begin{pmatrix}
(P - z\cdot 1)^{-1} & 0 & 0 \\
0 & 0 & -1 \\
0 & -1 & 0
\end{pmatrix}
=
\begin{pmatrix}
1 & 0 & 0 \\
Y + \tfrac{X}{2} & 1 & 0 \\
X & 0 & 1
\end{pmatrix}^{-1}
\begin{pmatrix}
-z & X & Y + \tfrac{X}{2} \\
X & 0 & -1 \\
Y + \tfrac{X}{2} & -1 & 0
\end{pmatrix}^{-1}
\begin{pmatrix}
1 & Y + \tfrac{X}{2} & X \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}^{-1}.
$$

Thus
$$
(P - z\cdot 1)^{-1}
=
\big[(\hat{P} - \Lambda(z))^{-1}\big]_{1,1},
$$

where
$$
\hat{P} :=
\begin{pmatrix}
0 & X & Y + \tfrac{X}{2} \\
X & 0 & -1 \\
Y + \tfrac{X}{2} & -1 & 0
\end{pmatrix},
\quad
\Lambda(z) :=
\begin{pmatrix}
z & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{pmatrix}.
$$

Now $\hat{P}$ is a linear polynomial in $X$ and $Y$ with matrix coefficients:
$$
\hat{P}
=
\begin{pmatrix}
0 & 0 & 0 \\
0 & 0 & -1 \\
0 & -1 & 0
\end{pmatrix} \cdot 1
+
\begin{pmatrix}
0 & 1 & \tfrac{1}{2} \\
1 & 0 & 0 \\
\tfrac{1}{2} & 0 & 0
\end{pmatrix} \cdot X
+
\begin{pmatrix}
0 & 0 & 1 \\
0 & 0 & 0 \\
1 & 0 & 0
\end{pmatrix} \cdot Y.
$$


This can then be dealt with by *operator-valued free probability theory*, which allows to calculate the distribution of such matrix-valued linear combinations.

Note also that the above factorization of $P$ might have looked very special, but indeed for any polynomial (and actually also for any non-commutative rational function) $P$ there exists (via a concrete algorithm) such a linearization $\hat{P}$.