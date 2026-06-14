---
source: "[[High-Dimensional Analysis- Random Matrices and Machine Learning.pdf]]"
tags:
  - 随机矩阵与机器学习_random-matrices-and-machine-learning
  - Wishart矩阵_wishart-matrix
  - 随机矩阵_random-matrix
  - 协方差估计_covariance-estimation
---

>[!definition] Definition. *Wishart matrix*
>$x \sim N(0, \Sigma_{p\times p})$ and denote by $X = (x_1 \ x_2  \dots  x_n) \in \mathbb{R}^{p \times n}$ the data matrix. (Obverse $n$ times for $x_{p\times 1}$ 观察 $n$ 次 $p$ 维参数).
>$$
>\hat{\Sigma} := \frac{1}{n}XX^T = \frac{1}{n}\sum_{k=1}^n x_k x_k^T,
>$$
>is an estimator for $\Sigma$.

![[4 Wishart Random Matrices-1772624976366.webp|494]]
### 4.1. Concentration for the largest eigenvalue of Wishart matrices

>[!remark]
>- 数据集矩阵最大的奇异值是一个满足 L 条件的范数, 所以它集中在它的期望值附近.
>- 接着我们还需要思考这个期望值到底是多少.

We want to understand the spectral properties of $\hat{\Sigma}$, i.e. the eigenvalues of $\hat{\Sigma}$, or the singular values of $X$. Let us first restrict to the largest singular value:
$$
\|\cdot\|: \mathbb{R}^{p \times n} \cong \mathbb{R}^{pn} \to \mathbb{R}, \quad \|X\| := \sup_{v \in \mathbb{R}^n, \|v\|=1} \|Xv\|_2
$$
It is Lipschitz: for $X_1, X_2 \in \mathbb{R}^{p \times n}$,
$$
\|X_1+X_2\| = \sup_{v \in \mathbb{R}^n, \|v\|=1} \|(X_1+X_2)v\|_2 \leq \|X_1\| + \|X_2\|,
$$
$$
\|X_1\| - \|X_2\| \leq \underbrace{\|X_1 - X_2\| \leq 1 \cdot\|X_1 - X_2\|_{\mathbb{R}^{pn}}}_{\|X\| \leq \underbrace{\|X\|_F := \sqrt{\sum_{i=1}^p \sum_{j=1}^n |x_{ij}|^2}}_{\text{Frobenius norm}},}.
$$

Thus we can apply our Gaussian concentration inequality for Lipschitz functions. Note that $X \sim N(0, I_{pn})$ corresponds to $\Sigma = I_p$.

>[!theo] Theorem. 4.1. 标准数据一定可以中心化
>Suppose $X = (x_{ij})_{\substack{i=1, \dots, p \\ j=1, \dots, n}}$ is a standard Gaussian random matrix, i.e.  $x_{ij}\ i.i.d. \sim N(0, 1)$. Then
$$
\mathrm{P} \{ |\|X\| - E[\|X\|]| \geq \alpha \} \leq 2 \exp \left( -\frac{\alpha^2}{2} \right).
$$
^bjj1x5

***Proof.***
By [[3 Concentration of Gaussian random vectors for non-linear Lipschitz function#^fx0z71|Theorem. 3.2 Gaussian concentration for Lipschitz functions]]

>[!remark] ***Question*** 中心值是多少?

 ==***Step 1***== Simpler Question
 Note that we can write
$$
\|X\| = \max_{\substack{v \in \mathbb{R}^n, w \in \mathbb{R}^p \\ \|v\|=1=\|w\|}} \langle X_{p\times n}v_{n\times 1}, w_{p\times 1} \rangle
$$
But there are infinitely many terms. Too hard to control! So we go to finitely many conditions by approximating all $v$ and $w$ by elements from $\varepsilon$-nets.
It's a bit easier to do this for balls than for spheres, so let us write
$$
\|X\| = \max_{\substack{\|v\| \leq 1 \\ \|w\| \leq 1}} \langle Xv, w \rangle
$$
and let $\mathcal{N}$ now be an $\varepsilon$-net for
$$
B_n = \{v \in \mathbb{R}^n : \|v\| \leq 1\},
$$
i.e. $\mathcal{N} \subset B_n$ such that
$$
\forall v \in B_n \quad \exists \tilde{v} \in \mathcal{N} : \|v - \tilde{v}\| < \varepsilon.
$$
==***Step 2***==
We want $\mathcal{N}$ to be as small as possible; it is easy to see that there exists an $\varepsilon$-net $\mathcal{N}$ with
$$
|\mathcal{N}| \leq \left( \frac{2}{\varepsilon} + 1 \right)^n.
$$
Construct an $\varepsilon$-net $\mathcal{N}=\{v_1, v_2,  \dots,v_N\}$ by choosing a new point $v_k$ such that
$$
B_n \left( v_k, \frac{\varepsilon}{2} \right) \cap B_n \left( v_i, \frac{\varepsilon}{2} \right) = \emptyset \quad \forall\,  i = 1, \dots, k-1.
$$
Thus all $B_n \left( v_k, \frac{\varepsilon}{2} \right)$ are disjoint and
$$
\bigcup_k B_n \left( v_k, \frac{\varepsilon}{2} \right) \subset B_n \left( 0, 1 + \frac{\varepsilon}{2} \right),
$$
$$
N \cdot \text{vol} \left( B_n \left( \frac{\varepsilon}{2} \right) \right) \leq \text{vol} \left( B_n \left( 1 + \frac{\varepsilon}{2} \right) \right),
$$
$$
N \le \frac{\text{vol}\left(B_n\left(1 + \frac{\varepsilon}{2}\right)\right)}{\text{vol}\left(B_n\left(\frac{\varepsilon}{2}\right)\right)} = \frac{\left(1 + \frac{\varepsilon}{2}\right)^n}{\left(\frac{\varepsilon}{2}\right)^n} = \left(\frac{2}{\varepsilon} + 1\right)^n.
$$
你会注意到由于体积大小的限制, $N$ 只能是有限的. 这确保了 $\mathcal{N} = \{v_1, \dots, v_N\}$ 是一个 $\varepsilon$-net，因为如果存在一个点 $v$ 使得对于所有 $k = 1, \dots, N$ 都有 $\|v - v_k\| \ge \varepsilon$，那么这个 $v$ 就可以在我们的构造中被选作 $v_{N+1}$. 于是我们便构造出了满足条件的 $\varepsilon$-net.

==***Step 3***==
Set $\varepsilon = \frac{1}{4}$ then $\frac{2}{\varepsilon} + 1 = 9$.

Then we can choose an $\varepsilon$-net $\mathcal{N}$ for $\mathrm{B}_n$ and an $\varepsilon$-net $\mathcal{M}$ for $\mathrm{B}_p$ with $|\mathcal{N}| \le 9^n$ and $|\mathcal{M}| \le 9^p$.

Let $v \in \mathrm{B}_n$ and $w \in \mathrm{B}_p$ be the maximizer for $\|X\| = \langle Xv, w \rangle$ (note that finite dimensions & compact unit ball $\rightarrow$ the supremum is indeed a maximum);
then there exist $\tilde{v} \in \mathcal{N}$ and $\tilde{w} \in \mathcal{M}$ such that
$$
\|v - \tilde{v}\| < \varepsilon \quad \text{and} \quad \|w - \tilde{w}\| < \varepsilon.
$$
Then
$$
\begin{aligned}
\langle Xv, w \rangle &= \langle X(\tilde{v} + (v - \tilde{v})), \tilde{w} + (w - \tilde{w}) \rangle \\
&= \langle X\tilde{v}, \tilde{w} \rangle + \langle X\tilde{v}, w - \tilde{w} \rangle + \langle X(v - \tilde{v}), w \rangle \\
&\le \langle X\tilde{v}, \tilde{w} \rangle + \|X\| \cdot \underbrace{\|\tilde{v}\|}_{\leq 1} \cdot \underbrace{\|w - \tilde{w}\|}_{< \varepsilon} + \|X\| \cdot \underbrace{\|v - \tilde{v}\|}_{< \varepsilon} \cdot \underbrace{\|w\|}_{\leq 1} \\
&\le \langle X\tilde{v}, \tilde{w} \rangle + 2\varepsilon \|X\|,\\
\rightarrow \|X\| &\le \max_{\substack{\tilde{v} \in \mathcal{N} \\
\tilde{w} \in \mathcal{M}}} \langle X\tilde{v}, \tilde{w} \rangle + 2\varepsilon \|X\|\\
\rightarrow \|X\| &\le \frac{1}{1 - 2\varepsilon}  \max_{\substack{\tilde{v} \in \mathcal{N} \\
\tilde{w} \in \mathcal{M}}} \langle X\tilde{v}, \tilde{w} \rangle \le \frac{1}{1 - 2\varepsilon}  \max_{\substack{v \in \mathcal{N} \\ w \in \mathcal{M}}} \langle Xv, w \rangle.
\end{aligned}
$$
Thus we have
$$
\|X\| \ge \alpha\rightarrow\exists\ v \in \mathcal{N},w \in \mathcal{M}\ \text{s.t} \ \langle Xv, w \rangle \ge \alpha(1 - 2\varepsilon)
$$
$$
\begin{aligned}
\mathrm{P} \left\{ \|X\| \ge \alpha \right\} &\le \mathrm{P} \left\{ \bigcup_{\substack{v \in \mathcal{N} \\ w \in \mathcal{M}}} \{ \langle Xv, w \rangle \ge \alpha(1 - 2\varepsilon) \} \right\} \\
&\le \sum_{\substack{v \in \mathcal{N} \\ w \in \mathcal{M}}} \mathrm{P} \left\{ \langle Xv, w \rangle \ge \alpha(1 - 2\varepsilon) \right\}.
\end{aligned}
$$
Hence we need now the concentration inequality only for the finitely many summands on the right-hand side. Note that
$$
\begin{aligned} \langle Xv, w \rangle &= \sum_{i=1}^p  (X v)_i w_i =\underbrace{\sum_{i=1}^p \sum_{j=1}^n x_{ij} v_j w_i}_{x_{ij} \ i.i.d\sim N(0, 1)},\\
&\sim N(0,\sigma^2)\\
\sigma^2 &= \sum_{i=1}^p \sum_{j=1}^n v_j^2 w_i^2 = \|v\|^2 \cdot \|w\|^2 \le 1,\end{aligned}
$$

thus
$$
\mathrm{P} \{ \langle Xv, w \rangle \ge \alpha \} \le \frac{1}{2} \exp \left( -\frac{\alpha^2}{2} \right),
$$
or now with $\alpha$ replaced with $\alpha(1 - 2\varepsilon) = \frac{1}{2}\alpha$ for $\varepsilon = \frac{1}{4}$,

$$
\mathrm{P} \left\{ \langle Xv, w \rangle \ge \frac{\alpha}{2} \right\} \le \frac{1}{2} \exp \left( -\frac{\alpha^2}{8} \right).
$$

This then yields

$$
\begin{aligned}
\mathrm{P} \{ \|X\| \ge \alpha \} &\le |\mathcal{N}| \cdot |\mathcal{M}| \cdot \frac{1}{2} \exp \left( -\frac{\alpha^2}{8} \right) \\
&= \frac{1}{2} \cdot 9^n \cdot 9^p \cdot \exp \left( -\frac{\alpha^2}{8} \right) \\
(9<e^3)\quad&\le \frac{1}{2} \exp \left( 3n + 3p - \frac{\alpha^2}{8} \right) \\
&= \frac{1}{2} \exp \left( -\frac{1}{8}(\alpha^2 - 24n - 24p) \right).
\end{aligned}
$$

Put $\alpha = \sqrt{24}(\sqrt{n} + \sqrt{p}) + u$, then $\alpha^2 \ge u^2 + 24n + 24p$ and thus
$$
\mathrm{P} \left\{ \|X\| \ge \sqrt{24}(\sqrt{n} + \sqrt{p}) + u \right\} \le \frac{1}{2} \exp \left( -\frac{u^2}{8} \right).
$$
![[4 Wishart Random Matrices-1773058455477.webp|445]]
it is a rough estimate.

We will not pursue this any further, but instead we will now look at the collection of all singular values of $X$ or of all eigenvalues of $\hat{\Sigma}$; i.e., we want now to understand the asymptotics of the histograms of the eigenvalues.

### 4.2. Eigenvalue distribution of Wishart matrices & Marchenko-Pastur law

Set $x_1, \dots, x_n \in \mathbb{R}^p$ are independent vectors with $x_k \sim N(0, I_p)$ & $X = (x_1 \ x_2 \ \dots \ x_n) \in \mathbb{R}^{p \times n}$, then
$$
\hat{\Sigma} := \frac{1}{n} XX^T = \frac{1}{n} \sum_{k=1}^n x_k x_k^T \in \mathbb{R}^{p \times p}.
$$
What can we say about the eigenvalues of $\hat{\Sigma}$?

So first let us try to shrink it to Lipschitz condition.

Let $\lambda_1(A) \le \lambda_2(A) \le \dots \le \lambda_p(A)$ be the eigenvalues of a symmetric matrix $A = A^T \in \mathbb{R}^{p \times p}$.
![[4 Wishart Random Matrices-1773059107308.webp|456]]
Then one has, as for the maximal eigenvalue,
$\lambda_i(A) + \lambda_i(B-A) \leqslant \lambda_i(B) = \lambda_i(A+(B-A)) \leqslant \lambda_i(A) + \lambda_n(B-A)$
$\lambda_i(B-A) \leqslant \lambda_i(B) - \lambda_i(A) \leqslant \lambda_n(B-A)$
$|\lambda_i(B) - \lambda_i(A)| \leqslant \max(|\lambda_1(B-A)|, |\lambda_n(B-A)|)$
$|\lambda_i(A) - \lambda_i(B)| \leqslant \max(|\lambda_n(A-B)|, |\lambda_1(A-B)|)$
Thus
$$
|\lambda_i(A) - \lambda_i(B)| \le_{\substack{\text{Weyl's}\\ \text{Inequality}}} \|A - B\| \le \|A - B\|_F,
$$
i.e. the maps $A \mapsto \lambda_i(A)$ for $i = 1, \dots, p$ are Lipschitz and thus also the map
$$
A \mapsto (\lambda_1(A), \dots, \lambda_p(A))
$$
is Lipschitz.

However, since $X$ is our matrix with independent Gaussian entries, we are interested in the mapping

$$
X \mapsto \left( \lambda_1 \left( \frac{1}{n} X_{p\times n}X^T_{n\times p} \right), \dots, \lambda_p \left( \frac{1}{n} XX^T \right) \right).
$$

For this, the Lipschitz constant is modified as follows:
$$
\begin{aligned}
\left| \lambda_i \left( \frac{1}{n} XX^T \right) - \lambda_i \left( \frac{1}{n} YY^T \right) \right| &\le \frac{1}{n} \|XX^T - YY^T\| \\
&= \frac{1}{n} \|XX^T - XY^T + XY^T - YY^T\| \\
&\le \frac{1}{n} \left( \|X\| \cdot \underbrace{\|X^T - Y^T\|}_{\le \|X-Y\|_F} + \underbrace{\|X - Y\|}_{\le \|X-Y\|_F} \cdot \|Y^T\| \right) \\
&\le \frac{1}{n} (\|X\| + \|Y\|) \cdot \|X - Y\|_F \\
&\le \underbrace{\frac{2}{n} \max(\|X\|, \|Y\|)}_{\text{L is bounded by}} \cdot \|X - Y\|_F.
\end{aligned}
$$

Note that the estimate $\|X\| \le \|X\|_F = \|X\|_{\mathbb{R}^{pn}}$ is not helpful, since we know that $\|X\|_{\mathbb{R}^{pn}} \sim \sqrt{pn}$. [[2 Gaussian random vectors and linear concentration of Chebyshev and Bernstein type#^lrt756|(Reason)]] is not a useful bound.

But let us have a closer look on this, as it also reveals the difference between classical and modern regimes.
* In the classical regime,
$$
L \sim \frac{2}{n} \underbrace{\sqrt{pn}}_{\color{red}{\substack{p \text{ is fixed}\\ n \to \infty}}} \sim c \frac{1}{\sqrt{n}},
$$
which would give good concentration.
* But in the modern regime,
$$
L \sim \underbrace{2 \frac{1}{n} \sqrt{\gamma n \cdot n}}_{\color{red}\substack{\frac p n\xrightarrow{p,n\rightarrow \infty} \gamma\\ \gamma \text{ is fixed}}} = 2\sqrt{\gamma} \sim \text{constant},
$$
which does not give good concentration.
- So let's keep the operator norm $\|X\|$ in $L$ in Section 4.1; for this we already know that we have good concentration around $\sim c(\sqrt{n} + \sqrt{p}) \sim c(1 + \sqrt{\gamma})\sqrt{n}$ and thus with high probability
$$
L \sim 2 \frac{1}{n} c(1 + \sqrt{\gamma})\sqrt{n} \sim \tilde{c} \frac{1}{\sqrt{n}}.
$$
By [[3 Concentration of Gaussian random vectors for non-linear Lipschitz function#^fx0z71|Theorem 3.2]], this then gives concentration of $\lambda_i(\hat{\Sigma})$ around its expected value with
$$
2 \exp \left( -\frac{\alpha^2}{2L^2} \right) \sim 2 \exp \left( -\frac{n\alpha^2}{2\tilde{c}^2} \right)
$$
This means in the modern regime, the eigenvalue distribution of $\hat{\Sigma} = \frac{1}{n} XX^T$ concentrates on its average (The scaling factor $\frac{1}{n}$ in $\hat{\Sigma}$ ensures that we have a limit for $n \to \infty$), then we can get:

![[4 Wishart Random Matrices-1772177853460.webp|583]]

>[!theo] Theorem. 4.2 Marchenko-Pastur Law [^mp]
>Let $X \in \mathbb{R}^{p \times n} \cong \mathbb{R}^{np}$ & $(X)_{ij} \ \text{i.i.d.} \sim N(0, \sigma^2)$. If $\frac{p}{n} \xrightarrow{n \to \infty} \gamma \in (0, 1]$. (观察次数 $\geq$ 数据维数)
>Then the histogram of the eigenvalues of $\hat{\Sigma} = \frac{1}{n} XX^T$ converges to the Marchenko-Pastur density
>$$
>\psi_{\text{MP}}(x) = \underbrace{\frac{1}{2\pi \sigma^2 \gamma}}_{\substack{\text{Normalization}\\ \text{Coefficient}}} \cdot \frac{\sqrt{(\lambda_+ - x)(x - \lambda_-)}}{x}
>$$
>where $\lambda_+ = \sigma^2 (1 + \sqrt{\gamma})^2$ &  $\lambda_- = \sigma^2 (1 - \sqrt{\gamma})^2$.

***Remark.***
- 这里的归一化常数不是伽马分布那种强行归一化计算出来的, 而是通过自洽方程自然推导出来的.
- Note that the statement is of the form
$$
\frac{1}{p} \sum_{i=1}^{p} f(\lambda_i) \xrightarrow[p=\gamma n]{n \to \infty} \int f(t) \psi_{\text{MP}}(t) \, dt, \quad (1)
$$
where $f = \mathbb{1}_{[a,b]}$ and $\lambda_1, \dots, \lambda_p$ are the eigenvalues of $\hat{\Sigma}$.
Proving (1) directly is not so clear, but can be achieved by proving analogous statements for other classes of functions. Instead of proving (1) for
- (i) all $f = \mathbb{1}_{[a,b]}$ for all $a < b$, (直接看落在 $[a, b]$ 区间的特征值占比)
- (ii) all moments $f(t) = t^n$ for all $n \in \mathbb{N}$,  (证明特征值的各阶矩（$t^n$ 的平均值）都对得上)
- (iii) all resolvents $f(t) = \frac{1}{t-z}$ for all $z \in \mathbb{C}_+$. ($\mathbb{C}_+$ denotes the complex upper half plane.) (证明预解式 $f(t) = \frac{1}{t-z}$ 的平均值对得上)
By concentration, it suffices to prove in each case the version for the average, i.e. one has to prove
$$
\frac{1}{p} \sum_{i=1}^{p} \mathbb{E}[f(\lambda_i)] \to \int f(t) \psi_{\text{MP}}(t) \, dt. \quad (2)
$$
Note for this that $\frac{1}{t-z}$ and $t^n$ (if we restrict them to a compact interval) are Lipschitz functions.

***Proof.*** **(by Self-consistent equation)**
***Step 1***
>[!definition] Definition. 4.4 Stieltjes Transform
>- For Wishart matrices $\hat{\Sigma} = \frac{1}{n}XX^T \in \mathbb{R}^{p \times p}$
>$$
>S_n(z) := E \left[ \frac{1}{p}\text{tr}[(\hat{\Sigma} - zI_p)^{-1}] \right] \quad \text{for} \quad z \in \mathbb{C} \setminus \mathbb{R}.
>$$
>- For the Marchenko-Pastur distribution $\psi_{\text{MP}}$
>$$
>S_{\text{MP}}(z) := \int \frac{1}{t-z} \psi_{\text{MP}} \, dt \quad \text{for} \quad z \in \mathbb{C} \setminus \mathbb{R}.
>$$

So what we have to prove is the convergence of the Stieltjes transforms:
$$
S_n(z) \xrightarrow[p=\gamma n]{n \to \infty} S_{\text{MP}}(z) \quad \text{for all} \quad z \in \mathbb{C} \setminus \mathbb{R}.
$$
And take some prepares for it:

>[!lemma] Lemma. 4.3
>Let $A \in \mathbb{R}^{p \times p}$ be a symmetric matrix with eigenvalues $\lambda_1, \dots, \lambda_p$. Then, for any $z \in \mathbb{C} \setminus \{\lambda_1, \dots, \lambda_p\}$, we have
>$$
>\frac{1}{p} \sum_{i=1}^{p} \frac{1}{\lambda_i - z} = \operatorname{ntr} \left[ (A - zI_p)^{-1} \right],
>$$
>where $\operatorname{ntr} = \frac{1}{p} \operatorname{Tr}$ is the normalized trace on $\mathbb{R}^{p \times p}$.

By [[Char2 Wigner Ensemble & SemiCircle Law#^v7zq7x|The eigenvalues of Resolvent]]

>[!lemma] Lemma. 4.6 很重要的一个技巧
>Let $x \sim N(0, I_{p})$ be a standard Gaussian vector in $\mathbb{R}^{p}$ and $B \in \mathbb{R}^{p \times p}$ a deterministic or random matrix independent from $x$. Then
>$$
>E_{x}[x^{T} B x]=\operatorname{Tr}(B).
>$$

>[!remark]- ***Proof.***
Let $B=\left(b_{i j}\right)_{i, j=1}^{p}$. We have
$$
E_{x}\left[x^{T} B x\right]=E_{x}[\langle x, B x\rangle]=\sum_{i, j=1}^{p} E\left[x_{i} b_{i j} x_{j}\right]=\sum_{i, j=1}^{p} b_{i j} \delta_{i j}=\sum_{i=1}^{p} b_{i i}=\operatorname{Tr}(B) .
$$

***Step 2***
By [[Char2 Wigner Ensemble & SemiCircle Law#^yxh4qp|Resolvent]] we can get
$$
\begin{aligned}
S_n(z) &\stackrel{\text{def}}{=} \mathbb{E} \frac{\text{tr}(\hat{\Sigma}-zI_p)^{-1}}{p} = \mathbb{E} \frac{\sum_{i=1}^{p} \frac{1}{\lambda_i-z}}{p} \\
S_n(z) &= \mathbb{E} \frac{\text{tr}(\hat{\Sigma}-zI_p)^{-1}}{p} \\
&= \mathbb{E} \frac{\text{tr}(\frac{1}{n}XX^T-zI_p)^{-1}}{p} \\
&= \mathbb{E} \frac{n}{p} \text{tr}(XX^T-znI_p)^{-1} \\
&= \mathbb{E} \frac{n}{p} \text{tr}(\underbrace{\sum_{i=1}^{n} x_i x_i^T - znI_p}_{\color{red}B})^{-1} \\
&= \mathbb{E} \frac{n}{p} \text{tr}(\underbrace{\sum_{i=1}^{n-1} x_i x_i^T - znI_p}_{\color{blue}A} + \underbrace{x_n x_n^T}_{X \sim N(0, I_p)})^{-1}
\end{aligned}
$$
By [[Char1 矩阵基础知识#^rzmtuw|Sherman-Morrison Formula]] we can get
$$
\begin{aligned}
x_n^T B^{-1} x_n &= x_n^T (A + x_n x_n^T)^{-1} x_n \\
(\text{S-M Formula})&= x_n^T \left( A^{-1} - \frac{A^{-1} x_n x_n^T A^{-1}}{1 + x_n^T A^{-1} x_n} \right) x_n \\
&= x_n^T A^{-1} x_n - \frac{(x_n^T A^{-1} x_n)(x_n^T A^{-1} x_n)}{1 + x_n^T A^{-1} x_n} \\
(x_n^T A^{-1} x_n = C) &= C - \frac{C^2}{1 + C} = \frac{C}{1 + C} \\
&= \frac{x_n^T A^{-1} x_n}{1 + x_n^T A^{-1} x_n} \\
 &\rightarrow \mathbb{E} \frac{x_n^T A^{-1} x_n}{1 + x_n^T A^{-1} x_n} = \frac{\mathbb{E} x_n^T A^{-1} x_n}{1 + \mathbb{E} x_n^T A^{-1} x_n} \\
(\text{Lemma 4.6}) &= \frac{\text{tr}(A^{-1})}{1 + \text{tr}(A^{-1})}
\end{aligned}
$$
In fact, $x_n$ can be replaced by $x_k\sim N(0,I_p)$ where $k$ is arbitrary. Thus we have

$$
\begin{aligned} \frac{\text{tr}(A^{-1})}{1 + \text{tr}(A^{-1})} &\approx x_k^T B^{-1} x_k = \text{tr}(\underbrace{x_k^T B^{-1} x_k}_{\mathbb{R}^1}) = \text{tr}(\underbrace{x_k x_k^T\cdot B^{-1}}_{\mathbb{R}^{p \times p}}) \\ n \cdot \frac{\text{tr}(A^{-1})}{1 + \text{tr}(A^{-1})} &\approx \sum_{k=1}^{n} \text{tr}(x_k x_k^T B^{-1}) = \text{tr} \left( \sum_{k=1}^{n} x_k x_k^T \cdot B^{-1} \right) \\ &= \text{tr} \big( \underbrace{\sum_{k=1}^{n-1} x_k x_k^T - n z I_p }_{A}+ x_n x_n^T + n z I_p \big) B^{-1} \\ &= \text{tr}(B + n z I_p) B^{-1} \\ &= p + n z \text{ tr} B^{-1} \end{aligned}
$$

So left and right side becomes:
$$
\begin{aligned}
\text{tr}(B^{-1}) &\approx \mathbb{E} \text{ tr} B^{-1} = \frac{p}{n} s_n(z) \\
\text{tr}(A^{-1}) &= \text{tr}(\sum_{k=1}^{n-1} x_k x_k^T - n z I_p) \\
&\approx \mathbb{E}(\text{tr}(\sum_{k=1}^{n-1} x_k x_k^T - (n-1) z I_p)) \\
&= \frac{p}{n-1} s_{n-1}(z) \\
&\approx \frac{p}{n} s_n(z)
\end{aligned}
$$
Finally we can calculate ^wkgll1
$$
\begin{aligned}
n \cdot \frac{\text{tr} A^{-1}}{1 + \text{tr} A^{-1}} &\approx p + nz \cdot \text{tr} B^{-1} \quad (\text{复杂度 } \gamma = \frac{p}{n}) \\
\frac{\text{tr} A^{-1}}{1 + \text{tr} A^{-1}} &\approx \gamma + z \cdot \text{tr} B^{-1} \\
\frac{\gamma s_n(z)}{1 + \gamma s_n(z)} &\approx \gamma + z \cdot \gamma \cdot s_n(z) \\
\lim_{\substack{n \to \infty \\ p = \gamma \cdot n}} \left( \frac{s_n(z)}{1 + \gamma s_n(z)} \right) &\approx \lim_{\substack{n \to \infty \\ p = \gamma \cdot n}} (1 + z s_n(z)) \\
\color{red}\frac{s(z)}{1 + \gamma s(z)} &\color{red}\approx 1 + z s(z) \\
0 &= \gamma z s(z)^2 - (1 - z - \gamma) s(z) + 1 \\
\Rightarrow s(z) &= \frac{1 - z - \gamma + \sqrt{(z - \gamma +)(z - \gamma -)}}{2 \gamma z}
\end{aligned}
$$
But it turn out without density, then we find a way to calculate out the density of MP distribution:

***Step 3***
>[!lemma] **Lemma 4.7 (Stieltjes Inversion Formula).**
>Let $\psi$ be a continuous probability density on $\mathbb{R}$. Then its Stieltjes transform
>$$
>S(z) := \int \frac{1}{t - z} \psi(t) \, dt \quad \text{for} \quad z \in \mathbb{C}_+
>$$
>has a continuous extension to $\mathbb{C}_+ \cup \mathbb{R}$ and
>$$
>\psi(t) = \frac{1}{\pi} \text{Im}(S(t)) = \frac{1}{\pi} \lim_{\varepsilon \to 0} \text{Im}(S(t + i\varepsilon)).
>$$

>[!remark]- ***Proof.***
For all $z \in \mathbb{C} \setminus \{0\}$, we have
>$$
>\text{Im} \left( \frac{1}{z} \right) = \frac{1}{2i} \left( \frac{1}{z} - \frac{1}{\bar{z}} \right) = \frac{1}{2i} \cdot \frac{\bar{z} - z}{z \cdot \bar{z}} = -\frac{\text{Im}(z)}{|z|^2}
>$$
and thus
>$$
>\begin{aligned} \frac{1}{\pi} \text{Im}(S(t + i\varepsilon)) &= \frac{1}{\pi} \int \text{Im} \left( \frac{1}{s - (t + i\varepsilon)} \right) \psi(s) \, ds \\ &= \frac{1}{\pi} \int \frac{\varepsilon}{(t - s)^2 + \varepsilon^2} \psi(s) \, ds \\ &\xrightarrow{\varepsilon \searrow 0} \int \delta_t(s) \psi(s) \, ds = \psi(t). \quad \square \end{aligned}
>$$

Apply this to
$$
S(z) = \frac{1 - z - \gamma + \sqrt{(z - \gamma_+)(z - \gamma_-)}}{2 \gamma z},
$$
then we get the form of the Marchenko-Pastur distribution as claimed in Theorem 4.2
$$
\begin{aligned}
\frac{1}{\pi} \text{Im}(S(t)) &= \frac{1}{\pi} \frac{\text{Im}(\sqrt{(t - \gamma_+)(t - \gamma_-)})}{2 \gamma t} \\
&= \begin{cases} \frac{1}{2 \pi \gamma t} \sqrt{(\gamma_+ - t)(t - \gamma_-)}, & t \in [\gamma_-, \gamma_+], \\ 0, & \text{otherwise.} \end{cases}
\end{aligned}
$$

![[4 Wishart Random Matrices-1773209233653.webp|379]]



[^MP]: VA Marchenko and LA Pastur, *The distribution of eigenvalues in certain sets of random matrices math, Math.* USSR-Sbornik 1 (1967), 457–483.

