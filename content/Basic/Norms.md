---
tags:
  - 数值分析_numerical-analysis
  - 矩阵范数_matrix-norm
  - 扰动分析_perturbation-analysis
  - 条件数_condition-number
---
# Chapter 7 Norms and Perturbation Analysis

> *"The central question of perturbation theory is: How does a function change when its argument is subject to a perturbation?"*
> -- G. Stewart and J. G. Sun [29]

It is well-known that the length of vectors in a vector space is captured by norms. In this chapter, we employ norms to analyze perturbation effects on numerical solutions of linear systems.

## Common Matrix Norms

设 $A \in \mathbb{C}^{m \times n}$, 其奇异值为 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(m,n)} \geq 0$.

>[!definition] Spectral norm (2-norm) 矩阵放大规模
>$$\|A\|_{op}=\|A\|_2 := \max_{\mathbf{x} \neq \mathbf{0}} \frac{\|A\mathbf{x}\|_2}{\|\mathbf{x}\|_2} = \sigma_1(A).$$
>即 $A$ 的最大奇异值. 它是由向量 2-norm 诱导的算子范数.

>[!definition] Frobenius norm 矩阵总能量
>$$\|A\|_F := \left( \sum_{i,j} |a_{ij}|^2 \right)^{1/2} = \left( \sum_{k=1}^{\min(m,n)} \sigma_k^2 \right)^{1/2} = [\operatorname{tr}(A^*A)]^{1/2}.$$
>Frobenius 范数不是算子范数, 但满足 submultiplicativity $\|AB\|_F \leq \|A\|_F \|B\|_F$, 且具有酉不变性.

>[!definition] Nuclear norm (trace norm)
>$$\|A\|_* := \sum_{k=1}^{\min(m,n)} \sigma_k = \operatorname{tr}\!\left(\sqrt{A^*A}\right).$$
>核范数是谱范数的对偶范数: $\|A\|_* = \max_{\|B\|_2 \leq 1} \operatorname{Re}\,\operatorname{tr}(B^*A)$. 在低秩矩阵恢复中常用作秩函数的凸松弛.

三者之间的关系: $\|A\|_2 \leq \|A\|_F \leq \|A\|_* \leq \sqrt{\min(m,n)}\,\|A\|_F \leq \min(m,n)\,\|A\|_2$.


## 核范数

这是一个非常深刻的类比。要理解核范数（Nuclear Norm），最快的方法是把它看作是**“矩阵界的 $L_1$ 范数”**。

在高维统计和压缩感知中，我们经常追求“简单”的模型。对于向量，简单意味着**稀疏性**（Sparse）；对于矩阵，简单则意味着**低秩性**（Low-rank）。


###### 1. 从 $L_0$ 到 $L_1$：向量的启示

在处理向量 $x$ 时：

- **$L_0$ 范数**：计算非零元素的个数。我们想最小化它来获得稀疏解，但它是非凸的，极难优化（NP-hard）。
- **$L_1$ 范数**：$\sum |x_i|$。数学家发现，它是 $L_0$ 的**最优凸近似**。最小化 $L_1$ 会在坐标轴上产生“尖点”，迫使很多分量变成 $0$。

###### 2. 从 Rank 到 Nuclear Norm：矩阵的跨越

矩阵的**秩 (Rank)** 对应向量的 $L_0$ 范数——它计算的是非零奇异值的个数。
- **Rank(A)**：非凸，阶梯状函数，无法求导，优化起来非常痛苦。
- __核范数 $\|A\|_* = \sum \sigma_i$_*：它是奇异值序列的 $L_1$ 范数。

因为核范数是所有奇异值的**绝对值之和**，当你试图减小核范数时，你实际上是在挤压每一个奇异值。由于凸优化的几何特性，这种“挤压”会优先让那些较小的、不重要的奇异值直接归零。

---

### 3. 核心应用场景的深层逻辑

#### A. 矩阵补全 (Netflix 问题)

假设有一个 $1000 \times 1000$ 的矩阵，代表用户对电影的评分。大部分格子是空的。

- **直觉**：人类的品味其实只有几种模式（动作片迷、文艺片迷等）。这意味着这个大矩阵本质上是由几个向量组合出来的，即它是**低秩**的。
- **做法**：我们在补全空格时，要求补全后的矩阵**核范数最小**。这会迫使矩阵通过最少量的“特征模式”来解释已有的评分，从而推测出空位的分值。

#### B. 稳健主成分分析 (RPCA)

想象一段监控视频。

- **背景**：几乎不动，随时间变化呈现**低秩结构**（每一帧都差不多）。
- **行人**：突然出现，只占画面一小块，在空间上是**稀疏**的。
- **模型**：$D（观测数据） = L（低秩背景） + S（稀疏前景）$。
- **优化**：$\min \|L\|_* + \lambda \|S\|_1$。
    通过最小化核范数 $\|L\|_*$，我们强迫背景必须保持高度的一致性（低秩），从而把跳动的行人挤到了 $S$ 里面。

---

### 4. 为什么说它衡量“内在维度”？

我们可以把矩阵看作一个数据集。

- 如果一个矩阵的**谱范数**（最大奇异值）很大，但**核范数**和它差不多，说明这个矩阵几乎只有一个主成分，能量极其集中。
    
- 如果一个矩阵的**核范数**远大于**谱范数**，说明它的能量分散在很多个奇异值上，结构非常复杂，维度很高。
    

**核范数越小，意味着奇异值分布越“尖锐”：** 绝大部分能量被极少数几个奇异值带走了。这就像一个压缩包，虽然看起来解压后很大，但核心信息量（核范数）其实很小。

---

### 总结：范数与“简单性”

|**目标**|**衡量标准**|**凸近似工具**|**物理意义**|
|---|---|---|---|
|**向量稀疏**|$L_0$ (非零元素数)|**$L_1$ 范数**|只有少数特征起作用|
|**矩阵低秩**|Rank (奇异值个数)|**核范数 $\\| \cdot \\|$**|只有少数模态/模式起作用|

你在研究随机矩阵时，如果涉及到**样本协方差矩阵的降维**或者**噪声去除**，核范数就是那个帮你从杂乱无章的噪声中剥离出“真实结构”的精密手术刀。

你目前在做的毕业论文，是否涉及到了这类“凸松弛（Convex Relaxation）”的证明？比如证明在什么采样条件下，最小化核范数一定能还原出真实的秩？

## 7.1 Norms

Recall that the magnitude of any scalar number is easily measured by its modulus, i.e., $|c|$ for $c \in \mathbb{C}$. The same can be done for vectors in $\mathbb{C}^n$ and matrices in $\mathbb{C}^{n \times n}$ through a nonnegative measure called the norm. Usually, norms are used to measure errors in matrix computations. We introduce vector norms first and then matrix norms.

>[!definition] Definition 7.1.
>A norm is a function that assigns to each vector $\mathbf{x} \in \mathbb{C}^n$ a real number $\|\mathbf{x}\|$, called the norm of $\mathbf{x}$, such that the following three axioms are satisfied:
>1. $\|\mathbf{x}\| \geq 0$ for all $\mathbf{x} \in \mathbb{C}^n$, and $\|\mathbf{x}\| = 0$ if and only if $\mathbf{x} = \mathbf{0}$.
>2. $\|\alpha \mathbf{x}\| = |\alpha| \cdot \|\mathbf{x}\|$ for all $\mathbf{x} \in \mathbb{C}^n$ and $\alpha \in \mathbb{C}$.
>3. $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$ for all $\mathbf{x}, \mathbf{y} \in \mathbb{C}^n$.

Let $\mathbf{x} = [x_1, x_2, \ldots, x_n]^T \in \mathbb{C}^n$. A useful class of vector norms is the $p$-norm, defined as
$$\|\mathbf{x}\|_p := \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p} \tag{7.1}$$
for $p \geq 1$.

>[!example] Example 7.1.
>In (7.1), let $p \to \infty$ and show that
>$$\|\mathbf{x}\|_\infty = \max_{1 \leq i \leq n} |x_i|.$$

***Proof.*** Note that
$$\max_{1 \leq i \leq n} |x_i| = \left( \max_{1 \leq i \leq n} |x_i|^p \right)^{1/p} \leq \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p} = \|\mathbf{x}\|_p$$
and
$$\|\mathbf{x}\|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p} \leq \left( n \max_{1 \leq i \leq n} |x_i|^p \right)^{1/p} = n^{1/p} \max_{1 \leq i \leq n} |x_i|.$$
Then
$$\|\mathbf{x}\|_\infty = \max_{1 \leq i \leq n} |x_i|, \qquad \text{as } p \to \infty. \qquad \square$$

The following $p$-norms are the most widely used norms in practice:
$$\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|, \qquad \|\mathbf{x}\|_2 = \left( \sum_{i=1}^{n} |x_i|^2 \right)^{1/2}, \qquad \|\mathbf{x}\|_\infty = \max_{1 \leq i \leq n} |x_i|.$$

We now turn to matrix norms.

>[!definition] Definition 7.2.
>A matrix norm is a function that assigns to each $A \in \mathbb{C}^{n \times n}$ a real number $\|A\|$, called the norm of $A$, such that the following four axioms are satisfied:
>1. $\|A\| \geq 0$ for all $A \in \mathbb{C}^{n \times n}$, and $\|A\| = 0$ if and only if $A = 0$.
>2. $\|\alpha A\| = |\alpha| \cdot \|A\|$ for all $A \in \mathbb{C}^{n \times n}$ and $\alpha \in \mathbb{C}$.
>3. $\|A + B\| \leq \|A\| + \|B\|$ for all $A, B \in \mathbb{C}^{n \times n}$.
>4. $\|AB\| \leq \|A\| \cdot \|B\|$ for all $A, B \in \mathbb{C}^{n \times n}$.

***Remark.*** A matrix norm must satisfy four axioms, whereas a vector norm only requires three. The former three axioms of a matrix norm coincide with those of vector norms. The crucial fourth axiom (d) specifically governs the behavior of norm under matrix multiplication, preserving the algebraic structure essential for matrix computations.

>[!definition] Definition 7.3.
>If a vector norm $\|\cdot\|_v$ and a matrix norm $\|\cdot\|_M$ satisfy
>$$\|A\mathbf{x}\|_v \leq \|A\|_M \|\mathbf{x}\|_v$$
>for $A \in \mathbb{C}^{n \times n}$ and $\mathbf{x} \in \mathbb{C}^n$, then these norms are called compatible.

For any vector norm $\|\cdot\|_v$, a matrix norm $\|\cdot\|_M$ induced by $\|\cdot\|_v$ can be defined naturally in the following way:
$$\|A\|_M := \max_{\mathbf{x} \neq \mathbf{0}} \frac{\|A\mathbf{x}\|_v}{\|\mathbf{x}\|_v} = \max_{\|\mathbf{x}\|_v = 1} \|A\mathbf{x}\|_v.$$

One can check that $\|\cdot\|_M$ defined above satisfies the four axioms in Definition 7.2. See Exercise 7.4. Moreover, $\|\cdot\|_v$ and $\|\cdot\|_M$ are compatible. The most commonly used matrix norms are the matrix $p$-norms induced by the corresponding vector $p$-norms for $p = 1, 2, \infty$.

>[!theo] Theorem 7.1.
>Let $A = [a_{ij}] \in \mathbb{C}^{n \times n}$, $\mathbf{r}_i$ be the $i$th row of $A$, and $\mathbf{c}_j$ be the $j$th column of $A$. Then
>1. $\|A\|_1 = \max_{\|\mathbf{x}\|_1 = 1} \|A\mathbf{x}\|_1 = \max_{1 \leq j \leq n} \sum_{i=1}^{n} |a_{ij}| = \max_{1 \leq j \leq n} \|\mathbf{c}_j\|_1$.
>2. $\|A\|_\infty = \max_{\|\mathbf{x}\|_\infty = 1} \|A\mathbf{x}\|_\infty = \max_{1 \leq i \leq n} \sum_{j=1}^{n} |a_{ij}| = \max_{1 \leq i \leq n} \|\mathbf{r}_i\|_1$.
>3. $\|A\|_2 = \max_{\|\mathbf{x}\|_2 = 1} \|A\mathbf{x}\|_2 = \sqrt{\lambda_{\max}(A^*A)}$, where $\lambda_{\max}(A^*A)$ denotes the largest eigenvalue of $A^*A$.

***Proof.*** We only prove (i) and (iii). For (i), let
$$\|\mathbf{c}_{j_0}\|_1 = \max_{1 \leq j \leq n} \|\mathbf{c}_j\|_1.$$
Then for any vector $\mathbf{x} = [x_1, x_2, \ldots, x_n]^T \in \mathbb{C}^n$ which satisfies
$$\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i| = 1,$$
we have
$$\|A\mathbf{x}\|_1 = \left\| \sum_{j=1}^{n} x_j \mathbf{c}_j \right\|_1 \leq \sum_{j=1}^{n} |x_j| \cdot \|\mathbf{c}_j\|_1 \leq \left(\sum_{j=1}^{n} |x_j|\right) \cdot \max_{1 \leq j \leq n} \|\mathbf{c}_j\|_1 = \|\mathbf{c}_{j_0}\|_1.$$
Let $\mathbf{e}_{j_0}$ denote the $j_0$th column of the $n$-by-$n$ identity matrix $I$ and then
$$\|A\mathbf{e}_{j_0}\|_1 = \|\mathbf{c}_{j_0}\|_1.$$
Consequently,
$$\|A\|_1 = \max_{\|\mathbf{x}\|_1 = 1} \|A\mathbf{x}\|_1 = \|\mathbf{e}_{j_0}\|_1 = \max_{1 \leq j \leq n} \|\mathbf{c}_j\|_1 = \max_{1 \leq j \leq n} \sum_{i=1}^{n} |a_{ij}|.$$

For (iii), note that
$$\|A\|_2^2 = \max_{\|\mathbf{x}\|_2 = 1} \|A\mathbf{x}\|_2^2 = \max_{\|\mathbf{x}\|_2 = 1} (A\mathbf{x})^*(A\mathbf{x}) = \max_{\|\mathbf{x}\|_2 = 1} \mathbf{x}^*(A^*A)\mathbf{x}.$$
Since $A^*A$ is Hermitian positive semidefinite, its eigenvalues can be ordered as:
$$0 \leq \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n.$$
Let $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n \in \mathbb{C}^n$ be the orthonormal eigenvectors corresponding to $\lambda_1, \lambda_2, \ldots, \lambda_n$, respectively. Then for any vector $\mathbf{x} \in \mathbb{C}^n$ with $\|\mathbf{x}\|_2 = 1$, we have
$$\mathbf{x} = \sum_{i=1}^{n} \alpha_i \mathbf{v}_i, \qquad \sum_{i=1}^{n} |\alpha_i|^2 = 1.$$
Therefore,
$$\mathbf{x}^* A^* A \mathbf{x} = \sum_{i=1}^{n} \lambda_i |\alpha_i|^2 \leq \lambda_n.$$
On the other hand, let $\mathbf{x} = \mathbf{v}_n$, we have
$$\mathbf{x}^* A^* A \mathbf{x} = \mathbf{v}_n^* A^* A \mathbf{v}_n = \mathbf{v}_n^* \lambda_n \mathbf{v}_n = \lambda_n.$$
Thus,
$$\|A\|_2^2 = \max_{\|\mathbf{x}\|_2 = 1} \|A\mathbf{x}\|_2^2 = \lambda_n = \lambda_{\max}(A^*A)$$
and (iii) holds. $\square$

Another useful norm is the Frobenius norm defined by
$$\|A\|_F := \left( \sum_{j=1}^{n} \sum_{i=1}^{n} |a_{ij}|^2 \right)^{1/2} = \left( \sum_{j=1}^{n} \|\mathbf{c}_j\|_2^2 \right)^{1/2} = \left( \sum_{i=1}^{n} \|\mathbf{r}_i\|_2^2 \right)^{1/2} = [\operatorname{tr}(A^*A)]^{1/2} = \left[ \sum_{k=1}^{n} \lambda_k(A^*A) \right]^{1/2},$$
where $\operatorname{tr}(A^*A)$ is the trace of the matrix $A^*A$.

>[!example] Example 7.2.
>Show that the Frobenius norm is a matrix norm.

***Proof.*** We only show that the Frobenius norm satisfies Axioms (c) and (d) in Definition 7.2. Let $A = [a_{ij}]$ and $B = [b_{ij}]$ be $n \times n$ matrices. For (c), it follows from the Cauchy-Schwarz inequality that
$$\|A + B\|_F^2 = \sum_{j=1}^{n} \sum_{i=1}^{n} |a_{ij} + b_{ij}|^2 \leq \sum_{j=1}^{n} \sum_{i=1}^{n} (|a_{ij}|^2 + |b_{ij}|^2 + 2|a_{ij}||b_{ij}|)$$
$$= \|A\|_F^2 + \|B\|_F^2 + 2 \sum_{j=1}^{n} \sum_{i=1}^{n} |a_{ij}||b_{ij}|$$
$$\leq \|A\|_F^2 + \|B\|_F^2 + 2 \left[ \left( \sum_{j=1}^{n} \sum_{i=1}^{n} |a_{ij}|^2 \right) \left( \sum_{j=1}^{n} \sum_{i=1}^{n} |b_{ij}|^2 \right) \right]^{1/2}$$
$$= \|A\|_F^2 + \|B\|_F^2 + 2\|A\|_F \|B\|_F = \left( \|A\|_F + \|B\|_F \right)^2$$
and (c) holds.

For (d), it follows by the Cauchy-Schwarz inequality again that
$$\|AB\|_F^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} \left| \sum_{k=1}^{n} a_{ik} b_{kj} \right|^2 \leq \sum_{i=1}^{n} \sum_{j=1}^{n} \left( \sum_{p=1}^{n} |a_{ip}|^2 \right) \left( \sum_{q=1}^{n} |b_{qj}|^2 \right)$$
$$= \left( \sum_{i=1}^{n} \sum_{p=1}^{n} |a_{ip}|^2 \right) \left( \sum_{j=1}^{n} \sum_{q=1}^{n} |b_{qj}|^2 \right) = \|A\|_F^2 \cdot \|B\|_F^2$$
and (d) holds. $\square$

One of the most important properties of norms $\|\cdot\|_2$ and $\|\cdot\|_F$ is the unitary invariance.

>[!theo] Theorem 7.2.
>For any unitary matrices $Q$ and $Z$, we have
>$$\|A\|_2 = \|QAZ\|_2, \qquad \|A\|_F = \|QAZ\|_F.$$

***Proof.*** Let $\mathbf{y} = Q\mathbf{x}$ with $\mathbf{x} \in \mathbb{C}^n$. For the vector norm $\|\cdot\|_2$, we have
$$\|\mathbf{y}\|_2^2 = \mathbf{y}^*\mathbf{y} = (Q\mathbf{x})^*(Q\mathbf{x}) = \mathbf{x}^*Q^*Q\mathbf{x} = \mathbf{x}^*\mathbf{x} = \|\mathbf{x}\|_2^2.$$

For the matrix norm $\|\cdot\|_2$, we have by Theorem 7.1 (iii),
$$\|QAZ\|_2^2 = \lambda_{\max}[(QAZ)^*(QAZ)] = \lambda_{\max}(Z^*A^*AZ) = \lambda_{\max}(A^*A) = \|A\|_2^2.$$

For the norm $\|\cdot\|_F$, let $W = AZ$ and $\mathbf{w}_j$ be the $j$th column of $W$. By the unitary invariance of the vector norm $\|\cdot\|_2$, we obtain
$$\|QAZ\|_F^2 = \|QW\|_F^2 = \sum_{j=1}^{n} \|Q\mathbf{w}_j\|_2^2 = \sum_{j=1}^{n} \|\mathbf{w}_j\|_2^2 = \|W\|_F^2 = \|AZ\|_F^2.$$

It remains to show that $\|AZ\|_F^2 = \|A\|_F^2$. Let $\mathbf{r}_i$ be the $i$th row of $A$. Again from the unitary invariance of the vector norm $\|\cdot\|_2$,
$$\|AZ\|_F^2 = \sum_{i=1}^{n} \|\mathbf{r}_i Z\|_2^2 = \sum_{i=1}^{n} \|Z^* \mathbf{r}_i^*\|_2^2 = \sum_{i=1}^{n} \|\mathbf{r}_i^*\|_2^2 = \|A\|_F^2. \qquad \square$$

Next, we introduce the following definitions of the spectrum and spectral radius of a matrix.

>[!definition] Definition 7.4.
>Let $A \in \mathbb{C}^{n \times n}$. Then the set of all the eigenvalues of $A$ is called the spectrum of $A$ and
>$$\rho(A) = \max\{|\lambda| \;:\; \lambda \text{ belongs to the spectrum of } A\}$$
>is called the spectral radius of $A$.

Before establishing the relationship between the spectral radius and matrix norms, we introduce the following fundamental theorem.

>[!theo] Theorem 7.3. (Jordan decomposition theorem)
>Let $A \in \mathbb{C}^{n \times n}$. Then there exists a nonsingular matrix $X \in \mathbb{C}^{n \times n}$ such that
>$$X^{-1}AX = \operatorname{diag}(J_1, J_2, \ldots, J_p) := \begin{bmatrix} J_1 & \mathbf{0} & \cdots & \mathbf{0} \\ \mathbf{0} & J_2 & \cdots & \mathbf{0} \\ \vdots & \vdots & \ddots & \vdots \\ \mathbf{0} & \mathbf{0} & \cdots & J_p \end{bmatrix}, \tag{7.2}$$
>where
>$$J_i = \begin{bmatrix} \lambda_i & 1 & 0 & \cdots & 0 \\ 0 & \lambda_i & 1 & \ddots & \vdots \\ \vdots & 0 & \ddots & \ddots & 0 \\ \vdots & & \ddots & \ddots & 1 \\ 0 & \cdots & \cdots & 0 & \lambda_i \end{bmatrix} \in \mathbb{C}^{n_i \times n_i}, \qquad 1 \leq i \leq p, \tag{7.3}$$
>with $n_1 + n_2 + \cdots + n_p = n$. The matrix $\operatorname{diag}(J_1, J_2, \ldots, J_p)$ is called the Jordan canonical form of $A$ and each $J_i$ is called a Jordan block. If $A$ is a real matrix with only real eigenvalues, then the matrix $X$ can also be chosen to be real.

>[!theo] Theorem 7.4.
>Let $A \in \mathbb{C}^{n \times n}$. Then
>1. For any matrix norm, we have
>$$\rho(A) \leq \|A\|.$$
>2. For any $\epsilon > 0$, there exists a norm $\|\cdot\|_\epsilon$ defined on $\mathbb{C}^{n \times n}$ such that
>$$\|A\|_\epsilon \leq \rho(A) + \epsilon.$$

***Proof.*** For (i), let $\mathbf{x} \in \mathbb{C}^n$ satisfy
$$\mathbf{x} \neq \mathbf{0}, \qquad A\mathbf{x} = \lambda\mathbf{x}, \qquad |\lambda| = \rho(A).$$
Then
$$\rho(A)\|\mathbf{x}\mathbf{e}_1^T\| = \|\lambda\mathbf{x}\mathbf{e}_1^T\| = \|A\mathbf{x}\mathbf{e}_1^T\| \leq \|A\| \cdot \|\mathbf{x}\mathbf{e}_1^T\|.$$
Hence
$$\rho(A) \leq \|A\|.$$

For (ii), by Theorem 7.3, there exists a nonsingular matrix $X \in \mathbb{C}^{n \times n}$ such that
$$X^{-1}AX = \begin{bmatrix} \lambda_1 & \delta_1 \\ & \lambda_2 & \delta_2 \\ & & \ddots & \ddots \\ & & & \lambda_{n-1} & \delta_{n-1} \\ & & & & \lambda_n \end{bmatrix},$$
where $\delta_i = 1$ or $0$. For any given $\epsilon > 0$, let
$$D_\epsilon = \operatorname{diag}(1, \epsilon, \epsilon^2, \ldots, \epsilon^{n-1}).$$
Then
$$D_\epsilon^{-1} X^{-1} A X D_\epsilon = \begin{bmatrix} \lambda_1 & \epsilon\delta_1 \\ & \lambda_2 & \epsilon\delta_2 \\ & & \ddots & \ddots \\ & & & \lambda_{n-1} & \epsilon\delta_{n-1} \\ & & & & \lambda_n \end{bmatrix}.$$
Now, define
$$\|G\|_\epsilon := \|D_\epsilon^{-1} X^{-1} G X D_\epsilon\|_\infty, \qquad G \in \mathbb{C}^{n \times n}. \tag{7.4}$$
It can be verified that $\|\cdot\|_\epsilon$ is a matrix norm (see Exercise 7.10). Therefore,
$$\|A\|_\epsilon = \|D_\epsilon^{-1} X^{-1} A X D_\epsilon\|_\infty \leq \max\left\{ \max_{1 \leq i \leq n-1} (|\lambda_i| + |\epsilon\delta_i|), |\lambda_n| \right\} \leq \rho(A) + \epsilon. \qquad \square$$

>[!definition] Definition 7.5.
>For a matrix $A = [a_{ij}] \in \mathbb{C}^{n \times n}$ and a sequence of matrices $\{A_{(k)}\}$, where $A_{(k)} = [a_{ij}^{(k)}] \in \mathbb{C}^{n \times n}$, we say that $\lim_{k \to \infty} A_{(k)} = A$ if $\lim_{k \to \infty} a_{ij}^{(k)} = a_{ij}$ for $1 \leq i, j \leq n$.

***Remark.*** In fact, for any matrix norm, we have
$$\lim_{k \to \infty} A_{(k)} = A \iff \lim_{k \to \infty} \|A_{(k)} - A\| = 0.$$
For more details, we refer to [18].

>[!theo] Theorem 7.5.
>Let $A \in \mathbb{C}^{n \times n}$. Then $\lim_{k \to \infty} A^k = \mathbf{0}$ if and only if $\rho(A) < 1$.

***Proof.*** We first assume that $\lim_{k \to \infty} A^k = \mathbf{0}$, which implies
$$\lim_{k \to \infty} \|A^k\| = 0. \tag{7.5}$$
Let $\lambda$ be an eigenvalue of $A$ such that $\rho(A) = |\lambda|$. Then $\lambda^k$ is an eigenvalue of $A^k$ for any $k$. By Theorem 7.4 (i), we have for any $k$,
$$\rho(A)^k = |\lambda|^k = |\lambda^k| \leq \rho(A^k) \leq \|A^k\|. \tag{7.6}$$
It follows from (7.6) and (7.5) that
$$0 \leq \lim_{k \to \infty} \rho(A)^k \leq \lim_{k \to \infty} \|A^k\| = 0.$$
Therefore,
$$\lim_{k \to \infty} \rho(A)^k = 0,$$
which implies $\rho(A) < 1$.

Conversely, assume that $\rho(A) < 1$. By Theorem 7.4 (ii), there exists a matrix norm $\|\cdot\|_\epsilon$ such that $\|A\|_\epsilon < 1$. Therefore,
$$0 \leq \|A^k\|_\epsilon \leq \|A\|_\epsilon^k \to 0, \qquad \text{as } k \to \infty,$$
which implies $\lim_{k \to \infty} A^k = \mathbf{0}$. $\square$

## 7.2 Perturbation analysis of linear systems

The following definition is a generalization of Definition 1.1.

>[!definition] Definition 7.6.
>Let $\|\cdot\|$ be any vector norm and $\hat{\mathbf{x}}$ be an approximation of $\mathbf{x}$. The absolute error in $\hat{\mathbf{x}}$ is defined by $\|\mathbf{x} - \hat{\mathbf{x}}\|$, and the relative error in $\hat{\mathbf{x}}$ is defined by $\|\mathbf{x} - \hat{\mathbf{x}}\| / \|\mathbf{x}\|$. Analogously, the concepts of absolute error and relative error can be extended to matrix case.

Before carrying out the perturbation analysis, we first consider the following simple example.

>[!example] Example 7.3.
>Let $A\mathbf{x} = \mathbf{b}$ be given by
>$$\begin{bmatrix} 2.0001 & 1.9999 \\ 1.9999 & 2.0001 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 4 \\ 4 \end{bmatrix}.$$
>The solution of this linear system is $\mathbf{x} = [1, 1]^T$. If there is a small perturbation in $\mathbf{b}$ given by
>$$\beta = \begin{bmatrix} 1 \times 10^{-4} \\ -1 \times 10^{-4} \end{bmatrix},$$
>the original system becomes $A\hat{\mathbf{x}} = \hat{\mathbf{b}}$:
>$$\begin{bmatrix} 2.0001 & 1.9999 \\ 1.9999 & 2.0001 \end{bmatrix} \begin{bmatrix} \hat{x}_1 \\ \hat{x}_2 \end{bmatrix} = \begin{bmatrix} 4.0001 \\ 3.9999 \end{bmatrix},$$
>where $\hat{\mathbf{b}} = \mathbf{b} + \beta$. The solution of this perturbed system is $\hat{\mathbf{x}} = [1.5, 0.5]^T$. We therefore have
>$$\frac{\|\mathbf{x} - \hat{\mathbf{x}}\|_\infty}{\|\mathbf{x}\|_\infty} = \frac{1}{2} \tag{7.7}$$
>and
>$$\frac{\|\mathbf{b} - \hat{\mathbf{b}}\|_\infty}{\|\mathbf{b}\|_\infty} = \frac{1}{40000}. \tag{7.8}$$
>It follows that the relative error in $\hat{\mathbf{x}}$ is 20000 times of that in $\hat{\mathbf{b}}$.

Consequently when solving a linear system $A\mathbf{x} = \mathbf{b}$, a good measurement, which can tell us how sensitive the computed solution is to small perturbations in $A$ and $\mathbf{b}$, is needed. This sensitivity is quantified by the condition number of matrix $A$.

>[!definition] Definition 7.7.
>Let $\|\cdot\|$ be any norm of matrix and $A$ be a nonsingular matrix. The condition number of $A$ is defined by
>$$\kappa(A) := \|A\| \cdot \|A^{-1}\|. \tag{7.9}$$

Obviously, the condition number depends on the matrix norm used. For any $p$-norm $\|\cdot\|_p$, let
$$\kappa_p(A) := \|A\|_p \|A^{-1}\|_p.$$
Then
$$\kappa_p(A) = \|A\|_p \|A^{-1}\|_p \geq \|A \cdot A^{-1}\|_p = \|I\|_p = 1.$$

In the following discussion, the vector norm and the matrix norm are always assumed to be compatible. The next theorem gives the effect of perturbations of the vector $\mathbf{b}$ on the solution of $A\mathbf{x} = \mathbf{b}$ in terms of the condition number of $A$.

>[!theo] Theorem 7.6.
>Let $A$ be a nonsingular matrix. If $A\mathbf{x} = \mathbf{b}$ and $A\hat{\mathbf{x}} = \hat{\mathbf{b}}$, where $\hat{\mathbf{b}}$ is a perturbed vector of $\mathbf{b}$, then
>$$\frac{\|\mathbf{x} - \hat{\mathbf{x}}\|}{\|\mathbf{x}\|} \leq \kappa(A) \frac{\|\mathbf{b} - \hat{\mathbf{b}}\|}{\|\mathbf{b}\|}. \tag{7.10}$$

***Proof.*** Since
$$\mathbf{x} = A^{-1}\mathbf{b}, \qquad \hat{\mathbf{x}} = A^{-1}\hat{\mathbf{b}},$$
we have
$$\|\mathbf{x} - \hat{\mathbf{x}}\| = \|A^{-1}(\mathbf{b} - \hat{\mathbf{b}})\| \leq \|A^{-1}\| \cdot \|\mathbf{b} - \hat{\mathbf{b}}\|. \tag{7.11}$$
It follows from $\|\mathbf{b}\| = \|A\mathbf{x}\| \leq \|A\| \cdot \|\mathbf{x}\|$ that
$$\frac{1}{\|\mathbf{x}\|} \leq \frac{\|A\|}{\|\mathbf{b}\|}. \tag{7.12}$$
Combining (7.11), (7.12), and (7.9) yields
$$\frac{\|\mathbf{x} - \hat{\mathbf{x}}\|}{\|\mathbf{x}\|} \leq \kappa(A) \frac{\|\mathbf{b} - \hat{\mathbf{b}}\|}{\|\mathbf{b}\|}. \qquad \square$$

Let us consider Example 7.3 again. The condition number of $A$ is given by
$$\kappa_\infty(A) = \|A\|_\infty \|A^{-1}\|_\infty = 4 \times 5000 = 20000, \tag{7.13}$$
which gives a bound of the relative error in $\hat{\mathbf{x}}$ by using (7.10), (7.13), and (7.8) as follows:
$$\frac{\|\mathbf{x} - \hat{\mathbf{x}}\|_\infty}{\|\mathbf{x}\|_\infty} \leq \kappa_\infty(A) \frac{\|\mathbf{b} - \hat{\mathbf{b}}\|_\infty}{\|\mathbf{b}\|_\infty} = 20000 \cdot \frac{1}{40000} = \frac{1}{2}.$$
This estimate coincides with (7.7). Since $A$ is nearly singular, $\|A^{-1}\|_\infty$ is very large and so is $\kappa_\infty(A)$. As a result, small perturbations in $\mathbf{b}$ can lead to substantial relative errors in the solution $\hat{\mathbf{x}}$.

The following theorem quantifies how perturbations in both the matrix $A$ and the vector $\mathbf{b}$ affect the solution of the linear system $A\mathbf{x} = \mathbf{b}$, expressed in terms of the condition number of $A$.

>[!theo] Theorem 7.7.
>Let $A$ be a nonsingular matrix and $\hat{A}$ be a perturbed matrix of $A$ such that
>$$\|A - \hat{A}\| \cdot \|A^{-1}\| < 1.$$
>If $A\mathbf{x} = \mathbf{b}$ and $\hat{A}\hat{x} = \hat{b}$ where $\hat{b}$ is a perturbed vector of $b$, then
>$$\frac{\|\mathbf{x} - \hat{\mathbf{x}}\|}{\|\mathbf{x}\|} \leq \frac{\kappa(A)}{1 - \kappa(A)\frac{\|A - \hat{A}\|}{\|A\|}} \left( \frac{\|A - \hat{A}\|}{\|A\|} + \frac{\|\mathbf{b} - \hat{\mathbf{b}}\|}{\|\mathbf{b}\|} \right).$$

***Proof.*** Let
$$E = A - \hat{A}, \qquad \beta = \mathbf{b} - \hat{\mathbf{b}}.$$
It follows by subtracting $A\mathbf{x} = \mathbf{b}$ from $\hat{A}\hat{x} = \hat{b}$ that
$$A(\mathbf{x} - \hat{\mathbf{x}}) = -E\hat{\mathbf{x}} + \beta.$$
Furthermore,
$$\frac{\|\mathbf{x} - \hat{\mathbf{x}}\|}{\|\mathbf{x}\|} \leq \|A^{-1}E\| \frac{\|\hat{\mathbf{x}}\|}{\|\mathbf{x}\|} + \|A^{-1}\| \frac{\|A\mathbf{x}\|}{\|\mathbf{x}\|} \frac{\|\beta\|}{\|\mathbf{b}\|}.$$
By using
$$\|\hat{\mathbf{x}}\| \leq \|\hat{\mathbf{x}} - \mathbf{x}\| + \|\mathbf{x}\|, \qquad \|A\mathbf{x}\| \leq \|A\| \cdot \|\mathbf{x}\|,$$
we then have
$$\frac{\|\mathbf{x} - \hat{\mathbf{x}}\|}{\|\mathbf{x}\|} \leq \|A^{-1}E\| \frac{\|\mathbf{x} - \hat{\mathbf{x}}\|}{\|\mathbf{x}\|} + \|A^{-1}E\| + \|A^{-1}\| \cdot \|A\| \frac{\|\beta\|}{\|\mathbf{b}\|},$$
i.e.,
$$(1 - \|A^{-1}E\|) \frac{\|\mathbf{x} - \hat{\mathbf{x}}\|}{\|\mathbf{x}\|} \leq \|A^{-1}E\| + \kappa(A) \frac{\|\beta\|}{\|\mathbf{b}\|}.$$
Since
$$\|A^{-1}E\| \leq \|A^{-1}\| \cdot \|E\| = \|A^{-1}\| \cdot \|A - \hat{A}\| < 1,$$
we obtain
$$\frac{\|\mathbf{x} - \hat{\mathbf{x}}\|}{\|\mathbf{x}\|} \leq (1 - \|A^{-1}E\|)^{-1} \left( \|A^{-1}E\| + \kappa(A) \frac{\|\beta\|}{\|\mathbf{b}\|} \right).$$
By using
$$\|A^{-1}E\| \leq \|A^{-1}\| \cdot \|E\| = \kappa(A) \frac{\|E\|}{\|A\|},$$
we finally have
$$\frac{\|\mathbf{x} - \hat{\mathbf{x}}\|}{\|\mathbf{x}\|} \leq \frac{\kappa(A)}{1 - \kappa(A)\frac{\|E\|}{\|A\|}} \left( \frac{\|E\|}{\|A\|} + \frac{\|\beta\|}{\|\mathbf{b}\|} \right). \qquad \square$$

***Remark.*** A matrix $A$ is called well-conditioned when its condition number $\kappa(A)$ is small, and ill-conditioned when $\kappa(A)$ is large. Theorems 7.6 and 7.7 show that for a well-conditioned matrix, the relative error in the computed solution $\hat{\mathbf{x}}$ remains small provided the relative errors in both $\hat{A}$ and $\hat{\mathbf{b}}$ are small. However, for an ill-conditioned matrix, the relative error in $\hat{\mathbf{x}}$ can be very large even though the relative errors in both $\hat{A}$ and $\hat{\mathbf{b}}$ are small.

## Exercises

**7.1.** Let $\mathbf{x} \in \mathbb{C}^n$. Show that
1. $\|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1 \leq \sqrt{n}\|\mathbf{x}\|_2$.
2. $\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \sqrt{n}\|\mathbf{x}\|_\infty$.

**7.2.** Let $A$ be a symmetric positive definite matrix. Define $\|\mathbf{x}\|_A := \sqrt{\mathbf{x}^T A \mathbf{x}}$ for any $\mathbf{x} \in \mathbb{R}^n$. Show that it is a vector norm, called the energy norm.

**7.3.** Show that if
$$A = \begin{bmatrix} I & Z \\ \mathbf{0} & I \end{bmatrix},$$
where $I$ is an $n$-by-$n$ identity matrix, then
$$\kappa_F(A) = \|A\|_F \|A^{-1}\|_F = 2n + \|Z\|_F^2.$$

**7.4.** Show that $\|A\|_M = \max_{\mathbf{x} \neq \mathbf{0}} \dfrac{\|A\mathbf{x}\|_v}{\|\mathbf{x}\|_v}$ is a matrix norm, where $\|\cdot\|_v$ is a vector norm.

**7.5.** Let $A \in \mathbb{C}^{n \times n}$. Show that
$$\|A\|_\infty = \max_{1 \leq i \leq n} \sum_{j=1}^{n} |a_{ij}| = \max_{1 \leq i \leq n} \|\mathbf{r}_i\|_1,$$
where $\mathbf{r}_1, \mathbf{r}_2, \ldots, \mathbf{r}_n$ are rows of $A$.

**7.6.** Let $A = [a_{ij}]$ be an $n$-by-$n$ matrix. Define
$$\|A\| = \max_{i,j} |a_{ij}|.$$
Is $\|\cdot\|$ a matrix norm? Give reasons for your answer.

**7.7.** Let $A$ be nonsingular. Show that
$$\|A^{-1}\|_2^{-1} = \min_{\|\mathbf{x}\|_2 = 1} \|A\mathbf{x}\|_2.$$

**7.8.** Let $A, B \in \mathbb{C}^{n \times n}$. Show that
1. $\|AB\|_F \leq \|A\|_2 \|B\|_F$.
2. $\|AB\|_F \leq \|A\|_F \|B\|_2$.
3. $\|A\|_2^2 \leq \|A\|_1 \|A\|_\infty$.

**7.9.** Show that
$$\left\| \begin{bmatrix} A_{11} & 0 \\ 0 & A_{22} \end{bmatrix} \right\|_2 \leq \left\| \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix} \right\|_2.$$

**7.10.** Show that $\|\cdot\|_\epsilon$ defined by (7.4) is a matrix norm.

**7.11.** Show that if
$$Y = \begin{bmatrix} I & Z \\ \mathbf{0} & I \end{bmatrix},$$
then
$$\kappa_2(Y) \equiv \|Y\|_2 \|Y^{-1}\|_2 = \frac{1}{2} \left( 2 + \sigma^2 + \sqrt{4\sigma^2 + \sigma^4} \right),$$
where $\sigma = \|Z\|_2$.

**7.12.** Let $A \in \mathbb{R}^{n \times n}$ be nonsingular and let $\mathbf{x}, \mathbf{y}, \mathbf{z} \in \mathbb{R}^n$ such that $A\mathbf{x} = \mathbf{b}$ and $A\mathbf{y} = \mathbf{b} + \mathbf{z}$. Show that
$$\frac{\|\mathbf{z}\|_2}{\|A\|_2} \leq \|\mathbf{x} - \mathbf{y}\|_2 \leq \|A^{-1}\|_2 \|\mathbf{z}\|_2.$$
