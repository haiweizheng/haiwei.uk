---
tags:
  - 线性代数
  - 矩阵分解
  - 极分解
  - SVD
  - linear-algebra
  - matrix-decomposition
  - polar-decomposition
  - singular-value-decomposition
source: https://zhuanlan.zhihu.com/p/198941358
---

## 矩阵的极分解

本文着重介绍 $n$ 阶**实方阵**的正交半正定分解。

> [!lemma] 引理 1（实方阵的奇异值分解）
> 设 $A$ 是 $n$ 阶实方阵，则一定存在两个 $n$ 阶正交矩阵 $P, Q$，使得：
> $$A = Q \begin{pmatrix} \Delta & O \\ O & O \end{pmatrix} P$$
> 其中 $\Delta = \mathrm{diag}(\mu_1, \mu_2, \dots, \mu_s)$，$\mu_1 \ge \mu_2 \ge \dots \ge \mu_s > 0$。

**证：** 此时 $AA^T$ 是半正定的，设 $AA^T$ 的非零特征值为 $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_s > 0$，令 $\mu_j = \sqrt{\lambda_j}, j = 1, 2, \dots, s$，则一定存在正交矩阵 $Q$ 使得：

$$Q^T AA^T Q = \begin{pmatrix} \Delta^2 & O \\ O & O \end{pmatrix} = \begin{pmatrix} \Delta & O \\ O & O \end{pmatrix} \begin{pmatrix} \Delta & O \\ O & O \end{pmatrix}$$

令 $Q = (Q_1, Q_2)$，其中 $Q_1$ 是 $n \times s$ 矩阵，$Q_2$ 是 $n \times (n-s)$ 矩阵，则：

$$Q^T AA^T Q = \begin{pmatrix} Q_1^T \\ Q_2^T \end{pmatrix} AA^T (Q_1, Q_2) = \begin{pmatrix} Q_1^T AA^T Q_1 & Q_1^T AA^T Q_2 \\ Q_2^T AA^T Q_1 & Q_2^T AA^T Q_2 \end{pmatrix} = \begin{pmatrix} \Delta^2 & O \\ O & O \end{pmatrix}$$

此时 $Q_1^T AA^T Q_1 = \Delta^2, Q_2^T AA^T Q_2 = O$。

令 $V_1^T = \Delta^{-1} Q_1^T A$，则 $V_1^T V_1 = \Delta^{-1} Q_1^T AA^T Q_1 \Delta^{-1} = E_s$，从而 $V_1$ 的 $s$ 个列向量是单位正交向量组，它总是能扩充为 $\mathbb{R}^n$ 的标准正交基，从而一定存在 $n \times (n-s)$ 矩阵 $V_2$ 满足 $V = (V_1, V_2)$ 是正交矩阵。

由此可证得 $Q^T A V = \begin{pmatrix} \Delta & O \\ O & O \end{pmatrix}$，取 $P = V^T$ 即可得证。$\blacksquare$

> [!lemma] 引理 2
> 设 $B, C$ 均为 $n$ 阶半正定矩阵，且 $B^2 = C^2$，则 $B = C$。

**证：** 此时 $B, C$ 有相同特征值，且同一特征值的（代数）重数相等。

设 $B$ 的两两不同的特征值为 $\lambda_1, \lambda_2, \dots, \lambda_s$，重数分别为 $n_1, n_2, \dots, n_s$，则一定存在正交矩阵 $P, Q$ 使得

$$B = P^T \begin{pmatrix} \lambda_1 E_{n_1} & & & \\ & \lambda_2 E_{n_2} & & \\ & & \ddots & \\ & & & \lambda_s E_{n_s} \end{pmatrix} P, \quad C = Q^T \begin{pmatrix} \lambda_1 E_{n_1} & & & \\ & \lambda_2 E_{n_2} & & \\ & & \ddots & \\ & & & \lambda_s E_{n_s} \end{pmatrix} Q$$

由 $B^2 = C^2$，得

$$P^T \begin{pmatrix} \lambda_1^2 E_{n_1} & & & \\ & \lambda_2^2 E_{n_2} & & \\ & & \ddots & \\ & & & \lambda_s^2 E_{n_s} \end{pmatrix} P = Q^T \begin{pmatrix} \lambda_1^2 E_{n_1} & & & \\ & \lambda_2^2 E_{n_2} & & \\ & & \ddots & \\ & & & \lambda_s^2 E_{n_s} \end{pmatrix} Q$$

则

$$QP^T \begin{pmatrix} \lambda_1^2 E_{n_1} & & & \\ & \lambda_2^2 E_{n_2} & & \\ & & \ddots & \\ & & & \lambda_s^2 E_{n_s} \end{pmatrix} = \begin{pmatrix} \lambda_1^2 E_{n_1} & & & \\ & \lambda_2^2 E_{n_2} & & \\ & & \ddots & \\ & & & \lambda_s^2 E_{n_s} \end{pmatrix} QP^T$$

此时一定有 $QP^T$ 是分块对角矩阵

$$QP^T = \begin{pmatrix} M_1 & & & \\ & M_2 & & \\ & & \ddots & \\ & & & M_s \end{pmatrix}$$

其中 $M_j$ 是 $n_j$ 阶矩阵，从而 $QP^T$ 与

$$\begin{pmatrix} \lambda_1 E_{n_1} & & & \\ & \lambda_2 E_{n_2} & & \\ & & \ddots & \\ & & & \lambda_s E_{n_s} \end{pmatrix}$$

可交换。从而

$$\begin{aligned}
B &= P^T \begin{pmatrix} \lambda_1 E_{n_1} & & & \\ & \lambda_2 E_{n_2} & & \\ & & \ddots & \\ & & & \lambda_s E_{n_s} \end{pmatrix} P \\
&= Q^T QP^T \begin{pmatrix} \lambda_1 E_{n_1} & & & \\ & \lambda_2 E_{n_2} & & \\ & & \ddots & \\ & & & \lambda_s E_{n_s} \end{pmatrix} P Q^T Q \\
&= Q^T \begin{pmatrix} \lambda_1 E_{n_1} & & & \\ & \lambda_2 E_{n_2} & & \\ & & \ddots & \\ & & & \lambda_s E_{n_s} \end{pmatrix} QP^T P Q^T Q \\
&= Q^T \begin{pmatrix} \lambda_1 E_{n_1} & & & \\ & \lambda_2 E_{n_2} & & \\ & & \ddots & \\ & & & \lambda_s E_{n_s} \end{pmatrix} Q \\
&= C
\end{aligned}$$

从而 $B = C$。$\blacksquare$

> [!theorem] 定理 1（实方阵的极分解）
> 设 $A$ 是 $n$ 阶实方阵，则一定存在正交矩阵 $Q$ 以及唯一的半正定矩阵 $T$，使得 $A = QT$，并且当 $A$ 可逆时，$Q$ 也是唯一的（此时 $T$ 是正定的）。

**证：**

**存在性：** 设 $A$ 的奇异值分解为

$$A = Q_1 \begin{pmatrix} \Delta & O \\ O & O \end{pmatrix} Q_2$$

取

$$Q = Q_1 Q_2, \quad T = Q_2^T \begin{pmatrix} \Delta & O \\ O & O \end{pmatrix} Q_2$$

即可得证。

**唯一性：** 设 $A$ 还有极分解 $A = PS$，则：

$$A^T A = (PS)^T (PS) = S P^T P S = S^2$$
$$A^T A = (QT)^T (QT) = T Q^T Q T = T^2$$

从而 $S^2 = T^2$。由于 $S, T$ 均为半正定矩阵，由引理 2 可知 $S = T$。

此时有 $PS = QT$。若加上 $A$ 可逆的条件，则 $S, T$ 可逆，从而 $P = Q$。$\blacksquare$

> [!theorem] 定理 2（复方阵的极分解）
> 设 $A$ 是 $n$ 阶复方阵，则一定存在酉矩阵 $U$ 以及唯一的半正定 Hermite 矩阵 $T$，使得 $A = UT$，并且当 $A$ 可逆时，$U$ 也是唯一的（此时 $T$ 是正定 Hermite 矩阵）。
