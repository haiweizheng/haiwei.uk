---
draft: true
---
![[SVD-1775226622135.webp]]

>[!proposition] Symmetric matrix spectral theorem
>Let $M \in \mathbb{R}^{n\times n}$ be symmetric. Then there exist an orthogonal matrix $Q \in \mathbb{R}^{n\times n}$ and a diagonal matrix $\Lambda$ such that
>$$M = Q\Lambda Q^\top.$$
>In particular, eigenvectors corresponding to distinct eigenvalues are orthogonal.

***Proof.*** By Schur decomposition,
$$M = QTQ^\top,$$
where $Q^\top Q = I$ and $T$ is upper triangular. Since $M^\top = M$,
$$QTQ^\top = M = M^\top = (QTQ^\top)^\top = QT^\top Q^\top,$$
hence
$$T = T^\top.$$
Thus $T$ is both upper triangular and symmetric, so $T$ is diagonal. Write
$$T=\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n),$$
then
$$M=Q\Lambda Q^\top.$$
If $Mx=\lambda x$ and $My=\mu y$ with $\lambda\ne\mu$, then
$$\lambda x^\top y = (Mx)^\top y = x^\top M^\top y = x^\top My = \mu x^\top y,$$
so
$$ (\lambda-\mu)x^\top y=0 \implies x^\top y=0.$$

>[!proposition] Positive semidefinite structure of $A^\top A$ and $AA^\top$
Let $A_{m\times n} \in \mathbb{R}^{m\times n}$. Then
$$A^\top A \in \mathbb{R}^{n\times n},\quad AA^\top \in \mathbb{R}^{m\times m}$$
are symmetric positive semidefinite. Moreover, $A^\top A$ and $AA^\top$ have the same nonzero eigenvalues with the same algebraic multiplicities.

***Proof.*** For any $x \in \mathbb{R}^n$ and $y \in \mathbb{R}^m$,
$$x^\top A^\top A x = (Ax)^\top (Ax)=\|Ax\|_2^2 \ge 0,$$
$$y^\top AA^\top y = (A^\top y)^\top (A^\top y)=\|A^\top y\|_2^2 \ge 0.$$
Hence $A^\top A$ and $AA^\top$ are positive semidefinite.

For $\lambda \ne 0$,
$$\det(\lambda I_n-A^\top A)=\lambda^{\,n}\det\!\left(I_n-\lambda^{-1}A^\top A\right),$$
and by Sylvester's determinant identity,
$$\det\!\left(I_n-\lambda^{-1}A^\top A\right)=\det\!\left(I_m-\lambda^{-1}AA^\top\right).$$
Therefore
$$\det(\lambda I_n-A^\top A)=\lambda^{\,n-m}\det(\lambda I_m-AA^\top).$$
Thus $A^\top A$ and $AA^\top$ have the same nonzero roots of their characteristic polynomials, with the same multiplicities.

>[!proposition] Singular values and eigenvalues
设 $\lambda_1,\dots,\lambda_r$ 是 $A^\top A$ 的全部正特征值，其中 $r=\operatorname{rank}(A)$，则 $A$ 的奇异值定义为
$$\sigma_i = \sqrt{\lambda_i},\quad i=1,\dots,r.$$
等价地，$\sigma_i^2$ 也是 $AA^\top$ 的全部正特征值。因此，奇异值刻画的是矩阵在线性变换下对不同正交方向的拉伸强度。

>[!proposition] Singular value decomposition
>对任意矩阵 $A_{m\times n} \in \mathbb{R}^{m\times n}$，存在正交矩阵 $U \in \mathbb{R}^{m\times m}$、$V \in \mathbb{R}^{n\times n}$ 与对角型矩阵 $\Sigma \in \mathbb{R}^{m\times n}$，使得
>$$A = U\Sigma V^\top.$$
>其中：
>- $U$ 的列向量是 $AA^\top$ 的一组标准正交特征向量。
>- $V$ 的列向量是 $A^\top A$ 的一组标准正交特征向量。
>- $\Sigma$ 的对角元为 $A$ 的奇异值 $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$。

>[!remark] Geometric interpretation
>分解
>$$A = U\Sigma V^\top$$
>可以理解为三个连续步骤的复合：
>- $V^\top$ 先将输入坐标系旋转到右奇异向量基。
>- $\Sigma$ 再沿各个标准坐标轴做独立拉伸或压缩，拉伸倍数正是奇异值。
>- $U$ 最后把结果旋转到左奇异向量基所对应的方向。
>因此，SVD 将任意线性变换分解为“旋转/反射 + 轴向缩放 + 旋转/反射”的标准形式。

>[!proposition] Low-rank approximation viewpoint
若
$$A = U\Sigma V^\top = \sum_{i=1}^r \sigma_i u_i v_i^\top,$$
则奇异值分解还给出了矩阵的最佳低秩逼近结构。特别地，截断 SVD
$$A_k = \sum_{i=1}^k \sigma_i u_i v_i^\top$$
是在 Frobenius 范数和谱范数意义下对 $A$ 的最佳秩-$k$ 逼近。
