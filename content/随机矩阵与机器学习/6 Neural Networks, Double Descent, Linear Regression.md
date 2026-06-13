---
source: "[[High-Dimensional Analysis- Random Matrices and Machine Learning.pdf]]"
tags:
  - 随机矩阵与机器学习_random-matrices-and-machine-learning
  - 神经网络_neural-network
  - 双下降_double-descent
  - 线性回归_linear-regression
---

### 6.1 Neural Network

>[!definition] Definition. Neural Network
>A function
>$$\begin{aligned} f(x) &:\ \mathbb{R}^p\rightarrow \mathbb{R}\\
>&= w \cdot\sigma(W_L \dots \sigma(W_3\,\sigma(W_2 \,\sigma(W_1 x)))) \end{aligned}$$



![[IMG_0410 2.jpg]]
![[Gemini_Generated_Image_c8wcuzc8wcuzc8wc.png]]
![[Gemini_Generated_Image_i1k2u8i1k2u8i1k2.png]]


Now consider the simplest neural network without hidden layer and without a non-linearity (i.e. *linear regression*):
$$y = f(x) = wx\quad Y = wX,$$where $(X, Y)$ are given and we look for the "best" $w$ to model this. 

Assume we have a linear relation, but have noise in the measurements, thus we see
$$\hat{Y}_{1\times n} = w_{1\times p}X_{p\times n} + N_{1\times n},$$
where $N$ is some Gaussian noise. We want to find the best $\hat{w}$ such that we have
$$\hat{Y}_{1\times n} = \hat{w}_{1\times p}X_{p\times n},$$
where we have a system of $n$ linear equations for $p$ variables. 

## 6.3. Linear regression: over-determined case ($n > p$)

In the case $n > p$, $\hat{Y} = \hat{w}X$ typically will have no solution and we approximate the non-linear relation (the non-linearity here is created by the noise.) between $\hat{Y}$ and $X$ by the method of least squares: instead of $\hat{Y} = \hat{w}X$ we try to solve *normal equation*
$$\hat{Y}X^T = \hat{w}XX^T;$$
characterizing a $\hat{w}$ such that $\|\hat{Y}-\hat{w}X\|$ is minimal. Typically, i.e., if $\text{rank}(X) = p < n$, the matrix $XX^T \in \mathbb{R}^{p \times p}$ is invertible, thus
$$\hat{w} = \hat{Y}X^T(XX^T)^{-1}.$$
Thus we have made the error
$$\begin{aligned}  
\|w - \hat{w}\|^2 &= \|w -\underbrace{\hat{Y}}_{=wX+N} X^T(XX^T)^{-1}\|^2\\
&= \|w - \underbrace{wXX^T(XX^T)^{-1}}_{=1} - NX^T(XX^T)^{-1}\|^2\\
&= \|NX^T(XX^T)^{-1}\|^2\\
&= \underbrace{N}_{1\times p}X^T(XX^T)^{-1}(XX^T)^{-1}X\underbrace{N^T}_{p\times 1},\\
\end{aligned}$$
since $\|a\|^2 = aa^*$ for all $a \in \mathbb{R}^{1 \times p}$. Assume now that $N \sim \mathcal{N}(0, \sigma^2 I_n)$ is a Gaussian noise and average the error over $N$:
$$\begin{aligned}  
E_N [\|w - \hat{w}\|^2] &= E_N [NX^T(XX^T)^{-2}XN^T]\\
&\stackrel{4.6}{=} \sigma^2 \cdot \text{Tr}\left(X^T(XX^T)^{-2}X\right)\\
&= \sigma^2 \cdot \text{Tr}\left(XX^T(XX^T)^{-2}\right)\\
&= \sigma^2 \cdot \text{Tr}\left((XX^T)^{-1}\right)\\
&= \sigma^2 \frac{p}{n} [\frac 1p\text{Tr}  \left( \frac{1}{n} XX^T \right)^{-1} ].
\end{aligned}$$

Assume now ==that $X$ is a standard Gaussian random matrix==, then $\hat{\Sigma} = \frac{1}{n} XX^T$ is a Wishart matrix and the above error converges for $n, p = \gamma n \to \infty$ to $\sigma^2 \gamma S(0)$, where
$$S(z) = \int \frac{1}{t-z} \psi_{\text{MP}} \, dt$$
is the Stieltjes transform of the Marchenko-Pastur distribution. We know from the proof of Theorem 4.2 that $S(z)$ satisfies the equation
$$1 + zS(z) = \frac{S(z)}{1 + \gamma S(z)},$$
i.e., for $z = 0$ (note that $S$ has a continuous extension to $\mathbb{R}$ for $\gamma < 1$)
$$1 = \frac{S(0)}{1 + \gamma S(0)}, \quad \text{so} \quad S(0) = \frac{1}{1 - \gamma}$$
and thus
$$E [\|w - \hat{w}\|^2] = \sigma^2 \frac{\gamma}{1 - \gamma}.$$

## 6.4. Linear regression: under-determined case ($n < p$)

>[!definition] Definition. Pseudo-inverse
>For all matrix $A\in M^{m\times n}$, $\exists ! A^+\in M^{m\times n}$ that satisfies Moore-Penrose conditions: 
>1. $AA^+A = A$ and $A^+AA^+ = A^+$
>2. $(AA^+)^H = AA^+$ and $(A^+A)^H = A^+A$

设 $A = U\Sigma V^T$ 为 SVD，其中 $\Sigma = \text{diag}(\sigma_1, \dots, \sigma_r, 0, \dots, 0)$，$r = \text{rank}(A)$. 则伪逆为
$$A^+ = V\Sigma^+ U^T,$$
其中 $\Sigma^+$ 将每个非零奇异值取倒数，零奇异值保持为零，再转置：
$$\Sigma^+ = \text{diag}(1/\sigma_1, \dots, 1/\sigma_r, 0, \dots, 0)^T.$$
直觉上，$A$ 在非零奇异值方向上的作用是拉伸 $\sigma_i$ 倍，伪逆就是在这些方向上缩回 $1/\sigma_i$ 倍，而在零奇异值方向（$A$ 的零空间）上不做任何事. 

> [!remark]- 伪逆实战应用
> **广义逆 $A^+$**：是**尽力而为**。
>- 如果方程**解太多**（矮胖矩阵）：它帮你选一个**最低调（范数最小）**的解。
>- 如果方程**没解**（瘦高矩阵）：它帮你找一个**最接近（残差最小）**的解。
>- **口诀：** “多解取其短（小范数），无解取其近（最小二乘）”。
>  1. 解决“超定方程组” (Overdetermined Systems)
> **场景：** 实验数据点多于未知数 ($m > n$)，由于噪声干扰，没有完美解能经过所有点。
> * **核心逻辑：** 最小化残差的平方和 $\|Ax - b\|^2$。
> * **公式：** $A^+ = (A^T A)^{-1} A^T$
> * **应用：** **最小二乘拟合 (OLS)**、传感器数据校准、线性回归。
> 2. 解决“欠定方程组” (Underdetermined Systems)
> **场景：** 方程太少，未知数太多 ($m < n$)，存在无数个可行解（如 Lemma 6.1）。
> * **核心逻辑：** 在所有可行解中挑选出**范数（能量）最小**的那个解 $\|x\|$。
> * **公式：** $A^+ = A^T (AA^T)^{-1}$
> * **应用：** **信号重建**、压缩感知 (Compressed Sensing)、推荐系统。
> 3. 矩阵去噪与数据压缩 (Noise Reduction)
> **场景：** 原始矩阵包含大量噪声，或者由于相关性极高导致接近奇异 (Singular)。
> * **核心逻辑：** 结合 **SVD**，通过伪逆丢弃极小的奇异值，提取数据的主骨架。
> * **应用：** **主成分分析 (PCA)**、图像压缩、滤除背景噪声。
> 4. 机器人冗余运动控制 (Redundancy Control)
> **场景：** 机械臂关节数多于自由度，存在无数种动作姿态可以达到同一个目标点。
> * **核心逻辑：** 优化关节运动路径，使关节旋转的总能量/位移最小。
> * **应用：** **逆运动学求解 (Inverse Kinematics)**、避障算法。
> 5. 高维统计与机器学习 (High-dim Statistics)
> **场景：** 特征数 $p$ 远大于样本数 $n$ ($p \gg n$)，协方差矩阵不可逆。
> * **核心逻辑：** 在矩阵不满秩时，提供一个数值稳定的“代用逆”。
> * **应用：** **精密矩阵估计**、LDA 判别分析、岭回归 (Ridge Regression) 的极限形式。

In the case $n < p$, $\hat{Y} = \hat{w}X$ typically will have infinitely many solutions and we will choose the one with the smallest norm. The same formula that replace the inverse by the pseudo-inverse:
$$\hat{w} = \hat{Y}X^T(XX^T)^+ = \hat{Y}(X^T X)^{-1}X^T,$$
where the last equation holds only if $\text{rank}(X) = n < p$. 
>[!lemma] Lemma. 6.1
>Let $A \in \mathbb{R}^{n \times p}$ with $n < p$ and $\text{rank}(A) = n$. Then
>- $AA^T \in \mathbb{R}^{n \times n}$ is invertible. 
>- For $y_{n\times 1}$, $x_0 := A^T(AA^T)^{-1}y$ is a **solution** of $Ax = y$ and with **smallest norm**.

***Remark.*** How to find it?
- Notice it is for $\hat{Y}^T = X^T_{n\times p}\hat{w}^T$
- This $\hat{w}$ then is a solution of $\hat{Y} = \hat{w}X$ and has smallest norm among all infinitely many solutions. Notice that
$$\hat{Y}^T_{n\times1}=X^T_{n\times p}\hat{\omega}^T_{n\times 1} $$

***Proof.*** 
***Step 1***
To verify $\text{rank}(AA^T) = \text{rank}(A^T) = n$
$$
\begin{aligned}
\forall x \in & \text{Null}(AA^T),  \\
&AA^Tx = 0 \rightarrow x^TAA^Tx = 0 \rightarrow \|A^Tx\|^2 = 0 \rightarrow A^Tx = 0 \\
\forall x \in & \text{Null}(A^T),  \\
& A^Tx = 0 \rightarrow AA^Tx = 0
\end{aligned}
$$
Thus $\text{rank}(A^T) = n - \dim \text{Null}(A^T) = n - \dim \text{Null}(AA^T) = \text{rank}(AA^T)$
***Step 2***
① Existence: $Ax_0 = AA^T(AA^T)^{-1}y = y$
② Minimum:
$$ \begin{aligned} \|x\|^2 = \langle x, x \rangle &= \langle (x - x_0) + x_0, (x - x_0) + x_0 \rangle \\ 
&= \langle x - x_0, x - x_0 \rangle + \langle x_0, x_0 \rangle + \underbrace{2 \langle x - x_0, x_0 \rangle} \\ 
&\phantom{= \langle x - x_0, x - x_0 \rangle + \langle x_0, x_0 \rangle +}= \langle x - x_0, x_0 \rangle \\ 
&\phantom{= \langle x - x_0, x - x_0 \rangle + \langle x_0, x_0 \rangle +}= \langle x - x_0, A^T(AA^T)^{-1}y \rangle \\ 
&\phantom{= \langle x - x_0, x - x_0 \rangle + \langle x_0, x_0 \rangle +}= \langle A(x - x_0), (AA^T)^{-1}y \rangle = 0 \\ 
&= \|x - x_0\|^2 + \|x_0\|^2 \geq \|x_0\|^2 \end{aligned} $$

So in this case, the BEST solution is 
$$\hat{w} = \hat{Y}(X^T X)^{-1}X^T.$$
where $\hat{w}$ exactly matches the given noisy data $\hat{Y} = wX + N$. 

Let us again consider the error
$$
\begin{aligned}
\|w - \hat{w}\|^2 &= \|w - (wX + N)(X^TX)^{-1}X^T\|^2 \\
&= \|\underbrace{w - wX(X^TX)^{-1}X^T}_{A} - \underbrace{N(X^TX)^{-1}X^T}_{B}\|^2 \\
&= \|A\|^2 + \|B\|^2 + \underbrace{2 \langle A, B \rangle} \\
&\phantom{\|A\|^2 + \|B\|^2 +=} = [w - wX(X^TX)^{-1}X^T][N(X^TX)^{-1}X^T]^T \\
&\phantom{\|A\|^2 + \|B\|^2 +=}= w[I - X(X^TX)^{-1}X^T][X(X^TX)^{-1}N^T] \\
&\phantom{\|A\|^2 + \|B\|^2 +=}= w[\underbrace{X - X(X^TX)^{-1}X^TX}_{0}](X^TX)^{-1}N^T =0\\
&= \underbrace{\|A\|^2}_{\text{bias term}} + \underbrace{\|B\|^2}_{\text{variance term}}
\end{aligned}
$$

① For $B$:
$$
\begin{aligned}
\mathbb{E}_N \|B\|^2 &= \mathbb{E}_N \|N(X^TX)^{-1}X^T\|^2 \\
&= \mathbb{E}_N (N_{1 \times n}(X^TX)^{-1}X^T_{n \times p})(N(X^TX)^{-1}X^T)^T \\
&= \mathbb{E}_N (N(X^TX)^{-1}X^TX(X^TX)^{-1}N^T \\
&= \mathbb{E}_N (N \cdot (X^TX)^{-1}N^T) \\
&= \sigma^2 \text{tr}(X^TX)^{-1}_{p \times p} \\
&= \sigma^2 \cdot \frac{n}{p} \cdot \underbrace{\left[ \frac{1}{n} \text{tr} \left( \frac{1}{p} X^TX \right)^{-1} \right]} \\
& \qquad \qquad X^T_{n \times p} = B_{n \times p} (n < p) \\
& \qquad \qquad S(z) = \frac{1}{n} \text{tr} \left( \frac{1}{p} B_{n \times p} B^T-zI_n \right)^{-1}\\
&\rightarrow \sigma^2 \cdot \frac{n}{p} \cdot S(0) \\
&\rightarrow \sigma^2 \frac{1}{\gamma}\cdot\frac{1}{1 - \frac{1}{\gamma}} = \sigma^2 \frac{1}{\gamma - 1}
\end{aligned}
$$
Notice here the roles of $n$ and $p$ are exchanged! Thus $\frac n p =\frac 1 \gamma$
$$1 + zS(z) = \frac{S(z)}{1 + \frac 1\gamma S(z)},$$
i.e., for $z = 0$ (note that $S$ has a continuous extension to $\mathbb{R}$ for $\frac 1 \gamma < 1$)
$$1 = \frac{S(0)}{1 + \frac 1\gamma S(0)}, \quad \text{so} \quad S(0) = \frac{1}{1 - \frac 1\gamma}$$

② For $A$:
$$
\begin{aligned}
\|w - wX(X^TX)^{-1}X^T\|^2 &= \underbrace{(w - wX(X^TX)^{-1}X^T)}_{w_{1 \times n}} (w - wX(X^TX)^{-1}X^T)^T \\
&= (w - wX(X^TX)^{-1}X^T) (w^T - X(X^TX)^{-1}X^T w^T) \\
&= ww^T - 2wX(X^TX)^{-1}X^T w^T + wX\underbrace{(X^TX)^{-1}X^TX}_{(X^TX)^{-1}X^TX=I} (X^TX)^{-1}X^T w^T \\
&= ww^T - wX(X^TX)^{-1}X^T w^T \\
&= \|w\|^2 \underbrace{(1 - \frac{w}{\|w\|} X(X^TX)^{-1}X^T \frac{w^T}{\|w\|})}_{\text{rotational symmetric}}
\end{aligned}
$$
Since Rotational Symmetric of $X(X^TX)^{-1}X^T$,
$$
\begin{aligned}
tr(X(X^TX)^{-1}X^T) &= \sum e_i^T (X(X^TX)^{-1}X^T) e_i \\
&\overset{RS}{=} n \ e_i^T (X(X^TX)^{-1}X^T) e_i \\
&\overset{RS}{=} n \ \underbrace{\frac{w}{\|w\|} (X(X^TX)^{-1}X^T) \frac{w^T}{\|w\|}}_{w_{1 \times n}}
\end{aligned}
$$
Back to original formula:
$$
\begin{aligned}
\|w - wX(X^TX)^{-1}X^T\|^2 &= \|w\|^2 (1 - \frac{w}{\|w\|} X(X^TX)^{-1}X^T \frac{w^T}{\|w\|})\\
&= \|w\|^2 \cdot \left( 1 - \frac{1}{p} \operatorname{tr}(X(X^TX)^{-1}X^T) \right) \\ &= \|w\|^2 \cdot \left( 1 - \frac{1}{p} \operatorname{tr}(X^TX)(X^TX)^{-1} \right) \\ &= \|w\|^2 \cdot \left( 1 - \frac{n}{p} \right) \xrightarrow[p = \gamma n]{n \to \infty} \|w\|^2 \cdot \left( 1 - \frac{1}{\gamma} \right) 
\end{aligned}
$$
## 6.5. Double descent for linear regression

Combining all the results, we get
$$
E \left[ \|w - \hat{w}\|^2 \right] = \begin{cases} \sigma^2 \frac{\gamma}{1 - \gamma}, & \gamma < 1, \\ \sigma^2 \frac{1}{\gamma - 1} + \|w\|^2 \left( 1 - \frac{1}{\gamma} \right), & \gamma > 1. \end{cases}
$$
![[6 Neural Networks, Double Descent, Linear Regression-1773481313343.webp|625]]


## 6.6. Adding layers and non-linearities

Consider now a more interesting neural network by adding one layer and non-linearities. 

![[6 Neural Networks, Double Descent, Linear Regression-1773481740489.webp|588]]
$$
\mathbb{R} \xleftarrow{w} \mathbb{R}^m \xleftarrow{W} \mathbb{R}^p : f
$$

So we have now $f(x) = w \sigma Wx$ and $f(X) = w \sigma WX$. If we don't learn $W$, but only $w$, then this is still linear regression, but not on $X$, but on the "random features" $F := \sigma WX$ and the performance depends on the eigenvalue distribution of $FF^T$.

So we should now address the question whether we can understand the (combined) effect of

* multiplying two random matrices and
* applying non-linear functions entry-wise to random matrices

on the eigenvalue distribution? The first problem is part of classical random matrix theory, the second problem, in combination with the first one, is new and gives rise to "non-linear random matrix theory". Let us be a bit more precise on this.

(i) Consider $F = WX$, where both $W$ and $X$ are Gaussian random matrices (thus $WW^T$ and $XX^T$ are Wishart matrices), independent from each other. Then $FF^T = WXX^T W^T$ has the same distribution as a Wishart matrix $YY^T$, where $Y$ is the data matrix of vectors $y = Wx$ where $x \sim N(0, I_p)$, so $y \sim N(0, WW^T)$. Put $\Sigma := WW^T \in \mathbb{R}^{m \times m}$. Then one can determine a fixed-point equation for the Stieltjes transform of $FF^T$; similar (but more general) as in Exercise 4 of Assignment 4.

(ii) Consider $F = \sigma X$. Then the entries of $F$ are still i.i.d., but their common distribution is not Gaussian any more, but the push-forward of Gauss under $\sigma$. However, for the validity of the Marchenko-Pastur law **Theorem 4.2** one does not need the Gaussian distribution, the essential input is independence 且有二阶矩. Thus the distribution of $F$ is still Marchenko-Pastur.


Both (i) and (ii) are thus within the realm of classical random matrix theory, but if we consider now the combination (iii) $F = \sigma WX$, then we have to move into new "non-linear" random matrix territory! We will address this in the next section.

