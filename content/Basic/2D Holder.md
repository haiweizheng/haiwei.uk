---
draft: true
---
“2D Hölder 不等式”通常就是把普通 Hölder 不等式用在二维指标或二维区域上.
对矩阵/二维数组版本: 若 $p,q>1$, 且 $\frac1p+\frac1q=1$, 则
$$\sum_i\sum_j |a_{ij}b_{ij}| \le \left(\sum_i\sum_j |a_{ij}|^p\right)^{1/p}\left(\sum_i\sum_j |b_{ij}|^q\right)^{1/q}.$$
对二维积分版本: 若 $D\subset \mathbb R^2$, 则
$$\int_D |f(x,y)g(x,y)|\,\mathrm{d}x\mathrm{d}y \le \left(\int_D |f(x,y)|^p\,\mathrm{d}x\mathrm{d}y\right)^{1/p}\left(\int_D |g(x,y)|^q\,\mathrm{d}x\mathrm{d}y\right)^{1/q}.$$
其中 $q=\frac p{p-1}$, 称为 $p$ 的共轭指数. 特别地, 当 $p=q=2$ 时, 就是二维形式的 Cauchy-Schwarz 不等式.
***Proof.*** 以求和版本为例. 令
$$A=\left(\sum_{i,j}|a_{ij}|^p\right)^{1/p},\qquad B=\left(\sum_{i,j}|b_{ij}|^q\right)^{1/q}.$$
若 $A=0$ 或 $B=0$, 结论显然成立. 否则定义
$$x_{ij}=\frac{|a_{ij}|}{A},\qquad y_{ij}=\frac{|b_{ij}|}{B}.$$
于是 $\sum_{i,j}x_{ij}^p=1$, $\sum_{i,j}y_{ij}^q=1$.
用 Young 不等式: 对任意 $u,v\ge 0$, 有 $uv\le \frac{u^p}{p}+\frac{v^q}{q}$. 因此
$$\begin{aligned}\sum_{i,j}x_{ij}y_{ij} &\le \sum_{i,j}\left(\frac{x_{ij}^p}{p}+\frac{y_{ij}^q}{q}\right) \\ &= \frac1p\sum_{i,j}x_{ij}^p+\frac1q\sum_{i,j}y_{ij}^q \\ &= \frac1p+\frac1q=1.\end{aligned}$$
于是
$$\sum_{i,j}|a_{ij}b_{ij}|=AB\sum_{i,j}x_{ij}y_{ij}\le AB,$$
即
$$\sum_{i,j}|a_{ij}b_{ij}| \le \left(\sum_{i,j}|a_{ij}|^p\right)^{1/p}\left(\sum_{i,j}|b_{ij}|^q\right)^{1/q}.$$
二维积分版本完全同理: 把 $\sum_{i,j}$ 换成 $\int_D$, 把 $a_{ij},b_{ij}$ 换成 $f(x,y),g(x,y)$ 即可. 关键点是 Hölder 不等式并不依赖于“维度”, 二维只是把指标从 $i$ 变成 $(i,j)$, 或者把积分区域放在 $\mathbb R^2$ 中.
