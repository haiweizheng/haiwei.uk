---
title: Attention
publish: true
tags:
  - 注意力机制_attention
  - 深度学习_deep-learning
  - transformer
---
# 1. Query & Key & Value

在注意力机制中，原始数据转换成三个对象：Query, Key, Value. 而 Attention 就是在根据 Query 和 Key 的匹配程度获得权重，给 Value 分配权重，然后做加权求和。

***Example. 1*** 一个简单例子
![[Basic/attachments/attention-qkv-01.svg]]

| $Q\backslash K$ | $k_1=\{Li,Hua,60\}$ | $k_2=\{Wu,Ming,20\}$ | $k_3=\{Li,Meng,40\}$ |
| --------------- | ------------------- | -------------------- | -------------------- |
| 姓为 "Wu"         | 0                   | 1                    | 0                    |
| 年龄较大            | 0.7                 | 0.05                 | 0.25                 |
| 性别为 "男"         | unknown / 0.2       | unknown / 0.4        | unknown / 0.2        |
<div style="text-align: center;"><strong><em>Table.</em></strong> 固定 <em>Q</em> 时, 不同 Key 下的注意力权重.</div>

- 当问到姓为 "Wu" 的时候, 注意力权重集中于第二个 Key $k_2$ , 这时候我们主要关心 Wu 姓用户的电费 (Value).
- 当问到年龄较大的时候, 注意力权重在年纪较大的 Key 大, 在年纪较小的 Key 小, 这时我们主要关心年纪大的用户的电费 (Value).
- 当问到性别为男时, Key 并不包含性别 tag, 故注意力权重溃散到 $k_1,k_2,k_3$.

当固定 $Q$ 为 "年龄较大" 时, 注意力为
$$\mathrm{Attention}=0.7\times v_1+0.05\times v_2+0.25\times v_3.$$

> [!definition] Definition. Attention pool
> - 给定一个 Query $q$ 和一组 Key-Value 对 $\{(k_i,v_i)\}_{i=1}^n$, 注意力为:
> $$\mathrm{Attention}(q,\{(k_i,v_i)\}_{i=1}^n)=\sum_{i=1}^n \alpha(q,k_i)v_i,$$
> $\alpha(q,k_i)$ 是 query $q$ 分配给 key $k_i$ 的注意力权重，需满足:
> $$\alpha(q,k_i)\ge 0,\quad \sum_{i=1}^n \alpha(q,k_i)=1.$$

实际上, 注意力权重即 ==Q 和 K 之间的匹配程度/相关程度== 有不同的表示方法, 接下来给出两个例子:
- 来自 *Theory of Probability & Its Applications* ([[#^ref-nadaraya-1964|Nadaraya (1964)]]) 的 Nadaraya-Watson regression
$$\alpha(q,k_i)=\alpha(x,x_i)=\sum_{i=1}^n \frac{K(x-x_i)}{\sum_{j=1}^n K(x-x_j)}$$
- 来自 *Attention is all you need* ([[#^ref-vaswani-2017|Vaswani et al. (2017)]]) 的 scaled dot-product attention
$$
\alpha(Q,K)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)
$$

# 2. N-W 核回归 Nadaraya-Watson regression
***Step 1*** 对由
$$y_i=2\sin(x_i)+x_i^{0.8}+\epsilon_i,\quad \epsilon_i \sim \mathcal{N}(0,0.5^2).$$
生成的数据集 $\{(x_1,y_1),\ldots,(x_{50},y_{50})\}$ 做回归.

***Step 2*** 最简单的预测方法是取平均：
$$f(x)=\frac{1}{n}\sum_{i=1}^n y_i=\sum_{i=1}^n \frac{1}{n} y_i.$$
此时 Query 为 $x$, Key 为 $x_i$, Value 为 $y_i$. 注意力权重为 $\alpha(q,k_i)=\alpha(x,x_i)=\frac 1 n$. 如下图, 拟合效果并不好, 因为我们没有有效的使用 Query 和 Key.

```tikz
\begin{document}
\begin{tikzpicture}[every node/.style={font=\large}]
% Temporary data for the TikZ plot below. Generated with seed=42 from y_i=2sin(x_i)+x_i^0.8+epsilon_i, epsilon_i~N(0,0.5^2).
% x_train = [0.219, 0.319, 0.471, 0.641, 0.650, 0.699, 0.771, 0.947, 0.973, 1.135, 1.136, 1.442, 1.562, 1.629, 1.773, 1.852, 1.854, 1.937, 2.186, 2.194, 2.217, 2.252, 2.334, 2.348, 2.379, 2.773, 3.158, 3.219, 3.349, 3.412, 3.415, 3.487, 3.501, 3.724, 3.790, 3.806, 3.870, 3.892, 3.930, 4.024, 4.114, 4.138, 4.161, 4.163, 4.293, 4.466, 4.634, 4.838, 4.853, 4.878]
% y_train = [1.071, 1.062, 1.599, 2.211, 1.189, 1.877, 1.972, 2.262, 2.494, 3.666, 2.489, 3.807, 2.587, 3.307, 3.622, 3.852, 3.915, 3.961, 3.329, 3.268, 3.916, 3.373, 2.778, 2.839, 2.923, 3.231, 2.547, 2.738, 2.004, 2.214, 2.444, 1.885, 2.249, 1.432, 1.514, 1.489, 1.023, 1.845, 1.335, 1.508, 1.688, 1.659, 1.758, 1.375, 1.169, 1.331, 0.572, 0.821, 0.897, 1.082]
% y_mean = 2.224
\node at (4.0,5.35) {mean pooling regression};
\draw[gray!25, line width=0.35pt] (0.0,0) -- (0.0,4.8);
\draw[gray!25, line width=0.35pt] (1.6,0) -- (1.6,4.8);
\draw[gray!25, line width=0.35pt] (3.2,0) -- (3.2,4.8);
\draw[gray!25, line width=0.35pt] (4.8,0) -- (4.8,4.8);
\draw[gray!25, line width=0.35pt] (6.4,0) -- (6.4,4.8);
\draw[gray!25, line width=0.35pt] (8.0,0) -- (8.0,4.8);
\draw[gray!25, line width=0.35pt] (0,0.0) -- (8.0,0.0);
\draw[gray!25, line width=0.35pt] (0,0.8) -- (8.0,0.8);
\draw[gray!25, line width=0.35pt] (0,1.6) -- (8.0,1.6);
\draw[gray!25, line width=0.35pt] (0,2.4) -- (8.0,2.4);
\draw[gray!25, line width=0.35pt] (0,3.2) -- (8.0,3.2);
\draw[gray!25, line width=0.35pt] (0,4.0) -- (8.0,4.0);
\draw[gray!25, line width=0.35pt] (0,4.8) -- (8.0,4.8);
\draw[->, thick] (0,0.8) -- (8.45,0.8) node[right] {$x$};
\draw[->, thick] (0,0) -- (0,5.05) node[above] {$y$};
\node[below] at (1.6,0.8) {$1$};
\node[below] at (3.2,0.8) {$2$};
\node[below] at (4.8,0.8) {$3$};
\node[below] at (6.4,0.8) {$4$};
\node[below] at (8.0,0.8) {$5$};
\node[left] at (0,0.0) {$-1$};
\node[left] at (0,0.8) {$0$};
\node[left] at (0,1.6) {$1$};
\node[left] at (0,2.4) {$2$};
\node[left] at (0,3.2) {$3$};
\node[left] at (0,4.0) {$4$};
\node[left] at (0,4.8) {$5$};
\draw[blue!65!black, thick] plot[smooth] coordinates {(0.000,0.800) (0.200,1.151) (0.400,1.460) (0.600,1.751) (0.800,2.027) (1.000,2.285) (1.200,2.526) (1.400,2.747) (1.600,2.946) (1.800,3.123) (2.000,3.275) (2.200,3.402) (2.400,3.503) (2.600,3.577) (2.800,3.626) (3.000,3.649) (3.200,3.648) (3.400,3.623) (3.600,3.575) (3.800,3.508) (4.000,3.423) (4.200,3.322) (4.400,3.208) (4.600,3.084) (4.800,2.952) (5.000,2.817) (5.200,2.681) (5.400,2.547) (5.600,2.418) (5.800,2.298) (6.000,2.189) (6.200,2.093) (6.400,2.014) (6.600,1.954) (6.800,1.914) (7.000,1.896) (7.200,1.901) (7.400,1.930) (7.600,1.984) (7.800,2.062) (8.000,2.165)};
\draw[red!75!black, very thick, dashed] (0,2.579) -- (8.0,2.579);
\fill[orange!70!white, draw=orange!80!black] (0.350,1.657) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (0.511,1.650) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (0.753,2.079) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (1.025,2.569) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (1.039,1.751) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (1.118,2.302) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (1.234,2.377) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (1.516,2.610) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (1.557,2.795) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (1.815,3.733) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (1.818,2.791) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (2.307,3.846) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (2.499,2.870) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (2.607,3.445) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (2.836,3.697) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (2.964,3.881) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (2.966,3.932) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (3.100,3.969) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (3.497,3.463) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (3.511,3.414) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (3.547,3.933) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (3.603,3.498) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (3.734,3.022) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (3.756,3.071) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (3.806,3.138) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (4.437,3.384) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (5.053,2.838) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (5.151,2.990) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (5.359,2.403) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (5.460,2.571) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (5.464,2.755) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (5.579,2.308) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (5.602,2.600) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (5.958,1.946) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.065,2.011) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.089,1.992) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.192,1.619) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.227,2.276) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.289,1.868) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.438,2.006) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.582,2.151) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.621,2.127) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.658,2.206) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.661,1.900) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.869,1.736) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (7.145,1.865) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (7.414,1.258) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (7.740,1.457) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (7.766,1.518) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (7.805,1.665) circle (1.6pt);
\draw[blue!65!black, thick] (6.2,4.60) -- (6.75,4.60);
\node[anchor=west] at (6.85,4.60) {Truth};
\draw[red!75!black, very thick, dashed] (6.2,4.00) -- (6.75,4.00);
\node[anchor=west] at (6.85,4.00) {Pred};
\end{tikzpicture}
\end{document}
```

***Step 3*** [[#^ref-nadaraya-1964|Nadaraya (1964)]] 和 [[#^ref-watson-1964|Watson (1964)]] 提出了 Nadaraya-Watson kernel regression 方法：
$$\begin{aligned} f(x)&=\sum_{i=1}^n \underbrace{{\color{blue}\frac{K(x-x_i)}{\sum_{j=1}^n K(x-x_j)}}}y_i,\quad K\text{ is kernel} \\ &=\sum_{i=1}^n \underbrace{{\color{blue}\alpha(x,x_i)}}_{{\color{blue}\text{注意力权重}}}y_i. \end{aligned}$$
When $K$ is Gaussian kernel
$$K(u)=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{u^2}{2}\right),$$
we have
$$\begin{aligned} f(x)&=\sum_{i=1}^n \alpha(x,x_i)y_i \\ &=\sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}(x-x_i)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}(x-x_j)^2\right)}y_i \\ &=\sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}(x-x_i)^2\right)y_i. \end{aligned}$$
可以看到，新模型的预测曲线更加平滑，并且比平均汇聚更接近真实函数.

```tikz
\begin{document}
\begin{tikzpicture}[every node/.style={font=\large}]
\node at (4.0,5.35) {Nadaraya-Watson kernel regression};
\draw[gray!25, line width=0.35pt] (0.0,0) -- (0.0,4.8);
\draw[gray!25, line width=0.35pt] (1.6,0) -- (1.6,4.8);
\draw[gray!25, line width=0.35pt] (3.2,0) -- (3.2,4.8);
\draw[gray!25, line width=0.35pt] (4.8,0) -- (4.8,4.8);
\draw[gray!25, line width=0.35pt] (6.4,0) -- (6.4,4.8);
\draw[gray!25, line width=0.35pt] (8.0,0) -- (8.0,4.8);
\draw[gray!25, line width=0.35pt] (0,0.0) -- (8.0,0.0);
\draw[gray!25, line width=0.35pt] (0,0.8) -- (8.0,0.8);
\draw[gray!25, line width=0.35pt] (0,1.6) -- (8.0,1.6);
\draw[gray!25, line width=0.35pt] (0,2.4) -- (8.0,2.4);
\draw[gray!25, line width=0.35pt] (0,3.2) -- (8.0,3.2);
\draw[gray!25, line width=0.35pt] (0,4.0) -- (8.0,4.0);
\draw[gray!25, line width=0.35pt] (0,4.8) -- (8.0,4.8);
\draw[->, thick] (0,0.8) -- (8.45,0.8) node[right] {$x$};
\draw[->, thick] (0,0) -- (0,5.05) node[above] {$y$};
\node[below] at (1.6,0.8) {$1$};
\node[below] at (3.2,0.8) {$2$};
\node[below] at (4.8,0.8) {$3$};
\node[below] at (6.4,0.8) {$4$};
\node[below] at (8.0,0.8) {$5$};
\node[left] at (0,0.0) {$-1$};
\node[left] at (0,0.8) {$0$};
\node[left] at (0,1.6) {$1$};
\node[left] at (0,2.4) {$2$};
\node[left] at (0,3.2) {$3$};
\node[left] at (0,4.0) {$4$};
\node[left] at (0,4.8) {$5$};
\draw[blue!65!black, thick] plot[smooth] coordinates {(0.000,0.800) (0.200,1.151) (0.400,1.460) (0.600,1.751) (0.800,2.027) (1.000,2.285) (1.200,2.526) (1.400,2.747) (1.600,2.946) (1.800,3.123) (2.000,3.275) (2.200,3.402) (2.400,3.503) (2.600,3.577) (2.800,3.626) (3.000,3.649) (3.200,3.648) (3.400,3.623) (3.600,3.575) (3.800,3.508) (4.000,3.423) (4.200,3.322) (4.400,3.208) (4.600,3.084) (4.800,2.952) (5.000,2.817) (5.200,2.681) (5.400,2.547) (5.600,2.418) (5.800,2.298) (6.000,2.189) (6.200,2.093) (6.400,2.014) (6.600,1.954) (6.800,1.914) (7.000,1.896) (7.200,1.901) (7.400,1.930) (7.600,1.984) (7.800,2.062) (8.000,2.165)};
\draw[red!75!black, very thick, dashed] plot[smooth] coordinates {(0.000,2.554) (0.200,2.599) (0.400,2.645) (0.600,2.692) (0.800,2.741) (1.000,2.789) (1.200,2.837) (1.400,2.884) (1.600,2.928) (1.800,2.970) (2.000,3.008) (2.200,3.041) (2.400,3.068) (2.600,3.087) (2.800,3.098) (3.000,3.101) (3.200,3.093) (3.400,3.075) (3.600,3.047) (3.800,3.010) (4.000,2.963) (4.200,2.909) (4.400,2.849) (4.600,2.784) (4.800,2.716) (5.000,2.647) (5.200,2.579) (5.400,2.512) (5.600,2.448) (5.800,2.388) (6.000,2.331) (6.200,2.279) (6.400,2.230) (6.600,2.185) (6.800,2.143) (7.000,2.105) (7.200,2.069) (7.400,2.036) (7.600,2.006) (7.800,1.977) (8.000,1.950)};
\fill[orange!70!white, draw=orange!80!black] (0.350,1.657) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (0.510,1.650) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (0.754,2.079) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (1.026,2.569) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (1.040,1.751) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (1.118,2.302) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (1.234,2.378) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (1.515,2.610) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (1.557,2.795) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (1.816,3.733) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (1.818,2.791) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (2.307,3.846) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (2.499,2.870) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (2.606,3.446) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (2.837,3.698) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (2.963,3.882) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (2.966,3.932) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (3.099,3.969) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (3.498,3.463) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (3.510,3.414) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (3.547,3.933) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (3.603,3.498) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (3.734,3.022) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (3.757,3.071) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (3.806,3.138) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (4.437,3.385) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (5.053,2.838) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (5.150,2.990) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (5.358,2.403) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (5.459,2.571) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (5.464,2.755) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (5.579,2.308) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (5.602,2.599) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (5.958,1.946) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.064,2.011) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.090,1.991) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.192,1.618) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.227,2.276) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.288,1.868) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.438,2.006) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.582,2.150) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.621,2.127) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.658,2.206) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.661,1.900) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (6.869,1.735) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (7.146,1.865) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (7.414,1.258) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (7.741,1.457) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (7.765,1.518) circle (1.6pt);
\fill[orange!70!white, draw=orange!80!black] (7.805,1.666) circle (1.6pt);
\draw[blue!65!black, thick] (6.2,4.60) -- (6.75,4.60);
\node[anchor=west] at (6.85,4.60) {Truth};
\draw[red!75!black, very thick, dashed] (6.2,4.00) -- (6.75,4.00);
\node[anchor=west] at (6.85,4.00) {Pred};
\end{tikzpicture}
\end{document}
```

# 3. 缩放内积注意力 scaled dot-product attention
It is a part of Transformer from *Attention is all you need* ([[#^ref-vaswani-2017|Vaswani et al. (2017)]]):
$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$
我们简单解释一下这个流程. 一句话说, 注意力就是让每个 token 根据自己的 Query 去和所有 token 的 Key 做匹配, 得到一行权重, 再用这行权重对所有 Value 做加权求和.
- 输入文本先被 tokenizer 切成 token, 每个 token 变成一行 embedding vector.
- 对同一个输入矩阵 $X$ 做三次线性变换, 得到 $Q,K,V$.
- $QK^\top$ 的第 $(i,j)$ 个元素是 $q_i\cdot k_j^\top$, 表示第 $i$ 个 token 对第 $j$ 个 token 的关注分数.
- softmax 对每一行做归一化, 得到注意力权重 $\alpha_{ij}$, 因此 $\sum_j \alpha_{ij}=1$.
- 最后用这一行权重加权所有 value vector: $o_i=\sum_j \alpha_{ij}v_j$. 这就是第 $i$ 个 token 汇总上下文后的表示.
<div style="width: 100%; max-width: 900px; box-sizing: border-box; overflow-x: hidden; margin: 10px auto 8px auto; font-family: 'Songti SC', 'STSong', 'SimSun', serif; color: var(--text-normal); line-height: 1.45;">
  <!-- GPT-2 attention structure diagram. Math formulas are kept outside HTML so Obsidian can render LaTeX. -->
  <div style="text-align: center; font-size: 1.15em; font-weight: 700; margin-bottom: 8px;">GPT-2 128M, consider one attention head</div>
  <div style="display: flex; justify-content: center; margin-bottom: 6px; max-width: 100%; box-sizing: border-box;">
    <div style="max-width: 100%; box-sizing: border-box; border: 1px solid var(--background-modifier-border); border-radius: 10px; padding: 8px 16px; background: var(--background-primary); font-size: 1.05em; overflow-wrap: anywhere;">I'm going to visualize ...</div>
  </div>
  <div style="text-align: center; color: var(--text-muted); margin: 2px 0 4px 0;">↓ tokenizer (分词器)</div>
  <div style="display: flex; justify-content: center; align-items: center; gap: 6px; flex-wrap: wrap; max-width: 100%; box-sizing: border-box; margin-bottom: 6px;">
    <span style="border: 2px solid #1e88e5; border-radius: 6px; padding: 4px 10px; color: #1e88e5; background: rgba(30,136,229,0.08);">I'm</span>
    <span style="border: 2px solid #1e88e5; border-radius: 6px; padding: 4px 10px; color: #1e88e5; background: rgba(30,136,229,0.08);">going</span>
    <span style="border: 2px solid #1e88e5; border-radius: 6px; padding: 4px 10px; color: #1e88e5; background: rgba(30,136,229,0.08);">to</span>
    <span style="border: 2px solid #1e88e5; border-radius: 6px; padding: 4px 10px; color: #1e88e5; background: rgba(30,136,229,0.08);">visual</span>
    <span style="border: 2px solid #1e88e5; border-radius: 6px; padding: 4px 10px; color: #1e88e5; background: rgba(30,136,229,0.08);">-ize</span>
    <span style="padding: 4px 6px; color: var(--text-muted);">...</span>
    <span style="color: #1e88e5; font-weight: 600; margin-left: 4px;">1024 tokens</span>
  </div>
  <div style="text-align: center; color: var(--text-muted); margin: 2px 0 6px 0;">↓ embedding</div>
  <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; width: 100%; max-width: 650px; box-sizing: border-box; margin: 0 auto 8px auto;">
    <div style="flex: 1 1 220px; min-width: 0; max-width: 320px; box-sizing: border-box; border: 1px solid rgba(239,83,80,0.55); border-radius: 10px; padding: 8px 10px; background: rgba(239,83,80,0.08); text-align: center;">
      <div style="color: #ef5350; font-weight: 700;">token embedding</div>
      <div style="font-size: 0.92em; color: var(--text-muted);">what the word means</div>
    </div>
    <div style="flex: 1 1 220px; min-width: 0; max-width: 320px; box-sizing: border-box; border: 1px solid rgba(239,83,80,0.55); border-radius: 10px; padding: 8px 10px; background: rgba(239,83,80,0.08); text-align: center;">
      <div style="color: #ef5350; font-weight: 700;">position embedding</div>
      <div style="font-size: 0.92em; color: var(--text-muted);">where the word is located</div>
    </div>
  </div>
  <div style="text-align: center; color: var(--text-muted); margin: 2px 0 4px 0;">↓ add them together</div>
</div>

```tikz
\begin{document}
\begin{tikzpicture}[every node/.style={font=\large}]
\draw[thick, rounded corners=4pt, fill=green!8] (0,0) rectangle (2.2,3);
\draw[gray!40, line width=0.4pt] (0.25,0.45) -- (1.95,0.45);
\draw[gray!40, line width=0.4pt] (0.25,0.80) -- (1.95,0.80);
\draw[gray!40, line width=0.4pt] (0.25,1.15) -- (1.95,1.15);
\draw[gray!40, line width=0.4pt] (0.25,1.50) -- (1.95,1.50);
\draw[gray!40, line width=0.4pt] (0.25,1.85) -- (1.95,1.85);
\draw[gray!40, line width=0.4pt] (0.25,2.20) -- (1.95,2.20);
\draw[gray!40, line width=0.4pt] (0.25,2.55) -- (1.95,2.55);
\node[rotate=90] at (-0.45,1.5) {1024 tokens};
\node at (1.1,-0.35) {1024 x 768};
\node at (1.1,-0.75) {$X$};
\node at (2.75,1.5) {$=$};
\node[text=red!70!black] at (4.3,3.70) {token};
\node[text=red!70!black] at (4.3,3.38) {embedding};
\draw[thick, rounded corners=4pt, fill=red!8] (3.2,0) rectangle (5.4,3);
\draw[gray!40, line width=0.4pt] (3.45,0.45) -- (5.15,0.45);
\draw[gray!40, line width=0.4pt] (3.45,0.80) -- (5.15,0.80);
\draw[gray!40, line width=0.4pt] (3.45,1.15) -- (5.15,1.15);
\draw[gray!40, line width=0.4pt] (3.45,1.50) -- (5.15,1.50);
\draw[gray!40, line width=0.4pt] (3.45,1.85) -- (5.15,1.85);
\draw[gray!40, line width=0.4pt] (3.45,2.20) -- (5.15,2.20);
\draw[gray!40, line width=0.4pt] (3.45,2.55) -- (5.15,2.55);
\node at (4.3,-0.35) {1024 x 768};
\node at (4.3,-0.75) {$E_{\mathrm{token}}$};
\node at (5.95,1.5) {$+$};
\node[text=blue!70!black] at (7.5,3.70) {position};
\node[text=blue!70!black] at (7.5,3.38) {embedding};
\draw[thick, rounded corners=4pt, fill=blue!8] (6.4,0) rectangle (8.6,3);
\draw[gray!40, line width=0.4pt] (6.65,0.45) -- (8.35,0.45);
\draw[gray!40, line width=0.4pt] (6.65,0.80) -- (8.35,0.80);
\draw[gray!40, line width=0.4pt] (6.65,1.15) -- (8.35,1.15);
\draw[gray!40, line width=0.4pt] (6.65,1.50) -- (8.35,1.50);
\draw[gray!40, line width=0.4pt] (6.65,1.85) -- (8.35,1.85);
\draw[gray!40, line width=0.4pt] (6.65,2.20) -- (8.35,2.20);
\draw[gray!40, line width=0.4pt] (6.65,2.55) -- (8.35,2.55);
\node at (7.5,-0.35) {1024 x 768};
\node at (7.5,-0.75) {$E_{\mathrm{position}}$};
\end{tikzpicture}
\end{document}
```

<div style="text-align: center; color: var(--text-muted); font-size: 0.9em; margin-top: 6px; font-family: 'Songti SC', 'STSong', 'SimSun', serif;">1024 Tokens 对应 1024 行 embedding vector, 注意这里的 X 包含了 token 的含义信息和在文本中的位置信息</div>
<div style="text-align: center; color: var(--text-muted); margin: 2px 0 4px 0; font-family: 'Songti SC', 'STSong', 'SimSun', serif;">↓ linear projections</div>

```tikz
\begin{document}
\begin{tikzpicture}[every node/.style={font=\large}]
\node[text=purple!70!black] at (1.0,4.65) {Query};
\node at (1.0,4.20) {$Q$};
\node at (1.0,3.72) {$=XW^Q_{768\mathrm{x}64}$};
\draw[thick, rounded corners=4pt, fill=purple!8] (0,0) rectangle (2,3);
\draw[gray!40, line width=0.4pt] (0.25,0.55) -- (1.75,0.55);
\draw[gray!40, line width=0.4pt] (0.25,1.05) -- (1.75,1.05);
\draw[gray!40, line width=0.4pt] (0.25,1.55) -- (1.75,1.55);
\draw[gray!40, line width=0.4pt] (0.25,2.05) -- (1.75,2.05);
\draw[gray!40, line width=0.4pt] (0.25,2.55) -- (1.75,2.55);
\node[fill=purple!8, inner sep=1pt] at (1.0,2.55) {$q_1$};
\node[fill=purple!8, inner sep=1pt] at (1.0,2.05) {$q_2$};
\node[fill=purple!8, inner sep=1pt] at (1.0,1.55) {$q_3$};
\node[fill=purple!8, inner sep=1pt] at (1.0,1.05) {$\cdots$};
\node[fill=purple!8, inner sep=1pt] at (1.0,0.55) {$q_{1024}$};
\node[rotate=90] at (-0.45,1.5) {1024 tokens};
\node at (1.0,-0.35) {1024 x 64};
\node[text=orange!80!black] at (4.1,4.65) {Key};
\node at (4.1,4.20) {$K$};
\node at (4.1,3.72) {$=XW^K_{768\mathrm{x}64}$};
\draw[thick, rounded corners=4pt, fill=orange!8] (3.1,0) rectangle (5.1,3);
\draw[gray!40, line width=0.4pt] (3.35,0.55) -- (4.85,0.55);
\draw[gray!40, line width=0.4pt] (3.35,1.05) -- (4.85,1.05);
\draw[gray!40, line width=0.4pt] (3.35,1.55) -- (4.85,1.55);
\draw[gray!40, line width=0.4pt] (3.35,2.05) -- (4.85,2.05);
\draw[gray!40, line width=0.4pt] (3.35,2.55) -- (4.85,2.55);
\node[fill=orange!8, inner sep=1pt] at (4.1,2.55) {$k_1$};
\node[fill=orange!8, inner sep=1pt] at (4.1,2.05) {$k_2$};
\node[fill=orange!8, inner sep=1pt] at (4.1,1.55) {$k_3$};
\node[fill=orange!8, inner sep=1pt] at (4.1,1.05) {$\cdots$};
\node[fill=orange!8, inner sep=1pt] at (4.1,0.55) {$k_{1024}$};
\node at (4.1,-0.35) {1024 x 64};
\node[text=teal!70!black] at (7.2,4.65) {Value};
\node at (7.2,4.20) {$V$};
\node at (7.2,3.72) {$=XW^V_{768\mathrm{x}64}$};
\draw[thick, rounded corners=4pt, fill=teal!8] (6.2,0) rectangle (8.2,3);
\draw[gray!40, line width=0.4pt] (6.45,0.55) -- (7.95,0.55);
\draw[gray!40, line width=0.4pt] (6.45,1.05) -- (7.95,1.05);
\draw[gray!40, line width=0.4pt] (6.45,1.55) -- (7.95,1.55);
\draw[gray!40, line width=0.4pt] (6.45,2.05) -- (7.95,2.05);
\draw[gray!40, line width=0.4pt] (6.45,2.55) -- (7.95,2.55);
\node[fill=teal!8, inner sep=1pt] at (7.2,2.55) {$v_1$};
\node[fill=teal!8, inner sep=1pt] at (7.2,2.05) {$v_2$};
\node[fill=teal!8, inner sep=1pt] at (7.2,1.55) {$v_3$};
\node[fill=teal!8, inner sep=1pt] at (7.2,1.05) {$\cdots$};
\node[fill=teal!8, inner sep=1pt] at (7.2,0.55) {$v_{1024}$};
\node at (7.2,-0.35) {1024 x 64};
\end{tikzpicture}
\end{document}
```

<div style="text-align: center; color: var(--text-muted); font-size: 0.9em; margin-top: 6px; font-family: 'Songti SC', 'STSong', 'SimSun', serif;">这三个 W 是可以将 X 转换为特定问题(Q), 特征标签(K), 所需信息(V), 它们是随着神经网络迭代而更新的参数矩阵</div>
<div style="text-align: center; color: var(--text-muted); margin: 2px 0 4px 0; font-family: 'Songti SC', 'STSong', 'SimSun', serif;">↓ dot-product</div>

```tikz
\begin{document}
\begin{tikzpicture}[every node/.style={font=\large}]
\node[draw=purple!50, rounded corners=4pt, fill=purple!6, inner sep=8pt] at (0,2.16) {$QK^\top=(q_i\cdot k_j^\top)_{i,j}$};
\node at (2.35,2.16) {$=$};
\fill[yellow!14] (3.2,3.60) rectangle (14.85,4.32);
\fill[red!8] (3.2,0) rectangle (4.1,4.32);
\draw[thick, rounded corners=4pt] (3.2,0) rectangle (14.85,4.32);
\draw[gray!45, line width=0.4pt] (4.1,0) -- (4.1,4.32);
\draw[gray!45, line width=0.4pt] (6.25,0) -- (6.25,4.32);
\draw[gray!45, line width=0.4pt] (8.40,0) -- (8.40,4.32);
\draw[gray!45, line width=0.4pt] (10.55,0) -- (10.55,4.32);
\draw[gray!45, line width=0.4pt] (12.70,0) -- (12.70,4.32);
\draw[gray!45, line width=0.4pt] (3.2,0.72) -- (14.85,0.72);
\draw[gray!45, line width=0.4pt] (3.2,1.44) -- (14.85,1.44);
\draw[gray!45, line width=0.4pt] (3.2,2.16) -- (14.85,2.16);
\draw[gray!45, line width=0.4pt] (3.2,2.88) -- (14.85,2.88);
\draw[gray!45, line width=0.4pt] (3.2,3.60) -- (14.85,3.60);
\node at (3.65,3.96) {$\color{red!70!black}q_i\backslash\color{yellow!60!orange} k_j$};
\node[text=yellow!60!orange] at (5.18,3.96) {$k_1$};
\node[text=yellow!60!orange] at (7.33,3.96) {$k_2$};
\node[text=yellow!60!orange] at (9.48,3.96) {$k_3$};
\node[text=yellow!60!orange] at (11.63,3.96) {$\cdots$};
\node[text=yellow!60!orange] at (13.78,3.96) {$k_{1024}$};
\node[text=red!70!black] at (3.65,3.24) {$q_1$};
\node at (5.18,3.24) {$\scriptstyle q_1\cdot k_1^\top$};
\node at (7.33,3.24) {$\scriptstyle q_1\cdot k_2^\top$};
\node at (9.48,3.24) {$\scriptstyle q_1\cdot k_3^\top$};
\node at (11.63,3.24) {$\cdots$};
\node at (13.78,3.24) {$\scriptstyle q_1\cdot k_{1024}^\top$};
\node[text=red!70!black] at (3.65,2.52) {$q_2$};
\node at (5.18,2.52) {$\scriptstyle q_2\cdot k_1^\top$};
\node at (7.33,2.52) {$\scriptstyle q_2\cdot k_2^\top$};
\node at (9.48,2.52) {$\scriptstyle q_2\cdot k_3^\top$};
\node at (11.63,2.52) {$\cdots$};
\node at (13.78,2.52) {$\scriptstyle q_2\cdot k_{1024}^\top$};
\node[text=red!70!black] at (3.65,1.80) {$q_3$};
\node at (5.18,1.80) {$\scriptstyle q_3\cdot k_1^\top$};
\node at (7.33,1.80) {$\scriptstyle q_3\cdot k_2^\top$};
\node at (9.48,1.80) {$\scriptstyle q_3\cdot k_3^\top$};
\node at (11.63,1.80) {$\cdots$};
\node at (13.78,1.80) {$\scriptstyle q_3\cdot k_{1024}^\top$};
\node[text=red!70!black] at (3.65,1.08) {$\vdots$};
\node at (5.18,1.08) {$\vdots$};
\node at (7.33,1.08) {$\vdots$};
\node at (9.48,1.08) {$\vdots$};
\node at (11.63,1.08) {$\ddots$};
\node at (13.78,1.08) {$\vdots$};
\node[text=red!70!black] at (3.65,0.36) {$q_{1024}$};
\node at (5.18,0.36) {$\scriptstyle q_{1024}\cdot k_1^\top$};
\node at (7.33,0.36) {$\scriptstyle q_{1024}\cdot k_2^\top$};
\node at (9.48,0.36) {$\scriptstyle q_{1024}\cdot k_3^\top$};
\node at (11.63,0.36) {$\cdots$};
\node at (13.78,0.36) {$\scriptstyle q_{1024}\cdot k_{1024}^\top$};
\node at (9.0,-0.55) {1024 x 1024 attention-score table};
\end{tikzpicture}
\end{document}
```

```tikz
\begin{document}
\begin{tikzpicture}[every node/.style={font=\large}]
\node[draw=green!50!black, rounded corners=4pt, fill=green!6, inner sep=8pt] at (0,2.16) {$A=\mathrm{softmax}(QK^\top)=(\alpha_{ij})$};
\node at (2.95,2.16) {$=$};
\fill[yellow!14] (3.6,3.60) rectangle (14.9,4.32);
\fill[red!8] (3.6,0) rectangle (4.5,4.32);
\fill[green!8] (12.65,0) rectangle (14.9,4.32);
\draw[thick, rounded corners=4pt] (3.6,0) rectangle (14.9,4.32);
\draw[gray!45, line width=0.4pt] (4.5,0) -- (4.5,4.32);
\draw[gray!45, line width=0.4pt] (6.10,0) -- (6.10,4.32);
\draw[gray!45, line width=0.4pt] (7.70,0) -- (7.70,4.32);
\draw[gray!45, line width=0.4pt] (9.30,0) -- (9.30,4.32);
\draw[gray!45, line width=0.4pt] (10.90,0) -- (10.90,4.32);
\draw[gray!45, line width=0.4pt] (12.65,0) -- (12.65,4.32);
\draw[gray!45, line width=0.4pt] (3.6,0.72) -- (14.9,0.72);
\draw[gray!45, line width=0.4pt] (3.6,1.44) -- (14.9,1.44);
\draw[gray!45, line width=0.4pt] (3.6,2.16) -- (14.9,2.16);
\draw[gray!45, line width=0.4pt] (3.6,2.88) -- (14.9,2.88);
\draw[gray!45, line width=0.4pt] (3.6,3.60) -- (14.9,3.60);
\node at (4.05,3.96) {$\color{red!70!black}q_i\backslash\color{yellow!60!orange} k_j$};
\node[text=yellow!60!orange] at (5.30,3.96) {$k_1$};
\node[text=yellow!60!orange] at (6.90,3.96) {$k_2$};
\node[text=yellow!60!orange] at (8.50,3.96) {$k_3$};
\node[text=yellow!60!orange] at (10.10,3.96) {$\cdots$};
\node[text=yellow!60!orange] at (11.78,3.96) {$k_{1024}$};
\node[text=green!50!black] at (13.78,3.96) {row sum};
\node[text=red!70!black] at (4.05,3.24) {$q_1$};
\node at (5.30,3.24) {$\alpha_{11}$};
\node at (6.90,3.24) {$\alpha_{12}$};
\node at (8.50,3.24) {$\alpha_{13}$};
\node at (10.10,3.24) {$\cdots$};
\node at (11.78,3.24) {$\alpha_{1,1024}$};
\node at (13.78,3.24) {$\scriptstyle \sum_j\alpha_{1j}=1$};
\node[text=red!70!black] at (4.05,2.52) {$q_2$};
\node at (5.30,2.52) {$\alpha_{21}$};
\node at (6.90,2.52) {$\alpha_{22}$};
\node at (8.50,2.52) {$\alpha_{23}$};
\node at (10.10,2.52) {$\cdots$};
\node at (11.78,2.52) {$\alpha_{2,1024}$};
\node at (13.78,2.52) {$\scriptstyle \sum_j\alpha_{2j}=1$};
\node[text=red!70!black] at (4.05,1.80) {$q_3$};
\node at (5.30,1.80) {$\alpha_{31}$};
\node at (6.90,1.80) {$\alpha_{32}$};
\node at (8.50,1.80) {$\alpha_{33}$};
\node at (10.10,1.80) {$\cdots$};
\node at (11.78,1.80) {$\alpha_{3,1024}$};
\node at (13.78,1.80) {$\scriptstyle \sum_j\alpha_{3j}=1$};
\node[text=red!70!black] at (4.05,1.08) {$\vdots$};
\node at (5.30,1.08) {$\vdots$};
\node at (6.90,1.08) {$\vdots$};
\node at (8.50,1.08) {$\vdots$};
\node at (10.10,1.08) {$\ddots$};
\node at (11.78,1.08) {$\vdots$};
\node at (13.78,1.08) {$\vdots$};
\node[text=red!70!black] at (4.05,0.36) {$q_{1024}$};
\node at (5.30,0.36) {$\alpha_{1024,1}$};
\node at (6.90,0.36) {$\alpha_{1024,2}$};
\node at (8.50,0.36) {$\alpha_{1024,3}$};
\node at (10.10,0.36) {$\cdots$};
\node at (11.78,0.36) {$\alpha_{1024,1024}$};
\node at (13.78,0.36) {$\scriptstyle \sum_j\alpha_{1024,j}=1$};
\end{tikzpicture}
\end{document}
```


<div style="text-align: center; color: var(--text-muted); margin: 2px 0 4px 0; font-family: 'Songti SC', 'STSong', 'SimSun', serif;">↓ scaled dot-product attention</div>

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.$$

<div style="text-align: center; color: var(--text-muted); font-size: 0.9em; margin-top: 6px; font-family: 'Songti SC', 'STSong', 'SimSun', serif;">这里的 d_k, M 和多头为了简化问题就暂不论述了</div>

## References

- Nadaraya, E. A. (1964). On estimating regression. *Theory of Probability & Its Applications*, 9(1), 141–142. ^ref-nadaraya-1964
- Watson, G. S. (1964). Smooth regression analysis. *Sankhyā: The Indian Journal of Statistics, Series A*, 359–372. ^ref-watson-1964
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30. ^ref-vaswani-2017
