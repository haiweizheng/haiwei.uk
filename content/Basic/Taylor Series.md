---
draft: true
tags:
  - 分析
  - Taylor级数
  - 余项
  - 海森矩阵
  - 优化
  - analysis
  - taylor-series
  - remainder
  - hessian-matrix
  - optimization
---

## 余项
>[!definition] 泰勒展开式与余项
>$$f(x) = \underbrace{\sum_{i=0}^{n} \frac1{i!}f^{(i)}(0)\cdot (x-0)^i}_{P_n(x)} + R_n(x)$$
>1. **皮亚诺余项** (Peano Remainder)：$R_n(x) = o((x - x_0)^n)$
>2. **拉格朗日余项** (Lagrange Remainder)：$R_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!} (x - x_0)^{n+1}$
>3. **积分型余项** (Integral Form)：$R_n(x) = \int_{a}^{x} \frac{f^{(n+1)}(t)}{n!} (x - t)^n dt$
>$$R_n(x) = f^{(n+1)}(\xi) \cdot \frac{1}{n!} \int_{a}^{x} (x-t)^n \, dt = \frac{f^{(n+1)}(\xi)}{(n+1)!} (x-a)^{n+1}$$

***Proof.***
Integrate by parts
$$
\begin{aligned}
f(x) - f(a) &= \int_{a}^{x} f'(t) \, dt \\
f(x) &= f(a) + \int_{a}^{x} f'(t) \, d(t-x) \\
&= f(a) + f'(t)(t-x) \Big|_{a}^{x} - \int_{a}^{x} (t-x) \, df'(t) \\
&= \boxed{\color{blue}f(a)} + \boxed{\color{green}f'(a)(x-a)} + \underbrace{\boxed{\color{red}\int_{a}^{x} (x-t) \, df'(t)}}_{\displaystyle \color{red}R_1(x)}
\end{aligned}
$$
Then we know
$$
\begin{aligned}
\int_{a}^{x} (x-t) f''(t) \, dt &= \int_{a}^{x} f''(t) \, d\left( -\frac{(x-t)^2}{2} \right) \\
&= f''(t) \left( -\frac{(x-t)^2}{2} \right) \Big|_{a}^{x} + \int_{a}^{x} \frac{(x-t)^2}{2} \, df''(t) \\
&= \boxed{\color{orange}f''(a) \frac{(x-a)^2}{2}} + \underbrace{\boxed{\color{magenta}\int_{a}^{x} \frac{(x-t)^2}{2} \, df''(t)}}_{\displaystyle \color{magenta}R_2(x)}
\end{aligned}
$$
By induction
$$
\begin{aligned}
R_n(x) &= \int_{a}^{x} \frac{(x-t)^n}{n!} \, df^{(n)}(t) \\
&= \int_{a}^{x} \frac{(x-t)^n}{n!} f^{(n+1)}(t) \, dt
\end{aligned}
$$
$\boxed{\mathbf{QED}}$

## 中值定理

##### The 1st Mean Value Theorem for Integrals

>[!theo] Theorem. The 1st Mean Value Theorem for Integrals
>>- **条件**：$f(x)\in C^0[a, b]$ 
>>- **结论**：在开区间 $(a, b)$ 内至少存在一点 $\xi$，使得：$$\int_{a}^{b} f(x) \, dx = f(\xi)(b - a)$$
>
>>- **条件**：$f(x)$ 和 $g(x)$ 在 $[a, b]$ 上连续, $g(x)$ 在 $[a, b]$ 上**不变号**（即始终 $\ge 0$ 或始终 $\le 0$）。$g(x)$ 可以看作是一个“权重函数”。
>>- **结论**：在 $(a, b)$ 内至少存在一点 $\xi$，使得：$$\int_{a}^{b} f(x)g(x) \, dx = f(\xi) \int_{a}^{b} g(x) \, dx$$

***Proof.***
$f(x) \in [m, M] \ (x \in [a, b])\Rightarrow m(b-a) \leq \underbrace{\int_{a}^{b} f(x)dx}_{(b-a)f(\xi)} \leq M(b-a)$ 
When $g(x) \geqslant 0 \quad \forall x \in [a, b]$, we have $m \int_a^b g(x) \, dx \leqslant \underbrace{\int_a^b f \cdot g \, dx}_{f(\xi) \int_a^b g(x) \, dx} \leqslant M \int_a^b g(x) \, dx$

##### The 2nd Mean Value Theorem for Integrals

>[!theorem] The 2nd Mean Value Theorem for Integrals
>
>**Bonnet 型**
>
>若 $f(x)$ 在 $[a, b]$ 上**单调递减**且 $f(x) \geqslant 0$，且 $g(x)$ 在 $[a, b]$ 上可积，则存在 $\xi \in [a, b]$，使得：
>$$\int_a^b f(x)g(x) \, dx = f(a) \int_a^\xi g(x) \, dx$$
>若 $f(x)$ 在 $[a, b]$ 上**单调递增**且 $f(x) \geqslant 0$，则存在 $\xi \in [a, b]$，使得：
>$$\int_a^b f(x)g(x) \, dx = f(b) \int_\xi^b g(x) \, dx$$
>
>**Weierstrass 型**
>
>$f(x)$ 在 $[a, b]$ 上**单调**（无需非负），$g(x)$ 在 $[a, b]$ 上可积，则存在 $\xi \in [a, b]$，使得：
>$$\int_a^b f(x)g(x) \, dx = f(a) \int_a^\xi g(x) \, dx + f(b) \int_\xi^b g(x) \, dx$$

***Proof.***
When $f(x) \uparrow [a, b], f(x) \geqslant 0$, then $f(x)$ is integral, set $G(t) = \int_a^t g(x)  dx$
$$
\begin{aligned}
\int f g \, dx &= \int_a^b f(x) \, dG(x) \\
&= f(b) \cdot G(b) - \int_a^b G(x) \, df(x) \\
&= f(b) \int_a^b g(x) \, dx + \int_a^b G(x)\underbrace{ \, d(-f(x))}_{\geqslant 0} \\
&= f(b) \int_a^b g(x) \, dx + G(\xi) \int_a^b -df(x) \\
&= f(b) \int_a^b g(x) \, dx + \int_a^\xi g(x) \, dx \cdot (f(a) - f(b)) \\
&= f(b) \int_\xi^b g(x) \, dx + f(a) \int_a^\xi g(x) \, dx
\end{aligned}
$$

## 多元情况

##### Definition. 雅可比矩阵 (Jacobian Matrix)

>[!definition] Definition. 雅可比矩阵 (Jacobian Matrix)
>设 $f: \mathbb{R}^n \to \mathbb{R}^m$，$f = (f_1, f_2, \ldots, f_m)^T$，其中每个 $f_i$ 在某点 $\mathbf{x}_0$ 处可微。
>$$J(\mathbf{x}_0) = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix}$$
>特别地，当 $m = 1$（即 $f: \mathbb{R}^n \to \mathbb{R}$）时，雅可比矩阵就是梯度向量（行向量）：
>$$\nabla f(\mathbf{x}_0) = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right)$$

##### Definition. 海森矩阵 (Hessian Matrix)

>[!definition] Definition. 海森矩阵 (Hessian Matrix)
>设 $f: \mathbb{R}^n \to \mathbb{R}$ 在某点 $\mathbf{x}_0$ 处二阶可微，海森矩阵定义为：
>$$H(\mathbf{x}_0) = \begin{pmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{pmatrix}$$
>假设所有二阶偏导数连续，海森矩阵是**对称矩阵**。

>[!theo] Theorem. 海森矩阵与极值判断
>设 $\mathbf{x}_0$ 是 $f$ 的驻点（即 $\nabla f(\mathbf{x}_0) = \mathbf{0}$），且 $H(\mathbf{x}_0)$ 存在。
>
>- 若 $H(\mathbf{x}_0)$ **正定**，则 $\mathbf{x}_0$ 是 $f$ 的**严格极小值点**。
>- 若 $H(\mathbf{x}_0)$ **负定**，则 $\mathbf{x}_0$ 是 $f$ 的**严格极大值点**。
>- 若 $H(\mathbf{x}_0)$ **不定**（既有正特征值，也有负特征值），则 $\mathbf{x}_0$ 是 $f$ 的**鞍点**。
>- 若 $H(\mathbf{x}_0)$ **半正定**或**半负定**，判别需进一步分析。

>[!remark]
>海森矩阵的正定性在优化算法中的应用
>
>海森矩阵的正定性在判断优化算法可行性时非常有用：
>
>- **二阶充分条件**：若 $H(\mathbf{x}_0)$ 正定，则函数的二阶偏导数恒大于 $0$，函数的一阶导数始终处于递增状态。
>- **凸性判断**：函数为凸函数的充要条件为其海森矩阵半正定，其中多元函数的海森矩阵半正定类似于一元函数的二阶导数非负。
>- **收敛性保证**：在牛顿法等梯度方法中，利用海森矩阵的正定性可以便捷地判断函数是否具有凸性，从而判断是否可收敛到局部或全局最优解。


##### 梯度下降
$$
\begin{aligned}
f(x) &\approx f(x_0) + f'(x_0)(x - x_0) \\
0 &= f(x_0) + f'(x_0)(x - x_0) \\
&\Rightarrow x = x_0 - \frac{f(x_0)}{f'(x_0)} \\
&\Rightarrow x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
\end{aligned}
$$

##### Newton 法 (自动步长)
$$
\begin{aligned}
f(x) &\approx f(x_n) + f'(x_n)(x - x_n) + \frac{1}{2} f''(x_n)(x - x_n)^2 \\
f'(x) &= f'(x_n) + f''(x_n)(x - x_n) = 0 \\
&\Rightarrow x = x_n - \frac{f'(x_n)}{f''(x_n)} \\
&\Rightarrow x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)} \\
&\Rightarrow x_{n+1} = x_n - \frac{\nabla f(x_n)}{H_f(x_n)}
\end{aligned}
$$
![[Taylor Series-1775457819064.webp]]

**注意：** 虽然牛顿法“收敛”快（迭代次数少），但由于每次都要计算海森矩阵并求逆，单次迭代的**计算成本**非常高, 实际工程中常使用拟牛顿法 (Quasi-Newton) 来折中。

##### 多元函数的 Taylor 展开
$$f(x) = f(x_0) + f'(x_0)(x - x_0) + \frac{1}{2!} f''(x_0)(x - x_0)^2 + \cdots$$
① For $f: \mathbb{R}^n \to \mathbb{R} \quad \forall x = (x_1, \dots, x_n)^T, \Delta x = (\Delta x_1, \dots, \Delta x_n)^T$
$$\begin{aligned}
f(x + \Delta x) &= f(x) + \sum_{i=1}^n \Delta x_i \cdot \frac{\partial f}{\partial x_i}(x) \\
&\quad + \frac{1}{2!} \sum_{i,j=1}^n \Delta x_i \Delta x_j \frac{\partial^2 f}{\partial x_i x_j}(x) \\
&\quad + \frac{1}{3!} \sum_{i,j,k=1}^n \Delta x_i \Delta x_j \Delta x_k \frac{\partial^3 f}{\partial x_i \partial x_j \partial x_k}(x) + \cdots \\[0.5em]
f(x + \Delta x) &= f(x) + \nabla f(x)^T \Delta x + \frac{1}{2!} \Delta x^T H(x) \Delta x \\
&\quad + \underbrace{\frac{1}{3!} \nabla^3 f(x) (\Delta x \otimes \Delta x \otimes \Delta x)}_{\text{Tensor}} + \cdots
\end{aligned}$$

② For $f: \mathbb{R}^{m \times n} \to \mathbb{R}, \forall M_{m \times n}$，Frobenius 范数和内积：
$$\begin{aligned}
\|M\|_F &= \sqrt{\sum_{i=1}^m \sum_{j=1}^n |M_{ij}|^2} \\
\langle A, B \rangle_F &= \text{Tr}(A^T B) = \sum_{i=1}^m \sum_{j=1}^n a_{ij} b_{ij}= \sum_{i=1}^m \sum_{j=1}^n \overline{a_{ij}} b_{ij} \\

\end{aligned}$$
$$\begin{aligned}
f(W + \Delta W) &= f(W) + \langle \nabla f(W), \Delta W \rangle_F \\
&\quad + \frac{1}{2!} \Delta W^T H(W) \Delta W + \cdots
\end{aligned}$$
$$\begin{aligned}
f(W + \Delta W) &= f(W) \\
&\quad + \langle \nabla f(W), \Delta W \rangle_F \quad \text{(一阶：向量内积)} \\
&\quad + \frac{1}{2} \langle \nabla^2 f(W), \Delta W \otimes \Delta W \rangle_F \quad \text{(二阶：张量内积)} \\
&\quad + \frac{1}{6} \langle \nabla^3 f(W), \Delta W \otimes \Delta W \otimes \Delta W \rangle_F \quad \text{(三阶：张量内积)}\\
&\quad +\cdots
\end{aligned}$$

## 二次上界

>[!lemma] Descent Lemma (二次上界)
> 若 $f: \mathbb{R}^{m\times n} \to \mathbb{R}$ 是 Frobenius 范数 $L$-Lipschitz 光滑的
>$$\|\nabla f(W) - \nabla f(W')\|_F \leq L\|W - W'\|_F$$ 
>则对任意 $W, W' \in \mathbb{R}^{m\times n}$,
> $$f(W') \leq f(W) + \langle \nabla f(W),\, W' - W\rangle_F + \frac{L}{2}\|W' - W\|_F^2$$
> 即 $f$ 被以 $W$ 为中心的二次函数从上方扣住, "碗"的曲率由 $L$ 决定.

***Proof.*** 由微积分基本定理, 沿从 $W$ 到 $W'$ 的直线路径积分:

$$\begin{aligned} f(W') - f(W) &= \int_{W}^{W'} \langle \nabla f(t),\, I\rangle_F\, dt \\ &= \int_0^{W' - W} \langle \nabla f(W + t),\, I\rangle_F\, dt \\ &= \int_0^1 \langle \nabla f(W + t(W' - W)),\, I\rangle_F\, d(W' - W)t \\ &= \int_0^1 \langle \nabla f(W + t(W' - W)),\, W' - W\rangle_F\, dt\\ &= \langle\nabla f(W), W' - W\rangle_F \\ &\quad + \int_0^1 \langle \nabla f(W + t(W'-W)) - \nabla f(W),\, W' - W\rangle_F\, dt \end{aligned}$$

对第二项用 Cauchy-Schwarz 不等式和 Lipschitz 光滑性:
$$\begin{aligned}
&\left|\int_0^1 \langle \nabla f(W + t(W'-W)) - \nabla f(W),\, W' - W\rangle_F\, dt\right| \\
&\quad\leq \int_0^1 \|\nabla f(W + t(W'-W)) - \nabla f(W)\|_F \cdot \|W' - W\|_F\, dt \\
&\quad\leq \int_0^1 L t \|W' - W\|_F \cdot \|W' - W\|_F \, dt \\
&\quad= \int_0^1 L t \|W' - W\|_F^2 \, dt \\
&\quad= \frac{L}{2}\|W' - W\|_F^2
\end{aligned}$$
则
$$f(W') - f(W) \leq \langle\nabla f(W), W'-W\rangle_F + \frac{L}{2}\|W' - W\|_F^2. \quad \blacksquare$$

**应用 1: GD 的学习率上限**
代入 GD 一步 $W' = W - \eta\nabla f(W)$:
$$\begin{aligned} f(W') &\leq f(W) - \eta\|\nabla f(W)\|_F^2 + \frac{L\eta^2}{2}\|\nabla f(W)\|_F^2 \\ &= f(W) - \eta\left(1 - \frac{L\eta}{2}\right)\|\nabla f(W)\|_F^2 \end{aligned}$$
要保证下降需要 $1 - L\eta/2 > 0$, 即学习率上限 $\eta < 2/L$, 最优选择 (求导) $\eta^* = 1/L$.

**应用 2: Muon 证明的起点**
Theorem 4.1 证明 (B.1 节) 第一步直接用 descent lemma 展开 $f(W_t) - f(W_{t+1})$:
$$\mathbb{E}[f(W_t) - f(W_{t+1})] \geq \mathbb{E}\left[\eta\langle\nabla f(W_t), U_t V_t^\top\rangle_F - \frac{L}{2}\eta^2 \|U_t V_t^\top\|_F^2\right]$$
这正是 Assumption 3.1 通过 descent lemma 直接给出的 "每步下降量下界".


## 算法验证 牛顿法与简单的梯度下降

### Gradient Descent

>[!example] Problem 3 (Condition number and convergence rate)
>Let $f(x,y) = 50x^2 + y^2$.
>(a) Write out the gradient descent iteration with step size $\eta$.
>(b) Find the range of $\eta$ for convergence.
>(c) Show that the convergence rate depends on the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ and derive the bound $\|x_k - x^*\| \le \left(\frac{\kappa - 1}{\kappa + 1}\right)^k \|x_0 - x^*\|$.

>[!example] Problem 4 (Saddle point escape)
>Consider $f(x,y) = x^2 - y^2$. Starting from $x_0 = (\epsilon, \epsilon)$ with small $\epsilon > 0$, describe the behavior of gradient descent. Why does GD fail here, and how does adding noise (SGD) help?

>[!example] Problem 5 (Non-convex landscape)
>Let $f(x) = x \sin(x)$ on $[0, 10\pi]$. Run gradient descent from $x_0 = 8$ with $\eta = 0.05$. Which local minimum does it reach? Is it the global minimum? Discuss the role of initialization.

### Newton's Method (Root Finding)

>[!example] Problem 6 (Basic Newton-Raphson)
>Use Newton's method to find $\sqrt{7}$: formulate it as finding the root of $f(x) = x^2 - 7$. Starting from $x_0 = 3$, compute $x_1, x_2, x_3$ and compare with $\sqrt{7} \approx 2.6457513\ldots$

>[!example] Problem 7 (Convergence failure)
>Apply Newton's method to $f(x) = x^{1/3}$ starting from $x_0 = 1$. Show that $|x_{k+1}| = 2|x_k|$ and the method diverges. What goes wrong geometrically?

>[!example] Problem 8 (Multiplicity and convergence order)
>For $f(x) = (x-1)^3$, show that Newton's method starting from any $x_0 \neq 1$ converges to $x^* = 1$ but only linearly (not quadratically). Prove the rate is $|x_{k+1} - 1| = \frac{2}{3}|x_k - 1|$. How can you modify Newton's method to restore quadratic convergence when the root has known multiplicity $m$?

### Newton's Method (Optimization)

>[!example] Problem 9 (Quadratic in one step)
>Show that Newton's method for minimizing a quadratic $f(x) = \frac{1}{2}x^T A x - b^T x$ ($A \succ 0$) converges in exactly one step from any initial point.

>[!example] Problem 10 (Rosenbrock function: Newton vs GD)
>Let $f(x,y) = 100(y - x^2)^2 + (1-x)^2$.
>(a) Compute the gradient and Hessian.
>(b) Explain why gradient descent is slow near the minimum (hint: condition number).
>(c) Run Newton's method from $(0,0)$. Does it converge? Is the Hessian always positive definite along the path?

>[!example] Problem 11 (Damped Newton and line search)
>For $f(x) = e^x - 2x$:
>(a) Find the minimizer analytically.
>(b) Newton's method gives $x_{k+1} = x_k - \frac{e^{x_k} - 2}{e^{x_k}}$. Starting from $x_0 = 10$, show that pure Newton overshoots. Propose a line search modification $x_{k+1} = x_k - t_k \frac{f'(x_k)}{f''(x_k)}$ and explain how to choose $t_k$ (e.g. Armijo condition).

>[!example] Problem 12 (High-dimensional least squares)
>Consider $f(x) = \frac{1}{2}\|Ax - b\|^2$ with $A \in \mathbb{R}^{m \times n}$, $m \gg n$.
>(a) Write down the Newton step. How does it relate to solving the normal equations?
>(b) What is the per-iteration cost of Newton vs gradient descent?
>(c) When $n$ is large, why might quasi-Newton methods (BFGS) or conjugate gradient be preferred?

## 参考

[海森矩阵与优化算法](https://xishansnow.github.io/posts/2a7b098e.html)