---
source: "[[High-Dimensional Analysis- Random Matrices and Machine Learning.pdf]]"
tags:
  - 随机矩阵与机器学习_random-matrices-and-machine-learning
  - 梯度下降_gradient-descent
  - 神经切线核_neural-tangent-kernel
  - NTK_neural-tangent-kernel
---

Consider our random feature neural network function
$$Y = f(X) = w\sigma(W_1X)=w_{1\times m}\sigma((W_1)_{m\times p}X_{p\times n}).$$
We have a kind of an understanding of how the statistical properties of $X$ influence the statistical properties of $Y$ if $W_1$ and $w$ are fixed (e.g., $W_1$ is randomly chosen). But the crux of a neural network is to find $w$ and $W_1$ through "learning" such that a given set of data $\hat{Y} = f(X)$ is described best. In the above we can
- fix $W_1$ (deterministically or randomly) and only learn $w$ (*linear regression*)
- learn both $w$ and $W_1$ (*feature learning*).
## 8.1 Gradient descent for linear regression

Let us reconsider our linear regression problem in the over-parameterized (= under-determined 数据太少) case. Given $X \in \mathbb{R}^{p \times n}$ and $\hat{Y} \in \mathbb{R}^{1 \times n}$, we seek $\hat{w} \in \mathbb{R}^{1 \times p}$ such that $\hat{Y} = \hat{w}X$ in the under-determined case $n < p$.

There are typically (i.e. if $\mathrm{rank}(X) = n$) infinitely many solutions and the “best” (i.e., the one with the smallest norm) is given by
$$\hat{w} = \hat{Y}(X^TX)^{-1}X^T. \tag{3}$$
Since taking inverses of big matrices is usually not a good idea, even here it is better to have an algorithm to approximate the solution via iterations, for any $w$:
$$\begin{aligned}
\hat{Y}_{1\times n} &= \hat{w}_{1\times p}X_{p\times n}\\
\|\hat{Y} - wX\|^2 
&= (\hat{Y} - wX)(\hat{Y} - wX)^T \\
&= \hat{Y}\hat{Y}^T - \underbrace{\hat{Y}X^T}_{1\times p} 
w^T - w\underbrace{X\hat{Y}^T}_{p\times 1} + w_{1\times p}XX^Tw^T\\
\nabla_w \|\hat{Y} - wX\|^2 &= -2\hat{Y}X^T + 2wXX^T.
\end{aligned}$$

***Proof.***
$$\begin{aligned}
\nabla_w \underbrace{(-\hat{Y}X^T)}_{a_{1 \times p}} w^T &= \nabla_w (\sum_{i=1}^p a_i w_i) = (a_1, \dots, a_p) = a = -\hat{Y}X^T \\
\nabla_w w(-X\hat{Y}^T) &= \nabla_w (-\hat{Y}X^T)^T w^T = -\hat{Y}X^T \\
\nabla_w (wXX^T w^T) &= \nabla_w (\sum_{i,j=1}^p w_i (XX^T)_{ij} w_j) \\
&= (\sum_{j=1}^p (XX^T)_{1j} w_j + \sum_{i=1}^p w_i (XX^T)_{i1}, \dots) \\
&= (\sum_{j=1}^p 2(XX^T)_{1j} w_j, \dots) \\
&= (\sum_{j=1}^p 2(XX^T)_{j1} \cdot w_j, \dots) \\
&= 2wXX^T
\end{aligned}$$
Thus
$$\nabla_w \|\hat{Y} - \hat{w}X\|^2 = 0\rightarrow\hat{w}\ \boxed{XX^T}_{p\times p} = \hat{Y}X^T,$$
which gives the least square solution
- Over-determined case $n\geq p$, $\hat{w} = \hat{Y}X^T(XX^T)^{-1}$
- Under-determined case $n\leq p$
$$
\hat{w} = \hat{Y}X^T(XX^T)^+ = \hat{Y}(X^TX)^{-1}X^T
$$
where $A^+$ is the pseudo-inverse of $A$.

Starting from some $w^{(0)}$ take
$$
w^{(t+1)} := w^{(t)} - \eta \nabla_w \|\hat{Y} - w^{(t)}X\|^2
= w^{(t)} - \eta \cdot 2\left(w^{(t)}XX^T - \hat{Y}X^T\right),
$$
where $t \in \mathbb{N}_0$ is the time and $\eta$ is the “step size” or “learning rate”. For $\eta$ small enough this algorithm will converge. In the over-determined case, it converges to the least square solution, in the under-determined case it converges to a solution, but this might not be the best one. Thus it is advantageous to improve the algorithm by regularization.

>[!example] Example. 8.1 梯度下降只能更新那些在损失函数中“有贡献”的参数
>$$\hat{Y}=f(x) = wx = (\theta_1 \ \theta_2)
\begin{pmatrix} t_1 \\ t_2 \end{pmatrix}
= \theta_1 t_1 + \theta_2 t_2.$$
>Given that$$x = \begin{pmatrix} 0 \\ 1 \end{pmatrix}, \quad \hat{Y} = (1).$$
Thus we want to find the best $\hat{w} = (\hat{\theta}_1 \ \hat{\theta}_2)$ such that $\hat{Y} = \hat{w}X$.
>- Analytical solution:$$\hat{w} = \hat{Y}(X^TX)^{-1}X^T
= 1 \cdot \left((0 \ 1)\begin{pmatrix} 0 \\ 1 \end{pmatrix}\right)^{-1} (0 \ 1)
= (0 \ 1),$$
i.e. $\hat{\theta}_1 = 0$ and $\hat{\theta}_2 = 1$, which is clearly the solution with the smallest norm.
>- Numerical solution:
>$$\begin{aligned} 
w^{(t)} &= (\theta_1^{(t)} \ \theta_2^{(t)})\\
\begin{pmatrix}\theta_1^{(t+1)} & \theta_2^{(t+1)}\end{pmatrix}
&= w^{(t+1)} = w^{(t)} - \eta \nabla_w \|\hat{Y} - w^{(t)}X\|^2\\
&= w^{(t)} - \eta \cdot 2\left(w^{(t)}\begin{pmatrix}0 & 0 \\0 & 1\end{pmatrix}- (0 \ 1)\right)\\
&= \left(\theta_1^{(t)}, \ \theta_2^{(t)} - 2\eta(\theta_2^{(t)} - 1)\right).\\
\theta_1^{(t+1)} &= \theta_1^{(t)}, \quad \theta_2^{(t+1)} - 1 = (\theta_2^{(t)} - 1)(1 - 2\eta).
\end{aligned}$$
>Hence, if $|1 - 2\eta| < 1$, $\theta_2$ converges to the right solution $\hat{\theta}_2 = 1$, but $\theta_1$ does not change at all. So if we start with $\theta_1^{(0)} \ne 0$, we will not converge to the solution with the smallest norm.

在原始最小二乘问题 $\min_{\hat{w}} \|\hat{Y} - \hat{w}X\|^2$ 中，当 $XX^T$ 不可逆时解不唯一 (哪怕可逆也有可能条件数爆炸高)，梯度下降也未必收敛到最小范数解. 正则化通过在目标函数上加一个惩罚项 $\lambda\|w\|^2$ 来约束解的复杂度：
- $\lambda$ 越大，解的范数越小但拟合误差越大；
- $\lambda$ 越小，越接近无约束的最小二乘解. 
$\lambda I$ 将 $XX^T$ 的所有特征值抬高 $\lambda$，保证解的唯一性和数值稳定性. 

>[!definition] Definition. 8.2 Ridge Regression
>Instead of minimizing$$\mathcal{L}(w) := \|\hat{Y} - wX\|^2,$$
we add a penalty term for large norms, i.e. we want to minimize$$\mathcal{L}_\lambda(w) := \|\hat{Y} - wX\|^2 + \lambda \|w\|^2\quad \text{for some } \lambda > 0.$$
The minimizer of this is determined by$$\begin{aligned}  
0 = \nabla_w \mathcal{L}_\lambda(w)
&= 2(wXX^T - \hat{Y}X^T) + \lambda \nabla_w \underbrace{(ww^T)}_{2w} \\
&= \underbrace{2\left(w(XX^T + \lambda I) - \hat{Y}X^T\right)}_{\hat{w}(XX^T + \lambda I) = \hat{Y}X^T.} 
\end{aligned}$$
>Note that for $\lambda > 0$, the matrix $XX^T + \lambda I$ is always invertible even if $XX^T$ is not invertible, since $XX^T$ is positive semi-definite. So we have a unique solution$$\hat{w} = \hat{Y}X^T(XX^T + \lambda I)^{-1},$$
>- does not interpolate exactly anymore, but we allow small (depending on $\lambda$) errors
>- is “nice” (has small norm, or more general some good smoothness properties).
>- Gradient descent with the “ridged” gradient converges to this solution. Note also that
$$\lim_{\lambda \downarrow 0} X^T(XX^T + \lambda I)^{-1} = X^+,$$the pseudo-inverse of $X$, and the ridge regression solution converges for $\lambda \downarrow 0$ to the regression solution.

>[!example] Example. 8.3 (Ridge regression for Example 8.1)
>The unique solution $\hat{w}_\lambda$ for $\lambda$ is
>$$\begin{aligned}\hat{w}_\lambda &= \hat{Y}X^T(XX^T + \lambda I)^{-1}
>= 1 \cdot (0 \ 1)\left(\begin{pmatrix}0 & 0 \\ 0 & 1\end{pmatrix} + \begin{pmatrix}\lambda & 0 \\ 0 & \lambda\end{pmatrix}\right)^{-1}\\
>&= (0 \ 1)\begin{pmatrix}\frac{1}{\lambda} & 0 \\ 0 & \frac{1}{1+\lambda}\end{pmatrix}
>= \left(0 \ \frac{1}{1+\lambda}\right) \xrightarrow{\lambda \downarrow 0} (0 \ 1) = \hat{w}.\end{aligned}$$
>Gradient descent iteration gives now
>$$\begin{aligned}w_\lambda^{(t+1)} &= w_\lambda^{(t)} - 2\eta \left((0 \ \theta_2^{(t)}) + \lambda w_\lambda^{(t)} - (0 \ 1)\right),\\
>\theta_1^{(t+1)} &= \theta_1^{(t)} - 2\eta \lambda \theta_1^{(t)} = \theta_1^{(t)}(1 - 2\eta \lambda),\\
>\theta_2^{(t+1)} &= \theta_2^{(t)} - 2\eta \left((\theta_2^{(t)})(1+\lambda) - 1\right).\end{aligned}$$
>In particular, we have
>$$\theta_1^{(t)} \to 0 \quad \text{if } |1 - 2\eta\lambda| < 1$$
>and
>$$\theta_2^{(t)} \to \frac{1}{1+\lambda} \quad \text{if } |1 - 2\eta(1+\lambda)| < 1,$$
>since
>$$\theta_2^{(t+1)} - \frac{1}{1+\lambda}
>= \left(\theta_2^{(t)} - \frac{1}{1+\lambda}\right)(1 - 2\eta(1+\lambda)).$$

## 8.2. Gradient descent for feature learning

Now let's go beyond linear regression and allow all parameters to be trained. We consider
$$f(x) = \underbrace{w_{1\times m}}_{a^T} \cdot \sigma \left( \underbrace{\frac{1}{\sqrt{p}}(W_1)_{m\times p}}_{W_{m\times p}} x_{p\times 1} \right)$$
$$f_{\theta}(x) = a^T \cdot \sigma(Wx)$$
depending on the parameters $\theta := \{a, W\}$ (a $(m + m \cdot p)$-dimensional vector). 

For given $n$ observations $(\hat{x}_1, \hat{y}_1), \dots, (\hat{x}_n, \hat{y}_n) \in \mathbb{R}^p \times \mathbb{R},$ we measure deviation from this by the loss function
$$\mathcal{L}(\theta) = \frac{1}{2} \sum_{k=1}^n (f_{\theta}(\hat{x}_k) - \hat{y}_k)^2.\tag{MSE}$$
We want to minimize this by changing $\theta$ via gradient descent (discrete)
$$\theta(t + 1) = \theta(t) - \eta \nabla_{\theta} \mathcal{L}(\theta(t)),$$
Then by renaming (continuing)
$$\lim_{\Delta t\rightarrow 0}\frac{\theta(t + \Delta t)-\theta(t)}{\Delta t}=  - \eta  \nabla_{\theta} \mathcal{L}(\theta(t)),$$
Any change in $\theta$ induces a change in $f_t := f_{\theta(t)}$, which is our main concern. We have
$$\begin{aligned}  
\frac{\mathrm{d} f_t(x)}{\mathrm{d} t} := \frac{\mathrm{d} f_{\theta(t)}(x)}{\mathrm{d} t} &= \nabla_{\theta} f_{\theta(t)}(x)^T \cdot \frac{\mathrm{d} \theta(t)}{\mathrm{d} t} =  \nabla_{\theta} f_{\theta(t)}(x)^T \cdot[- \eta \underbrace{\nabla_{\theta} \mathcal{L}(\theta(t))}_{} 
].\\
\nabla_{\theta} \mathcal{L}(\theta_t) &= \sum_{k=1}^n \nabla_{\theta} \frac{1}{2} (f_{\theta_t}(\hat{x}_k) - \hat{y}_k)^2 = \boxed{\sum_{k=1}^n (f_t(\hat{x}_k) - \hat{y}_k) \nabla_{\theta} f_t(\hat{x}_k)}\\
\frac{\mathrm{d} f_t(x)}{\mathrm{d} t} &= -\eta \sum_{k=1}^n \nabla_{\theta} f_t(x)^T \cdot \nabla_{\theta} f_t(\hat{x}_k) \cdot (f_t(\hat{x}_k) - \hat{y}_k).\\
&= -\underbrace{\eta}_{\text{步长}} \sum_{k=1}^n \underbrace{\boxed{\nabla_{\theta} f_t(x)^T \cdot \nabla_{\theta} f_t(\hat{x}_k)}}_{\text{相似度度量}} \cdot \underbrace{(f_t(\hat{x}_k) - \hat{y}_k)}_{\text{训练误差项}}.
\end{aligned}$$

## 8.3. Neural tangent kernel

>[!definition] Definition. *neural tangent kernel*
>$$k_t(x, \tilde{x}) := \nabla_{\theta} f_t(x)^T \cdot \nabla_{\theta} f_t(\tilde{x})=\langle x,\tilde{x} \rangle_{\nabla_{\theta}f_t(\cdot)},$$which was introduced by Jacot, Gabriel, and Hongler [JGH18] in 2018.

A priori (先验) $k_t$ is a probabilistic object (随机对象) which depends on time $t$. Unless we can say more about it, the above is just a compact and useless way of writing down the time-evolution in an abstract way.

In the large *Layer Width* limit ($W_{m\times p}$ 隐藏层宽度=神经元数量) $m \to \infty$, $k_t$ ($k_t(X, \hat{X})$) converges to a limit object $k$, which
* is deterministic (which we can believe by concentration),
* is independent of time (变换相对于 $m$ 非常微弱, which is not so clear right now, maybe later there will be more on this)
* stays away from zero, i.e. $k \geq \delta I$ for some $\delta > 0$, thus has only strictly positive eigenvalues.

Note that with
$$\begin{aligned}  
\hat{X} &= (\hat{x}_1 \quad \dots \quad \hat{x}_n) \quad \text{and} \quad \hat{Y} = \begin{pmatrix} \hat{y}_1 \\ \vdots \\ \hat{y}_n \end{pmatrix}\\
\frac{\mathrm{d} f_t(x)}{\mathrm{d} t} &= -\eta \nabla_\theta f_t(x)^T \cdot \sum_{k=1}^n \nabla_\theta f_t(\hat{x}_k) \cdot (f_t(\hat{x}_k) - \hat{y}_k) \\ &= -\eta \nabla_\theta f_t(x)^T \cdot \nabla_\theta f_t(\hat{X}) \cdot (f_t(\hat{X}) - \hat{Y}) \\
\frac{\mathrm{d} f_t(\hat{X})}{\mathrm{d} t} &= -\eta \nabla_\theta f_t(\hat{X})^T \cdot \nabla_\theta f_t(\hat{X}) \cdot (f_t(\hat{X}) - \hat{Y}),\\
\frac{\mathrm{d} (f_t(\hat{X}) - \hat{Y})}{\mathrm{d} t} &= -\eta \underbrace{\nabla_\theta f_t(\hat{X})^T \cdot \nabla_\theta f_t(\hat{X})}_{\substack{\approx k(\hat{X}, \hat{X})\text{ is constant}\\ \text{and } \geq \delta I}} \cdot (f_t(\hat{X}) - \hat{Y}),
\end{aligned}$$
Set $u(t)=(f_t(\hat{X}) - \hat{Y})_{n\times 1}$ and $$k(\hat{X}, \hat{X})=K= \begin{bmatrix} 
k(\hat{x}_1, \hat{x}_1) & \dots & k(\hat{x}_1, \hat{x}_n) \\ 
\vdots & \ddots & \vdots \\ 
k(\hat{x}_n, \hat{x}_1) & \dots & k(\hat{x}_n, \hat{x}_n) 
\end{bmatrix}$$
so $f_t(\hat{X})$ converges exponentially to $\hat{Y}$:
$$(f_t(\hat{X}) - \hat{Y}) = (f_0(\hat{X}) - \hat{Y}) \exp(-\eta t k(\hat{X}, \hat{X})) ,$$
### 8.4. Test error in the random feature model

Thus the training error goes to zero; but how about the test error? What is the prediction for $t \to \infty$ for arbitrary, "unseen" data $x$? By 
$$\begin{aligned}  
\frac{\mathrm{d} f_t(x)}{\mathrm{d} t} &= -\eta \nabla_\theta f_t(x)^T \cdot \sum_{k=1}^n \nabla_\theta f_t(\hat{x}_k) \cdot (f_t(\hat{x}_k) - \hat{y}_k) \\ &= -\eta \nabla_\theta f_t(x)^T \cdot \nabla_\theta f_t(\hat{X}) \cdot (f_t(\hat{X}) - \hat{Y}) \\
&= -\eta k(x, \hat{X})_{1\times n}(f_t(\hat{X}) - \hat{Y})_{n\times 1} \\
&= -\eta k(x, \hat{X}) \cdot(f_0(\hat{X}) - \hat{Y})\cdot\exp(-\eta t k(\hat{X}, \hat{X})) ,
\end{aligned}$$
and
$$\begin{aligned}  
u(t)&=(f_t(\hat{X}) - \hat{Y}) = (f_0(\hat{X}) - \hat{Y}) \exp(-\eta t k(\hat{X}, \hat{X})) ,\\
&= u(0)\cdot\exp(-\eta t K) \\
f_t(x) &= \int_0^t -\eta k(x, \hat{X}) \cdot u(0)\cdot \exp(-\eta t K)  \, \mathrm{d}t\\
&= -\eta k(x, \hat{X})\cdot u(0)\cdot \left( \int \exp(-\eta t K) \, \mathrm{d}t \right)\\
&= -\eta k(x, \hat{X})\cdot u(0)\cdot \left( \frac{1}{-\eta K}\exp(-\eta tK)+C_* \right)\\
&=  k(x, \hat{X})\cdot\frac 1K\cdot u(0)\cdot  \exp(-\eta tK) +C_\star\\
&= k(x, \hat{X}) \cdot k(\hat{X}, \hat{X})^{-1} \exp(-\eta t k(\hat{X}, \hat{X})) \cdot (f_0(\hat{X}) - \hat{Y}) + C.
\end{aligned}$$
For $t = 0$, we have
$$f_0(x) = C + k(x, \hat{X}) \cdot k(\hat{X}, \hat{X})^{-1}(f_0(\hat{X}) - \hat{Y}),$$
so
$$
\begin{aligned}
f_t(x) &= [f_0(x) - k(x, \hat{X}) \cdot k(\hat{X}, \hat{X})^{-1} \cdot (f_0(\hat{X}) - \hat{Y})] \\
&\quad + k(x, \hat{X}) \cdot k(\hat{X}, \hat{X})^{-1} \exp(-\eta t k(\hat{X}, \hat{X})) \cdot (f_0(\hat{X}) - \hat{Y}) \\
&= f_0(x) + k(x, \hat{X}) \cdot k(\hat{X}, \hat{X})^{-1} (\exp(-\eta t k(\hat{X}, \hat{X})) - 1) \cdot (f_0(\hat{X}) - \hat{Y})\\
f_\infty(x) &= f_0(x) + k(x, \hat{X}) \cdot k(\hat{X}, \hat{X})^{-1} \cdot (\hat{Y} - f_0(\hat{X})).
\end{aligned}
$$
Applying some centering we can restrict to the case where $f_0 = 0$, thus
$$
f_\infty(x) = k(x, \hat{X}) k(\hat{X}, \hat{X})^{-1} \hat{Y}.
$$
We now have to prescribe some model for the unseen data, like
$$
y = g(x) = a_T^T \sigma_T(W_T x) + N,
$$
where the subindex $T$ stands for "teacher" and $N$ is some noise. Then the test error is
$$
E_{\text{test}} = E_x \left[ (g(x) - f_\infty(x))^2 \right] = E_x \left[ (g(x) - k(x, \hat{X}) k(\hat{X}, \hat{X})^{-1} \hat{Y})^2 \right].
$$
In principle, this can be expressed as a complicated, but manageable (namely rational) function in the involved random matrices. In particular, note:

(i) The quantities $\nabla_\theta f_\theta(x)$ and thus $k(x, \tilde{x})$ can be given explicitly: from
$$f_\theta(x) = a^T \sigma(Wx)$$
we get as in linear regression
$$\nabla_a f_\theta(x) = \sigma(Wx).$$
But what is $\nabla_w f_\theta(x)$? Write
$$a = \begin{pmatrix} a_1 \\ \vdots \\ a_m \end{pmatrix} \quad \text{and} \quad W = \begin{pmatrix} w_1^T \\ \vdots \\ w_m^T \end{pmatrix} \quad \text{for} \quad w_i \in \mathbb{R}^p,$$
then
$$f_\theta(x) = \sum_{i=1}^m a_i \sigma(w_i^T x)$$
$$\nabla_{w_i} f_\theta(x) = a_i \sigma'(w_i^T x) \cdot x,$$
thus
$$
\begin{aligned}
k(x, \tilde{x}) &= \nabla_a f_\theta(x)^T \cdot \nabla_a f_\theta(\tilde{x}) + \nabla_w f_\theta(x)^T \cdot \nabla_w f_\theta(\tilde{x}) \\
&= \sigma(Wx)^T \sigma(W\tilde{x}) + \sum_{i=1}^m a_i^2 \sigma'(w_i^T x)^T \sigma'(w_i^T \tilde{x}) x^T \tilde{x}
\end{aligned}
$$
and thus for the data matrices
$$
\begin{aligned}
k(X, \tilde{X}) &= \sigma(WX)^T \sigma(W\tilde{X}) + X^T \tilde{X} \odot \sum_{i=1}^m a_i^2 \sigma'(w_i^T X)^T \sigma'(w_i^T \tilde{X}) \\
&= \sigma(WX)^T \sigma(W\tilde{X}) +  \sigma'(WX)^T \text{diag}(a)^2 \sigma'(W\tilde{X})\odot  X^T \tilde{X},
\end{aligned}
$$
where $\odot$ is the Hadamard product and
$$\text{diag}(a) = \begin{pmatrix} a_1 & & 0 \\ & \ddots & \\ 0 & & a_m \end{pmatrix}.$$

(ii) By the Gaussian equivalence principle from **Section 7.6**, we can replace non-linear random matrices like $\sigma(WX)$ by linear+noise random matrices $\alpha WX + \beta Z$.

We will not go more into those calculations. For details one should see **[AP20]**.

## 8.5. Concentration of the neural tangent kernel

We still should get a better understanding of the claimed asymptotic properties of the neural tangent kernel (NTK). Consider for simplicity the model
$$f_{\theta}(x) = \frac{1}{\sqrt{m}} a^{T} \sigma(W x) = \frac{1}{\sqrt{m}} \sum_{i=1}^{m} a_{i} \sigma(w_{i}^{T} x),$$
where we only optimize over $W$ and keep $a$ fixed, thus $\theta = \{W\}$. For asymptotic statements we have to be precise about our normalizations; we choose:
* $a_{i}$ uniformly on $\{-1, +1\}$, so that $\|a\| \sim 1$,
* $W$ as a standard Gaussian random matrix, i.e. each $w_{i} \sim N(0, I_{p})$, and
* $\|x\| = 1$.
The kernel $k$ is then given by$$k(x, \tilde{x}) = \nabla_{w} f_{\theta}(x)^{T} \cdot \nabla_{w} f_{\theta}(\tilde{x}) = x^{T} \tilde{x} \cdot \frac{1}{m} \sum_{i=1}^{m} \underbrace{a_{i}^{2}}_{=1} \sigma'(w_{i}^{T} x) \sigma'(w_{i}^{T} \tilde{x}).$$Note that $\sigma'(w_{i}^{T} x) \sigma'(w_{i}^{T} \tilde{x})$ is independent for different $i$ and has the same distribution for each $i$, thus they are i.i.d.. In particular, for $v \sim N(0, I_{p})$, we have
$$\frac{1}{m} \sum_{i=1}^{m} \underbrace{a_{i}^{2}}_{=1} \sigma'(w_{i}^{T} x) \sigma'(w_{i}^{T} \tilde{x}) \xrightarrow{m \to \infty} E_{v} [\sigma'(v^{T} x) \sigma'(v^{T} \tilde{x})]$$
by the law of large numbers. Define the limiting NTK $k^{*}$ by
$$k^{*}(x, \tilde{x}) = E_{v} [\sigma'(v^{T} x) \sigma'(v^{T} \tilde{x})] x^{T} \tilde{x}, \tag{5}$$
then by the above and by concentration, with high probability for sufficiently large $m$ we have$$|k(x, \tilde{x}) - k^{*}(x, \tilde{x})| < \epsilon.$$

>[!example] Example. 
>In some cases one can also calculate the limiting NTK $k^{*}$. Let us consider $\sigma(\cdot) = \text{ReLU}=\max(0,\cdot)$, then$$\sigma'(t) = \begin{cases} 1, & t > 0, \\ 0, & t < 0 \end{cases}.$$
>Let $\|x\| = 1 = \|\tilde{x}\|$. What is $E_{v} [1_{\{v^{T} x > 0\}} \cdot 1_{\{v^{T} \tilde{x} > 0\}}]$? Note that $t_{1} = v^{T} x$ and $t_{2} = v^{T} \tilde{x}$ are two Gaussian vectors with covariance$$\Sigma = \begin{pmatrix} 1 & \alpha \\ \alpha & 1 \end{pmatrix} \quad \text{where} \quad \alpha = x^{T} \tilde{x},$$thus$$\Sigma^{-1} = \frac{1}{1 - \alpha^{2}} \begin{pmatrix} 1 & -\alpha \\ -\alpha & 1 \end{pmatrix}$$and we have the joint density$$\begin{aligned} \psi(t_{1}, t_{2}) &= \frac{1}{2\pi (\det(\Sigma))^{\frac{1}{2}}} \exp \left( -\frac{1}{2} \left\langle \begin{pmatrix} t_{1} \\ t_{2} \end{pmatrix}, \Sigma^{-1} \begin{pmatrix} t_{1} \\ t_{2} \end{pmatrix} \right\rangle \right) \\ &= \frac{1}{2\pi \sqrt{1 - \alpha^{2}}} \exp \left( -\frac{1}{2} \frac{t_{1}^{2} + t_{2}^{2} - 2\alpha t_{1} t_{2}}{1 - \alpha^{2}} \right). \end{aligned}$$
>So we have to calculate$$E_{v} [1_{\{v^{T} x > 0\}} \cdot 1_{\{v^{T} \tilde{x} > 0\}}] = \int_{0}^{\infty} \int_{0}^{\infty} \psi(t_{1}, t_{2}) dt_{1} dt_{2}.$$
This can be done by manipulating the integrals, but we prefer here another approach
without explicit integration: This is the same problem if we restrict to the plane spanned
by $x$ and $\tilde{x}$, so we can assume that $x, \tilde{x}, v \in \mathbb{R}^2$ and $v \sim N(0, I_2)$. Since $v^T x > 0$ if and
only if $\frac{v^T}{\|v\|} \cdot x > 0$, we can replace $v \sim N(0, I_2)$ by $\frac{v^T}{\|v\|}$ from the uniform distribution on
$S^1 = \{\exp(\text{i}\varphi) \mid 0 \le \varphi \le 2\pi\}$. So what we want is, for given $x, \tilde{x} \in \mathbb{R}^2$:$$P \{\exp(\text{i}\varphi) : \langle \exp(\text{i}\varphi), x \rangle \ge 0 \text{ and } \langle \exp(\text{i}\varphi), \tilde{x} \rangle \ge 0\}.$$
>So$$k^*(x, \tilde{x}) = x^T \tilde{x} \frac{\pi - \arccos(x^T \tilde{x})}{2\pi}.$$


## 8.6 Evolution of the neural tangent kernel under training

Now consider the evolution under training. First we show that the weight vectors do not change much. Recall that we have (put $\eta = 1$)

$$
\frac{d\theta(t)}{dt}
= - \nabla_\theta \mathcal{L}(\theta(t))
= - \sum_{k=1}^{n} \big(f_t(\hat{x}_k) - \hat{y}_k\big)\,\nabla_\theta f_t(\hat{x}_k).
$$

For $\theta = w_i$ we get

$$
\nabla_{w_i} f_t(\hat{x}_k)
= a_i \sigma'(w_i^T \hat{x}_k)\,\hat{x}_k \cdot \frac{1}{\sqrt{m}},
$$

thus

$$
\frac{dw_i}{dt}
= - \frac{1}{\sqrt{m}} \sum_{k=1}^{n}
\big(f_t(\hat{x}_k) - \hat{y}_k\big)\,
a_i \sigma'(w_i^T \hat{x}_k)\,\hat{x}_k.
$$

Now consider the evolution of the weights: since $w_i^T \hat{x}_k$ is a Gaussian variable of variance $\|\hat{x}_k\| = 1$, we have

$$
\begin{aligned}
\|w_i(t) - w_i(0)\|_2
&= \left\| \int_0^t \frac{dw_i(\tau)}{d\tau}\, d\tau \right\|_2 \\
&= \left\| \int_0^t \frac{1}{\sqrt{m}} \sum_{k=1}^{n}
\big(f_t(\hat{x}_k) - \hat{y}_k\big)\,
a_i \sigma'(w_i^T \hat{x}_k)\,\hat{x}_k \, d\tau \right\|_2 \\
&\le \frac{1}{\sqrt{m}} \sum_{k=1}^{n} \int_0^t [\text{order }1]\, d\tau \\
&\sim \text{order } \frac{t \cdot n}{\sqrt{m}},
\end{aligned}
$$

which is small if $m \to \infty$ for fixed $t$ and $n$.


Now consider the change in the kernel: since $x^T \tilde{x} \le \|x\|\cdot\|\tilde{x}\| = 1$, we have

$$
\begin{aligned}
|k_t(x,\tilde{x}) - k_0(x,\tilde{x})|
&= \left| x^T \tilde{x} \frac{1}{m} \sum_{i=1}^{m}
\Big(\sigma'(w_i(t)^T x)\sigma'(w_i(t)^T \tilde{x})
- \sigma'(w_i(0)^T x)\sigma'(w_i(0)^T \tilde{x}) \Big) \right| \\
&\le \frac{1}{m} \sum_{i=1}^{m}
\Big|
\sigma'(w_i(t)^T x)\sigma'(w_i(t)^T \tilde{x})
- \sigma'(w_i(t)^T x)\sigma'(w_i(0)^T \tilde{x})
\\
&\quad + \sigma'(w_i(t)^T x)\sigma'(w_i(0)^T \tilde{x})
- \sigma'(w_i(0)^T x)\sigma'(w_i(0)^T \tilde{x})
\Big| \\
&\le \frac{1}{m} \sum_{i=1}^{m}
\Big(
\max|\sigma'(w_i(t)^T x)|
\cdot |\sigma'(w_i(t)^T \tilde{x}) - \sigma'(w_i(0)^T \tilde{x})|
\\
&\quad +
|\sigma'(w_i(t)^T x) - \sigma'(w_i(0)^T x)|
\cdot \max|\sigma'(w_i(0)^T \tilde{x})|
\Big) \\
&\le \frac{1}{m} \sum_{i=1}^{m}
\Big(
\max|\sigma'(\cdot)| \cdot \max|\sigma''(\cdot)|
\cdot \|(w_i(t)-w_i(0))^T \tilde{x}\|
\\
&\quad +
\max|\sigma''(\cdot)| \cdot \|(w_i(t)-w_i(0))^T x\|
\cdot \max|\sigma'(\cdot)|
\Big).
\end{aligned}
$$

Since

$$
\|(w_i(t)^T - w_i(0)^T)\tilde{x}\|
\sim \frac{t \cdot n}{\sqrt{m}} \cdot \|\tilde{x}\|
= \frac{t \cdot n}{\sqrt{m}},
$$

we have that

$$
|k_t(x,\tilde{x}) - k_0(x,\tilde{x})|
= \text{order } \frac{t \cdot n}{\sqrt{m}} \to 0.
$$
If we put $k_{ij} = k(\hat{x}_i,\hat{x}_j)$ such that $k = (k_{ij})$ is an $n \times n$ matrix, then also in operator norm
$$
\begin{aligned}
\|k(t) - k(0)\|
&\le \|k(t) - k(0)\|_F \\
&= \left( \sum_{i,j=1}^{n} |k_{ij}(t) - k_{ij}(0)|^2 \right)^{1/2} \\
&\le \left( \sum_{i,j=1}^{n}
\left[\text{order } \frac{t \cdot n}{\sqrt{m}}\right]^2 \right)^{1/2} \\
&\sim \frac{t n^2}{\sqrt{m}} \to 0.
\end{aligned}
$$
for fixed $t$ and $n$.


## 8.7. Boundedness away from zero of the neural tangent kernel

In order to see that the limiting NTK $k^*$ according to Equation (5) (and thus also its approximations in high dimensions) has only positive eigenvalues which are bounded away from zero, i.e., $k^* \ge \delta \cdot I$, one should note:

- $k^*$ is essentially diagonal, $k^*(x, \tilde{x}) \approx 0$ since two vectors in high dimension are with high probability almost orthogonal, and

- $k^*(x, x) = \mathbb{E}_v \left[ \sigma'(v^T x)\sigma'(v^T x) \right] \ge \delta$ in general.

As an example for the latter statement, let us check this concretely for the case $\sigma = \mathrm{ReLU}$; then

$$
k^*(x, x) = \frac{\pi - \arccos(1)}{2\pi}
= \frac{\pi - \frac{\pi}{2}}{2\pi}
= \frac{1}{4}.
$$
