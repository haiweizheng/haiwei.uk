---
source: "[[High-Dimensional Analysis- Random Matrices and Machine Learning.pdf]]"
tags:
  - 随机矩阵与机器学习_random-matrices-and-machine-learning
  - 集中不等式_concentration-inequality
  - Lipschitz函数_lipschitz-function
  - 高斯随机向量_gaussian-random-vector
---

>本章的目标是借助 Lipschitz 函数来构造非线性函数的集中现象

>*A random variable that depends (in a ‘smooth’ way) on the influence of many independent variables (but not too much on any of them) is essentially constant.* 
>*如果一个随机变量“平滑地”依赖于许多独立的变量，且没有任何一个单变量能起决定性作用，那么这个随机变量的值基本上就是一个常数（即高度集中）*
>-Michel Talagrand

> [!example] 
>$$f(t_1, \dots, t_p) = \frac{1}{p}(t_1^2 + \dots + t_p^2) \quad \text{and} \quad g(t_1, \dots, t_p) = t_1^2$$
> satisfy $E[f(x)] = 1 = E[g(x)]$; $f$ concentrates about 1 for large $p$; but $g$ has, independent of $p$, always the same spread-out distribution.

>[!definition] 定义 L-Lipschitz Function
>A function $f : \mathbb{R}^p \to \mathbb{R}^m$ is $L$-Lipschitz, if
>$$\|f(x) - f(y)\| \leq L\|x - y\| \quad \text{for all} \quad x, y \in \mathbb{R}^p.$$

>[!example]- Example. 
>1. a smooth (i.e., differentiable) function $f$ is Lipschitz iff $\nabla f(x) = (\frac{\partial f(x)}{\partial t_1}, \dots, \frac{\partial f(x)}{\partial t_p})^\top$ is bounded.
>2. componentwise application of Lipschitz functions is Lipschitz: if $f_1, \dots, f_p : \mathbb{R} \to \mathbb{R}$ are $L$-Lipschitz, then
>$$f : \mathbb{R}^p \to \mathbb{R}^p \quad   f(t_1, \dots, t_p) = (f_1(t_1), \dots, f_p(t_p))$$is also $L$-Lipschitz: with $x = (t_1, \dots, t_p)$ and $y = (s_1, \dots, s_p)$ we have
>$$\begin{aligned}
\|f(x) - f(y)\|^2 &= \left\| \begin{pmatrix} f_1(t_1) - f_1(s_1) \\ \vdots \\ f_p(t_p) - f_p(s_p) \end{pmatrix} \right\|^2 \\
&= \underbrace{|f_1(t_1) - f_1(s_1)|^2}_{\leq L^2 |t_1 - s_1|^2} + \dots + \underbrace{|f_p(t_p) - f_p(s_p)|^2}_{\leq L^2 |t_p - s_p|^2} \\
&\leq L^2 \left( |t_1 - s_1|^2 + \dots + |t_p - s_p|^2 \right) \\
&= L^2 \|x - y\|^2\end{aligned}$$
>and thus $\|f(x) - f(y)\| \leq L\|x - y\|$.
>1. composition of Lipschitz functions is Lipschitz:$$
|g(f(x)) - g(f(y))| \leq L_2 \underbrace{\|f(x) - f(y)\|}_{\leq L_1 \|x - y\|} \leq L_1 \cdot L_2 \|x - y\|,$$

>[!theo] Theorem. 3.2 Gaussian concentration for Lipschitz functions
>Let $f : \mathbb{R}^p \to \mathbb{R}$ be an $L$-Lipschitz function. For $x \sim N(0, I_p)$ for any $\alpha \geq 0$: 
>$$ \mathbb{P}\left\{ |f(x) - \mathbb{E}[f(x)]| \geq \alpha \right\} \leq 2 \exp\left( -\frac{\alpha^2}{2L^2} \right). $$^fx0z71

***Proof.***
***Step 1***
Assume that $g=f-Ef(x)$,
$$|g(x)-g(y)|=|[f(x)-Ef(x)]-[f(y)-Ef(y)]|\leq L|x-y|$$
thus it suffices to prove
$$\mathrm{P} \{ x \in \mathbb{R}^p : |g(x)| \geq \alpha \} \leq 2 \exp \left( -\frac{\alpha^2}{2L^2} \right).$$
re-denote it by $f(x)$.
Since
$$
\{  |f(x)| \geq \alpha \} = \{  f(x) \geq \alpha \} \cup \{  -f(x) \geq \alpha \}
$$
it suffices to prove
$$
\mathrm{P} \{  f(x) \geq \alpha \} \leq \exp \left( -\frac{\alpha^2}{2L^2} \right).
$$
*“为了能用一个超炫酷的技巧来做证明，我们先做个合理的平滑假设，并且接受最终算出来的常数稍稍差那么一点点。”*
Then $\forall \lambda>0$ we have
$$ \begin{aligned} \mathbb{P}\left\{x \in \mathbb{R}^p : f(x) \geq \alpha\right\} &= \mathbb{P}\left\{x \in \mathbb{R}^p : \exp(\lambda f(x)) \geq \exp(\lambda \alpha)\right\} \\ &\leq \frac{\mathbb{E}\left[\exp(\lambda f(x))\right]}{\exp(\lambda \alpha)}. \end{aligned}$$

***Step 2***
Then consider how to deal with the inequality above. Set 
$$\begin{aligned} \mathbb{E}\left[\exp(\lambda f(x))\right] &=  \underbrace{\mathbb{E}\left[\exp(\lambda f(x))\right] \cdot 1}_{\mathbf{Jensen}\ \mathbb{E}\left[\exp(-\lambda f)\right]\geq\left[\exp(-\lambda \mathbb{E}f)\right]\geq 1} \\ &\leq \mathbb{E}\left[\exp(\lambda f(x))\right] \cdot \mathbb{E}\left[\exp(-\lambda f(y))\right] \\ &= \mathbb{E}\left[\exp(\lambda f(x)) \cdot \exp(-\lambda f(y))\right] \\ &= \mathbb{E}\left[\exp\left(\lambda\left(f(x) - f(y)\right)\right)\right], \end{aligned}$$ where $y$ i.i.d with $x$, i.e., $\begin{pmatrix} x \\ y \end{pmatrix} \in \mathbb{R}^{2p}$ is a $2p$-dimensional Gaussian random vector. Why do we introduce $y$ ? Because we can write $$f(x) - f(y) = \int_0^{\frac{\pi}{2}} \frac{\partial}{\partial\theta} f\big(\underbrace{\cos(\theta)y + \sin(\theta)x}_{=:x(\theta)}\big)\, \mathrm{d}\theta = \int_0^{\frac{\pi}{2}} \nabla f(x(\theta)) \cdot x'(\theta)\, \mathrm{d}\theta$$ where 
$$\begin{pmatrix} x(\theta) \\ y(\theta) \end{pmatrix} = \begin{pmatrix} \sin\theta & \cos\theta \\ \cos\theta & -\sin\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}$$
is $2p$-dimensional Gaussian random vector too.

**Example.** 
$$\frac{1}{\sqrt{2}}(x + y) \sim N(0, I_p)$$$$ \frac{1}{\sqrt{2}}(x - y) \sim N(0, I_p) $$ are independent. 
**Proof of Lemma 3.4.** 
By assumption, $(x, y) \in \mathbb{R}^{2p}$ has density $$ \psi(x, y) = \frac{1}{(2\pi)^{\frac{p}{2}}} \exp\left(-\frac{1}{2}\|x\|^2\right) \cdot \frac{1}{(2\pi)^{\frac{p}{2}}} \exp\left(-\frac{1}{2}\|y\|^2\right) $$ $$ = \frac{1}{(2\pi)^p} \exp\left(-\frac{1}{2}\left(\|x\|^2 + \|y\|^2\right)\right) $$ $$ = \frac{1}{(2\pi)^p} \exp\left(-\frac{1}{2}\left\|\begin{pmatrix} x \\ y \end{pmatrix}\right\|^2\right), $$ i.e. $ \begin{pmatrix} x \\ y \end{pmatrix} \sim N(0, I_{2p}) $. The replacement $ \begin{pmatrix} x \\ y \end{pmatrix} \rightsquigarrow \begin{pmatrix} x(\theta) \\ y(\theta) \end{pmatrix} $ is a unitary transformation in $ \mathbb{R}^{2p} $: with $$ U = \begin{pmatrix} \cos(\theta) & \sin(\theta) \\ -\sin(\theta) & \cos(\theta) \end{pmatrix} \in \mathbb{R}^{2 \times 2} $$ as well as $$ x(\theta) = (t_1(\theta), \dots, t_p(\theta)) $$ and $$ y(\theta) = (s_1(\theta), \dots, s_p(\theta)), $$ we have $$\begin{pmatrix} t_1(\theta) \\ s_1(\theta) \\ t_2(\theta) \\ s_2(\theta) \\ \vdots \\ t_p(\theta) \\ s_p(\theta) \end{pmatrix} = \underbrace{ \begin{pmatrix} U & & & 0 \\ & U & & \\ & & \ddots & \\ 0 & & & U \end{pmatrix} }_{=: V} \cdot \begin{pmatrix} t_1 \\ s_1 \\ t_2 \\ s_2 \\ \vdots \\ t_p \\ s_p \end{pmatrix}$$ and thus by the behaviour of Gaussian random vectors under linear transformations (see Assignment 2, Exercise 4) $$\begin{pmatrix} t_1(\theta) \\ s_1(\theta) \\ t_2(\theta) \\ s_2(\theta) \\ \vdots \\ t_p(\theta) \\ s_p(\theta) \end{pmatrix} \sim N(0, \underbrace{V I_{2p} V^T}_{=I_{2p}}),$$ i.e. for each $\theta$ we again have a standard Gaussian random vector in $\mathbb{R}^{2p}$ $\boxed{\mathbf{QED}}$


### 3.3. Generalizations of concentration inequalities
The concentration inequalities are of the form: $\text{input} \xrightarrow{\text{function}} \text{output}$, where:
* the input are independent variables, which up to now were Gaussian;
* the function is linear or Lipschitz;
* the output concentrates like a Gaussian distribution $\sim \exp(-\alpha^2 \cdot c)$.
Note that in the linear situation the assumption on Gaussianity can be generalized: the main ingredient was the control of $E[\exp(\lambda g(t))]$.
