---
source: "[[High-Dimensional Analysis- Random Matrices and Machine Learning.pdf]]"
tags:
  - 随机矩阵与机器学习_random-matrices-and-machine-learning
  - 高斯随机向量_gaussian-random-vector
  - 集中不等式_concentration-inequality
  - Chebyshev不等式_chebyshev-inequality
  - Bernstein不等式_bernstein-inequality
---

### 2.1. Gaussian random vectors
>[!definition] 定义 2.1 Standard Gaussian Vector $x = (t_1, \dots, t_p) \in \mathbb{R}^p$ 
$t_1, \dots, t_p$ are  $i.i.d. \sim N(0, 1)$
We denote this by $x \sim N(0, I_p)$, where  $I_p$ the $p \times p$ identity matrix, is the *matrix of covariances*. 

### 2.2. Concentration of the norm
>[!remark]
>我们会发现 $||x||_2^2$ 是卡方分布且 $||x||_2^2\xrightarrow{N\rightarrow \infty}E||x||_2^2= p$ ^lrt756

>[!theo] Theorem.  2.2  
>$x \sim N(0, I_p)$. Then, for $0 \leq \varepsilon \leq \sqrt{p}$,$$\mathrm{P} \left\{ x \in \mathbb{R}^p : | \|x\| - \sqrt{p} | \geq \varepsilon \right\} \leq 2 \exp \left( -\frac{\varepsilon^2}{16} \right),$$

Thus
$$\mathrm{P} \left\{  | \|\frac{1}{\sqrt{p}}x\| - 1 | \geq \varepsilon \right\}=\mathrm{P} \left\{  | \|x\| - \sqrt p | \geq \sqrt p\cdot\varepsilon \right\} \leq 2 \exp \left( -\frac{p\varepsilon^2}{16} \right).$$
As we can know
$$
\begin{aligned}
\mathrm{P} \left\{ \left| \|x\| - \sqrt{p} \right| \geq \varepsilon \right\} &\leq \mathrm{P} \left\{ \left| \|x\| - \sqrt{p} \right| \cdot (\|x\| + \sqrt{p}) \geq \varepsilon \sqrt{p} \right\} \\
&= \mathrm{P} \left\{ \left| \|x\|^2 - p \right| \geq \varepsilon \sqrt{p} \right\}.
\end{aligned}
$$
***Step 1*** A naive try
A simple try by Chi-Square Distribution:
$$
\begin{aligned}
\mathbb{P}(|\|x\|^2 - p| \geq \varepsilon \sqrt{p}) & \leq \frac{1}{(\varepsilon \sqrt{p})^2} \mathbb{E}(|\|x\|^2 - p|)^2 \\
&= \frac{1}{\varepsilon^2 p} p \cdot \mathbb{E}(x_1^2 - 1)^2 \\
&= \frac{1}{\varepsilon^2} \mathbb{E}(x_1^4 - 2x_1^2 + 1) \\
&= \frac{1}{\varepsilon^2} (3 - 2 + 1) = \frac{2}{\varepsilon^2}
\end{aligned}
$$
It is NOT enough for our need!

***Step 2*** A further try

- ***Step 2.1***
A way to solve question of this type. Assume that 
* $y = (s_1, \dots, s_p)$ has ==independent== coordinates, i.e. $\psi(s_1, \dots, s_p) = \psi_1(s_1) \cdots \psi_p(s_p)$;
* $f$ is a ==sum== of functions of the coordinates, i.e. $f(s_1, \dots, s_p) = f_1(s_1) + \cdots + f_p(s_p)$.
When $\forall \lambda > 0$
$$\begin{aligned}
\mathrm{P} \{f(y) \geq \alpha\} &= \mathrm{P} \{\exp(\lambda f(y)) \geq \exp(\lambda \alpha)\} \leq \frac{E[\exp(\lambda f(y))]}{\exp(\lambda \alpha)}\\
\rightarrow &\leq \inf_{\lambda > 0} \exp(-\lambda \alpha) E[\exp(\lambda f(y))].
\end{aligned}$$
- ***Step 2.2***
Set $f=||x||^2-p=\sum(t^2_i-1)$
By ***Step 2.1*** we have
$$\begin{aligned} \mathrm{P} \{f(x) \geq \alpha\} &\leq \inf_{ \lambda >0} \exp(-\lambda \alpha) E [\exp(\lambda (t^2_1-1))]^p  \end{aligned}$$
As we know, the moment generation function of $t \sim N(0,1)$ is
$$
\begin{aligned}
E(\exp(\lambda t))
&=\frac{1}{\sqrt{2\pi}} \int_{\mathbb{R}} \exp(\lambda t) \exp \left( -\frac{t^2}{2} \right) \mathrm{d}t \\
&= \frac{1}{\sqrt{2\pi}} \int_{\mathbb{R}} \exp \left( -\frac{(t - \lambda)^2}{2} \right) \exp \left( \frac{\lambda^2}{2} \right) \mathrm{d}t \\
&= \exp \left( \frac{\lambda^2}{2} \right) \underbrace{\frac{1}{\sqrt{2\pi}} \int_{\mathbb{R}} \exp \left( -\frac{(t - \lambda)^2}{2} \right) \mathrm{d}t}_{=1} \\
&= \exp \left( \frac{\lambda^2}{2} \right).
\end{aligned}
$$
Then we can get
$$\begin{aligned}  E [\exp(\lambda(t^2 - 1))]  &= \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{+\infty} \exp(\lambda(t^2 - 1)) \exp \left( -\frac{t^2}{2} \right) \mathrm{d}t \\ &= \frac{1}{\sqrt{2\pi}} \exp(-\lambda) \int_{-\infty}^{+\infty} \exp \left( -\frac{t^2}{2}(1 - 2\lambda) \right) \mathrm{d}t \\ 
(s = t\sqrt{1 − 2λ})&= \frac{1}{\sqrt{2\pi}} \exp(-\lambda) \underbrace{\int_{-\infty}^{+\infty} \exp \left( -\frac{s^2}{2} \right) \mathrm{d}s}_{=\sqrt{2\pi}} \cdot \frac{1}{\sqrt{1 - 2\lambda}} \\ 
&= \frac{\exp(-\lambda)}{\sqrt{1 - 2\lambda}}. \end{aligned}$$
- $\lambda < \frac12$ holds.
- $\lambda \geq \frac12$ does not hold.
Thus
$$\begin{aligned} \mathrm{P} \{f(x) \geq \alpha\} &\leq \inf_{0 < \lambda < \frac{1}{2}} \exp(-\lambda \alpha) E [\exp(\lambda (t^2_1-1))]^p \\ &= \inf_{0 < \lambda < \frac{1}{2}} \exp(-\lambda \alpha) \frac{\exp(-\lambda p)}{(1 - 2\lambda)^{\frac{p}{2}}}, \end{aligned}$$

***Step 3*** 
>[!lemma] Lemma. 
>**Lemma 2.5.** Let $t \sim N(0,1)$ be a one-dimensional standard Gaussian random variable. Then$$E [\exp(\lambda(t^2 - 1))] \leq \exp(4\lambda^2) \quad \text{for all} \quad |\lambda| \leq \frac{1}{4}.$$

***Proof.***
*Way 1* Analytic
$$-\lambda - \frac{1}{2} \ln(1-2\lambda) \leq 4\lambda^2$$
$$f(\lambda) = \lambda + \frac{1}{2} \ln(1-2\lambda) + 4\lambda^2 \geq 0 \quad (-\frac14\leq\lambda\leq \frac14)$$

***Step 4***
$$\begin{aligned} \mathbb{P} \{| \|x\|^2 - p | \geq \varepsilon \sqrt{p}\} 
&= \mathbb{P} \{f(y) \geq \varepsilon \sqrt{p}\} + \mathbb{P} \{f(y) \leq -\varepsilon \sqrt{p}\} \\ &= 2 \mathbb{P} \{f(y) \geq \varepsilon \sqrt{p}\} \\ &\leq 2 \inf_{0 < \lambda < \frac{1}{2}} \exp(-\lambda \varepsilon \sqrt{p}) \underbrace{\mathbb{E} \left[ \exp(\lambda(t^2 - 1)) \right]^p}_{\leq \exp(4\lambda^2) \text{ for } \lambda \leq \frac{1}{4}} \\ &\leq 2 \inf_{0 < \lambda \leq \frac{1}{4}} \exp(-\lambda \varepsilon \sqrt{p} + 4\lambda^2 p). \end{aligned}$$
Take min-value in $\lambda =\frac{\varepsilon}{8\sqrt p}\leq \frac14\ (0 \leq \varepsilon \leq \sqrt{p})$, the proposition holds.