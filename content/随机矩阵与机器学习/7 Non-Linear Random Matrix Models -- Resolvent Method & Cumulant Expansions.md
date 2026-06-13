---
source: "[[High-Dimensional Analysis- Random Matrices and Machine Learning.pdf]]"
tags:
  - 随机矩阵与机器学习_random-matrices-and-machine-learning
  - 非线性随机矩阵_non-linear-random-matrix
  - 预解式方法_resolvent-method
  - 累积量展开_cumulant-expansion
---

## 7.1. Distribution of the random features model

We consider our "random features" model and we want to understand the distribution of the features $f$ – in particular the eigenvalues of their covariance estimator $FF^T$.
![[7.1.jpg|469]]

We will try to understand the statement as well as the idea and the tools of the proof of the following theorem. For the proof we will follow the ideas from {[[High-Dimensional Analysis- Random Matrices and Machine Learning.pdf#page=131&offset=71,688,0|PS21]]}.

>[!theo] Theorem. 7.1 (Pennington and Worah {[[High-Dimensional Analysis- Random Matrices and Machine Learning.pdf#page=131&offset=71,617,0|PW17]]}, Benigni and Péché {[[High-Dimensional Analysis- Random Matrices and Machine Learning.pdf#page=130&offset=71,554,0|BP21]]}). 
>Let $X \in \mathbb{R}^{p \times n}$ and $W \in \mathbb{R}^{m \times p}$ be standard Gaussian random matrices and $\sigma : \mathbb{R} \to \mathbb{R}$ be a sufficiently nice function (e.g. analytical function), which is centered with respect to the Gaussian distribution, i.e.$$\int_{\mathbb{R}} \sigma(t) \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{t^2}{2}\right) dt = 0.$$
>We put $$F := \sigma\left(\frac{1}{\sqrt{p}} W_{m\times p}X_{p\times n}\right) \in \mathbb{R}^{m \times n} \quad \text{and} \quad M := \frac{1}{n} FF^T \in \mathbb{R}^{m \times m}.$$
>Then, in the limit $p, n, m \to \infty$ such that $\frac{m}{n} \to \gamma$ (终数据复杂度) and $\frac{p}{m} \to \tilde{\gamma}$ (数据原始比值), the Stieltjes transform of $M$ converges to a limit$$S(z) = \lim_{\substack{\frac{m}{n} \to \gamma \\ \frac{p}{m} \to \tilde{\gamma}}} E\left[\frac 1m\text{tr}\left(\frac{1}{n} FF^T - zI_m\right)^{-1}\right]$$and this $S(z)$ satisfies the following quartic equation:$$\begin{aligned}
1 + zS(z) = \theta_1 S(z)\Big(1 - \gamma(1 + zS(z))\Big) &- \frac{\theta_2}{\tilde{\gamma}}(1 + zS(z))\Big(1 - \gamma(1 + zS(z))\Big) \\
&+ \frac{\theta_2(\theta_1 - \theta_2)}{\tilde{\gamma}} S(z)^2\Big(1 - \gamma(1 + zS(z))\Big)^2,
\end{aligned}$$
>where$$\theta_1 = \theta_1(\sigma) := \int_{\mathbb{R}} \sigma(t)^2 \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{t^2}{2}\right) dt$$
>$$\theta_2 = \theta_2(\sigma) := \left(\int_{\mathbb{R}} \sigma'(t) \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{t^2}{2}\right) dt\right)^2.$$

>[!remark]
>1. If $\theta_2 = 0$ and $\theta_1 = 1$, this reduces to$$1 + zS(z) = S(z)\Big(1 - \gamma(1 + zS(z))\Big),$$which is the equation for Marchenko-Pastur, compare our proof of Theorem 4.2. Thus in this case $F \overset{d}{=} Z$, where $Z \in \mathbb{R}^{m \times n}$ is a standard Gaussian random matrix.
>2. If $\theta_1 = \theta_2$ and $\theta_2 = 1$, this reduces to the cubic equation$$1 + zS(z) = S(z)\Big(1 - \gamma(1 + zS(z))\Big)\left(1 - \frac{1}{\tilde{\gamma}}(1 + zS(z))\right).$$
>Note that $\theta_1 = \theta_2$ is given for $\sigma(t) = \sqrt{\theta_2} \cdot t$ (Actually, $θ_1 = θ_2$ can only happen for linear $σ$, see Assignment 5.) thus this special non-linear case corresponds to the linear situation $F \overset{d}{=} \frac{\sqrt{\theta_2}}{\sqrt{p}} WX$.
>3. It is not obvious from the general form of the equation, but one can show (and we will come back to this in Section 7.6) that in general one actually has the "independent" combination of those two special cases, i.e.$$F \overset{d}{=} \frac{\sqrt{\theta_2}}{\sqrt{p}} WX + \sqrt{\theta_1 - \theta_2} Z.$$
>Thus the effect of the non-linearity is to produce some additional noise.

## 7.2. Proof of Marchenko-Pastur law via Stein's identity
The new way to prove MP Law allows us then to also deal with products and non-linearities.

***Proof.***
***Step 1***
Let $X = (x_{ij}) \in \mathbb{R}^{p \times n}$ be our standard Gaussian random matrix, then we want to calculate, in the limit $p, n \to \infty$ with $\frac{p}{n} \to \gamma$, the Stieltjes transform of $A = \frac{1}{n} XX^T$, given by

$$
S(z) = E \left[\frac 1p \text{tr}  \underbrace{(A - zI_p)^{-1}}_{\substack{R(z) := (A - zI_p)^{-1}\\ \text{Resolvent}}}  \right] = \underbrace{E \left[ \frac 1p \text{tr}  \left( \frac{1}{n} XX^T - zI_p \right)^{-1}  \right]}_{ E[\frac 1p\text{tr}(R(z))]}.
$$

We have $(A - zI_p)R(z) = I_p$, i.e. $I_p + zR(z) = AR(z)$. Applying $E[\frac 1p \text{tr}(\cdot)]$ to both sides, we get
$$\begin{aligned}  
1 + zS(z) &= E[\frac 1p\text{tr}(AR(z))] = \frac{1}{np} E[\text{Tr}(XX^T R(z))] \\
&= \frac{1}{np} \sum_{\substack{i=1,\dots,p \\ j=1,\dots,n}} E\left[ x_{ij} [X^T R(z)]_{ji} \right],
\end{aligned}$$
where $X^T R(z)$ is a function of all $x_{kl}$. Thus we need a formula to deal with such expectations. We will first present this "Stein's identity" and then later continue with our proof. $\square$

***Step 2***
>[!lemma] Lemma. 7.2 Stein's Identity
>- $t_1, \dots, t_k \ \text{i.i.d.}\sim N(0,1)$
>- $h : \mathbb{R}^k \to \mathbb{R}$ a nice function (e.g. $C^\infty$). 
>
>Then $\forall\ i = 1, \dots, k$:
>$$E[t_i h(t_1, \dots, t_k)] = E[\underbrace{\partial_i h(t_1, \dots, t_k)}_{\partial_i = \frac{\partial}{\partial t_i}}],$$
>$$E[X h(X)] = \sigma^2 E[h'(X)]\quad X \sim N(0, \sigma^2)$$

***Proof.***
The main argument happens for $k = 1$; it is just partial integration:
$$
\begin{aligned}
E[th(t)] &= \frac{1}{\sqrt{2\pi}} \int t h(t) \exp\left(-\frac{t^2}{2}\right) dt \\
&= \frac{1}{\sqrt{2\pi}} \int h(t) \cdot \underbrace{t \exp\left(-\frac{t^2}{2}\right)}_{= \left(-\exp\left(-\frac{t^2}{2}\right)\right)'} dt \\
&= \frac{1}{\sqrt{2\pi}} \int h'(t) \exp\left(-\frac{t^2}{2}\right) dt - \frac{1}{\sqrt{2\pi}} \left[ h(t) \exp\left(-\frac{t^2}{2}\right) \right]_{-\infty}^{+\infty} \\
&= \frac{1}{\sqrt{2\pi}} \int h'(t) \exp\left(-\frac{t^2}{2}\right) dt \\
&= E[h'(t)]
\end{aligned}
$$
since the last summand in the partial integration is zero by the assumption on $h$. For general $k$, just do partial integration for the $i$-th coordinate. $\square$

***Step 3***
In our setting, this now gives
$$
\begin{aligned}
E[x_{ij} [X^T R(z)]_{ji}] &= E[\partial_{ij} [X^T R(z)]_{ji}] \\
&= E\left[ \partial_{ij} \sum_{k=1}^p [X^T]_{jk} [R(z)]_{ki} \right] \\
&= E\left[ \partial_{ij} \sum_{k=1}^p x_{kj} [R(z)]_{ki} \right] \\
&= E\left[ \sum_{k=1}^p \frac{\partial x_{kj}}{\partial x_{ij}} \cdot [R(z)]_{ki} + x_{kj} \frac{\partial [R(z)]_{ki}}{\partial x_{ij}} \right] \\
&= E\left[ \sum_{k=1}^p \delta_{ik} [R(z)]_{ki} + x_{kj} \frac{\partial [R(z)]_{ki}}{\partial x_{ij}} \right] \\
&= E[[R(z)]_{ii}] + \sum_{k=1}^p E\left[ x_{kj} \frac{\partial [R(z)]_{ki}}{\partial x_{ij}} \right]
\end{aligned}
$$

From Assignment 5 we know
$$
\frac{\partial [R(z)]_{kl}}{\partial x_{ij}} = -\frac{1}{n} \left( [R(z)]_{ki} \cdot [X^T R(z)]_{jl} + [R(z)X]_{kj} [R(z)]_{il} \right),
$$
thus, by setting $l = i$, we get

$$
E \left[ x_{kj} \frac{\partial [R(z)]_{ki}}{\partial x_{ij}} \right] = -\frac{1}{n} E \left[ x_{kj} [R(z)]_{ki} [X^T R(z)]_{ji} + x_{kj} [R(z)X]_{kj} [R(z)]_{ii} \right]
$$
and therefore
$$
\begin{aligned}
1 + zS(z) &= \frac{1}{np} \sum_{\substack{i=1,\dots,p \\ j=1,\dots,n}} E[x_{ij} [X^T R(z)]_{ji}] \\
&= \frac{1}{np} \sum_{\substack{i=1,\dots,p \\ j=1,\dots,n}} \left( E[[R(z)]_{ii}] + \sum_{k=1}^p E\left[ x_{kj} \frac{\partial [R(z)]_{ki}}{\partial x_{ij}} \right] \right) \\
&=  \underbrace{\frac{1}{np}\sum_{\substack{i=1,\dots,p \\ j=1,\dots,n}} E[[R(z)]_{ii}]}_{\color{red} A} -  \underbrace{\frac{1}{n^2 p}\sum_{\substack{i=1,\dots,p \\ j=1,\dots,n \\ k=1,\dots,p}} E[x_{kj} [R(z)]_{ki} [X^T R(z)]_{ji}]}_{\color{orange}B} \\
&- \underbrace{\frac{1}{n^2 p} \sum_{\substack{i=1,\dots,p \\ j=1,\dots,n \\ k=1,\dots,p}} E[x_{kj} [R(z)X]_{kj} [R(z)]_{ii}]}_{\color{blue}C}.
\end{aligned}
$$

① For $\color{red} A$
$$
\begin{aligned}
\color{red}{A} &= \frac{1}{n p} \sum_{\substack{i=1, \dots, p \\ j=1, \dots, n}} E\left[[R(z)]_{i i}\right] \\
& = \frac{1}{n p} \sum_{j=1}^n E\left[\sum_{i=1}^p[R(z)]_{i i}\right]\\ &= \frac{1}{n p} \cdot n \cdot E[\operatorname{Tr}(R(z))] \\
& = \frac{1}{p} E[p \operatorname{tr}(R(z))] \\
& = E[\operatorname{tr}(R(z))] = S(z) .
\end{aligned}
$$
② For $\color{orange}B$
$$\begin{aligned}
\color{orange}B &=
\frac{1}{n^2 p} \sum_{\substack{i=1, \dots, p \\ j=1, \dots, n \\ k=1, \dots, p}} E\left[x_{k j}[R(z)]_{k i}\left[X^T R(z)\right]_{j i}\right] \\
& = \frac{1}{n^2 p} \sum_{\substack{i=1, \dots, p \\ j=1, \dots, n \\ k=1, \dots, p}} E\left[\left[X^T\right]_{j k}[R(z)]_{k i}\left[R(z)^T X\right]_{i j}\right] \\
& = \frac{1}{n^2 p} \sum_{j=1}^n E\left[\left[X^T R(z) R(z)^T X\right]_{j j}\right] \\
& = \frac{1}{n^2 p} E\left[\operatorname{Tr}\left(X^T R(z) R(z)^T X\right)\right] \\
& = \frac{1}{n^2 p} E\left[\operatorname{Tr}\left(X X^T R(z) R(z)^T\right)\right] \\
& = \frac{1}{n} E\left[\frac 1p\operatorname{Tr}\left(\frac{X X^T}{n} R(z) R(z)^T\right)\right] \xrightarrow{n \rightarrow \infty} 0 .
\end{aligned}$$
Here we say about the last limit: Set the eigenvalues of $\frac{X X^T}{n}$ is $\lambda_1,\cdots,\lambda_p$, then the eigenvalues of $R(z) R(z)^T=R^2(z)$ is $\frac 1{(\lambda_i-z)^2}$. Since $\frac{X X^T}{n}$ and $R^2(z)$ share same eigenvectors, the eigenvalues of $\frac{X X^T}{n}\cdot R^2(z)$ is $\frac {\lambda_i}{(\lambda_i-z)^2}$. By the M-P law, we know $\operatorname{Tr}\left(\frac{X X^T}{n} R(z) R(z)^T\right)$ is bounded when $n\rightarrow \infty$.

③ For $\color{blue} C$ (这里 tr 是 ntr 没有修改过)
$$\begin{aligned}
\color{blue} C &=
\frac{1}{n^2 p} \sum_{\substack{i=1, \dots, p \\ j=1, \dots, n \\ k=1, \dots, p}} E \left[ x_{kj} [R(z)X]_{kj} [R(z)]_{ii} \right] \\
&= \frac{1}{n^2 p} E \left[ \sum_{k=1}^p \sum_{j=1}^n x_{kj} [R(z)X]_{kj} \sum_{i=1}^p [R(z)]_{ii} \right] \\
&= \frac{1}{n^2 p} E \left[ \sum_{k=1}^p \sum_{j=1}^n x_{kj} [R(z)X]_{kj} \text{Tr}(R(z)) \right] \\
&= E \left[ \text{tr}(R(z)) \frac{1}{n^2} \sum_{k=1}^p \sum_{j=1}^n [X^T]_{jk} [R(z)X]_{kj} \right] \\
&= E \left[ \text{tr}(R(z)) \frac{1}{n^2} \sum_{j=1}^n [X^T R(z) X]_{jj} \right] \\
&= E \left[ \text{tr}(R(z)) \frac{1}{n^2} \text{Tr}(X^T R(z) X) \right] \\
&= E \left[ \text{tr}(R(z)) \frac{1}{n^2} \text{Tr}(XX^T R(z)) \right] \\
&= E \left[ \text{tr}(R(z)) \frac{p}{n} \text{tr} \left( \frac{XX^T}{n} R(z) \right) \right] \\
&\approx E \left[E \left[ \text{tr}(R(z)) \right] \cdot \frac{p}{n}  \text{tr} \left( \frac{XX^T}{n} R(z) \right) \right] \\
&\approx E \left[ \text{tr}(R(z)) \right] \cdot \frac{p}{n} E \left[ \text{tr} \left( \frac{XX^T}{n} R(z) \right) \right] \\
&= S(z) \cdot \gamma \cdot E \left[ \text{tr}(AR(z)) \right],
\end{aligned}$$

Note that we need concentration to asymptotically factorize the expectation of a product! Remember that we have $AR(z) = I_p + zR(z)$, so
$$E\left[\text{tr}(AR(z))\right] = E\left[\text{tr}(I_p + zR(z))\right] = 1 + zS(z),$$
so
$$\frac{1}{n^2 p} \sum_{\substack{i=1, \dots, p \\ j=1, \dots, n \\ k=1, \dots, p}} E \left[ x_{kj} [R(z)X]_{kj} [R(z)]_{ii} \right] \approx S(z) \cdot \gamma \cdot (1 + zS(z)).$$
Putting everything together, we get in the limit
$$1 + zS(z) = S(z) - \gamma S(z) \cdot (1 + zS(z)),$$
i.e.
$$\gamma z S(z)^2 + (z + \gamma - 1)S(z) + 1 = 0.$$
This is the same equation as we derived in the proof of the Marchenko-Pastur law (cf. Theorem 4.2). $\square$

## 7.3. Extension of Stein’s identity to cumulant expansion

Now we want to extend this approach from $X$ to $F = \sigma \left( \frac{1}{\sqrt{p}} WX \right) \in \mathbb{R}^{m \times n}$. So we start as before, with
$$S(z) = E \left[ \frac{1}{m}\text{Tr}  \left( \underbrace{\frac{1}{n} FF^T}_{m\times m} 
 - zI_m \right)^{-1}  \right] = E[\frac{1}{m}\text{Tr}(R(z))]$$
and the equation
$$\begin{aligned}
1 + zS(z) &= E \left[ \frac{1}{m}\text{Tr} \left( \frac{1}{n} FF^T R(z) \right) \right] \\
&= \frac{1}{nm} E \left[ \text{Tr} \left( FF^T R(z) \right) \right] \\
&= \frac{1}{nm} \sum_{\substack{i=1, \dots, m \\ j=1, \dots, n}} \underbrace{E \left[ f_{ij} [F^T R(z)]_{ji} \right]}_{F = (f_{ij})} 
,
\end{aligned}$$

The problem is now that the $f_{ij}$ are neither Gaussian nor (and this is more serious) independent any more! So we have to face the question: Do we still have a version of Stein’s identity for such a general case?

Recall the one-dimensional case of Stein’s identity: If $t$ is a standard Gaussian random variable, then
$$E[th(t)] = E[h'(t)].$$
For general distributions, one can try to
* keep the RHS and change the LHS; this leads to the theory of Stein score functions
* <u>keep the LHS and change the RHS; this leads to cumulant expansions and is what we think here.</u>

In order to get an idea what $E[th(t)]$ could be in general, we will consider it for special functions of the form $h_s(t) = \exp(its)$ and then get the general case by Fourier decomposition. Defining constants $\kappa_\ell$ via

>[!definition] Definition. Cumulant Generating Function, CGF
>Set $t$ is r.v. By Characteristic Function, we can get
>$$\log \varphi_t(s)=\log(E[\underbrace{\exp(is\cdot t)}_{h_s(t) = \exp(ist)} ]) =: \sum_{\ell=1}^\infty \frac{\kappa_\ell}{\ell!} (is)^\ell,$$
>- $\kappa_1$ 反映均值
>- $\kappa_2$ 反映方差
>- For Gaussian distribution, $\kappa_3=\kappa_4=\cdots=0$

$$\begin{aligned}
e^{isT} &= 1 + isT + \frac{(isT)^2}{2!} + \frac{(isT)^3}{3!} + \cdots\\
\varphi_T(s) = E[e^{isT}] &= 1 + \sum_{\ell=1}^\infty \frac{1}{\ell!} (is)^\ell\cdot \mu_\ell\\
\log \varphi_T(s)=\log E[e^{isT}] &=\underbrace{\log(1+z)}_{z = \sum_{\ell=1}^\infty \frac{1}{\ell!} (it)^\ell\cdot \mu_\ell} \\
&= z - \frac{z^2}{2} + \frac{z^3}{3} - \cdots \\
\text{(rearrange) }&=\sum_{\ell=1}^\infty \underbrace{(\cdots\cdots\cdots)}_{\text{Set it as }\frac{\kappa_\ell}{\ell!}} \cdot(is)^\ell \\
\text{(Disguised as Taylor-s) }&=\sum_{\ell=1}^\infty \kappa_\ell\frac{1}{\ell!} \cdot(is)^\ell
\end{aligned}$$
For our def, we have
$$\begin{aligned}
\mathbb{E} \left( t h_s(t) \right) &= \mathbb{E} \left( t e^{is\cdot t} \right) = \mathbb{E} \underbrace{\left( -i \frac{d}{ds} \exp(is\cdot t) \right)}_{}    \\
&\phantom{\mathbb{E} \left( t h_s(t) \right) = \mathbb{E}} \frac{d}{ds} \exp(is\cdot t) = \frac{i}{i} \cdot it \exp(is\cdot t) \\
&= -i \mathbb{E} it \cdot \mathbb{E} \exp(is\cdot t) \\
&= -i \cdot \frac{d}{ds} \boxed{\log \mathbb{E} e^{is\cdot t}} \cdot \mathbb{E} e^{is\cdot t} \\
&= -i \cdot \frac{d}{ds} \sum_{l=1}^{\infty} \frac{\kappa_l}{l!} (is)^l \cdot \mathbb{E} e^{is\cdot t} \\
&= \sum_{l=1}^{\infty} \frac{\kappa_l}{(l-1)!} (is)^{l-1} \mathbb{E} e^{is\cdot t} \\
&= \mathbb{E} \sum_{l=0}^{\infty} \frac{\kappa_{l+1}}{l!} (is)^l e^{is\cdot t} 
= \mathbb{E} \sum_{l=0}^{\infty} \frac{\kappa_{l+1}}{l!} \frac{d^l}{dt^l} e^{is\cdot t} \\
&= \mathbb{E} \sum_{l=0}^{\infty} \frac{\kappa_{l+1}}{l!} h_s^{(l)}(t)
\end{aligned}$$

By Fourier decomposition, this then goes over to “arbitrary functions”:
>[!lemma] Lemma. 7.3 Extension of Stein’s identity I
>Let $t$ be a random variable such that all of its moments exist. Then we define its “cumulants” $\kappa_\ell$ by
>$$\log \varphi_T(s)=\log(E[\exp(is\cdot t)]) = \sum_{\ell=1}^\infty \frac{\kappa_\ell}{\ell!} (is)^\ell,$$
>and for a smooth function $h : \mathbb{R} \to \mathbb{R}$ we then have
$$E[th(t)] = \sum_{\ell=0}^\infty \frac{\kappa_{\ell+1}}{\ell!} E[h^{(\ell)}(t)].$$

***Remark.*** 
Note that this is consistent with Stein’s identity. If $t$ is a standard Gaussian random variable, then we have
$$E[\exp(its)] = \exp \left( -\frac{s^2}{2} \right),$$
thus
$$\log(E[\exp(its)]) = -\frac{s^2}{2},$$
which means that all $\kappa_\ell$ are zero except $\kappa_2 = 1$; but then Lemma 7.3 reduces to
$$E[th(t)] = \frac{\kappa_2}{1!} E[h'(t)] = E[h'(t)].$$
We now need the multivariate version of all this. This works in the same way.

>[!definition] Definition. 7.4 Multivariate version of Cumulant Generating Function
>Let $t_1, \dots, t_k$ be a collection of random variables.
>$$\begin{aligned}\varphi_{(t_1, \dots, t_k)}(s)&= \sum_{\vec{l}} \frac{1}{\vec{l}!} \mathbb{E}(t_1^{l_1} \dots t_k^{l_k}) \cdot (is)^{\vec{l}}\\
\log(\varphi_{(t_1, \dots, t_k)}(s))&= \sum_{|\vec{l}|_1 \neq 0} \kappa_{\vec{l}} \frac{1}{\vec{l}!}(is)^{\vec{l}}   \\\end{aligned}$$
>where $\vec{\ell} = (\ell_1, \dots, \ell_k)$ is a multi-index, and $\vec\ell! = \ell_1! \dots \ell_k!$ $s^{\vec{l}} = s_1^{l_1} \dots s_k^{l_k}$ .

***Remark.***
$$
\begin{aligned}
\varphi_{(t_1, \dots, t_k)}(s) &= \mathbb{E} e^{\sum_{j=1}^k i s_j t_j} = \mathbb{E} e^{i \sum_{j=1}^k s_j t_j} \\
&= \mathbb{E} \sum_{n=0}^\infty \frac{1}{n!} i^n \underbrace{(s_1 t_1 + \dots + s_k t_k)^n}_{\text{Multi-nominal}} \\
&= \mathbb{E} \sum_{n=0}^\infty \frac{i^n}{n!} \sum_{l_1 + \dots + l_k = n} \frac{n!}{l_1! \dots l_k!} (t_1 s_1)^{l_1} \dots (t_k s_k)^{l_k} \\
&= \sum_{n=0}^\infty i^n \sum_{l_1 + \dots + l_k = n} \frac{s_1^{l_1} \dots s_k^{l_k} \mathbb{E}(t_1^{l_1} \dots t_k^{l_k})}{l_1! \dots l_k!} \\
&\phantom{==} \text{Set } \vec{l} = (l_1, \dots, l_k), s^{\vec{l}} = s_1^{l_1} \dots s_k^{l_k}, \vec{l}! = l_1! \dots l_k! \\
&= \sum_{n=0}^\infty i^n \sum_{|\vec{l}|_1 = n} \frac{s^{\vec{l}}}{\vec{l}!} \mathbb{E}(t_1^{l_1} \dots t_k^{l_k}) \\
&= \boxed{\sum_{n=0}^\infty \sum_{|\vec{l}|_1 = n}} \frac{(is)^{\vec{l}}}{\vec{l}!} \mathbb{E}(t_1^{l_1} \dots t_k^{l_k}) \\
&= \sum_{\vec{l}} \frac{1}{\vec{l}!} \mathbb{E}(t_1^{l_1} \dots t_k^{l_k}) \cdot (is)^{\vec{l}}
\end{aligned}
$$
$$
\begin{aligned}
\log(\varphi_{(t_1, \dots, t_k)}(s)) &= \log \Big[ 1 + \underbrace{\sum_{|\vec{l}|_1 \neq 0} \frac{1}{\vec{l}!} \mathbb{E}(t_1^{l_1} \dots t_k^{l_k}) \cdot (is)^{\vec{l}}}_{z} \Big] \\
&= z - \frac{z^2}{2} + \frac{z^3}{3} - \dots \\
&= \sum_{|\vec{l}|_1 \neq 0} \kappa_{\vec{l}} \frac{1}{\vec{l}!} (is)^{\vec{l}}
\end{aligned}
$$
 The $\kappa$ are actually multi-linear mappings in the random variables, i.e. the coefficient of $s_{i_1} \dots s_{i_m}$ is $\kappa(t_{i_1}, \dots, t_{i_m})$, which ensures symmetric.
$$\kappa_{\vec{l}}=\kappa(\underbrace{t_1,\cdots, t_1}_{l_1}, \dots, \underbrace{t_k,\cdots, t_k}_{l_k})$$

>[!lemma] Lemma. 7.5 Extension of Stein’s identity II
>Let $t_1, \dots, t_k$ be a collection of random variables and $\kappa$ their cumulants. For smooth functions $h : \mathbb{R}^k \to \mathbb{R}$ we then have
$$E[t_i h(t_1, \dots, t_k)] = \sum_{|\ell|_1 \ge 0} \sum_{i_1, \dots, i_{|\ell|_1}=1}^k \frac{\kappa(t_i, t_{i_1}, \dots, t_{i_\ell})}{\ell!} E[\partial_{i_1} \dots \partial_{i_\ell} h(t_1, \dots, t_k)].$$
>where $\partial_{i_k}=\partial/\partial t_{i_k}$

***Example.***
(1) For $h = 1$, only $\ell = 0$ contributes, so
$$E[t_i] = E[t_i \cdot 1] = \frac{\kappa(t_i)}{0!} \underbrace{E[1]}_{=1} = \kappa(t_i).$$
(2) For $h(t_1, \dots, t_k) = t_j$ we have
$$\begin{aligned}
E[t_i t_j] &= \underbrace{\kappa(t_i) \cdot E[t_j]}_{\text{from } \ell=0} + \underbrace{\sum_{i_1=1}^k \kappa(t_i, t_{i_1}) E[\partial_{i_1} t_j]}_{\text{from } \ell=1} \\
&= \kappa(t_i) \cdot \kappa(t_j) + \sum_{i_1=1}^k \kappa(t_i, t_{i_1}) \delta_{i_1 j} \\
&= \kappa(t_i) \cdot \kappa(t_j) + \kappa(t_i, t_j),\\
&= \underbrace{\kappa(t_i, t_j)}_{\sqcup} + \underbrace{\kappa(t_i)\kappa(t_j)}_{| \ |}
\end{aligned}
$$
where the two summands correspond to the two partitions of $\{t_i, t_j\}$ drawn below them.
(3) For $h(t_1, \dots, t_k) = t_j t_r$ we have
$$\begin{aligned}
E[t_i t_j t_r] &= \kappa(t_i)E[t_j t_r] + \sum_{i_1=1}^k \kappa(t_i, t_{i_1})E[\partial_{i_1} t_j t_r] + \sum_{i_1, i_2=1}^k \frac{1}{2} \kappa(t_i, t_{i_1}, t_{i_2})E[\partial_{i_1} \partial_{i_2} t_j t_r] \\
&= \kappa(t_i)E[t_j t_r] + \kappa(t_i, t_j)E[t_r] + \kappa(t_i, t_r)E[t_j] + \frac{1}{2}(\kappa(t_i, t_j, t_r) + \kappa(t_i, t_r, t_j)) \\
&= \kappa(t_i)E[t_j t_r] + \kappa(t_i, t_j)E[t_r] + \kappa(t_i, t_r)E[t_j] + \kappa(t_i, t_j, t_r) \\
&= \kappa(t_i, t_j, t_r) + \kappa(t_i)\kappa(t_j, t_r) + \kappa(t_i, t_j)\kappa(t_r) + \kappa(t_i, t_r)\kappa(t_j) + \kappa(t_i)\kappa(t_j)\kappa(t_r)
\end{aligned}$$
(corresponding respectively to the partitions └┴┘, │ └┘, └┘ │, └┬┘, and │ │ │ of $\{t_i, t_j, t_r\}$), since $E[\partial_{i_1} t_j t_r] = \delta_{i_1 j} t_r + \delta_{i_1 r} t_j$ and $E[\partial_{i_1} \partial_{i_2} t_j t_r] = \delta_{i_1 j} \delta_{i_2 r} + \delta_{i_1 r} \delta_{i_2 j}$.)

>[!theo] Theorem. 7.6 Moment-cumulant formula 矩累计公式
>$$E[t_{i_1} \cdot t_{i_2} \cdot \dots \cdot t_{i_n}] = \sum_{\pi \in \mathcal{P}(n)} \kappa_\pi(t_{i_1}, t_{i_2}, \dots, t_{i_n}),$$where $\pi = \{V_1, \dots, V_r\}$ is a partition of the set $\{1, \dots, n\}$ and$$\kappa_\pi = \kappa_{V_1}(\dots) \kappa_{V_2}(\dots) \cdot \dots \cdot \kappa_{V_r}(\dots),$$

## 7.4. Cumulants and their properties and uses

>[!definition]- Definition. $\pi = \{V_1, \dots, V_r\}$ a *partition*
>(1) We call $\pi = \{V_1, \dots, V_r\}$ a *partition* of the set $S$ if
>* $V_i \neq \emptyset$ and $V_i \subset S$ for all $i$,
>* $V_i \cap V_j = \emptyset$ for all $i \neq j$, and
>* $V_1 \cup \dots \cup V_r = S$.
>We call $V_1, \dots, V_r$ the *blocks* of $\pi$. Given two elements $p, q \in S$, we write $p \sim_\pi q$ if $p$ and $q$ belong to the same block of $\pi$.
>
>(2) The set of all partitions of $S$ is denoted by $\mathcal{P}(S)$. If $S = \{1, \dots, n\}$, we write $\mathcal{P}(n) = \mathcal{P}(\{1, \dots, n\})$. Note that $\mathcal{P}(n)$ has a "smallest" element
$$0_n := \{\{1\}, \dots, \{n\}\} \in \mathcal{P}(n)$$
and a "largest" element
$$1_n := \{\{1, \dots, n\}\} \in \mathcal{P}(n).$$

Often we use a graphical representation of a partition $\pi$:
![[7 Non-Linear Random Matrix Models -- Resolvent Method & Cumulant Expansions-1773906320671.webp|616]]
In the following we consider an algebra $\mathcal{A}$ of our random variables, for which the expectation $E : \mathcal{A} \to \mathbb{R}$ is defined. For example, if we have a collection of random variables $t_1, \dots, t_r$, then
$$\mathcal{A} = \left\{ \sum_{n \ge 0} \sum_{i(1), \dots, i(n)=1}^r \alpha_{i(1), \dots, i(n)} t_{i(1)} \cdot \dots \cdot t_{i(n)} \mid \alpha_{i(1), \dots, i(n)} \in \mathbb{C} \right\}$$
is the collection of all polynomials in the random variables. Given such $(\mathcal{A}, E)$ we know all moments $E(t)$ for all $t \in \mathcal{A}$. But for many questions it is better to rewrite the information about moments into other objects, so-called cumulants.

**Notation 7.8.** For $\pi = \{V_1, \dots, V_r\} \in \mathcal{P}(n)$ and $t_1, \dots, t_n \in \mathcal{A}$ we put
$$E_\pi(t_1, \dots, t_n) = \prod_{V_i \in \pi} E(t_1, \dots, t_n \mid V_i),$$
where for $V = \{i_1 < i_2 < \dots < i_s\}$
$$E(t_1, \dots, t_n \mid V) = E(t_{i_1} t_{i_2} \dots t_{i_s}).$$
More general, if we have a collection $(\kappa_n)_{n \in \mathbb{N}}$ of $n$-linear functions
$$\kappa_n : \mathcal{A}^n \to \mathbb{R}, \quad (t_1, \dots, t_n) \mapsto \kappa_n(t_1, \dots, t_n),$$
then we define in the same way their multiplicative extension to all $\mathcal{P}(n)$ by
$$\kappa_\pi(t_1, \dots, t_n) = \prod_{V \in \pi} \kappa_{\#V}(t_1, \dots, t_n \mid V),$$
where for $V = \{i_1 < i_2 < \dots < i_s\}$
$$\kappa_{\#V}(t_1, \dots, t_n \mid V) = \kappa_s(t_{i_1}, t_{i_2}, \dots, t_{i_s}).$$

>[!definition] **Definition 7.9.** 
>Given $(\mathcal{A}, E)$ we define the *cumulants* of the random variables in $\mathcal{A}$ as $\kappa_n : \mathcal{A}^n \to \mathbb{R}$ by the moment-cumulant formulas: for all $n \in \mathbb{N}$ and all $t_1, \dots, t_n \in \mathcal{A}$ $$E(t_1 \cdot \dots \cdot t_n) = \sum_{\pi \in \mathcal{P}(n)} \kappa_\pi(t_1, \dots, t_n). \tag{MCF}$$These MCF define the $\kappa_n$'s recursively as $n$-linear functionals.

**Example 7.10.** 
(1) $n = 1$: we have $E(t_1) = \kappa_1(t_1)$ and thus $\kappa_1(t_1) = E(t_1)$, corresponding to the partition $|$ of $\{t_1\}$.
(2) $n = 2$: we have
$$\begin{aligned} 
E(t_1 t_2) &= \kappa_2(t_1, t_2) & \begin{matrix} t_1 & t_2 \\ 
\sqcup & \end{matrix} \\ &+ \kappa_1(t_1)\kappa_1(t_2) & \begin{matrix} | & | \end{matrix} \end{aligned}$$
and thus
$$\kappa_2(t_1, t_2) = E(t_1 t_2) - \kappa_1(t_1)\kappa_1(t_2) = E(t_1 t_2) - E(t_1)E(t_2),$$
which is the covariance of $(t_1, t_2)$.
(3) $n = 3$: we have
![[7 Non-Linear Random Matrix Models -- Resolvent Method & Cumulant Expansions-1773923495106.webp]]
We see that we can write the cumulants, similar as in the MCF, via a summation over $\mathcal{P}(n)$, <u>but now we get non-trivial coefficients</u>. This rewriting of the MCF is a general version of the inclusion-exclusion principle, abstractly known as Möbius inversion.

>[!theo] **Theorem 7.11.** Möbius inversion
>The recursive definition of the cumulants via the MCF is equivalent to the explicit formula
$$\kappa_n(t_1, \dots, t_n) = \sum_{\pi \in \mathcal{P}(n)} (-1)^{\#\pi - 1} (\#\pi - 1)! E_\pi(t_1, \dots, t_n). \tag{CMF}$$
>If $\pi=\{V_1,V_2,V_3\}$, then $\#\pi-1 =2$. The relevance of the cumulants is that they characterize independence.

>[!theo] **Theorem 7.12.** 
>Consider in $(\mathcal{A}, E)$ subsets $T_i \subset \mathcal{A} \ (i \in I)$ of random variables. Then the following are equivalent:
>1. $T_i$ are independent;
>2. Mixed cumulants in the $T_i$ vanish: $\kappa_n(t_1, \dots, t_n) = 0$ whenever $t_j \in T_{i(j)}$ and there exist $\ell, k$ such that $i(\ell) \neq i(k)$.

***Proof.*** 
Let us only check (ii)$\Rightarrow$(i); namely that vanishing of mixed cumulants gives us factorization of moments. We do this via a telling example; consider $E[t_1 s_1 t_2 s_2 s_3 t_3 t_4]$, where mixed moments in $\{t_1, t_2, t_3, t_4\}$ and $\{s_1, s_2, s_3\}$ vanish. We have
$$E[t_1 s_1 t_2 s_2 s_3 t_3 t_4] = \sum_{\pi} \kappa_{\pi},$$
but in the sum partitions like $\sqcup \sqcup \quad \sqcup \quad \sqcup$ are not included, since blocks are not allowed to connect a $t_i$ with an $s_j$. On the other hand, partitions like $\sqcup \sqcup \mid \quad \sqcup$ are included. So we have $\pi = \pi_s \cup \pi_t$, where $\pi_s$ is a partition of $\{s_1, s_2, s_3\}$ and $\pi_t$ is a partition of $\{t_1, t_2, t_3, t_4\}$. Continuing the computation, we get

$$\begin{aligned} E[t_1 s_1 t_2 s_2 s_3 t_3 t_4] &= \sum_{\pi} \kappa_{\pi}(t_1, s_1, t_2, s_2, s_3, t_4, t_5) \\ &= \sum_{\pi_s \cup \pi_t} \kappa_{\pi_s \cup \pi_t}(t_1, s_1, t_2, s_2, s_3, t_4, t_5) \\ &= \sum_{\pi_s \cup \pi_t} \kappa_{\pi_s}(s_1, s_2, s_3) \kappa_{\pi_t}(t_1, t_2, t_3, t_4) \\ &= \left( \sum_{\pi_s} \kappa_{\pi_s}(s_1, s_2, s_3) \right) \left( \sum_{\pi_t} \kappa_{\pi_t}(t_1, t_2, t_3, t_4) \right) \\ &= E(s_1 s_2 s_3) \cdot E(t_1 t_2 t_3 t_4). \end{aligned} $$

***Remark.*** 
1. Note that, as for moments, cumulants do not change under permutation of arguments; e.g.
$$\kappa_3(t_1, t_2, t_3) = \kappa_3(t_1, t_3, t_2) = \kappa_3(t_2, t_1, t_3)$$
etc. since the terms in the CMF are mapped to each other under such permutations.
2. Cumulants seem to be more complicated than moments. So, why do we want to use them? Here are some answers to this question.
   - Expansions around special situations (like independent Gaussians 除了二阶 cumulant, 其他高阶 cumulant 都是 0.) are easier to deal with.
   - Almost factorization of moments is hard to work with, almost vanishing (i.e. smallness) of cumulants is much better for estimates.
3. In order to be able to make really good use of cumulants, we also have to understand their multiplicative structure.


>[!theorem] Theorem 7.13. 变量相乘时的累积量计算规则
>Consider $n$ random variables and multiply them together in $m$ groups
>$$
>\begin{aligned}
>T_1 &= t_1 t_2 \cdot \dots \cdot t_{i(1)}, \\
>T_2 &= t_{i(1)+1} \cdot \dots \cdot t_{i(2)}, \\
>&\vdots \\
>T_m &= t_{i(m-1)+1} \cdot \dots \cdot t_{i(m)},
>\end{aligned}
>$$
>i.e.
>$$
>\underbrace{t_1 t_2 \cdot \dots \cdot t_{i(1)}}_{T_1} \cdot \underbrace{t_{i(1)+1} \cdot \dots \cdot t_{i(2)}}_{T_2} \cdot \dots \cdot \underbrace{t_{i(m-1)+1} \cdot \dots \cdot t_{i(m)}}_{T_m}.
>$$
>Then we have
>$$
>\kappa_m(T_1, \dots, T_m) = \sum_{\substack{\pi \in \mathcal{P}(n) \\ \pi \text{ connects all the} \\ \text{groups together}}} \kappa_\pi(t_1, t_2, \dots, t_n).
>$$

***Example.*** 
1. No products: if $T_i = t_i$ for all $i$, then
$$
\begin{aligned}
\kappa_n(t_1, \dots, t_n) &= \sum_{\substack{\pi \in \mathcal{P}(n) \\ \pi \text{ connects all the} \\ \text{groups together}}} \kappa_\pi(t_1, t_2, \dots, t_n) \\
&= \kappa_n(t_1, \dots, t_n),
\end{aligned}
$$
since only $1_n$ connects everything.
2. One product: if $T = t_1 \cdot \dots \cdot t_n$, then
$$
\begin{aligned}
\kappa_1(T) &= \sum_{\substack{\pi \in \mathcal{P}(n) \\ \pi \text{ connects all the} \\ \text{groups together}}} \kappa_\pi(t_1, t_2, \dots, t_n) \\
&= \sum_{\pi \in \mathcal{P}(n)} \kappa_\pi(t_1, \dots, t_n) = E(t_1 \cdot \dots \cdot t_n),
\end{aligned}
$$
since all partitions "connect" the only block. This agrees with
$$
\kappa_1(T) = E(T) = E(t_1 \cdot \dots \cdot t_n).
$$
3. We have
$$
\kappa_3(t_1 t_2, t_3, t_4) = \kappa_4(t_1, t_2, t_3, t_4) + \kappa_3(t_1, t_3, t_4)\kappa_1(t_2) + \dots
$$
according to the connecting partitions $\text{ \textvisiblespace \textvisiblespace \textvisiblespace }$, $\text{ \textvisiblespace } | \text{ \textvisiblespace }$, $\text{ \textvisiblespace } \lfloor \text{ \textvisiblespace }$, $\lfloor \text{ \textvisiblespace } \rfloor$, and $| \text{ \textvisiblespace }$.

Note also that our definition of the cumulants via the MCF leads directly to our cumulant expansion from **Proposition 7.5** if we choose $h$ as a moment. Namely, take
$$
h(t_1, \dots, t_k) = t_{r(1)} \cdot \dots \cdot t_{r(n)},
$$
then
$$
E[t_i h(t_1, \dots, t_k)] = E[t_i t_{r(1)} \cdot \dots \cdot t_{r(n)}] = \sum_{\pi \in \mathcal{P}(n+1)} \kappa_\pi(t_i, t_{r(1)}, \dots, t_{r(n)}).
$$
Now write $\pi = V \cup (\pi \setminus V)$, where $V = \{1, j_1, \dots, j_\ell\}$ is the block of $\pi$ containing 1. Then $E[t_i h(t_1, \dots, t_k)]$
$$
\begin{aligned}
&= \sum_{\pi \in \mathcal{P}(n+1)} \kappa_\pi(t_i, t_{r(1)}, \dots, t_{r(n)}) \\
&= \sum_{\ell \ge 0} \sum_{j_1, \dots, j_\ell} \sum_{\pi \setminus V} \kappa_{\ell+1}(t_i, t_{r(j_1)}, \dots, t_{r(j_\ell)}) \kappa_{\pi \setminus V}(t_{r(1)}, \dots, t_{r(n)} \mid \{1, \dots, n\} \setminus V) \\
&= \sum_{\ell \ge 0} \sum_{j_1, \dots, j_\ell} \kappa_{\ell+1}(t_i, t_{r(j_1)}, \dots, t_{r(j_\ell)}) E(t_{r(1)}, \dots, t_{r(n)} \mid \{1, \dots, n\} \setminus V) \\
&= \sum_{\ell \ge 0} \sum_{i_1, \dots, i_\ell} \sum_{j_1, \dots, j_\ell} \kappa_{\ell+1}(t_i, t_{i_1}, \dots, t_{i_\ell}) \cdot E\left[ t_{r(1)} \dots \frac{\partial t_{r(j_1)}}{\partial t_{i_1}} t_{r(j_1+1)} \dots \frac{\partial t_{r(j_\ell)}}{\partial t_{i_\ell}} \dots t_{r(n)} \right] \\
&= \sum_{\ell \ge 0} \sum_{i_1, \dots, i_\ell=1}^k \kappa_{\ell+1}(t_i, t_{i_1}, \dots, t_{i_\ell}) \cdot \frac{1}{\ell!} E[\partial_{i_1} \dots \partial_{i_\ell} t_{r(1)} \dots t_{r(n)}] \\
&= \sum_{\ell \ge 0} \sum_{i_1, \dots, i_\ell=1}^k \frac{\kappa_{\ell+1}(t_i, t_{i_1}, \dots, t_{i_\ell})}{\ell!} E[\partial_{i_1} \dots \partial_{i_\ell} h(t_1, \dots, t_k)].
\end{aligned}
$$

## 7.5 Cumulants and Stieltjes transform for the random feature model

>[!proposition] Proposition 7.15.
>Let $W \in \mathbb{R}^{m \times p}$ and $X \in \mathbb{R}^{p \times n}$ be independent standard Gaussian matrices. Then the cumulants of
>$$(g_{ij}) = G := \frac{1}{\sqrt{p}} W \cdot X \in \mathbb{R}^{m \times n}$$
>are given by
>$$\kappa_r(g_{i_1 j_1}, g_{i_2 j_2}, \dots, g_{i_r j_r}) = \frac{1}{p^{\frac{r}{2}-1}} \cdot M,$$
>where $M$ is the number of permutations $\sigma \in S_r$ such that
>$$(\overbracket[1pt]{i_{\sigma(1)} \underbracket[1pt]{j_{\sigma(1)})(j_{\sigma(2)}} \underbracket[1pt]{i_{\sigma(2)})(i_{\sigma(3)}} \underbracket[1pt]{j_{\sigma(3)}) \dots} \dots \underbracket[1pt]{\dots (j_{\sigma(r)}} i_{\sigma(r)}})$$

Let us now use this structure of the cumulants in the cumulant expansion to calculate
$$S(z) = E \left[ \frac 1m\text{Tr} \underbrace{\left( \frac{G_{m\times n}G^T_{n\times m}}{n} - zI_m \right)^{-1}}_{=:R(z)} \right]$$
via
$$\begin{aligned} 1 &+ zS(z)\\ 
&= \frac{1}{nm} \sum_{i,j} \boxed{E\left(g_{ij} [G^T R(z)]_{ji}\right)} \\ 
&= \frac{1}{nm} \sum_{i,j} \sum_{\ell} \frac{1}{\ell!} \kappa_{\ell+1}(g_{ij}, g_{p_1 q_1}, g_{p_2 q_2}, \dots, g_{p_\ell q_\ell}) \cdot E\left(\partial_{p_1 q_1} \partial_{p_2 q_2} \dots \partial_{p_\ell q_\ell} [G^T R(z)]_{ji}\right) \\ 
&= \frac{1}{nm} \sum_{r} \frac{1}{p^{r-1}} \sum_{\substack{i_1, \dots, i_r \\ j_1, \dots, j_r}} E\left(\partial_{i_2 j_1} \partial_{i_2 j_2} \dots \partial_{i_1 j_r} [G^T R(z)]_{j_1 i_1}\right). \end{aligned}$$

This results in a term for $r = 1$, which was treated in the second proof of the Marchenko-Pastur law after <font color="red">Lemma 7.2</font>, and terms for $r > 1$. For the latter ones, one does two of the partial integrations and reduces it to versions of $1 + zS(z)$. After quite a bit of approximations and technicalities (which we prefer not to do here), this finally gives the equation
$$1 + zS(z) = S(z) \cdot (1 - \gamma(1 + zS(z))) \cdot \left(1 - \frac{1}{\tilde{\gamma}}(1 + zS(z))\right),$$
which is the special case $\theta_1 = \theta_2 = 1$ of <font color="red">Theorem 7.1</font>.

Now let us finally consider the effect of the non-linearity $\sigma$. The main observation is that the qualitative cumulant structure of $G$ is preserved for $F = \sigma(G) = (f_{ij})$.

>[!proposition] Proposition. 7.16. 
>In leading order we have for the cumulants of the $\{f_{ij}\}$:
>1. only cumulants with a cyclic structure $≠ 0$
>2. odd cumulants $=0$
>3. 总强度：$$\begin{aligned}  
>\kappa_2(f_{ij}, f_{ij}) &= \theta_1(\sigma)= E[\sigma(g)^2]\\
>&= \int_{\mathbb{R}} \sigma(t)^2 \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{t^2}{2}\right) dt\\
>\end{aligned}$$
>4. 线性强度：for disjoint $i_1, j_1, \dots, i_r, j_r$, $r > 1$，$$\kappa_{2r}(f_{i_1 j_1}, f_{i_2 j_1}, f_{i_2 j_2}, \dots, f_{i_1 j_r}) = \frac{1}{p^{r-1}} \theta_2(\sigma)^r$$
>$$\theta_2(\sigma) = \left(E[\sigma'(g)]\right)^2 = \left( \int_{\mathbb{R}} \sigma'(t) \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{t^2}{2}\right) dt \right)^2$$

## 7.6. The Gaussian equivalence principle for the non-linear random feature model

在分析极其复杂的非线性模型（如神经网络的特征层）时，我们可以用一个更简单的**线性模型加上独立噪声**来替代它，而两者的宏观统计特性（如特征值分布）在数学上是完全一致的 

Note that getting the final formula for $S(z)$ out of the cumulants might not be easy, but from Proposition 7.5 on the form of the cumulants it is very easy to see that
$$F = \sigma \left( \frac{W}{\sqrt{p}} X \right) \quad \text{and} \quad \tilde{F} = \sqrt{\theta_2} \frac{W}{\sqrt{p}} X + \sqrt{\theta_1 - \theta_2} Z = (\tilde{f}_{ij})$$
$$\tilde{F} = \sqrt{\theta_2} \underbrace{\frac{W}{\sqrt{p}} X}_{\text{线性部分}} + \sqrt{\theta_1 - \theta_2} \underbrace{Z}_{\text{独立噪声}}$$
- **$\sqrt{\theta_2}$**：衡量了激活函数 $\sigma$ 的“线性贡献” 。
- **$\sqrt{\theta_1 - \theta_2}$**：衡量了非线性带来的“额外噪声”
- **$Z$**：是一个与 $W, X$ 都独立的高斯随机矩阵 

have in leading order the same cumulants and thus $\frac{1}{n} F F^T$ and $\frac{1}{n} \tilde{F} \tilde{F}^T$ have the same asymptotic eigenvalue distribution. Such statements are known as Gaussian equivalence principle. Let us check this. We have for $r > 1$
$$\kappa_{2r}(\tilde{f}_{i_1 j_1}, \dots, \tilde{f}_{i_r j_r}) = \theta_2^r \cdot \kappa_{2r}(g_{i_1 j_1}, \dots, g_{i_r j_r}) = \theta_2^r \cdot \frac{1}{p^{r-1}}$$
and
$$\kappa_2(\tilde{f}_{ij}, \tilde{f}_{ij}) = \theta_2 \underbrace{\kappa_2(g_{ij}, g_{ij})}_{=1} + (\theta_1 - \theta_2) \underbrace{\kappa_2(z_{ij}, z_{ij})}_{=1} = \theta_2 + \theta_1 - \theta_2 = \theta_1.$$

Note that mixed cumulants in $WX$ and $Z$ vanish and that $Z$ only has second-order cumulants.

Note also that in the asymptotic calculation in the summations over $i_1, j_1, \dots, i_r, j_r$ one can neglect terms where some of the indices are the same. That's good because we do not really have much control over terms like $\kappa_4(f_{11}, f_{11}, f_{12}, f_{12})$.