---
source: "[[High-Dimensional Analysis- Random Matrices and Machine Learning.pdf]]"
tags:
  - 随机矩阵与机器学习_random-matrices-and-machine-learning
  - 信号噪声模型_spiked-signal-noise-model
  - BBP相变_bbp-transition
  - 特征值_eigenvalue
---

>[!theo] Theorem. 5.1 BBP Transition [^1][^2]
>Consider, for fixed $\mu \geq 0$, as above a covariance matrix of the form $\Sigma = I_p + \mu u u^T\ (||u||=1)$.
>Then, in the asymptotic regime $\frac{p}{n} \to \gamma \in (0, 1]$, the largest eigenvalue $\lambda$ of $\hat{\Sigma}=\frac 1n XX^T$ is given by
>$$
>\lambda = \begin{cases} 1 + \mu + \gamma \frac{1+\mu}{\mu}, & \text{if } \mu > \sqrt{\gamma}, \\ \gamma_+ = (1 + \sqrt{\gamma})^2, & \text{if } \mu \leq \sqrt{\gamma}. \end{cases}
>$$

***Remark.***
- The eigenvalue of $\Sigma = I_p + \mu u u^T\ (||u||=1)$ is? In fact we know $\mu u u^T\sim (\mu, u)$, then $I_p+\mu u u^T\sim (1+\mu, u)$ and other eigenvalues are 1.
- $1+\mu$ is the theoretical (theore~~m~~tical) value, $\gamma \frac{1+\mu}{\mu}$ 分别表示幅度(左)和扭曲度(右)
- Note that $1 + \mu + \gamma \frac{1+\mu}{\mu} > \gamma_+$ for $\mu > \sqrt{\gamma}$ and for $\mu = \sqrt{\gamma}$ we have
$$
1 + \sqrt{\gamma} + \gamma \frac{1 + \sqrt{\gamma}}{\sqrt{\gamma}} = 1 + 2\sqrt{\gamma} + \gamma = (1 + \sqrt{\gamma})^2 = \gamma_+,
$$
- ![[Gemini_Generated_Image_puayi6puayi6puay.png|365]]

***Proof.***
***Step 1***
As we know, $x \sim N(\mu, \Sigma)$, $y \sim N(0, I_p)$, then $X = \Sigma^{\frac{1}{2}} y + \mu$.

We set $x_k \sim \Sigma^{\frac{1}{2}} y_k$ and $y_k \sim N(0, I_p)$, then $X = \Sigma^{\frac{1}{2}} Y$. Notice that $\frac{1}{n} Y Y^T$ is Wishart matrix and can get its $\psi_{mp}$ on $[\gamma_-, \gamma_+]$

Then set $\lambda \in \text{eigenvalue-set}(\hat{\Sigma}) - [\gamma_-, \gamma_+]$, we have
$$
\begin{aligned}
0 &= |\hat{\Sigma} - \lambda I_p| = |\frac{1}{n} X X^T - \lambda I_p| \\
&= |\frac{1}{n} \Sigma^{\frac{1}{2}} Y Y^T \Sigma^{\frac{1}{2}} - \lambda I_p| \\
&= |\Sigma^{\frac{1}{2}}| \cdot |\frac{1}{n} Y Y^T - \lambda \cdot \Sigma^{-\frac{1}{2}} I_p \Sigma^{-\frac{1}{2}}| \cdot |\Sigma^{\frac{1}{2}}| \\
&= \underbrace{|\Sigma|}_{\neq 0} \cdot |\frac{1}{n} Y Y^T - \lambda \Sigma^{-1}| \\
0 &= |\frac{1}{n} Y Y^T - \lambda \Sigma^{-1}| \quad (\Sigma^{-1} = (I_p + \mu \cdot u u^T)^{-1}) \\
&\phantom{= |\frac{1}{n} Y Y^T - \lambda \Sigma^{-1}| \quad (\Sigma^{-1} }= I_p^{-1} - \frac{\mu \cdot I_p^{-1} u \cdot u^T I_p^{-1}}{1 + \mu \cdot u^T I_p^{-1} u} \\
&\phantom{= |\frac{1}{n} Y Y^T - \lambda \Sigma^{-1}| \quad (\Sigma^{-1} }= I_p - \frac{\mu \cdot u \cdot u^T}{1 + \mu u^T u} = I_p - \frac{\mu u u^T}{1 + \mu}\\
&= \left| \left( \frac{1}{n} YY^T - \lambda I_p \right) + \frac{\lambda \mu u u^T}{1 + \mu} \right| \\
&= \underbrace{\left| \frac{1}{n} YY^T - \lambda I_p \right|}_{\substack{\neq 0\\\lambda \notin [\gamma_-, \gamma_+]}} \cdot \left| I_p + \left( \frac{1}{n} YY^T - \lambda I_p \right)^{-1} \frac{\lambda \mu u u^T}{1 + \mu} \right| \\
\end{aligned}
$$
***Step 2***
Ok now take a break and review a lemma:
![[Sylvester' Determinant Identity#^icw6w8]]
Proof is given in [[Sylvester' Determinant Identity]]
Then
$$
\begin{aligned}
0 &= | I_p + \underbrace{\frac{\lambda \mu}{1 + \mu} (\frac{1}{n} Y Y^T - \lambda I_p)^{-1} u}_{\mathbb{R}^{p \times 1}} \cdot \underbrace{u^T}_{\mathbb{R}^{1 \times p}} | \\
&= | I_1 + u^T \cdot \frac{\lambda \mu}{1 + \mu} (\frac{1}{n} Y Y^T - \lambda I_p)^{-1} \cdot u | \\
&= 1 + \frac{\lambda \mu}{1 + \mu} u^T (\frac{1}{n} Y Y^T - \lambda I_p)^{-1} u \\
&= 1 + \frac{\lambda \mu}{1 + \mu} \langle u, (\frac{1}{n} Y Y^T - \lambda I_p)^{-1} u \rangle \\
&\phantom{= 1 + \frac{\lambda \mu}{1 + \mu} }S(\lambda) = \mathbb{E}\text{tr} (\frac{1}{n} Y Y^T - \lambda I_p)^{-1} \\
&\phantom{= 1 + \frac{\lambda \mu}{1 + \mu}S(\lambda) }= \mathbb{E}\frac{1}{p} \sum_{k=1}^{p} \underbrace{\langle u_k, (\frac{1}{n} Y Y^T - \lambda I_p)^{-1} u_k \rangle}_{\text{orthonormal basis } \{u_1, \dots, u_p\}} \\
&\phantom{= 1 + \frac{\lambda \mu}{1 + \mu}S(\lambda) }= \underbrace{\mathbb{E}\langle u_1, (\frac{1}{n} Y Y^T - \lambda I_p)^{-1} u_1 \rangle}_{\substack{\text{Since rotational symmetry,}\\ (\frac{1}{n} Y Y^T - \lambda I_p)^{-1} \\ \text{not perfers any direction}\\ \text{in expectation}}}\\
&\phantom{= 1 + \frac{\lambda \mu}{1 + \mu}S(\lambda) }\approx \langle u_1, (\frac{1}{n} Y Y^T - \lambda I_p)^{-1} u_1 \rangle\\
0&\approx 1 + \frac{\lambda \mu}{1 + \mu} S(\lambda) \\
\boxed{λS(λ)} &= −\frac{1+\mu}{\mu}
\end{aligned}
$$

***Step 3***
As we know, $\boxed{λS(λ)} = −\frac{1+\mu}{\mu}$, then we want to find connection between $\lambda,γ,\mu$, so we should find a way to substitute $\boxed{λS(λ)}$ by $\lambda,γ,\mu$.
In Chapter 4 [[4 Wishart Random Matrices#^wkgll1|formula]], we can get another formula:
$$
\begin{aligned}1 + \lambda S(\lambda) &= \frac{S(\lambda)}{1 + \gamma S(\lambda)}\\
S(\lambda) &= 1 + (\lambda + \gamma) S(\lambda) + \lambda \gamma S^2(\lambda)\\
1 + \gamma S(\lambda) &= S(\lambda) (1 + \lambda - \lambda \gamma S(\lambda))\\
\frac{1 + \gamma S(\lambda)}{S(\lambda)} &= 1 + \lambda - \lambda \gamma S(\lambda)\\
&\Downarrow\\
1 + \lambda S(\lambda) &= \frac{1}{1 - \lambda - \gamma \lambda S(\lambda)}\\
&\Downarrow \text{ By } \lambda S(\lambda) = -\frac{1 + \mu}{\mu}\\
-\frac{1}{\mu} = 1 - \frac{1 + \mu}{\mu} &= \frac{1}{1 - \lambda + \gamma \frac{1 + \mu}{\mu}}\\
1 - \lambda + \gamma + \frac{\gamma}{\mu} &= -\mu\\
\lambda &= 1 + \mu + \gamma \frac{\mu + 1}{\mu}
\end{aligned}
$$

[^1]:Jinho Baik and Jack W Silverstein, Eigenvalues of large sample covariance matrices of spiked population models, *Journal of multivariate analysis 97 (2006)*, no. 6, 1382–1408.
[^2]:Jinho Baik, Gérard Ben Arous, and Sandrine Péché, Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices, *Annals of Probability (2005)*, 1643–1697.


