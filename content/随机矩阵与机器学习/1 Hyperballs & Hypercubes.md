---
source: "[[High-Dimensional Analysis- Random Matrices and Machine Learning.pdf]]"
tags:
  - 随机矩阵与机器学习_random-matrices-and-machine-learning
  - 高维几何_high-dimensional-geometry
  - 超球_hyperball
  - 超立方体_hypercube
---

### 1.1 半径为 $R$ 的高维球的体积 $\text{vol}(B_p(R))$ 的表达式

Set
$$
\text{B}_p(R) := \{x = (t_1, \dots, t_p) \in \mathbb{R}^p : \underbrace{t_1^2 + \dots + t_p^2}_{=:\|x\|^2} \le R^2\},
$$
How about the volumes of hyperballs? We have
$$
\begin{aligned} \text{vol}(\text{B}_p(R)) &= \int \dots \int\limits_{t_1^2 + \dots + t_p^2 \le R} \text{d}t_1 \dots \text{d}t_p \\ &= \int \dots \int\limits_{(Rs_1)^2 + \dots + (Rs_p)^2 \le R} \text{d}(Rs_1) \dots \text{d}(Rs_p) \\ &= R^p \int \dots \int\limits_{s_1^2 + \dots + s_p^2 \le 1} \text{d}s_1 \dots \text{d}s_p \\ &= R^p \text{vol}(\text{B}_p(1)). \end{aligned}
$$
Set $B_p=B_p(1)$, then
$$
\begin{aligned} \text{vol}(\text{B}_p) &= \int \dots \int\limits_{t_1^2 + \dots + t_p^2 \le 1} \text{d}t_1 \dots \text{d}t_p \\ &= \int \dots \int\limits_{t_1^2 + \dots + t_{p-2}^2 + r^2 \sin^2(\varphi) + r^2 \cos^2(\varphi) \le 1} \text{d}t_1 \dots \text{d}t_{p-2} r \, \text{d}r \, \text{d}\varphi \\ &= \int_0^{2\pi} \int_0^1 \left( \int \dots \int\limits_{t_1^2 + \dots + t_{p-2}^2 \le 1-r^2} \text{d}t_1 \dots \text{d}t_{p-2} \right) r \, \text{d}r \, \text{d}\varphi \\ &= \int_0^{2\pi} \int_0^1 \text{vol}(\text{B}_{p-2}(\sqrt{1-r^2})) r \, \text{d}r \, \text{d}\varphi \\ &= \int_0^{2\pi} \int_0^1 \text{vol}(\text{B}_{p-2}) (1-r^2)^{\frac{p-2}{2}} r \, \text{d}r \, \text{d}\varphi \\ &= \text{vol}(\text{B}_{p-2}) \cdot 2\pi \cdot \int_0^1 r(1-r^2)^{\frac{p-2}{2}} \text{d}r \\ &= \text{vol}(\text{B}_{p-2}) \cdot 2\pi \cdot \underbrace{\left[ (1-r^2)^{\frac{p}{2}} \left( -\frac{1}{2} \cdot \frac{2}{p} \right) \right]_0^1}_{=\frac{1}{p}} \\ &= \frac{2\pi}{p} \text{vol}(\text{B}_{p-2}). \end{aligned}
$$
Iterating the recursion for even $p = 2k$ gives
$$
\begin{aligned}\text{vol}(\text{B}_{2k}) &= \frac{2\pi}{2k} \cdot \text{vol}(\text{B}_{2(k-1)}) = \frac{2\pi}{2k} \cdot \frac{2\pi}{2(k-1)} \cdot \text{vol}(\text{B}_{2(k-2)}) = \dots = \frac{\pi^k}{k!} \cdot \text{vol}(\text{B}_0)\\ &=\frac{\pi^k}{k!} \cdot 1=\frac{\pi^k}{k!} = \frac{\pi^{\frac{p}{2}}}{(\frac{p}{2})!} \quad \quad \text{for} \ p=2k \end{aligned}
$$
For odd $p = 2k + 1$ gives
$$
\begin{aligned} \text{vol}(\text{B}_{2k+1}) &= \frac{2\pi}{2k+1} \cdot \frac{2\pi}{2k-1} \cdot \dots \cdot \frac{2\pi}{3} \cdot \underbrace{\text{vol}(\text{B}_1)}_{=2} \\ &= \frac{\pi^k}{(k+\frac{1}{2}) \cdot (k-\frac{1}{2}) \cdot \dots \cdot \frac{3}{2} \cdot \frac{1}{2}} \\ &= \frac{\pi^{\frac{p}{2}}}{\frac{p}{2} \cdot (\frac{p}{2}-1) \cdot \dots \cdot \frac{3}{2} \cdot \frac{1}{2} \cdot \pi^{\frac{1}{2}}}. \end{aligned}
$$
In conclusion, we have

> [!theo] Theorem 1.1. ==The volume of the hyperball==
$$
\text{vol}(\text{B}_p(R)) = \frac{\pi^{\frac{p}{2}}}{\Gamma \left( \frac{p}{2} + 1 \right)} \cdot R^p,
$$
 By [[Gamma Function & Stirling Formula#Gamma Function]], which behaves asymptotically as
$$
\text{vol}(\text{B}_p(R)) \sim \frac{1}{\sqrt{p\pi}} \cdot \left( \frac{2\pi\text{e}R^2}{p} \right)^{\frac{p}{2}} \quad \text{for} \quad p \to \infty.
$$


> [!lemma] Corollary 1.2.
> (1) For any fixed radius $R > 0$:
$$
\lim_{p\to\infty} \text{vol}(\text{B}_p(R)) = 0.
$$
(2) In order for $\text{B}_p(R)$ to have volume 1, the radius has to scale with the dimension as $R \sim \sqrt{\frac{p}{2\pi\text{e}}}$.

![[1 Hyperballs & Hypercubes-1770376173963.webp]]

> [!remark]
> In high dimensions:
>*(i) the volume of the ball is concentrated close to its surface;*
>*(ii) almost all of the volume of the ball lies near its equator (note: there are many equators and this holds for all of them!);*
>*(iii) any two vectors in the ball are almost orthogonal.*
>In small dimensions these statements are clearly not true.

---

### 1.2 高维球体的体积主要集中在球表面

Fix $\varepsilon > 0$, then we have for the random vectors $x \in \mathrm{B}_p$ that
$$
\begin{aligned}
\mathrm{P}(\|x\| > 1 - \varepsilon) &= \frac{\text{vol}(\{x \in \mathrm{B}_p : \|x\| > 1 - \varepsilon\})}{\text{vol}(\mathrm{B}_p)} \\
&= \frac{\text{vol}(\mathrm{B}_p) - \text{vol}(\{x \in \mathrm{B}_p : \|x\| \leq 1 - \varepsilon\})}{\text{vol}(\mathrm{B}_p)} \\
&= 1 - \frac{\text{vol}(\mathrm{B}_p(1 - \varepsilon))}{\text{vol}(\mathrm{B}_p(1))} \\
&= 1 - (1 - \varepsilon)^p \xrightarrow{p \to \infty} 1 \quad \text{if } \varepsilon \text{ is fixed.}
\end{aligned}
$$

If one wants the volume in the "skin" constant, one has to scale $\varepsilon$ with $p$ like $\varepsilon = \frac{1}{p}$:
$$
\mathrm{P}\left(\|x\| > 1 - \frac{1}{p}\right) = 1 - \underbrace{\left(1 - \frac{1}{p}\right)^p}_{\approx \frac{1}{\mathrm{e}} \text{ for } p \text{ large}} \approx 1 - \frac{1}{\mathrm{e}} \approx 62\%.
$$
Often, one likes to write the concentration via exponential estimates in the dimension. For this one has the following estimate.

> [!lemma] Lemma 1.4
> For $p \geq 1$ and $0 < \varepsilon \leq 1$ we have $(1 - \varepsilon)^p \leq \exp(-\varepsilon p).$

$0 \leq 1 +(- \varepsilon) \leq \exp(-\varepsilon)$ by taking the $p$-th power.

Then we can write our concentration of the norm estimate as follows.
> [!theo] Theorem 1.5
> For $p \geq 1$ and $0 < \varepsilon \leq 1$ we have
> $$
> P(x \in \mathrm{B}_p : \|x\| > 1 - \varepsilon) = 1 - (1 - \varepsilon)^p \geq 1 - \exp(-\varepsilon p)
> $$
> and
> $$
> P\left(x \in \mathrm{B}_p : \|x\| > 1 - \frac{1}{p}\right) \geq 1 - \frac{1}{\mathrm{e}}.
> $$

---

### 1.3 高维球体的体积主要集中在极轴上

Consider $x = (t_1, \dots, t_p) \in \text{B}_p$ and choose arbitrarily $t_p$ as the north direction.
$$
\begin{aligned}
P(|t_p| \leq \varepsilon) &= \frac{\text{vol}(\{x \in \mathrm{B}_p \mid t_p \in (-\varepsilon, \varepsilon)\})}{\text{vol}(\mathrm{B}_p)} \\
&= \frac{1}{\text{vol}(\mathrm{B}_p)} \int_{-\varepsilon}^{\varepsilon} \int \dots \int_{t_1^2 + \dots + t_{p-1}^2 \leq 1-t^2} \mathrm{d}t_1 \dots \mathrm{d}t_{p-1} \, \mathrm{d}t \\
&= \frac{1}{\text{vol}(\mathrm{B}_p)} \int_{-\varepsilon}^{\varepsilon} \text{vol}(\mathrm{B}_{p-1}(\sqrt{1-t^2})) \, \mathrm{d}t \\
&= \frac{\text{vol}(\mathrm{B}_{p-1})}{\text{vol}(\mathrm{B}_p)} \int_{-\varepsilon}^{\varepsilon} (1-t^2)^{\frac{p-1}{2}} \, \mathrm{d}t \\
&= \frac{\frac{\pi^{\frac{p-1}{2}}}{\Gamma(\frac{p-1}{2}+1)}}{\frac{\pi^{\frac{p}{2}}}{\Gamma(\frac{p}{2}+1)}} \int_{-\varepsilon}^{\varepsilon} (1-t^2)^{\frac{p-1}{2}} \, \mathrm{d}t \\
&= \frac{1}{\sqrt{\pi}} \frac{\Gamma(\frac{p}{2}+1)}{\Gamma(\frac{p-1}{2}+1)} \int_{-\varepsilon}^{\varepsilon} (1-t^2)^{\frac{p-1}{2}} \, \mathrm{d}t \\
&\sim \frac{1}{\sqrt{\pi}} \frac{\sqrt{2\pi \frac{p}{2}} \left( \frac{p}{2\mathrm{e}} \right)^{\frac{p}{2}}}{\sqrt{2\pi \frac{p-1}{2}} \left( \frac{p-1}{2\mathrm{e}} \right)^{\frac{p-1}{2}}} \int_{-\varepsilon}^{\varepsilon} (1-t^2)^{\frac{p-1}{2}} \, \mathrm{d}t \\
&\sim \frac{1}{\sqrt{\pi}} \sqrt{\frac{p}{p-1}} \left( \frac{p}{p-1} \right)^{\frac{p}{2}} \left( \frac{p-1}{2\mathrm{e}} \right)^{\frac{1}{2}} \int_{-\varepsilon}^{\varepsilon} (1-t^2)^{\frac{p-1}{2}} \, \mathrm{d}t \\
&\sim \frac{1}{\sqrt{\pi}} \sqrt{\frac{p}{2\mathrm{e}}} {\underbrace{\left( 1 - \frac{1}{p} \right)^{-\frac{p}{2}}}_{\sim \exp\left(\frac{1}{2}\right) = \sqrt{\mathrm{e}}}} \int_{-\varepsilon}^{\varepsilon} (1-t^2)^{\frac{p-1}{2}} \, \mathrm{d}t \\ &\sim \frac{1}{\sqrt{\pi}} \sqrt{\frac{p}{2}} \int_{-\varepsilon}^{\varepsilon} (1-t^2)^{\frac{p-1}{2}} \, \mathrm{d}t \\ &\sim \frac{1}{\sqrt{\pi}} \sqrt{\frac{p}{2}} \int_{-\varepsilon}^{\varepsilon} (1-t^2)^{\frac{p}{2}} \, \mathrm{d}t \\ &= \frac{1}{\sqrt{\pi}} \int_{-\varepsilon\sqrt{\frac{p}{2}}}^{+\varepsilon\sqrt{\frac{p}{2}}} \underbrace{\left( 1 - \frac{2s^2}{p} \right)^{\frac{p}{2}}}_{\sim \exp(-s^2)} \, \mathrm{d}s \quad \left( \text{substituting } t = \frac{s}{\sqrt{p/2}} \right) \\ &\sim \frac{1}{\sqrt{\pi}} \int_{-\infty}^{+\infty} \exp(-s^2) \, \mathrm{d}s = 1

\end{aligned}
$$

---

### 1.4 高维向量几乎正交

Consider two vectors $x, y$ in the unit ball. What is the probability that they are close to being orthogonal? We have
$$
\begin{aligned}
P \left( x, y \in \mathrm{B}_p : \frac{|\langle x, y \rangle|}{\|x\| \cdot \|y\|} \leq \varepsilon \right) &= \frac{\text{vol} \left( \left\{ (x, y) \in \mathrm{B}_p \times \mathrm{B}_p : \frac{|\langle x, y \rangle|}{\|x\| \cdot \|y\|} \leq \varepsilon \right\} \right)}{\underbrace{\text{vol}(\mathrm{B}_p \times \mathrm{B}_p)}_{= \text{vol}(\mathrm{B}_p)^2}} \\
&= \frac{\text{vol} \left( \left\{ x \in \mathrm{B}_p : \frac{|\langle x, y \rangle|}{\|x\| \cdot \|y\|} \leq \varepsilon \right\} \right)}{\text{vol}(\mathrm{B}_p)} \ \text{for any fixed y } \in \mathrm{B}_p \\
&= \frac{\text{vol} \left( \left\{ x \in \mathrm{B}_p : \frac{|\langle x, y \rangle|}{\|x\| \cdot \|y\|} \leq \varepsilon \right\} \right)}{\text{vol}(\mathrm{B}_p)} \ \text{for} \ y = (0, \dots, 0, 1) \in \mathrm{B}_p \\
&= \frac{\text{vol} \left( \left\{ x \in \mathrm{B}_p : \frac{|x_p|}{\|x\|} \leq \varepsilon \right\} \right)}{\text{vol}(\mathrm{B}_p)} \\
&= \frac{\text{vol} \left( \left\{ x = (t_1, \dots, t_p) \in \mathrm{B}_p : \frac{|t_p|}{\|x\|} \leq \varepsilon \right\} \right)}{\text{vol}(\mathrm{B}_p)}.
\end{aligned}
$$

Define $\eta$ via $\frac{\eta}{1-\eta} = \varepsilon$ (note that $\varepsilon \approx \eta$ for small $\varepsilon$), then we know
$$
P(A)=P(|t_p| \leq \eta) \geq 1 - \exp \left( -\frac{p-1}{2} \cdot \eta^2 \right) \cdot \sqrt{2\pi}=a
$$
$$
P(B)=P(\|x\| > 1 - \eta) \geq 1 - \exp (-p\eta)=b
$$
thus
$$
\begin{aligned} \mathbb{P}(A \cap B) &=P(|t_p| \leq \eta \text{ and } \|x\| > 1 - \eta)= 1 - \mathbb{P}((A \cap B)^c) = 1 - \mathbb{P}(A^c \cup B^c) \\ &\geq 1 - \mathbb{P}(A^c) - \mathbb{P}(B^c) \\
&\geq 1 - a - b=\exp \left( -\frac{p-1}{2} \cdot \eta^2 \right) \cdot \sqrt{2\pi} - \exp (-p\eta) \end{aligned}
$$
Since $|t_p| \leq \eta$ and $\|x\| > 1 - \eta$ together imply
$$
\frac{|t_p|}{\|x\|} \leq \frac{\eta}{1-\eta} = \varepsilon,
$$
we have

$$
\begin{aligned}
P \left( x, y \in \mathrm{B}_p : \frac{|\langle x, y \rangle|}{\|x\| \cdot \|y\|} \leq \varepsilon \right)
&\geq P(|t_p| \leq \eta \text{ and } \|x\| > 1 - \eta) \\
&\geq 1 - \exp \left( -\frac{p-1}{2} \cdot \eta^2 \right) \cdot \sqrt{2\pi} - \exp (-p\eta). \\
\end{aligned}
$$

***END***
