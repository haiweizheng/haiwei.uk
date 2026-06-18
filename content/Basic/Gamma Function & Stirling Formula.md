---
tags:
  - 分析
  - 特殊函数
  - Gamma函数
  - Stirling公式
  - analysis
  - special-functions
  - gamma-function
  - stirling-formula
---

### Stirling Formula

> [!lemma] Lemma Stirling approximation
$$n! = \sqrt{2\pi n} \left( \frac{n}{\text{e}} \right)^n (1 + o(1)) \quad \text{as } n \to \infty.$$ 

### Gamma Function

>[!lemma] Gamma function
>The *gamma function* extends the notion of factorial for all real numbers, and even for all complex numbers $z$ whose real part is positive. 
>$$\Gamma(z) := \int_0^\infty t^{z-1}\text{e}^{-t}\text{d}t. \tag{1.30}$$
>And
>$$\Gamma(n+1) =n\cdot\Gamma(n) =n! \quad \Gamma(1) = 1\quad \Gamma(\frac12) = \sqrt\pi $$
Stirling approximation  remains valid for the gamma function:
>$$\Gamma(z+1) = \sqrt{2\pi z} \left( \frac{z}{\text{e}} \right)^z (1 + o(1)) \quad \text{as } \mathbb{R} \ni z \to \infty. \tag{1.31}$$ ^nlk37x

Thus
$$\Gamma(n) \approx \sqrt{2\pi (n-1)} \left( \frac{n-1}{e} \right)^{n-1}\Gamma(n) \approx \sqrt{\frac{2\pi}{n}} \left( \frac{n}{e} \right)^{n}$$
$$\ln \Gamma(n) \approx n \ln n - n$$
### Note

Note from GTM295/Durreet
![[Gamma Function & Striling Formula-1770390779945.webp|574x505]]
![[Gamma Function & Striling Formula-1770390690150.webp|574x672]]