---
draft: true
tags:
  - 线性代数
  - 行列式
  - Sylvester恒等式
  - Schur补
  - linear-algebra
  - determinant
  - sylvester-identity
  - schur-complement
---

>[!lemma] Lemma.  (Sylvester’s Determinant Identity). 
>Consider $A \in \mathbb{R}^{p \times n}$ and $B \in \mathbb{R}^{n \times p}$. Then
$$\det(I_p + AB) = \det(I_n + BA).$$ ^icw6w8

***Proof.*** 
Use “Schur complement”: ^4azsbm
$$
\begin{pmatrix} I_p & 0 \\ -B & I_n \end{pmatrix} \begin{pmatrix} I_p & A \\ B & I_n \end{pmatrix} \begin{pmatrix} I_p & -A \\ 0 & I_n \end{pmatrix} = \begin{pmatrix} I_p & 0 \\ 0 & I_n - BA \end{pmatrix}
$$

$$
\left( \begin{array}{cc|cc} I_p & 0 & I_p & \\ -B & I_n & & I_n \end{array} \right) \to \left( \begin{array}{cc|cc} I_p & 0 & I_p & 0 \\ -B + B I_p & I_n & B I_p & I_n \end{array} \right) \to \begin{pmatrix} I_p & 0 \\ B & I_n \end{pmatrix}
$$

$$
\begin{aligned}
\begin{pmatrix} I_p & A \\ B & I_n \end{pmatrix} &= \begin{pmatrix} I_p & 0 \\ B & I_n \end{pmatrix} \begin{pmatrix} I_p & 0 \\ 0 & I_n - BA \end{pmatrix} \begin{pmatrix} I_p & A \\ 0 & I_n \end{pmatrix} \\
&= \begin{pmatrix} I_p & A \\ 0 & I_n \end{pmatrix} \begin{pmatrix} I_p - AB & 0 \\ 0 & I_n \end{pmatrix} \begin{pmatrix} I_p & 0 \\ B & I_n \end{pmatrix}.
\end{aligned}
$$

Then calculate it. $\boxed{\mathbf{QED}}$

