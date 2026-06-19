---
draft: true
---
# Chapter 6: Direct Methods for Linear Systems

"Solving linear equations is the single most important problem of linear algebra and probably of applied mathematics in general." — K. Lange [28]

Most computational problems in science and engineering ultimately reduce to solving linear systems of equations. Gaussian elimination, introduced in Chapter 1.5, is one popular method for solving linear systems. A direct method yields the exact solution (in exact arithmetic) within a finite number of computational steps. In contrast, an iterative method produces an approximation solution after a finite number of iterations.

In this chapter, we introduce a direct method called LU factorization for solving linear systems $Ax = b$ with an arbitrary square matrix $A$. This method is particularly well-suited for dense, medium-sized linear systems. Essentially, the factorization can be viewed as the matrix formulation of Gaussian elimination.

## 6.1 Review

In this section, we recall some basic knowledge of matrix computations and then discuss the operation cost of Cramer's rule and Gaussian elimination.

### 6.1.1 Basic knowledge

Let $\mathbb{R}^n$ ($\mathbb{C}^n$) denote the set of real (complex) $n$-vectors. Vectors are always columns of the vector space $\mathbb{R}^{n \times n}$ ($\mathbb{C}^{n \times n}$) to denote all real (complex) $n \times n$ matrices. Matrices are denoted by upper-case letters $A, B, C, D, \ldots$, and vectors are denoted by lowercase letters $a, b, c, \ldots$, etc.

Let $(A)_{ij} = a_{ij}$ denote the $(i, j)$ entry of a matrix $A = [a_{ij}]$. An $n \times n$ diagonal matrix is denoted by

$$\text{diag}(a_{11}, a_{22}, \ldots, a_{nn}) = \begin{bmatrix} a_{11} & 0 & \cdots & 0 \\ 0 & a_{22} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & a_{nn} \end{bmatrix}.$$

We use $I$ for the identity matrix. The symbol $\mathbf{0}$ denotes the $n$ column of $I$.

Recall that a matrix $A \in \mathbb{R}^{n \times n}$ is symmetric if $A^T = A$, where $A^T$ is the transpose. A matrix $A \in \mathbb{C}^{n \times n}$ is Hermitian if $A^* = A$, where $A^*$ is the conjugate transpose of $A$. A Hermitian matrix $A \in \mathbb{C}^{n \times n}$ is positive definite if $\langle x, Ax \rangle > 0$ for any nonzero vector $x \in \mathbb{C}^n$. A matrix $A \in \mathbb{C}^{n \times n}$ is Hermitian if $A^* = A$. The set of eigenvalues of $A$ (if they exist a nonzero vector $x \in \mathbb{C}^n$ such that $Ax = \lambda x$, where $\lambda \in \mathbb{C}$. Let $\lambda_{\min}(A)$ and $\lambda_{\max}(A)$ denote the smallest and largest eigenvalue of a Hermitian matrix $A$ respectively. All eigenvalues of any Hermitian matrix are real [18]. Let $\sigma_k(A)$ and $\sigma_{\max}(A)$ denote the $k$-th and largest singular value of a matrix, respectively.

### 6.1.2 Operation cost of Cramer's rule and Gaussian elimination

Consider a linear system $Ax = b$, where $A = [a_{ij}]$ is an $n \times n$ nonsingular (invertible) matrix and $x = [x_1, x_2, \ldots, x_n]^T$ is an unknown vector. Between Cramer's rule and Gaussian elimination [21, 22] significantly impacts computational efficiency. Below is a detailed comparison of their operation costs.

If we use Cramer's rule, then the solution is given by

$$x_1 = \frac{\det(A_1)}{\det(A)}, \quad x_2 = \frac{\det(A_2)}{\det(A)}, \quad \ldots, \quad x_n = \frac{\det(A_n)}{\det(A)},$$

where $A_1, A_2, \ldots, A_n$ are $n \times n$ matrices obtained by replacing the $i$-th column of $A$ by the vector $b$. Hence we should compute $n + 1$ determinants.

Recall that for an $n \times n$ matrix $M = [m_{ij}]$, the determinant of $M$ is defined by

$$\det(M) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \, m_{1\sigma(1)} m_{2\sigma(2)} \cdots m_{n\sigma(n)},$$

where $(i_1, i_2, \ldots, i_n)$ is a permutation of the set $\{1, 2, \ldots, n\}$. Therefore, $\det(n) = n!$ multiplications are needed in computation of $n + 1$ determinants.

For instance, when $n = 25$, by using a computer with 10 billion operations per second, it costs

$$\frac{26 \times 25!}{10^9 \times 3600 \times 24 \times 365} \approx 39.6 \text{ billion years}$$

to solve the linear system.

By using Gaussian elimination, we need to transform the $n \times n + 1$ augmented matrix $[A \mid b]$ to (reduced) row-echelon form. Suppose that $a_{11} \neq 0$, otherwise we can interchange the top row with another, and the other entries below row 1 become zero. If $a_{1j} \neq 0$ for $2 \le j \le n$, it requires $(n - 1)$ multiplications to the first equation of Gaussian elimination is given by

$$\sum_{i=1}^{n-1} (i(n - i)) = \sum_{i=1}^{n-1} in - \sum_{i=1}^{n-1} i^2 = \sum_{i=1}^n i^2 = n \left( n + 1 \right)(2n + 1) - n = O(n^3).$$

Here, the notation $O(n^4)$ indicates that it grows at the same rate as $n^3$.

Resolving the above 25-by-25 linear system via Gaussian elimination on the same machine from a computational perspective, Gaussian elimination is generally preferred over Cramer's rule for solving linear systems.

## 6.2 LU factorization

For a nonsingular matrix $A$, if $A$ can be factorized as $A = LU$, where $L$ is a lower triangular matrix and $U$ is an upper triangular matrix, then $A$ of [1]. In fact, the LU factorization is essentially the matrix formulation of Gaussian elimination. By LU factorization, one can find the solution of $Lx = b$ and $Ux = y$. Therefore, we begin our treatment of [1] with triangular linear systems.

### 6.2.1 Triangular linear systems

We first consider the following nonsingular lower triangular linear system

$$Ly = b,$$

where $b = [b_1, b_2, \ldots, b_n]^T$ is a given vector, $y = [y_1, y_2, \ldots, y_n]^T$ is an unknown vector, and $L = [l_{ij}] \in \mathbb{R}^{n \times n}$ is given by

$$L = \begin{bmatrix} l_{11} & 0 & \cdots & 0 \\ l_{21} & l_{22} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ l_{n1} & l_{n2} & l_{n3} & \cdots & l_{nn} \end{bmatrix},$$

with $l_{ij} \neq 0$ for $1 \le i \le n$. More precisely, system (6.2) can be written as

$$\begin{cases} l_{11}y_1 = b_1 \\ l_{21}y_1 + l_{22}y_2 = b_2 \\ l_{31}y_1 + l_{32}y_2 + l_{33}y_3 = b_3 \\ \vdots \\ l_{n1}y_1 + l_{n2}y_2 + l_{n3}y_3 + \cdots + l_{nn}y_n = b_n. \end{cases}$$

By using the first equation in (6.3), we have

$$y_1 = \frac{b_1}{l_{11}}.$$

Similarly, by the second equation in (6.3), we obtain

$$y_2 = \frac{1}{l_{22}} \left( b_2 - l_{21}y_1 \right).$$

This algorithm is called forward substitution which requires $O(n^2)$ operations.

Next, we consider the following nonsingular upper triangular linear system

$$Ux = y,$$

where $x = [x_1, x_2, \ldots, x_n]^T$ is an unknown vector and $U \in \mathbb{R}^{n \times n}$ is given by

$$U = \begin{bmatrix} u_{11} & u_{12} & u_{13} & \cdots & u_{1n} \\ 0 & u_{22} & u_{23} & \cdots & u_{2n} \\ 0 & 0 & u_{33} & \cdots & u_{3n} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & u_{nn} \end{bmatrix},$$

with $u_{ij} \neq 0$ for $1 \le i \le n$. System (6.4) can be written as

$$\begin{cases} u_{11}x_1 + u_{12}x_2 + u_{13}x_3 + \cdots + u_{1n}x_n = y_1 \\ u_{22}x_2 + u_{23}x_3 + \cdots + u_{2n}x_n = y_2 \\ u_{33}x_3 + \cdots + u_{3n}x_n = y_3 \\ \vdots \\ u_{nn}x_n = y_n. \end{cases}$$

Beginning with the last equation in (6.5), we can obtain $x_n, x_{n-1}, \ldots, x_1$ step by step. $x_n = \frac{y_n}{u_{nn}}$, and $x_i$ is given by

$$x_i = \frac{1}{u_{ii}} \left( y_i - \sum_{j=i+1}^{n} u_{ij}x_j \right),$$

for $i = n - 1, \ldots, 2, 1$. This algorithm is called backward substitution which also requires $O(n^2)$ operations.

For a general linear system

$$Ax = b,$$

where $A \in \mathbb{R}^{n \times n}$ and $b \in \mathbb{R}^n$ are given, if $A$ can be factorized as $A = LU$, then the solution of (6.6) can be found by the following two steps:

(i) Find the solution of $Ly = b$ by forward substitution.

(ii) Find the solution of $Ux = y$ by backward substitution.

### 6.2.2 Gaussian transform matrix

Let

$$L_k = I - L_k e_k^T,$$

where $I \in \mathbb{R}^{n \times n}$ is the identity matrix, $L_k = [0, \ldots, 0, l_{k+1,k}, \ldots, l_{n,k}]^T \in \mathbb{R}^n$, and $e_k$ is the $k$-th column of $I$. Then for each $k$:

$$L_k = \begin{bmatrix} 1 & 0 & \cdots & \cdots & 0 \\ \vdots & \ddots & \vdots & \vdots & \vdots \\ \vdots & \vdots & 1 & \vdots & \vdots \\ \vdots & \vdots & -l_{k+1,k} & \ddots & \vdots \\ \vdots & \vdots & -l_{n,k} & \cdots & 1 \end{bmatrix}.$$

We call a Gaussian transform matrix. Such a matrix is a unit lower triangular matrix with nonzero diagonal entries being all 1.

For any given vector $x = [x_1, x_2, \ldots, x_n]^T$, we have

$$L_k x = [x_1, \ldots, x_k, x_{k+1} - l_{k+1,k} x_k, \ldots, x_n - l_{n,k} x_k, \ldots, 0]^T,$$

provided $l_n = l/x_i, \text{ for } k + 1 \le i \le n$ with $z_i \neq 0$. It is easy to check that

$$L_k^{-1} = I + L_k e_k^T.$$

By noting that $e_k^T L_k = 0$, we can use Gaussian transform matrix $L_k = I - L_k e_k^T$ where

$$L_k = [0, \ldots, 0, l_{k+1,k}, \ldots, l_{n,k}]^T \in \mathbb{R}^n, \quad l_n = l/x_i, \text{ for } k + 1 \le i \le n,$$

such that the last $n - k$ entries in the $k$-th column of $L_k A^{(k-1)}$ become zeros. Therefore, we have

$$\begin{bmatrix} A_k^{(k-1)} & A_k^{(k-1)} \\ 0 & A_{22}^{(k)} \end{bmatrix},$$

where $A_k^{(k)}$ is a $k \times k$ upper triangular matrix. After $n - 1$ steps, an upper triangular matrix $A^{(n-1)}$ is obtained. Let

$$L = (L_n - ... L_2L_1)^{-1}, \quad U = A^{(n-1)}.$$

Then $A = LU$. Now we show that $L$ is a unit lower triangular matrix. Noting that $\Sigma L_i^{-1} L_i^{-1} = (I + L_k e_k^T) \Sigma (I + L_k e_k^T)$, it follows that

$$\begin{bmatrix} A_k^{(k-1)} & A_k^{(k-1)} \\ 0 & A_{22}^{(k)} \end{bmatrix},$$

where

$$A_{22}^{(k)} = \begin{bmatrix} 0 & \cdots & * \\ \vdots & \ddots & \vdots \\ 0 & \cdots & 0 \end{bmatrix} + \begin{bmatrix} a_{22}^{(2)} & \cdots & * \\ \vdots & \ddots & \vdots \\ 0 & \cdots & a_{n,n-1}^{(n-1)} \end{bmatrix}.$$

### 6.2.3 Computation of LU factorization

Let us first consider a simple example before studying the LU factorization in the general case.

***Example 6.1.*** Compute the LU factorization of

$$A = \begin{bmatrix} 1 & 4 & 5 \\ 2 & 6 & 16 \\ 3 & 9 & 26 \end{bmatrix}.$$

***Solution.*** By using the Gaussian transform matrix

$$L_1 = \begin{bmatrix} 1 & 0 & 0 \\ -2 & 1 & 0 \\ -3 & 0 & 1 \end{bmatrix},$$

we have

$$L_1 A = \begin{bmatrix} 1 & 4 & 5 \\ 0 & -2 & 6 \\ 0 & -3 & 27 \end{bmatrix}.$$

Followed by using the Gaussian transform matrix

$$L_2 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & -4 & 1 \end{bmatrix},$$

we obtain

$$L_2 L_1 A = \begin{bmatrix} 1 & 4 & 5 \\ 0 & -2 & 6 \\ 0 & -8 & 27 \end{bmatrix}.$$

Finally, we derive $A = LU$, where

$$L = (L_2 L_1)^{-1} = L_1^{-1} L_2^{-1} = \begin{bmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 3 & 4 & 1 \end{bmatrix}, \quad U = \begin{bmatrix} 1 & 4 & 5 \\ 0 & -2 & 6 \\ 0 & -8 & 27 \end{bmatrix}.$$

Based on the above discussion, we have the following MATLAB-style pseudocode for the algorithm:

***Algorithm 6.1.*** (LU factorization)

$$\begin{aligned}
&\text{for } k = 1 : n - 1 \\
&\quad \text{for } i = k + 1 : n, l(i, k) = a(i, k)/a(k, k) \\
&\quad \text{for } j = k + 1 : n \\
&\quad \quad a(i, j) = a(i, j) - l(i, k) \cdot a(k, j) \\
&\quad \text{end} \\
&\text{end}
\end{aligned}$$

For a general $n \times n$ matrix and from the pseudocode, the number of multiplications and divisions in LU factorization is

$$\sum_{k=1}^{n-1} (n - k) = \sum_{j=1}^{n-1} j = \sum_{j=1}^{n-1} j = n - \frac{1}{3}n(n + 1)(2n + 1) - n = O(n^3).$$

Note that in an LU factorization, the diagonal entries $a_{ii}^{(k)}, 1 \le k \le n - 1$, are required to be non-zero. We have the following theorem:

***Theorem 6.1.*** The entries $a_{11}^{(0)}, a_{22}^{(1)}, \ldots, a_{nn}^{(n-1)}$ are nonzero if and only if all the leading principal submatrices $A_1, A_2, \ldots, A_n$ of $A$ are nonsingular.

***Proof.*** By induction, for $k = 1$, it is obviously true. Assume that the statement is true for $k - 1$. Under the condition that $A_1, A_2, \ldots, A_{k-1}$ are nonsingular, we show that $A_k$ is nonsingular if and only if $a_{kk}^{(k-1)} \neq 0$. It follows from the induction hypothesis that $A_1, A_2, \ldots, A_{k-1}$ are nonsingular. By the $k - 1$ Gaussian transform matrices $L_1, L_2, \ldots, L_{k-1}$, we obtain

$$A^{(k-1)} = L_{k-1} \cdots L_2 L_1 A = \begin{bmatrix} A_1^{(k-1)} & A_{12}^{(k-1)} \\ 0 & A_{22}^{(k-1)} \end{bmatrix},$$

where

$$A_{11}^{(k-1)} = \begin{bmatrix} a_{11}^{(0)} & \cdots & * \\ \vdots & \ddots & * \\ 0 & \cdots & a_{kk}^{(k-1)} \end{bmatrix}.$$

Since $L_i, L_2, \ldots, L_{k-1}$ are unit lower triangular matrices, we immediately know that

$$\det(A_k) = a_{11}^{(0)} \det(A_{22}^{(k-1)}) \neq 0 \quad \iff \quad a_{kk}^{(k-1)} \neq 0.$$

Then, we have

***Theorem 6.2.*** (LU factorization) If all the leading principal submatrices of a square matrix $A$ are nonsingular, then there exists a unique factorization of $A$ in the form $A = LU$, where $L$ is a unit lower triangular matrix and $U$ is an upper triangular matrix.

## 6.3 LU factorization with pivoting

Prior to discussing the pivoting technique, we study the LU factorization in the form $P_1 A P_2 = LU$, where $P_1$ and $P_2$ are permutation matrices.

### 6.3.1 LU factorization with permutations

If a nonsingular matrix does not satisfy the conditions in Theorem 6.2, permutation strategies become necessary in the process of LU factorization. Therefore, we first introduce the following definition:

***Definition 6.1.*** A permutation matrix $P$ is obtained by permuting the rows or columns of an identity matrix.

For instance,

$$A = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{bmatrix}.$$

Some key properties of permutation matrices are included in the following lemma. The proof of the lemma is straightforward.

***Lemma 6.1.*** Let $P$ and $P_b$ be in $n \times n$ permutation matrices and $X$ be any $n \times n$ matrix. Then:

(a) $PX$ permutes the rows of $X$ and $XP$ permutes the columns of $X$.

(b) $P^{-1} = P^T$.

(c) $\det(P) = 1$.

(d) $PP$ is also a permutation matrix.

***Theorem 6.3.*** Let $A \in \mathbb{R}^{n \times n}$ be nonsingular. Then there exist right-hand left-hand permutation matrices $P_1$ and $P_2$, a unit lower triangular matrix $L$, and an upper triangular matrix $U$ such that

$$P_1 A P_2 = LU.$$

***Proof.*** We use induction on $n$. If $n = 1$, it is obviously true. Assume that $A$ is an $n \times n$ nonsingular matrix. It has a nonzero entry. Choose permutation matrices $P_1$ and $P_2$ such that the (1, 1) th position of $P_1 A P_2$ is nonzero. Let

$$P_1 A P_2 = \begin{bmatrix} a_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix},$$

where $A_{22}$ is an $(n - 1) \times (n - 1)$ matrix, and $A_2^2$ are $(n - 1) \times 1$-matrices. We then seek a factorization of the form

$$P_1 A P_2 = \begin{bmatrix} 1 & 0 \\ L_{21} & I \end{bmatrix} \begin{bmatrix} u_{11} & U_{12} \\ L_{21}u_{11} & L_{21}u_{12} + A_{22} \end{bmatrix},$$

where $L_{21}u_{11} = A_{21}$ and $u_{11} = A_{11}$. With $u_{12} = A_{12}$ and $A_{22}$ (cf. (6.9)) we get a factorization of the form

$$P_1 A P_2 = \begin{bmatrix} 1 & 0 \\ L_{21} & I \end{bmatrix} \begin{bmatrix} u_{11} & U_{12} \\ 0 & A_{22}' \end{bmatrix} = \begin{bmatrix} u_{11} & U_{12} \\ L_{21}u_{11} & L_{21}u_{12} + A_{22} \end{bmatrix},$$

Let $L_k, L_2, \ldots, L_{k-1}$ denote the $k$-th loading principal submatrices of $L_k, L_2, \ldots, L_{k-1}$, it follows from (6.8) that

$$(L_k, \ldots) (L_2)(L_1) A_k = \begin{bmatrix} A_{11}^{(k-1)} & A_{12}^{(k-1)} \\ 0 & A_{22}^{(k-1)} \end{bmatrix}.$$

Since $L_k, L_2, \ldots, L_{k-1}$ are unit lower triangular matrices, we immediately know that

$$\det(A_k) = a_{11}^{(0)} \det(A_{22}^{(k-1)}) \neq 0 \quad \iff \quad \det(A_{22}) \neq 0.$$

Then, we have

***Theorem 6.4.*** Let $A \in \mathbb{R}^{n \times n}$ be symmetric positive definite. Then there exists a right to-sign permutation matrix $P_1$ and $P_2$, a unit lower triangular matrix $L$, and an upper triangular matrix $U$ such that

$$P_1 A P_2 = LU.$$

Obviously, $P_1$ and $P_2$ are permutation matrices. Consequently, a desired LU factorization of $A$ is obtained:

$$P_1 A P_2 = \begin{bmatrix} 1 & 0 \\ L_{21} & I \end{bmatrix} \begin{bmatrix} u_{11} & U_{12} \\ 0 & U_{22}' \end{bmatrix}.$$

### 6.3.2 Pivoting technique

The LU factorization with pivoting is one of the most fundamental algorithms in numerical linear algebra. We begin with the following simple example.

***Example 6.2.*** Solve

$$\begin{bmatrix} 5 \times 10^{-17} & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$

by LU factorization.

***Solution.*** By LU factorization with the 15-digit decimal floating-point arithmetic, we have

$$L = \begin{bmatrix} 1 & 0 \\ 2 \times 10^{-17} & 1 \end{bmatrix}, \quad U = \begin{bmatrix} 5 \times 10^{-17} & 1 \\ 0 & -2 \times 10^{16} \end{bmatrix}.$$

Then the numerical solution is

$$x = [L_2 x_2]^T = [0, 1]^T,$$

which is not a good approximation to the exact solution

$$x = [x_1, x_2]^T = \left[ \frac{1 - 10^{-16}}{1 - 2 \times 10^{-17}}, \frac{1 - 10^{-16}}{1 - 2 \times 10^{-17}} \right]^T.$$

Interchanging the first and the second equation yields

$$\begin{bmatrix} 1 & 1 \\ 5 \times 10^{-17} & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}.$$

Then the numerical solution is $x = [1, 1]^T$, which is a good approximation to the exact solution.

In the LU factorization, the diagonal entry selected during the elimination process is called the pivot. Even if the pivot is nonzero but very small, it can result in severe round-off errors. See Example 6.2. To mitigate such errors, we need to find the entry with the maximum modulus. This strategy is called complete pivoting. The corresponding algorithm is outlined below.

***Algorithm 6.2.*** (LU factorization with complete pivoting)

$$\begin{aligned}
&\text{for } k = 1 : n - 1 \\
&\quad \text{choose } p, q \text{ }(k \le p \le n) \text{ and} (k \le q \le n) \text{ such that } |A(k, q)| \ge \max_{i \ge k, j \ge k} |A(i, j)| \\
&\quad \text{end if } A(k, k) \neq 0 \\
&\quad \text{else swap} \\
&\quad \quad A(k, :) \leftrightarrow A(p, :) \\
&\quad \quad A(:, k) \leftrightarrow A(:, q) \\
&\quad \text{end} \\
&\quad \text{for } i = k + 1 : n \\
&\quad \quad \text{compute} p = P \\
&\quad \quad A(i : n, k) = A(i : n, k) - A(i : n, 1 : k - 1) / A(k, k) \\
&\quad \text{end} \\
&\text{end}
\end{aligned}$$

It is clear that the operation cost of the Cholesky factorization remains $O(n^3)$.

## 6.4 Cholesky factorization

We now consider the LU factorization for symmetric positive definite matrices.

***Theorem 6.4.*** Let $A \in \mathbb{R}^{n \times n}$ be symmetric positive definite. Then there exists a lower triangular matrix $L \in \mathbb{R}^{n \times n}$ with positive diagonal entries such that

$$A = LL^T.$$

This factorization is called the Cholesky factorization.

***Proof.*** Since $A$ is positive definite, all the principal submatrices of $A$ positive definite. It follows from Theorem 6.2 that there exist unit lower triangular matrix $L$ and an upper triangular matrix $U$ such that

$$A = LU.$$

Let

$$D = \text{diag}(u_{11}, u_{22}, \ldots, u_{nn}),$$

where $u_{11}, u_{22}, \ldots, u_{nn}$ are diagonal entries of $U$. We then have

$$U^T D U^T = A^T = A = LDU.$$

Therefore,

$$U^T D = D U^T L D.$$

Note that $LU^{-1}$ is a unit upper triangular matrix and $D^{-1} U D$ is a lower triangular matrix. Hence

$$U^T - 1 = D^{-1} U L D.$$

which implies $U = L^T$. Then

$$A = LD L^T.$$

The matrix $D$ is positive definite because $A$ is positive definite. It follows that

$$u_{ii} > 0 \text{ for all } i. \text{ Let } L = D \cdot \text{diag}(\sqrt{u_{11}}, \sqrt{u_{22}}, \ldots, \sqrt{u_{nn}}).$$

We finally have

$$A = LL^T.$$

Thus, when a matrix $A$ is symmetric positive definite, we can find the solution of the system $Ax = b$ by the following three steps:

(i) Find the Cholesky factorization of $A = LL^T$.

(ii) Find the solution of $Ly = b$.

(iii) Find the solution of $L^T x = y$.

In fact, the matrix $L$ can be computed directly by computing corresponding elements in the equation $A = LL^T$. We have the following algorithm.

***Algorithm 6.3.*** (Cholesky factorization)

$$\begin{aligned}
&\text{for } k = 1 : n \\
&\quad A(k : n, k) = A(k : n, k) - A(k : n, 1 : k - 1) / A(k, k) \\
&\quad \text{for } j = k + 1 : n \\
&\quad \quad A(j : n, j) = A(j : n, j) - A(j : n, 1 : k - 1) A(j, j) - A(j : n, 1 : k - 1) A(j : n, 1 : k - 1)^T \\
&\quad \text{end}
\end{aligned}$$

It is clear that the operation cost of the Cholesky factorization remains $O(n^3)$.

## Exercises

6.1. Prove Lemma 6.1.

6.2. Let $L$ be a nonsingular symmetric matrix and $A = LDM^T$, where $L$ and $M$ are unit lower triangular matrices and $D$ is a diagonal matrix. Show that $L = M$.

6.3. Let $A$ be symmetric positive definite. Show that:

(a) $u_k > 0$ for all $k$; and max $|a_{ij}| = \max_{k} > 0$.

(b) $|a_{ij}| < (a_{ii}a_{jj})^{1/2}$.

6.4. Show that the $LDL^T$ factorization of a symmetric positive definite matrix $A$ is unique; where $L$ is a unit lower triangular matrix.

6.5. Let $A$ be symmetric positive definite. Find an algorithm for computing an upper triangular matrix $U$ such that $A = U^T U$.

6.6. Let $A = [a_{ij}] \in \mathbb{R}^{n \times n}$ be a strictly diagonally dominant matrix, satisfying

$$|a_{ii}| > \sum_{j \neq i} |a_{ij}|$$

for $1 \le i \le n$. Show that:

(a) $A$ is nonsingular.

(b) If $A$ is symmetric with positive diagonal entries, then $A$ is positive definite.

6.7. Let

$$A = \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix},$$

with $A_{11}$ being a $k \times k$ nonsingular matrix. Then

$$S = A_{22} - A_{21}A_{11}^{-1}A_{12}$$

is called the Schur complement of $A_{11}$ in $A$. Show that after $k$ steps of an LU factorization without pivoting, $A_{22}^{(k)} = S$.

6.8. Let $A$ be a symmetric positive definite matrix. At the end of the first step of an LU factorization, we have

$$\begin{bmatrix} u_{11} & a_1^T \\ 0 & A_{22} \end{bmatrix}.$$

Prove that $A_{22}$ is also symmetric positive definite.

6.9. Let $A$ be a strictly diagonally dominant matrix. After one step of an LU factorization, we have

$$\begin{bmatrix} u_{11} & u_{12} \\ 0 & A_{22} \end{bmatrix}.$$

Show that $A_{22}$ is also strictly diagonally dominant.

6.10. Show that if $PA Q = LU$ is obtained via LU factorization with complete pivoting, where $U = [u_{ij}]$, then

$$|u_{ij}| \ge |u_{ij}|$$

for $1 \le i \le j \le n$.

6.11. Let $H = A + iB$ be a Hermitian positive definite matrix, where $A, B \in \mathbb{R}^{n \times n}$, and $i := \sqrt{-1}$.

(a) Show that

$$C = \begin{bmatrix} A & -B \\ B & A \end{bmatrix}$$

is symmetric positive definite.

(b) How to solve

$$(A + iB)(x + iy) = b + ic, \quad x, y, b, c \in \mathbb{R}^n$$

by real number computation only?

6.12. Show that if a singular matrix $A \in \mathbb{R}^{n \times n}$ has a unique LU factorization, then the leading principal submatrix $A_k$ of $A$ is nonsingular for $1 \le k \le n - 1$.

6.13. Let $S, T \in \mathbb{R}^{n \times n}$ be upper triangular matrices such that

$$(ST - \lambda I)x = b$$

is a nonsingular system. Find an algorithm of $O(n^2)$ operations for computing $x$. Refer to [25].
