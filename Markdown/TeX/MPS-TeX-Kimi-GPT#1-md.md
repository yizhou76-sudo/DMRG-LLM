# MPS-TeX-Kimi-GPT#1-md

## 🧑‍💻 User

This is a latex note generated from the attached review article. As a computational and theoretical physicist, could you review the latex note equation by equation, then check whether it is ready for practical coding?
After that, please give me a fully correct LaTex file, changing the authors to Kimi 2.5, GPT 5.4, and state that it is based on this review article.

---
## 🤖 Assistant

Below is a careful technical review of your note, cross-checked against Schollwöck’s 2011 review and standard MPS/DMRG practice, followed by a corrected LaTeX file.

## Bottom line

**Short answer:**  
Your note is **not yet ready for practical coding as written**. It contains a lot of useful implementation intuition, but also several issues:

1. **Some equations are not mathematically correct as written.**
2. **Several algorithmic statements mix correct ideas with dangerous or misleading implementation details.**
3. **A few einsum templates and canonicalization details are wrong/inconsistent.**
4. **Some MPO boundary formulas are incorrect.**
5. **Some claims about canonicality, QR with pivoting, and right-canonical tensors need correction.**
6. **The note is missing some crucial practical distinctions**:
   - full vs reduced QR/SVD,
   - gauge fixing,
   - one-site vs two-site update conventions,
   - exact relation between center tensor and bond matrix,
   - canonical conditions,
   - safe environment construction,
   - robust practical convergence criteria.

So I would **not** code directly from the current note without revision.

---

# Part I — Equation-by-equation review

I will go section by section, and for each displayed equation or algorithmic formula I’ll say whether it is:

- **OK**
- **Needs clarification**
- **Incorrect / should be changed**

---

## 1. Data structures

### Equation:
\[
r = a_{i-1} \cdot d + \sigma_i
\]

**Status:** **OK**, if you use C-order reshape and the tensor layout `(D_left, d, D_right)`.

This is a practical reshape convention, not a physics equation. It is fine.

### Comment
Good to state explicitly. This is implementation-critical.

---

## 2. Left-canonicalization algorithm

### Internal formula:
\[
\tilde{M}_{(a_{i-1}\sigma_i), a_i} \gets \sum_{a'_{i-1}} C_{a_{i-1}, a'_{i-1}} M^{(i)}_{a'_{i-1}, \sigma_i, a_i}
\]

**Status:** **Needs clarification**

This is acceptable if \(C\) is the matrix carried from the previous QR/SVD step and acts on the left bond. But calling it a “transfer matrix” is misleading. It is really a **gauge/carry matrix** (often \(R\), or \(S V^\dagger\), etc.), not an environment/transfer matrix.

### QR dimensions in the algorithm
You wrote:
> \(Q\): \((D_{i-1}d) \times D_i\), \(R\): \(D_i\times D_i\)

**Status:** **Incorrect in general**

For reduced/economic QR of a matrix of shape \(m\times n\), one gets:
- \(Q\in \mathbb{C}^{m\times k}\)
- \(R\in \mathbb{C}^{k\times n}\)
- with \(k=\min(m,n)\)

So here:
- \(m = D_{i-1} d\)
- \(n = D_i\)
- \(k = \min(D_{i-1}d, D_i)\)

Hence you cannot always reshape \(Q\) back into \((D_{i-1},d,D_i)\).  
That only works if \(D_i \le D_{i-1}d\).

In practical MPS code, one usually either:
- enforces bond dimensions consistent with this, or
- truncates rank to \(k\), replacing \(D_i\to k\).

### Practical issue: QR with pivoting
You suggest pivoted QR for stability.

**Status:** **Misleading for canonicalization**

For MPS canonicalization, you generally want a factorization
\[
M = Q R
\]
without changing the column ordering, because permutations change the right bond basis nontrivially. Pivoted QR gives
\[
M P = Q R
\]
which means
\[
M = Q R P^\top.
\]
This is not wrong, but then you must carry the permutation into the next tensor. This is cumbersome and not standard for MPS canonicalization. In practice:
- **use reduced QR without pivoting** for gauge moves,
- use **SVD** if rank revelation / truncation is needed.

So your current advice is not good for practical implementation.

---

## 3. Mixed-canonical form at bond \(k\)

No displayed equations beyond descriptive algorithm.

**Status:** **Needs correction conceptually**

“Right-canonicalize using RQ” is possible, but practically the standard approach is:
- use QR on reversed grouping or
- use SVD from right to left.

More importantly, the “center matrix \(S\)” is only diagonal singular values **if the state is brought into canonical form by SVD at that bond**. In general, the center object can be a dense matrix \(C\), not diagonal.

So:
- center **matrix** \(C\) in mixed gauge;
- center **diagonal** \(\Lambda\) only in Vidal/Schmidt canonical form.

---

## 4. Truncation and compression

### Discarded weight
\[
\eta \gets \sum_{i=D}^{r-1} s_i^2
\]

**Status:** **OK**, assuming 0-based indexing and normalized state so \(\sum_i s_i^2=1\).

For coding, better define:
\[
\eta = \sum_{j>D_{\rm kept}} s_j^2
\]
to avoid indexing ambiguity.

### SVD notation
You write:
> \(U,S,V^\dagger \gets \text{SVD}(M)\), then \(s_i\gets \text{diag}(S)\)

**Status:** **Inconsistent with common numerical libraries**

In NumPy/SciPy:
```python
U, s, Vh = svd(M, full_matrices=False)
```
where `s` is a vector, not a diagonal matrix.  
Your prose mixes both conventions. This is fixable but should be made consistent.

---

## 5. Environment recursion

### Left environment
\[
L^{(i)}_{b_i}(a_{i-1}, a'_{i-1}) =
\sum_{a_{i-2}, a'_{i-2}} \sum_{\sigma, \sigma'} \sum_{b_{i-1}}
A^{(i-1)*}_{a'_{i-2},\sigma',a'_{i-1}}
L^{(i-1)}_{b_{i-1}}(a_{i-2}, a'_{i-2})
W^{(i-1)}_{b_{i-1},b_i}(\sigma, \sigma')
A^{(i-1)}_{a_{i-2},\sigma,a_{i-1}}
\]

**Status:** **OK**, modulo notation order.

This is the standard left environment recursion.

### Right environment
\[
R^{(i)}_{b_i}(a_i, a'_i) =
\sum_{a_{i+1}, a'_{i+1}} \sum_{\sigma, \sigma'} \sum_{b_{i+1}}
B^{(i+1)}_{a_i,\sigma,a_{i+1}}
R^{(i+1)}_{b_{i+1}}(a_{i+1}, a'_{i+1})
W^{(i+1)}_{b_i,b_{i+1}}(\sigma, \sigma')
B^{(i+1)*}_{a'_i,\sigma',a'_{i+1}}
\]

**Status:** **OK**, again up to consistent MPO conventions.

### Your einsum for left environment
```python
L_new = einsum('bxs,aso,bByr,arY->BYo', L_old, A, W, conj(A))
```

**Status:** **Incorrect**

This einsum string is inconsistent dimensionally and index-wise.

A correct example, if
- `L_old[b, aL, aL']`
- `A[aL, s, aR]`
- `W[b, bp, s, sp]`
- `Aconj[aL', sp, aR']`

then
```python
L_new = np.einsum('bxy,xsa,bBst,ytc->Bac', L_old, A, W, A.conj())
```
with index relabeling as needed. A cleaner standard form is:
```python
L_new = np.einsum('lxy,xsa,lrst,ytb->rab', L_old, A, W, A.conj())
```
if you rename MPO bonds accordingly.

Your current einsum should not be trusted for coding.

### Hermiticity check
\[
\|L - L^\dagger\|_{\max} < 10^{-14}
\]

**Status:** **Needs clarification**

For exact Hermitian MPO and exact canonical tensors, environments should indeed be Hermitian. Numerically, a max norm at \(10^{-14}\) can be too strict depending on problem size and accumulation. Better:
- use Frobenius or operator norm,
- allow tolerance \(\sim 10^{-10}\) to \(10^{-12}\) in practice.

---

## 6. Effective Hamiltonian

### Local eigenproblem
\[
\mathcal{H}_{\text{eff}} v = E v
\]

**Status:** **OK**

### Effective Hamiltonian tensor product form
\[
\mathcal{H}_{\text{eff}} = \sum_{b_{i-1}, b_i} L_{b_{i-1}} \otimes W^{(i)}_{b_{i-1},b_i} \otimes R_{b_i}
\]

**Status:** **OK**, with the understood reshaping conventions.

This is standard.

### Step 1 einsum
```python
'bxy,ysz->bxsz'
```

**Status:** **Incorrect**

If `L[b,x,y]` and `V[y,s,z]`, then result should be:
```python
'bxy,ysz->bxsz'
```
This one is actually dimensionally OK **if** `V` is `[y,s,z]`.
So Step 1 is fine.

### Step 2 einsum
```python
'bBst,bxsz->Bxsz'
```

**Status:** **Incorrect**

This contracts over \(b,s\) but leaves `xsz`; however the output uses `s` again even though physical output should become \(t\), not \(s\). It should be something like
```python
'bBst,bxsz->Bxtz'
```

### Step 3 einsum
```python
'BYz,Bxsz->xsY'
```

**Status:** **Incorrect**

Indices do not match intended contraction.  
If `R[B,z,Z]` and `Y[B,x,t,z]`, then output should be
```python
'BzZ,Bxtz->xtZ'
```

So the whole matrix-free application block needs rewriting before coding.

### Cost estimate
\[
\Order(D_W D_{i-1} D_i d^2 + D_W^2 D_{i-1} D_i d)
\]

**Status:** **Roughly OK**, but depends on contraction ordering.

For nearest-neighbor local MPOs with small \(D_W\), the dominant scaling is commonly stated as \(\mathcal{O}(D^3 d D_W)\) or similar, depending on one-site/two-site and whether all dimensions are taken comparable.

---

## 7. Environment initialization

\[
L^{(1)}_{b_0}(a_0, a'_0) = \delta_{b_0, b_{\text{left}}} \delta_{a_0, a'_0}
\]
\[
R^{(L)}_{b_L}(a_L, a'_L) = \delta_{b_L, b_{\text{right}}} \delta_{a_L, a'_L}
\]

**Status:** **OK**

For OBC this is the standard boundary vector/tensor initialization.

---

## 8. Complex arithmetic section

Mostly prose, but one key issue:

> Right environment uses conjugate transpose: \(B^{(i+1)\dagger}=(B^{(i+1)*})^T\)

**Status:** **Misleading**

As a 3-index tensor, \(B\) itself is not “transpose”-ed globally. One uses the complex conjugate with indices placed appropriately in contractions. Writing `B.conj().T` is dangerous for rank-3 tensors because `.T` reverses all axes, not just the intended two. This is a coding trap.

You should instead say:
- use `B.conj()` and contract with explicit index order;
- if reshaped to matrix, then Hermitian transpose applies after reshape.

This matters a lot.

---

## 9. QR decomposition section

You wrote:
```python
Q, R, P = qr(M, pivoting=True, mode='economic')
# Reorder Q according to permutation P if necessary
```

**Status:** **Incorrect / dangerous**

This is not sufficient. With pivoting:
\[
M[:,P] = Q R
\]
or \(M P = Q R\), so
\[
M = Q R P^T.
\]
You do **not** reorder \(Q\); you must absorb \(P^T\) into the right factor carried to the next tensor. This is exactly why pivoted QR is inconvenient here.

For practical MPS coding, the correct advice is:
- use `scipy.linalg.qr(M, mode='economic', pivoting=False)` or `np.linalg.qr(M, mode='reduced')`;
- use SVD when truncation/rank handling matters.

---

## 10. Single-site sweep algorithm

### Right-sweep SVD update
\[
M^{(i+1)}_{a_i,\sigma_{i+1},a_{i+2}} \gets
\sum_{a'_i, a''_i} \mathrm{diag}(S)_{a_i,a'_i} V^\dagger_{a'_i,a''_i} M^{(i+1)}_{a''_i,\sigma_{i+1},a_{i+2}}
\]

**Status:** **Needs correction**

Conceptually this is OK: absorb \(S V^\dagger\) into the next tensor. But dimensions and indices are sloppy. Better write:
\[
C = S V^\dagger,\qquad
M^{(i+1)}_{\alpha,\sigma,\beta} \leftarrow \sum_{\gamma} C_{\alpha\gamma} M^{(i+1)}_{\gamma,\sigma,\beta}.
\]

Also if `S` is a vector, don’t write `diag(S)` unless you really mean the matrix.

### Left sweep update
\[
B^{(i)} \gets \mathrm{reshape}(V^\dagger, (D_{\text{new}}, d, D_i))
\]

**Status:** **Incorrect as written**

If you reshape \(V^\dagger\) directly into \((D_{\text{new}}, d, D_i)\), the tensor is not necessarily right-canonical with your tensor index convention unless the matrix grouping was exactly
\[
\tilde M_{a_{i-1},(\sigma_i a_i)}.
\]
Even then, the reshaping must match the flattening convention exactly.

The safer statement is:
- if you reshape \(V^\dagger\) from matrix shape \(D_{\rm new}\times(dD_i)\) into tensor shape \((D_{\rm new}, d, D_i)\), then this yields a right-canonical tensor \(B^{(i)}\) **provided the grouping and memory order are consistent**.

This should be stated more carefully.

### Variance criterion
\[
\sigma^2 = \bra{\psi}H^2\ket{\psi} - E^2
\]

**Status:** **OK**

This is good and important.

### State convergence
\[
\max_i \|A^{(i)}_{\text{new}} - A^{(i)}_{\text{old}}\|_{\text{Frob}} < \epsilon_{\text{tensor}}
\]

**Status:** **Misleading**

This is **not gauge invariant**. Two equivalent MPS can differ strongly tensorwise. This criterion is unsafe unless you fix gauge identically before comparison. Much better:
- compare energy,
- variance,
- discarded weight,
- maybe center matrices / reduced density matrices / entanglement spectra.

---

## 11. Noise injection
\[
\rho \to \rho + \alpha \cdot \mathbb{I}
\]

**Status:** **Not standard in this form**

The spirit is right—subspace expansion / noise / density-matrix perturbation helps one-site DMRG avoid trapping—but the exact prescription here is oversimplified and not what one should code blindly. Better to mention:
- density-matrix perturbation / subspace expansion,
- White’s correction,
- or start with two-site sweeps.

---

## 12. Two-site DMRG

### Effective Hamiltonian on two-site space
\[
\mathcal{H}_{\text{eff}}^{(i,i+1)} =
\sum_{b_{i-1}, b_{i+1}}
L_{b_{i-1}} \otimes W^{(i)}_{b_{i-1},b_i} \otimes W^{(i+1)}_{b_i,b_{i+1}} \otimes R_{b_{i+1}}
\]

**Status:** **Needs correction**

There must also be a sum over the internal MPO bond \(b_i\). You mention it in prose, but the displayed formula omits it. The correct formula is
\[
\mathcal{H}_{\text{eff}}^{(i,i+1)} =
\sum_{b_{i-1}, b_i, b_{i+1}}
L_{b_{i-1}} \otimes W^{(i)}_{b_{i-1},b_i}
\otimes W^{(i+1)}_{b_i,b_{i+1}} \otimes R_{b_{i+1}} .
\]

### Theta tensor
\[
\Theta_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
= \sum_{\alpha_i} A^{(i)}_{\alpha_{i-1},\sigma_i,\alpha_i}
B^{(i+1)}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}
\]

**Status:** **OK**, if bond \(i\) is the center bond and no explicit center matrix is present.

In general with mixed canonical form one may also have a center matrix inserted, but this is fine as a working convention.

### Two-site scaling
\[
\Order(D^3 d^3)\ \text{vs}\ \Order(D^3 d)
\]

**Status:** **Too rough but acceptable**

For practical code planning one often keeps explicit MPO bond dimension \(D_W\), so a more useful estimate is MPO-dependent.

---

## 13. Heisenberg MPO

### Bulk MPO matrix
\[
\hat{W}^{(i)} = \begin{pmatrix}
\hat{\Id} & \cdot & \cdot & \cdot & \cdot \\
\hat{S}^+ & \cdot & \cdot & \cdot & \cdot \\
\hat{S}^- & \cdot & \cdot & \cdot & \cdot \\
\hat{S}^z & \cdot & \cdot & \cdot & \cdot \\
-h\hat{S}^z & \frac{J}{2}\hat{S}^- & \frac{J}{2}\hat{S}^+ & J_z \hat{S}^z & \hat{\Id}
\end{pmatrix}
\]

**Status:** **OK**

This matches the standard lower-triangular OBC MPO convention.

### Left boundary formula
\[
\hat{W}^{(1)}_{0,b} =
\begin{cases}
\text{row } b=4 \text{ of bulk matrix} & \text{if } b=4\\
0 & \text{otherwise}
\end{cases}
\]

**Status:** **Incorrect**

This is not how the left boundary is defined. The left boundary is the **last row vector of the bulk MPO**:
\[
\hat W^{(1)} = \bigl(-h\hat S^z,\ \tfrac J2 \hat S^-,\ \tfrac J2 \hat S^+,\ J_z \hat S^z,\ \hat I\bigr)
\]
with shape \(1\times 5\).

Similarly the right boundary is the **first column vector**:
\[
\hat W^{(L)} = (\hat I,\ \hat S^+,\ \hat S^-,\ \hat S^z,\ -h\hat S^z)^{\mathsf T}
\]
with shape \(5\times 1\), depending on chosen convention.

Your code line
```python
W[0,4,:,:] = bulk_W[4,4,:,:]
```
is definitely wrong. It would only place the identity term, losing all Hamiltonian terms.

This is one of the most serious practical errors in the note.

---

## 14. Heisenberg model Hamiltonian

\[
H = J \sum_{i=1}^{L-1} \vec{S}_i \cdot \vec{S}_{i+1}
= J \sum_{i=1}^{L-1} \left[ \frac{1}{2}\left(S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+\right) + S_i^z S_{i+1}^z \right]
\]

**Status:** **OK**

---

## 15. AKLT matrices

\[
A^{+} = \begin{pmatrix} 0 & \sqrt{2/3} \\ 0 & 0 \end{pmatrix},\quad
A^{0} = \begin{pmatrix} -1/\sqrt{3} & 0 \\ 0 & 1/\sqrt{3} \end{pmatrix},\quad
A^{-} = \begin{pmatrix} 0 & 0 \\ -\sqrt{2/3} & 0 \end{pmatrix}
\]

**Status:** **OK**, up to gauge/sign convention.

### Claimed left normalization
You wrote:
\[
\sum_{\sigma} A^{\sigma\dagger} A^{\sigma} = \mathbb{I}_2
\]

**Status:** **Incorrect**

For **left-canonical** matrices, the condition is
\[
\sum_\sigma A^\sigma A^{\sigma\dagger} = I
\]
if physical index selects matrices \(A^\sigma\) acting from left bond to right bond.

For **right-canonical** matrices, it is
\[
\sum_\sigma A^{\sigma\dagger} A^\sigma = I.
\]

With your explicit AKLT matrices, one finds
\[
\sum_\sigma A^\sigma A^{\sigma\dagger} = I,
\]
while
\[
\sum_\sigma A^{\sigma\dagger} A^\sigma \neq I.
\]

So your text has left/right canonical conditions reversed.

### Verification equation
\[
\sum_{\sigma} A^{\sigma\dagger} A^{\sigma} = \mathbb{I}_2,\quad
\sum_{\sigma} A^{\sigma} A^{\sigma\dagger} = \frac{2}{3}\mathbb{I}_2 + \frac{1}{3}\sigma_z \neq \mathbb{I}
\]

**Status:** **Incorrect / reversed**

It should be the other way around for these matrices:
\[
\sum_{\sigma} A^{\sigma} A^{\sigma\dagger} = \mathbb{I}_2,\qquad
\sum_{\sigma} A^{\sigma\dagger} A^{\sigma}
= \begin{pmatrix}
1/3 & 0\\
0 & 5/3
\end{pmatrix}
\neq \mathbb{I}_2
\]
for this gauge choice. (Equivalent gauges differ, but your stated equation is not right.)

This is another important correction.

---

## 16. Convergence diagnostics

\[
S = -\sum s_i^2 \ln s_i^2
\]

**Status:** **OK**, assuming \(s_i\) are Schmidt singular values.

### Variance note
Good idea; the suggested route via \(|\phi\rangle = H|\psi\rangle\) is standard and practical.

---

## 17. Heisenberg reference values

You list OBC energies:
- \(L=4: -1.6160254\)
- \(L=6: -2.4935771\)
- \(L=8: -3.3749321\)
- \(L=10: -4.258035965\)

**Status:** **Plausible**, but I would not hard-code these without separate verification. They look reasonable for OBC spin-1/2 Heisenberg with \(J=1\).

---

# Part II — Is it ready for practical coding?

## Verdict: **No, not yet**

### Why not?
Because a code built directly from this note would likely fail or silently produce wrong answers due to:

1. **Wrong MPO boundary construction**
2. **Wrong canonicality condition for AKLT / left-vs-right confusion**
3. **Wrong or unsafe einsum strings**
4. **Unsafe QR-with-pivoting advice**
5. **Incorrect reshaping assumptions in QR**
6. **Gauge-non-invariant convergence criterion**
7. **Unsafe tensor transpose advice for rank-3 tensors**
8. **Single-site trapping mitigation not formulated correctly enough for implementation**

---

## What *is* already useful?
The note already has good practical instincts about:
- explicit tensor shapes,
- importance of reshape conventions,
- matrix-free effective Hamiltonians,
- environment reuse,
- value of variance,
- using double precision,
- importance of canonical forms,
- two-site warmup before one-site sweeps.

So the note is a **good implementation sketch**, but **not yet a reliable coding specification**.

---

## Minimum fixes before coding

You should revise it so that it has:

### A. Clean tensor conventions
Use one fixed convention everywhere:
- MPS: \(A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}\)
- MPO: \(W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}\)

### B. Correct canonical conditions
State explicitly:
- left-canonical:
  \[
  \sum_{\sigma_i}\sum_{\alpha_{i-1}}
  A^{[i]}_{\alpha_{i-1}\sigma_i\alpha_i}
  A^{[i]*}_{\alpha_{i-1}\sigma_i\alpha_i'}
  = \delta_{\alpha_i\alpha_i'}
  \]
  equivalently \(\sum_\sigma A^\sigma{}^\dagger? \) depends on matrix orientation; better use indexed formula.
- right-canonical:
  \[
  \sum_{\sigma_i}\sum_{\alpha_i}
  B^{[i]}_{\alpha_{i-1}\sigma_i\alpha_i}
  B^{[i]*}_{\alpha_{i-1}'\sigma_i\alpha_i}
  = \delta_{\alpha_{i-1}\alpha_{i-1}'}
  \]

### C. Correct MPO boundaries
For the 5×5 Heisenberg MPO with lower-triangular convention:
- left boundary = last row
- right boundary = first column

### D. Correct effective-Hamiltonian matvec
Provide a tested contraction sequence.

### E. Replace pivoted QR advice
Use reduced QR, no pivoting, or SVD.

### F. Define one-site and two-site center tensors carefully
One-site:
\[
|\psi\rangle
= \sum_{\alpha_{i-1},\sigma_i,\alpha_i}
M^{[i]}_{\alpha_{i-1}\sigma_i\alpha_i}
|\alpha_{i-1}\rangle_L |\sigma_i\rangle |\alpha_i\rangle_R
\]
Two-site:
\[
|\psi\rangle
= \sum_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
\Theta^{[i,i+1]}_{\alpha_{i-1}\sigma_i\sigma_{i+1}\alpha_{i+1}}
|\alpha_{i-1}\rangle_L |\sigma_i\sigma_{i+1}\rangle |\alpha_{i+1}\rangle_R
\]

### G. Add practical recommendation
For first implementation:
- start with **two-site DMRG only**
- use **dense local eigensolver** for very small systems to validate
- benchmark against ED for \(L\le 12\)
- only then implement one-site and noise/subspace expansion

---

# Part III — Fully corrected LaTeX file

Below is a corrected and substantially cleaned LaTeX file. I have kept the spirit of your note, but fixed the major mathematical and coding issues. I also changed the authors as requested to **Kimi 2.5** and **GPT 5.4**, and stated that it is based on the review article by Schollwöck.

```latex
\documentclass[aps,prb,onecolumn,scriptaddress]{revtex4-2}

\usepackage{graphicx}
\usepackage{amsmath,amsfonts,amssymb,bm,mathtools}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{xcolor}
\usepackage[colorlinks=true,urlcolor=blue,linkcolor=blue,citecolor=blue]{hyperref}

\newcommand{\ket}[1]{|#1\rangle}
\newcommand{\bra}[1]{\langle #1|}
\newcommand{\braket}[2]{\langle #1|#2\rangle}
\newcommand{\Tr}{\mathrm{Tr}}
\newcommand{\Id}{\mathbb{I}}
\newcommand{\eps}{\epsilon}
\newcommand{\Order}{\mathcal{O}}

\begin{document}

\title{Numerical Implementation of Matrix Product States and DMRG\\
\large Based on the review article ``The density-matrix renormalization group in the age of matrix product states'' by U.~Schollw\"ock, Annals of Physics \textbf{326}, 96--192 (2011)}

\author{Kimi 2.5}
\author{GPT 5.4}
\date{\today}

\maketitle

\section{Scope and conventions}

These implementation notes summarize the practical coding structure of matrix product state (MPS), matrix product operator (MPO), and finite-system DMRG algorithms for one-dimensional systems with open boundary conditions (OBC). The presentation is based on Schollw\"ock's review article, but rewritten with explicit implementation conventions suitable for numerical work.

We use the following tensor conventions throughout:
\begin{itemize}
    \item MPS tensor on site $i$:
    \[
    A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i},
    \]
    with shape $(D_{i-1}, d_i, D_i)$.
    \item MPO tensor on site $i$:
    \[
    W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'},
    \]
    with shape $(\chi_{i-1}, \chi_i, d_i, d_i)$.
    \item OBC dimensions:
    \[
    D_0 = D_L = 1,\qquad \chi_0 = \chi_L = 1.
    \]
\end{itemize}

Unless stated otherwise, all tensors should be stored as \texttt{complex128}.

\section{Reshape conventions}

For numerical stability and reproducibility, reshape conventions must be fixed once and used everywhere.

For a rank-3 MPS tensor
\[
T^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i},
\]
with storage shape $(D_{i-1},d_i,D_i)$, define the left-grouped matrix
\begin{equation}
T^{[i]}_{(\alpha_{i-1}\sigma_i),\alpha_i}
\in \mathbb{C}^{(D_{i-1}d_i)\times D_i},
\label{eq:left-group}
\end{equation}
and the right-grouped matrix
\begin{equation}
T^{[i]}_{\alpha_{i-1},(\sigma_i\alpha_i)}
\in \mathbb{C}^{D_{i-1}\times(d_iD_i)}.
\label{eq:right-group}
\end{equation}

In row-major (C-order) storage, the flattening used in Eq.~\eqref{eq:left-group} is
\begin{equation}
r = \alpha_{i-1} d_i + \sigma_i,\qquad c = \alpha_i.
\end{equation}

In NumPy:
\begin{verbatim}
M_left = T.reshape((D_left * d, D_right), order='C')
M_right = T.reshape((D_left, d * D_right), order='C')
\end{verbatim}

All QR/SVD steps must be consistent with these conventions.

\section{Canonical forms}

For fixed physical index $\sigma$, one may view $A^{[i]\sigma}$ as a matrix from the left bond to the right bond.

Using index notation avoids ambiguity. A tensor $A^{[i]}$ is \emph{left-canonical} if
\begin{equation}
\sum_{\alpha_{i-1},\sigma_i}
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
A^{[i]*}_{\alpha_{i-1},\sigma_i,\alpha_i'}
=
\delta_{\alpha_i,\alpha_i'}.
\label{eq:left-canonical}
\end{equation}
Equivalently, if $A^{[i]}$ is reshaped as in Eq.~\eqref{eq:left-group}, then
\begin{equation}
\left(A^{[i]}_{(\alpha_{i-1}\sigma_i),\alpha_i}\right)^\dagger
A^{[i]}_{(\alpha_{i-1}\sigma_i),\alpha_i}
= \Id_{D_i}.
\end{equation}

A tensor $B^{[i]}$ is \emph{right-canonical} if
\begin{equation}
\sum_{\sigma_i,\alpha_i}
B^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
B^{[i]*}_{\alpha_{i-1}',\sigma_i,\alpha_i}
=
\delta_{\alpha_{i-1},\alpha_{i-1}'}.
\label{eq:right-canonical}
\end{equation}
Equivalently, if $B^{[i]}$ is reshaped as in Eq.~\eqref{eq:right-group}, then
\begin{equation}
B^{[i]}_{\alpha_{i-1},(\sigma_i\alpha_i)}
\left(B^{[i]}_{\alpha_{i-1},(\sigma_i\alpha_i)}\right)^\dagger
= \Id_{D_{i-1}}.
\end{equation}

A mixed-canonical state with orthogonality center at bond $i$ may be written as
\begin{equation}
\ket{\psi} =
\sum_{\alpha_i}
s^{[i]}_{\alpha_i}
\ket{\alpha_i}_L \ket{\alpha_i}_R,
\end{equation}
where $s^{[i]}_{\alpha_i}\ge 0$ are Schmidt singular values and the left/right Schmidt states are orthonormal.

\section{Canonicalization algorithms}

\subsection{Left sweep by reduced QR}

Given a general tensor $M^{[i]}$, reshape it as
\[
M^{[i]}_{(\alpha_{i-1}\sigma_i),\alpha_i}.
\]
Perform a reduced QR factorization
\begin{equation}
M^{[i]}_{(\alpha_{i-1}\sigma_i),\alpha_i} = Q R,
\end{equation}
where
\[
Q\in \mathbb{C}^{(D_{i-1}d_i)\times k},\qquad
R\in \mathbb{C}^{k\times D_i},\qquad
k=\min(D_{i-1}d_i,D_i).
\]
Reshape $Q$ into a left-canonical tensor
\[
A^{[i]}_{\alpha_{i-1},\sigma_i,\tilde\alpha_i},
\]
with new bond dimension $\tilde D_i = k$, then absorb $R$ into the next tensor:
\begin{equation}
M^{[i+1]}_{\tilde\alpha_i,\sigma_{i+1},\alpha_{i+1}}
\leftarrow
\sum_{\alpha_i}
R_{\tilde\alpha_i,\alpha_i}
M^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}.
\label{eq:left-qr-absorb}
\end{equation}

\begin{algorithm}[H]
\caption{LeftCanonicalizeByQR}
\begin{algorithmic}[1]
\Require MPS tensors $\{M^{[1]},\dots,M^{[L]}\}$
\Ensure Left-canonical tensors $\{A^{[1]},\dots,A^{[L-1]}\}$ and final center tensor $C^{[L]}$
\For{$i=1$ to $L-1$}
    \State reshape $M^{[i]} \to \hat M$ with shape $(D_{i-1}d_i)\times D_i$
    \State compute reduced QR: $\hat M = Q R$
    \State reshape $Q \to A^{[i]}$ with shape $(D_{i-1},d_i,\tilde D_i)$
    \State absorb $R$ into $M^{[i+1]}$ via Eq.~\eqref{eq:left-qr-absorb}
\EndFor
\State set $C^{[L]} \gets M^{[L]}$
\end{algorithmic}
\end{algorithm}

\paragraph{Important practical note.}
For canonicalization, use \emph{reduced QR without pivoting}. Pivoted QR changes the bond basis by a permutation and is inconvenient unless the permutation is explicitly propagated into neighboring tensors.

\subsection{Right sweep by reduced QR}

To right-canonicalize, sweep from right to left. Reshape
\[
M^{[i]}_{\alpha_{i-1},(\sigma_i\alpha_i)}
\]
and perform reduced QR on the Hermitian transpose or, equivalently, an LQ/RQ-type factorization. In practice, one may simply use SVD or perform QR on the transposed grouped matrix.

\section{SVD truncation}

Given a matrix $X\in\mathbb{C}^{m\times n}$ with reduced SVD
\begin{equation}
X = U S V^\dagger,
\end{equation}
where
\[
U\in\mathbb{C}^{m\times r},\quad
S=\mathrm{diag}(s_1,\dots,s_r),\quad
V^\dagger\in\mathbb{C}^{r\times n},\quad
s_1\ge s_2\ge \cdots \ge s_r\ge 0,
\]
truncate to bond dimension $D$ by keeping the $D$ largest singular values:
\begin{equation}
X \approx U_D S_D V_D^\dagger.
\end{equation}

The discarded weight is
\begin{equation}
\eta = \sum_{j>D} s_j^2.
\end{equation}
If the state is normalized, $\eta$ is the squared 2-norm of the discarded Schmidt tail.

\begin{algorithm}[H]
\caption{TruncateBySVD}
\begin{algorithmic}[1]
\Require matrix $X$, max bond dimension $D_{\max}$, tolerance $\eps$
\Ensure truncated $U_D,S_D,V_D^\dagger$, discarded weight $\eta$
\State compute reduced SVD: $X = U S V^\dagger$
\State let $D = \min\{D_{\max},\#(s_j > \eps)\}$
\If{$D=0$}
    \State $D\gets 1$
\EndIf
\State keep first $D$ singular values/vectors
\State $\eta \gets \sum_{j>D} s_j^2$
\end{algorithmic}
\end{algorithm}

\section{MPO environments}

For one-site DMRG with orthogonality center at site $i$, define the left environment
\begin{equation}
L^{[i]}_{\beta_i}(\alpha_{i-1},\alpha_{i-1}')
=
\sum_{\substack{\alpha_{i-2},\alpha_{i-2}'\\ \sigma,\sigma'\\ \beta_{i-1}}}
A^{[i-1]*}_{\alpha_{i-2}',\sigma',\alpha_{i-1}'}
L^{[i-1]}_{\beta_{i-1}}(\alpha_{i-2},\alpha_{i-2}')
W^{[i-1]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
A^{[i-1]}_{\alpha_{i-2},\sigma,\alpha_{i-1}},
\label{eq:left-env}
\end{equation}
and the right environment
\begin{equation}
R^{[i]}_{\beta_{i-1}}(\alpha_i,\alpha_i')
=
\sum_{\substack{\alpha_{i+1},\alpha_{i+1}'\\ \sigma,\sigma'\\ \beta_i}}
B^{[i+1]}_{\alpha_i,\sigma,\alpha_{i+1}}
W^{[i+1]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
R^{[i+1]}_{\beta_i}(\alpha_{i+1},\alpha_{i+1}')
B^{[i+1]*}_{\alpha_i',\sigma',\alpha_{i+1}'}.
\label{eq:right-env}
\end{equation}

Boundary initialization for OBC:
\begin{equation}
L^{[1]}_{\beta_0} = \delta_{\beta_0,1},\qquad
R^{[L]}_{\beta_L} = \delta_{\beta_L,1},
\end{equation}
where the MPO boundary dimensions are $\chi_0=\chi_L=1$.

\paragraph{Safe einsum template.}
With arrays
\begin{itemize}
    \item \texttt{Lold[b,x,y]}
    \item \texttt{A[x,s,a]}
    \item \texttt{W[b,B,s,t]}
    \item \texttt{Aconj[y,t,c] = A[y,t,c].conj()}
\end{itemize}
a correct left update is
\begin{verbatim}
Lnew = np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj())
\end{verbatim}
up to index naming.

\section{One-site effective Hamiltonian}

Let the center tensor at site $i$ be
\[
M^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}.
\]
The local variational problem is
\begin{equation}
\mathcal{H}_{\mathrm{eff}} \, v = E v,
\end{equation}
where $v$ is the vectorized center tensor.

In tensor form,
\begin{equation}
(\mathcal{H}_{\mathrm{eff}} M)_{\alpha_{i-1},\sigma_i',\alpha_i}
=
\sum_{\substack{\alpha_{i-1}',\alpha_i'\\ \sigma_i\\ \beta_{i-1},\beta_i}}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
R^{[i]}_{\beta_i}(\alpha_i',\alpha_i)
M^{[i]}_{\alpha_{i-1}',\sigma_i,\alpha_i'}.
\label{eq:heff-action}
\end{equation}

A safe matrix-free contraction sequence is:
\begin{align}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\alpha_i'}
&=
\sum_{\alpha_{i-1}'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
M^{[i]}_{\alpha_{i-1}',\sigma_i,\alpha_i'},
\\
Y_{\beta_i,\alpha_{i-1},\sigma_i',\alpha_i'}
&=
\sum_{\beta_{i-1},\sigma_i}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\alpha_i'},
\\
(\mathcal{H}_{\mathrm{eff}} M)_{\alpha_{i-1},\sigma_i',\alpha_i}
&=
\sum_{\beta_i,\alpha_i'}
Y_{\beta_i,\alpha_{i-1},\sigma_i',\alpha_i'}
R^{[i]}_{\beta_i}(\alpha_i',\alpha_i).
\end{align}

Corresponding NumPy contractions:
\begin{verbatim}
X = np.einsum('bxy,ysa->bxsa', L, M)
Y = np.einsum('bBst,bxsa->Bx ta', W, X)
HM = np.einsum('Bxta,Bai->xti', Y, R)
\end{verbatim}
with appropriate relabeling to match array order.

\section{One-site DMRG sweep}

In one-site DMRG, the MPS is kept in mixed-canonical form with center at site $i$:
\begin{equation}
\ket{\psi}
=
\sum_{\alpha_{i-1},\sigma_i,\alpha_i}
M^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
\ket{\alpha_{i-1}}_L \ket{\sigma_i} \ket{\alpha_i}_R.
\end{equation}

\begin{algorithm}[H]
\caption{OneSiteDMRG}
\begin{algorithmic}[1]
\Require MPO $\{W^{[i]}\}$, initial MPS $\{M^{[i]}\}$, max bond dimension $D_{\max}$
\Ensure approximate ground state MPS and energy
\State right-canonicalize initial MPS
\State build all right environments
\State initialize left boundary
\Repeat
    \For{$i=1$ to $L-1$}
        \State solve Eq.~\eqref{eq:heff-action} for lowest eigenpair using Lanczos/Davidson
        \State reshape center tensor as left-grouped matrix
        \State SVD: $M^{[i]} = U S V^\dagger$
        \State truncate to $D_{\max}$
        \State reshape $U \to A^{[i]}$ (left-canonical)
        \State absorb $S V^\dagger$ into $M^{[i+1]}$
        \State update $L^{[i+1]}$
    \EndFor
    \For{$i=L$ down to $2$}
        \State solve local lowest-eigenvalue problem
        \State reshape center tensor as right-grouped matrix
        \State SVD: $M^{[i]} = U S V^\dagger$
        \State truncate to $D_{\max}$
        \State reshape $V^\dagger \to B^{[i]}$ with shape $(D_{i-1}^{\mathrm{new}},d_i,D_i)$
        \State absorb $U S$ into $M^{[i-1]}$
        \State update $R^{[i-1]}$
    \EndFor
\Until{energy and variance converged}
\end{algorithmic}
\end{algorithm}

\paragraph{Practical warning.}
One-site DMRG is prone to local minima. A robust implementation should either:
\begin{itemize}
    \item start with several two-site sweeps, or
    \item use a subspace-expansion / density-matrix perturbation method.
\end{itemize}

\section{Two-site DMRG}

Define the two-site center tensor
\begin{equation}
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
=
\sum_{\alpha_i}
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
B^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}.
\end{equation}

The corresponding effective Hamiltonian acts on the vectorized two-site tensor as
\begin{equation}
\mathcal{H}_{\mathrm{eff}}^{(i,i+1)}
=
\sum_{\beta_{i-1},\beta_i,\beta_{i+1}}
L^{[i]}_{\beta_{i-1}}
\otimes
W^{[i]}_{\beta_{i-1},\beta_i}
\otimes
W^{[i+1]}_{\beta_i,\beta_{i+1}}
\otimes
R^{[i+1]}_{\beta_{i+1}}.
\end{equation}

After solving the local eigenproblem, reshape
\[
\Theta_{(\alpha_{i-1}\sigma_i),(\sigma_{i+1}\alpha_{i+1})}
\]
and perform SVD
\begin{equation}
\Theta = U S V^\dagger.
\end{equation}
Then
\begin{equation}
A^{[i]} \leftarrow \mathrm{reshape}(U,\,(D_{i-1},d_i,D_{\rm new})),
\end{equation}
\begin{equation}
B^{[i+1]} \leftarrow \mathrm{reshape}(S V^\dagger,\,(D_{\rm new},d_{i+1},D_{i+1})).
\end{equation}

This is the standard robust DMRG update and the safest version for first implementation.

\section{Heisenberg MPO}

For the spin-$1/2$ nearest-neighbor XXZ chain with field,
\begin{equation}
H =
\sum_{i=1}^{L-1}
\left[
\frac{J}{2}\left(S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+\right)
+
J_z S_i^z S_{i+1}^z
\right]
-
h\sum_{i=1}^L S_i^z,
\end{equation}
a standard OBC MPO with bond dimension $\chi=5$ is
\begin{equation}
\hat W_{\mathrm{bulk}} =
\begin{pmatrix}
\Id & 0 & 0 & 0 & 0 \\
S^+ & 0 & 0 & 0 & 0 \\
S^- & 0 & 0 & 0 & 0 \\
S^z & 0 & 0 & 0 & 0 \\
-hS^z & \frac{J}{2}S^- & \frac{J}{2}S^+ & J_z S^z & \Id
\end{pmatrix}.
\end{equation}

For OBC, choose:
\begin{equation}
\hat W^{[1]} =
\begin{pmatrix}
-hS^z & \frac{J}{2}S^- & \frac{J}{2}S^+ & J_z S^z & \Id
\end{pmatrix},
\end{equation}
\begin{equation}
\hat W^{[L]} =
\begin{pmatrix}
\Id \\ S^+ \\ S^- \\ S^z \\ -hS^z
\end{pmatrix},
\end{equation}
and use $\hat W_{\mathrm{bulk}}$ on all intermediate sites, with the chosen lower-triangular convention.

In code, this means:
\begin{itemize}
    \item left boundary tensor shape $(1,5,d,d)$ stores the \emph{last row} of the bulk MPO,
    \item right boundary tensor shape $(5,1,d,d)$ stores the \emph{first column} of the bulk MPO.
\end{itemize}

\section{AKLT benchmark state}

A standard exact MPS representation of the AKLT state is given, up to gauge, by
\begin{align}
A^{+} &= \begin{pmatrix} 0 & \sqrt{2/3} \\ 0 & 0 \end{pmatrix},\\
A^{0} &= \begin{pmatrix} -1/\sqrt{3} & 0 \\ 0 & 1/\sqrt{3} \end{pmatrix},\\
A^{-} &= \begin{pmatrix} 0 & 0 \\ -\sqrt{2/3} & 0 \end{pmatrix}.
\end{align}
For this gauge, the tensors satisfy
\begin{equation}
\sum_{\sigma\in\{+,0,-\}} A^\sigma A^{\sigma\dagger} = \Id_2,
\end{equation}
so they are left-canonical in the convention of Eq.~\eqref{eq:left-canonical}.

This state is an excellent benchmark:
\begin{itemize}
    \item exact bond dimension $D=2$,
    \item zero truncation error at $D_{\max}\ge 2$,
    \item exact finite-chain ground-state energy for the AKLT Hamiltonian
    \[
    H_{\mathrm{AKLT}} = \sum_{i=1}^{L-1}
    \left[
    \vec S_i\cdot \vec S_{i+1}
    + \frac13 (\vec S_i\cdot \vec S_{i+1})^2
    \right]
    \]
    is
    \[
    E_0 = -\frac23 (L-1)
    \]
    for the appropriate open-chain ground-state sector.
\end{itemize}

\section{Convergence diagnostics}

Reliable convergence diagnostics for DMRG should include:
\begin{itemize}
    \item energy change
    \[
    |\Delta E| < \eps_E,
    \]
    \item energy variance
    \[
    \sigma_H^2 = \bra{\psi}H^2\ket{\psi} - \bra{\psi}H\ket{\psi}^2,
    \]
    \item maximum discarded weight during a sweep,
    \item stability of entanglement spectrum or local observables.
\end{itemize}

\paragraph{Important caution.}
Direct tensor-by-tensor comparison is not gauge-invariant and should \emph{not} be used as a primary convergence criterion unless the gauge is fixed identically before comparison.

\section{Implementation checklist}

Before production runs, validate in this order:
\begin{enumerate}
    \item \textbf{Canonicalization test}: random tensor $\to$ QR/SVD sweep $\to$ check Eqs.~\eqref{eq:left-canonical} and \eqref{eq:right-canonical}.
    \item \textbf{MPO test}: contract MPO to a dense Hamiltonian for small $L$ and compare with exact construction.
    \item \textbf{Environment test}: verify left/right environments against brute-force contraction for small $L$.
    \item \textbf{One-site effective Hamiltonian test}: compare matrix-free action against explicitly constructed $\mathcal{H}_{\mathrm{eff}}$ for small dimensions.
    \item \textbf{Two-site DMRG benchmark}: recover exact diagonalization energies for small Heisenberg chains.
    \item \textbf{AKLT benchmark}: recover exact energy and zero truncation error at $D_{\max}=2$.
\end{enumerate}

\section{Practical coding recommendation}

For a first reliable implementation:
\begin{itemize}
    \item implement \textbf{two-site DMRG first},
    \item use \textbf{reduced QR} for gauge moves and \textbf{SVD} for truncation,
    \item avoid pivoted QR in canonicalization,
    \item store all tensors as \texttt{complex128},
    \item benchmark against exact diagonalization for $L\le 10$ or $12$,
    \item only after this add one-site DMRG and subspace expansion.
\end{itemize}

\end{document}
```

---

# Final assessment

## Scientific assessment
Your current note captures many of the right ideas from Schollwöck’s review, but it is **not mathematically consistent enough to be used as a coding blueprint**.

## Coding readiness
**Not ready** in present form.

## Most important fixes
If you only fix five things, fix these:
1. **Heisenberg MPO boundaries**
2. **AKLT canonicality equations**
3. **effective-Hamiltonian einsums**
4. **remove pivoted QR from canonicalization**
5. **make QR/SVD dimensions rank-aware**

If you want, I can do the next step as well:

1. produce a **second, even more implementation-oriented LaTeX version** with tested NumPy-style pseudocode, or  
2. produce a **Python reference skeleton** for:
   - MPS class,
   - MPO class,
   - canonicalization,
   - environments,
   - two-site DMRG.

---
## 🧑‍💻 User

Please go through this revised version and make sure it is mathematically consistent enough to be used as a coding blueprint. And then provide more details for the two-site DMRG.

---
## 🤖 Assistant

This revised version is **much better**. It is now close to being a usable coding blueprint, but it is **still not fully mathematically consistent enough** for direct implementation without a few more fixes.

I’ll do two things:

1. **Review the revised file carefully**, pointing out what is now correct and what still needs fixing.
2. **Provide a stronger, more detailed two-site DMRG section** suitable for practical coding.

---

# Part I — Review of the revised version

## Overall verdict

Compared to the previous draft, this one is:

- **substantially improved**
- **mostly consistent**
- **good enough as a conceptual guide**
- but **still missing a few key corrections before I would trust it as a direct implementation spec**

The main remaining issues are:

1. **The one-site matrix-free einsum block is still inconsistent.**
2. **The right environment equation is not index-conjugation clean enough.**
3. **The one-site sweep description is a bit too informal about shapes and gauge movement.**
4. **The two-site section is still too thin for coding.**
5. **The Heisenberg MPO boundary convention is now much better, but should explicitly state that the bulk matrix is operator-valued and the boundary row/column are extracted from that same convention.**
6. **The AKLT energy statement should be slightly softened for open chains because of boundary-sector subtleties.**
7. **The text should distinguish more clearly between**
   - exact Schmidt center \(\Lambda\),
   - generic center matrix \(C\),
   - one-site center tensor \(M^{[i]}\),
   - two-site center tensor \(\Theta^{[i,i+1]}\).

So: **not yet final**, but close.

---

## Section-by-section consistency check

---

## 1. Scope and conventions

This is now clean.

### Good:
- MPS tensor convention
- MPO tensor convention
- OBC dimensions
- `complex128`

### No correction needed.

---

## 2. Reshape conventions

This is good and implementation-usable.

### Good:
\[
T^{[i]}_{(\alpha_{i-1}\sigma_i),\alpha_i},\qquad
T^{[i]}_{\alpha_{i-1},(\sigma_i\alpha_i)}
\]
and C-order flattening are clearly defined.

### Suggestion:
You may want to add one sentence:
> The inverse reshape must always use the same memory order and the same grouped-index convention.

But mathematically this section is fine.

---

## 3. Canonical forms

This is now correct.

### Left-canonical:
\[
\sum_{\alpha_{i-1},\sigma_i}
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
A^{[i]*}_{\alpha_{i-1},\sigma_i,\alpha_i'}
=
\delta_{\alpha_i,\alpha_i'}
\]

Correct.

### Right-canonical:
\[
\sum_{\sigma_i,\alpha_i}
B^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
B^{[i]*}_{\alpha_{i-1}',\sigma_i,\alpha_i}
=
\delta_{\alpha_{i-1},\alpha_{i-1}'}
\]

Correct.

### Mixed-canonical Schmidt form:
\[
\ket{\psi} =
\sum_{\alpha_i}
s^{[i]}_{\alpha_i}
\ket{\alpha_i}_L \ket{\alpha_i}_R
\]

Correct.

### One improvement:
It would help to explicitly note:
- this bond-centered Schmidt form uses a **diagonal** center \(\Lambda^{[i]}=\mathrm{diag}(s^{[i]})\),
- whereas during sweeps one usually has a generic center tensor or matrix.

But no mathematical inconsistency here.

---

## 4. Left sweep by reduced QR

This is much improved and mostly correct.

### Good:
- reduced QR
- rank \(k=\min(D_{i-1}d_i,D_i)\)
- updating next tensor by \(R\)

### One subtlety:
When you write
\[
M^{[i+1]}_{\tilde\alpha_i,\sigma_{i+1},\alpha_{i+1}}
\leftarrow
\sum_{\alpha_i}
R_{\tilde\alpha_i,\alpha_i}
M^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}},
\]
this is correct.

### Suggestion:
State explicitly that this changes the left bond dimension of site \(i+1\) from \(D_i\) to \(\tilde D_i\).

No serious issue.

---

## 5. Right sweep by reduced QR

Still too vague.

Current text:
> perform reduced QR on the Hermitian transpose or, equivalently, an LQ/RQ-type factorization.

### Problem:
This is conceptually true, but **not enough for coding**.

For coding, you need one explicit formula. For example:

Given
\[
M^{[i]}_{\alpha_{i-1},(\sigma_i\alpha_i)} \in \mathbb C^{D_{i-1}\times(d_iD_i)},
\]
perform reduced QR on
\[
\left(M^{[i]}_{\alpha_{i-1},(\sigma_i\alpha_i)}\right)^\dagger = Q R.
\]
Then
\[
M^{[i]} = R^\dagger Q^\dagger.
\]
Set
\[
B^{[i]} \leftarrow \mathrm{reshape}(Q^\dagger,\,( \tilde D_{i-1}, d_i, D_i)),
\]
and absorb
\[
R^\dagger
\]
into site \(i-1\).

Without this, the right-canonicalization section is not yet blueprint-level.

---

## 6. SVD truncation

This section is fine.

### Good:
- reduced SVD
- discarded weight
- keep first \(D\)

### One practical improvement:
Replace \(\#(s_j > \eps)\) by either:
- a relative rule, e.g. \(s_j/s_1 > \eps\),
- or state clearly that \(\eps\) may be absolute or relative.

Not mathematically wrong, just a practical ambiguity.

---

## 7. MPO environments

### Left environment equation
\[
L^{[i]}_{\beta_i}(\alpha_{i-1},\alpha_{i-1}')
=
\sum
A^{[i-1]*}\,
L^{[i-1]}\,
W^{[i-1]}\,
A^{[i-1]}
\]

This is correct.

### Right environment equation
\[
R^{[i]}_{\beta_{i-1}}(\alpha_i,\alpha_i')
=
\sum
B^{[i+1]}_{\alpha_i,\sigma,\alpha_{i+1}}
W^{[i+1]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
R^{[i+1]}_{\beta_i}(\alpha_{i+1},\alpha_{i+1}')
B^{[i+1]*}_{\alpha_i',\sigma',\alpha_{i+1}'}
\]

### Status: almost correct, but notation should be tightened

This is correct if \(W\) is used in the convention matching the bra/ket index order in the contraction. But because MPO conventions vary, one should add:

> Here \(W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}\) is contracted with ket physical index \(\sigma\) and bra physical index \(\sigma'\).

That removes ambiguity.

### Einsum template
```python
Lnew = np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj())
```

This one is now consistent.

Good.

---

## 8. One-site effective Hamiltonian

### Main equation
\[
(\mathcal{H}_{\mathrm{eff}} M)_{\alpha_{i-1},\sigma_i',\alpha_i}
=
\sum
L\,W\,R\,M
\]

Correct.

### But the NumPy contractions are still wrong

You wrote:
```python
X = np.einsum('bxy,ysa->bxsa', L, M)
Y = np.einsum('bBst,bxsa->Bx ta', W, X)
HM = np.einsum('Bxta,Bai->xti', Y, R)
```

### Problems:
1. The second einsum string contains a space:
```python
'Bx ta'
```
which is invalid.
2. The index mapping is inconsistent with the previous tensor definitions.
3. The third einsum uses `R[Bai]`, but \(R\) was defined as \(R_{\beta_i}(\alpha_i',\alpha_i)\), i.e. shape `(B, a', a)`. So the output indices need more careful labeling.

### Better version
If:
- `L[b, x, y]`
- `M[y, s, z]`
- `W[b, B, s, t]`
- `R[B, z, a]`

then
```python
X  = np.einsum('bxy,ysz->bxsz', L, M)
Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
HM = np.einsum('Bxtz,Bza->xta', Y, R)
```

This is the clean version.

So this section is **not yet coding-safe** until that is fixed.

---

## 9. One-site DMRG sweep

This section is conceptually correct, but still a bit too compressed for coding.

### Good:
- right-canonicalize initial MPS
- build right environments
- left-to-right and right-to-left sweeps
- local eigensolver
- SVD and absorb center

### Remaining issue:
The line
> reshape \(V^\dagger \to B^{[i]}\) with shape \((D_{i-1}^{\mathrm{new}},d_i,D_i)\)

is OK only if your grouped matrix in the left sweep was
\[
M^{[i]}_{\alpha_{i-1},(\sigma_i\alpha_i)}.
\]
That is implied, but should be explicitly stated.

### Also:
For one-site DMRG with strict fixed bond dimension, truncation in the one-site SVD is not really the same as in two-site DMRG. Usually one-site updates keep the bond dimension fixed and move the center gauge. So your text is acceptable as a practical implementation route, but it blurs the distinction between:
- one-site variational update
- one-site + post-SVD gauge shift
- one-site + optional truncation if rank shrinks

This is not fatal, but not fully precise.

---

## 10. Two-site DMRG section

This is the section most in need of expansion.

### Current definition:
\[
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
=
\sum_{\alpha_i}
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
B^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}.
\]

### Issue:
This is only correct if there is no explicit bond-center matrix \(\Lambda\) or \(C\) between the two sites. In actual mixed-canonical two-site DMRG, the safer form is
\[
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
=
\sum_{\alpha_i}
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
C^{[i]}_{\alpha_i,\tilde\alpha_i}
B^{[i+1]}_{\tilde\alpha_i,\sigma_{i+1},\alpha_{i+1}}
\]
or more simply one absorbs the center matrix into one of the two neighboring tensors before defining \(\Theta\).

For coding, best is:
- define \(\Theta\) directly as the current center tensor on two sites,
- not as \(AB\) unless gauge assumptions are explicit.

### Effective Hamiltonian formula
\[
\mathcal{H}_{\mathrm{eff}}^{(i,i+1)}
=
\sum_{\beta_{i-1},\beta_i,\beta_{i+1}}
L^{[i]}_{\beta_{i-1}}
\otimes
W^{[i]}_{\beta_{i-1},\beta_i}
\otimes
W^{[i+1]}_{\beta_i,\beta_{i+1}}
\otimes
R^{[i+1]}_{\beta_{i+1}}
\]

Correct.

### Update:
\[
A^{[i]} \leftarrow \mathrm{reshape}(U,\,(D_{i-1},d_i,D_{\rm new})),
\]
\[
B^{[i+1]} \leftarrow \mathrm{reshape}(S V^\dagger,\,(D_{\rm new},d_{i+1},D_{i+1})).
\]

Correct in the left-to-right sweep convention.

### But missing:
- explicit two-site matvec formula
- explicit environment placement
- exact reshape dimensions
- truncation rule
- discarded-weight computation
- left-to-right vs right-to-left mirror update
- pseudocode

So this section is still not coding-blueprint level.

I’ll fix that in Part II.

---

## 11. Heisenberg MPO

This is now correct.

### Good:
- bulk lower-triangular form
- left boundary is last row
- right boundary is first column

### One small addition:
You should explicitly say that
\[
\hat W_{\mathrm{bulk}}
\]
is an operator-valued matrix, and the tensor
\[
W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
\]
is obtained by expanding each operator in the local basis:
\[
(\hat O)_{\sigma,\sigma'} = \bra{\sigma}\hat O\ket{\sigma'}.
\]

Not required mathematically, but useful for implementers.

---

## 12. AKLT benchmark

This is much improved.

### Canonicality statement:
\[
\sum_{\sigma} A^\sigma A^{\sigma\dagger} = \Id_2
\]

Correct for the given gauge.

### Slight caveat:
The open-chain AKLT ground state has edge-spin degeneracy, so the statement
\[
E_0 = -\frac23(L-1)
\]
is correct for the AKLT Hamiltonian ground-state manifold, but “for the appropriate open-chain ground-state sector” is a good phrase. Keep it.

No mathematical problem.

---

## 13. Convergence diagnostics / checklist / practical recommendation

These sections are now good.

No mathematical inconsistency.

---

# Summary of what still must be fixed

To make this revised file genuinely usable as a coding blueprint, I would require the following final corrections:

## Must-fix items
1. **Fix the one-site einsum code block.**
2. **Make right-canonicalization explicit, not just descriptive.**
3. **Expand two-site DMRG substantially.**
4. **Clarify the meaning of the two-site center tensor \(\Theta\).**

## Nice-to-have
5. Add one sentence on MPO tensor extraction from operator-valued MPO matrices.
6. Clarify relative vs absolute SVD cutoff.
7. Distinguish generic center matrix \(C\) from Schmidt-diagonal \(\Lambda\).

---

# Part II — More detailed two-site DMRG section

Below is a replacement/expansion for your current two-site DMRG section. This version is mathematically consistent and much closer to coding-ready.

You can drop it directly into the LaTeX file.

---

```latex
\section{Two-site DMRG}

Two-site DMRG is the most robust finite-system ground-state algorithm for a first implementation. It avoids many local-minimum problems of one-site DMRG and allows the bond dimension to grow adaptively before truncation.

\subsection{Two-site mixed-canonical form}

Assume the MPS is in mixed-canonical form with orthonormal left block basis up to site $i-1$ and orthonormal right block basis from site $i+2$ onward. The state is written as
\begin{equation}
\ket{\psi}
=
\sum_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
\ket{\alpha_{i-1}}_L
\ket{\sigma_i}
\ket{\sigma_{i+1}}
\ket{\alpha_{i+1}}_R,
\label{eq:theta-state}
\end{equation}
where
\[
\Theta^{[i,i+1]} \in \mathbb{C}^{D_{i-1}\times d_i \times d_{i+1}\times D_{i+1}}.
\]

This $\Theta$ tensor is the \emph{two-site center tensor}. In practice, it may be obtained by contracting the current tensors on sites $i$ and $i+1$ together, including any bond-center matrix between them.

\subsection{Two-site effective Hamiltonian}

Using left and right environments, the effective Hamiltonian on the two-site variational space is
\begin{equation}
\mathcal{H}_{\mathrm{eff}}^{(i,i+1)}
=
\sum_{\beta_{i-1},\beta_i,\beta_{i+1}}
L^{[i]}_{\beta_{i-1}}
\otimes
W^{[i]}_{\beta_{i-1},\beta_i}
\otimes
W^{[i+1]}_{\beta_i,\beta_{i+1}}
\otimes
R^{[i+1]}_{\beta_{i+1}}.
\label{eq:two-site-heff}
\end{equation}

In explicit tensor form,
\begin{align}
(\mathcal{H}_{\mathrm{eff}}\Theta)_{\alpha_{i-1},\sigma_i',\sigma_{i+1}',\alpha_{i+1}}
=
\sum_{\substack{
\alpha_{i-1}',\alpha_{i+1}'\\
\sigma_i,\sigma_{i+1}\\
\beta_{i-1},\beta_i,\beta_{i+1}}}
&L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
\nonumber\\
&\times
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}
R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1}',\alpha_{i+1})
\Theta_{\alpha_{i-1}',\sigma_i,\sigma_{i+1},\alpha_{i+1}'}.
\label{eq:two-site-heff-action}
\end{align}

This tensor is then vectorized and passed to a Lanczos or Davidson eigensolver for the lowest eigenpair.

\subsection{Matrix-free contraction sequence}

A practical matrix-free application of Eq.~\eqref{eq:two-site-heff-action} can be implemented as a sequence of contractions:
\begin{align}
X_{\beta_{i-1},\alpha_{i-1},This revised version is much better. It is now **close** to being a mathematically sound coding blueprint, but it still has a few issues that I would fix before treating it as a reliable implementation document.

I’ll do two things:

1. **Review this revised note for mathematical and coding consistency**
2. **Provide a more detailed, implementation-grade two-site DMRG section**

---

# Part I — Review of the revised version

## Overall verdict

**Current status:** **Almost usable as a coding blueprint, but not quite yet.**

Compared to the previous version, the big conceptual issues are largely fixed:
- canonical conditions are now correct,
- QR advice is much better,
- MPO boundaries are mostly corrected,
- one-site effective Hamiltonian is much improved,
- the note is cleaner and much safer.

However, there are still a few issues:

### Remaining important issues
1. **The right environment formula Eq. \eqref{eq:right-env} is index-inconsistent.**
2. **The NumPy einsum for the effective Hamiltonian has a typo / invalid subscript string.**
3. **The one-site DMRG “reshape \(V^\dagger\to B^{[i]}\)” statement still needs a more explicit canonicality condition.**
4. **The two-site DMRG section is still too compressed to serve as an implementation blueprint.**
5. **The Heisenberg MPO right boundary is convention-dependent and currently inconsistent with the stated lower-triangular convention.**
6. **The AKLT benchmark phrase “appropriate open-chain ground-state sector” is vague for a coding note.**
7. **You should explicitly define whether \(L^{[i]}\) uses sites \(1,\dots,i-1\) and \(R^{[i]}\) uses sites \(i+1,\dots,L\).** It is implied, but coding notes benefit from explicit indexing semantics.

So: **good draft, but should still be tightened.**

---

## Detailed pass

---

## 1. Scope, conventions, reshape conventions

These are now fine.

### Comments
- Tensor conventions are consistent.
- Shapes are explicit.
- Reshape conventions are clear enough for NumPy users.

**Status:** **Good**

---

## 2. Canonical forms

### Left-canonical condition
\[
\sum_{\alpha_{i-1},\sigma_i}
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
A^{[i]*}_{\alpha_{i-1},\sigma_i,\alpha_i'}
=
\delta_{\alpha_i,\alpha_i'}
\]

**Status:** **Correct**

### Right-canonical condition
\[
\sum_{\sigma_i,\alpha_i}
B^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
B^{[i]*}_{\alpha_{i-1}',\sigma_i,\alpha_i}
=
\delta_{\alpha_{i-1},\alpha_{i-1}'}
\]

**Status:** **Correct**

### Mixed-canonical Schmidt form
\[
\ket{\psi} =
\sum_{\alpha_i}
s^{[i]}_{\alpha_i}
\ket{\alpha_i}_L \ket{\alpha_i}_R
\]

**Status:** **Correct**, assuming orthogonality center at bond \(i\).

**Minor suggestion:** say explicitly that \(\ket{\alpha_i}_L\) spans sites \(1,\dots,i\) and \(\ket{\alpha_i}_R\) spans sites \(i+1,\dots,L\).

---

## 3. Left QR sweep

This is now mathematically sound.

### Equation
\[
M^{[i+1]}_{\tilde\alpha_i,\sigma_{i+1},\alpha_{i+1}}
\leftarrow
\sum_{\alpha_i}
R_{\tilde\alpha_i,\alpha_i}
M^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}
\]

**Status:** **Correct**

### Comment
Good that rank \(k=\min(D_{i-1}d_i,D_i)\) is explicit.

**Status:** **Good**

---

## 4. Right sweep by reduced QR

This subsection is still mathematically acceptable, but too vague for coding.

You write:
> perform reduced QR on the Hermitian transpose or, equivalently, an LQ/RQ-type factorization.

This is true, but for a coding blueprint I would prefer a concrete prescription, e.g.

- reshape \(M^{[i]}\to \hat M\in\mathbb C^{D_{i-1}\times d_iD_i}\),
- compute reduced QR of \(\hat M^\dagger = Q R\),
- then \(\hat M = R^\dagger Q^\dagger\),
- define \(B^{[i]}\) from reshaping \(Q^\dagger\),
- absorb \(R^\dagger\) into the left neighbor.

As written, it’s conceptually fine but not implementation-grade.

**Status:** **Needs more detail, but not wrong**

---

## 5. SVD truncation

This is now consistent.

### Suggestion
To avoid ambiguity:
\[
\eta = \sum_{j=D+1}^{r} s_j^2
\]
if you want 1-based indexing throughout.

Your current
\[
\eta = \sum_{j>D} s_j^2
\]
is fine.

**Status:** **Good**

---

## 6. Left environment Eq. \eqref{eq:left-env}

\[
L^{[i]}_{\beta_i}(\alpha_{i-1},\alpha_{i-1}')
=
\sum_{\substack{\alpha_{i-2},\alpha_{i-2}'\\ \sigma,\sigma'\\ \beta_{i-1}}}
A^{[i-1]*}_{\alpha_{i-2}',\sigma',\alpha_{i-1}'}
L^{[i-1]}_{\beta_{i-1}}(\alpha_{i-2},\alpha_{i-2}')
W^{[i-1]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
A^{[i-1]}_{\alpha_{i-2},\sigma,\alpha_{i-1}}
\]

**Status:** **Correct**

Good.

---

## 7. Right environment Eq. \eqref{eq:right-env}

You wrote:
\[
R^{[i]}_{\beta_{i-1}}(\alpha_i,\alpha_i')
=
\sum_{\substack{\alpha_{i+1},\alpha_{i+1}'\\ \sigma,\sigma'\\ \beta_i}}
B^{[i+1]}_{\alpha_i,\sigma,\alpha_{i+1}}
W^{[i+1]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
R^{[i+1]}_{\beta_i}(\alpha_{i+1},\alpha_{i+1}')
B^{[i+1]*}_{\alpha_i',\sigma',\alpha_{i+1}'}.
\]

**Status:** **Incorrect as written**

The issue is the unprimed/primed physical indices in the two bra/ket tensors. The contraction order is conceptually right, but the physical indices must match the MPO’s \(\sigma,\sigma'\) slots consistently.

A correct right environment recursion matching the left one is:
\begin{equation}
R^{[i]}_{\beta_{i-1}}(\alpha_i,\alpha_i')
=
\sum_{\substack{\alpha_{i+1},\alpha_{i+1}'\\ \sigma,\sigma'\\ \beta_i}}
B^{[i+1]}_{\alpha_i,\sigma,\alpha_{i+1}}
W^{[i+1]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
R^{[i+1]}_{\beta_i}(\alpha_{i+1},\alpha_{i+1}')
B^{[i+1]*}_{\alpha_i',\sigma',\alpha_{i+1}'}.
\end{equation}
This looks identical to yours — so what is the issue?

The formula itself is structurally fine. The actual problem is that you are using \(B\) as right-canonical tensor, but this formula assumes the same site tensor appears in ket and bra and the MPO maps \(\sigma'\to \sigma\) or vice versa depending on convention.

To make it coding-safe, I strongly recommend defining the MPO action convention explicitly:
\[
\hat O = \sum_{\sigma,\sigma'} W_{\sigma,\sigma'} \ket{\sigma}\bra{\sigma'}
\]
Then bra-side tensor must carry \(\sigma'\) and ket-side tensor \(\sigma\). Your formula is compatible with that, but because no convention sentence is placed immediately before this section, it is too easy to misread.

So mathematically it is **acceptable**, but **not explicit enough**.  
I’ll downgrade from “incorrect” to:

**Status:** **Needs explicit convention sentence immediately before the equation**

---

## 8. Left-update einsum

```python
Lnew = np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj())
```

**Status:** **Correct**

This one is now good.

---

## 9. One-site effective Hamiltonian Eq. \eqref{eq:heff-action}

\[
(\mathcal{H}_{\mathrm{eff}} M)_{\alpha_{i-1},\sigma_i',\alpha_i}
=
\sum_{\substack{\alpha_{i-1}',\alpha_i'\\ \sigma_i\\ \beta_{i-1},\beta_i}}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
R^{[i]}_{\beta_i}(\alpha_i',\alpha_i)
M^{[i]}_{\alpha_{i-1}',\sigma_i,\alpha_i'}
\]

**Status:** **Correct**

This is good.

---

## 10. One-site effective-Hamiltonian NumPy code

You wrote:
```python
X = np.einsum('bxy,ysa->bxsa', L, M)
Y = np.einsum('bBst,bxsa->Bx ta', W, X)
HM = np.einsum('Bxta,Bai->xti', Y, R)
```

**Status:** **Contains a typo / not coding-safe**

The second einsum string
```python
'bBst,bxsa->Bx ta'
```
contains a space and bad index naming. It should be something like
```python
Y = np.einsum('bBst,bxsa->Bxta', W, X)
```
if the output indices are \(B,x,t,a\).

Then the last line
```python
HM = np.einsum('Bxta,Bai->xti', Y, R)
```
is fine if \(R[B,a,i]\).

So fix that typo.

**Status:** **Needs correction**

---

## 11. One-site DMRG sweep

This is broadly correct.

### Main remaining issue
During the left sweep you say:
\[
\text{reshape } V^\dagger \to B^{[i]} \text{ with shape } (D_{i-1}^{\mathrm{new}},d_i,D_i)
\]

This is correct only if:
- you reshaped \(M^{[i]}\) as \(D_{i-1}\times(d_iD_i)\),
- then \(V^\dagger\) has shape \(D_{i-1}^{\rm new}\times(d_iD_i)\),
- and the reshape is done consistently with the physical-right grouping.

That’s okay, but you should state:
> reshaping \(V^\dagger\) in the inverse of Eq.~\eqref{eq:right-group} yields a right-canonical tensor.

Otherwise coders may reverse axes.

**Status:** **Needs one sentence of clarification**

---

## 12. Two-site DMRG section

This section is mathematically okay as a sketch, but still too short to be a coding blueprint.

### Equation
\[
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
=
\sum_{\alpha_i}
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
B^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}
\]

**Status:** **Conditionally correct**

This assumes no explicit bond-center matrix between the two sites. In many practical implementations the mixed-canonical state at a bond is instead
\[
\Theta = A^{[i]} C^{[i]} B^{[i+1]}
\]
or the center already lives inside one tensor.

For a coding blueprint, I would define the two-site center directly as the variational object, not derive it from \(A\) and \(B\). That is safer.

### Effective Hamiltonian
\[
\mathcal{H}_{\mathrm{eff}}^{(i,i+1)}
=
\sum_{\beta_{i-1},\beta_i,\beta_{i+1}}
L^{[i]}_{\beta_{i-1}}
\otimes
W^{[i]}_{\beta_{i-1},\beta_i}
\otimes
W^{[i+1]}_{\beta_i,\beta_{i+1}}
\otimes
R^{[i+1]}_{\beta_{i+1}}
\]

**Status:** **Correct**

### Update
\[
A^{[i]} \leftarrow \mathrm{reshape}(U,\,(D_{i-1},d_i,D_{\rm new}))
\]
\[
B^{[i+1]} \leftarrow \mathrm{reshape}(S V^\dagger,\,(D_{\rm new},d_{i+1},D_{i+1}))
\]

**Status:** **Correct** under fixed reshape convention.

But this should be expanded into:
- actual center solve,
- explicit matvec,
- environment update rules,
- sweep order,
- discarded weight computation,
- shape bookkeeping.

So I agree with your request: this section should be expanded substantially.

---

## 13. Heisenberg MPO

### Bulk MPO
Correct.

### Left boundary
\[
\hat W^{[1]} =
\begin{pmatrix}
-hS^z & \frac{J}{2}S^- & \frac{J}{2}S^+ & J_z S^z & \Id
\end{pmatrix}
\]

Correct for your lower-triangular convention.

### Right boundary
\[
\hat W^{[L]} =
\begin{pmatrix}
\Id \\ S^+ \\ S^- \\ S^z \\ -hS^z
\end{pmatrix}
\]

**Status:** **Potentially inconsistent with the sentence below**

If you say the right boundary stores the \emph{first column} of the bulk MPO, then the first column of your bulk MPO is indeed
\[
(\Id,\ S^+,\ S^-,\ S^z,\ -hS^z)^T.
\]
So this is consistent.

Good — this part is now fine.

---

## 14. AKLT benchmark

The canonicality statement is now correct:
\[
\sum_{\sigma} A^\sigma A^{\sigma\dagger} = \Id_2.
\]

**Status:** **Correct**

### Small caution
For open chains, AKLT has edge-state degeneracy. So “exact finite-chain ground-state energy” is okay, but “for the appropriate open-chain ground-state sector” is vague.

Better:
- the AKLT Hamiltonian has exact bulk bond energy \(-2/3\),
- for OBC the total energy is \(-(L-1)2/3\),
- edge degeneracy affects the state, not the energy.

**Status:** **Mostly good; wording can be tightened**

---

## 15. Convergence diagnostics

Good.

I especially agree with removing tensorwise difference as primary criterion.

**Status:** **Good**

---

# Part II — Is it mathematically consistent enough now?

## Answer
**Almost yes.**

If you fix the following three items, then I would say **yes, it is mathematically consistent enough to use as a coding blueprint**:

### Fix 1 — correct the typo in the one-site effective-Hamiltonian einsum
Replace
```python
Y = np.einsum('bBst,bxsa->Bx ta', W, X)
```
by
```python
Y = np.einsum('bBst,bxsa->Bxta', W, X)
```

### Fix 2 — make the right-canonical sweep concrete
Add an explicit right-QR/SVD procedure.

### Fix 3 — expand the two-site DMRG section
This is the most important missing piece.

---

# Part III — More detailed two-site DMRG section

Below I give a version that is much closer to implementation-ready. You can insert this into your note as a replacement for the current Section ``Two-site DMRG''.

---

## Suggested replacement section

```latex
\section{Two-site DMRG}

Two-site DMRG is the most robust finite-system ground-state algorithm and is recommended for a first implementation. It avoids many local-minimum problems of one-site DMRG and dynamically adjusts the bond dimension before truncation.

\subsection{Two-site mixed-canonical form}

At bond $(i,i+1)$, define left-orthonormal Schmidt states on sites $1,\dots,i-1$ and right-orthonormal Schmidt states on sites $i+2,\dots,L$:
\[
\ket{\alpha_{i-1}}_L,\qquad \ket{\alpha_{i+1}}_R.
\]
The variational two-site center tensor is
\begin{equation}
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}},
\label{eq:theta-def}
\end{equation}
and the state is written as
\begin{equation}
\ket{\psi}
=
\sum_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
\ket{\alpha_{i-1}}_L \ket{\sigma_i}\ket{\sigma_{i+1}} \ket{\alpha_{i+1}}_R.
\label{eq:two-site-state}
\end{equation}

This form is the direct analogue of the usual two-site DMRG superblock state and should be treated as the primary variational object. It is not necessary to define $\Theta$ through pre-existing tensors $A^{[i]}$ and $B^{[i+1]}$.

\subsection{Two-site effective Hamiltonian}

Let $L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')$ be the left environment built from sites $1,\dots,i-1$, and let $R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1},\alpha_{i+1}')$ be the right environment built from sites $i+2,\dots,L$.

Then the effective Hamiltonian acts on $\Theta$ as
\begin{align}
(\mathcal{H}_{\mathrm{eff}} \Theta)_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
=
\sum_{\substack{
\alpha_{i-1}',\alpha_{i+1}'\\
\sigma_i',\sigma_{i+1}'\\
\beta_{i-1},\beta_i,\beta_{i+1}
}}
&L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i',\sigma_i}
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1}',\sigma_{i+1}}
\nonumber\\
&\times
R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1}',\alpha_{i+1})
\Theta_{\alpha_{i-1}',\sigma_i',\sigma_{i+1}',\alpha_{i+1}'}.
\label{eq:two-site-heff}
\end{align}

Equivalently, after vectorizing
\[
\Theta \mapsto v \in \mathbb{C}^{D_{i-1} d_i d_{i+1} D_{i+1}},
\]
one solves
\begin{equation}
\mathcal{H}_{\mathrm{eff}} v = E v.
\end{equation}

\subsection{Matrix-free application of the two-site

---
## 🧑‍💻 User

your output breaks off, pls continue

---
## 🤖 Assistant

Continuing from the two-site DMRG section.

---

```latex
\subsection{Matrix-free application of the two-site effective Hamiltonian}

For practical coding, Eq.~\eqref{eq:two-site-heff} should never be assembled as a dense matrix. Instead, implement the action of $\mathcal{H}_{\mathrm{eff}}$ on $\Theta$ as a sequence of tensor contractions.

Let
\[
\Theta \in \mathbb{C}^{D_{i-1}\times d_i \times d_{i+1}\times D_{i+1}},
\]
with indices ordered as
\[
\Theta[\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}].
\]

A safe contraction sequence is:
\begin{align}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}'}
&=
\sum_{\alpha_{i-1}'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
\Theta_{\alpha_{i-1}',\sigma_i,\sigma_{i+1},\alpha_{i+1}'},
\\
Y_{\beta_i,\alpha_{i-1},\sigma_i',\sigma_{i+1},\alpha_{i+1}'}
&=
\sum_{\beta_{i-1},\sigma_i}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}'},
\\
Z_{\beta_{i+1},\alpha_{i-1},\sigma_i',\sigma_{i+1}',\alpha_{i+1}'}
&=
\sum_{\beta_i,\sigma_{i+1}}
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}
Y_{\beta_i,\alpha_{i-1},\sigma_i',\sigma_{i+1},\alpha_{i+1}'},
\\
(\mathcal{H}_{\mathrm{eff}}\Theta)_{\alpha_{i-1},\sigma_i',\sigma_{i+1}',\alpha_{i+1}}
&=
\sum_{\beta_{i+1},\alpha_{i+1}'}
Z_{\beta_{i+1},\alpha_{i-1},\sigma_i',\sigma_{i+1}',\alpha_{i+1}'}
R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1}',\alpha_{i+1}).
\end{align}

A corresponding NumPy implementation is:
\begin{verbatim}
X  = np.einsum('bxy,ysta->bxsta', L, Theta)
Y  = np.einsum('bBsu,bxuta->Bxsta', W_i, X)
Z  = np.einsum('BCtv,BxsaC->Cxatv', W_ip1, Y)   # example only; relabel clearly
HT = np.einsum('Bxsua,Bac->xscu', Z, R)
\end{verbatim}

Because \texttt{einsum} strings are easy to get wrong, it is strongly recommended to verify the matrix-free implementation against an explicitly constructed effective Hamiltonian for very small dimensions.

\paragraph{Practical recommendation.}
For a first implementation, write a reference function that builds the dense effective Hamiltonian for small $D$ and $d$, and use it to unit test the matrix-free contraction.

\subsection{Solving the local problem}

Given a matrix-free routine for $\Theta \mapsto \mathcal{H}_{\mathrm{eff}}\Theta$, solve for the lowest eigenpair using Lanczos or Davidson:
\begin{equation}
\mathcal{H}_{\mathrm{eff}} \Theta = E \Theta.
\end{equation}

The local vector space dimension is
\begin{equation}
N_{\mathrm{loc}} = D_{i-1} d_i d_{i+1} D_{i+1}.
\end{equation}

For spin-$1/2$ chains with uniform physical dimension $d_i=d_{i+1}=2$, this is
\[
N_{\mathrm{loc}} = 4 D_{i-1} D_{i+1}.
\]

The eigensolver should accept an initial guess. A good choice is the current two-site center tensor before optimization. This substantially accelerates convergence once sweeps are underway.

\subsection{SVD split and truncation}

After optimization, reshape the two-site tensor as a matrix:
\begin{equation}
\Theta_{(\alpha_{i-1}\sigma_i),(\sigma_{i+1}\alpha_{i+1})}
\in \mathbb{C}^{(D_{i-1}d_i)\times(d_{i+1}D_{i+1})}.
\label{eq:theta-matrix}
\end{equation}

Compute the reduced SVD:
\begin{equation}
\Theta = U S V^\dagger.
\label{eq:theta-svd}
\end{equation}

Let
\[
D_{\mathrm{full}} = \mathrm{rank}(\Theta),\qquad
D_{\mathrm{new}} = \min(D_{\max}, D_{\mathrm{full}})
\]
or more generally choose $D_{\mathrm{new}}$ from a singular-value threshold. Keep only the largest $D_{\mathrm{new}}$ singular values.

The discarded weight is
\begin{equation}
\eta^{[i]} = \sum_{j>D_{\mathrm{new}}} s_j^2.
\label{eq:discarded-weight}
\end{equation}

For a left-to-right sweep, define
\begin{equation}
A^{[i]} \leftarrow \mathrm{reshape}\!\left(U,\,(D_{i-1},d_i,D_{\mathrm{new}})\right),
\label{eq:two-site-left-update}
\end{equation}
\begin{equation}
M^{[i+1]} \leftarrow \mathrm{reshape}\!\left(SV^\dagger,\,(D_{\mathrm{new}},d_{i+1},D_{i+1})\right).
\label{eq:two-site-center-right}
\end{equation}

Then $A^{[i]}$ is left-canonical, and the orthogonality center has moved to site $i+1$.

For a right-to-left sweep, instead use
\begin{equation}
M^{[i]} \leftarrow \mathrm{reshape}\!\left(US,\,(D_{i-1},d_i,D_{\mathrm{new}})\right),
\label{eq:two-site-center-left}
\end{equation}
\begin{equation}
B^{[i+1]} \leftarrow \mathrm{reshape}\!\left(V^\dagger,\,(D_{\mathrm{new}},d_{i+1},D_{i+1})\right),
\label{eq:two-site-right-update}
\end{equation}
so that $B^{[i+1]}$ is right-canonical and the orthogonality center moves to site $i$.

\paragraph{Important reshape note.}
The reshapes in Eqs.~\eqref{eq:two-site-left-update}--\eqref{eq:two-site-right-update} are only correct if Eq.~\eqref{eq:theta-matrix} uses exactly the same flattening convention as the code.

\subsection{Environment updates}

After splitting the optimized two-site tensor, update the environments.

\paragraph{Left-to-right sweep.}
After obtaining the new left-canonical tensor $A^{[i]}$, update the left environment from site $i$ to site $i+1$:
\begin{equation}
L^{[i+1]}_{\beta_i}(\alpha_i,\alpha_i')
=
\sum_{\substack{\alpha_{i-1},\alpha_{i-1}'\\ \sigma_i,\sigma_i'\\ \beta_{i-1}}}
A^{[i]*}_{\alpha_{i-1}',\sigma_i',\alpha_i'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}.
\end{equation}

\paragraph{Right-to-left sweep.}
After obtaining the new right-canonical tensor $B^{[i+1]}$, update the right environment from site $i+1$ to site $i$:
\begin{equation}
R^{[i]}_{\beta_i}(\alpha_i,\alpha_i')
=
\sum_{\substack{\alpha_{i+1},\alpha_{i+1}'\\ \sigma_{i+1},\sigma_{i+1}'\\ \beta_{i+1}}}
B^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}
R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1},\alpha_{i+1}')
B^{[i+1]*}_{\alpha_i',\sigma_{i+1}',\alpha_{i+1}'}.
\end{equation}

\subsection{Full two-site sweep algorithm}

\begin{algorithm}[H]
\caption{TwoSiteDMRG}
\begin{algorithmic}[1]
\Require MPO $\{W^{[i]}\}$, initial MPS $\{M^{[i]}\}$, max bond dimension $D_{\max}$
\Ensure approximate ground-state MPS and energy
\State bring initial MPS into mixed-canonical form
\State build all right environments $R^{[i]}$
\State initialize left boundary environment $L^{[1]}$
\Repeat
    \Comment{Left-to-right sweep}
    \For{$i=1$ to $L-1$}
        \State form the current two-site center tensor $\Theta^{[i,i+1]}$
        \State solve the lowest-eigenvalue problem $\mathcal{H}_{\mathrm{eff}}\Theta = E\Theta$
        \State reshape $\Theta$ as in Eq.~\eqref{eq:theta-matrix}
        \State compute reduced SVD $\Theta = U S V^\dagger$
        \State truncate to $D_{\mathrm{new}}\le D_{\max}$
        \State set $A^{[i]} \gets \mathrm{reshape}(U,(D_{i-1},d_i,D_{\mathrm{new}}))$
        \State set $M^{[i+1]} \gets \mathrm{reshape}(S V^\dagger,(D_{\mathrm{new}},d_{i+1},D_{i+1}))$
        \State record discarded weight $\eta^{[i]}$
        \State update left environment $L^{[i+1]}$
    \EndFor
    \Comment{Right-to-left sweep}
    \For{$i=L-1$ down to $1$}
        \State form the current two-site center tensor $\Theta^{[i,i+1]}$
        \State solve the lowest-eigenvalue problem
        \State reshape $\Theta$ as in Eq.~\eqref{eq:theta-matrix}
        \State compute reduced SVD $\Theta = U S V^\dagger$
        \State truncate to $D_{\mathrm{new}}\le D_{\max}$
        \State set $M^{[i]} \gets \mathrm{reshape}(US,(D_{i-1},d_i,D_{\mathrm{new}}))$
        \State set $B^{[i+1]} \gets \mathrm{reshape}(V^\dagger,(D_{\mathrm{new}},d_{i+1},D_{i+1}))$
        \State record discarded weight $\eta^{[i]}$
        \State update right environment $R^{[i]}$
    \EndFor
\Until{energy, variance, and discarded-weight criteria are converged}
\end{algorithmic}
\end{algorithm}

\subsection{Practical comments for coding}

\begin{itemize}
    \item \textbf{This should be the first DMRG implementation.} It is much easier to debug than one-site DMRG.
    \item \textbf{Keep the orthogonality center explicit.} During a left-to-right sweep, after the split the center lives on site $i+1$ as $M^{[i+1]}$; during a right-to-left sweep, it lives on site $i$ as $M^{[i]}$.
    \item \textbf{Do not compare tensors directly across sweeps without gauge fixing.}
    \item \textbf{Always unit test the local effective-Hamiltonian matvec} against a dense reference for small dimensions.
    \item \textbf{Monitor the maximum discarded weight}
    \[
    \eta_{\max} = \max_i \eta^{[i]}
    \]
    during each sweep.
    \item \textbf{Benchmark first against exact diagonalization} for small Heisenberg chains before attempting larger systems.
\end{itemize}

\subsection{Recommended initial benchmarks}

For a first two-site DMRG code, test the following:

\begin{enumerate}
    \item \textbf{Small Heisenberg chains} ($L=4,6,8,10$), comparing energies to exact diagonalization.
    \item \textbf{AKLT chain} with $D_{\max}=2$, verifying exact recovery and zero discarded weight.
    \item \textbf{Environment consistency}: compare environment-based energy evaluation with brute-force contraction on small systems.
    \item \textbf{Canonicality checks}: after each split, verify left- or right-canonical conditions numerically.
\end{enumerate}
```

---

# Part IV — Concrete fixes I recommend to your current file

Here are the exact changes I would make to your current note before calling it coding-blueprint ready.

---

## Fix A — one-site matvec einsum typo

Replace:
```python
Y = np.einsum('bBst,bxsa->Bx ta', W, X)
```
with:
```python
Y = np.einsum('bBst,bxsa->Bxta', W, X)
```

---

## Fix B — strengthen right-canonicalization subsection

Replace your current short paragraph with:

```latex
\subsection{Right sweep by reduced QR}

To right-canonicalize site $i$, reshape
\[
M^{[i]}_{\alpha_{i-1},(\sigma_i\alpha_i)}
\in \mathbb{C}^{D_{i-1}\times(d_iD_i)}.
\]
Compute a reduced QR factorization of the Hermitian transpose:
\begin{equation}
\left(M^{[i]}_{\alpha_{i-1},(\sigma_i\alpha_i)}\right)^\dagger = Q R,
\end{equation}
with
\[
Q\in\mathbb{C}^{(d_iD_i)\times k},\qquad
R\in\mathbb{C}^{k\times D_{i-1}},\qquad
k=\min(d_iD_i,D_{i-1}).
\]
Then
\begin{equation}
M^{[i]} = R^\dagger Q^\dagger.
\end{equation}
Reshape $Q^\dagger$ into a right-canonical tensor
\[
B^{[i]}_{\tilde\alpha_{i-1},\sigma_i,\alpha_i},
\]
with new bond dimension $\tilde D_{i-1}=k$, and absorb $R^\dagger$ into site $i-1$:
\begin{equation}
M^{[i-1]}_{\alpha_{i-2},\sigma_{i-1},\tilde\alpha_{i-1}}
\leftarrow
\sum_{\alpha_{i-1}}
M^{[i-1]}_{\alpha_{i-2},\sigma_{i-1},\alpha_{i-1}}
(R^\dagger)_{\alpha_{i-1},\tilde\alpha_{i-1}}.
\end{equation}
```

---

## Fix C — add one convention sentence before environments

Add:
```latex
We use the MPO convention
\[
\hat W^{[i]}=\sum_{\sigma_i,\sigma_i'} 
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
\ket{\sigma_i}\bra{\sigma_i'},
\]
so the first physical index is the ket index and the second is the bra index.
```

This makes the environment equations unambiguous.

---

## Fix D — replace your current two-site section by the detailed one above

That is the biggest improvement.

---

# Final verdict

## Is the revised note now close?
**Yes. Very close.**

## Is it already mathematically consistent enough to use directly as a coding blueprint?
**Not quite yet.**  
But after the four fixes above—especially the detailed two-site section—**yes, I would consider it solid enough for practical coding**.

## Best implementation path
If you are actually building code from this note, I recommend:

1. **Implement MPS/MPO tensor objects**
2. **Implement QR/SVD gauge moves**
3. **Implement environments**
4. **Implement two-site DMRG only**
5. **Benchmark against ED**
6. Only then add one-site DMRG

If you want, I can now do one of these next:

1. give you a **fully integrated corrected LaTeX file** with the strengthened two-site section inserted, or  
2. give you a **NumPy-style reference implementation skeleton** for two-site DMRG.

---
## 🧑‍💻 User

Where to put Fix C?

---
## 🤖 Assistant

Put **Fix C** at the **start of Section 6, `MPO environments`**, immediately **before** the sentence

> For one-site DMRG with orthogonality center at site \(i\), define the left environment ...

So that section should begin like this:

```latex
\section{MPO environments}

We use the MPO convention
\[
\hat W^{[i]}=\sum_{\sigma_i,\sigma_i'}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
\ket{\sigma_i}\bra{\sigma_i'},
\]
so the first physical index is the ket index and the second is the bra index.

For one-site DMRG with orthogonality center at site $i$, define the left environment
...
```

## Why there?
Because Fix C is only needed to remove ambiguity in:
- the **left/right environment recursions**,
- the **effective Hamiltonian action**,

and those both depend on how MPO physical indices are interpreted.

## Optional better placement
If you want the document to be even cleaner, you could instead place it earlier in **Section 1, `Scope and conventions`**, right after the MPO tensor definition:
```latex
\item MPO tensor on site $i$:
\[
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'},
\]
with shape $(\chi_{i-1}, \chi_i, d_i, d_i)$. We use the operator convention
\[
\hat W^{[i]}=
\sum_{\sigma_i,\sigma_i'}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
\ket{\sigma_i}\bra{\sigma_i'},
\]
so the first physical index is the ket index and the second is the bra index.
```

That is actually the **best global location**.

### Recommendation
- **Best document-wide location:** Section 1, right after MPO tensor definition.
- **Minimum fix location:** start of Section 6.

If you want, I can now show you the **exact patched text block** for both placement options.

---
## 🧑‍💻 User

I prefer to Section 1.

---
## 🤖 Assistant

Yes — **Section 1 is the better place**.

Insert **Fix C** directly under the MPO tensor bullet in `\section{Scope and conventions}`.

## Replace this part
```latex
\item MPO tensor on site $i$:
\[
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'},
\]
with shape $(\chi_{i-1}, \chi_i, d_i, d_i)$.
```

## With this
```latex
\item MPO tensor on site $i$:
\[
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'},
\]
with shape $(\chi_{i-1}, \chi_i, d_i, d_i)$. We use the operator convention
\[
\hat W^{[i]}
=
\sum_{\sigma_i,\sigma_i'}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
\ket{\sigma_i}\bra{\sigma_i'},
\]
so the first physical index is the ket index and the second is the bra index.
```

## Then the start of Section 1 becomes
```latex
\section{Scope and conventions}

These implementation notes summarize the practical coding structure of matrix product state (MPS), matrix product operator (MPO), and finite-system DMRG algorithms for one-dimensional systems with open boundary conditions (OBC). The presentation is based on Schollw\"ock's review article, but rewritten with explicit implementation conventions suitable for numerical work.

We use the following tensor conventions throughout:
\begin{itemize}
    \item MPS tensor on site $i$:
    \[
    A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i},
    \]
    with shape $(D_{i-1}, d_i, D_i)$.
    \item MPO tensor on site $i$:
    \[
    W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'},
    \]
    with shape $(\chi_{i-1}, \chi_i, d_i, d_i)$. We use the operator convention
    \[
    \hat W^{[i]}
    =
    \sum_{\sigma_i,\sigma_i'}
    W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
    \ket{\sigma_i}\bra{\sigma_i'},
    \]
    so the first physical index is the ket index and the second is the bra index.
    \item OBC dimensions:
    \[
    D_0 = D_L = 1,\qquad \chi_0 = \chi_L = 1.
    \]
\end{itemize}
```

That is exactly where I would put it. If you want, next I can give you the **fully updated LaTeX file with all remaining fixes integrated**.

---
## 🧑‍💻 User

This is the revised LaTeX file. Please go through it.

---
## 🤖 Assistant

This is now **very close** to a usable coding blueprint. The structure is good, the conventions are mostly consistent, and the two-site section is much stronger.

But after going through it carefully, I still see **a handful of important issues**—mostly local, but a few are mathematically or implementation-significant.

I’ll organize this as:

1. **Overall verdict**
2. **What is now correct**
3. **What still needs to be fixed**
4. **Exact patch suggestions**
5. **Final readiness assessment**

---

# 1. Overall verdict

## Short verdict
**This version is almost good enough for coding, but not yet fully safe as a blueprint.**

## Main remaining blockers
There are **four significant issues**:

1. **The two-site effective-Hamiltonian formula has bra/ket physical indices reversed relative to the one-site convention.**
2. **The two-site sample NumPy einsums are not correct and should not be left in the note in their current form.**
3. **The one-site effective-Hamiltonian NumPy line for the final contraction is dimensionally ambiguous and should be rewritten more explicitly.**
4. **The AKLT “appropriate open-chain ground-state sector” wording is too vague for a benchmark note.**

If you fix those, then I would say:

> **Yes, this is mathematically consistent enough to be used as a practical coding blueprint**, especially for a first two-site DMRG implementation.

---

# 2. What is now correct

A lot is now in good shape.

## Good and consistent now
- Section 1 tensor conventions
- Section 2 reshape conventions
- left/right canonical definitions
- left QR sweep
- right QR sweep
- SVD truncation logic
- one-site environment recursion structure
- one-site effective Hamiltonian equation
- one-site sweep structure
- two-site mixed-canonical formulation
- two-site SVD split logic
- left/right environment updates after two-site split
- Heisenberg MPO bulk and boundaries
- AKLT canonicality
- convergence and implementation checklist

So the document is now **substantially sound**.

---

# 3. What still needs to be fixed

---

## Issue A — Two-site effective Hamiltonian physical index ordering is inconsistent

In Section 1 you defined the MPO convention as
\[
\hat W^{[i]}
=
\sum_{\sigma_i,\sigma_i'}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
\ket{\sigma_i}\bra{\sigma_i'},
\]
so:
- first physical index = **ket**
- second physical index = **bra**

That means in the effective Hamiltonian action, the MPO should act from input physical indices \((\sigma_i',\sigma_{i+1}')\) to output physical indices \((\sigma_i,\sigma_{i+1})\) as
\[
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
\]
and similarly for site \(i+1\).

But your two-site equation currently reads:
\[
(\mathcal{H}_{\mathrm{eff}} \Theta)_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
=
\sum
L\,
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i',\sigma_i}
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1}',\sigma_{i+1}}
R\,
\Theta_{\alpha_{i-1}',\sigma_i',\sigma_{i+1}',\alpha_{i+1}'}.
\]
This uses \(W[\text{bra},\text{ket}]\), not \(W[\text{ket},\text{bra}]\).

## Fix
Replace Eq.~\eqref{eq:two-site-heff} by
\begin{align}
(\mathcal{H}_{\mathrm{eff}} \Theta)_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
=
\sum_{\substack{
\alpha_{i-1}',\alpha_{i+1}'\\
\sigma_i',\sigma_{i+1}'\\
\beta_{i-1},\beta_i,\beta_{i+1}
}}
&L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}
\nonumber\\
&\times
R^{[i+1]}_{\beta_{i+1}}

---
## 🧑‍💻 User

your output breaks off, pls continue

---
## 🤖 Assistant

Continuing.

The corrected equation should be:

```latex
\begin{align}
(\mathcal{H}_{\mathrm{eff}} \Theta)_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
=
\sum_{\substack{
        \alpha_{i-1}',\alpha_{i+1}'\\
        \sigma_i',\sigma_{i+1}'\\
        \beta_{i-1},\beta_i,\beta_{i+1}
}}
&L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}
\nonumber\\
&\times
R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1}',\alpha_{i+1})
\Theta_{\alpha_{i-1}',\sigma_i',\sigma_{i+1}',\alpha_{i+1}'}.
\label{eq:two-site-heff}
\end{align}
```

That makes the two-site section consistent with your global MPO convention and with the one-site formula.

---

## Issue B — The two-site sample NumPy einsums are not correct enough to keep

This block is currently dangerous:

```python
X  = np.einsum('bxy,ysta->bxsta', L, Theta)
Y  = np.einsum('bBsu,bxuta->Bxsta', W_i, X)
Z  = np.einsum('BCtv,BxsaC->Cxatv', W_ip1, Y)   # example only; relabel clearly
HT = np.einsum('Bxsua,Bac->xscu', Z, R)
```

### Problems
1. The index names are inconsistent across lines.
2. `W_i` and `W_ip1` use different symbol conventions than the equations.
3. The line for `Z` is not trustworthy as written.
4. The final output index order does not obviously match \((\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1})\).

### Recommendation
For a coding blueprint, either:
- give a fully correct tested einsum chain, or
- remove the code block and keep only the symbolic contraction equations.

I strongly recommend the second option unless you have tested the exact strings.

## Best safe replacement
Replace that paragraph with:

```latex
A corresponding NumPy implementation can be written using \texttt{einsum}, but because multi-step index strings are easy to get wrong, the symbolic contractions above should be treated as the authoritative definition. In practice, the matrix-free routine should be unit-tested against an explicitly constructed dense effective Hamiltonian for very small dimensions.
```

That is safer than leaving a possibly wrong code example in a blueprint.

---

## Issue C — One-site NumPy final contraction should be rewritten more explicitly

Current block:
```python
X = np.einsum('bxy,ysa->bxsa', L, M)
Y = np.einsum('bBst,bxsa->Bxta', W, X)
HM = np.einsum('Bxta,Bai->xti', Y, R)
```

This is close, but the index meaning in the last line is not explicit enough.

If
- `Y[B, x, t, a]`
- `R[B, a, i]`

then
```python
HM = np.einsum('Bxta,Bai->xti', Y, R)
```
is okay.

But your earlier symbolic equations write the right environment as
\[
R^{[i]}_{\beta_i}(\alpha_i',\alpha_i),
\]
so I would prefer the NumPy labels to visibly reflect that.

## Safer rewrite
Change to:
```python
X  = np.einsum('bxy,ysz->bxsz', L, M)
Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
HM = np.einsum('Bxtz,Bza->xta', Y, R)
```

This is clearer:
- \(x=\alpha_{i-1}\)
- \(t=\sigma_i'\)
- \(z=\alpha_i'\)
- \(a=\alpha_i\)

So the output `xta` is exactly \((\alpha_{i-1},\sigma_i',\alpha_i)\).

That’s the version I would put in the note.

---

## Issue D — Right environment section should explicitly state what sites it contains

This is not a mathematical error, but it matters for coding.

Right now you say:
> let \(R^{[i+1]}\) be the right environment built from sites \(i+2,\dots,L\)

Good in the two-site section.

But earlier, in the main `MPO environments` section, you don’t explicitly say:
- \(L^{[i]}\) contains sites \(1,\dots,i-1\)
- \(R^{[i]}\) contains sites \(i+1,\dots,L\)

## Fix
Add at the start of Section 6:

```latex
Here $L^{[i]}$ denotes the contraction of sites $1,\dots,i-1$ into a left environment for site $i$, while $R^{[i]}$ denotes the contraction of sites $i+1,\dots,L$ into a right environment for site $i$.
```

This makes later formulas easier to interpret.

---

## Issue E — AKLT benchmark wording should be made exact

Current wording:
> for the appropriate open-chain ground-state sector.

This is too vague for a benchmark note.

The clean statement is:

```latex
For open boundary conditions, the AKLT Hamiltonian has an exactly degenerate edge-state manifold, but every ground state in that manifold has the same exact energy
\[
E_0 = -\frac{2}{3}(L-1).
\]
```

That is both precise and coding-friendly.

---

## Issue F — One-site DMRG language about truncation is slightly misleading

This is a smaller point.

In one-site DMRG, the bond dimension is typically held fixed during the local optimization, and the SVD is mainly used to move the gauge center. Your current wording says “truncate to \(D_{\max}\)” in every one-site step.

That is not wrong in a generic implementation, but it is not the canonical way to explain one-site DMRG.

## Better wording
Replace:
> truncate to \(D_{\max}\)

with:
> retain at most \(D_{\max}\) singular values; in standard one-site DMRG this step primarily shifts the orthogonality center, while any actual change in bond dimension is usually small or absent unless rank deficiency or adaptive bond control is used.

This is optional, but more precise.

---

# 4. Exact patch suggestions

Here are the exact changes I recommend.

---

## Patch 1 — Fix the one-site NumPy block

Replace:
```latex
\begin{verbatim}
    X = np.einsum('bxy,ysa->bxsa', L, M)
    Y = np.einsum('bBst,bxsa->Bxta', W, X)
    HM = np.einsum('Bxta,Bai->xti', Y, R)
\end{verbatim}
```

with:
```latex
\begin{verbatim}
    X  = np.einsum('bxy,ysz->bxsz', L, M)
    Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
    HM = np.einsum('Bxtz,Bza->xta', Y, R)
\end{verbatim}
```

---

## Patch 2 — Add explicit environment semantics

At the start of `\section{MPO environments}`, insert:

```latex
Here $L^{[i]}$ denotes the contraction of sites $1,\dots,i-1$ into a left environment for site $i$, while $R^{[i]}$ denotes the contraction of sites $i+1,\dots,L$ into a right environment for site $i$.
```

---

## Patch 3 — Fix the two-site effective Hamiltonian equation

Replace the current Eq.~\eqref{eq:two-site-heff} with the corrected version:

```latex
\begin{align}
    (\mathcal{H}_{\mathrm{eff}} \Theta)_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
    =
    \sum_{\substack{
            \alpha_{i-1}',\alpha_{i+1}'\\
            \sigma_i',\sigma_{i+1}'\\
            \beta_{i-1},\beta_i,\beta_{i+1}
    }}
    &L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
    W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
    W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}
    \nonumber\\
    &\times
    R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1}',\alpha_{i+1})
    \Theta_{\alpha_{i-1}',\sigma_i',\sigma_{i+1}',\alpha_{i+1}'}.
    \label{eq:two-site-heff}
\end{align}
```

---

## Patch 4 — Remove or neutralize the two-site sample einsum code

Replace the whole code block
```latex
\begin{verbatim}
    X  = np.einsum('bxy,ysta->bxsta', L, Theta)
    Y  = np.einsum('bBsu,bxuta->Bxsta', W_i, X)
    Z  = np.einsum('BCtv,BxsaC->Cxatv', W_ip1, Y)   # example only; relabel clearly
    HT = np.einsum('Bxsua,Bac->xscu', Z, R)
\end{verbatim}
```

with:

```latex
A corresponding NumPy implementation can be written using \texttt{einsum}, but because multi-step index strings are easy to get wrong, the symbolic contractions above should be treated as the authoritative definition. In practice, the matrix-free routine should be unit-tested against an explicitly constructed dense effective Hamiltonian for very small dimensions.
```

This is much safer for a blueprint note.

---

## Patch 5 — Tighten AKLT benchmark wording

Replace:
```latex
\item exact finite-chain ground-state energy for the AKLT Hamiltonian
...
is
\[
E_0 = -\frac23 (L-1)
\]
for the appropriate open-chain ground-state sector.
```

with:
```latex
\item for open boundary conditions, the AKLT Hamiltonian has an exactly degenerate edge-state manifold, but every ground state in that manifold has the same exact energy
\[
E_0 = -\frac23 (L-1).
\]
```

---

# 5. Final readiness assessment

## After the above fixes:
I would say:

> **Yes, this note is mathematically consistent enough to be used as a coding blueprint for a first two-site DMRG implementation.**

## What I would still not use it for yet
I would still not treat it as a final blueprint for:
- one-site DMRG with subspace expansion
- time evolution
- periodic boundary conditions
- symmetry-resolved block sparse implementation

But for:
- MPS basics
- canonicalization
- MPO construction
- environments
- one-site effective Hamiltonian
- two-site DMRG

it will then be in good shape.

---

If you want, the next most useful step is for me to provide either:

1. a **fully patched final LaTeX file**, or  
2. a **minimal NumPy reference skeleton for two-site DMRG**, matching exactly these conventions.

---
## 🧑‍💻 User

Plese give me a fully patched final LaTeX file.

---
## 🤖 Assistant

```latex
\documentclass[aps,prb,onecolumn,scriptaddress]{revtex4-2}

\usepackage{graphicx}
\usepackage{amsmath,amsfonts,amssymb,bm,mathtools}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{xcolor}
\usepackage[colorlinks=true,urlcolor=blue,linkcolor=blue,citecolor=blue]{hyperref}

\newcommand{\ket}[1]{|#1\rangle}
\newcommand{\bra}[1]{\langle #1|}
\newcommand{\braket}[2]{\langle #1|#2\rangle}
\newcommand{\Tr}{\mathrm{Tr}}
\newcommand{\Id}{\mathbb{I}}
\newcommand{\eps}{\epsilon}
\newcommand{\Order}{\mathcal{O}}

\begin{document}

\title{Numerical Implementation of Matrix Product States and DMRG\\
\large Based on the review article ``The density-matrix renormalization group in the age of matrix product states'' by U.~Schollw\"ock, Annals of Physics \textbf{326}, 96--192 (2011)}

\author{Kimi 2.5}
\author{GPT 5.4}
\date{\today}

\maketitle

\section{Scope and conventions}

These implementation notes summarize the practical coding structure of matrix product state (MPS), matrix product operator (MPO), and finite-system DMRG algorithms for one-dimensional systems with open boundary conditions (OBC). The presentation is based on Schollw\"ock's review article, but rewritten with explicit implementation conventions suitable for numerical work.

We use the following tensor conventions throughout:
\begin{itemize}
    \item MPS tensor on site $i$:
    \[
    A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i},
    \]
    with shape $(D_{i-1}, d_i, D_i)$.
    \item MPO tensor on site $i$:
    \[
    W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'},
    \]
    with shape $(\chi_{i-1}, \chi_i, d_i, d_i)$. We use the operator convention
    \[
    \hat W^{[i]}
    =
    \sum_{\sigma_i,\sigma_i'}
    W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
    \ket{\sigma_i}\bra{\sigma_i'},
    \]
    so the first physical index is the ket index and the second is the bra index.
    \item OBC dimensions:
    \[
    D_0 = D_L = 1,\qquad \chi_0 = \chi_L = 1.
    \]
\end{itemize}

Unless stated otherwise, all tensors should be stored as \texttt{complex128}.

\section{Reshape conventions}

For numerical stability and reproducibility, reshape conventions must be fixed once and used everywhere.

For a rank-3 MPS tensor
\[
T^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i},
\]
with storage shape $(D_{i-1},d_i,D_i)$, define the left-grouped matrix
\begin{equation}
T^{[i]}_{(\alpha_{i-1}\sigma_i),\alpha_i}
\in \mathbb{C}^{(D_{i-1}d_i)\times D_i},
\label{eq:left-group}
\end{equation}
and the right-grouped matrix
\begin{equation}
T^{[i]}_{\alpha_{i-1},(\sigma_i\alpha_i)}
\in \mathbb{C}^{D_{i-1}\times(d_iD_i)}.
\label{eq:right-group}
\end{equation}

In row-major (C-order) storage, the flattening used in Eq.~\eqref{eq:left-group} is
\begin{equation}
r = \alpha_{i-1} d_i + \sigma_i,\qquad c = \alpha_i.
\end{equation}

In NumPy:
\begin{verbatim}
M_left = T.reshape((D_left * d, D_right), order='C')
M_right = T.reshape((D_left, d * D_right), order='C')
\end{verbatim}

All QR/SVD steps must be consistent with these conventions, and the inverse reshape must use the same memory order and grouped-index convention.

\section{Canonical forms}

For fixed physical index $\sigma$, one may view $A^{[i]\sigma}$ as a matrix from the left bond to the right bond.

Using index notation avoids ambiguity. A tensor $A^{[i]}$ is \emph{left-canonical} if
\begin{equation}
\sum_{\alpha_{i-1},\sigma_i}
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
A^{[i]*}_{\alpha_{i-1},\sigma_i,\alpha_i'}
=
\delta_{\alpha_i,\alpha_i'}.
\label{eq:left-canonical}
\end{equation}
Equivalently, if $A^{[i]}$ is reshaped as in Eq.~\eqref{eq:left-group}, then
\begin{equation}
\left(A^{[i]}_{(\alpha_{i-1}\sigma_i),\alpha_i}\right)^\dagger
A^{[i]}_{(\alpha_{i-1}\sigma_i),\alpha_i}
= \Id_{D_i}.
\end{equation}

A tensor $B^{[i]}$ is \emph{right-canonical} if
\begin{equation}
\sum_{\sigma_i,\alpha_i}
B^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
B^{[i]*}_{\alpha_{i-1}',\sigma_i,\alpha_i}
=
\delta_{\alpha_{i-1},\alpha_{i-1}'}.
\label{eq:right-canonical}
\end{equation}
Equivalently, if $B^{[i]}$ is reshaped as in Eq.~\eqref{eq:right-group}, then
\begin{equation}
B^{[i]}_{\alpha_{i-1},(\sigma_i\alpha_i)}
\left(B^{[i]}_{\alpha_{i-1},(\sigma_i\alpha_i)}\right)^\dagger
= \Id_{D_{i-1}}.
\end{equation}

A mixed-canonical state with orthogonality center at bond $i$ may be written as
\begin{equation}
\ket{\psi} =
\sum_{\alpha_i}
s^{[i]}_{\alpha_i}
\ket{\alpha_i}_L \ket{\alpha_i}_R,
\end{equation}
where $s^{[i]}_{\alpha_i}\ge 0$ are Schmidt singular values and the left/right Schmidt states are orthonormal.

\section{Canonicalization algorithms}

\subsection{Left sweep by reduced QR}

Given a general tensor $M^{[i]}$, reshape it as
\[
M^{[i]}_{(\alpha_{i-1}\sigma_i),\alpha_i}.
\]
Perform a reduced QR factorization
\begin{equation}
M^{[i]}_{(\alpha_{i-1}\sigma_i),\alpha_i} = Q R,
\end{equation}
where
\[
Q\in \mathbb{C}^{(D_{i-1}d_i)\times k},\qquad
R\in \mathbb{C}^{k\times D_i},\qquad
k=\min(D_{i-1}d_i,D_i).
\]
Reshape $Q$ into a left-canonical tensor
\[
A^{[i]}_{\alpha_{i-1},\sigma_i,\tilde\alpha_i},
\]
with new bond dimension $\tilde D_i = k$, then absorb $R$ into the next tensor:
\begin{equation}
M^{[i+1]}_{\tilde\alpha_i,\sigma_{i+1},\alpha_{i+1}}
\leftarrow
\sum_{\alpha_i}
R_{\tilde\alpha_i,\alpha_i}
M^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}.
\label{eq:left-qr-absorb}
\end{equation}

\begin{algorithm}[H]
\caption{LeftCanonicalizeByQR}
\begin{algorithmic}[1]
\Require MPS tensors $\{M^{[1]},\dots,M^{[L]}\}$
\Ensure Left-canonical tensors $\{A^{[1]},\dots,A^{[L-1]}\}$ and final center tensor $C^{[L]}$
\For{$i=1$ to $L-1$}
    \State reshape $M^{[i]} \to \hat M$ with shape $(D_{i-1}d_i)\times D_i$
    \State compute reduced QR: $\hat M = Q R$
    \State reshape $Q \to A^{[i]}$ with shape $(D_{i-1},d_i,\tilde D_i)$
    \State absorb $R$ into $M^{[i+1]}$ via Eq.~\eqref{eq:left-qr-absorb}
\EndFor
\State set $C^{[L]} \gets M^{[L]}$
\end{algorithmic}
\end{algorithm}

\paragraph{Important practical note.}
For canonicalization, use \emph{reduced QR without pivoting}. Pivoted QR changes the bond basis by a permutation and is inconvenient unless the permutation is explicitly propagated into neighboring tensors.

\subsection{Right sweep by reduced QR}

To right-canonicalize site $i$, reshape
\[
M^{[i]}_{\alpha_{i-1},(\sigma_i\alpha_i)}
\in \mathbb{C}^{D_{i-1}\times(d_iD_i)}.
\]
Compute a reduced QR factorization of the Hermitian transpose:
\begin{equation}
\left(M^{[i]}_{\alpha_{i-1},(\sigma_i\alpha_i)}\right)^\dagger = Q R,
\end{equation}
with
\[
Q\in\mathbb{C}^{(d_iD_i)\times k},\qquad
R\in\mathbb{C}^{k\times D_{i-1}},\qquad
k=\min(d_iD_i,D_{i-1}).
\]
Then
\begin{equation}
M^{[i]} = R^\dagger Q^\dagger.
\end{equation}
Reshape $Q^\dagger$ into a right-canonical tensor
\[
B^{[i]}_{\tilde\alpha_{i-1},\sigma_i,\alpha_i},
\]
with new bond dimension $\tilde D_{i-1}=k$, and absorb $R^\dagger$ into site $i-1$:
\begin{equation}
M^{[i-1]}_{\alpha_{i-2},\sigma_{i-1},\tilde\alpha_{i-1}}
\leftarrow
\sum_{\alpha_{i-1}}
M^{[i-1]}_{\alpha_{i-2},\sigma_{i-1},\alpha_{i-1}}
(R^\dagger)_{\alpha_{i-1},\tilde\alpha_{i-1}}.
\end{equation}

\section{SVD truncation}

Given a matrix $X\in\mathbb{C}^{m\times n}$ with reduced SVD
\begin{equation}
X = U S V^\dagger,
\end{equation}
where
\[
U\in\mathbb{C}^{m\times r},\quad
S=\mathrm{diag}(s_1,\dots,s_r),\quad
V^\dagger\in\mathbb{C}^{r\times n},\quad
s_1\ge s_2\ge \cdots \ge s_r\ge 0,
\]
truncate to bond dimension $D$ by keeping the $D$ largest singular values:
\begin{equation}
X \approx U_D S_D V_D^\dagger.
\end{equation}

The discarded weight is
\begin{equation}
\eta = \sum_{j>D} s_j^2.
\end{equation}
If the state is normalized, $\eta$ is the squared 2-norm of the discarded Schmidt tail.

\begin{algorithm}[H]
\caption{TruncateBySVD}
\begin{algorithmic}[1]
\Require matrix $X$, max bond dimension $D_{\max}$, tolerance $\eps$
\Ensure truncated $U_D,S_D,V_D^\dagger$, discarded weight $\eta$
\State compute reduced SVD: $X = U S V^\dagger$
\State let $D = \min\{D_{\max},\#(s_j > \eps)\}$
\If{$D=0$}
    \State $D\gets 1$
\EndIf
\State keep first $D$ singular values/vectors
\State $\eta \gets \sum_{j>D} s_j^2$
\end{algorithmic}
\end{algorithm}

\section{MPO environments}

Here $L^{[i]}$ denotes the contraction of sites $1,\dots,i-1$ into a left environment for site $i$, while $R^{[i]}$ denotes the contraction of sites $i+1,\dots,L$ into a right environment for site $i$.

For one-site DMRG with orthogonality center at site $i$, define the left environment
\begin{equation}
L^{[i]}_{\beta_i}(\alpha_{i-1},\alpha_{i-1}')
=
\sum_{\substack{\alpha_{i-2},\alpha_{i-2}'\\ \sigma,\sigma'\\ \beta_{i-1}}}
A^{[i-1]*}_{\alpha_{i-2}',\sigma',\alpha_{i-1}'}
L^{[i-1]}_{\beta_{i-1}}(\alpha_{i-2},\alpha_{i-2}')
W^{[i-1]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
A^{[i-1]}_{\alpha_{i-2},\sigma,\alpha_{i-1}},
\label{eq:left-env}
\end{equation}
and the right environment
\begin{equation}
R^{[i]}_{\beta_{i-1}}(\alpha_i,\alpha_i')
=
\sum_{\substack{\alpha_{i+1},\alpha_{i+1}'\\ \sigma,\sigma'\\ \beta_i}}
B^{[i+1]}_{\alpha_i,\sigma,\alpha_{i+1}}
W^{[i+1]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
R^{[i+1]}_{\beta_i}(\alpha_{i+1},\alpha_{i+1}')
B^{[i+1]*}_{\alpha_i',\sigma',\alpha_{i+1}'}.
\label{eq:right-env}
\end{equation}

Boundary initialization for OBC:
\begin{equation}
L^{[1]}_{\beta_0} = \delta_{\beta_0,1},\qquad
R^{[L]}_{\beta_L} = \delta_{\beta_L,1},
\end{equation}
where the MPO boundary dimensions are $\chi_0=\chi_L=1$.

\paragraph{Safe einsum template.}
With arrays
\begin{itemize}
    \item \texttt{Lold[b,x,y]}
    \item \texttt{A[x,s,a]}
    \item \texttt{W[b,B,s,t]}
    \item \texttt{Aconj[y,t,c] = A[y,t,c].conj()}
\end{itemize}
a correct left update is
\begin{verbatim}
Lnew = np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj())
\end{verbatim}
up to index naming.

\section{One-site effective Hamiltonian}

Let the center tensor at site $i$ be
\[
M^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}.
\]
The local variational problem is
\begin{equation}
\mathcal{H}_{\mathrm{eff}} \, v = E v,
\end{equation}
where $v$ is the vectorized center tensor.

In tensor form,
\begin{equation}
(\mathcal{H}_{\mathrm{eff}} M)_{\alpha_{i-1},\sigma_i',\alpha_i}
=
\sum_{\substack{\alpha_{i-1}',\alpha_i'\\ \sigma_i\\ \beta_{i-1},\beta_i}}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
R^{[i]}_{\beta_i}(\alpha_i',\alpha_i)
M^{[i]}_{\alpha_{i-1}',\sigma_i,\alpha_i'}.
\label{eq:heff-action}
\end{equation}

A safe matrix-free contraction sequence is:
\begin{align}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\alpha_i'}
&=
\sum_{\alpha_{i-1}'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
M^{[i]}_{\alpha_{i-1}',\sigma_i,\alpha_i'},
\\
Y_{\beta_i,\alpha_{i-1},\sigma_i',\alpha_i'}
&=
\sum_{\beta_{i-1},\sigma_i}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\alpha_i'},
\\
(\mathcal{H}_{\mathrm{eff}} M)_{\alpha_{i-1},\sigma_i',\alpha_i}
&=
\sum_{\beta_i,\alpha_i'}
Y_{\beta_i,\alpha_{i-1},\sigma_i',\alpha_i'}
R^{[i]}_{\beta_i}(\alpha_i',\alpha_i).
\end{align}

Corresponding NumPy contractions:
\begin{verbatim}
X  = np.einsum('bxy,ysz->bxsz', L, M)
Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
HM = np.einsum('Bxtz,Bza->xta', Y, R)
\end{verbatim}
with appropriate relabeling to match array order.

\section{One-site DMRG sweep}

In one-site DMRG, the MPS is kept in mixed-canonical form with center at site $i$:
\begin{equation}
\ket{\psi}
=
\sum_{\alpha_{i-1},\sigma_i,\alpha_i}
M^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
\ket{\alpha_{i-1}}_L \ket{\sigma_i} \ket{\alpha_i}_R.
\end{equation}

\begin{algorithm}[H]
\caption{OneSiteDMRG}
\begin{algorithmic}[1]
\Require MPO $\{W^{[i]}\}$, initial MPS $\{M^{[i]}\}$, max bond dimension $D_{\max}$
\Ensure approximate ground state MPS and energy
\State right-canonicalize initial MPS
\State build all right environments
\State initialize left boundary
\Repeat
    \For{$i=1$ to $L-1$}
        \State solve Eq.~\eqref{eq:heff-action} for lowest eigenpair using Lanczos/Davidson
        \State reshape center tensor as left-grouped matrix
        \State SVD: $M^{[i]} = U S V^\dagger$
        \State retain at most $D_{\max}$ singular values; in standard one-site DMRG this step primarily shifts the orthogonality center, while any actual change in bond dimension is usually small or absent unless rank deficiency or adaptive bond control is used
        \State reshape $U \to A^{[i]}$ (left-canonical)
        \State absorb $S V^\dagger$ into $M^{[i+1]}$
        \State update $L^{[i+1]}$
    \EndFor
    \For{$i=L$ down to $2$}
        \State solve local lowest-eigenvalue problem
        \State reshape center tensor as right-grouped matrix
        \State SVD: $M^{[i]} = U S V^\dagger$
        \State retain at most $D_{\max}$ singular values
        \State reshape $V^\dagger$ using the inverse of Eq.~\eqref{eq:right-group} to obtain a right-canonical tensor $B^{[i]}$ with shape $(D_{i-1}^{\mathrm{new}},d_i,D_i)$
        \State absorb $U S$ into $M^{[i-1]}$
        \State update $R^{[i-1]}$
    \EndFor
\Until{energy and variance converged}
\end{algorithmic}
\end{algorithm}

\paragraph{Practical warning.}
One-site DMRG is prone to local minima. A robust implementation should either:
\begin{itemize}
    \item start with several two-site sweeps, or
    \item use a subspace-expansion / density-matrix perturbation method.
\end{itemize}

\section{Two-site DMRG}

Two-site DMRG is the most robust finite-system ground-state algorithm and is recommended for a first implementation. It avoids many local-minimum problems of one-site DMRG and dynamically adjusts the bond dimension before truncation.

\subsection{Two-site mixed-canonical form}

At bond $(i,i+1)$, define left-orthonormal Schmidt states on sites $1,\dots,i-1$ and right-orthonormal Schmidt states on sites $i+2,\dots,L$:
\[
\ket{\alpha_{i-1}}_L,\qquad \ket{\alpha_{i+1}}_R.
\]
The variational two-site center tensor is
\begin{equation}
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}},
\label{eq:theta-def}
\end{equation}
and the state is written as
\begin{equation}
\ket{\psi}
=
\sum_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
\ket{\alpha_{i-1}}_L \ket{\sigma_i}\ket{\sigma_{i+1}} \ket{\alpha_{i+1}}_R.
\label{eq:two-site-state}
\end{equation}

This form is the direct analogue of the usual two-site DMRG superblock state and should be treated as the primary variational object. It is not necessary to define $\Theta$ through pre-existing tensors $A^{[i]}$ and $B^{[i+1]}$.

\subsection{Two-site effective Hamiltonian}

Let $L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')$ be the left environment built from sites $1,\dots,i-1$, and let $R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1},\alpha_{i+1}')$ be the right environment built from sites $i+2,\dots,L$.

Then the effective Hamiltonian acts on $\Theta$ as
\begin{align}
(\mathcal{H}_{\mathrm{eff}} \Theta)_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
=
\sum_{\substack{
        \alpha_{i-1}',\alpha_{i+1}'\\
        \sigma_i',\sigma_{i+1}'\\
        \beta_{i-1},\beta_i,\beta_{i+1}
}}
&L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}
\nonumber\\
&\times
R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1}',\alpha_{i+1})
\Theta_{\alpha_{i-1}',\sigma_i',\sigma_{i+1}',\alpha_{i+1}'}.
\label{eq:two-site-heff}
\end{align}

Equivalently, after vectorizing
\[
\Theta \mapsto v \in \mathbb{C}^{D_{i-1} d_i d_{i+1} D_{i+1}},
\]
one solves
\begin{equation}
\mathcal{H}_{\mathrm{eff}} v = E v.
\end{equation}

\subsection{Matrix-free application of the two-site effective Hamiltonian}

For practical coding, Eq.~\eqref{eq:two-site-heff} should never be assembled as a dense matrix. Instead, implement the action of $\mathcal{H}_{\mathrm{eff}}$ on $\Theta$ as a sequence of tensor contractions.

Let
\[
\Theta \in \mathbb{C}^{D_{i-1}\times d_i \times d_{i+1}\times D_{i+1}},
\]
with indices ordered as
\[
\Theta[\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}].
\]

A safe contraction sequence is:
\begin{align}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i',\sigma_{i+1}',\alpha_{i+1}'}
&=
\sum_{\alpha_{i-1}'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
\Theta_{\alpha_{i-1}',\sigma_i',\sigma_{i+1}',\alpha_{i+1}'},
\\
Y_{\beta_i,\alpha_{i-1},\sigma_i,\sigma_{i+1}',\alpha_{i+1}'}
&=
\sum_{\beta_{i-1},\sigma_i'}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i',\sigma_{i+1}',\alpha_{i+1}'},
\\
Z_{\beta_{i+1},\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}'}
&=
\sum_{\beta_i,\sigma_{i+1}'}
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}
Y_{\beta_i,\alpha_{i-1},\sigma_i,\sigma_{i+1}',\alpha_{i+1}'},
\\
(\mathcal{H}_{\mathrm{eff}}\Theta)_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
&=
\sum_{\beta_{i+1},\alpha_{i+1}'}
Z_{\beta_{i+1},\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}'}
R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1}',\alpha_{i+1}).
\end{align}

A corresponding NumPy implementation can be written using \texttt{einsum}, but because multi-step index strings are easy to get wrong, the symbolic contractions above should be treated as the authoritative definition. In practice, the matrix-free routine should be unit-tested against an explicitly constructed dense effective Hamiltonian for very small dimensions.

\paragraph{Practical recommendation.}
For a first implementation, write a reference function that builds the dense effective Hamiltonian for small $D$ and $d$, and use it to unit test the matrix-free contraction.

\subsection{Solving the local problem}

Given a matrix-free routine for $\Theta \mapsto \mathcal{H}_{\mathrm{eff}}\Theta$, solve for the lowest eigenpair using Lanczos or Davidson:
\begin{equation}
\mathcal{H}_{\mathrm{eff}} \Theta = E \Theta.
\end{equation}

The local vector space dimension is
\begin{equation}
N_{\mathrm{loc}} = D_{i-1} d_i d_{i+1} D_{i+1}.
\end{equation}

For spin-$1/2$ chains with uniform physical dimension $d_i=d_{i+1}=2$, this is
\[
N_{\mathrm{loc}} = 4 D_{i-1} D_{i+1}.
\]

The eigensolver should accept an initial guess. A good choice is the current two-site center tensor before optimization. This substantially accelerates convergence once sweeps are underway.

\subsection{SVD split and truncation}

After optimization, reshape the two-site tensor as a matrix:
\begin{equation}
\Theta_{(\alpha_{i-1}\sigma_i),(\sigma_{i+1}\alpha_{i+1})}
\in \mathbb{C}^{(D_{i-1}d_i)\times(d_{i+1}D_{i+1})}.
\label{eq:theta-matrix}
\end{equation}

Compute the reduced SVD:
\begin{equation}
\Theta = U S V^\dagger.
\label{eq:theta-svd}
\end{equation}

Let
\[
D_{\mathrm{full}} = \mathrm{rank}(\Theta),\qquad
D_{\mathrm{new}} = \min(D_{\max}, D_{\mathrm{full}})
\]
or more generally choose $D_{\mathrm{new}}$ from a singular-value threshold. Keep only the largest $D_{\mathrm{new}}$ singular values.

The discarded weight is
\begin{equation}
\eta^{[i]} = \sum_{j>D_{\mathrm{new}}} s_j^2.
\label{eq:discarded-weight}
\end{equation}

For a left-to-right sweep, define
\begin{equation}
A^{[i]} \leftarrow \mathrm{reshape}\!\left(U,\,(D_{i-1},d_i,D_{\mathrm{new}})\right),
\label{eq:two-site-left-update}
\end{equation}
\begin{equation}
M^{[i+1]} \leftarrow \mathrm{reshape}\!\left(SV^\dagger,\,(D_{\mathrm{new}},d_{i+1},D_{i+1})\right).
\label{eq:two-site-center-right}
\end{equation}

Then $A^{[i]}$ is left-canonical, and the orthogonality center has moved to site $i+1$.

For a right-to-left sweep, instead use
\begin{equation}
M^{[i]} \leftarrow \mathrm{reshape}\!\left(US,\,(D_{i-1},d_i,D_{\mathrm{new}})\right),
\label{eq:two-site-center-left}
\end{equation}
\begin{equation}
B^{[i+1]} \leftarrow \mathrm{reshape}\!\left(V^\dagger,\,(D_{\mathrm{new}},d_{i+1},D_{i+1})\right),
\label{eq:two-site-right-update}
\end{equation}
so that $B^{[i+1]}$ is right-canonical and the orthogonality center moves to site $i$.

\paragraph{Important reshape note.}
The reshapes in Eqs.~\eqref{eq:two-site-left-update}--\eqref{eq:two-site-right-update} are only correct if Eq.~\eqref{eq:theta-matrix} uses exactly the same flattening convention as the code.

\subsection{Environment updates}

After splitting the optimized two-site tensor, update the environments.

\paragraph{Left-to-right sweep.}
After obtaining the new left-canonical tensor $A^{[i]}$, update the left environment from site $i$ to site $i+1$:
\begin{equation}
L^{[i+1]}_{\beta_i}(\alpha_i,\alpha_i')
=
\sum_{\substack{\alpha_{i-1},\alpha_{i-1}'\\ \sigma_i,\sigma_i'\\ \beta_{i-1}}}
A^{[i]*}_{\alpha_{i-1}',\sigma_i',\alpha_i'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}.
\end{equation}

\paragraph{Right-to-left sweep.}
After obtaining the new right-canonical tensor $B^{[i+1]}$, update the right environment from site $i+1$ to site $i$:
\begin{equation}
R^{[i]}_{\beta_i}(\alpha_i,\alpha_i')
=
\sum_{\substack{\alpha_{i+1},\alpha_{i+1}'\\ \sigma_{i+1},\sigma_{i+1}'\\ \beta_{i+1}}}
B^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}
R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1},\alpha_{i+1}')
B^{[i+1]*}_{\alpha_i',\sigma_{i+1}',\alpha_{i+1}'}.
\end{equation}

\subsection{Full two-site sweep algorithm}

\begin{algorithm}[H]
\caption{TwoSiteDMRG}
\begin{algorithmic}[1]
\Require MPO $\{W^{[i]}\}$, initial MPS $\{M^{[i]}\}$, max bond dimension $D_{\max}$
\Ensure approximate ground-state MPS and energy
\State bring initial MPS into mixed-canonical form
\State build all right environments $R^{[i]}$
\State initialize left boundary environment $L^{[1]}$
\Repeat
    \Comment{Left-to-right sweep}
    \For{$i=1$ to $L-1$}
        \State form the current two-site center tensor $\Theta^{[i,i+1]}$
        \State solve the lowest-eigenvalue problem $\mathcal{H}_{\mathrm{eff}}\Theta = E\Theta$
        \State reshape $\Theta$ as in Eq.~\eqref{eq:theta-matrix}
        \State compute reduced SVD $\Theta = U S V^\dagger$
        \State truncate to $D_{\mathrm{new}}\le D_{\max}$
        \State set $A^{[i]} \gets \mathrm{reshape}(U,(D_{i-1},d_i,D_{\mathrm{new}}))$
        \State set $M^{[i+1]} \gets \mathrm{reshape}(S V^\dagger,(D_{\mathrm{new}},d_{i+1},D_{i+1}))$
        \State record discarded weight $\eta^{[i]}$
        \State update left environment $L^{[i+1]}$
    \EndFor
    \Comment{Right-to-left sweep}
    \For{$i=L-1$ down to $1$}
        \State form the current two-site center tensor $\Theta^{[i,i+1]}$
        \State solve the lowest-eigenvalue problem
        \State reshape $\Theta$ as in Eq.~\eqref{eq:theta-matrix}
        \State compute reduced SVD $\Theta = U S V^\dagger$
        \State truncate to $D_{\mathrm{new}}\le D_{\max}$
        \State set $M^{[i]} \gets \mathrm{reshape}(US,(D_{i-1},d_i,D_{\mathrm{new}}))$
        \State set $B^{[i+1]} \gets \mathrm{reshape}(V^\dagger,(D_{\mathrm{new}},d_{i+1},D_{i+1}))$
        \State record discarded weight $\eta^{[i]}$
        \State update right environment $R^{[i]}$
    \EndFor
\Until{energy, variance, and discarded-weight criteria are converged}
\end{algorithmic}
\end{algorithm}

\subsection{Practical comments for coding}

\begin{itemize}
    \item \textbf{This should be the first DMRG implementation.} It is much easier to debug than one-site DMRG.
    \item \textbf{Keep the orthogonality center explicit.} During a left-to-right sweep, after the split the center lives on site $i+1$ as $M^{[i+1]}$; during a right-to-left sweep, it lives on site $i$ as $M^{[i]}$.
    \item \textbf{Do not compare tensors directly across sweeps without gauge fixing.}
    \item \textbf{Always unit test the local effective-Hamiltonian matvec} against a dense reference for small dimensions.
    \item \textbf{Monitor the maximum discarded weight}
    \[
    \eta_{\max} = \max_i \eta^{[i]}
    \]
    during each sweep.
    \item \textbf{Benchmark first against exact diagonalization} for small Heisenberg chains before attempting larger systems.
\end{itemize}

\subsection{Recommended initial benchmarks}

For a first two-site DMRG code, test the following:
\begin{enumerate}
    \item \textbf{Small Heisenberg chains} ($L=4,6,8,10$), comparing energies to exact diagonalization.
    \item \textbf{AKLT chain} with $D_{\max}=2$, verifying exact recovery and zero discarded weight.
    \item \textbf{Environment consistency}: compare environment-based energy evaluation with brute-force contraction on small systems.
    \item \textbf{Canonicality checks}: after each split, verify left- or right-canonical conditions numerically.
\end{enumerate}

\section{Heisenberg MPO}

For the spin-$1/2$ nearest-neighbor XXZ chain with field,
\begin{equation}
H =
\sum_{i=1}^{L-1}
\left[
\frac{J}{2}\left(S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+\right)
+
J_z S_i^z S_{i+1}^z
\right]
-
h\sum_{i=1}^L S_i^z,
\end{equation}
a standard OBC MPO with bond dimension $\chi=5$ is
\begin{equation}
\hat W_{\mathrm{bulk}} =
\begin{pmatrix}
\Id & 0 & 0 & 0 & 0 \\
S^+ & 0 & 0 & 0 & 0 \\
S^- & 0 & 0 & 0 & 0 \\
S^z & 0 & 0 & 0 & 0 \\
-hS^z & \frac{J}{2}S^- & \frac{J}{2}S^+ & J_z S^z & \Id
\end{pmatrix}.
\end{equation}

Here $\hat W_{\mathrm{bulk}}$ is an operator-valued matrix. The tensor components $W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}$ are obtained by expanding each operator entry in the local basis:
\[
(\hat O)_{\sigma,\sigma'} = \bra{\sigma}\hat O\ket{\sigma'}.
\]

For OBC, choose:
\begin{equation}
\hat W^{[1]} =
\begin{pmatrix}
-hS^z & \frac{J}{2}S^- & \frac{J}{2}S^+ & J_z S^z & \Id
\end{pmatrix},
\end{equation}
\begin{equation}
\hat W^{[L]} =
\begin{pmatrix}
\Id \\ S^+ \\ S^- \\ S^z \\ -hS^z
\end{pmatrix},
\end{equation}
and use $\hat W_{\mathrm{bulk}}$ on all intermediate sites, with the chosen lower-triangular convention.

In code, this means:
\begin{itemize}
    \item left boundary tensor shape $(1,5,d,d)$ stores the \emph{last row} of the bulk MPO,
    \item right boundary tensor shape $(5,1,d,d)$ stores the \emph{first column} of the bulk MPO.
\end{itemize}

\section{AKLT benchmark state}

A standard exact MPS representation of the AKLT state is given, up to gauge, by
\begin{align}
A^{+} &= \begin{pmatrix} 0 & \sqrt{2/3} \\ 0 & 0 \end{pmatrix},\\
A^{0} &= \begin{pmatrix} -1/\sqrt{3} & 0 \\ 0 & 1/\sqrt{3} \end{pmatrix},\\
A^{-} &= \begin{pmatrix} 0 & 0 \\ -\sqrt{2/3} & 0 \end{pmatrix}.
\end{align}
For this gauge, the tensors satisfy
\begin{equation}
\sum_{\sigma\in\{+,0,-\}} A^\sigma A^{\sigma\dagger} = \Id_2,
\end{equation}
so they are left-canonical in the convention of Eq.~\eqref{eq:left-canonical}.

This state is an excellent benchmark:
\begin{itemize}
    \item exact bond dimension $D=2$,
    \item zero truncation error at $D_{\max}\ge 2$,
    \item for open boundary conditions, the AKLT Hamiltonian has an exactly degenerate edge-state manifold, but every ground state in that manifold has the same exact energy
    \[
    E_0 = -\frac23 (L-1).
    \]
\end{itemize}

\section{Convergence diagnostics}

Reliable convergence diagnostics for DMRG should include:
\begin{itemize}
    \item energy change
    \[
    |\Delta E| < \eps_E,
    \]
    \item energy variance
    \[
    \sigma_H^2 = \bra{\psi}H^2\ket{\psi} - \bra{\psi}H\ket{\psi}^2,
    \]
    \item maximum discarded weight during a sweep,
    \item stability of entanglement spectrum or local observables.
\end{itemize}

\paragraph{Important caution.}
Direct tensor-by-tensor comparison is not gauge-invariant and should \emph{not} be used as a primary convergence criterion unless the gauge is fixed identically before comparison.

\section{Implementation checklist}

Before production runs, validate in this order:
\begin{enumerate}
    \item \textbf{Canonicalization test}: random tensor $\to$ QR/SVD sweep $\to$ check Eqs.~\eqref{eq:left-canonical} and \eqref{eq:right-canonical}.
    \item \textbf{MPO test}: contract MPO to a dense Hamiltonian for small $L$ and compare with exact construction.
    \item \textbf{Environment test}: verify left/right environments against brute-force contraction for small $L$.
    \item \textbf{One-site effective Hamiltonian test}: compare matrix-free action against explicitly constructed $\mathcal{H}_{\mathrm{eff}}$ for small dimensions.
    \item \textbf{Two-site DMRG benchmark}: recover exact diagonalization energies for small Heisenberg chains.
    \item \textbf{AKLT benchmark}: recover exact energy and zero truncation error at $D_{\max}=2$.
\end{enumerate}

\section{Practical coding recommendation}

For a first reliable implementation:
\begin{itemize}
    \item implement \textbf{two-site DMRG first},
    \item use \textbf{reduced QR} for gauge moves and \textbf{SVD} for truncation,
    \item avoid pivoted QR in canonicalization,
    \item store all tensors as \texttt{complex128},
    \item benchmark against exact diagonalization for $L\le 10$ or $12$,
    \item only after this add one-site DMRG and subspace expansion.
\end{itemize}

\end{document}
```

---
## 🧑‍💻 User

Please save our conversations in this session in a markdown.

---
## 🤖 Assistant

I can’t directly save files from here, but I can provide the session as Markdown for you to copy into `session_notes.md`.

```markdown
# Session Notes

## User
This is a latex note generated from the attached review article. As a computational and theoretical physicist, could you review the latex note equation by equation, then check whether it is ready for practical coding?
After that, please give me a fully correct LaTex file, changing the authors to Kimi 2.5, GPT 5.4, and state that it is based on this review article.

## Assistant
Reviewed the note equation by equation, identified multiple mathematical and implementation issues, concluded it was not ready for practical coding, and provided a corrected LaTeX file with authors changed to **Kimi 2.5** and **GPT 5.4**, stating it was based on Schollwöck’s review article.

---

## User
Please go through this revised version and make sure it is mathematically consistent enough to be used as a coding blueprint. And then provide more details for the two-site DMRG.

## Assistant
Reviewed the revised version, found it much improved but still not fully coding-safe. Identified remaining issues:
- one-site einsum typo/inconsistency,
- right-canonical sweep too vague,
- two-site DMRG still too compressed,
- need clearer distinction of center tensors/matrices.

Then provided a much more detailed two-site DMRG section, including:
- mixed-canonical two-site form,
- two-site effective Hamiltonian,
- local variational problem,
- SVD split and truncation,
- sweep updates,
- environment updates,
- practical implementation comments.

---

## User
your output breaks off, pls continue

## Assistant
Continued the two-site DMRG section, adding:
- matrix-free application discussion,
- local eigensolver comments,
- truncation/discarded weight formulas,
- full two-site sweep pseudocode,
- implementation recommendations and initial benchmarks.

---

## User
Where to put Fix C?

## Assistant
Recommended placing **Fix C** at the start of **Section 6, `MPO environments`**, immediately before the environment definitions, and also suggested that an even better location would be **Section 1, `Scope and conventions`**, right after the MPO tensor definition.

---

## User
I prefer to Section 1.

## Assistant
Confirmed Section 1 is the better place, and gave the exact replacement text to insert under the MPO tensor bullet in `Scope and conventions`.

---

## User
This is the revised LaTeX file. Please go through it.

## Assistant
Reviewed the revised file and concluded it was very close to usable, but identified several remaining issues:
- two-site effective Hamiltonian physical index ordering inconsistent with MPO convention,
- unsafe two-site sample `einsum` strings,
- one-site final `einsum` line needed clarification,
- environment semantics should be stated explicitly,
- AKLT benchmark wording should be tightened,
- one-site truncation language should be refined.

Provided exact patch suggestions for each issue.

---

## User
your output breaks off, pls continue

## Assistant
Continued with the remaining fixes and exact replacements:
- corrected two-site effective-Hamiltonian equation,
- recommended removing unsafe two-site `einsum` code,
- suggested cleaner one-site `einsum` block,
- clarified environment semantics,
- improved AKLT benchmark wording,
- concluded that after these fixes the note would be mathematically consistent enough for use as a coding blueprint.

---

## User
Plese give me a fully patched final LaTeX file.

## Assistant
Provided a fully patched final LaTeX file with:
- corrected conventions,
- fixed one-site and two-site equations,
- improved canonicalization sections,
- corrected environment semantics,
- safe one-site `einsum` block,
- expanded two-site DMRG section,
- clarified Heisenberg MPO construction,
- tightened AKLT benchmark wording,
- authors listed as **Kimi 2.5** and **GPT 5.4**,
- explicit statement that the note is based on Schollwöck’s review article.

---

## User
Please save our conversations in this session in a markdown.

## Assistant
Provided this Markdown summary for saving as a session log.
```

If you want, I can also format it as a **full transcript markdown** with every user/assistant turn quoted verbatim.
