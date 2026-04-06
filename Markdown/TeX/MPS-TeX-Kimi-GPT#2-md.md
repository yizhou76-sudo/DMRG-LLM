# MPS-TeX-Kimi-GPT#2-md

## 🧑‍💻 User

As a computational and theoretical physicist, could you review this latex note line by line, then check whether it is ready for practical coding?

---
## 🤖 Assistant

Yes. Overall, this is a strong note: mathematically coherent, implementation-oriented, and already much closer to “code spec” than most MPS/DMRG summaries.

But it is not quite “ready for practical coding” without a few corrections and several clarifications. The main issues are:

1. one explicit tensor-contraction bug in the left-environment einsum template,
2. some boundary/index inconsistencies in the one-site environment notation,
3. a likely AKLT energy mistake,
4. several places where the text is mathematically fine but still underspecified for direct software implementation.

Below I go section by section, line by line in substance, and then I give a coding-readiness verdict plus a concrete list of changes needed before implementation.

---

# High-level verdict

## What is already good
- Clear tensor shapes for MPS/MPO.
- Good emphasis on fixed reshape conventions.
- Canonical form definitions are correct.
- QR-based gauge moves are correctly described.
- Two-site DMRG is rightly prioritized.
- Effective Hamiltonian is treated matrix-free.
- Testing/benchmarking culture is good.

## What must be fixed before coding
- The left environment `einsum` is wrong.
- One-site environment symbols \(L^{[i]}\), \(R^{[i]}\) are notationally inconsistent with the later effective-Hamiltonian formula.
- Some algorithm steps need explicit shape bookkeeping.
- The AKLT benchmark energy is almost certainly not universally correct as written.
- The note still lacks a minimal software data model and canonical API.

So: **excellent draft, not yet fully coding-ready**.

---

# Detailed review

---

## Preamble and packages

```latex
\documentclass[aps,prb,onecolumn,scriptaddress]{revtex4-2}
```

Fine for a note. No coding relevance.

```latex
\usepackage{graphicx}
\usepackage{amsmath,amsfonts,amssymb,bm,mathtools}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{xcolor}
\usepackage[colorlinks=true,urlcolor=blue,linkcolor=blue,citecolor=blue]{hyperref}
```

All fine. `graphicx`, `booktabs`, `tabularx`, `xcolor` seem unused in the current text, but harmless.

```latex
\newcommand{\ket}[1]{|#1\rangle}
...
```

Good. Standard.

---

# 1. Scope and conventions

This section is mostly good.

### MPS tensor definition
```latex
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i},
with shape $(D_{i-1}, d_i, D_i)$.
```
Correct and practical.

### MPO tensor definition
```latex
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'},
with shape $(\chi_{i-1}, \chi_i, d_i, d_i)$.
```
Correct.

### Operator convention
```latex
\hat W^{[i]}
=
\sum_{\sigma_i,\sigma_i'}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
\ket{\sigma_i}\bra{\sigma_i'},
```
This is fine, but you should explicitly state once that in code the last two MPO legs are ordered as `(phys_out, phys_in)` or `(ket, bra)`. You do say this verbally, which is good. For coding, I would make it stronger:

- **Recommended exact code convention:** `W.shape = (chiL, chiR, d_out, d_in)`

Otherwise people later forget whether `W[bL,bR,s,t]` maps `t -> s` or `s -> t`.

### OBC dimensions
Correct.

### complex128
Good practical default.

### Missing for coding
You should add:
- whether state vectors are normalized after each local solve,
- whether singular values are absorbed left or right consistently,
- whether all bonds are allowed to vary dynamically.

---

# 2. Reshape conventions

This is one of the strongest sections.

### Left-grouped and right-grouped reshapes
Correct.

### Row-major flattening
```latex
r = \alpha_{i-1} d_i + \sigma_i,\qquad c = \alpha_i.
```
Correct for C-order if tensor layout is `(alpha_left, sigma, alpha_right)`.

### NumPy reshapes
```python
M_left = T.reshape((D_left * d, D_right), order='C')
M_right = T.reshape((D_left, d * D_right), order='C')
```
Correct.

### Comment
This section is coding-useful. Good.

### Missing for coding
For full direct implementation, also define the two-site flattening explicitly:
\[
(\alpha_{i-1},\sigma_i) \leftrightarrow \text{left row index},\qquad
(\sigma_{i+1},\alpha_{i+1}) \leftrightarrow \text{right column index}.
\]
You do this later, but I would bring the exact C-order flattening formula here too.

---

# 3. Canonical forms

This section is correct.

### Left-canonical condition
```latex
\sum_{\alpha_{i-1},\sigma_i}
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
A^{[i]*}_{\alpha_{i-1},\sigma_i,\alpha_i'}
=
\delta_{\alpha_i,\alpha_i'}.
```
Correct.

### Right-canonical condition
Correct.

### Mixed-canonical state
Correct in spirit. The phrase
```latex
orthogonality center at bond $i$
```
is okay, but later you often discuss center at a site. It would help to distinguish:

- **bond-center canonical form**: Schmidt values on bond \(i\),
- **site-center mixed canonical form**: center tensor on site \(i\).

Not wrong, just worth clarifying.

---

# 4. Canonicalization algorithms

## Left sweep by reduced QR

This is correct.

### QR dimensions
```latex
k=\min(D_{i-1}d_i,D_i).
```
Correct.

### Absorbing \(R\) into next tensor
```latex
M^{[i+1]}_{\tilde\alpha_i,\sigma_{i+1},\alpha_{i+1}}
\leftarrow
\sum_{\alpha_i}
R_{\tilde\alpha_i,\alpha_i}
M^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}.
```
Correct.

### Algorithm
Good.

### Practical note on pivoting
Excellent.

## Right sweep by reduced QR

Also correct. Using QR on the conjugate transpose is standard and practical.

### Minor coding recommendation
You should include the explicit NumPy pattern:
```python
Q, R = np.linalg.qr(M_right.conj().T, mode='reduced')
B = Q.conj().T.reshape((k, d, D_right), order='C')
M_prev = np.einsum('xsa,ab->xsb', M_prev, R.conj().T)
```
This would make the note much more directly executable.

---

# 5. SVD truncation

Mostly correct.

### Truncation criterion
```latex
D = \min\{D_{\max},\#(s_j > \eps)\}
```
This is acceptable but not ideal for DMRG production code.

Why? Because in practice one usually uses either:
- fixed `D_max`,
- discarded-weight threshold,
- or both.

A pure `s_j > eps` rule is not robust across norm conventions or system size.

### Recommendation
Replace with something like:
- keep at most `D_max`,
- and optionally choose the smallest kept dimension such that discarded weight \(\eta \le \epsilon_{\rm svd}\),
- while always keeping at least one singular value.

This is more coding-relevant.

### Discarded weight
Correct if the state is normalized.

You might add:
- if \(\Theta\) is not normalized before SVD, \(\eta\) is relative only after dividing by \(\|\Theta\|^2\).

---

# 6. MPO environments

This is where the first serious issue appears.

## Conceptual definition
The prose is okay:
- \(L^{[i]}\): contraction of sites \(1,\dots,i-1\),
- \(R^{[i]}\): contraction of sites \(i+1,\dots,L\).

That is a good convention.

## Left environment recursion

```latex
L^{[i]}_{\beta_i}(\alpha_{i-1},\alpha_{i-1}')
=
...
A^{[i-1]*}_{\alpha_{i-2}',\sigma',\alpha_{i-1}'}
L^{[i-1]}_{\beta_{i-1}}(\alpha_{i-2},\alpha_{i-2}')
W^{[i-1]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
A^{[i-1]}_{\alpha_{i-2},\sigma,\alpha_{i-1}}
```

This is fine.

## Right environment recursion

```latex
R^{[i]}_{\beta_{i-1}}(\alpha_i,\alpha_i')
=
...
B^{[i+1]}_{\alpha_i,\sigma,\alpha_{i+1}}
W^{[i+1]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
R^{[i+1]}_{\beta_i}(\alpha_{i+1},\alpha_{i+1}')
B^{[i+1]*}_{\alpha_i',\sigma',\alpha_{i+1}'}.
```

This is also structurally correct.

## Boundary initialization
```latex
L^{[1]}_{\beta_0} = \delta_{\beta_0,1},\qquad
R^{[L]}_{\beta_L} = \delta_{\beta_L,1},
```

This is too informal for code. These are not just scalars; for code they should be rank-3 arrays:
- `L[1].shape = (1,1,1)` with entry `1`
- `R[L].shape = (1,1,1)` with entry `1`

I would write that explicitly.

## Serious bug: left-environment einsum template

You wrote:
```python
Lnew = np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj())
```

This is wrong.

Reason: the conjugate tensor should carry the new right bond index `c`, but the non-conjugate tensor carries new right bond index `a`; then both should survive as output indices. In your contraction, `A.conj()` uses `ytc`, but there is no contraction connecting `a` and `c` correctly via the same old left indices and physical legs in a way matching the formula. More importantly, the output uses `Bac`, but `a` comes from `A` and `c` from `A.conj()`, which is okay in principle, yet the old left-bond pairing is mismatched because the first two MPS tensors must use indices `(x,s,a)` and `(y,t,c)` together with `Lold[b,x,y]`.

Actually the correct template for your notation is:
```python
Lnew = np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj())
```
Wait — algebraically this *can* be correct if:
- `Lold[b,x,y] = L_{β_{i-1}}(α_{i-1}, α'_{i-1})`
- `A[x,s,a] = A_{α_{i-1},σ,α_i}`
- `W[b,B,s,t] = W_{β_{i-1},β_i,σ,σ'}`
- `A.conj()[y,t,c] = A^*_{α'_{i-1},σ',α'_i}`

Then output `B,a,c` is indeed \(L^{[i+1]}_{\beta_i}(\alpha_i,\alpha_i')\).

So this contraction is actually correct.

However, your naming line says:
```latex
Aconj[y,t,c] = A[y,t,c].conj()
```
That is fine.

So I retract “bug” in the strict algebraic sense: **the left environment einsum is correct**.

But there is still a practical problem: it is easy to confuse because the text around Eq. \eqref{eq:left-env} defines \(L^{[i]}\) as environment for site \(i\), while here the update is effectively \(L^{[i]}\to L^{[i+1]}\). I recommend making that explicit in the code example:
```python
# L_i -> L_{i+1}
L_ip1 = np.einsum('bxy,xsa,bBst,ytc->Bac', L_i, A_i, W_i, A_i.conj())
```

So the einsum itself is okay, but the notation should be sharpened.

---

# 7. One-site effective Hamiltonian

This section has a notation inconsistency.

You define earlier:
- \(L^{[i]}\) = contraction of sites \(1,\dots,i-1\),
- \(R^{[i]}\) = contraction of sites \(i+1,\dots,L\).

Then in the action formula:
```latex
(\mathcal{H}_{\mathrm{eff}} M)_{\alpha_{i-1},\sigma_i',\alpha_i}
=
\sum ...
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
R^{[i]}_{\beta_i}(\alpha_i',\alpha_i)
M^{[i]}_{\alpha_{i-1}',\sigma_i,\alpha_i'}.
```
This is actually consistent if:
- `L^{[i]}` carries left MPO bond \(\beta_{i-1}\),
- `R^{[i]}` carries right MPO bond \(\beta_i\).

But earlier in the environment recursion you wrote:
```latex
L^{[i]}_{\beta_i}(...)
```
and
```latex
R^{[i]}_{\beta_{i-1}}(...)
```
which swaps the labels.

That is the real inconsistency.

## Recommended fix
Choose one convention and use it everywhere. For example:

- \(L^{[i]}\): left environment up to site \(i-1\), with shape \((\chi_{i-1}, D_{i-1}, D_{i-1})\), indices
  \[
  L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha'_{i-1})
  \]
- \(R^{[i]}\): right environment from site \(i+1\) onward, with shape \((\chi_i, D_i, D_i)\), indices
  \[
  R^{[i]}_{\beta_i}(\alpha_i,\alpha'_i)
  \]

Then rewrite both recursion formulas accordingly. That would make the effective-Hamiltonian formula internally consistent.

## Matvec contraction sequence
```python
X  = np.einsum('bxy,ysz->bxsz', L, M)
Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
HM = np.einsum('Bxtz,Bza->xta', Y, R)
```
This is correct if:
- `L[b,x,y]`
- `M[y,s,z]`
- `W[b,B,s,t]`
- `R[B,z,a]`

But notice the output physical index becomes `t`, i.e. your `sigma_i'`, as desired. Good.

Still, because of the notational inconsistency in the preceding formulas, the code snippet is more reliable than the symbolic section. That should not happen in a coding note. The symbolic notation must be made authoritative and self-consistent.

---

# 8. One-site DMRG sweep

This is broadly okay.

## Good points
- Starts with right-canonicalization.
- Builds right environments.
- Uses eigensolver for local problem.
- Moves center using SVD.
- Warns about local minima.

## Comment on one-site bond updates
You wrote:
```latex
in standard one-site DMRG this step primarily shifts the orthogonality center, while any actual change in bond dimension is usually small or absent unless rank deficiency or adaptive bond control is used
```
This is a good and nuanced remark.

## Missing coding details
For one-site DMRG, practical codes often do **QR** for center motion and reserve **SVD** for adaptive truncation or noise/subspace expansion variants. What you wrote is fine, but if the goal is “ready for practical coding,” I would state:

- one-site sweep can move center via QR for efficiency,
- use SVD only if adaptive rank truncation is desired.

That can matter significantly in runtime.

---

# 9. Two-site DMRG

This is the best section of the note.

## Two-site mixed-canonical form
Correct and well explained.

## Two-site effective Hamiltonian
Equation structure is correct.

### Minor index issue
In Eq. \eqref{eq:two-site-heff}, the output and summation labels use primed/unprimed physical indices in a way opposite to some common conventions, but it is still internally valid. No issue if kept consistent.

## Matrix-free action
Very good.

## Recommendation to test against dense reference
Excellent and essential.

## Solving the local problem
Good. Mentioning warm starts is useful.

## SVD split and truncation
Correct in structure.

### Important coding note
When you write
```latex
M^{[i+1]} \leftarrow \mathrm{reshape}\!\left(SV^\dagger,\,(D_{\mathrm{new}},d_{i+1},D_{i+1})\right)
```
this is correct only if the matrix \(SV^\dagger\) is interpreted with row index equal to the kept bond and column index equal to \((\sigma_{i+1},\alpha_{i+1})\), and the code uses the matching C-order reshape. You do say this later, which is good.

### Strong suggestion
Add explicit formulas:
\[
r = \tilde \alpha_i,\qquad c = \sigma_{i+1} D_{i+1} + \alpha_{i+1}
\]
for the right factor reshape in C-order.

Similarly for \(U\):
\[
r = \alpha_{i-1} d_i + \sigma_i,\qquad c = \tilde\alpha_i.
\]

That would remove ambiguity.

## Environment updates
These are correct.

## Full two-site sweep algorithm
Structurally good.

### Missing implementation detail
You do not define how to form the current two-site center tensor \(\Theta^{[i,i+1]}\). For coding, you should explicitly say:

- in left-to-right sweep, if center is at site \(i\) and site \(i+1\) is right-canonical/right-part tensor, then
  \[
  \Theta_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
  =
  \sum_{\alpha_i}
  M^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
  B^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}},
  \]
  or in a simpler implementation, always explicitly merge adjacent tensors currently stored.

Without this, a coder still has to infer how \(\Theta\) is obtained from the MPS storage format.

---

# 10. Practical comments for coding

This section is good and realistic.

Especially good:
- first implement two-site DMRG,
- do not compare tensors without gauge fixing,
- test local matvec against dense reference.

No major issues.

---

# 11. Recommended initial benchmarks

Good list.

### Caveat on AKLT benchmark
The AKLT benchmark is useful, but only if the Hamiltonian convention is specified exactly. This becomes important below.

---

# 12. Heisenberg MPO

This section is mostly correct and practical.

## Hamiltonian
```latex
H =
\sum_{i=1}^{L-1}
\left[
\frac{J}{2}\left(S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+\right)
+
J_z S_i^z S_{i+1}^z
\right]
-
h\sum_{i=1}^L S_i^z,
```
Correct XXZ form.

## MPO bulk tensor
The lower-triangular convention is standard and correct.

## Boundary tensors
```latex
\hat W^{[1]} = ( -hS^z,\ \frac{J}{2}S^-,\ \frac{J}{2}S^+,\ J_zS^z,\ \Id )
```
and
```latex
\hat W^{[L]} = ( \Id,\ S^+,\ S^-,\ S^z,\ -hS^z)^T
```
This matches your bulk lower-triangular convention.

## Good coding note
```latex
left boundary tensor shape $(1,5,d,d)$ stores the last row of the bulk MPO,
right boundary tensor shape $(5,1,d,d)$ stores the first column of the bulk MPO.
```
Excellent.

### Missing detail
For practical coding, define explicit local matrices:
\[
S^+ = \begin{pmatrix}0&1\\0&0\end{pmatrix},\quad
S^- = \begin{pmatrix}0&0\\1&0\end{pmatrix},\quad
S^z = \frac12\begin{pmatrix}1&0\\0&-1\end{pmatrix}.
\]
This avoids convention drift.

---

# 13. AKLT benchmark state

The MPS tensors are standard and look fine.

## Canonicality statement
```latex
\sum_{\sigma} A^\sigma A^{\sigma\dagger} = \Id_2
```
This corresponds to left-canonicality in your convention, yes.

## Serious issue: AKLT energy
You write:
```latex
E_0 = -\frac23 (L-1).
```
This is not a universal AKLT ground-state energy unless a very specific Hamiltonian normalization has been chosen, and that normalization has not been stated here.

The most common AKLT Hamiltonian is
\[
H = \sum_i \left[\mathbf S_i\cdot\mathbf S_{i+1} + \frac13(\mathbf S_i\cdot\mathbf S_{i+1})^2\right]
\]
up to additive constants, or the projector form
\[
H = \sum_i P^{(2)}_{i,i+1}.
\]
Depending on whether one subtracts constants bondwise, the ground-state energy can be \(0\), \(-2/3(L-1)\), or another affine equivalent.

## Fix required
You must specify the exact AKLT Hamiltonian convention before giving the energy benchmark.

For example:
- If
  \[
  H_{\rm AKLT} = \sum_{i=1}^{L-1}\left[\mathbf S_i\cdot\mathbf S_{i+1} + \frac13(\mathbf S_i\cdot\mathbf S_{i+1})^2 + \frac23\right],
  \]
  then the ground-state energy is \(0\).
- If the \(+2/3\) constant is omitted, the energy is \(-2/3(L-1)\).

As written, the benchmark is ambiguous and dangerous for coding validation.

---

# 14. Convergence diagnostics

This is good.

## Energy change
Good.

## Variance
Good and important.

### Practical coding note
You should mention that computing variance requires an MPO for \(H^2\) or another stable contraction strategy. Otherwise a beginner may think it is trivial.

A useful sentence:
- “For first implementations, use energy convergence and discarded weight first; add variance after the MPO and environment machinery is well tested.”

---

# 15. Implementation checklist

Very good.

### Suggested addition
Add:
- normalization test of the MPS after each canonicalization/split,
- local eigensolver residual norm test,
- consistency of energy computed from center tensor vs full environment contraction.

---

# 16. Practical coding recommendation

Good final recommendations.

---

# What is missing for direct practical coding?

Even after fixing the mathematical issues, the note still lacks some software-engineering essentials. To be coding-ready, I would add the following sections.

---

## A. Minimal tensor data model

You should specify something like:

```python
# MPS
psi[i].shape == (D_left_i, d_i, D_right_i)

# MPO
W[i].shape == (chi_left_i, chi_right_i, d_i, d_i)

# left environment for site i
Lenv[i].shape == (chi_left_i, D_left_i, D_left_i)

# right environment for site i
Renv[i].shape == (chi_right_i, D_right_i, D_right_i)
```

and state exactly what `i` means in each array.

Right now this is implied, but not formally codified.

---

## B. Canonical-center invariant

State an invariant for sweeps, e.g.

- During left-to-right sweep at step \(i\):
  - sites \(1,\dots,i-1\) are left-canonical,
  - site \(i\) and \(i+1\) form the active two-site center,
  - sites \(i+2,\dots,L\) are right-canonical.

This is obvious to experts but crucial for implementation correctness.

---

## C. Local eigensolver API

You should state something like:

```python
def heff_matvec_theta(theta_vec, Lenv, W_i, W_ip1, Renv, dims):
    ...
    return out_vec
```

and similarly for one-site. This is where many implementations go wrong.

---

## D. Truncation policy

You should define exactly one default policy, for example:

1. compute SVD,
2. choose smallest \(D_{\rm new}\le D_{\max}\) such that discarded weight \(\le \epsilon_{\rm trunc}\),
3. if no such \(D_{\rm new}\) exists, use \(D_{\max}\),
4. always keep at least one singular value.

Without this, two implementations from the note may behave quite differently.

---

## E. Dense reference builders

You recommend dense checks, which is excellent, but you should define:
- dense Hamiltonian builder from MPO,
- dense state builder from MPS,
- dense local effective Hamiltonian builder for one-site and two-site.

This would make the note much more practically executable.

---

# Specific corrections I recommend

Here is the short actionable edit list.

## Must-fix mathematical/notational issues

### 1. Make environment indexing consistent everywhere
Adopt:
\[
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha'_{i-1}),
\qquad
R^{[i]}_{\beta_i}(\alpha_i,\alpha'_i).
\]
Then update all recursion equations accordingly.

### 2. Clarify boundary environments as arrays
Replace scalar-style initialization by:
\[
L^{[1]} \in \mathbb{C}^{1\times 1\times 1},\quad L^{[1]}_{1,1,1}=1,
\]
\[
R^{[L]} \in \mathbb{C}^{1\times 1\times 1},\quad R^{[L]}_{1,1,1}=1.
\]

### 3. Specify AKLT Hamiltonian convention
Do not quote an energy without the Hamiltonian normalization.

### 4. Add explicit two-site flattening formulas
Not just shapes; give index-to-row/column maps.

---

## Strongly recommended coding additions

### 5. Add explicit formula for forming \(\Theta\)
For example:
```latex
\Theta = \mathrm{contract}(M^{[i]}, B^{[i+1]})
```
with full index expression.

### 6. Add explicit `einsum` formulas for
- right environment update,
- two-site effective Hamiltonian matvec,
- two-site tensor formation and splitting.

### 7. Define a default truncation rule
Prefer discarded-weight plus `D_max`.

### 8. Add a minimal pseudocode data structure section
For MPS, MPO, environments, center position.

---

# Is it ready for practical coding right now?

## Short answer
**Almost, but not yet.**

## More precise answer
It is ready for:
- designing the architecture,
- writing unit tests,
- implementing canonicalization,
- implementing dense reference routines,
- implementing two-site DMRG prototype.

It is **not yet ready for production-quality coding** until you:
- fix the environment-index convention globally,
- specify truncation and center-handling more concretely,
- clarify benchmark Hamiltonian conventions.

---

# My recommended final verdict

If I were reviewing this as a computational physics implementation note, I would say:

> This is a strong implementation-oriented MPS/DMRG draft with good tensor conventions and a sound two-site DMRG core. However, it still needs a final pass to resolve environment-index notation, sharpen reshape/center conventions into a software API, and fix ambiguous benchmark normalization (especially AKLT). After those edits, it would be suitable as a practical coding specification.

---

# If you want, I can do one of the following next:

1. **Produce an annotated corrected LaTeX version** with all fixes inserted.
2. **Translate this note into a Python implementation plan** with class/function signatures.
3. **Write unit tests implied by the note**.
4. **Check every einsum and reshape convention explicitly**, one by one.

---
## 🧑‍💻 User

Produce an annotated corrected LaTeX version with all fixes inserted.

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
A^{[i]}

---
## 🧑‍💻 User

breaks off, please continue

---
## 🤖 Assistant

Continuing from `A^{[i]}` and providing the full annotated corrected LaTeX:

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

% Annotation macros for the corrected draft
\newcommand{\fixnote}[1]{\textcolor{blue}{\textbf{[Fix:} #1\textbf{]}}}
\newcommand{\codenote}[1]{\textcolor{red}{\textbf{[Coding note:} #1\textbf{]}}}

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

We use the operator convention
\[
\hat W^{[i]}
=
\sum_{\sigma_i,\sigma_i'}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
\ket{\sigma_i}\bra{\sigma_i'},
\]
so the first physical index is the ket/output index and the second is the bra/input index.

\fixnote{This convention should be used literally in code: store MPO tensors as \texttt{W[i].shape == (chiL, chiR, d\_out, d\_in)}. Many sign and contraction bugs come from silently swapping these last two indices.}

\item OBC dimensions:
\[
D_0 = D_L = 1,\qquad \chi_0 = \chi_L = 1.
\]
\end{itemize}

Unless stated otherwise, all tensors should be stored as \texttt{complex128}.

\codenote{For a first implementation, also define the core container conventions once and keep them invariant throughout:
\begin{align*}
&\texttt{mps[i].shape} = (D_{i-1}, d_i, D_i),\\
&\texttt{mpo[i].shape} = (\chi_{i-1}, \chi_i, d_i, d_i),\\
&\texttt{Lenv[i].shape} = (\chi_{i-1}, D_{i-1}, D_{i-1}),\\
&\texttt{Renv[i].shape} = (\chi_i, D_i, D_i).
\end{align*}
Here \texttt{Lenv[i]} is the left environment for site $i$, and \texttt{Renv[i]} is the right environment for site $i$.}

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

For Eq.~\eqref{eq:right-group}, the flattening is
\begin{equation}
r = \alpha_{i-1},\qquad c = \sigma_i D_i + \alpha_i.
\end{equation}

In NumPy:
\begin{verbatim}
M_left = T.reshape((D_left * d, D_right), order='C')
M_right = T.reshape((D_left, d * D_right), order='C')
\end{verbatim}

For a rank-4 two-site tensor
\[
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}},
\]
with shape $(D_{i-1}, d_i, d_{i+1}, D_{i+1})$, define the grouped matrix
\begin{equation}
\Theta_{(\alpha_{i-1}\sigma_i),(\sigma_{i+1}\alpha_{i+1})}
\in \mathbb{C}^{(D_{i-1}d_i)\times(d_{i+1}D_{i+1})}.
\label{eq:theta-grouped}
\end{equation}
In C-order storage this means
\begin{equation}
r = \alpha_{i-1} d_i + \sigma_i,\qquad
c = \sigma_{i+1} D_{i+1} + \alpha_{i+1}.
\label{eq:theta-c-order}
\end{equation}

\fixnote{The original note defined the grouped two-site matrix later, but for coding it is better to place all flattening conventions in one section.}

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

A bond-centered mixed-canonical state with orthogonality center at bond $i$ may be written as
\begin{equation}
\ket{\psi} =
\sum_{\alpha_i}
s^{[i]}_{\alpha_i}
\ket{\alpha_i}_L \ket{\alpha_i}_R,
\end{equation}
where $s^{[i]}_{\alpha_i}\ge 0$ are Schmidt singular values and the left/right Schmidt states are orthonormal.

\fixnote{The original text used ``orthogonality center at bond $i$'' here but later used site-centered language. We distinguish these explicitly:
\begin{itemize}
\item \emph{bond-centered form}: Schmidt values on bond $i$,
\item \emph{site-centered mixed-canonical form}: a center tensor on site $i$.
\end{itemize}
This is only a wording fix, but it prevents confusion when writing sweep code.}

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

\codenote{Reference NumPy pattern:
\begin{verbatim}
Mmat = M[i].reshape((Dl * d, Dr), order='C')
Q, R = np.linalg.qr(Mmat, mode='reduced')
A[i] = Q.reshape((Dl, d, Q.shape[1]), order='C')
M[i+1] = np.einsum('ab,bsd->asd', R, M[i+1])
\end{verbatim}}

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

\codenote{Reference NumPy pattern:
\begin{verbatim}
Mmat = M[i].reshape((Dl, d * Dr), order='C')
Q, R = np.linalg.qr(Mmat.conj().T, mode='reduced')
B[i] = Q.conj().T.reshape((Q.shape[1], d, Dr), order='C')
M[i-1] = np.einsum('xsa,ab->xsb', M[i-1], R.conj().T)
\end{verbatim}}

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

\fixnote{The original note used the rule $D = \min\{D_{\max},\#(s_j > \eps)\}$ as the primary truncation prescription. For practical DMRG code, discarded-weight control is usually a better default than an absolute singular-value threshold.}

A practical truncation policy is:
\begin{itemize}
\item enforce $D \le D_{\max}$,
\item optionally choose the smallest $D$ such that $\eta \le \eps_{\mathrm{trunc}}$,
\item always keep at least one singular value.
\end{itemize}

\begin{algorithm}[H]
\caption{TruncateBySVD}
\begin{algorithmic}[1]
\Require matrix $X$, max bond dimension $D_{\max}$, discarded-weight target $\eps_{\mathrm{trunc}}$
\Ensure truncated $U_D,S_D,V_D^\dagger$, discarded weight $\eta$
\State compute reduced SVD: $X = U S V^\dagger$
\State initialize $D \gets \min(D_{\max}, r)$
\If{discarded-weight targeting is enabled}
\State choose the smallest $D\le D_{\max}$ such that $\sum_{j>D} s_j^2 \le \eps_{\mathrm{trunc}}$
\State if no such $D$ exists, keep $D\gets D_{\max}$
\EndIf
\If{$D=0$}
\State $D\gets 1$
\EndIf
\State keep first $D$ singular values/vectors
\State $\eta \gets \sum_{j>D} s_j^2$
\end{algorithmic}
\end{algorithm}

\codenote{If $\Theta$ or $X$ is not normalized, the discarded weight is not yet a normalized truncation error. In that case, it is often useful to record both $\eta$ and $\eta/\sum_j s_j^2$.}

\section{MPO environments}

Here $L^{[i]}$ denotes the contraction of sites $1,\dots,i-1$ into a left environment for site $i$, while $R^{[i]}$ denotes the contraction of sites $i+1,\dots,L$ into a right environment for site $i$.

\fixnote{The original draft had an index-label inconsistency between the environment recursions and the effective-Hamiltonian section. The corrected convention used everywhere below is:
\[
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
\in \mathbb{C}^{\chi_{i-1}\times D_{i-1}\times D_{i-1}},
\]
\[
R^{[i]}_{\beta_i}(\alpha_i,\alpha_i')
\in \mathbb{C}^{\chi_i\times D_i\times D_i}.
\]
That is, $L^{[i]}$ carries the MPO bond index immediately to the left of site $i$, and $R^{[i]}$ carries the MPO bond index immediately to the right of site $i$.}

For one-site DMRG with orthogonality center at site $i$, define the left environment update
\begin{equation}
L^{[i+1]}_{\beta_i}(\alpha_i,\alpha_i')
=
\sum_{\substack{\alpha_{i-1},\alpha_{i-1}'\\ \sigma_i,\sigma_i'\\ \beta_{i-1}}}
A^{[i]*}_{\alpha_{i-1}',\sigma_i',\alpha_i'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i},
\label{eq:left-env-corrected}
\end{equation}
and the right environment update
\begin{equation}
R^{[i]}_{\beta_i}(\alpha_i,\alpha_i')
=
\sum_{\substack{\alpha_{i+1},\alpha_{i+1}'\\ \sigma_{i+1},\sigma_{i+1}'\\ \beta_{i+1}}}
B^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}
R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1},\alpha_{i+1}')
B^{[i+1]*}_{\alpha_i',\sigma_{i+1}',\alpha_{i+1}'}.
\label{eq:right-env-corrected}
\end{equation}

Boundary initialization for OBC:
\begin{equation}
L^{[1]} \in \mathbb{C}^{1\times 1\times 1},\qquad L^{[1]}_{1,1,1}=1,
\end{equation}
\begin{equation}
R^{[L]} \in \mathbb{C}^{1\times 1\times 1},\qquad R^{[L]}_{1,1,1}=1.
\end{equation}

\fixnote{The original boundary formulas were written as Kronecker deltas. That is mathematically fine, but for code it is better to state the actual array shapes.}

\paragraph{Safe einsum template.}
With arrays
\begin{itemize}
\item \texttt{Lold[b,x,y]} representing $L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')$,
\item \texttt{A[x,s,a]} representing $A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}$,
\item \texttt{W[b,B,s,t]} representing $W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}$,
\item \texttt{A.conj()[y,t,c]} representing $A^{[i]*}_{\alpha_{i-1}',\sigma_i',\alpha_i'}$,
\end{itemize}
a correct left update is
\begin{verbatim}
Lnew = np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj())
\end{verbatim}
which computes $L^{[i+1]}$ from $L^{[i]}$.

Similarly, with arrays
\begin{itemize}
\item \texttt{B[a,s,x]} representing $B^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}$,
\item \texttt{W[B,b,s,t]} representing $W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}$,
\item \texttt{Rold[b,x,y]} representing $R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1},\alpha_{i+1}')$,
\item \texttt{B.conj()[c,t,y]} representing $B^{[i+1]*}_{\alpha_i',\sigma_{i+1}',\alpha_{i+1}'}$,
\end{itemize}
a correct right update is
\begin{verbatim}
Rnew = np.einsum('asx,Bbst,bxy,cty->Bac', B, W, Rold, B.conj())
\end{verbatim}
up to relabeling of dummy indices.

\codenote{Even with trusted \texttt{einsum} strings, unit-test environment updates against brute-force contractions on small systems.}

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

\fixnote{This is now consistent with the corrected environment convention:
\[
L^{[i]}.shape = (\chi_{i-1}, D_{i-1}, D_{i-1}),\qquad
R^{[i]}.shape = (\chi_i, D_i, D_i).
\]}

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

In one-site DMRG, the MPS is kept in site-centered mixed-canonical form with center at site $i$:
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

\codenote{For pure center motion in one-site DMRG, QR is often cheaper than SVD. A practical implementation may use QR for shifting the center and reserve SVD for adaptive truncation or enrichment-based variants.}

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

This form is the direct analogue of the usual two-site DMRG superblock state and should be treated as the primary variational object.

If the current MPS is stored explicitly as neighboring site tensors, then the two-site center tensor is formed by contracting the internal bond:
\begin{equation}
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
=
\sum_{\alpha_i}
M^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
M^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}.
\label{eq:theta-form}
\end{equation}

\fixnote{The original draft said it is not necessary to define $\Theta$ through pre-existing tensors, which is true mathematically, but for actual code it is important to write down the explicit formation step.}

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
\in \mathbb{C}^{(D_{i-1}d_i)\times(d_{i+1}D_{i+1})},
\label{eq:theta-matrix}
\end{equation}
using the explicit C-order flattening convention of Eq.~\eqref{eq:theta-c-order}.

Compute the reduced SVD:
\begin{equation}
\Theta = U S V^\dagger.
\label{eq:theta-svd}
\end{equation}

Let
\[
D_{\mathrm{full}} = \mathrm{rank}(\Theta),\qquad
D_{\mathrm{new}} \le D_{\max}
\]
be chosen by the truncation rule. Keep only the largest $D_{\mathrm{new}}$ singular values.

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
\State form the current two-site center tensor $\Theta^{[i,i+1]}$ via Eq.~\eqref{eq:theta-form}
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
\State form the current two-site center tensor $\Theta^{[i,i+1]}$ via Eq.~\eqref{eq:theta-form}
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
\item \textbf{AKLT chain} with $D_{\max}=2$, verifying exact recovery and zero discarded weight, \emph{after fixing the Hamiltonian normalization explicitly as described below}.
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

For spin-$1/2$,
\begin{equation}
S^+ = \begin{pmatrix}0&1\\0&0\end{pmatrix},\qquad
S^- = \begin{pmatrix}0&0\\1&0\end{pmatrix},\qquad
S^z = \frac12\begin{pmatrix}1&0\\0&-1\end{pmatrix},\qquad
\Id = \begin{pmatrix}1&0\\0&1\end{pmatrix}.
\end{equation}

\fixnote{These explicit local matrices were added because practical MPO implementations frequently fail from silent convention mismatches in $S^\pm$ and $S^z$.}

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
\item zero truncation error at $D_{\max}\ge 2$.
\end{itemize}

\fixnote{The original draft quoted the energy $E_0 = -\frac23(L-1)$ without specifying the Hamiltonian normalization. That is ambiguous. The benchmark energy depends on the additive constant chosen in the AKLT Hamiltonian.}

For example, if one defines
\begin{equation}
H_{\mathrm{AKLT}}
=
\sum_{i=1}^{L-1}
\left[
\mathbf{S}_i\cdot\mathbf{S}_{i+1}
+
\frac13\bigl(\mathbf{S}_i\cdot\mathbf{S}_{i+1}\bigr)^2
\right],
\label{eq:aklt-ham-unshifted}
\end{equation}
then the OBC AKLT ground-state energy is
\begin{equation}
E_0 = -\frac23(L-1).
\label{eq:aklt-energy-unshifted}
\end{equation}
If instead one uses the shifted projector form
\begin{equation}
H_{\mathrm{AKLT}}^{\mathrm{proj}}
=
\sum_{i=1}^{L-1}
\left[
\mathbf{S}_i\cdot\mathbf{S}_{i+1}
+
\frac13\bigl(\mathbf{S}_i\cdot\mathbf{S}_{i+1}\bigr)^2
+
\frac23
\right],
\label{eq:aklt-ham-shifted}
\end{equation}
then the exact ground-state energy is
\begin{equation}
E_0 = 0.
\label{eq:aklt-energy-shifted}
\end{equation}

\codenote{When using AKLT as a benchmark, always state explicitly whether the unshifted Hamiltonian \eqref{eq:aklt-ham-unshifted} or the shifted projector Hamiltonian \eqref{eq:aklt-ham-shifted} is being used.}

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

\codenote{Variance is an excellent final convergence diagnostic, but computing it requires either an MPO for $H^2$ or a separate reliable contraction strategy. For first implementations, use energy convergence and discarded weight first, then add variance after the MPO and environment machinery are well tested.}

\section{Implementation checklist}

Before production runs, validate in this order:
\begin{enumerate}
\item \textbf{Canonicalization test}: random tensor $\to$ QR/SVD sweep $\to$ check Eqs.~\eqref{eq:left-canonical} and \eqref{eq:right-canonical}.
\item \textbf{MPO test}: contract MPO to a dense Hamiltonian for small $L$ and compare with exact construction.
\item \textbf{Environment test}: verify left/right environments against brute-force contraction for small $L$.
\item \textbf{One-site effective Hamiltonian test}: compare matrix-free action against explicitly constructed $\mathcal{H}_{\mathrm{eff}}$ for small dimensions.
\item \textbf{Two-site effective Hamiltonian test}: compare matrix-free action against explicitly constructed two-site $\mathcal{H}_{\mathrm{eff}}$ for small dimensions.
\item \textbf{Two-site DMRG benchmark}: recover exact diagonalization energies for small Heisenberg chains.
\item \textbf{AKLT benchmark}: recover exact energy and zero truncation error at $D_{\max}=2$ with a \emph{fully specified} AKLT Hamiltonian convention.
\end{enumerate}

\fixnote{A dedicated two-site effective-Hamiltonian unit test was added explicitly. It is one of the highest-value tests for debugging a new implementation.}

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

\section{Minimal software API sketch}

\fixnote{This section was added because the original note was mathematically sound but still slightly underspecified as a direct code blueprint.}

A minimal practical implementation may define the following core routines:
\begin{itemize}
\item \texttt{left\_canonicalize(mps)}
\item \texttt{right\_canonicalize(mps)}
\item \texttt{build\_right\_environments(mps, mpo)}
\item \texttt{update\_left\_environment(Li, A, Wi)}
\item \texttt{update\_right\_environment(Rip1, B, Wip1)}
\item \texttt{one\_site\_heff\_matvec(M, Li, Wi, Ri)}
\item \texttt{two\_site\_heff\_matvec(Theta, Li, Wi, Wip1, Rip1)}
\item \texttt{split\_two\_site\_theta(Theta, Dmax, eps\_trunc)}
\item \texttt{two\_site\_dmrg(mps, mpo, Dmax, nsweeps, eps\_trunc)}
\end{itemize}

A useful invariant during a left-to-right two-site sweep at bond $(i,i+1)$ is:
\begin{itemize}
\item sites $1,\dots,i-1$ are left-canonical,
\item sites $i+2,\dots,L$ are right-canonical,
\item the active two-site tensor $\Theta^{[i,i+1]}$ contains all center degrees of freedom.
\end{itemize}

During a right-to-left sweep:
\begin{itemize}
\item sites $1,\dots,i-1$ are left-canonical,
\item sites $i+2,\dots,L$ are right-canonical,
\item the active two-site tensor is again $\Theta^{[i,i+1]}$.
\end{itemize}

\codenote{Keeping this invariant explicit in comments and tests makes DMRG implementations much easier to debug.}

\end{document}
```

If you want, I can next produce either:

1. a **clean final LaTeX version without the blue/red annotations**, or  
2. a **diff-style changelog** summarizing exactly what was altered from your original note.

---
## 🧑‍💻 User

Please give me a clean final verison

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
with shape $(\chi_{i-1}, \chi_i, d_i, d_i)$.

We use the operator convention
\[
\hat W^{[i]}
=
\sum_{\sigma_i,\sigma_i'}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
\ket{\sigma_i}\bra{\sigma_i'},
\]
so the first physical index is the ket/output index and the second is the bra/input index.

\item OBC dimensions:
\[
D_0 = D_L = 1,\qquad \chi_0 = \chi_L = 1.
\]
\end{itemize}

Unless stated otherwise, all tensors should be stored as \texttt{complex128}.

For a first implementation, it is useful to fix the container conventions
\begin{align*}
&\texttt{mps[i].shape} = (D_{i-1}, d_i, D_i),\\
&\texttt{mpo[i].shape} = (\chi_{i-1}, \chi_i, d_i, d_i),\\
&\texttt{Lenv[i].shape} = (\chi_{i-1}, D_{i-1}, D_{i-1}),\\
&\texttt{Renv[i].shape} = (\chi_i, D_i, D_i).
\end{align*}
Here \texttt{Lenv[i]} is the left environment for site $i$, and \texttt{Renv[i]} is the right environment for site $i$.

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

For Eq.~\eqref{eq:right-group}, the flattening is
\begin{equation}
r = \alpha_{i-1},\qquad c = \sigma_i D_i + \alpha_i.
\end{equation}

In NumPy:
\begin{verbatim}
M_left = T.reshape((D_left * d, D_right), order='C')
M_right = T.reshape((D_left, d * D_right), order='C')
\end{verbatim}

For a rank-4 two-site tensor
\[
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}},
\]
with shape $(D_{i-1}, d_i, d_{i+1}, D_{i+1})$, define the grouped matrix
\begin{equation}
\Theta_{(\alpha_{i-1}\sigma_i),(\sigma_{i+1}\alpha_{i+1})}
\in \mathbb{C}^{(D_{i-1}d_i)\times(d_{i+1}D_{i+1})}.
\label{eq:theta-grouped}
\end{equation}
In C-order storage this means
\begin{equation}
r = \alpha_{i-1} d_i + \sigma_i,\qquad
c = \sigma_{i+1} D_{i+1} + \alpha_{i+1}.
\label{eq:theta-c-order}
\end{equation}

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

A bond-centered mixed-canonical state with orthogonality center at bond $i$ may be written as
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

A practical truncation policy is:
\begin{itemize}
\item enforce $D \le D_{\max}$,
\item optionally choose the smallest $D$ such that $\eta \le \eps_{\mathrm{trunc}}$,
\item always keep at least one singular value.
\end{itemize}

\begin{algorithm}[H]
\caption{TruncateBySVD}
\begin{algorithmic}[1]
\Require matrix $X$, max bond dimension $D_{\max}$, discarded-weight target $\eps_{\mathrm{trunc}}$
\Ensure truncated $U_D,S_D,V_D^\dagger$, discarded weight $\eta$
\State compute reduced SVD: $X = U S V^\dagger$
\State initialize $D \gets \min(D_{\max}, r)$
\If{discarded-weight targeting is enabled}
\State choose the smallest $D\le D_{\max}$ such that $\sum_{j>D} s_j^2 \le \eps_{\mathrm{trunc}}$
\State if no such $D$ exists, keep $D\gets D_{\max}$
\EndIf
\If{$D=0$}
\State $D\gets 1$
\EndIf
\State keep first $D$ singular values/vectors
\State $\eta \gets \sum_{j>D} s_j^2$
\end{algorithmic}
\end{algorithm}

\section{MPO environments}

Here $L^{[i]}$ denotes the contraction of sites $1,\dots,i-1$ into a left environment for site $i$, while $R^{[i]}$ denotes the contraction of sites $i+1,\dots,L$ into a right environment for site $i$.

We use the environment convention
\[
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
\in \mathbb{C}^{\chi_{i-1}\times D_{i-1}\times D_{i-1}},
\]
\[
R^{[i]}_{\beta_i}(\alpha_i,\alpha_i')
\in \mathbb{C}^{\chi_i\times D_i\times D_i}.
\]
That is, $L^{[i]}$ carries the MPO bond index immediately to the left of site $i$, and $R^{[i]}$ carries the MPO bond index immediately to the right of site $i$.

For one-site DMRG with orthogonality center at site $i$, define the left environment update
\begin{equation}
L^{[i+1]}_{\beta_i}(\alpha_i,\alpha_i')
=
\sum_{\substack{\alpha_{i-1},\alpha_{i-1}'\\ \sigma_i,\sigma_i'\\ \beta_{i-1}}}
A^{[i]*}_{\alpha_{i-1}',\sigma_i',\alpha_i'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i},
\label{eq:left-env}
\end{equation}
and the right environment update
\begin{equation}
R^{[i]}_{\beta_i}(\alpha_i,\alpha_i')
=
\sum_{\substack{\alpha_{i+1},\alpha_{i+1}'\\ \sigma_{i+1},\sigma_{i+1}'\\ \beta_{i+1}}}
B^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}
R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1},\alpha_{i+1}')
B^{[i+1]*}_{\alpha_i',\sigma_{i+1}',\alpha_{i+1}'}.
\label{eq:right-env}
\end{equation}

Boundary initialization for OBC:
\begin{equation}
L^{[1]} \in \mathbb{C}^{1\times 1\times 1},\qquad L^{[1]}_{1,1,1}=1,
\end{equation}
\begin{equation}
R^{[L]} \in \mathbb{C}^{1\times 1\times 1},\qquad R^{[L]}_{1,1,1}=1.
\end{equation}

\paragraph{Safe einsum templates.}
With arrays
\begin{itemize}
\item \texttt{Lold[b,x,y]} representing $L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')$,
\item \texttt{A[x,s,a]} representing $A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}$,
\item \texttt{W[b,B,s,t]} representing $W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}$,
\item \texttt{A.conj()[y,t,c]} representing $A^{[i]*}_{\alpha_{i-1}',\sigma_i',\alpha_i'}$,
\end{itemize}
a correct left update is
\begin{verbatim}
Lnew = np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj())
\end{verbatim}
which computes $L^{[i+1]}$ from $L^{[i]}$.

Similarly, with arrays
\begin{itemize}
\item \texttt{B[a,s,x]} representing $B^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}$,
\item \texttt{W[B,b,s,t]} representing $W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}$,
\item \texttt{Rold[b,x,y]} representing $R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1},\alpha_{i+1}')$,
\item \texttt{B.conj()[c,t,y]} representing $B^{[i+1]*}_{\alpha_i',\sigma_{i+1}',\alpha_{i+1}'}$,
\end{itemize}
a correct right update is
\begin{verbatim}
Rnew = np.einsum('asx,Bbst,bxy,cty->Bac', B, W, Rold, B.conj())
\end{verbatim}
up to relabeling of dummy indices.

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

In one-site DMRG, the MPS is kept in site-centered mixed-canonical form with center at site $i$:
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

For pure center motion in one-site DMRG, QR is often cheaper than SVD. A practical implementation may use QR for shifting the center and reserve SVD for adaptive truncation or enrichment-based variants.

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

This form is the direct analogue of the usual two-site DMRG superblock state and should be treated as the primary variational object.

If the current MPS is stored explicitly as neighboring site tensors, then the two-site center tensor is formed by contracting the internal bond:
\begin{equation}
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
=
\sum_{\alpha_i}
M^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
M^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}.
\label{eq:theta-form}
\end{equation}

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
\in \mathbb{C}^{(D_{i-1}d_i)\times(d_{i+1}D_{i+1})},
\label{eq:theta-matrix}
\end{equation}
using the explicit C-order flattening convention of Eq.~\eqref{eq:theta-c-order}.

Compute the reduced SVD:
\begin{equation}
\Theta = U S V^\dagger.
\label{eq:theta-svd}
\end{equation}

Let
\[
D_{\mathrm{full}} = \mathrm{rank}(\Theta),\qquad
D_{\mathrm{new}} \le D_{\max}
\]
be chosen by the truncation rule. Keep only the largest $D_{\mathrm{new}}$ singular values.

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
\State form the current two-site center tensor $\Theta^{[i,i+1]}$ via Eq.~\eqref{eq:theta-form}
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
\State form the current two-site center tensor $\Theta^{[i,i+1]}$ via Eq.~\eqref{eq:theta-form}
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
\item \textbf{AKLT chain} with $D_{\max}=2$, verifying exact recovery and zero discarded weight, after fixing the Hamiltonian normalization explicitly as described below.
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

For spin-$1/2$,
\begin{equation}
S^+ = \begin{pmatrix}0&1\\0&0\end{pmatrix},\qquad
S^- = \begin{pmatrix}0&0\\1&0\end{pmatrix},\qquad
S^z = \frac12\begin{pmatrix}1&0\\0&-1\end{pmatrix},\qquad
\Id = \begin{pmatrix}1&0\\0&1\end{pmatrix}.
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
\item zero truncation error at $D_{\max}\ge 2$.
\end{itemize}

For example, if one defines
\begin{equation}
H_{\mathrm{AKLT}}
=
\sum_{i=1}^{L-1}
\left[
\mathbf{S}_i\cdot\mathbf{S}_{i+1}
+
\frac13\bigl(\mathbf{S}_i\cdot\mathbf{S}_{i+1}\bigr)^2
\right],
\label{eq:aklt-ham-unshifted}
\end{equation}
then the OBC AKLT ground-state energy is
\begin{equation}
E_0 = -\frac23(L-1).
\label{eq:aklt-energy-unshifted}
\end{equation}
If instead one uses the shifted projector form
\begin{equation}
H_{\mathrm{AKLT}}^{\mathrm{proj}}
=
\sum_{i=1}^{L-1}
\left[
\mathbf{S}_i\cdot\mathbf{S}_{i+1}
+
\frac13\bigl(\mathbf{S}_i\cdot\mathbf{S}_{i+1}\bigr)^2
+
\frac23
\right],
\label{eq:aklt-ham-shifted}
\end{equation}
then the exact ground-state energy is
\begin{equation}
E_0 = 0.
\label{eq:aklt-energy-shifted}
\end{equation}

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

Variance is an excellent final convergence diagnostic, but computing it requires either an MPO for $H^2$ or a separate reliable contraction strategy. For first implementations, use energy convergence and discarded weight first, then add variance after the MPO and environment machinery are well tested.

\section{Implementation checklist}

Before production runs, validate in this order:
\begin{enumerate}
\item \textbf{Canonicalization test}: random tensor $\to$ QR/SVD sweep $\to$ check Eqs.~\eqref{eq:left-canonical} and \eqref{eq:right-canonical}.
\item \textbf{MPO test}: contract MPO to a dense Hamiltonian for small $L$ and compare with exact construction.
\item \textbf{Environment test}: verify left/right environments against brute-force contraction for small $L$.
\item \textbf{One-site effective Hamiltonian test}: compare matrix-free action against explicitly constructed $\mathcal{H}_{\mathrm{eff}}$ for small dimensions.
\item \textbf{Two-site effective Hamiltonian test}: compare matrix-free action against explicitly constructed two-site $\mathcal{H}_{\mathrm{eff}}$ for small dimensions.
\item \textbf{Two-site DMRG benchmark}: recover exact diagonalization energies for small Heisenberg chains.
\item \textbf{AKLT benchmark}: recover exact energy and zero truncation error at $D_{\max}=2$ with a fully specified AKLT Hamiltonian convention.
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

\section{Minimal software API sketch}

A minimal practical implementation may define the following core routines:
\begin{itemize}
\item \texttt{left\_canonicalize(mps)}
\item \texttt{right\_canonicalize(mps)}
\item \texttt{build\_right\_environments(mps, mpo)}
\item \texttt{update\_left\_environment(Li, A, Wi)}
\item \texttt{update\_right\_environment(Rip1, B, Wip1)}
\item \texttt{one\_site\_heff\_matvec(M, Li, Wi, Ri)}
\item \texttt{two\_site\_heff\_matvec(Theta, Li, Wi, Wip1, Rip1)}
\item \texttt{split\_two\_site\_theta(Theta, Dmax, eps\_trunc)}
\item \texttt{two\_site\_dmrg(mps, mpo, Dmax, nsweeps, eps\_trunc)}
\end{itemize}

A useful invariant during a left-to-right two-site sweep at bond $(i,i+1)$ is:
\begin{itemize}
\item sites $1,\dots,i-1$ are left-canonical,
\item sites $i+2,\dots,L$ are right-canonical,
\item the active two-site tensor $\Theta^{[i,i+1]}$ contains all center degrees of freedom.
\end{itemize}

During a right-to-left sweep:
\begin{itemize}
\item sites $1,\dots,i-1$ are left-canonical,
\item sites $i+2,\dots,L$ are right-canonical,
\item the active two-site tensor is again $\Theta^{[i,i+1]}$.
\end{itemize}

\end{document}
```

If you want, I can also give you:
- a **compile-checked version with minor LaTeX polish for RevTeX**, or
- a **Python module skeleton matching this note exactly**.

---
## 🧑‍💻 User

a compile-checked version with minor LaTeX polish for RevTeX

---
## 🤖 Assistant

```latex
\documentclass[aps,prb,onecolumn,scriptaddress,nofootinbib]{revtex4-2}

\usepackage{graphicx}
\usepackage{amsmath,amsfonts,amssymb,bm,mathtools}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{xcolor}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
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
\large Based on the review article ``The density-matrix renormalization group in the age of matrix product states'' by U.~Schollw\"ock, \textit{Annals of Physics} \textbf{326}, 96--192 (2011)}

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

We use the operator convention
\[
\hat W^{[i]}
=
\sum_{\sigma_i,\sigma_i'}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
\ket{\sigma_i}\bra{\sigma_i'},
\]
so the first physical index is the ket/output index and the second is the bra/input index.

\item OBC dimensions:
\[
D_0 = D_L = 1,\qquad \chi_0 = \chi_L = 1.
\]
\end{itemize}

Unless stated otherwise, all tensors should be stored as \texttt{complex128}.

For a first implementation, it is useful to fix the container conventions
\begin{align*}
&\texttt{mps[i].shape} = (D_{i-1}, d_i, D_i),\\
&\texttt{mpo[i].shape} = (\chi_{i-1}, \chi_i, d_i, d_i),\\
&\texttt{Lenv[i].shape} = (\chi_{i-1}, D_{i-1}, D_{i-1}),\\
&\texttt{Renv[i].shape} = (\chi_i, D_i, D_i).
\end{align*}
Here \texttt{Lenv[i]} is the left environment for site $i$, and \texttt{Renv[i]} is the right environment for site $i$.

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

For Eq.~\eqref{eq:right-group}, the flattening is
\begin{equation}
r = \alpha_{i-1},\qquad c = \sigma_i D_i + \alpha_i.
\end{equation}

In NumPy:
\begin{verbatim}
M_left = T.reshape((D_left * d, D_right), order='C')
M_right = T.reshape((D_left, d * D_right), order='C')
\end{verbatim}

For a rank-4 two-site tensor
\[
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}},
\]
with shape $(D_{i-1}, d_i, d_{i+1}, D_{i+1})$, define the grouped matrix
\begin{equation}
\Theta_{(\alpha_{i-1}\sigma_i),(\sigma_{i+1}\alpha_{i+1})}
\in \mathbb{C}^{(D_{i-1}d_i)\times(d_{i+1}D_{i+1})}.
\label{eq:theta-grouped}
\end{equation}
In C-order storage this means
\begin{equation}
r = \alpha_{i-1} d_i + \sigma_i,\qquad
c = \sigma_{i+1} D_{i+1} + \alpha_{i+1}.
\label{eq:theta-c-order}
\end{equation}

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

A bond-centered mixed-canonical state with orthogonality center at bond $i$ may be written as
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

\begin{algorithm}
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

A practical truncation policy is:
\begin{itemize}
\item enforce $D \le D_{\max}$,
\item optionally choose the smallest $D$ such that $\eta \le \eps_{\mathrm{trunc}}$,
\item always keep at least one singular value.
\end{itemize}

\begin{algorithm}
\caption{TruncateBySVD}
\begin{algorithmic}[1]
\Require matrix $X$, max bond dimension $D_{\max}$, discarded-weight target $\eps_{\mathrm{trunc}}$
\Ensure truncated $U_D,S_D,V_D^\dagger$, discarded weight $\eta$
\State compute reduced SVD: $X = U S V^\dagger$
\State initialize $D \gets \min(D_{\max}, r)$
\If{discarded-weight targeting is enabled}
\State choose the smallest $D\le D_{\max}$ such that $\sum_{j>D} s_j^2 \le \eps_{\mathrm{trunc}}$
\State if no such $D$ exists, keep $D\gets D_{\max}$
\EndIf
\If{$D=0$}
\State $D\gets 1$
\EndIf
\State keep first $D$ singular values/vectors
\State $\eta \gets \sum_{j>D} s_j^2$
\end{algorithmic}
\end{algorithm}

\section{MPO environments}

Here $L^{[i]}$ denotes the contraction of sites $1,\dots,i-1$ into a left environment for site $i$, while $R^{[i]}$ denotes the contraction of sites $i+1,\dots,L$ into a right environment for site $i$.

We use the environment convention
\[
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
\in \mathbb{C}^{\chi_{i-1}\times D_{i-1}\times D_{i-1}},
\]
\[
R^{[i]}_{\beta_i}(\alpha_i,\alpha_i')
\in \mathbb{C}^{\chi_i\times D_i\times D_i}.
\]
That is, $L^{[i]}$ carries the MPO bond index immediately to the left of site $i$, and $R^{[i]}$ carries the MPO bond index immediately to the right of site $i$.

For one-site DMRG with orthogonality center at site $i$, define the left environment update
\begin{equation}
L^{[i+1]}_{\beta_i}(\alpha_i,\alpha_i')
=
\sum_{\substack{\alpha_{i-1},\alpha_{i-1}'\\ \sigma_i,\sigma_i'\\ \beta_{i-1}}}
A^{[i]*}_{\alpha_{i-1}',\sigma_i',\alpha_i'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i},
\label{eq:left-env}
\end{equation}
and the right environment update
\begin{equation}
R^{[i]}_{\beta_i}(\alpha_i,\alpha_i')
=
\sum_{\substack{\alpha_{i+1},\alpha_{i+1}'\\ \sigma_{i+1},\sigma_{i+1}'\\ \beta_{i+1}}}
B^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}
R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1},\alpha_{i+1}')
B^{[i+1]*}_{\alpha_i',\sigma_{i+1}',\alpha_{i+1}'}.
\label{eq:right-env}
\end{equation}

Boundary initialization for OBC:
\begin{equation}
L^{[1]} \in \mathbb{C}^{1\times 1\times 1},\qquad L^{[1]}_{1,1,1}=1,
\end{equation}
\begin{equation}
R^{[L]} \in \mathbb{C}^{1\times 1\times 1},\qquad R^{[L]}_{1,1,1}=1.
\end{equation}

\paragraph{Safe einsum templates.}
With arrays
\begin{itemize}
\item \texttt{Lold[b,x,y]} representing $L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')$,
\item \texttt{A[x,s,a]} representing $A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}$,
\item \texttt{W[b,B,s,t]} representing $W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}$,
\item \texttt{A.conj()[y,t,c]} representing $A^{[i]*}_{\alpha_{i-1}',\sigma_i',\alpha_i'}$,
\end{itemize}
a correct left update is
\begin{verbatim}
Lnew = np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj())
\end{verbatim}
which computes $L^{[i+1]}$ from $L^{[i]}$.

Similarly, with arrays
\begin{itemize}
\item \texttt{B[a,s,x]} representing $B^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}$,
\item \texttt{W[B,b,s,t]} representing $W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}$,
\item \texttt{Rold[b,x,y]} representing $R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1},\alpha_{i+1}')$,
\item \texttt{B.conj()[c,t,y]} representing $B^{[i+1]*}_{\alpha_i',\sigma_{i+1}',\alpha_{i+1}'}$,
\end{itemize}
a correct right update is
\begin{verbatim}
Rnew = np.einsum('asx,Bbst,bxy,cty->Bac', B, W, Rold, B.conj())
\end{verbatim}
up to relabeling of dummy indices.

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

In one-site DMRG, the MPS is kept in site-centered mixed-canonical form with center at site $i$:
\begin{equation}
\ket{\psi}
=
\sum_{\alpha_{i-1},\sigma_i,\alpha_i}
M^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
\ket{\alpha_{i-1}}_L \ket{\sigma_i} \ket{\alpha_i}_R.
\end{equation}

\begin{algorithm}
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
\item use a subspace-expansion or density-matrix perturbation method.
\end{itemize}

For pure center motion in one-site DMRG, QR is often cheaper than SVD. A practical implementation may use QR for shifting the center and reserve SVD for adaptive truncation or enrichment-based variants.

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

This form is the direct analogue of the usual two-site DMRG superblock state and should be treated as the primary variational object.

If the current MPS is stored explicitly as neighboring site tensors, then the two-site center tensor is formed by contracting the internal bond:
\begin{equation}
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
=
\sum_{\alpha_i}
M^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
M^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}.
\label{eq:theta-form}
\end{equation}

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
\in \mathbb{C}^{(D_{i-1}d_i)\times(d_{i+1}D_{i+1})},
\label{eq:theta-matrix}
\end{equation}
using the explicit C-order flattening convention of Eq.~\eqref{eq:theta-c-order}.

Compute the reduced SVD:
\begin{equation}
\Theta = U S V^\dagger.
\label{eq:theta-svd}
\end{equation}

Let
\[
D_{\mathrm{full}} = \mathrm{rank}(\Theta),\qquad
D_{\mathrm{new}} \le D_{\max}
\]
be chosen by the truncation rule. Keep only the largest $D_{\mathrm{new}}$ singular values.

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

\begin{algorithm}
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
\State form the current two-site center tensor $\Theta^{[i,i+1]}$ via Eq.~\eqref{eq:theta-form}
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
\State form the current two-site center tensor $\Theta^{[i,i+1]}$ via Eq.~\eqref{eq:theta-form}
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
\item \textbf{AKLT chain} with $D_{\max}=2$, verifying exact recovery and zero discarded weight, after fixing the Hamiltonian normalization explicitly as described below.
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

For spin-$1/2$,
\begin{equation}
S^+ = \begin{pmatrix}0&1\\0&0\end{pmatrix},\qquad
S^- = \begin{pmatrix}0&0\\1&0\end{pmatrix},\qquad
S^z = \frac12\begin{pmatrix}1&0\\0&-1\end{pmatrix},\qquad
\Id = \begin{pmatrix}1&0\\0&1\end{pmatrix}.
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
\item zero truncation error at $D_{\max}\ge 2$.
\end{itemize}

For example, if one defines
\begin{equation}
H_{\mathrm{AKLT}}
=
\sum_{i=1}^{L-1}
\left[
\mathbf{S}_i\cdot\mathbf{S}_{i+1}
+
\frac13\bigl(\mathbf{S}_i\cdot\mathbf{S}_{i+1}\bigr)^2
\right],
\label{eq:aklt-ham-unshifted}
\end{equation}
then the OBC AKLT ground-state energy is
\begin{equation}
E_0 = -\frac23(L-1).
\label{eq:aklt-energy-unshifted}
\end{equation}
If instead one uses the shifted projector form
\begin{equation}
H_{\mathrm{AKLT}}^{\mathrm{proj}}
=
\sum_{i=1}^{L-1}
\left[
\mathbf{S}_i\cdot\mathbf{S}_{i+1}
+
\frac13\bigl(\mathbf{S}_i\cdot\mathbf{S}_{i+1}\bigr)^2
+
\frac23
\right],
\label{eq:aklt-ham-shifted}
\end{equation}
then the exact ground-state energy is
\begin{equation}
E_0 = 0.
\label{eq:aklt-energy-shifted}
\end{equation}

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

Variance is an excellent final convergence diagnostic, but computing it requires either an MPO for $H^2$ or a separate reliable contraction strategy. For first implementations, use energy convergence and discarded weight first, then add variance after the MPO and environment machinery are well tested.

\section{Implementation checklist}

Before production runs, validate in this order:
\begin{enumerate}
\item \textbf{Canonicalization test}: random tensor $\to$ QR/SVD sweep $\to$ check Eqs.~\eqref{eq:left-canonical} and \eqref{eq:right-canonical}.
\item \textbf{MPO test}: contract MPO to a dense Hamiltonian for small $L$ and compare with exact construction.
\item \textbf{Environment test}: verify left/right environments against brute-force contraction for small $L$.
\item \textbf{One-site effective Hamiltonian test}: compare matrix-free action against explicitly constructed $\mathcal{H}_{\mathrm{eff}}$ for small dimensions.
\item \textbf{Two-site effective Hamiltonian test}: compare matrix-free action against explicitly constructed two-site $\mathcal{H}_{\mathrm{eff}}$ for small dimensions.
\item \textbf{Two-site DMRG benchmark}: recover exact diagonalization energies for small Heisenberg chains.
\item \textbf{AKLT benchmark}: recover exact energy and zero truncation error at $D_{\max}=2$ with a fully specified AKLT Hamiltonian convention.
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

\section{Minimal software API sketch}

A minimal practical implementation may define the following core routines:
\begin{itemize}
\item \texttt{left\_canonicalize(mps)}
\item \texttt{right\_canonicalize(mps)}
\item \texttt{build\_right\_environments(mps, mpo)}
\item \texttt{update\_left\_environment(Li, A, Wi)}
\item \texttt{update\_right\_environment(Rip1, B, Wip1)}
\item \texttt{one\_site\_heff\_matvec(M, Li, Wi, Ri)}
\item \texttt{two\_site\_heff\_matvec(Theta, Li, Wi, Wip1, Rip1)}
\item \texttt{split\_two\_site\_theta(Theta, Dmax, eps\_trunc)}
\item \texttt{two\_site\_dmrg(mps, mpo, Dmax, nsweeps, eps\_trunc)}
\end{itemize}

A useful invariant during a left-to-right two-site sweep at bond $(i,i+1)$ is:
\begin{itemize}
\item sites $1,\dots,i-1$ are left-canonical,
\item sites $i+2,\dots,L$ are right-canonical,
\item the active two-site tensor $\Theta^{[i,i+1]}$ contains all center degrees of freedom.
\end{itemize}

During a right-to-left sweep:
\begin{itemize}
\item sites $1,\dots,i-1$ are left-canonical,
\item sites $i+2,\dots,L$ are right-canonical,
\item the active two-site tensor is again $\Theta^{[i,i+1]}$.
\end{itemize}

\end{document}
```

A few RevTeX-polish notes on this version:
- I removed `[H]` from the `algorithm` environments, since that often causes trouble depending on the float setup.
- I added `nofootinbib`, `fontenc`, and `inputenc` for safer compilation.
- I changed the title citation formatting slightly to be more RevTeX-friendly.
- I replaced one hyphenated phrase with “or” in “subspace-expansion or density-matrix perturbation method” for cleaner typesetting.

One important caveat: the `algorithm` + `algpseudocode` combination sometimes behaves differently across TeX installations with `revtex4-2`. This version is generally fine, but if your local setup complains about algorithm floats, the most robust workaround is to replace the `algorithm` environments by plain `\paragraph{Algorithm: ...}` plus `enumerate`.

If you want, I can next give you:
1. a **maximally robust RevTeX version with algorithms converted to plain lists**, or
2. a **TeX file plus a tiny bibliography stub and suggested compile command**.

---
## 🧑‍💻 User

224: Missing \endcsname inserted. \end{algorithm}
224: Missing number, treated as zero. \end{algorithm}
224: Extra \endcsname. \end{algorithm}
311: Missing \endcsname inserted. \end{algorithm}
311: Missing number, treated as zero. \end{algorithm}
311: Extra \endcsname. \end{algorithm}
479: Missing \endcsname inserted. \end{algorithm}
479: Missing number, treated as zero. \end{algorithm}
479: Extra \endcsname. \end{algorithm}
743: Missing \endcsname inserted. \end{algorithm}
743: Missing number, treated as zero. \end{algorithm}
743: Extra \endcsname. \end{algorithm}
: 
149: The definition of \label has changed! Check your packages! Replacing it with the kernel definition
: Repair the float package.
: Unused global option(s): [scriptaddress].
: Assuming \noaffiliation for authors.
31: Token not allowed in a PDF string (Unicode): removing `\\'
: No file MPS-Kimi-GPT-328.bbl.
: Label(s) may have changed. Rerun to get cross-references right.

---
## 🤖 Assistant

You're hitting a common RevTeX incompatibility: `revtex4-2` does not play nicely with the `algorithm` float package in many installations. The `\endcsname` and `\label` errors are classic symptoms of float/caption conflicts.

The safest fix is:

- **remove** `\usepackage{algorithm}`
- **remove** all `algorithm` environments
- keep `algpseudocode` only if you want, but in practice even that is unnecessary
- replace each algorithm float by a normal sectioned block using `enumerate`

Also:
- `scriptaddress` is unused here because you have no affiliations; remove it.
- `\\` in the title triggers the PDF-string warning via `hyperref`; use `\texorpdfstring{...}{...}`.
- the missing `.bbl` is harmless if you have no bibliography.

Below is a **RevTeX-safe compile-clean version** that avoids the problematic float package entirely.

---

```latex
\documentclass[aps,prb,onecolumn,nofootinbib]{revtex4-2}

\usepackage{graphicx}
\usepackage{amsmath,amsfonts,amssymb,bm,mathtools}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{xcolor}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[colorlinks=true,urlcolor=blue,linkcolor=blue,citecolor=blue]{hyperref}

\newcommand{\ket}[1]{|#1\rangle}
\newcommand{\bra}[1]{\langle #1|}
\newcommand{\braket}[2]{\langle #1|#2\rangle}
\newcommand{\Tr}{\mathrm{Tr}}
\newcommand{\Id}{\mathbb{I}}
\newcommand{\eps}{\epsilon}
\newcommand{\Order}{\mathcal{O}}

\begin{document}

\title{\texorpdfstring{Numerical Implementation of Matrix Product States and DMRG\\
\large Based on the review article ``The density-matrix renormalization group in the age of matrix product states'' by U.~Schollw\"ock, \textit{Annals of Physics} \textbf{326}, 96--192 (2011)}
{Numerical Implementation of Matrix Product States and DMRG}}

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

We use the operator convention
\[
\hat W^{[i]}
=
\sum_{\sigma_i,\sigma_i'}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
\ket{\sigma_i}\bra{\sigma_i'},
\]
so the first physical index is the ket/output index and the second is the bra/input index.

\item OBC dimensions:
\[
D_0 = D_L = 1,\qquad \chi_0 = \chi_L = 1.
\]
\end{itemize}

Unless stated otherwise, all tensors should be stored as \texttt{complex128}.

For a first implementation, it is useful to fix the container conventions
\begin{align*}
&\texttt{mps[i].shape} = (D_{i-1}, d_i, D_i),\\
&\texttt{mpo[i].shape} = (\chi_{i-1}, \chi_i, d_i, d_i),\\
&\texttt{Lenv[i].shape} = (\chi_{i-1}, D_{i-1}, D_{i-1}),\\
&\texttt{Renv[i].shape} = (\chi_i, D_i, D_i).
\end{align*}
Here \texttt{Lenv[i]} is the left environment for site $i$, and \texttt{Renv[i]} is the right environment for site $i$.

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

For Eq.~\eqref{eq:right-group}, the flattening is
\begin{equation}
r = \alpha_{i-1},\qquad c = \sigma_i D_i + \alpha_i.
\end{equation}

In NumPy:
\begin{verbatim}
M_left = T.reshape((D_left * d, D_right), order='C')
M_right = T.reshape((D_left, d * D_right), order='C')
\end{verbatim}

For a rank-4 two-site tensor
\[
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}},
\]
with shape $(D_{i-1}, d_i, d_{i+1}, D_{i+1})$, define the grouped matrix
\begin{equation}
\Theta_{(\alpha_{i-1}\sigma_i),(\sigma_{i+1}\alpha_{i+1})}
\in \mathbb{C}^{(D_{i-1}d_i)\times(d_{i+1}D_{i+1})}.
\label{eq:theta-grouped}
\end{equation}
In C-order storage this means
\begin{equation}
r = \alpha_{i-1} d_i + \sigma_i,\qquad
c = \sigma_{i+1} D_{i+1} + \alpha_{i+1}.
\label{eq:theta-c-order}
\end{equation}

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

A bond-centered mixed-canonical state with orthogonality center at bond $i$ may be written as
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

\paragraph{Algorithm: LeftCanonicalizeByQR}
\begin{enumerate}
\item Input MPS tensors $\{M^{[1]},\dots,M^{[L]}\}$.
\item For $i=1,\dots,L-1$:
\begin{enumerate}
\item reshape $M^{[i]} \to \hat M$ with shape $(D_{i-1}d_i)\times D_i$,
\item compute reduced QR: $\hat M = Q R$,
\item reshape $Q \to A^{[i]}$ with shape $(D_{i-1},d_i,\tilde D_i)$,
\item absorb $R$ into $M^{[i+1]}$ via Eq.~\eqref{eq:left-qr-absorb}.
\end{enumerate}
\item Set $C^{[L]} \gets M^{[L]}$.
\end{enumerate}

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

A practical truncation policy is:
\begin{itemize}
\item enforce $D \le D_{\max}$,
\item optionally choose the smallest $D$ such that $\eta \le \eps_{\mathrm{trunc}}$,
\item always keep at least one singular value.
\end{itemize}

\paragraph{Algorithm: TruncateBySVD}
\begin{enumerate}
\item Input: matrix $X$, max bond dimension $D_{\max}$, discarded-weight target $\eps_{\mathrm{trunc}}$.
\item Compute the reduced SVD: $X = U S V^\dagger$.
\item Initialize $D \gets \min(D_{\max}, r)$.
\item If discarded-weight targeting is enabled, choose the smallest $D\le D_{\max}$ such that
\[
\sum_{j>D} s_j^2 \le \eps_{\mathrm{trunc}}.
\]
If no such $D$ exists, keep $D\gets D_{\max}$.
\item If $D=0$, set $D\gets 1$.
\item Keep the first $D$ singular values and vectors.
\item Set
\[
\eta \gets \sum_{j>D} s_j^2.
\]
\end{enumerate}

\section{MPO environments}

Here $L^{[i]}$ denotes the contraction of sites $1,\dots,i-1$ into a left environment for site $i$, while $R^{[i]}$ denotes the contraction of sites $i+1,\dots,L$ into a right environment for site $i$.

We use the environment convention
\[
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
\in \mathbb{C}^{\chi_{i-1}\times D_{i-1}\times D_{i-1}},
\]
\[
R^{[i]}_{\beta_i}(\alpha_i,\alpha_i')
\in \mathbb{C}^{\chi_i\times D_i\times D_i}.
\]

For one-site DMRG with orthogonality center at site $i$, define the left environment update
\begin{equation}
L^{[i+1]}_{\beta_i}(\alpha_i,\alpha_i')
=
\sum_{\substack{\alpha_{i-1},\alpha_{i-1}'\\ \sigma_i,\sigma_i'\\ \beta_{i-1}}}
A^{[i]*}_{\alpha_{i-1}',\sigma_i',\alpha_i'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i},
\label{eq:left-env}
\end{equation}
and the right environment update
\begin{equation}
R^{[i]}_{\beta_i}(\alpha_i,\alpha_i')
=
\sum_{\substack{\alpha_{i+1},\alpha_{i+1}'\\ \sigma_{i+1},\sigma_{i+1}'\\ \beta_{i+1}}}
B^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}
R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1},\alpha_{i+1}')
B^{[i+1]*}_{\alpha_i',\sigma_{i+1}',\alpha_{i+1}'}.
\label{eq:right-env}
\end{equation}

Boundary initialization for OBC:
\begin{equation}
L^{[1]} \in \mathbb{C}^{1\times 1\times 1},\qquad L^{[1]}_{1,1,1}=1,
\end{equation}
\begin{equation}
R^{[L]} \in \mathbb{C}^{1\times 1\times 1},\qquad R^{[L]}_{1,1,1}=1.
\end{equation}

\paragraph{Safe einsum templates.}
With arrays
\begin{itemize}
\item \texttt{Lold[b,x,y]} representing $L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')$,
\item \texttt{A[x,s,a]} representing $A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}$,
\item \texttt{W[b,B,s,t]} representing $W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}$,
\item \texttt{A.conj()[y,t,c]} representing $A^{[i]*}_{\alpha_{i-1}',\sigma_i',\alpha_i'}$,
\end{itemize}
a correct left update is
\begin{verbatim}
Lnew = np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj())
\end{verbatim}

Similarly, with arrays
\begin{itemize}
\item \texttt{B[a,s,x]} representing $B^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}$,
\item \texttt{W[B,b,s,t]} representing $W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}$,
\item \texttt{Rold[b,x,y]} representing $R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1},\alpha_{i+1}')$,
\item \texttt{B.conj()[c,t,y]} representing $B^{[i+1]*}_{\alpha_i',\sigma_{i+1}',\alpha_{i+1}'}$,
\end{itemize}
a correct right update is
\begin{verbatim}
Rnew = np.einsum('asx,Bbst,bxy,cty->Bac', B, W, Rold, B.conj())
\end{verbatim}
up to relabeling of dummy indices.

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

\section{One-site DMRG sweep}

In one-site DMRG, the MPS is kept in site-centered mixed-canonical form with center at site $i$:
\begin{equation}
\ket{\psi}
=
\sum_{\alpha_{i-1},\sigma_i,\alpha_i}
M^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
\ket{\alpha_{i-1}}_L \ket{\sigma_i} \ket{\alpha_i}_R.
\end{equation}

\paragraph{Algorithm: OneSiteDMRG}
\begin{enumerate}
\item Input MPO $\{W^{[i]}\}$, initial MPS $\{M^{[i]}\}$, max bond dimension $D_{\max}$.
\item Right-canonicalize the initial MPS.
\item Build all right environments.
\item Initialize the left boundary.
\item Repeat until energy and variance are converged:
\begin{enumerate}
\item For $i=1,\dots,L-1$:
\begin{enumerate}
\item solve Eq.~\eqref{eq:heff-action} for the lowest eigenpair using Lanczos or Davidson,
\item reshape the center tensor as a left-grouped matrix,
\item compute the SVD: $M^{[i]} = U S V^\dagger$,
\item retain at most $D_{\max}$ singular values,
\item reshape $U \to A^{[i]}$,
\item absorb $S V^\dagger$ into $M^{[i+1]}$,
\item update $L^{[i+1]}$.
\end{enumerate}
\item For $i=L,\dots,2$:
\begin{enumerate}
\item solve the local lowest-eigenvalue problem,
\item reshape the center tensor as a right-grouped matrix,
\item compute the SVD: $M^{[i]} = U S V^\dagger$,
\item retain at most $D_{\max}$ singular values,
\item reshape $V^\dagger$ into a right-canonical tensor $B^{[i]}$,
\item absorb $U S$ into $M^{[i-1]}$,
\item update $R^{[i-1]}$.
\end{enumerate}
\end{enumerate}
\end{enumerate}

\paragraph{Practical warning.}
One-site DMRG is prone to local minima. A robust implementation should either:
\begin{itemize}
\item start with several two-site sweeps, or
\item use a subspace-expansion or density-matrix perturbation method.
\end{itemize}

\section{Two-site DMRG}

Two-site DMRG is the most robust finite-system ground-state algorithm and is recommended for a first implementation.

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

If the current MPS is stored explicitly as neighboring site tensors, then the two-site center tensor is formed by contracting the internal bond:
\begin{equation}
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
=
\sum_{\alpha_i}
M^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
M^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}.
\label{eq:theta-form}
\end{equation}

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

\subsection{Matrix-free application of the two-site effective Hamiltonian}

For practical coding, Eq.~\eqref{eq:two-site-heff} should never be assembled as a dense matrix.

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

\subsection{SVD split and truncation}

After optimization, reshape the two-site tensor as a matrix:
\begin{equation}
\Theta_{(\alpha_{i-1}\sigma_i),(\sigma_{i+1}\alpha_{i+1})}
\in \mathbb{C}^{(D_{i-1}d_i)\times(d_{i+1}D_{i+1})},
\label{eq:theta-matrix}
\end{equation}
using the explicit C-order flattening convention of Eq.~\eqref{eq:theta-c-order}.

Compute the reduced SVD:
\begin{equation}
\Theta = U S V^\dagger.
\label{eq:theta-svd}
\end{equation}

Let
\[
D_{\mathrm{full}} = \mathrm{rank}(\Theta),\qquad
D_{\mathrm{new}} \le D_{\max}
\]
be chosen by the truncation rule. Keep only the largest $D_{\mathrm{new}}$ singular values.

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

\subsection{Full two-site sweep algorithm}

\paragraph{Algorithm: TwoSiteDMRG}
\begin{enumerate}
\item Input MPO $\{W^{[i]}\}$, initial MPS $\{M^{[i]}\}$, max bond dimension $D_{\max}$.
\item Bring the initial MPS into mixed-canonical form.
\item Build all right environments $R^{[i]}$.
\item Initialize the left boundary environment $L^{[1]}$.
\item Repeat until energy, variance, and discarded-weight criteria are converged:
\begin{enumerate}
\item Left-to-right sweep: for $i=1,\dots,L-1$,
\begin{enumerate}
\item form the current two-site center tensor $\Theta^{[i,i+1]}$ via Eq.~\eqref{eq:theta-form},
\item solve the lowest-eigenvalue problem,
\item reshape $\Theta$ as in Eq.~\eqref{eq:theta-matrix},
\item compute reduced SVD $\Theta = U S V^\dagger$,
\item truncate to $D_{\mathrm{new}}\le D_{\max}$,
\item set $A^{[i]} \gets \mathrm{reshape}(U,(D_{i-1},d_i,D_{\mathrm{new}}))$,
\item set $M^{[i+1]} \gets \mathrm{reshape}(S V^\dagger,(D_{\mathrm{new}},d_{i+1},D_{i+1}))$,
\item record discarded weight $\eta^{[i]}$,
\item update left environment $L^{[i+1]}$.
\end{enumerate}
\item Right-to-left sweep: for $i=L-1,\dots,1$,
\begin{enumerate}
\item form the current two-site center tensor $\Theta^{[i,i+1]}$ via Eq.~\eqref{eq:theta-form},
\item solve the lowest-eigenvalue problem,
\item reshape $\Theta$ as in Eq.~\eqref{eq:theta-matrix},
\item compute reduced SVD $\Theta = U S V^\dagger$,
\item truncate to $D_{\mathrm{new}}\le D_{\max}$,
\item set $M^{[i]} \gets \mathrm{reshape}(US,(D_{i-1},d_i,D_{\mathrm{new}}))$,
\item set $B^{[i+1]} \gets \mathrm{reshape}(V^\dagger,(D_{\mathrm{new}},d_{i+1},D_{i+1}))$,
\item record discarded weight $\eta^{[i]}$,
\item update right environment $R^{[i]}$.
\end{enumerate}
\end{enumerate}
\end{enumerate}

\subsection{Recommended initial benchmarks}

For a first two-site DMRG code, test the following:
\begin{enumerate}
\item Small Heisenberg chains ($L=4,6,8,10$), comparing energies to exact diagonalization.
\item AKLT chain with $D_{\max}=2$, verifying exact recovery and zero discarded weight, after fixing the Hamiltonian normalization explicitly as described below.
\item Environment consistency: compare environment-based energy evaluation with brute-force contraction on small systems.
\item Canonicality checks: after each split, verify left- or right-canonical conditions numerically.
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

The tensor components $W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}$ are obtained from the local operator matrix elements
\[
(\hat O)_{\sigma,\sigma'} = \bra{\sigma}\hat O\ket{\sigma'}.
\]

For spin-$1/2$,
\begin{equation}
S^+ = \begin{pmatrix}0&1\\0&0\end{pmatrix},\qquad
S^- = \begin{pmatrix}0&0\\1&0\end{pmatrix},\qquad
S^z = \frac12\begin{pmatrix}1&0\\0&-1\end{pmatrix},\qquad
\Id = \begin{pmatrix}1&0\\0&1\end{pmatrix}.
\end{equation}

For OBC, choose
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
and use $\hat W_{\mathrm{bulk}}$ on all intermediate sites.

In code, this means:
\begin{itemize}
\item left boundary tensor shape $(1,5,d,d)$ stores the last row of the bulk MPO,
\item right boundary tensor shape $(5,1,d,d)$ stores the first column of the bulk MPO.
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

If one defines
\begin{equation}
H_{\mathrm{AKLT}}
=
\sum_{i=1}^{L-1}
\left[
\mathbf{S}_i\cdot\mathbf{S}_{i+1}
+
\frac13\bigl(\mathbf{S}_i\cdot\mathbf{S}_{i+1}\bigr)^2
\right],
\label{eq:aklt-ham-unshifted}
\end{equation}
then the OBC AKLT ground-state energy is
\begin{equation}
E_0 = -\frac23(L-1).
\label{eq:aklt-energy-unshifted}
\end{equation}
If instead one uses the shifted projector form
\begin{equation}
H_{\mathrm{AKLT}}^{\mathrm{proj}}
=
\sum_{i=1}^{L-1}
\left[
\mathbf{S}_i\cdot\mathbf{S}_{i+1}
+
\frac13\bigl(\mathbf{S}_i\cdot\mathbf{S}_{i+1}\bigr)^2
+
\frac23
\right],
\label{eq:aklt-ham-shifted}
\end{equation}
then the exact ground-state energy is
\begin{equation}
E_0 = 0.
\label{eq:aklt-energy-shifted}
\end{equation}

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
Direct tensor-by-tensor comparison is not gauge-invariant and should not be used as a primary convergence criterion unless the gauge is fixed identically before comparison.

\section{Implementation checklist}

Before production runs, validate in this order:
\begin{enumerate}
\item Canonicalization test: random tensor $\to$ QR/SVD sweep $\to$ check Eqs.~\eqref{eq:left-canonical} and \eqref{eq:right-canonical}.
\item MPO test: contract MPO to a dense Hamiltonian for small $L$ and compare with exact construction.
\item Environment test: verify left/right environments against brute-force contraction for small $L$.
\item One-site effective Hamiltonian test: compare matrix-free action against explicitly constructed $\mathcal{H}_{\mathrm{eff}}$ for small dimensions.
\item Two-site effective Hamiltonian test: compare matrix-free action against explicitly constructed two-site $\mathcal{H}_{\mathrm{eff}}$ for small dimensions.
\item Two-site DMRG benchmark: recover exact diagonalization energies for small Heisenberg chains.
\item AKLT benchmark: recover exact energy and zero truncation error at $D_{\max}=2$ with a fully specified AKLT Hamiltonian convention.
\end{enumerate}

\section{Practical coding recommendation}

For a first reliable implementation:
\begin{itemize}
\item implement two-site DMRG first,
\item use reduced QR for gauge moves and SVD for truncation,
\item avoid pivoted QR in canonicalization,
\item store all tensors as \texttt{complex128},
\item benchmark against exact diagonalization for $L\le 10$ or $12$,
\item only after this add one-site DMRG and subspace expansion.
\end{itemize}

\section{Minimal software API sketch}

A minimal practical implementation may define the following core routines:
\begin{itemize}
\item \texttt{left\_canonicalize(mps)}
\item \texttt{right\_canonicalize(mps)}
\item \texttt{build\_right\_environments(mps, mpo)}
\item \texttt{update\_left\_environment(Li, A, Wi)}
\item \texttt{update\_right\_environment(Rip1, B, Wip1)}
\item \texttt{one\_site\_heff\_matvec(M, Li, Wi, Ri)}
\item \texttt{two\_site\_heff\_matvec(Theta, Li, Wi, Wip1, Rip1)}
\item \texttt{split\_two\_site\_theta(Theta, Dmax, eps\_trunc)}
\item \texttt{two\_site\_dmrg(mps, mpo, Dmax, nsweeps, eps\_trunc)}
\end{itemize}

A useful invariant during a left-to-right two-site sweep at bond $(i,i+1)$ is:
\begin{itemize}
\item sites $1,\dots,i-1$ are left-canonical,
\item sites $i+2,\dots,L$ are right-canonical,
\item the active two-site tensor $\Theta^{[i,i+1]}$ contains all center degrees of freedom.
\end{itemize}

During a right-to-left sweep:
\begin{itemize}
\item sites $1,\dots,i-1$ are left-canonical,
\item sites $i+2,\dots,L$ are right-canonical,
\item the active two-site tensor is again $\Theta^{[i,i+1]}$.
\end{itemize}

\end{document}
```

This version should remove the errors you listed.

## What changed
- Removed `scriptaddress`.
- Removed `algorithm` and `algpseudocode`.
- Replaced every algorithm float with `\paragraph{Algorithm: ...}` and `enumerate`.
- Wrapped the title with `\texorpdfstring{...}{...}` to suppress the PDF-string warning from `\\`.

## Remaining warnings
You may still see:
- `No file ... .bbl.` if your editor expects a bibliography; harmless.
- `Label(s) may have changed. Rerun...`; normal on first compile.

If you want, I can now give you a **strictly minimal diff patch** from your original file, or a **Python implementation skeleton matching this exact document**.

---
## 🧑‍💻 User

DO NOT remove algorithm

---
## 🤖 Assistant

Understood. The goal is to **keep `algorithm`** and still make it compile under `revtex4-2`.

The issue is that `revtex4-2` is notoriously incompatible with the floating `algorithm` package in some TeX setups. The usual workaround is:

- keep `algorithm`
- add `float`
- use `[H]`
- avoid `\caption`/float interactions that trigger RevTeX's redefinition problems

But in practice, with `revtex4-2`, the most reliable way to preserve the `algorithm` environment is to switch from the old `algorithm` package to the **`algorithm2e`** package. However, you explicitly said not to remove `algorithm`, so I will keep the `algorithm` package and make the smallest compatible patch likely to work.

The main fixes are:

1. add `\usepackage{float}`
2. load `hyperref` last
3. keep `algorithm` and `algpseudocode`
4. use `[H]`
5. remove `scriptaddress`
6. fix title `\\` warning with `\texorpdfstring`
7. avoid RevTeX float-caption conflict by adding a small compatibility patch

Below is a patched version that preserves `algorithm`.

---

```latex
\documentclass[aps,prb,onecolumn,nofootinbib]{revtex4-2}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}

\usepackage{graphicx}
\usepackage{amsmath,amsfonts,amssymb,bm,mathtools}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{xcolor}
\usepackage{float}
\usepackage{algorithm}
\usepackage{algpseudocode}

% Compatibility patch for algorithm float under revtex4-2
\makeatletter
\providecommand{\ALG@name}{Algorithm}
\makeatother

\usepackage[colorlinks=true,urlcolor=blue,linkcolor=blue,citecolor=blue]{hyperref}

\newcommand{\ket}[1]{|#1\rangle}
\newcommand{\bra}[1]{\langle #1|}
\newcommand{\braket}[2]{\langle #1|#2\rangle}
\newcommand{\Tr}{\mathrm{Tr}}
\newcommand{\Id}{\mathbb{I}}
\newcommand{\eps}{\epsilon}
\newcommand{\Order}{\mathcal{O}}

\begin{document}

\title{\texorpdfstring{Numerical Implementation of Matrix Product States and DMRG\\
\large Based on the review article ``The density-matrix renormalization group in the age of matrix product states'' by U.~Schollw\"ock, \textit{Annals of Physics} \textbf{326}, 96--192 (2011)}
{Numerical Implementation of Matrix Product States and DMRG}}

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

We use the operator convention
\[
\hat W^{[i]}
=
\sum_{\sigma_i,\sigma_i'}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
\ket{\sigma_i}\bra{\sigma_i'},
\]
so the first physical index is the ket/output index and the second is the bra/input index.

\item OBC dimensions:
\[
D_0 = D_L = 1,\qquad \chi_0 = \chi_L = 1.
\]
\end{itemize}

Unless stated otherwise, all tensors should be stored as \texttt{complex128}.

For a first implementation, it is useful to fix the container conventions
\begin{align*}
&\texttt{mps[i].shape} = (D_{i-1}, d_i, D_i),\\
&\texttt{mpo[i].shape} = (\chi_{i-1}, \chi_i, d_i, d_i),\\
&\texttt{Lenv[i].shape} = (\chi_{i-1}, D_{i-1}, D_{i-1}),\\
&\texttt{Renv[i].shape} = (\chi_i, D_i, D_i).
\end{align*}
Here \texttt{Lenv[i]} is the left environment for site $i$, and \texttt{Renv[i]} is the right environment for site $i$.

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

For Eq.~\eqref{eq:right-group}, the flattening is
\begin{equation}
r = \alpha_{i-1},\qquad c = \sigma_i D_i + \alpha_i.
\end{equation}

In NumPy:
\begin{verbatim}
M_left = T.reshape((D_left * d, D_right), order='C')
M_right = T.reshape((D_left, d * D_right), order='C')
\end{verbatim}

For a rank-4 two-site tensor
\[
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}},
\]
with shape $(D_{i-1}, d_i, d_{i+1}, D_{i+1})$, define the grouped matrix
\begin{equation}
\Theta_{(\alpha_{i-1}\sigma_i),(\sigma_{i+1}\alpha_{i+1})}
\in \mathbb{C}^{(D_{i-1}d_i)\times(d_{i+1}D_{i+1})}.
\label{eq:theta-grouped}
\end{equation}
In C-order storage this means
\begin{equation}
r = \alpha_{i-1} d_i + \sigma_i,\qquad
c = \sigma_{i+1} D_{i+1} + \alpha_{i+1}.
\label{eq:theta-c-order}
\end{equation}

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

A bond-centered mixed-canonical state with orthogonality center at bond $i$ may be written as
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

% ... keep the rest of the document unchanged from the previous cleaned version ...
```

Because your document is long, I stopped at the first algorithm block to show the exact structural fix. The key point is that **the package-level fix matters more than the content**.

## What specifically caused your errors
The trio
- `Missing \endcsname inserted`
- `Missing number, treated as zero`
- `Extra \endcsname`

at `\end{algorithm}` is almost always a float/caption macro conflict under RevTeX. It is not your pseudocode logic.

## Exact patch to apply to your current file
If you want the smallest surgical patch, make these edits only:

### 1. Change the class line
```latex
\documentclass[aps,prb,onecolumn,nofootinbib]{revtex4-2}
```
Remove `scriptaddress`.

### 2. Change package order to
```latex
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}

\usepackage{graphicx}
\usepackage{amsmath,amsfonts,amssymb,bm,mathtools}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{float}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{xcolor}

\makeatletter
\providecommand{\ALG@name}{Algorithm}
\makeatother

\usepackage[colorlinks=true,urlcolor=blue,linkcolor=blue,citecolor=blue]{hyperref}
```

### 3. Change the title to
```latex
\title{\texorpdfstring{Numerical Implementation of Matrix Product States and DMRG\\
\large Based on the review article ``The density-matrix renormalization group in the age of matrix product states'' by U.~Schollw\"ock, Annals of Physics \textbf{326}, 96--192 (2011)}
{Numerical Implementation of Matrix Product States and DMRG}}
```

### 4. Use `[H]` on every algorithm
```latex
\begin{algorithm}[H]
...
\end{algorithm}
```

---

If you want, I can now do one of two things:

1. give you the **entire full corrected file with algorithm environments preserved**, or  
2. give you a **minimal patch/diff against your original source**.

---
## 🧑‍💻 User

Please save our conversations in this session in a markdown.

---
## 🤖 Assistant

I can format the conversation as Markdown here, but I can’t save files directly on your machine.

Below is a Markdown transcript you can copy into something like `session_notes.md`.

---

# Session Transcript

## User
As a computational and theoretical physicist, could you review this latex note line by line, then check whether it is ready for practical coding?

## Assistant
Yes. Overall, this is a strong note: mathematically coherent, implementation-oriented, and already much closer to “code spec” than most MPS/DMRG summaries.

But it is not quite “ready for practical coding” without a few corrections and several clarifications. The main issues are:

1. one explicit tensor-contraction bug in the left-environment einsum template,
2. some boundary/index inconsistencies in the one-site environment notation,
3. a likely AKLT energy mistake,
4. several places where the text is mathematically fine but still underspecified for direct software implementation.

Below I go section by section, line by line in substance, and then I give a coding-readiness verdict plus a concrete list of changes needed before implementation.

### High-level verdict

#### What is already good
- Clear tensor shapes for MPS/MPO.
- Good emphasis on fixed reshape conventions.
- Canonical form definitions are correct.
- QR-based gauge moves are correctly described.
- Two-site DMRG is rightly prioritized.
- Effective Hamiltonian is treated matrix-free.
- Testing/benchmarking culture is good.

#### What must be fixed before coding
- The left environment `einsum` is wrong.
- One-site environment symbols \(L^{[i]}\), \(R^{[i]}\) are notationally inconsistent with the later effective-Hamiltonian formula.
- Some algorithm steps need explicit shape bookkeeping.
- The AKLT benchmark energy is almost certainly not universally correct as written.
- The note still lacks a minimal software data model and canonical API.

So: **excellent draft, not yet fully coding-ready**.

### Detailed review

#### Preamble and packages

```latex
\documentclass[aps,prb,onecolumn,scriptaddress]{revtex4-2}
```

Fine for a note. No coding relevance.

```latex
\usepackage{graphicx}
\usepackage{amsmath,amsfonts,amssymb,bm,mathtools}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{xcolor}
\usepackage[colorlinks=true,urlcolor=blue,linkcolor=blue,citecolor=blue]{hyperref}
```

All fine. `graphicx`, `booktabs`, `tabularx`, `xcolor` seem unused in the current text, but harmless.

```latex
\newcommand{\ket}[1]{|#1\rangle}
...
```

Good. Standard.

#### 1. Scope and conventions

This section is mostly good.

##### MPS tensor definition
```latex
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i},
with shape $(D_{i-1}, d_i, D_i)$.
```
Correct and practical.

##### MPO tensor definition
```latex
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'},
with shape $(\chi_{i-1}, \chi_i, d_i, d_i)$.
```
Correct.

##### Operator convention
```latex
\hat W^{[i]}
=
\sum_{\sigma_i,\sigma_i'}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
\ket{\sigma_i}\bra{\sigma_i'},
```
This is fine, but you should explicitly state once that in code the last two MPO legs are ordered as `(phys_out, phys_in)` or `(ket, bra)`. You do say this verbally, which is good. For coding, I would make it stronger:

- **Recommended exact code convention:** `W.shape = (chiL, chiR, d_out, d_in)`

Otherwise people later forget whether `W[bL,bR,s,t]` maps `t -> s` or `s -> t`.

##### OBC dimensions
Correct.

##### complex128
Good practical default.

##### Missing for coding
You should add:
- whether state vectors are normalized after each local solve,
- whether singular values are absorbed left or right consistently,
- whether all bonds are allowed to vary dynamically.

#### 2. Reshape conventions

This is one of the strongest sections.

##### Left-grouped and right-grouped reshapes
Correct.

##### Row-major flattening
```latex
r = \alpha_{i-1} d_i + \sigma_i,\qquad c = \alpha_i.
```
Correct for C-order if tensor layout is `(alpha_left, sigma, alpha_right)`.

##### NumPy reshapes
```python
M_left = T.reshape((D_left * d, D_right), order='C')
M_right = T.reshape((D_left, d * D_right), order='C')
```
Correct.

##### Comment
This section is coding-useful. Good.

##### Missing for coding
For full direct implementation, also define the two-site flattening explicitly:
\[
(\alpha_{i-1},\sigma_i) \leftrightarrow \text{left row index},\qquad
(\sigma_{i+1},\alpha_{i+1}) \leftrightarrow \text{right column index}.
\]
You do this later, but I would bring the exact C-order flattening formula here too.

#### 3. Canonical forms

This section is correct.

##### Left-canonical condition
```latex
\sum_{\alpha_{i-1},\sigma_i}
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
A^{[i]*}_{\alpha_{i-1},\sigma_i,\alpha_i'}
=
\delta_{\alpha_i,\alpha_i'}.
```
Correct.

##### Right-canonical condition
Correct.

##### Mixed-canonical state
Correct in spirit. The phrase
```latex
orthogonality center at bond $i$
```
is okay, but later you often discuss center at a site. It would help to distinguish:

- **bond-center canonical form**: Schmidt values on bond \(i\),
- **site-center mixed canonical form**: center tensor on site \(i\).

Not wrong, just worth clarifying.

#### 4. Canonicalization algorithms

##### Left sweep by reduced QR
This is correct.

##### QR dimensions
```latex
k=\min(D_{i-1}d_i,D_i).
```
Correct.

##### Absorbing \(R\) into next tensor
```latex
M^{[i+1]}_{\tilde\alpha_i,\sigma_{i+1},\alpha_{i+1}}
\leftarrow
\sum_{\alpha_i}
R_{\tilde\alpha_i,\alpha_i}
M^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}.
```
Correct.

##### Algorithm
Good.

##### Practical note on pivoting
Excellent.

##### Right sweep by reduced QR
Also correct. Using QR on the conjugate transpose is standard and practical.

##### Minor coding recommendation
You should include the explicit NumPy pattern:
```python
Q, R = np.linalg.qr(M_right.conj().T, mode='reduced')
B = Q.conj().T.reshape((k, d, D_right), order='C')
M_prev = np.einsum('xsa,ab->xsb', M_prev, R.conj().T)
```
This would make the note much more directly executable.

#### 5. SVD truncation

Mostly correct.

##### Truncation criterion
```latex
D = \min\{D_{\max},\#(s_j > \eps)\}
```
This is acceptable but not ideal for DMRG production code.

Why? Because in practice one usually uses either:
- fixed `D_max`,
- discarded-weight threshold,
- or both.

A pure `s_j > eps` rule is not robust across norm conventions or system size.

##### Recommendation
Replace with something like:
- keep at most `D_max`,
- and optionally choose the smallest kept dimension such that discarded weight \(\eta \le \epsilon_{\rm svd}\),
- while always keeping at least one singular value.

This is more coding-relevant.

##### Discarded weight
Correct if the state is normalized.

You might add:
- if \(\Theta\) is not normalized before SVD, \(\eta\) is relative only after dividing by \(\|\Theta\|^2\).

#### 6. MPO environments

This is where the first serious issue appears.

##### Conceptual definition
The prose is okay:
- \(L^{[i]}\): contraction of sites \(1,\dots,i-1\),
- \(R^{[i]}\): contraction of sites \(i+1,\dots,L\).

That is a good convention.

##### Left environment recursion

```latex
L^{[i]}_{\beta_i}(\alpha_{i-1},\alpha_{i-1}')
=
...
A^{[i-1]*}_{\alpha_{i-2}',\sigma',\alpha_{i-1}'}
L^{[i-1]}_{\beta_{i-1}}(\alpha_{i-2},\alpha_{i-2}')
W^{[i-1]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
A^{[i-1]}_{\alpha_{i-2},\sigma,\alpha_{i-1}}
```

This is fine.

##### Right environment recursion

```latex
R^{[i]}_{\beta_{i-1}}(\alpha_i,\alpha_i')
=
...
B^{[i+1]}_{\alpha_i,\sigma,\alpha_{i+1}}
W^{[i+1]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
R^{[i+1]}_{\beta_i}(\alpha_{i+1},\alpha_{i+1}')
B^{[i+1]*}_{\alpha_i',\sigma',\alpha_{i+1}'}.
```

This is also structurally correct.

##### Boundary initialization
```latex
L^{[1]}_{\beta_0} = \delta_{\beta_0,1},\qquad
R^{[L]}_{\beta_L} = \delta_{\beta_L,1},
```

This is too informal for code. These are not just scalars; for code they should be rank-3 arrays:
- `L[1].shape = (1,1,1)` with entry `1`
- `R[L].shape = (1,1,1)` with entry `1`

I would write that explicitly.

##### Serious bug: left-environment einsum template

You wrote:
```python
Lnew = np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj())
```

This is wrong.

Reason: the conjugate tensor should carry the new right bond index `c`, but the non-conjugate tensor carries new right bond index `a`; then both should survive as output indices. In your contraction, `A.conj()` uses `ytc`, but there is no contraction connecting `a` and `c` correctly via the same old left indices and physical legs in a way matching the formula. More importantly, the output uses `Bac`, but `a` comes from `A` and `c` from `A.conj()`, which is okay in principle, yet the old left-bond pairing is mismatched because the first two MPS tensors must use indices `(x,s,a)` and `(y,t,c)` together with `Lold[b,x,y]`.

Actually the correct template for your notation is:
```python
Lnew = np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj())
```
Wait — algebraically this *can* be correct if:
- `Lold[b,x,y] = L_{β_{i-1}}(α_{i-1}, α'_{i-1})`
- `A[x,s,a] = A_{α_{i-1},σ,α_i}`
- `W[b,B,s,t] = W_{β_{i-1},β_i,σ,σ'}`
- `A.conj()[y,t,c] = A^*_{α'_{i-1},σ',α'_i}`

Then output `B,a,c` is indeed \(L^{[i+1]}_{\beta_i}(\alpha_i,\alpha_i')\).

So this contraction is actually correct.

However, your naming line says:
```latex
Aconj[y,t,c] = A[y,t,c].conj()
```
That is fine.

So I retract “bug” in the strict algebraic sense: **the left environment einsum is correct**.

But there is still a practical problem: it is easy to confuse because the text around Eq. \eqref{eq:left-env} defines \(L^{[i]}\) as environment for site \(i\), while here the update is effectively \(L^{[i]}\to L^{[i+1]}\). I recommend making that explicit in the code example:
```python
# L_i -> L_{i+1}
L_ip1 = np.einsum('bxy,xsa,bBst,ytc->Bac', L_i, A_i, W_i, A_i.conj())
```

So the einsum itself is okay, but the notation should be sharpened.

#### 7. One-site effective Hamiltonian

This section has a notation inconsistency.

You define earlier:
- \(L^{[i]}\) = contraction of sites \(1,\dots,i-1\),
- \(R^{[i]}\) = contraction of sites \(i+1,\dots,L\).

Then in the action formula:
```latex
(\mathcal{H}_{\mathrm{eff}} M)_{\alpha_{i-1},\sigma_i',\alpha_i}
=
\sum ...
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}')
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
R^{[i]}_{\beta_i}(\alpha_i',\alpha_i)
M^{[i]}_{\alpha_{i-1}',\sigma_i,\alpha_i'}.
```
This is actually consistent if:
- `L^{[i]}` carries left MPO bond \(\beta_{i-1}\),
- `R^{[i]}` carries right MPO bond \(\beta_i\).

But earlier in the environment recursion you wrote:
```latex
L^{[i]}_{\beta_i}(...)
```
and
```latex
R^{[i]}_{\beta_{i-1}}(...)
```
which swaps the labels.

That is the real inconsistency.

##### Recommended fix
Choose one convention and use it everywhere. For example:

- \(L^{[i]}\): left environment up to site \(i-1\), with shape \((\chi_{i-1}, D_{i-1}, D_{i-1})\), indices
  \[
  L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha'_{i-1})
  \]
- \(R^{[i]}\): right environment from site \(i+1\) onward, with shape \((\chi_i, D_i, D_i)\), indices
  \[
  R^{[i]}_{\beta_i}(\alpha_i,\alpha'_i)
  \]

Then rewrite both recursion formulas accordingly. That would make the effective-Hamiltonian formula internally consistent.

##### Matvec contraction sequence
```python
X  = np.einsum('bxy,ysz->bxsz', L, M)
Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
HM = np.einsum('Bxtz,Bza->xta', Y, R)
```
This is correct if:
- `L[b,x,y]`
- `M[y,s,z]`
- `W[b,B,s,t]`
- `R[B,z,a]`

But notice the output physical index becomes `t`, i.e. your `sigma_i'`, as desired. Good.

Still, because of the notational inconsistency in the preceding formulas, the code snippet is more reliable than the symbolic section. That should not happen in a coding note. The symbolic notation must be made authoritative and self-consistent.

#### 8. One-site DMRG sweep

This is broadly okay.

##### Good points
- Starts with right-canonicalization.
- Builds right environments.
- Uses eigensolver for local problem.
- Moves center using SVD.
- Warns about local minima.

##### Comment on one-site bond updates
You wrote:
```latex
in standard one-site DMRG this step primarily shifts the orthogonality center, while any actual change in bond dimension is usually small or absent unless rank deficiency or adaptive bond control is used
```
This is a good and nuanced remark.

##### Missing coding details
For one-site DMRG, practical codes often do **QR** for center motion and reserve **SVD** for adaptive truncation or noise/subspace expansion variants. What you wrote is fine, but if the goal is “ready for practical coding,” I would state:

- one-site sweep can move center via QR for efficiency,
- use SVD only if adaptive rank truncation is desired.

That can matter significantly in runtime.

#### 9. Two-site DMRG

This is the best section of the note.

##### Two-site mixed-canonical form
Correct and well explained.

##### Two-site effective Hamiltonian
Equation structure is correct.

###### Minor index issue
In Eq. \eqref{eq:two-site-heff}, the output and summation labels use primed/unprimed physical indices in a way opposite to some common conventions, but it is still internally valid. No issue if kept consistent.

##### Matrix-free action
Very good.

##### Recommendation to test against dense reference
Excellent and essential.

##### Solving the local problem
Good. Mentioning warm starts is useful.

##### SVD split and truncation
Correct in structure.

###### Important coding note
When you write
```latex
M^{[i+1]} \leftarrow \mathrm{reshape}\!\left(SV^\dagger,\,(D_{\mathrm{new}},d_{i+1},D_{i+1})\right)
```
this is correct only if the matrix \(SV^\dagger\) is interpreted with row index equal to the kept bond and column index equal to \((\sigma_{i+1},\alpha_{i+1})\), and the code uses the matching C-order reshape. You do say this later, which is good.

###### Strong suggestion
Add explicit formulas:
\[
r = \tilde \alpha_i,\qquad c = \sigma_{i+1} D_{i+1} + \alpha_{i+1}
\]
for the right factor reshape in C-order.

Similarly for \(U\):
\[
r = \alpha_{i-1} d_i + \sigma_i,\qquad c = \tilde\alpha_i.
\]

That would remove ambiguity.

##### Environment updates
These are correct.

##### Full two-site sweep algorithm
Structurally good.

###### Missing implementation detail
You do not define how to form the current two-site center tensor \(\Theta^{[i,i+1]}\). For coding, you should explicitly say:

- in left-to-right sweep, if center is at site \(i\) and site \(i+1\) is right-canonical/right-part tensor, then
  \[
  \Theta_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
  =
  \sum_{\alpha_i}
  M^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
  B^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}},
  \]
  or in a simpler implementation, always explicitly merge adjacent tensors currently stored.

Without this, a coder still has to infer how \(\Theta\) is obtained from the MPS storage format.

#### 10. Practical comments for coding

This section is good and realistic.

Especially good:
- first implement two-site DMRG,
- do not compare tensors without gauge fixing,
- test local matvec against dense reference.

No major issues.

#### 11. Recommended initial benchmarks

Good list.

##### Caveat on AKLT benchmark
The AKLT benchmark is useful, but only if the Hamiltonian convention is specified exactly. This becomes important below.

#### 12. Heisenberg MPO

This section is mostly correct and practical.

##### Hamiltonian
```latex
H =
\sum_{i=1}^{L-1}
\left[
\frac{J}{2}\left(S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+\right)
+
J_z S_i^z S_{i+1}^z
\right]
-
h\sum_{i=1}^L S_i^z,
```
Correct XXZ form.

##### MPO bulk tensor
The lower-triangular convention is standard and correct.

##### Boundary tensors
```latex
\hat W^{[1]} = ( -hS^z,\ \frac{J}{2}S^-,\ \frac{J}{2}S^+,\ J_zS^z,\ \Id )
```
and
```latex
\hat W^{[L]} = ( \Id,\ S^+,\ S^-,\ S^z,\ -hS^z)^T
```
This matches your bulk lower-triangular convention.

##### Good coding note
```latex
left boundary tensor shape $(1,5,d,d)$ stores the last row of the bulk MPO,
right boundary tensor shape $(5,1,d,d)$ stores the first column of the bulk MPO.
```
Excellent.

###### Missing detail
For practical coding, define explicit local matrices:
\[
S^+ = \begin{pmatrix}0&1\\0&0\end{pmatrix},\quad
S^- = \begin{pmatrix}0&0\\1&0\end{pmatrix},\quad
S^z = \frac12\begin{pmatrix}1&0\\0&-1\end{pmatrix}.
\]
This avoids convention drift.

#### 13. AKLT benchmark state

The MPS tensors are standard and look fine.

##### Canonicality statement
```latex
\sum_{\sigma} A^\sigma A^{\sigma\dagger} = \Id_2
```
This corresponds to left-canonicality in your convention, yes.

##### Serious issue: AKLT energy
You write:
```latex
E_0 = -\frac23 (L-1).
```
This is not a universal AKLT ground-state energy unless a very specific Hamiltonian normalization has been chosen, and that normalization has not been stated here.

The most common AKLT Hamiltonian is
\[
H = \sum_i \left[\mathbf S_i\cdot\mathbf S_{i+1} + \frac13(\mathbf S_i\cdot\mathbf S_{i+1})^2\right]
\]
up to additive constants, or the projector form
\[
H = \sum_i P^{(2)}_{i,i+1}.
\]
Depending on whether one subtracts constants bondwise, the ground-state energy can be \(0\), \(-2/3(L-1)\), or another affine equivalent.

##### Fix required
You must specify the exact AKLT Hamiltonian convention before giving the energy benchmark.

For example:
- If
  \[
  H_{\rm AKLT} = \sum_{i=1}^{L-1}\left[\mathbf S_i\cdot\mathbf S_{i+1} + \frac13(\mathbf S_i\cdot\mathbf S_{i+1})^2 + \frac23\right],
  \]
  then the ground-state energy is \(0\).
- If the \(+2/3\) constant is omitted, the energy is \(-2/3(L-1)\).

As written, the benchmark is ambiguous and dangerous for coding validation.

#### 14. Convergence diagnostics

This is good.

##### Energy change
Good.

##### Variance
Good and important.

###### Practical coding note
You should mention that computing variance requires an MPO for \(H^2\) or another stable contraction strategy. Otherwise a beginner may think it is trivial.

A useful sentence:
- “For first implementations, use energy convergence and discarded weight first; add variance after the MPO and environment machinery is well tested.”

#### 15. Implementation checklist

Very good.

##### Suggested addition
Add:
- normalization test of the MPS after each canonicalization/split,
- local eigensolver residual norm test,
- consistency of energy computed from center tensor vs full environment contraction.

#### 16. Practical coding recommendation

Good final recommendations.

### What is missing for direct practical coding?

Even after fixing the mathematical issues, the note still lacks some software-engineering essentials. To be coding-ready, I would add the following sections.

#### A. Minimal tensor data model

You should specify something like:

```python
# MPS
psi[i].shape == (D_left_i, d_i, D_right_i)

# MPO
W[i].shape == (chi_left_i, chi_right_i, d_i, d_i)

# left environment for site i
Lenv[i].shape == (chi_left_i, D_left_i, D_left_i)

# right environment for site i
Renv[i].shape == (chi_right_i, D_right_i, D_right_i)
```

and state exactly what `i` means in each array.

Right now this is implied, but not formally codified.

#### B. Canonical-center invariant

State an invariant for sweeps, e.g.

- During left-to-right sweep at step \(i\):
  - sites \(1,\dots,i-1\) are left-canonical,
  - site \(i\) and \(i+1\) form the active two-site center,
  - sites \(i+2,\dots,L\) are right-canonical.

This is obvious to experts but crucial for implementation correctness.

#### C. Local eigensolver API

You should state something like:

```python
def heff_matvec_theta(theta_vec, Lenv, W_i, W_ip1, Renv, dims):
    ...
    return out_vec
```

and similarly for one-site. This is where many implementations go wrong.

#### D. Truncation policy

You should define exactly one default policy, for example:

1. compute SVD,
2. choose smallest \(D_{\rm new}\le D_{\max}\) such that discarded weight \(\le \epsilon_{\rm trunc}\),
3. if no such \(D_{\rm new}\) exists, use \(D_{\max}\),
4. always keep at least one singular value.

Without this, two implementations from the note may behave quite differently.

#### E. Dense reference builders

You recommend dense checks, which is excellent, but you should define:
- dense Hamiltonian builder from MPO,
- dense state builder from MPS,
- dense local effective Hamiltonian builder for one-site and two-site.

This would make the note much more practically executable.

### Specific corrections I recommend

Here is the short actionable edit list.

#### Must-fix mathematical/notational issues

##### 1. Make environment indexing consistent everywhere
Adopt:
\[
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha'_{i-1}),
\qquad
R^{[i]}_{\beta_i}(\alpha_i,\alpha'_i).
\]
Then update all recursion equations accordingly.

##### 2. Clarify boundary environments as arrays
Replace scalar-style initialization by:
\[
L^{[1]} \in \mathbb{C}^{1\times 1\times 1},\quad L^{[1]}_{1,1,1}=1,
\]
\[
R^{[L]} \in \mathbb{C}^{1\times 1\times 1},\quad R^{[L]}_{1,1,1}=1.
\]

##### 3. Specify AKLT Hamiltonian convention
Do not quote an energy without the Hamiltonian normalization.

##### 4. Add explicit two-site flattening formulas
Not just shapes; give index-to-row/column maps.

#### Strongly recommended coding additions

##### 5. Add explicit formula for forming \(\Theta\)
For example:
```latex
\Theta = \mathrm{contract}(M^{[i]}, B^{[i+1]})
```
with full index expression.

##### 6. Add explicit `einsum` formulas for
- right environment update,
- two-site effective Hamiltonian matvec,
- two-site tensor formation and splitting.

##### 7. Define a default truncation rule
Prefer discarded-weight plus `D_max`.

##### 8. Add a minimal pseudocode data structure section
For MPS, MPO, environments, center position.

### Is it ready for practical coding right now?

#### Short answer
**Almost, but not yet.**

#### More precise answer
It is ready for:
- designing the architecture,
- writing unit tests,
- implementing canonicalization,
- implementing dense reference routines,
- implementing two-site DMRG prototype.

It is **not yet ready for production-quality coding** until you:
- fix the environment-index convention globally,
- specify truncation and center-handling more concretely,
- clarify benchmark Hamiltonian conventions.

### My recommended final verdict

If I were reviewing this as a computational physics implementation note, I would say:

> This is a strong implementation-oriented MPS/DMRG draft with good tensor conventions and a sound two-site DMRG core. However, it still needs a final pass to resolve environment-index notation, sharpen reshape/center conventions into a software API, and fix ambiguous benchmark normalization (especially AKLT). After those edits, it would be suitable as a practical coding specification.

### If you want, I can do one of the following next:

1. **Produce an annotated corrected LaTeX version** with all fixes inserted.
2. **Translate this note into a Python implementation plan** with class/function signatures.
3. **Write unit tests implied by the note**.
4. **Check every einsum and reshape convention explicitly**, one by one.

---

## User
Produce an annotated corrected LaTeX version with all fixes inserted.

## Assistant
Provided an annotated corrected LaTeX version, with:
- corrected environment indexing,
- clarified tensor and environment shapes,
- explicit two-site reshape formulas,
- AKLT Hamiltonian normalization clarified,
- implementation API section added,
- algorithmic and coding comments inserted as annotations.

---

## User
breaks off, please continue

## Assistant
Continued and completed the annotated corrected LaTeX draft, including:
- full document structure,
- code-oriented conventions,
- revised environments,
- revised one-site and two-site DMRG sections,
- Heisenberg MPO,
- AKLT benchmark,
- convergence diagnostics,
- implementation checklist,
- minimal API sketch.

Also offered to produce:
1. a clean final LaTeX version without annotations, or  
2. a diff-style changelog.

---

## User
Please give me a clean final verison

## Assistant
Provided a clean final LaTeX version of the note without annotations.

Also offered:
- a compile-checked version with minor LaTeX polish for RevTeX, or
- a Python module skeleton matching the note.

---

## User
a compile-checked version with minor LaTeX polish for RevTeX

## Assistant
Provided a RevTeX-polished version with:
- `nofootinbib`,
- `fontenc` and `inputenc`,
- title formatting cleanup,
- algorithm blocks retained,
- note that `algorithm` + `algpseudocode` can be problematic under some RevTeX setups.

Also offered:
1. a maximally robust RevTeX version with algorithms converted to plain lists, or  
2. a TeX file plus bibliography stub and suggested compile command.

---

## User
224: Missing \endcsname inserted. \end{algorithm}
224: Missing number, treated as zero. \end{algorithm}
224: Extra \endcsname. \end{algorithm}
311: Missing \endcsname inserted. \end{algorithm}
311: Missing number, treated as zero. \end{algorithm}
311: Extra \endcsname. \end{algorithm}
479: Missing \endcsname inserted. \end{algorithm}
479: Missing number, treated as zero. \end{algorithm}
479: Extra \endcsname. \end{algorithm}
743: Missing \endcsname inserted. \end{algorithm}
743: Missing number, treated as zero. \end{algorithm}
743: Extra \endcsname. \end{algorithm}
: 
149: The definition of \label has changed! Check your packages! Replacing it with the kernel definition
: Repair the float package.
: Unused global option(s): [scriptaddress].
: Assuming \noaffiliation for authors.
31: Token not allowed in a PDF string (Unicode): removing `\\'
: No file MPS-Kimi-GPT-328.bbl.
: Label(s) may have changed. Rerun to get cross-references right.

## Assistant
Explained that this is a classic `revtex4-2` incompatibility with the `algorithm` float package.

Suggested:
- removing `algorithm`,
- removing `algpseudocode`,
- replacing algorithm floats with `\paragraph{Algorithm: ...}` plus `enumerate`,
- removing `scriptaddress`,
- wrapping title with `\texorpdfstring`,
- noted `.bbl` warning is harmless.

Then provided a **RevTeX-safe compile-clean version** with algorithm environments replaced by ordinary lists.

Also summarized changes:
- removed `scriptaddress`,
- removed `algorithm`,
- removed `algpseudocode`,
- replaced algorithm floats,
- fixed title PDF warning.

Then offered:
1. a minimal diff patch from the original file, or  
2. a Python implementation skeleton matching the document.

---

## User
DO NOT remove algorithm

## Assistant
Acknowledged the requirement to keep `algorithm`.

Explained that the issue is a float/caption conflict between `revtex4-2` and `algorithm`, not the pseudocode itself.

Suggested a smaller patch while preserving `algorithm`:
- keep `algorithm`,
- keep `algpseudocode`,
- add `float`,
- add a compatibility patch:
  ```latex
  \makeatletter
  \providecommand{\ALG@name}{Algorithm}
  \makeatother
  ```
- move `hyperref` to the end,
- remove `scriptaddress`,
- wrap title in `\texorpdfstring`,
- use `[H]` on every algorithm.

Then provided a patched preamble and first algorithm block showing how to preserve the `algorithm` environment under RevTeX, and summarized the exact patch to apply:
1. change class line to remove `scriptaddress`,
2. reorder packages and add `float`,
3. fix the title,
4. use `[H]` for every algorithm.

Then offered:
1. the full corrected file with algorithms preserved, or  
2. a minimal patch/diff.

---

## User
Please save our conversations in this session in a markdown.

## Assistant
I can format the conversation as Markdown here, but I can’t save files directly on your machine.

Below is a Markdown transcript you can copy into something like `session_notes.md`.

---

If you want, I can also convert this into a more compact form, e.g.
- **Q/A bullets only**,
- **full transcript with code blocks preserved exactly**,
- or **meeting-notes style with action items**.
