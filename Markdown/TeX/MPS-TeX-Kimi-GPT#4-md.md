# MPS-TeX-Kimi-GPT#4-md

## 🧑‍💻 User

Could you find where is the MPO fomula for AKLT in this review article?

---
## 🤖 Assistant

Yes. In Schollwöck’s review, the **AKLT formula is given as an MPS, not as an MPO**.

### Where to find it
Look in:

- **Section 4.1.5 — “The AKLT state as a matrix product state”**
- around **pages 119–120** of the Annals of Physics version you pasted.

### Key equations there
The AKLT construction is written explicitly starting with the Hamiltonian

- **Eq. (76)**

then the spin-1 basis and singlet construction

- **Eqs. (77)–(81)**

and finally the MPS form

- **Eq. (86)**:  
  \[
  |\psi\rangle=\sum_{\sigma}\mathrm{Tr}\!\left(M^{\sigma_1}\Sigma M^{\sigma_2}\Sigma\cdots M^{\sigma_L}\Sigma\right)|\sigma\rangle
  \]

then simplified to

- **Eq. (88)**:  
  \[
  |\psi\rangle=\sum_{\sigma}\mathrm{Tr}\!\left(\tilde A^{\sigma_1}\tilde A^{\sigma_2}\cdots \tilde A^{\sigma_L}\right)|\sigma\rangle
  \]

with normalized matrices finally given in

- **Eq. (89)**:
  \[
  A^{+}=\begin{pmatrix}0&\sqrt{2/3}\\0&0\end{pmatrix},\quad
  A^{0}=\begin{pmatrix}-1/\sqrt3&0\\0&1/\sqrt3\end{pmatrix},\quad
  A^{-}=\begin{pmatrix}0&0\\-\sqrt{2/3}&0\end{pmatrix}.
  \]

### Important clarification
If you were specifically looking for an **MPO formula for the AKLT Hamiltonian**, that is **not given explicitly** as a special AKLT example. The review gives:

- the **AKLT state as an MPS** in **Section 4.1.5**
- the **general MPO formalism** later in **Section 5**
- and an explicit MPO example mainly for the **Heisenberg Hamiltonian**, around **Eq. (182)** onward.

So:

- **AKLT state as MPS** → Section 4.1.5, Eqs. (76)–(89)
- **general MPO construction** → Section 5
- **AKLT-specific MPO** → not explicitly singled out in the review

If you want, I can also help you **construct the AKLT Hamiltonian as an MPO** from the general rules in Section 5.

---
## 🧑‍💻 User

Great. This is a latex note generated from the review article. Please go through it and check whether it construct the AKLT Hamiltonian as an MPO. If not, please do it and add to this note.

---
## 🤖 Assistant

I checked the note: it **does not** construct the **AKLT Hamiltonian as an MPO**. It only gives:

- the generic **Heisenberg/XXZ MPO** in the section *Heisenberg MPO*,
- the **AKLT state as an exact MPS** in *AKLT benchmark state*.

So yes, something is missing if you want the **AKLT Hamiltonian MPO** explicitly in the note.

---

## What should be added

For the spin-1 AKLT chain,
\[
H_{\mathrm{AKLT}}
=
\sum_{i=1}^{L-1}
\left[
\mathbf S_i\cdot \mathbf S_{i+1}
+\frac13 \bigl(\mathbf S_i\cdot \mathbf S_{i+1}\bigr)^2
\right].
\]

A convenient MPO is obtained by using the standard “finite-state automaton” construction for nearest-neighbor Hamiltonians.

Define the two-site operator
\[
h_{i,i+1}
=
\mathbf S_i\cdot \mathbf S_{i+1}
+\frac13 \bigl(\mathbf S_i\cdot \mathbf S_{i+1}\bigr)^2.
\]
Write
\[
X \equiv \mathbf S_i\cdot \mathbf S_{i+1}
=
\frac12(S_i^+S_{i+1}^-+S_i^-S_{i+1}^+)+S_i^zS_{i+1}^z.
\]
Then
\[
X^2
=
\sum_{a,b} A_i^{(a)}A_i^{(b)} \otimes B_{i+1}^{(a)}B_{i+1}^{(b)},
\]
if
\[
X=\sum_a A_i^{(a)}\otimes B_{i+1}^{(a)}.
\]

A compact choice is to use the operator list
\[
\{O_a\}
=
\left\{
\frac{1}{\sqrt2}S^+,\,
\frac{1}{\sqrt2}S^-,\,
S^z
\right\},
\]
so that
\[
X=\sum_{a=1}^3 O_a \otimes O_a^\dagger,
\]
with
\[
O_1^\dagger=\frac{1}{\sqrt2}S^-,
\qquad
O_2^\dagger=\frac{1}{\sqrt2}S^+,
\qquad
O_3^\dagger=S^z.
\]

Then the AKLT MPO can be written with bond dimension
\[
\chi = 1+3+3+1 = 8.
\]

---

## LaTeX block to insert

I suggest inserting the following **new section** right after your current section **“Heisenberg MPO”** and before **“MPO environments”**.

```latex
\section{AKLT Hamiltonian as an MPO}

For the spin-1 AKLT chain,
\begin{equation}
	H_{\mathrm{AKLT}}
	=
	\sum_{i=1}^{L-1}
	\left[
	\mathbf S_i\cdot \mathbf S_{i+1}
	+\frac13 \bigl(\mathbf S_i\cdot \mathbf S_{i+1}\bigr)^2
	\right],
	\label{eq:aklt-ham}
\end{equation}
with
\begin{equation}
	\mathbf S_i\cdot \mathbf S_{i+1}
	=
	\frac12\left(S_i^+S_{i+1}^- + S_i^-S_{i+1}^+\right)
	+
	S_i^z S_{i+1}^z.
	\label{eq:aklt-bilinear}
\end{equation}

A convenient MPO construction uses the operator list
\begin{equation}
	O_1 = \frac{1}{\sqrt2}S^+,\qquad
	O_2 = \frac{1}{\sqrt2}S^-,\qquad
	O_3 = S^z,
\end{equation}
and
\begin{equation}
	\bar O_1 = \frac{1}{\sqrt2}S^-,\qquad
	\bar O_2 = \frac{1}{\sqrt2}S^+,\qquad
	\bar O_3 = S^z.
\end{equation}
Then
\begin{equation}
	\mathbf S_i\cdot \mathbf S_{i+1}
	=
	\sum_{a=1}^3 O_a^{[i]} \bar O_a^{[i+1]}.
\end{equation}
Therefore
\begin{equation}
	\bigl(\mathbf S_i\cdot \mathbf S_{i+1}\bigr)^2
	=
	\sum_{a=1}^3\sum_{b=1}^3
	\left(O_a O_b\right)^{[i]}
	\left(\bar O_a \bar O_b\right)^{[i+1]}.
\end{equation}

Hence the two-site AKLT interaction can be written as
\begin{equation}
	h_{i,i+1}
	=
	\sum_{a=1}^3 O_a^{[i]} \bar O_a^{[i+1]}
	+
	\frac13
	\sum_{a=1}^3\sum_{b=1}^3
	\left(O_a O_b\right)^{[i]}
	\left(\bar O_a \bar O_b\right)^{[i+1]}.
	\label{eq:aklt-local-factorized}
\end{equation}

This gives an exact nearest-neighbor MPO with bond dimension
\[
\chi = 1 + 3 + 3 + 1 = 8.
\]

Define the row vector of local operators
\begin{equation}
	\bm O =
	\begin{pmatrix}
		O_1 & O_2 & O_3
	\end{pmatrix},
\end{equation}
the column vector
\begin{equation}
	\bar{\bm O} =
	\begin{pmatrix}
		\bar O_1 \\ \bar O_2 \\ \bar O_3
	\end{pmatrix},
\end{equation}
and the $3\times 3$ operator-valued matrix
\begin{equation}
	Q_{ab} = O_a O_b,
	\qquad
	\bar Q_{ab} = \bar O_a \bar O_b.
\end{equation}

Then an operator-valued bulk MPO tensor is
\begin{equation}
	\hat W_{\mathrm{AKLT,bulk}}
	=
	\begin{pmatrix}
		\Id & 0 & 0 & 0 \\
		\bar{\bm O} & 0 & 0 & 0 \\
		\bar Q & 0 & 0 & 0 \\
		0 & \bm O & \frac13 \bm O^{(2)} & \Id
	\end{pmatrix},
	\label{eq:aklt-mpo-block}
\end{equation}
where the block sizes are $1|3|3|1$, and
\begin{equation}
	\bm O^{(2)} =
	\begin{pmatrix}
		O_1 & O_2 & O_3
	\end{pmatrix}.
\end{equation}

Written explicitly as an $8\times 8$ operator-valued matrix, this is
\begin{equation}
	\hat W_{\mathrm{AKLT,bulk}}
	=
	\begin{pmatrix}
		\Id & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
		\bar O_1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
		\bar O_2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
		\bar O_3 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
		\bar O_1\bar O_1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
		\bar O_2\bar O_2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
		\bar O_3\bar O_3 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
		0 & O_1 & O_2 & O_3 & \frac13 O_1 & \frac13 O_2 & \frac13 O_3 & \Id
	\end{pmatrix},
	\label{eq:aklt-mpo-explicit-bad}
\end{equation}
but for actual implementation one should instead use the block form
Eq.~\eqref{eq:aklt-mpo-block} together with the full $3\times3$ matrix $\bar Q$, i.e.
\begin{equation}
	\bar Q =
	\begin{pmatrix}
		\bar O_1\bar O_1 & \bar O_1\bar O_2 & \bar O_1\bar O_3 \\
		\bar O_2\bar O_1 & \bar O_2\bar O_2 & \bar O_2\bar O_3 \\
		\bar O_3\bar O_1 & \bar O_3\bar O_2 & \bar O_3\bar O_3
	\end{pmatrix}.
\end{equation}

Equivalently, the clean implementation form is
\begin{equation}
	\hat W_{\mathrm{AKLT,bulk}}
	=
	\begin{pmatrix}
		\Id & 0_{1\times 3} & 0_{1\times 3} & 0 \\
		\bar{\bm O} & 0_{3\times 3} & 0_{3\times 3} & 0_{3\times 1} \\
		\bar Q & 0_{3\times 3} & 0_{3\times 3} & 0_{3\times 1} \\
		0 & \bm O & \frac13 \bm O & \Id
	\end{pmatrix},
	\label{eq:aklt-mpo-clean}
\end{equation}
with the understanding that the products in $\bar Q$ generate the
$\bigl(\mathbf S_i\cdot \mathbf S_{i+1}\bigr)^2$ term.

For OBC, choose boundary tensors
\begin{equation}
	\hat W^{[1]}_{\mathrm{AKLT}}
	=
	\begin{pmatrix}
		0 & \bm O & \frac13 \bm O & \Id
	\end{pmatrix},
\end{equation}
and
\begin{equation}
	\hat W^{[L]}_{\mathrm{AKLT}}
	=
	\begin{pmatrix}
		\Id \\ \bar{\bm O} \\ \bar Q \\ 0
	\end{pmatrix}.
\end{equation}
Contracting these tensors produces exactly Eq.~\eqref{eq:aklt-ham}.

\subsection{Local spin-1 operators}

In the local basis $\{\ket{+1},\ket{0},\ket{-1}\}$, use
\begin{equation}
	S^z =
	\begin{pmatrix}
		1 & 0 & 0 \\
		0 & 0 & 0 \\
		0 & 0 & -1
	\end{pmatrix},
\end{equation}
\begin{equation}
	S^+ =
	\sqrt2
	\begin{pmatrix}
		0 & 1 & 0 \\
		0 & 0 & 1 \\
		0 & 0 & 0
	\end{pmatrix},
	\qquad
	S^- =
	\sqrt2
	\begin{pmatrix}
		0 & 0 & 0 \\
		1 & 0 & 0 \\
		0 & 1 & 0
	\end{pmatrix}.
\end{equation}

\paragraph{Implementation note.}
The most transparent coding strategy is:
\begin{enumerate}
	\item define $O_a$ and $\bar O_a$,
	\item build the $3\times 3$ table of products $\bar Q_{ab}=\bar O_a\bar O_b$,
	\item assemble the operator-valued $8\times 8$ block tensor of Eq.~\eqref{eq:aklt-mpo-clean},
	\item convert each operator entry into local matrix elements
	\[
	W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
	=
	\bra{\sigma}\hat W^{[i]}_{\beta_{i-1},\beta_i}\ket{\sigma'}.
	\]
\end{enumerate}

\paragraph{Recommended unit tests.}
For $L=2,3,4$:
\begin{itemize}
	\item contract the MPO to a dense matrix,
	\item compare against the exact Hamiltonian built from
	\[
	h_{i,i+1}
	=
	\mathbf S_i\cdot \mathbf S_{i+1}
	+\frac13(\mathbf S_i\cdot \mathbf S_{i+1})^2,
	\]
	using Kronecker products,
	\item verify that the known AKLT MPS in the note has energy
	\[
	E_0 = -\frac23(L-1)
	\]
	up to numerical precision.
\end{itemize}
```

---

## Important correction to the explicit \(8\times 8\) form

In the block discussion above, the **correct logical structure** is the block form in Eq. `\eqref{eq:aklt-mpo-clean}`.  
The fully expanded \(8\times 8\) form is easy to write incorrectly if one flattens the \(3\times3\) block \( \bar Q \) carelessly.

So for the note, I strongly recommend:

- **keep the block form**
- **do not include a flattened explicit 8×8 matrix unless you also specify the exact index ordering of the 3×3 block**

That is the cleanest implementation document.

---

## Even better: a simpler exact MPO with bond dimension 14

If you want a **completely explicit scalar-index MPO with no block ambiguity**, one can instead use a bond dimension
\[
\chi=1+3+9+1=14,
\]
which is slightly larger but trivial to code:

- states:
  - 1 start state
  - 3 channels for the bilinear term
  - 9 channels for the biquadratic term
  - 1 terminal state

This avoids packing the 3×3 operator block into a 3-dimensional subspace. If you want, I can write that version too; for implementation notes it is often the most foolproof.

---

## Short verdict on your current note

### Present:
- MPS canonical forms: yes
- generic XXZ/Heisenberg MPO: yes
- AKLT state MPS: yes

### Missing:
- AKLT Hamiltonian MPO: **missing**

---

## My recommendation

Add a new section with:

1. AKLT Hamiltonian definition  
2. spin-1 local matrices  
3. factorized decomposition of \( \mathbf S\cdot \mathbf S \)  
4. exact MPO construction  
5. small-system dense-MPO unit test

If you want, I can now do one of two things:

1. **rewrite the AKLT MPO section into a cleaner final LaTeX version** that fits your note’s style exactly, or  
2. give you a **fully explicit 14-channel MPO** that is easier to implement directly in code.

---
## 🧑‍💻 User

1

---
## 🤖 Assistant

Great — here is a **cleaner final LaTeX section** in the style of your note, ready to paste in.

I recommend inserting it **after** `\section{Heisenberg MPO}` and **before** `\section{MPO environments}`.

---

```latex
\section{AKLT Hamiltonian MPO}

For the spin-1 AKLT chain, the Hamiltonian is
\begin{equation}
	H_{\mathrm{AKLT}}
	=
	\sum_{i=1}^{L-1}
	\left[
	\mathbf S_i\cdot \mathbf S_{i+1}
	+
	\frac13\left(\mathbf S_i\cdot \mathbf S_{i+1}\right)^2
	\right].
	\label{eq:aklt-hamiltonian}
\end{equation}
This is the Hamiltonian whose exact ground state is the AKLT MPS given later in these notes.

\subsection{Local spin-1 operators}

In the local basis
\[
\{\ket{+1},\ket{0},\ket{-1}\},
\]
use the standard spin-1 matrices
\begin{equation}
	S^z
	=
	\begin{pmatrix}
		1 & 0 & 0 \\
		0 & 0 & 0 \\
		0 & 0 & -1
	\end{pmatrix},
\end{equation}
\begin{equation}
	S^+
	=
	\sqrt{2}
	\begin{pmatrix}
		0 & 1 & 0 \\
		0 & 0 & 1 \\
		0 & 0 & 0
	\end{pmatrix},
	\qquad
	S^-
	=
	\sqrt{2}
	\begin{pmatrix}
		0 & 0 & 0 \\
		1 & 0 & 0 \\
		0 & 1 & 0
	\end{pmatrix}.
\end{equation}

Define the bilinear bond operator
\begin{equation}
	X_{i,i+1}
	\equiv
	\mathbf S_i\cdot \mathbf S_{i+1}
	=
	\frac12\left(S_i^+S_{i+1}^- + S_i^-S_{i+1}^+\right)
	+
	S_i^z S_{i+1}^z.
	\label{eq:aklt-bilinear-bond}
\end{equation}
Then
\begin{equation}
	h_{i,i+1}^{\mathrm{AKLT}}
	=
	X_{i,i+1}
	+
	\frac13 X_{i,i+1}^2,
	\label{eq:aklt-local-bond}
\end{equation}
and
\[
H_{\mathrm{AKLT}} = \sum_{i=1}^{L-1} h_{i,i+1}^{\mathrm{AKLT}}.
\]

\subsection{Factorized form of the bond interaction}

Introduce the operator lists
\begin{equation}
	O_1 = \frac{1}{\sqrt{2}}S^+,\qquad
	O_2 = \frac{1}{\sqrt{2}}S^-,\qquad
	O_3 = S^z,
\end{equation}
and
\begin{equation}
	\bar O_1 = \frac{1}{\sqrt{2}}S^-,\qquad
	\bar O_2 = \frac{1}{\sqrt{2}}S^+,\qquad
	\bar O_3 = S^z.
\end{equation}
Then Eq.~\eqref{eq:aklt-bilinear-bond} becomes
\begin{equation}
	X_{i,i+1}
	=
	\sum_{a=1}^3 O_a^{[i]} \bar O_a^{[i+1]}.
	\label{eq:X-factorized}
\end{equation}
Therefore
\begin{equation}
	X_{i,i+1}^2
	=
	\sum_{a=1}^3 \sum_{b=1}^3
	\left(O_a O_b\right)^{[i]}
	\left(\bar O_a \bar O_b\right)^{[i+1]}.
	\label{eq:X2-factorized}
\end{equation}
Substituting Eqs.~\eqref{eq:X-factorized} and \eqref{eq:X2-factorized} into Eq.~\eqref{eq:aklt-local-bond}, we obtain
\begin{equation}
	h_{i,i+1}^{\mathrm{AKLT}}
	=
	\sum_{a=1}^3 O_a^{[i]} \bar O_a^{[i+1]}
	+
	\frac13
	\sum_{a=1}^3 \sum_{b=1}^3
	\left(O_a O_b\right)^{[i]}
	\left(\bar O_a \bar O_b\right)^{[i+1]}.
	\label{eq:aklt-factorized-bond}
\end{equation}

\subsection{Exact MPO construction}

Equation~\eqref{eq:aklt-factorized-bond} is an exact nearest-neighbor MPO. A convenient operator-valued bulk MPO has bond dimension
\[
\chi = 1 + 3 + 9 + 1 = 14.
\]
This slightly larger bond dimension is chosen because it leads to a completely explicit implementation without block-index ambiguities.

Use the auxiliary MPO bond states in the order
\[
1
\;\big|\;
\{a\}_{a=1}^3
\;\big|\;
\{(a,b)\}_{a,b=1}^3
\;\big|\;
14.
\]
Then the bulk operator-valued MPO tensor is
\begin{equation}
	\hat W_{\mathrm{AKLT,bulk}}
	=
	\begin{pmatrix}
		\Id & 0 & 0 & 0 \\
		\bar{\bm O} & 0 & 0 & 0 \\
		\bar{\bm Q} & 0 & 0 & 0 \\
		0 & \bm O & \frac13 \bm O^{(2)} & \Id
	\end{pmatrix},
	\label{eq:aklt-mpo-block}
\end{equation}
where the block sizes are $1|3|9|1$, and
\begin{equation}
	\bar{\bm O}
	=
	\begin{pmatrix}
		\bar O_1 \\ \bar O_2 \\ \bar O_3
	\end{pmatrix},
	\qquad
	\bm O
	=
	\begin{pmatrix}
		O_1 & O_2 & O_3
	\end{pmatrix},
\end{equation}
\begin{equation}
	\bar{\bm Q}
	=
	\begin{pmatrix}
		\bar O_1 \bar O_1 \\
		\bar O_1 \bar O_2 \\
		\bar O_1 \bar O_3 \\
		\bar O_2 \bar O_1 \\
		\bar O_2 \bar O_2 \\
		\bar O_2 \bar O_3 \\
		\bar O_3 \bar O_1 \\
		\bar O_3 \bar O_2 \\
		\bar O_3 \bar O_3
	\end{pmatrix},
\end{equation}
and
\begin{equation}
	\bm O^{(2)}
	=
	\begin{pmatrix}
		O_1 O_1 &
		O_1 O_2 &
		O_1 O_3 &
		O_2 O_1 &
		O_2 O_2 &
		O_2 O_3 &
		O_3 O_1 &
		O_3 O_2 &
		O_3 O_3
	\end{pmatrix}.
\end{equation}

In words:
\begin{itemize}
	\item the channel $1\to a$ starts the bilinear term via $\bar O_a$,
	\item the channel $1\to(a,b)$ starts the biquadratic term via $\bar O_a\bar O_b$,
	\item the channel $a\to 14$ finishes the bilinear term via $O_a$,
	\item the channel $(a,b)\to 14$ finishes the biquadratic term via $\frac13 O_a O_b$,
	\item the channel $14\to 14$ propagates the identity.
\end{itemize}
Contracting the MPO therefore generates exactly
\[
\sum_i h_{i,i+1}^{\mathrm{AKLT}}.
\]

For OBC, use boundary tensors
\begin{equation}
	\hat W_{\mathrm{AKLT}}^{[1]}
	=
	\begin{pmatrix}
		0 & \bm O & \frac13 \bm O^{(2)} & \Id
	\end{pmatrix},
	\label{eq:aklt-left-boundary}
\end{equation}
and
\begin{equation}
	\hat W_{\mathrm{AKLT}}^{[L]}
	=
	\begin{pmatrix}
		\Id \\ \bar{\bm O} \\ \bar{\bm Q} \\ 0
	\end{pmatrix}.
	\label{eq:aklt-right-boundary}
\end{equation}
Thus:
\begin{itemize}
	\item the left boundary tensor has shape $(1,14,d,d)$,
	\item the right boundary tensor has shape $(14,1,d,d)$,
	\item each bulk tensor has shape $(14,14,d,d)$ with $d=3$.
\end{itemize}

\subsection{Tensor components}

As in the Heisenberg MPO section, the operator-valued entries are converted into numerical MPO tensor components by
\begin{equation}
	W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
	=
	\bra{\sigma}
	\hat W^{[i]}_{\beta_{i-1},\beta_i}
	\ket{\sigma'}.
\end{equation}
Here $\sigma,\sigma'\in\{+1,0,-1\}$ and the local physical dimension is $d=3$.

\subsection{Implementation remarks}

A transparent implementation strategy is:
\begin{enumerate}
	\item construct the local spin-1 matrices $S^+,S^-,S^z$,
	\item define $O_a$ and $\bar O_a$,
	\item build the $9$ operators $\bar O_a\bar O_b$,
	\item build the $9$ operators $O_a O_b$,
	\item assemble the operator-valued boundary and bulk MPOs from
	Eqs.~\eqref{eq:aklt-mpo-block}, \eqref{eq:aklt-left-boundary}, and \eqref{eq:aklt-right-boundary},
	\item expand each operator entry into local matrix elements.
\end{enumerate}

\paragraph{Recommended unit tests.}
For $L=2,3,4$:
\begin{itemize}
	\item contract the MPO to a dense matrix,
	\item compare against the Hamiltonian built directly from Eq.~\eqref{eq:aklt-hamiltonian},
	\item verify that the AKLT MPS given later in these notes has energy
	\[
	E_0 = -\frac{2}{3}(L-1)
	\]
	for open chains, up to numerical precision.
\end{itemize}
```

---

## Why I used \(\chi=14\) here

This is the cleanest implementation-oriented choice because it avoids ambiguous internal block flattening.

- bilinear channels: \(3\)
- biquadratic channels: \(9\)
- plus start/end identity channels

So:
\[
1+3+9+1=14.
\]

This is a bit larger than the most compressed possible form, but much safer for a coding note.

---

## One small issue in your note unrelated to AKLT MPO

In your current **AKLT benchmark state** section, you write:

> “For this gauge, the tensors satisfy  
> \(\sum_\sigma A^\sigma A^{\sigma\dagger}=\Id_2\),  
> so they are right-canonical in the convention of Eq. (right-canonical).”

But earlier your right-canonical condition was
\[
\sum_{\sigma,\alpha_i}
B_{\alpha_{i-1},\sigma,\alpha_i}
B^*_{\alpha'_{i-1},\sigma,\alpha_i}
=
\delta_{\alpha_{i-1},\alpha'_{i-1}},
\]
which indeed corresponds to
\[
\sum_\sigma B^\sigma B^{\sigma\dagger}=\Id.
\]
So that part is consistent.

---

If you want, I can next do either of these:

1. give you the **same section but with a short Python/NumPy snippet** that builds the AKLT MPO tensor, or  
2. integrate this section into your full `.tex` and clean up surrounding cross-references.

---
## 🧑‍💻 User

So this version is ready for coding?

---
## 🤖 Assistant

Yes — **the \(\chi=14\) version is ready for coding**.

But let me be precise about what that means:

## Ready for coding
It is ready in the sense that it gives you an **unambiguous exact MPO construction** for the AKLT Hamiltonian:
- local basis fixed,
- local spin-1 matrices fixed,
- factorization fixed,
- auxiliary bond channels fixed,
- left/bulk/right MPO structure fixed.

That is enough to implement directly.

## What you still need to do in code
You still have to choose an explicit indexing convention for the 14 MPO bond states. For example:

- channel 0: start
- channels 1,2,3: bilinear channels \(a=1,2,3\)
- channels 4,\dots,12: biquadratic channels \((a,b)\) in lexicographic order
- channel 13: final/identity channel

For coding, I recommend this exact ordering:
\[
0,\;
1\equiv a=1,\;2\equiv a=2,\;3\equiv a=3,\;
4\equiv(1,1),\;5\equiv(1,2),\;6\equiv(1,3),\;7\equiv(2,1),\;8\equiv(2,2),\;9\equiv(2,3),\;10\equiv(3,1),\;11\equiv(3,2),\;12\equiv(3,3),\;
13.
\]

Then the MPO tensors are straightforward.

---

# Coding form

Let
```python
d = 3
chi = 14
```

and define local operators
```python
Sz = np.array([[1,0,0],[0,0,0],[0,0,-1]], dtype=np.complex128)
Sp = np.sqrt(2) * np.array([[0,1,0],[0,0,1],[0,0,0]], dtype=np.complex128)
Sm = np.sqrt(2) * np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.complex128)
I  = np.eye(3, dtype=np.complex128)
```

Then
```python
O = [Sp/np.sqrt(2), Sm/np.sqrt(2), Sz]
Obar = [Sm/np.sqrt(2), Sp/np.sqrt(2), Sz]
```

and the nine products
```python
O2 = [O[a] @ O[b] for a in range(3) for b in range(3)]
Obar2 = [Obar[a] @ Obar[b] for a in range(3) for b in range(3)]
```

---

# Bulk MPO tensor

Using your convention
\[
W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
\]
with shape `(chi_left, chi_right, d, d)`, the bulk tensor is:

```python
def aklt_mpo_bulk():
    W = np.zeros((14, 14, 3, 3), dtype=np.complex128)

    # 0 -> 0 : identity propagation from the start channel
    W[0,0] = I

    # start bilinear channels: 0 -> 1..3
    for a in range(3):
        W[0,1+a] = Obar[a]

    # start biquadratic channels: 0 -> 4..12
    for n in range(9):
        W[0,4+n] = Obar2[n]

    # finish bilinear channels: 1..3 -> 13
    for a in range(3):
        W[1+a,13] = O[a]

    # finish biquadratic channels: 4..12 -> 13
    for n in range(9):
        W[4+n,13] = (1.0/3.0) * O2[n]

    # 13 -> 13 : identity propagation after term completion
    W[13,13] = I

    return W
```

---

# Boundary tensors for OBC

Left boundary shape `(1,14,3,3)`:
```python
def aklt_mpo_left():
    W = np.zeros((1, 14, 3, 3), dtype=np.complex128)

    for a in range(3):
        W[0,1+a] = O[a]

    for n in range(9):
        W[0,4+n] = (1.0/3.0) * O2[n]

    W[0,13] = I
    return W
```

Right boundary shape `(14,1,3,3)`:
```python
def aklt_mpo_right():
    W = np.zeros((14, 1, 3, 3), dtype=np.complex128)

    W[0,0] = I

    for a in range(3):
        W[1+a,0] = Obar[a]

    for n in range(9):
        W[4+n,0] = Obar2[n]

    return W
```

---

# Full MPO builder

```python
def aklt_mpo(L):
    assert L >= 2
    mpo = [None] * L
    mpo[0] = aklt_mpo_left()
    for i in range(1, L-1):
        mpo[i] = aklt_mpo_bulk()
    mpo[-1] = aklt_mpo_right()
    return mpo
```

---

# Important check: does this generate the correct Hamiltonian?

Yes, with the above channel logic:

- left boundary starts a term with \(O_a\) or \(\frac13 O_aO_b\),
- bulk/right completes it with \(\bar O_a\) or \(\bar O_a\bar O_b\),
- identity propagates completed strings.

This produces
\[
\sum_i \left( \sum_a O_a^{[i]} \bar O_a^{[i+1]}
+ \frac13 \sum_{a,b} (O_aO_b)^{[i]} (\bar O_a\bar O_b)^{[i+1]} \right),
\]
which is exactly
\[
\sum_i \left[\mathbf S_i\cdot\mathbf S_{i+1}+\frac13(\mathbf S_i\cdot\mathbf S_{i+1})^2\right].
\]

---

# One caution
The **LaTeX section is conceptually correct**, but for coding I trust the **explicit channel-index implementation above** more than the symbolic block matrix. That’s normal: code needs flattening choices; LaTeX often suppresses them.

So my practical answer is:

- **Yes, the construction is ready for coding**
- but for actual implementation, use the **explicit index-resolved version above**

---

# Recommended minimal tests
Before using it in DMRG:

1. Build dense \(H\) from the MPO for \(L=2,3,4\)
2. Build dense \(H\) directly from
   \[
   h_{i,i+1}= \mathbf S_i\cdot\mathbf S_{i+1}+\frac13(\mathbf S_i\cdot\mathbf S_{i+1})^2
   \]
3. Compare entrywise:
   ```python
   np.linalg.norm(H_mpo - H_exact)
   ```
   should be near machine precision.

4. Check the ground-state energy for small OBC chains:
   \[
   E_0 = -\frac23(L-1)
   \]
   for the AKLT state MPS.

---

If you want, I can now give you a **drop-in LaTeX code block with this explicit channel ordering written in the note**, so the note becomes directly implementation-grade rather than just conceptually correct.

---
## 🧑‍💻 User

At this stage, do not add python code.

---
## 🤖 Assistant

Understood. Then the answer is:

## Short answer
**Almost, but not fully.**

The LaTeX section I gave is **conceptually exact and implementation-safe at the mathematical level**, but if your goal is that the note itself be **fully coding-ready without further interpretation**, then it still needs **one more thing**:

- an **explicit ordering of the 14 MPO bond channels**.

Without that, a coder still has to decide how the \(3\) bilinear channels and \(9\) biquadratic channels are flattened into indices \(1,\dots,14\). That is easy, but it is still a choice. For a true implementation note, that choice should be fixed in the text.

So:

- **Mathematically ready**: yes
- **Directly coding-ready from the note alone**: not yet fully, unless we add the channel-order convention

---

## What is missing
The section currently says the bond dimension is
\[
\chi = 1+3+9+1 = 14,
\]
and explains the blocks, but it does **not** explicitly define the mapping
\[
\beta = 0,1,\dots,13
\]
to
- start channel,
- bilinear channels \(a=1,2,3\),
- biquadratic channels \((a,b)\),
- terminal channel.

A code needs that mapping fixed.

---

## Minimal fix to make it fully coding-ready
Add a short paragraph like this inside the AKLT MPO section.

```latex
\paragraph{Explicit MPO bond-index convention.}
For implementation, fix the MPO bond states in the following order:
\begin{equation}
	\beta = 0
	\quad\text{(start)},
\end{equation}
\begin{equation}
	\beta = 1,2,3
	\quad\leftrightarrow\quad
	a=1,2,3,
\end{equation}
\begin{equation}
	\beta = 4,\dots,12
	\quad\leftrightarrow\quad
	(a,b)\in\{(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)\},
\end{equation}
in lexicographic order, and
\begin{equation}
	\beta = 13
	\quad\text{(terminal / identity-propagation channel)}.
\end{equation}
With this convention, the bulk MPO tensor has shape $(14,14,3,3)$, the left boundary tensor has shape $(1,14,3,3)$, and the right boundary tensor has shape $(14,1,3,3)$.
```

That would make the note fully unambiguous for coding.

---

## One more subtle point
There is another reason I would not yet call the previous LaTeX section “fully coding-ready”: the symbolic block form
\[
\hat W_{\mathrm{AKLT,bulk}}
=
\begin{pmatrix}
\Id & 0 & 0 & 0\\
\bar{\bm O} & 0 & 0 & 0\\
\bar{\bm Q} & 0 & 0 & 0\\
0 & \bm O & \frac13\bm O^{(2)} & \Id
\end{pmatrix}
\]
is good mathematically, but for coding notes it is best to state explicitly that:

- \(\bar{\bm Q}\) is a **9-component column** ordered lexicographically in \((a,b)\),
- \(\bm O^{(2)}\) is a **9-component row** in the **same order**.

Otherwise a programmer could accidentally mismatch the order in the left and right factors.

So I would add one sentence:

```latex
The ordering of the $9$ composite channels in $\bar{\bm Q}$ and $\bm O^{(2)}$ must be identical; throughout these notes we use lexicographic order in $(a,b)$.
```

---

# Final verdict

## If you do not add anything else:
I would say:
- **good mathematically**
- **not yet fully implementation-grade**

## If you add the two short clarifications above:
then I would say:
- **yes, fully ready for coding**

---

If you want, I can now give you a **small replacement patch** to the LaTeX section — just the extra implementation-grade wording, no Python, no extra derivation.

---
## 🧑‍💻 User

Give me give you a small replacement patch to the LaTeX section 

---
## 🤖 Assistant

Here is a **small replacement patch** that makes the AKLT MPO section fully implementation-grade **without adding any Python**.

You can paste these pieces into the section I gave before.

---

## Patch 1: replace the paragraph introducing the bond dimension

Replace this part:
```latex
Equation~\eqref{eq:aklt-factorized-bond} is an exact nearest-neighbor MPO. A convenient operator-valued bulk MPO has bond dimension
\[
\chi = 1 + 3 + 9 + 1 = 14.
\]
This slightly larger bond dimension is chosen because it leads to a completely explicit implementation without block-index ambiguities.
```

with:

```latex
Equation~\eqref{eq:aklt-factorized-bond} is an exact nearest-neighbor MPO. A convenient operator-valued bulk MPO has bond dimension
\[
\chi = 1 + 3 + 9 + 1 = 14.
\]
This slightly larger bond dimension is chosen because it leads to a completely explicit implementation without block-index ambiguities.

\paragraph{Explicit MPO bond-index convention.}
For implementation, the MPO bond index is ordered as follows:
\begin{equation}
	\beta = 0
	\qquad \text{start channel},
\end{equation}
\begin{equation}
	\beta = 1,2,3
	\qquad \leftrightarrow \qquad
	a=1,2,3,
\end{equation}
\begin{equation}
	\beta = 4,\dots,12
	\qquad \leftrightarrow \qquad
	(a,b)\in
	\{(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)\},
	\label{eq:aklt-lexicographic-order}
\end{equation}
taken in lexicographic order, and
\begin{equation}
	\beta = 13
	\qquad \text{terminal / identity-propagation channel}.
\end{equation}
This convention is used consistently in the bulk tensor and in both boundary tensors.
```

---

## Patch 2: replace the paragraph just before the bulk MPO matrix

Replace this part:
```latex
Use the auxiliary MPO bond states in the order
\[
1
\;\big|\;
\{a\}_{a=1}^3
\;\big|\;
\{(a,b)\}_{a,b=1}^3
\;\big|\;
14.
\]
Then the bulk operator-valued MPO tensor is
```

with:

```latex
Using the bond-index convention above, the bulk operator-valued MPO tensor is
```

---

## Patch 3: replace the definitions of \(\bar{\bm Q}\) and \(\bm O^{(2)}\)

Replace this block:
```latex
\begin{equation}
	\bar{\bm Q}
	=
	\begin{pmatrix}
		\bar O_1 \bar O_1 \\
		\bar O_1 \bar O_2 \\
		\bar O_1 \bar O_3 \\
		\bar O_2 \bar O_1 \\
		\bar O_2 \bar O_2 \\
		\bar O_2 \bar O_3 \\
		\bar O_3 \bar O_1 \\
		\bar O_3 \bar O_2 \\
		\bar O_3 \bar O_3
	\end{pmatrix},
\end{equation}
and
\begin{equation}
	\bm O^{(2)}
	=
	\begin{pmatrix}
		O_1 O_1 &
		O_1 O_2 &
		O_1 O_3 &
		O_2 O_1 &
		O_2 O_2 &
		O_2 O_3 &
		O_3 O_1 &
		O_3 O_2 &
		O_3 O_3
	\end{pmatrix}.
\end{equation}
```

with:

```latex
\begin{equation}
	\bar{\bm Q}
	=
	\begin{pmatrix}
		\bar O_1 \bar O_1 \\
		\bar O_1 \bar O_2 \\
		\bar O_1 \bar O_3 \\
		\bar O_2 \bar O_1 \\
		\bar O_2 \bar O_2 \\
		\bar O_2 \bar O_3 \\
		\bar O_3 \bar O_1 \\
		\bar O_3 \bar O_2 \\
		\bar O_3 \bar O_3
	\end{pmatrix},
\end{equation}
and
\begin{equation}
	\bm O^{(2)}
	=
	\begin{pmatrix}
		O_1 O_1 &
		O_1 O_2 &
		O_1 O_3 &
		O_2 O_1 &
		O_2 O_2 &
		O_2 O_3 &
		O_3 O_1 &
		O_3 O_2 &
		O_3 O_3
	\end{pmatrix}.
\end{equation}
The ordering of the $9$ entries in $\bar{\bm Q}$ and $\bm O^{(2)}$ is the same lexicographic ordering defined in Eq.~\eqref{eq:aklt-lexicographic-order}. This ordering must be used identically on the left and right of the MPO construction.
```

---

## Patch 4: replace the “In words” bullet list with a slightly more explicit one

Replace:
```latex
In words:
\begin{itemize}
	\item the channel $1\to a$ starts the bilinear term via $\bar O_a$,
	\item the channel $1\to(a,b)$ starts the biquadratic term via $\bar O_a\bar O_b$,
	\item the channel $a\to 14$ finishes the bilinear term via $O_a$,
	\item the channel $(a,b)\to 14$ finishes the biquadratic term via $\frac13 O_a O_b$,
	\item the channel $14\to 14$ propagates the identity.
\end{itemize}
Contracting the MPO therefore generates exactly
\[
\sum_i h_{i,i+1}^{\mathrm{AKLT}}.
\]
```

with:

```latex
In words:
\begin{itemize}
	\item the channel $0\to a$ with $a=1,2,3$ starts the bilinear term via $\bar O_a$,
	\item the channel $0\to(a,b)$ starts the biquadratic term via $\bar O_a\bar O_b$,
	\item the channel $a\to 13$ finishes the bilinear term via $O_a$,
	\item the channel $(a,b)\to 13$ finishes the biquadratic term via $\frac13 O_a O_b$,
	\item the channel $13\to 13$ propagates the identity after a term has been completed.
\end{itemize}
With the index convention above, contracting the MPO generates exactly
\[
\sum_{i=1}^{L-1} h_{i,i+1}^{\mathrm{AKLT}}
=
\sum_{i=1}^{L-1}
\left[
\mathbf S_i\cdot \mathbf S_{i+1}
+
\frac13\left(\mathbf S_i\cdot \mathbf S_{i+1}\right)^2
\right].
\]
```

---

## Patch 5: replace the last bullet list about tensor shapes with a more coding-oriented version

Replace:
```latex
Thus:
\begin{itemize}
	\item the left boundary tensor has shape $(1,14,d,d)$,
	\item the right boundary tensor has shape $(14,1,d,d)$,
	\item each bulk tensor has shape $(14,14,d,d)$ with $d=3$.
\end{itemize}
```

with:

```latex
Thus, in the tensor convention of these notes,
\begin{itemize}
	\item the left boundary tensor has shape $(1,14,3,3)$,
	\item each bulk tensor has shape $(14,14,3,3)$,
	\item the right boundary tensor has shape $(14,1,3,3)$.
\end{itemize}
The MPO physical indices are ordered as
\[
W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'},
\]
with $\sigma$ the ket index and $\sigma'$ the bra index, exactly as in the Heisenberg MPO section.
```

---

# Result
After these five small replacements, the section is:

- mathematically correct,
- internally consistent with your note’s tensor conventions,
- **fully ready for coding** without forcing the programmer to make hidden indexing choices.

If you want, I can also give you the **full AKLT section already patched and merged into one clean LaTeX block** so you can paste it in one shot.

---
## 🧑‍💻 User

give me the full AKLT section already patched and merged into one clean LaTeX block

---
## 🤖 Assistant

```latex
\section{AKLT Hamiltonian MPO}

For the spin-1 AKLT chain, the Hamiltonian is
\begin{equation}
	H_{\mathrm{AKLT}}
	=
	\sum_{i=1}^{L-1}
	\left[
	\mathbf S_i\cdot \mathbf S_{i+1}
	+
	\frac13\left(\mathbf S_i\cdot \mathbf S_{i+1}\right)^2
	\right].
	\label{eq:aklt-hamiltonian}
\end{equation}
This is the Hamiltonian whose exact ground state is the AKLT MPS given later in these notes.

\subsection{Local spin-1 operators}

In the local basis
\[
\{\ket{+1},\ket{0},\ket{-1}\},
\]
use the standard spin-1 matrices
\begin{equation}
	S^z
	=
	\begin{pmatrix}
		1 & 0 & 0 \\
		0 & 0 & 0 \\
		0 & 0 & -1
	\end{pmatrix},
\end{equation}
\begin{equation}
	S^+
	=
	\sqrt{2}
	\begin{pmatrix}
		0 & 1 & 0 \\
		0 & 0 & 1 \\
		0 & 0 & 0
	\end{pmatrix},
	\qquad
	S^-
	=
	\sqrt{2}
	\begin{pmatrix}
		0 & 0 & 0 \\
		1 & 0 & 0 \\
		0 & 1 & 0
	\end{pmatrix}.
\end{equation}

Define the bilinear bond operator
\begin{equation}
	X_{i,i+1}
	\equiv
	\mathbf S_i\cdot \mathbf S_{i+1}
	=
	\frac12\left(S_i^+S_{i+1}^- + S_i^-S_{i+1}^+\right)
	+
	S_i^z S_{i+1}^z.
	\label{eq:aklt-bilinear-bond}
\end{equation}
Then
\begin{equation}
	h_{i,i+1}^{\mathrm{AKLT}}
	=
	X_{i,i+1}
	+
	\frac13 X_{i,i+1}^2,
	\label{eq:aklt-local-bond}
\end{equation}
and
\[
H_{\mathrm{AKLT}} = \sum_{i=1}^{L-1} h_{i,i+1}^{\mathrm{AKLT}}.
\]

\subsection{Factorized form of the bond interaction}

Introduce the operator lists
\begin{equation}
	O_1 = \frac{1}{\sqrt{2}}S^+,\qquad
	O_2 = \frac{1}{\sqrt{2}}S^-,\qquad
	O_3 = S^z,
\end{equation}
and
\begin{equation}
	\bar O_1 = \frac{1}{\sqrt{2}}S^-,\qquad
	\bar O_2 = \frac{1}{\sqrt{2}}S^+,\qquad
	\bar O_3 = S^z.
\end{equation}
Then Eq.~\eqref{eq:aklt-bilinear-bond} becomes
\begin{equation}
	X_{i,i+1}
	=
	\sum_{a=1}^3 O_a^{[i]} \bar O_a^{[i+1]}.
	\label{eq:X-factorized}
\end{equation}
Therefore
\begin{equation}
	X_{i,i+1}^2
	=
	\sum_{a=1}^3 \sum_{b=1}^3
	\left(O_a O_b\right)^{[i]}
	\left(\bar O_a \bar O_b\right)^{[i+1]}.
	\label{eq:X2-factorized}
\end{equation}
Substituting Eqs.~\eqref{eq:X-factorized} and \eqref{eq:X2-factorized} into Eq.~\eqref{eq:aklt-local-bond}, we obtain
\begin{equation}
	h_{i,i+1}^{\mathrm{AKLT}}
	=
	\sum_{a=1}^3 O_a^{[i]} \bar O_a^{[i+1]}
	+
	\frac13
	\sum_{a=1}^3 \sum_{b=1}^3
	\left(O_a O_b\right)^{[i]}
	\left(\bar O_a \bar O_b\right)^{[i+1]}.
	\label{eq:aklt-factorized-bond}
\end{equation}

\subsection{Exact MPO construction}

Equation~\eqref{eq:aklt-factorized-bond} is an exact nearest-neighbor MPO. A convenient operator-valued bulk MPO has bond dimension
\[
\chi = 1 + 3 + 9 + 1 = 14.
\]
This slightly larger bond dimension is chosen because it leads to a completely explicit implementation without block-index ambiguities.

\paragraph{Explicit MPO bond-index convention.}
For implementation, the MPO bond index is ordered as follows:
\begin{equation}
	\beta = 0
	\qquad \text{start channel},
\end{equation}
\begin{equation}
	\beta = 1,2,3
	\qquad \leftrightarrow \qquad
	a=1,2,3,
\end{equation}
\begin{equation}
	\beta = 4,\dots,12
	\qquad \leftrightarrow \qquad
	(a,b)\in
	\{(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)\},
	\label{eq:aklt-lexicographic-order}
\end{equation}
taken in lexicographic order, and
\begin{equation}
	\beta = 13
	\qquad \text{terminal / identity-propagation channel}.
\end{equation}
This convention is used consistently in the bulk tensor and in both boundary tensors.

Using the bond-index convention above, the bulk operator-valued MPO tensor is
\begin{equation}
	\hat W_{\mathrm{AKLT,bulk}}
	=
	\begin{pmatrix}
		\Id & 0 & 0 & 0 \\
		\bar{\bm O} & 0 & 0 & 0 \\
		\bar{\bm Q} & 0 & 0 & 0 \\
		0 & \bm O & \frac13 \bm O^{(2)} & \Id
	\end{pmatrix},
	\label{eq:aklt-mpo-block}
\end{equation}
where the block sizes are $1|3|9|1$, and
\begin{equation}
	\bar{\bm O}
	=
	\begin{pmatrix}
		\bar O_1 \\ \bar O_2 \\ \bar O_3
	\end{pmatrix},
	\qquad
	\bm O
	=
	\begin{pmatrix}
		O_1 & O_2 & O_3
	\end{pmatrix},
\end{equation}
\begin{equation}
	\bar{\bm Q}
	=
	\begin{pmatrix}
		\bar O_1 \bar O_1 \\
		\bar O_1 \bar O_2 \\
		\bar O_1 \bar O_3 \\
		\bar O_2 \bar O_1 \\
		\bar O_2 \bar O_2 \\
		\bar O_2 \bar O_3 \\
		\bar O_3 \bar O_1 \\
		\bar O_3 \bar O_2 \\
		\bar O_3 \bar O_3
	\end{pmatrix},
\end{equation}
and
\begin{equation}
	\bm O^{(2)}
	=
	\begin{pmatrix}
		O_1 O_1 &
		O_1 O_2 &
		O_1 O_3 &
		O_2 O_1 &
		O_2 O_2 &
		O_2 O_3 &
		O_3 O_1 &
		O_3 O_2 &
		O_3 O_3
	\end{pmatrix}.
\end{equation}
The ordering of the $9$ entries in $\bar{\bm Q}$ and $\bm O^{(2)}$ is the same lexicographic ordering defined in Eq.~\eqref{eq:aklt-lexicographic-order}. This ordering must be used identically on the left and right of the MPO construction.

In words:
\begin{itemize}
	\item the channel $0\to a$ with $a=1,2,3$ starts the bilinear term via $\bar O_a$,
	\item the channel $0\to(a,b)$ starts the biquadratic term via $\bar O_a\bar O_b$,
	\item the channel $a\to 13$ finishes the bilinear term via $O_a$,
	\item the channel $(a,b)\to 13$ finishes the biquadratic term via $\frac13 O_a O_b$,
	\item the channel $13\to 13$ propagates the identity after a term has been completed.
\end{itemize}
With the index convention above, contracting the MPO generates exactly
\[
\sum_{i=1}^{L-1} h_{i,i+1}^{\mathrm{AKLT}}
=
\sum_{i=1}^{L-1}
\left[
\mathbf S_i\cdot \mathbf S_{i+1}
+
\frac13\left(\mathbf S_i\cdot \mathbf S_{i+1}\right)^2
\right].
\]

For OBC, use boundary tensors
\begin{equation}
	\hat W_{\mathrm{AKLT}}^{[1]}
	=
	\begin{pmatrix}
		0 & \bm O & \frac13 \bm O^{(2)} & \Id
	\end{pmatrix},
	\label{eq:aklt-left-boundary}
\end{equation}
and
\begin{equation}
	\hat W_{\mathrm{AKLT}}^{[L]}
	=
	\begin{pmatrix}
		\Id \\ \bar{\bm O} \\ \bar{\bm Q} \\ 0
	\end{pmatrix}.
	\label{eq:aklt-right-boundary}
\end{equation}
Thus, in the tensor convention of these notes,
\begin{itemize}
	\item the left boundary tensor has shape $(1,14,3,3)$,
	\item each bulk tensor has shape $(14,14,3,3)$,
	\item the right boundary tensor has shape $(14,1,3,3)$.
\end{itemize}
The MPO physical indices are ordered as
\[
W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'},
\]
with $\sigma$ the ket index and $\sigma'$ the bra index, exactly as in the Heisenberg MPO section.

\subsection{Tensor components}

As in the Heisenberg MPO section, the operator-valued entries are converted into numerical MPO tensor components by
\begin{equation}
	W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
	=
	\bra{\sigma}
	\hat W^{[i]}_{\beta_{i-1},\beta_i}
	\ket{\sigma'}.
\end{equation}
Here $\sigma,\sigma'\in\{+1,0,-1\}$ and the local physical dimension is $d=3$.

\subsection{Implementation remarks}

A transparent implementation strategy is:
\begin{enumerate}
	\item construct the local spin-1 matrices $S^+,S^-,S^z$,
	\item define $O_a$ and $\bar O_a$,
	\item build the $9$ operators $\bar O_a\bar O_b$,
	\item build the $9$ operators $O_a O_b$,
	\item assemble the operator-valued boundary and bulk MPOs from
	Eqs.~\eqref{eq:aklt-mpo-block}, \eqref{eq:aklt-left-boundary}, and \eqref{eq:aklt-right-boundary},
	\item expand each operator entry into local matrix elements.
\end{enumerate}

\paragraph{Recommended unit tests.}
For $L=2,3,4$:
\begin{itemize}
	\item contract the MPO to a dense matrix,
	\item compare against the Hamiltonian built directly from Eq.~\eqref{eq:aklt-hamiltonian},
	\item verify that the AKLT MPS given later in these notes has energy
	\[
	E_0 = -\frac{2}{3}(L-1)
	\]
	for open chains, up to numerical precision.
\end{itemize}
```

---
## 🧑‍💻 User

Is it completed?

---
## 🤖 Assistant

Not quite — **one correction is still needed**.

## The issue
In the block MPO I gave, the **left boundary** and **bulk orientation** are not fully consistent with the propagation convention used in your Heisenberg MPO section.

In your note, the XXZ MPO uses the convention:

- left boundary = **row vector**
- right boundary = **column vector**
- bulk propagates from left bond index \(\beta_{i-1}\) to right bond index \(\beta_i\)

For the AKLT MPO, to produce
\[
\sum_i \left[ X_{i,i+1} + \frac13 X_{i,i+1}^2 \right],
\]
the operator inserted on the **left site** of a bond must be the left factor, and the operator inserted on the **right site** must be the right factor.

That means the clean automaton convention should be:

- from the left boundary / start state, emit the **left-site operators**
  \[
  O_a,\qquad \frac13 O_a O_b
  \]
- then, on the following site, complete with
  \[
  \bar O_a,\qquad \bar O_a\bar O_b.
  \]

In the previous section, I had these directions reversed in the block presentation.

---

# Corrected full section

This is the **completed and corrected** version.  
This one is implementation-grade and consistent with your note’s MPO convention.

```latex
\section{AKLT Hamiltonian MPO}

For the spin-1 AKLT chain, the Hamiltonian is
\begin{equation}
	H_{\mathrm{AKLT}}
	=
	\sum_{i=1}^{L-1}
	\left[
	\mathbf S_i\cdot \mathbf S_{i+1}
	+
	\frac13\left(\mathbf S_i\cdot \mathbf S_{i+1}\right)^2
	\right].
	\label{eq:aklt-hamiltonian}
\end{equation}
This is the Hamiltonian whose exact ground state is the AKLT MPS given later in these notes.

\subsection{Local spin-1 operators}

In the local basis
\[
\{\ket{+1},\ket{0},\ket{-1}\},
\]
use the standard spin-1 matrices
\begin{equation}
	S^z
	=
	\begin{pmatrix}
		1 & 0 & 0 \\
		0 & 0 & 0 \\
		0 & 0 & -1
	\end{pmatrix},
\end{equation}
\begin{equation}
	S^+
	=
	\sqrt{2}
	\begin{pmatrix}
		0 & 1 & 0 \\
		0 & 0 & 1 \\
		0 & 0 & 0
	\end{pmatrix},
	\qquad
	S^-
	=
	\sqrt{2}
	\begin{pmatrix}
		0 & 0 & 0 \\
		1 & 0 & 0 \\
		0 & 1 & 0
	\end{pmatrix}.
\end{equation}

Define the bilinear bond operator
\begin{equation}
	X_{i,i+1}
	\equiv
	\mathbf S_i\cdot \mathbf S_{i+1}
	=
	\frac12\left(S_i^+S_{i+1}^- + S_i^-S_{i+1}^+\right)
	+
	S_i^z S_{i+1}^z.
	\label{eq:aklt-bilinear-bond}
\end{equation}
Then
\begin{equation}
	h_{i,i+1}^{\mathrm{AKLT}}
	=
	X_{i,i+1}
	+
	\frac13 X_{i,i+1}^2,
	\label{eq:aklt-local-bond}
\end{equation}
and
\[
H_{\mathrm{AKLT}} = \sum_{i=1}^{L-1} h_{i,i+1}^{\mathrm{AKLT}}.
\]

\subsection{Factorized form of the bond interaction}

Introduce the operator lists
\begin{equation}
	O_1 = \frac{1}{\sqrt{2}}S^+,\qquad
	O_2 = \frac{1}{\sqrt{2}}S^-,\qquad
	O_3 = S^z,
\end{equation}
and
\begin{equation}
	\bar O_1 = \frac{1}{\sqrt{2}}S^-,\qquad
	\bar O_2 = \frac{1}{\sqrt{2}}S^+,\qquad
	\bar O_3 = S^z.
\end{equation}
Then Eq.~\eqref{eq:aklt-bilinear-bond} becomes
\begin{equation}
	X_{i,i+1}
	=
	\sum_{a=1}^3 O_a^{[i]} \bar O_a^{[i+1]}.
	\label{eq:X-factorized}
\end{equation}
Therefore
\begin{equation}
	X_{i,i+1}^2
	=
	\sum_{a=1}^3 \sum_{b=1}^3
	\left(O_a O_b\right)^{[i]}
	\left(\bar O_a \bar O_b\right)^{[i+1]}.
	\label{eq:X2-factorized}
\end{equation}
Substituting Eqs.~\eqref{eq:X-factorized} and \eqref{eq:X2-factorized} into Eq.~\eqref{eq:aklt-local-bond}, we obtain
\begin{equation}
	h_{i,i+1}^{\mathrm{AKLT}}
	=
	\sum_{a=1}^3 O_a^{[i]} \bar O_a^{[i+1]}
	+
	\frac13
	\sum_{a=1}^3 \sum_{b=1}^3
	\left(O_a O_b\right)^{[i]}
	\left(\bar O_a \bar O_b\right)^{[i+1]}.
	\label{eq:aklt-factorized-bond}
\end{equation}

\subsection{Exact MPO construction}

Equation~\eqref{eq:aklt-factorized-bond} is an exact nearest-neighbor MPO. A convenient MPO has bond dimension
\[
\chi = 1 + 3 + 9 + 1 = 14.
\]
This slightly larger bond dimension is chosen because it leads to a completely explicit implementation without block-index ambiguities.

\paragraph{Explicit MPO bond-index convention.}
For implementation, the MPO bond index is ordered as follows:
\begin{equation}
	\beta = 0
	\qquad \text{start channel},
\end{equation}
\begin{equation}
	\beta = 1,2,3
	\qquad \leftrightarrow \qquad
	a=1,2,3,
\end{equation}
\begin{equation}
	\beta = 4,\dots,12
	\qquad \leftrightarrow \qquad
	(a,b)\in
	\{(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)\},
	\label{eq:aklt-lexicographic-order}
\end{equation}
taken in lexicographic order, and
\begin{equation}
	\beta = 13
	\qquad \text{terminal / identity-propagation channel}.
\end{equation}
This convention is used consistently in the bulk tensor and in both boundary tensors.

Define
\begin{equation}
	\bm O
	=
	\begin{pmatrix}
		O_1 & O_2 & O_3
	\end{pmatrix},
	\qquad
	\bar{\bm O}
	=
	\begin{pmatrix}
		\bar O_1 \\ \bar O_2 \\ \bar O_3
	\end{pmatrix},
\end{equation}
\begin{equation}
	\bm O^{(2)}
	=
	\begin{pmatrix}
		O_1 O_1 &
		O_1 O_2 &
		O_1 O_3 &
		O_2 O_1 &
		O_2 O_2 &
		O_2 O_3 &
		O_3 O_1 &
		O_3 O_2 &
		O_3 O_3
	\end{pmatrix},
\end{equation}
and
\begin{equation}
	\bar{\bm O}^{(2)}
	=
	\begin{pmatrix}
		\bar O_1 \bar O_1 \\
		\bar O_1 \bar O_2 \\
		\bar O_1 \bar O_3 \\
		\bar O_2 \bar O_1 \\
		\bar O_2 \bar O_2 \\
		\bar O_2 \bar O_3 \\
		\bar O_3 \bar O_1 \\
		\bar O_3 \bar O_2 \\
		\bar O_3 \bar O_3
	\end{pmatrix}.
\end{equation}
The ordering of the $9$ entries in $\bm O^{(2)}$ and $\bar{\bm O}^{(2)}$ is the same lexicographic ordering defined in Eq.~\eqref{eq:aklt-lexicographic-order}.

A bulk operator-valued MPO tensor is then
\begin{equation}
	\hat W_{\mathrm{AKLT,bulk}}
	=
	\begin{pmatrix}
		\Id & \bm O & \frac13 \bm O^{(2)} & 0 \\
		0 & 0 & 0 & \bar{\bm O} \\
		0 & 0 & 0 & \bar{\bm O}^{(2)} \\
		0 & 0 & 0 & \Id
	\end{pmatrix},
	\label{eq:aklt-mpo-block}
\end{equation}
where the block sizes are $1|3|9|1$.

In words:
\begin{itemize}
	\item the channel $0\to a$ with $a=1,2,3$ starts the bilinear term via $O_a$,
	\item the channel $0\to(a,b)$ starts the biquadratic term via $\frac13 O_a O_b$,
	\item the channel $a\to 13$ finishes the bilinear term via $\bar O_a$,
	\item the channel $(a,b)\to 13$ finishes the biquadratic term via $\bar O_a \bar O_b$,
	\item the channel $13\to 13$ propagates the identity after a term has been completed.
\end{itemize}
With the index convention above, contracting the MPO generates exactly
\[
\sum_{i=1}^{L-1} h_{i,i+1}^{\mathrm{AKLT}}
=
\sum_{i=1}^{L-1}
\left[
\mathbf S_i\cdot \mathbf S_{i+1}
+
\frac13\left(\mathbf S_i\cdot \mathbf S_{i+1}\right)^2
\right].
\]

For OBC, use boundary tensors
\begin{equation}
	\hat W_{\mathrm{AKLT}}^{[1]}
	=
	\begin{pmatrix}
		\Id & \bm O & \frac13 \bm O^{(2)} & 0
	\end{pmatrix},
	\label{eq:aklt-left-boundary}
\end{equation}
and
\begin{equation}
	\hat W_{\mathrm{AKLT}}^{[L]}
	=
	\begin{pmatrix}
		0 \\ \bar{\bm O} \\ \bar{\bm O}^{(2)} \\ \Id
	\end{pmatrix}.
	\label{eq:aklt-right-boundary}
\end{equation}
Thus, in the tensor convention of these notes,
\begin{itemize}
	\item the left boundary tensor has shape $(1,14,3,3)$,
	\item each bulk tensor has shape $(14,14,3,3)$,
	\item the right boundary tensor has shape $(14,1,3,3)$.
\end{itemize}
The MPO physical indices are ordered as
\[
W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'},
\]
with $\sigma$ the ket index and $\sigma'$ the bra index, exactly as in the Heisenberg MPO section.

\subsection{Tensor components}

As in the Heisenberg MPO section, the operator-valued entries are converted into numerical MPO tensor components by
\begin{equation}
	W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'}
	=
	\bra{\sigma}
	\hat W^{[i]}_{\beta_{i-1},\beta_i}
	\ket{\sigma'}.
\end{equation}
Here $\sigma,\sigma'\in\{+1,0,-1\}$ and the local physical dimension is $d=3$.

\subsection{Implementation remarks}

A transparent implementation strategy is:
\begin{enumerate}
	\item construct the local spin-1 matrices $S^+,S^-,S^z$,
	\item define $O_a$ and $\bar O_a$,
	\item build the $9$ operators $O_a O_b$,
	\item build the $9$ operators $\bar O_a\bar O_b$,
	\item assemble the operator-valued boundary and bulk MPOs from
	Eqs.~\eqref{eq:aklt-mpo-block}, \eqref{eq:aklt-left-boundary}, and \eqref{eq:aklt-right-boundary},
	\item expand each operator entry into local matrix elements.
\end{enumerate}

\paragraph{Recommended unit tests.}
For $L=2,3,4$:
\begin{itemize}
	\item contract the MPO to a dense matrix,
	\item compare against the Hamiltonian built directly from Eq.~\eqref{eq:aklt-hamiltonian},
	\item verify that the AKLT MPS given later in these notes has energy
	\[
	E_0 = -\frac{2}{3}(L-1)
	\]
	for open chains, up to numerical precision.
\end{itemize}
```

---

## Final answer
So now: **yes, this version is completed**.

It is:

- mathematically correct,
- consistent with your MPO convention,
- explicit enough for coding,
- and avoids hidden indexing choices.

If you want, I can next do one more pass just to check that this new section is fully consistent with the rest of your document’s notation and doesn’t conflict with your existing Heisenberg MPO section.

---
## 🧑‍💻 User

Now is it ready?

---
## 🤖 Assistant

Yes — **now it is ready**.

More precisely, the AKLT MPO section is now:

- **mathematically correct**,  
- **consistent with your note’s MPO convention**
  \[
  W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'},
  \]
- **explicit enough for coding**, because:
  - the local basis is fixed,
  - the spin-1 operators are fixed,
  - the factorization is fixed,
  - the MPO bond dimension is fixed,
  - the **14 bond channels are explicitly ordered**,
  - left/bulk/right boundary tensors are specified,
  - the ordering of the 9 composite channels is fixed lexicographically.

So as a LaTeX implementation note, **yes, it is ready**.

---

## Very small caveat
“Ready” here means:

- ready to implement directly,
- not that the implementation is guaranteed bug-free without testing.

You should still do the unit tests already listed in the section:

1. contract MPO to dense matrix for \(L=2,3,4\),
2. compare with direct dense Hamiltonian,
3. verify AKLT MPS energy
   \[
   E_0=-\frac23(L-1).
   \]

That is standard and unavoidable for any MPO implementation.

---

## Final verdict
**Yes, the section is now ready to be added to the note and used for coding.**

If you want, the next useful step would be:
- I can check whether the **AKLT benchmark state section** should be slightly cross-linked to this new MPO section, so the note reads more smoothly.

---
## 🧑‍💻 User

Please go through this revised version once more.

---
## 🤖 Assistant

I went through the revised file carefully. Overall it is in **good shape**, and the **AKLT MPO section is now conceptually correct and consistent with your MPO convention**.

However, there are still **a few things I would fix before calling the note fully polished**. The most important ones are not about the AKLT MPO itself, but about consistency and one potentially misleading statement.

---

# Main conclusions

## AKLT MPO section
**Status: good.**  
The revised AKLT MPO section is now:
- consistent with OBC row/column MPO convention,
- explicit enough for coding,
- clear about bond-index ordering,
- mathematically correct.

So on the AKLT MPO question: **yes, that part is now fine**.

---

# Things I recommend fixing

## 1. `\braket` is defined incorrectly
You currently have
```latex
\newcommand{\braket}[2]{\langle #1|#2\rangle}
```
This is wrong: it produces `\langle x|y\rangle`, but the command name `\braket` usually suggests a single argument like `\braket{\psi|\phi}` or, if two-argument, should be `\langle #1 | #2 \rangle`.

### Fix
Replace it with
```latex
\newcommand{\braket}[2]{\langle #1 \mid #2 \rangle}
```

This is minor, but worth correcting.

---

## 2. One-site DMRG right-canonical reshape statement is a bit dangerous
In the one-site DMRG algorithm you write:
```latex
\State reshape $V^\dagger$ using the inverse of Eq.~\eqref{eq:right-group} to obtain a right-canonical tensor $B^{[i]}$ with shape $(D_{i-1}^{\mathrm{new}},d_i,D_i)$
```
This is basically right, but the notation is a little sloppy because after SVD of the right-grouped matrix,
\[
M^{[i]}_{\alpha_{i-1},(\sigma_i\alpha_i)} = U S V^\dagger,
\]
the right-canonical tensor comes from reshaping \(V^\dagger\), whose shape is
\[
(D_{i-1}^{\rm new},\, d_i D_i).
\]
That is okay — but the symbol \(D_{i-1}^{\mathrm{new}}\) should be made consistent with the rank/truncation at that step. Right now it appears without being explicitly defined in that algorithm block.

### Suggested clarification
Replace that line by:
```latex
\State reshape $V^\dagger$ with shape $(D_{i-1}^{\mathrm{new}},\, d_i D_i)$ using the inverse of Eq.~\eqref{eq:right-group} to obtain a right-canonical tensor $B^{[i]}$ of shape $(D_{i-1}^{\mathrm{new}}, d_i, D_i)$
```

That makes it unambiguous.

---

## 3. AKLT benchmark: “zero truncation error at \(D_{\max}\ge 2\)” is a bit too strong
You write:
```latex
\item zero truncation error at $D_{\max}\ge 2$,
```
This is true **if the algorithm stays in the exact AKLT manifold / exact gauge / exact arithmetic**, but as a practical DMRG statement it can be misleading:
- two-site DMRG may temporarily generate larger intermediate bond dimensions before truncation,
- numerical errors or gauge choices may spoil exact zero discarded weight at machine precision,
- the exact AKLT ground space for OBC is degenerate, and if one is not careful, the returned state may not be exactly the simple \(D=2\) MPS gauge you wrote.

### Better wording
I recommend:
```latex
\item the AKLT ground state admits an exact MPS representation with bond dimension $D=2$,
```
and optionally
```latex
\item therefore a correct implementation should recover the ground-state energy exactly (up to numerical precision) once $D_{\max}\ge 2$,
```
That is safer and more implementation-oriented.

---

## 4. AKLT benchmark: edge-state degeneracy statement is fine, but can be sharpened
You write:
```latex
\item for open boundary conditions, the AKLT Hamiltonian has an exactly degenerate edge-state manifold, but every ground state in that manifold has the same exact energy
\[
E_0 = -\frac23 (L-1).
\]
```
This is correct. If you want, you could mention that for OBC the ground-state manifold is **fourfold degenerate**. Not necessary, but useful.

Suggested version:
```latex
\item for open boundary conditions, the AKLT Hamiltonian has a fourfold degenerate ground-state manifold associated with spin-$1/2$ edge modes, and every ground state in that manifold has exact energy
\[
E_0 = -\frac23 (L-1).
\]
```

---

## 5. Heisenberg MPO section: wording about row/column storage is okay, but slightly imprecise
You write:
```latex
\item left boundary tensor shape $(1,5,d,d)$ stores the \emph{last row} of the bulk MPO,
\item right boundary tensor shape $(5,1,d,d)$ stores the \emph{first column} of the bulk MPO.
```
This is consistent with your chosen lower-triangular convention, but it is easy for readers to confuse “first/last” because it depends on whether one thinks of the operator-valued matrix or tensor index direction.

Not wrong, but if you want it sharper, say:
```latex
\item the left boundary tensor is the row vector that selects the bottom row of the bulk operator-valued MPO,
\item the right boundary tensor is the column vector that selects the leftmost column of the bulk operator-valued MPO.
```
This is optional.

---

## 6. Minor notation inconsistency: sometimes \(d\), sometimes \(d_i\)
In most of the document you use \(d_i\) carefully, but in several places you revert to \(d\), e.g.
```latex
\item left boundary tensor shape $(1,5,d,d)$ ...
```
and in some local dimension statements.

This is okay if all sites are homogeneous, but the note earlier sets up the general case with \(d_i\). I suggest either:
- consistently assume uniform local dimension and say so once, or
- keep \(d_i\) and \(d_{i+1}\) throughout.

For spin chains it is harmless, but for polish I’d tighten this.

---

## 7. One-site DMRG algorithm: the sentence about truncation is correct but a bit wordy inside the algorithm environment
This line:
```latex
\State retain at most $D_{\max}$ singular values; in standard one-site DMRG this step mainly regauges the state and shifts the orthogonality center, but does not reliably enlarge the bond space
```
is conceptually good, but too long for an algorithm line.

I’d move the explanation outside the algorithm and leave the algorithm line shorter:

```latex
\State retain at most $D_{\max}$ singular values
```

and after the algorithm add:
```latex
\paragraph{Remark.}
In standard one-site DMRG, this SVD step mainly regauges the state and shifts the orthogonality center; unlike two-site DMRG, it does not naturally enlarge the accessible bond space, which is one reason why one-site DMRG is more prone to local minima unless combined with subspace expansion or density-matrix perturbation.
```

This is a style issue, not a correctness issue.

---

# Minimal must-fix list
If you only want the essentials, I would definitely fix these three:

### (a) fix `\braket`
```latex
\newcommand{\braket}[2]{\langle #1 \mid #2 \rangle}
```

### (b) soften the AKLT truncation claim
Replace
```latex
\item zero truncation error at $D_{\max}\ge 2$,
```
with
```latex
\item the AKLT ground state admits an exact MPS representation with bond dimension $D=2$,
```

### (c) slightly clarify the one-site right-canonical reshape line
Replace that algorithm line as suggested above.

---

# Final verdict
## On your original question
For the AKLT MPO: **yes, the revised version now contains a valid construction**.

## On the note as a whole
It is **very close**, but I would still make the few fixes above before considering it fully cleaned up.

If you want, I can provide a **small exact patch** containing only the lines that should be changed, with before/after replacements.

---
## 🧑‍💻 User

Please provide these exact patches

---
## 🤖 Assistant

Here are the **exact patches** I recommend.

---

## Patch 1: fix `\braket`

### Replace
```latex
\newcommand{\braket}[2]{\langle #1|#2\rangle}
```

### With
```latex
\newcommand{\braket}[2]{\langle #1 \mid #2 \rangle}
```

---

## Patch 2: clarify the one-site DMRG right-canonical reshape line

In the `OneSiteDMRG` algorithm, find this line:

### Replace
```latex
\State reshape $V^\dagger$ using the inverse of Eq.~\eqref{eq:right-group} to obtain a right-canonical tensor $B^{[i]}$ with shape $(D_{i-1}^{\mathrm{new}},d_i,D_i)$
```

### With
```latex
\State reshape $V^\dagger$ with shape $(D_{i-1}^{\mathrm{new}},\, d_i D_i)$ using the inverse of Eq.~\eqref{eq:right-group} to obtain a right-canonical tensor $B^{[i]}$ of shape $(D_{i-1}^{\mathrm{new}}, d_i, D_i)$
```

---

## Patch 3: shorten the long one-site DMRG algorithm line

In the same `OneSiteDMRG` algorithm, find the two occurrences of:

### Replace
```latex
\State retain at most $D_{\max}$ singular values; in standard one-site DMRG this step mainly regauges the state and shifts the orthogonality center, but does not reliably enlarge the bond space
```

### With
```latex
\State retain at most $D_{\max}$ singular values
```

Then, immediately **after** the `algorithm` environment for `OneSiteDMRG`, add:

```latex
\paragraph{Remark.}
In standard one-site DMRG, the SVD step mainly regauges the state and shifts the orthogonality center; unlike two-site DMRG, it does not naturally enlarge the accessible bond space. This is one reason why one-site DMRG is more prone to local minima unless combined with subspace expansion or density-matrix perturbation.
```

---

## Patch 4: soften the AKLT truncation claim

In the `AKLT benchmark state` section, find this bullet list entry:

### Replace
```latex
\item zero truncation error at $D_{\max}\ge 2$,
```

### With
```latex
\item the AKLT ground state admits an exact MPS representation with bond dimension $D=2$,
```

---

## Patch 5: sharpen the AKLT edge-state statement

In the same `AKLT benchmark state` section, replace the last bullet.

### Replace
```latex
\item for open boundary conditions, the AKLT Hamiltonian has an exactly degenerate edge-state manifold, but every ground state in that manifold has the same exact energy
\[
E_0 = -\frac23 (L-1).
\]
```

### With
```latex
\item for open boundary conditions, the AKLT Hamiltonian has a fourfold degenerate ground-state manifold associated with spin-$1/2$ edge modes, and every ground state in that manifold has exact energy
\[
E_0 = -\frac23 (L-1).
\]
```

---

# Optional polish patch

This one is optional, but I recommend it.

## Patch 6: improve wording in the Heisenberg MPO boundary description

In the `Heisenberg MPO` section, find:

### Replace
```latex
\item left boundary tensor shape $(1,5,d,d)$ stores the \emph{last row} of the bulk MPO,
\item right boundary tensor shape $(5,1,d,d)$ stores the \emph{first column} of the bulk MPO.
```

### With
```latex
\item the left boundary tensor has shape $(1,5,d,d)$ and is the row vector selecting the bottom row of the bulk operator-valued MPO,
\item the right boundary tensor has shape $(5,1,d,d)$ and is the column vector selecting the leftmost column of the bulk operator-valued MPO.
```

---

# Summary of must-fix patches
If you want only the essential ones, apply:
- Patch 1
- Patch 2
- Patch 4
- Patch 5

If you want the note polished for readability too, also apply:
- Patch 3
- Patch 6

If you want, I can also give you a **single merged corrected block** for the entire `AKLT benchmark state` section and the `OneSiteDMRG` section so you can paste them directly.

---
## 🧑‍💻 User

Please check this version

---
## 🤖 Assistant

I checked this version carefully. It is **very close**, but there are still **two definite issues** and **one recommended cleanup**.

---

# 1. Definite issue: AKLT benchmark duplicates the \(D=2\) statement

In `\section{AKLT benchmark state}` you currently have

```latex
\item exact bond dimension $D=2$,
\item the AKLT ground state admits an exact MPS representation with bond dimension $D=2$,
```

These say essentially the same thing twice.

## Fix
Replace the whole bullet list by:

```latex
\begin{itemize}
	\item the AKLT ground state admits an exact MPS representation with bond dimension $D=2$,
	\item for open boundary conditions, the AKLT Hamiltonian has a fourfold degenerate ground-state manifold associated with spin-$1/2$ edge modes, and every ground state in that manifold has exact energy
	\[
	E_0 = -\frac23 (L-1).
	\]
\end{itemize}
```

---

# 2. Definite issue: Implementation checklist still says “zero truncation error at \(D_{\max}=2\)”

At the end, in the checklist, you still have:

```latex
\item \textbf{AKLT benchmark}: recover exact energy and zero truncation error at $D_{\max}=2$.
```

This is the statement we previously softened because it is too strong as a practical implementation claim.

## Fix
Replace it with:

```latex
\item \textbf{AKLT benchmark}: recover the exact ground-state energy at $D_{\max}=2$ and verify that the ground state is represented with bond dimension $D=2$ up to numerical precision.
```

That is a robust implementation test.

---

# 3. Recommended cleanup: one-site DMRG uses \(D_{i-1}^{\mathrm{new}}\) without prior definition

In the algorithm line
```latex
\State reshape $V^\dagger$ with shape $(D_{i-1}^{\mathrm{new}},\, d_i D_i)$ ...
```
this is understandable, but \(D_{i-1}^{\mathrm{new}}\) is introduced on the fly.

This is not wrong, but for clarity I recommend adding one short sentence immediately before the algorithm:

```latex
At each SVD step in one-site DMRG, let $D_{i-1}^{\mathrm{new}}$ denote the number of retained singular values after truncation on bond $(i-1,i)$.
```

This is optional but improves readability.

---

# 4. AKLT MPO section itself: checked again

I re-checked the logic of the revised AKLT MPO section. It is now fine:

- left boundary starts with \(\Id,\bm O,\frac13\bm O^{(2)}\),
- bulk propagates unfinished terms to the terminal channel,
- right boundary closes with \(\bar{\bm O}, \bar{\bm O}^{(2)}, \Id\),
- the ordering of the 9 \((a,b)\) channels is fixed.

So on the original substantive question: **yes, the AKLT Hamiltonian is now properly included as an MPO**.

---

# Exact patches

## Patch A: AKLT benchmark bullet list

### Replace
```latex
\begin{itemize}
	\item exact bond dimension $D=2$,
	\item the AKLT ground state admits an exact MPS representation with bond dimension $D=2$,
	\item for open boundary conditions, the AKLT Hamiltonian has a fourfold degenerate ground-state manifold associated with spin-$1/2$ edge modes, and every ground state in that manifold has exact energy
	\[
	E_0 = -\frac23 (L-1).
	\]
\end{itemize}
```

### With
```latex
\begin{itemize}
	\item the AKLT ground state admits an exact MPS representation with bond dimension $D=2$,
	\item for open boundary conditions, the AKLT Hamiltonian has a fourfold degenerate ground-state manifold associated with spin-$1/2$ edge modes, and every ground state in that manifold has exact energy
	\[
	E_0 = -\frac23 (L-1).
	\]
\end{itemize}
```

---

## Patch B: implementation checklist AKLT line

### Replace
```latex
\item \textbf{AKLT benchmark}: recover exact energy and zero truncation error at $D_{\max}=2$.
```

### With
```latex
\item \textbf{AKLT benchmark}: recover the exact ground-state energy at $D_{\max}=2$ and verify that the ground state is represented with bond dimension $D=2$ up to numerical precision.
```

---

## Patch C: optional clarity sentence before one-site algorithm

Insert immediately before
```latex
\begin{algorithm}[H]
	\caption{OneSiteDMRG}
```

the sentence:

```latex
At each SVD step in one-site DMRG, let $D_{i-1}^{\mathrm{new}}$ denote the number of retained singular values after truncation on bond $(i-1,i)$.
```

---

# Final verdict

## Substantive answer
- **AKLT MPO section:** good
- **Document overall:** almost done

## After applying Patch A and Patch B
I would consider this **ready**.

If you want, I can do one final pass for **LaTeX compilation hygiene only** — checking for duplicated labels, awkward spacing, and notation consistency.

---
## 🧑‍💻 User

pls do final pass

---
## 🤖 Assistant

Here is a final-pass check focused on **compilation hygiene, notation consistency, and small editorial issues**. The document is basically ready, but I recommend the following final tweaks.

---

## 1) Apply the two content fixes from the previous pass

These are the only substantive “must-fix” items left.

### (a) AKLT benchmark duplicated bullet
Replace
```latex
\begin{itemize}
	\item exact bond dimension $D=2$,
	\item the AKLT ground state admits an exact MPS representation with bond dimension $D=2$,
	\item for open boundary conditions, the AKLT Hamiltonian has a fourfold degenerate ground-state manifold associated with spin-$1/2$ edge modes, and every ground state in that manifold has exact energy
	\[
	E_0 = -\frac23 (L-1).
	\]
\end{itemize}
```
with
```latex
\begin{itemize}
	\item the AKLT ground state admits an exact MPS representation with bond dimension $D=2$,
	\item for open boundary conditions, the AKLT Hamiltonian has a fourfold degenerate ground-state manifold associated with spin-$1/2$ edge modes, and every ground state in that manifold has exact energy
	\[
	E_0 = -\frac23 (L-1).
	\]
\end{itemize}
```

### (b) Checklist line
Replace
```latex
\item \textbf{AKLT benchmark}: recover exact energy and zero truncation error at $D_{\max}=2$.
```
with
```latex
\item \textbf{AKLT benchmark}: recover the exact ground-state energy at $D_{\max}=2$ and verify that the ground state is represented with bond dimension $D=2$ up to numerical precision.
```

---

## 2) Add the one-site definition sentence for \(D_{i-1}^{\mathrm{new}}\)

Not mandatory for compilation, but recommended for clarity.

Insert immediately before
```latex
\begin{algorithm}[H]
	\caption{OneSiteDMRG}
```
the sentence
```latex
At each SVD step in one-site DMRG, let $D_{i-1}^{\mathrm{new}}$ denote the number of retained singular values after truncation on bond $(i-1,i)$.
```

---

## 3) Minor notation consistency: `\bra{\sigma}\hat O\ket{\sigma'}` vs ket/bra interpretation

Your MPO convention is consistent throughout:
\[
\hat W^{[i]} = \sum_{\sigma,\sigma'} W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'} \ket{\sigma}\bra{\sigma'}.
\]
And later you define components via
\[
W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'} = \bra{\sigma}\hat W^{[i]}_{\beta_{i-1},\beta_i}\ket{\sigma'}.
\]
That is correct.

So **no change needed**, but I checked this specifically because this is where many notes become inconsistent. Yours is fine.

---

## 4) Compilation hygiene: `\bm O` and `\bar{\bm O}` are okay

You use `\bm O`, `\bar{\bm O}`, `\bm O^{(2)}`, `\bar{\bm O}^{(2)}` in the AKLT MPO section. This is fine with `\usepackage{bm}` already loaded.

**No change needed.**

---

## 5) Equation labels: checked for obvious collisions

I checked the label names in the pasted version. I do **not** see duplicated labels among the visible equations. In particular:
- `eq:theta-def` and `eq:theta-def-alg` are distinct,
- `eq:two-site-state` and `eq:two-site-state-alg` are distinct,
- AKLT labels are distinct from Heisenberg labels.

**No obvious duplicate-label problem.**

---

## 6) Slight wording cleanup in AKLT benchmark section

Not necessary, but this reads a bit smoother:

### Current
```latex
A standard exact MPS representation of the AKLT state is given, up to gauge, by
```

### Recommended
```latex
An exact MPS representation of the AKLT state is, up to gauge, given by
```

This is purely stylistic.

---

## 7) Slight wording cleanup in `Reshape conventions`

You define “Typical inverse reshapes”
```latex
A  = U.reshape((D_left, d, D_new), order='C')
B  = Vh.reshape((D_new, d, D_right), order='C')
```
This is okay, but only if the grouped matrix was formed consistently. Since you already emphasized consistency, this is acceptable.

**No change needed.**

---

## 8) Algorithm package compatibility

You use both:
```latex
\usepackage{algorithm}
\usepackage{algpseudocode}
```
and
```latex
\providecommand{\ALG@name}{Algorithm}
```
This is standard and should compile fine in `revtex4-2`.

**No change needed.**

---

## 9) A very small style issue: `$$ ... $$`

You have a couple of display equations written with double dollar signs in the matrix-free eigensolver section:
```latex
$$
M^{[i]} \in \mathbb{C}^{D_{i-1}\times d_i \times D_i},
$$
```
and
```latex
$$
N_{\mathrm{loc}} = D_{i-1} d_i D_i.
$$
```

These will compile, but in LaTeX it is better to use `\[ ... \]`.

### Recommended replacements
Replace
```latex
$$
M^{[i]} \in \mathbb{C}^{D_{i-1}\times d_i \times D_i},
$$
```
with
```latex
\[
M^{[i]} \in \mathbb{C}^{D_{i-1}\times d_i \times D_i},
\]
```
and replace
```latex
$$
N_{\mathrm{loc}} = D_{i-1} d_i D_i.
$$
```
with
```latex
\[
N_{\mathrm{loc}} = D_{i-1} d_i D_i.
\]
```

This is not a correctness issue, but it is good LaTeX hygiene.

---

## 10) Final ready status

After applying:
- Patch A (AKLT duplicated bullet),
- Patch B (checklist line),
- Patch C (optional one-site clarification),
- and optionally replacing the two `$$...$$` displays,

I would consider the document **ready**.

---

# Minimal final patch set

If you want the minimal final set, do exactly these:

### Patch A
```latex
\begin{itemize}
	\item the AKLT ground state admits an exact MPS representation with bond dimension $D=2$,
	\item for open boundary conditions, the AKLT Hamiltonian has a fourfold degenerate ground-state manifold associated with spin-$1/2$ edge modes, and every ground state in that manifold has exact energy
	\[
	E_0 = -\frac23 (L-1).
	\]
\end{itemize}
```

### Patch B
```latex
\item \textbf{AKLT benchmark}: recover the exact ground-state energy at $D_{\max}=2$ and verify that the ground state is represented with bond dimension $D=2$ up to numerical precision.
```

### Patch C
```latex
At each SVD step in one-site DMRG, let $D_{i-1}^{\mathrm{new}}$ denote the number of retained singular values after truncation on bond $(i-1,i)$.
```

### Patch D (optional LaTeX hygiene)
Replace the two `$$...$$` displays by `\[...\]`.

---

## Final verdict
**Yes — after those tiny last fixes, this note is ready.**

If you want, I can provide the **exact corrected text for the AKLT benchmark section and the checklist section as a single copy-paste block**.

---
## 🧑‍💻 User

provide the exact corrected text for the AKLT benchmark section and the checklist section as a single copy-paste block

---
## 🤖 Assistant

```latex
\section{AKLT benchmark state}

An exact MPS representation of the AKLT state is, up to gauge, given by
\begin{align}
	A^{+} &= \begin{pmatrix} 0 & \sqrt{2/3} \\ 0 & 0 \end{pmatrix},\\
	A^{0} &= \begin{pmatrix} -1/\sqrt{3} & 0 \\ 0 & 1/\sqrt{3} \end{pmatrix},\\
	A^{-} &= \begin{pmatrix} 0 & 0 \\ -\sqrt{2/3} & 0 \end{pmatrix}.
\end{align}
For this gauge, the tensors satisfy
\begin{equation}
	\sum_{\sigma\in\{+,0,-\}} A^\sigma A^{\sigma\dagger} = \Id_2,
\end{equation}
so they are \emph{right-canonical} in the convention of Eq.~\eqref{eq:right-canonical}.

This state is an excellent benchmark:
\begin{itemize}
	\item the AKLT ground state admits an exact MPS representation with bond dimension $D=2$,
	\item for open boundary conditions, the AKLT Hamiltonian has a fourfold degenerate ground-state manifold associated with spin-$1/2$ edge modes, and every ground state in that manifold has exact energy
	\[
	E_0 = -\frac23 (L-1).
	\]
\end{itemize}

\section{Implementation checklist}

Before production runs, validate in this order:
\begin{enumerate}
	\item \textbf{Canonicalization test}: random tensor $\to$ QR/SVD sweep $\to$ check Eqs.~\eqref{eq:left-canonical} and \eqref{eq:right-canonical}.
	\item \textbf{MPO test}: contract MPO to a dense Hamiltonian for small $L$ and compare with exact construction.
	\item \textbf{Environment test}: verify left/right environments against brute-force contraction for small $L$.
	\item \textbf{Hermiticity test of the matrix-free local solver}: verify numerically for random small tensors that
	\[
	\langle x, \mathcal{H}_{\mathrm{eff}} y\rangle
	=
	\langle \mathcal{H}_{\mathrm{eff}} x, y\rangle.
	\]
	\item \textbf{One-site effective Hamiltonian test}: compare matrix-free action against explicitly constructed $\mathcal{H}_{\mathrm{eff}}$ for small dimensions.
	\item \textbf{Two-site effective Hamiltonian test}: compare matrix-free action against explicitly constructed $\mathcal{H}_{\mathrm{eff}}$ for small dimensions.
	\item \textbf{Two-site DMRG benchmark}: recover exact diagonalization energies for small Heisenberg chains.
	\item \textbf{AKLT benchmark}: recover the exact ground-state energy at $D_{\max}=2$ and verify that the ground state is represented with bond dimension $D=2$ up to numerical precision.
\end{enumerate}
```

---
## 🧑‍💻 User

Is it completed? where to put it?

---
## 🤖 Assistant

Yes — **this replacement block is completed**.

## Where to put it
It is meant to **replace two existing sections** in your file:

1. replace the current section
```latex
\section{AKLT benchmark state}
...
```

2. replace the current section
```latex
\section{Implementation checklist}
...
```

So:

- find `\section{AKLT benchmark state}`
- delete everything from there **through the end of** the current `\section{Implementation checklist}`
- paste the new block in exactly that place

## More explicitly
In your current file, replace this whole region:

```latex
\section{AKLT benchmark state}

... old text ...

\section{Implementation checklist}

... old text ...
```

with the new block I gave.

---

## Order relative to the rest of the document
The correct order after replacement should remain:

```latex
\section{One-site DMRG}
...
\section{AKLT benchmark state}
...
\section{Implementation checklist}
...
\section{Practical coding recommendation}
...
```

So the new block goes **between**
- `\section{One-site DMRG}`
and
- `\section{Practical coding recommendation}`

exactly where the old two sections already are.

---

## Final status
After inserting that block, your document is essentially complete.

### One optional last cleanup
You still have two `$$ ... $$` displays in the `Matrix-free eigensolver interface for local effective Hamiltonians` section. They will compile, but if you want proper LaTeX style, replace them with `\[ ... \]`.

If you want, I can give you those exact two tiny replacements too.
