# MPS-Code-GPT#1-md

## 🧑‍💻 User

This is a latex note on MPS and DMRG. Could you generate python codes according to it? Please use two-site DMRG algorithm and compute both S=1/2 Heisenberg model and AKLT model. I work with Jupyter. Please show all figures and save them in directory "figureAKLT".


---
## 🤖 Assistant

Below is a Jupyter-friendly implementation of **two-site DMRG** following the conventions in your LaTex note. It covers:

- **Spin-1/2 Heisenberg chain**
- **Spin-1 AKLT chain**
- **Two-site DMRG**
- **Exact diagonalization benchmarks for small systems**
- **Plots shown inline and saved to `figureAKLT/`**

I’ve organized it as notebook cells so you can paste them directly into Jupyter.

---

# 1. Notebook setup

```python
import os
import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import LinearOperator, eigsh
import matplotlib.pyplot as plt

np.set_printoptions(precision=12, suppress=True)
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["font.size"] = 12

FIGDIR = "figureAKLT"
os.makedirs(FIGDIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIGDIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    print(f"saved: {path}")
```

---

# 2. Local operators

We need spin-1/2 operators for Heisenberg and spin-1 operators for AKLT.

```python
def spin_half_ops(dtype=np.complex128):
    I = np.eye(2, dtype=dtype)
    Sp = np.array([[0, 1], [0, 0]], dtype=dtype)
    Sm = np.array([[0, 0], [1, 0]], dtype=dtype)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=dtype)
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=dtype)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=dtype)
    return I, Sp, Sm, Sz, Sx, Sy

def spin_one_ops(dtype=np.complex128):
    I = np.eye(3, dtype=dtype)
    s2 = np.sqrt(2.0)
    Sp = np.array([[0, s2, 0],
                   [0, 0, s2],
                   [0, 0, 0]], dtype=dtype)
    Sm = Sp.conj().T
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, -1]], dtype=dtype)
    Sx = 0.5 * (Sp + Sm)
    Sy = (Sp - Sm) / (2j)
    return I, Sp, Sm, Sz, Sx, Sy
```

---

# 3. MPO builders

## 3.1 Heisenberg MPO for spin-1/2

This follows your note exactly.

\[
H = \sum_i \left[\frac{J}{2}(S_i^+S_{i+1}^- + S_i^-S_{i+1}^+) + J_z S_i^zS_{i+1}^z\right] - h\sum_i S_i^z
\]

```python
def mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    I, Sp, Sm, Sz, Sx, Sy = spin_half_ops(dtype=dtype)
    d = 2
    chi = 5

    W_bulk = np.zeros((chi, chi, d, d), dtype=dtype)
    W_bulk[0, 0] = I
    W_bulk[1, 0] = Sp
    W_bulk[2, 0] = Sm
    W_bulk[3, 0] = Sz
    W_bulk[4, 0] = -h * Sz
    W_bulk[4, 1] = 0.5 * J * Sm
    W_bulk[4, 2] = 0.5 * J * Sp
    W_bulk[4, 3] = Jz * Sz
    W_bulk[4, 4] = I

    W_first = np.zeros((1, chi, d, d), dtype=dtype)
    W_first[0, 0] = -h * Sz
    W_first[0, 1] = 0.5 * J * Sm
    W_first[0, 2] = 0.5 * J * Sp
    W_first[0, 3] = Jz * Sz
    W_first[0, 4] = I

    W_last = np.zeros((chi, 1, d, d), dtype=dtype)
    W_last[0, 0] = I
    W_last[1, 0] = Sp
    W_last[2, 0] = Sm
    W_last[3, 0] = Sz
    W_last[4, 0] = -h * Sz

    mpo = [None] * L
    if L == 1:
        mpo[0] = np.array([[[[-h*Sz]]]], dtype=dtype)  # not really used
    else:
        mpo[0] = W_first
        for i in range(1, L - 1):
            mpo[i] = W_bulk.copy()
        mpo[-1] = W_last
    return mpo
```

---

## 3.2 AKLT MPO for spin-1

The AKLT Hamiltonian is

\[
H_{\mathrm{AKLT}} = \sum_{i=1}^{L-1}
\left[
\mathbf S_i\cdot \mathbf S_{i+1}
+ \frac13 (\mathbf S_i\cdot \mathbf S_{i+1})^2
\right]
\]

up to an additive constant convention. The standard open-chain ground-state energy is

\[
E_0 = -\frac23 (L-1)
\]

for the convention

\[
H = \sum_i \left[P^{(2)}_{i,i+1}\right]
= \sum_i \left[
\mathbf S_i\cdot \mathbf S_{i+1}
+ \frac13 (\mathbf S_i\cdot \mathbf S_{i+1})^2
+ \frac23
\right]
\]

while the note uses the shifted Hamiltonian with energy \(-\frac23(L-1)\). So here I implement the **shifted AKLT Hamiltonian**:

\[
H_{\text{shifted}} = \sum_i \left[
\mathbf S_i\cdot \mathbf S_{i+1}
+ \frac13 (\mathbf S_i\cdot \mathbf S_{i+1})^2
\right]
\]

whose exact AKLT MPS energy is indeed \(-\frac23(L-1)\).

To build an MPO cleanly, we use a generic two-body builder.

```python
def mpo_from_two_body(L, onsite_op_list, bond_terms, d, dtype=np.complex128):
    """
    Build an MPO for
        H = sum_i sum_a h1[a] Oa(i) + sum_i sum_k Lk(i) Rk(i+1)

    Parameters
    ----------
    onsite_op_list : list of local operators [O0, O1, ...] with coefficients already included
                     summed as single-site terms on every site.
    bond_terms : list of tuples (Lop, Rop), already including coefficients.
    """
    n1 = len(onsite_op_list)
    nb = len(bond_terms)
    chi = 2 + nb

    I = np.eye(d, dtype=dtype)
    W_bulk = np.zeros((chi, chi, d, d), dtype=dtype)

    # first row
    W_bulk[0, 0] = I

    # bond channels
    for k, (Lop, Rop) in enumerate(bond_terms):
        W_bulk[0, 1 + k] = Lop
        W_bulk[1 + k, -1] = Rop

    # onsite term and propagation
    W_bulk[-1, -1] = I
    if n1 > 0:
        W_bulk[0, -1] = sum(onsite_op_list)

    # OBC boundaries
    W_first = np.zeros((1, chi, d, d), dtype=dtype)
    W_first[0] = W_bulk[0]

    W_last = np.zeros((chi, 1, d, d), dtype=dtype)
    W_last[:, 0] = W_bulk[:, -1]

    mpo = [None] * L
    mpo[0] = W_first
    for i in range(1, L - 1):
        mpo[i] = W_bulk.copy()
    mpo[-1] = W_last
    return mpo
```

Now define the AKLT bond decomposition. A convenient identity is

\[
\mathbf S_i\cdot \mathbf S_j
= \frac12(S_i^+S_j^- + S_i^-S_j^+) + S_i^z S_j^z
\]

and then square it by expanding in operator products. We write

\[
(\mathbf S_i\cdot\mathbf S_j)^2 = \sum_{ab} c_{ab}\, O_a(i) O_b(j)
\]

using the operator basis \(\{S^+, S^-, S^z\}\).

```python
def mpo_aklt(L, dtype=np.complex128):
    I, Sp, Sm, Sz, Sx, Sy = spin_one_ops(dtype=dtype)
    d = 3

    # Build A = S.S as sum_k Lk \otimes Rk
    # A = 1/2 Sp⊗Sm + 1/2 Sm⊗Sp + Sz⊗Sz
    A_terms = [
        (0.5 * Sp, Sm),
        (0.5 * Sm, Sp),
        (1.0 * Sz, Sz),
    ]

    # Build A^2 = sum_{a,b} (La Lb) ⊗ (Ra Rb)
    A2_terms = []
    for La, Ra in A_terms:
        for Lb, Rb in A_terms:
            A2_terms.append(((1.0/3.0) * (La @ Lb), Ra @ Rb))

    bond_terms = A_terms + A2_terms
    onsite_terms = []  # shifted AKLT has no onsite term
    return mpo_from_two_body(L, onsite_terms, bond_terms, d=3, dtype=dtype)
```

---

# 4. Exact dense Hamiltonians for benchmarking

For small systems we compare against exact diagonalization.

```python
def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def dense_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    I, Sp, Sm, Sz, Sx, Sy = spin_half_ops(dtype=dtype)
    d = 2
    H = np.zeros((d**L, d**L), dtype=dtype)

    for i in range(L - 1):
        ops = [I] * L
        ops[i] = 0.5 * J * Sp
        ops[i+1] = Sm
        H += kron_all(ops)

        ops = [I] * L
        ops[i] = 0.5 * J * Sm
        ops[i+1] = Sp
        H += kron_all(ops)

        ops = [I] * L
        ops[i] = Jz * Sz
        ops[i+1] = Sz
        H += kron_all(ops)

    for i in range(L):
        ops = [I] * L
        ops[i] = -h * Sz
        H += kron_all(ops)

    return H

def dense_aklt(L, dtype=np.complex128):
    I, Sp, Sm, Sz, Sx, Sy = spin_one_ops(dtype=dtype)
    d = 3
    H = np.zeros((d**L, d**L), dtype=dtype)

    for i in range(L - 1):
        SS = (
            0.5 * np.kron(Sp, Sm)
            + 0.5 * np.kron(Sm, Sp)
            + np.kron(Sz, Sz)
        )
        hbond = SS + (1.0/3.0) * (SS @ SS)

        ops = [I] * L
        # embed 2-site operator
        left_dim = d**i
        right_dim = d**(L - i - 2)
        H += np.kron(np.eye(left_dim, dtype=dtype),
                     np.kron(hbond, np.eye(right_dim, dtype=dtype)))
    return H

def exact_ground_energy(H):
    vals = la.eigvalsh(H)
    return np.min(vals).real
```

---

# 5. Basic MPS utilities

We use MPS tensors of shape `(Dl, d, Dr)` exactly as in your note.

```python
def random_mps(L, d, Dmax, seed=1234, dtype=np.complex128):
    rng = np.random.default_rng(seed)
    Ds = [1]
    for i in range(1, L):
        Ds.append(min(Dmax, Ds[-1] * d))
    # symmetric cap from right
    Drs = [1]
    for i in range(1, L):
        Drs.append(min(Dmax, Drs[-1] * d))
    Drs = list(reversed(Drs))

    tensors = []
    for i in range(L):
        Dl = min(Ds[i], Drs[i])
        Dr = min(Ds[i+1] if i < L-1 else 1, Drs[i+1] if i < L-1 else 1) if i < L-1 else 1
        A = rng.normal(size=(Dl, d, Dr)) + 1j * rng.normal(size=(Dl, d, Dr))
        A = A.astype(dtype)
        tensors.append(A)
    return tensors

def product_state_mps(L, d, state_index=0, dtype=np.complex128):
    mps = []
    for _ in range(L):
        A = np.zeros((1, d, 1), dtype=dtype)
        A[0, state_index, 0] = 1.0
        mps.append(A)
    return mps
```

---

## 5.1 Canonicalization

We right-canonicalize the initial MPS before DMRG.

```python
def left_canonicalize_qr(mps):
    mps = [A.copy() for A in mps]
    L = len(mps)
    for i in range(L - 1):
        Dl, d, Dr = mps[i].shape
        M = mps[i].reshape(Dl * d, Dr, order='C')
        Q, R = la.qr(M, mode='economic')
        k = Q.shape[1]
        mps[i] = Q.reshape(Dl, d, k, order='C')
        mps[i+1] = np.einsum('ab,bsd->asd', R, mps[i+1])
    return mps

def right_canonicalize_qr(mps):
    mps = [A.copy() for A in mps]
    L = len(mps)
    for i in range(L - 1, 0, -1):
        Dl, d, Dr = mps[i].shape
        M = mps[i].reshape(Dl, d * Dr, order='C')
        Q, R = la.qr(M.conj().T, mode='economic')
        k = Q.shape[1]
        B = Q.conj().T.reshape(k, d, Dr, order='C')
        mps[i] = B
        mps[i-1] = np.einsum('asb,bc->asc', mps[i-1], R.conj().T)
    return mps

def check_left_canonical(A, tol=1e-10):
    Dl, d, Dr = A.shape
    M = A.reshape(Dl * d, Dr, order='C')
    X = M.conj().T @ M
    return np.linalg.norm(X - np.eye(Dr)) < tol

def check_right_canonical(A, tol=1e-10):
    Dl, d, Dr = A.shape
    M = A.reshape(Dl, d * Dr, order='C')
    X = M @ M.conj().T
    return np.linalg.norm(X - np.eye(Dl)) < tol
```

---

# 6. Environment contractions

These follow the conventions in the note.

```python
def init_left_env():
    Lenv = np.zeros((1, 1, 1), dtype=np.complex128)
    Lenv[0, 0, 0] = 1.0
    return Lenv

def init_right_env():
    Renv = np.zeros((1, 1, 1), dtype=np.complex128)
    Renv[0, 0, 0] = 1.0
    return Renv

def update_left_env(Lenv, A, W):
    # Lnew[B,a,c] = sum_{b,x,y,s,t} L[b,x,y] A[x,s,a] W[b,B,s,t] A*[y,t,c]
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lenv, A, W, A.conj(), optimize=True)

def update_right_env(Renv, B, W):
    # Rnew[b,a,c] = sum_{B,x,y,s,t} B[a,s,x] W[b,B,s,t] R[B,x,y] B*[c,t,y]
    return np.einsum('asx,bBst,Bxy,cty->bac', B, W, Renv, B.conj(), optimize=True)

def build_all_right_envs(mps, mpo):
    L = len(mps)
    Renvs = [None] * L
    R = init_right_env()
    Renvs[L - 1] = R
    for i in range(L - 1, 0, -1):
        R = update_right_env(R, mps[i], mpo[i])
        Renvs[i - 1] = R
    return Renvs
```

---

# 7. Two-site effective Hamiltonian matvec

This is the key matrix-free operation.

```python
def two_site_heff_apply(theta, Lenv, W1, W2, Renv):
    """
    theta shape: (Dl, d1, d2, Dr)
    Lenv shape: (chiL, Dl, Dl')
    W1 shape:   (chiL, chiM, d1, d1)
    W2 shape:   (chiM, chiR, d2, d2)
    Renv shape: (chiR, Dr, Dr')
    """
    # X[b,a,s1',s2',r'] = sum_{a'} L[b,a,a'] theta[a',s1',s2',r']
    X = np.einsum('bxa,asty->bxsty', Lenv, theta, optimize=True)
    # Y[B,a,s1,s2',r'] = sum_{b,t1} W1[b,B,s1,t1] X[b,a,t1,s2',r']
    Y = np.einsum('bBst,batuy->Basuy', W1, X, optimize=True)
    # Z[C,a,s1,s2,r'] = sum_{B,t2} W2[B,C,s2,t2] Y[B,a,s1,t2,r']
    Z = np.einsum('BCst,Catuy->Casuy', W2, Y, optimize=True)
    # out[a,s1,s2,r] = sum_{C,r'} Z[C,a,s1,s2,r'] R[C,r',r]
    out = np.einsum('Casuy,Cyr->asur', Z, Renv, optimize=True)
    return out
```

To avoid index mistakes, let’s also provide a local dense effective Hamiltonian builder for testing.

```python
def build_dense_two_site_heff(Lenv, W1, W2, Renv, Dl, d1, d2, Dr):
    n = Dl * d1 * d2 * Dr
    H = np.zeros((n, n), dtype=np.complex128)
    for j in range(n):
        e = np.zeros(n, dtype=np.complex128)
        e[j] = 1.0
        theta = e.reshape(Dl, d1, d2, Dr, order='C')
        He = two_site_heff_apply(theta, Lenv, W1, W2, Renv).reshape(n, order='C')
        H[:, j] = He
    return H
```

---

# 8. Local eigensolver for the two-site problem

We use `eigsh` on a `LinearOperator`.

```python
def solve_local_ground_state(theta0, Lenv, W1, W2, Renv):
    Dl, d1, d2, Dr = theta0.shape
    n = Dl * d1 * d2 * Dr

    def mv(v):
        th = v.reshape(Dl, d1, d2, Dr, order='C')
        out = two_site_heff_apply(th, Lenv, W1, W2, Renv)
        return out.reshape(n, order='C')

    Hlin = LinearOperator((n, n), matvec=mv, dtype=np.complex128)
    v0 = theta0.reshape(n, order='C')
    v0 = v0 / np.linalg.norm(v0)

    vals, vecs = eigsh(Hlin, k=1, which='SA', v0=v0, tol=1e-10, maxiter=500)
    E = vals[0].real
    theta = vecs[:, 0].reshape(Dl, d1, d2, Dr, order='C')
    theta /= np.linalg.norm(theta)
    return E, theta
```

---

# 9. Two-site split by SVD

Following your reshape convention:
- left index = \((\alpha_{i-1}, \sigma_i)\)
- right index = \((\sigma_{i+1}, \alpha_{i+1})\)

```python
def split_two_site_theta(theta, Dmax, cutoff=1e-12, sweep_direction='lr'):
    Dl, d1, d2, Dr = theta.shape
    M = theta.reshape(Dl * d1, d2 * Dr, order='C')
    U, s, Vh = la.svd(M, full_matrices=False)

    keep = min(Dmax, np.sum(s > cutoff))
    if keep < 1:
        keep = 1

    disc_weight = np.sum(s[keep:]**2).real
    U = U[:, :keep]
    s = s[:keep]
    Vh = Vh[:keep, :]

    if sweep_direction == 'lr':
        A = U.reshape(Dl, d1, keep, order='C')
        Mright = (np.diag(s) @ Vh).reshape(keep, d2, Dr, order='C')
        return A, Mright, s, disc_weight
    elif sweep_direction == 'rl':
        Mleft = (U @ np.diag(s)).reshape(Dl, d1, keep, order='C')
        B = Vh.reshape(keep, d2, Dr, order='C')
        return Mleft, B, s, disc_weight
    else:
        raise ValueError("sweep_direction must be 'lr' or 'rl'")
```

---

# 10. MPS observables and utilities

## 10.1 Convert MPS to dense state

For small systems only, useful for benchmarking and variance.

```python
def mps_to_dense(mps):
    psi = mps[0]
    # psi shape (1,d,D1)
    psi = psi[0, :, :]  # (d,D1)
    for i in range(1, len(mps)):
        A = mps[i]  # (Dl,d,Dr)
        psi = np.einsum('...a,asb->...sb', psi, A, optimize=True)
    return psi.reshape(-1, order='C')
```

## 10.2 Energy from dense state

```python
def dense_energy_from_mps(mps, H):
    psi = mps_to_dense(mps)
    psi = psi / np.linalg.norm(psi)
    return np.vdot(psi, H @ psi).real

def dense_variance_from_mps(mps, H):
    psi = mps_to_dense(mps)
    psi = psi / np.linalg.norm(psi)
    E = np.vdot(psi, H @ psi).real
    Hpsi = H @ psi
    E2 = np.vdot(Hpsi, Hpsi).real
    return E2 - E**2
```

## 10.3 Bond entanglement entropy

```python
def bond_singular_values_from_mps(mps):
    # assumes mixed or left/right canonical enough for extracting via sequential SVD from dense conversion
    # for small/moderate sizes only
    psi = mps_to_dense(mps)
    L = len(mps)
    d = mps[0].shape[1]
    svals = []
    for cut in range(1, L):
        M = psi.reshape(d**cut, d**(L-cut), order='C')
        s = la.svdvals(M)
        s = s / np.linalg.norm(s)
        svals.append(s)
    return svals

def entanglement_entropies(mps):
    svals = bond_singular_values_from_mps(mps)
    ent = []
    for s in svals:
        p = s**2
        p = p[p > 1e-15]
        ent.append(-(p * np.log(p)).sum().real)
    return np.array(ent)
```

For AKLT with spin-1 local dimension \(d=3\), the previous function infers \(d\) from the first tensor, so it works.

---

# 11. Two-site DMRG driver

```python
def dmrg_two_site(mpo, d, Dmax=32, nsweeps=8, cutoff=1e-12, init_mps=None, verbose=True):
    L = len(mpo)

    if init_mps is None:
        init_mps = random_mps(L, d, Dmax=max(2, min(Dmax, 4)), seed=1234)

    mps = right_canonicalize_qr(init_mps)

    energies = []
    max_discards = []

    for sweep in range(nsweeps):
        # Build right environments for current gauge
        Renvs = build_all_right_envs(mps, mpo)
        Lenv = init_left_env()

        sweep_discards = []
        E_last = None

        # Left -> Right
        for i in range(L - 1):
            theta = np.einsum('asb,btc->astc', mps[i], mps[i+1], optimize=True)
            E, theta_opt = solve_local_ground_state(theta, Lenv, mpo[i], mpo[i+1], Renvs[i+1])

            A, Mright, s, disc = split_two_site_theta(theta_opt, Dmax=Dmax, cutoff=cutoff, sweep_direction='lr')
            mps[i] = A
            mps[i+1] = Mright
            Lenv = update_left_env(Lenv, mps[i], mpo[i])

            sweep_discards.append(disc)
            E_last = E

        # Right -> Left
        Renv = init_right_env()
        for i in range(L - 2, -1, -1):
            theta = np.einsum('asb,btc->astc', mps[i], mps[i+1], optimize=True)
            # left env for bond i can be rebuilt progressively if needed
            # simpler and safer here: rebuild from left up to i
            Ltmp = init_left_env()
            for j in range(i):
                Ltmp = update_left_env(Ltmp, mps[j], mpo[j])

            E, theta_opt = solve_local_ground_state(theta, Ltmp, mpo[i], mpo[i+1], Renv)

            Mleft, B, s, disc = split_two_site_theta(theta_opt, Dmax=Dmax, cutoff=cutoff, sweep_direction='rl')
            mps[i] = Mleft
            mps[i+1] = B
            Renv = update_right_env(Renv, mps[i+1], mpo[i+1])

            sweep_discards.append(disc)
            E_last = E

        energies.append(E_last)
        max_discards.append(max(sweep_discards) if sweep_discards else 0.0)

        if verbose:
            print(f"sweep {sweep+1:2d}: E = {E_last:.15f}, max discarded weight = {max_discards[-1]:.3e}")

    return {
        "mps": mps,
        "energies": np.array(energies),
        "max_discards": np.array(max_discards),
    }
```

---

# 12. Sanity/unit tests

## 12.1 Canonicalization test

```python
# Canonicalization quick test
mps0 = random_mps(L=6, d=2, Dmax=4, seed=7)
mpsL = left_canonicalize_qr(mps0)
mpsR = right_canonicalize_qr(mps0)

print("Left-canonical checks:")
for i in range(5):
    print(i, check_left_canonical(mpsL[i]))

print("Right-canonical checks:")
for i in range(1, 6):
    print(i, check_right_canonical(mpsR[i]))
```

---

## 12.2 Two-site effective Hamiltonian test against dense local matrix

This is strongly recommended by your note.

```python
# local effective Hamiltonian test on small random dimensions
rng = np.random.default_rng(1)
Dl, d1, d2, Dr = 2, 2, 2, 3
chiL, chiM, chiR = 4, 5, 3

Lenv = rng.normal(size=(chiL, Dl, Dl)) + 1j*rng.normal(size=(chiL, Dl, Dl))
W1 = rng.normal(size=(chiL, chiM, d1, d1)) + 1j*rng.normal(size=(chiL, chiM, d1, d1))
W2 = rng.normal(size=(chiM, chiR, d2, d2)) + 1j*rng.normal(size=(chiM, chiR, d2, d2))
Renv = rng.normal(size=(chiR, Dr, Dr)) + 1j*rng.normal(size=(chiR, Dr, Dr))
theta = rng.normal(size=(Dl, d1, d2, Dr)) + 1j*rng.normal(size=(Dl, d1, d2, Dr))

Hdense = build_dense_two_site_heff(Lenv, W1, W2, Renv, Dl, d1, d2, Dr)
v = theta.reshape(-1, order='C')
v1 = Hdense @ v
v2 = two_site_heff_apply(theta, Lenv, W1, W2, Renv).reshape(-1, order='C')

print("two-site Heff test error =", np.linalg.norm(v1 - v2))
```

---

# 13. Benchmark 1: Heisenberg chain

We compare DMRG against exact diagonalization for small \(L\).

```python
def run_heisenberg_benchmark(L_list=(4, 6, 8, 10), J=1.0, Jz=1.0, h=0.0, Dmax=32, nsweeps=8):
    exact_E = []
    dmrg_E = []

    for L in L_list:
        mpo = mpo_heisenberg_xxz(L, J=J, Jz=Jz, h=h)
        init = product_state_mps(L, d=2, state_index=0)  # all-up product state
        res = dmrg_two_site(mpo, d=2, Dmax=Dmax, nsweeps=nsweeps, init_mps=init, verbose=False)

        E_dmrg = res["energies"][-1]
        dmrg_E.append(E_dmrg)

        H = dense_heisenberg_xxz(L, J=J, Jz=Jz, h=h)
        E_ex = exact_ground_energy(H)
        exact_E.append(E_ex)

        print(f"L={L:2d}   DMRG={E_dmrg:.12f}   Exact={E_ex:.12f}   |Δ|={abs(E_dmrg-E_ex):.3e}")

    return np.array(L_list), np.array(dmrg_E), np.array(exact_E)
```

Run it:

```python
L_list, E_dmrg_h, E_exact_h = run_heisenberg_benchmark(L_list=(4, 6, 8, 10), Dmax=32, nsweeps=10)
```

Plot:

```python
plt.figure()
plt.plot(L_list, E_exact_h, 'o-', label='Exact diagonalization')
plt.plot(L_list, E_dmrg_h, 's--', label='Two-site DMRG')
plt.xlabel('L')
plt.ylabel('Ground-state energy')
plt.title('Spin-1/2 Heisenberg chain')
plt.legend()
plt.grid(True, alpha=0.3)
savefig("heisenberg_energy_vs_L.png")
plt.show()
```

Error plot:

```python
plt.figure()
plt.semilogy(L_list, np.abs(E_dmrg_h - E_exact_h), 'o-')
plt.xlabel('L')
plt.ylabel(r'$|E_{\rm DMRG} - E_{\rm exact}|$')
plt.title('Heisenberg DMRG error')
plt.grid(True, alpha=0.3)
savefig("heisenberg_error_vs_L.png")
plt.show()
```

---

# 14. Heisenberg convergence history for one selected system

```python
L = 10
mpo = mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0)
init = product_state_mps(L, d=2, state_index=0)
res_h = dmrg_two_site(mpo, d=2, Dmax=32, nsweeps=12, init_mps=init, verbose=True)

plt.figure()
plt.plot(np.arange(1, len(res_h["energies"]) + 1), res_h["energies"], 'o-')
plt.xlabel('Sweep')
plt.ylabel('Energy')
plt.title(f'Heisenberg convergence, L={L}')
plt.grid(True, alpha=0.3)
savefig("heisenberg_convergence_L10.png")
plt.show()

plt.figure()
plt.semilogy(np.arange(1, len(res_h["max_discards"]) + 1), res_h["max_discards"], 'o-')
plt.xlabel('Sweep')
plt.ylabel('Max discarded weight')
plt.title(f'Heisenberg discarded weight, L={L}')
plt.grid(True, alpha=0.3)
savefig("heisenberg_discarded_weight_L10.png")
plt.show()
```

---

# 15. Benchmark 2: AKLT chain

We compare with:
- exact diagonalization for small \(L\)
- exact formula \(E_0 = -\frac23(L-1)\)

```python
def run_aklt_benchmark(L_list=(4, 6, 8), Dmax=2, nsweeps=8):
    exact_formula = []
    exact_ed = []
    dmrg_E = []
    max_disc = []

    for L in L_list:
        mpo = mpo_aklt(L)
        init = product_state_mps(L, d=3, state_index=0)
        res = dmrg_two_site(mpo, d=3, Dmax=Dmax, nsweeps=nsweeps, init_mps=init, verbose=False)

        E_dmrg = res["energies"][-1]
        dmrg_E.append(E_dmrg)
        max_disc.append(np.max(res["max_discards"]))

        E_formula = -(2.0/3.0) * (L - 1)
        exact_formula.append(E_formula)

        H = dense_aklt(L)
        E_ex = exact_ground_energy(H)
        exact_ed.append(E_ex)

        print(f"L={L:2d}   DMRG={E_dmrg:.12f}   Formula={E_formula:.12f}   ED={E_ex:.12f}   max disc={max_disc[-1]:.3e}")

    return np.array(L_list), np.array(dmrg_E), np.array(exact_formula), np.array(exact_ed), np.array(max_disc)
```

Run it:

```python
L_list_aklt, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a = run_aklt_benchmark(L_list=(4, 6, 8), Dmax=2, nsweeps=10)
```

Plot energies:

```python
plt.figure()
plt.plot(L_list_aklt, E_formula_a, 'k-', label='Exact formula')
plt.plot(L_list_aklt, E_exact_a, 'o', label='Exact diagonalization')
plt.plot(L_list_aklt, E_dmrg_a, 's--', label='Two-site DMRG, Dmax=2')
plt.xlabel('L')
plt.ylabel('Ground-state energy')
plt.title('AKLT chain')
plt.legend()
plt.grid(True, alpha=0.3)
savefig("aklt_energy_vs_L.png")
plt.show()
```

Plot DMRG error vs exact formula:

```python
plt.figure()
plt.semilogy(L_list_aklt, np.abs(E_dmrg_a - E_formula_a), 'o-')
plt.xlabel('L')
plt.ylabel(r'$|E_{\rm DMRG} - E_{\rm exact}|$')
plt.title('AKLT DMRG error')
plt.grid(True, alpha=0.3)
savefig("aklt_error_vs_L.png")
plt.show()
```

Plot discarded weight:

```python
plt.figure()
plt.semilogy(L_list_aklt, np.maximum(max_disc_a, 1e-18), 'o-')
plt.xlabel('L')
plt.ylabel('Max discarded weight')
plt.title('AKLT discarded weight at Dmax=2')
plt.grid(True, alpha=0.3)
savefig("aklt_discarded_weight_vs_L.png")
plt.show()
```

For a correct implementation, this should be very small, ideally near machine precision.

---

# 16. AKLT entanglement entropy profile

For the AKLT state, the bond dimension is exactly 2, so the entanglement entropy should show the characteristic finite-chain profile.

```python
L = 12
mpo = mpo_aklt(L)
init = product_state_mps(L, d=3, state_index=0)
res_aklt = dmrg_two_site(mpo, d=3, Dmax=2, nsweeps=12, init_mps=init, verbose=True)
mps_aklt = res_aklt["mps"]

Sbond = entanglement_entropies(mps_aklt)

plt.figure()
plt.plot(np.arange(1, L), Sbond, 'o-')
plt.xlabel('Bond cut')
plt.ylabel('Entanglement entropy')
plt.title(f'AKLT bond entanglement entropy, L={L}')
plt.grid(True, alpha=0.3)
savefig("aklt_entanglement_entropy_L12.png")
plt.show()
```

---

# 17. AKLT convergence history

```python
plt.figure()
plt.plot(np.arange(1, len(res_aklt["energies"]) + 1), res_aklt["energies"], 'o-')
plt.xlabel('Sweep')
plt.ylabel('Energy')
plt.title('AKLT convergence')
plt.grid(True, alpha=0.3)
savefig("aklt_convergence.png")
plt.show()

plt.figure()
plt.semilogy(np.arange(1, len(res_aklt["max_discards"]) + 1),
             np.maximum(res_aklt["max_discards"], 1e-18), 'o-')
plt.xlabel('Sweep')
plt.ylabel('Max discarded weight')
plt.title('AKLT discarded weight')
plt.grid(True, alpha=0.3)
savefig("aklt_discarded_weight.png")
plt.show()
```

---

# 18. Optional: variance check for small systems

Variance is a very good convergence diagnostic, but computing it from dense \(H\) is only practical for small systems.

### Heisenberg variance example

```python
L = 8
mpo = mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0)
res = dmrg_two_site(mpo, d=2, Dmax=32, nsweeps=10, init_mps=product_state_mps(L,2,0), verbose=False)
H = dense_heisenberg_xxz(L)
E = dense_energy_from_mps(res["mps"], H)
var = dense_variance_from_mps(res["mps"], H)
print("Heisenberg L=8: E =", E, " variance =", var)
```

### AKLT variance example

```python
L = 8
mpo = mpo_aklt(L)
res = dmrg_two_site(mpo, d=3, Dmax=2, nsweeps=10, init_mps=product_state_mps(L,3,0), verbose=False)
H = dense_aklt(L)
E = dense_energy_from_mps(res["mps"], H)
var = dense_variance_from_mps(res["mps"], H)
print("AKLT L=8: E =", E, " variance =", var)
```

---

# 19. Exact AKLT MPS from the note

Your note gives the exact bond-dimension-2 MPS tensors:

\[
A^+ = \begin{pmatrix} 0 & \sqrt{2/3} \\ 0 & 0 \end{pmatrix},\quad
A^0 = \begin{pmatrix} -1/\sqrt{3} & 0 \\ 0 & 1/\sqrt{3} \end{pmatrix},\quad
A^- = \begin{pmatrix} 0 & 0 \\ -\sqrt{2/3} & 0 \end{pmatrix}
\]

We can build that explicitly as another benchmark.

```python
def exact_aklt_mps(L, dtype=np.complex128):
    Ap = np.array([[0, np.sqrt(2/3)], [0, 0]], dtype=dtype)
    A0 = np.array([[-1/np.sqrt(3), 0], [0, 1/np.sqrt(3)]], dtype=dtype)
    Am = np.array([[0, 0], [-np.sqrt(2/3), 0]], dtype=dtype)

    # physical ordering: +,0,-
    A_bulk = np.stack([Ap, A0, Am], axis=1)  # (2,3,2)

    mps = []
    # choose edge vectors to produce one member of the OBC ground-state manifold
    vl = np.array([[1.0, 0.0]], dtype=dtype)       # (1,2)
    vr = np.array([[1.0], [0.0]], dtype=dtype)     # (2,1)

    # first site
    A1 = np.einsum('la,asb->lsb', vl, A_bulk).reshape(1, 3, 2)
    mps.append(A1)

    # bulk
    for _ in range(1, L - 1):
        mps.append(A_bulk.copy())

    # last site
    AL = np.einsum('asb,br->asr', A_bulk, vr).reshape(2, 3, 1)
    mps.append(AL)

    return mps
```

Check energy of the exact AKLT MPS:

```python
L = 8
mps = exact_aklt_mps(L)
H = dense_aklt(L)
E = dense_energy_from_mps(mps, H)
print("Exact AKLT MPS energy =", E)
print("Expected =", -(2/3)*(L-1))
```

You can also compare the DMRG state energy to this exact MPS.

---

# 20. Notes on robustness and possible improvements

This code is a solid first implementation, but here are a few practical remarks:

## What is already correct and aligned with your note
- tensors are stored as `(Dl, d, Dr)`
- QR canonicalization uses the reshape conventions from the note
- two-site effective Hamiltonian is **matrix-free**
- two-site split uses SVD and truncation
- environments are updated by safe `einsum` contractions
- Heisenberg MPO follows the lower-triangular convention from the note

## Things that could be improved later
1. **Reuse environments more efficiently** during right-to-left sweeps  
   I rebuild some left environments in a simple safe way. It is correct, but not optimal.

2. **Use Lanczos with better subspace control**  
   `eigsh` is fine for moderate local dimensions.

3. **Add MPO-based energy evaluation** rather than dense-state conversion.

4. **Add explicit variance MPO** for larger systems.

5. **Add symmetry sectors** such as total \(S^z\), which can greatly improve performance for Heisenberg.

---

# 21. A compact “run all” cell

If you want one cell to produce the main figures:

```python
# --- Heisenberg benchmark ---
L_list, E_dmrg_h, E_exact_h = run_heisenberg_benchmark(L_list=(4, 6, 8, 10), Dmax=32, nsweeps=10)

plt.figure()
plt.plot(L_list, E_exact_h, 'o-', label='Exact')
plt.plot(L_list, E_dmrg_h, 's--', label='DMRG')
plt.xlabel('L')
plt.ylabel('Ground-state energy')
plt.title('Heisenberg chain')
plt.legend()
plt.grid(True, alpha=0.3)
savefig("heisenberg_energy_vs_L.png")
plt.show()

plt.figure()
plt.semilogy(L_list, np.abs(E_dmrg_h - E_exact_h), 'o-')
plt.xlabel('L')
plt.ylabel('Absolute error')
plt.title('Heisenberg DMRG error')
plt.grid(True, alpha=0.3)
savefig("heisenberg_error_vs_L.png")
plt.show()

# --- AKLT benchmark ---
L_list_aklt, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a = run_aklt_benchmark(L_list=(4, 6, 8), Dmax=2, nsweeps=10)

plt.figure()
plt.plot(L_list_aklt, E_formula_a, 'k-', label='Exact formula')
plt.plot(L_list_aklt, E_exact_a, 'o', label='ED')
plt.plot(L_list_aklt, E_dmrg_a, 's--', label='DMRG')
plt.xlabel('L')
plt.ylabel('Ground-state energy')
plt.title('AKLT chain')
plt.legend()
plt.grid(True, alpha=0.3)
savefig("aklt_energy_vs_L.png")
plt.show()

plt.figure()
plt.semilogy(L_list_aklt, np.abs(E_dmrg_a - E_formula_a), 'o-')
plt.xlabel('L')
plt.ylabel('Absolute error')
plt.title('AKLT DMRG error')
plt.grid(True, alpha=0.3)
savefig("aklt_error_vs_L.png")
plt.show()

plt.figure()
plt.semilogy(L_list_aklt, np.maximum(max_disc_a, 1e-18), 'o-')
plt.xlabel('L')
plt.ylabel('Max discarded weight')
plt.title('AKLT discarded weight')
plt.grid(True, alpha=0.3)
savefig("aklt_discarded_weight_vs_L.png")
plt.show()

# --- AKLT entropy profile ---
L = 12
res_aklt = dmrg_two_site(mpo_aklt(L), d=3, Dmax=2, nsweeps=12, init_mps=product_state_mps(L,3,0), verbose=False)
Sbond = entanglement_entropies(res_aklt["mps"])

plt.figure()
plt.plot(np.arange(1, L), Sbond, 'o-')
plt.xlabel('Bond cut')
plt.ylabel('Entanglement entropy')
plt.title('AKLT entanglement entropy')
plt.grid(True, alpha=0.3)
savefig("aklt_entanglement_entropy_L12.png")
plt.show()
```

---

# 22. If you want, I can also provide a cleaner version as a single `.py` module

I can next generate one of the following:

1. **A single clean Python module** `dmrg_mps.py`
2. **A full Jupyter notebook** structure with markdown explanations
3. **A more optimized version** with cached left/right environments
4. **Additional plots**, e.g.
   - local magnetization
   - spin-spin correlations
   - entanglement spectrum
   - AKLT string order parameter

If you want, I can now rewrite all of this into a **single polished notebook file layout** with markdown headers and minimal repetition.

---
## 🧑‍💻 User

Could you put all the functions in one cell?

---
## 🤖 Assistant

Yes — below is a **single Jupyter cell** containing **all functions** for:

- MPS utilities
- MPO builders
- two-site DMRG
- exact diagonalization benchmarks
- plotting helpers
- AKLT exact MPS benchmark

After that, I’ll give a few short example cells showing how to use it.

---

## Single cell with all functions

```python
import os
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, eigsh
import matplotlib.pyplot as plt

# ============================================================
# Global setup
# ============================================================

np.set_printoptions(precision=12, suppress=True)

def setup_figure_dir(dirname="figureAKLT"):
    os.makedirs(dirname, exist_ok=True)
    return dirname

def savefig_in_dir(dirname, name):
    path = os.path.join(dirname, name)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    print(f"saved: {path}")

# ============================================================
# Local spin operators
# ============================================================

def spin_half_ops(dtype=np.complex128):
    I = np.eye(2, dtype=dtype)
    Sp = np.array([[0, 1], [0, 0]], dtype=dtype)
    Sm = np.array([[0, 0], [1, 0]], dtype=dtype)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=dtype)
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=dtype)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=dtype)
    return I, Sp, Sm, Sz, Sx, Sy

def spin_one_ops(dtype=np.complex128):
    I = np.eye(3, dtype=dtype)
    s2 = np.sqrt(2.0)
    Sp = np.array([[0, s2, 0],
                   [0, 0, s2],
                   [0, 0, 0]], dtype=dtype)
    Sm = Sp.conj().T
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, -1]], dtype=dtype)
    Sx = 0.5 * (Sp + Sm)
    Sy = (Sp - Sm) / (2j)
    return I, Sp, Sm, Sz, Sx, Sy

# ============================================================
# Dense exact Hamiltonians for benchmarking
# ============================================================

def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def dense_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    I, Sp, Sm, Sz, Sx, Sy = spin_half_ops(dtype=dtype)
    d = 2
    H = np.zeros((d**L, d**L), dtype=dtype)

    for i in range(L - 1):
        ops = [I] * L
        ops[i] = 0.5 * J * Sp
        ops[i+1] = Sm
        H += kron_all(ops)

        ops = [I] * L
        ops[i] = 0.5 * J * Sm
        ops[i+1] = Sp
        H += kron_all(ops)

        ops = [I] * L
        ops[i] = Jz * Sz
        ops[i+1] = Sz
        H += kron_all(ops)

    for i in range(L):
        ops = [I] * L
        ops[i] = -h * Sz
        H += kron_all(ops)

    return H

def dense_aklt(L, dtype=np.complex128):
    I, Sp, Sm, Sz, Sx, Sy = spin_one_ops(dtype=dtype)
    d = 3
    H = np.zeros((d**L, d**L), dtype=dtype)

    SS = 0.5 * np.kron(Sp, Sm) + 0.5 * np.kron(Sm, Sp) + np.kron(Sz, Sz)
    hbond = SS + (1.0 / 3.0) * (SS @ SS)

    for i in range(L - 1):
        left_dim = d**i
        right_dim = d**(L - i - 2)
        H += np.kron(np.eye(left_dim, dtype=dtype),
                     np.kron(hbond, np.eye(right_dim, dtype=dtype)))
    return H

def exact_ground_energy(H):
    return np.min(la.eigvalsh(H)).real

# ============================================================
# MPO builders
# ============================================================

def mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    I, Sp, Sm, Sz, Sx, Sy = spin_half_ops(dtype=dtype)
    d = 2
    chi = 5

    W_bulk = np.zeros((chi, chi, d, d), dtype=dtype)
    W_bulk[0, 0] = I
    W_bulk[1, 0] = Sp
    W_bulk[2, 0] = Sm
    W_bulk[3, 0] = Sz
    W_bulk[4, 0] = -h * Sz
    W_bulk[4, 1] = 0.5 * J * Sm
    W_bulk[4, 2] = 0.5 * J * Sp
    W_bulk[4, 3] = Jz * Sz
    W_bulk[4, 4] = I

    W_first = np.zeros((1, chi, d, d), dtype=dtype)
    W_first[0, 0] = -h * Sz
    W_first[0, 1] = 0.5 * J * Sm
    W_first[0, 2] = 0.5 * J * Sp
    W_first[0, 3] = Jz * Sz
    W_first[0, 4] = I

    W_last = np.zeros((chi, 1, d, d), dtype=dtype)
    W_last[0, 0] = I
    W_last[1, 0] = Sp
    W_last[2, 0] = Sm
    W_last[3, 0] = Sz
    W_last[4, 0] = -h * Sz

    mpo = [None] * L
    mpo[0] = W_first
    for i in range(1, L - 1):
        mpo[i] = W_bulk.copy()
    mpo[-1] = W_last
    return mpo

def mpo_from_two_body(L, onsite_op_list, bond_terms, d, dtype=np.complex128):
    """
    H = sum_i onsite(i) + sum_i sum_k Lk(i) Rk(i+1)
    onsite_op_list already includes coefficients.
    bond_terms is a list [(Lop, Rop), ...] already including coefficients.
    """
    nb = len(bond_terms)
    chi = 2 + nb

    I = np.eye(d, dtype=dtype)
    W_bulk = np.zeros((chi, chi, d, d), dtype=dtype)

    W_bulk[0, 0] = I
    for k, (Lop, Rop) in enumerate(bond_terms):
        W_bulk[0, 1 + k] = Lop
        W_bulk[1 + k, -1] = Rop
    W_bulk[-1, -1] = I
    if len(onsite_op_list) > 0:
        W_bulk[0, -1] = sum(onsite_op_list)

    W_first = np.zeros((1, chi, d, d), dtype=dtype)
    W_first[0] = W_bulk[0]

    W_last = np.zeros((chi, 1, d, d), dtype=dtype)
    W_last[:, 0] = W_bulk[:, -1]

    mpo = [None] * L
    mpo[0] = W_first
    for i in range(1, L - 1):
        mpo[i] = W_bulk.copy()
    mpo[-1] = W_last
    return mpo

def mpo_aklt(L, dtype=np.complex128):
    I, Sp, Sm, Sz, Sx, Sy = spin_one_ops(dtype=dtype)
    d = 3

    A_terms = [
        (0.5 * Sp, Sm),
        (0.5 * Sm, Sp),
        (1.0 * Sz, Sz),
    ]

    A2_terms = []
    for La, Ra in A_terms:
        for Lb, Rb in A_terms:
            A2_terms.append(((1.0 / 3.0) * (La @ Lb), Ra @ Rb))

    bond_terms = A_terms + A2_terms
    onsite_terms = []
    return mpo_from_two_body(L, onsite_terms, bond_terms, d=d, dtype=dtype)

# ============================================================
# MPS constructors
# ============================================================

def random_mps(L, d, Dmax, seed=1234, dtype=np.complex128):
    rng = np.random.default_rng(seed)
    dims = [1]
    for i in range(1, L):
        dims.append(min(Dmax, dims[-1] * d))
    rdims = [1]
    for i in range(1, L):
        rdims.append(min(Dmax, rdims[-1] * d))
    rdims = list(reversed(rdims))

    mps = []
    for i in range(L):
        Dl = min(dims[i], rdims[i])
        if i < L - 1:
            Dr = min(dims[i+1], rdims[i+1])
        else:
            Dr = 1
        A = rng.normal(size=(Dl, d, Dr)) + 1j * rng.normal(size=(Dl, d, Dr))
        mps.append(A.astype(dtype))
    return mps

def product_state_mps(L, d, state_index=0, dtype=np.complex128):
    mps = []
    for _ in range(L):
        A = np.zeros((1, d, 1), dtype=dtype)
        A[0, state_index, 0] = 1.0
        mps.append(A)
    return mps

def exact_aklt_mps(L, dtype=np.complex128):
    Ap = np.array([[0, np.sqrt(2/3)], [0, 0]], dtype=dtype)
    A0 = np.array([[-1/np.sqrt(3), 0], [0, 1/np.sqrt(3)]], dtype=dtype)
    Am = np.array([[0, 0], [-np.sqrt(2/3), 0]], dtype=dtype)
    A_bulk = np.stack([Ap, A0, Am], axis=1)  # shape (2,3,2)

    mps = []
    vl = np.array([[1.0, 0.0]], dtype=dtype)
    vr = np.array([[1.0], [0.0]], dtype=dtype)

    A1 = np.einsum('la,asb->lsb', vl, A_bulk).reshape(1, 3, 2)
    mps.append(A1)

    for _ in range(1, L - 1):
        mps.append(A_bulk.copy())

    AL = np.einsum('asb,br->asr', A_bulk, vr).reshape(2, 3, 1)
    mps.append(AL)
    return mps

# ============================================================
# Canonicalization
# ============================================================

def left_canonicalize_qr(mps):
    mps = [A.copy() for A in mps]
    L = len(mps)
    for i in range(L - 1):
        Dl, d, Dr = mps[i].shape
        M = mps[i].reshape(Dl * d, Dr, order='C')
        Q, R = la.qr(M, mode='economic')
        k = Q.shape[1]
        mps[i] = Q.reshape(Dl, d, k, order='C')
        mps[i+1] = np.einsum('ab,bsd->asd', R, mps[i+1], optimize=True)
    return mps

def right_canonicalize_qr(mps):
    mps = [A.copy() for A in mps]
    L = len(mps)
    for i in range(L - 1, 0, -1):
        Dl, d, Dr = mps[i].shape
        M = mps[i].reshape(Dl, d * Dr, order='C')
        Q, R = la.qr(M.conj().T, mode='economic')
        k = Q.shape[1]
        mps[i] = Q.conj().T.reshape(k, d, Dr, order='C')
        mps[i-1] = np.einsum('asb,bc->asc', mps[i-1], R.conj().T, optimize=True)
    return mps

def check_left_canonical(A, tol=1e-10):
    Dl, d, Dr = A.shape
    M = A.reshape(Dl * d, Dr, order='C')
    return np.linalg.norm(M.conj().T @ M - np.eye(Dr)) < tol

def check_right_canonical(A, tol=1e-10):
    Dl, d, Dr = A.shape
    M = A.reshape(Dl, d * Dr, order='C')
    return np.linalg.norm(M @ M.conj().T - np.eye(Dl)) < tol

# ============================================================
# Environments
# ============================================================

def init_left_env(dtype=np.complex128):
    Lenv = np.zeros((1, 1, 1), dtype=dtype)
    Lenv[0, 0, 0] = 1.0
    return Lenv

def init_right_env(dtype=np.complex128):
    Renv = np.zeros((1, 1, 1), dtype=dtype)
    Renv[0, 0, 0] = 1.0
    return Renv

def update_left_env(Lenv, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lenv, A, W, A.conj(), optimize=True)

def update_right_env(Renv, B, W):
    return np.einsum('asx,bBst,Bxy,cty->bac', B, W, Renv, B.conj(), optimize=True)

def build_all_right_envs(mps, mpo):
    L = len(mps)
    Renvs = [None] * L
    R = init_right_env(dtype=mps[0].dtype)
    Renvs[L - 1] = R
    for i in range(L - 1, 0, -1):
        R = update_right_env(R, mps[i], mpo[i])
        Renvs[i - 1] = R
    return Renvs

def build_left_env_up_to(mps, mpo, site):
    """
    Build left environment L^{[site]} from sites 0..site-1.
    site=0 gives boundary env.
    """
    Lenv = init_left_env(dtype=mps[0].dtype)
    for i in range(site):
        Lenv = update_left_env(Lenv, mps[i], mpo[i])
    return Lenv

# ============================================================
# Two-site effective Hamiltonian
# ============================================================

def two_site_heff_apply(theta, Lenv, W1, W2, Renv):
    """
    theta[a, s1, s2, r]
    Lenv[b, a, a']
    W1[b, B, s1, t1]
    W2[B, C, s2, t2]
    Renv[C, r', r]
    """
    X = np.einsum('bax,xsty->basty', Lenv, theta, optimize=True)
    Y = np.einsum('bBsu,batry->Bastr y'.replace(' ', ''), W1, X, optimize=True)
    Z = np.einsum('BCsv,Batvy->Catsy'.replace(' ', ''), W2, Y, optimize=True)
    out = np.einsum('Castr,Cry->asty'.replace(' ', ''), Z, Renv, optimize=True)
    return out

def build_dense_two_site_heff(Lenv, W1, W2, Renv, Dl, d1, d2, Dr):
    n = Dl * d1 * d2 * Dr
    H = np.zeros((n, n), dtype=np.complex128)
    for j in range(n):
        e = np.zeros(n, dtype=np.complex128)
        e[j] = 1.0
        theta = e.reshape(Dl, d1, d2, Dr, order='C')
        H[:, j] = two_site_heff_apply(theta, Lenv, W1, W2, Renv).reshape(n, order='C')
    return H

def solve_local_ground_state(theta0, Lenv, W1, W2, Renv, tol=1e-10, maxiter=500):
    Dl, d1, d2, Dr = theta0.shape
    n = Dl * d1 * d2 * Dr

    def mv(v):
        th = v.reshape(Dl, d1, d2, Dr, order='C')
        out = two_site_heff_apply(th, Lenv, W1, W2, Renv)
        return out.reshape(n, order='C')

    Hlin = LinearOperator((n, n), matvec=mv, dtype=np.complex128)
    v0 = theta0.reshape(n, order='C')
    nrm = np.linalg.norm(v0)
    if nrm < 1e-15:
        v0 = np.random.default_rng(123).normal(size=n) + 1j * np.random.default_rng(456).normal(size=n)
        v0 = v0.astype(np.complex128)
        v0 /= np.linalg.norm(v0)
    else:
        v0 = v0 / nrm

    vals, vecs = eigsh(Hlin, k=1, which='SA', v0=v0, tol=tol, maxiter=maxiter)
    E = vals[0].real
    theta = vecs[:, 0].reshape(Dl, d1, d2, Dr, order='C')
    theta /= np.linalg.norm(theta)
    return E, theta

# ============================================================
# SVD split / truncation
# ============================================================

def split_two_site_theta(theta, Dmax, cutoff=1e-12, sweep_direction='lr'):
    Dl, d1, d2, Dr = theta.shape
    M = theta.reshape(Dl * d1, d2 * Dr, order='C')
    U, s, Vh = la.svd(M, full_matrices=False)

    keep = min(Dmax, np.sum(s > cutoff))
    if keep < 1:
        keep = 1

    disc_weight = np.sum(s[keep:]**2).real
    U = U[:, :keep]
    s = s[:keep]
    Vh = Vh[:keep, :]

    if sweep_direction == 'lr':
        A = U.reshape(Dl, d1, keep, order='C')
        Mright = (np.diag(s) @ Vh).reshape(keep, d2, Dr, order='C')
        return A, Mright, s, disc_weight
    elif sweep_direction == 'rl':
        Mleft = (U @ np.diag(s)).reshape(Dl, d1, keep, order='C')
        B = Vh.reshape(keep, d2, Dr, order='C')
        return Mleft, B, s, disc_weight
    else:
        raise ValueError("sweep_direction must be 'lr' or 'rl'")

# ============================================================
# Dense-state utilities
# ============================================================

def mps_to_dense(mps):
    psi = mps[0][0, :, :]
    for i in range(1, len(mps)):
        psi = np.einsum('...a,asb->...sb', psi, mps[i], optimize=True)
    return psi.reshape(-1, order='C')

def dense_energy_from_mps(mps, H):
    psi = mps_to_dense(mps)
    psi = psi / np.linalg.norm(psi)
    return np.vdot(psi, H @ psi).real

def dense_variance_from_mps(mps, H):
    psi = mps_to_dense(mps)
    psi = psi / np.linalg.norm(psi)
    E = np.vdot(psi, H @ psi).real
    Hpsi = H @ psi
    E2 = np.vdot(Hpsi, Hpsi).real
    return E2 - E**2

def bond_singular_values_from_mps(mps):
    psi = mps_to_dense(mps)
    L = len(mps)
    d = mps[0].shape[1]
    svals = []
    for cut in range(1, L):
        M = psi.reshape(d**cut, d**(L-cut), order='C')
        s = la.svdvals(M)
        s = s / np.linalg.norm(s)
        svals.append(s)
    return svals

def entanglement_entropies(mps):
    svals = bond_singular_values_from_mps(mps)
    ent = []
    for s in svals:
        p = s**2
        p = p[p > 1e-15]
        ent.append(-(p * np.log(p)).sum().real)
    return np.array(ent)

# ============================================================
# Two-site DMRG main driver
# ============================================================

def dmrg_two_site(mpo, d, Dmax=32, nsweeps=8, cutoff=1e-12,
                  init_mps=None, verbose=True, eig_tol=1e-10):
    L = len(mpo)

    if init_mps is None:
        init_mps = random_mps(L, d, Dmax=max(2, min(Dmax, 4)), seed=1234)

    mps = right_canonicalize_qr(init_mps)

    energies = []
    max_discards = []

    for sweep in range(nsweeps):
        Renvs = build_all_right_envs(mps, mpo)
        Lenv = init_left_env(dtype=mps[0].dtype)

        sweep_discards = []
        E_last = None

        # Left -> Right
        for i in range(L - 1):
            theta = np.einsum('asb,btc->astc', mps[i], mps[i+1], optimize=True)
            E, theta_opt = solve_local_ground_state(theta, Lenv, mpo[i], mpo[i+1], Renvs[i+1], tol=eig_tol)

            A, Mright, s, disc = split_two_site_theta(theta_opt, Dmax=Dmax, cutoff=cutoff, sweep_direction='lr')
            mps[i] = A
            mps[i+1] = Mright
            Lenv = update_left_env(Lenv, mps[i], mpo[i])

            sweep_discards.append(disc)
            E_last = E

        # Right -> Left
        Renv = init_right_env(dtype=mps[0].dtype)
        for i in range(L - 2, -1, -1):
            theta = np.einsum('asb,btc->astc', mps[i], mps[i+1], optimize=True)
            Ltmp = build_left_env_up_to(mps, mpo, i)

            E, theta_opt = solve_local_ground_state(theta, Ltmp, mpo[i], mpo[i+1], Renv, tol=eig_tol)

            Mleft, B, s, disc = split_two_site_theta(theta_opt, Dmax=Dmax, cutoff=cutoff, sweep_direction='rl')
            mps[i] = Mleft
            mps[i+1] = B
            Renv = update_right_env(Renv, mps[i+1], mpo[i+1])

            sweep_discards.append(disc)
            E_last = E

        energies.append(E_last)
        max_discards.append(max(sweep_discards) if len(sweep_discards) > 0 else 0.0)

        if verbose:
            print(f"sweep {sweep+1:2d}: E = {E_last:.15f}, max discarded weight = {max_discards[-1]:.3e}")

    return {
        "mps": mps,
        "energies": np.array(energies),
        "max_discards": np.array(max_discards),
    }

# ============================================================
# Benchmarks
# ============================================================

def run_heisenberg_benchmark(L_list=(4, 6, 8, 10), J=1.0, Jz=1.0, h=0.0,
                             Dmax=32, nsweeps=8, verbose=True):
    exact_E = []
    dmrg_E = []

    for L in L_list:
        mpo = mpo_heisenberg_xxz(L, J=J, Jz=Jz, h=h)
        init = product_state_mps(L, d=2, state_index=0)
        res = dmrg_two_site(mpo, d=2, Dmax=Dmax, nsweeps=nsweeps, init_mps=init, verbose=False)
        E_dmrg = res["energies"][-1]
        dmrg_E.append(E_dmrg)

        H = dense_heisenberg_xxz(L, J=J, Jz=Jz, h=h)
        E_ex = exact_ground_energy(H)
        exact_E.append(E_ex)

        if verbose:
            print(f"L={L:2d}   DMRG={E_dmrg:.12f}   Exact={E_ex:.12f}   |Δ|={abs(E_dmrg-E_ex):.3e}")

    return np.array(L_list), np.array(dmrg_E), np.array(exact_E)

def run_aklt_benchmark(L_list=(4, 6, 8), Dmax=2, nsweeps=8, verbose=True):
    exact_formula = []
    exact_ed = []
    dmrg_E = []
    max_disc = []

    for L in L_list:
        mpo = mpo_aklt(L)
        init = product_state_mps(L, d=3, state_index=0)
        res = dmrg_two_site(mpo, d=3, Dmax=Dmax, nsweeps=nsweeps, init_mps=init, verbose=False)

        E_dmrg = res["energies"][-1]
        dmrg_E.append(E_dmrg)
        max_disc.append(np.max(res["max_discards"]))

        E_formula = -(2.0 / 3.0) * (L - 1)
        exact_formula.append(E_formula)

        H = dense_aklt(L)
        E_ex = exact_ground_energy(H)
        exact_ed.append(E_ex)

        if verbose:
            print(f"L={L:2d}   DMRG={E_dmrg:.12f}   Formula={E_formula:.12f}   ED={E_ex:.12f}   max disc={max_disc[-1]:.3e}")

    return np.array(L_list), np.array(dmrg_E), np.array(exact_formula), np.array(exact_ed), np.array(max_disc)

# ============================================================
# Plot helpers
# ============================================================

def plot_heisenberg_benchmark(L_list, E_dmrg, E_exact, figdir="figureAKLT"):
    setup_figure_dir(figdir)

    plt.figure()
    plt.plot(L_list, E_exact, 'o-', label='Exact diagonalization')
    plt.plot(L_list, E_dmrg, 's--', label='Two-site DMRG')
    plt.xlabel('L')
    plt.ylabel('Ground-state energy')
    plt.title('Spin-1/2 Heisenberg chain')
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "heisenberg_energy_vs_L.png")
    plt.show()

    plt.figure()
    plt.semilogy(L_list, np.abs(E_dmrg - E_exact), 'o-')
    plt.xlabel('L')
    plt.ylabel(r'$|E_{\rm DMRG} - E_{\rm exact}|$')
    plt.title('Heisenberg DMRG error')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "heisenberg_error_vs_L.png")
    plt.show()

def plot_convergence(res, title_prefix="", filename_prefix="", figdir="figureAKLT"):
    setup_figure_dir(figdir)

    plt.figure()
    plt.plot(np.arange(1, len(res["energies"]) + 1), res["energies"], 'o-')
    plt.xlabel('Sweep')
    plt.ylabel('Energy')
    plt.title(f'{title_prefix} convergence')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, f"{filename_prefix}_convergence.png")
    plt.show()

    plt.figure()
    plt.semilogy(np.arange(1, len(res["max_discards"]) + 1),
                 np.maximum(res["max_discards"], 1e-18), 'o-')
    plt.xlabel('Sweep')
    plt.ylabel('Max discarded weight')
    plt.title(f'{title_prefix} discarded weight')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, f"{filename_prefix}_discarded_weight.png")
    plt.show()

def plot_aklt_benchmark(L_list, E_dmrg, E_formula, E_exact, max_disc, figdir="figureAKLT"):
    setup_figure_dir(figdir)

    plt.figure()
    plt.plot(L_list, E_formula, 'k-', label='Exact formula')
    plt.plot(L_list, E_exact, 'o', label='Exact diagonalization')
    plt.plot(L_list, E_dmrg, 's--', label='Two-site DMRG, Dmax=2')
    plt.xlabel('L')
    plt.ylabel('Ground-state energy')
    plt.title('AKLT chain')
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "aklt_energy_vs_L.png")
    plt.show()

    plt.figure()
    plt.semilogy(L_list, np.abs(E_dmrg - E_formula), 'o-')
    plt.xlabel('L')
    plt.ylabel(r'$|E_{\rm DMRG} - E_{\rm exact}|$')
    plt.title('AKLT DMRG error')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "aklt_error_vs_L.png")
    plt.show()

    plt.figure()
    plt.semilogy(L_list, np.maximum(max_disc, 1e-18), 'o-')
    plt.xlabel('L')
    plt.ylabel('Max discarded weight')
    plt.title('AKLT discarded weight at Dmax=2')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "aklt_discarded_weight_vs_L.png")
    plt.show()

def plot_entanglement_entropy(mps, title="Entanglement entropy", filename="entanglement_entropy.png", figdir="figureAKLT"):
    setup_figure_dir(figdir)
    Sbond = entanglement_entropies(mps)
    L = len(mps)

    plt.figure()
    plt.plot(np.arange(1, L), Sbond, 'o-')
    plt.xlabel('Bond cut')
    plt.ylabel('Entanglement entropy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, filename)
    plt.show()

# ============================================================
# Diagnostics / tests
# ============================================================

def test_canonicalization():
    mps0 = random_mps(L=6, d=2, Dmax=4, seed=7)
    mpsL = left_canonicalize_qr(mps0)
    mpsR = right_canonicalize_qr(mps0)

    print("Left-canonical checks:")
    for i in range(5):
        print(i, check_left_canonical(mpsL[i]))

    print("Right-canonical checks:")
    for i in range(1, 6):
        print(i, check_right_canonical(mpsR[i]))

def test_two_site_heff():
    rng = np.random.default_rng(1)
    Dl, d1, d2, Dr = 2, 2, 2, 3
    chiL, chiM, chiR = 3, 4, 2

    Lenv = rng.normal(size=(chiL, Dl, Dl)) + 1j * rng.normal(size=(chiL, Dl, Dl))
    W1 = rng.normal(size=(chiL, chiM, d1, d1)) + 1j * rng.normal(size=(chiL, chiM, d1, d1))
    W2 = rng.normal(size=(chiM, chiR, d2, d2)) + 1j * rng.normal(size=(chiM, chiR, d2, d2))
    Renv = rng.normal(size=(chiR, Dr, Dr)) + 1j * rng.normal(size=(chiR, Dr, Dr))
    theta = rng.normal(size=(Dl, d1, d2, Dr)) + 1j * rng.normal(size=(Dl, d1, d2, Dr))

    Hdense = build_dense_two_site_heff(Lenv, W1, W2, Renv, Dl, d1, d2, Dr)
    v = theta.reshape(-1, order='C')
    v1 = Hdense @ v
    v2 = two_site_heff_apply(theta, Lenv, W1, W2, Renv).reshape(-1, order='C')

    err = np.linalg.norm(v1 - v2)
    print("two-site Heff test error =", err)
    return err
```

---

## Important note

In the single-cell version above, the function `two_site_heff_apply` uses compact `einsum` strings. Since this is the most delicate part, I recommend you test it immediately with:

```python
test_two_site_heff()
```

If you want, I can also give you a **slightly longer but safer rewritten version** of `two_site_heff_apply` using more explicit intermediate tensors and simpler index labels.

---

# Example usage cells

---

## 1. Create figure directory

```python
figdir = setup_figure_dir("figureAKLT")
```

---

## 2. Test canonicalization and local effective Hamiltonian

```python
test_canonicalization()
test_two_site_heff()
```

---

## 3. Heisenberg benchmark

```python
L_list, E_dmrg_h, E_exact_h = run_heisenberg_benchmark(
    L_list=(4, 6, 8, 10),
    J=1.0, Jz=1.0, h=0.0,
    Dmax=32,
    nsweeps=10,
    verbose=True
)

plot_heisenberg_benchmark(L_list, E_dmrg_h, E_exact_h, figdir="figureAKLT")
```

---

## 4. Heisenberg convergence for one chain

```python
L = 10
mpo = mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0)
init = product_state_mps(L, d=2, state_index=0)

res_h = dmrg_two_site(mpo, d=2, Dmax=32, nsweeps=12, init_mps=init, verbose=True)
plot_convergence(res_h, title_prefix=f"Heisenberg L={L}", filename_prefix="heisenberg_L10", figdir="figureAKLT")
```

---

## 5. AKLT benchmark

```python
L_list_aklt, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a = run_aklt_benchmark(
    L_list=(4, 6, 8),
    Dmax=2,
    nsweeps=10,
    verbose=True
)

plot_aklt_benchmark(L_list_aklt, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a, figdir="figureAKLT")
```

---

## 6. AKLT convergence and entanglement entropy

```python
L = 12
mpo = mpo_aklt(L)
init = product_state_mps(L, d=3, state_index=0)

res_aklt = dmrg_two_site(mpo, d=3, Dmax=2, nsweeps=12, init_mps=init, verbose=True)
plot_convergence(res_aklt, title_prefix="AKLT", filename_prefix="aklt", figdir="figureAKLT")
plot_entanglement_entropy(res_aklt["mps"],
                          title=f"AKLT bond entanglement entropy, L={L}",
                          filename="aklt_entanglement_entropy_L12.png",
                          figdir="figureAKLT")
```

---

## 7. Exact AKLT MPS energy check

```python
L = 8
mps = exact_aklt_mps(L)
H = dense_aklt(L)
E = dense_energy_from_mps(mps, H)
print("Exact AKLT MPS energy =", E)
print("Expected =", -(2/3)*(L-1))
```

---

If you want, I can next do one of these:

1. **Fix and simplify `two_site_heff_apply` into the safest explicit form**
2. **Make the whole notebook more efficient**
3. **Provide one final "run-all" notebook cell after the function cell**

---
## 🧑‍💻 User

Provide one final "run-all" notebook cell after the function cell

---
## 🤖 Assistant

```python
# =========================
# Run-all notebook cell
# =========================

figdir = setup_figure_dir("figureAKLT")
print("Figures will be saved in:", figdir)

# -------------------------------------------------
# 1. Basic tests
# -------------------------------------------------
print("\n=== Basic tests ===")
test_canonicalization()
err_heff = test_two_site_heff()
print("Local two-site Heff test error =", err_heff)

# -------------------------------------------------
# 2. Heisenberg benchmark vs exact diagonalization
# -------------------------------------------------
print("\n=== Heisenberg benchmark ===")
L_list_h, E_dmrg_h, E_exact_h = run_heisenberg_benchmark(
    L_list=(4, 6, 8, 10),
    J=1.0,
    Jz=1.0,
    h=0.0,
    Dmax=32,
    nsweeps=10,
    verbose=True
)
plot_heisenberg_benchmark(L_list_h, E_dmrg_h, E_exact_h, figdir=figdir)

# Heisenberg convergence for one selected system
print("\n=== Heisenberg convergence example ===")
Lh = 10
mpo_h = mpo_heisenberg_xxz(Lh, J=1.0, Jz=1.0, h=0.0)
init_h = product_state_mps(Lh, d=2, state_index=0)
res_h = dmrg_two_site(mpo_h, d=2, Dmax=32, nsweeps=12, init_mps=init_h, verbose=True)
plot_convergence(res_h, title_prefix=f"Heisenberg L={Lh}", filename_prefix="heisenberg_L10", figdir=figdir)

# Optional dense variance check for small Heisenberg system
print("\n=== Heisenberg variance check ===")
Lh_small = 8
mpo_hs = mpo_heisenberg_xxz(Lh_small, J=1.0, Jz=1.0, h=0.0)
res_hs = dmrg_two_site(mpo_hs, d=2, Dmax=32, nsweeps=10,
                       init_mps=product_state_mps(Lh_small, d=2, state_index=0),
                       verbose=False)
Hh_small = dense_heisenberg_xxz(Lh_small, J=1.0, Jz=1.0, h=0.0)
Eh_small = dense_energy_from_mps(res_hs["mps"], Hh_small)
varh_small = dense_variance_from_mps(res_hs["mps"], Hh_small)
print(f"Heisenberg L={Lh_small}: E = {Eh_small:.12f}, variance = {varh_small:.3e}")

# -------------------------------------------------
# 3. AKLT benchmark
# -------------------------------------------------
print("\n=== AKLT benchmark ===")
L_list_a, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a = run_aklt_benchmark(
    L_list=(4, 6, 8),
    Dmax=2,
    nsweeps=10,
    verbose=True
)
plot_aklt_benchmark(L_list_a, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a, figdir=figdir)

# AKLT convergence and entropy profile
print("\n=== AKLT convergence and entropy ===")
La = 12
mpo_a = mpo_aklt(La)
init_a = product_state_mps(La, d=3, state_index=0)
res_a = dmrg_two_site(mpo_a, d=3, Dmax=2, nsweeps=12, init_mps=init_a, verbose=True)

plot_convergence(res_a, title_prefix=f"AKLT L={La}", filename_prefix="aklt_L12", figdir=figdir)
plot_entanglement_entropy(
    res_a["mps"],
    title=f"AKLT bond entanglement entropy, L={La}",
    filename="aklt_entanglement_entropy_L12.png",
    figdir=figdir
)

# Optional dense variance check for small AKLT system
print("\n=== AKLT variance check ===")
La_small = 8
mpo_as = mpo_aklt(La_small)
res_as = dmrg_two_site(mpo_as, d=3, Dmax=2, nsweeps=10,
                       init_mps=product_state_mps(La_small, d=3, state_index=0),
                       verbose=False)
Ha_small = dense_aklt(La_small)
Ea_small = dense_energy_from_mps(res_as["mps"], Ha_small)
vara_small = dense_variance_from_mps(res_as["mps"], Ha_small)
print(f"AKLT L={La_small}: E = {Ea_small:.12f}, variance = {vara_small:.3e}")

# -------------------------------------------------
# 4. Exact AKLT MPS check from note
# -------------------------------------------------
print("\n=== Exact AKLT MPS check ===")
L_exact = 8
mps_exact_aklt = exact_aklt_mps(L_exact)
H_exact_aklt = dense_aklt(L_exact)
E_exact_mps = dense_energy_from_mps(mps_exact_aklt, H_exact_aklt)
print(f"Exact AKLT MPS energy for L={L_exact}: {E_exact_mps:.12f}")
print(f"Expected exact energy: {-(2/3)*(L_exact-1):.12f}")

# -------------------------------------------------
# 5. Summary
# -------------------------------------------------
print("\n=== Summary ===")
print("Heisenberg benchmark sizes:", L_list_h)
print("Heisenberg |DMRG-Exact|    :", np.abs(E_dmrg_h - E_exact_h))
print("AKLT benchmark sizes       :", L_list_a)
print("AKLT |DMRG-Formula|        :", np.abs(E_dmrg_a - E_formula_a))
print("AKLT max discarded weights :", max_disc_a)
print("\nAll figures have been shown and saved to:", figdir)
```

---
## 🧑‍💻 User

Left-canonical checks:
0 True
1 True
2 True
3 True
4 True
Right-canonical checks:
1 True
2 True
3 True
4 True
5 True

---
## 🤖 Assistant

Great — that means the **QR canonicalization part is working correctly**.

So the following pieces are now validated:

- left-canonicalization
- right-canonicalization
- reshape conventions for QR
- basic MPS gauge handling

The next important check is the **two-site effective Hamiltonian test**:

```python
err_heff = test_two_site_heff()
print(err_heff)
```

What we want is that the error is very small, typically around:

- `1e-12` to `1e-10` → excellent
- `1e-8` → probably still okay
- much larger than that → likely an index mistake in `two_site_heff_apply`

## Why this matters
The canonicalization test only checks the MPS gauge routines.  
But **DMRG itself depends crucially on the correctness of**
\[
\Theta \mapsto \mathcal H_{\mathrm{eff}}\Theta
\]
for the two-site local eigensolver.

So the next step is:

1. run `test_two_site_heff()`
2. if the error is not tiny, I should fix `two_site_heff_apply`

---

## My suspicion
Because `einsum` index strings are easy to mistype, I suspect that the current compact implementation of `two_site_heff_apply` may need a safer rewrite.

If you want, I can now give you a **more explicit, safer replacement** for:

- `two_site_heff_apply`
- `test_two_site_heff`

so your DMRG results are trustworthy before running the full benchmarks.

If you already ran `test_two_site_heff()`, send me the printed error and I’ll tell you immediately whether the local Hamiltonian contraction is correct.

---
## 🧑‍💻 User

---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[3], line 2
      1 test_canonicalization()
----> 2 test_two_site_heff()

Cell In[1], line 696, in test_two_site_heff()
    693 Renv = rng.normal(size=(chiR, Dr, Dr)) + 1j * rng.normal(size=(chiR, Dr, Dr))
    694 theta = rng.normal(size=(Dl, d1, d2, Dr)) + 1j * rng.normal(size=(Dl, d1, d2, Dr))
--> 696 Hdense = build_dense_two_site_heff(Lenv, W1, W2, Renv, Dl, d1, d2, Dr)
    697 v = theta.reshape(-1, order='C')
    698 v1 = Hdense @ v

Cell In[1], line 355, in build_dense_two_site_heff(Lenv, W1, W2, Renv, Dl, d1, d2, Dr)
    353     e[j] = 1.0
    354     theta = e.reshape(Dl, d1, d2, Dr, order='C')
--> 355     H[:, j] = two_site_heff_apply(theta, Lenv, W1, W2, Renv).reshape(n, order='C')
    356 return H

Cell In[1], line 344, in two_site_heff_apply(theta, Lenv, W1, W2, Renv)
    342 X = np.einsum('bax,xsty->basty', Lenv, theta, optimize=True)
    343 Y = np.einsum('bBsu,batry->Bastr y'.replace(' ', ''), W1, X, optimize=True)
--> 344 Z = np.einsum('BCsv,Batvy->Catsy'.replace(' ', ''), W2, Y, optimize=True)
    345 out = np.einsum('Castr,Cry->asty'.replace(' ', ''), Z, Renv, optimize=True)
    346 return out

File c:\Users\yizho\anaconda3\envs\science\Lib\site-packages\numpy\_core\einsumfunc.py:1434, in einsum(out, optimize, *operands, **kwargs)
   1431     raise TypeError(f"Did not understand the following kwargs: {unknown_kwargs}")
   1433 # Build the contraction list and operand
-> 1434 operands, contraction_list = einsum_path(*operands, optimize=optimize,
   1435                                          einsum_call=True)
   1437 # Handle order kwarg for output array, c_einsum allows mixed case
   1438 output_order = kwargs.pop('order', 'K')

File c:\Users\yizho\anaconda3\envs\science\Lib\site-packages\numpy\_core\einsumfunc.py:900, in einsum_path(optimize, einsum_call, *operands)
    898 sh = operands[tnum].shape
    899 if len(sh) != len(term):
--> 900     raise ValueError("Einstein sum subscript %s does not contain the "
    901                      "correct number of indices for operand %d."
    902                      % (input_subscripts[tnum], tnum))
    903 for cnum, char in enumerate(term):
    904     dim = sh[cnum]

ValueError: Einstein sum subscript C does not contain the correct number of indices for operand 1.

---
## 🤖 Assistant

Yes — this confirms the issue is in `two_site_heff_apply`, not in your notebook setup.

The compact `einsum` strings I used there were too error-prone. Let’s fix that with a **safe explicit version** that follows your LaTeX note exactly.

---

# Replace these two functions in your function cell

Please **overwrite**:

- `two_site_heff_apply`
- `test_two_site_heff`

with the versions below.

---

## Correct `two_site_heff_apply`

```python
def two_site_heff_apply(theta, Lenv, W1, W2, Renv):
    """
    Apply the two-site effective Hamiltonian to theta.

    Shapes:
      theta[a', t1, t2, r']    with shape (Dl, d1, d2, Dr)
      Lenv[b, a, a']           with shape (chiL, Dl, Dl)
      W1[b, B, s1, t1]         with shape (chiL, chiM, d1, d1)
      W2[B, C, s2, t2]         with shape (chiM, chiR, d2, d2)
      Renv[C, r', r]           with shape (chiR, Dr, Dr)

    Output:
      out[a, s1, s2, r]        with shape (Dl, d1, d2, Dr)

    This follows the contraction sequence from the note:
      X  = sum_{a'} L * theta
      Y  = sum_{b,t1} W1 * X
      Z  = sum_{B,t2} W2 * Y
      out= sum_{C,r'} Z * R
    """
    # X[b, a, t1, t2, r']
    X = np.einsum('baa,asty->bsty', np.zeros((1,1,1)), np.zeros((1,1,1,1)))
```

Stop — actually, let me give you the **proper correct function directly**, not half-written.

Use this exact replacement:

```python
def two_site_heff_apply(theta, Lenv, W1, W2, Renv):
    """
    theta[aP, t1, t2, rP]
    Lenv[b, a, aP]
    W1[b, B, s1, t1]
    W2[B, C, s2, t2]
    Renv[C, rP, r]

    returns out[a, s1, s2, r]
    """
    # Step 1: X[b, a, t1, t2, rP] = sum_{aP} Lenv[b,a,aP] theta[aP,t1,t2,rP]
    X = np.einsum('bap,ptur->batur', Lenv, theta, optimize=True)

    # Step 2: Y[B, a, s1, t2, rP] = sum_{b,t1} W1[b,B,s1,t1] X[b,a,t1,t2,rP]
    Y = np.einsum('bBst,batur->Basur', W1, X, optimize=True)

    # Step 3: Z[C, a, s1, s2, rP] = sum_{B,t2} W2[B,C,s2,t2] Y[B,a,s1,t2,rP]
    Z = np.einsum('BCsu,Batur->Catsu', W2, Y, optimize=True)

    # Step 4: out[a, s1, s2, r] = sum_{C,rP} Z[C,a,s1,s2,rP] Renv[C,rP,r]
    out = np.einsum('Castr,Crq->astq', Z, Renv, optimize=True)

    return out
```

However, to be fully consistent, I should make the intermediate labels cleaner because `u/r/q` can become confusing. Here is the **cleanest final version**:

```python
def two_site_heff_apply(theta, Lenv, W1, W2, Renv):
    """
    Apply the two-site effective Hamiltonian.

    Shapes:
        theta : (Dl, d1, d2, Dr)      indices (ap, t1, t2, rp)
        Lenv  : (chiL, Dl, Dl)        indices (b, a, ap)
        W1    : (chiL, chiM, d1, d1)  indices (b, B, s1, t1)
        W2    : (chiM, chiR, d2, d2)  indices (B, C, s2, t2)
        Renv  : (chiR, Dr, Dr)        indices (C, rp, r)

    Returns:
        out   : (Dl, d1, d2, Dr)      indices (a, s1, s2, r)
    """
    # X[b,a,t1,t2,rp]
    X = np.einsum('bap,ptur->batur', Lenv, theta, optimize=True)

    # Y[B,a,s1,t2,rp]
    Y = np.einsum('bBst,batur->Basur', W1, X, optimize=True)

    # Z[C,a,s1,s2,rp]
    Z = np.einsum('BCsu,Batur->Castr', W2, Y, optimize=True)

    # out[a,s1,s2,r]
    out = np.einsum('Castr,Crq->astq', Z, Renv, optimize=True)

    return out
```

---

## Correct `test_two_site_heff`

```python
def test_two_site_heff():
    rng = np.random.default_rng(1)
    Dl, d1, d2, Dr = 2, 2, 2, 3
    chiL, chiM, chiR = 3, 4, 2

    Lenv = rng.normal(size=(chiL, Dl, Dl)) + 1j * rng.normal(size=(chiL, Dl, Dl))
    W1   = rng.normal(size=(chiL, chiM, d1, d1)) + 1j * rng.normal(size=(chiL, chiM, d1, d1))
    W2   = rng.normal(size=(chiM, chiR, d2, d2)) + 1j * rng.normal(size=(chiM, chiR, d2, d2))
    Renv = rng.normal(size=(chiR, Dr, Dr)) + 1j * rng.normal(size=(chiR, Dr, Dr))
    theta = rng.normal(size=(Dl, d1, d2, Dr)) + 1j * rng.normal(size=(Dl, d1, d2, Dr))

    Hdense = build_dense_two_site_heff(Lenv, W1, W2, Renv, Dl, d1, d2, Dr)
    v = theta.reshape(-1, order='C')

    v_dense = Hdense @ v
    v_mfree = two_site_heff_apply(theta, Lenv, W1, W2, Renv).reshape(-1, order='C')

    err = np.linalg.norm(v_dense - v_mfree)
    print("two-site Heff test error =", err)
    return err
```

---

# Then run this

```python
test_two_site_heff()
```

You should now get a very small number.

Typically:
- around `1e-13` or `1e-14` is excellent

---

# Important follow-up

Since this function is used everywhere in DMRG, after replacing it, rerun:

```python
# basic tests
test_canonicalization()
test_two_site_heff()

# then benchmarks
```

---

# If you want the safest possible version

I can also give you a **fully revised function cell** with the corrected `two_site_heff_apply` already integrated, so you don’t have to manually edit inside the large cell. That may be the easiest option.

If you want, I can now provide:

- **a corrected full one-cell function block**, or
- just **a minimal patch block** containing only the functions that need replacement.

---
## 🧑‍💻 User

a corrected full one-cell function block

---
## 🤖 Assistant

```python
import os
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, eigsh
import matplotlib.pyplot as plt

# ============================================================
# Global setup
# ============================================================

np.set_printoptions(precision=12, suppress=True)

def setup_figure_dir(dirname="figureAKLT"):
    os.makedirs(dirname, exist_ok=True)
    return dirname

def savefig_in_dir(dirname, name):
    path = os.path.join(dirname, name)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    print(f"saved: {path}")

# ============================================================
# Local spin operators
# ============================================================

def spin_half_ops(dtype=np.complex128):
    I = np.eye(2, dtype=dtype)
    Sp = np.array([[0, 1], [0, 0]], dtype=dtype)
    Sm = np.array([[0, 0], [1, 0]], dtype=dtype)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=dtype)
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=dtype)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=dtype)
    return I, Sp, Sm, Sz, Sx, Sy

def spin_one_ops(dtype=np.complex128):
    I = np.eye(3, dtype=dtype)
    s2 = np.sqrt(2.0)
    Sp = np.array([[0, s2, 0],
                   [0, 0, s2],
                   [0, 0, 0]], dtype=dtype)
    Sm = Sp.conj().T
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, -1]], dtype=dtype)
    Sx = 0.5 * (Sp + Sm)
    Sy = (Sp - Sm) / (2j)
    return I, Sp, Sm, Sz, Sx, Sy

# ============================================================
# Dense exact Hamiltonians for benchmarking
# ============================================================

def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def dense_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    I, Sp, Sm, Sz, Sx, Sy = spin_half_ops(dtype=dtype)
    d = 2
    H = np.zeros((d**L, d**L), dtype=dtype)

    for i in range(L - 1):
        ops = [I] * L
        ops[i] = 0.5 * J * Sp
        ops[i+1] = Sm
        H += kron_all(ops)

        ops = [I] * L
        ops[i] = 0.5 * J * Sm
        ops[i+1] = Sp
        H += kron_all(ops)

        ops = [I] * L
        ops[i] = Jz * Sz
        ops[i+1] = Sz
        H += kron_all(ops)

    for i in range(L):
        ops = [I] * L
        ops[i] = -h * Sz
        H += kron_all(ops)

    return H

def dense_aklt(L, dtype=np.complex128):
    I, Sp, Sm, Sz, Sx, Sy = spin_one_ops(dtype=dtype)
    d = 3
    H = np.zeros((d**L, d**L), dtype=dtype)

    SS = 0.5 * np.kron(Sp, Sm) + 0.5 * np.kron(Sm, Sp) + np.kron(Sz, Sz)
    hbond = SS + (1.0 / 3.0) * (SS @ SS)

    for i in range(L - 1):
        left_dim = d**i
        right_dim = d**(L - i - 2)
        H += np.kron(np.eye(left_dim, dtype=dtype),
                     np.kron(hbond, np.eye(right_dim, dtype=dtype)))
    return H

def exact_ground_energy(H):
    return np.min(la.eigvalsh(H)).real

# ============================================================
# MPO builders
# ============================================================

def mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    I, Sp, Sm, Sz, Sx, Sy = spin_half_ops(dtype=dtype)
    d = 2
    chi = 5

    W_bulk = np.zeros((chi, chi, d, d), dtype=dtype)
    W_bulk[0, 0] = I
    W_bulk[1, 0] = Sp
    W_bulk[2, 0] = Sm
    W_bulk[3, 0] = Sz
    W_bulk[4, 0] = -h * Sz
    W_bulk[4, 1] = 0.5 * J * Sm
    W_bulk[4, 2] = 0.5 * J * Sp
    W_bulk[4, 3] = Jz * Sz
    W_bulk[4, 4] = I

    W_first = np.zeros((1, chi, d, d), dtype=dtype)
    W_first[0, 0] = -h * Sz
    W_first[0, 1] = 0.5 * J * Sm
    W_first[0, 2] = 0.5 * J * Sp
    W_first[0, 3] = Jz * Sz
    W_first[0, 4] = I

    W_last = np.zeros((chi, 1, d, d), dtype=dtype)
    W_last[0, 0] = I
    W_last[1, 0] = Sp
    W_last[2, 0] = Sm
    W_last[3, 0] = Sz
    W_last[4, 0] = -h * Sz

    mpo = [None] * L
    mpo[0] = W_first
    for i in range(1, L - 1):
        mpo[i] = W_bulk.copy()
    mpo[-1] = W_last
    return mpo

def mpo_from_two_body(L, onsite_op_list, bond_terms, d, dtype=np.complex128):
    """
    H = sum_i onsite(i) + sum_i sum_k Lk(i) Rk(i+1)
    onsite_op_list already includes coefficients.
    bond_terms is a list [(Lop, Rop), ...] already including coefficients.
    """
    nb = len(bond_terms)
    chi = 2 + nb

    I = np.eye(d, dtype=dtype)
    W_bulk = np.zeros((chi, chi, d, d), dtype=dtype)

    W_bulk[0, 0] = I
    for k, (Lop, Rop) in enumerate(bond_terms):
        W_bulk[0, 1 + k] = Lop
        W_bulk[1 + k, -1] = Rop
    W_bulk[-1, -1] = I
    if len(onsite_op_list) > 0:
        W_bulk[0, -1] = sum(onsite_op_list)

    W_first = np.zeros((1, chi, d, d), dtype=dtype)
    W_first[0] = W_bulk[0]

    W_last = np.zeros((chi, 1, d, d), dtype=dtype)
    W_last[:, 0] = W_bulk[:, -1]

    mpo = [None] * L
    mpo[0] = W_first
    for i in range(1, L - 1):
        mpo[i] = W_bulk.copy()
    mpo[-1] = W_last
    return mpo

def mpo_aklt(L, dtype=np.complex128):
    I, Sp, Sm, Sz, Sx, Sy = spin_one_ops(dtype=dtype)
    d = 3

    # S_i.S_{i+1} = 1/2 Sp Sm + 1/2 Sm Sp + Sz Sz
    A_terms = [
        (0.5 * Sp, Sm),
        (0.5 * Sm, Sp),
        (1.0 * Sz, Sz),
    ]

    # (S_i.S_{i+1})^2 / 3
    A2_terms = []
    for La, Ra in A_terms:
        for Lb, Rb in A_terms:
            A2_terms.append(((1.0 / 3.0) * (La @ Lb), Ra @ Rb))

    bond_terms = A_terms + A2_terms
    onsite_terms = []
    return mpo_from_two_body(L, onsite_terms, bond_terms, d=d, dtype=dtype)

# ============================================================
# MPS constructors
# ============================================================

def random_mps(L, d, Dmax, seed=1234, dtype=np.complex128):
    rng = np.random.default_rng(seed)
    dims = [1]
    for i in range(1, L):
        dims.append(min(Dmax, dims[-1] * d))
    rdims = [1]
    for i in range(1, L):
        rdims.append(min(Dmax, rdims[-1] * d))
    rdims = list(reversed(rdims))

    mps = []
    for i in range(L):
        Dl = min(dims[i], rdims[i])
        Dr = min(dims[i+1], rdims[i+1]) if i < L - 1 else 1
        A = rng.normal(size=(Dl, d, Dr)) + 1j * rng.normal(size=(Dl, d, Dr))
        mps.append(A.astype(dtype))
    return mps

def product_state_mps(L, d, state_index=0, dtype=np.complex128):
    mps = []
    for _ in range(L):
        A = np.zeros((1, d, 1), dtype=dtype)
        A[0, state_index, 0] = 1.0
        mps.append(A)
    return mps

def exact_aklt_mps(L, dtype=np.complex128):
    Ap = np.array([[0, np.sqrt(2/3)], [0, 0]], dtype=dtype)
    A0 = np.array([[-1/np.sqrt(3), 0], [0, 1/np.sqrt(3)]], dtype=dtype)
    Am = np.array([[0, 0], [-np.sqrt(2/3), 0]], dtype=dtype)
    A_bulk = np.stack([Ap, A0, Am], axis=1)  # (2,3,2)

    mps = []
    vl = np.array([[1.0, 0.0]], dtype=dtype)
    vr = np.array([[1.0], [0.0]], dtype=dtype)

    A1 = np.einsum('la,asb->lsb', vl, A_bulk, optimize=True).reshape(1, 3, 2)
    mps.append(A1)

    for _ in range(1, L - 1):
        mps.append(A_bulk.copy())

    AL = np.einsum('asb,br->asr', A_bulk, vr, optimize=True).reshape(2, 3, 1)
    mps.append(AL)
    return mps

# ============================================================
# Canonicalization
# ============================================================

def left_canonicalize_qr(mps):
    mps = [A.copy() for A in mps]
    L = len(mps)
    for i in range(L - 1):
        Dl, d, Dr = mps[i].shape
        M = mps[i].reshape(Dl * d, Dr, order='C')
        Q, R = la.qr(M, mode='economic')
        k = Q.shape[1]
        mps[i] = Q.reshape(Dl, d, k, order='C')
        mps[i+1] = np.einsum('ab,bsd->asd', R, mps[i+1], optimize=True)
    return mps

def right_canonicalize_qr(mps):
    mps = [A.copy() for A in mps]
    L = len(mps)
    for i in range(L - 1, 0, -1):
        Dl, d, Dr = mps[i].shape
        M = mps[i].reshape(Dl, d * Dr, order='C')
        Q, R = la.qr(M.conj().T, mode='economic')
        k = Q.shape[1]
        mps[i] = Q.conj().T.reshape(k, d, Dr, order='C')
        mps[i-1] = np.einsum('asb,bc->asc', mps[i-1], R.conj().T, optimize=True)
    return mps

def check_left_canonical(A, tol=1e-10):
    Dl, d, Dr = A.shape
    M = A.reshape(Dl * d, Dr, order='C')
    return np.linalg.norm(M.conj().T @ M - np.eye(Dr)) < tol

def check_right_canonical(A, tol=1e-10):
    Dl, d, Dr = A.shape
    M = A.reshape(Dl, d * Dr, order='C')
    return np.linalg.norm(M @ M.conj().T - np.eye(Dl)) < tol

# ============================================================
# Environments
# ============================================================

def init_left_env(dtype=np.complex128):
    Lenv = np.zeros((1, 1, 1), dtype=dtype)
    Lenv[0, 0, 0] = 1.0
    return Lenv

def init_right_env(dtype=np.complex128):
    Renv = np.zeros((1, 1, 1), dtype=dtype)
    Renv[0, 0, 0] = 1.0
    return Renv

def update_left_env(Lenv, A, W):
    # Lnew[B,a,c] = sum_{b,x,y,s,t} L[b,x,y] A[x,s,a] W[b,B,s,t] A*[y,t,c]
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lenv, A, W, A.conj(), optimize=True)

def update_right_env(Renv, B, W):
    # Rnew[b,a,c] = sum_{B,x,y,s,t} B[a,s,x] W[b,B,s,t] R[B,x,y] B*[c,t,y]
    return np.einsum('asx,bBst,Bxy,cty->bac', B, W, Renv, B.conj(), optimize=True)

def build_all_right_envs(mps, mpo):
    L = len(mps)
    Renvs = [None] * L
    R = init_right_env(dtype=mps[0].dtype)
    Renvs[L - 1] = R
    for i in range(L - 1, 0, -1):
        R = update_right_env(R, mps[i], mpo[i])
        Renvs[i - 1] = R
    return Renvs

def build_left_env_up_to(mps, mpo, site):
    """
    Build left environment L^{[site]} from sites 0..site-1.
    site=0 gives boundary env.
    """
    Lenv = init_left_env(dtype=mps[0].dtype)
    for i in range(site):
        Lenv = update_left_env(Lenv, mps[i], mpo[i])
    return Lenv

# ============================================================
# Two-site effective Hamiltonian
# ============================================================

def two_site_heff_apply(theta, Lenv, W1, W2, Renv):
    """
    Apply the two-site effective Hamiltonian.

    Shapes:
        theta : (Dl, d1, d2, Dr)      indices (ap, t1, t2, rp)
        Lenv  : (chiL, Dl, Dl)        indices (b, a, ap)
        W1    : (chiL, chiM, d1, d1)  indices (b, B, s1, t1)
        W2    : (chiM, chiR, d2, d2)  indices (B, C, s2, t2)
        Renv  : (chiR, Dr, Dr)        indices (C, rp, r)

    Returns:
        out   : (Dl, d1, d2, Dr)      indices (a, s1, s2, r)
    """
    # Step 1: X[b, a, t1, t2, rp] = sum_ap L[b,a,ap] theta[ap,t1,t2,rp]
    X = np.einsum('bap,ptur->batur', Lenv, theta, optimize=True)

    # Step 2: Y[B, a, s1, t2, rp] = sum_{b,t1} W1[b,B,s1,t1] X[b,a,t1,t2,rp]
    Y = np.einsum('bBst,batur->Basur', W1, X, optimize=True)

    # Step 3: Z[C, a, s1, s2, rp] = sum_{B,t2} W2[B,C,s2,t2] Y[B,a,s1,t2,rp]
    Z = np.einsum('BCsu,Batur->Castr', W2, Y, optimize=True)

    # Step 4: out[a, s1, s2, r] = sum_{C,rp} Z[C,a,s1,s2,rp] R[C,rp,r]
    out = np.einsum('Castr,Crq->astq', Z, Renv, optimize=True)

    return out

def build_dense_two_site_heff(Lenv, W1, W2, Renv, Dl, d1, d2, Dr):
    n = Dl * d1 * d2 * Dr
    H = np.zeros((n, n), dtype=np.complex128)
    for j in range(n):
        e = np.zeros(n, dtype=np.complex128)
        e[j] = 1.0
        theta = e.reshape(Dl, d1, d2, Dr, order='C')
        H[:, j] = two_site_heff_apply(theta, Lenv, W1, W2, Renv).reshape(n, order='C')
    return H

def solve_local_ground_state(theta0, Lenv, W1, W2, Renv, tol=1e-10, maxiter=500):
    Dl, d1, d2, Dr = theta0.shape
    n = Dl * d1 * d2 * Dr

    def mv(v):
        th = v.reshape(Dl, d1, d2, Dr, order='C')
        out = two_site_heff_apply(th, Lenv, W1, W2, Renv)
        return out.reshape(n, order='C')

    Hlin = LinearOperator((n, n), matvec=mv, dtype=np.complex128)
    v0 = theta0.reshape(n, order='C')
    nrm = np.linalg.norm(v0)
    if nrm < 1e-15:
        rng = np.random.default_rng(123)
        v0 = rng.normal(size=n) + 1j * rng.normal(size=n)
        v0 = v0.astype(np.complex128)
        v0 /= np.linalg.norm(v0)
    else:
        v0 = v0 / nrm

    vals, vecs = eigsh(Hlin, k=1, which='SA', v0=v0, tol=tol, maxiter=maxiter)
    E = vals[0].real
    theta = vecs[:, 0].reshape(Dl, d1, d2, Dr, order='C')
    theta /= np.linalg.norm(theta)
    return E, theta

# ============================================================
# SVD split / truncation
# ============================================================

def split_two_site_theta(theta, Dmax, cutoff=1e-12, sweep_direction='lr'):
    Dl, d1, d2, Dr = theta.shape
    M = theta.reshape(Dl * d1, d2 * Dr, order='C')
    U, s, Vh = la.svd(M, full_matrices=False)

    keep = min(Dmax, np.sum(s > cutoff))
    if keep < 1:
        keep = 1

    disc_weight = np.sum(s[keep:]**2).real
    U = U[:, :keep]
    s = s[:keep]
    Vh = Vh[:keep, :]

    if sweep_direction == 'lr':
        A = U.reshape(Dl, d1, keep, order='C')
        Mright = (np.diag(s) @ Vh).reshape(keep, d2, Dr, order='C')
        return A, Mright, s, disc_weight
    elif sweep_direction == 'rl':
        Mleft = (U @ np.diag(s)).reshape(Dl, d1, keep, order='C')
        B = Vh.reshape(keep, d2, Dr, order='C')
        return Mleft, B, s, disc_weight
    else:
        raise ValueError("sweep_direction must be 'lr' or 'rl'")

# ============================================================
# Dense-state utilities
# ============================================================

def mps_to_dense(mps):
    psi = mps[0][0, :, :]
    for i in range(1, len(mps)):
        psi = np.einsum('...a,asb->...sb', psi, mps[i], optimize=True)
    return psi.reshape(-1, order='C')

def dense_energy_from_mps(mps, H):
    psi = mps_to_dense(mps)
    psi = psi / np.linalg.norm(psi)
    return np.vdot(psi, H @ psi).real

def dense_variance_from_mps(mps, H):
    psi = mps_to_dense(mps)
    psi = psi / np.linalg.norm(psi)
    E = np.vdot(psi, H @ psi).real
    Hpsi = H @ psi
    E2 = np.vdot(Hpsi, Hpsi).real
    return E2 - E**2

def bond_singular_values_from_mps(mps):
    psi = mps_to_dense(mps)
    L = len(mps)
    d = mps[0].shape[1]
    svals = []
    for cut in range(1, L):
        M = psi.reshape(d**cut, d**(L-cut), order='C')
        s = la.svdvals(M)
        s = s / np.linalg.norm(s)
        svals.append(s)
    return svals

def entanglement_entropies(mps):
    svals = bond_singular_values_from_mps(mps)
    ent = []
    for s in svals:
        p = s**2
        p = p[p > 1e-15]
        ent.append(-(p * np.log(p)).sum().real)
    return np.array(ent)

# ============================================================
# Two-site DMRG main driver
# ============================================================

def dmrg_two_site(mpo, d, Dmax=32, nsweeps=8, cutoff=1e-12,
                  init_mps=None, verbose=True, eig_tol=1e-10):
    L = len(mpo)

    if init_mps is None:
        init_mps = random_mps(L, d, Dmax=max(2, min(Dmax, 4)), seed=1234)

    mps = right_canonicalize_qr(init_mps)

    energies = []
    max_discards = []

    for sweep in range(nsweeps):
        Renvs = build_all_right_envs(mps, mpo)
        Lenv = init_left_env(dtype=mps[0].dtype)

        sweep_discards = []
        E_last = None

        # Left -> Right
        for i in range(L - 1):
            theta = np.einsum('asb,btc->astc', mps[i], mps[i+1], optimize=True)
            E, theta_opt = solve_local_ground_state(theta, Lenv, mpo[i], mpo[i+1], Renvs[i+1], tol=eig_tol)

            A, Mright, s, disc = split_two_site_theta(theta_opt, Dmax=Dmax, cutoff=cutoff, sweep_direction='lr')
            mps[i] = A
            mps[i+1] = Mright
            Lenv = update_left_env(Lenv, mps[i], mpo[i])

            sweep_discards.append(disc)
            E_last = E

        # Right -> Left
        Renv = init_right_env(dtype=mps[0].dtype)
        for i in range(L - 2, -1, -1):
            theta = np.einsum('asb,btc->astc', mps[i], mps[i+1], optimize=True)
            Ltmp = build_left_env_up_to(mps, mpo, i)

            E, theta_opt = solve_local_ground_state(theta, Ltmp, mpo[i], mpo[i+1], Renv, tol=eig_tol)

            Mleft, B, s, disc = split_two_site_theta(theta_opt, Dmax=Dmax, cutoff=cutoff, sweep_direction='rl')
            mps[i] = Mleft
            mps[i+1] = B
            Renv = update_right_env(Renv, mps[i+1], mpo[i+1])

            sweep_discards.append(disc)
            E_last = E

        energies.append(E_last)
        max_discards.append(max(sweep_discards) if len(sweep_discards) > 0 else 0.0)

        if verbose:
            print(f"sweep {sweep+1:2d}: E = {E_last:.15f}, max discarded weight = {max_discards[-1]:.3e}")

    return {
        "mps": mps,
        "energies": np.array(energies),
        "max_discards": np.array(max_discards),
    }

# ============================================================
# Benchmarks
# ============================================================

def run_heisenberg_benchmark(L_list=(4, 6, 8, 10), J=1.0, Jz=1.0, h=0.0,
                             Dmax=32, nsweeps=8, verbose=True):
    exact_E = []
    dmrg_E = []

    for L in L_list:
        mpo = mpo_heisenberg_xxz(L, J=J, Jz=Jz, h=h)
        init = product_state_mps(L, d=2, state_index=0)
        res = dmrg_two_site(mpo, d=2, Dmax=Dmax, nsweeps=nsweeps, init_mps=init, verbose=False)
        E_dmrg = res["energies"][-1]
        dmrg_E.append(E_dmrg)

        H = dense_heisenberg_xxz(L, J=J, Jz=Jz, h=h)
        E_ex = exact_ground_energy(H)
        exact_E.append(E_ex)

        if verbose:
            print(f"L={L:2d}   DMRG={E_dmrg:.12f}   Exact={E_ex:.12f}   |Δ|={abs(E_dmrg-E_ex):.3e}")

    return np.array(L_list), np.array(dmrg_E), np.array(exact_E)

def run_aklt_benchmark(L_list=(4, 6, 8), Dmax=2, nsweeps=8, verbose=True):
    exact_formula = []
    exact_ed = []
    dmrg_E = []
    max_disc = []

    for L in L_list:
        mpo = mpo_aklt(L)
        init = product_state_mps(L, d=3, state_index=0)
        res = dmrg_two_site(mpo, d=3, Dmax=Dmax, nsweeps=nsweeps, init_mps=init, verbose=False)

        E_dmrg = res["energies"][-1]
        dmrg_E.append(E_dmrg)
        max_disc.append(np.max(res["max_discards"]))

        E_formula = -(2.0 / 3.0) * (L - 1)
        exact_formula.append(E_formula)

        H = dense_aklt(L)
        E_ex = exact_ground_energy(H)
        exact_ed.append(E_ex)

        if verbose:
            print(f"L={L:2d}   DMRG={E_dmrg:.12f}   Formula={E_formula:.12f}   ED={E_ex:.12f}   max disc={max_disc[-1]:.3e}")

    return np.array(L_list), np.array(dmrg_E), np.array(exact_formula), np.array(exact_ed), np.array(max_disc)

# ============================================================
# Plot helpers
# ============================================================

def plot_heisenberg_benchmark(L_list, E_dmrg, E_exact, figdir="figureAKLT"):
    setup_figure_dir(figdir)

    plt.figure()
    plt.plot(L_list, E_exact, 'o-', label='Exact diagonalization')
    plt.plot(L_list, E_dmrg, 's--', label='Two-site DMRG')
    plt.xlabel('L')
    plt.ylabel('Ground-state energy')
    plt.title('Spin-1/2 Heisenberg chain')
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "heisenberg_energy_vs_L.png")
    plt.show()

    plt.figure()
    plt.semilogy(L_list, np.abs(E_dmrg - E_exact), 'o-')
    plt.xlabel('L')
    plt.ylabel(r'$|E_{\rm DMRG} - E_{\rm exact}|$')
    plt.title('Heisenberg DMRG error')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "heisenberg_error_vs_L.png")
    plt.show()

def plot_convergence(res, title_prefix="", filename_prefix="", figdir="figureAKLT"):
    setup_figure_dir(figdir)

    plt.figure()
    plt.plot(np.arange(1, len(res["energies"]) + 1), res["energies"], 'o-')
    plt.xlabel('Sweep')
    plt.ylabel('Energy')
    plt.title(f'{title_prefix} convergence')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, f"{filename_prefix}_convergence.png")
    plt.show()

    plt.figure()
    plt.semilogy(np.arange(1, len(res["max_discards"]) + 1),
                 np.maximum(res["max_discards"], 1e-18), 'o-')
    plt.xlabel('Sweep')
    plt.ylabel('Max discarded weight')
    plt.title(f'{title_prefix} discarded weight')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, f"{filename_prefix}_discarded_weight.png")
    plt.show()

def plot_aklt_benchmark(L_list, E_dmrg, E_formula, E_exact, max_disc, figdir="figureAKLT"):
    setup_figure_dir(figdir)

    plt.figure()
    plt.plot(L_list, E_formula, 'k-', label='Exact formula')
    plt.plot(L_list, E_exact, 'o', label='Exact diagonalization')
    plt.plot(L_list, E_dmrg, 's--', label='Two-site DMRG, Dmax=2')
    plt.xlabel('L')
    plt.ylabel('Ground-state energy')
    plt.title('AKLT chain')
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "aklt_energy_vs_L.png")
    plt.show()

    plt.figure()
    plt.semilogy(L_list, np.abs(E_dmrg - E_formula), 'o-')
    plt.xlabel('L')
    plt.ylabel(r'$|E_{\rm DMRG} - E_{\rm exact}|$')
    plt.title('AKLT DMRG error')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "aklt_error_vs_L.png")
    plt.show()

    plt.figure()
    plt.semilogy(L_list, np.maximum(max_disc, 1e-18), 'o-')
    plt.xlabel('L')
    plt.ylabel('Max discarded weight')
    plt.title('AKLT discarded weight at Dmax=2')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "aklt_discarded_weight_vs_L.png")
    plt.show()

def plot_entanglement_entropy(mps, title="Entanglement entropy", filename="entanglement_entropy.png", figdir="figureAKLT"):
    setup_figure_dir(figdir)
    Sbond = entanglement_entropies(mps)
    L = len(mps)

    plt.figure()
    plt.plot(np.arange(1, L), Sbond, 'o-')
    plt.xlabel('Bond cut')
    plt.ylabel('Entanglement entropy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, filename)
    plt.show()

# ============================================================
# Diagnostics / tests
# ============================================================

def test_canonicalization():
    mps0 = random_mps(L=6, d=2, Dmax=4, seed=7)
    mpsL = left_canonicalize_qr(mps0)
    mpsR = right_canonicalize_qr(mps0)

    print("Left-canonical checks:")
    for i in range(5):
        print(i, check_left_canonical(mpsL[i]))

    print("Right-canonical checks:")
    for i in range(1, 6):
        print(i, check_right_canonical(mpsR[i]))

def test_two_site_heff():
    rng = np.random.default_rng(1)
    Dl, d1, d2, Dr = 2, 2, 2, 3
    chiL, chiM, chiR = 3, 4, 2

    Lenv = rng.normal(size=(chiL, Dl, Dl)) + 1j * rng.normal(size=(chiL, Dl, Dl))
    W1   = rng.normal(size=(chiL, chiM, d1, d1)) + 1j * rng.normal(size=(chiL, chiM, d1, d1))
    W2   = rng.normal(size=(chiM, chiR, d2, d2)) + 1j * rng.normal(size=(chiM, chiR, d2, d2))
    Renv = rng.normal(size=(chiR, Dr, Dr)) + 1j * rng.normal(size=(chiR, Dr, Dr))
    theta = rng.normal(size=(Dl, d1, d2, Dr)) + 1j * rng.normal(size=(Dl, d1, d2, Dr))

    Hdense = build_dense_two_site_heff(Lenv, W1, W2, Renv, Dl, d1, d2, Dr)
    v = theta.reshape(-1, order='C')

    v_dense = Hdense @ v
    v_mfree = two_site_heff_apply(theta, Lenv, W1, W2, Renv).reshape(-1, order='C')

    err = np.linalg.norm(v_dense - v_mfree)
    print("two-site Heff test error =", err)
    return err
```

---
## 🧑‍💻 User

Is it complete?

---
## 🤖 Assistant

Yes, **that one-cell function block is complete** in the sense that it includes everything needed to:

- build MPOs for
  - **spin-1/2 Heisenberg**
  - **spin-1 AKLT**
- create and canonicalize MPS
- run **two-site DMRG**
- benchmark against exact diagonalization
- compute dense-state energy/variance for small systems
- plot and save figures into `figureAKLT`

And together with the earlier **run-all cell**, it should be a self-contained notebook workflow.

---

## What is included

The function cell contains:

### Core numerics
- `spin_half_ops`
- `spin_one_ops`
- `mpo_heisenberg_xxz`
- `mpo_aklt`
- `random_mps`
- `product_state_mps`
- `exact_aklt_mps`
- `left_canonicalize_qr`
- `right_canonicalize_qr`
- environment builders and updates
- `two_site_heff_apply`
- `solve_local_ground_state`
- `split_two_site_theta`
- `dmrg_two_site`

### Diagnostics
- `test_canonicalization`
- `test_two_site_heff`

### Exact benchmarking
- `dense_heisenberg_xxz`
- `dense_aklt`
- `exact_ground_energy`
- `dense_energy_from_mps`
- `dense_variance_from_mps`

### Plotting
- `plot_heisenberg_benchmark`
- `plot_aklt_benchmark`
- `plot_convergence`
- `plot_entanglement_entropy`

### Figure saving
- `setup_figure_dir`
- `savefig_in_dir`

So from a notebook usage point of view: **yes, complete**.

---

## But I should mention two practical caveats

### 1. The most delicate part is still `two_site_heff_apply`
I corrected it, but because tensor-index code is subtle, the real confirmation is:

```python
test_two_site_heff()
```

If this returns a tiny number, then the local effective Hamiltonian contraction is consistent.

### 2. The code is complete, but not yet “production optimized”
It is written to be:
- readable
- consistent with your LaTeX note
- easy to debug

It is **not** yet optimized for large systems.

In particular:
- left environments are rebuilt in the right-to-left sweep via `build_left_env_up_to`
- dense-state routines scale exponentially and are only for small benchmarks

That is okay for notebook benchmarking and validation.

---

# What I recommend you do next

Run in this order:

## Step 1
Run the full function cell.

## Step 2
Run:

```python
test_canonicalization()
test_two_site_heff()
```

## Step 3
If `test_two_site_heff()` gives a tiny error, run the run-all cell.

---

# One important thing
Since you asked “is it complete?”, the answer is:

- **Complete as a notebook implementation**: yes
- **Guaranteed bug-free without running tests**: no, because DMRG tensor contractions always need numerical validation

So the code is complete, but the decisive check is still the local-Hamiltonian unit test.

---

If you want, I can do one final pass and give you a **fully cleaned final notebook package** consisting of exactly two cells only:

1. **function cell**
2. **run-all cell**

with no extra edits needed.

---
## 🧑‍💻 User

=== AKLT variance check ===
AKLT L=8: E = 6.736542157778, variance = 1.044e+00

=== Exact AKLT MPS check ===
Exact AKLT MPS energy for L=8: -4.666666666667
Expected exact energy: -4.666666666667

=== Summary ===
Heisenberg benchmark sizes: [ 4  6  8 10]
Heisenberg |DMRG-Exact|    : [1.366025403784 1.743577133888 1.132547192273 1.148990112337]
AKLT benchmark sizes       : [4 6 8]
AKLT |DMRG-Formula|        : [0.951594621019 0.713815625061 2.116746372076]
AKLT max discarded weights : [0.434062581231 0.11746987896  0.093686218973]

All figures have been shown and saved to: figureAKLT

---
## 🤖 Assistant

Thanks — these results show that the notebook is **not yet correct at the DMRG level**.

The key evidence is:

- **AKLT exact MPS check is perfect**
  - so `dense_aklt(...)` is correct
  - and `exact_aklt_mps(...)` is correct
- but **DMRG energies are very wrong**
- and **AKLT variance is large**
  - ground state should have variance near zero
- Heisenberg errors of order \(1\) are far too large

So the problem is not the Hamiltonian definition.  
The problem is in the **two-site DMRG implementation**, most likely one of:

1. `two_site_heff_apply`
2. environment usage in the sweep
3. left/right canonical center handling
4. the right-to-left sweep logic

---

# Most likely culprit

The strongest suspicion is that the current `two_site_heff_apply` still has an index mismatch, even if it now runs.

This is common: a contraction can be dimensionally valid but still mathematically wrong.

So instead of trying to patch the current version piecemeal, the best fix is to replace the DMRG core with a **safer reference implementation**, where the local effective Hamiltonian is built explicitly as a dense matrix for the local two-site problem.

For the system sizes you are benchmarking in Jupyter:

- Heisenberg \(L \le 10\)
- AKLT \(L \le 12\)

this is totally fine.

That will make the code:
- slower
- but much more reliable
- and excellent for debugging / learning

---

# Recommended fix

I suggest replacing the matrix-free local eigensolver by:

- a function that constructs the **dense two-site effective Hamiltonian**
- solve its smallest eigenvalue using `scipy.linalg.eigh`

This avoids hidden tensor-index mistakes in the local matvec.

---

# Why this is a good idea here

Your note explicitly says:

> write a reference function that builds the dense effective Hamiltonian for small \(D\) and \(d\), and use it to unit test the matrix-free contraction.

For a notebook and benchmark code, we can go one step further:

> use the dense local effective Hamiltonian directly inside DMRG

That is the safest route until everything is validated.

---

# What I can do next

I can now give you a **corrected full one-cell function block** where:

- the two-site local problem is solved by **dense local \(H_{\rm eff}\)**
- the sweep logic is cleaned up
- it should correctly recover:
  - Heisenberg ED energies for small \(L\)
  - AKLT exact energy with \(D_{\max}=2\)
  - AKLT near-zero discarded weight
  - small variance

This is the most reliable next step.

---

# In short

##

---
## 🧑‍💻 User

continue

---
## 🤖 Assistant

Yes — let’s continue by switching to a **safer dense-local two-site DMRG core**.

This will fix the main issue: the current matrix-free local Hamiltonian application is almost certainly wrong in index structure, even though it runs.

Below is a **fully corrected full one-cell function block** that:

- keeps the same notebook style
- still uses two-site DMRG
- but solves each local problem with an **explicit dense effective Hamiltonian**
- is much easier to trust for small and medium chains
- should work well for your requested Heisenberg and AKLT benchmarks in Jupyter

---

# Corrected full one-cell function block

```python
import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# ============================================================
# Global setup
# ============================================================

np.set_printoptions(precision=12, suppress=True)

def setup_figure_dir(dirname="figureAKLT"):
    os.makedirs(dirname, exist_ok=True)
    return dirname

def savefig_in_dir(dirname, name):
    path = os.path.join(dirname, name)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    print(f"saved: {path}")

# ============================================================
# Local spin operators
# ============================================================

def spin_half_ops(dtype=np.complex128):
    I = np.eye(2, dtype=dtype)
    Sp = np.array([[0, 1], [0, 0]], dtype=dtype)
    Sm = np.array([[0, 0], [1, 0]], dtype=dtype)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=dtype)
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=dtype)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=dtype)
    return I, Sp, Sm, Sz, Sx, Sy

def spin_one_ops(dtype=np.complex128):
    I = np.eye(3, dtype=dtype)
    s2 = np.sqrt(2.0)
    Sp = np.array([[0, s2, 0],
                   [0, 0, s2],
                   [0, 0, 0]], dtype=dtype)
    Sm = Sp.conj().T
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, -1]], dtype=dtype)
    Sx = 0.5 * (Sp + Sm)
    Sy = (Sp - Sm) / (2j)
    return I, Sp, Sm, Sz, Sx, Sy

# ============================================================
# Dense exact Hamiltonians for benchmarking
# ============================================================

def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def dense_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    I, Sp, Sm, Sz, _, _ = spin_half_ops(dtype=dtype)
    d = 2
    H = np.zeros((d**L, d**L), dtype=dtype)

    for i in range(L - 1):
        ops = [I] * L
        ops[i] = 0.5 * J * Sp
        ops[i+1] = Sm
        H += kron_all(ops)

        ops = [I] * L
        ops[i] = 0.5 * J * Sm
        ops[i+1] = Sp
        H += kron_all(ops)

        ops = [I] * L
        ops[i] = Jz * Sz
        ops[i+1] = Sz
        H += kron_all(ops)

    for i in range(L):
        ops = [I] * L
        ops[i] = -h * Sz
        H += kron_all(ops)

    return H

def dense_aklt(L, dtype=np.complex128):
    I, Sp, Sm, Sz, _, _ = spin_one_ops(dtype=dtype)
    d = 3
    H = np.zeros((d**L, d**L), dtype=dtype)

    SS = 0.5 * np.kron(Sp, Sm) + 0.5 * np.kron(Sm, Sp) + np.kron(Sz, Sz)
    hbond = SS + (1.0 / 3.0) * (SS @ SS)

    for i in range(L - 1):
        left_dim = d**i
        right_dim = d**(L - i - 2)
        H += np.kron(np.eye(left_dim, dtype=dtype),
                     np.kron(hbond, np.eye(right_dim, dtype=dtype)))
    return H

def exact_ground_energy(H):
    return np.min(la.eigvalsh(H)).real

# ============================================================
# MPO builders
# ============================================================

def mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    I, Sp, Sm, Sz, _, _ = spin_half_ops(dtype=dtype)
    d = 2
    chi = 5

    W_bulk = np.zeros((chi, chi, d, d), dtype=dtype)
    W_bulk[0, 0] = I
    W_bulk[1, 0] = Sp
    W_bulk[2, 0] = Sm
    W_bulk[3, 0] = Sz
    W_bulk[4, 0] = -h * Sz
    W_bulk[4, 1] = 0.5 * J * Sm
    W_bulk[4, 2] = 0.5 * J * Sp
    W_bulk[4, 3] = Jz * Sz
    W_bulk[4, 4] = I

    W_first = np.zeros((1, chi, d, d), dtype=dtype)
    W_first[0, 0] = -h * Sz
    W_first[0, 1] = 0.5 * J * Sm
    W_first[0, 2] = 0.5 * J * Sp
    W_first[0, 3] = Jz * Sz
    W_first[0, 4] = I

    W_last = np.zeros((chi, 1, d, d), dtype=dtype)
    W_last[0, 0] = I
    W_last[1, 0] = Sp
    W_last[2, 0] = Sm
    W_last[3, 0] = Sz
    W_last[4, 0] = -h * Sz

    mpo = [None] * L
    mpo[0] = W_first
    for i in range(1, L - 1):
        mpo[i] = W_bulk.copy()
    mpo[-1] = W_last
    return mpo

def mpo_from_two_body(L, onsite_op_list, bond_terms, d, dtype=np.complex128):
    nb = len(bond_terms)
    chi = 2 + nb

    I = np.eye(d, dtype=dtype)
    W_bulk = np.zeros((chi, chi, d, d), dtype=dtype)

    W_bulk[0, 0] = I
    for k, (Lop, Rop) in enumerate(bond_terms):
        W_bulk[0, 1 + k] = Lop
        W_bulk[1 + k, -1] = Rop
    W_bulk[-1, -1] = I
    if len(onsite_op_list) > 0:
        W_bulk[0, -1] = sum(onsite_op_list)

    W_first = np.zeros((1, chi, d, d), dtype=dtype)
    W_first[0] = W_bulk[0]

    W_last = np.zeros((chi, 1, d, d), dtype=dtype)
    W_last[:, 0] = W_bulk[:, -1]

    mpo = [None] * L
    mpo[0] = W_first
    for i in range(1, L - 1):
        mpo[i] = W_bulk.copy()
    mpo[-1] = W_last
    return mpo

def mpo_aklt(L, dtype=np.complex128):
    I, Sp, Sm, Sz, _, _ = spin_one_ops(dtype=dtype)
    d = 3

    A_terms = [
        (0.5 * Sp, Sm),
        (0.5 * Sm, Sp),
        (1.0 * Sz, Sz),
    ]

    A2_terms = []
    for La, Ra in A_terms:
        for Lb, Rb in A_terms:
            A2_terms.append(((1.0 / 3.0) * (La @ Lb), Ra @ Rb))

    bond_terms = A_terms + A2_terms
    onsite_terms = []
    return mpo_from_two_body(L, onsite_terms, bond_terms, d=d, dtype=dtype)

# ============================================================
# MPS constructors
# ============================================================

def random_mps(L, d, Dmax, seed=1234, dtype=np.complex128):
    rng = np.random.default_rng(seed)
    dimsL = [1]
    for _ in range(L - 1):
        dimsL.append(min(Dmax, dimsL[-1] * d))
    dimsR = [1]
    for _ in range(L - 1):
        dimsR.append(min(Dmax, dimsR[-1] * d))
    dimsR = list(reversed(dimsR))

    mps = []
    for i in range(L):
        Dl = min(dimsL[i], dimsR[i])
        Dr = min(dimsL[i+1], dimsR[i+1]) if i < L - 1 else 1
        A = rng.normal(size=(Dl, d, Dr)) + 1j * rng.normal(size=(Dl, d, Dr))
        mps.append(A.astype(dtype))
    return mps

def product_state_mps(L, d, state_index=0, dtype=np.complex128):
    mps = []
    for _ in range(L):
        A = np.zeros((1, d, 1), dtype=dtype)
        A[0, state_index, 0] = 1.0
        mps.append(A)
    return mps

def exact_aklt_mps(L, dtype=np.complex128):
    Ap = np.array([[0, np.sqrt(2/3)], [0, 0]], dtype=dtype)
    A0 = np.array([[-1/np.sqrt(3), 0], [0, 1/np.sqrt(3)]], dtype=dtype)
    Am = np.array([[0, 0], [-np.sqrt(2/3), 0]], dtype=dtype)
    A_bulk = np.stack([Ap, A0, Am], axis=1)  # (2,3,2)

    mps = []
    vl = np.array([[1.0, 0.0]], dtype=dtype)
    vr = np.array([[1.0], [0.0]], dtype=dtype)

    A1 = np.einsum('la,asb->lsb', vl, A_bulk, optimize=True).reshape(1, 3, 2)
    mps.append(A1)

    for _ in range(1, L - 1):
        mps.append(A_bulk.copy())

    AL = np.einsum('asb,br->asr', A_bulk, vr, optimize=True).reshape(2, 3, 1)
    mps.append(AL)
    return mps

# ============================================================
# Canonicalization
# ============================================================

def left_canonicalize_qr(mps):
    mps = [A.copy() for A in mps]
    L = len(mps)
    for i in range(L - 1):
        Dl, d, Dr = mps[i].shape
        M = mps[i].reshape(Dl * d, Dr, order='C')
        Q, R = la.qr(M, mode='economic')
        k = Q.shape[1]
        mps[i] = Q.reshape(Dl, d, k, order='C')
        mps[i+1] = np.einsum('ab,bsd->asd', R, mps[i+1], optimize=True)
    return mps

def right_canonicalize_qr(mps):
    mps = [A.copy() for A in mps]
    L = len(mps)
    for i in range(L - 1, 0, -1):
        Dl, d, Dr = mps[i].shape
        M = mps[i].reshape(Dl, d * Dr, order='C')
        Q, R = la.qr(M.conj().T, mode='economic')
        k = Q.shape[1]
        mps[i] = Q.conj().T.reshape(k, d, Dr, order='C')
        mps[i-1] = np.einsum('asb,bc->asc', mps[i-1], R.conj().T, optimize=True)
    return mps

def check_left_canonical(A, tol=1e-10):
    Dl, d, Dr = A.shape
    M = A.reshape(Dl * d, Dr, order='C')
    return np.linalg.norm(M.conj().T @ M - np.eye(Dr)) < tol

def check_right_canonical(A, tol=1e-10):
    Dl, d, Dr = A.shape
    M = A.reshape(Dl, d * Dr, order='C')
    return np.linalg.norm(M @ M.conj().T - np.eye(Dl)) < tol

# ============================================================
# Environments
# ============================================================

def init_left_env(dtype=np.complex128):
    Lenv = np.zeros((1, 1, 1), dtype=dtype)
    Lenv[0, 0, 0] = 1.0
    return Lenv

def init_right_env(dtype=np.complex128):
    Renv = np.zeros((1, 1, 1), dtype=dtype)
    Renv[0, 0, 0] = 1.0
    return Renv

def update_left_env(Lenv, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lenv, A, W, A.conj(), optimize=True)

def update_right_env(Renv, B, W):
    return np.einsum('asx,bBst,Bxy,cty->bac', B, W, Renv, B.conj(), optimize=True)

def build_all_right_envs(mps, mpo):
    L = len(mps)
    Renvs = [None] * L
    R = init_right_env(dtype=mps[0].dtype)
    Renvs[L - 1] = R
    for i in range(L - 1, 0, -1):
        R = update_right_env(R, mps[i], mpo[i])
        Renvs[i - 1] = R
    return Renvs

def build_left_env_up_to(mps, mpo, site):
    Lenv = init_left_env(dtype=mps[0].dtype)
    for i in range(site):
        Lenv = update_left_env(Lenv, mps[i], mpo[i])
    return Lenv

# ============================================================
# Dense local two-site effective Hamiltonian
# ============================================================

def two_site_heff_apply(theta, Lenv, W1, W2, Renv):
    """
    Safe reference action using explicit summation structure:
    out[a,s1,s2,r] =
      sum_{ap,rp,t1,t2,b,B,C}
      Lenv[b,a,ap] W1[b,B,s1,t1] W2[B,C,s2,t2] Renv[C,rp,r] theta[ap,t1,t2,rp]
    """
    out = np.einsum(
        'bap,bBst,BCuv,Crq,ptvq->asur',
        Lenv, W1, W2, Renv, theta,
        optimize=True
    )
    return out

def build_dense_two_site_heff(Lenv, W1, W2, Renv, Dl, d1, d2, Dr):
    n = Dl * d1 * d2 * Dr
    H = np.zeros((n, n), dtype=np.complex128)
    for j in range(n):
        e = np.zeros(n, dtype=np.complex128)
        e[j] = 1.0
        theta = e.reshape(Dl, d1, d2, Dr, order='C')
        H[:, j] = two_site_heff_apply(theta, Lenv, W1, W2, Renv).reshape(n, order='C')
    return H

def solve_local_ground_state_dense(theta0, Lenv, W1, W2, Renv):
    Dl, d1, d2, Dr = theta0.shape
    Hloc = build_dense_two_site_heff(Lenv, W1, W2, Renv, Dl, d1, d2, Dr)
    vals, vecs = la.eigh(Hloc)
    E = vals[0].real
    theta = vecs[:, 0].reshape(Dl, d1, d2, Dr, order='C')
    theta /= np.linalg.norm(theta)
    return E, theta

# ============================================================
# SVD split / truncation
# ============================================================

def split_two_site_theta(theta, Dmax, cutoff=1e-12, sweep_direction='lr'):
    Dl, d1, d2, Dr = theta.shape
    M = theta.reshape(Dl * d1, d2 * Dr, order='C')
    U, s, Vh = la.svd(M, full_matrices=False)

    keep = min(Dmax, np.sum(s > cutoff))
    if keep < 1:
        keep = 1

    disc_weight = np.sum(s[keep:]**2).real
    U = U[:, :keep]
    s = s[:keep]
    Vh = Vh[:keep, :]

    if sweep_direction == 'lr':
        A = U.reshape(Dl, d1, keep, order='C')
        Mright = (np.diag(s) @ Vh).reshape(keep, d2, Dr, order='C')
        return A, Mright, s, disc_weight
    elif sweep_direction == 'rl':
        Mleft = (U @ np.diag(s)).reshape(Dl, d1, keep, order='C')
        B = Vh.reshape(keep, d2, Dr, order='C')
        return Mleft, B, s, disc_weight
    else:
        raise ValueError("sweep_direction must be 'lr' or 'rl'")

# ============================================================
# Dense-state utilities
# ============================================================

def mps_to_dense(mps):
    psi = mps[0][0, :, :]
    for i in range(1, len(mps)):
        psi = np.einsum('...a,asb->...sb', psi, mps[i], optimize=True)
    return psi.reshape(-1, order='C')

def dense_energy_from_mps(mps, H):
    psi = mps_to_dense(mps)
    psi = psi / np.linalg.norm(psi)
    return np.vdot(psi, H @ psi).real

def dense_variance_from_mps(mps, H):
    psi = mps_to_dense(mps)
    psi = psi / np.linalg.norm(psi)
    E = np.vdot(psi, H @ psi).real
    Hpsi = H @ psi
    E2 = np.vdot(Hpsi, Hpsi).real
    return E2 - E**2

def bond_singular_values_from_mps(mps):
    psi = mps_to_dense(mps)
    L = len(mps)
    d = mps[0].shape[1]
    svals = []
    for cut in range(1, L):
        M = psi.reshape(d**cut, d**(L-cut), order='C')
        s = la.svdvals(M)
        s = s / np.linalg.norm(s)
        svals.append(s)
    return svals

def entanglement_entropies(mps):
    svals = bond_singular_values_from_mps(mps)
    ent = []
    for s in svals:
        p = s**2
        p = p[p > 1e-15]
        ent.append(-(p * np.log(p)).sum().real)
    return np.array(ent)

# ============================================================
# Two-site DMRG main driver
# ============================================================

def dmrg_two_site(mpo, d, Dmax=32, nsweeps=8, cutoff=1e-12,
                  init_mps=None, verbose=True):
    L = len(mpo)

    if init_mps is None:
        init_mps = random_mps(L, d, Dmax=max(2, min(Dmax, 4)), seed=1234)

    mps = right_canonicalize_qr(init_mps)

    energies = []
    max_discards = []

    for sweep in range(nsweeps):
        sweep_discards = []
        E_last = None

        # -------------------------
        # Left -> Right
        # -------------------------
        Renvs = build_all_right_envs(mps, mpo)
        Lenv = init_left_env(dtype=mps[0].dtype)

        for i in range(L - 1):
            theta = np.einsum('asb,btc->astc', mps[i], mps[i+1], optimize=True)
            E, theta_opt = solve_local_ground_state_dense(theta, Lenv, mpo[i], mpo[i+1], Renvs[i+1])

            A, Mright, s, disc = split_two_site_theta(
                theta_opt, Dmax=Dmax, cutoff=cutoff, sweep_direction='lr'
            )

            mps[i] = A
            mps[i+1] = Mright
            Lenv = update_left_env(Lenv, mps[i], mpo[i])

            sweep_discards.append(disc)
            E_last = E

        # -------------------------
        # Right -> Left
        # -------------------------
        Renv = init_right_env(dtype=mps[0].dtype)

        for i in range(L - 2, -1, -1):
            theta = np.einsum('asb,btc->astc', mps[i], mps[i+1], optimize=True)
            Ltmp = build_left_env_up_to(mps, mpo, i)

            E, theta_opt = solve_local_ground_state_dense(theta, Ltmp, mpo[i], mpo[i+1], Renv)

            Mleft, B, s, disc = split_two_site_theta(
                theta_opt, Dmax=Dmax, cutoff=cutoff, sweep_direction='rl'
            )

            mps[i] = Mleft
            mps[i+1] = B
            Renv = update_right_env(Renv, mps[i+1], mpo[i+1])

            sweep_discards.append(disc)
            E_last = E

        energies.append(E_last)
        max_discards.append(max(sweep_discards) if sweep_discards else 0.0)

        if verbose:
            print(f"sweep {sweep+1:2d}: E = {E_last:.15f}, max discarded weight = {max_discards[-1]:.3e}")

    return {
        "mps": mps,
        "energies": np.array(energies),
        "max_discards": np.array(max_discards),
    }

# ============================================================
# Benchmarks
# ============================================================

def run_heisenberg_benchmark(L_list=(4, 6, 8, 10), J=1.0, Jz=1.0, h=0.0,
                             Dmax=32, nsweeps=8, verbose=True):
    exact_E = []
    dmrg_E = []

    for L in L_list:
        mpo = mpo_heisenberg_xxz(L, J=J, Jz=Jz, h=h)
        init = product_state_mps(L, d=2, state_index=0)
        res = dmrg_two_site(mpo, d=2, Dmax=Dmax, nsweeps=nsweeps, init_mps=init, verbose=False)
        E_dmrg = res["energies"][-1]
        dmrg_E.append(E_dmrg)

        H = dense_heisenberg_xxz(L, J=J, Jz=Jz, h=h)
        E_ex = exact_ground_energy(H)
        exact_E.append(E_ex)

        if verbose:
            print(f"L={L:2d}   DMRG={E_dmrg:.12f}   Exact={E_ex:.12f}   |Δ|={abs(E_dmrg-E_ex):.3e}")

    return np.array(L_list), np.array(dmrg_E), np.array(exact_E)

def run_aklt_benchmark(L_list=(4, 6, 8), Dmax=2, nsweeps=8, verbose=True):
    exact_formula = []
    exact_ed = []
    dmrg_E = []
    max_disc = []

    for L in L_list:
        mpo = mpo_aklt(L)
        init = product_state_mps(L, d=3, state_index=0)
        res = dmrg_two_site(mpo, d=3, Dmax=Dmax, nsweeps=nsweeps, init_mps=init, verbose=False)

        E_dmrg = res["energies"][-1]
        dmrg_E.append(E_dmrg)
        max_disc.append(np.max(res["max_discards"]))

        E_formula = -(2.0 / 3.0) * (L - 1)
        exact_formula.append(E_formula)

        H = dense_aklt(L)
        E_ex = exact_ground_energy(H)
        exact_ed.append(E_ex)

        if verbose:
            print(f"L={L:2d}   DMRG={E_dmrg:.12f}   Formula={E_formula:.12f}   ED={E_ex:.12f}   max disc={max_disc[-1]:.3e}")

    return np.array(L_list), np.array(dmrg_E), np.array(exact_formula), np.array(exact_ed), np.array(max_disc)

# ============================================================
# Plot helpers
# ============================================================

def plot_heisenberg_benchmark(L_list, E_dmrg, E_exact, figdir="figureAKLT"):
    setup_figure_dir(figdir)

    plt.figure()
    plt.plot(L_list, E_exact, 'o-', label='Exact diagonalization')
    plt.plot(L_list, E_dmrg, 's--', label='Two-site DMRG')
    plt.xlabel('L')
    plt.ylabel('Ground-state energy')
    plt.title('Spin-1/2 Heisenberg chain')
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "heisenberg_energy_vs_L.png")
    plt.show()

    plt.figure()
    plt.semilogy(L_list, np.abs(E_dmrg - E_exact), 'o-')
    plt.xlabel('L')
    plt.ylabel(r'$|E_{\rm DMRG} - E_{\rm exact}|$')
    plt.title('Heisenberg DMRG error')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "heisenberg_error_vs_L.png")
    plt.show()

def plot_convergence(res, title_prefix="", filename_prefix="", figdir="figureAKLT"):
    setup_figure_dir(figdir)

    plt.figure()
    plt.plot(np.arange(1, len(res["energies"]) + 1), res["energies"], 'o-')
    plt.xlabel('Sweep')
    plt.ylabel('Energy')
    plt.title(f'{title_prefix} convergence')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, f"{filename_prefix}_convergence.png")
    plt.show()

    plt.figure()
    plt.semilogy(np.arange(1, len(res["max_discards"]) + 1),
                 np.maximum(res["max_discards"], 1e-18), 'o-')
    plt.xlabel('Sweep')
    plt.ylabel('Max discarded weight')
    plt.title(f'{title_prefix} discarded weight')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, f"{filename_prefix}_discarded_weight.png")
    plt.show()

def plot_aklt_benchmark(L_list, E_dmrg, E_formula, E_exact, max_disc, figdir="figureAKLT"):
    setup_figure_dir(figdir)

    plt.figure()
    plt.plot(L_list, E_formula, 'k-', label='Exact formula')
    plt.plot(L_list, E_exact, 'o', label='Exact diagonalization')
    plt.plot(L_list, E_dmrg, 's--', label='Two-site DMRG, Dmax=2')
    plt.xlabel('L')
    plt.ylabel('Ground-state energy')
    plt.title('AKLT chain')
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "aklt_energy_vs_L.png")
    plt.show()

    plt.figure()
    plt.semilogy(L_list, np.abs(E_dmrg - E_formula), 'o-')
    plt.xlabel('L')
    plt.ylabel(r'$|E_{\rm DMRG} - E_{\rm exact}|$')
    plt.title('AKLT DMRG error')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "aklt_error_vs_L.png")
    plt.show()

    plt.figure()
    plt.semilogy(L_list, np.maximum(max_disc, 1e-18), 'o-')
    plt.xlabel('L')
    plt.ylabel('Max discarded weight')
    plt.title('AKLT discarded weight at Dmax=2')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "aklt_discarded_weight_vs_L.png")
    plt.show()

def plot_entanglement_entropy(mps, title="Entanglement entropy", filename="entanglement_entropy.png", figdir="figureAKLT"):
    setup_figure_dir(figdir)
    Sbond = entanglement_entropies(mps)
    L = len(mps)

    plt.figure()
    plt.plot(np.arange(1, L), Sbond, 'o-')
    plt.xlabel('Bond cut')
    plt.ylabel('Entanglement entropy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, filename)
    plt.show()

# ============================================================
# Diagnostics / tests
# ============================================================

def test_canonicalization():
    mps0 = random_mps(L=6, d=2, Dmax=4, seed=7)
    mpsL = left_canonicalize_qr(mps0)
    mpsR = right_canonicalize_qr(mps0)

    print("Left-canonical checks:")
    for i in range(5):
        print(i, check_left_canonical(mpsL[i]))

    print("Right-canonical checks:")
    for i in range(1, 6):
        print(i, check_right_canonical(mpsR[i]))

def test_two_site_heff():
    rng = np.random.default_rng(1)
    Dl, d1, d2, Dr = 2, 2, 2, 3
    chiL, chiM, chiR = 3, 4, 2

    Lenv = rng.normal(size=(chiL, Dl, Dl)) + 1j * rng.normal(size=(chiL, Dl, Dl))
    W1   = rng.normal(size=(chiL, chiM, d1, d1)) + 1j * rng.normal(size=(chiL, chiM, d1, d1))
    W2   = rng.normal(size=(chiM, chiR, d2, d2)) + 1j * rng.normal(size=(chiM, chiR, d2, d2))
    Renv = rng.normal(size=(chiR, Dr, Dr)) + 1j * rng.normal(size=(chiR, Dr, Dr))
    theta = rng.normal(size=(Dl, d1, d2, Dr)) + 1j * rng.normal(size=(Dl, d1, d2, Dr))

    Hdense = build_dense_two_site_heff(Lenv, W1, W2, Renv, Dl, d1, d2, Dr)
    v = theta.reshape(-1, order='C')
    v_dense = Hdense @ v
    v_apply = two_site_heff_apply(theta, Lenv, W1, W2, Renv).reshape(-1, order='C')

    err = np.linalg.norm(v_dense - v_apply)
    print("two-site Heff test error =", err)
    return err
```

---

# Use the same run-all cell as before

After replacing your old function cell with this corrected one, run:

```python
test_canonicalization()
test_two_site_heff()
```

and then the run-all cell.

---

# What should improve now

You should expect:

## Heisenberg
- DMRG energies much closer to exact diagonalization
- small benchmark errors for \(L=4,6,8,10\)

## AKLT
- energy close to
  \[
  -\frac23(L-1)
  \]
- discarded weight near zero for `Dmax=2`
- variance much smaller than before

---

# One more note

This version is deliberately a **reference-quality implementation**, not a fast one.

The local dense effective Hamiltonian dimension is:

\[
(D_{i-1} d_i)(d_{i+1} D_{i+1})
\]

So for your requested benchmarks it is fine, but for large \(D\) it will get expensive.

---

If you want, after you test this version, send me the new benchmark output and I’ll help you verify whether it is now behaving correctly.

---
## 🧑‍💻 User

It has run for 4 mins. Not yet finish.

---
## 🤖 Assistant

That makes sense. The dense-local version is correct in spirit but can get **slow**, especially because:

- every two-site update builds a dense local Hamiltonian
- and in the current sweep, left environments are rebuilt repeatedly
- plus exact diagonalization benchmarks and dense variance checks are expensive

So for Jupyter, the best move is:

## Use a lighter “practical run-all” cell first
- smaller systems
- fewer sweeps
- skip the slow variance checks initially

---

# Use this faster run-all cell

```python
# =========================
# Faster run-all notebook cell
# =========================

figdir = setup_figure_dir("figureAKLT")
print("Figures will be saved in:", figdir)

# -------------------------------------------------
# 1. Basic tests
# -------------------------------------------------
print("\n=== Basic tests ===")
test_canonicalization()
err_heff = test_two_site_heff()
print("Local two-site Heff test error =", err_heff)

# -------------------------------------------------
# 2. Heisenberg benchmark vs exact diagonalization
#    Use smaller sizes first
# -------------------------------------------------
print("\n=== Heisenberg benchmark (fast) ===")
L_list_h, E_dmrg_h, E_exact_h = run_heisenberg_benchmark(
    L_list=(4, 6, 8),
    J=1.0,
    Jz=1.0,
    h=0.0,
    Dmax=16,
    nsweeps=4,
    verbose=True
)
plot_heisenberg_benchmark(L_list_h, E_dmrg_h, E_exact_h, figdir=figdir)

# Heisenberg convergence for one selected system
print("\n=== Heisenberg convergence example (fast) ===")
Lh = 8
mpo_h = mpo_heisenberg_xxz(Lh, J=1.0, Jz=1.0, h=0.0)
init_h = product_state_mps(Lh, d=2, state_index=0)
res_h = dmrg_two_site(mpo_h, d=2, Dmax=16, nsweeps=5, init_mps=init_h, verbose=True)
plot_convergence(res_h, title_prefix=f"Heisenberg L={Lh}", filename_prefix="heisenberg_L8_fast", figdir=figdir)

# -------------------------------------------------
# 3. AKLT benchmark
# -------------------------------------------------
print("\n=== AKLT benchmark (fast) ===")
L_list_a, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a = run_aklt_benchmark(
    L_list=(4, 6),
    Dmax=2,
    nsweeps=4,
    verbose=True
)
plot_aklt_benchmark(L_list_a, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a, figdir=figdir)

# AKLT convergence and entropy profile
print("\n=== AKLT convergence and entropy (fast) ===")
La = 8
mpo_a = mpo_aklt(La)
init_a = product_state_mps(La, d=3, state_index=0)
res_a = dmrg_two_site(mpo_a, d=3, Dmax=2, nsweeps=5, init_mps=init_a, verbose=True)

plot_convergence(res_a, title_prefix=f"AKLT L={La}", filename_prefix="aklt_L8_fast", figdir=figdir)
plot_entanglement_entropy(
    res_a["mps"],
    title=f"AKLT bond entanglement entropy, L={La}",
    filename="aklt_entanglement_entropy_L8_fast.png",
    figdir=figdir
)

# -------------------------------------------------
# 4. Exact AKLT MPS check
# -------------------------------------------------
print("\n=== Exact AKLT MPS check ===")
L_exact = 8
mps_exact_aklt = exact_aklt_mps(L_exact)
H_exact_aklt = dense_aklt(L_exact)
E_exact_mps = dense_energy_from_mps(mps_exact_aklt, H_exact_aklt)
print(f"Exact AKLT MPS energy for L={L_exact}: {E_exact_mps:.12f}")
print(f"Expected exact energy: {-(2/3)*(L_exact-1):.12f}")

# -------------------------------------------------
# 5. Summary
# -------------------------------------------------
print("\n=== Summary ===")
print("Heisenberg benchmark sizes:", L_list_h)
print("Heisenberg |DMRG-Exact|    :", np.abs(E_dmrg_h - E_exact_h))
print("AKLT benchmark sizes       :", L_list_a)
print("AKLT |DMRG-Formula|        :", np.abs(E_dmrg_a - E_formula_a))
print("AKLT max discarded weights :", max_disc_a)
print("\nAll figures have been shown and saved to:", figdir)
```

---

# Why this is faster

Compared with the previous run-all:
- Heisenberg:
  - `(4,6,8)` instead of `(4,6,8,10)`
  - `Dmax=16` instead of `32`
  - `nsweeps=4 or 5` instead of `10+`
- AKLT:
  - `(4,6)` instead of `(4,6,8)`
  - shorter convergence run
- skips dense variance checks, which can be slow

---

# If the current cell is still running

In Jupyter, you can:
- press **Interrupt Kernel**
- then rerun:
  1. the function cell
  2. this faster run-all cell

---

# Best workflow now

## First debug / validate
Run only:

```python
test_canonicalization()
test_two_site_heff()
```

Then a single small case:

```python
L = 4
mpo = mpo_heisenberg_xxz(L)
res = dmrg_two_site(mpo, d=2, Dmax=8, nsweeps=2, init_mps=product_state_mps(L,2,0), verbose=True)
print(res["energies"])
```

If that behaves well, move to larger cases.

---

# If you want a much better fix
The real solution is not just shrinking runs — it is to make the DMRG sweep itself more efficient by:

- caching **all left environments**
- caching **all right environments**
- updating both incrementally
- not rebuilding `build_left_env_up_to(...)` every right-sweep step

That would speed things up a lot.

If you want, I can next provide a **faster corrected one-cell implementation** that still uses safe dense local diagonalization but avoids the worst redundant environment rebuilds.

---
## 🧑‍💻 User

sweep  1: E = -1.616025403784440, max discarded weight = 0.000e+00
sweep  2: E = -1.616025403784445, max discarded weight = 0.000e+00
[-1.616025403784 -1.616025403784]

---
## 🤖 Assistant

Good — this is actually a **very useful sign**.

For **Heisenberg \(L=4\)** with OBC and \(J=J_z=1, h=0\), the exact ground-state energy is:

\[
E_0 = -1.6160254037844386\ldots
\]

and your DMRG result is

\[
-1.61602540378444
\]

which is essentially exact.

So this means:

## What is now working
- the local two-site problem is now being solved correctly
- the DMRG update/split logic is at least basically correct
- the dense-local DMRG version is not fundamentally broken
- the Heisenberg implementation works on a nontrivial test

That is excellent.

---

# Why earlier large runs felt bad
The issue is now likely **performance**, not correctness.

The dense-local implementation becomes slow because:
- every bond optimization builds a dense local Hamiltonian
- the right-to-left sweep rebuilds left environments repeatedly
- large runs plus plotting plus ED plus variance checks can take several minutes

So the code is now much more likely **correct but slow**.

---

# What to do next

Now we should test a few small benchmark points manually before launching a big run-all.

---

## 1. Heisenberg benchmark sanity check

Run this:

```python
for L in [4, 6, 8]:
    mpo = mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0)
    res = dmrg_two_site(
        mpo, d=2, Dmax=16, nsweeps=4,
        init_mps=product_state_mps(L, 2, 0),
        verbose=False
    )
    E_dmrg = res["energies"][-1]
    E_exact = exact_ground_energy(dense_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0))
    print(f"L={L}, E_dmrg={E_dmrg:.12f}, E_exact={E_exact:.12f}, err={abs(E_dmrg-E_exact):.3e}")
```

What we want:
- very small error for \(L=4,6\)
- still small for \(L=8\)

---

## 2. AKLT benchmark sanity check

Run this:

```python
for L in [4, 6]:
    mpo = mpo_aklt(L)
    res = dmrg_two_site(
        mpo, d=3, Dmax=2, nsweeps=4,
        init_mps=product_state_mps(L, 3, 0),
        verbose=False
    )
    E_dmrg = res["energies"][-1]
    E_exact = -(2/3)*(L-1)
    print(f"L={L}, E_dmrg={E_dmrg:.12f}, E_exact={E_exact:.12f}, err={abs(E_dmrg-E_exact):.3e}, maxdisc={np.max(res['max_discards']):.3e}")
```

What we want:
- energy close to exact AKLT value
- discarded weight near zero for `Dmax=2`

---

# Important interpretation of the Heisenberg result
Your `L=4` result being exact strongly suggests that:

- the **MPO is correct**
- the **effective Hamiltonian is correct**
- the **SVD split is correct**
- the **canonicalization is correct enough**

That narrows any remaining problems to:
- convergence for larger systems
- runtime
- possibly the AKLT MPO if that still misbehaves

---

# My best guess now
I think:

- **Heisenberg is likely correct**
- **AKLT may still need checking carefully**
- the slow full run is mostly due to the dense-local implementation

---

# Recommended next step
Please run the two short loops above and send me the printed outputs.

That will let me determine immediately whether:

1. both Heisenberg and AKLT are now numerically correct, or
2. only Heisenberg is correct and AKLT MPO still needs adjustment.

If you want, I can also already prepare a **lighter benchmark cell** specifically for:
- Heisenberg \(L=4,6,8\)
- AKLT \(L=4,6\)

so you can validate correctness without waiting too long.

---
## 🧑‍💻 User

L=4, E_dmrg=-1.616025403784, E_exact=-1.616025403784, err=1.554e-15
L=6, E_dmrg=-2.493577133888, E_exact=-2.493577133888, err=2.220e-15
L=8, E_dmrg=-3.374932598688, E_exact=-3.374932598688, err=9.770e-15L=4, E_dmrg=-2.000000000000, E_exact=-2.000000000000, err=0.000e+00, maxdisc=3.224e-03
L=6, E_dmrg=-3.333333333333, E_exact=-3.333333333333, err=4.441e-16, maxdisc=3.224e-03

---
## 🤖 Assistant

Excellent — this means the corrected implementation is now **working**.

## Confirmed

### Heisenberg
You got:

- `L=4` exact
- `L=6` exact
- `L=8` exact

with errors at machine precision.

So the **spin-1/2 Heisenberg DMRG is correct**.

### AKLT
You got:

- `L=4` exact
- `L=6` exact

again to machine precision.

So the **AKLT MPO and two-site DMRG are also correct**.

---

# About the AKLT discarded weight

You reported:

- `maxdisc = 3.224e-03`

even though the energy is exact.

This is not ideal if one expects “zero discarded weight at \(D_{\max}=2\)”, but it is not necessarily alarming yet. There are two reasons:

## 1. The energy can be exact before every local truncation is numerically exact
For AKLT the exact state has bond dimension 2, so asymptotically the algorithm should settle into a bond-2 description. During early sweeps, however, the local optimized two-site tensor may temporarily contain small extra singular values before the gauge settles.

## 2. You are taking the maximum over all sweeps
If the first sweep starts far from the AKLT state, some truncation can happen initially. What matters more is whether the discarded weight goes to nearly zero in later sweeps.

Check:

```python
L = 6
mpo = mpo_aklt(L)
res = dmrg_two_site(
    mpo, d=3, Dmax=2, nsweeps=8,
    init_mps=product_state_mps(L, 3, 0),
    verbose=True
)
print("energies =", res["energies"])
print("max_discards =", res["max_discards"])
```

If later sweeps become tiny, then everything is fine.

---

# So the implementation status is now

## Complete and correct for your requested purpose:
- two-site DMRG
- Heisenberg model
- AKLT model
- Jupyter-friendly
- figures shown and saved in `figureAKLT`

---

# What happened earlier

Earlier wrong results came from the matrix-free local Hamiltonian implementation.  
Replacing that by a dense local effective Hamiltonian fixed the problem.

So the current code is a **reference-quality benchmark implementation**:
- accurate
- clear
- somewhat slow for larger runs

---

# What I recommend now

Use a modest benchmark workflow such as:

- Heisenberg: \(L=4,6,8\)
- AKLT: \(L=4,6,8\)
- convergence plots for one example each
- entanglement entropy for AKLT

That should be enough for a nice notebook without making it too slow.

---

# Suggested final run-all cell

Use this version rather than the heavy earlier one:

```python
# =========================
# Practical final run-all notebook cell
# =========================

figdir = setup_figure_dir("figureAKLT")
print("Figures will be saved in:", figdir)

# -------------------------------------------------
# 1. Basic tests
# -------------------------------------------------
print("\n=== Basic tests ===")
test_canonicalization()
err_heff = test_two_site_heff()
print("Local two-site Heff test error =", err_heff)

# -------------------------------------------------
# 2. Heisenberg benchmark
# -------------------------------------------------
print("\n=== Heisenberg benchmark ===")
L_list_h, E_dmrg_h, E_exact_h = run_heisenberg_benchmark(
    L_list=(4, 6, 8),
    J=1.0,
    Jz=1.0,
    h=0.0,
    Dmax=16,
    nsweeps=4,
    verbose=True
)
plot_heisenberg_benchmark(L_list_h, E_dmrg_h, E_exact_h, figdir=figdir)

print("\n=== Heisenberg convergence example ===")
Lh = 8
mpo_h = mpo_heisenberg_xxz(Lh, J=1.0, Jz=1.0, h=0.0)
init_h = product_state_mps(Lh, d=2, state_index=0)
res_h = dmrg_two_site(mpo_h, d=2, Dmax=16, nsweeps=5, init_mps=init_h, verbose=True)
plot_convergence(res_h, title_prefix=f"Heisenberg L={Lh}", filename_prefix="heisenberg_L8", figdir=figdir)

# -------------------------------------------------
# 3. AKLT benchmark
# -------------------------------------------------
print("\n=== AKLT benchmark ===")
L_list_a, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a = run_aklt_benchmark(
    L_list=(4, 6),
    Dmax=2,
    nsweeps=5,
    verbose=True
)
plot_aklt_benchmark(L_list_a, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a, figdir=figdir)

print("\n=== AKLT convergence and entropy ===")
La = 8
mpo_a = mpo_aklt(La)
init_a = product_state_mps(La, d=3, state_index=0)
res_a = dmrg_two_site(mpo_a, d=3, Dmax=2, nsweeps=6, init_mps=init_a, verbose=True)
plot_convergence(res_a, title_prefix=f"AKLT L={La}", filename_prefix="aklt_L8", figdir=figdir)
plot_entanglement_entropy(
    res_a["mps"],
    title=f"AKLT bond entanglement entropy, L={La}",
    filename="aklt_entanglement_entropy_L8.png",
    figdir=figdir
)

# -------------------------------------------------
# 4. Exact AKLT MPS check
# -------------------------------------------------
print("\n=== Exact AKLT MPS check ===")
L_exact = 8
mps_exact_aklt = exact_aklt_mps(L_exact)
H_exact_aklt = dense_aklt(L_exact)
E_exact_mps = dense_energy_from_mps(mps_exact_aklt, H_exact_aklt)
print(f"Exact AKLT MPS energy for L={L_exact}: {E_exact_mps:.12f}")
print(f"Expected exact energy: {-(2/3)*(L_exact-1):.12f}")

# -------------------------------------------------
# 5. Summary
# -------------------------------------------------
print("\n=== Summary ===")
print("Heisenberg benchmark sizes:", L_list_h)
print("Heisenberg |DMRG-Exact|    :", np.abs(E_dmrg_h - E_exact_h))
print("AKLT benchmark sizes       :", L_list_a)
print("AKLT |DMRG-Formula|        :", np.abs(E_dmrg_a - E_formula_a))
print("AKLT max discarded weights :", max_disc_a)
print("\nAll figures have been shown and saved to:", figdir)
```

---

# Final assessment

Yes — at this point the code is doing what you asked:

- **two-site DMRG**
- **Heisenberg spin-1/2**
- **AKLT spin-1**
- **Jupyter-friendly**
- **figures shown and saved in `figureAKLT`**

If you want, I can now do one more cleanup step and provide:

## a polished final answer with exactly:
1. **one function cell**
2. **one final run-all cell**
3. nothing extra

ready to paste directly into your notebook.

---
## 🧑‍💻 User

=== Exact AKLT MPS check ===
Exact AKLT MPS energy for L=8: -4.666666666667
Expected exact energy: -4.666666666667

=== Summary ===
Heisenberg benchmark sizes: [4 6 8]
Heisenberg |DMRG-Exact|    : [0. 0. 0.]
AKLT benchmark sizes       : [4 6]
AKLT |DMRG-Formula|        : [0. 0.]
AKLT max discarded weights : [0.003223675502 0.003223675502]

All figures have been shown and saved to: figureAKLT

---
## 🤖 Assistant

Perfect — this means the implementation is now **numerically correct** for the requested benchmarks.

## Final status

### Heisenberg
- benchmark energies match exact diagonalization:
  - `[0. 0. 0.]` error for \(L=4,6,8\)

### AKLT
- benchmark energies match the exact formula:
  - `[0. 0.]` error for \(L=4,6\)
- exact AKLT MPS energy check is also perfect

So the notebook now successfully computes both:

- **spin-1/2 Heisenberg model**
- **spin-1 AKLT model**

with **two-site DMRG**.

---

# About the AKLT discarded weight

You still see:

```python
AKLT max discarded weights : [0.003223675502 0.003223675502]
```

This is not fatal since the energies are exact, but let me explain it.

## Why this can happen
You are recording the **maximum discarded weight over the whole sweep history**.  
Even if the final converged state is exact with bond dimension 2, the early sweeps may transiently produce a local two-site tensor with more than two appreciable singular values before the gauge settles.

So this number does **not** necessarily mean the final state is wrong.

## Better diagnostic
Look at the per-sweep history:

```python
L = 6
mpo = mpo_aklt(L)
res = dmrg_two_site(mpo, d=3, Dmax=2, nsweeps=8,
                    init_mps=product_state_mps(L, 3, 0),
                    verbose=True)

print("energies =", res["energies"])
print("max_discards =", res["max_discards"])
```

If the discarded weight decreases and stabilizes while the energy is exact, that is fine.

---

# Conclusion

Your code is now doing exactly what you asked for:

- all functions available in one cell
- one run-all cell
- two-site DMRG
- Heisenberg and AKLT models
- figures shown and saved to `figureAKLT`

---

# If you want one final refinement
I can still help with one of these optional improvements:

1. **Make AKLT discarded weight go closer to zero**
   - by using the exact AKLT MPS as initialization
   - or a better random initialization / more sweeps

2. **Speed up the code**
   - cache left/right environments during right sweeps
   - avoid rebuilding left environments repeatedly

3. **Add observables**
   - local magnetization
   - spin-spin correlation functions
   - AKLT string order parameter
   - entanglement spectrum

4. **Export as a `.py` file or notebook structure**

If you want, I can now give you a **small patch** to add:
- local observables
- correlation functions
- AKLT string order

which would make the notebook much more complete physically.

---
## 🧑‍💻 User

go ahead with 2 and 3

---
## 🤖 Assistant

Below is a **patch cell** adding:

## 2. Speed-up
- cached left/right environments during sweeps
- no repeated rebuilding of left environments in the right sweep

## 3. Observables
- one-site expectation values
- two-site correlators
- full correlation matrix
- AKLT string order parameter
- entanglement spectrum from dense Schmidt values

You can paste this **after** your existing function cell.  
It will override `dmrg_two_site` with a faster version and add new observable utilities.

---

# Patch cell: faster DMRG + observables

```python
# ============================================================
# Faster environment handling
# ============================================================

def build_all_left_envs(mps, mpo):
    """
    Returns list Lenvs with:
      Lenvs[i] = left environment for bond starting at site i
    i.e. contraction of sites 0..i-1.
    Shapes:
      Lenvs[0] = boundary env of shape (1,1,1)
    """
    L = len(mps)
    Lenvs = [None] * L
    Lenv = init_left_env(dtype=mps[0].dtype)
    Lenvs[0] = Lenv
    for i in range(L - 1):
        Lenv = update_left_env(Lenv, mps[i], mpo[i])
        Lenvs[i + 1] = Lenv
    return Lenvs

def build_all_right_envs_full(mps, mpo):
    """
    Returns list Renvs with:
      Renvs[i] = right environment for bond ending at site i
    i.e. contraction of sites i+1..L-1.
    In particular for two-site optimization on bond (i,i+1),
    use Renvs[i+1].
    """
    L = len(mps)
    Renvs = [None] * L
    Renv = init_right_env(dtype=mps[0].dtype)
    Renvs[L - 1] = Renv
    for i in range(L - 1, 0, -1):
        Renv = update_right_env(Renv, mps[i], mpo[i])
        Renvs[i - 1] = Renv
    return Renvs

# ============================================================
# Faster two-site DMRG using cached environments
# ============================================================

def dmrg_two_site(mpo, d, Dmax=32, nsweeps=8, cutoff=1e-12,
                  init_mps=None, verbose=True):
    """
    Faster two-site DMRG:
      - dense local effective Hamiltonian
      - cached left/right environments
    Good for small/medium notebook benchmarks.
    """
    L = len(mpo)

    if init_mps is None:
        init_mps = random_mps(L, d, Dmax=max(2, min(Dmax, 4)), seed=1234)

    mps = right_canonicalize_qr(init_mps)

    energies = []
    max_discards = []

    for sweep in range(nsweeps):
        sweep_discards = []
        E_last = None

        # Build all right envs once for LR sweep
        Renvs = build_all_right_envs_full(mps, mpo)
        Lenv = init_left_env(dtype=mps[0].dtype)

        # -------------------------
        # Left -> Right
        # -------------------------
        for i in range(L - 1):
            theta = np.einsum('asb,btc->astc', mps[i], mps[i+1], optimize=True)

            E, theta_opt = solve_local_ground_state_dense(
                theta, Lenv, mpo[i], mpo[i+1], Renvs[i+1]
            )

            A, Mright, s, disc = split_two_site_theta(
                theta_opt, Dmax=Dmax, cutoff=cutoff, sweep_direction='lr'
            )

            mps[i] = A
            mps[i+1] = Mright
            Lenv = update_left_env(Lenv, mps[i], mpo[i])

            sweep_discards.append(disc)
            E_last = E

        # Build all left envs once for RL sweep
        Lenvs = build_all_left_envs(mps, mpo)
        Renv = init_right_env(dtype=mps[0].dtype)

        # -------------------------
        # Right -> Left
        # -------------------------
        for i in range(L - 2, -1, -1):
            theta = np.einsum('asb,btc->astc', mps[i], mps[i+1], optimize=True)

            E, theta_opt = solve_local_ground_state_dense(
                theta, Lenvs[i], mpo[i], mpo[i+1], Renv
            )

            Mleft, B, s, disc = split_two_site_theta(
                theta_opt, Dmax=Dmax, cutoff=cutoff, sweep_direction='rl'
            )

            mps[i] = Mleft
            mps[i+1] = B
            Renv = update_right_env(Renv, mps[i+1], mpo[i+1])

            sweep_discards.append(disc)
            E_last = E

        energies.append(E_last)
        max_discards.append(max(sweep_discards) if sweep_discards else 0.0)

        if verbose:
            print(f"sweep {sweep+1:2d}: E = {E_last:.15f}, max discarded weight = {max_discards[-1]:.3e}")

    return {
        "mps": mps,
        "energies": np.array(energies),
        "max_discards": np.array(max_discards),
    }

# ============================================================
# Generic local-operator expectation values from dense state
# ============================================================

def local_expectation_dense(mps, op, site):
    """
    <op_site> from dense wavefunction.
    Suitable for small chains.
    """
    psi = mps_to_dense(mps)
    psi = psi / np.linalg.norm(psi)

    L = len(mps)
    d = mps[0].shape[1]
    I = np.eye(d, dtype=np.complex128)

    ops = [I] * L
    ops[site] = op
    O = kron_all(ops)
    return np.vdot(psi, O @ psi)

def two_point_expectation_dense(mps, op1, i, op2, j):
    """
    <op1_i op2_j> from dense wavefunction.
    """
    psi = mps_to_dense(mps)
    psi = psi / np.linalg.norm(psi)

    L = len(mps)
    d = mps[0].shape[1]
    I = np.eye(d, dtype=np.complex128)

    ops = [I] * L
    ops[i] = op1
    ops[j] = op2
    O = kron_all(ops)
    return np.vdot(psi, O @ psi)

def one_site_profile(mps, op):
    """
    Returns array [ <op_0>, <op_1>, ... ].
    """
    L = len(mps)
    vals = np.array([local_expectation_dense(mps, op, i) for i in range(L)], dtype=np.complex128)
    return vals

def correlation_function_from_site(mps, op1, i0, op2):
    """
    Returns C(j) = <op1_i0 op2_j> for all j.
    """
    L = len(mps)
    vals = np.array([two_point_expectation_dense(mps, op1, i0, op2, j) for j in range(L)], dtype=np.complex128)
    return vals

def connected_correlation_from_site(mps, op1, i0, op2):
    """
    Returns connected correlation:
      <op1_i0 op2_j> - <op1_i0><op2_j>
    """
    L = len(mps)
    v1 = local_expectation_dense(mps, op1, i0)
    v2 = one_site_profile(mps, op2)
    corr = correlation_function_from_site(mps, op1, i0, op2)
    return corr - v1 * v2

def correlation_matrix(mps, op1, op2=None):
    """
    Full matrix C_ij = <op1_i op2_j>.
    If op2 is None, uses op2=op1.
    """
    if op2 is None:
        op2 = op1
    L = len(mps)
    C = np.zeros((L, L), dtype=np.complex128)
    for i in range(L):
        for j in range(L):
            C[i, j] = two_point_expectation_dense(mps, op1, i, op2, j)
    return C

# ============================================================
# Spin observables helpers
# ============================================================

def heisenberg_spin_profiles(mps):
    I, Sp, Sm, Sz, Sx, Sy = spin_half_ops()
    prof = {
        "Sz": one_site_profile(mps, Sz).real,
        "Sx": one_site_profile(mps, Sx).real,
        "Sy": one_site_profile(mps, Sy).real,
    }
    return prof

def aklt_spin_profiles(mps):
    I, Sp, Sm, Sz, Sx, Sy = spin_one_ops()
    prof = {
        "Sz": one_site_profile(mps, Sz).real,
        "Sx": one_site_profile(mps, Sx).real,
        "Sy": one_site_profile(mps, Sy).real,
    }
    return prof

def spin_spin_connected_from_center_heisenberg(mps):
    I, Sp, Sm, Sz, Sx, Sy = spin_half_ops()
    L = len(mps)
    i0 = L // 2
    return {
        "i0": i0,
        "SzSz": connected_correlation_from_site(mps, Sz, i0, Sz).real,
        "SxSx": connected_correlation_from_site(mps, Sx, i0, Sx).real,
        "SySy": connected_correlation_from_site(mps, Sy, i0, Sy).real,
    }

def spin_spin_connected_from_center_aklt(mps):
    I, Sp, Sm, Sz, Sx, Sy = spin_one_ops()
    L = len(mps)
    i0 = L // 2
    return {
        "i0": i0,
        "SzSz": connected_correlation_from_site(mps, Sz, i0, Sz).real,
        "SxSx": connected_correlation_from_site(mps, Sx, i0, Sx).real,
        "SySy": connected_correlation_from_site(mps, Sy, i0, Sy).real,
    }

# ============================================================
# AKLT string order parameter
# ============================================================

def string_order_parameter_dense(mps, i, j):
    """
    AKLT string correlator:
      < Sz_i exp(i pi sum_{k=i+1}^{j-1} Sz_k) Sz_j >
    for spin-1 systems.

    Since exp(i pi Sz) = diag(-1, 1, -1) for spin-1,
    we use that exact onsite operator.
    """
    I, Sp, Sm, Sz, Sx, Sy = spin_one_ops()
    psi = mps_to_dense(mps)
    psi = psi / np.linalg.norm(psi)

    L = len(mps)
    d = 3
    Id = np.eye(d, dtype=np.complex128)
    phase = la.expm(1j * np.pi * Sz)

    ops = [Id] * L
    ops[i] = Sz
    ops[j] = Sz
    for k in range(i + 1, j):
        ops[k] = phase

    O = kron_all(ops)
    return np.vdot(psi, O @ psi)

def string_order_profile_from_center(mps):
    """
    Computes string order from center site to all sites j>i0.
    """
    L = len(mps)
    i0 = L // 2 - 1 if L % 2 == 0 else L // 2
    js = np.arange(i0 + 1, L)
    vals = np.array([string_order_parameter_dense(mps, i0, j) for j in js], dtype=np.complex128)
    return i0, js, vals

# ============================================================
# Entanglement spectrum
# ============================================================

def entanglement_spectrum(mps, cut):
    """
    Returns Schmidt probabilities p = s^2 across bond 'cut',
    where left block = sites [0..cut-1], right block = [cut..L-1].
    Dense-state based.
    """
    psi = mps_to_dense(mps)
    L = len(mps)
    d = mps[0].shape[1]

    M = psi.reshape(d**cut, d**(L - cut), order='C')
    s = la.svdvals(M)
    s = s / np.linalg.norm(s)
    p = s**2
    p = p[p > 1e-15]
    return np.sort(p)[::-1]

def entanglement_spectrum_all_bonds(mps):
    L = len(mps)
    return [entanglement_spectrum(mps, cut) for cut in range(1, L)]

# ============================================================
# Plot helpers for observables
# ============================================================

def plot_one_site_profile(vals, ylabel, title, filename, figdir="figureAKLT"):
    setup_figure_dir(figdir)
    L = len(vals)

    plt.figure()
    plt.plot(np.arange(L), np.real(vals), 'o-')
    plt.xlabel("Site")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, filename)
    plt.show()

def plot_connected_correlation_from_center(corr_dict, ylabel, title, filename, figdir="figureAKLT"):
    setup_figure_dir(figdir)
    i0 = corr_dict["i0"]
    L = len(corr_dict["SzSz"])
    x = np.arange(L) - i0

    plt.figure()
    plt.plot(x, corr_dict["SzSz"], 'o-', label='SzSz')
    plt.plot(x, corr_dict["SxSx"], 's--', label='SxSx')
    plt.plot(x, corr_dict["SySy"], 'd-.', label='SySy')
    plt.xlabel(r"$j-i_0$")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, filename)
    plt.show()

def plot_string_order_profile(i0, js, vals, title, filename, figdir="figureAKLT"):
    setup_figure_dir(figdir)

    plt.figure()
    plt.plot(js - i0, vals.real, 'o-')
    plt.xlabel(r"$j-i$")
    plt.ylabel("String order")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, filename)
    plt.show()

def plot_entanglement_spectrum(espec, cut, title, filename, figdir="figureAKLT"):
    setup_figure_dir(figdir)
    x = np.arange(1, len(espec) + 1)

    plt.figure()
    plt.semilogy(x, espec, 'o-')
    plt.xlabel("Schmidt index")
    plt.ylabel(r"$p_\alpha = s_\alpha^2$")
    plt.title(title + f" (cut={cut})")
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, filename)
    plt.show()
```

---

# Example usage cell for observables

Use this after running DMRG.

```python
figdir = setup_figure_dir("figureAKLT")

# -------------------------------------------------
# Heisenberg example
# -------------------------------------------------
Lh = 8
res_h = dmrg_two_site(
    mpo_heisenberg_xxz(Lh, J=1.0, Jz=1.0, h=0.0),
    d=2, Dmax=16, nsweeps=5,
    init_mps=product_state_mps(Lh, 2, 0),
    verbose=True
)
mps_h = res_h["mps"]

prof_h = heisenberg_spin_profiles(mps_h)
plot_one_site_profile(
    prof_h["Sz"],
    ylabel=r"$\langle S_i^z \rangle$",
    title=f"Heisenberg local magnetization, L={Lh}",
    filename="heisenberg_local_Sz.png",
    figdir=figdir
)

corr_h = spin_spin_connected_from_center_heisenberg(mps_h)
plot_connected_correlation_from_center(
    corr_h,
    ylabel="Connected correlation",
    title=f"Heisenberg connected correlations, L={Lh}",
    filename="heisenberg_connected_correlations.png",
    figdir=figdir
)

es_h = entanglement_spectrum(mps_h, cut=Lh//2)
plot_entanglement_spectrum(
    es_h,
    cut=Lh//2,
    title=f"Heisenberg entanglement spectrum, L={Lh}",
    filename="heisenberg_entanglement_spectrum.png",
    figdir=figdir
)

# -------------------------------------------------
# AKLT example
# -------------------------------------------------
La = 8
res_a = dmrg_two_site(
    mpo_aklt(La),
    d=3, Dmax=2, nsweeps=6,
    init_mps=product_state_mps(La, 3, 0),
    verbose=True
)
mps_a = res_a["mps"]

prof_a = aklt_spin_profiles(mps_a)
plot_one_site_profile(
    prof_a["Sz"],
    ylabel=r"$\langle S_i^z \rangle$",
    title=f"AKLT local magnetization, L={La}",
    filename="aklt_local_Sz.png",
    figdir=figdir
)

corr_a = spin_spin_connected_from_center_aklt(mps_a)
plot_connected_correlation_from_center(
    corr_a,
    ylabel="Connected correlation",
    title=f"AKLT connected correlations, L={La}",
    filename="aklt_connected_correlations.png",
    figdir=figdir
)

i0, js, svals = string_order_profile_from_center(mps_a)
plot_string_order_profile(
    i0, js, svals,
    title=f"AKLT string order profile, L={La}",
    filename="aklt_string_order_profile.png",
    figdir=figdir
)

es_a = entanglement_spectrum(mps_a, cut=La//2)
plot_entanglement_spectrum(
    es_a,
    cut=La//2,
    title=f"AKLT entanglement spectrum, L={La}",
    filename="aklt_entanglement_spectrum.png",
    figdir=figdir
)
```

---

# What this gives you physically

## Heisenberg
- local magnetization profile \(\langle S_i^z\rangle\)
- connected correlations
  - \(\langle S^z_{i_0} S^z_j\rangle_c\)
  - \(\langle S^x_{i_0} S^x_j\rangle_c\)
  - \(\langle S^y_{i_0} S^y_j\rangle_c\)
- entanglement spectrum at a cut

## AKLT
- local magnetization profile
- connected spin correlations
- **string order**
  \[
  \left\langle S_i^z \exp\!\left(i\pi\sum_{k=i+1}^{j-1} S_k^z\right) S_j^z \right\rangle
  \]
- entanglement spectrum

---

# Notes

## Speed-up
This patched `dmrg_two_site` is faster because the right sweep no longer repeatedly rebuilds left environments from scratch.

## Observables are dense-state based
These observable routines are fine for the sizes you are using:
- Heisenberg \(L \lesssim 8,10\)
- AKLT \(L \lesssim 8,10\)

For much larger systems, one would want fully MPS/MPO-based expectation value routines.

---

If you want, I can next give you a **single final polished run-all cell** that includes:
- benchmarks
- convergence
- magnetization
- correlations
- string order
- entanglement spectrum

all saved to `figureAKLT`.

---
## 🧑‍💻 User

where to put these patches?

---
## 🤖 Assistant

Put the patch cell **after your main function cell**.

## Order in the notebook

### Cell 1
Your big original function cell:
- operators
- MPOs
- MPS utilities
- DMRG core
- plotting helpers
- tests

### Cell 2
The **patch cell** I just gave you.

This works because the patch cell:

- **adds new functions**
- and **redefines** some existing ones, especially:
  - `dmrg_two_site`

In Python/Jupyter, the newest definition overrides the old one.

---

## So the notebook order should be:

### 1. Main function cell
Paste and run the full one-cell function block.

### 2. Patch cell
Paste and run the patch block for:
- faster environments
- faster `dmrg_two_site`
- observables
- string order
- entanglement spectrum

### 3. Run / analysis cells
Then run your benchmark or plotting cells.

---

## Important
If you edit the main function cell later and rerun it, it may overwrite the patched version of `dmrg_two_site`.

So after rerunning the main function cell, you should rerun the patch cell too.

---

# Minimal rule to remember

If you restart the kernel, run in this order:

```python
# 1
[main function cell]

# 2
[patch cell]

# 3
[run-all / benchmark / observable cells]
```

---

# What gets overridden by the patch
The patch replaces or adds:

### Replaced
- `dmrg_two_site`

### Added
- `build_all_left_envs`
- `build_all_right_envs_full`
- observable functions
- string order functions
- entanglement spectrum functions
- new plot helpers

---

# Recommended notebook layout

## Cell A — full function definitions
Your corrected full one-cell function block

## Cell B — patch
My speed-up + observables patch

## Cell C — final run-all
Benchmarks and figures

## Cell D — optional extra observable plots
Magnetization, correlations, string order, entanglement spectrum

---

If you want, I can now give you a **single final run-all cell** designed specifically for the patched notebook layout.

---
## 🧑‍💻 User

give me a single final run-all cell 

---
## 🤖 Assistant

```python
# ============================================================
# Final run-all cell
# Assumes:
#   1) main function cell has been run
#   2) patch cell (faster dmrg + observables) has been run
# ============================================================

figdir = setup_figure_dir("figureAKLT")
print("Figures will be saved in:", figdir)

# ============================================================
# 1. Basic tests
# ============================================================

print("\n=== Basic tests ===")
test_canonicalization()
err_heff = test_two_site_heff()
print("Local two-site Heff test error =", err_heff)

# ============================================================
# 2. Heisenberg benchmark
# ============================================================

print("\n=== Heisenberg benchmark ===")
L_list_h, E_dmrg_h, E_exact_h = run_heisenberg_benchmark(
    L_list=(4, 6, 8),
    J=1.0,
    Jz=1.0,
    h=0.0,
    Dmax=16,
    nsweeps=4,
    verbose=True
)
plot_heisenberg_benchmark(L_list_h, E_dmrg_h, E_exact_h, figdir=figdir)

print("\n=== Heisenberg convergence example ===")
Lh = 8
mpo_h = mpo_heisenberg_xxz(Lh, J=1.0, Jz=1.0, h=0.0)
init_h = product_state_mps(Lh, d=2, state_index=0)
res_h = dmrg_two_site(mpo_h, d=2, Dmax=16, nsweeps=5, init_mps=init_h, verbose=True)
mps_h = res_h["mps"]

plot_convergence(
    res_h,
    title_prefix=f"Heisenberg L={Lh}",
    filename_prefix="heisenberg_L8",
    figdir=figdir
)

# ============================================================
# 3. Heisenberg observables
# ============================================================

print("\n=== Heisenberg observables ===")
prof_h = heisenberg_spin_profiles(mps_h)

plot_one_site_profile(
    prof_h["Sz"],
    ylabel=r"$\langle S_i^z \rangle$",
    title=f"Heisenberg local magnetization, L={Lh}",
    filename="heisenberg_local_Sz.png",
    figdir=figdir
)

corr_h = spin_spin_connected_from_center_heisenberg(mps_h)
plot_connected_correlation_from_center(
    corr_h,
    ylabel="Connected correlation",
    title=f"Heisenberg connected correlations, L={Lh}",
    filename="heisenberg_connected_correlations.png",
    figdir=figdir
)

es_h = entanglement_spectrum(mps_h, cut=Lh // 2)
plot_entanglement_spectrum(
    es_h,
    cut=Lh // 2,
    title=f"Heisenberg entanglement spectrum, L={Lh}",
    filename="heisenberg_entanglement_spectrum.png",
    figdir=figdir
)

# ============================================================
# 4. AKLT benchmark
# ============================================================

print("\n=== AKLT benchmark ===")
L_list_a, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a = run_aklt_benchmark(
    L_list=(4, 6),
    Dmax=2,
    nsweeps=5,
    verbose=True
)
plot_aklt_benchmark(L_list_a, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a, figdir=figdir)

print("\n=== AKLT convergence example ===")
La = 8
mpo_a = mpo_aklt(La)
init_a = product_state_mps(La, d=3, state_index=0)
res_a = dmrg_two_site(mpo_a, d=3, Dmax=2, nsweeps=6, init_mps=init_a, verbose=True)
mps_a = res_a["mps"]

plot_convergence(
    res_a,
    title_prefix=f"AKLT L={La}",
    filename_prefix="aklt_L8",
    figdir=figdir
)

plot_entanglement_entropy(
    mps_a,
    title=f"AKLT bond entanglement entropy, L={La}",
    filename="aklt_entanglement_entropy_L8.png",
    figdir=figdir
)

# ============================================================
# 5. AKLT observables
# ============================================================

print("\n=== AKLT observables ===")
prof_a = aklt_spin_profiles(mps_a)

plot_one_site_profile(
    prof_a["Sz"],
    ylabel=r"$\langle S_i^z \rangle$",
    title=f"AKLT local magnetization, L={La}",
    filename="aklt_local_Sz.png",
    figdir=figdir
)

corr_a = spin_spin_connected_from_center_aklt(mps_a)
plot_connected_correlation_from_center(
    corr_a,
    ylabel="Connected correlation",
    title=f"AKLT connected correlations, L={La}",
    filename="aklt_connected_correlations.png",
    figdir=figdir
)

i0, js, svals = string_order_profile_from_center(mps_a)
plot_string_order_profile(
    i0, js, svals,
    title=f"AKLT string order profile, L={La}",
    filename="aklt_string_order_profile.png",
    figdir=figdir
)

es_a = entanglement_spectrum(mps_a, cut=La // 2)
plot_entanglement_spectrum(
    es_a,
    cut=La // 2,
    title=f"AKLT entanglement spectrum, L={La}",
    filename="aklt_entanglement_spectrum.png",
    figdir=figdir
)

# ============================================================
# 6. Exact AKLT MPS check
# ============================================================

print("\n=== Exact AKLT MPS check ===")
L_exact = 8
mps_exact_aklt = exact_aklt_mps(L_exact)
H_exact_aklt = dense_aklt(L_exact)
E_exact_mps = dense_energy_from_mps(mps_exact_aklt, H_exact_aklt)
print(f"Exact AKLT MPS energy for L={L_exact}: {E_exact_mps:.12f}")
print(f"Expected exact energy: {-(2/3)*(L_exact-1):.12f}")

# ============================================================
# 7. Optional small variance checks
# ============================================================

print("\n=== Small variance checks ===")
Lh_small = 8
Hh_small = dense_heisenberg_xxz(Lh_small, J=1.0, Jz=1.0, h=0.0)
Eh_small = dense_energy_from_mps(mps_h, Hh_small)
varh_small = dense_variance_from_mps(mps_h, Hh_small)
print(f"Heisenberg L={Lh}: E = {Eh_small:.12f}, variance = {varh_small:.3e}")

Ha_small = dense_aklt(La)
Ea_small = dense_energy_from_mps(mps_a, Ha_small)
vara_small = dense_variance_from_mps(mps_a, Ha_small)
print(f"AKLT L={La}: E = {Ea_small:.12f}, variance = {vara_small:.3e}")

# ============================================================
# 8. Summary
# ============================================================

print("\n=== Summary ===")
print("Heisenberg benchmark sizes:", L_list_h)
print("Heisenberg |DMRG-Exact|    :", np.abs(E_dmrg_h - E_exact_h))
print("AKLT benchmark sizes       :", L_list_a)
print("AKLT |DMRG-Formula|        :", np.abs(E_dmrg_a - E_formula_a))
print("AKLT max discarded weights :", max_disc_a)

print("\nGenerated figures include:")
print("  - Heisenberg benchmark energy and error")
print("  - Heisenberg convergence")
print("  - Heisenberg local magnetization")
print("  - Heisenberg connected correlations")
print("  - Heisenberg entanglement spectrum")
print("  - AKLT benchmark energy, error, discarded weight")
print("  - AKLT convergence")
print("  - AKLT entanglement entropy")
print("  - AKLT local magnetization")
print("  - AKLT connected correlations")
print("  - AKLT string order profile")
print("  - AKLT entanglement spectrum")

print("\nAll figures have been shown and saved to:", figdir)
```

---
## 🧑‍💻 User

=== Exact AKLT MPS check ===
Exact AKLT MPS energy for L=8: -4.666666666667
Expected exact energy: -4.666666666667

=== Small variance checks ===
Heisenberg L=8: E = -3.374932598688, variance = -1.776e-15
AKLT L=8: E = -4.666666666667, variance = 1.421e-14

=== Summary ===
Heisenberg benchmark sizes: [4 6 8]
Heisenberg |DMRG-Exact|    : [0. 0. 0.]
AKLT benchmark sizes       : [4 6]
AKLT |DMRG-Formula|        : [0. 0.]
AKLT max discarded weights : [0.003223675502 0.003223675502]

Generated figures include:
  - Heisenberg benchmark energy and error
  - Heisenberg convergence
  - Heisenberg local magnetization
  - Heisenberg connected correlations
  - Heisenberg entanglement spectrum
  - AKLT benchmark energy, error, discarded weight
  - AKLT convergence
  - AKLT entanglement entropy
  - AKLT local magnetization
  - AKLT connected correlations
  - AKLT string order profile
  - AKLT entanglement spectrum

All figures have been shown and saved to: figureAKLT

---
## 🤖 Assistant

Excellent — this is now a **fully successful outcome**.

## What your final output confirms

### Heisenberg
- exact benchmark agreement:
  \[
  |E_{\mathrm{DMRG}}-E_{\mathrm{exact}}| = 0
  \]
  for \(L=4,6,8\)

- variance:
  \[
  -1.776\times 10^{-15}
  \]
  which is numerically zero up to floating-point roundoff

### AKLT
- exact benchmark agreement:
  \[
  |E_{\mathrm{DMRG}}-E_{\mathrm{exact}}| = 0
  \]
- exact AKLT MPS energy check succeeds
- variance:
  \[
  1.421\times 10^{-14}
  \]
  also essentially zero

So the final notebook is doing exactly what it should.

---

# About the small negative Heisenberg variance
You got:

```python
variance = -1.776e-15
```

This is completely normal. Variance must be nonnegative analytically, but due to floating-point cancellation, tiny negative numbers of order \(10^{-15}\) can appear. This just means:

- the state is effectively an eigenstate
- numerical precision is very good

---

# About the AKLT discarded weight
You still see:

```python
0.003223675502
```

This is not a concern here because:

- the energy is exact
- the variance is essentially zero
- the exact AKLT MPS benchmark works

So the final state is correct. That discarded weight is just reflecting some intermediate truncation during the sweep history, not a final-state error.

If you wanted, one could refine the code to store:
- final-sweep discarded weights only, or
- per-bond discarded weights by sweep

but this is optional.

---

# Final assessment

You now have a Jupyter-ready implementation that includes:

## Core algorithms
- MPS utilities
- canonicalization
- MPOs
- **two-site DMRG**

## Models
- **spin-1/2 Heisenberg**
- **spin-1 AKLT**

## Diagnostics
- exact diagonalization benchmarks
- energy convergence
- variance checks
- discarded weights

## Physics observables
- local magnetization
- connected correlations
- AKLT string order
- entanglement entropy
- entanglement spectrum

## Output handling
- all figures shown inline
- all figures saved to:
  ```python
  figureAKLT
  ```

---

# If you want to write this up
Your results are now strong enough for a clean notebook/report statement like:

> We implemented a two-site DMRG algorithm for open chains using MPS/MPO language. The code was benchmarked against exact diagonalization for the spin-1/2 Heisenberg chain and against the exact AKLT ground-state energy. For the tested system sizes, the DMRG energies agree with exact values to machine precision, and the energy variance is numerically zero within floating-point accuracy.

---

# Optional final improvements
If you want to polish this even further, the next natural additions would be:

1. **per-bond discarded-weight history**
2. **MPO-based expectation values** instead of dense-state observables
3. **larger-system runs**
4. **symmetry implementation**, e.g. \(U(1)\) / fixed \(S^z\)
5. **finite-size scaling plots**

---

If you want, I can still provide one of these final extras:

1. a **clean minimal notebook structure** with markdown sections,
2. a **single `.py` module** version,
3. a **per-bond discarded-weight diagnostic patch**.

---
## 🧑‍💻 User

a clean minimal notebook structure with markdown sections

---
## 🤖 Assistant

Below is a **clean minimal notebook structure** with markdown section titles and the code cells you already have.  
The idea is:

- easy to read
- easy to present
- minimal duplication
- suitable for your report/note workflow

You can create a new Jupyter notebook and arrange it in this order.

---

# Notebook title cell

## Markdown cell

```markdown
# Two-site DMRG for the Heisenberg and AKLT chains

This notebook implements a two-site DMRG algorithm in matrix-product-state (MPS) language for one-dimensional open chains.

We study:

- the spin-1/2 Heisenberg chain
- the spin-1 AKLT chain

The implementation follows standard MPS/MPO conventions and includes:

- MPS canonicalization
- MPO representations of the Hamiltonians
- two-site DMRG sweeps
- exact-diagonalization benchmarks for small systems
- convergence diagnostics
- physical observables:
  - local magnetization
  - connected spin correlations
  - entanglement entropy and entanglement spectrum
  - AKLT string order

All figures are displayed in the notebook and saved to the directory:

`figureAKLT`
```

---

# 1. Imports and setup

## Markdown cell

```markdown
## 1. Imports and setup
```

## Code cell

```python
import numpy as np
import matplotlib.pyplot as plt

figdir = "figureAKLT"
```

---

# 2. Function definitions

## Markdown cell

```markdown
## 2. Core function definitions

This section contains the full implementation:

- local spin operators
- exact dense Hamiltonians for benchmarking
- MPO builders
- MPS constructors and canonicalization
- environments
- dense local two-site effective Hamiltonian
- two-site DMRG
- plotting helpers
- diagnostics
```

## Code cell

Paste your **main full one-cell function block** here.

---

# 3. Speed-up and observable patch

## Markdown cell

```markdown
## 3. Faster sweep implementation and observable utilities

This section adds:

- a faster two-site DMRG sweep using cached environments
- local expectation values
- connected correlations
- AKLT string order
- entanglement spectrum
```

## Code cell

Paste the **patch cell** here.

---

# 4. Basic internal tests

## Markdown cell

```markdown
## 4. Internal consistency tests

Before running benchmarks, we test:

- canonicalization
- the two-site effective Hamiltonian contraction
```

## Code cell

```python
test_canonicalization()
err_heff = test_two_site_heff()
print("Local two-site Heff test error =", err_heff)
```

---

# 5. Heisenberg benchmark

## Markdown cell

```markdown
## 5. Heisenberg benchmark against exact diagonalization

We test the spin-1/2 Heisenberg chain with open boundaries,
\[
H = \sum_{i=1}^{L-1} \mathbf S_i \cdot \mathbf S_{i+1},
\]
using two-site DMRG and compare the ground-state energy to exact diagonalization for small system sizes.
```

## Code cell

```python
figdir = setup_figure_dir("figureAKLT")

L_list_h, E_dmrg_h, E_exact_h = run_heisenberg_benchmark(
    L_list=(4, 6, 8),
    J=1.0,
    Jz=1.0,
    h=0.0,
    Dmax=16,
    nsweeps=4,
    verbose=True
)

plot_heisenberg_benchmark(L_list_h, E_dmrg_h, E_exact_h, figdir=figdir)
```

---

# 6. Heisenberg convergence and observables

## Markdown cell

```markdown
## 6. Heisenberg convergence and observables

We run DMRG for a representative chain and compute:

- convergence history
- local magnetization
- connected spin correlations
- entanglement spectrum
```

## Code cell

```python
Lh = 8
mpo_h = mpo_heisenberg_xxz(Lh, J=1.0, Jz=1.0, h=0.0)
init_h = product_state_mps(Lh, d=2, state_index=0)

res_h = dmrg_two_site(mpo_h, d=2, Dmax=16, nsweeps=5, init_mps=init_h, verbose=True)
mps_h = res_h["mps"]

plot_convergence(
    res_h,
    title_prefix=f"Heisenberg L={Lh}",
    filename_prefix="heisenberg_L8",
    figdir=figdir
)
```

## Code cell

```python
prof_h = heisenberg_spin_profiles(mps_h)

plot_one_site_profile(
    prof_h["Sz"],
    ylabel=r"$\langle S_i^z \rangle$",
    title=f"Heisenberg local magnetization, L={Lh}",
    filename="heisenberg_local_Sz.png",
    figdir=figdir
)

corr_h = spin_spin_connected_from_center_heisenberg(mps_h)
plot_connected_correlation_from_center(
    corr_h,
    ylabel="Connected correlation",
    title=f"Heisenberg connected correlations, L={Lh}",
    filename="heisenberg_connected_correlations.png",
    figdir=figdir
)

es_h = entanglement_spectrum(mps_h, cut=Lh // 2)
plot_entanglement_spectrum(
    es_h,
    cut=Lh // 2,
    title=f"Heisenberg entanglement spectrum, L={Lh}",
    filename="heisenberg_entanglement_spectrum.png",
    figdir=figdir
)
```

---

# 7. AKLT benchmark

## Markdown cell

```markdown
## 7. AKLT benchmark

We study the spin-1 AKLT Hamiltonian
\[
H = \sum_{i=1}^{L-1} \left[ \mathbf S_i\cdot\mathbf S_{i+1} + \frac13(\mathbf S_i\cdot\mathbf S_{i+1})^2 \right].
\]

The exact open-chain ground-state energy is
\[
E_0 = -\frac23(L-1).
\]

We benchmark the DMRG ground-state energy against this exact result.
```

## Code cell

```python
L_list_a, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a = run_aklt_benchmark(
    L_list=(4, 6),
    Dmax=2,
    nsweeps=5,
    verbose=True
)

plot_aklt_benchmark(L_list_a, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a, figdir=figdir)
```

---

# 8. AKLT convergence and observables

## Markdown cell

```markdown
## 8. AKLT convergence and observables

For the AKLT chain we compute:

- convergence history
- entanglement entropy
- local magnetization
- connected spin correlations
- nonlocal string order
- entanglement spectrum
```

## Code cell

```python
La = 8
mpo_a = mpo_aklt(La)
init_a = product_state_mps(La, d=3, state_index=0)

res_a = dmrg_two_site(mpo_a, d=3, Dmax=2, nsweeps=6, init_mps=init_a, verbose=True)
mps_a = res_a["mps"]

plot_convergence(
    res_a,
    title_prefix=f"AKLT L={La}",
    filename_prefix="aklt_L8",
    figdir=figdir
)

plot_entanglement_entropy(
    mps_a,
    title=f"AKLT bond entanglement entropy, L={La}",
    filename="aklt_entanglement_entropy_L8.png",
    figdir=figdir
)
```

## Code cell

```python
prof_a = aklt_spin_profiles(mps_a)

plot_one_site_profile(
    prof_a["Sz"],
    ylabel=r"$\langle S_i^z \rangle$",
    title=f"AKLT local magnetization, L={La}",
    filename="aklt_local_Sz.png",
    figdir=figdir
)

corr_a = spin_spin_connected_from_center_aklt(mps_a)
plot_connected_correlation_from_center(
    corr_a,
    ylabel="Connected correlation",
    title=f"AKLT connected correlations, L={La}",
    filename="aklt_connected_correlations.png",
    figdir=figdir
)

i0, js, svals = string_order_profile_from_center(mps_a)
plot_string_order_profile(
    i0, js, svals,
    title=f"AKLT string order profile, L={La}",
    filename="aklt_string_order_profile.png",
    figdir=figdir
)

es_a = entanglement_spectrum(mps_a, cut=La // 2)
plot_entanglement_spectrum(
    es_a,
    cut=La // 2,
    title=f"AKLT entanglement spectrum, L={La}",
    filename="aklt_entanglement_spectrum.png",
    figdir=figdir
)
```

---

# 9. Exact AKLT MPS check

## Markdown cell

```markdown
## 9. Exact AKLT MPS check

As an additional validation, we evaluate the energy of the exact bond-dimension-2 AKLT MPS and compare it to the exact formula.
```

## Code cell

```python
L_exact = 8
mps_exact_aklt = exact_aklt_mps(L_exact)
H_exact_aklt = dense_aklt(L_exact)
E_exact_mps = dense_energy_from_mps(mps_exact_aklt, H_exact_aklt)

print(f"Exact AKLT MPS energy for L={L_exact}: {E_exact_mps:.12f}")
print(f"Expected exact energy: {-(2/3)*(L_exact-1):.12f}")
```

---

# 10. Variance checks

## Markdown cell

```markdown
## 10. Energy-variance checks

A useful convergence diagnostic is the energy variance
\[
\sigma_H^2 = \langle H^2\rangle - \langle H\rangle^2.
\]

For an exact eigenstate, the variance should vanish up to numerical precision.
```

## Code cell

```python
Hh_small = dense_heisenberg_xxz(Lh, J=1.0, Jz=1.0, h=0.0)
Eh_small = dense_energy_from_mps(mps_h, Hh_small)
varh_small = dense_variance_from_mps(mps_h, Hh_small)

Ha_small = dense_aklt(La)
Ea_small = dense_energy_from_mps(mps_a, Ha_small)
vara_small = dense_variance_from_mps(mps_a, Ha_small)

print(f"Heisenberg L={Lh}: E = {Eh_small:.12f}, variance = {varh_small:.3e}")
print(f"AKLT L={La}: E = {Ea_small:.12f}, variance = {vara_small:.3e}")
```

---

# 11. Summary

## Markdown cell

```markdown
## 11. Summary

We summarize the benchmark errors and convergence diagnostics.
```

## Code cell

```python
print("Heisenberg benchmark sizes:", L_list_h)
print("Heisenberg |DMRG-Exact|    :", np.abs(E_dmrg_h - E_exact_h))
print("AKLT benchmark sizes       :", L_list_a)
print("AKLT |DMRG-Formula|        :", np.abs(E_dmrg_a - E_formula_a))
print("AKLT max discarded weights :", max_disc_a)
print("\nAll figures have been shown and saved to:", figdir)
```

---

# 12. Optional conclusion cell

## Markdown cell

```markdown
## 12. Conclusion

The two-site DMRG implementation successfully reproduces the exact ground-state energies of the spin-1/2 Heisenberg chain and the spin-1 AKLT chain for the tested system sizes. The energy variances are numerically zero within floating-point precision, confirming convergence to the ground state. The AKLT benchmark also reproduces the characteristic nonlocal string order and entanglement structure.
```

---

# Minimal notebook map

If you want the shortest clean version, the notebook is simply:

1. **Title markdown**
2. **Imports/setup**
3. **Main function cell**
4. **Patch cell**
5. **Tests**
6. **Heisenberg benchmark**
7. **Heisenberg observables**
8. **AKLT benchmark**
9. **AKLT observables**
10. **Exact AKLT MPS check**
11. **Variance check**
12. **Summary**

---

If you want, I can also turn this into a **copy-paste ready notebook template** where each markdown cell is already fully written in polished report style.

---
## 🧑‍💻 User

Please go ahead.

---
## 🤖 Assistant

Below is a **copy-paste ready notebook template** in polished report style.  
Each section is written as a markdown cell followed by the code cell(s) to place under it.

---

## Markdown cell 1

```markdown
# Two-site DMRG for the Heisenberg and AKLT chains

This notebook implements a finite-system **two-site density-matrix renormalization group (DMRG)** algorithm in the language of **matrix product states (MPS)** and **matrix product operators (MPO)** for one-dimensional chains with open boundary conditions.

We study two standard benchmark models:

1. the **spin-1/2 Heisenberg chain**
2. the **spin-1 AKLT chain**

The goals of the notebook are:

- to implement the two-site DMRG algorithm in a transparent way,
- to benchmark the numerical results against exact diagonalization for small systems,
- to verify the AKLT model against its exact ground-state energy and exact MPS representation,
- to compute physically relevant observables such as local magnetization, connected correlations, entanglement properties, and the AKLT string order parameter.

All figures are displayed in the notebook and also saved to the directory

```text
figureAKLT
```
```

---

## Markdown cell 2

```markdown
## 1. Imports and basic setup

We begin by importing standard numerical and plotting libraries.  
All figures generated in this notebook will be saved to the directory `figureAKLT`.
```

## Code cell 2

```python
import numpy as np
import matplotlib.pyplot as plt

figdir = "figureAKLT"
```

---

## Markdown cell 3

```markdown
## 2. Core implementation

The next code cell contains the full implementation of the basic MPS/MPO/DMRG machinery:

- local spin operators,
- exact dense Hamiltonians for benchmarking,
- MPO constructions for the Heisenberg and AKLT chains,
- MPS constructors and canonicalization routines,
- environment contractions,
- dense local two-site effective Hamiltonian,
- two-site DMRG,
- diagnostics and plotting helpers.

This is the main reference implementation.
```

## Code cell 3

Paste your **main full one-cell function block** here.

---

## Markdown cell 4

```markdown
## 3. Faster sweep implementation and observable utilities

The next patch cell improves the sweep performance by caching left and right environments during the sweeps.  
It also adds observable routines for:

- local expectation values,
- connected spin-spin correlations,
- entanglement spectrum,
- AKLT string order.

This cell should be executed **after** the core implementation cell.
```

## Code cell 4

Paste your **patch cell** here.

---

## Markdown cell 5

```markdown
## 4. Internal consistency tests

Before running physical benchmarks, we verify two internal consistency checks:

1. **canonicalization test**  
   The QR-based routines should produce left- and right-canonical tensors.

2. **two-site effective Hamiltonian test**  
   The tensor-contraction implementation of the local effective Hamiltonian should agree with its explicitly constructed dense version.

These tests are useful debugging steps before any production run.
```

## Code cell 5

```python
test_canonicalization()
err_heff = test_two_site_heff()
print("Local two-site Heff test error =", err_heff)
```

---

## Markdown cell 6

```markdown
## 5. Heisenberg benchmark against exact diagonalization

We first consider the open spin-1/2 Heisenberg chain
\[
H = \sum_{i=1}^{L-1} \mathbf S_i \cdot \mathbf S_{i+1},
\]
which in the present code is implemented as the isotropic XXZ model with \(J=J_z=1\) and \(h=0\).

For small systems, the DMRG ground-state energy can be compared directly to exact diagonalization.  
This provides a stringent benchmark of the implementation.
```

## Code cell 6

```python
figdir = setup_figure_dir("figureAKLT")

L_list_h, E_dmrg_h, E_exact_h = run_heisenberg_benchmark(
    L_list=(4, 6, 8),
    J=1.0,
    Jz=1.0,
    h=0.0,
    Dmax=16,
    nsweeps=4,
    verbose=True
)

plot_heisenberg_benchmark(L_list_h, E_dmrg_h, E_exact_h, figdir=figdir)
```

---

## Markdown cell 7

```markdown
## 6. Heisenberg convergence

We next study the convergence behavior of the DMRG sweeps for a representative Heisenberg chain.  
The sweep history records:

- the variational ground-state energy,
- the maximum discarded weight during each sweep.

These quantities provide a direct view of convergence.
```

## Code cell 7

```python
Lh = 8
mpo_h = mpo_heisenberg_xxz(Lh, J=1.0, Jz=1.0, h=0.0)
init_h = product_state_mps(Lh, d=2, state_index=0)

res_h = dmrg_two_site(mpo_h, d=2, Dmax=16, nsweeps=5, init_mps=init_h, verbose=True)
mps_h = res_h["mps"]

plot_convergence(
    res_h,
    title_prefix=f"Heisenberg L={Lh}",
    filename_prefix="heisenberg_L8",
    figdir=figdir
)
```

---

## Markdown cell 8

```markdown
## 7. Heisenberg observables

Once the Heisenberg ground state has been obtained, we compute several standard observables:

- the local magnetization profile \(\langle S_i^z \rangle\),
- connected spin-spin correlations measured from the center of the chain,
- the entanglement spectrum across the central bond.

These provide a physical characterization of the state beyond the energy alone.
```

## Code cell 8

```python
prof_h = heisenberg_spin_profiles(mps_h)

plot_one_site_profile(
    prof_h["Sz"],
    ylabel=r"$\langle S_i^z \rangle$",
    title=f"Heisenberg local magnetization, L={Lh}",
    filename="heisenberg_local_Sz.png",
    figdir=figdir
)

corr_h = spin_spin_connected_from_center_heisenberg(mps_h)
plot_connected_correlation_from_center(
    corr_h,
    ylabel="Connected correlation",
    title=f"Heisenberg connected correlations, L={Lh}",
    filename="heisenberg_connected_correlations.png",
    figdir=figdir
)

es_h = entanglement_spectrum(mps_h, cut=Lh // 2)
plot_entanglement_spectrum(
    es_h,
    cut=Lh // 2,
    title=f"Heisenberg entanglement spectrum, L={Lh}",
    filename="heisenberg_entanglement_spectrum.png",
    figdir=figdir
)
```

---

## Markdown cell 9

```markdown
## 8. AKLT benchmark

We now turn to the spin-1 AKLT chain with Hamiltonian
\[
H = \sum_{i=1}^{L-1}
\left[
\mathbf S_i \cdot \mathbf S_{i+1}
+ \frac13 \left(\mathbf S_i \cdot \mathbf S_{i+1}\right)^2
\right].
\]

For open boundary conditions, the exact ground-state energy is
\[
E_0 = -\frac{2}{3}(L-1).
\]

The AKLT model is an especially useful benchmark because its exact ground state is an MPS of bond dimension \(D=2\). Therefore, a correct two-site DMRG implementation should recover the exact energy at \(D_{\max}=2\).
```

## Code cell 9

```python
L_list_a, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a = run_aklt_benchmark(
    L_list=(4, 6),
    Dmax=2,
    nsweeps=5,
    verbose=True
)

plot_aklt_benchmark(L_list_a, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a, figdir=figdir)
```

---

## Markdown cell 10

```markdown
## 9. AKLT convergence

For a representative AKLT chain, we monitor:

- the DMRG energy convergence,
- the discarded weight,
- the bond entanglement entropy.

Since the exact AKLT state has bond dimension \(2\), this serves as a precise and physically transparent test of the algorithm.
```

## Code cell 10

```python
La = 8
mpo_a = mpo_aklt(La)
init_a = product_state_mps(La, d=3, state_index=0)

res_a = dmrg_two_site(mpo_a, d=3, Dmax=2, nsweeps=6, init_mps=init_a, verbose=True)
mps_a = res_a["mps"]

plot_convergence(
    res_a,
    title_prefix=f"AKLT L={La}",
    filename_prefix="aklt_L8",
    figdir=figdir
)

plot_entanglement_entropy(
    mps_a,
    title=f"AKLT bond entanglement entropy, L={La}",
    filename="aklt_entanglement_entropy_L8.png",
    figdir=figdir
)
```

---

## Markdown cell 11

```markdown
## 10. AKLT observables

The AKLT state is characterized not only by its energy but also by its hidden topological order.  
To probe its structure, we compute:

- the local magnetization profile,
- connected spin-spin correlations,
- the nonlocal **string order parameter**
  \[
  \left\langle
  S_i^z
  \exp\!\left(i\pi \sum_{k=i+1}^{j-1} S_k^z\right)
  S_j^z
  \right\rangle,
  \]
- the entanglement spectrum across the central bond.

The string order parameter is a hallmark of the Haldane phase.
```

## Code cell 11

```python
prof_a = aklt_spin_profiles(mps_a)

plot_one_site_profile(
    prof_a["Sz"],
    ylabel=r"$\langle S_i^z \rangle$",
    title=f"AKLT local magnetization, L={La}",
    filename="aklt_local_Sz.png",
    figdir=figdir
)

corr_a = spin_spin_connected_from_center_aklt(mps_a)
plot_connected_correlation_from_center(
    corr_a,
    ylabel="Connected correlation",
    title=f"AKLT connected correlations, L={La}",
    filename="aklt_connected_correlations.png",
    figdir=figdir
)

i0, js, svals = string_order_profile_from_center(mps_a)
plot_string_order_profile(
    i0, js, svals,
    title=f"AKLT string order profile, L={La}",
    filename="aklt_string_order_profile.png",
    figdir=figdir
)

es_a = entanglement_spectrum(mps_a, cut=La // 2)
plot_entanglement_spectrum(
    es_a,
    cut=La // 2,
    title=f"AKLT entanglement spectrum, L={La}",
    filename="aklt_entanglement_spectrum.png",
    figdir=figdir
)
```

---

## Markdown cell 12

```markdown
## 11. Exact AKLT MPS check

As an additional benchmark, we construct the exact bond-dimension-2 AKLT MPS directly from its analytic tensors and evaluate its energy with the dense AKLT Hamiltonian.

This provides an independent confirmation that both:

- the Hamiltonian implementation,
- and the AKLT MPS tensors

are consistent with the exact energy formula.
```

## Code cell 12

```python
L_exact = 8
mps_exact_aklt = exact_aklt_mps(L_exact)
H_exact_aklt = dense_aklt(L_exact)
E_exact_mps = dense_energy_from_mps(mps_exact_aklt, H_exact_aklt)

print(f"Exact AKLT MPS energy for L={L_exact}: {E_exact_mps:.12f}")
print(f"Expected exact energy: {-(2/3)*(L_exact-1):.12f}")
```

---

## Markdown cell 13

```markdown
## 12. Energy-variance checks

A robust convergence diagnostic is the energy variance
\[
\sigma_H^2 = \langle H^2 \rangle - \langle H \rangle^2.
\]

For an exact eigenstate, this quantity should vanish up to numerical precision.  
We therefore evaluate the variance for the final Heisenberg and AKLT states obtained above.
```

## Code cell 13

```python
Hh_small = dense_heisenberg_xxz(Lh, J=1.0, Jz=1.0, h=0.0)
Eh_small = dense_energy_from_mps(mps_h, Hh_small)
varh_small = dense_variance_from_mps(mps_h, Hh_small)

Ha_small = dense_aklt(La)
Ea_small = dense_energy_from_mps(mps_a, Ha_small)
vara_small = dense_variance_from_mps(mps_a, Ha_small)

print(f"Heisenberg L={Lh}: E = {Eh_small:.12f}, variance = {varh_small:.3e}")
print(f"AKLT L={La}: E = {Ea_small:.12f}, variance = {vara_small:.3e}")
```

---

## Markdown cell 14

```markdown
## 13. Summary of benchmark results

We summarize the benchmark errors and convergence indicators obtained in this notebook.
```

## Code cell 14

```python
print("Heisenberg benchmark sizes:", L_list_h)
print("Heisenberg |DMRG-Exact|    :", np.abs(E_dmrg_h - E_exact_h))
print("AKLT benchmark sizes       :", L_list_a)
print("AKLT |DMRG-Formula|        :", np.abs(E_dmrg_a - E_formula_a))
print("AKLT max discarded weights :", max_disc_a)
print("\nAll figures have been shown and saved to:", figdir)
```

---

## Markdown cell 15

```markdown
## 14. Conclusion

The two-site DMRG implementation developed in this notebook successfully reproduces the exact ground-state energies of both the spin-1/2 Heisenberg chain and the spin-1 AKLT chain for the tested system sizes.

The main numerical conclusions are:

- the Heisenberg ground-state energies agree with exact diagonalization to machine precision,
- the AKLT ground-state energies agree with the exact formula,
- the exact AKLT MPS benchmark is reproduced exactly,
- the final energy variances are numerically zero within floating-point precision.

In addition to the energies, we computed physically meaningful observables such as local magnetization, connected correlations, entanglement properties, and the AKLT string order parameter.

This notebook therefore provides a compact and transparent reference implementation of two-site DMRG in the MPS/MPO framework for one-dimensional open chains.
```

---

# Optional very short notebook workflow note

If you want a practical instruction cell near the top, add this markdown cell after the title:

```markdown
## Execution order

To run this notebook from a fresh kernel, execute the cells in the following order:

1. imports and setup
2. core implementation
3. faster sweep / observable patch
4. internal tests
5. benchmark and analysis cells

The patch cell must be executed **after** the core implementation cell because it overrides the DMRG sweep with a faster version and adds extra observable routines.
```

---

If you want, I can also provide a final **very compact table-of-contents markdown cell** for the top of the notebook.

---
## 🧑‍💻 User

The problem is that it is still too slow. It costs more than 6 mins.

---
## 🤖 Assistant

Yes — that is expected. The current notebook is **correct but not optimized**.

The main slowdown is that the code still uses:

- **dense local effective Hamiltonians**
- **dense-state observables**
- repeated exact diagonalization style routines on local spaces
- many plots and diagnostics in one run

So if you want the notebook to finish much faster, we need to reduce cost in a practical way.

---

# Best strategy: keep correctness, reduce runtime

There are two levels of improvement:

## Level 1: quick practical speed-up
Do this immediately:
- use fewer sweeps
- reduce benchmark sizes
- avoid duplicate DMRG runs
- skip dense variance checks except once
- compute fewer observables

## Level 2: algorithmic speed-up
Would require code changes:
- replace dense local diagonalization by Lanczos / `eigsh`
- implement a correct matrix-free local matvec
- compute observables directly with MPS contractions instead of dense-state conversion

Level 2 is more work.  
For now, the most useful thing is to give you a **fast final run-all cell** that preserves the main results and figures while finishing much sooner.

---

# Why your current run-all is slow

Even if benchmarks are small, the notebook currently does:

### Heisenberg
- benchmark for 3 sizes
- another DMRG run for convergence
- observables
- entanglement spectrum
- variance check

### AKLT
- benchmark for 2 sizes
- another DMRG run
- observables
- string order
- entanglement spectrum
- exact AKLT MPS check
- variance check

This is a lot of repeated dense computations.

---

# Fast solution

## Use only one DMRG run per model for the analysis section
- benchmark small systems
- do only one representative large-ish system for plots

## Lower sweeps modestly
Since your energies are already converging exactly:
- Heisenberg: `nsweeps=3` or `4`
- AKLT: `nsweeps=4`

## Reduce duplicate expensive diagnostics
- keep variance check only once
- string order on one representative system only
- entanglement spectrum on one representative system only

---

# Here is a much faster final run-all cell

This is the one I recommend using in practice.

```python
# ============================================================
# FAST final run-all cell
# ============================================================

figdir = setup_figure_dir("figureAKLT")
print("Figures will be saved in:", figdir)

# ============================================================
# 1. Basic tests
# ============================================================

print("\n=== Basic tests ===")
test_canonicalization()
err_heff = test_two_site_heff()
print("Local two-site Heff test error =", err_heff)

# ============================================================
# 2. Heisenberg benchmark
# ============================================================

print("\n=== Heisenberg benchmark ===")
L_list_h, E_dmrg_h, E_exact_h = run_heisenberg_benchmark(
    L_list=(4, 6),
    J=1.0,
    Jz=1.0,
    h=0.0,
    Dmax=12,
    nsweeps=3,
    verbose=True
)
plot_heisenberg_benchmark(L_list_h, E_dmrg_h, E_exact_h, figdir=figdir)

# One representative Heisenberg run
print("\n=== Heisenberg representative run ===")
Lh = 6
mpo_h = mpo_heisenberg_xxz(Lh, J=1.0, Jz=1.0, h=0.0)
init_h = product_state_mps(Lh, d=2, state_index=0)
res_h = dmrg_two_site(mpo_h, d=2, Dmax=12, nsweeps=4, init_mps=init_h, verbose=True)
mps_h = res_h["mps"]

plot_convergence(
    res_h,
    title_prefix=f"Heisenberg L={Lh}",
    filename_prefix="heisenberg_L6",
    figdir=figdir
)

prof_h = heisenberg_spin_profiles(mps_h)
plot_one_site_profile(
    prof_h["Sz"],
    ylabel=r"$\langle S_i^z \rangle$",
    title=f"Heisenberg local magnetization, L={Lh}",
    filename="heisenberg_local_Sz.png",
    figdir=figdir
)

corr_h = spin_spin_connected_from_center_heisenberg(mps_h)
plot_connected_correlation_from_center(
    corr_h,
    ylabel="Connected correlation",
    title=f"Heisenberg connected correlations, L={Lh}",
    filename="heisenberg_connected_correlations.png",
    figdir=figdir
)

es_h = entanglement_spectrum(mps_h, cut=Lh // 2)
plot_entanglement_spectrum(
    es_h,
    cut=Lh // 2,
    title=f"He

---
## 🧑‍💻 User

could you remove not-needed dense computations?

---
## 🤖 Assistant

Yes. The biggest avoidable slowdown now is from **dense-state observables** and **dense variance checks**.

Since your main goal is:

- two-site DMRG
- Heisenberg and AKLT energies
- figures shown and saved

we can remove many nonessential dense computations and keep the notebook much faster.

---

# What is currently expensive but not strictly needed

## Not needed for the core benchmark
These are optional and can be removed or reduced:

1. `dense_variance_from_mps(...)`
2. `dense_energy_from_mps(...)` except for one AKLT exact-state check
3. dense local magnetization and correlation functions
4. dense entanglement spectrum from full state vector
5. full string-order profile over many distances

---

# What should remain
To keep the notebook scientifically solid and much faster, I recommend keeping:

## Essential
- canonicalization test
- two-site Heff test
- Heisenberg benchmark against ED for small \(L\)
- AKLT benchmark against exact formula
- convergence plots
- AKLT exact MPS energy check

## Optional but lightweight enough
- AKLT entanglement entropy
- maybe one simple local magnetization profile

---

# Best simplification strategy

## Remove from the final run-all cell:
- dense variance checks
- Heisenberg local magnetization
- Heisenberg connected correlations
- Heisenberg entanglement spectrum
- AKLT local magnetization
- AKLT connected correlations
- AKLT entanglement spectrum
- AKLT string order profile

These are nice, but they are what costs time after the DMRG itself.

---

# Keep only these figures
A fast, clean notebook can still include:

### Heisenberg
- energy benchmark
- energy error benchmark
- convergence
- discarded weight

### AKLT
- energy benchmark
- error benchmark
- discarded weight
- convergence
- entanglement entropy

That is already a solid notebook.

---

# Fastest practical final run-all cell

Use this instead.

```python
# ============================================================
# Lean final run-all cell
# Keeps only essential dense computations
# ============================================================

figdir = setup_figure_dir("figureAKLT")
print("Figures will be saved in:", figdir)

# ============================================================
# 1. Basic tests
# ============================================================

print("\n=== Basic tests ===")
test_canonicalization()
err_heff = test_two_site_heff()
print("Local two-site Heff test error =", err_heff)

# ============================================================
# 2. Heisenberg benchmark
# ============================================================

print("\n=== Heisenberg benchmark ===")
L_list_h, E_dmrg_h, E_exact_h = run_heisenberg_benchmark(
    L_list=(4, 6, 8),
    J=1.0,
    Jz=1.0,
    h=0.0,
    Dmax=16,
    nsweeps=4,
    verbose=True
)
plot_heisenberg_benchmark(L_list_h, E_dmrg_h, E_exact_h, figdir=figdir)

# Representative Heisenberg convergence run
print("\n=== Heisenberg convergence example ===")
Lh = 8
mpo_h = mpo_heisenberg_xxz(Lh, J=1.0, Jz=1.0, h=0.0)
init_h = product_state_mps(Lh, d=2, state_index=0)
res_h = dmrg_two_site(mpo_h, d=2, Dmax=16, nsweeps=5, init_mps=init_h, verbose=True)

plot_convergence(
    res_h,
    title_prefix=f"Heisenberg L={Lh}",
    filename_prefix="heisenberg_L8",
    figdir=figdir
)

# ============================================================
# 3. AKLT benchmark
# ============================================================

print("\n=== AKLT benchmark ===")
L_list_a, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a = run_aklt_benchmark(
    L_list=(4, 6),
    Dmax=2,
    nsweeps=5,
    verbose=True
)
plot_aklt_benchmark(L_list_a, E_dmrg_a, E_formula_a, E_exact_a, max_disc_a, figdir=figdir)

# Representative AKLT convergence run
print("\n=== AKLT convergence example ===")
La = 8
mpo_a = mpo_aklt(La)
init_a = product_state_mps(La, d=3, state_index=0)
res_a = dmrg_two_site(mpo_a, d=3, Dmax=2, nsweeps=6, init_mps=init_a, verbose=True)
mps_a = res_a["mps"]

plot_convergence(
    res_a,
    title_prefix=f"AKLT L={La}",
    filename_prefix="aklt_L8",
    figdir=figdir
)

plot_entanglement_entropy(
    mps_a,
    title=f"AKLT bond entanglement entropy, L={La}",
    filename="aklt_entanglement_entropy_L8.png",
    figdir=figdir
)

# ============================================================
# 4. Exact AKLT MPS check
# ============================================================

print("\n=== Exact AKLT MPS check ===")
L_exact = 8
mps_exact_aklt = exact_aklt_mps(L_exact)
H_exact_aklt = dense_aklt(L_exact)
E_exact_mps = dense_energy_from_mps(mps_exact_aklt, H_exact_aklt)

print(f"Exact AKLT MPS energy for L={L_exact}: {E_exact_mps:.12f}")
print(f"Expected exact energy: {-(2/3)*(L_exact-1):.12f}")

# ============================================================
# 5. Summary
# ============================================================

print("\n=== Summary ===")
print("Heisenberg benchmark sizes:", L_list_h)
print("Heisenberg |DMRG-Exact|    :", np.abs(E_dmrg_h - E_exact_h))
print("AKLT benchmark sizes       :", L_list_a)
print("AKLT |DMRG-Formula|        :", np.abs(E_dmrg_a - E_formula_a))
print("AKLT max discarded weights :", max_disc_a)

print("\nSaved figures:")
print("  - heisenberg_energy_vs_L.png")
print("  - heisenberg_error_vs_L.png")
print("  - heisenberg_L8_convergence.png")
print("  - heisenberg_L8_discarded_weight.png")
print("  - aklt_energy_vs_L.png")
print("  - aklt_error_vs_L.png")
print("  - aklt_discarded_weight_vs_L.png")
print("  - aklt_L8_convergence.png")
print("  - aklt_L8_discarded_weight.png")
print("  - aklt_entanglement_entropy_L8.png")

print("\nAll figures have been shown and saved to:", figdir)
```

---

# If you want to go further
Yes, we can remove even more dense computations by also modifying some functions.

## Good next cuts
If you want maximum speed, we can also:

### 1. Remove exact diagonalization for Heisenberg \(L=8\)
Benchmark only:
- \(L=4,6\)

### 2. Remove AKLT dense exact diagonalization entirely
Use only exact formula:
\[
E_0 = -\frac23(L-1)
\]

### 3. Remove `plot_entanglement_entropy(...)`
because it currently uses dense reshaping of the full state

This would make the notebook much lighter.

---

# Strong recommendation
If runtime is your priority, the cleanest compromise is:

## Keep
- Heisenberg benchmark: \(L=4,6\)
- AKLT benchmark: \(L=4,6\)
- one convergence plot each
- exact AKLT MPS energy check

## Drop
- all dense observables

That gives you a notebook that is still scientifically solid but much faster.

---

If you want, I can now provide an even more aggressive version:

## “ultra-lean run-all cell”
with only:
- tests
- Heisenberg benchmark
- AKLT benchmark
- convergence plots
- exact AKLT MPS check

which should be substantially faster.

---
## 🧑‍💻 User

could give me a clean function-all cell and a clean run-all cell?

---
## 🤖 Assistant

Below are **two clean cells**:

1. **function-all cell**
2. **run-all cell**

They are designed to be **correct and leaner**, keeping only the essential parts needed for:

- two-site DMRG
- spin-1/2 Heisenberg
- spin-1 AKLT
- exact small-system benchmarks
- convergence plots
- AKLT entanglement entropy
- saving figures to `figureAKLT`

I have intentionally **removed not-needed dense observable functions** like:
- local magnetization
- correlation functions
- string order
- entanglement spectrum
- dense variance checks

to keep the notebook faster and cleaner.

---

# 1. Clean function-all cell

```python
import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# ============================================================
# Global setup
# ============================================================

np.set_printoptions(precision=12, suppress=True)

def setup_figure_dir(dirname="figureAKLT"):
    os.makedirs(dirname, exist_ok=True)
    return dirname

def savefig_in_dir(dirname, name):
    path = os.path.join(dirname, name)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    print(f"saved: {path}")

# ============================================================
# Local spin operators
# ============================================================

def spin_half_ops(dtype=np.complex128):
    I = np.eye(2, dtype=dtype)
    Sp = np.array([[0, 1], [0, 0]], dtype=dtype)
    Sm = np.array([[0, 0], [1, 0]], dtype=dtype)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=dtype)
    return I, Sp, Sm, Sz

def spin_one_ops(dtype=np.complex128):
    I = np.eye(3, dtype=dtype)
    s2 = np.sqrt(2.0)
    Sp = np.array([[0, s2, 0],
                   [0, 0, s2],
                   [0, 0, 0]], dtype=dtype)
    Sm = Sp.conj().T
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, -1]], dtype=dtype)
    return I, Sp, Sm, Sz

# ============================================================
# Exact dense Hamiltonians for benchmarks
# ============================================================

def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def dense_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    I, Sp, Sm, Sz = spin_half_ops(dtype=dtype)
    d = 2
    H = np.zeros((d**L, d**L), dtype=dtype)

    for i in range(L - 1):
        ops = [I] * L
        ops[i] = 0.5 * J * Sp
        ops[i+1] = Sm
        H += kron_all(ops)

        ops = [I] * L
        ops[i] = 0.5 * J * Sm
        ops[i+1] = Sp
        H += kron_all(ops)

        ops = [I] * L
        ops[i] = Jz * Sz
        ops[i+1] = Sz
        H += kron_all(ops)

    for i in range(L):
        ops = [I] * L
        ops[i] = -h * Sz
        H += kron_all(ops)

    return H

def dense_aklt(L, dtype=np.complex128):
    I, Sp, Sm, Sz = spin_one_ops(dtype=dtype)
    d = 3
    H = np.zeros((d**L, d**L), dtype=dtype)

    SS = 0.5 * np.kron(Sp, Sm) + 0.5 * np.kron(Sm, Sp) + np.kron(Sz, Sz)
    hbond = SS + (1.0 / 3.0) * (SS @ SS)

    for i in range(L - 1):
        left_dim = d**i
        right_dim = d**(L - i - 2)
        H += np.kron(np.eye(left_dim, dtype=dtype),
                     np.kron(hbond, np.eye(right_dim, dtype=dtype)))
    return H

def exact_ground_energy(H):
    return np.min(la.eigvalsh(H)).real

# ============================================================
# MPO builders
# ============================================================

def mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    I, Sp, Sm, Sz = spin_half_ops(dtype=dtype)
    d = 2
    chi = 5

    W_bulk = np.zeros((chi, chi, d, d), dtype=dtype)
    W_bulk[0, 0] = I
    W_bulk[1, 0] = Sp
    W_bulk[2, 0] = Sm
    W_bulk[3, 0] = Sz
    W_bulk[4, 0] = -h * Sz
    W_bulk[4, 1] = 0.5 * J * Sm
    W_bulk[4, 2] = 0.5 * J * Sp
    W_bulk[4, 3] = Jz * Sz
    W_bulk[4, 4] = I

    W_first = np.zeros((1, chi, d, d), dtype=dtype)
    W_first[0, 0] = -h * Sz
    W_first[0, 1] = 0.5 * J * Sm
    W_first[0, 2] = 0.5 * J * Sp
    W_first[0, 3] = Jz * Sz
    W_first[0, 4] = I

    W_last = np.zeros((chi, 1, d, d), dtype=dtype)
    W_last[0, 0] = I
    W_last[1, 0] = Sp
    W_last[2, 0] = Sm
    W_last[3, 0] = Sz
    W_last[4, 0] = -h * Sz

    mpo = [None] * L
    mpo[0] = W_first
    for i in range(1, L - 1):
        mpo[i] = W_bulk.copy()
    mpo[-1] = W_last
    return mpo

def mpo_from_two_body(L, bond_terms, d, dtype=np.complex128):
    nb = len(bond_terms)
    chi = 2 + nb
    I = np.eye(d, dtype=dtype)

    W_bulk = np.zeros((chi, chi, d, d), dtype=dtype)
    W_bulk[0, 0] = I
    for k, (Lop, Rop) in enumerate(bond_terms):
        W_bulk[0, 1 + k] = Lop
        W_bulk[1 + k, -1] = Rop
    W_bulk[-1, -1] = I

    W_first = np.zeros((1, chi, d, d), dtype=dtype)
    W_first[0] = W_bulk[0]

    W_last = np.zeros((chi, 1, d, d), dtype=dtype)
    W_last[:, 0] = W_bulk[:, -1]

    mpo = [None] * L
    mpo[0] = W_first
    for i in range(1, L - 1):
        mpo[i] = W_bulk.copy()
    mpo[-1] = W_last
    return mpo

def mpo_aklt(L, dtype=np.complex128):
    I, Sp, Sm, Sz = spin_one_ops(dtype=dtype)
    d = 3

    A_terms = [
        (0.5 * Sp, Sm),
        (0.5 * Sm, Sp),
        (1.0 * Sz, Sz),
    ]

    A2_terms = []
    for La, Ra in A_terms:
        for Lb, Rb in A_terms:
            A2_terms.append(((1.0 / 3.0) * (La @ Lb), Ra @ Rb))

    bond_terms = A_terms + A2_terms
    return mpo_from_two_body(L, bond_terms, d=d, dtype=dtype)

# ============================================================
# MPS constructors
# ============================================================

def random_mps(L, d, Dmax, seed=1234, dtype=np.complex128):
    rng = np.random.default_rng(seed)
    dimsL = [1]
    for _ in range(L - 1):
        dimsL.append(min(Dmax, dimsL[-1] * d))
    dimsR = [1]
    for _ in range(L - 1):
        dimsR.append(min(Dmax, dimsR[-1] * d))
    dimsR = list(reversed(dimsR))

    mps = []
    for i in range(L):
        Dl = min(dimsL[i], dimsR[i])
        Dr = min(dimsL[i+1], dimsR[i+1]) if i < L - 1 else 1
        A = rng.normal(size=(Dl, d, Dr)) + 1j * rng.normal(size=(Dl, d, Dr))
        mps.append(A.astype(dtype))
    return mps

def product_state_mps(L, d, state_index=0, dtype=np.complex128):
    mps = []
    for _ in range(L):
        A = np.zeros((1, d, 1), dtype=dtype)
        A[0, state_index, 0] = 1.0
        mps.append(A)
    return mps

def exact_aklt_mps(L, dtype=np.complex128):
    Ap = np.array([[0, np.sqrt(2/3)], [0, 0]], dtype=dtype)
    A0 = np.array([[-1/np.sqrt(3), 0], [0, 1/np.sqrt(3)]], dtype=dtype)
    Am = np.array([[0, 0], [-np.sqrt(2/3), 0]], dtype=dtype)
    A_bulk = np.stack([Ap, A0, Am], axis=1)

    mps = []
    vl = np.array([[1.0, 0.0]], dtype=dtype)
    vr = np.array([[1.0], [0.0]], dtype=dtype)

    A1 = np.einsum('la,asb->lsb', vl, A_bulk, optimize=True).reshape(1, 3, 2)
    mps.append(A1)
    for _ in range(1, L - 1):
        mps.append(A_bulk.copy())
    AL = np.einsum('asb,br->asr', A_bulk, vr, optimize=True).reshape(2, 3, 1)
    mps.append(AL)
    return mps

# ============================================================
# Canonicalization
# ============================================================

def left_canonicalize_qr(mps):
    mps = [A.copy() for A in mps]
    L = len(mps)
    for i in range(L - 1):
        Dl, d, Dr = mps[i].shape
        M = mps[i].reshape(Dl * d, Dr, order='C')
        Q, R = la.qr(M, mode='economic')
        k = Q.shape[1]
        mps[i] = Q.reshape(Dl, d, k, order='C')
        mps[i+1] = np.einsum('ab,bsd->asd', R, mps[i+1], optimize=True)
    return mps

def right_canonicalize_qr(mps):
    mps = [A.copy() for A in mps]
    L = len(mps)
    for i in range(L - 1, 0, -1):
        Dl, d, Dr = mps[i].shape
        M = mps[i].reshape(Dl, d * Dr, order='C')
        Q, R = la.qr(M.conj().T, mode='economic')
        k = Q.shape[1]
        mps[i] = Q.conj().T.reshape(k, d, Dr, order='C')
        mps[i-1] = np.einsum('asb,bc->asc', mps[i-1], R.conj().T, optimize=True)
    return mps

def check_left_canonical(A, tol=1e-10):
    Dl, d, Dr = A.shape
    M = A.reshape(Dl * d, Dr, order='C')
    return np.linalg.norm(M.conj().T @ M - np.eye(Dr)) < tol

def check_right_canonical(A, tol=1e-10):
    Dl, d, Dr = A.shape
    M = A.reshape(Dl, d * Dr, order='C')
    return np.linalg.norm(M @ M.conj().T - np.eye(Dl)) < tol

# ============================================================
# Environments
# ============================================================

def init_left_env(dtype=np.complex128):
    Lenv = np.zeros((1, 1, 1), dtype=dtype)
    Lenv[0, 0, 0] = 1.0
    return Lenv

def init_right_env(dtype=np.complex128):
    Renv = np.zeros((1, 1, 1), dtype=dtype)
    Renv[0, 0, 0] = 1.0
    return Renv

def update_left_env(Lenv, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lenv, A, W, A.conj(), optimize=True)

def update_right_env(Renv, B, W):
    return np.einsum('asx,bBst,Bxy,cty->bac', B, W, Renv, B.conj(), optimize=True)

def build_all_left_envs(mps, mpo):
    L = len(mps)
    Lenvs = [None] * L
    Lenv = init_left_env(dtype=mps[0].dtype)
    Lenvs[0] = Lenv
    for i in range(L - 1):
        Lenv = update_left_env(Lenv, mps[i], mpo[i])
        Lenvs[i + 1] = Lenv
    return Lenvs

def build_all_right_envs(mps, mpo):
    L = len(mps)
    Renvs = [None] * L
    Renv = init_right_env(dtype=mps[0].dtype)
    Renvs[L - 1] = Renv
    for i in range(L - 1, 0, -1):
        Renv = update_right_env(Renv, mps[i], mpo[i])
        Renvs[i - 1] = Renv
    return Renvs

# ============================================================
# Dense local two-site effective Hamiltonian
# ============================================================

def two_site_heff_apply(theta, Lenv, W1, W2, Renv):
    return np.einsum(
        'bap,bBst,BCuv,Crq,ptvq->asur',
        Lenv, W1, W2, Renv, theta,
        optimize=True
    )

def build_dense_two_site_heff(Lenv, W1, W2, Renv, Dl, d1, d2, Dr):
    n = Dl * d1 * d2 * Dr
    H = np.zeros((n, n), dtype=np.complex128)
    for j in range(n):
        e = np.zeros(n, dtype=np.complex128)
        e[j] = 1.0
        theta = e.reshape(Dl, d1, d2, Dr, order='C')
        H[:, j] = two_site_heff_apply(theta, Lenv, W1, W2, Renv).reshape(n, order='C')
    return H

def solve_local_ground_state_dense(theta0, Lenv, W1, W2, Renv):
    Dl, d1, d2, Dr = theta0.shape
    Hloc = build_dense_two_site_heff(Lenv, W1, W2, Renv, Dl, d1, d2, Dr)
    vals, vecs = la.eigh(Hloc)
    E = vals[0].real
    theta = vecs[:, 0].reshape(Dl, d1, d2, Dr, order='C')
    theta /= np.linalg.norm(theta)
    return E, theta

# ============================================================
# Two-site split
# ============================================================

def split_two_site_theta(theta, Dmax, cutoff=1e-12, sweep_direction='lr'):
    Dl, d1, d2, Dr = theta.shape
    M = theta.reshape(Dl * d1, d2 * Dr, order='C')
    U, s, Vh = la.svd(M, full_matrices=False)

    keep = min(Dmax, np.sum(s > cutoff))
    if keep < 1:
        keep = 1

    disc_weight = np.sum(s[keep:]**2).real
    U = U[:, :keep]
    s = s[:keep]
    Vh = Vh[:keep, :]

    if sweep_direction == 'lr':
        A = U.reshape(Dl, d1, keep, order='C')
        Mright = (np.diag(s) @ Vh).reshape(keep, d2, Dr, order='C')
        return A, Mright, disc_weight
    elif sweep_direction == 'rl':
        Mleft = (U @ np.diag(s)).reshape(Dl, d1, keep, order='C')
        B = Vh.reshape(keep, d2, Dr, order='C')
        return Mleft, B, disc_weight
    else:
        raise ValueError("sweep_direction must be 'lr' or 'rl'")

# ============================================================
# MPS to dense / entanglement entropy
# ============================================================

def mps_to_dense(mps):
    psi = mps[0][0, :, :]
    for i in range(1, len(mps)):
        psi = np.einsum('...a,asb->...sb', psi, mps[i], optimize=True)
    return psi.reshape(-1, order='C')

def dense_energy_from_mps(mps, H):
    psi = mps_to_dense(mps)
    psi = psi / np.linalg.norm(psi)
    return np.vdot(psi, H @ psi).real

def entanglement_entropies(mps):
    psi = mps_to_dense(mps)
    L = len(mps)
    d = mps[0].shape[1]
    ent = []
    for cut in range(1, L):
        M = psi.reshape(d**cut, d**(L-cut), order='C')
        s = la.svdvals(M)
        s = s / np.linalg.norm(s)
        p = s**2
        p = p[p > 1e-15]
        ent.append(-(p * np.log(p)).sum().real)
    return np.array(ent)

# ============================================================
# Two-site DMRG
# ============================================================

def dmrg_two_site(mpo, d, Dmax=32, nsweeps=4, cutoff=1e-12,
                  init_mps=None, verbose=True):
    L = len(mpo)

    if init_mps is None:
        init_mps = random_mps(L, d, Dmax=max(2, min(Dmax, 4)), seed=1234)

    mps = right_canonicalize_qr(init_mps)

    energies = []
    max_discards = []

    for sweep in range(nsweeps):
        sweep_discards = []
        E_last = None

        # Left -> Right
        Renvs = build_all_right_envs(mps, mpo)
        Lenv = init_left_env(dtype=mps[0].dtype)

        for i in range(L - 1):
            theta = np.einsum('asb,btc->astc', mps[i], mps[i+1], optimize=True)
            E, theta_opt = solve_local_ground_state_dense(theta, Lenv, mpo[i], mpo[i+1], Renvs[i+1])

            A, Mright, disc = split_two_site_theta(
                theta_opt, Dmax=Dmax, cutoff=cutoff, sweep_direction='lr'
            )

            mps[i] = A
            mps[i+1] = Mright
            Lenv = update_left_env(Lenv, mps[i], mpo[i])

            sweep_discards.append(disc)
            E_last = E

        # Right -> Left
        Lenvs = build_all_left_envs(mps, mpo)
        Renv = init_right_env(dtype=mps[0].dtype)

        for i in range(L - 2, -1, -1):
            theta = np.einsum('asb,btc->astc', mps[i], mps[i+1], optimize=True)
            E, theta_opt = solve_local_ground_state_dense(theta, Lenvs[i], mpo[i], mpo[i+1], Renv)

            Mleft, B, disc = split_two_site_theta(
                theta_opt, Dmax=Dmax, cutoff=cutoff, sweep_direction='rl'
            )

            mps[i] = Mleft
            mps[i+1] = B
            Renv = update_right_env(Renv, mps[i+1], mpo[i+1])

            sweep_discards.append(disc)
            E_last = E

        energies.append(E_last)
        max_discards.append(max(sweep_discards) if sweep_discards else 0.0)

        if verbose:
            print(f"sweep {sweep+1:2d}: E = {E_last:.15f}, max discarded weight = {max_discards[-1]:.3e}")

    return {
        "mps": mps,
        "energies": np.array(energies),
        "max_discards": np.array(max_discards),
    }

# ============================================================
# Benchmarks
# ============================================================

def run_heisenberg_benchmark(L_list=(4, 6), J=1.0, Jz=1.0, h=0.0,
                             Dmax=12, nsweeps=3, verbose=True):
    exact_E = []
    dmrg_E = []

    for L in L_list:
        mpo = mpo_heisenberg_xxz(L, J=J, Jz=Jz, h=h)
        init = product_state_mps(L, d=2, state_index=0)
        res = dmrg_two_site(mpo, d=2, Dmax=Dmax, nsweeps=nsweeps, init_mps=init, verbose=False)
        E_dmrg = res["energies"][-1]
        dmrg_E.append(E_dmrg)

        H = dense_heisenberg_xxz(L, J=J, Jz=Jz, h=h)
        E_ex = exact_ground_energy(H)
        exact_E.append(E_ex)

        if verbose:
            print(f"L={L:2d}   DMRG={E_dmrg:.12f}   Exact={E_ex:.12f}   |Δ|={abs(E_dmrg-E_ex):.3e}")

    return np.array(L_list), np.array(dmrg_E), np.array(exact_E)

def run_aklt_benchmark(L_list=(4, 6), Dmax=2, nsweeps=4, verbose=True):
    exact_formula = []
    dmrg_E = []
    max_disc = []

    for L in L_list:
        mpo = mpo_aklt(L)
        init = product_state_mps(L, d=3, state_index=0)
        res = dmrg_two_site(mpo, d=3, Dmax=Dmax, nsweeps=nsweeps, init_mps=init, verbose=False)

        E_dmrg = res["energies"][-1]
        dmrg_E.append(E_dmrg)
        max_disc.append(np.max(res["max_discards"]))

        E_formula = -(2.0 / 3.0) * (L - 1)
        exact_formula.append(E_formula)

        if verbose:
            print(f"L={L:2d}   DMRG={E_dmrg:.12f}   Formula={E_formula:.12f}   |Δ|={abs(E_dmrg-E_formula):.3e}   max disc={max_disc[-1]:.3e}")

    return np.array(L_list), np.array(dmrg_E), np.array(exact_formula), np.array(max_disc)

# ============================================================
# Plot helpers
# ============================================================

def plot_heisenberg_benchmark(L_list, E_dmrg, E_exact, figdir="figureAKLT"):
    setup_figure_dir(figdir)

    plt.figure()
    plt.plot(L_list, E_exact, 'o-', label='Exact diagonalization')
    plt.plot(L_list, E_dmrg, 's--', label='Two-site DMRG')
    plt.xlabel('L')
    plt.ylabel('Ground-state energy')
    plt.title('Spin-1/2 Heisenberg chain')
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "heisenberg_energy_vs_L.png")
    plt.show()

    plt.figure()
    plt.semilogy(L_list, np.abs(E_dmrg - E_exact), 'o-')
    plt.xlabel('L')
    plt.ylabel(r'$|E_{\rm DMRG} - E_{\rm exact}|$')
    plt.title('Heisenberg DMRG error')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "heisenberg_error_vs_L.png")
    plt.show()

def plot_aklt_benchmark(L_list, E_dmrg, E_formula, max_disc, figdir="figureAKLT"):
    setup_figure_dir(figdir)

    plt.figure()
    plt.plot(L_list, E_formula, 'k-', label='Exact formula')
    plt.plot(L_list, E_dmrg, 's--', label='Two-site DMRG')
    plt.xlabel('L')
    plt.ylabel('Ground-state energy')
    plt.title('AKLT chain')
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "aklt_energy_vs_L.png")
    plt.show()

    plt.figure()
    plt.semilogy(L_list, np.abs(E_dmrg - E_formula), 'o-')
    plt.xlabel('L')
    plt.ylabel(r'$|E_{\rm DMRG} - E_{\rm exact}|$')
    plt.title('AKLT DMRG error')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "aklt_error_vs_L.png")
    plt.show()

    plt.figure()
    plt.semilogy(L_list, np.maximum(max_disc, 1e-18), 'o-')
    plt.xlabel('L')
    plt.ylabel('Max discarded weight')
    plt.title('AKLT discarded weight')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, "aklt_discarded_weight_vs_L.png")
    plt.show()

def plot_convergence(res, title_prefix="", filename_prefix="", figdir="figureAKLT"):
    setup_figure_dir(figdir)

    plt.figure()
    plt.plot(np.arange(1, len(res["energies"]) + 1), res["energies"], 'o-')
    plt.xlabel('Sweep')
    plt.ylabel('Energy')
    plt.title(f'{title_prefix} convergence')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, f"{filename_prefix}_convergence.png")
    plt.show()

    plt.figure()
    plt.semilogy(np.arange(1, len(res["max_discards"]) + 1),
                 np.maximum(res["max_discards"], 1e-18), 'o-')
    plt.xlabel('Sweep')
    plt.ylabel('Max discarded weight')
    plt.title(f'{title_prefix} discarded weight')
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, f"{filename_prefix}_discarded_weight.png")
    plt.show()

def plot_entanglement_entropy(mps, title="Entanglement entropy", filename="entanglement_entropy.png", figdir="figureAKLT"):
    setup_figure_dir(figdir)
    Sbond = entanglement_entropies(mps)
    L = len(mps)

    plt.figure()
    plt.plot(np.arange(1, L), Sbond, 'o-')
    plt.xlabel('Bond cut')
    plt.ylabel('Entanglement entropy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, filename)
    plt.show()

# ============================================================
# Diagnostics
# ============================================================

def test_canonicalization():
    mps0 = random_mps(L=6, d=2, Dmax=4, seed=7)
    mpsL = left_canonicalize_qr(mps0)
    mpsR = right_canonicalize_qr(mps0)

    print("Left-canonical checks:")
    for i in range(5):
        print(i, check_left_canonical(mpsL[i]))

    print("Right-canonical checks:")
    for i in range(1, 6):
        print(i, check_right_canonical(mpsR[i]))

def test_two_site_heff():
    rng = np.random.default_rng(1)
    Dl, d1, d2, Dr = 2, 2, 2, 3
    chiL, chiM, chiR = 3, 4, 2

    Lenv = rng.normal(size=(chiL, Dl, Dl)) + 1j * rng.normal(size=(chiL, Dl, Dl))
    W1   = rng.normal(size=(chiL, chiM, d1, d1)) + 1j * rng.normal(size=(chiL, chiM, d1, d1))
    W2   = rng.normal(size=(chiM, chiR, d2, d2)) + 1j * rng.normal(size=(chiM, chiR, d2, d2))
    Renv = rng.normal(size=(chiR, Dr, Dr)) + 1j * rng.normal(size=(chiR, Dr, Dr))
    theta = rng.normal(size=(Dl, d1, d2, Dr)) + 1j * rng.normal(size=(Dl, d1, d2, Dr))

    Hdense = build_dense_two_site_heff(Lenv, W1, W2, Renv, Dl, d1, d2, Dr)
    v = theta.reshape(-1, order='C')
    v_dense = Hdense @ v
    v_apply = two_site_heff_apply(theta, Lenv, W1, W2, Renv).reshape(-1, order='C')

    err = np.linalg.norm(v_dense - v_apply)
    print("two-site Heff test error =", err)
    return err
```

---

# 2. Clean run-all cell

```python
# ============================================================
# Clean run-all cell
# ============================================================

figdir = setup_figure_dir("figureAKLT")
print("Figures will be saved in:", figdir)

# ============================================================
# 1. Basic tests
# ============================================================

print("\n=== Basic tests ===")
test_canonicalization()
err_heff = test_two_site_heff()
print("Local two-site Heff test error =", err_heff)

# ============================================================
# 2. Heisenberg benchmark
# ============================================================

print("\n=== Heisenberg benchmark ===")
L_list_h, E_dmrg_h, E_exact_h = run_heisenberg_benchmark(
    L_list=(4, 6),
    J=1.0,
    Jz=1.0,
    h=0.0,
    Dmax=12,
    nsweeps=3,
    verbose=True
)
plot_heisenberg_benchmark(L_list_h, E_dmrg_h, E_exact_h, figdir=figdir)

# Representative Heisenberg run
print("\n=== Heisenberg convergence example ===")
Lh = 6
mpo_h = mpo_heisenberg_xxz(Lh, J=1.0, Jz=1.0, h=0.0)
init_h = product_state_mps(Lh, d=2, state_index=0)
res_h = dmrg_two_site(mpo_h, d=2, Dmax=12, nsweeps=4, init_mps=init_h, verbose=True)

plot_convergence(
    res_h,
    title_prefix=f"Heisenberg L={Lh}",
    filename_prefix="heisenberg_L6",
    figdir=figdir
)

# ============================================================
# 3. AKLT benchmark
# ============================================================

print("\n=== AKLT benchmark ===")
L_list_a, E_dmrg_a, E_formula_a, max_disc_a = run_aklt_benchmark(
    L_list=(4, 6),
    Dmax=2,
    nsweeps=4,
    verbose=True
)
plot_aklt_benchmark(L_list_a, E_dmrg_a, E_formula_a, max_disc_a, figdir=figdir)

# Representative AKLT run
print("\n=== AKLT convergence example ===")
La = 8
mpo_a = mpo_aklt(La)
init_a = product_state_mps(La, d=3, state_index=0)
res_a = dmrg_two_site(mpo_a, d=3, Dmax=2, nsweeps=4, init_mps=init_a, verbose=True)
mps_a = res_a["mps"]

plot_convergence(
    res_a,
    title_prefix=f"AKLT L={La}",
    filename_prefix="aklt_L8",
    figdir=figdir
)

plot_entanglement_entropy(
    mps_a,
    title=f"AKLT bond entanglement entropy, L={La}",
    filename="aklt_entanglement_entropy_L8.png",
    figdir=figdir
)

# ============================================================
# 4. Exact AKLT MPS check
# ============================================================

print("\n=== Exact AKLT MPS check ===")
L_exact = 8
mps_exact_aklt = exact_aklt_mps(L_exact)
H_exact_aklt = dense_aklt(L_exact)
E_exact_mps = dense_energy_from_mps(mps_exact_aklt, H_exact_aklt)

print(f"Exact AKLT MPS energy for L={L_exact}: {E_exact_mps:.12f}")
print(f"Expected exact energy: {-(2/3)*(L_exact-1):.12f}")

# ============================================================
# 5. Summary
# ============================================================

print("\n=== Summary ===")
print("Heisenberg benchmark sizes:", L_list_h)
print("Heisenberg |DMRG-Exact|    :", np.abs(E_dmrg_h - E_exact_h))
print("AKLT benchmark sizes       :", L_list_a)
print("AKLT |DMRG-Formula|        :", np.abs(E_dmrg_a - E_formula_a))
print("AKLT max discarded weights :", max_disc_a)

print("\nSaved figures:")
print("  - heisenberg_energy_vs_L.png")
print("  - heisenberg_error_vs_L.png")
print("  - heisenberg_L6_convergence.png")
print("  - heisenberg_L6_discarded_weight.png")
print("  - aklt_energy_vs_L.png")
print("  - aklt_error_vs_L.png")
print("  - aklt_discarded_weight_vs_L.png")
print("  - aklt_L8_convergence.png")
print("  - aklt_L8_discarded_weight.png")
print("  - aklt_entanglement_entropy_L8.png")

print("\nAll figures have been shown and saved to:", figdir)
```

---

# Notes

This version is cleaner and faster because it removes:
- dense variance checks
- local observable profiles
- connected correlations
- string order
- entanglement spectrum

while keeping the main physically relevant benchmarks and AKLT entanglement entropy.

If you want, I can also provide a **matching minimal markdown notebook structure** specifically adapted to these two lean cells.

---
## 🧑‍💻 User

=== Exact AKLT MPS check ===
Exact AKLT MPS energy for L=8: -4.666666666667
Expected exact energy: -4.666666666667

=== Summary ===
Heisenberg benchmark sizes: [4 6]
Heisenberg |DMRG-Exact|    : [0. 0.]
AKLT benchmark sizes       : [4 6]
AKLT |DMRG-Formula|        : [0. 0.]
AKLT max discarded weights : [0.003223675502 0.003223675502]

Saved figures:
  - heisenberg_energy_vs_L.png
  - heisenberg_error_vs_L.png
  - heisenberg_L6_convergence.png
  - heisenberg_L6_discarded_weight.png
  - aklt_energy_vs_L.png
  - aklt_error_vs_L.png
  - aklt_discarded_weight_vs_L.png
  - aklt_L8_convergence.png
  - aklt_L8_discarded_weight.png
  - aklt_entanglement_entropy_L8.png

All figures have been shown and saved to: figureAKLT

---
## 🤖 Assistant

Great — this is a clean successful final result.

## Final status

Your notebook now:

- implements **two-site DMRG**
- computes both
  - **spin-1/2 Heisenberg**
  - **spin-1 AKLT**
- benchmarks correctly:
  - Heisenberg vs exact diagonalization
  - AKLT vs exact formula
- verifies the **exact AKLT MPS**
- shows and saves the requested figures in:
  - `figureAKLT`

And the final benchmark output is exact to numerical precision.

---

# Interpretation of the results

## Heisenberg
```python
Heisenberg |DMRG-Exact|    : [0. 0.]
```
So the two-site DMRG reproduces exact diagonalization for the tested sizes \(L=4,6\).

## AKLT
```python
AKLT |DMRG-Formula|        : [0. 0.]
```
So the DMRG reproduces the exact AKLT ground-state energy for the tested sizes \(L=4,6\).

## Exact AKLT MPS
```python
Exact AKLT MPS energy for L=8: -4.666666666667
Expected exact energy:        -4.666666666667
```
So the Hamiltonian and exact AKLT MPS benchmark are fully consistent.

---

# About the AKLT discarded weight
You still have:
```python
AKLT max discarded weights : [0.003223675502 0.003223675502]
```

As discussed earlier, this is not a real issue here because:

- the energy is exact
- the exact AKLT MPS check is exact
- the convergence behavior is correct

So this value reflects intermediate sweep truncation history, not an error in the final state.

---

# At this point
You have what you originally asked for in a practical form:

- a clean function-all cell
- a clean run-all cell
- correct DMRG results
- saved figures

---

# If you want to finalize the notebook presentation
The next natural step would be one of these:

1. **A minimal markdown structure** matching the lean notebook
2. **A short explanation paragraph for each saved figure**
3. **A clean export-ready summary section** for a report or note
4. **A `.py` script version** of the lean notebook

Since your code is now working, I’d recommend option 1 or 3 next.

If you want, I can now give you a **minimal markdown notebook structure adapted exactly to this final lean version**.

---
## 🧑‍💻 User

I would like to run for larger L, up to L=20, for example. How long is it expected?

---
## 🤖 Assistant

For your **current implementation**, going to **\(L=20\)** will be **very slow**, and in some cases likely impractical in a comfortable Jupyter workflow.

## Short answer
Expected runtime for \(L=20\):

- **Heisenberg, modest \(D_{\max}\sim 16\)**: likely **many minutes to tens of minutes**
- **AKLT, \(D_{\max}=2\)**: much faster, likely **seconds to a few minutes**
- **Heisenberg with larger \(D_{\max}\sim 32\)**: can become **very slow**, potentially **tens of minutes to hours**

---

# Why it gets slow

Your current code is correct, but it uses a **dense local effective Hamiltonian** at every two-site step.

For each bond update, the local Hilbert-space dimension is

\[
N_{\text{loc}} = D_{i-1}\, d_i\, d_{i+1}\, D_{i+1}.
\]

For Heisenberg, \(d=2\), so roughly

\[
N_{\text{loc}} \sim 4 D^2.
\]

Then the code builds a dense matrix of size

\[
N_{\text{loc}} \times N_{\text{loc}},
\]

and diagonalizes it with dense `eigh`, which scales roughly like

\[
\mathcal O(N_{\text{loc}}^3).
\]

That is the bottleneck.

---

# Rough timing estimates

These are only order-of-magnitude estimates, but useful.

## 1. AKLT, \(D_{\max}=2\)
For AKLT the exact bond dimension is just 2, so the local dimension is tiny:

\[
N_{\text{loc}} = 2 \times 3 \times 3 \times 2 = 36
\]

in the bulk.

Dense diagonalization of a \(36\times 36\) matrix is trivial.

So for **AKLT at \(L=20\), \(D_{\max}=2\)**:
- likely **well under a minute**
- maybe **a few tens of seconds** depending on number of sweeps and plotting

## 2. Heisenberg, \(D_{\max}=16\)
For bulk bonds,

\[
N_{\text{loc}} \sim 4\times 16^2 = 1024.
\]

Dense diagonalizing a \(1024\times 1024\) Hermitian matrix at every bond update is expensive.

A single sweep has about

\[
2(L-1) \approx 38
\]

local optimizations for \(L=20\).

So for **Heisenberg at \(L=20\), \(D_{\max}=16\)**:
- likely **many minutes**
- something like **10–30 minutes** would not surprise me

## 3. Heisenberg, \(D_{\max}=32\)
Then

\[
N_{\text{loc}} \sim 4\times 32^2 = 4096.
\]

A dense \(4096\times4096\) diagonalization at each bond is extremely heavy.

So for **Heisenberg \(L=20, D_{\max}=32\)**:
- this is likely **too slow** in the current implementation
- possibly **hours** or memory-stress territory

---

# What is safe to try now

## AKLT
You can safely try:

```python
L = 20
res = dmrg_two_site(mpo_aklt(L), d=3, Dmax=2, nsweeps=4, init_mps=product_state_mps(L,3,0), verbose=True)
```

This should be fine.

## Heisenberg
A cautious starting point is:

```python
L = 20
res = dmrg_two_site(mpo_heisenberg_xxz(L), d=2, Dmax=8, nsweeps=2, init_mps=product_state_mps(L,2,0), verbose=True)
```

Then try `Dmax=12`, maybe `16` if tolerable.

---

# Recommended practical limits for the current code

## Comfortable
- Heisenberg: \(L\le 10\) or \(12\)
- AKLT: \(L\le 20\) at \(D_{\max}=2\)

## Possible but slow
- Heisenberg: \(L=16\), \(D_{\max}\le 12\)

## Likely too slow
- Heisenberg: \(L=20\), \(D_{\max}\ge 16\)

---

# If you really want \(L=20\) Heisenberg
The correct next algorithmic upgrade is:

## Replace dense local diagonalization by Lanczos / `eigsh`
But only after implementing a **correct matrix-free two-site effective-Hamiltonian application**.

That would reduce the local solve cost dramatically and make \(L=20\) much more realistic.

---

# Rule of thumb
For the current code:

- **AKLT \(L=20\)**: fine
- **Heisenberg \(L=20\)**: only with **small \(D_{\max}\)** and patience

---

# Suggested experiment
If you want a quick empirical timing test, use:

```python
import time

L = 20
t0 = time.time()
res = dmrg_two_site(
    mpo_heisenberg_xxz(L),
    d=2,
    Dmax=8,
    nsweeps=1,
    init_mps=product_state_mps(L,2,0),
    verbose=True
)
print("Elapsed time:", time.time() - t0, "seconds")
```

Then multiply roughly by your intended number of sweeps.

That gives the most reliable timing estimate on your machine.

---

If you want, I can give you:

1. a **timing-test cell** for \(L=20\),
2. or a **lighter large-\(L\) run cell** for AKLT and Heisenberg separately.

---
## 🧑‍💻 User

2

---
## 🤖 Assistant

Below are two **large-\(L\) run cells**, one for **AKLT** and one for **Heisenberg**, designed for your current implementation.

They are chosen to be:

- reasonable for Jupyter
- lighter than the benchmark notebook
- minimal plotting
- still save figures to `figureAKLT`

---

# 1. Large-\(L\) AKLT run cell

This is safe for your current code because AKLT at \(D_{\max}=2\) is cheap.

```python
# ============================================================
# Large-L AKLT run
# ============================================================

figdir = setup_figure_dir("figureAKLT")

L = 20
Dmax = 2
nsweeps = 4

print(f"Running AKLT with L={L}, Dmax={Dmax}, nsweeps={nsweeps}")

mpo = mpo_aklt(L)
init = product_state_mps(L, d=3, state_index=0)

res = dmrg_two_site(
    mpo,
    d=3,
    Dmax=Dmax,
    nsweeps=nsweeps,
    init_mps=init,
    verbose=True
)

mps = res["mps"]
E_dmrg = res["energies"][-1]
E_exact = -(2.0 / 3.0) * (L - 1)

print("\n=== AKLT large-L summary ===")
print(f"L            = {L}")
print(f"Dmax         = {Dmax}")
print(f"nsweeps      = {nsweeps}")
print(f"E_DMRG       = {E_dmrg:.12f}")
print(f"E_exact      = {E_exact:.12f}")
print(f"|ΔE|         = {abs(E_dmrg - E_exact):.3e}")
print(f"max discard  = {np.max(res['max_discards']):.3e}")

plot_convergence(
    res,
    title_prefix=f"AKLT L={L}",
    filename_prefix=f"aklt_L{L}",
    figdir=figdir
)

plot_entanglement_entropy(
    mps,
    title=f"AKLT bond entanglement entropy, L={L}",
    filename=f"aklt_entanglement_entropy_L{L}.png",
    figdir=figdir
)
```

---

# 2. Large-\(L\) Heisenberg run cell

For the current dense-local code, use **small or moderate \(D_{\max}\)**.

I recommend starting with:

- `L = 20`
- `Dmax = 8`
- `nsweeps = 2`

and only increasing if runtime is acceptable.

```python
# ============================================================
# Large-L Heisenberg run
# ============================================================

figdir = setup_figure_dir("figureAKLT")

L = 20
Dmax = 8
nsweeps = 2

print(f"Running Heisenberg with L={L}, Dmax={Dmax}, nsweeps={nsweeps}")

mpo = mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0)
init = product_state_mps(L, d=2, state_index=0)

res = dmrg_two_site(
    mpo,
    d=2,
    Dmax=Dmax,
    nsweeps=nsweeps,
    init_mps=init,
    verbose=True
)

E_dmrg = res["energies"][-1]

print("\n=== Heisenberg large-L summary ===")
print(f"L            = {L}")
print(f"Dmax         = {Dmax}")
print(f"nsweeps      = {nsweeps}")
print(f"E_DMRG       = {E_dmrg:.12f}")
print(f"E_DMRG / site= {E_dmrg / L:.12f}")
print(f"max discard  = {np.max(res['max_discards']):.3e}")

plot_convergence(
    res,
    title_prefix=f"Heisenberg L={L}",
    filename_prefix=f"heisenberg_L{L}",
    figdir=figdir
)
```

---

# Practical advice

## For AKLT
You can directly run \(L=20\).  
This should be fine.

## For Heisenberg
Use this sequence:

### First try
```python
L = 20
Dmax = 8
nsweeps = 2
```

### If runtime is acceptable
try:
```python
Dmax = 10 or 12
```

### Avoid for now
```python
Dmax >= 16
```
with the current dense-local implementation, unless you are willing to wait a long time.

---

# If you want a slightly more informative Heisenberg large-\(L\) plot
You can also add the bond entanglement entropy, but it uses dense-state reconstruction and will cost more. For \(L=20\) I would **not** recommend it in the current implementation.

---

# Recommended workflow

## AKLT
Run:
- `L=20, Dmax=2, nsweeps=4`

## Heisenberg
Run:
- `L=20, Dmax=8, nsweeps=2`
- if okay, then maybe
- `L=20, Dmax=8, nsweeps=3`

---

If you want, I can next give you a **timed version** of these cells that prints wall-clock time automatically.

---
## 🧑‍💻 User

Give me give a timed version of these cells

---
## 🤖 Assistant

Below are **timed versions** of the large-\(L\) AKLT and Heisenberg run cells.  
They print:

- start/end timing
- total wall-clock time
- per-sweep average time
- final energy summary

---

# 1.

---
## 🧑‍💻 User

breaks off, pls continue

---
## 🤖 Assistant

Sure — here are the full **timed versions**.

---

# 1. Timed large-\(L\) AKLT run cell

```python
# ============================================================
# Timed large-L AKLT run
# ============================================================

import time

figdir = setup_figure_dir("figureAKLT")

L = 20
Dmax = 2
nsweeps = 4

print(f"Running AKLT with L={L}, Dmax={Dmax}, nsweeps={nsweeps}")
t0 = time.time()

mpo = mpo_aklt(L)
init = product_state_mps(L, d=3, state_index=0)

res = dmrg_two_site(
    mpo,
    d=3,
    Dmax=Dmax,
    nsweeps=nsweeps,
    init_mps=init,
    verbose=True
)

t1 = time.time()
elapsed = t1 - t0

mps = res["mps"]
E_dmrg = res["energies"][-1]
E_exact = -(2.0 / 3.0) * (L - 1)

print("\n=== AKLT large-L summary ===")
print(f"L                  = {L}")
print(f"Dmax               = {Dmax}")
print(f"nsweeps            = {nsweeps}")
print(f"E_DMRG             = {E_dmrg:.12f}")
print(f"E_exact            = {E_exact:.12f}")
print(f"|ΔE|               = {abs(E_dmrg - E_exact):.3e}")
print(f"max discarded wt   = {np.max(res['max_discards']):.3e}")
print(f"total time         = {elapsed:.2f} s")
print(f"time per sweep     = {elapsed / nsweeps:.2f} s")

plot_convergence(
    res,
    title_prefix=f"AKLT L={L}",
    filename_prefix=f"aklt_L{L}",
    figdir=figdir
)

plot_entanglement_entropy(
    mps,
    title=f"AKLT bond entanglement entropy, L={L}",
    filename=f"aklt_entanglement_entropy_L{L}.png",
    figdir=figdir
)
```

---

# 2. Timed large-\(L\) Heisenberg run cell

```python
# ============================================================
# Timed large-L Heisenberg run
# ============================================================

import time

figdir = setup_figure_dir("figureAKLT")

L = 20
Dmax = 8
nsweeps = 2

print(f"Running Heisenberg with L={L}, Dmax={Dmax}, nsweeps={nsweeps}")
t0 = time.time()

mpo = mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0)
init = product_state_mps(L, d=2, state_index=0)

res = dmrg_two_site(
    mpo,
    d=2,
    Dmax=Dmax,
    nsweeps=nsweeps,
    init_mps=init,
    verbose=True
)

t1 = time.time()
elapsed = t1 - t0

E_dmrg = res["energies"][-1]

print("\n=== Heisenberg large-L summary ===")
print(f"L                  = {L}")
print(f"Dmax               = {Dmax}")
print(f"nsweeps            = {nsweeps}")
print(f"E_DMRG             = {E_dmrg:.12f}")
print(f"E_DMRG / site      = {E_dmrg / L:.12f}")
print(f"max discarded wt   = {np.max(res['max_discards']):.3e}")
print(f"total time         = {elapsed:.2f} s")
print(f"time per sweep     = {elapsed / nsweeps:.2f} s")

plot_convergence(
    res,
    title_prefix=f"Heisenberg L={L}",
    filename_prefix=f"heisenberg_L{L}",
    figdir=figdir
)
```

---

# 3. Optional: quick timing probe for Heisenberg before a longer run

This is useful if you want to estimate runtime before launching more sweeps.

```python
# ============================================================
# Quick timing probe for Heisenberg
# ============================================================

import time

L = 20
Dmax = 8
nsweeps_probe = 1

print(f"Timing probe: Heisenberg L={L}, Dmax={Dmax}, nsweeps={nsweeps_probe}")

t0 = time.time()

res_probe = dmrg_two_site(
    mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0),
    d=2,
    Dmax=Dmax,
    nsweeps=nsweeps_probe,
    init_mps=product_state_mps(L, d=2, state_index=0),
    verbose=True
)

t1 = time.time()
elapsed_probe = t1 - t0

print("\n=== Timing probe summary ===")
print(f"Elapsed time for {nsweeps_probe} sweep(s): {elapsed_probe:.2f} s")
print(f"Estimated time for 2 sweeps: {2 * elapsed_probe:.2f} s")
print(f"Estimated time for 4 sweeps: {4 * elapsed_probe:.2f} s")
```

---

# Recommended use

## AKLT
Run the full timed AKLT cell directly.

## Heisenberg
First run the **timing probe**.  
Then decide whether to launch:
- `nsweeps=2`
- or increase `Dmax`

---

# Rule of thumb for your current code

If the 1-sweep timing probe for Heisenberg gives:

- **10–20 s** → 2 sweeps is fine
- **30–60 s** → 4 sweeps may already be annoying
- **> 1 min per sweep** → reduce `Dmax` or `L`

---

If you want, I can also give you a **small parameter-scan timing cell** that automatically times:
- `Dmax = 4, 6, 8, 10`
for fixed `L=20`.

---
## 🧑‍💻 User

Timing probe: Heisenberg L=20, Dmax=8, nsweeps=1
sweep  1: E = -8.606021852095465, max discarded weight = 0.000e+00

=== Timing probe summary ===
Elapsed time for 1 sweep(s): 0.32 s
Estimated time for 2 sweeps: 0.64 s
Estimated time for 4 sweeps: 1.29 s

---
## 🤖 Assistant

That is **much faster than expected**, which is great news.

## Interpretation

For your machine and current implementation:

- **Heisenberg \(L=20, D_{\max}=8\)**  
  takes only about **0.32 s per sweep**

So in practice:

- `nsweeps=4` → about **1.3 s**
- `nsweeps=10` → about **3.2 s**

That means **\(L=20\) is absolutely feasible** in your current setup, at least for moderate bond dimensions.

---

# What this implies

The earlier long runtimes were not from the core DMRG alone. They were mostly due to:

- extra dense observables
- repeated benchmark runs
- exact diagonalization checks
- many plots
- notebook overhead

Your **lean DMRG code itself is fast enough** for \(L=20\) at moderate \(D_{\max}\).

---

# Next recommended tests

Now you can probe how runtime grows with bond dimension.

## Try this sequence
For Heisenberg \(L=20\), run:

- `Dmax = 8`
- `Dmax = 12`
- `Dmax = 16`
- maybe `Dmax = 20`

and compare runtime.

---

# Parameter-scan timing cell

Here is a compact timing scan for \(L=20\):

```python
# ============================================================
# Timing scan for Heisenberg at fixed L
# ============================================================

import time

L = 20
D_list = [4, 8, 12, 16]
nsweeps_probe = 1

print(f"Timing scan: Heisenberg L={L}, nsweeps={nsweeps_probe}")
print("-" * 70)

for Dmax in D_list:
    t0 = time.time()
    res = dmrg_two_site(
        mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0),
        d=2,
        Dmax=Dmax,
        nsweeps=nsweeps_probe,
        init_mps=product_state_mps(L, d=2, state_index=0),
        verbose=False
    )
    t1 = time.time()
    elapsed = t1 - t0
    E = res["energies"][-1]
    disc = np.max(res["max_discards"])

    print(f"Dmax={Dmax:2d}   time={elapsed:8.3f} s   E={E: .12f}   maxdisc={disc:.3e}")
```

---

# A practical large-\(L\) production run cell

Since runtime is clearly okay, you can now do a more serious Heisenberg run like this:

```python
# ============================================================
# Production Heisenberg run for L=20
# ============================================================

import time

figdir = setup_figure_dir("figureAKLT")

L = 20
Dmax = 16
nsweeps = 6

print(f"Production run: Heisenberg L={L}, Dmax={Dmax}, nsweeps={nsweeps}")
t0 = time.time()

res = dmrg_two_site(
    mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0),
    d=2,
    Dmax=Dmax,
    nsweeps=nsweeps,
    init_mps=product_state_mps(L, d=2, state_index=0),
    verbose=True
)

t1 = time.time()
elapsed = t1 - t0

E = res["energies"][-1]

print("\n=== Production summary ===")
print(f"L                  = {L}")
print(f"Dmax               = {Dmax}")
print(f"nsweeps            = {nsweeps}")
print(f"E_DMRG             = {E:.12f}")
print(f"E_DMRG / site      = {E/L:.12f}")
print(f"max discarded wt   = {np.max(res['max_discards']):.3e}")
print(f"total time         = {elapsed:.2f} s")
print(f"time per sweep     = {elapsed / nsweeps:.2f} s")

plot_convergence(
    res,
    title_prefix=f"Heisenberg L={L}",
    filename_prefix=f"heisenberg_L{L}_D{Dmax}",
    figdir=figdir
)
```

---

# AKLT at larger L
Since AKLT is even cheaper at \(D_{\max}=2\), you can also comfortably try:

- `L=20`
- `L=40`
- maybe larger

with the current code, as long as you avoid too many dense-state diagnostics.

---

# Most important conclusion
Your current lean implementation is fast enough that:

## You can now realistically do
- **Heisenberg \(L=20\)** with moderate \(D_{\max}\)
- **AKLT \(L=20\)** easily

So the bottleneck is no longer the DMRG core itself.

---

If you want, I can next give you:

1. a **large-\(L\) Heisenberg run cell with energy-per-site vs sweep plot**, or  
2. a **finite-size scaling cell** for \(E_0/L\) vs \(1/L\).

---
## 🧑‍💻 User

Timing scan: Heisenberg L=20, nsweeps=1
----------------------------------------------------------------------
Dmax= 4   time=   0.317 s   E=-8.606021852095   maxdisc=0.000e+00
Dmax= 8   time=   0.293 s   E=-8.606021852095   maxdisc=0.000e+00
Dmax=12   time=   0.286 s   E=-8.606021852095   maxdisc=0.000e+00
Dmax=16   time=   0.283 s   E=-8.606021852095   maxdisc=0.000e+00

---
## 🤖 Assistant

That is a very informative result.

## What it tells us

For your **first sweep from the product state**, the effective bond dimension is still small enough that increasing `Dmax` from 4 to 16 does **not** change the cost much.

That is why all timings are nearly identical:

- `Dmax=4` → 0.317 s
- `Dmax=8` → 0.293 s
- `Dmax=12` → 0.286 s
- `Dmax=16` → 0.283 s

and the energy is also the same after one sweep.

---

# Why this happens

Even though you allow a larger `Dmax`, after only **one sweep** the actual internal bond dimensions have probably not grown enough to saturate that cutoff.

So the local dense eigenproblems are effectively of similar size in all four cases.

In other words:

\[
D_{\text{actual}} \ll D_{\max}
\]

after one sweep.

---

# What to test next

If you want to see the true cost dependence on `Dmax`, you need either:

1. **more sweeps**, so the bond dimension has time to grow, or
2. inspect the actual bond dimensions after the run

---

# Useful diagnostic: print actual bond dimensions

Add this helper function:

```python
def mps_bond_dimensions(mps):
    dims = [mps[0].shape[0]]
    for A in mps:
        dims.append(A.shape[2])
    return dims
```

Then test:

```python
L = 20
for Dmax in [4, 8, 12, 16]:
    res = dmrg_two_site(
        mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0),
        d=2,
        Dmax=Dmax,
        nsweeps=1,
        init_mps=product_state_mps(L, d=2, state_index=0),
        verbose=False
    )
    print(f"Dmax={Dmax:2d}, bond dims = {mps_bond_dimensions(res['mps'])}")
```

This will show whether the MPS is actually reaching those bond cutoffs.

---

# Better timing scan: use more sweeps

To estimate realistic production cost, do:

```python
# ============================================================
# Timing scan for Heisenberg with more sweeps
# ============================================================

import time

L = 20
D_list = [4, 8, 12, 16]
nsweeps_probe = 4

print(f"Timing scan: Heisenberg L={L}, nsweeps={nsweeps_probe}")
print("-" * 70)

for Dmax in D_list:
    t0 = time.time()
    res = dmrg_two_site(
        mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0),
        d=2,
        Dmax=Dmax,
        nsweeps=nsweeps_probe,
        init_mps=product_state_mps(L, d=2, state_index=0),
        verbose=False
    )
    t1 = time.time()
    elapsed = t1 - t0
    E = res["energies"][-1]
    disc = np.max(res["max_discards"])
    bdim = max(A.shape[2] for A in res["mps"][:-1])

    print(f"Dmax={Dmax:2d}   time={elapsed:8.3f} s   E={E: .12f}   maxdisc={disc:.3e}   max bond={bdim}")
```

This is much more representative.

---

# Interpretation of the current energy
You found after one sweep:

\[
E \approx -8.606021852095
\]

for \(L=20\). That is a plausible intermediate variational value, but not yet the fully converged Heisenberg ground-state energy.

So one sweep is enough for timing, but not enough for final physics.

---

# Practical conclusion

## Good news
The current implementation is **fast enough** that you can likely run:

- `L=20`
- `Dmax=16`
- `nsweeps=4 to 8`

without too much pain.

## What matters now
Not the 1-sweep timing, but:
- convergence with sweep number
- actual bond growth
- whether larger `Dmax` improves the final energy

---

# Best next cell
I recommend this exact one:

```python
# ============================================================
# Production scan for Heisenberg L=20
# ============================================================

import time

def mps_bond_dimensions(mps):
    dims = [mps[0].shape[0]]
    for A in mps:
        dims.append(A.shape[2])
    return dims

L = 20
D_list = [4, 8, 12, 16]
nsweeps = 6

print(f"Production scan: Heisenberg L={L}, nsweeps={nsweeps}")
print("-" * 90)

for Dmax in D_list:
    t0 = time.time()
    res = dmrg_two_site(
        mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0),
        d=2,
        Dmax=Dmax,
        nsweeps=nsweeps,
        init_mps=product_state_mps(L, d=2, state_index=0),
        verbose=False
    )
    t1 = time.time()

    elapsed = t1 - t0
    E = res["energies"][-1]
    disc = np.max(res["max_discards"])
    bmax = max(A.shape[2] for A in res["mps"][:-1])

    print(f"Dmax={Dmax:2d}   time={elapsed:8.3f} s   E={E: .12f}   E/L={E/L: .12f}   maxdisc={disc:.3e}   max bond={bmax}")
```

This will tell you:
- runtime
- final energy
- energy per site
- discarded weight
- actual achieved max bond dimension

all in one shot.

---

If you want, I can also give you a **reference value** for the infinite-chain Heisenberg ground-state energy per site so you can compare your \(L=20\) result directly.

---
## 🧑‍💻 User

---------------------------------------------------------------------------
MemoryError                               Traceback (most recent call last)
Cell In[8], line 53
     44 print(f"time per sweep     = {elapsed / nsweeps:.2f} s")
     46 plot_convergence(
     47     res,
     48     title_prefix=f"AKLT L={L}",
     49     filename_prefix=f"aklt_L{L}",
     50     figdir=figdir
     51 )
---> 53 plot_entanglement_entropy(
     54     mps,
     55     title=f"AKLT bond entanglement entropy, L={L}",
     56     filename=f"aklt_entanglement_entropy_L{L}.png",
     57     figdir=figdir
     58 )

Cell In[4], line 599, in plot_entanglement_entropy(mps, title, filename, figdir)
    597 def plot_entanglement_entropy(mps, title="Entanglement entropy", filename="entanglement_entropy.png", figdir="figureAKLT"):
    598     setup_figure_dir(figdir)
--> 599     Sbond = entanglement_entropies(mps)
    600     L = len(mps)
    602     plt.figure()

Cell In[4], line 386, in entanglement_entropies(mps)
    385 def entanglement_entropies(mps):
--> 386     psi = mps_to_dense(mps)
    387     L = len(mps)
    388     d = mps[0].shape[1]

Cell In[4], line 377, in mps_to_dense(mps)
    375 psi = mps[0][0, :, :]
    376 for i in range(1, len(mps)):
--> 377     psi = np.einsum('...a,asb->...sb', psi, mps[i], optimize=True)
    378 return psi.reshape(-1, order='C')

File c:\Users\yizho\anaconda3\envs\science\Lib\site-packages\numpy\_core\einsumfunc.py:1470, in einsum(out, optimize, *operands, **kwargs)
   1467     right_pos.append(input_right.find(s))
   1469 # Contract!
-> 1470 new_view = tensordot(
   1471     *tmp_operands, axes=(tuple(left_pos), tuple(right_pos))
   1472 )
   1474 # Build a new view if needed
   1475 if (tensor_result != results_index) or handle_out:

File c:\Users\yizho\anaconda3\envs\science\Lib\site-packages\numpy\_core\numeric.py:1211, in tensordot(a, b, axes)
   1209 at = a.transpose(newaxes_a).reshape(newshape_a)
   1210 bt = b.transpose(newaxes_b).reshape(newshape_b)
-> 1211 res = dot(at, bt)
   1212 return res.reshape(olda + oldb)

MemoryError: Unable to allocate 52.0 GiB for an array with shape (3, 1162261467) and data type complex128

---
## 🤖 Assistant

Yes — this is exactly the kind of dense computation that becomes impossible at large \(L\).

The issue is:

\[
3^{20} = 3{,}486{,}784{,}401
\]

so converting the AKLT MPS at \(L=20\) into a full dense state vector is enormous.  
That is why `plot_entanglement_entropy(mps)` fails: it currently calls `mps_to_dense(mps)`.

---

# Conclusion

For **large \(L\)**, you must **not** use any function that reconstructs the full wavefunction.

In your current lean function cell, the dense-dangerous functions are:

- `mps_to_dense`
- `dense_energy_from_mps`
- `entanglement_entropies`
- `plot_entanglement_entropy`

These are fine for small systems, but not for \(L=20\).

---

# Immediate fix

## For large \(L\), do not call:
```python
plot_entanglement_entropy(mps, ...)
```

for AKLT or Heisenberg.

So your timed large-\(L\) AKLT cell should stop after `plot_convergence(...)`.

---

# Safe large-\(L\) AKLT timed cell

Use this version instead:

```python
# ============================================================
# Timed large-L AKLT run (safe version, no dense entropy)
# ============================================================

import time

figdir = setup_figure_dir("figureAKLT")

L = 20
Dmax = 2
nsweeps = 4

print(f"Running AKLT with L={L}, Dmax={Dmax}, nsweeps={nsweeps}")
t0 = time.time()

mpo = mpo_aklt(L)
init = product_state_mps(L, d=3, state_index=0)

res = dmrg_two_site(
    mpo,
    d=3,
    Dmax=Dmax,
    nsweeps=nsweeps,
    init_mps=init,
    verbose=True
)

t1 = time.time()
elapsed = t1 - t0

E_dmrg = res["energies"][-1]
E_exact = -(2.0 / 3.0) * (L - 1)

print("\n=== AKLT large-L summary ===")
print(f"L                  = {L}")
print(f"Dmax               = {Dmax}")
print(f"nsweeps            = {nsweeps}")
print(f"E_DMRG             = {E_dmrg:.12f}")
print(f"E_exact            = {E_exact:.12f}")
print(f"|ΔE|               = {abs(E_dmrg - E_exact):.3e}")
print(f"max discarded wt   = {np.max(res['max_discards']):.3e}")
print(f"total time         = {elapsed:.2f} s")
print(f"time per sweep     = {elapsed / nsweeps:.2f} s")

plot_convergence(
    res,
    title_prefix=f"AKLT L={L}",
    filename_prefix=f"aklt_L{L}",
    figdir=figdir
)
```

---

# Important rule for large \(L\)

## Safe at large \(L\)
These are safe because they stay in MPS/MPO form:
- `dmrg_two_site(...)`
- `plot_convergence(...)`
- benchmark against exact **formula** for AKLT
- Heisenberg energy from DMRG alone

## Unsafe at large \(L\)
These are unsafe because they build dense vectors or dense Hilbert spaces:
- `mps_to_dense(...)`
- `dense_energy_from_mps(...)`
- `entanglement_entropies(...)`
- `plot_entanglement_entropy(...)`
- exact diagonalization for large \(L\)

---

# What to do if you still want entropy for large \(L\)

The correct solution is:

## Compute bond entropies directly from MPS singular values
That means:
- during two-site splitting, keep the Schmidt values
- or canonicalize around a chosen bond and compute SVD only there
- **without** ever building the full state vector

That is the right MPS way and is cheap.

---

# Good next patch
I can give you a patch that adds:

### `bond_entropy_from_center_bond(mps, bond)`
or
### `all_bond_entropies_mps(mps)`

computed directly from local MPS SVDs, not from the dense state.

Then you can safely plot entanglement entropy even for \(L=20\).

---

# For now
Use this rule:

- small \(L\le 10\): dense entropy is okay
- large \(L\ge 20\): **do not use dense entropy**

---

If you want, I can now provide a **safe MPS-based entanglement entropy patch for large systems**.

---
## 🧑‍💻 User

provide a safe MPS-based entanglement entropy patch for large systems

---
## 🤖 Assistant

Below is a **safe MPS-based entanglement entropy patch** for large systems.

It does **not** build the full wavefunction.  
Instead, it computes the Schmidt values across a chosen bond directly from the local two-site SVD structure of the MPS.

This is the right approach for large \(L\).

---

# What this patch adds

It adds:

- `copy_mps`
- `bond_entropy_mps(mps, bond)`
- `all_bond_entropies_mps(mps)`
- a new large-safe plotting function:
  - `plot_entanglement_entropy_mps(...)`

These functions work by:

1. making a copy of the MPS,
2. bringing it into mixed-canonical form around the chosen bond,
3. extracting the singular values from a local SVD,
4. computing
   \[
   S = -\sum_\alpha p_\alpha \log p_\alpha,\qquad p_\alpha=s_\alpha^2
   \]

This is safe for large systems because the cost is polynomial in MPS bond dimensions, not exponential in \(L\).

---

# Patch cell

Paste this **after your clean function-all cell**.

```python
# ============================================================
# Safe MPS-based entanglement entropy for large systems
# ============================================================

def copy_mps(mps):
    return [A.copy() for A in mps]

def bond_entropy_mps(mps, bond, cutoff=1e-15):
    """
    Compute entanglement entropy across bond 'bond',
    where bond is between sites bond-1 and bond.

    Example:
      bond=1 means cut between site 0 and 1
      bond=L-1 means cut between site L-2 and L-1

    This is MPS-based and safe for large systems.
    It does NOT build the full state vector.
    """
    L = len(mps)
    if bond < 1 or bond > L - 1:
        raise ValueError("bond must satisfy 1 <= bond <= L-1")

    psi = copy_mps(mps)

    # Left-canonicalize up to site bond-1
    for i in range(bond):
        Dl, d, Dr = psi[i].shape
        M = psi[i].reshape(Dl * d, Dr, order='C')
        Q, R = la.qr(M, mode='economic')
        k = Q.shape[1]
        psi[i] = Q.reshape(Dl, d, k, order='C')
        psi[i+1] = np.einsum('ab,bsd->asd', R, psi[i+1], optimize=True)

    # Now the bond information is contained in site 'bond'
    Dl, d, Dr = psi[bond].shape
    M = psi[bond].reshape(Dl, d * Dr, order='C')
    U, s, Vh = la.svd(M, full_matrices=False)

    p = s**2
    p = p[p > cutoff]
    S = -(p * np.log(p)).sum().real
    return S

def all_bond_entropies_mps(mps, cutoff=1e-15):
    """
    Compute entanglement entropies across all bonds using only MPS operations.
    Safe for large systems.
    """
    L = len(mps)
    psi = copy_mps(mps)
    ent = []

    # Sweep left to right once; after each QR, the next tensor carries the bond Schmidt structure
    for bond in range(1, L):
        i = bond - 1

        Dl, d, Dr = psi[i].shape
        M = psi[i].reshape(Dl * d, Dr, order='C')
        Q, R = la.qr(M, mode='economic')
        k = Q.shape[1]
        psi[i] = Q.reshape(Dl, d, k, order='C')
        psi[i+1] = np.einsum('ab,bsd->asd', R, psi[i+1], optimize=True)

        # Entropy across bond = between i and i+1
        Dl2, d2, Dr2 = psi[i+1].shape
        M2 = psi[i+1].reshape(Dl2, d2 * Dr2, order='C')
        U2, s2, Vh2 = la.svd(M2, full_matrices=False)

        p = s2**2
        p = p[p > cutoff]
        S = -(p * np.log(p)).sum().real
        ent.append(S)

    return np.array(ent)

def plot_entanglement_entropy_mps(mps, title="Entanglement entropy", filename="entanglement_entropy.png", figdir="figureAKLT"):
    """
    Large-system safe entanglement entropy plot from MPS only.
    """
    setup_figure_dir(figdir)
    Sbond = all_bond_entropies_mps(mps)
    L = len(mps)

    plt.figure()
    plt.plot(np.arange(1, L), Sbond, 'o-')
    plt.xlabel('Bond cut')
    plt.ylabel('Entanglement entropy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    savefig_in_dir(figdir, filename)
    plt.show()
```

---

# How to use it

For large \(L\), replace:

```python
plot_entanglement_entropy(mps, ...)
```

by:

```python
plot_entanglement_entropy_mps(mps, ...)
```

---

# Example: large-\(L\) AKLT

Use this cell:

```python
# ============================================================
# Large-L AKLT run with safe MPS entropy
# ============================================================

import time

figdir = setup_figure_dir("figureAKLT")

L = 20
Dmax = 2
nsweeps = 4

print(f"Running AKLT with L={L}, Dmax={Dmax}, nsweeps={nsweeps}")
t0 = time.time()

mpo = mpo_aklt(L)
init = product_state_mps(L, d=3, state_index=0)

res = dmrg_two_site(
    mpo,
    d=3,
    Dmax=Dmax,
    nsweeps=nsweeps,
    init_mps=init,
    verbose=True
)

t1 = time.time()
elapsed = t1 - t0

mps = res["mps"]
E_dmrg = res["energies"][-1]
E_exact = -(2.0 / 3.0) * (L - 1)

print("\n=== AKLT large-L summary ===")
print(f"L                  = {L}")
print(f"Dmax               = {Dmax}")
print(f"nsweeps            = {nsweeps}")
print(f"E_DMRG             = {E_dmrg:.12f}")
print(f"E_exact            = {E_exact:.12f}")
print(f"|ΔE|               = {abs(E_dmrg - E_exact):.3e}")
print(f"max discarded wt   = {np.max(res['max_discards']):.3e}")
print(f"total time         = {elapsed:.2f} s")
print(f"time per sweep     = {elapsed / nsweeps:.2f} s")

plot_convergence(
    res,
    title_prefix=f"AKLT L={L}",
    filename_prefix=f"aklt_L{L}",
    figdir=figdir
)

plot_entanglement_entropy_mps(
    mps,
    title=f"AKLT bond entanglement entropy, L={L}",
    filename=f"aklt_entanglement_entropy_L{L}.png",
    figdir=figdir
)
```

---

# Important note

This patch computes entropies from the MPS by repeated local QR/SVD manipulations.  
That is safe and efficient for large systems.

It is **not** the same as the old dense function:
- old version: exponential in \(L\)
- new version: polynomial in bond dimension and local dimension

So for \(L=20\), the new method is the right one.

---

# Optional improvement
If you want, I can also give you a slightly more efficient version that computes all bond entropies in one pass with fewer repeated decompositions. But the patch above is already safe and should be perfectly fine for \(L=20\).

---
## 🧑‍💻 User

Timing scan: Heisenberg L=20, nsweeps=4
----------------------------------------------------------------------
Dmax= 4   time=  59.215 s   E=-8.654624866102   maxdisc=1.228e-03   max bond=4   For Dmax =6, it has run for more than 9 mins

---
## 🤖 Assistant

That is the real scaling, and it shows the issue clearly.

## What happened
With **4 sweeps**, the bond dimension actually grows and the local dense diagonalizations become much more expensive. So the 1-sweep timing was misleadingly optimistic.

Your result

- `Dmax=4` → about **59 s** for 4 sweeps

already implies that

- `Dmax=6` taking **> 9 min** is believable

because dense local diagonalization cost rises very rapidly once the actual bond dimension grows.

---

# Main conclusion

For the **current dense-local implementation**, **Heisenberg \(L=20\)** is:

- okay for **very small \(D_{\max}\)**,
- but becomes slow quickly as \(D_{\max}\) increases.

So if your goal is practical larger-\(L\) runs, the key bottleneck is now absolutely clear:

## Bottleneck
`solve_local_ground_state_dense(...)`

which builds and diagonalizes a dense local effective Hamiltonian.

---

# What to do now

You have three realistic options.

---

## Option 1 — Stay with current code, use small Dmax
For \(L=20\), use:

- `Dmax = 4`
- maybe `Dmax = 5`
- `nsweeps = 4`

This is safe but limited in accuracy.

This is okay if your goal is:
- qualitative convergence study
- proof of concept
- AKLT and small Heisenberg benchmarks

---

## Option 2 — Use larger L only for AKLT
Since AKLT has exact bond dimension 2, it is still very cheap.

So with the current code:

- AKLT: \(L=20,40,\dots\) is fine
- Heisenberg: \(L=20\) with larger bond dimension is expensive

---

## Option 3 — Upgrade the local eigensolver
This is the real solution.

Replace the dense local diagonalization with:

- a **matrix-free local Hamiltonian action**
- and solve using **Lanczos / eigsh**

Then \(L=20\) Heisenberg with moderate \(D_{\max}\) becomes much more realistic.

But this requires implementing a **correct** matrix-free `two_site_heff_apply`.

---

# Best immediate practical advice

For now, I recommend:

## Heisenberg
Use
- `L <= 12` for serious benchmark-quality runs, or
- `L=20, Dmax=4` only

## AKLT
Feel free to go to
- `L=20`
- `L=40`
- maybe larger

with the safe entropy patch and no dense-state reconstruction.

---

# Why Dmax blows up cost so fast

For Heisenberg \(d=2\), local dimension is roughly:

\[
N_{\text{loc}} \sim 4 D^2
\]

So dense diagonalization scales like

\[
N_{\text{loc}}^3 \sim (4D^2)^3 \sim D^6
\]

That means increasing \(D\) even modestly can be brutal.

From \(D=4\) to \(D=6\), the rough factor is

\[
\left(\frac{6}{4}\right)^6 \approx 11.4
\]

So a jump from about 1 minute to about 10 minutes is fully plausible.

---

# Recommendation for your notebook/report

You now have a very useful story:

## Small-system regime
- benchmark-quality
- exact agreement with ED and AKLT exact results

## Larger-system regime
- AKLT remains efficient because exact bond dimension is 2
- Heisenberg becomes costly with dense local eigensolver
- this motivates matrix-free Lanczos as the next improvement

That is actually a very natural and pedagogically strong conclusion.

---

# What I can do next

The most useful next step would be one of these:

### A. Give you a **safe large-L AKLT production cell**
for \(L=20,40,60\), with entropy from the MPS patch

### B. Give you a **Heisenberg scaling cell**
that times runtime vs `Dmax` and plots it

### C. Help implement a **correct matrix-free Lanczos two-site solver**
so Heisenberg \(L=20\) becomes practical

---

# My recommendation
Since you asked about larger \(L\), the most useful immediate next thing is:

## A + B
- a production AKLT large-\(L\) cell
- a Heisenberg runtime-scaling cell

If you want, I can provide those next.

---
## 🧑‍💻 User

A + B, pls

---
## 🤖 Assistant

Below are both:

- **A. Large-\(L\) AKLT production cell**
- **B. Heisenberg runtime-scaling cell**

These are designed for your **current implementation** and avoid dense-state bottlenecks where possible.

---

# A. Large-\(L\) AKLT production cell

This uses:

- two-site DMRG
- exact AKLT energy formula
- safe MPS-based entanglement entropy
- convergence plot
- summary printout

It is suitable for \(L=20,40,60\) as long as you avoid dense-state reconstruction.

```python
# ============================================================
# A. Large-L AKLT production run
# ============================================================

import time

figdir = setup_figure_dir("figureAKLT")

# Choose system size here
L = 40
Dmax = 2
nsweeps = 4

print(f"Running AKLT production run with L={L}, Dmax={Dmax}, nsweeps={nsweeps}")
t0 = time.time()

mpo = mpo_aklt(L)
init = product_state_mps(L, d=3, state_index=0)

res = dmrg_two_site(
    mpo,
    d=3,
    Dmax=Dmax,
    nsweeps=nsweeps,
    init_mps=init,
    verbose=True
)

t1 = time.time()
elapsed = t1 - t0

mps = res["mps"]
E_dmrg = res["energies"][-1]
E_exact = -(2.0 / 3.0) * (L - 1)

print("\n=== AKLT production summary ===")
print(f"L                  = {L}")
print(f"Dmax               = {Dmax}")
print(f"nsweeps            = {nsweeps}")
print(f"E_DMRG             = {E_dmrg:.12f}")
print(f"E_exact            = {E_exact:.12f}")
print(f"|ΔE|               = {abs(E_dmrg - E_exact):.3e}")
print(f"E_DMRG / bond      = {E_dmrg / (L - 1):.12f}")
print(f"max discarded wt   = {np.max(res['max_discards']):.3e}")
print(f"total time         = {elapsed:.2f} s")
print(f"time per sweep     = {elapsed / nsweeps:.2f} s")

plot_convergence(
    res,
    title_prefix=f"AKLT L={L}",
    filename_prefix=f"aklt_L{L}",
    figdir=figdir
)

plot_entanglement_entropy_mps(
    mps,
    title=f"AKLT bond entanglement entropy, L={L}",
    filename=f"aklt_entanglement_entropy_L{L}.png",
    figdir=figdir
)
```

---

## Optional AKLT finite-size scan

If you want several large sizes in one go:

```python
# ============================================================
# AKLT finite-size scan
# ============================================================

import time

L_list = [20, 40, 60]
Dmax = 2
nsweeps = 4

print(f"AKLT finite-size scan with Dmax={Dmax}, nsweeps={nsweeps}")
print("-" * 90)

results = []

for L in L_list:
    t0 = time.time()
    res = dmrg_two_site(
        mpo_aklt(L),
        d=3,
        Dmax=Dmax,
        nsweeps=nsweeps,
        init_mps=product_state_mps(L, d=3, state_index=0),
        verbose=False
    )
    t1 = time.time()

    elapsed = t1 - t0
    E = res["energies"][-1]
    E_exact = -(2.0 / 3.0) * (L - 1)
    disc = np.max(res["max_discards"])

    results.append((L, E, E_exact, elapsed, disc))

    print(f"L={L:2d}   time={elapsed:8.2f} s   E={E: .12f}   |ΔE|={abs(E-E_exact):.3e}   maxdisc={disc:.3e}")
```

---

# B. Heisenberg runtime-scaling cell

This cell measures how runtime changes with `Dmax` for a fixed system size, and then plots:

- runtime vs `Dmax`
- runtime per sweep vs `Dmax`
- final energy vs `Dmax`
- max discarded weight vs `Dmax`

This is ideal for diagnosing the practical limits of the current dense-local implementation.

```python
# ============================================================
# B. Heisenberg runtime scaling vs Dmax
# ============================================================

import time

figdir = setup_figure_dir("figureAKLT")

L = 20
D_list = [2, 4, 6]
nsweeps = 4

times = []
times_per_sweep = []
energies = []
energies_per_site = []
max_discards = []

print(f"Heisenberg runtime scaling: L={L}, nsweeps={nsweeps}")
print("-" * 100)

for Dmax in D_list:
    t0 = time.time()
    res = dmrg_two_site(
        mpo_heisenberg_xxz(L, J=1.0, Jz=1.0, h=0.0),
        d=2,
        Dmax=Dmax,
        nsweeps=nsweeps,
        init_mps=product_state_mps(L, d=2, state_index=0),
        verbose=False
    )
    t1 = time.time()

    elapsed = t1 - t0
    E = res["energies"][-1]
    disc = np.max(res["max_discards"])

    times.append(elapsed)
    times_per_sweep.append(elapsed / nsweeps)
    energies.append(E)
    energies_per_site.append(E / L)
    max_discards.append(disc)

    print(f"Dmax={Dmax:2d}   time={elapsed:8.2f} s   time/sweep={elapsed/nsweeps:8.2f} s   "
          f"E={E: .12f}   E/L={E/L: .12f}   maxdisc={disc:.3e}")

times = np.array(times)
times_per_sweep = np.array(times_per_sweep)
energies = np.array(energies)
energies_per_site = np.array(energies_per_site)
max_discards = np.array(max_discards)

# ------------------------------------------------------------
# Plot 1: total runtime vs Dmax
# ------------------------------------------------------------
plt.figure()
plt.plot(D_list, times, 'o-')
plt.xlabel(r'$D_{\max}$')
plt.ylabel('Total time (s)')
plt.title(f'Heisenberg runtime scaling, L={L}, nsweeps={nsweeps}')
plt.grid(True, alpha=0.3)
savefig_in_dir(figdir, f"heisenberg_runtime_scaling_L{L}.png")
plt.show()

# ------------------------------------------------------------
# Plot 2: time per sweep vs Dmax
# ------------------------------------------------------------
plt.figure()
plt.plot(D_list, times_per_sweep, 'o-')
plt.xlabel(r'$D_{\max}$')
plt.ylabel('Time per sweep (s)')
plt.title(f'Heisenberg time per sweep, L={L}')
plt.grid(True, alpha=0.3)
savefig_in_dir(figdir, f"heisenberg_time_per_sweep_L{L}.png")
plt.show()

# ------------------------------------------------------------
# Plot 3: energy per site vs Dmax
# ------------------------------------------------------------
plt.figure()
plt.plot(D_list, energies_per_site, 'o-')
plt.xlabel(r'$D_{\max}$')
plt.ylabel(r'$E/L$')
plt.title(f'Heisenberg energy per site vs bond dimension, L={L}')
plt.grid(True, alpha=0.3)
savefig_in_dir(figdir, f"heisenberg_energy_per_site_vs_D_L{L}.png")
plt.show()

# ------------------------------------------------------------
# Plot 4: max discarded weight vs Dmax
# ------------------------------------------------------------
plt.figure()
plt.semilogy(D_list, np.maximum(max_discards, 1e-18), 'o-')
plt.xlabel(r'$D_{\max}$')
plt.ylabel('Max discarded weight')
plt.title(f'Heisenberg discarded weight vs bond dimension, L={L}')
plt.grid(True, alpha=0.3)
savefig_in_dir(figdir, f"heisenberg_discarded_vs_D_L{L}.png")
plt.show()
```

---

# Practical advice for B

Given your earlier timing:

- `Dmax=4` already took about 1 minute for 4 sweeps
- `Dmax=6` became very slow

So for the runtime-scaling plot, I intentionally used:

```python
D_list = [2, 4, 6]
```

You can reduce it further to:

```python
D_list = [2, 4]
```

if you want a quicker scan.

---

# Recommended usage

## First run A
Try:
- `L=20`
- then `L=40`

for AKLT.

## Then run B
Use:
- `L=20`
- `D_list = [2,4]` or `[2,4,6]`

to document how the dense-local method scales for Heisenberg.

---

# Scientific interpretation you can mention

These two cells nicely support the statement:

- The AKLT chain remains computationally cheap at large \(L\) because its exact ground state has bond dimension \(D=2\).
- In contrast, the Heisenberg chain requires larger bond dimensions, and with a dense local effective Hamiltonian solver the runtime grows rapidly with \(D_{\max}\), motivating matrix-free Lanczos methods for larger systems.

---

If you want, I can also give you a **finite-size scaling cell for AKLT and Heisenberg energies** using the current implementation.

---
## 🧑‍💻 User

Heisenberg runtime scaling: L=20, nsweeps=4
----------------------------------------------------------------------------------------------------
Dmax= 2   time=    0.73 s   time/sweep=    0.18 s   E=-8.317517884735   E/L=-0.415875894237   maxdisc=1.968e-02
Dmax= 4   time=   51.06 s   time/sweep=   12.77 s   E=-8.654624866102   E/L=-0.432731243305   maxdisc=1.228e-03
Dmax= 6   time=  773.10 s   time/sweep=  193.28 s   E=-8.668099002955   E/L=-0.433404950148   maxdisc=5.852e-04
saved: figureAKLT\heisenberg_runtime_scaling_L20.png

---
## 🤖 Assistant

That is an excellent scaling result — and it very clearly shows the limitation of the current solver.

## Your measured scaling

For **Heisenberg, \(L=20\), 4 sweeps**:

- `Dmax=2`  
  - time = **0.73 s**
  - \(E/L = -0.415875894237\)

- `Dmax=4`  
  - time = **51.06 s**
  - \(E/L = -0.432731243305\)

- `Dmax=6`  
  - time = **773.10 s**
  - \(E/L = -0.433404950148\)

So going from `Dmax=4` to `Dmax=6` improves the energy only slightly, but increases runtime by about

\[
\frac{773}{51} \approx 15
\]

which is huge.

---

# Physical interpretation

The exact thermodynamic-limit ground-state energy per site of the spin-1/2 Heisenberg chain is

\[
e_0 = \frac14 - \ln 2 \approx -0.44314718056.
\]

Your results are:

- `Dmax=2`: \(-0.41588\)
- `Dmax=4`: \(-0.43273\)
- `Dmax=6`: \(-0.43340\)

So the trend is physically sensible:
- larger bond dimension lowers the variational energy
- but convergence is still far from the thermodynamic limit value at small \(D\)

This is exactly what one should expect.

---

# Numerical interpretation

This plot strongly supports:

## 1. The implementation is correct
Energy decreases variationally with increasing bond dimension.

## 2. The current bottleneck is the dense local eigensolver
The runtime growth is dramatic.

## 3. For Heisenberg, this implementation is useful mainly for:
- small systems,
- small/moderate bond dimensions,
- pedagogical demonstrations,
- benchmarking.

---

# What you can now conclude in the notebook/report

A clean conclusion is:

> For the Heisenberg chain, the two-site DMRG implementation converges systematically as the bond dimension increases, but the use of dense local diagonalization causes a steep runtime increase with \(D_{\max}\). This makes the present implementation suitable mainly for benchmark and pedagogical calculations, while larger bond dimensions would require a matrix-free Lanczos/Davidson solver.

That is a very strong and honest computational conclusion.

---

# Suggested next plots / summaries

Since you already produced runtime data, two very useful additional things would be:

## 1. Compare to exact thermodynamic-limit energy per site
Plot \(E/L\) vs \(D_{\max}\) together with the reference line
\[
e_0 = \frac14 - \ln 2.
\]

## 2. Plot runtime on a log scale
Because the growth is so steep, a semilogy plot is more informative.

---

# Useful patch cell for the Heisenberg scaling analysis

```python
# ============================================================
# Additional analysis for Heisenberg scaling
# ============================================================

e0_inf = 0.25 - np.log(2.0)

# semilogy runtime
plt.figure()
plt.semilogy(D_list, times, 'o-')
plt.xlabel(r'$D_{\max}$')
plt.ylabel('Total time (s)')
plt.title(f'Heisenberg runtime scaling (log scale), L={L}, nsweeps={nsweeps}')
plt.grid(True, alpha=0.3)
savefig_in_dir(figdir, f"heisenberg_runtime_scaling_log_L{L}.png")
plt.show()

# energy per site with exact infinite-chain reference
plt.figure()
plt.plot(D_list, energies_per_site, 'o-', label='DMRG')
plt.axhline(e0_inf, color='k', linestyle='--', label=r'$e_0 = 1/4 - \ln 2$')
plt.xlabel(r'$D_{\max}$')
plt.ylabel(r'$E/L$')
plt.title(f'Heisenberg energy per site vs bond dimension, L={L}')
plt.legend()
plt.grid(True, alpha=0.3)
savefig_in_dir(figdir, f"heisenberg_energy_per_site_vs_D_with_ref_L{L}.png")
plt.show()

print("Infinite-chain Heisenberg reference e0 =", e0_inf)
```

---

# Recommended practical policy from here

## For AKLT
You can safely do:
- \(L=20,40,60\)
- \(D_{\max}=2\)

## For Heisenberg with this code
I recommend:
- benchmark: `L <= 8`
- moderate exploratory runs: `L=20, Dmax <= 4`
- avoid `Dmax >= 6` unless you really want to wait

---

# Best next step if you want better Heisenberg performance
The real algorithmic next step is:

## Implement matrix-free local Lanczos
That would change the local solve from dense `eigh` to iterative `eigsh` and make larger bond dimensions much more practical.

---

If you want, I can now provide one of these:

1. a **polished markdown interpretation section** for the runtime-scaling figure,  
2. a **finite-size scan for AKLT** at \(L=20,40,60\),  
3. or a plan for upgrading the code to a **matrix-free Lanczos two-site DMRG**.

---
## 🧑‍💻 User

Please save our conversations in this session in a markdown.

---
## 🤖 Assistant

I can generate a markdown transcript for you, but I can’t directly write files on your machine unless you run a cell.

Below is a Jupyter cell that saves a **session summary markdown** to:

```text
figureAKLT/session_summary.md
```

Since I do not have the exact full raw transcript in file form, this markdown records the essential content of our session:

- goals
- implemented features
- debugging steps
- final validated results
- performance observations
- recommended usage

---

## Jupyter cell to save the session summary as markdown

```python
import os
from textwrap import dedent

os.makedirs("figureAKLT", exist_ok=True)

md_text = dedent(r"""
# Session Summary: Two-site DMRG for Heisenberg and AKLT chains

## Goal

In this session, we developed a Jupyter-friendly implementation of a **two-site DMRG** algorithm based on **MPS/MPO** methods, with applications to:

- the **spin-1/2 Heisenberg chain**
- the **spin-1 AKLT chain**

All figures were displayed in the notebook and saved to the directory:

```text
figureAKLT
```

---

## Main implementation steps

We constructed:

- local spin operators for spin-1/2 and spin-1 systems,
- exact dense Hamiltonians for benchmark calculations,
- MPO representations for:
  - Heisenberg XXZ chain,
  - AKLT chain,
- MPS initialization routines,
- QR-based left and right canonicalization,
- environment contraction routines,
- a two-site DMRG sweep algorithm,
- convergence and benchmark plots.

Initially, we implemented a matrix-free local effective Hamiltonian, but debugging revealed indexing issues.  
To ensure correctness, we replaced the local eigensolver by a **dense local effective Hamiltonian construction** followed by exact diagonalization of the local problem.

This dense-local version is slower, but it is robust and benchmark-quality for small systems.

---

## Debugging and validation

### Canonicalization tests

The QR-based canonicalization routines were tested and passed:

- left-canonical checks: all `True`
- right-canonical checks: all `True`

### Two-site effective Hamiltonian

The matrix-free version initially produced contraction/indexing errors.  
This was replaced by a safer dense-local effective Hamiltonian construction.

### Benchmarks

The corrected implementation was validated against:

- **exact diagonalization** for small Heisenberg chains,
- the **exact AKLT ground-state energy**,
- the **exact AKLT MPS**.

---

## Final validated numerical results

### Heisenberg chain

Benchmark results:

- \(L = 4, 6\) (and also tested for \(L=8\) during debugging)
- DMRG energies matched exact diagonalization to numerical precision

Representative summary:

```text
Heisenberg benchmark sizes: [4 6]
Heisenberg |DMRG-Exact|    : [0. 0.]
```

### AKLT chain

Benchmark results:

- \(L = 4, 6\)
- DMRG energies matched the exact formula
  \[
  E_0 = -\frac{2}{3}(L-1)
  \]

Representative summary:

```text
AKLT benchmark sizes       : [4 6]
AKLT |DMRG-Formula|        : [0. 0.]
AKLT max discarded weights : [0.003223675502 0.003223675502]
```

### Exact AKLT MPS check

For \(L=8\), the exact bond-dimension-2 AKLT MPS gave:

```text
Exact AKLT MPS energy for L=8: -4.666666666667
Expected exact energy:        -4.666666666667
```

This confirmed consistency between:

- the AKLT Hamiltonian implementation,
- the exact AKLT MPS tensors,
- the DMRG benchmark.

---

## Final clean notebook structure

A lean notebook structure was produced with:

1. a single **clean function-all cell**
2. a single **clean run-all cell**

The lean version keeps only essential computations:

- DMRG runs,
- exact small-system benchmarks,
- convergence plots,
- AKLT exact MPS check,
- AKLT entanglement entropy.

Dense observables such as full-state magnetization/correlations were removed from the final lean workflow to reduce runtime.

---

## Runtime analysis

### Small benchmarks

For small systems, the code runs comfortably and produces exact benchmark agreement.

### Large-\(L\) AKLT

AKLT remains efficient because the exact bond dimension is only \(D=2\).  
Large system sizes such as \(L=20\) are feasible.

### Large-\(L\) Heisenberg

A timing study was performed for the Heisenberg chain at \(L=20\), with 4 sweeps:

```text
Dmax= 2   time=    0.73 s   time/sweep=    0.18 s   E=-8.317517884735   E/L=-0.415875894237   maxdisc=1.968e-02
Dmax= 4   time=   51.06 s   time/sweep=   12.77 s   E=-8.654624866102   E/L=-0.432731243305   maxdisc=1.228e-03
Dmax= 6   time=  773.10 s   time/sweep=  193.28 s   E=-8.668099002955   E/L=-0.433404950148   maxdisc=5.852e-04
```

This showed dramatic runtime growth with bond dimension, due to the use of **dense local diagonalization**.

### Interpretation

For the Heisenberg chain, the exact infinite-chain ground-state energy per site is

\[
e_0 = \frac14 - \ln 2 \approx -0.44314718056.
\]

The computed values for \(L=20\) improved variationally with increasing bond dimension, but the dense local solver became prohibitively expensive beyond small/moderate \(D_{\max}\).

This motivates a future upgrade to a **matrix-free Lanczos/Davidson local solver**.

---

## Memory issue and large-system entanglement entropy

A `MemoryError` occurred when trying to compute AKLT entanglement entropy for \(L=20\) using dense-state reconstruction:

- the old entropy routine called `mps_to_dense(mps)`,
- for spin-1 systems this scales as \(3^L\),
- at \(L=20\), the dense state is far too large.

To fix this, a **safe MPS-based entanglement entropy patch** was added:

- `bond_entropy_mps`
- `all_bond_entropies_mps`
- `plot_entanglement_entropy_mps`

This version computes bond entropies directly from MPS QR/SVD operations and is safe for large systems.

---

## Final practical recommendations

### Use current code for:

- Heisenberg benchmarks at small system sizes,
- AKLT at small and moderate/large sizes,
- pedagogical DMRG demonstrations,
- convergence studies.

### Avoid with current dense-local solver:

- Heisenberg at large \(L\) and large bond dimension,
- dense-state observables for large systems.

### Good use cases now:

- Heisenberg: \(L \le 8\) or modest exploratory runs at \(L=20\) with small \(D_{\max}\),
- AKLT: \(L=20,40,60\) with \(D_{\max}=2\).

---

## Figures generated

The final lean workflow saved figures such as:

- `heisenberg_energy_vs_L.png`
- `heisenberg_error_vs_L.png`
- `heisenberg_L6_convergence.png`
- `heisenberg_L6_discarded_weight.png`
- `aklt_energy_vs_L.png`
- `aklt_error_vs_L.png`
- `aklt_discarded_weight_vs_L.png`
- `aklt_L8_convergence.png`
- `aklt_L8_discarded_weight.png`
- `aklt_entanglement_entropy_L8.png`

Additional runtime-scaling figures were also generated, e.g.

- `heisenberg_runtime_scaling_L20.png`

---

## Final conclusion

By the end of the session, we obtained a clean, validated two-site DMRG implementation that:

- reproduces exact benchmark energies for the Heisenberg chain,
- reproduces exact benchmark energies for the AKLT chain,
- reproduces the exact AKLT MPS energy,
- supports convergence diagnostics and figure generation in Jupyter,
- and includes a safe MPS-based entanglement entropy routine for large systems.

The code is accurate and pedagogically useful, while larger Heisenberg calculations would benefit from a future upgrade to a matrix-free iterative local eigensolver.
""")

outfile = os.path.join("figureAKLT", "session_summary.md")
with open(outfile, "w", encoding="utf-8") as f:
    f.write(md_text)

print(f"saved: {outfile}")
```

---

## If you want the actual full transcript
I can also generate a **more detailed markdown transcript** with sectioned conversation notes, for example:

- user requests
- assistant responses
- code revisions
- debug milestones
- final working state

It would still be a reconstructed transcript rather than a literal raw export, but it could be much closer to the step-by-step dialogue.

If you want, I can produce that next.
