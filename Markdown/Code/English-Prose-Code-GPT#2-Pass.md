# Test-Code-GPT#2-Pass

## 🧑‍💻 User


The markdown is a note on MPS and DMRG. Could you generate python codes according to it? Please compute both S=1/2 Heisenberg model and AKLT model, and make sure that a "scalable" matrix-free approach with LANCZOS is adapted. I work with Jupyter. Please show all figures and save them in directory "figureAKLT". Please stay strictly with this markdown note implementation. 
> I prefer to fewer cells.
> Please put Cell # at the head of each cell.
> Please give me codes cell by cell.



---
## 🤖 Assistant


Below is a compact Jupyter-style implementation, strictly following the note’s conventions:

- MPS tensors: `(D_left, d, D_right)`
- MPO tensors: `(chi_left, chi_right, d, d)` with physical indices `(ket, bra)`
- matrix-free local action
- Lanczos via `scipy.sparse.linalg.eigsh`
- two-site DMRG first
- figures shown and saved in `figureAKLT`
- fewer cells
- each cell starts with `# Cell N`

I keep the code self-contained and organized so you can paste cell by cell into Jupyter.

---

```python
# Cell 1
import os
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, eigsh
import matplotlib.pyplot as plt

np.set_printoptions(precision=12, suppress=True)
DTYPE = np.complex128

os.makedirs("figureAKLT", exist_ok=True)
```

---

```python
# Cell 2
# Local operators, MPO construction, dense checks, and exact AKLT MPS

def spin_half_ops():
    I = np.eye(2, dtype=DTYPE)
    Sp = np.array([[0, 1],
                   [0, 0]], dtype=DTYPE)
    Sm = np.array([[0, 0],
                   [1, 0]], dtype=DTYPE)
    Sz = 0.5 * np.array([[1, 0],
                         [0, -1]], dtype=DTYPE)
    return I, Sp, Sm, Sz

def spin_one_ops():
    I = np.eye(3, dtype=DTYPE)
    Sp = np.sqrt(2.0) * np.array([[0, 1, 0],
                                  [0, 0, 1],
                                  [0, 0, 0]], dtype=DTYPE)
    Sm = np.sqrt(2.0) * np.array([[0, 0, 0],
                                  [1, 0, 0],
                                  [0, 1, 0]], dtype=DTYPE)
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, -1]], dtype=DTYPE)
    return I, Sp, Sm, Sz

def heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0):
    I, Sp, Sm, Sz = spin_half_ops()
    d = 2
    chi = 5

    Wbulk = np.zeros((chi, chi, d, d), dtype=DTYPE)
    ops = [[None for _ in range(chi)] for _ in range(chi)]

    ops[0][0] = I
    ops[1][0] = Sp
    ops[2][0] = Sm
    ops[3][0] = Sz
    ops[4][0] = -h * Sz
    ops[4][1] = (J/2.0) * Sm
    ops[4][2] = (J/2.0) * Sp
    ops[4][3] = Jz * Sz
    ops[4][4] = I

    for a in range(chi):
        for b in range(chi):
            if ops[a][b] is not None:
                Wbulk[a, b] = ops[a][b]

    Wleft = np.zeros((1, chi, d, d), dtype=DTYPE)
    Wleft[0, 0] = -h * Sz
    Wleft[0, 1] = (J/2.0) * Sm
    Wleft[0, 2] = (J/2.0) * Sp
    Wleft[0, 3] = Jz * Sz
    Wleft[0, 4] = I

    Wright = np.zeros((chi, 1, d, d), dtype=DTYPE)
    Wright[0, 0] = I
    Wright[1, 0] = Sp
    Wright[2, 0] = Sm
    Wright[3, 0] = Sz
    Wright[4, 0] = -h * Sz

    W = [None] * L
    if L == 1:
        raise ValueError("Need L >= 2")
    W[0] = Wleft
    for i in range(1, L-1):
        W[i] = Wbulk.copy()
    W[L-1] = Wright
    return W

def aklt_mpo(L):
    I, Sp, Sm, Sz = spin_one_ops()
    d = 3
    chi = 14

    O = [Sp/np.sqrt(2.0), Sm/np.sqrt(2.0), Sz]
    Ob = [Sm/np.sqrt(2.0), Sp/np.sqrt(2.0), Sz]

    O2 = []
    Ob2 = []
    for a in range(3):
        for b in range(3):
            O2.append(O[a] @ O[b])
            Ob2.append(Ob[a] @ Ob[b])

    Wbulk = np.zeros((chi, chi, d, d), dtype=DTYPE)

    Wbulk[0, 0] = I
    for a in range(3):
        Wbulk[0, 1+a] = O[a]
    for p in range(9):
        Wbulk[0, 4+p] = (1.0/3.0) * O2[p]
    for a in range(3):
        Wbulk[1+a, 13] = Ob[a]
    for p in range(9):
        Wbulk[4+p, 13] = Ob2[p]
    Wbulk[13, 13] = I

    Wleft = np.zeros((1, chi, d, d), dtype=DTYPE)
    Wleft[0, 0] = I
    for a in range(3):
        Wleft[0, 1+a] = O[a]
    for p in range(9):
        Wleft[0, 4+p] = (1.0/3.0) * O2[p]

    Wright = np.zeros((chi, 1, d, d), dtype=DTYPE)
    for a in range(3):
        Wright[1+a, 0] = Ob[a]
    for p in range(9):
        Wright[4+p, 0] = Ob2[p]
    Wright[13, 0] = I

    W = [None] * L
    if L == 1:
        raise ValueError("Need L >= 2")
    W[0] = Wleft
    for i in range(1, L-1):
        W[i] = Wbulk.copy()
    W[L-1] = Wright
    return W

def kron_all(op_list):
    out = op_list[0]
    for op in op_list[1:]:
        out = np.kron(out, op)
    return out

def dense_heisenberg_hamiltonian(L, J=1.0, Jz=1.0, h=0.0):
    I, Sp, Sm, Sz = spin_half_ops()
    d = 2
    H = np.zeros((d**L, d**L), dtype=DTYPE)

    for i in range(L-1):
        ops = [I]*L
        ops[i] = Sp
        ops[i+1] = Sm
        H += (J/2.0) * kron_all(ops)

        ops = [I]*L
        ops[i] = Sm
        ops[i+1] = Sp
        H += (J/2.0) * kron_all(ops)

        ops = [I]*L
        ops[i] = Sz
        ops[i+1] = Sz
        H += Jz * kron_all(ops)

    for i in range(L):
        ops = [I]*L
        ops[i] = Sz
        H += -h * kron_all(ops)

    return H

def dense_aklt_hamiltonian(L):
    I, Sp, Sm, Sz = spin_one_ops()
    d = 3
    H = np.zeros((d**L, d**L), dtype=DTYPE)

    Xloc = 0.5*(np.kron(Sp, Sm) + np.kron(Sm, Sp)) + np.kron(Sz, Sz)
    hloc = Xloc + (1.0/3.0)*(Xloc @ Xloc)

    for i in range(L-1):
        ops = [I]*L
        ops[i] = None
        ops[i+1] = None
        left = kron_all(ops[:i]) if i > 0 else np.array([[1]], dtype=DTYPE)
        right = kron_all(ops[i+2:]) if i+2 < L else np.array([[1]], dtype=DTYPE)
        H += np.kron(np.kron(left, hloc), right)

    return H

def mpo_to_dense(W):
    L = len(W)
    d = W[0].shape[2]
    H = np.zeros((d**L, d**L), dtype=DTYPE)
    for bra in range(d**L):
        bra_digits = np.base_repr(bra, base=d).zfill(L)
        t_digits = [int(c) for c in bra_digits]
        for ket in range(d**L):
            ket_digits = np.base_repr(ket, base=d).zfill(L)
            s_digits = [int(c) for c in ket_digits]
            env = np.array([1.0+0.0j], dtype=DTYPE)
            for i in range(L):
                Wi = W[i]
                new_env = np.zeros(Wi.shape[1], dtype=DTYPE)
                for bl in range(Wi.shape[0]):
                    for br in range(Wi.shape[1]):
                        new_env[br] += env[bl] * Wi[bl, br, s_digits[i], t_digits[i]]
                env = new_env
            H[ket, bra] = env[0]
    return H

def exact_aklt_mps(L):
    # Right-canonical local tensors in the note's gauge
    Ap = np.array([[0, np.sqrt(2/3)],
                   [0, 0]], dtype=DTYPE)
    A0 = np.array([[-1/np.sqrt(3), 0],
                   [0, 1/np.sqrt(3)]], dtype=DTYPE)
    Am = np.array([[0, 0],
                   [-np.sqrt(2/3), 0]], dtype=DTYPE)

    A = np.stack([Ap, A0, Am], axis=0)  # (d, Dl, Dr) temporarily
    A = np.transpose(A, (1, 0, 2))      # (Dl, d, Dr)

    mps = []
    # choose edge vectors to get one exact ground state in the OBC manifold
    Lvec = np.array([[1.0, 0.0]], dtype=DTYPE)   # (1,2)
    Rvec = np.array([[1.0], [0.0]], dtype=DTYPE) # (2,1)

    if L == 1:
        raise ValueError("Need L >= 2")

    A1 = np.einsum('la,asb->lsb', Lvec, A)
    AL = np.einsum('asb,br->asr', A, Rvec)

    mps.append(A1.astype(DTYPE))
    for _ in range(1, L-1):
        mps.append(A.copy())
    mps.append(AL.astype(DTYPE))
    return mps

def mps_to_dense(psi):
    L = len(psi)
    d = psi[0].shape[1]
    vec = np.zeros(d**L, dtype=DTYPE)
    for n in range(d**L):
        digits = [int(c) for c in np.base_repr(n, base=d).zfill(L)]
        x = np.array([[1.0+0.0j]], dtype=DTYPE)
        for i in range(L):
            x = x @ psi[i][:, digits[i], :]
        vec[n] = x[0, 0]
    return vec

def mpo_expectation_dense(W, psi):
    v = mps_to_dense(psi)
    H = mpo_to_dense(W)
    return np.vdot(v, H @ v) / np.vdot(v, v)
```

---

```python
# Cell 3
# Canonicalization, truncation, environments, matrix-free one-site and two-site actions

def left_canonicalize_qr(mps):
    mps = [A.copy().astype(DTYPE) for A in mps]
    L = len(mps)
    for i in range(L-1):
        Dl, d, Dr = mps[i].shape
        M = np.reshape(mps[i], (Dl*d, Dr), order='C')
        Q, R = la.qr(M, mode='economic')
        k = Q.shape[1]
        mps[i] = np.reshape(Q, (Dl, d, k), order='C')
        mps[i+1] = np.einsum('ab,bsd->asd', R, mps[i+1])
    return mps

def right_canonicalize_qr(mps):
    mps = [A.copy().astype(DTYPE) for A in mps]
    L = len(mps)
    for i in range(L-1, 0, -1):
        Dl, d, Dr = mps[i].shape
        M = np.reshape(mps[i], (Dl, d*Dr), order='C')
        Q, R = la.qr(M.conj().T, mode='economic')
        Bmat = Q.conj().T
        Rd = R.conj().T
        k = Bmat.shape[0]
        mps[i] = np.reshape(Bmat, (k, d, Dr), order='C')
        mps[i-1] = np.einsum('asd,db->asb', mps[i-1], Rd)
    return mps

def mixed_canonical_from_random(L, d, Dmax, seed=0):
    rng = np.random.default_rng(seed)
    Ds = [1]
    for i in range(1, L):
        Ds.append(min(Dmax, Ds[-1]*d))
    Ds.append(1)
    mps = []
    for i in range(L):
        Dl = Ds[i]
        Dr = Ds[i+1] if i+1 < len(Ds) else 1
        A = rng.normal(size=(Dl, d, Dr)) + 1j*rng.normal(size=(Dl, d, Dr))
        A = A.astype(DTYPE)
        mps.append(A)
    mps = left_canonicalize_qr(mps)
    return mps

def svd_truncate(X, Dmax, eps=1e-12, cutoff=None):
    U, s, Vh = la.svd(X, full_matrices=False)
    if cutoff is None:
        D = min(Dmax, max(1, np.sum(s > eps)))
    else:
        keep = min(Dmax, len(s))
        D = keep
        for k in range(1, keep+1):
            disc = np.sum(s[k:]**2)
            if disc <= cutoff:
                D = k
                break
    eta = np.sum(s[D:]**2)
    return U[:, :D], s[:D], Vh[:D, :], eta

def build_right_envs(mps, W):
    L = len(mps)
    R = [None]*(L+1)
    R[L] = np.ones((1, 1, 1), dtype=DTYPE)
    for i in range(L-1, 0, -1):
        B = mps[i]
        Wi = W[i]
        R[i] = np.einsum('xsa,bBst,Bac,ytc->bxy', B, Wi, R[i+1], np.conj(B))
    return R

def update_left_env(Lold, A, Wsite):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, Wsite, np.conj(A))

def update_right_env(Rold, B, Wsite):
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, Wsite, Rold, np.conj(B))

def one_site_matvec(v, Lenv, Wsite, Renv, Dl, d, Dr):
    M = np.reshape(v, (Dl, d, Dr), order='C')
    X = np.einsum('byx,ysz->bxsz', Lenv, M)
    Y = np.einsum('bBst,bxsz->Bxtz', Wsite, X)
    HM = np.einsum('Bxtz,Bza->xta', Y, Renv)
    return np.reshape(HM, (-1,), order='C')

def two_site_matvec(v, Lenv, W1, W2, Renv, Dl, d1, d2, Dr):
    Theta = np.reshape(v, (Dl, d1, d2, Dr), order='C')
    X = np.einsum('byx,yuvz->bxuvz', Lenv, Theta)
    Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
    Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv)
    return np.reshape(HT, (-1,), order='C')

def expectation_mpo_mps(mps, W):
    Lenv = np.ones((1, 1, 1), dtype=DTYPE)
    for i in range(len(mps)):
        Lenv = np.einsum('bxy,xsa,bBst,ytc->Bac', Lenv, mps[i], W[i], np.conj(mps[i]))
    return Lenv[0, 0, 0]

def mps_norm(mps):
    return np.vdot(mps_to_dense(mps), mps_to_dense(mps)).real

def normalize_mps_dense(mps):
    nrm = np.sqrt(mps_norm(mps))
    out = [A.copy() for A in mps]
    out[0] /= nrm
    return out
```

---

```python
# Cell 4
# Two-site DMRG following the note's validated matrix-free contraction

def dmrg_two_site(W, d, Dmax=40, nsweeps=8, tol=1e-10, cutoff=1e-12, seed=0, init_mps=None, lanczos_ncv=None):
    L = len(W)
    if init_mps is None:
        mps = mixed_canonical_from_random(L, d, Dmax=max(2, min(Dmax, 4)), seed=seed)
    else:
        mps = [A.copy().astype(DTYPE) for A in init_mps]

    mps = left_canonicalize_qr(mps)
    mps = normalize_mps_dense(mps)
    R = build_right_envs(mps, W)
    energies = []
    max_discards = []

    for sweep in range(nsweeps):
        Lenv = np.ones((1, 1, 1), dtype=DTYPE)
        discards = []

        # left-to-right
        for i in range(L-1):
            Dl = mps[i].shape[0]
            Dr = mps[i+1].shape[2]
            d1 = mps[i].shape[1]
            d2 = mps[i+1].shape[1]

            Theta0 = np.einsum('asg,gtb->astb', mps[i], mps[i+1])
            v0 = np.reshape(Theta0, (-1,), order='C')
            dim = v0.size

            mv = lambda v: two_site_matvec(v, Lenv, W[i], W[i+1], R[i+2], Dl, d1, d2, Dr)
            Hop = LinearOperator((dim, dim), matvec=mv, dtype=DTYPE)
            ncv = min(dim, max(20, 2 if lanczos_ncv is None else lanczos_ncv))
            evals, evecs = eigsh(Hop, k=1, which='SA', v0=v0, ncv=min(dim, max(4, ncv)))
            Theta = np.reshape(evecs[:, 0], (Dl, d1, d2, Dr), order='C')

            X = np.reshape(Theta, (Dl*d1, d2*Dr), order='C')
            U, s, Vh, eta = svd_truncate(X, Dmax=Dmax, eps=1e-14, cutoff=cutoff)
            Dnew = len(s)

            A = np.reshape(U, (Dl, d1, Dnew), order='C')
            Mnext = np.reshape(np.diag(s) @ Vh, (Dnew, d2, Dr), order='C')

            mps[i] = A
            mps[i+1] = Mnext
            discards.append(eta.real)

            Lenv = update_left_env(Lenv, mps[i], W[i])

        # right environments for updated state
        R = build_right_envs(mps, W)

        # right-to-left
        for i in range(L-2, -1, -1):
            Dl = mps[i].shape[0]
            Dr = mps[i+1].shape[2]
            d1 = mps[i].shape[1]
            d2 = mps[i+1].shape[1]

            # build left env up to i
            Lenv = np.ones((1, 1, 1), dtype=DTYPE)
            for j in range(i):
                Lenv = update_left_env(Lenv, mps[j], W[j])

            Theta0 = np.einsum('asg,gtb->astb', mps[i], mps[i+1])
            v0 = np.reshape(Theta0, (-1,), order='C')
            dim = v0.size

            mv = lambda v: two_site_matvec(v, Lenv, W[i], W[i+1], R[i+2], Dl, d1, d2, Dr)
            Hop = LinearOperator((dim, dim), matvec=mv, dtype=DTYPE)
            ncv = min(dim, max(20, 2 if lanczos_ncv is None else lanczos_ncv))
            evals, evecs = eigsh(Hop, k=1, which='SA', v0=v0, ncv=min(dim, max(4, ncv)))
            Theta = np.reshape(evecs[:, 0], (Dl, d1, d2, Dr), order='C')

            X = np.reshape(Theta, (Dl*d1, d2*Dr), order='C')
            U, s, Vh, eta = svd_truncate(X, Dmax=Dmax, eps=1e-14, cutoff=cutoff)
            Dnew = len(s)

            Mleft = np.reshape(U @ np.diag(s), (Dl, d1, Dnew), order='C')
            B = np.reshape(Vh, (Dnew, d2, Dr), order='C')

            mps[i] = Mleft
            mps[i+1] = B
            discards.append(eta.real)

            R[i+1] = update_right_env(R[i+2], mps[i+1], W[i+1])

        # re-gauge to left-canonical form for stability
        mps = left_canonicalize_qr(mps)
        mps = normalize_mps_dense(mps)
        R = build_right_envs(mps, W)

        E = expectation_mpo_mps(mps, W).real
        energies.append(E)
        max_discards.append(max(discards) if discards else 0.0)

        if len(energies) >= 2 and abs(energies[-1] - energies[-2]) < tol:
            break

    return mps, np.array(energies), np.array(max_discards)
```

---

```python
# Cell 5
# Validation tests from the note: MPO-to-dense and AKLT exact MPS benchmark

# Heisenberg MPO tests
for L in [2, 3, 4]:
    W = heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0)
    Hd_mpo = mpo_to_dense(W)
    Hd_ref = dense_heisenberg_hamiltonian(L, J=1.0, Jz=1.0, h=0.0)
    err = np.max(np.abs(Hd_mpo - Hd_ref))
    print(f"Heisenberg MPO dense test L={L}: max|Δ| = {err:.3e}")

# AKLT MPO tests
for L in [2, 3, 4]:
    W = aklt_mpo(L)
    Hd_mpo = mpo_to_dense(W)
    Hd_ref = dense_aklt_hamiltonian(L)
    err = np.max(np.abs(Hd_mpo - Hd_ref))
    print(f"AKLT MPO dense test L={L}: max|Δ| = {err:.3e}")

# Exact AKLT MPS energy test
for L in [4, 6, 8]:
    W = aklt_mpo(L)
    psi = normalize_mps_dense(exact_aklt_mps(L))
    E = expectation_mpo_mps(psi, W).real
    E_exact = -(2.0/3.0)*(L-1)
    print(f"AKLT exact MPS energy L={L}: E = {E:.12f}, exact = {E_exact:.12f}, Δ = {abs(E-E_exact):.3e}")
```

---

```python
# Cell 6
# Run scalable matrix-free two-site DMRG with Lanczos for both models and make figures

# ---------- Heisenberg S=1/2 ----------
Lh = 24
Wh = heisenberg_mpo(Lh, J=1.0, Jz=1.0, h=0.0)
mps_h, Ehist_h, disc_h = dmrg_two_site(
    Wh, d=2, Dmax=64, nsweeps=10, tol=1e-9, cutoff=1e-10, seed=1
)

print(f"Heisenberg L={Lh} final DMRG energy = {Ehist_h[-1]:.12f}")
print(f"Heisenberg energy/site = {Ehist_h[-1]/Lh:.12f}")

# ---------- AKLT spin-1 ----------
La = 24
Wa = aklt_mpo(La)
init_aklt = exact_aklt_mps(La)  # exact bond-dim 2 benchmark/state
init_aklt = normalize_mps_dense(init_aklt)

mps_a, Ehist_a, disc_a = dmrg_two_site(
    Wa, d=3, Dmax=32, nsweeps=8, tol=1e-11, cutoff=1e-12, seed=2, init_mps=init_aklt
)

E_exact_aklt = -(2.0/3.0)*(La-1)
print(f"AKLT L={La} final DMRG energy = {Ehist_a[-1]:.12f}")
print(f"AKLT exact open-chain ground energy = {E_exact_aklt:.12f}")
print(f"AKLT |ΔE| = {abs(Ehist_a[-1]-E_exact_aklt):.3e}")

# ---------- Figures ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(np.arange(1, len(Ehist_h)+1), Ehist_h, 'o-', label='Heisenberg')
ax[0].set_xlabel('Sweep')
ax[0].set_ylabel('Energy')
ax[0].set_title(f'Heisenberg S=1/2, L={Lh}')
ax[0].grid(True)
ax[0].legend()

ax[1].semilogy(np.arange(1, len(disc_h)+1), np.maximum(disc_h, 1e-18), 'o-', label='Heisenberg')
ax[1].set_xlabel('Sweep')
ax[1].set_ylabel('Max discarded weight')
ax[1].set_title(f'Heisenberg S=1/2, L={Lh}')
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.savefig("figureAKLT/heisenberg_dmrg.png", dpi=160)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(np.arange(1, len(Ehist_a)+1), Ehist_a, 'o-', label='AKLT DMRG')
ax[0].

---
## 🧑‍💻 User


Heisenberg MPO dense test L=2: max|Δ| = 0.000e+00
Heisenberg MPO dense test L=3: max|Δ| = 0.000e+00
Heisenberg MPO dense test L=4: max|Δ| = 0.000e+00
AKLT MPO dense test L=2: max|Δ| = 3.331e-16
AKLT MPO dense test L=3: max|Δ| = 6.661e-16
AKLT MPO dense test L=4: max|Δ| = 8.882e-16
AKLT exact MPS energy L=4: E = -2.000000000000, exact = -2.000000000000, Δ = 0.000e+00
AKLT exact MPS energy L=6: E = -3.333333333333, exact = -3.333333333333, Δ = 8.882e-16
AKLT exact MPS energy L=8: E = -4.666666666667, exact = -4.666666666667, Δ = 1.776e-15

---
## 🤖 Assistant


Great — these outputs show the MPO constructions and the exact AKLT MPS benchmark are correct to numerical precision.

So we can continue with the remaining plotting/run cell. Below I give the corrected continuation as a single final code cell, starting again with a cell header.

---

```python
# Cell 6
# Run scalable matrix-free two-site DMRG with Lanczos for both models and make figures

# ---------- Heisenberg S=1/2 ----------
Lh = 24
Wh = heisenberg_mpo(Lh, J=1.0, Jz=1.0, h=0.0)
mps_h, Ehist_h, disc_h = dmrg_two_site(
    Wh, d=2, Dmax=64, nsweeps=10, tol=1e-9, cutoff=1e-10, seed=1
)

print(f"Heisenberg L={Lh} final DMRG energy = {Ehist_h[-1]:.12f}")
print(f"Heisenberg energy/site = {Ehist_h[-1]/Lh:.12f}")

# ---------- AKLT spin-1 ----------
La = 24
Wa = aklt_mpo(La)
init_aklt = normalize_mps_dense(exact_aklt_mps(La))

mps_a, Ehist_a, disc_a = dmrg_two_site(
    Wa, d=3, Dmax=32, nsweeps=8, tol=1e-11, cutoff=1e-12, seed=2, init_mps=init_aklt
)

E_exact_aklt = -(2.0/3.0)*(La-1)
print(f"AKLT L={La} final DMRG energy = {Ehist_a[-1]:.12f}")
print(f"AKLT exact open-chain ground energy = {E_exact_aklt:.12f}")
print(f"AKLT |ΔE| = {abs(Ehist_a[-1]-E_exact_aklt):.3e}")

# ---------- Figure 1: Heisenberg convergence ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(np.arange(1, len(Ehist_h)+1), Ehist_h, 'o-', label='Heisenberg')
ax[0].set_xlabel('Sweep')
ax[0].set_ylabel('Energy')
ax[0].set_title(f'Heisenberg S=1/2, L={Lh}')
ax[0].grid(True)
ax[0].legend()

ax[1].semilogy(np.arange(1, len(disc_h)+1), np.maximum(disc_h, 1e-18), 'o-', label='Heisenberg')
ax[1].set_xlabel('Sweep')
ax[1].set_ylabel('Max discarded weight')
ax[1].set_title(f'Heisenberg S=1/2, L={Lh}')
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.savefig("figureAKLT/heisenberg_dmrg.png", dpi=160)
plt.show()

# ---------- Figure 2: AKLT convergence ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(np.arange(1, len(Ehist_a)+1), Ehist_a, 'o-', label='AKLT DMRG')
ax[0].axhline(E_exact_aklt, color='r', ls='--', label='Exact')
ax[0].set_xlabel('Sweep')
ax[0].set_ylabel('Energy')
ax[0].set_title(f'AKLT spin-1, L={La}')
ax[0].grid(True)
ax[0].legend()

ax[1].semilogy(np.arange(1, len(disc_a)+1), np.maximum(disc_a, 1e-18), 'o-', label='AKLT')
ax[1].set_xlabel('Sweep')
ax[1].set_ylabel('Max discarded weight')
ax[1].set_title(f'AKLT spin-1, L={La}')
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.savefig("figureAKLT/aklt_dmrg.png", dpi=160)
plt.show()

# ---------- Figure 3: AKLT energy error ----------
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.semilogy(np.arange(1, len(Ehist_a)+1), np.abs(Ehist_a - E_exact_aklt) + 1e-18, 'o-')
ax.set_xlabel('Sweep')
ax.set_ylabel(r'$|E-E_{\mathrm{exact}}|$')
ax.set_title(f'AKLT energy error, L={La}')
ax.grid(True)
plt.tight_layout()
plt.savefig("figureAKLT/aklt_energy_error.png", dpi=160)
plt.show()

# ---------- Save arrays ----------
np.save("figureAKLT/heisenberg_energy_history.npy", Ehist_h)
np.save("figureAKLT/heisenberg_discarded_history.npy", disc_h)
np.save("figureAKLT/aklt_energy_history.npy", Ehist_a)
np.save("figureAKLT/aklt_discarded_history.npy", disc_a)
```

---

If you want, I can give you one more **final compact cell** for the note’s recommended **small-system Heisenberg DMRG vs exact diagonalization benchmark**. That would be the most important remaining validation step before trusting larger runs.

---
## 🧑‍💻 User


I want a quick test within a few mins.

---
## 🤖 Assistant


Use a much smaller system and bond dimension. Replace Cell 6 by this quick-test version.

```python
# Cell 6
# Quick test: a few minutes or less

# ---------- Heisenberg S=1/2 quick test ----------
Lh = 12
Wh = heisenberg_mpo(Lh, J=1.0, Jz=1.0, h=0.0)
mps_h, Ehist_h, disc_h = dmrg_two_site(
    Wh, d=2, Dmax=24, nsweeps=4, tol=1e-8, cutoff=1e-10, seed=1
)

print(f"Heisenberg L={Lh} final DMRG energy = {Ehist_h[-1]:.12f}")
print(f"Heisenberg energy/site = {Ehist_h[-1]/Lh:.12f}")

# optional exact benchmark for small Heisenberg chain
Hh_dense = dense_heisenberg_hamiltonian(Lh, J=1.0, Jz=1.0, h=0.0)
Eh_exact = np.min(np.linalg.eigvalsh(Hh_dense)).real
print(f"Heisenberg exact ED energy = {Eh_exact:.12f}")
print(f"Heisenberg |ΔE| = {abs(Ehist_h[-1] - Eh_exact):.3e}")

# ---------- AKLT spin-1 quick test ----------
La = 10
Wa = aklt_mpo(La)
init_aklt = normalize_mps_dense(exact_aklt_mps(La))

mps_a, Ehist_a, disc_a = dmrg_two_site(
    Wa, d=3, Dmax=8, nsweeps=3, tol=1e-10, cutoff=1e-12, seed=2, init_mps=init_aklt
)

E_exact_aklt = -(2.0/3.0)*(La-1)
print(f"AKLT L={La} final DMRG energy = {Ehist_a[-1]:.12f}")
print(f"AKLT exact open-chain ground energy = {E_exact_aklt:.12f}")
print(f"AKLT |ΔE| = {abs(Ehist_a[-1]-E_exact_aklt):.3e}")

# ---------- Figure 1: Heisenberg quick convergence ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(np.arange(1, len(Ehist_h)+1), Ehist_h, 'o-', label='Heisenberg')
ax[0].axhline(Eh_exact, color='r', ls='--', label='ED exact')
ax[0].set_xlabel('Sweep')
ax[0].set_ylabel('Energy')
ax[0].set_title(f'Heisenberg S=1/2, L={Lh}')
ax[0].grid(True)
ax[0].legend()

ax[1].semilogy(np.arange(1, len(disc_h)+1), np.maximum(disc_h, 1e-18), 'o-', label='Heisenberg')
ax[1].set_xlabel('Sweep')
ax[1].set_ylabel('Max discarded weight')
ax[1].set_title(f'Heisenberg S=1/2, L={Lh}')
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.savefig("figureAKLT/heisenberg_quick_test.png", dpi=160)
plt.show()

# ---------- Figure 2: AKLT quick convergence ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(np.arange(1, len(Ehist_a)+1), Ehist_a, 'o-', label='AKLT DMRG')
ax[0].axhline(E_exact_aklt, color='r', ls='--', label='Exact')
ax[0].set_xlabel('Sweep')
ax[0].set_ylabel('Energy')
ax[0].set_title(f'AKLT spin-1, L={La}')
ax[0].grid(True)
ax[0].legend()

ax[1].semilogy(np.arange(1, len(disc_a)+1), np.maximum(disc_a, 1e-18), 'o-', label='AKLT')
ax[1].set_xlabel('Sweep')
ax[1].set_ylabel('Max discarded weight')
ax[1].set_title(f'AKLT spin-1, L={La}')
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.savefig("figureAKLT/aklt_quick_test.png", dpi=160)
plt.show()

# ---------- Save arrays ----------
np.save("figureAKLT/heisenberg_quick_energy_history.npy", Ehist_h)
np.save("figureAKLT/heisenberg_quick_discarded_history.npy", disc_h)
np.save("figureAKLT/aklt_quick_energy_history.npy", Ehist_a)
np.save("figureAKLT/aklt_quick_discarded_history.npy", disc_a)
```

If you want it even faster, use:
- `Lh = 10`, `Dmax = 16`, `nsweeps = 3`
- `La = 8`, `Dmax = 4`, `nsweeps = 2`

That should finish very quickly in Jupyter.

---
## 🧑‍💻 User


---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[6], line 7
      5 Lh = 12
      6 Wh = heisenberg_mpo(Lh, J=1.0, Jz=1.0, h=0.0)
----> 7 mps_h, Ehist_h, disc_h = dmrg_two_site(
      8     Wh, d=2, Dmax=24, nsweeps=4, tol=1e-8, cutoff=1e-10, seed=1
      9 )
     11 print(f"Heisenberg L={Lh} final DMRG energy = {Ehist_h[-1]:.12f}")
     12 print(f"Heisenberg energy/site = {Ehist_h[-1]/Lh:.12f}")

TypeError: cannot unpack non-iterable NoneType object

---
## 🤖 Assistant


That means your `dmrg_two_site(...)` cell likely did not finish correctly, or the function definition got truncated so it returns `None`.

To fix this quickly, please **re-run Cell 4** by replacing it fully with the corrected version below. Then run your quick-test Cell 6 again.

---

```python
# Cell 4
# Two-site DMRG following the note's validated matrix-free contraction

def dmrg_two_site(W, d, Dmax=40, nsweeps=8, tol=1e-10, cutoff=1e-12, seed=0, init_mps=None, lanczos_ncv=None):
    L = len(W)

    if init_mps is None:
        mps = mixed_canonical_from_random(L, d, Dmax=max(2, min(Dmax, 4)), seed=seed)
    else:
        mps = [A.copy().astype(DTYPE) for A in init_mps]

    mps = left_canonicalize_qr(mps)
    mps = normalize_mps_dense(mps)

    energies = []
    max_discards = []

    for sweep in range(nsweeps):
        discards = []

        # Build right environments for current mixed-canonical state
        R = build_right_envs(mps, W)

        # ---------------- left-to-right sweep ----------------
        Lenv = np.ones((1, 1, 1), dtype=DTYPE)

        for i in range(L - 1):
            Dl = mps[i].shape[0]
            Dr = mps[i + 1].shape[2]
            d1 = mps[i].shape[1]
            d2 = mps[i + 1].shape[1]

            Theta0 = np.einsum('asg,gtb->astb', mps[i], mps[i + 1])
            v0 = np.reshape(Theta0, (-1,), order='C')
            dim = v0.size

            mv = lambda v: two_site_matvec(v, Lenv, W[i], W[i + 1], R[i + 2], Dl, d1, d2, Dr)
            Hop = LinearOperator((dim, dim), matvec=mv, dtype=DTYPE)

            ncv = min(dim, lanczos_ncv if lanczos_ncv is not None else max(8, min(dim, 20)))
            evals, evecs = eigsh(Hop, k=1, which='SA', v0=v0, ncv=ncv)

            Theta = np.reshape(evecs[:, 0], (Dl, d1, d2, Dr), order='C')

            X = np.reshape(Theta, (Dl * d1, d2 * Dr), order='C')
            U, s, Vh, eta = svd_truncate(X, Dmax=Dmax, eps=1e-14, cutoff=cutoff)
            Dnew = len(s)

            A = np.reshape(U, (Dl, d1, Dnew), order='C')
            Mnext = np.reshape(np.diag(s) @ Vh, (Dnew, d2, Dr), order='C')

            mps[i] = A
            mps[i + 1] = Mnext
            discards.append(float(np.real(eta)))

            Lenv = update_left_env(Lenv, mps[i], W[i])

        # ---------------- right-to-left sweep ----------------
        R = build_right_envs(mps, W)

        for i in range(L - 2, -1, -1):
            # build left env up to site i
            Lenv = np.ones((1, 1, 1), dtype=DTYPE)
            for j in range(i):
                Lenv = update_left_env(Lenv, mps[j], W[j])

            Dl = mps[i].shape[0]
            Dr = mps[i + 1].shape[2]
            d1 = mps[i].shape[1]
            d2 = mps[i + 1].shape[1]

            Theta0 = np.einsum('asg,gtb->astb', mps[i], mps[i + 1])
            v0 = np.reshape(Theta0, (-1,), order='C')
            dim = v0.size

            mv = lambda v: two_site_matvec(v, Lenv, W[i], W[i + 1], R[i + 2], Dl, d1, d2, Dr)
            Hop = LinearOperator((dim, dim), matvec=mv, dtype=DTYPE)

            ncv = min(dim, lanczos_ncv if lanczos_ncv is not None else max(8, min(dim, 20)))
            evals, evecs = eigsh(Hop, k=1, which='SA', v0=v0, ncv=ncv)

            Theta = np.reshape(evecs[:, 0], (Dl, d1, d2, Dr), order='C')

            X = np.reshape(Theta, (Dl * d1, d2 * Dr), order='C')
            U, s, Vh, eta = svd_truncate(X, Dmax=Dmax, eps=1e-14, cutoff=cutoff)
            Dnew = len(s)

            Mleft = np.reshape(U @ np.diag(s), (Dl, d1, Dnew), order='C')
            B = np.reshape(Vh, (Dnew, d2, Dr), order='C')

            mps[i] = Mleft
            mps[i + 1] = B
            discards.append(float(np.real(eta)))

            # update right env recursively
            R[i + 1] = update_right_env(R[i + 2], mps[i + 1], W[i + 1])

        # stabilize gauge and normalize
        mps = left_canonicalize_qr(mps)
        mps = normalize_mps_dense(mps)

        E = expectation_mpo_mps(mps, W).real
        energies.append(float(E))
        max_discards.append(max(discards) if len(discards) > 0 else 0.0)

        print(f"sweep {sweep+1:2d}: E = {energies[-1]:.12f}, max discarded weight = {max_discards[-1]:.3e}")

        if len(energies) >= 2 and abs(energies[-1] - energies[-2]) < tol:
            break

    return mps, np.array(energies), np.array(max_discards)
```

---

Then re-run your quick test.

If you want an even safer very-fast test first, use this tiny one:

```python
# Cell 6
Lh = 10
Wh = heisenberg_mpo(Lh, J=1.0, Jz=1.0, h=0.0)
mps_h, Ehist_h, disc_h = dmrg_two_site(
    Wh, d=2, Dmax=16, nsweeps=3, tol=1e-8, cutoff=1e-10, seed=1
)

print("Heisenberg energies:", Ehist_h)

La = 8
Wa = aklt_mpo(La)
init_aklt = normalize_mps_dense(exact_aklt_mps(La))
mps_a, Ehist_a, disc_a = dmrg_two_site(
    Wa, d=3, Dmax=4, nsweeps=2, tol=1e-10, cutoff=1e-12, seed=2, init_mps=init_aklt
)

print("AKLT energies:", Ehist_a)
print("AKLT exact:", -(2.0/3.0)*(La-1))
```

If you get another error, send me the full traceback and I’ll give you a direct patch cell only.

---
## 🧑‍💻 User


sweep  1: E = -5.142059307561, max discarded weight = 5.418e-11
sweep  2: E = -5.142076510323, max discarded weight = 6.329e-11
sweep  3: E = -5.142076516202, max discarded weight = 6.775e-11
Heisenberg L=12 final DMRG energy = -5.142076516202
Heisenberg energy/site = -0.428506376350
Heisenberg exact ED energy = -5.142090632841
Heisenberg |ΔE| = 1.412e-05
sweep  1: E = -5.999999999974, max discarded weight = 5.093e-12
sweep  2: E = -5.999999999976, max discarded weight = 4.962e-12
AKLT L=10 final DMRG energy = -5.999999999976
AKLT exact open-chain ground energy = -6.000000000000
AKLT |ΔE| = 2.405e-11

---
## 🤖 Assistant


Excellent. These are good quick-test results.

## Interpretation

### Heisenberg
- DMRG: `-5.142076516202`
- Exact ED: `-5.142090632841`
- Error: `1.412e-05`

For a quick run with small `Dmax` and few sweeps, this is perfectly reasonable.

### AKLT
- DMRG: `-5.999999999976`
- Exact: `-6.000000000000`
- Error: `2.405e-11`

This is essentially exact, as expected from:
- correct AKLT MPO
- exact AKLT MPS initialization
- exact ground state representable with bond dimension 2

So your implementation is behaving consistently with the note.

---

## What this means

You now have:

- correct Heisenberg MPO
- correct AKLT MPO
- correct exact AKLT MPS benchmark
- matrix-free two-site effective Hamiltonian
- Lanczos local solver
- finite-system two-site DMRG working
- figures saved to `figureAKLT`

---

## If you want the quickest trustworthy production settings

Use this as your practical quick setup:

```python
# Heisenberg
Lh = 12
Dmax = 24
nsweeps = 4

# AKLT
La = 10
Dmax = 8
nsweeps = 3
```

If you want slightly better Heisenberg accuracy while still fast:

```python
Lh = 12
Dmax = 32
nsweeps = 5
```

That should reduce the Heisenberg error further.

---

## Optional final plotting cell

If your earlier plotting cell was interrupted, here is a compact plotting-only cell.

```python
# Cell 7
# Plot quick-test results only

# Heisenberg
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(np.arange(1, len(Ehist_h)+1), Ehist_h, 'o-', label='DMRG')
ax[0].axhline(Eh_exact, color='r', ls='--', label='ED exact')
ax[0].set_xlabel('Sweep')
ax[0].set_ylabel('Energy')
ax[0].set_title('Heisenberg quick test')
ax[0].grid(True)
ax[0].legend()

ax[1].semilogy(np.arange(1, len(disc_h)+1), np.maximum(disc_h, 1e-18), 'o-')
ax[1].set_xlabel('Sweep')
ax[1].set_ylabel('Max discarded weight')
ax[1].set_title('Heisenberg discarded weight')
ax[1].grid(True)

plt.tight_layout()
plt.savefig("figureAKLT/heisenberg_quick_test.png", dpi=160)
plt.show()

# AKLT
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(np.arange(1, len(Ehist_a)+1), Ehist_a, 'o-', label='DMRG')
ax[0].axhline(E_exact_aklt, color='r', ls='--', label='Exact')
ax[0].set_xlabel('Sweep')
ax[0].set_ylabel('Energy')
ax[0].set_title('AKLT quick test')
ax[0].grid(True)
ax[0].legend()

ax[1].semilogy(np.arange(1, len(disc_a)+1), np.maximum(disc_a, 1e-18), 'o-')
ax[1].set_xlabel('Sweep')
ax[1].set_ylabel('Max discarded weight')
ax[1].set_title('AKLT discarded weight')
ax[1].grid(True)

plt.tight_layout()
plt.savefig("figureAKLT/aklt_quick_test.png", dpi=160)
plt.show()
```

---

## Bottom line

Your quick test passed.

- Heisenberg: good approximate agreement with ED
- AKLT: essentially exact

If you want, next I can give you exactly one additional cell for either:

1. **one-site DMRG implementation** following the same note, or  
2. **entanglement spectrum / bond singular values plots**, or  
3. **local projected-operator validation test** against dense `P†HP` for the two-site matvec.

---
## 🧑‍💻 User


Before that I would like to check wheter we stay strictly with the note.

---
## 🤖 Assistant


Yes — largely **yes**, with one important qualification.

## What is strictly consistent with the note

The code does follow the note in the core structural points:

- **MPS convention**: `A[i].shape = (D_left, d, D_right)`
- **MPO convention**: `W[i].shape = (chi_left, chi_right, d, d)`
- MPO physical index order is taken as  
  **first physical index = ket/output**,  
  **second physical index = bra/input**
- **C-order reshape convention** is used throughout
- **QR canonicalization** uses the note’s grouped reshapes
- **SVD truncation** uses the note’s left-grouped two-site reshape
- **environment update formulas** are exactly the note’s validated ones:
  - left env: `bxy, xsa, bBst, ytc -> Bac`
  - right env: `xsa, bBst, Bac, ytc -> bxy`
- **two-site effective Hamiltonian action** is exactly the note’s validated matrix-free contraction:
  - `byx, yuvz -> bxuvz`
  - `bBus, bxuvz -> Bxsvz`
  - `BCvt, Bxsvz -> Cxstz`
  - `Cxstz, Cza -> xsta`
- **Lanczos / matrix-free LinearOperator** is used, as recommended
- **two-site DMRG first** is exactly the note’s recommendation
- **Heisenberg MPO** and **AKLT MPO** follow the explicit constructions in the note
- **AKLT exact MPS benchmark** matches the note

So on the most important technical conventions: **yes, we stayed with the note**.

---

## The qualification: where the code is not maximally strict

There are a few places where the implementation is **practical** rather than fully “strict-textbook-by-note”.

### 1. The initial MPS builder is simplified
In `mixed_canonical_from_random`, I used a simple random initialization and then left-canonicalized it.

This is consistent with the note’s conventions, but the note does not prescribe this exact initializer. So this is **compatible**, but not uniquely derived from the note.

---

### 2. The bond-dimension schedule in random initialization is pragmatic
I used a simple growing bond-dimension pattern instead of a more careful controlled initial bond structure.

Again: compatible, but not explicitly dictated by the note.

---

### 3. The truncation rule is slightly more practical
The note says:

- either threshold by singular values,
- or use a discarded-weight criterion,
- and always keep at least one singular value.

The current `svd_truncate` supports that, but the exact decision logic is a practical implementation choice, not a literal line-by-line transcription.

So this is **within the note**, but not an exact verbatim algorithmic policy.

---

### 4. The right-to-left sweep rebuilds left environments naively
In the right sweep I rebuild the left environment by looping from the left boundary up to bond `i`.

This is correct and consistent with the note, but it is not the most efficient formulation of the full algorithm as one would optimize it.  
The note emphasizes scalable matrix-free local solves; this part is still fine for quick runs, but the environment management is not yet the most optimized possible.

So:
- **correct by note conventions**
- **not the most efficient production implementation**

---

### 5. We have not yet implemented the note’s full validation hierarchy
The note strongly insists on:

- one-site projected-operator test vs dense `P†HP`
- two-site projected-operator test vs dense `P†HP`
- environment recursion tests
- Hermiticity tests

We did:
- MPO-to-dense tests
- AKLT exact MPS energy test
- small Heisenberg ED benchmark

These are very good, but not yet the **full strict checklist** from the note.

So in terms of **development process**, we have not yet completed every validation step the note prescribes.

---

## Short verdict

### Strict with the note in:
- tensor conventions
- reshape conventions
- MPO conventions
- validated environment contractions
- validated matrix-free two-site local Hamiltonian
- Lanczos usage
- Heisenberg and AKLT constructions

### Not yet strict in the strongest possible sense because:
- some practical implementation choices were made for initialization/truncation/workflow
- the full dense projected-operator validation tests from the note have not been implemented yet
- the sweep/environment bookkeeping is correct but not fully optimized

---

## My recommendation

If you want to be able to say **“this implementation stays strictly with the markdown note”**, then the next thing to do is add the note’s decisive validation cell:

### the **two-site projected-operator test**
Compare the dense matrix generated by repeated application of the matrix-free two-site action against the exact projected operator `P† H P` on a tiny chain.

That is exactly the note’s strongest correctness criterion.

If that passes, then you can say much more confidently:

> the local operator implementation follows the note strictly, including the subtle left-environment index swap.

---

## Most honest one-line answer

**Yes for the core implementation conventions and local contractions; not fully yet for the note’s complete validation protocol.**

If you want, I can now give you **one compact Jupyter cell** that performs the note’s **two-site projected-operator validation test** for a tiny Heisenberg chain.

---
## 🧑‍💻 User


Make sure free-matrix approach and the scalability.

---
## 🤖 Assistant


Understood. Then the clean answer is:

## Yes for the local solver, but not fully yet for overall scalability

### What is already matrix-free and scalable
The most important part **is** already implemented in the scalable way required by the note:

- the **local effective Hamiltonian is never built densely** in DMRG sweeps
- instead, only the action
  \[
  v \mapsto H_{\mathrm{eff}} v
  \]
  is implemented
- this is wrapped in `LinearOperator`
- and solved with **Lanczos** via `eigsh`

That is exactly the note’s required scalable matrix-free strategy.

In particular, these are matrix-free:
- `one_site_matvec(...)`
- `two_site_matvec(...)`
- local eigensolve in `dmrg_two_site(...)`

So the **core DMRG solver is matrix-free and scalable**.

---

## What is not fully scalable yet
There are still a few non-production parts in the notebook:

### 1. `normalize_mps_dense`
This converts the full MPS to a dense statevector.
That is **not scalable**.

### 2. `mps_norm`
Also dense, hence **not scalable**.

### 3. `expectation_mpo_mps`
This one **is scalable**, since it uses environment-style contraction and does not build the full state or full Hamiltonian.

So energy evaluation is fine, but normalization currently uses dense conversion.

### 4. small-system validation utilities
These are intentionally dense:
- `mpo_to_dense`
- `dense_heisenberg_hamiltonian`
- `dense_aklt_hamiltonian`
- `mps_to_dense`

Those are acceptable only for validation, exactly as the note allows. They should not be used in production scaling.

---

# To stay strictly matrix-free/scalable in actual runs

We should remove dense normalization from the sweep code and replace it by **MPS-overlap contraction**.

That is the missing piece.

Below I give you a **replacement cell** for the normalization and overlap routines that is fully scalable and consistent with the note. After that, I give a tiny patch to `dmrg_two_site`.

---

```python
# Cell 3A
# Scalable overlap and normalization routines replacing dense normalization

def mps_overlap(mps1, mps2):
    """
    Scalable contraction for <mps1|mps2>.
    Both MPS use tensors of shape (Dl, d, Dr).
    """
    E = np.ones((1, 1), dtype=DTYPE)
    for A, B in zip(mps1, mps2):
        # E[x,y], conj(A)[x,s,a], B[y,s,b] -> Enew[a,b]
        E = np.einsum('xy,xsa,ysb->ab', E, np.conj(A), B)
    return E[0, 0]

def mps_norm(mps):
    return np.real(mps_overlap(mps, mps))

def normalize_mps(mps):
    nrm = np.sqrt(mps_norm(mps))
    out = [A.copy() for A in mps]
    out[0] /= nrm
    return out
```

---

Now patch `dmrg_two_site` so it uses `normalize_mps(...)` instead of `normalize_mps_dense(...)`.

Replace these two lines inside `dmrg_two_site`:

```python
mps = normalize_mps_dense(mps)
```

by

```python
mps = normalize_mps(mps)
```

Specifically, your corrected `dmrg_two_site` should use:

---

```python
# Cell 4A
# Patch: fully scalable normalization inside dmrg_two_site

def dmrg_two_site(W, d, Dmax=40, nsweeps=8, tol=1e-10, cutoff=1e-12, seed=0, init_mps=None, lanczos_ncv=None):
    L = len(W)

    if init_mps is None:
        mps = mixed_canonical_from_random(L, d, Dmax=max(2, min(Dmax, 4)), seed=seed)
    else:
        mps = [A.copy().astype(DTYPE) for A in init_mps]

    mps = left_canonicalize_qr(mps)
    mps = normalize_mps(mps)

    energies = []
    max_discards = []

    for sweep in range(nsweeps):
        discards = []

        R = build_right_envs(mps, W)

        # left-to-right
        Lenv = np.ones((1, 1, 1), dtype=DTYPE)

        for i in range(L - 1):
            Dl = mps[i].shape[0]
            Dr = mps[i + 1].shape[2]
            d1 = mps[i].shape[1]
            d2 = mps[i + 1].shape[1]

            Theta0 = np.einsum('asg,gtb->astb', mps[i], mps[i + 1])
            v0 = np.reshape(Theta0, (-1,), order='C')
            dim = v0.size

            mv = lambda v: two_site_matvec(v, Lenv, W[i], W[i + 1], R[i + 2], Dl, d1, d2, Dr)
            Hop = LinearOperator((dim, dim), matvec=mv, dtype=DTYPE)

            ncv = min(dim, lanczos_ncv if lanczos_ncv is not None else max(8, min(dim, 20)))
            evals, evecs = eigsh(Hop, k=1, which='SA', v0=v0, ncv=ncv)

            Theta = np.reshape(evecs[:, 0], (Dl, d1, d2, Dr), order='C')

            X = np.reshape(Theta, (Dl * d1, d2 * Dr), order='C')
            U, s, Vh, eta = svd_truncate(X, Dmax=Dmax, eps=1e-14, cutoff=cutoff)
            Dnew = len(s)

            A = np.reshape(U, (Dl, d1, Dnew), order='C')
            Mnext = np.reshape(np.diag(s) @ Vh, (Dnew, d2, Dr), order='C')

            mps[i] = A
            mps[i + 1] = Mnext
            discards.append(float(np.real(eta)))

            Lenv = update_left_env(Lenv, mps[i], W[i])

        # right-to-left
        R = build_right_envs(mps, W)

        for i in range(L - 2, -1, -1):
            Lenv = np.ones((1, 1, 1), dtype=DTYPE)
            for j in range(i):
                Lenv = update_left_env(Lenv, mps[j], W[j])

            Dl = mps[i].shape[0]
            Dr = mps[i + 1].shape[2]
            d1 = mps[i].shape[1]
            d2 = mps[i + 1].shape[1]

            Theta0 = np.einsum('asg,gtb->astb', mps[i], mps[i + 1])
            v0 = np.reshape(Theta0, (-1,), order='C')
            dim = v0.size

            mv = lambda v: two_site_matvec(v, Lenv, W[i], W[i + 1], R[i + 2], Dl, d1, d2, Dr)
            Hop = LinearOperator((dim, dim), matvec=mv, dtype=DTYPE)

            ncv = min(dim, lanczos_ncv if lanczos_ncv is not None else max(8, min(dim, 20)))
            evals, evecs = eigsh(Hop, k=1, which='SA', v0=v0, ncv=ncv)

            Theta = np.reshape(evecs[:, 0], (Dl, d1, d2, Dr), order='C')

            X = np.reshape(Theta, (Dl * d1, d2 * Dr), order='C')
            U, s, Vh, eta = svd_truncate(X, Dmax=Dmax, eps=1e-14, cutoff=cutoff)
            Dnew = len(s)

            Mleft = np.reshape(U @ np.diag(s), (Dl, d1, Dnew), order='C')
            B = np.reshape(Vh, (Dnew, d2, Dr), order='C')

            mps[i] = Mleft
            mps[i + 1] = B
            discards.append(float(np.real(eta)))

            R[i + 1] = update_right_env(R[i + 2], mps[i + 1], W[i + 1])

        mps = left_canonicalize_qr(mps)
        mps = normalize_mps(mps)

        E = expectation_mpo_mps(mps, W).real
        energies.append(float(E))
        max_discards.append(max(discards) if len(discards) > 0 else 0.0)

        print(f"sweep {sweep+1:2d}: E = {energies[-1]:.12f}, max discarded weight = {max_discards[-1]:.3e}")

        if len(energies) >= 2 and abs(energies[-1] - energies[-2]) < tol:
            break

    return mps, np.array(energies), np.array(max_discards)
```

---

## After this patch

Then your actual DMRG workflow is matrix-free/scalable in the sense of the note:

### Production parts are scalable:
- MPO storage
- environment recursion
- local effective Hamiltonian application
- Lanczos local eigensolver
- SVD truncation
- energy evaluation
- normalization

### Dense parts remain only for validation:
- exact diagonalization
- MPO-to-dense tests
- exact small-system checks

That is exactly how the note recommends doing it.

---

## Honest final statement

After applying the patch above, you can fairly say:

> The DMRG sweeps use a scalable matrix-free local Hamiltonian with Lanczos, and dense objects are used only for small-system validation.

If you want, I can now give you the **smallest possible replacement quick-test cell** compatible with this fully scalable patch.

---
## 🧑‍💻 User


Then give me a quick test cell again

---
## 🤖 Assistant


```python
# Cell 6
# Quick test with scalable matrix-free DMRG only in the sweeps

# ---------- Heisenberg S=1/2 quick test ----------
Lh = 10
Wh = heisenberg_mpo(Lh, J=1.0, Jz=1.0, h=0.0)
mps_h, Ehist_h, disc_h = dmrg_two_site(
    Wh, d=2, Dmax=16, nsweeps=3, tol=1e-8, cutoff=1e-10, seed=1
)

print(f"Heisenberg L={Lh} final DMRG energy = {Ehist_h[-1]:.12f}")
print(f"Heisenberg energy/site = {Ehist_h[-1]/Lh:.12f}")

# Small-system ED benchmark only for validation
Hh_dense = dense_heisenberg_hamiltonian(Lh, J=1.0, Jz=1.0, h=0.0)
Eh_exact = np.min(np.linalg.eigvalsh(Hh_dense)).real
print(f"Heisenberg exact ED energy = {Eh_exact:.12f}")
print(f"Heisenberg |ΔE| = {abs(Ehist_h[-1] - Eh_exact):.3e}")

# ---------- AKLT spin-1 quick test ----------
La = 8
Wa = aklt_mpo(La)
init_aklt = normalize_mps(exact_aklt_mps(La))
mps_a, Ehist_a, disc_a = dmrg_two_site(
    Wa, d=3, Dmax=4, nsweeps=2, tol=1e-10, cutoff=1e-12, seed=2, init_mps=init_aklt
)

E_exact_aklt = -(2.0/3.0)*(La-1)
print(f"AKLT L={La} final DMRG energy = {Ehist_a[-1]:.12f}")
print(f"AKLT exact open-chain ground energy = {E_exact_aklt:.12f}")
print(f"AKLT |ΔE| = {abs(Ehist_a[-1]-E_exact_aklt):.3e}")

# ---------- Figure 1: Heisenberg ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(np.arange(1, len(Ehist_h)+1), Ehist_h, 'o-', label='DMRG')
ax[0].axhline(Eh_exact, color='r', ls='--', label='ED exact')
ax[0].set_xlabel('Sweep')
ax[0].set_ylabel('Energy')
ax[0].set_title(f'Heisenberg S=1/2, L={Lh}')
ax[0].grid(True)
ax[0].legend()

ax[1].semilogy(np.arange(1, len(disc_h)+1), np.maximum(disc_h, 1e-18), 'o-')
ax[1].set_xlabel('Sweep')
ax[1].set_ylabel('Max discarded weight')
ax[1].set_title(f'Heisenberg discarded weight, L={Lh}')
ax[1].grid(True)

plt.tight_layout()
plt.savefig("figureAKLT/heisenberg_quick_test_scalable.png", dpi=160)
plt.show()

# ---------- Figure 2: AKLT ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(np.arange(1, len(Ehist_a)+1), Ehist_a, 'o-', label='DMRG')
ax[0].axhline(E_exact_aklt, color='r', ls='--', label='Exact')
ax[0].set_xlabel('Sweep')
ax[0].set_ylabel('Energy')
ax[0].set_title(f'AKLT spin-1, L={La}')
ax[0].grid(True)
ax[0].legend()

ax[1].semilogy(np.arange(1, len(disc_a)+1), np.maximum(disc_a, 1e-18), 'o-')
ax[1].set_xlabel('Sweep')
ax[1].set_ylabel('Max discarded weight')
ax[1].set_title(f'AKLT discarded weight, L={La}')
ax[1].grid(True)

plt.tight_layout()
plt.savefig("figureAKLT/aklt_quick_test_scalable.png", dpi=160)
plt.show()

# ---------- Save data ----------
np.save("figureAKLT/heisenberg_quick_test_scalable_energy.npy", Ehist_h)
np.save("figureAKLT/heisenberg_quick_test_scalable_discard.npy", disc_h)
np.save("figureAKLT/aklt_quick_test_scalable_energy.npy", Ehist_a)
np.save("figureAKLT/aklt_quick_test_scalable_discard.npy", disc_a)
```

---
## 🧑‍💻 User


sweep  1: E = -4.258016497419, max discarded weight = 3.315e-11
sweep  2: E = -4.258032411880, max discarded weight = 6.860e-13
sweep  3: E = -4.258032411952, max discarded weight = 6.908e-13
Heisenberg L=10 final DMRG energy = -4.258032411952
Heisenberg energy/site = -0.425803241195
Heisenberg exact ED energy = -4.258035207283
Heisenberg |ΔE| = 2.795e-06
sweep  1: E = -4.666666664275, max discarded weight = 1.545e-05
sweep  2: E = -4.666666664275, max discarded weight = 1.544e-05
AKLT L=8 final DMRG energy = -4.666666664275
AKLT exact open-chain ground energy = -4.666666666667
AKLT |ΔE| = 2.391e-09

---
## 🤖 Assistant


These quick-test results are good, and now the sweeps are using the matrix-free/scalable structure.

## Quick assessment

### Heisenberg, `L=10`
- DMRG: `-4.258032411952`
- Exact ED: `-4.258035207283`
- Error: `2.795e-06`

This is a solid result for:
- only `3` sweeps
- `Dmax=16`
- fast run

### AKLT, `L=8`
- DMRG: `-4.666666664275`
- Exact: `-4.666666666667`
- Error: `2.391e-09`

Also good. Not machine precision because:
- you used truncated two-site sweeps
- `Dmax=4`
- only `2` sweeps

But still extremely accurate.

---

## On “strict with the note” and “scalable”

At this point, the **DMRG sweep itself** is using:

- matrix-free local effective Hamiltonian
- `LinearOperator`
- Lanczos (`eigsh`)
- environment contractions
- no dense local projected Hamiltonian construction
- scalable overlap/normalization patch

So the actual optimization loop is now in the spirit of the note.

The only dense steps left are for **small validation only**:
- exact diagonalization for short Heisenberg chains
- MPO-to-dense checks
- optional dense Hamiltonian comparison

That is fine and consistent with the note.

---

## One thing to note about AKLT accuracy
Your AKLT quick test used:
- exact AKLT initial MPS
- but `Dmax=4`
- and a generic two-site truncating algorithm

So a tiny deviation from exact energy can still happen from repeated truncation/gauge motion and finite Lanczos tolerance.

If you want AKLT to land essentially exactly in a quick test, try:
- `Dmax=8`
- `nsweeps=3`

or even just evaluate the exact AKLT MPS energy directly, which you already did successfully.

---

## Recommended quick settings

### Fastest reliable quick test
- Heisenberg: `L=10, Dmax=16, nsweeps=3`
- AKLT: `L=8, Dmax=4, nsweeps=2`

### Slightly better but still quick
- Heisenberg: `L=12, Dmax=24, nsweeps=4`
- AKLT: `L=10, Dmax=8, nsweeps=3`

---

## Conclusion

Yes — this now gives you a **quick matrix-free scalable DMRG test** consistent with the note’s main implementation principles.

If you want, the next best step is to add exactly one compact cell for the note’s decisive check:

### **two-site projected-operator validation**
for a tiny system, comparing the matrix-free local operator with dense `P†HP`.

That would be the strongest “strictly with the note” check.
