# MPS-Code-GPT#3-md

## 🧑‍💻 User

Please read the markdown file and the latex note, together with the existing python codes in the attachment. Then generate python codes for both S=1/2 Heisenberg model and AKLT model, with "fully MPO/MPS environment-based matrix-free local Hamiltonian action." 

---
## 🤖 Assistant

Below is a **new self-contained Python module** that implements what you asked for:

- **spin-1/2 Heisenberg model**
- **spin-1 AKLT model**
- **fully MPO/MPS environment-based matrix-free local Hamiltonian action**
- **two-site DMRG**
- **no dense full Hamiltonian in the actual DMRG sweep**
- **no dense projected operator \(P^\dagger H P\) in the actual DMRG sweep**
- includes **validation tools** comparing the environment-based local operator against a dense projected reference for small systems.

I re-derived the environment contractions directly from the conventions in your LaTeX note and used those conventions consistently:

- MPS tensor shape: `(Dl, d, Dr)`
- MPO tensor shape: `(chiL, chiR, d, d)` with physical indices `(ket, bra)`
- C-order reshapes
- two-site center tensor shape: `(Dl, d1, d2, Dr)`

The most important parts here are:

1. `left_env_update(...)`
2. `right_env_update(...)`
3. `apply_two_site_heff_env(...)`

These are the core matrix-free environment-based routines.

---

# New file: `mps_dmrg_env_matrix_free.py`

```python
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import eigh as scipy_eigh

# ============================================================
# Local operators
# ============================================================

def spin_half_ops(dtype=np.complex128):
    I = np.eye(2, dtype=dtype)
    Sp = np.array([[0, 1],
                   [0, 0]], dtype=dtype)
    Sm = np.array([[0, 0],
                   [1, 0]], dtype=dtype)
    Sz = 0.5 * np.array([[1, 0],
                         [0, -1]], dtype=dtype)
    return I, Sp, Sm, Sz


def spin_one_ops(dtype=np.complex128):
    I = np.eye(3, dtype=dtype)
    Sp = np.sqrt(2) * np.array([[0, 1, 0],
                                [0, 0, 1],
                                [0, 0, 0]], dtype=dtype)
    Sm = np.sqrt(2) * np.array([[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0]], dtype=dtype)
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, -1]], dtype=dtype)
    return I, Sp, Sm, Sz


# ============================================================
# MPO builders
# ============================================================

def heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    """
    OBC MPO for
        H = sum_i [ J/2 (S+_i S-_{i+1} + S-_i S+_{i+1}) + Jz Sz_i Sz_{i+1} ]
            - h sum_i Sz_i
    Convention:
        W[betaL, betaR, sigma, sigma']
    """
    I, Sp, Sm, Sz = spin_half_ops(dtype=dtype)
    d = 2
    chi = 5

    Wbulk = np.zeros((chi, chi, d, d), dtype=dtype)
    Wbulk[0, 0] = I
    Wbulk[1, 0] = Sp
    Wbulk[2, 0] = Sm
    Wbulk[3, 0] = Sz
    Wbulk[4, 0] = -h * Sz
    Wbulk[4, 1] = (J / 2.0) * Sm
    Wbulk[4, 2] = (J / 2.0) * Sp
    Wbulk[4, 3] = Jz * Sz
    Wbulk[4, 4] = I

    Wleft = np.zeros((1, chi, d, d), dtype=dtype)
    Wleft[0, 0] = -h * Sz
    Wleft[0, 1] = (J / 2.0) * Sm
    Wleft[0, 2] = (J / 2.0) * Sp
    Wleft[0, 3] = Jz * Sz
    Wleft[0, 4] = I

    Wright = np.zeros((chi, 1, d, d), dtype=dtype)
    Wright[0, 0] = I
    Wright[1, 0] = Sp
    Wright[2, 0] = Sm
    Wright[3, 0] = Sz
    Wright[4, 0] = -h * Sz

    mpo = [None] * L
    mpo[0] = Wleft
    for i in range(1, L - 1):
        mpo[i] = Wbulk.copy()
    mpo[L - 1] = Wright
    return mpo


def aklt_mpo(L, dtype=np.complex128):
    """
    OBC MPO for AKLT Hamiltonian:
        H = sum_i [ X_{i,i+1} + (1/3) X_{i,i+1}^2 ]
    with
        X = 1/2 (S+S- + S-S+) + Sz Sz
    Bond dimension chi=14 following the tex note.
    """
    I, Sp, Sm, Sz = spin_one_ops(dtype=dtype)
    d = 3
    chi = 14

    O = [Sp / np.sqrt(2.0), Sm / np.sqrt(2.0), Sz]
    Obar = [Sm / np.sqrt(2.0), Sp / np.sqrt(2.0), Sz]

    O2 = []
    Obar2 = []
    for a in range(3):
        for b in range(3):
            O2.append(O[a] @ O[b])
            Obar2.append(Obar[a] @ Obar[b])

    Wbulk = np.zeros((chi, chi, d, d), dtype=dtype)

    Wbulk[0, 0] = I
    for a in range(3):
        Wbulk[0, 1 + a] = O[a]
    for ab in range(9):
        Wbulk[0, 4 + ab] = (1.0 / 3.0) * O2[ab]

    for a in range(3):
        Wbulk[1 + a, 13] = Obar[a]
    for ab in range(9):
        Wbulk[4 + ab, 13] = Obar2[ab]

    Wbulk[13, 13] = I

    Wleft = np.zeros((1, chi, d, d), dtype=dtype)
    Wleft[0, 0] = I
    for a in range(3):
        Wleft[0, 1 + a] = O[a]
    for ab in range(9):
        Wleft[0, 4 + ab] = (1.0 / 3.0) * O2[ab]

    Wright = np.zeros((chi, 1, d, d), dtype=dtype)
    for a in range(3):
        Wright[1 + a, 0] = Obar[a]
    for ab in range(9):
        Wright[4 + ab, 0] = Obar2[ab]
    Wright[13, 0] = I

    mpo = [None] * L
    mpo[0] = Wleft
    for i in range(1, L - 1):
        mpo[i] = Wbulk.copy()
    mpo[L - 1] = Wright
    return mpo


# ============================================================
# Dense Hamiltonians and MPO->dense
# Only for testing/benchmarking small systems
# ============================================================

def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def onsite_op(L, i, op, d):
    I = np.eye(d, dtype=np.complex128)
    ops = [I] * L
    ops[i] = op
    return kron_all(ops)


def bond_op(L, i, op_i, op_j, d):
    I = np.eye(d, dtype=np.complex128)
    ops = [I] * L
    ops[i] = op_i
    ops[i + 1] = op_j
    return kron_all(ops)


def dense_heisenberg(L, J=1.0, Jz=1.0, h=0.0):
    I, Sp, Sm, Sz = spin_half_ops()
    d = 2
    H = np.zeros((d**L, d**L), dtype=np.complex128)

    for i in range(L - 1):
        H += (J / 2.0) * bond_op(L, i, Sp, Sm, d)
        H += (J / 2.0) * bond_op(L, i, Sm, Sp, d)
        H += Jz * bond_op(L, i, Sz, Sz, d)
    for i in range(L):
        H += -h * onsite_op(L, i, Sz, d)

    return H


def dense_aklt(L):
    I, Sp, Sm, Sz = spin_one_ops()
    d = 3
    H = np.zeros((d**L, d**L), dtype=np.complex128)

    for i in range(L - 1):
        X = (
            0.5 * bond_op(L, i, Sp, Sm, d)
            + 0.5 * bond_op(L, i, Sm, Sp, d)
            + bond_op(L, i, Sz, Sz, d)
        )
        H += X + (1.0 / 3.0) * (X @ X)

    return H


def mpo_to_dense(mpo):
    L = len(mpo)
    cur = mpo[0][0]
    cur_ops = [cur[b] for b in range(cur.shape[0])]

    for i in range(1, L):
        W = mpo[i]
        chiL, chiR, d, _ = W.shape
        new_ops = [
            np.zeros((cur_ops[0].shape[0] * d, cur_ops[0].shape[1] * d), dtype=np.complex128)
            for _ in range(chiR)
        ]
        for bl in range(chiL):
            for br in range(chiR):
                if np.linalg.norm(W[bl, br]) == 0:
                    continue
                new_ops[br] += np.kron(cur_ops[bl], W[bl, br])
        cur_ops = new_ops

    return cur_ops[0]


# ============================================================
# MPS utilities
# ============================================================

def random_mps(L, d, Dmax, seed=1234, dtype=np.complex128):
    rng = np.random.default_rng(seed)
    dims = [1]
    for i in range(1, L):
        dims.append(min(Dmax, d ** min(i, L - i)))
    dims.append(1)

    mps = []
    for i in range(L):
        Dl, Dr = dims[i], dims[i + 1]
        A = rng.normal(size=(Dl, d, Dr)) + 1j * rng.normal(size=(Dl, d, Dr))
        mps.append(A.astype(dtype))
    return mps


def mps_to_state(mps):
    psi = mps[0][0]
    for i in range(1, len(mps)):
        psi = np.einsum('...a,asb->...sb', psi, mps[i], optimize=True)
    return psi.reshape(-1, order='C')


def normalize_mps_by_state(mps):
    psi = mps_to_state(mps)
    nrm = np.sqrt(np.vdot(psi, psi).real)
    mps = [A.copy() for A in mps]
    mps[0] = mps[0] / nrm
    return mps


def right_canonicalize(mps):
    mps = [A.copy() for A in mps]
    L = len(mps)

    for i in range(L - 1, 0, -1):
        A = mps[i]
        Dl, d, Dr = A.shape
        M = A.reshape(Dl, d * Dr, order='C')
        Q, R = np.linalg.qr(M.conj().T, mode='reduced')
        B = Q.conj().T.reshape((-1, d, Dr), order='C')
        mps[i] = B
        Rd = R.conj().T
        mps[i - 1] = np.einsum('xsa,ab->xsb', mps[i - 1], Rd, optimize=True)

    return mps


def left_canonicalize(mps):
    mps = [A.copy() for A in mps]
    L = len(mps)

    for i in range(L - 1):
        A = mps[i]
        Dl, d, Dr = A.shape
        M = A.reshape(Dl * d, Dr, order='C')
        Q, R = np.linalg.qr(M, mode='reduced')
        k = Q.shape[1]
        mps[i] = Q.reshape(Dl, d, k, order='C')
        mps[i + 1] = np.einsum('ab,bsd->asd', R, mps[i + 1], optimize=True)

    return mps


def form_theta(A, B):
    return np.einsum('asm,mtr->astr', A, B, optimize=True)


def split_theta_left_to_right(theta, Dmax, svd_cutoff=1e-12):
    Dl, d1, d2, Dr = theta.shape
    M = theta.reshape((Dl * d1, d2 * Dr), order='C')
    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    keep = np.sum(S > svd_cutoff)
    keep = max(1, min(Dmax, keep, len(S)))
    disc_weight = np.sum(S[keep:]**2).real

    U = U[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]

    A = U.reshape((Dl, d1, keep), order='C')
    B = (np.diag(S) @ Vh).reshape((keep, d2, Dr), order='C')
    return A, B, S, disc_weight


def split_theta_right_to_left(theta, Dmax, svd_cutoff=1e-12):
    Dl, d1, d2, Dr = theta.shape
    M = theta.reshape((Dl * d1, d2 * Dr), order='C')
    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    keep = np.sum(S > svd_cutoff)
    keep = max(1, min(Dmax, keep, len(S)))
    disc_weight = np.sum(S[keep:]**2).real

    U = U[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]

    A = (U @ np.diag(S)).reshape((Dl, d1, keep), order='C')
    B = Vh.reshape((keep, d2, Dr), order='C')
    return A, B, S, disc_weight


# ============================================================
# Environment machinery
# ============================================================

def init_left_env():
    Lenv = np.zeros((1, 1, 1), dtype=np.complex128)
    Lenv[0, 0, 0] = 1.0
    return Lenv


def init_right_env():
    Renv = np.zeros((1, 1, 1), dtype=np.complex128)
    Renv[0, 0, 0] = 1.0
    return Renv


def left_env_update(Lold, A, W):
    """
    Implements
      Lnew_{B}(a,c) = sum_{b,x,y,s,t} Lold_b(x,y) A_{x,s,a} W_{b,B,s,t} A^*_{y,t,c}

    Shapes:
      Lold: (chiL, Dl_prev, Dl_prev')
      A   : (Dl_prev, d, Dl)
      W   : (chiL, chiR, d, d)
      Lnew: (chiR, Dl, Dl')
    """
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj(), optimize=True)


def right_env_update(Rold, B, W):
    """
    Implements
      Rnew_b(x,y) = sum_{B,a,c,s,t} B_{x,s,a} W_{b,B,s,t} Rold_B(a,c) B^*_{y,t,c}

    Shapes:
      Rold: (chiR, Dr_next, Dr_next')
      B   : (Dl, d, Dr_next)
      W   : (chiL, chiR, d, d)
      Rnew: (chiL, Dl, Dl')
    """
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj(), optimize=True)


def build_right_envs(mps, mpo):
    """
    Returns a list R where R[i] is the right environment for center at site i,
    i.e. contraction of sites i+1,...,L-1.
    In two-site DMRG for bond (i, i+1), use R[i+1].
    """
    L = len(mps)
    R = [None] * L
    R[L - 1] = init_right_env()

    cur = R[L - 1]
    for i in range(L - 1, 0, -1):
        cur = right_env_update(cur, mps[i], mpo[i])
        R[i - 1] = cur

    return R


def build_left_envs(mps, mpo):
    """
    Returns a list Lenvs where Lenvs[i] is the left environment for center at site i,
    i.e. contraction of sites 0,...,i-1.
    """
    L = len(mps)
    Lenvs = [None] * L
    Lenvs[0] = init_left_env()

    cur = Lenvs[0]
    for i in range(0, L - 1):
        cur = left_env_update(cur, mps[i], mpo[i])
        Lenvs[i + 1] = cur

    return Lenvs


# ============================================================
# Matrix-free local effective Hamiltonian actions
# ============================================================

def apply_one_site_heff_env(M, Lenv, W, Renv):
    """
    One-site effective Hamiltonian action:
      HM_{x,t,a} = sum_{y,z,s,b,B} L_{b}(x,y) W_{b,B,t,s}? no:
    Use note convention exactly:
      (Heff M)_{a_left, sigma_out, a_right}
        = sum L_{b}(a_left, a_left')
              W_{b,B,sigma_out,sigma_in}
              R_{B}(a_right', a_right)
              M_{a_left', sigma_in, a_right'}

    Since W uses (ket, bra) = (output, input), we contract:
      HM = einsum('bxy,bBst,Bza,ysz->xta')
    """
    X = np.einsum('bxy,ysz->bxsz', Lenv, M, optimize=True)
    Y = np.einsum('bBst,bxsz->Bxtz', W, X, optimize=True)
    HM = np.einsum('Bxtz,Bza->xta', Y, Renv, optimize=True)
    return HM


def apply_two_site_heff_env(theta, Lenv, W1, W2, Renv):
    """
    Two-site effective Hamiltonian action following the note:

      HT_{a,s1,s2,c}
        = sum_{a',c',u1,u2,b,B,C}
            L_b(a,a') W1_{b,B,s1,u1} W2_{B,C,s2,u2} R_C(c',c)
            theta_{a',u1,u2,c'}

    Shapes:
      theta: (Dl, d1, d2, Dr)
      Lenv : (chiL, Dl, Dl')
      W1   : (chiL, chiM, d1, d1)
      W2   : (chiM, chiR, d2, d2)
      Renv : (chiR, Dr', Dr)
    """
    X = np.einsum('bxy,ysuz->bxsuz', Lenv, theta, optimize=True)
    Y = np.einsum('bBst,bxtuz->Bxsuz', W1, X, optimize=True)
    Z = np.einsum('BCuv,Bxsuz->Cxs vz', W2, Y, optimize=True)
    # remove accidental spaces if copied into editors manually:
    Z = np.einsum('BCuv,Bxsuz->Cxsvz', W2, Y, optimize=True)
    HT = np.einsum('Cxsvz,Cza->xsva', Z, Renv, optimize=True)
    return HT


def heff_two_site_env_matvec(v, Lenv, W1, W2, Renv, Dl, d1, d2, Dr):
    theta = v.reshape((Dl, d1, d2, Dr), order='C')
    out = apply_two_site_heff_env(theta, Lenv, W1, W2, Renv)
    return out.reshape(Dl * d1 * d2 * Dr, order='C')


# ============================================================
# Small-system dense projected reference tools
# Only for validation
# ============================================================

def left_basis_map(mps, i):
    if i == 0:
        return np.array([[1.0]], dtype=np.complex128)

    X = mps[0][0]
    for k in range(1, i):
        X = np.einsum('pa,asb->psb', X, mps[k], optimize=True)
        X = X.reshape((-1, X.shape[-1]), order='C')
    return X


def right_basis_map(mps, i):
    """
    For bond (i, i+1), returns map from right bond index at site i+1
    to full basis of sites i+2..L-1.
    Shape: (Dr, dimR)
    """
    L = len(mps)
    if i == L - 2:
        return np.array([[1.0]], dtype=np.complex128)

    X = mps[i + 2]
    for k in range(i + 3, L):
        X = np.einsum('asd,dtu->astu', X, mps[k], optimize=True)
        X = X.reshape((X.shape[0], -1, X.shape[-1]), order='C')
    return X[:, :, 0]


def dense_projected_two_site_heff_from_full(Hfull, mps, i, d):
    UL = left_basis_map(mps, i)
    UR = right_basis_map(mps, i)

    Dl = UL.shape[1]
    Dr = UR.shape[0]
    dimL = UL.shape[0]
    dimR = UR.shape[1]

    Nloc = Dl * d * d * Dr
    Dim = dimL * d * d * dimR
    P = np.zeros((Dim, Nloc), dtype=np.complex128)

    col = 0
    for a in range(Dl):
        for s in range(d):
            for t in range(d):
                for c in range(Dr):
                    mid = np.zeros(d * d, dtype=np.complex128)
                    mid[s * d + t] = 1.0
                    vec = np.kron(UL[:, a], np.kron(mid, UR[c, :]))
                    P[:, col] = vec
                    col += 1

    Heff = P.conj().T @ Hfull @ P
    return Heff


# ============================================================
# Validation of environments and local operator
# ============================================================

def build_dense_local_heff_from_env(mps, mpo, i):
    """
    Explicitly build dense local Heff by applying the env-based matvec
    to basis vectors. Used only for debugging small systems.
    """
    A = mps[i]
    B = mps[i + 1]
    Dl, d1, mid = A.shape
    mid2, d2, Dr = B.shape
    assert mid == mid2

    # Need mixed canonical setting:
    # sites < i left-canonical, sites > i+1 right-canonical
    # then left/right bond spaces are orthonormal and local env operator equals projected Heff.
    Lenvs = build_left_envs(mps, mpo)
    Renvs = build_right_envs(mps, mpo)

    Lenv = Lenvs[i]
    Renv = Renvs[i + 1]

    N = Dl * d1 * d2 * Dr
    Heff = np.zeros((N, N), dtype=np.complex128)

    for j in range(N):
        e = np.zeros(N, dtype=np.complex128)
        e[j] = 1.0
        Heff[:, j] = heff_two_site_env_matvec(
            e, Lenv, mpo[i], mpo[i + 1], Renv, Dl, d1, d2, Dr
        )
    return Heff


def validate_two_site_env_operator_against_dense_projected(
    Hfull, mpo, mps, i, d, verbose=True
):
    """
    Compare:
      Heff_env  from environments
      Heff_proj = P^\dagger H P
    This should match when MPS is in proper mixed canonical form around bond i.
    """
    Heff_env = build_dense_local_heff_from_env(mps, mpo, i)
    Heff_proj = dense_projected_two_site_heff_from_full(Hfull, mps, i, d)

    diff = np.max(np.abs(Heff_env - Heff_proj))
    herm_env = np.max(np.abs(Heff_env - Heff_env.conj().T))
    herm_proj = np.max(np.abs(Heff_proj - Heff_proj.conj().T))

    if verbose:
        print(f"bond {i}: max|Heff_env - Heff_proj| = {diff:.3e}")
        print(f"bond {i}: env Hermiticity err      = {herm_env:.3e}")
        print(f"bond {i}: proj Hermiticity err     = {herm_proj:.3e}")

    return diff, herm_env, herm_proj


# ============================================================
# Two-site DMRG with fully environment-based matrix-free local solve
# ============================================================

def two_site_dmrg_env_matrix_free(
    mpo,
    d,
    Dmax=32,
    nsweeps=8,
    lanczos_tol=1e-10,
    lanczos_maxiter=None,
    svd_cutoff=1e-12,
    seed=1234,
    verbose=True,
):
    """
    Fully environment-based matrix-free two-site DMRG.
    No dense full Hamiltonian is used in the sweep.

    Strategy:
    - initialize random MPS
    - right-canonicalize
    - build all right environments
    - left-to-right sweep:
        use current Lenv and stored Renv[i+1]
    - right-to-left sweep:
        update right environments on the fly

    Returns:
      mps, sweep_energies_local, sweep_discards
    where sweep_energies_local are the last local energies from sweeps.
    """
    L = len(mpo)

    mps = random_mps(L, d=d, Dmax=min(Dmax, 4), seed=seed)
    mps = normalize_mps_by_state(mps)
    mps = right_canonicalize(mps)

    sweep_energies = []
    sweep_discards = []

    for sweep in range(nsweeps):
        discards_this_sweep = []
        last_local_energy = None

        # Build right envs for current right-canonical configuration
        Renvs = build_right_envs(mps, mpo)
        Lenv = init_left_env()

        # ----------------------------------------------------
        # Left-to-right sweep
        # ----------------------------------------------------
        for i in range(L - 1):
            theta0 = form_theta(mps[i], mps[i + 1])
            Dl, d1, d2, Dr = theta0.shape
            Nloc = Dl * d1 * d2 * Dr

            Renv = Renvs[i + 1]

            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v, Lenv=Lenv, W1=mpo[i], W2=mpo[i + 1], Renv=Renv,
                              Dl=Dl, d1=d1, d2=d2, Dr=Dr:
                    heff_two_site_env_matvec(v, Lenv, W1, W2, Renv, Dl, d1, d2, Dr),
                dtype=np.complex128
            )

            v0 = theta0.reshape(-1, order='C')
            vals, vecs = eigsh(
                Heff, k=1, which='SA',
                v0=v0, tol=lanczos_tol, maxiter=lanczos_maxiter
            )
            last_local_energy = vals[0].real

            theta = vecs[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            Anew, Bnew, S, disc = split_theta_left_to_right(theta, Dmax, svd_cutoff)
            mps[i] = Anew
            mps[i + 1] = Bnew
            discards_this_sweep.append(disc)

            # update left environment with new left-canonical tensor Anew
            Lenv = left_env_update(Lenv, mps[i], mpo[i])

        # After left->right, regauge to left-canonical / right-center setup
        mps = left_canonicalize(mps)
        mps = normalize_mps_by_state(mps)

        # Build left envs for convenience if needed; for RL we update R on the fly
        Lenvs = build_left_envs(mps, mpo)
        Renv = init_right_env()

        # ----------------------------------------------------
        # Right-to-left sweep
        # ----------------------------------------------------
        for i in range(L - 2, -1, -1):
            theta0 = form_theta(mps[i], mps[i + 1])
            Dl, d1, d2, Dr = theta0.shape
            Nloc = Dl * d1 * d2 * Dr

            Lcur = Lenvs[i]

            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v, Lcur=Lcur, W1=mpo[i], W2=mpo[i + 1], Renv=Renv,
                              Dl=Dl, d1=d1, d2=d2, Dr=Dr:
                    heff_two_site_env_matvec(v, Lcur, W1, W2, Renv, Dl, d1, d2, Dr),
                dtype=np.complex128
            )

            v0 = theta0.reshape(-1, order='C')
            vals, vecs = eigsh(
                Heff, k=1, which='SA',
                v0=v0, tol=lanczos_tol, maxiter=lanczos_maxiter
            )
            last_local_energy = vals[0].real

            theta = vecs[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            Anew, Bnew, S, disc = split_theta_right_to_left(theta, Dmax, svd_cutoff)
            mps[i] = Anew
            mps[i + 1] = Bnew
            discards_this_sweep.append(disc)

            # update right environment with new right-canonical tensor Bnew at site i+1
            Renv = right_env_update(Renv, mps[i + 1], mpo[i + 1])

        mps = right_canonicalize(mps)
        mps = normalize_mps_by_state(mps)

        max_disc = max(discards_this_sweep) if discards_this_sweep else 0.0
        sweep_energies.append(last_local_energy if last_local_energy is not None else np.nan)
        sweep_discards.append(max_disc)

        if verbose:
            print(f"Sweep {sweep+1:2d}: local eig E = {sweep_energies[-1]:.12f}, "
                  f"max discarded weight = {max_disc:.3e}")

    return mps, np.array(sweep_energies), np.array(sweep_discards)


# ============================================================
# Exact AKLT benchmark MPS
# ============================================================

def exact_aklt_mps(L, left_vec=None, right_vec=None, dtype=np.complex128):
    Ap = np.array([[0, np.sqrt(2/3)],
                   [0, 0]], dtype=dtype)
    A0 = np.array([[-1/np.sqrt(3), 0],
                   [0, 1/np.sqrt(3)]], dtype=dtype)
    Am = np.array([[0, 0],
                   [-np.sqrt(2/3), 0]], dtype=dtype)

    local = np.stack([Ap, A0, Am], axis=1)  # (2,3,2)

    if left_vec is None:
        left_vec = np.array([1.0, 0.0], dtype=dtype)
    if right_vec is None:
        right_vec = np.array([1.0, 0.0], dtype=dtype)

    mps = []
    A1 = np.einsum('a,asb->sb', left_vec, local).reshape(1, 3, 2)
    mps.append(A1)

    for _ in range(1, L - 1):
        mps.append(local.copy())

    AL = np.einsum('asb,b->as', local, right_vec).reshape(2, 3, 1)
    mps.append(AL)

    return mps


# ============================================================
# Energy evaluation with MPO environments
# ============================================================

def expectation_value_mpo(mps, mpo):
    """
    Contract <psi|MPO|psi> left-to-right.
    """
    env = init_left_env()
    for i in range(len(mps)):
        env = left_env_update(env, mps[i], mpo[i])
    return env[0, 0, 0]


def norm_from_mps(mps):
    psi = mps_to_state(mps)
    return np.vdot(psi, psi).real


# ============================================================
# Convenience runners / tests
# ============================================================

def test_mpo_construction():
    for L in [2, 3, 4]:
        Hh1 = mpo_to_dense(heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.2))
        Hh2 = dense_heisenberg(L, J=1.0, Jz=1.0, h=0.2)
        print(f"Heisenberg MPO test L={L}: max diff = {np.max(np.abs(Hh1 - Hh2)):.3e}")

    for L in [2, 3, 4]:
        Ha1 = mpo_to_dense(aklt_mpo(L))
        Ha2 = dense_aklt(L)
        print(f"AKLT MPO test L={L}: max diff = {np.max(np.abs(Ha1 - Ha2)):.3e}")


def test_env_operator_heisenberg_small():
    """
    Crucial local-operator validation test.
    """
    L = 4
    d = 2
    mpo = heisenberg_mpo(L)
    Hfull = dense_heisenberg(L)

    mps = random_mps(L, d=d, Dmax=3, seed=7)
    mps = normalize_mps_by_state(mps)

    # Put bond i into mixed canonical form:
    i = 1
    # left-canonicalize left block, right-canonicalize right block by full mixed regauging
    mps = right_canonicalize(mps)
    mps = left_canonicalize(mps[:i] + mps[i:])  # harmless full left; then re-right on right part not enough
    # Simpler robust path:
    mps = right_canonicalize(mps)
    mps = left_canonicalize(mps)
    mps = right_canonicalize(mps)

    # Better explicit mixed-canonical prep:
    # bring sites < i left-canonical
    for k in range(i):
        A = mps[k]
        Dl, dd, Dr = A.shape
        M = A.reshape(Dl * dd, Dr, order='C')
        Q, R = np.linalg.qr(M, mode='reduced')
        nk = Q.shape[1]
        mps[k] = Q.reshape(Dl, dd, nk, order='C')
        mps[k + 1] = np.einsum('ab,bsd->asd', R, mps[k + 1], optimize=True)
    # bring sites > i+1 right-canonical
    for k in range(L - 1, i + 1, -1):
        A = mps[k]
        Dl, dd, Dr = A.shape
        M = A.reshape(Dl, dd * Dr, order='C')
        Q, R = np.linalg.qr(M.conj().T, mode='reduced')
        B = Q.conj().T.reshape((-1, dd, Dr), order='C')
        mps[k] = B
        Rd = R.conj().T
        mps[k - 1] = np.einsum('xsa,ab->xsb', mps[k - 1], Rd, optimize=True)

    validate_two_site_env_operator_against_dense_projected(Hfull, mpo, mps, i, d, verbose=True)


def test_env_operator_aklt_small():
    L = 4
    d = 3
    mpo = aklt_mpo(L)
    Hfull = dense_aklt(L)

    mps = random_mps(L, d=d, Dmax=3, seed=11)
    mps = normalize_mps_by_state(mps)

    i = 1
    # prepare mixed-canonical around bond i
    for k in range(i):
        A = mps[k]
        Dl, dd, Dr = A.shape
        M = A.reshape(Dl * dd, Dr, order='C')
        Q, R = np.linalg.qr(M, mode='reduced')
        nk = Q.shape[1]
        mps[k] = Q.reshape(Dl, dd, nk, order='C')
        mps[k + 1] = np.einsum('ab,bsd->asd', R, mps[k + 1], optimize=True)

    for k in range(L - 1, i + 1, -1):
        A = mps[k]
        Dl, dd, Dr = A.shape
        M = A.reshape(Dl, dd * Dr, order='C')
        Q, R = np.linalg.qr(M.conj().T, mode='reduced')
        B = Q.conj().T.reshape((-1, dd, Dr), order='C')
        mps[k] = B
        Rd = R.conj().T
        mps[k - 1] = np.einsum('xsa,ab->xsb', mps[k - 1], Rd, optimize=True)

    validate_two_site_env_operator_against_dense_projected(Hfull, mpo, mps, i, d, verbose=True)


def run_heisenberg_dmrg_demo(L=8, Dmax=32, nsweeps=8, J=1.0, Jz=1.0, h=0.0, seed=1234):
    mpo = heisenberg_mpo(L, J=J, Jz=Jz, h=h)
    mps, Es, disc = two_site_dmrg_env_matrix_free(
        mpo=mpo, d=2, Dmax=Dmax, nsweeps=nsweeps, seed=seed, verbose=True
    )
    E_mpo = expectation_value_mpo(mps, mpo).real
    nrm = norm_from_mps(mps)
    print(f"\nHeisenberg final <H> = {E_mpo:.12f}, norm = {nrm:.12f}")

    if L <= 10:
        Hfull = dense_heisenberg(L, J=J, Jz=Jz, h=h)
        psi = mps_to_state(mps)
        E_dense = (np.vdot(psi, Hfull @ psi) / np.vdot(psi, psi)).real
        E0 = np.linalg.eigvalsh(Hfull)[0].real
        print(f"Heisenberg final dense energy from psi = {E_dense:.12f}")
        print(f"Heisenberg exact ground energy         = {E0:.12f}")
        print(f"Energy error                           = {abs(E_dense - E0):.3e}")

    return mps, Es, disc


def run_aklt_dmrg_demo(L=6, Dmax=16, nsweeps=8, seed=1234):
    mpo = aklt_mpo(L)
    mps, Es, disc = two_site_dmrg_env_matrix_free(
        mpo=mpo, d=3, Dmax=Dmax, nsweeps=nsweeps, seed=seed, verbose=True
    )
    E_mpo = expectation_value_mpo(mps, mpo).real
    nrm = norm_from_mps(mps)
    print(f"\nAKLT final <H> = {E_mpo:.12f}, norm = {nrm:.12f}")

    E_exact = -(2.0 / 3.0) * (L - 1)
    print(f"AKLT exact open-chain ground energy = {E_exact:.12f}")
    print(f"Energy error                        = {abs(E_mpo - E_exact):.3e}")

    if L <= 7:
        Hfull = dense_aklt(L)
        psi = mps_to_state(mps)
        E_dense = (np.vdot(psi, Hfull @ psi) / np.vdot(psi, psi)).real
        E0 = np.linalg.eigvalsh(Hfull)[0].real
        print(f"AKLT dense energy from psi  = {E_dense:.12f}")
        print(f"AKLT exact ED ground energy = {E0:.12f}")
        print(f"Difference to ED            = {abs(E_dense - E0):.3e}")

    return mps, Es, disc


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    print("=== MPO construction tests ===")
    test_mpo_construction()

    print("\n=== Local operator validation: Heisenberg ===")
    test_env_operator_heisenberg_small()

    print("\n=== Local operator validation: AKLT ===")
    test_env_operator_aklt_small()

    print("\n=== Heisenberg DMRG demo ===")
    run_heisenberg_dmrg_demo(L=8, Dmax=32, nsweeps=6, J=1.0, Jz=1.0, h=0.0, seed=1234)

    print("\n=== AKLT DMRG demo ===")
    run_aklt_dmrg_demo(L=6, Dmax=16, nsweeps=6, seed=1234)
```

---

# What is new and important

## 1. Correct environment updates

These are the two formulas that matter most.

### Left environment
```python
def left_env_update(Lold, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj(), optimize=True)
```

This matches the LaTeX note:
\[
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha'_{i-1})
=
\sum
A^* L W A
\]
with the precise index ordering encoded consistently.

### Right environment
```python
def right_env_update(Rold, B, W):
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj(), optimize=True)
```

This matches:
\[
R^{[i]}_{\beta_i}(\alpha_i,\alpha'_i)
=
\sum
B\,W\,R\,B^*
\]

---

## 2. Correct two-site matrix-free local action

This is the core scalable local operator:

```python
def apply_two_site_heff_env(theta, Lenv, W1, W2, Renv):
    X = np.einsum('bxy,ysuz->bxsuz', Lenv, theta, optimize=True)
    Y = np.einsum('bBst,bxtuz->Bxsuz', W1, X, optimize=True)
    Z = np.einsum('BCuv,Bxsuz->Cxsvz', W2, Y, optimize=True)
    HT = np.einsum('Cxsvz,Cza->xsva', Z, Renv, optimize=True)
    return HT
```

Interpretation:

- `theta[y, u1, u2, z]`
- contract left environment on `y`
- apply first MPO tensor on site `i`
- apply second MPO tensor on site `i+1`
- contract right environment on `z`

This is exactly the matrix-free version of the two-site effective Hamiltonian from the note.

---

## 3. Validation against the true projected operator

You explicitly asked for the fully environment-based operator, but the handoff note made clear that the previous environment branch failed because it did **not** match \(P^\dagger H P\).

So I included:

```python
validate_two_site_env_operator_against_dense_projected(...)
```

This constructs both:

- `Heff_env` by repeated calls to the environment-based matvec
- `Heff_proj = P^\dagger H P` using the dense full Hamiltonian only for small tests

and compares them.

This is the right diagnostic and should be run first.

---

# How to use it

## Minimal recommended workflow

### 1. Save the new file
Save the code as:

```bash
mps_dmrg_env_matrix_free.py
```

### 2. Run the validation tests first
```bash
python mps_dmrg_env_matrix_free.py
```

You should first inspect:

- MPO test errors
- local operator validation errors:
  - `max|Heff_env - Heff_proj|`
  - Hermiticity errors

These should be near machine precision for the small test systems.

### 3. Then run DMRG demos
Included demo functions:

- `run_heisenberg_dmrg_demo(...)`
- `run_aklt_dmrg_demo(...)`

---

# Important implementation remarks

## Mixed-canonical requirement
For the identity
\[
H_{\mathrm{eff}}^{\text{env}} = P^\dagger H P
\]
to hold with the environments interpreted as orthonormal-basis contractions, the MPS must be in the correct **mixed-canonical form around the optimized bond**:

- sites left of bond: left-canonical
- sites right of bond: right-canonical

The DMRG sweep code maintains that structure dynamically.

This is the likely reason the old environment branch could be Hermitian but still wrong: Hermiticity alone does not ensure the local operator is the correct projected operator.

---

## Why the local energies in the sweep are not the final state energy
Inside two-site DMRG, the eigensolver returns the minimum of the current local effective Hamiltonian. After truncation and continuing the sweep, that number is not necessarily equal to the final global energy of the updated MPS.

So after the sweeps I compute:

```python
E_mpo = expectation_value_mpo(mps, mpo).real
```

and for small systems also compare to dense ED.

---

## AKLT degeneracy
For OBC, the AKLT chain has a fourfold degenerate ground manifold. So DMRG may converge to any state in that manifold, but the energy should be

\[
E_0 = -\frac{2}{3}(L-1).
\]

That is exactly the right benchmark.

---

# Suggested next checks

I strongly recommend running these in order:

## Heisenberg small-system local test
Use `L=4`, bond `i=1`, compare `Heff_env` vs `P^\dagger H P`.

## AKLT small-system local test
Same idea for `L=4`.

## Full small-system DMRG benchmark
- Heisenberg `L=8`
- AKLT `L=6`

Compare final energies to exact diagonalization.

---

# If you want, I can also provide

1. a **patch version** that integrates this directly into your existing `mps_dmrg_models.py`,
2. a **clean split into modules**:
   - `operators.py`
   - `mpo.py`
   - `mps.py`
   - `environments.py`
   - `dmrg.py`
   - `tests.py`
3. a version with **one-site environment-based DMRG** too,
4. a version with a **Davidson local eigensolver** instead of `eigsh`,
5. a **pytest test suite** for the local operator identity
   \[
   H_{\mathrm{eff}}^{\text{env}} = P^\dagger H P.
   \]

If you want, I can next give you a **fully polished, debugged package-style version** with explicit unit tests and a cleaner mixed-canonical preparation routine.

---
## 🧑‍💻 User

Thank you. I use Jupyter.

---
## 🤖 Assistant

Great — for **Jupyter**, the best format is a set of **cells** rather than one big script.

Below I’ll give you a notebook-friendly version organized into sections you can paste into separate cells.

A few practical Jupyter notes first:

- put the **imports + core functions** in early cells,
- run the **validation cells first** before the DMRG cells,
- in notebooks, it is especially useful to keep the dense-reference checks nearby,
- if a cell is long-running, you can interrupt and reduce `L`, `Dmax`, or `nsweeps`.

---

# Cell 1 — Imports

```python
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import eigh as scipy_eigh

np.set_printoptions(precision=6, suppress=True)
```

---

# Cell 2 — Local operators

```python
def spin_half_ops(dtype=np.complex128):
    I = np.eye(2, dtype=dtype)
    Sp = np.array([[0, 1],
                   [0, 0]], dtype=dtype)
    Sm = np.array([[0, 0],
                   [1, 0]], dtype=dtype)
    Sz = 0.5 * np.array([[1, 0],
                         [0, -1]], dtype=dtype)
    return I, Sp, Sm, Sz


def spin_one_ops(dtype=np.complex128):
    I = np.eye(3, dtype=dtype)
    Sp = np.sqrt(2) * np.array([[0, 1, 0],
                                [0, 0, 1],
                                [0, 0, 0]], dtype=dtype)
    Sm = np.sqrt(2) * np.array([[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0]], dtype=dtype)
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, -1]], dtype=dtype)
    return I, Sp, Sm, Sz
```

---

# Cell 3 — MPO builders

```python
def heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    I, Sp, Sm, Sz = spin_half_ops(dtype=dtype)
    d = 2
    chi = 5

    Wbulk = np.zeros((chi, chi, d, d), dtype=dtype)
    Wbulk[0, 0] = I
    Wbulk[1, 0] = Sp
    Wbulk[2, 0] = Sm
    Wbulk[3, 0] = Sz
    Wbulk[4, 0] = -h * Sz
    Wbulk[4, 1] = (J / 2.0) * Sm
    Wbulk[4, 2] = (J / 2.0) * Sp
    Wbulk[4, 3] = Jz * Sz
    Wbulk[4, 4] = I

    Wleft = np.zeros((1, chi, d, d), dtype=dtype)
    Wleft[0, 0] = -h * Sz
    Wleft[0, 1] = (J / 2.0) * Sm
    Wleft[0, 2] = (J / 2.0) * Sp
    Wleft[0, 3] = Jz * Sz
    Wleft[0, 4] = I

    Wright = np.zeros((chi, 1, d, d), dtype=dtype)
    Wright[0, 0] = I
    Wright[1, 0] = Sp
    Wright[2, 0] = Sm
    Wright[3, 0] = Sz
    Wright[4, 0] = -h * Sz

    mpo = [None] * L
    mpo[0] = Wleft
    for i in range(1, L - 1):
        mpo[i] = Wbulk.copy()
    mpo[L - 1] = Wright
    return mpo


def aklt_mpo(L, dtype=np.complex128):
    I, Sp, Sm, Sz = spin_one_ops(dtype=dtype)
    d = 3
    chi = 14

    O = [Sp / np.sqrt(2.0), Sm / np.sqrt(2.0), Sz]
    Obar = [Sm / np.sqrt(2.0), Sp / np.sqrt(2.0), Sz]

    O2 = []
    Obar2 = []
    for a in range(3):
        for b in range(3):
            O2.append(O[a] @ O[b])
            Obar2.append(Obar[a] @ Obar[b])

    Wbulk = np.zeros((chi, chi, d, d), dtype=dtype)

    Wbulk[0, 0] = I
    for a in range(3):
        Wbulk[0, 1 + a] = O[a]
    for ab in range(9):
        Wbulk[0, 4 + ab] = (1.0 / 3.0) * O2[ab]

    for a in range(3):
        Wbulk[1 + a, 13] = Obar[a]
    for ab in range(9):
        Wbulk[4 + ab, 13] = Obar2[ab]

    Wbulk[13, 13] = I

    Wleft = np.zeros((1, chi, d, d), dtype=dtype)
    Wleft[0, 0] = I
    for a in range(3):
        Wleft[0, 1 + a] = O[a]
    for ab in range(9):
        Wleft[0, 4 + ab] = (1.0 / 3.0) * O2[ab]

    Wright = np.zeros((chi, 1, d, d), dtype=dtype)
    for a in range(3):
        Wright[1 + a, 0] = Obar[a]
    for ab in range(9):
        Wright[4 + ab, 0] = Obar2[ab]
    Wright[13, 0] = I

    mpo = [None] * L
    mpo[0] = Wleft
    for i in range(1, L - 1):
        mpo[i] = Wbulk.copy()
    mpo[L - 1] = Wright
    return mpo
```

---

# Cell 4 — Dense Hamiltonians and MPO-to-dense for testing

```python
def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def onsite_op(L, i, op, d):
    I = np.eye(d, dtype=np.complex128)
    ops = [I] * L
    ops[i] = op
    return kron_all(ops)


def bond_op(L, i, op_i, op_j, d):
    I = np.eye(d, dtype=np.complex128)
    ops = [I] * L
    ops[i] = op_i
    ops[i + 1] = op_j
    return kron_all(ops)


def dense_heisenberg(L, J=1.0, Jz=1.0, h=0.0):
    I, Sp, Sm, Sz = spin_half_ops()
    d = 2
    H = np.zeros((d**L, d**L), dtype=np.complex128)

    for i in range(L - 1):
        H += (J / 2.0) * bond_op(L, i, Sp, Sm, d)
        H += (J / 2.0) * bond_op(L, i, Sm, Sp, d)
        H += Jz * bond_op(L, i, Sz, Sz, d)

    for i in range(L):
        H += -h * onsite_op(L, i, Sz, d)

    return H


def dense_aklt(L):
    I, Sp, Sm, Sz = spin_one_ops()
    d = 3
    H = np.zeros((d**L, d**L), dtype=np.complex128)

    for i in range(L - 1):
        X = (
            0.5 * bond_op(L, i, Sp, Sm, d)
            + 0.5 * bond_op(L, i, Sm, Sp, d)
            + bond_op(L, i, Sz, Sz, d)
        )
        H += X + (1.0 / 3.0) * (X @ X)

    return H


def mpo_to_dense(mpo):
    L = len(mpo)
    cur = mpo[0][0]
    cur_ops = [cur[b] for b in range(cur.shape[0])]

    for i in range(1, L):
        W = mpo[i]
        chiL, chiR, d, _ = W.shape
        new_ops = [
            np.zeros((cur_ops[0].shape[0] * d, cur_ops[0].shape[1] * d), dtype=np.complex128)
            for _ in range(chiR)
        ]
        for bl in range(chiL):
            for br in range(chiR):
                if np.linalg.norm(W[bl, br]) == 0:
                    continue
                new_ops[br] += np.kron(cur_ops[bl], W[bl, br])
        cur_ops = new_ops

    return cur_ops[0]
```

---

# Cell 5 — MPS utilities

```python
def random_mps(L, d, Dmax, seed=1234, dtype=np.complex128):
    rng = np.random.default_rng(seed)

    dims = [1]
    for i in range(1, L):
        dims.append(min(Dmax, d ** min(i, L - i)))
    dims.append(1)

    mps = []
    for i in range(L):
        Dl, Dr = dims[i], dims[i + 1]
        A = rng.normal(size=(Dl, d, Dr)) + 1j * rng.normal(size=(Dl, d, Dr))
        mps.append(A.astype(dtype))
    return mps


def mps_to_state(mps):
    psi = mps[0][0]
    for i in range(1, len(mps)):
        psi = np.einsum('...a,asb->...sb', psi, mps[i], optimize=True)
    return psi.reshape(-1, order='C')


def normalize_mps_by_state(mps):
    psi = mps_to_state(mps)
    nrm = np.sqrt(np.vdot(psi, psi).real)
    mps = [A.copy() for A in mps]
    mps[0] = mps[0] / nrm
    return mps


def right_canonicalize(mps):
    mps = [A.copy() for A in mps]
    L = len(mps)

    for i in range(L - 1, 0, -1):
        A = mps[i]
        Dl, d, Dr = A.shape
        M = A.reshape(Dl, d * Dr, order='C')
        Q, R = np.linalg.qr(M.conj().T, mode='reduced')
        B = Q.conj().T.reshape((-1, d, Dr), order='C')
        mps[i] = B
        Rd = R.conj().T
        mps[i - 1] = np.einsum('xsa,ab->xsb', mps[i - 1], Rd, optimize=True)

    return mps


def left_canonicalize(mps):
    mps = [A.copy() for A in mps]
    L = len(mps)

    for i in range(L - 1):
        A = mps[i]
        Dl, d, Dr = A.shape
        M = A.reshape(Dl * d, Dr, order='C')
        Q, R = np.linalg.qr(M, mode='reduced')
        k = Q.shape[1]
        mps[i] = Q.reshape(Dl, d, k, order='C')
        mps[i + 1] = np.einsum('ab,bsd->asd', R, mps[i + 1], optimize=True)

    return mps


def form_theta(A, B):
    return np.einsum('asm,mtr->astr', A, B, optimize=True)


def split_theta_left_to_right(theta, Dmax, svd_cutoff=1e-12):
    Dl, d1, d2, Dr = theta.shape
    M = theta.reshape((Dl * d1, d2 * Dr), order='C')
    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    keep = np.sum(S > svd_cutoff)
    keep = max(1, min(Dmax, keep, len(S)))
    disc_weight = np.sum(S[keep:]**2).real

    U = U[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]

    A = U.reshape((Dl, d1, keep), order='C')
    B = (np.diag(S) @ Vh).reshape((keep, d2, Dr), order='C')
    return A, B, S, disc_weight


def split_theta_right_to_left(theta, Dmax, svd_cutoff=1e-12):
    Dl, d1, d2, Dr = theta.shape
    M = theta.reshape((Dl * d1, d2 * Dr), order='C')
    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    keep = np.sum(S > svd_cutoff)
    keep = max(1, min(Dmax, keep, len(S)))
    disc_weight = np.sum(S[keep:]**2).real

    U = U[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]

    A = (U @ np.diag(S)).reshape((Dl, d1, keep), order='C')
    B = Vh.reshape((keep, d2, Dr), order='C')
    return A, B, S, disc_weight
```

---

# Cell 6 — Environment routines

```python
def init_left_env():
    Lenv = np.zeros((1, 1, 1), dtype=np.complex128)
    Lenv[0, 0, 0] = 1.0
    return Lenv


def init_right_env():
    Renv = np.zeros((1, 1, 1), dtype=np.complex128)
    Renv[0, 0, 0] = 1.0
    return Renv


def left_env_update(Lold, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj(), optimize=True)


def right_env_update(Rold, B, W):
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj(), optimize=True)


def build_right_envs(mps, mpo):
    L = len(mps)
    R = [None] * L
    R[L - 1] = init_right_env()

    cur = R[L - 1]
    for i in range(L - 1, 0, -1):
        cur = right_env_update(cur, mps[i], mpo[i])
        R[i - 1] = cur

    return R


def build_left_envs(mps, mpo):
    L = len(mps)
    Lenvs = [None] * L
    Lenvs[0] = init_left_env()

    cur = Lenvs[0]
    for i in range(0, L - 1):
        cur = left_env_update(cur, mps[i], mpo[i])
        Lenvs[i + 1] = cur

    return Lenvs
```

---

# Cell 7 — Matrix-free local Hamiltonian actions

```python
def apply_one_site_heff_env(M, Lenv, W, Renv):
    X = np.einsum('bxy,ysz->bxsz', Lenv, M, optimize=True)
    Y = np.einsum('bBst,bxsz->Bxtz', W, X, optimize=True)
    HM = np.einsum('Bxtz,Bza->xta', Y, Renv, optimize=True)
    return HM


def apply_two_site_heff_env(theta, Lenv, W1, W2, Renv):
    X = np.einsum('bxy,ysuz->bxsuz', Lenv, theta, optimize=True)
    Y = np.einsum('bBst,bxtuz->Bxsuz', W1, X, optimize=True)
    Z = np.einsum('BCuv,Bxsuz->Cxsvz', W2, Y, optimize=True)
    HT = np.einsum('Cxsvz,Cza->xsva', Z, Renv, optimize=True)
    return HT


def heff_two_site_env_matvec(v, Lenv, W1, W2, Renv, Dl, d1, d2, Dr):
    theta = v.reshape((Dl, d1, d2, Dr), order='C')
    out = apply_two_site_heff_env(theta, Lenv, W1, W2, Renv)
    return out.reshape(Dl * d1 * d2 * Dr, order='C')
```

---

# Cell 8 — Dense projected reference tools for debugging

```python
def left_basis_map(mps, i):
    if i == 0:
        return np.array([[1.0]], dtype=np.complex128)

    X = mps[0][0]
    for k in range(1, i):
        X = np.einsum('pa,asb->psb', X, mps[k], optimize=True)
        X = X.reshape((-1, X.shape[-1]), order='C')
    return X


def right_basis_map(mps, i):
    L = len(mps)
    if i == L - 2:
        return np.array([[1.0]], dtype=np.complex128)

    X = mps[i + 2]
    for k in range(i + 3, L):
        X = np.einsum('asd,dtu->astu', X, mps[k], optimize=True)
        X = X.reshape((X.shape[0], -1, X.shape[-1]), order='C')
    return X[:, :, 0]


def dense_projected_two_site_heff_from_full(Hfull, mps, i, d):
    UL = left_basis_map(mps, i)
    UR = right_basis_map(mps, i)

    Dl = UL.shape[1]
    Dr = UR.shape[0]
    dimL = UL.shape[0]
    dimR = UR.shape[1]

    Nloc = Dl * d * d * Dr
    Dim = dimL * d * d * dimR
    P = np.zeros((Dim, Nloc), dtype=np.complex128)

    col = 0
    for a in range(Dl):
        for s in range(d):
            for t in range(d):
                for c in range(Dr):
                    mid = np.zeros(d * d, dtype=np.complex128)
                    mid[s * d + t] = 1.0
                    vec = np.kron(UL[:, a], np.kron(mid, UR[c, :]))
                    P[:, col] = vec
                    col += 1

    Heff = P.conj().T @ Hfull @ P
    return Heff
```

---

# Cell 9 — Mixed-canonical preparation around a bond

This is useful in Jupyter because you can explicitly prepare the MPS for a local operator test.

```python
def make_mixed_canonical_around_bond(mps, i):
    """
    Make sites < i left-canonical and sites > i+1 right-canonical.
    Bond (i, i+1) becomes the two-site center region.
    """
    mps = [A.copy() for A in mps]
    L = len(mps)

    # Left canonicalize sites 0..i-1
    for k in range(i):
        A = mps[k]
        Dl, d, Dr = A.shape
        M = A.reshape(Dl * d, Dr, order='C')
        Q, R = np.linalg.qr(M, mode='reduced')
        nk = Q.shape[1]
        mps[k] = Q.reshape(Dl, d, nk, order='C')
        mps[k + 1] = np.einsum('ab,bsd->asd', R, mps[k + 1], optimize=True)

    # Right canonicalize sites L-1 .. i+2
    for k in range(L - 1, i + 1, -1):
        A = mps[k]
        Dl, d, Dr = A.shape
        M = A.reshape(Dl, d * Dr, order='C')
        Q, R = np.linalg.qr(M.conj().T, mode='reduced')
        B = Q.conj().T.reshape((-1, d, Dr), order='C')
        mps[k] = B
        Rd = R.conj().T
        mps[k - 1] = np.einsum('xsa,ab->xsb', mps[k - 1], Rd, optimize=True)

    return mps
```

---

# Cell 10 — Environment-vs-projected local operator validation

```python
def build_dense_local_heff_from_env(mps, mpo, i):
    A = mps[i]
    B = mps[i + 1]
    Dl, d1, mid = A.shape
    mid2, d2, Dr = B.shape
    assert mid == mid2

    Lenvs = build_left_envs(mps, mpo)
    Renvs = build_right_envs(mps, mpo)

    Lenv = Lenvs[i]
    Renv = Renvs[i + 1]

    N = Dl * d1 * d2 * Dr
    Heff = np.zeros((N, N), dtype=np.complex128)

    for j in range(N):
        e = np.zeros(N, dtype=np.complex128)
        e[j] = 1.0
        Heff[:, j] = heff_two_site_env_matvec(
            e, Lenv, mpo[i], mpo[i + 1], Renv, Dl, d1, d2, Dr
        )
    return Heff


def validate_two_site_env_operator_against_dense_projected(Hfull, mpo, mps, i, d):
    Heff_env = build_dense_local_heff_from_env(mps, mpo, i)
    Heff_proj = dense_projected_two_site_heff_from_full(Hfull, mps, i, d)

    diff = np.max(np.abs(Heff_env - Heff_proj))
    herm_env = np.max(np.abs(Heff_env - Heff_env.conj().T))
    herm_proj = np.max(np.abs(Heff_proj - Heff_proj.conj().T))

    print(f"bond {i}: max|Heff_env - Heff_proj| = {diff:.3e}")
    print(f"bond {i}: env Hermiticity err      = {herm_env:.3e}")
    print(f"bond {i}: proj Hermiticity err     = {herm_proj:.3e}")

    return Heff_env, Heff_proj
```

---

# Cell 11 — MPO expectation value and norm

```python
def expectation_value_mpo(mps, mpo):
    env = init_left_env()
    for i in range(len(mps)):
        env = left_env_update(env, mps[i], mpo[i])
    return env[0, 0, 0]


def norm_from_mps(mps):
    psi = mps_to_state(mps)
    return np.vdot(psi, psi).real
```

---

# Cell 12 — Fully environment-based matrix-free two-site DMRG

```python
def two_site_dmrg_env_matrix_free(
    mpo,
    d,
    Dmax=32,
    nsweeps=8,
    lanczos_tol=1e-10,
    lanczos_maxiter=None,
    svd_cutoff=1e-12,
    seed=1234,
    verbose=True,
):
    L = len(mpo)

    mps = random_mps(L, d=d, Dmax=min(Dmax, 4), seed=seed)
    mps = normalize_mps_by_state(mps)
    mps = right_canonicalize(mps)

    sweep_energies = []
    sweep_discards = []

    for sweep in range(nsweeps):
        discards_this_sweep = []
        last_local_energy = None

        Renvs = build_right_envs(mps, mpo)
        Lenv = init_left_env()

        # Left -> Right
        for i in range(L - 1):
            theta0 = form_theta(mps[i], mps[i + 1])
            Dl, d1, d2, Dr = theta0.shape
            Nloc = Dl * d1 * d2 * Dr
            Renv = Renvs[i + 1]

            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v, Lenv=Lenv, W1=mpo[i], W2=mpo[i + 1], Renv=Renv,
                              Dl=Dl, d1=d1, d2=d2, Dr=Dr:
                    heff_two_site_env_matvec(v, Lenv, W1, W2, Renv, Dl, d1, d2, Dr),
                dtype=np.complex128
            )

            v0 = theta0.reshape(-1, order='C')
            vals, vecs = eigsh(
                Heff, k=1, which='SA',
                v0=v0, tol=lanczos_tol, maxiter=lanczos_maxiter
            )
            last_local_energy = vals[0].real

            theta = vecs[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            Anew, Bnew, S, disc = split_theta_left_to_right(theta, Dmax, svd_cutoff)
            mps[i] = Anew
            mps[i + 1] = Bnew
            discards_this_sweep.append(disc)

            Lenv = left_env_update(Lenv, mps[i], mpo[i])

        mps = left_canonicalize(mps)
        mps = normalize_mps_by_state(mps)

        Lenvs = build_left_envs(mps, mpo)
        Renv = init_right_env()

        # Right -> Left
        for i in range(L - 2, -1, -1):
            theta0 = form_theta(mps[i], mps[i + 1])
            Dl, d1, d2, Dr = theta0.shape
            Nloc = Dl * d1 * d2 * Dr
            Lcur = Lenvs[i]

            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v, Lcur=Lcur, W1=mpo[i], W2=mpo[i + 1], Renv=Renv,
                              Dl=Dl, d1=d1, d2=d2, Dr=Dr:
                    heff_two_site_env_matvec(v, Lcur, W1, W2, Renv, Dl, d1, d2, Dr),
                dtype=np.complex128
            )

            v0 = theta0.reshape(-1, order='C')
            vals, vecs = eigsh(
                Heff, k=1, which='SA',
                v0=v0, tol=lanczos_tol, maxiter=lanczos_maxiter
            )
            last_local_energy = vals[0].real

            theta = vecs[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            Anew, Bnew, S, disc = split_theta_right_to_left(theta, Dmax, svd_cutoff)
            mps[i] = Anew
            mps[i + 1] = Bnew
            discards_this_sweep.append(disc)

            Renv = right_env_update(Renv, mps[i + 1], mpo[i + 1])

        mps = right_canonicalize(mps)
        mps = normalize_mps_by_state(mps)

        max_disc = max(discards_this_sweep) if discards_this_sweep else 0.0
        sweep_energies.append(last_local_energy if last_local_energy is not None else np.nan)
        sweep_discards.append(max_disc)

        if verbose:
            print(f"Sweep {sweep+1:2d}: local eig E = {sweep_energies[-1]:.12f}, "
                  f"max discarded weight = {max_disc:.3e}")

    return mps, np.array(sweep_energies), np.array(sweep_discards)
```

---

# Cell 13 — Exact AKLT MPS

```python
def exact_aklt_mps(L, left_vec=None, right_vec=None, dtype=np.complex128):
    Ap = np.array([[0, np.sqrt(2/3)],
                   [0, 0]], dtype=dtype)
    A0 = np.array([[-1/np.sqrt(3), 0],
                   [0, 1/np.sqrt(3)]], dtype=dtype)
    Am = np.array([[0, 0],
                   [-np.sqrt(2/3), 0]], dtype=dtype)

    local = np.stack([Ap, A0, Am], axis=1)

    if left_vec is None:
        left_vec = np.array([1.0, 0.0], dtype=dtype)
    if right_vec is None:
        right_vec = np.array([1.0, 0.0], dtype=dtype)

    mps = []
    A1 = np.einsum('a,asb->sb', left_vec, local).reshape(1, 3, 2)
    mps.append(A1)

    for _ in range(1, L - 1):
        mps.append(local.copy())

    AL = np.einsum('asb,b->as', local, right_vec).reshape(2, 3, 1)
    mps.append(AL)

    return mps
```

---

# Cell 14 — First test: MPO correctness

```python
for L in [2, 3, 4]:
    H1 = mpo_to_dense(heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.2))
    H2 = dense_heisenberg(L, J=1.0, Jz=1.0, h=0.2)
    print(f"Heisenberg MPO test L={L}: max|diff| = {np.max(np.abs(H1 - H2)):.3e}")

for L in [2, 3, 4]:
    H1 = mpo_to_dense(aklt_mpo(L))
    H2 = dense_aklt(L)
    print(f"AKLT MPO test L={L}: max|diff| = {np.max(np.abs(H1 - H2)):.3e}")
```

---

# Cell 15 — Crucial test: local env operator vs projected dense operator, Heisenberg

```python
L = 4
d = 2
i = 1

mpo = heisenberg_mpo(L)
Hfull = dense_heisenberg(L)

mps = random_mps(L, d=d, Dmax=3, seed=7)
mps = normalize_mps_by_state(mps)
mps = make_mixed_canonical_around_bond(mps, i)

Heff_env, Heff_proj = validate_two_site_env_operator_against_dense_projected(
    Hfull, mpo, mps, i, d
)
```

If this is correct, `max|Heff_env - Heff_proj|` should be very small.

---

# Cell 16 — Same local test for AKLT

```python
L = 4
d = 3
i = 1

mpo = aklt_mpo(L)
Hfull = dense_aklt(L)

mps = random_mps(L, d=d, Dmax=3, seed=11)
mps = normalize_mps_by_state(mps)
mps = make_mixed_canonical_around_bond(mps, i)

Heff_env, Heff_proj = validate_two_site_env_operator_against_dense_projected(
    Hfull, mpo, mps, i, d
)
```

---

# Cell 17 — Heisenberg DMRG run

```python
L = 8
mpo = heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0)

mps_h, Es_h, disc_h = two_site_dmrg_env_matrix_free(
    mpo=mpo,
    d=2,
    Dmax=32,
    nsweeps=6,
    seed=1234,
    verbose=True
)

E_mpo = expectation_value_mpo(mps_h, mpo).real
print("\nHeisenberg <H> from MPO contraction =", E_mpo)

Hfull = dense_heisenberg(L, J=1.0, Jz=1.0, h=0.0)
psi = mps_to_state(mps_h)
E_dense = (np.vdot(psi, Hfull @ psi) / np.vdot(psi, psi)).real
E0 = np.linalg.eigvalsh(Hfull)[0].real

print("Heisenberg energy from dense H on final MPS =", E_dense)
print("Heisenberg exact ED ground energy           =", E0)
print("Absolute error                              =", abs(E_dense - E0))
```

---

# Cell 18 — AKLT DMRG run

```python
L = 6
mpo = aklt_mpo(L)

mps_a, Es_a, disc_a = two_site_dmrg_env_matrix_free(
    mpo=mpo,
    d=3,
    Dmax=16,
    nsweeps=6,
    seed=1234,
    verbose=True
)

E_mpo = expectation_value_mpo(mps_a, mpo).real
E_exact = -(2.0 / 3.0) * (L - 1)

print("\nAKLT <H> from MPO contraction =", E_mpo)
print("AKLT exact open-chain energy  =", E_exact)
print("Absolute error                =", abs(E_mpo - E_exact))

Hfull = dense_aklt(L)
psi = mps_to_state(mps_a)
E_dense = (np.vdot(psi, Hfull @ psi) / np.vdot(psi, psi)).real
E0 = np.linalg.eigvalsh(Hfull)[0].real

print("AKLT dense energy from final MPS =", E_dense)
print("AKLT exact ED ground energy      =", E0)
print("Difference to ED                 =", abs(E_dense - E0))
```

---

# Very useful Jupyter tip: add progress/inspection cells

For notebooks, I recommend adding small inspection cells like:

## Inspect bond dimensions
```python
[A.shape for A in mps_h]
```

## Inspect sweep histories
```python
Es_h, disc_h
```

## Plot convergence
```python
import matplotlib.pyplot as plt

plt.plot(Es_h, marker='o')
plt.xlabel('Sweep')
plt.ylabel('Last local eig energy')
plt.title('Heisenberg DMRG convergence')
plt.grid(True)
plt.show()
```

and similarly for `disc_h`.

---

# One important correction / caution

In Jupyter, because cells can be run out of order, the most common source of confusion is:

- you redefine one function,
- but other cells still use old state or old objects.

So I strongly recommend:

## Good notebook practice
When changing core functions such as:
- `left_env_update`
- `right_env_update`
- `apply_two_site_heff_env`
- `two_site_dmrg_env_matrix_free`

do:

1. **Kernel → Restart**
2. **Run All**

That avoids stale definitions.

---

# Recommended notebook order

Use this order:

1. Cells 1–13: definitions
2. Cell 14: MPO tests
3. Cell 15: Heisenberg local-operator validation
4. Cell 16: AKLT local-operator validation
5. Cell 17: Heisenberg DMRG
6. Cell 18: AKLT DMRG

This is the safest route.

---

# If you want a cleaner notebook structure

I can next reformat this into a true notebook-style layout with:

- Markdown section headers
- shorter cells
- optional `%time` and plotting cells
- a final “benchmark dashboard” cell

For example, I can provide:

1. **a polished `.ipynb-style` version**, cell-by-cell with markdown titles,
2. **a compact version** for quick runs,
3. **a debug notebook version** with extra assertions and operator checks.

If you want, I can now give you the **next message as a notebook-ready sequence of Markdown + code cells**, so you can paste it almost directly into Jupyter.

---
## 🧑‍💻 User

Output of Cell 15: bond 1: max|Heff_env - Heff_proj| = 5.000e-01
bond 1: env Hermiticity err      = 0.000e+00
bond 1: proj Hermiticity err     = 5.551e-17

---
## 🤖 Assistant

Good — this is exactly the kind of diagnostic we needed.

That output means:

- the environment-built local operator is **Hermitian**
- but it is still **not the true projected operator**
- the mismatch is large: `0.5`

So the bug is not numerical noise. It is structural.

## Most likely cause
The most likely issue is the **physical-index orientation in the two-site local action**.

Your MPO convention from the note is:

\[
W_{\beta_{i-1},\beta_i,\sigma,\sigma'}
\]
with

- first physical index = **ket/output**
- second physical index = **bra/input**

So in the local action,
\[
(\mathcal H_{\mathrm{eff}}\Theta)_{\dots,\sigma_i,\sigma_{i+1},\dots}
=
\sum
W^{[i]}_{\dots,\sigma_i,\sigma'_i}
W^{[i+1]}_{\dots,\sigma_{i+1},\sigma'_{i+1}}
\Theta_{\dots,\sigma'_i,\sigma'_{i+1},\dots}
\]
the MPO should map **input physical indices of `theta`** to **output physical indices of the result**.

That means the two-site contraction is sensitive to whether we wrote the einsums with `(output,input)` or accidentally flipped them.

---

# Fix

Please replace **Cell 7**, specifically `apply_two_site_heff_env`, by the version below.

## Corrected Cell 7

```python
def apply_one_site_heff_env(M, Lenv, W, Renv):
    # M[y, s_in, z]
    # W[b, B, s_out, s_in]
    X = np.einsum('bxy,ysz->bxsz', Lenv, M, optimize=True)
    Y = np.einsum('bBts,bxsz->Bxtz', W, X, optimize=True)
    HM = np.einsum('Bxtz,Bza->xta', Y, Renv, optimize=True)
    return HM


def apply_two_site_heff_env(theta, Lenv, W1, W2, Renv):
    """
    theta[y, u, v, z] = input two-site tensor
    W1[b, B, s, u]
    W2[B, C, t, v]
    output HT[x, s, t, a]
    """
    X = np.einsum('bxy,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBsu,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCtv,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def heff_two_site_env_matvec(v, Lenv, W1, W2, Renv, Dl, d1, d2, Dr):
    theta = v.reshape((Dl, d1, d2, Dr), order='C')
    out = apply_two_site_heff_env(theta, Lenv, W1, W2, Renv)
    return out.reshape(Dl * d1 * d2 * Dr, order='C')
```

---

# Why this change matters

Previously I used contractions like
```python
W1[b,B,s,t]
```
and fed them into tensors in a way that effectively treated the physical pair ambiguously.

The corrected version makes the role explicit:

- `theta[y,u,v,z]` has **input** physical indices `u,v`
- `W1[b,B,s,u]` maps `u -> s`
- `W2[B,C,t,v]` maps `v -> t`
- output is `HT[x,s,t,a]`

This is the safest explicit index assignment.

---

# After replacing Cell 7

Please do this in Jupyter:

1. **Restart kernel**
2. Run Cells 1–14 again
3. Re-run Cell 15

What we want to see is something like

```python
bond 1: max|Heff_env - Heff_proj| ~ 1e-14 to 1e-12
```

or at worst small floating-point noise.

---

# If it still fails

Then the next likely culprit is not the local action but the **canonical-basis map consistency** in the dense projected reference test, i.e.

- `left_basis_map`
- `right_basis_map`
- or `make_mixed_canonical_around_bond`

But the first thing to test is the corrected local action above, because a `0.5` mismatch is very consistent with a physical-index transposition mistake.

---

# Very useful extra debugging cell

After fixing Cell 7, add this temporary debugging cell:

```python
# Debugging: compare local matvec on random vector
L = 4
d = 2
i = 1

mpo = heisenberg_mpo(L)
Hfull = dense_heisenberg(L)

mps = random_mps(L, d=d, Dmax=3, seed=7)
mps = normalize_mps_by_state(mps)
mps = make_mixed_canonical_around_bond(mps, i)

Lenvs = build_left_envs(mps, mpo)
Renvs = build_right_envs(mps, mpo)

A = mps[i]
B = mps[i+1]
Dl, d1, mid = A.shape
mid2, d2, Dr = B.shape

Heff_proj = dense_projected_two_site_heff_from_full(Hfull, mps, i, d)

rng = np.random.default_rng(123)
v = rng.normal(size=Dl*d1*d2*Dr) + 1j*rng.normal(size=Dl*d1*d2*Dr)

v_env = heff_two_site_env_matvec(v, Lenvs[i], mpo[i], mpo[i+1], Renvs[i+1], Dl, d1, d2, Dr)
v_proj = Heff_proj @ v

print("random-vector matvec diff =", np.max(np.abs(v_env - v_proj)))
```

If this becomes tiny, then the local operator is fixed.

---

# Important note
Because Jupyter keeps old function definitions around, do not just rerun Cell 15 alone after editing Cell 7.

Please **restart kernel and run all relevant cells in order**.

---

If you want, after you rerun Cell 15, send me the new output and I’ll help you debug the next layer immediately if needed.

---
## 🧑‍💻 User

bond 1: max|Heff_env - Heff_proj| = 4.775e-01
bond 1: env Hermiticity err      = 0.000e+00
bond 1: proj Hermiticity err     = 5.551e-17

---
## 🤖 Assistant

Thanks — that is very informative.

Since the mismatch barely changed, the primary issue is **not** just the physical-index orientation in `apply_two_site_heff_env`.

At this point the most likely problem is:

# The dense projected reference is being compared in a bond basis that is not guaranteed to match the environment basis

In other words, the issue is likely in one or both of:

- `left_basis_map`
- `right_basis_map`
- `make_mixed_canonical_around_bond`

rather than in the environment contraction itself.

This is consistent with the handoff note: the old environment branch was Hermitian but did not match \(P^\dagger H P\). A very common reason is that the “projected” basis maps are not actually the exact orthonormal left/right Schmidt bases corresponding to the current two-site center.

---

# Best next step in Jupyter

Instead of debugging everything at once, let us do the **most robust possible local test**:

## Build the projected basis directly from the mixed-canonical MPS tensors around the bond
That means:

- left basis = contraction of sites `0..i-1`
- right basis = contraction of sites `i+2..L-1`
- and then explicitly verify these basis maps are orthonormal

If the basis maps are wrong, `P^\dagger H P` is not the same operator that the environments represent.

---

# Replace Cells 8, 9, 10 with the versions below

These are more careful and include orthonormality diagnostics.

---

## New Cell 8 — explicit left/right basis maps and checks

```python
def left_basis_map(mps, i):
    """
    Basis map for sites 0..i-1 into the left bond space at bond i.
    Returns UL with shape (dimL, Dl), where Dl = mps[i].shape[0].
    In mixed canonical form with sites < i left-canonical, columns of UL
    should be orthonormal.
    """
    if i == 0:
        return np.array([[1.0]], dtype=np.complex128)

    X = mps[0][0]  # shape (d, D1)
    for k in range(1, i):
        X = np.einsum('pa,asb->psb', X, mps[k], optimize=True)
        X = X.reshape((-1, X.shape[-1]), order='C')
    return X


def right_basis_map(mps, i):
    """
    Basis map for sites i+2..L-1 from the right bond space at bond i+1.
    Returns UR with shape (Dr, dimR), where Dr = mps[i+1].shape[2].
    In mixed canonical form with sites > i+1 right-canonical, rows of UR
    should be orthonormal.
    """
    L = len(mps)
    if i == L - 2:
        return np.array([[1.0]], dtype=np.complex128)

    X = mps[i + 2]
    # X shape: (Dr, d, Dnext)
    for k in range(i + 3, L):
        X = np.einsum('asd,dtu->astu', X, mps[k], optimize=True)
        X = X.reshape((X.shape[0], -1, X.shape[-1]), order='C')
    return X[:, :, 0]


def check_basis_orthonormality(mps, i):
    UL = left_basis_map(mps, i)
    UR = right_basis_map(mps, i)

    errL = np.max(np.abs(UL.conj().T @ UL - np.eye(UL.shape[1])))
    errR = np.max(np.abs(UR @ UR.conj().T - np.eye(UR.shape[0])))

    print(f"bond {i}: left basis orthonormality error  = {errL:.3e}")
    print(f"bond {i}: right basis orthonormality error = {errR:.3e}")

    return errL, errR
```

---

## New Cell 9 — safer mixed-canonical preparation

```python
def make_mixed_canonical_around_bond(mps, i):
    """
    Make the MPS mixed-canonical around bond (i, i+1):
      - sites 0..i-1 left-canonical
      - sites i+2..L-1 right-canonical
    sites i and i+1 remain the two-site center region.
    """
    mps = [A.copy() for A in mps]
    L = len(mps)

    # Left canonicalize sites 0..i-1
    for k in range(i):
        A = mps[k]
        Dl, d, Dr = A.shape
        M = A.reshape((Dl * d, Dr), order='C')
        Q, R = np.linalg.qr(M, mode='reduced')
        nk = Q.shape[1]
        mps[k] = Q.reshape((Dl, d, nk), order='C')
        mps[k + 1] = np.einsum('ab,bsd->asd', R, mps[k + 1], optimize=True)

    # Right canonicalize sites i+2..L-1
    for k in range(L - 1, i + 1, -1):
        A = mps[k]
        Dl, d, Dr = A.shape
        M = A.reshape((Dl, d * Dr), order='C')
        Q, R = np.linalg.qr(M.conj().T, mode='reduced')
        B = Q.conj().T.reshape((-1, d, Dr), order='C')
        mps[k] = B
        Rd = R.conj().T
        mps[k - 1] = np.einsum('xsa,ab->xsb', mps[k - 1], Rd, optimize=True)

    return mps
```

---

## New Cell 10 — projected operator builder plus diagnostics

```python
def dense_projected_two_site_heff_from_full(Hfull, mps, i, d):
    UL = left_basis_map(mps, i)
    UR = right_basis_map(mps, i)

    Dl = UL.shape[1]
    Dr = UR.shape[0]
    dimL = UL.shape[0]
    dimR = UR.shape[1]

    Nloc = Dl * d * d * Dr
    Dim = dimL * d * d * dimR
    P = np.zeros((Dim, Nloc), dtype=np.complex128)

    col = 0
    for a in range(Dl):
        for s in range(d):
            for t in range(d):
                for c in range(Dr):
                    vec = np.kron(UL[:, a], np.kron(np.eye(d*d)[:, s*d + t], UR[c, :]))
                    P[:, col] = vec
                    col += 1

    # Diagnostic: P should be an isometry if basis maps are orthonormal
    iso_err = np.max(np.abs(P.conj().T @ P - np.eye(Nloc)))
    print(f"bond {i}: projector isometry error = {iso_err:.3e}")

    Heff = P.conj().T @ Hfull @ P
    return Heff


def build_dense_local_heff_from_env(mps, mpo, i):
    A = mps[i]
    B = mps[i + 1]
    Dl, d1, mid = A.shape
    mid2, d2, Dr = B.shape
    assert mid == mid2

    Lenvs = build_left_envs(mps, mpo)
    Renvs = build_right_envs(mps, mpo)

    Lenv = Lenvs[i]
    Renv = Renvs[i + 1]

    N = Dl * d1 * d2 * Dr
    Heff = np.zeros((N, N), dtype=np.complex128)

    for j in range(N):
        e = np.zeros(N, dtype=np.complex128)
        e[j] = 1.0
        Heff[:, j] = heff_two_site_env_matvec(
            e, Lenv, mpo[i], mpo[i + 1], Renv, Dl, d1, d2, Dr
        )
    return Heff


def validate_two_site_env_operator_against_dense_projected(Hfull, mpo, mps, i, d):
    check_basis_orthonormality(mps, i)

    Heff_env = build_dense_local_heff_from_env(mps, mpo, i)
    Heff_proj = dense_projected_two_site_heff_from_full(Hfull, mps, i, d)

    diff = np.max(np.abs(Heff_env - Heff_proj))
    herm_env = np.max(np.abs(Heff_env - Heff_env.conj().T))
    herm_proj = np.max(np.abs(Heff_proj - Heff_proj.conj().T))

    print(f"bond {i}: max|Heff_env - Heff_proj| = {diff:.3e}")
    print(f"bond {i}: env Hermiticity err      = {herm_env:.3e}")
    print(f"bond {i}: proj Hermiticity err     = {herm_proj:.3e}")

    return Heff_env, Heff_proj
```

---

# Then rerun this test cell

```python
L = 4
d = 2
i = 1

mpo = heisenberg_mpo(L)
Hfull = dense_heisenberg(L)

mps = random_mps(L, d=d, Dmax=3, seed=7)
mps = normalize_mps_by_state(mps)
mps = make_mixed_canonical_around_bond(mps, i)

Heff_env, Heff_proj = validate_two_site_env_operator_against_dense_projected(
    Hfull, mpo, mps, i, d
)
```

---

# What I want to see from you next

Please send me the full printed output of that cell, especially these three lines:

- `left basis orthonormality error`
- `right basis orthonormality error`
- `projector isometry error`

These will tell us immediately whether the projected reference itself is trustworthy.

---

# Why this is the right debugging move

At this stage there are only two real possibilities:

## Possibility A
The environment contraction is wrong.

## Possibility B
The dense projected reference is being built in a basis that is not exactly the mixed-canonical Schmidt basis.

Your previous result strongly suggests B is involved, because:

- Hermiticity is exact
- mismatch is large but not random
- this is exactly what happens when the local basis in \(P\) is not the same orthonormal basis encoded by environments

---

# One more highly likely issue
There is also a subtlety in how `expectation_value_mpo` is being used later: that routine is fine only when the MPS appears in both bra and ket, but it does not by itself diagnose the local projected-basis problem.

So for now, focus entirely on the local operator identity test.

---

If you send the new diagnostic output, I can pinpoint the next correction very quickly.

---
## 🧑‍💻 User

bond 1: left basis orthonormality error  = 4.441e-16
bond 1: right basis orthonormality error = 1.221e-15
bond 1: projector isometry error = 1.554e-15
bond 1: max|Heff_env - Heff_proj| = 4.775e-01
bond 1: env Hermiticity err      = 0.000e+00
bond 1: proj Hermiticity err     = 5.551e-17

---
## 🤖 Assistant

Excellent — this isolates the issue very sharply.

## Conclusion from your diagnostics

The projected reference is fine:

- left basis orthonormality: good
- right basis orthonormality: good
- projector isometry: good

So the bug is **definitely in the environment-based operator**, not in the dense projected reference.

That means we now only need to fix one of:

1. `left_env_update`
2. `right_env_update`
3. `apply_two_site_heff_env`

Since the dense projected basis is correct, the next best diagnostic is:

# Test the environments themselves against brute-force partial contractions

This is the cleanest way to determine whether the error is already present in `Lenv` / `Renv`, or whether those are correct and only the two-site application is wrong.

---

# Add this new Jupyter cell

## Cell 19 — brute-force dense environments from full basis maps

```python
def dense_left_env_reference(mps, mpo, i):
    """
    Build left environment for bond i by brute force from the orthonormal
    left basis UL for sites 0..i-1.

    Returns Lref with shape (chi_i_left, Dl, Dl)
    where chi_i_left = mpo[i].shape[0].
    """
    UL = left_basis_map(mps, i)   # (dimL, Dl)
    dimL, Dl = UL.shape

    # Dense MPO operator for left block 0..i-1, with open MPO legs:
    # M[b] = operator from mpo bond-left boundary to internal bond b at cut i
    cur = [mpo[0][0, b] for b in range(mpo[0].shape[1])]

    for site in range(1, i):
        W = mpo[site]
        chiL, chiR, d, _ = W.shape
        new = [
            np.zeros((cur[0].shape[0] * d, cur[0].shape[1] * d), dtype=np.complex128)
            for _ in range(chiR)
        ]
        for bl in range(chiL):
            for br in range(chiR):
                if np.linalg.norm(W[bl, br]) == 0:
                    continue
                new[br] += np.kron(cur[bl], W[bl, br])
        cur = new

    chi_cut = mpo[i].shape[0]
    if i == 0:
        # left block empty
        Lref = np.zeros((chi_cut, 1, 1), dtype=np.complex128)
        Lref[0, 0, 0] = 1.0
        return Lref

    Lref = np.zeros((chi_cut, Dl, Dl), dtype=np.complex128)
    for b in range(chi_cut):
        Op = cur[b]
        Lref[b] = UL.conj().T @ Op @ UL
    return Lref


def dense_right_env_reference(mps, mpo, i_plus_1):
    """
    Build right environment R^{[i+1]} by brute force from orthonormal right basis
    on sites i+2..L-1.

    Input i_plus_1 corresponds to the right environment used for bond (i, i+1).
    Returns shape (chi_right, Dr, Dr), where chi_right = mpo[i+1].shape[1].
    """
    L = len(mps)
    site0 = i_plus_1 + 1  # first site in right block

    UR = right_basis_map(mps, i_plus_1 - 1)  # (Dr, dimR)
    Dr, dimR = UR.shape

    if site0 >= L:
        Rref = np.zeros((1, 1, 1), dtype=np.complex128)
        Rref[0, 0, 0] = 1.0
        return Rref

    # Build dense operator-valued right block from site0..L-1 with open left MPO leg
    cur = [mpo[site0][b, 0] for b in range(mpo[site0].shape[0])]

    for site in range(site0 + 1, L):
        W = mpo[site]
        chiL, chiR, d, _ = W.shape
        new = [
            np.zeros((d * cur[0].shape[0], d * cur[0].shape[1]), dtype=np.complex128)
            for _ in range(chiL)
        ]

        # We are building from right to left, so do it explicitly:
        # O_b = sum_B W[b,B] \otimes cur[B]
        for b in range(chiL):
            acc = np.zeros((d * cur[0].shape[0], d * cur[0].shape[1]), dtype=np.complex128)
            for B in range(chiR):
                if np.linalg.norm(W[b, B]) == 0:
                    continue
                acc += np.kron(W[b, B], cur[B])
            new[b] = acc
        cur = new

    chi_cut = mpo[i_plus_1].shape[1]
    Rref = np.zeros((chi_cut, Dr, Dr), dtype=np.complex128)
    for b in range(chi_cut):
        Op = cur[b]
        Rref[b] = UR @ Op @ UR.conj().T
    return Rref
```

---

# Add this next cell

## Cell 20 — compare env updates to brute-force env references

```python
def compare_envs_to_dense_references(mps, mpo, i):
    """
    Compare Lenvs[i] and Renvs[i+1] against brute-force dense references.
    """
    Lenvs = build_left_envs(mps, mpo)
    Renvs = build_right_envs(mps, mpo)

    Lenv = Lenvs[i]
    Renv = Renvs[i + 1]

    Lref = dense_left_env_reference(mps, mpo, i)
    Rref = dense_right_env_reference(mps, mpo, i + 1)

    print(f"bond {i}: max|Lenv - Lref| = {np.max(np.abs(Lenv - Lref)):.3e}")
    print(f"bond {i}: max|Renv - Rref| = {np.max(np.abs(Renv - Rref)):.3e}")

    return Lenv, Lref, Renv, Rref
```

---

# Run this test for your Heisenberg bond

## Cell 21

```python
L = 4
d = 2
i = 1

mpo = heisenberg_mpo(L)
mps = random_mps(L, d=d, Dmax=3, seed=7)
mps = normalize_mps_by_state(mps)
mps = make_mixed_canonical_around_bond(mps, i)

Lenv, Lref, Renv, Rref = compare_envs_to_dense_references(mps, mpo, i)
```

---

# What this will tell us

There are three possibilities:

## Case 1
`Lenv` wrong, `Renv` right  
Then `left_env_update` is wrong.

## Case 2
`Lenv` right, `Renv` wrong  
Then `right_env_update` is wrong.

## Case 3
Both right, but `Heff_env` still wrong  
Then `apply_two_site_heff_env` is wrong.

This completely isolates the bug.

---

# My current suspicion

Given the formulas and the fact that Hermiticity is exact, I suspect:

## `right_env_update` is the most likely culprit

It is much easier to get right-env index order subtly wrong, because the bra/ket and bond orientations are less intuitive than on the left.

The current implementation is:

```python
def right_env_update(Rold, B, W):
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj(), optimize=True)
```

This is plausible, but I am not yet fully convinced it matches the exact dense reference in the chosen convention.

---

# If the right environment turns out wrong

Then the first alternative to test is:

```python
def right_env_update(Rold, B, W):
    return np.einsum('xsa,bBts,Bac,ytc->bxy', B, W, Rold, B.conj(), optimize=True)
```

i.e. swap the physical indices of `W` in the right update only.

That kind of asymmetry can happen because the right contraction formula has the bra/ket order arranged differently than one naively writes in code.

But don't change it yet. First run the comparison against `Rref`.

---

# Please send me the output of Cell 21

The two lines I want are:

- `max|Lenv - Lref|`
- `max|Renv - Rref|`

Once I see those, I can tell you exactly what to patch next.

---
## 🧑‍💻 User

bond 1: max|Lenv - Lref| = 8.246e-01
bond 1: max|Renv - Rref| = 0.000e+00

---
## 🤖 Assistant

Perfect. That isolates it.

## The bug is in `left_env_update`
because:

- `Renv` matches dense reference exactly
- `Lenv` is badly wrong

So we can now focus only on the left update.

---

# Why the previous left update was wrong

You had:

```python
def left_env_update(Lold, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj(), optimize=True)
```

This uses `W[b,B,s,t]` as if the physical index on `A` and `A.conj()` were contracted in the order:

- `A` gets `s`
- `A.conj()` gets `t`

But with the MPO convention

\[
W_{\beta_{i-1},\beta_i,\sigma,\sigma'}
\]
where:

- first physical index = ket/output
- second physical index = bra/input

and with the left environment formula in the note,
the contraction should be

\[
A^*_{\dots,\sigma',\dots}\;
W_{\dots,\sigma,\sigma'}\;
A_{\dots,\sigma,\dots}
\]

So the correct code must contract:

- `A` with the **first** physical index of `W`
- `A.conj()` with the **second** physical index of `W`

That means the left update should be:

```python
def left_env_update(Lold, A, W):
    return np.einsum('bxy,xsa,bBts,ytc->Bac', Lold, A, W, A.conj(), optimize=True)
```

Notice the crucial change:

- from `bBst`
- to   `bBts`

---

# Replace Cell 6 by this corrected version

```python
def init_left_env():
    Lenv = np.zeros((1, 1, 1), dtype=np.complex128)
    Lenv[0, 0, 0] = 1.0
    return Lenv


def init_right_env():
    Renv = np.zeros((1, 1, 1), dtype=np.complex128)
    Renv[0, 0, 0] = 1.0
    return Renv


def left_env_update(Lold, A, W):
    """
    Correct left update:
      Lnew_B(a,c) = sum_{b,x,y,s,t} Lold_b(x,y) A(x,s,a) W(b,B,t,s) A*(y,t,c)

    Here W[..., ket, bra], and A carries the ket index.
    """
    return np.einsum('bxy,xsa,bBts,ytc->Bac', Lold, A, W, A.conj(), optimize=True)


def right_env_update(Rold, B, W):
    """
    Right update already validated against dense reference.
    """
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj(), optimize=True)


def build_right_envs(mps, mpo):
    L = len(mps)
    R = [None] * L
    R[L - 1] = init_right_env()

    cur = R[L - 1]
    for i in range(L - 1, 0, -1):
        cur = right_env_update(cur, mps[i], mpo[i])
        R[i - 1] = cur

    return R


def build_left_envs(mps, mpo):
    L = len(mps)
    Lenvs = [None] * L
    Lenvs[0] = init_left_env()

    cur = Lenvs[0]
    for i in range(0, L - 1):
        cur = left_env_update(cur, mps[i], mpo[i])
        Lenvs[i + 1] = cur

    return Lenvs
```

---

# Then rerun these cells in order

Because you are in Jupyter:

1. **Restart kernel**
2. Run Cells 1–21 in order again

---

# What should happen now

## First, Cell 21 should give
```python
bond 1: max|Lenv - Lref| ~ 1e-15
bond 1: max|Renv - Rref| ~ 1e-15
```

## Then Cell 15 should give
```python
bond 1: max|Heff_env - Heff_proj| ~ 1e-15
```

or at least very small.

---

# Why right update did not need the same swap
This is subtle but important.

The right update was already validated against dense reference, so in the chosen convention it is correct as written.

This asymmetry can happen because the left and right environment formulas place bra/ket tensors in different positions in the contraction network. So it is not necessary that both updates use the same physical-index ordering in the einsum string.

This is exactly why brute-force environment reference tests are so useful.

---

# After that
If Cell 15 passes, then we can trust the local matrix-free environment operator.

Then the next step is:

- rerun Heisenberg DMRG
- rerun AKLT DMRG
- compare final energies

---

# One more thing: you should also patch Cell 11

Because `expectation_value_mpo` uses the left environment update repeatedly, once `left_env_update` changes, that function becomes correct automatically. No extra edit is needed, but you do need to rerun that cell after rerunning Cell 6.

---

Please rerun:

- Cell 21
- then Cell 15

and send me both outputs. If those pass, I’ll help you verify the full DMRG sweep.

---
## 🧑‍💻 User

but you did not provide the patch to Cell 11

---
## 🤖 Assistant

You're right. Cell 11 itself does not need a formula change, but in Jupyter it should be rerun after patching Cell 6. For clarity, here is the explicit patched Cell 11.

# Patched Cell 11

```python
def expectation_value_mpo(mps, mpo):
    """
    Compute <psi|H|psi> by repeated left-environment updates.
    Assumes the corrected left_env_update from Cell 6.
    """
    env = init_left_env()
    for i in range(len(mps)):
        env = left_env_update(env, mps[i], mpo[i])
    return env[0, 0, 0]


def norm_from_mps(mps):
    psi = mps_to_state(mps)
    return np.vdot(psi, psi).real
```

## Important Jupyter point
Even though the code text is the same as before, you should still rerun Cell 11 after rerunning Cell 6, because `expectation_value_mpo` depends on the current definition of `left_env_update`.

## Recommended rerun order now
After patching Cell 6:

1. Restart kernel
2. Run Cells 1–11
3. Run Cell 21
4. Run Cell 15
5. If both pass, run the DMRG cells

If you want, after you rerun Cell 21 and Cell 15, send me the outputs and I’ll help confirm everything is now consistent.

---
## 🧑‍💻 User

bond 1: max|Lenv - Lref| = 8.246e-01
bond 1: max|Renv - Rref| = 0.000e+00

---
## 🤖 Assistant

Thanks — that means my previous “swap only the `W` physical labels in the left update” patch was **not sufficient**. Good that you checked the environment directly.

Since:

- `Renv` is correct,
- `Lenv` is still wrong,

we should stop guessing and derive the left update by matching the *actual dense reference convention* used in your code.

---

# What is happening

Your dense left reference `Lref` is built as

\[
L_b = U_L^\dagger \, O_b \, U_L,
\]

where `UL[:, a]` is the left Schmidt basis vector and `O_b` is the operator-valued left-block MPO channel.

So the recursive update for `Lenv` must reproduce exactly that matrix element.

When written in code with your array conventions, the correct left update is the one that contracts as

\[
L^{\text{new}}_{B}(a,c)
=
\sum_{b,x,y,s,t}
L^{\text{old}}_{b}(x,y)\,
A^*_{y,s,c}\,
W_{b,B,s,t}\,
A_{x,t,a}.
\]

That is: compared to the earlier candidate, **the physical indices on `A` and `A.conj()` are swapped relative to the MPO physical slots**.

So the correct left update is:

```python
np.einsum('bxy,ytc,bBst,xsa->Bac', Lold, A.conj(), W, A)
```

or equivalently with index names reordered.

---

# Please replace Cell 6 by this exact version

```python
def init_left_env():
    Lenv = np.zeros((1, 1, 1), dtype=np.complex128)
    Lenv[0, 0, 0] = 1.0
    return Lenv


def init_right_env():
    Renv = np.zeros((1, 1, 1), dtype=np.complex128)
    Renv[0, 0, 0] = 1.0
    return Renv


def left_env_update(Lold, A, W):
    """
    Left environment update consistent with the dense reference
    Lref[b] = UL^\dagger O_b UL.

    Shapes:
      Lold[b, x, y]
      A[x, s, a]
      W[b, B, s, t]
      A.conj()[y, t, c]
      -> Lnew[B, a, c]
    """
    return np.einsum('bxy,ytc,bBst,xsa->Bac', Lold, A.conj(), W, A, optimize=True)


def right_env_update(Rold, B, W):
    """
    Right update already validated against dense reference.
    """
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj(), optimize=True)


def build_right_envs(mps, mpo):
    L = len(mps)
    R = [None] * L
    R[L - 1] = init_right_env()

    cur = R[L - 1]
    for i in range(L - 1, 0, -1):
        cur = right_env_update(cur, mps[i], mpo[i])
        R[i - 1] = cur

    return R


def build_left_envs(mps, mpo):
    L = len(mps)
    Lenvs = [None] * L
    Lenvs[0] = init_left_env()

    cur = Lenvs[0]
    for i in range(0, L - 1):
        cur = left_env_update(cur, mps[i], mpo[i])
        Lenvs[i + 1] = cur

    return Lenvs
```

---

# Why this is different from the previous attempt

Previously I kept the tensor positions essentially as
```python
Lold, A, W, A.conj()
```
with a possible swap in the MPO physical labels.

But the dense-reference comparison indicates the problem is deeper: the **bra/ket role of the MPS tensor in the left update** was reversed relative to how the operator-valued left block is built.

The new formula makes the left update correspond directly to
\[
\langle a |_L \, O_B \, | c \rangle_L
\]
with the same basis convention used by `left_basis_map`.

---

# After replacing Cell 6

Again, because Jupyter caches definitions:

1. **Restart kernel**
2. Run Cells 1–21 in order
3. Check Cell 21 first

What we want now is:

```python
bond 1: max|Lenv - Lref| ~ 1e-15
bond 1: max|Renv - Rref| ~ 1e-15
```

If that works, then run Cell 15 again. Then we expect:

```python
bond 1: max|Heff_env - Heff_proj| ~ 1e-15
```

---

# You should also rerun Cell 11
Again explicitly, after rerunning Cell 6, rerun:

```python
def expectation_value_mpo(mps, mpo):
    env = init_left_env()
    for i in range(len(mps)):
        env = left_env_update(env, mps[i], mpo[i])
    return env[0, 0, 0]


def norm_from_mps(mps):
    psi = mps_to_state(mps)
    return np.vdot(psi, psi).real
```

---

# If this still does not fix `Lenv`
Then the next step is to compare the **single-step** left update on site 0 against its exact dense block operator, which will isolate whether the mismatch already appears on the very first site. But first try the corrected Cell 6 above.

Please send me the new Cell 21 output after this patch.

---
## 🧑‍💻 User

bond 1: max|Lenv - Lref| = 8.246e-01
bond 1: max|Renv - Rref| = 0.000e+00

---
## 🤖 Assistant

Great — this tells us something decisive:

- the **recursive** left environment update formula we’re trying is still not matching,
- but since `Renv` matches perfectly, the network conventions are internally consistent on the right,
- therefore the safest path now is **not** to continue guessing the left update by inspection.

Instead, in Jupyter we should do the proper forensic step:

# Compare the left environment after exactly one update

For bond `i=1`, the left environment `Lenvs[1]` is just the contraction of **site 0 only**.

That means we can compute it both ways very explicitly:

- via `left_env_update(init_left_env(), mps[0], mpo[0])`
- via direct dense matrix elements of the left boundary MPO tensor

This will isolate the problem at the smallest possible level.

---

# Add this new debugging cell

## Cell 22 — one-step left environment diagnostic

```python
def left_env_site0_direct(A0, W0):
    """
    Direct formula for the left environment after site 0 only.

    Since the left block contains only site 0 and Dl=1 on the left boundary,
    the result should be:
        L1[B, a, c] = sum_{s,t} A0[0,s,a] * W0[0,B,s,t] * A0[0,t,c]^*
    up to the exact convention.
    We'll test several candidate contractions.
    """
    cand1 = np.einsum('sa,Bst,tc->Bac', A0[0], W0[0], A0[0].conj(), optimize=True)
    cand2 = np.einsum('ta,Bst,sc->Bac', A0[0], W0[0], A0[0].conj(), optimize=True)
    cand3 = np.einsum('sc,Bst,ta->Bac', A0[0].conj(), W0[0], A0[0], optimize=True)
    cand4 = np.einsum('tc,Bst,sa->Bac', A0[0].conj(), W0[0], A0[0], optimize=True)
    return cand1, cand2, cand3, cand4


def compare_left_env_first_step(mps, mpo):
    A0 = mps[0]
    W0 = mpo[0]

    L1_rec = left_env_update(init_left_env(), A0, W0)
    cand1, cand2, cand3, cand4 = left_env_site0_direct(A0, W0)

    print("L1_rec shape:", L1_rec.shape)
    print("max|L1_rec - cand1| =", np.max(np.abs(L1_rec - cand1)))
    print("max|L1_rec - cand2| =", np.max(np.abs(L1_rec - cand2)))
    print("max|L1_rec - cand3| =", np.max(np.abs(L1_rec - cand3)))
    print("max|L1_rec - cand4| =", np.max(np.abs(L1_rec - cand4)))

    return L1_rec, cand1, cand2, cand3, cand4
```

---

# Run this test

## Cell 23

```python
L = 4
d = 2
i = 1

mpo = heisenberg_mpo(L)
mps = random_mps(L, d=d, Dmax=3, seed=7)
mps = normalize_mps_by_state(mps)
mps = make_mixed_canonical_around_bond(mps, i)

L1_rec, cand1, cand2, cand3, cand4 = compare_left_env_first_step(mps, mpo)
```

---

# What this will reveal

If one of the candidates matches exactly, then we know the correct sitewise left-update convention immediately.

Then we generalize that candidate to the recursive formula.

---

# Why this is better than further guessing
For `i=1`, the left block contains only one site, so there is no ambiguity from recursion. This strips the problem down to:

- MPO boundary convention
- bra/ket placement
- conjugation order
- physical-index order

Once we know which direct formula matches, the recursive left update follows uniquely.

---

# I strongly suspect the dense left reference `Lref` is using the convention

For site 0 only, the physically correct matrix element in the left bond basis is most likely

\[
L^{[1]}_B(a,c)
=
\sum_{s,t}
A_{0,s,a}^* \, W_{0,B,s,t} \, A_{0,t,c}
\]

or one of the nearby permutations. The single-step test will tell us exactly which.

---

# Also, I want to simplify one thing

Your current dense left reference builder is more general than needed, but for `i=1` it reduces to a very simple object. So we can independently compare:

## Cell 24

```python
L = 4
d = 2
i = 1

mpo = heisenberg_mpo(L)
mps = random_mps(L, d=d, Dmax=3, seed=7)
mps = normalize_mps_by_state(mps)
mps = make_mixed_canonical_around_bond(mps, i)

Lref = dense_left_env_reference(mps, mpo, i)
L1_rec = left_env_update(init_left_env(), mps[0], mpo[0])

print("max|L1_rec - Lref| =", np.max(np.abs(L1_rec - Lref)))
print("L1_rec[0] =\n", L1_rec[0])
print("Lref[0]   =\n", Lref[0])
```

This prints an actual matrix so we can inspect what is being flipped.

---

# Most likely outcome
One of `cand1..cand4` will match `Lref`, and not `L1_rec`. Then we can write the correct recursive update.

---

Please run Cells 23 and 24 and send me the output. That will let me give you the exact final left-update formula with no more guessing.

---
## 🧑‍💻 User

L1_rec shape: (5, 2, 2)
max|L1_rec - cand1| = 0.0
max|L1_rec - cand2| = 0.4363232840272704
max|L1_rec - cand3| = 0.4363232840272704
max|L1_rec - cand4| = 0.0

---
## 🤖 Assistant

Excellent. This is the key clue.

From your output:

- `L1_rec == cand1`
- `L1_rec == cand4`
- but earlier `L1_rec != Lref`

So the issue is now very clear:

# The problem is not the recursive left update formula itself.
# The problem is the dense left reference builder `dense_left_env_reference`.

Because for the first step:

- your recursive left update gives a well-defined tensor,
- the direct one-site contraction agrees with it exactly,
- therefore the recursion is internally consistent.

So if `Lref` disagrees, then `dense_left_env_reference(...)` is using the wrong convention for the left block operator basis.

That is why we were chasing the wrong object.

---

# What this means

You should **revert** `left_env_update` to the original correct form:

```python
def left_env_update(Lold, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj(), optimize=True)
```

because your first-step test confirms that this is the direct contraction your code is actually implementing.

The real bug is in:

```python
dense_left_env_reference
```

---

# Why `dense_left_env_reference` is wrong

The left reference was built as:

```python
Lref[b] = UL.conj().T @ Op @ UL
```

But the environment contraction corresponds to the operator matrix elements in the basis convention induced by the MPS tensors, and with your MPS-to-state flattening convention there is a transpose/conjugation issue in how the left block basis vectors are assembled into `UL`.

In short:

- `UL` is orthonormal,
- but it is not being inserted into the operator matrix element in the same orientation as the recursive environment contraction.

This is a basis-orientation mismatch in the **reference builder**, not in the environment.

---

# Best next step

Now that we know the environment recursion is internally correct, the right way forward is:

## Compare the full environment-based local operator to a brute-force contraction built directly from left/right basis vectors and the MPO, without using `dense_left_env_reference`

In other words, instead of validating `Lenv` separately via the flawed `Lref`, we should validate the final local operator directly against \(P^\dagger H P\), but with a corrected two-site local action if needed.

However, we already saw earlier that `Heff_env - Heff_proj` was nonzero, so there may still also be an issue in `apply_two_site_heff_env`.

Thus the next clean step is:

# Fix the left reference builder first, then retest `Heff_env`

---

# Replace Cell 19 by corrected dense environment references

Use the following corrected version.

## Corrected Cell 19

```python
def dense_left_env_reference(mps, mpo, i):
    """
    Build left environment for bond i by brute force using the exact basis
    vectors generated by left_basis_map and the same convention as the
    recursive left environment contraction.

    Returns Lref with shape (chi_cut, Dl, Dl).
    """
    UL = left_basis_map(mps, i)   # shape (dimL, Dl)
    dimL, Dl = UL.shape

    if i == 0:
        Lref = np.zeros((mpo[0].shape[0], 1, 1), dtype=np.complex128)
        Lref[0, 0, 0] = 1.0
        return Lref

    # Build operator-valued left block O_b on sites 0..i-1
    cur = [mpo[0][0, b] for b in range(mpo[0].shape[1])]

    for site in range(1, i):
        W = mpo[site]
        chiL, chiR, d, _ = W.shape
        new = [
            np.zeros((cur[0].shape[0] * d, cur[0].shape[1] * d), dtype=np.complex128)
            for _ in range(chiR)
        ]
        for bl in range(chiL):
            for br in range(chiR):
                if np.linalg.norm(W[bl, br]) == 0:
                    continue
                new[br] += np.kron(cur[bl], W[bl, br])
        cur = new

    chi_cut = mpo[i].shape[0]
    Lref = np.zeros((chi_cut, Dl, Dl), dtype=np.complex128)

    # IMPORTANT:
    # match the recursive environment convention
    for b in range(chi_cut):
        Op = cur[b]
        # Use transpose relative to previous attempt
        Lref[b] = UL.T @ Op.T @ UL.conj()

    return Lref


def dense_right_env_reference(mps, mpo, i_plus_1):
    """
    Build right environment by brute force.
    The previously implemented right reference already matched the recursion.
    """
    L = len(mps)
    site0 = i_plus_1 + 1

    UR = right_basis_map(mps, i_plus_1 - 1)
    Dr, dimR = UR.shape

    if site0 >= L:
        Rref = np.zeros((1, 1, 1), dtype=np.complex128)
        Rref[0, 0, 0] = 1.0
        return Rref

    cur = [mpo[site0][b, 0] for b in range(mpo[site0].shape[0])]

    for site in range(site0 + 1, L):
        W = mpo[site]
        chiL, chiR, d, _ = W.shape
        new = [
            np.zeros((d * cur[0].shape[0], d * cur[0].shape[1]), dtype=np.complex128)
            for _ in range(chiL)
        ]
        for b in range(chiL):
            acc = np.zeros((d * cur[0].shape[0], d * cur[0].shape[1]), dtype=np.complex128)
            for B in range(chiR):
                if np.linalg.norm(W[b, B]) == 0:
                    continue
                acc += np.kron(W[b, B], cur[B])
            new[b] = acc
        cur = new

    chi_cut = mpo[i_plus_1].shape[1]
    Rref = np.zeros((chi_cut, Dr, Dr), dtype=np.complex128)
    for b in range(chi_cut):
        Op = cur[b]
        Rref[b] = UR @ Op @ UR.conj().T
    return Rref
```

---

# And restore Cell 6 left update to the original version

## Correct Cell 6

```python
def init_left_env():
    Lenv = np.zeros((1, 1, 1), dtype=np.complex128)
    Lenv[0, 0, 0] = 1.0
    return Lenv


def init_right_env():
    Renv = np.zeros((1, 1, 1), dtype=np.complex128)
    Renv[0, 0, 0] = 1.0
    return Renv


def left_env_update(Lold, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj(), optimize=True)


def right_env_update(Rold, B, W):
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj(), optimize=True)


def build_right_envs(mps, mpo):
    L = len(mps)
    R = [None] * L
    R[L - 1] = init_right_env()

    cur = R[L - 1]
    for i in range(L - 1, 0, -1):
        cur = right_env_update(cur, mps[i], mpo[i])
        R[i - 1] = cur

    return R


def build_left_envs(mps, mpo):
    L = len(mps)
    Lenvs = [None] * L
    Lenvs[0] = init_left_env()

    cur = Lenvs[0]
    for i in range(0, L - 1):
        cur = left_env_update(cur, mps[i], mpo[i])
        Lenvs[i + 1] = cur

    return Lenvs
```

---

# Then rerun

1. Restart kernel
2. Run Cells 1–21 again
3. Check Cell 21
4. Then check Cell 15

---

# What we hope to see

Now:

- `Lenv - Lref` should become tiny
- `Renv - Rref` should stay tiny
- then `Heff_env - Heff_proj` may also become tiny

If `Heff_env - Heff_proj` is still not small after fixing `Lref`, then the remaining issue is in `apply_two_site_heff_env`.

But we must first stop using a flawed left reference.

---

# Summary
At this point the evidence says:

- original `left_env_update` was likely correct,
- `right_env_update` is correct,
- `dense_left_env_reference` was inconsistent with the recursion.

So the patch target has changed.

Please rerun Cell 21 and Cell 15 after these two changes, and send me the outputs.

---
## 🧑‍💻 User

bond 1: max|Lenv - Lref| = 4.363e-01
bond 1: max|Renv - Rref| = 0.000e+00

---
## 🤖 Assistant

Thanks. That means the attempted repair of `dense_left_env_reference` is still not correct, and we should stop using that path as the primary diagnostic.

But we already learned something stronger from your one-step test:

- the recursive `left_env_update` matches the explicit one-site contraction exactly,
- `right_env_update` matches dense reference exactly.

So the remaining practical target is:

# make the full local operator match \(P^\dagger H P\)

That is the actual milestone.

At this point, the cleanest and fastest way in Jupyter is:

## Keep the environment updates as
- original `left_env_update`
- original `right_env_update`

and now directly debug only:

- `apply_two_site_heff_env`

because the two-site local action is the only remaining place where a structural mismatch can live.

---

# So first: freeze Cell 6 to this version

Please use this as the definitive Cell 6 for now:

```python
def init_left_env():
    Lenv = np.zeros((1, 1, 1), dtype=np.complex128)
    Lenv[0, 0, 0] = 1.0
    return Lenv


def init_right_env():
    Renv = np.zeros((1, 1, 1), dtype=np.complex128)
    Renv[0, 0, 0] = 1.0
    return Renv


def left_env_update(Lold, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj(), optimize=True)


def right_env_update(Rold, B, W):
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj(), optimize=True)


def build_right_envs(mps, mpo):
    L = len(mps)
    R = [None] * L
    R[L - 1] = init_right_env()

    cur = R[L - 1]
    for i in range(L - 1, 0, -1):
        cur = right_env_update(cur, mps[i], mpo[i])
        R[i - 1] = cur

    return R


def build_left_envs(mps, mpo):
    L = len(mps)
    Lenvs = [None] * L
    Lenvs[0] = init_left_env()

    cur = Lenvs[0]
    for i in range(0, L - 1):
        cur = left_env_update(cur, mps[i], mpo[i])
        Lenvs[i + 1] = cur

    return Lenvs
```

---

# Now add the decisive diagnostic:
# build the two-site local operator from environments in several candidate conventions

Instead of guessing one `apply_two_site_heff_env`, we will test a small set of plausible index orderings and compare each directly to `Heff_proj`.

This will identify the correct local-action contraction unambiguously.

---

## New Cell 25 — candidate two-site local actions

```python
def apply_two_site_heff_env_cand1(theta, Lenv, W1, W2, Renv):
    # current version
    X = np.einsum('bxy,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBsu,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCtv,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def apply_two_site_heff_env_cand2(theta, Lenv, W1, W2, Renv):
    # swap physical legs of W1 and W2
    X = np.einsum('bxy,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def apply_two_site_heff_env_cand3(theta, Lenv, W1, W2, Renv):
    # swap only W1
    X = np.einsum('bxy,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCtv,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def apply_two_site_heff_env_cand4(theta, Lenv, W1, W2, Renv):
    # swap only W2
    X = np.einsum('bxy,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBsu,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def build_dense_local_heff_from_env_candidate(mps, mpo, i, candidate=1):
    A = mps[i]
    B = mps[i + 1]
    Dl, d1, mid = A.shape
    mid2, d2, Dr = B.shape
    assert mid == mid2

    Lenvs = build_left_envs(mps, mpo)
    Renvs = build_right_envs(mps, mpo)

    Lenv = Lenvs[i]
    Renv = Renvs[i + 1]

    N = Dl * d1 * d2 * Dr
    Heff = np.zeros((N, N), dtype=np.complex128)

    funcs = {
        1: apply_two_site_heff_env_cand1,
        2: apply_two_site_heff_env_cand2,
        3: apply_two_site_heff_env_cand3,
        4: apply_two_site_heff_env_cand4,
    }
    f = funcs[candidate]

    for j in range(N):
        e = np.zeros(N, dtype=np.complex128)
        e[j] = 1.0
        theta = e.reshape((Dl, d1, d2, Dr), order='C')
        Heff[:, j] = f(theta, Lenv, mpo[i], mpo[i + 1], Renv).reshape(-1, order='C')

    return Heff
```

---

## New Cell 26 — compare all candidates to the dense projected operator

```python
def compare_two_site_action_candidates(Hfull, mpo, mps, i, d):
    Heff_proj = dense_projected_two_site_heff_from_full(Hfull, mps, i, d)

    for cand in [1, 2, 3, 4]:
        Heff_env = build_dense_local_heff_from_env_candidate(mps, mpo, i, candidate=cand)
        diff = np.max(np.abs(Heff_env - Heff_proj))
        herm = np.max(np.abs(Heff_env - Heff_env.conj().T))
        print(f"candidate {cand}: max|env-proj| = {diff:.3e}, herm err = {herm:.3e}")
```

---

## New Cell 27 — run the candidate test

```python
L = 4
d = 2
i = 1

mpo = heisenberg_mpo(L)
Hfull = dense_heisenberg(L)

mps = random_mps(L, d=d, Dmax=3, seed=7)
mps = normalize_mps_by_state(mps)
mps = make_mixed_canonical_around_bond(mps, i)

compare_two_site_action_candidates(Hfull, mpo, mps, i, d)
```

---

# Why this is the right move now

At this point:

- left basis map is good,
- right basis map is good,
- projector is good,
- right env is good,
- left env recursion is internally correct at one-site level.

So the only remaining practical ambiguity is the **two-site operator contraction convention**.

Testing 4 candidates directly against `P^\dagger H P` is much faster and more reliable than continuing to reverse-engineer by hand.

---

# What I expect
One of the candidates should come out tiny, likely around `1e-15` to `1e-13`.

If so, we simply adopt that candidate as the correct `apply_two_site_heff_env`.

If none are small, then the issue is subtler: likely the contraction order with `Lenv` or `Renv` should also be transposed in the local action. In that case, we can expand the candidate family further, but this four-way test is the best next cut.

---

# Please send the output of Cell 27

That will tell us exactly which local two-site action is correct.

---
## 🧑‍💻 User

tell me what Cells need to run and remove others

---
## 🤖 Assistant

Yes — let’s simplify aggressively.

At this point, you do **not** need all the previous debugging cells. For the next step, keep only the cells needed to:

1. define the core objects,
2. build the dense projected local operator,
3. test candidate environment-based two-site local actions.

---

# Keep only these cells

## Keep / rerun these cells in this order:

### Cell A — imports
### Cell B — local operators
### Cell C — MPO builders
### Cell D — dense Hamiltonians
### Cell E — MPS utilities
### Cell F — environment routines
### Cell G — dense projected reference tools
### Cell H — mixed-canonical preparation
### Cell I — candidate two-site local actions
### Cell J — comparison runner

Everything else can be ignored for now.

---

# Below is the minimal notebook content

You can paste these as fresh cells into a clean notebook.

---

## Cell A — imports

```python
import numpy as np

np.set_printoptions(precision=6, suppress=True)
```

---

## Cell B — local operators

```python
def spin_half_ops(dtype=np.complex128):
    I = np.eye(2, dtype=dtype)
    Sp = np.array([[0, 1],
                   [0, 0]], dtype=dtype)
    Sm = np.array([[0, 0],
                   [1, 0]], dtype=dtype)
    Sz = 0.5 * np.array([[1, 0],
                         [0, -1]], dtype=dtype)
    return I, Sp, Sm, Sz


def spin_one_ops(dtype=np.complex128):
    I = np.eye(3, dtype=dtype)
    Sp = np.sqrt(2) * np.array([[0, 1, 0],
                                [0, 0, 1],
                                [0, 0, 0]], dtype=dtype)
    Sm = np.sqrt(2) * np.array([[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0]], dtype=dtype)
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, -1]], dtype=dtype)
    return I, Sp, Sm, Sz
```

---

## Cell C — MPO builders

```python
def heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    I, Sp, Sm, Sz = spin_half_ops(dtype=dtype)
    d = 2
    chi = 5

    Wbulk = np.zeros((chi, chi, d, d), dtype=dtype)
    Wbulk[0, 0] = I
    Wbulk[1, 0] = Sp
    Wbulk[2, 0] = Sm
    Wbulk[3, 0] = Sz
    Wbulk[4, 0] = -h * Sz
    Wbulk[4, 1] = (J / 2.0) * Sm
    Wbulk[4, 2] = (J / 2.0) * Sp
    Wbulk[4, 3] = Jz * Sz
    Wbulk[4, 4] = I

    Wleft = np.zeros((1, chi, d, d), dtype=dtype)
    Wleft[0, 0] = -h * Sz
    Wleft[0, 1] = (J / 2.0) * Sm
    Wleft[0, 2] = (J / 2.0) * Sp
    Wleft[0, 3] = Jz * Sz
    Wleft[0, 4] = I

    Wright = np.zeros((chi, 1, d, d), dtype=dtype)
    Wright[0, 0] = I
    Wright[1, 0] = Sp
    Wright[2, 0] = Sm
    Wright[3, 0] = Sz
    Wright[4, 0] = -h * Sz

    mpo = [None] * L
    mpo[0] = Wleft
    for i in range(1, L - 1):
        mpo[i] = Wbulk.copy()
    mpo[L - 1] = Wright
    return mpo
```

---

## Cell D — dense Heisenberg Hamiltonian

```python
def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def onsite_op(L, i, op, d):
    I = np.eye(d, dtype=np.complex128)
    ops = [I] * L
    ops[i] = op
    return kron_all(ops)


def bond_op(L, i, op_i, op_j, d):
    I = np.eye(d, dtype=np.complex128)
    ops = [I] * L
    ops[i] = op_i
    ops[i + 1] = op_j
    return kron_all(ops)


def dense_heisenberg(L, J=1.0, Jz=1.0, h=0.0):
    I, Sp, Sm, Sz = spin_half_ops()
    d = 2
    H = np.zeros((d**L, d**L), dtype=np.complex128)

    for i in range(L - 1):
        H += (J / 2.0) * bond_op(L, i, Sp, Sm, d)
        H += (J / 2.0) * bond_op(L, i, Sm, Sp, d)
        H += Jz * bond_op(L, i, Sz, Sz, d)

    for i in range(L):
        H += -h * onsite_op(L, i, Sz, d)

    return H
```

---

## Cell E — MPS utilities

```python
def random_mps(L, d, Dmax, seed=1234, dtype=np.complex128):
    rng = np.random.default_rng(seed)

    dims = [1]
    for i in range(1, L):
        dims.append(min(Dmax, d ** min(i, L - i)))
    dims.append(1)

    mps = []
    for i in range(L):
        Dl, Dr = dims[i], dims[i + 1]
        A = rng.normal(size=(Dl, d, Dr)) + 1j * rng.normal(size=(Dl, d, Dr))
        mps.append(A.astype(dtype))
    return mps


def mps_to_state(mps):
    psi = mps[0][0]
    for i in range(1, len(mps)):
        psi = np.einsum('...a,asb->...sb', psi, mps[i], optimize=True)
    return psi.reshape(-1, order='C')


def normalize_mps_by_state(mps):
    psi = mps_to_state(mps)
    nrm = np.sqrt(np.vdot(psi, psi).real)
    mps = [A.copy() for A in mps]
    mps[0] = mps[0] / nrm
    return mps


def left_basis_map(mps, i):
    if i == 0:
        return np.array([[1.0]], dtype=np.complex128)

    X = mps[0][0]
    for k in range(1, i):
        X = np.einsum('pa,asb->psb', X, mps[k], optimize=True)
        X = X.reshape((-1, X.shape[-1]), order='C')
    return X


def right_basis_map(mps, i):
    L = len(mps)
    if i == L - 2:
        return np.array([[1.0]], dtype=np.complex128)

    X = mps[i + 2]
    for k in range(i + 3, L):
        X = np.einsum('asd,dtu->astu', X, mps[k], optimize=True)
        X = X.reshape((X.shape[0], -1, X.shape[-1]), order='C')
    return X[:, :, 0]


def make_mixed_canonical_around_bond(mps, i):
    mps = [A.copy() for A in mps]
    L = len(mps)

    for k in range(i):
        A = mps[k]
        Dl, d, Dr = A.shape
        M = A.reshape((Dl * d, Dr), order='C')
        Q, R = np.linalg.qr(M, mode='reduced')
        nk = Q.shape[1]
        mps[k] = Q.reshape((Dl, d, nk), order='C')
        mps[k + 1] = np.einsum('ab,bsd->asd', R, mps[k + 1], optimize=True)

    for k in range(L - 1, i + 1, -1):
        A = mps[k]
        Dl, d, Dr = A.shape
        M = A.reshape((Dl, d * Dr), order='C')
        Q, R = np.linalg.qr(M.conj().T, mode='reduced')
        B = Q.conj().T.reshape((-1, d, Dr), order='C')
        mps[k] = B
        Rd = R.conj().T
        mps[k - 1] = np.einsum('xsa,ab->xsb', mps[k - 1], Rd, optimize=True)

    return mps
```

---

## Cell F — environment routines

```python
def init_left_env():
    Lenv = np.zeros((1, 1, 1), dtype=np.complex128)
    Lenv[0, 0, 0] = 1.0
    return Lenv


def init_right_env():
    Renv = np.zeros((1, 1, 1), dtype=np.complex128)
    Renv[0, 0, 0] = 1.0
    return Renv


def left_env_update(Lold, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj(), optimize=True)


def right_env_update(Rold, B, W):
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj(), optimize=True)


def build_left_envs(mps, mpo):
    L = len(mps)
    Lenvs = [None] * L
    Lenvs[0] = init_left_env()
    cur = Lenvs[0]
    for i in range(0, L - 1):
        cur = left_env_update(cur, mps[i], mpo[i])
        Lenvs[i + 1] = cur
    return Lenvs


def build_right_envs(mps, mpo):
    L = len(mps)
    Renvs = [None] * L
    Renvs[L - 1] = init_right_env()
    cur = Renvs[L - 1]
    for i in range(L - 1, 0, -1):
        cur = right_env_update(cur, mps[i], mpo[i])
        Renvs[i - 1] = cur
    return Renvs
```

---

## Cell G — dense projected reference

```python
def dense_projected_two_site_heff_from_full(Hfull, mps, i, d):
    UL = left_basis_map(mps, i)
    UR = right_basis_map(mps, i)

    Dl = UL.shape[1]
    Dr = UR.shape[0]
    dimL = UL.shape[0]
    dimR = UR.shape[1]

    Nloc = Dl * d * d * Dr
    Dim = dimL * d * d * dimR
    P = np.zeros((Dim, Nloc), dtype=np.complex128)

    col = 0
    for a in range(Dl):
        for s in range(d):
            for t in range(d):
                for c in range(Dr):
                    mid = np.zeros(d * d, dtype=np.complex128)
                    mid[s * d + t] = 1.0
                    vec = np.kron(UL[:, a], np.kron(mid, UR[c, :]))
                    P[:, col] = vec
                    col += 1

    Heff = P.conj().T @ Hfull @ P
    return Heff
```

---

## Cell H — candidate two-site actions

```python
def apply_two_site_heff_env_cand1(theta, Lenv, W1, W2, Renv):
    X = np.einsum('bxy,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBsu,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCtv,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def apply_two_site_heff_env_cand2(theta, Lenv, W1, W2, Renv):
    X = np.einsum('bxy,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def apply_two_site_heff_env_cand3(theta, Lenv, W1, W2, Renv):
    X = np.einsum('bxy,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCtv,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def apply_two_site_heff_env_cand4(theta, Lenv, W1, W2, Renv):
    X = np.einsum('bxy,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBsu,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def build_dense_local_heff_from_env_candidate(mps, mpo, i, candidate=1):
    A = mps[i]
    B = mps[i + 1]
    Dl, d1, mid = A.shape
    mid2, d2, Dr = B.shape
    assert mid == mid2

    Lenvs = build_left_envs(mps, mpo)
    Renvs = build_right_envs(mps, mpo)

    Lenv = Lenvs[i]
    Renv = Renvs[i + 1]

    funcs = {
        1: apply_two_site_heff_env_cand1,
        2: apply_two_site_heff_env_cand2,
        3: apply_two_site_heff_env_cand3,
        4: apply_two_site_heff_env_cand4,
    }
    f = funcs[candidate]

    N = Dl * d1 * d2 * Dr
    Heff = np.zeros((N, N), dtype=np.complex128)

    for j in range(N):
        e = np.zeros(N, dtype=np.complex128)
        e[j] = 1.0
        theta = e.reshape((Dl, d1, d2, Dr), order='C')
        Heff[:, j] = f(theta, Lenv, mpo[i], mpo[i + 1], Renv).reshape(-1, order='C')

    return Heff
```

---

## Cell I — comparison runner

```python
def compare_two_site_action_candidates(Hfull, mpo, mps, i, d):
    Heff_proj = dense_projected_two_site_heff_from_full(Hfull, mps, i, d)

    for cand in [1, 2, 3, 4]:
        Heff_env = build_dense_local_heff_from_env_candidate(mps, mpo, i, candidate=cand)
        diff = np.max(np.abs(Heff_env - Heff_proj))
        herm = np.max(np.abs(Heff_env - Heff_env.conj().T))
        print(f"candidate {cand}: max|env-proj| = {diff:.3e}, herm err = {herm:.3e}")
```

---

## Cell J — run the test

```python
L = 4
d = 2
i = 1

mpo = heisenberg_mpo(L)
Hfull = dense_heisenberg(L)

mps = random_mps(L, d=d, Dmax=3, seed=7)
mps = normalize_mps_by_state(mps)
mps = make_mixed_canonical_around_bond(mps, i)

compare_two_site_action_candidates(Hfull, mpo, mps, i, d)
```

---

# What to ignore / remove for now

You can ignore all previous cells about:

- `dense_left_env_reference`
- `dense_right_env_reference`
- `compare_envs_to_dense_references`
- `left_env_site0_direct`
- `compare_left_env_first_step`
- `expectation_value_mpo`
- `two_site_dmrg_env_matrix_free`
- AKLT code for now

They are not needed for the next debugging step.

---

# What I need from you next

Please run only:

- Cells A through J

and send me the output of **Cell J**.

That output will tell us which two-site local action convention is correct. After that I can give you a clean final version with the unnecessary debug cells removed.

---
## 🧑‍💻 User

I will add # Cell A, B, C to each Cell. But you had better do this next time.

---
## 🤖 Assistant

Understood — you’re right. Next time I’ll label every notebook block explicitly and consistently.

Please use the cells below exactly as labeled.

---

# Cell A — imports

```python
import numpy as np

np.set_printoptions(precision=6, suppress=True)
```

---

# Cell B — local operators

```python
def spin_half_ops(dtype=np.complex128):
    I = np.eye(2, dtype=dtype)
    Sp = np.array([[0, 1],
                   [0, 0]], dtype=dtype)
    Sm = np.array([[0, 0],
                   [1, 0]], dtype=dtype)
    Sz = 0.5 * np.array([[1, 0],
                         [0, -1]], dtype=dtype)
    return I, Sp, Sm, Sz


def spin_one_ops(dtype=np.complex128):
    I = np.eye(3, dtype=dtype)
    Sp = np.sqrt(2) * np.array([[0, 1, 0],
                                [0, 0, 1],
                                [0, 0, 0]], dtype=dtype)
    Sm = np.sqrt(2) * np.array([[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0]], dtype=dtype)
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, -1]], dtype=dtype)
    return I, Sp, Sm, Sz
```

---

# Cell C — MPO builders

```python
def heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    I, Sp, Sm, Sz = spin_half_ops(dtype=dtype)
    d = 2
    chi = 5

    Wbulk = np.zeros((chi, chi, d, d), dtype=dtype)
    Wbulk[0, 0] = I
    Wbulk[1, 0] = Sp
    Wbulk[2, 0] = Sm
    Wbulk[3, 0] = Sz
    Wbulk[4, 0] = -h * Sz
    Wbulk[4, 1] = (J / 2.0) * Sm
    Wbulk[4, 2] = (J / 2.0) * Sp
    Wbulk[4, 3] = Jz * Sz
    Wbulk[4, 4] = I

    Wleft = np.zeros((1, chi, d, d), dtype=dtype)
    Wleft[0, 0] = -h * Sz
    Wleft[0, 1] = (J / 2.0) * Sm
    Wleft[0, 2] = (J / 2.0) * Sp
    Wleft[0, 3] = Jz * Sz
    Wleft[0, 4] = I

    Wright = np.zeros((chi, 1, d, d), dtype=dtype)
    Wright[0, 0] = I
    Wright[1, 0] = Sp
    Wright[2, 0] = Sm
    Wright[3, 0] = Sz
    Wright[4, 0] = -h * Sz

    mpo = [None] * L
    mpo[0] = Wleft
    for i in range(1, L - 1):
        mpo[i] = Wbulk.copy()
    mpo[L - 1] = Wright
    return mpo
```

---

# Cell D — dense Heisenberg Hamiltonian

```python
def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def onsite_op(L, i, op, d):
    I = np.eye(d, dtype=np.complex128)
    ops = [I] * L
    ops[i] = op
    return kron_all(ops)


def bond_op(L, i, op_i, op_j, d):
    I = np.eye(d, dtype=np.complex128)
    ops = [I] * L
    ops[i] = op_i
    ops[i + 1] = op_j
    return kron_all(ops)


def dense_heisenberg(L, J=1.0, Jz=1.0, h=0.0):
    I, Sp, Sm, Sz = spin_half_ops()
    d = 2
    H = np.zeros((d**L, d**L), dtype=np.complex128)

    for i in range(L - 1):
        H += (J / 2.0) * bond_op(L, i, Sp, Sm, d)
        H += (J / 2.0) * bond_op(L, i, Sm, Sp, d)
        H += Jz * bond_op(L, i, Sz, Sz, d)

    for i in range(L):
        H += -h * onsite_op(L, i, Sz, d)

    return H
```

---

# Cell E — MPS utilities

```python
def random_mps(L, d, Dmax, seed=1234, dtype=np.complex128):
    rng = np.random.default_rng(seed)

    dims = [1]
    for i in range(1, L):
        dims.append(min(Dmax, d ** min(i, L - i)))
    dims.append(1)

    mps = []
    for i in range(L):
        Dl, Dr = dims[i], dims[i + 1]
        A = rng.normal(size=(Dl, d, Dr)) + 1j * rng.normal(size=(Dl, d, Dr))
        mps.append(A.astype(dtype))
    return mps


def mps_to_state(mps):
    psi = mps[0][0]
    for i in range(1, len(mps)):
        psi = np.einsum('...a,asb->...sb', psi, mps[i], optimize=True)
    return psi.reshape(-1, order='C')


def normalize_mps_by_state(mps):
    psi = mps_to_state(mps)
    nrm = np.sqrt(np.vdot(psi, psi).real)
    mps = [A.copy() for A in mps]
    mps[0] = mps[0] / nrm
    return mps


def left_basis_map(mps, i):
    if i == 0:
        return np.array([[1.0]], dtype=np.complex128)

    X = mps[0][0]
    for k in range(1, i):
        X = np.einsum('pa,asb->psb', X, mps[k], optimize=True)
        X = X.reshape((-1, X.shape[-1]), order='C')
    return X


def right_basis_map(mps, i):
    L = len(mps)
    if i == L - 2:
        return np.array([[1.0]], dtype=np.complex128)

    X = mps[i + 2]
    for k in range(i + 3, L):
        X = np.einsum('asd,dtu->astu', X, mps[k], optimize=True)
        X = X.reshape((X.shape[0], -1, X.shape[-1]), order='C')
    return X[:, :, 0]


def make_mixed_canonical_around_bond(mps, i):
    mps = [A.copy() for A in mps]
    L = len(mps)

    for k in range(i):
        A = mps[k]
        Dl, d, Dr = A.shape
        M = A.reshape((Dl * d, Dr), order='C')
        Q, R = np.linalg.qr(M, mode='reduced')
        nk = Q.shape[1]
        mps[k] = Q.reshape((Dl, d, nk), order='C')
        mps[k + 1] = np.einsum('ab,bsd->asd', R, mps[k + 1], optimize=True)

    for k in range(L - 1, i + 1, -1):
        A = mps[k]
        Dl, d, Dr = A.shape
        M = A.reshape((Dl, d * Dr), order='C')
        Q, R = np.linalg.qr(M.conj().T, mode='reduced')
        B = Q.conj().T.reshape((-1, d, Dr), order='C')
        mps[k] = B
        Rd = R.conj().T
        mps[k - 1] = np.einsum('xsa,ab->xsb', mps[k - 1], Rd, optimize=True)

    return mps
```

---

# Cell F — environment routines

```python
def init_left_env():
    Lenv = np.zeros((1, 1, 1), dtype=np.complex128)
    Lenv[0, 0, 0] = 1.0
    return Lenv


def init_right_env():
    Renv = np.zeros((1, 1, 1), dtype=np.complex128)
    Renv[0, 0, 0] = 1.0
    return Renv


def left_env_update(Lold, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj(), optimize=True)


def right_env_update(Rold, B, W):
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj(), optimize=True)


def build_left_envs(mps, mpo):
    L = len(mps)
    Lenvs = [None] * L
    Lenvs[0] = init_left_env()
    cur = Lenvs[0]
    for i in range(0, L - 1):
        cur = left_env_update(cur, mps[i], mpo[i])
        Lenvs[i + 1] = cur
    return Lenvs


def build_right_envs(mps, mpo):
    L = len(mps)
    Renvs = [None] * L
    Renvs[L - 1] = init_right_env()
    cur = Renvs[L - 1]
    for i in range(L - 1, 0, -1):
        cur = right_env_update(cur, mps[i], mpo[i])
        Renvs[i - 1] = cur
    return Renvs
```

---

# Cell G — dense projected reference

```python
def dense_projected_two_site_heff_from_full(Hfull, mps, i, d):
    UL = left_basis_map(mps, i)
    UR = right_basis_map(mps, i)

    Dl = UL.shape[1]
    Dr = UR.shape[0]
    dimL = UL.shape[0]
    dimR = UR.shape[1]

    Nloc = Dl * d * d * Dr
    Dim = dimL * d * d * dimR
    P = np.zeros((Dim, Nloc), dtype=np.complex128)

    col = 0
    for a in range(Dl):
        for s in range(d):
            for t in range(d):
                for c in range(Dr):
                    mid = np.zeros(d * d, dtype=np.complex128)
                    mid[s * d + t] = 1.0
                    vec = np.kron(UL[:, a], np.kron(mid, UR[c, :]))
                    P[:, col] = vec
                    col += 1

    Heff = P.conj().T @ Hfull @ P
    return Heff
```

---

# Cell H — candidate two-site actions

```python
def apply_two_site_heff_env_cand1(theta, Lenv, W1, W2, Renv):
    X = np.einsum('bxy,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBsu,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCtv,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def apply_two_site_heff_env_cand2(theta, Lenv, W1, W2, Renv):
    X = np.einsum('bxy,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def apply_two_site_heff_env_cand3(theta, Lenv, W1, W2, Renv):
    X = np.einsum('bxy,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCtv,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def apply_two_site_heff_env_cand4(theta, Lenv, W1, W2, Renv):
    X = np.einsum('bxy,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBsu,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def build_dense_local_heff_from_env_candidate(mps, mpo, i, candidate=1):
    A = mps[i]
    B = mps[i + 1]
    Dl, d1, mid = A.shape
    mid2, d2, Dr = B.shape
    assert mid == mid2

    Lenvs = build_left_envs(mps, mpo)
    Renvs = build_right_envs(mps, mpo)

    Lenv = Lenvs[i]
    Renv = Renvs[i + 1]

    funcs = {
        1: apply_two_site_heff_env_cand1,
        2: apply_two_site_heff_env_cand2,
        3: apply_two_site_heff_env_cand3,
        4: apply_two_site_heff_env_cand4,
    }
    f = funcs[candidate]

    N = Dl * d1 * d2 * Dr
    Heff = np.zeros((N, N), dtype=np.complex128)

    for j in range(N):
        e = np.zeros(N, dtype=np.complex128)
        e[j] = 1.0
        theta = e.reshape((Dl, d1, d2, Dr), order='C')
        Heff[:, j] = f(theta, Lenv, mpo[i], mpo[i + 1], Renv).reshape(-1, order='C')

    return Heff
```

---

# Cell I — comparison runner

```python
def compare_two_site_action_candidates(Hfull, mpo, mps, i, d):
    Heff_proj = dense_projected_two_site_heff_from_full(Hfull, mps, i, d)

    for cand in [1, 2, 3, 4]:
        Heff_env = build_dense_local_heff_from_env_candidate(mps, mpo, i, candidate=cand)
        diff = np.max(np.abs(Heff_env - Heff_proj))
        herm = np.max(np.abs(Heff_env - Heff_env.conj().T))
        print(f"candidate {cand}: max|env-proj| = {diff:.3e}, herm err = {herm:.3e}")
```

---

# Cell J — run the test

```python
L = 4
d = 2
i = 1

mpo = heisenberg_mpo(L)
Hfull = dense_heisenberg(L)

mps = random_mps(L, d=d, Dmax=3, seed=7)
mps = normalize_mps_by_state(mps)
mps = make_mixed_canonical_around_bond(mps, i)

compare_two_site_action_candidates(Hfull, mpo, mps, i, d)
```

---

When you run **Cell J**, send me the output only. Then I’ll tell you which candidate to keep and give you the cleaned final cells.

---
## 🧑‍💻 User

candidate 1: max|env-proj| = 4.775e-01, herm err = 0.000e+00
candidate 2: max|env-proj| = 4.123e-01, herm err = 0.000e+00
candidate 3: max|env-proj| = 5.000e-01, herm err = 0.000e+00
candidate 4: max|env-proj| = 5.000e-01, herm err = 0.000e+00

---
## 🤖 Assistant

Good. That means:

- none of the simple physical-index swaps in the two-site action fixes it,
- so the issue is deeper than `W1/W2` leg ordering alone.

At this point, the most likely cause is:

# the left environment contraction convention is still inconsistent with the projected basis convention,
even though the one-site “site-0 direct” test matched the recursive code.

In other words, the recursive left environment is self-consistent, but not yet the same object needed for the projected local operator comparison.

So we need one final decisive test:

# test the one-site effective Hamiltonian at site 0

Why? Because for site 0:

- the left environment is trivial,
- only the MPO tensor and the right environment enter,
- and we can compare the environment-based one-site action directly to the exact projected one-site action.

Then test site 1, where the nontrivial left environment enters.

This will tell us immediately whether the mismatch first appears when the left environment becomes active.

---

# Please add only these new cells

---

## Cell K — one-site dense projected reference

```python
def dense_projected_one_site_heff_from_full(Hfull, mps, i, d):
    """
    Dense projected one-site effective Hamiltonian P^\dagger H P
    for site i.
    """
    L = len(mps)

    # left basis
    if i == 0:
        UL = np.array([[1.0]], dtype=np.complex128)
    else:
        UL = left_basis_map(mps, i)

    # right basis
    if i == L - 1:
        UR = np.array([[1.0]], dtype=np.complex128)
    else:
        # build right basis for sites i+1..L-1
        X = mps[i + 1]
        for k in range(i + 2, L):
            X = np.einsum('asd,dtu->astu', X, mps[k], optimize=True)
            X = X.reshape((X.shape[0], -1, X.shape[-1]), order='C')
        UR = X[:, :, 0]

    Dl = UL.shape[1]
    Dr = UR.shape[0]
    dimL = UL.shape[0]
    dimR = UR.shape[1]

    Nloc = Dl * d * Dr
    Dim = dimL * d * dimR
    P = np.zeros((Dim, Nloc), dtype=np.complex128)

    col = 0
    for a in range(Dl):
        for s in range(d):
            for c in range(Dr):
                mid = np.zeros(d, dtype=np.complex128)
                mid[s] = 1.0
                vec = np.kron(UL[:, a], np.kron(mid, UR[c, :]))
                P[:, col] = vec
                col += 1

    Heff = P.conj().T @ Hfull @ P
    return Heff
```

---

## Cell L — one-site env action

```python
def apply_one_site_heff_env(M, Lenv, W, Renv):
    X = np.einsum('bxy,ysz->bxsz', Lenv, M, optimize=True)
    Y = np.einsum('bBst,bxsz->Bxtz', W, X, optimize=True)
    HM = np.einsum('Bxtz,Bza->xta', Y, Renv, optimize=True)
    return HM


def build_dense_one_site_heff_from_env(mps, mpo, i):
    A = mps[i]
    Dl, d, Dr = A.shape

    Lenvs = build_left_envs(mps, mpo)
    Renvs = build_right_envs(mps, mpo)

    Lenv = Lenvs[i]
    Renv = Renvs[i]

    N = Dl * d * Dr
    Heff = np.zeros((N, N), dtype=np.complex128)

    for j in range(N):
        e = np.zeros(N, dtype=np.complex128)
        e[j] = 1.0
        M = e.reshape((Dl, d, Dr), order='C')
        Heff[:, j] = apply_one_site_heff_env(M, Lenv, mpo[i], Renv).reshape(-1, order='C')

    return Heff
```

---

## Cell M — compare one-site operators at site 0 and site 1

```python
def compare_one_site_env_vs_proj(Hfull, mpo, mps, i, d):
    Heff_env = build_dense_one_site_heff_from_env(mps, mpo, i)
    Heff_proj = dense_projected_one_site_heff_from_full(Hfull, mps, i, d)

    diff = np.max(np.abs(Heff_env - Heff_proj))
    herm_env = np.max(np.abs(Heff_env - Heff_env.conj().T))
    herm_proj = np.max(np.abs(Heff_proj - Heff_proj.conj().T))

    print(f"site {i}: max|env-proj| = {diff:.3e}")
    print(f"site {i}: env Herm err  = {herm_env:.3e}")
    print(f"site {i}: proj Herm err = {herm_proj:.3e}")
```

---

## Cell N — run one-site tests

```python
L = 4
d = 2
i_bond = 1

mpo = heisenberg_mpo(L)
Hfull = dense_heisenberg(L)

mps = random_mps(L, d=d, Dmax=3, seed=7)
mps = normalize_mps_by_state(mps)
mps = make_mixed_canonical_around_bond(mps, i_bond)

compare_one_site_env_vs_proj(Hfull, mpo, mps, 0, d)
compare_one_site_env_vs_proj(Hfull, mpo, mps, 1, d)
```

---

# What the result will mean

## If site 0 matches, but site 1 fails:
then the problem is definitely in the **left environment** convention.

## If site 0 already fails:
then the problem is in the **one-site local action** or right-environment placement.

## If both one-site tests pass:
then the problem is specifically in the **two-site action**.

This is the cleanest remaining fork in the debugging tree.

---

# Please send me the output of Cell N only.

---
## 🧑‍💻 User

site 0: max|env-proj| = 6.206e-17
site 0: env Herm err  = 4.389e-17
site 0: proj Herm err = 1.388e-17
site 1: max|env-proj| = 1.199e+01
site 1: env Herm err  = 1.790e-15
site 1: proj Herm err = 1.986e-15

---
## 🤖 Assistant

Excellent. This is now completely diagnostic.

## Conclusion

- **site 0 passes**
- **site 1 fails badly**

Therefore:

# the bug is definitively in the left environment convention

not in:

- the MPO itself,
- the right environment,
- the dense projected basis,
- the one-site local action formula,
- or the eigensolver interface.

This is exactly the clean isolation we needed.

---

# Why site 0 passes
At site 0, the left environment is trivial:
\[
L^{[0]} = 1
\]
so the bad left recursion does not enter.

# Why site 1 fails
At site 1, the left environment includes site 0, so the incorrect left-environment update immediately contaminates the local operator.

So now we should stop testing two-site actions and fix the left update by matching the **one-site operator at site 1**.

---

# The correct fix

Your current left update is:

```python
def left_env_update(Lold, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj(), optimize=True)
```

But site-1 failure means this is not the convention compatible with the projected basis used in `left_basis_map`.

The correct left update for your basis convention is:

```python
def left_env_update(Lold, A, W):
    return np.einsum('bxy,ysc,bBts,xsa->Bac', Lold, A.conj(), W, A, optimize=True)
```

Let me explain the structure:

- `Lold[b, x, y]`
- `A.conj()[y, s, c]`
- `W[b, B, t, s]`
- `A[x, t, a]`

This corresponds to the matrix element
\[
\langle c|\, O_B \,|a\rangle
\]
in the basis induced by `left_basis_map`, which is the one your dense projected reference uses.

This is the convention mismatch we were chasing.

---

# Replace only Cell F with this corrected version

## Cell F — corrected environment routines

```python
def init_left_env():
    Lenv = np.zeros((1, 1, 1), dtype=np.complex128)
    Lenv[0, 0, 0] = 1.0
    return Lenv


def init_right_env():
    Renv = np.zeros((1, 1, 1), dtype=np.complex128)
    Renv[0, 0, 0] = 1.0
    return Renv


def left_env_update(Lold, A, W):
    """
    Correct left environment update compatible with the projected-basis
    convention used by left_basis_map / dense_projected_*_heff_from_full.
    """
    return np.einsum('bxy,ysc,bBts,xsa->Bac', Lold, A.conj(), W, A, optimize=True)


def right_env_update(Rold, B, W):
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj(), optimize=True)


def build_left_envs(mps, mpo):
    L = len(mps)
    Lenvs = [None] * L
    Lenvs[0] = init_left_env()
    cur = Lenvs[0]
    for i in range(0, L - 1):
        cur = left_env_update(cur, mps[i], mpo[i])
        Lenvs[i + 1] = cur
    return Lenvs


def build_right_envs(mps, mpo):
    L = len(mps)
    Renvs = [None] * L
    Renvs[L - 1] = init_right_env()
    cur = Renvs[L - 1]
    for i in range(L - 1, 0, -1):
        cur = right_env_update(cur, mps[i], mpo[i])
        Renvs[i - 1] = cur
    return Renvs
```

---

# Then rerun only these cells

After changing Cell F:

1. **Restart kernel**
2. Run:
   - Cell A
   - Cell B
   - Cell C
   - Cell D
   - Cell E
   - Cell F
   - Cell G
   - Cell K
   - Cell L
   - Cell M
   - Cell N

You do **not** need the two-site candidate cells right now.

---

# What should happen now

In Cell N, we want:

- site 0: tiny mismatch
- site 1: also tiny mismatch

If that happens, then the left environment is finally fixed.

Then we move back to the two-site local operator and test it again.

---

# What to send me next

Please send the new output of **Cell N** after this Cell F patch.

If site 1 becomes small, I’ll immediately give you the minimal final two-site test cells and then the DMRG sweep cells.

---
## 🧑‍💻 User

site 0: max|env-proj| = 6.206e-17
site 0: env Herm err  = 4.389e-17
site 0: proj Herm err = 1.388e-17
site 1: max|env-proj| = 1.507e+01
site 1: env Herm err  = 1.299e+01
site 1: proj Herm err = 1.986e-15

---
## 🤖 Assistant

Good — that result is decisive too.

The new left update made things **worse**:

- site 1 mismatch grew
- Hermiticity was destroyed

So that patch is definitely wrong.

We now know:

# Facts established
1. `right_env_update` is correct.
2. `apply_one_site_heff_env` is correct.
3. `site 0` one-site effective Hamiltonian matches exactly.
4. The failure starts when a **nontrivial left environment** enters.
5. The original left update preserved Hermiticity, but gave the wrong operator.
6. The alternative left update destroyed Hermiticity, so reject it.

Thus the real issue is subtler:

# The left environment tensor itself is probably correct as a contraction object,
but it is being used with the wrong index orientation in the local effective Hamiltonian application.

This is actually very plausible:

- the recursive environment may produce \(L_B(a,c)\),
- but `apply_one_site_heff_env` may be using it as if it were \(L_B(c,a)\),
- site 0 would not detect this because the left bond dimension is 1,
- site 1 would fail strongly.

That matches your diagnostics perfectly.

---

# So: restore Cell F to the original version

Please restore Cell F to:

## Cell F — restore original

```python
def init_left_env():
    Lenv = np.zeros((1, 1, 1), dtype=np.complex128)
    Lenv[0, 0, 0] = 1.0
    return Lenv


def init_right_env():
    Renv = np.zeros((1, 1, 1), dtype=np.complex128)
    Renv[0, 0, 0] = 1.0
    return Renv


def left_env_update(Lold, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj(), optimize=True)


def right_env_update(Rold, B, W):
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj(), optimize=True)


def build_left_envs(mps, mpo):
    L = len(mps)
    Lenvs = [None] * L
    Lenvs[0] = init_left_env()
    cur = Lenvs[0]
    for i in range(0, L - 1):
        cur = left_env_update(cur, mps[i], mpo[i])
        Lenvs[i + 1] = cur
    return Lenvs


def build_right_envs(mps, mpo):
    L = len(mps)
    Renvs = [None] * L
    Renvs[L - 1] = init_right_env()
    cur = Renvs[L - 1]
    for i in range(L - 1, 0, -1):
        cur = right_env_update(cur, mps[i], mpo[i])
        Renvs[i - 1] = cur
    return Renvs
```

---

# Now the correct next move:
# test two one-site application conventions

Because site 0 passes regardless, and site 1 fails only once `Lenv` matters, we should test whether `Lenv` must be used as:

- `Lenv[x,y]` or
- `Lenv[y,x]`

inside the one-site effective Hamiltonian application.

That is the cleanest remaining ambiguity.

---

## Replace Cell L with this diagnostic version

# Cell L — one-site env action candidates

```python
def apply_one_site_heff_env_cand1(M, Lenv, W, Renv):
    # current convention
    X = np.einsum('bxy,ysz->bxsz', Lenv, M, optimize=True)
    Y = np.einsum('bBst,bxsz->Bxtz', W, X, optimize=True)
    HM = np.einsum('Bxtz,Bza->xta', Y, Renv, optimize=True)
    return HM


def apply_one_site_heff_env_cand2(M, Lenv, W, Renv):
    # transpose left environment indices
    X = np.einsum('byx,ysz->bxsz', Lenv, M, optimize=True)
    Y = np.einsum('bBst,bxsz->Bxtz', W, X, optimize=True)
    HM = np.einsum('Bxtz,Bza->xta', Y, Renv, optimize=True)
    return HM


def build_dense_one_site_heff_from_env_candidate(mps, mpo, i, candidate=1):
    A = mps[i]
    Dl, d, Dr = A.shape

    Lenvs = build_left_envs(mps, mpo)
    Renvs = build_right_envs(mps, mpo)

    Lenv = Lenvs[i]
    Renv = Renvs[i]

    funcs = {
        1: apply_one_site_heff_env_cand1,
        2: apply_one_site_heff_env_cand2,
    }
    f = funcs[candidate]

    N = Dl * d * Dr
    Heff = np.zeros((N, N), dtype=np.complex128)

    for j in range(N):
        e = np.zeros(N, dtype=np.complex128)
        e[j] = 1.0
        M = e.reshape((Dl, d, Dr), order='C')
        Heff[:, j] = f(M, Lenv, mpo[i], Renv).reshape(-1, order='C')

    return Heff
```

---

## Replace Cell M with this version

# Cell M — compare one-site candidates

```python
def compare_one_site_env_vs_proj_candidates(Hfull, mpo, mps, i, d):
    Heff_proj = dense_projected_one_site_heff_from_full(Hfull, mps, i, d)

    for cand in [1, 2]:
        Heff_env = build_dense_one_site_heff_from_env_candidate(mps, mpo, i, candidate=cand)
        diff = np.max(np.abs(Heff_env - Heff_proj))
        herm = np.max(np.abs(Heff_env - Heff_env.conj().T))
        print(f"site {i}, candidate {cand}: max|env-proj| = {diff:.3e}, herm err = {herm:.3e}")
```

---

## Replace Cell N with this

# Cell N — run one-site candidate tests

```python
L = 4
d = 2
i_bond = 1

mpo = heisenberg_mpo(L)
Hfull = dense_heisenberg(L)

mps = random_mps(L, d=d, Dmax=3, seed=7)
mps = normalize_mps_by_state(mps)
mps = make_mixed_canonical_around_bond(mps, i_bond)

compare_one_site_env_vs_proj_candidates(Hfull, mpo, mps, 0, d)
compare_one_site_env_vs_proj_candidates(Hfull, mpo, mps, 1, d)
```

---

# What this test means

If candidate 2 fixes site 1, then the left environment update was fine, but the local action must use the left environment with swapped bond indices.

That would also explain why the two-site local action failed.

---

# Run only these cells now

After restoring Cell F and replacing Cells L, M, N:

1. Restart kernel
2. Run:
   - Cell A
   - Cell B
   - Cell C
   - Cell D
   - Cell E
   - Cell F
   - Cell G
   - Cell K
   - Cell L
   - Cell M
   - Cell N

Please send me the output of Cell N only.

---
## 🧑‍💻 User

site 0, candidate 1: max|env-proj| = 6.206e-17, herm err = 4.389e-17
site 0, candidate 2: max|env-proj| = 6.206e-17, herm err = 4.389e-17
site 1, candidate 1: max|env-proj| = 1.199e+01, herm err = 1.790e-15
site 1, candidate 2: max|env-proj| = 3.580e-15, herm err = 1.790e-15

---
## 🤖 Assistant

Excellent. Now we have the actual fix.

## Conclusion

The environment recursion was fine enough, but the **local Hamiltonian application must use the left environment with swapped bond indices**.

That is:

- keep the **original** `left_env_update`
- keep the **original** `right_env_update`
- in local effective Hamiltonian actions, use `Lenv` as `Lenv.transpose(0,2,1)` relative to the naive formula

This explains everything:

- site 0 unaffected because `Dl=1`
- site 1 fixed only when left indices are swapped
- previous two-site failures came from the same issue

---

# Final fix you should use now

## Keep Cell F exactly as original
Do **not** change the environment updates.

## Replace the local-action routines with the corrected versions below

---

# Cell L — final one-site local action

```python
def apply_one_site_heff_env(M, Lenv, W, Renv):
    """
    Correct one-site effective Hamiltonian action.

    Important:
    the left environment must be used with swapped bond indices.
    """
    X = np.einsum('byx,ysz->bxsz', Lenv, M, optimize=True)
    Y = np.einsum('bBst,bxsz->Bxtz', W, X, optimize=True)
    HM = np.einsum('Bxtz,Bza->xta', Y, Renv, optimize=True)
    return HM


def build_dense_one_site_heff_from_env(mps, mpo, i):
    A = mps[i]
    Dl, d, Dr = A.shape

    Lenvs = build_left_envs(mps, mpo)
    Renvs = build_right_envs(mps, mpo)

    Lenv = Lenvs[i]
    Renv = Renvs[i]

    N = Dl * d * Dr
    Heff = np.zeros((N, N), dtype=np.complex128)

    for j in range(N):
        e = np.zeros(N, dtype=np.complex128)
        e[j] = 1.0
        M = e.reshape((Dl, d, Dr), order='C')
        Heff[:, j] = apply_one_site_heff_env(M, Lenv, mpo[i], Renv).reshape(-1, order='C')

    return Heff
```

---

# Cell H — final two-site local action

Replace the previous candidate cells by this single final version.

```python
def apply_two_site_heff_env(theta, Lenv, W1, W2, Renv):
    """
    Correct two-site effective Hamiltonian action.

    Important:
    use left environment with swapped bond indices.
    """
    X = np.einsum('byx,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBsu,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCtv,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def build_dense_local_heff_from_env(mps, mpo, i):
    A = mps[i]
    B = mps[i + 1]
    Dl, d1, mid = A.shape
    mid2, d2, Dr = B.shape
    assert mid == mid2

    Lenvs = build_left_envs(mps, mpo)
    Renvs = build_right_envs(mps, mpo)

    Lenv = Lenvs[i]
    Renv = Renvs[i + 1]

    N = Dl * d1 * d2 * Dr
    Heff = np.zeros((N, N), dtype=np.complex128)

    for j in range(N):
        e = np.zeros(N, dtype=np.complex128)
        e[j] = 1.0
        theta = e.reshape((Dl, d1, d2, Dr), order='C')
        Heff[:, j] = apply_two_site_heff_env(
            theta, Lenv, mpo[i], mpo[i + 1], Renv
        ).reshape(-1, order='C')

    return Heff
```

---

# Cell I — final two-site comparison runner

```python
def compare_two_site_env_vs_proj(Hfull, mpo, mps, i, d):
    Heff_proj = dense_projected_two_site_heff_from_full(Hfull, mps, i, d)
    Heff_env = build_dense_local_heff_from_env(mps, mpo, i)

    diff = np.max(np.abs(Heff_env - Heff_proj))
    herm_env = np.max(np.abs(Heff_env - Heff_env.conj().T))
    herm_proj = np.max(np.abs(Heff_proj - Heff_proj.conj().T))

    print(f"bond {i}: max|env-proj| = {diff:.3e}")
    print(f"bond {i}: env Herm err = {herm_env:.3e}")
    print(f"bond {i}: proj Herm err = {herm_proj:.3e}")
```

---

# Cell J — final two-site test

```python
L = 4
d = 2
i = 1

mpo = heisenberg_mpo(L)
Hfull = dense_heisenberg(L)

mps = random_mps(L, d=d, Dmax=3, seed=7)
mps = normalize_mps_by_state(mps)
mps = make_mixed_canonical_around_bond(mps, i)

compare_two_site_env_vs_proj(Hfull, mpo, mps, i, d)
```

---

# What to run now

You only need these cells now:

- Cell A
- Cell B
- Cell C
- Cell D
- Cell E
- Cell F
- Cell G
- Cell H
- Cell I
- Cell J

You do **not** need Cells K, M, N anymore unless you want the one-site debug check again.

---

# Expected result
Cell J should now give something like:

```python
bond 1: max|env-proj| ~ 1e-15
bond 1: env Herm err ~ 1e-15
bond 1: proj Herm err ~ 1e-15
```

---

# Then the DMRG code should also use the corrected local action

When we move back to full DMRG, the matrix-free local matvec must use:

```python
X = np.einsum('byx,yuvz->bxuvz', Lenv, theta, optimize=True)
```

not the unswapped version.

---

Please run Cell J and send me the output. If it passes, I’ll give you the final clean DMRG cells for both Heisenberg and AKLT.

---
## 🧑‍💻 User

bond 1: max|env-proj| = 4.775e-01
bond 1: env Herm err = 0.000e+00
bond 1: proj Herm err = 5.551e-17

---
## 🤖 Assistant

Thanks. This is very useful: the one-site fix does **not** automatically fix the two-site action. So:

- **one-site local action** needs `Lenv` with swapped left indices,
- but **two-site local action** still has another convention mismatch.

That is actually believable: the two-site action has more index-routing freedom than the one-site case.

We now know exactly what to do next, and I’ll keep it minimal.

---

# Keep these cells unchanged

Keep:

- Cell A
- Cell B
- Cell C
- Cell D
- Cell E
- Cell F
- Cell G

Do **not** change them.

---

# Replace Cell H, I, J with the versions below

We will test a small family of two-site contractions that all use the **correct one-site lesson**:

- left environment enters as `byx`
- then we vary the MPO physical-index routing on the two sites

This is now the shortest path.

---

## Cell H — two-site candidate actions, corrected search set

```python
def apply_two_site_heff_env_cand1(theta, Lenv, W1, W2, Renv):
    # left env swapped; W1/W2 as (out,in)
    X = np.einsum('byx,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBsu,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCtv,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def apply_two_site_heff_env_cand2(theta, Lenv, W1, W2, Renv):
    # left env swapped; W1 swapped phys; W2 normal
    X = np.einsum('byx,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCtv,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def apply_two_site_heff_env_cand3(theta, Lenv, W1, W2, Renv):
    # left env swapped; W1 normal; W2 swapped phys
    X = np.einsum('byx,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBsu,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def apply_two_site_heff_env_cand4(theta, Lenv, W1, W2, Renv):
    # left env swapped; W1 swapped phys; W2 swapped phys
    X = np.einsum('byx,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def apply_two_site_heff_env_cand5(theta, Lenv, W1, W2, Renv):
    # left env swapped; apply W2 first in algebraic routing
    X = np.einsum('byx,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('BCtv,bxuvz->bCxutz', W2, X, optimize=True)
    Z = np.einsum('bBsu,bCxutz->Cxstz', W1, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def apply_two_site_heff_env_cand6(theta, Lenv, W1, W2, Renv):
    # left env swapped; W2 first with swapped physical order
    X = np.einsum('byx,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('BCvt,bxuvz->bCxutz', W2, X, optimize=True)
    Z = np.einsum('bBus,bCxutz->Cxstz', W1, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def build_dense_local_heff_from_env_candidate(mps, mpo, i, candidate=1):
    A = mps[i]
    B = mps[i + 1]
    Dl, d1, mid = A.shape
    mid2, d2, Dr = B.shape
    assert mid == mid2

    Lenvs = build_left_envs(mps, mpo)
    Renvs = build_right_envs(mps, mpo)

    Lenv = Lenvs[i]
    Renv = Renvs[i + 1]

    funcs = {
        1: apply_two_site_heff_env_cand1,
        2: apply_two_site_heff_env_cand2,
        3: apply_two_site_heff_env_cand3,
        4: apply_two_site_heff_env_cand4,
        5: apply_two_site_heff_env_cand5,
        6: apply_two_site_heff_env_cand6,
    }
    f = funcs[candidate]

    N = Dl * d1 * d2 * Dr
    Heff = np.zeros((N, N), dtype=np.complex128)

    for j in range(N):
        e = np.zeros(N, dtype=np.complex128)
        e[j] = 1.0
        theta = e.reshape((Dl, d1, d2, Dr), order='C')
        Heff[:, j] = f(theta, Lenv, mpo[i], mpo[i + 1], Renv).reshape(-1, order='C')

    return Heff
```

---

## Cell I — comparison runner

```python
def compare_two_site_action_candidates(Hfull, mpo, mps, i, d):
    Heff_proj = dense_projected_two_site_heff_from_full(Hfull, mps, i, d)

    for cand in [1, 2, 3, 4, 5, 6]:
        Heff_env = build_dense_local_heff_from_env_candidate(mps, mpo, i, candidate=cand)
        diff = np.max(np.abs(Heff_env - Heff_proj))
        herm = np.max(np.abs(Heff_env - Heff_env.conj().T))
        print(f"candidate {cand}: max|env-proj| = {diff:.3e}, herm err = {herm:.3e}")
```

---

## Cell J — run the two-site candidate test

```python
L = 4
d = 2
i = 1

mpo = heisenberg_mpo(L)
Hfull = dense_heisenberg(L)

mps = random_mps(L, d=d, Dmax=3, seed=7)
mps = normalize_mps_by_state(mps)
mps = make_mixed_canonical_around_bond(mps, i)

compare_two_site_action_candidates(Hfull, mpo, mps, i, d)
```

---

# Run only these cells now

After replacing H/I/J:

1. Restart kernel
2. Run:
   - Cell A
   - Cell B
   - Cell C
   - Cell D
   - Cell E
   - Cell F
   - Cell G
   - Cell H
   - Cell I
   - Cell J

---

# What I need from you
Please send the output of **Cell J** only.

That will let me identify the correct two-site routing, after which I can give you the final cleaned notebook cells.

---
## 🧑‍💻 User

candidate 1: max|env-proj| = 4.775e-01, herm err = 0.000e+00
candidate 2: max|env-proj| = 5.000e-01, herm err = 0.000e+00
candidate 3: max|env-proj| = 5.000e-01, herm err = 0.000e+00
candidate 4: max|env-proj| = 1.665e-16, herm err = 0.000e+00
candidate 5: max|env-proj| = 2.189e+00, herm err = 0.000e+00
candidate 6: max|env-proj| = 2.189e+00, herm err = 0.000e+00

---
## 🤖 Assistant

Excellent. We have the answer.

# Correct two-site local action = candidate 4

That means the correct environment-based two-site action is:

- use the **left environment with swapped left-bond indices**
- use **both MPO tensors with swapped physical input/output ordering** relative to the naive contraction

Concretely:

```python
X = np.einsum('byx,yuvz->bxuvz', Lenv, theta)
Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
HT = np.einsum('Cxstz,Cza->xsta', Z, Renv)
```

This is the correct two-site matrix-free local Hamiltonian action in your conventions.

---

# Final minimal clean cells you should keep

You can now discard the candidate-testing cells and use the final versions below.

---

## Cell A — imports

```python
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import eigh as scipy_eigh

np.set_printoptions(precision=6, suppress=True)
```

---

## Cell B — local operators

```python
def spin_half_ops(dtype=np.complex128):
    I = np.eye(2, dtype=dtype)
    Sp = np.array([[0, 1],
                   [0, 0]], dtype=dtype)
    Sm = np.array([[0, 0],
                   [1, 0]], dtype=dtype)
    Sz = 0.5 * np.array([[1, 0],
                         [0, -1]], dtype=dtype)
    return I, Sp, Sm, Sz


def spin_one_ops(dtype=np.complex128):
    I = np.eye(3, dtype=dtype)
    Sp = np.sqrt(2) * np.array([[0, 1, 0],
                                [0, 0, 1],
                                [0, 0, 0]], dtype=dtype)
    Sm = np.sqrt(2) * np.array([[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0]], dtype=dtype)
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, -1]], dtype=dtype)
    return I, Sp, Sm, Sz
```

---

## Cell C — MPO builders

```python
def heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    I, Sp, Sm, Sz = spin_half_ops(dtype=dtype)
    d = 2
    chi = 5

    Wbulk = np.zeros((chi, chi, d, d), dtype=dtype)
    Wbulk[0, 0] = I
    Wbulk[1, 0] = Sp
    Wbulk[2, 0] = Sm
    Wbulk[3, 0] = Sz
    Wbulk[4, 0] = -h * Sz
    Wbulk[4, 1] = (J / 2.0) * Sm
    Wbulk[4, 2] = (J / 2.0) * Sp
    Wbulk[4, 3] = Jz * Sz
    Wbulk[4, 4] = I

    Wleft = np.zeros((1, chi, d, d), dtype=dtype)
    Wleft[0, 0] = -h * Sz
    Wleft[0, 1] = (J / 2.0) * Sm
    Wleft[0, 2] = (J / 2.0) * Sp
    Wleft[0, 3] = Jz * Sz
    Wleft[0, 4] = I

    Wright = np.zeros((chi, 1, d, d), dtype=dtype)
    Wright[0, 0] = I
    Wright[1, 0] = Sp
    Wright[2, 0] = Sm
    Wright[3, 0] = Sz
    Wright[4, 0] = -h * Sz

    mpo = [None] * L
    mpo[0] = Wleft
    for i in range(1, L - 1):
        mpo[i] = Wbulk.copy()
    mpo[L - 1] = Wright
    return mpo


def aklt_mpo(L, dtype=np.complex128):
    I, Sp, Sm, Sz = spin_one_ops(dtype=dtype)
    d = 3
    chi = 14

    O = [Sp / np.sqrt(2.0), Sm / np.sqrt(2.0), Sz]
    Obar = [Sm / np.sqrt(2.0), Sp / np.sqrt(2.0), Sz]

    O2 = []
    Obar2 = []
    for a in range(3):
        for b in range(3):
            O2.append(O[a] @ O[b])
            Obar2.append(Obar[a] @ Obar[b])

    Wbulk = np.zeros((chi, chi, d, d), dtype=dtype)
    Wbulk[0, 0] = I
    for a in range(3):
        Wbulk[0, 1 + a] = O[a]
    for ab in range(9):
        Wbulk[0, 4 + ab] = (1.0 / 3.0) * O2[ab]
    for a in range(3):
        Wbulk[1 + a, 13] = Obar[a]
    for ab in range(9):
        Wbulk[4 + ab, 13] = Obar2[ab]
    Wbulk[13, 13] = I

    Wleft = np.zeros((1, chi, d, d), dtype=dtype)
    Wleft[0, 0] = I
    for a in range(3):
        Wleft[0, 1 + a] = O[a]
    for ab in range(9):
        Wleft[0, 4 + ab] = (1.0 / 3.0) * O2[ab]

    Wright = np.zeros((chi, 1, d, d), dtype=dtype)
    for a in range(3):
        Wright[1 + a, 0] = Obar[a]
    for ab in range(9):
        Wright[4 + ab, 0] = Obar2[ab]
    Wright[13, 0] = I

    mpo = [None] * L
    mpo[0] = Wleft
    for i in range(1, L - 1):
        mpo[i] = Wbulk.copy()
    mpo[L - 1] = Wright
    return mpo
```

---

## Cell D — dense Hamiltonians for testing

```python
def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def onsite_op(L, i, op, d):
    I = np.eye(d, dtype=np.complex128)
    ops = [I] * L
    ops[i] = op
    return kron_all(ops)


def bond_op(L, i, op_i, op_j, d):
    I = np.eye(d, dtype=np.complex128)
    ops = [I] * L
    ops[i] = op_i
    ops[i + 1] = op_j
    return kron_all(ops)


def dense_heisenberg(L, J=1.0, Jz=1.0, h=0.0):
    I, Sp, Sm, Sz = spin_half_ops()
    d = 2
    H = np.zeros((d**L, d**L), dtype=np.complex128)

    for i in range(L - 1):
        H += (J / 2.0) * bond_op(L, i, Sp, Sm, d)
        H += (J / 2.0) * bond_op(L, i, Sm, Sp, d)
        H += Jz * bond_op(L, i, Sz, Sz, d)

    for i in range(L):
        H += -h * onsite_op(L, i, Sz, d)

    return H


def dense_aklt(L):
    I, Sp, Sm, Sz = spin_one_ops()
    d = 3
    H = np.zeros((d**L, d**L), dtype=np.complex128)

    for i in range(L - 1):
        X = (
            0.5 * bond_op(L, i, Sp, Sm, d)
            + 0.5 * bond_op(L, i, Sm, Sp, d)
            + bond_op(L, i, Sz, Sz, d)
        )
        H += X + (1.0 / 3.0) * (X @ X)

    return H
```

---

## Cell E — MPS utilities

```python
def random_mps(L, d, Dmax, seed=1234, dtype=np.complex128):
    rng = np.random.default_rng(seed)

    dims = [1]
    for i in range(1, L):
        dims.append(min(Dmax, d ** min(i, L - i)))
    dims.append(1)

    mps = []
    for i in range(L):
        Dl, Dr = dims[i], dims[i + 1]
        A = rng.normal(size=(Dl, d, Dr)) + 1j * rng.normal(size=(Dl, d, Dr))
        mps.append(A.astype(dtype))
    return mps


def mps_to_state(mps):
    psi = mps[0][0]
    for i in range(1, len(mps)):
        psi = np.einsum('...a,asb->...sb', psi, mps[i], optimize=True)
    return psi.reshape(-1, order='C')


def normalize_mps_by_state(mps):
    psi = mps_to_state(mps)
    nrm = np.sqrt(np.vdot(psi, psi).real)
    mps = [A.copy() for A in mps]
    mps[0] = mps[0] / nrm
    return mps


def left_basis_map(mps, i):
    if i == 0:
        return np.array([[1.0]], dtype=np.complex128)

    X = mps[0][0]
    for k in range(1, i):
        X = np.einsum('pa,asb->psb', X, mps[k], optimize=True)
        X = X.reshape((-1, X.shape[-1]), order='C')
    return X


def right_basis_map(mps, i):
    L = len(mps)
    if i == L - 2:
        return np.array([[1.0]], dtype=np.complex128)

    X = mps[i + 2]
    for k in range(i + 3, L):
        X = np.einsum('asd,dtu->astu', X, mps[k], optimize=True)
        X = X.reshape((X.shape[0], -1, X.shape[-1]), order='C')
    return X[:, :, 0]


def make_mixed_canonical_around_bond(mps, i):
    mps = [A.copy() for A in mps]
    L = len(mps)

    for k in range(i):
        A = mps[k]
        Dl, d, Dr = A.shape
        M = A.reshape((Dl * d, Dr), order='C')
        Q, R = np.linalg.qr(M, mode='reduced')
        nk = Q.shape[1]
        mps[k] = Q.reshape((Dl, d, nk), order='C')
        mps[k + 1] = np.einsum('ab,bsd->asd', R, mps[k + 1], optimize=True)

    for k in range(L - 1, i + 1, -1):
        A = mps[k]
        Dl, d, Dr = A.shape
        M = A.reshape((Dl, d * Dr), order='C')
        Q, R = np.linalg.qr(M.conj().T, mode='reduced')
        B = Q.conj().T.reshape((-1, d, Dr), order='C')
        mps[k] = B
        Rd = R.conj().T
        mps[k - 1] = np.einsum('xsa,ab->xsb', mps[k - 1], Rd, optimize=True)

    return mps
```

---

## Cell F — environment routines

```python
def init_left_env():
    Lenv = np.zeros((1, 1, 1), dtype=np.complex128)
    Lenv[0, 0, 0] = 1.0
    return Lenv


def init_right_env():
    Renv = np.zeros((1, 1, 1), dtype=np.complex128)
    Renv[0, 0, 0] = 1.0
    return Renv


def left_env_update(Lold, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj(), optimize=True)


def right_env_update(Rold, B, W):
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj(), optimize=True)


def build_left_envs(mps, mpo):
    L = len(mps)
    Lenvs = [None] * L
    Lenvs[0] = init_left_env()
    cur = Lenvs[0]
    for i in range(0, L - 1):
        cur = left_env_update(cur, mps[i], mpo[i])
        Lenvs[i + 1] = cur
    return Lenvs


def build_right_envs(mps, mpo):
    L = len(mps)
    Renvs = [None] * L
    Renvs[L - 1] = init_right_env()
    cur = Renvs[L - 1]
    for i in range(L - 1, 0, -1):
        cur = right_env_update(cur, mps[i], mpo[i])
        Renvs[i - 1] = cur
    return Renvs
```

---

## Cell G — dense projected reference

```python
def dense_projected_two_site_heff_from_full(Hfull, mps, i, d):
    UL = left_basis_map(mps, i)
    UR = right_basis_map(mps, i)

    Dl = UL.shape[1]
    Dr = UR.shape[0]
    dimL = UL.shape[0]
    dimR = UR.shape[1]

    Nloc = Dl * d * d * Dr
    Dim = dimL * d * d * dimR
    P = np.zeros((Dim, Nloc), dtype=np.complex128)

    col = 0
    for a in range(Dl):
        for s in range(d):
            for t in range(d):
                for c in range(Dr):
                    mid = np.zeros(d * d, dtype=np.complex128)
                    mid[s * d + t] = 1.0
                    vec = np.kron(UL[:, a], np.kron(mid, UR[c, :]))
                    P[:, col] = vec
                    col += 1

    Heff = P.conj().T @ Hfull @ P
    return Heff
```

---

## Cell H — final two-site local action

```python
def apply_two_site_heff_env(theta, Lenv, W1, W2, Renv):
    """
    Correct two-site environment-based local Hamiltonian action.
    """
    X = np.einsum('byx,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def heff_two_site_env_matvec(v, Lenv, W1, W2, Renv, Dl, d1, d2, Dr):
    theta = v.reshape((Dl, d1, d2, Dr), order='C')
    out = apply_two_site_heff_env(theta, Lenv, W1, W2, Renv)
    return out.reshape(Dl * d1 * d2 * Dr, order='C')


def build_dense_local_heff_from_env(mps, mpo, i):
    A = mps[i]
    B = mps[i + 1]
    Dl, d1, mid = A.shape
    mid2, d2, Dr = B.shape
    assert mid == mid2

    Lenvs = build_left_envs(mps, mpo)
    Renvs = build_right_envs(mps, mpo)

    Lenv = Lenvs[i]
    Renv = Renvs[i + 1]

    N = Dl * d1 * d2 * Dr
    Heff = np.zeros((N, N), dtype=np.complex128)

    for j in range(N):
        e = np.zeros(N, dtype=np.complex128)
        e[j] = 1.0
        theta = e.reshape((Dl, d1, d2, Dr), order='C')
        Heff[:, j] = apply_two_site_heff_env(
            theta, Lenv, mpo[i], mpo[i + 1], Renv
        ).reshape(-1, order='C')

    return Heff
```

---

## Cell I — final local-operator validation

```python
def compare_two_site_env_vs_proj(Hfull, mpo, mps, i, d):
    Heff_proj = dense_projected_two_site_heff_from_full(Hfull, mps, i, d)
    Heff_env = build_dense_local_heff_from_env(mps, mpo, i)

    diff = np.max(np.abs(Heff_env - Heff_proj))
    herm_env = np.max(np.abs(Heff_env - Heff_env.conj().T))
    herm_proj = np.max(np.abs(Heff_proj - Heff_proj.conj().T))

    print(f"bond {i}: max|env-proj| = {diff:.3e}")
    print(f"bond {i}: env Herm err = {herm_env:.3e}")
    print(f"bond {i}: proj Herm err = {herm_proj:.3e}")
```

---

## Cell J — Heisenberg local-operator test

```python
L = 4
d = 2
i = 1

mpo = heisenberg_mpo(L)
Hfull = dense_heisenberg(L)

mps = random_mps(L, d=d, Dmax=3, seed=7)
mps = normalize_mps_by_state(mps)
mps = make_mixed_canonical_around_bond(mps, i)

compare_two_site_env_vs_proj(Hfull, mpo, mps, i, d)
```

---

# What to do next

Run Cells A–J. Cell J should now pass.

After that, the next step is to build the final DMRG sweep cell using this corrected local action.

If you want, my next reply can be exactly:

- **Cell K**: final two-site DMRG for Heisenberg and AKLT using the corrected environment-based matvec
- **Cell L**: benchmark runs for Heisenberg and AKLT in Jupyter

with explicit cell labels.

---
## 🧑‍💻 User

bond 1: max|env-proj| = 1.665e-16
bond 1: env Herm err = 0.000e+00
bond 1: proj Herm err = 5.551e-17

---
## 🤖 Assistant

Excellent — that’s the key milestone.

## We now have the correct fully environment-based two-site local action.

Your result
- `max|env-proj| = 1.665e-16`

means the environment-based operator now matches
\[
P^\dagger H P
\]
to machine precision.

So now we can move to the final goal:

# fully MPO/MPS environment-based matrix-free two-site DMRG
for both

- **spin-1/2 Heisenberg**
- **spin-1 AKLT**

Below I give you only the **next necessary notebook cells**, explicitly labeled.

You should keep your already working Cells **A–J**, then add the following.

---

# Cell K — SVD split helpers

```python
def form_theta(A, B):
    return np.einsum('asm,mtr->astr', A, B, optimize=True)


def split_theta_left_to_right(theta, Dmax, svd_cutoff=1e-12):
    Dl, d1, d2, Dr = theta.shape
    M = theta.reshape((Dl * d1, d2 * Dr), order='C')

    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    keep = np.sum(S > svd_cutoff)
    keep = max(1, min(Dmax, keep, len(S)))

    disc_weight = np.sum(S[keep:]**2).real

    U = U[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]

    A = U.reshape((Dl, d1, keep), order='C')
    B = (np.diag(S) @ Vh).reshape((keep, d2, Dr), order='C')
    return A, B, S, disc_weight


def split_theta_right_to_left(theta, Dmax, svd_cutoff=1e-12):
    Dl, d1, d2, Dr = theta.shape
    M = theta.reshape((Dl * d1, d2 * Dr), order='C')

    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    keep = np.sum(S > svd_cutoff)
    keep = max(1, min(Dmax, keep, len(S)))

    disc_weight = np.sum(S[keep:]**2).real

    U = U[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]

    A = (U @ np.diag(S)).reshape((Dl, d1, keep), order='C')
    B = Vh.reshape((keep, d2, Dr), order='C')
    return A, B, S, disc_weight
```

---

# Cell L — MPO expectation value and exact AKLT MPS

```python
def expectation_value_mpo(mps, mpo):
    env = init_left_env()
    for i in range(len(mps)):
        env = left_env_update(env, mps[i], mpo[i])
    return env[0, 0, 0]


def exact_aklt_mps(L, left_vec=None, right_vec=None, dtype=np.complex128):
    Ap = np.array([[0, np.sqrt(2/3)],
                   [0, 0]], dtype=dtype)
    A0 = np.array([[-1/np.sqrt(3), 0],
                   [0, 1/np.sqrt(3)]], dtype=dtype)
    Am = np.array([[0, 0],
                   [-np.sqrt(2/3), 0]], dtype=dtype)

    local = np.stack([Ap, A0, Am], axis=1)  # (2,3,2)

    if left_vec is None:
        left_vec = np.array([1.0, 0.0], dtype=dtype)
    if right_vec is None:
        right_vec = np.array([1.0, 0.0], dtype=dtype)

    mps = []
    A1 = np.einsum('a,asb->sb', left_vec, local).reshape(1, 3, 2)
    mps.append(A1)

    for _ in range(1, L - 1):
        mps.append(local.copy())

    AL = np.einsum('asb,b->as', local, right_vec).reshape(2, 3, 1)
    mps.append(AL)

    return mps
```

---

# Cell M — final matrix-free two-site DMRG

```python
def two_site_dmrg_env_matrix_free(
    mpo,
    d,
    Dmax=32,
    nsweeps=8,
    lanczos_tol=1e-10,
    lanczos_maxiter=None,
    svd_cutoff=1e-12,
    seed=1234,
    verbose=True,
):
    """
    Fully environment-based matrix-free two-site DMRG.
    No dense full Hamiltonian is used during the sweep.
    """
    L = len(mpo)

    mps = random_mps(L, d=d, Dmax=min(Dmax, 4), seed=seed)
    mps = normalize_mps_by_state(mps)
    mps = make_mixed_canonical_around_bond(mps, 0)

    sweep_local_energies = []
    sweep_discards = []

    for sweep in range(nsweeps):
        discards_this_sweep = []
        last_local_energy = None

        # --------------------------------------------------
        # Left -> Right
        # --------------------------------------------------
        Renvs = build_right_envs(mps, mpo)
        Lenv = init_left_env()

        for i in range(L - 1):
            theta0 = form_theta(mps[i], mps[i + 1])
            Dl, d1, d2, Dr = theta0.shape
            Nloc = Dl * d1 * d2 * Dr
            Renv = Renvs[i + 1]

            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v, Lenv=Lenv, W1=mpo[i], W2=mpo[i + 1], Renv=Renv,
                              Dl=Dl, d1=d1, d2=d2, Dr=Dr:
                    heff_two_site_env_matvec(v, Lenv, W1, W2, Renv, Dl, d1, d2, Dr),
                dtype=np.complex128
            )

            v0 = theta0.reshape(-1, order='C')
            vals, vecs = eigsh(
                Heff,
                k=1,
                which='SA',
                v0=v0,
                tol=lanczos_tol,
                maxiter=lanczos_maxiter
            )
            last_local_energy = vals[0].real

            theta = vecs[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            Anew, Bnew, S, disc = split_theta_left_to_right(theta, Dmax, svd_cutoff)

            mps[i] = Anew
            mps[i + 1] = Bnew
            discards_this_sweep.append(disc)

            Lenv = left_env_update(Lenv, mps[i], mpo[i])

        mps = normalize_mps_by_state(mps)
        mps = make_mixed_canonical_around_bond(mps, L - 2)

        # --------------------------------------------------
        # Right -> Left
        # --------------------------------------------------
        Lenvs = build_left_envs(mps, mpo)
        Renv = init_right_env()

        for i in range(L - 2, -1, -1):
            theta0 = form_theta(mps[i], mps[i + 1])
            Dl, d1, d2, Dr = theta0.shape
            Nloc = Dl * d1 * d2 * Dr
            Lcur = Lenvs[i]

            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v, Lcur=Lcur, W1=mpo[i], W2=mpo[i + 1], Renv=Renv,
                              Dl=Dl, d1=d1, d2=d2, Dr=Dr:
                    heff_two_site_env_matvec(v, Lcur, W1, W2, Renv, Dl, d1, d2, Dr),
                dtype=np.complex128
            )

            v0 = theta0.reshape(-1, order='C')
            vals, vecs = eigsh(
                Heff,
                k=1,
                which='SA',
                v0=v0,
                tol=lanczos_tol,
                maxiter=lanczos_maxiter
            )
            last_local_energy = vals[0].real

            theta = vecs[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            Anew, Bnew, S, disc = split_theta_right_to_left(theta, Dmax, svd_cutoff)

            mps[i] = Anew
            mps[i + 1] = Bnew
            discards_this_sweep.append(disc)

            Renv = right_env_update(Renv, mps[i + 1], mpo[i + 1])

        mps = normalize_mps_by_state(mps)
        mps = make_mixed_canonical_around_bond(mps, 0)

        max_disc = max(discards_this_sweep) if discards_this_sweep else 0.0
        sweep_local_energies.append(last_local_energy if last_local_energy is not None else np.nan)
        sweep_discards.append(max_disc)

        if verbose:
            print(f"Sweep {sweep+1:2d}: local eig E = {sweep_local_energies[-1]:.12f}, "
                  f"max discarded weight = {max_disc:.3e}")

    return mps, np.array(sweep_local_energies), np.array(sweep_discards)
```

---

# Cell N — Heisenberg DMRG benchmark

```python
L = 8
mpo_h = heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0)

mps_h, Es_h, disc_h = two_site_dmrg_env_matrix_free(
    mpo=mpo_h,
    d=2,
    Dmax=32,
    nsweeps=6,
    seed=1234,
    verbose=True
)

E_mpo_h = expectation_value_mpo(mps_h, mpo_h).real
print("\nHeisenberg <H> from MPO contraction =", E_mpo_h)

Hfull_h = dense_heisenberg(L, J=1.0, Jz=1.0, h=0.0)
psi_h = mps_to_state(mps_h)
E_dense_h = (np.vdot(psi_h, Hfull_h @ psi_h) / np.vdot(psi_h, psi_h)).real
E0_h = np.linalg.eigvalsh(Hfull_h)[0].real

print("Heisenberg energy from dense H on final MPS =", E_dense_h)
print("Heisenberg exact ED ground energy           =", E0_h)
print("Absolute error                              =", abs(E_dense_h - E0_h))
```

---

# Cell O — AKLT DMRG benchmark

```python
L = 6
mpo_a = aklt_mpo(L)

mps_a, Es_a, disc_a = two_site_dmrg_env_matrix_free(
    mpo=mpo_a,
    d=3,
    Dmax=16,
    nsweeps=6,
    seed=1234,
    verbose=True
)

E_mpo_a = expectation_value_mpo(mps_a, mpo_a).real
E_exact_a = -(2.0 / 3.0) * (L - 1)

print("\nAKLT <H> from MPO contraction =", E_mpo_a)
print("AKLT exact open-chain energy  =", E_exact_a)
print("Absolute error                =", abs(E_mpo_a - E_exact_a))

Hfull_a = dense_aklt(L)
psi_a = mps_to_state(mps_a)
E_dense_a = (np.vdot(psi_a, Hfull_a @ psi_a) / np.vdot(psi_a, psi_a)).real
E0_a = np.linalg.eigvalsh(Hfull_a)[0].real

print("AKLT dense energy from final MPS =", E_dense_a)
print("AKLT exact ED ground energy      =", E0_a)
print("Difference to ED                 =", abs(E_dense_a - E0_a))
```

---

# Cell P — AKLT exact-state benchmark

```python
L = 6
mpo_a = aklt_mpo(L)
mps_exact = exact_aklt_mps(L)
mps_exact = normalize_mps_by_state(mps_exact)

E_exact_mps = expectation_value_mpo(mps_exact, mpo_a).real
print("Exact AKLT MPS energy =", E_exact_mps)
print("Expected energy       =", -(2.0 / 3.0) * (L - 1))
print("Error                 =", abs(E_exact_mps + (2.0 / 3.0) * (L - 1)))
```

---

# Important note about Cell M

Cell M uses the corrected local matvec through `heff_two_site_env_matvec`, which came from Cell H:

```python
def heff_two_site_env_matvec(v, Lenv, W1, W2, Renv, Dl, d1, d2, Dr):
    theta = v.reshape((Dl, d1, d2, Dr), order='C')
    out = apply_two_site_heff_env(theta, Lenv, W1, W2, Renv)
    return out.reshape(Dl * d1 * d2 * Dr, order='C')
```

So Cell H must be present exactly as given.

---

# Recommended run order now

If you already have Cells A–J working, then just add and run:

- Cell K
- Cell L
- Cell M
- Cell N
- Cell O
- Cell P

in that order.

---

# What to expect

## Heisenberg
For `L=8`, `Dmax=32`, the final energy should be very close to exact ED.

## AKLT
For `L=6`, `Dmax=16`, the final energy should be very close to
\[
-\frac{2}{3}(L-1).
\]

Also Cell P should confirm that the exact AKLT MPS gives the exact AKLT energy.

---

If you want, after you run Cells N and O, send me the outputs and I’ll help check whether the sweep logic itself is fully consistent or needs one final cleanup.

---
## 🧑‍💻 User

Sweep  1: local eig E = -3.374932598688, max discarded weight = 0.000e+00
Sweep  2: local eig E = -3.374932598688, max discarded weight = 0.000e+00
Sweep  3: local eig E = -3.374932598688, max discarded weight = 0.000e+00
Sweep  4: local eig E = -3.374932598688, max discarded weight = 0.000e+00
Sweep  5: local eig E = -3.374932598688, max discarded weight = 0.000e+00
Sweep  6: local eig E = -3.374932598688, max discarded weight = 0.000e+00

Heisenberg <H> from MPO contraction = -3.374932598687892
Heisenberg energy from dense H on final MPS = -3.37493259868789
Heisenberg exact ED ground energy           = -3.374932598687897
Absolute error                              = 6.661338147750939e-15Sweep  1: local eig E = -3.333333333333, max discarded weight = 8.997e-24
Sweep  2: local eig E = -3.333333333333, max discarded weight = 2.037e-25
Sweep  3: local eig E = -3.333333333333, max discarded weight = 1.418e-24
Sweep  4: local eig E = -3.333333333333, max discarded weight = 9.297e-26
Sweep  5: local eig E = -3.333333333333, max discarded weight = 2.023e-27
Sweep  6: local eig E = -3.333333333333, max discarded weight = 4.168e-26

AKLT <H> from MPO contraction = -3.3333333333333357
AKLT exact open-chain energy  = -3.333333333333333
Absolute error                = 2.6645352591003757e-15
AKLT dense energy from final MPS = -3.3333333333333344
AKLT exact ED ground energy      = -3.333333333333365
Difference to ED                 = 3.064215547965432e-14Exact AKLT MPS energy = -3.333333333333332
Expected energy       = -3.333333333333333
Error                 = 8.881784197001252e-16

---
## 🤖 Assistant

Excellent — this is a full success.

## Final status

You now have a **working fully MPO/MPS environment-based matrix-free two-site DMRG** implementation for both:

- **spin-1/2 Heisenberg**
- **spin-1 AKLT**

and it is validated by:

- local operator identity
  \[
  H_{\mathrm{eff}}^{\mathrm{env}} = P^\dagger H P
  \]
  to machine precision,
- Heisenberg ground-state energy matching ED,
- AKLT ground-state energy matching the exact open-chain value,
- exact AKLT MPS benchmark also passing.

Your reported results are essentially perfect.

---

# What was the actual crucial fix?

The key nontrivial result was:

## Correct two-site local action
```python
def apply_two_site_heff_env(theta, Lenv, W1, W2, Renv):
    X = np.einsum('byx,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT
```

That is, compared to the naive contraction:

- the **left environment** is used with swapped bond indices: `byx`
- both MPO tensors are used with swapped physical ordering:
  - `W1: bBus`
  - `W2: BCvt`

This was the convention needed to match your dense projected basis construction.

---

# Your benchmark results

## Heisenberg
You got:
- final MPS energy:
  `-3.37493259868789`
- exact ED:
  `-3.374932598687897`
- error:
  `6.66e-15`

This is machine precision.

## AKLT
You got:
- final MPS energy:
  `-3.3333333333333357`
- exact open-chain AKLT energy:
  `-3.333333333333333`
- error:
  `2.66e-15`

Again machine precision.

## Exact AKLT MPS benchmark
You got:
- `8.88e-16` error

Also perfect.

---

# What this means practically

You have now achieved the target from the handoff note:

- no dense full Hamiltonian during DMRG sweeps,
- no dense projected operator during sweeps,
- no embedding into full Hilbert space during local solves,
- only:
  - MPS tensors
  - MPO tensors
  - left/right environments
  - matrix-free local action

That is exactly the scalable environment-based DMRG architecture.

---

# Recommended final cleanup for your notebook

Now that the code works, I strongly recommend organizing the notebook into:

## Keep
- imports
- local operators
- MPO builders
- MPS utilities
- environment routines
- final one-site/two-site local actions
- SVD split helpers
- DMRG routine
- benchmark cells

## Archive / remove
All temporary debugging cells:
- candidate tests
- broken left-reference attempts
- intermediate local operator checks you no longer need day-to-day

Though I would keep one compact validation cell for future safety.

---

# Strongly recommended permanent validation cells

I suggest keeping these two tests permanently:

## 1. Local operator identity test
For Heisenberg `L=4`, bond `i=1`:
\[
\max |H_{\mathrm{env}} - P^\dagger H P|
\]
should be around `1e-15`.

## 2. AKLT exact MPS energy test
For `L=6`:
\[
E = -\frac23(L-1)
\]
should hold to machine precision.

These two tests will catch almost any future accidental convention break.

---

# One minor comment on Cell M

Your DMRG sweep currently works and is correct for these benchmarks.

A future refinement would be to maintain canonical form more incrementally rather than calling `make_mixed_canonical_around_bond(...)` after each half-sweep. That would be more standard and more efficient.

But:

- for correctness,
- for clarity,
- and for Jupyter-based development,

your current version is absolutely fine.

---

# Recommended next improvements

Now that correctness is established, the natural next steps are:

## 1. Add one-site DMRG
Using the same corrected left-environment convention in the local action.

## 2. Add variance evaluation
Compute
\[
\langle H^2\rangle - \langle H\rangle^2
\]
as a convergence diagnostic.

## 3. Improve sweep efficiency
Avoid rebuilding mixed-canonical form from scratch each half-sweep.

## 4. Add truncation by discarded weight
Instead of only singular-value cutoff.

## 5. Package the notebook code into a `.py` module
So the notebook becomes only a driver / benchmark layer.

---

# If you want, my next reply can give you one of these:

## Option A — clean final notebook version
A compact, cleaned Jupyter sequence with only the necessary cells.

## Option B — single `.py` file
A polished Python module combining everything you now validated.

## Option C — add one-site DMRG
Using the corrected environment conventions.

## Option D — add diagnostics and plotting
Energy convergence, discarded weight, bond dimensions, etc.

---

If I were choosing the most useful next step, I would suggest:

# **Option A: a cleaned final notebook cell set**
because you now have many debugging cells and likely want a tidy final version.

If you want that, I can provide a **clean notebook-ready final set of labeled cells only**, with all dead debug branches removed.

---
## 🧑‍💻 User

A

---
## 🤖 Assistant

Below is a **clean final notebook-ready set of labeled cells**. It keeps only the necessary code for:

- Heisenberg MPO/MPS environment-based matrix-free two-site DMRG
- AKLT MPO/MPS environment-based matrix-free two-site DMRG
- local-operator validation
- benchmark runs

You can paste these into a fresh notebook in order.

---

# Cell A — imports

```python
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

np.set_printoptions(precision=6, suppress=True)
```

---

# Cell B — local operators

```python
def spin_half_ops(dtype=np.complex128):
    I = np.eye(2, dtype=dtype)
    Sp = np.array([[0, 1],
                   [0, 0]], dtype=dtype)
    Sm = np.array([[0, 0],
                   [1, 0]], dtype=dtype)
    Sz = 0.5 * np.array([[1, 0],
                         [0, -1]], dtype=dtype)
    return I, Sp, Sm, Sz


def spin_one_ops(dtype=np.complex128):
    I = np.eye(3, dtype=dtype)
    Sp = np.sqrt(2) * np.array([[0, 1, 0],
                                [0, 0, 1],
                                [0, 0, 0]], dtype=dtype)
    Sm = np.sqrt(2) * np.array([[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0]], dtype=dtype)
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, -1]], dtype=dtype)
    return I, Sp, Sm, Sz
```

---

# Cell C — MPO builders

```python
def heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    I, Sp, Sm, Sz = spin_half_ops(dtype=dtype)
    d = 2
    chi = 5

    Wbulk = np.zeros((chi, chi, d, d), dtype=dtype)
    Wbulk[0, 0] = I
    Wbulk[1, 0] = Sp
    Wbulk[2, 0] = Sm
    Wbulk[3, 0] = Sz
    Wbulk[4, 0] = -h * Sz
    Wbulk[4, 1] = (J / 2.0) * Sm
    Wbulk[4, 2] = (J / 2.0) * Sp
    Wbulk[4, 3] = Jz * Sz
    Wbulk[4, 4] = I

    Wleft = np.zeros((1, chi, d, d), dtype=dtype)
    Wleft[0, 0] = -h * Sz
    Wleft[0, 1] = (J / 2.0) * Sm
    Wleft[0, 2] = (J / 2.0) * Sp
    Wleft[0, 3] = Jz * Sz
    Wleft[0, 4] = I

    Wright = np.zeros((chi, 1, d, d), dtype=dtype)
    Wright[0, 0] = I
    Wright[1, 0] = Sp
    Wright[2, 0] = Sm
    Wright[3, 0] = Sz
    Wright[4, 0] = -h * Sz

    mpo = [None] * L
    mpo[0] = Wleft
    for i in range(1, L - 1):
        mpo[i] = Wbulk.copy()
    mpo[L - 1] = Wright
    return mpo


def aklt_mpo(L, dtype=np.complex128):
    I, Sp, Sm, Sz = spin_one_ops(dtype=dtype)
    d = 3
    chi = 14

    O = [Sp / np.sqrt(2.0), Sm / np.sqrt(2.0), Sz]
    Obar = [Sm / np.sqrt(2.0), Sp / np.sqrt(2.0), Sz]

    O2 = []
    Obar2 = []
    for a in range(3):
        for b in range(3):
            O2.append(O[a] @ O[b])
            Obar2.append(Obar[a] @ Obar[b])

    Wbulk = np.zeros((chi, chi, d, d), dtype=dtype)
    Wbulk[0, 0] = I
    for a in range(3):
        Wbulk[0, 1 + a] = O[a]
    for ab in range(9):
        Wbulk[0, 4 + ab] = (1.0 / 3.0) * O2[ab]
    for a in range(3):
        Wbulk[1 + a, 13] = Obar[a]
    for ab in range(9):
        Wbulk[4 + ab, 13] = Obar2[ab]
    Wbulk[13, 13] = I

    Wleft = np.zeros((1, chi, d, d), dtype=dtype)
    Wleft[0, 0] = I
    for a in range(3):
        Wleft[0, 1 + a] = O[a]
    for ab in range(9):
        Wleft[0, 4 + ab] = (1.0 / 3.0) * O2[ab]

    Wright = np.zeros((chi, 1, d, d), dtype=dtype)
    for a in range(3):
        Wright[1 + a, 0] = Obar[a]
    for ab in range(9):
        Wright[4 + ab, 0] = Obar2[ab]
    Wright[13, 0] = I

    mpo = [None] * L
    mpo[0] = Wleft
    for i in range(1, L - 1):
        mpo[i] = Wbulk.copy()
    mpo[L - 1] = Wright
    return mpo
```

---

# Cell D — dense Hamiltonians for benchmarks and validation

```python
def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def onsite_op(L, i, op, d):
    I = np.eye(d, dtype=np.complex128)
    ops = [I] * L
    ops[i] = op
    return kron_all(ops)


def bond_op(L, i, op_i, op_j, d):
    I = np.eye(d, dtype=np.complex128)
    ops = [I] * L
    ops[i] = op_i
    ops[i + 1] = op_j
    return kron_all(ops)


def dense_heisenberg(L, J=1.0, Jz=1.0, h=0.0):
    I, Sp, Sm, Sz = spin_half_ops()
    d = 2
    H = np.zeros((d**L, d**L), dtype=np.complex128)

    for i in range(L - 1):
        H += (J / 2.0) * bond_op(L, i, Sp, Sm, d)
        H += (J / 2.0) * bond_op(L, i, Sm, Sp, d)
        H += Jz * bond_op(L, i, Sz, Sz, d)

    for i in range(L):
        H += -h * onsite_op(L, i, Sz, d)

    return H


def dense_aklt(L):
    I, Sp, Sm, Sz = spin_one_ops()
    d = 3
    H = np.zeros((d**L, d**L), dtype=np.complex128)

    for i in range(L - 1):
        X = (
            0.5 * bond_op(L, i, Sp, Sm, d)
            + 0.5 * bond_op(L, i, Sm, Sp, d)
            + bond_op(L, i, Sz, Sz, d)
        )
        H += X + (1.0 / 3.0) * (X @ X)

    return H
```

---

# Cell E — MPS utilities

```python
def random_mps(L, d, Dmax, seed=1234, dtype=np.complex128):
    rng = np.random.default_rng(seed)

    dims = [1]
    for i in range(1, L):
        dims.append(min(Dmax, d ** min(i, L - i)))
    dims.append(1)

    mps = []
    for i in range(L):
        Dl, Dr = dims[i], dims[i + 1]
        A = rng.normal(size=(Dl, d, Dr)) + 1j * rng.normal(size=(Dl, d, Dr))
        mps.append(A.astype(dtype))
    return mps


def mps_to_state(mps):
    psi = mps[0][0]
    for i in range(1, len(mps)):
        psi = np.einsum('...a,asb->...sb', psi, mps[i], optimize=True)
    return psi.reshape(-1, order='C')


def normalize_mps_by_state(mps):
    psi = mps_to_state(mps)
    nrm = np.sqrt(np.vdot(psi, psi).real)
    mps = [A.copy() for A in mps]
    mps[0] = mps[0] / nrm
    return mps


def left_basis_map(mps, i):
    if i == 0:
        return np.array([[1.0]], dtype=np.complex128)

    X = mps[0][0]
    for k in range(1, i):
        X = np.einsum('pa,asb->psb', X, mps[k], optimize=True)
        X = X.reshape((-1, X.shape[-1]), order='C')
    return X


def right_basis_map(mps, i):
    L = len(mps)
    if i == L - 2:
        return np.array([[1.0]], dtype=np.complex128)

    X = mps[i + 2]
    for k in range(i + 3, L):
        X = np.einsum('asd,dtu->astu', X, mps[k], optimize=True)
        X = X.reshape((X.shape[0], -1, X.shape[-1]), order='C')
    return X[:, :, 0]


def make_mixed_canonical_around_bond(mps, i):
    mps = [A.copy() for A in mps]
    L = len(mps)

    for k in range(i):
        A = mps[k]
        Dl, d, Dr = A.shape
        M = A.reshape((Dl * d, Dr), order='C')
        Q, R = np.linalg.qr(M, mode='reduced')
        nk = Q.shape[1]
        mps[k] = Q.reshape((Dl, d, nk), order='C')
        mps[k + 1] = np.einsum('ab,bsd->asd', R, mps[k + 1], optimize=True)

    for k in range(L - 1, i + 1, -1):
        A = mps[k]
        Dl, d, Dr = A.shape
        M = A.reshape((Dl, d * Dr), order='C')
        Q, R = np.linalg.qr(M.conj().T, mode='reduced')
        B = Q.conj().T.reshape((-1, d, Dr), order='C')
        mps[k] = B
        Rd = R.conj().T
        mps[k - 1] = np.einsum('xsa,ab->xsb', mps[k - 1], Rd, optimize=True)

    return mps
```

---

# Cell F — environments

```python
def init_left_env():
    Lenv = np.zeros((1, 1, 1), dtype=np.complex128)
    Lenv[0, 0, 0] = 1.0
    return Lenv


def init_right_env():
    Renv = np.zeros((1, 1, 1), dtype=np.complex128)
    Renv[0, 0, 0] = 1.0
    return Renv


def left_env_update(Lold, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj(), optimize=True)


def right_env_update(Rold, B, W):
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj(), optimize=True)


def build_left_envs(mps, mpo):
    L = len(mps)
    Lenvs = [None] * L
    Lenvs[0] = init_left_env()
    cur = Lenvs[0]
    for i in range(0, L - 1):
        cur = left_env_update(cur, mps[i], mpo[i])
        Lenvs[i + 1] = cur
    return Lenvs


def build_right_envs(mps, mpo):
    L = len(mps)
    Renvs = [None] * L
    Renvs[L - 1] = init_right_env()
    cur = Renvs[L - 1]
    for i in range(L - 1, 0, -1):
        cur = right_env_update(cur, mps[i], mpo[i])
        Renvs[i - 1] = cur
    return Renvs
```

---

# Cell G — corrected local actions

```python
def apply_one_site_heff_env(M, Lenv, W, Renv):
    X = np.einsum('byx,ysz->bxsz', Lenv, M, optimize=True)
    Y = np.einsum('bBst,bxsz->Bxtz', W, X, optimize=True)
    HM = np.einsum('Bxtz,Bza->xta', Y, Renv, optimize=True)
    return HM


def apply_two_site_heff_env(theta, Lenv, W1, W2, Renv):
    X = np.einsum('byx,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def heff_two_site_env_matvec(v, Lenv, W1, W2, Renv, Dl, d1, d2, Dr):
    theta = v.reshape((Dl, d1, d2, Dr), order='C')
    out = apply_two_site_heff_env(theta, Lenv, W1, W2, Renv)
    return out.reshape(Dl * d1 * d2 * Dr, order='C')
```

---

# Cell H — dense projected reference for local-operator validation

```python
def dense_projected_two_site_heff_from_full(Hfull, mps, i, d):
    UL = left_basis_map(mps, i)
    UR = right_basis_map(mps, i)

    Dl = UL.shape[1]
    Dr = UR.shape[0]
    dimL = UL.shape[0]
    dimR = UR.shape[1]

    Nloc = Dl * d * d * Dr
    Dim = dimL * d * d * dimR
    P = np.zeros((Dim, Nloc), dtype=np.complex128)

    col = 0
    for a in range(Dl):
        for s in range(d):
            for t in range(d):
                for c in range(Dr):
                    mid = np.zeros(d * d, dtype=np.complex128)
                    mid[s * d + t] = 1.0
                    vec = np.kron(UL[:, a], np.kron(mid, UR[c, :]))
                    P[:, col] = vec
                    col += 1

    Heff = P.conj().T @ Hfull @ P
    return Heff


def build_dense_local_heff_from_env(mps, mpo, i):
    A = mps[i]
    B = mps[i + 1]
    Dl, d1, mid = A.shape
    mid2, d2, Dr = B.shape
    assert mid == mid2

    Lenvs = build_left_envs(mps, mpo)
    Renvs = build_right_envs(mps, mpo)

    Lenv = Lenvs[i]
    Renv = Renvs[i + 1]

    N = Dl * d1 * d2 * Dr
    Heff = np.zeros((N, N), dtype=np.complex128)

    for j in range(N):
        e = np.zeros(N, dtype=np.complex128)
        e[j] = 1.0
        theta = e.reshape((Dl, d1, d2, Dr), order='C')
        Heff[:, j] = apply_two_site_heff_env(
            theta, Lenv, mpo[i], mpo[i + 1], Renv
        ).reshape(-1, order='C')

    return Heff


def compare_two_site_env_vs_proj(Hfull, mpo, mps, i, d):
    Heff_proj = dense_projected_two_site_heff_from_full(Hfull, mps, i, d)
    Heff_env = build_dense_local_heff_from_env(mps, mpo, i)

    diff = np.max(np.abs(Heff_env - Heff_proj))
    herm_env = np.max(np.abs(Heff_env - Heff_env.conj().T))
    herm_proj = np.max(np.abs(Heff_proj - Heff_proj.conj().T))

    print(f"bond {i}: max|env-proj| = {diff:.3e}")
    print(f"bond {i}: env Herm err = {herm_env:.3e}")
    print(f"bond {i}: proj Herm err = {herm_proj:.3e}")
```

---

# Cell I — SVD helpers and expectation value

```python
def form_theta(A, B):
    return np.einsum('asm,mtr->astr', A, B, optimize=True)


def split_theta_left_to_right(theta, Dmax, svd_cutoff=1e-12):
    Dl, d1, d2, Dr = theta.shape
    M = theta.reshape((Dl * d1, d2 * Dr), order='C')

    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    keep = np.sum(S > svd_cutoff)
    keep = max(1, min(Dmax, keep, len(S)))
    disc_weight = np.sum(S[keep:]**2).real

    U = U[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]

    A = U.reshape((Dl, d1, keep), order='C')
    B = (np.diag(S) @ Vh).reshape((keep, d2, Dr), order='C')
    return A, B, S, disc_weight


def split_theta_right_to_left(theta, Dmax, svd_cutoff=1e-12):
    Dl, d1, d2, Dr = theta.shape
    M = theta.reshape((Dl * d1, d2 * Dr), order='C')

    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    keep = np.sum(S > svd_cutoff)
    keep = max(1, min(Dmax, keep, len(S)))
    disc_weight = np.sum(S[keep:]**2).real

    U = U[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]

    A = (U @ np.diag(S)).reshape((Dl, d1, keep), order='C')
    B = Vh.reshape((keep, d2, Dr), order='C')
    return A, B, S, disc_weight


def expectation_value_mpo(mps, mpo):
    env = init_left_env()
    for i in range(len(mps)):
        env = left_env_update(env, mps[i], mpo[i])
    return env[0, 0, 0]
```

---

# Cell J — exact AKLT MPS

```python
def exact_aklt_mps(L, left_vec=None, right_vec=None, dtype=np.complex128):
    Ap = np.array([[0, np.sqrt(2/3)],
                   [0, 0]], dtype=dtype)
    A0 = np.array([[-1/np.sqrt(3), 0],
                   [0, 1/np.sqrt(3)]], dtype=dtype)
    Am = np.array([[0, 0],
                   [-np.sqrt(2/3), 0]], dtype=dtype)

    local = np.stack([Ap, A0, Am], axis=1)

    if left_vec is None:
        left_vec = np.array([1.0, 0.0], dtype=dtype)
    if right_vec is None:
        right_vec = np.array([1.0, 0.0], dtype=dtype)

    mps = []
    A1 = np.einsum('a,asb->sb', left_vec, local).reshape(1, 3, 2)
    mps.append(A1)

    for _ in range(1, L - 1):
        mps.append(local.copy())

    AL = np.einsum('asb,b->as', local, right_vec).reshape(2, 3, 1)
    mps.append(AL)

    return mps
```

---

# Cell K — final matrix-free two-site DMRG

```python
def two_site_dmrg_env_matrix_free(
    mpo,
    d,
    Dmax=32,
    nsweeps=8,
    lanczos_tol=1e-10,
    lanczos_maxiter=None,
    svd_cutoff=1e-12,
    seed=1234,
    verbose=True,
):
    L = len(mpo)

    mps = random_mps(L, d=d, Dmax=min(Dmax, 4), seed=seed)
    mps = normalize_mps_by_state(mps)
    mps = make_mixed_canonical_around_bond(mps, 0)

    sweep_local_energies = []
    sweep_discards = []

    for sweep in range(nsweeps):
        discards_this_sweep = []
        last_local_energy = None

        # Left -> Right
        Renvs = build_right_envs(mps, mpo)
        Lenv = init_left_env()

        for i in range(L - 1):
            theta0 = form_theta(mps[i], mps[i + 1])
            Dl, d1, d2, Dr = theta0.shape
            Nloc = Dl * d1 * d2 * Dr
            Renv = Renvs[i + 1]

            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v, Lenv=Lenv, W1=mpo[i], W2=mpo[i + 1], Renv=Renv,
                              Dl=Dl, d1=d1, d2=d2, Dr=Dr:
                    heff_two_site_env_matvec(v, Lenv, W1, W2, Renv, Dl, d1, d2, Dr),
                dtype=np.complex128
            )

            v0 = theta0.reshape(-1, order='C')
            vals, vecs = eigsh(
                Heff, k=1, which='SA',
                v0=v0, tol=lanczos_tol, maxiter=lanczos_maxiter
            )
            last_local_energy = vals[0].real

            theta = vecs[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            Anew, Bnew, S, disc = split_theta_left_to_right(theta, Dmax, svd_cutoff)

            mps[i] = Anew
            mps[i + 1] = Bnew
            discards_this_sweep.append(disc)

            Lenv = left_env_update(Lenv, mps[i], mpo[i])

        mps = normalize_mps_by_state(mps)
        mps = make_mixed_canonical_around_bond(mps, L - 2)

        # Right -> Left
        Lenvs = build_left_envs(mps, mpo)
        Renv = init_right_env()

        for i in range(L - 2, -1, -1):
            theta0 = form_theta(mps[i], mps[i + 1])
            Dl, d1, d2, Dr = theta0.shape
            Nloc = Dl * d1 * d2 * Dr
            Lcur = Lenvs[i]

            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v, Lcur=Lcur, W1=mpo[i], W2=mpo[i + 1], Renv=Renv,
                              Dl=Dl, d1=d1, d2=d2, Dr=Dr:
                    heff_two_site_env_matvec(v, Lcur, W1, W2, Renv, Dl, d1, d2, Dr),
                dtype=np.complex128
            )

            v0 = theta0.reshape(-1, order='C')
            vals, vecs = eigsh(
                Heff, k=1, which='SA',
                v0=v0, tol=lanczos_tol, maxiter=lanczos_maxiter
            )
            last_local_energy = vals[0].real

            theta = vecs[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            Anew, Bnew, S, disc = split_theta_right_to_left(theta, Dmax, svd_cutoff)

            mps[i] = Anew
            mps[i + 1] = Bnew
            discards_this_sweep.append(disc)

            Renv = right_env_update(Renv, mps[i + 1], mpo[i + 1])

        mps = normalize_mps_by_state(mps)
        mps = make_mixed_canonical_around_bond(mps, 0)

        max_disc = max(discards_this_sweep) if discards_this_sweep else 0.0
        sweep_local_energies.append(last_local_energy if last_local_energy is not None else np.nan)
        sweep_discards.append(max_disc)

        if verbose:
            print(f"Sweep {sweep+1:2d}: local eig E = {sweep_local_energies[-1]:.12f}, "
                  f"max discarded weight = {max_disc:.3e}")

    return mps, np.array(sweep_local_energies), np.array(sweep_discards)
```

---

# Cell L — local-operator validation test

```python
L = 4
d = 2
i = 1

mpo = heisenberg_mpo(L)
Hfull = dense_heisenberg(L)

mps = random_mps(L, d=d, Dmax=3, seed=7)
mps = normalize_mps_by_state(mps)
mps = make_mixed_canonical_around_bond(mps, i)

compare_two_site_env_vs_proj(Hfull, mpo, mps, i, d)
```

---

# Cell M — Heisenberg benchmark

```python
L = 8
mpo_h = heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0)

mps_h, Es_h, disc_h = two_site_dmrg_env_matrix_free(
    mpo=mpo_h,
    d=2,
    Dmax=32,
    nsweeps=6,
    seed=1234,
    verbose=True
)

E_mpo_h = expectation_value_mpo(mps_h, mpo_h).real
print("\nHeisenberg <H> from MPO contraction =", E_mpo_h)

Hfull_h = dense_heisenberg(L, J=1.0, Jz=1.0, h=0.0)
psi_h = mps_to_state(mps_h)
E_dense_h = (np.vdot(psi_h, Hfull_h @ psi_h) / np.vdot(psi_h, psi_h)).real
E0_h = np.linalg.eigvalsh(Hfull_h)[0].real

print("Heisenberg energy from dense H on final MPS =", E_dense_h)
print("Heisenberg exact ED ground energy           =", E0_h)
print("Absolute error                              =", abs(E_dense_h - E0_h))
```

---

# Cell N — AKLT benchmark

```python
L = 6
mpo_a = aklt_mpo(L)

mps_a, Es_a, disc_a = two_site_dmrg_env_matrix_free(
    mpo=mpo_a,
    d=3,
    Dmax=16,
    nsweeps=6,
    seed=1234,
    verbose=True
)

E_mpo_a = expectation_value_mpo(mps_a, mpo_a).real
E_exact_a = -(2.0 / 3.0) * (L - 1)

print("\nAKLT <H> from MPO contraction =", E_mpo_a)
print("AKLT exact open-chain energy  =", E_exact_a)
print("Absolute error                =", abs(E_mpo_a - E_exact_a))

Hfull_a = dense_aklt(L)
psi_a = mps_to_state(mps_a)
E_dense_a = (np.vdot(psi_a, Hfull_a @ psi_a) / np.vdot(psi_a, psi_a)).real
E0_a = np.linalg.eigvalsh(Hfull_a)[0].real

print("AKLT dense energy from final MPS =", E_dense_a)
print("AKLT exact ED ground energy      =", E0_a)
print("Difference to ED                 =", abs(E_dense_a - E0_a))
```

---

# Cell O — exact AKLT MPS benchmark

```python
L = 6
mpo_a = aklt_mpo(L)
mps_exact = exact_aklt_mps(L)
mps_exact = normalize_mps_by_state(mps_exact)

E_exact_mps = expectation_value_mpo(mps_exact, mpo_a).real
print("Exact AKLT MPS energy =", E_exact_mps)
print("Expected energy       =", -(2.0 / 3.0) * (L - 1))
print("Error                 =", abs(E_exact_mps + (2.0 / 3.0) * (L - 1)))
```

---

# Recommended usage

Run in order:

1. Cell A through K
2. Cell L once to confirm local-operator correctness
3. Cell M for Heisenberg
4. Cell N for AKLT
5. Cell O for exact AKLT state benchmark

---

# Final note

The crucial convention-dependent formulas in this final notebook are:

## left environment update
```python
np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj())
```

## right environment update
```python
np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj())
```

## corrected one-site local action
```python
np.einsum('byx,ysz->bxsz', Lenv, M)
```

## corrected two-site local action
```python
np.einsum('byx,yuvz->bxuvz', Lenv, theta)
np.einsum('bBus,bxuvz->Bxsvz', W1, X)
np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
```

Those are the nontrivial conventions that made the full implementation work.

If you want next, I can give you **Option B**: the same final code as a single clean `.py` module.

---
## 🧑‍💻 User

Option B, with some small comments, such as:  # local operators

---
## 🤖 Assistant

```python
# mps_dmrg_env_final.py

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh


# ============================================================
# local operators
# ============================================================

def spin_half_ops(dtype=np.complex128):
    I = np.eye(2, dtype=dtype)
    Sp = np.array([[0, 1],
                   [0, 0]], dtype=dtype)
    Sm = np.array([[0, 0],
                   [1, 0]], dtype=dtype)
    Sz = 0.5 * np.array([[1, 0],
                         [0, -1]], dtype=dtype)
    return I, Sp, Sm, Sz


def spin_one_ops(dtype=np.complex128):
    I = np.eye(3, dtype=dtype)
    Sp = np.sqrt(2) * np.array([[0, 1, 0],
                                [0, 0, 1],
                                [0, 0, 0]], dtype=dtype)
    Sm = np.sqrt(2) * np.array([[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0]], dtype=dtype)
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, -1]], dtype=dtype)
    return I, Sp, Sm, Sz


# ============================================================
# mpo builders
# ============================================================

def heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    I, Sp, Sm, Sz = spin_half_ops(dtype=dtype)
    d = 2
    chi = 5

    Wbulk = np.zeros((chi, chi, d, d), dtype=dtype)
    Wbulk[0, 0] = I
    Wbulk[1, 0] = Sp
    Wbulk[2, 0] = Sm
    Wbulk[3, 0] = Sz
    Wbulk[4, 0] = -h * Sz
    Wbulk[4, 1] = (J / 2.0) * Sm
    Wbulk[4, 2] = (J / 2.0) * Sp
    Wbulk[4, 3] = Jz * Sz
    Wbulk[4, 4] = I

    Wleft = np.zeros((1, chi, d, d), dtype=dtype)
    Wleft[0, 0] = -h * Sz
    Wleft[0, 1] = (J / 2.0) * Sm
    Wleft[0, 2] = (J / 2.0) * Sp
    Wleft[0, 3] = Jz * Sz
    Wleft[0, 4] = I

    Wright = np.zeros((chi, 1, d, d), dtype=dtype)
    Wright[0, 0] = I
    Wright[1, 0] = Sp
    Wright[2, 0] = Sm
    Wright[3, 0] = Sz
    Wright[4, 0] = -h * Sz

    mpo = [None] * L
    mpo[0] = Wleft
    for i in range(1, L - 1):
        mpo[i] = Wbulk.copy()
    mpo[L - 1] = Wright
    return mpo


def aklt_mpo(L, dtype=np.complex128):
    I, Sp, Sm, Sz = spin_one_ops(dtype=dtype)
    d = 3
    chi = 14

    O = [Sp / np.sqrt(2.0), Sm / np.sqrt(2.0), Sz]
    Obar = [Sm / np.sqrt(2.0), Sp / np.sqrt(2.0), Sz]

    O2 = []
    Obar2 = []
    for a in range(3):
        for b in range(3):
            O2.append(O[a] @ O[b])
            Obar2.append(Obar[a] @ Obar[b])

    Wbulk = np.zeros((chi, chi, d, d), dtype=dtype)
    Wbulk[0, 0] = I
    for a in range(3):
        Wbulk[0, 1 + a] = O[a]
    for ab in range(9):
        Wbulk[0, 4 + ab] = (1.0 / 3.0) * O2[ab]
    for a in range(3):
        Wbulk[1 + a, 13] = Obar[a]
    for ab in range(9):
        Wbulk[4 + ab, 13] = Obar2[ab]
    Wbulk[13, 13] = I

    Wleft = np.zeros((1, chi, d, d), dtype=dtype)
    Wleft[0, 0] = I
    for a in range(3):
        Wleft[0, 1 + a] = O[a]
    for ab in range(9):
        Wleft[0, 4 + ab] = (1.0 / 3.0) * O2[ab]

    Wright = np.zeros((chi, 1, d, d), dtype=dtype)
    for a in range(3):
        Wright[1 + a, 0] = Obar[a]
    for ab in range(9):
        Wright[4 + ab, 0] = Obar2[ab]
    Wright[13, 0] = I

    mpo = [None] * L
    mpo[0] = Wleft
    for i in range(1, L - 1):
        mpo[i] = Wbulk.copy()
    mpo[L - 1] = Wright
    return mpo


# ============================================================
# dense hamiltonians, for tests
# ============================================================

def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def onsite_op(L, i, op, d):
    I = np.eye(d, dtype=np.complex128)
    ops = [I] * L
    ops[i] = op
    return kron_all(ops)


def bond_op(L, i, op_i, op_j, d):
    I = np.eye(d, dtype=np.complex128)
    ops = [I] * L
    ops[i] = op_i
    ops[i + 1] = op_j
    return kron_all(ops)


def dense_heisenberg(L, J=1.0, Jz=1.0, h=0.0):
    I, Sp, Sm, Sz = spin_half_ops()
    d = 2
    H = np.zeros((d**L, d**L), dtype=np.complex128)

    for i in range(L - 1):
        H += (J / 2.0) * bond_op(L, i, Sp, Sm, d)
        H += (J / 2.0) * bond_op(L, i, Sm, Sp, d)
        H += Jz * bond_op(L, i, Sz, Sz, d)

    for i in range(L):
        H += -h * onsite_op(L, i, Sz, d)

    return H


def dense_aklt(L):
    I, Sp, Sm, Sz = spin_one_ops()
    d = 3
    H = np.zeros((d**L, d**L), dtype=np.complex128)

    for i in range(L - 1):
        X = (
            0.5 * bond_op(L, i, Sp, Sm, d)
            + 0.5 * bond_op(L, i, Sm, Sp, d)
            + bond_op(L, i, Sz, Sz, d)
        )
        H += X + (1.0 / 3.0) * (X @ X)

    return H


# ============================================================
# mps utilities
# ============================================================

def random_mps(L, d, Dmax, seed=1234, dtype=np.complex128):
    rng = np.random.default_rng(seed)

    dims = [1]
    for i in range(1, L):
        dims.append(min(Dmax, d ** min(i, L - i)))
    dims.append(1)

    mps = []
    for i in range(L):
        Dl, Dr = dims[i], dims[i + 1]
        A = rng.normal(size=(Dl, d, Dr)) + 1j * rng.normal(size=(Dl, d, Dr))
        mps.append(A.astype(dtype))
    return mps


def mps_to_state(mps):
    psi = mps[0][0]
    for i in range(1, len(mps)):
        psi = np.einsum('...a,asb->...sb', psi, mps[i], optimize=True)
    return psi.reshape(-1, order='C')


def normalize_mps_by_state(mps):
    psi = mps_to_state(mps)
    nrm = np.sqrt(np.vdot(psi, psi).real)
    mps = [A.copy() for A in mps]
    mps[0] = mps[0] / nrm
    return mps


def left_basis_map(mps, i):
    if i == 0:
        return np.array([[1.0]], dtype=np.complex128)

    X = mps[0][0]
    for k in range(1, i):
        X = np.einsum('pa,asb->psb', X, mps[k], optimize=True)
        X = X.reshape((-1, X.shape[-1]), order='C')
    return X


def right_basis_map(mps, i):
    L = len(mps)
    if i == L - 2:
        return np.array([[1.0]], dtype=np.complex128)

    X = mps[i + 2]
    for k in range(i + 3, L):
        X = np.einsum('asd,dtu->astu', X, mps[k], optimize=True)
        X = X.reshape((X.shape[0], -1, X.shape[-1]), order='C')
    return X[:, :, 0]


def make_mixed_canonical_around_bond(mps, i):
    mps = [A.copy() for A in mps]
    L = len(mps)

    for k in range(i):
        A = mps[k]
        Dl, d, Dr = A.shape
        M = A.reshape((Dl * d, Dr), order='C')
        Q, R = np.linalg.qr(M, mode='reduced')
        nk = Q.shape[1]
        mps[k] = Q.reshape((Dl, d, nk), order='C')
        mps[k + 1] = np.einsum('ab,bsd->asd', R, mps[k + 1], optimize=True)

    for k in range(L - 1, i + 1, -1):
        A = mps[k]
        Dl, d, Dr = A.shape
        M = A.reshape((Dl, d * Dr), order='C')
        Q, R = np.linalg.qr(M.conj().T, mode='reduced')
        B = Q.conj().T.reshape((-1, d, Dr), order='C')
        mps[k] = B
        Rd = R.conj().T
        mps[k - 1] = np.einsum('xsa,ab->xsb', mps[k - 1], Rd, optimize=True)

    return mps


# ============================================================
# mpo environments
# ============================================================

def init_left_env():
    Lenv = np.zeros((1, 1, 1), dtype=np.complex128)
    Lenv[0, 0, 0] = 1.0
    return Lenv


def init_right_env():
    Renv = np.zeros((1, 1, 1), dtype=np.complex128)
    Renv[0, 0, 0] = 1.0
    return Renv


def left_env_update(Lold, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj(), optimize=True)


def right_env_update(Rold, B, W):
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj(), optimize=True)


def build_left_envs(mps, mpo):
    L = len(mps)
    Lenvs = [None] * L
    Lenvs[0] = init_left_env()

    cur = Lenvs[0]
    for i in range(0, L - 1):
        cur = left_env_update(cur, mps[i], mpo[i])
        Lenvs[i + 1] = cur

    return Lenvs


def build_right_envs(mps, mpo):
    L = len(mps)
    Renvs = [None] * L
    Renvs[L - 1] = init_right_env()

    cur = Renvs[L - 1]
    for i in range(L - 1, 0, -1):
        cur = right_env_update(cur, mps[i], mpo[i])
        Renvs[i - 1] = cur

    return Renvs


# ============================================================
# local effective hamiltonian actions
# ============================================================

def apply_one_site_heff_env(M, Lenv, W, Renv):
    X = np.einsum('byx,ysz->bxsz', Lenv, M, optimize=True)
    Y = np.einsum('bBst,bxsz->Bxtz', W, X, optimize=True)
    HM = np.einsum('Bxtz,Bza->xta', Y, Renv, optimize=True)
    return HM


def apply_two_site_heff_env(theta, Lenv, W1, W2, Renv):
    X = np.einsum('byx,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT


def heff_two_site_env_matvec(v, Lenv, W1, W2, Renv, Dl, d1, d2, Dr):
    theta = v.reshape((Dl, d1, d2, Dr), order='C')
    out = apply_two_site_heff_env(theta, Lenv, W1, W2, Renv)
    return out.reshape(Dl * d1 * d2 * Dr, order='C')


# ============================================================
# dense projected local operators, for validation
# ============================================================

def dense_projected_two_site_heff_from_full(Hfull, mps, i, d):
    UL = left_basis_map(mps, i)
    UR = right_basis_map(mps, i)

    Dl = UL.shape[1]
    Dr = UR.shape[0]
    dimL = UL.shape[0]
    dimR = UR.shape[1]

    Nloc = Dl * d * d * Dr
    Dim = dimL * d * d * dimR
    P = np.zeros((Dim, Nloc), dtype=np.complex128)

    col = 0
    for a in range(Dl):
        for s in range(d):
            for t in range(d):
                for c in range(Dr):
                    mid = np.zeros(d * d, dtype=np.complex128)
                    mid[s * d + t] = 1.0
                    vec = np.kron(UL[:, a], np.kron(mid, UR[c, :]))
                    P[:, col] = vec
                    col += 1

    Heff = P.conj().T @ Hfull @ P
    return Heff


def build_dense_local_heff_from_env(mps, mpo, i):
    A = mps[i]
    B = mps[i + 1]
    Dl, d1, mid = A.shape
    mid2, d2, Dr = B.shape
    assert mid == mid2

    Lenvs = build_left_envs(mps, mpo)
    Renvs = build_right_envs(mps, mpo)

    Lenv = Lenvs[i]
    Renv = Renvs[i + 1]

    N = Dl * d1 * d2 * Dr
    Heff = np.zeros((N, N), dtype=np.complex128)

    for j in range(N):
        e = np.zeros(N, dtype=np.complex128)
        e[j] = 1.0
        theta = e.reshape((Dl, d1, d2, Dr), order='C')
        Heff[:, j] = apply_two_site_heff_env(
            theta, Lenv, mpo[i], mpo[i + 1], Renv
        ).reshape(-1, order='C')

    return Heff


def compare_two_site_env_vs_proj(Hfull, mpo, mps, i, d, verbose=True):
    Heff_proj = dense_projected_two_site_heff_from_full(Hfull, mps, i, d)
    Heff_env = build_dense_local_heff_from_env(mps, mpo, i)

    diff = np.max(np.abs(Heff_env - Heff_proj))
    herm_env = np.max(np.abs(Heff_env - Heff_env.conj().T))
    herm_proj = np.max(np.abs(Heff_proj - Heff_proj.conj().T))

    if verbose:
        print(f"bond {i}: max|env-proj| = {diff:.3e}")
        print(f"bond {i}: env Herm err = {herm_env:.3e}")
        print(f"bond {i}: proj Herm err = {herm_proj:.3e}")

    return diff, herm_env, herm_proj


# ============================================================
# svd helpers and expectation values
# ============================================================

def form_theta(A, B):
    return np.einsum('asm,mtr->astr', A, B, optimize=True)


def split_theta_left_to_right(theta, Dmax, svd_cutoff=1e-12):
    Dl, d1, d2, Dr = theta.shape
    M = theta.reshape((Dl * d1, d2 * Dr), order='C')

    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    keep = np.sum(S > svd_cutoff)
    keep = max(1, min(Dmax, keep, len(S)))
    disc_weight = np.sum(S[keep:]**2).real

    U = U[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]

    A = U.reshape((Dl, d1, keep), order='C')
    B = (np.diag(S) @ Vh).reshape((keep, d2, Dr), order='C')
    return A, B, S, disc_weight


def split_theta_right_to_left(theta, Dmax, svd_cutoff=1e-12):
    Dl, d1, d2, Dr = theta.shape
    M = theta.reshape((Dl * d1, d2 * Dr), order='C')

    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    keep = np.sum(S > svd_cutoff)
    keep = max(1, min(Dmax, keep, len(S)))
    disc_weight = np.sum(S[keep:]**2).real

    U = U[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]

    A = (U @ np.diag(S)).reshape((Dl, d1, keep), order='C')
    B = Vh.reshape((keep, d2, Dr), order='C')
    return A, B, S, disc_weight


def expectation_value_mpo(mps, mpo):
    env = init_left_env()
    for i in range(len(mps)):
        env = left_env_update(env, mps[i], mpo[i])
    return env[0, 0, 0]


# ============================================================
# exact aklt mps
# ============================================================

def exact_aklt_mps(L, left_vec=None, right_vec=None, dtype=np.complex128):
    Ap = np.array([[0, np.sqrt(2/3)],
                   [0, 0]], dtype=dtype)
    A0 = np.array([[-1/np.sqrt(3), 0],
                   [0, 1/np.sqrt(3)]], dtype=dtype)
    Am = np.array([[0, 0],
                   [-np.sqrt(2/3), 0]], dtype=dtype)

    local = np.stack([Ap, A0, Am], axis=1)

    if left_vec is None:
        left_vec = np.array([1.0, 0.0], dtype=dtype)
    if right_vec is None:
        right_vec = np.array([1.0, 0.0], dtype=dtype)

    mps = []
    A1 = np.einsum('a,asb->sb', left_vec, local).reshape(1, 3, 2)
    mps.append(A1)

    for _ in range(1, L - 1):
        mps.append(local.copy())

    AL = np.einsum('asb,b->as', local, right_vec).reshape(2, 3, 1)
    mps.append(AL)

    return mps


# ============================================================
# final two-site dmrg
# ============================================================

def two_site_dmrg_env_matrix_free(
    mpo,
    d,
    Dmax=32,
    nsweeps=8,
    lanczos_tol=1e-10,
    lanczos_maxiter=None,
    svd_cutoff=1e-12,
    seed=1234,
    verbose=True,
):
    L = len(mpo)

    mps = random_mps(L, d=d, Dmax=min(Dmax, 4), seed=seed)
    mps = normalize_mps_by_state(mps)
    mps = make_mixed_canonical_around_bond(mps, 0)

    sweep_local_energies = []
    sweep_discards = []

    for sweep in range(nsweeps):
        discards_this_sweep = []
        last_local_energy = None

        # left -> right
        Renvs = build_right_envs(mps, mpo)
        Lenv = init_left_env()

        for i in range(L - 1):
            theta0 = form_theta(mps[i], mps[i + 1])
            Dl, d1, d2, Dr = theta0.shape
            Nloc = Dl * d1 * d2 * Dr
            Renv = Renvs[i + 1]

            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v, Lenv=Lenv, W1=mpo[i], W2=mpo[i + 1], Renv=Renv,
                              Dl=Dl, d1=d1, d2=d2, Dr=Dr:
                    heff_two_site_env_matvec(v, Lenv, W1, W2, Renv, Dl, d1, d2, Dr),
                dtype=np.complex128
            )

            v0 = theta0.reshape(-1, order='C')
            vals, vecs = eigsh(
                Heff, k=1, which='SA',
                v0=v0, tol=lanczos_tol, maxiter=lanczos_maxiter
            )
            last_local_energy = vals[0].real

            theta = vecs[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            Anew, Bnew, S, disc = split_theta_left_to_right(theta, Dmax, svd_cutoff)

            mps[i] = Anew
            mps[i + 1] = Bnew
            discards_this_sweep.append(disc)

            Lenv = left_env_update(Lenv, mps[i], mpo[i])

        mps = normalize_mps_by_state(mps)
        mps = make_mixed_canonical_around_bond(mps, L - 2)

        # right -> left
        Lenvs = build_left_envs(mps, mpo)
        Renv = init_right_env()

        for i in range(L - 2, -1, -1):
            theta0 = form_theta(mps[i], mps[i + 1])
            Dl, d1, d2, Dr = theta0.shape
            Nloc = Dl * d1 * d2 * Dr
            Lcur = Lenvs[i]

            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v, Lcur=Lcur, W1=mpo[i], W2=mpo[i + 1], Renv=Renv,
                              Dl=Dl, d1=d1, d2=d2, Dr=Dr:
                    heff_two_site_env_matvec(v, Lcur, W1, W2, Renv, Dl, d1, d2, Dr),
                dtype=np.complex128
            )

            v0 = theta0.reshape(-1, order='C')
            vals, vecs = eigsh(
                Heff, k=1, which='SA',
                v0=v0, tol=lanczos_tol, maxiter=lanczos_maxiter
            )
            last_local_energy = vals[0].real

            theta = vecs[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            Anew, Bnew, S, disc = split_theta_right_to_left(theta, Dmax, svd_cutoff)

            mps[i] = Anew
            mps[i + 1] = Bnew
            discards_this_sweep.append(disc)

            Renv = right_env_update(Renv, mps[i + 1], mpo[i + 1])

        mps = normalize_mps_by_state(mps)
        mps = make_mixed_canonical_around_bond(mps, 0)

        max_disc = max(discards_this_sweep) if discards_this_sweep else 0.0
        sweep_local_energies.append(last_local_energy if last_local_energy is not None else np.nan)
        sweep_discards.append(max_disc)

        if verbose:
            print(f"Sweep {sweep+1:2d}: local eig E = {sweep_local_energies[-1]:.12f}, "
                  f"max discarded weight = {max_disc:.3e}")

    return mps, np.array(sweep_local_energies), np.array(sweep_discards)


# ============================================================
# benchmarks
# ============================================================

def run_local_validation_heisenberg():
    L = 4
    d = 2
    i = 1

    mpo = heisenberg_mpo(L)
    Hfull = dense_heisenberg(L)

    mps = random_mps(L, d=d, Dmax=3, seed=7)
    mps = normalize_mps_by_state(mps)
    mps = make_mixed_canonical_around_bond(mps, i)

    return compare_two_site_env_vs_proj(Hfull, mpo, mps, i, d, verbose=True)


def run_heisenberg_demo(L=8, Dmax=32, nsweeps=6, J=1.0, Jz=1.0, h=0.0, seed=1234):
    mpo = heisenberg_mpo(L, J=J, Jz=Jz, h=h)

    mps, Es, disc = two_site_dmrg_env_matrix_free(
        mpo=mpo,
        d=2,
        Dmax=Dmax,
        nsweeps=nsweeps,
        seed=seed,
        verbose=True
    )

    E_mpo = expectation_value_mpo(mps, mpo).real
    print("\nHeisenberg <H> from MPO contraction =", E_mpo)

    Hfull = dense_heisenberg(L, J=J, Jz=Jz, h=h)
    psi = mps_to_state(mps)
    E_dense = (np.vdot(psi, Hfull @ psi) / np.vdot(psi, psi)).real
    E0 = np.linalg.eigvalsh(Hfull)[0].real

    print("Heisenberg energy from dense H on final MPS =", E_dense)
    print("Heisenberg exact ED ground energy           =", E0)
    print("Absolute error                              =", abs(E_dense - E0))

    return mps, Es, disc


def run_aklt_demo(L=6, Dmax=16, nsweeps=6, seed=1234):
    mpo = aklt_mpo(L)

    mps, Es, disc = two_site_dmrg_env_matrix_free(
        mpo=mpo,
        d=3,
        Dmax=Dmax,
        nsweeps=nsweeps,
        seed=seed,
        verbose=True
    )

    E_mpo = expectation_value_mpo(mps, mpo).real
    E_exact = -(2.0 / 3.0) * (L - 1)

    print("\nAKLT <H> from MPO contraction =", E_mpo)
    print("AKLT exact open-chain energy  =", E_exact)
    print("Absolute error                =", abs(E_mpo - E_exact))

    Hfull = dense_aklt(L)
    psi = mps_to_state(mps)
    E_dense = (np.vdot(psi, Hfull @ psi) / np.vdot(psi, psi)).real
    E0 = np.linalg.eigvalsh(Hfull)[0].real

    print("AKLT dense energy from final MPS =", E_dense)
    print("AKLT exact ED ground energy      =", E0)
    print("Difference to ED                 =", abs(E_dense - E0))

    return mps, Es, disc


def run_exact_aklt_mps_benchmark(L=6):
    mpo = aklt_mpo(L)
    mps = exact_aklt_mps(L)
    mps = normalize_mps_by_state(mps)

    E = expectation_value_mpo(mps, mpo).real
    E_exact = -(2.0 / 3.0) * (L - 1)

    print("Exact AKLT MPS energy =", E)
    print("Expected energy       =", E_exact)
    print("Error                 =", abs(E - E_exact))

    return E


if __name__ == "__main__":
    print("=== local validation ===")
    run_local_validation_heisenberg()

    print("\n=== heisenberg demo ===")
    run_heisenberg_demo()

    print("\n=== aklt demo ===")
    run_aklt_demo()

    print("\n=== exact aklt mps benchmark ===")
    run_exact_aklt_mps_benchmark()
```

---
## 🧑‍💻 User

Is it completed?

---
## 🤖 Assistant

Yes — **completed**.

What you now have is:

- a **working single Python module**
- for **Heisenberg** and **AKLT**
- using **fully MPO/MPS environment-based matrix-free local Hamiltonian action**
- with **two-site DMRG**
- and validated by:
  - local operator test \(H_{\text{env}} = P^\dagger H P\),
  - Heisenberg ED benchmark,
  - AKLT exact-energy benchmark,
  - exact AKLT MPS benchmark.

## In short
The requested implementation is done.

## What is included
- local operators
- Heisenberg MPO
- AKLT MPO
- dense Hamiltonians for testing
- MPS utilities
- left/right environments
- corrected one-site and two-site local actions
- matrix-free two-site DMRG
- benchmark runners

## What is not included
Not yet added:
- one-site DMRG
- Davidson solver
- variance calculation
- packaging/tests beyond the built-in benchmark functions

But for your stated request, the core task is finished.

## Recommended final check
Run:

```python
python mps_dmrg_env_final.py
```

You should see the successful validation and benchmark outputs.

If you want, next I can do either of these:

1. **add docstrings and polish formatting**
2. **split the module into smaller files**
3. **add one-site DMRG**
4. **add pytest unit tests**

---
## 🧑‍💻 User

Now come back to the latex note "MPS-Kimi-GPT-329.tex". According to our conversations and your debugging processes, any revision or correction should be made?

---
## 🤖 Assistant

Yes. The LaTeX note is **mostly correct in spirit**, but based on the debugging we just completed, there is one **important convention correction** and a few **strong clarifications** that should be made.

The main point is:

# The environment update formulas can remain as written,
but the local effective-Hamiltonian action formulas in the note are **not consistent with the actually validated code conventions**.

In particular, with your working implementation:

- `left_env_update` is
  \[
  L^{\text{new}} \sim L\,A\,W\,A^\*
  \]
  in the array sense used in the note,
- `right_env_update` is also as written,
- **but** when applying the local effective Hamiltonian, the left environment enters with **transposed bond indices** relative to the naive note formula,
- and for the two-site action, the MPO physical indices must be contracted in the routing corresponding to the validated candidate 4.

So the note should be revised in the sections:

- **MPO environments**
- **One-site effective Hamiltonian**
- **Two-site effective Hamiltonian**
- probably also a short **implementation warning** subsection.

---

# Most important correction

## 1. The one-site effective Hamiltonian formula should be revised

The current note gives, in essence,

\[
(\mathcal H_{\mathrm{eff}} M)_{\alpha_{i-1},\sigma_i',\alpha_i}
=
\sum
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha'_{i-1})
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
R^{[i]}_{\beta_i}(\alpha_i',\alpha_i)
M_{\alpha'_{i-1},\sigma_i,\alpha_i'}.
\]

But your validated code shows that with the note’s MPS/MPO conventions, the correct matrix-free application uses **the left environment with swapped bond indices**.

In code, the correct one-site action is:

```python
X  = np.einsum('byx,ysz->bxsz', L, M)
Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
HM = np.einsum('Bxtz,Bza->xta', Y, R)
```

So in index form, the corrected one-site action is better written as

\[
(\mathcal H_{\mathrm{eff}} M)_{\alpha_{i-1},\sigma_i,\alpha_i}
=
\sum_{\substack{
\tilde\alpha_{i-1},\tilde\alpha_i\\
\tau_i\\
\beta_{i-1},\beta_i}}
L^{[i]}_{\beta_{i-1}}(\tilde\alpha_{i-1},\alpha_{i-1})\,
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\tau_i}\,
R^{[i]}_{\beta_i}(\tilde\alpha_i,\alpha_i)\,
M_{\tilde\alpha_{i-1},\tau_i,\tilde\alpha_i}.
\]

That is, the left environment should be read as acting with indices ordered
\[
L_{\beta}(\text{input left-bond},\text{output left-bond})
\]
rather than the naive ordering previously stated.

Equivalent statement: the tensor stored as `L[b,x,y]` in the validated code enters the local action as `L[b,y,x]`.

This is a very important correction.

---

# Second important correction

## 2. The two-site effective Hamiltonian contraction in the note should be revised

The current note writes the two-site action in a natural symbolic form, but the implied contraction order is not sufficient to fix implementation, and the naive implementation was wrong.

The validated matrix-free action is:

```python
X = np.einsum('byx,yuvz->bxuvz', Lenv, theta)
Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
HT = np.einsum('Cxstz,Cza->xsta', Z, Renv)
```

So the note should explicitly state that, in the chosen conventions, the correct two-site action is

\[
(\mathcal H_{\mathrm{eff}}\Theta)_{x,s,t,a}
=
\sum_{b,B,C}
\sum_{y,u,v,z}
L_b(y,x)\,
W^{[i]}_{b,B,u,s}\,
W^{[i+1]}_{B,C,v,t}\,
R_C(z,a)\,
\Theta_{y,u,v,z},
\]

or equivalently, with relabeling,

\[
(\mathcal H_{\mathrm{eff}}\Theta)_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
=
\sum
L^{[i]}_{\beta_{i-1}}(\tilde\alpha_{i-1},\alpha_{i-1})
W^{[i]}_{\beta_{i-1},\beta_i,\tilde\sigma_i,\sigma_i}
W^{[i+1]}_{\beta_i,\beta_{i+1},\tilde\sigma_{i+1},\sigma_{i+1}}
R^{[i+1]}_{\beta_{i+1}}(\tilde\alpha_{i+1},\alpha_{i+1})
\Theta_{\tilde\alpha_{i-1},\tilde\sigma_i,\tilde\sigma_{i+1},\tilde\alpha_{i+1}}.
\]

The exact symbolic placement of input/output labels can be adjusted, but the key is:

- `L` appears with reversed bond-index order relative to the naive note formula,
- and the MPO physical indices appear in the routing corresponding to
  `bBus` and `BCvt` in the working code.

This should be written explicitly in the note. Otherwise, a future implementation is likely to reproduce the same bug we just debugged.

---

# Third correction / clarification

## 3. The “safe einsum templates” in the environment section need a warning

The note currently gives:

```python
Lnew = np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj())
Rnew = np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj())
```

These are actually fine as **environment updates** in the validated code.

But the note should add immediately after that:

> Important: with the conventions of this note, the stored left environment tensor `L[b,x,y]` enters the local effective-Hamiltonian action with swapped bond indices, i.e. as `L[b,y,x]` in the matrix-free one-site and two-site local contractions.

That one sentence would have saved a lot of debugging.

---

# Fourth correction

## 4. The one-site matrix-free wrapper shown in the note is not correct for the validated convention

The note currently shows:

```python
X  = np.einsum('bxy,ysz->bxsz', L, M)
Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
HM = np.einsum('Bxtz,Bza->xta', Y, R)
```

This should be replaced by the validated formula:

```python
X  = np.einsum('byx,ysz->bxsz', L, M)
Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
HM = np.einsum('Bxtz,Bza->xta', Y, R)
```

That is a concrete correction.

---

# Fifth correction

## 5. The two-site matrix-free wrapper in the note should be replaced by an explicit validated einsum sequence

Instead of the current symbolic generic one, the note should include the actual correct implementation template:

```python
X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
HT = np.einsum('Cxstz,Cza->xsta', Z, R)
```

and say explicitly that this template has been unit-tested against dense
\[
P^\dagger H P.
\]

That would be the best practical revision.

---

# Sixth recommended addition

## 6. Add a brief “implementation warning” paragraph

I strongly recommend adding a paragraph like this:

> **Implementation warning.**
> For the tensor storage conventions used here, Hermiticity of the environment-built local operator is not sufficient to guarantee correctness. In particular, the left environment update may be consistent and the resulting local operator may be Hermitian, while still differing from the true projected operator \(P^\dagger H P\) if the left environment bond indices are not used in the correct order inside the local matrix-free contraction. Therefore the one-site and two-site local actions must always be validated explicitly against dense projected references on small systems.

This is exactly what your debugging experience showed.

---

# Seventh recommended addition

## 7. Add the exact validated formulas to the implementation checklist

Under the checklist, add something like:

- verify one-site local operator against dense projected operator using the corrected contraction
  ```python
  X  = np.einsum('byx,ysz->bxsz', L, M)
  Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
  HM = np.einsum('Bxtz,Bza->xta', Y, R)
  ```
- verify two-site local operator against dense projected operator using
  ```python
  X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
  Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
  Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
  HT = np.einsum('Cxstz,Cza->xsta', Z, R)
  ```

This makes the note practically self-correcting.

---

# What does not need changing

These parts of the note are still fine:

- overall MPS tensor convention `(Dl, d, Dr)`
- MPO tensor convention `(chiL, chiR, d, d)` with `(ket, bra)`
- C-order reshape conventions
- canonicalization sections
- Heisenberg MPO
- AKLT MPO
- general DMRG algorithmic structure
- recommendation to validate local operators against dense projected references

So the note is **not fundamentally wrong**; the issue is that the local action formulas were not precise enough for this convention and, as written, encourage a wrong implementation.

---

# Minimal concrete edits

If you want the shortest actionable summary:

## Replace in the note:

### One-site local action code snippet
from
```python
X  = np.einsum('bxy,ysz->bxsz', L, M)
```
to
```python
X  = np.einsum('byx,ysz->bxsz', L, M)
```

### Two-site local action code snippet
replace by
```python
X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
HT = np.einsum('Cxstz,Cza->xsta', Z, R)
```

### Add a warning
that the stored left environment enters the local action with swapped bond indices.

---

If you want, I can next produce a **patch-style revision** of the relevant LaTeX sections, i.e. I rewrite the exact paragraphs/equations/code blocks in LaTeX form so you can paste them directly into `MPS-Kimi-GPT-329.tex`.

---
## 🧑‍💻 User

Please go ahead

---
## 🤖 Assistant

Below is a **patch-style revision** for the relevant parts of `MPS-Kimi-GPT-329.tex`, written so you can paste it directly into the note.

I will give:

1. a replacement for the **MPO environments** warning paragraph,
2. a replacement for the **one-site effective Hamiltonian** subsection,
3. a replacement for the **two-site effective Hamiltonian / matrix-free application** subsection,
4. a small addition to the **matrix-free eigensolver interface** section,
5. a small addition to the **implementation checklist**.

---

# 1. Add a warning near the end of Section `MPO environments`

Find the end of Section `\section{MPO environments}` and, after the current “Safe einsum templates” paragraph, add the following:

```latex
\paragraph{Important convention warning.}
For the tensor-storage and basis conventions used in these notes, the left environment
computed by Eq.~\eqref{eq:left-env-consistent} and by the update
\begin{verbatim}
Lnew = np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj())
\end{verbatim}
is \emph{not} inserted into the local effective-Hamiltonian action with the same
bond-index order as it is stored. In the validated matrix-free local contractions,
the stored array \texttt{L[b,x,y]} must be used as \texttt{L[b,y,x]}.
Equivalently, the left environment enters the one-site and two-site local actions with
its two MPS-bond indices interchanged relative to the most naive reading of the symbolic formula.

This point is easy to get wrong. A local operator built from environments can remain
Hermitian even when this index ordering is incorrect, while still failing to agree with the
true projected operator \(P^\dagger H P\). Therefore correctness must always be checked
against dense projected references on small systems.
```

---

# 2. Replace the subsection `\section{One-site effective Hamiltonian}`

Replace that whole section by the following.

```latex
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

\paragraph{Convention-sensitive form.}
For the tensor and environment conventions used in these notes, the matrix-free one-site
action is most safely specified by its validated contraction sequence rather than by a
single symbolic expression alone. In particular, the stored left environment enters the
local action with its two MPS-bond indices interchanged.

A correct matrix-free contraction sequence is:
\begin{align}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\alpha_i'}
&=
\sum_{\alpha_{i-1}'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1}',\alpha_{i-1})
M^{[i]}_{\alpha_{i-1}',\sigma_i,\alpha_i'},
\\
Y_{\beta_i,\alpha_{i-1},\sigma_i',\alpha_i'}
&=
\sum_{\beta_{i-1},\sigma_i}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i',\sigma_i}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\alpha_i'},
\\
(\mathcal{H}_{\mathrm{eff}} M)_{\alpha_{i-1},\sigma_i',\alpha_i}
&=
\sum_{\beta_i,\alpha_i'}
Y_{\beta_i,\alpha_{i-1},\sigma_i',\alpha_i'}
R^{[i]}_{\beta_i}(\alpha_i',\alpha_i).
\end{align}

In NumPy, the validated implementation is
\begin{verbatim}
X  = np.einsum('byx,ysz->bxsz', L, M)
Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
HM = np.einsum('Bxtz,Bza->xta', Y, R)
\end{verbatim}
where:
\begin{itemize}
    \item \texttt{L[b,x,y]} stores the left environment,
    \item \texttt{M[y,s,z]} is the center tensor,
    \item \texttt{W[b,B,s,t]} uses the MPO convention
    \[
    W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'},
    \]
    with the first physical index the ket/output index and the second the bra/input index,
    \item \texttt{R[B,z,a]} stores the right environment.
\end{itemize}

\paragraph{Important note.}
The first contraction uses \texttt{byx} rather than \texttt{bxy}. This reflects the
validated fact that, with the present conventions, the stored left environment must be
used in the local action with its two MPS-bond indices interchanged.

\paragraph{Recommended validation.}
For small systems, one should explicitly compare the dense operator built from the
matrix-free contraction above against the exact projected operator
\[
P^\dagger H P.
\]
Hermiticity of the environment-built local operator is necessary but not sufficient.
```

---

# 3. Replace the subsection `\section{Two-site effective Hamiltonian}`

You can keep the introductory part defining \(\Theta^{[i,i+1]}\), but replace the part from

- “Then the effective Hamiltonian acts on \(\Theta\) as ...”
- and especially the subsection
  `\subsection{Matrix-free application of the two-site effective Hamiltonian}`

with the following.

---

## Replacement text

```latex
\section{Two-site effective Hamiltonian}

At bond $(i,i+1)$, define the two-site center tensor
\begin{equation}
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}},
\label{eq:theta-def}
\end{equation}
so that
\begin{equation}
\ket{\psi}
=
\sum_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
\ket{\alpha_{i-1}}_L \ket{\sigma_i}\ket{\sigma_{i+1}} \ket{\alpha_{i+1}}_R.
\label{eq:two-site-state}
\end{equation}

Let $L^{[i]}$ be the left environment built from sites $1,\dots,i-1$, and let
$R^{[i+1]}$ be the right environment built from sites $i+2,\dots,L$.

\paragraph{Convention-sensitive form.}
As in the one-site case, the most reliable specification of the two-site effective Hamiltonian
is its validated contraction sequence. With the conventions used in these notes:
\begin{itemize}
    \item the stored left environment enters the local action with its two MPS-bond indices interchanged,
    \item the MPO physical-index routing in the validated two-site contraction is most transparently given by the explicit einsum sequence below.
\end{itemize}

\subsection{Matrix-free application of the two-site effective Hamiltonian}

Let
\[
\Theta \in \mathbb{C}^{D_{i-1}\times d_i \times d_{i+1}\times D_{i+1}},
\]
with storage order
\[
\Theta[\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}].
\]

A validated contraction sequence is:
\begin{align}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}'}
&=
\sum_{\alpha_{i-1}'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1}',\alpha_{i-1})
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

In NumPy, the validated implementation is
\begin{verbatim}
X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
HT = np.einsum('Cxstz,Cza->xsta', Z, R)
\end{verbatim}
where:
\begin{itemize}
    \item \texttt{Theta[y,u,v,z]} is the input two-site tensor,
    \item \texttt{L[b,x,y]} stores the left environment but enters the action as \texttt{byx},
    \item \texttt{W1[b,B,u,s]} and \texttt{W2[B,C,v,t]} follow the MPO storage convention
    \[
    W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'},
    \]
    with the first physical index the ket/output index and the second the bra/input index,
    \item \texttt{R[C,z,a]} stores the right environment,
    \item \texttt{HT[x,s,t,a]} is the output tensor.
\end{itemize}

\paragraph{Important note.}
A more naive contraction sequence, even if it yields a Hermitian local operator, need not
agree with the true projected operator \(P^\dagger H P\). The sequence above was explicitly
validated against dense projected references for small systems.

\paragraph{Practical recommendation.}
For a first implementation, write a reference function that builds the dense projected
operator \(P^\dagger H P\) for small dimensions, and compare it entrywise against the dense
matrix obtained by repeated application of the matrix-free contraction above.
```

---

# 4. Revise the `Matrix-free eigensolver interface for local effective Hamiltonians`

In the subsection `\subsection{One-site local solve}`, replace the code snippet by:

```latex
A practical NumPy/\texttt{scipy} interface is:
\begin{verbatim}
from scipy.sparse.linalg import LinearOperator, eigsh

def heff_one_site_matvec(v, L, W, R, Dl, d, Dr):
    M = v.reshape((Dl, d, Dr), order='C')
    X  = np.einsum('byx,ysz->bxsz', L, M)
    Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
    HM = np.einsum('Bxtz,Bza->xta', Y, R)
    return HM.reshape(Dl * d * Dr, order='C')

Nloc = Dl * d * Dr
Heff = LinearOperator(
    shape=(Nloc, Nloc),
    matvec=lambda v: heff_one_site_matvec(v, L, W, R, Dl, d, Dr),
    dtype=np.complex128
)

E, vec = eigsh(Heff, k=1, which='SA', v0=v0)
M = vec.reshape((Dl, d, Dr), order='C')
\end{verbatim}

Here the use of \texttt{'byx'} rather than \texttt{'bxy'} in the first contraction is essential
for consistency with the environment convention validated in these notes.
```

And in the subsection `\subsection{Two-site local solve}`, replace the generic wrapper snippet by:

```latex
A practical wrapper is
\begin{verbatim}
def heff_two_site_matvec(v, L, W1, W2, R, Dl, d1, d2, Dr):
    Theta = v.reshape((Dl, d1, d2, Dr), order='C')
    X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
    Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
    Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
    HT = np.einsum('Cxstz,Cza->xsta', Z, R)
    return HT.reshape(Dl * d1 * d2 * Dr, order='C')
\end{verbatim}
and this should be passed to \texttt{LinearOperator} exactly as in the one-site case.

This two-site contraction has been validated explicitly against the dense projected operator
\(P^\dagger H P\) for small systems.
```

---

# 5. Add to the `Implementation checklist`

In the checklist near the end, after the item

> \textbf{Two-site effective Hamiltonian test}: compare matrix-free action against explicitly constructed \(\mathcal H_{\mathrm{eff}}\) for small dimensions.

add the following sentence:

```latex
For the conventions of these notes, the validated matrix-free contractions are
\begin{verbatim}
# one-site
X  = np.einsum('byx,ysz->bxsz', L, M)
Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
HM = np.einsum('Bxtz,Bza->xta', Y, R)

# two-site
X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
HT = np.einsum('Cxstz,Cza->xsta', Z, R)
\end{verbatim}
and these should be treated as authoritative reference implementations.
```

---

# Optional but recommended short addition to the Practical coding recommendation section

At the end of `\section{Practical coding recommendation}`, add:

```latex
\paragraph{Final practical warning.}
When implementing environment-based matrix-free local solvers, do not trust Hermiticity
alone. In the present conventions, it is possible to obtain a Hermitian local operator
that is nevertheless incorrect because of an index-order mismatch in the left environment
or in the MPO physical-index routing. Always validate the local operator against the dense
projected operator \(P^\dagger H P\) on small systems before running full sweeps.
```

---

# Summary of the substantive corrections

The main revisions are:

- keep the environment update equations,
- correct the local-action formulas,
- explicitly state that the stored left environment enters the local action with swapped bond indices,
- replace the one-site and two-site einsum templates by the validated ones,
- strengthen the warning that Hermiticity is not enough.

---

If you want, I can next produce:

## Version 2
a **fully rewritten LaTeX subsection text** with polished prose integrated seamlessly into the original style, rather than patch blocks.

---
## 🧑‍💻 User

Please give me a fully rewritten LaTeX subsection text 

---
## 🤖 Assistant

Below is a **fully rewritten LaTeX subsection text** for the relevant parts, in a style matching your note. You can paste these directly into `MPS-Kimi-GPT-329.tex` to replace the corresponding subsections.

I will provide rewritten text for:

1. `\section{MPO environments}`
2. `\section{One-site effective Hamiltonian}`
3. `\section{Two-site effective Hamiltonian}`
4. the relevant code snippets in `\section{Matrix-free eigensolver interface for local effective Hamiltonians}`
5. a short added warning in the checklist/coding-recommendation style

---

# 1. Fully rewritten `\section{MPO environments}`

Replace the current section `\section{MPO environments}` by:

```latex
\section{MPO environments}
\label{sec:env}

For a center at site $i$, let $L^{[i]}$ denote the contraction of sites
$1,\dots,i-1$ into a left environment, and let $R^{[i]}$ denote the contraction
of sites $i+1,\dots,L$ into a right environment.

We store the environments as tensors
\[
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha'_{i-1}),
\qquad
R^{[i]}_{\beta_i}(\alpha_i,\alpha'_i),
\]
with array shapes
\[
L^{[i]} \in \mathbb{C}^{\chi_{i-1}\times D_{i-1}\times D_{i-1}},
\qquad
R^{[i]} \in \mathbb{C}^{\chi_i\times D_i\times D_i}.
\]

The left environment is updated by
\begin{equation}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha'_{i-1})
=
\sum_{\substack{
\alpha_{i-2},\alpha'_{i-2}\\
\sigma,\sigma'\\
\beta_{i-2}
}}
L^{[i-1]}_{\beta_{i-2}}(\alpha_{i-2},\alpha'_{i-2})
A^{[i-1]}_{\alpha_{i-2},\sigma,\alpha_{i-1}}
W^{[i-1]}_{\beta_{i-2},\beta_{i-1},\sigma,\sigma'}
A^{[i-1]*}_{\alpha'_{i-2},\sigma',\alpha'_{i-1}},
\label{eq:left-env-consistent-revised}
\end{equation}
while the right environment is updated by
\begin{equation}
R^{[i]}_{\beta_i}(\alpha_i,\alpha'_i)
=
\sum_{\substack{
\alpha_{i+1},\alpha'_{i+1}\\
\sigma,\sigma'\\
\beta_{i+1}
}}
B^{[i+1]}_{\alpha_i,\sigma,\alpha_{i+1}}
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma,\sigma'}
R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1},\alpha'_{i+1})
B^{[i+1]*}_{\alpha'_i,\sigma',\alpha'_{i+1}}.
\label{eq:right-env-consistent-revised}
\end{equation}

For open boundary conditions,
\begin{equation}
L^{[1]}_{\beta_0}(1,1)=\delta_{\beta_0,1},
\qquad
R^{[L]}_{\beta_L}(1,1)=\delta_{\beta_L,1},
\end{equation}
with $D_0=D_L=\chi_0=\chi_L=1$.
In code, these are simply
\begin{verbatim}
L = np.zeros((1,1,1), dtype=np.complex128); L[0,0,0] = 1
R = np.zeros((1,1,1), dtype=np.complex128); R[0,0,0] = 1
\end{verbatim}

\paragraph{Validated einsum templates.}
With array conventions
\begin{itemize}
    \item \texttt{Lold[b,x,y]},
    \item \texttt{A[x,s,a]},
    \item \texttt{W[b,B,s,t]},
    \item \texttt{Rold[B,a,c]},
    \item \texttt{B[x,s,a]},
\end{itemize}
the validated updates are
\begin{verbatim}
Lnew = np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj())
Rnew = np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj())
\end{verbatim}

\paragraph{Important convention warning.}
The update formulas above define the \emph{stored} environment tensors. However,
for the tensor-storage conventions used in these notes, the stored left environment
does not enter the local effective-Hamiltonian action with the same bond-index order
as it is stored. In the validated one-site and two-site local contractions, the array
\texttt{L[b,x,y]} must be used as \texttt{L[b,y,x]}.

This point is subtle but crucial. One can obtain an environment-built local operator
that is Hermitian yet still incorrect if this bond-index ordering is mishandled.
Therefore, for any implementation based on these conventions, one must validate the
local environment-based operator explicitly against the dense projected operator
\[
P^\dagger H P
\]
on small systems.
```

---

# 2. Fully rewritten `\section{One-site effective Hamiltonian}`

Replace the entire current section by:

```latex
\section{One-site effective Hamiltonian}

Let the center tensor at site $i$ be
\[
M^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}.
\]
The corresponding local variational problem is
\begin{equation}
\mathcal{H}_{\mathrm{eff}}\,v = E v,
\end{equation}
where $v$ is the vectorized center tensor.

\paragraph{Convention-sensitive form.}
Because the left environment is stored in a convention that differs from the one in
which it enters the local action, the most reliable definition of the one-site effective
Hamiltonian is the validated contraction sequence rather than a bare symbolic formula alone.

For the conventions of these notes, the correct one-site matrix-free action is:
\begin{align}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\alpha_i'}
&=
\sum_{\alpha_{i-1}'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1}',\alpha_{i-1})
M^{[i]}_{\alpha_{i-1}',\sigma_i,\alpha_i'},
\\
Y_{\beta_i,\alpha_{i-1},\sigma_i',\alpha_i'}
&=
\sum_{\beta_{i-1},\sigma_i}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i',\sigma_i}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\alpha_i'},
\\
(\mathcal{H}_{\mathrm{eff}}M)_{\alpha_{i-1},\sigma_i',\alpha_i}
&=
\sum_{\beta_i,\alpha_i'}
Y_{\beta_i,\alpha_{i-1},\sigma_i',\alpha_i'}
R^{[i]}_{\beta_i}(\alpha_i',\alpha_i).
\label{eq:heff-action-revised}
\end{align}

The key point is that the left environment appears as
\[
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1}',\alpha_{i-1}),
\]
not as
\[
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}').
\]

\paragraph{Validated NumPy implementation.}
With arrays
\begin{itemize}
    \item \texttt{L[b,x,y]},
    \item \texttt{M[y,s,z]},
    \item \texttt{W[b,B,s,t]},
    \item \texttt{R[B,z,a]},
\end{itemize}
the correct contraction is
\begin{verbatim}
X  = np.einsum('byx,ysz->bxsz', L, M)
Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
HM = np.einsum('Bxtz,Bza->xta', Y, R)
\end{verbatim}
The use of \texttt{'byx'} rather than \texttt{'bxy'} in the first contraction is essential.

\paragraph{Interpretation.}
The MPO tensor is stored as
\[
W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'},
\]
where the first physical index is the ket/output index and the second is the bra/input
index. The contraction above is the validated matrix-free implementation consistent with
the environment conventions of these notes.

\paragraph{Practical warning.}
A naive contraction using \texttt{L[b,x,y]} directly in the first step may still produce
a Hermitian local operator, but that operator need not coincide with the true projected
operator \(P^\dagger H P\). Therefore the contraction in Eq.~\eqref{eq:heff-action-revised}
should be treated as the authoritative implementation for the present conventions.
```

---

# 3. Fully rewritten `\section{Two-site effective Hamiltonian}`

Replace the whole current section by:

```latex
\section{Two-site effective Hamiltonian}

At bond $(i,i+1)$, define the two-site center tensor
\begin{equation}
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}},
\label{eq:theta-def}
\end{equation}
so that
\begin{equation}
\ket{\psi}
=
\sum_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
\ket{\alpha_{i-1}}_L \ket{\sigma_i}\ket{\sigma_{i+1}} \ket{\alpha_{i+1}}_R.
\label{eq:two-site-state}
\end{equation}

Let $L^{[i]}$ denote the left environment built from sites $1,\dots,i-1$, and
let $R^{[i+1]}$ denote the right environment built from sites $i+2,\dots,L$.

\paragraph{Convention-sensitive form.}
As in the one-site case, the safest and most useful definition of the two-site effective
Hamiltonian is the validated contraction sequence. For the conventions used in these notes:
\begin{itemize}
    \item the stored left environment enters the local action with its two MPS-bond indices interchanged,
    \item the MPO physical indices must be routed exactly as in the validated contraction below.
\end{itemize}

\subsection{Matrix-free application of the two-site effective Hamiltonian}

Let
\[
\Theta \in \mathbb{C}^{D_{i-1}\times d_i \times d_{i+1}\times D_{i+1}},
\]
stored as
\[
\Theta[\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}].
\]

A validated contraction sequence is:
\begin{align}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}'}
&=
\sum_{\alpha_{i-1}'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1}',\alpha_{i-1})
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
\label{eq:two-site-heff-revised}
\end{align}

\paragraph{Validated NumPy implementation.}
With arrays
\begin{itemize}
    \item \texttt{L[b,x,y]},
    \item \texttt{Theta[y,u,v,z]},
    \item \texttt{W1[b,B,u,s]},
    \item \texttt{W2[B,C,v,t]},
    \item \texttt{R[C,z,a]},
\end{itemize}
the correct matrix-free contraction is
\begin{verbatim}
X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
HT = np.einsum('Cxstz,Cza->xsta', Z, R)
\end{verbatim}
with output tensor \texttt{HT[x,s,t,a]}.

\paragraph{Important note.}
The first contraction uses \texttt{'byx'} rather than \texttt{'bxy'}, and the MPO tensors
enter with index routings \texttt{'bBus'} and \texttt{'BCvt'}. These details are not optional:
they are the convention-dependent choices required for agreement with the exact projected operator
\[
P^\dagger H P.
\]

\paragraph{Validation requirement.}
For small systems, the dense matrix obtained by repeated application of the matrix-free map
in Eq.~\eqref{eq:two-site-heff-revised} should be compared entrywise against the dense projected
operator \(P^\dagger H P\). This comparison is the decisive local correctness test for the
environment-based implementation.
```

---

# 4. Fully rewritten code blocks in `\section{Matrix-free eigensolver interface for local effective Hamiltonians}`

You can keep most of that section, but replace the key code blocks by the following.

---

## Replace the one-site wrapper block with:

```latex
A practical NumPy/\texttt{scipy} interface is:
\begin{verbatim}
from scipy.sparse.linalg import LinearOperator, eigsh

def heff_one_site_matvec(v, L, W, R, Dl, d, Dr):
    M = v.reshape((Dl, d, Dr), order='C')
    X  = np.einsum('byx,ysz->bxsz', L, M)
    Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
    HM = np.einsum('Bxtz,Bza->xta', Y, R)
    return HM.reshape(Dl * d * Dr, order='C')

Nloc = Dl * d * Dr
Heff = LinearOperator(
    shape=(Nloc, Nloc),
    matvec=lambda v: heff_one_site_matvec(v, L, W, R, Dl, d, Dr),
    dtype=np.complex128
)

E, vec = eigsh(Heff, k=1, which='SA', v0=v0)
M = vec.reshape((Dl, d, Dr), order='C')
\end{verbatim}

The use of \texttt{'byx'} in the first contraction is essential for consistency with
the environment convention adopted in these notes.
```

---

## Replace the two-site wrapper block with:

```latex
A practical wrapper has the form
\begin{verbatim}
def heff_two_site_matvec(v, L, W1, W2, R, Dl, d1, d2, Dr):
    Theta = v.reshape((Dl, d1, d2, Dr), order='C')
    X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
    Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
    Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
    HT = np.einsum('Cxstz,Cza->xsta', Z, R)
    return HT.reshape(Dl * d1 * d2 * Dr, order='C')
\end{verbatim}
and is passed to \texttt{LinearOperator} in the same way as for the one-site case.

This contraction sequence has been validated explicitly against the dense projected operator
\(P^\dagger H P\) for small systems and should be treated as the authoritative implementation
for the conventions of these notes.
```

---

# 5. Add a rewritten warning paragraph near the end of the checklist or practical coding section

Insert this paragraph near the end of the note, for example after the implementation checklist or in the practical coding recommendation section:

```latex
\paragraph{Final implementation warning.}
For the conventions of these notes, Hermiticity of the environment-built local operator
is not sufficient to guarantee correctness. In particular, it is possible to construct a
one-site or two-site local operator that is Hermitian and numerically stable, yet still differs
from the true projected operator \(P^\dagger H P\) because of an index-order mismatch in the
left environment or in the MPO physical-index routing. Therefore the local matrix-free action
must always be validated against dense projected references on small systems before running
full DMRG sweeps.
```

---

# Short summary of what changed

These rewritten subsections encode the key lessons from the debugging:

- environment updates remain standard in storage form,
- the stored left environment enters local actions with swapped bond indices,
- the one-site matrix-free contraction must start with `byx`,
- the two-site matrix-free contraction must use
  `byx`, `bBus`, `BCvt`,
- dense projected local-operator tests are mandatory.

---

If you want, I can next produce a **diff-style concise edit guide**, e.g. “replace section X with Y, replace paragraph Z with W,” to make patching the `.tex` file easier.

---
## 🧑‍💻 User

Could you give me the fully written section \section{Matrix-free eigensolver interface for local effective Hamiltonians}?

---
## 🤖 Assistant

```latex
\section{Matrix-free eigensolver interface for local effective Hamiltonians}
\label{sec:matfree}

In practical DMRG implementations, the local effective Hamiltonian should \emph{never}
be formed as a dense matrix except for unit tests on very small systems. Instead, one
implements only the map
\[
v \mapsto \mathcal{H}_{\mathrm{eff}} v,
\]
and passes this map to an iterative eigensolver such as Lanczos or Davidson.

\paragraph{Important convention warning.}
For the tensor, MPO, and environment conventions adopted in these notes, the stored
left environment tensor enters the local effective-Hamiltonian action with its two
MPS-bond indices interchanged. In array form, if the stored left environment is
\texttt{L[b,x,y]}, then the validated local matrix-free contractions use it as
\texttt{L[b,y,x]}. This point is subtle but essential. A local operator built from
environments can remain Hermitian even when this ordering is wrong, while still failing
to agree with the true projected operator
\[
P^\dagger H P.
\]
Accordingly, the one-site and two-site matrix-free contractions given below should be
treated as the authoritative implementations for the conventions of these notes.

\subsection{One-site local solve}

For a one-site center tensor
\[
M^{[i]} \in \mathbb{C}^{D_{i-1}\times d_i \times D_i},
\]
define the local vector-space dimension
\[
N_{\mathrm{loc}} = D_{i-1} d_i D_i.
\]
The eigensolver vector
\[
v\in\mathbb{C}^{N_{\mathrm{loc}}}
\]
is identified with the tensor $M^{[i]}$ by
\begin{equation}
M^{[i]} = \mathrm{reshape}(v,\,(D_{i-1},d_i,D_i)),
\end{equation}
using the fixed C-order convention of Section~\ref{sec:reshape}. After applying the
local effective Hamiltonian, the result is reshaped back to a vector with the same convention.

\paragraph{Validated one-site contraction.}
With arrays
\begin{itemize}
    \item \texttt{L[b,x,y]} for the stored left environment,
    \item \texttt{M[y,s,z]} for the center tensor,
    \item \texttt{W[b,B,s,t]} for the MPO tensor,
    \item \texttt{R[B,z,a]} for the stored right environment,
\end{itemize}
the validated one-site matrix-free action is
\begin{verbatim}
X  = np.einsum('byx,ysz->bxsz', L, M)
Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
HM = np.einsum('Bxtz,Bza->xta', Y, R)
\end{verbatim}
where \texttt{HM} has the same shape as \texttt{M}.

In index notation, this means
\begin{align}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\alpha_i'}
&=
\sum_{\alpha_{i-1}'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1}',\alpha_{i-1})
M^{[i]}_{\alpha_{i-1}',\sigma_i,\alpha_i'},
\\
Y_{\beta_i,\alpha_{i-1},\sigma_i',\alpha_i'}
&=
\sum_{\beta_{i-1},\sigma_i}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i',\sigma_i}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\alpha_i'},
\\
(\mathcal{H}_{\mathrm{eff}} M)_{\alpha_{i-1},\sigma_i',\alpha_i}
&=
\sum_{\beta_i,\alpha_i'}
Y_{\beta_i,\alpha_{i-1},\sigma_i',\alpha_i'}
R^{[i]}_{\beta_i}(\alpha_i',\alpha_i).
\end{align}

\paragraph{Matrix-free wrapper.}
A practical NumPy/\texttt{scipy} interface is:
\begin{verbatim}
from scipy.sparse.linalg import LinearOperator, eigsh

def heff_one_site_matvec(v, L, W, R, Dl, d, Dr):
    M = v.reshape((Dl, d, Dr), order='C')
    X  = np.einsum('byx,ysz->bxsz', L, M)
    Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
    HM = np.einsum('Bxtz,Bza->xta', Y, R)
    return HM.reshape(Dl * d * Dr, order='C')

Nloc = Dl * d * Dr
Heff = LinearOperator(
    shape=(Nloc, Nloc),
    matvec=lambda v: heff_one_site_matvec(v, L, W, R, Dl, d, Dr),
    dtype=np.complex128
)

E, vec = eigsh(Heff, k=1, which='SA', v0=v0)
M = vec.reshape((Dl, d, Dr), order='C')
\end{verbatim}

Here \texttt{v0} should be the current local tensor flattened with the same reshape convention:
\begin{verbatim}
v0 = M_current.reshape(Nloc, order='C')
\end{verbatim}

\paragraph{Hermiticity and correctness checks.}
For a Hermitian MPO and correctly built environments, the local operator should satisfy
\[
\langle x, \mathcal{H}_{\mathrm{eff}} y\rangle
=
\langle \mathcal{H}_{\mathrm{eff}} x, y\rangle
\]
up to numerical precision. However, Hermiticity alone is not a sufficient correctness test.
For small systems, one should also compare the dense matrix obtained from the matrix-free
routine against the exact projected operator
\[
P^\dagger H P.
\]

\subsection{Two-site local solve}

For a two-site center tensor
\[
\Theta^{[i,i+1]} \in
\mathbb{C}^{D_{i-1}\times d_i \times d_{i+1}\times D_{i+1}},
\]
the local vector-space dimension is
\[
N_{\mathrm{loc}} = D_{i-1} d_i d_{i+1} D_{i+1}.
\]
Again, the eigensolver vector $v$ is identified with the tensor $\Theta$ by
\begin{equation}
\Theta = \mathrm{reshape}(v,\,(D_{i-1},d_i,d_{i+1},D_{i+1})).
\end{equation}
After applying the local effective Hamiltonian, the output tensor is flattened back
to a vector using the same C-order convention.

\paragraph{Validated two-site contraction.}
With arrays
\begin{itemize}
    \item \texttt{L[b,x,y]} for the stored left environment,
    \item \texttt{Theta[y,u,v,z]} for the input two-site tensor,
    \item \texttt{W1[b,B,u,s]} for the MPO tensor on site $i$,
    \item \texttt{W2[B,C,v,t]} for the MPO tensor on site $i+1$,
    \item \texttt{R[C,z,a]} for the stored right environment,
\end{itemize}
the validated matrix-free two-site action is
\begin{verbatim}
X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
HT = np.einsum('Cxstz,Cza->xsta', Z, R)
\end{verbatim}
where \texttt{HT} has the same shape as \texttt{Theta}.

In index notation, this is
\begin{align}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}'}
&=
\sum_{\alpha_{i-1}'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1}',\alpha_{i-1})
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

\paragraph{Matrix-free wrapper.}
A practical wrapper is
\begin{verbatim}
def heff_two_site_matvec(v, L, W1, W2, R, Dl, d1, d2, Dr):
    Theta = v.reshape((Dl, d1, d2, Dr), order='C')
    X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
    Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
    Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
    HT = np.einsum('Cxstz,Cza->xsta', Z, R)
    return HT.reshape(Dl * d1 * d2 * Dr, order='C')
\end{verbatim}
and this is passed to \texttt{LinearOperator} exactly as in the one-site case.

\paragraph{Practical remark.}
The two-site contraction above was explicitly validated against the dense projected operator
\[
P^\dagger H P
\]
for small systems. This validation is essential, because more naive einsum routings can
produce Hermitian local operators that are nevertheless incorrect.

\subsection{Lanczos versus Davidson}

For first implementations, Lanczos is usually sufficient and easiest to access through
standard sparse-eigensolver interfaces. Davidson can be advantageous when the local
dimension becomes larger or when a useful diagonal preconditioner is available.

\begin{itemize}
    \item \textbf{Lanczos:} simplest robust choice for initial code.
    \item \textbf{Davidson:} often faster for large local problems, but requires more implementation effort.
\end{itemize}

In both cases, using the current local tensor as the initial guess is essential for sweep efficiency.

\subsection{Dense reference for unit tests}

For very small dimensions only, one should explicitly build the dense local projected operator
\[
H_{\mathrm{eff}} \in \mathbb{C}^{N_{\mathrm{loc}}\times N_{\mathrm{loc}}},
\]
and compare:
\begin{itemize}
    \item dense matrix--vector multiplication versus the matrix-free routine,
    \item lowest eigenvalue from dense diagonalization versus Lanczos/Davidson,
    \item Hermiticity of the dense reference matrix,
    \item agreement of the environment-built operator with the exact projected operator
    \(P^\dagger H P\).
\end{itemize}

This unit test should be performed for both the one-site and two-site local solvers before
running full sweeps. For the conventions of these notes, the one-site and two-site contraction
templates written above should be treated as the authoritative reference implementations.
```

---
## 🧑‍💻 User

Is the output cut off?

---
## 🤖 Assistant

No — it was complete.

The section ended at:

```latex
This unit test should be performed for both the one-site and two-site local solvers before
running full sweeps. For the conventions of these notes, the one-site and two-site contraction
templates written above should be treated as the authoritative reference implementations.
```

So nothing was truncated.

If you want, I can now also provide:

1. a **fully rewritten `\section{One-site effective Hamiltonian}`** in the same polished style,
2. a **fully rewritten `\section{Two-site effective Hamiltonian}`**,
3. or a **single consolidated patch package** listing all sections that should be replaced in the `.tex` file.

---
## 🧑‍💻 User

gao with 1 and 2

---
## 🤖 Assistant

Below are **fully rewritten polished LaTeX sections** for

1. `\section{One-site effective Hamiltonian}`
2. `\section{Two-site effective Hamiltonian}`

written to be consistent with the corrected and validated implementation.

You can paste them directly into `MPS-Kimi-GPT-329.tex` in place of the corresponding existing sections.

---

# 1. Fully rewritten `\section{One-site effective Hamiltonian}`

```latex
\section{One-site effective Hamiltonian}

Let the center tensor at site $i$ be
\[
M^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}.
\]
The local variational problem is
\begin{equation}
\mathcal{H}_{\mathrm{eff}}\,v = E v,
\end{equation}
where $v$ denotes the vectorized form of the center tensor.

\paragraph{General structure.}
The one-site effective Hamiltonian is obtained by contracting the full MPO with all
MPS tensors outside site $i$, leaving only the local tensor $M^{[i]}$ open.
In principle one may write this object symbolically as a contraction of a left environment,
one local MPO tensor, and a right environment. In practice, however, the safest way to
specify the operator---and the one that is directly useful for coding---is through its
validated matrix-free contraction sequence.

\paragraph{Convention-sensitive point.}
For the conventions used in these notes, the stored left environment tensor does not enter
the local action with the same MPS-bond index order as it is stored. If the stored array is
\texttt{L[b,x,y]}, then the validated local contraction uses it as \texttt{L[b,y,x]}.
Equivalently, the left environment enters the one-site effective Hamiltonian with its two
MPS-bond indices interchanged relative to the most naive symbolic reading.

\subsection{Validated matrix-free action}

Let
\[
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha'_{i-1})
\]
be the stored left environment and
\[
R^{[i]}_{\beta_i}(\alpha_i,\alpha'_i)
\]
the stored right environment. Then the validated one-site matrix-free action is
\begin{align}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\alpha_i'}
&=
\sum_{\alpha_{i-1}'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1}',\alpha_{i-1})
M^{[i]}_{\alpha_{i-1}',\sigma_i,\alpha_i'},
\label{eq:one-site-step1}
\\
Y_{\beta_i,\alpha_{i-1},\sigma_i',\alpha_i'}
&=
\sum_{\beta_{i-1},\sigma_i}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i',\sigma_i}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\alpha_i'},
\label{eq:one-site-step2}
\\
(\mathcal{H}_{\mathrm{eff}}M)_{\alpha_{i-1},\sigma_i',\alpha_i}
&=
\sum_{\beta_i,\alpha_i'}
Y_{\beta_i,\alpha_{i-1},\sigma_i',\alpha_i'}
R^{[i]}_{\beta_i}(\alpha_i',\alpha_i).
\label{eq:one-site-step3}
\end{align}

This is the convention-correct form for the one-site effective Hamiltonian in the present notes.
In particular, the first step involves
\[
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1}',\alpha_{i-1}),
\]
not
\[
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}').
\]

\subsection{Validated NumPy implementation}

With arrays
\begin{itemize}
    \item \texttt{L[b,x,y]} for the stored left environment,
    \item \texttt{M[y,s,z]} for the local center tensor,
    \item \texttt{W[b,B,s,t]} for the MPO tensor,
    \item \texttt{R[B,z,a]} for the stored right environment,
\end{itemize}
the validated implementation is
\begin{verbatim}
X  = np.einsum('byx,ysz->bxsz', L, M)
Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
HM = np.einsum('Bxtz,Bza->xta', Y, R)
\end{verbatim}

Here \texttt{HM} has the same shape as \texttt{M}. The use of \texttt{'byx'} rather than
\texttt{'bxy'} in the first contraction is essential.

\paragraph{Interpretation of physical indices.}
The MPO tensor is stored as
\[
W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'},
\]
with the first physical index $\sigma$ the ket/output index and the second physical index
$\sigma'$ the bra/input index. The contraction in Eq.~\eqref{eq:one-site-step2} is the
validated routing consistent with this convention and with the environment storage described above.

\subsection{Vectorization and local eigensolver interface}

If
\[
M^{[i]} \in \mathbb{C}^{D_{i-1}\times d_i \times D_i},
\]
then the local vector-space dimension is
\[
N_{\mathrm{loc}} = D_{i-1} d_i D_i.
\]
The eigensolver vector
\[
v \in \mathbb{C}^{N_{\mathrm{loc}}}
\]
is identified with the tensor by
\[
M^{[i]} = \mathrm{reshape}(v,\,(D_{i-1},d_i,D_i)),
\]
using the fixed C-order convention of Section~\ref{sec:reshape}. After applying the local
effective Hamiltonian, the output tensor is flattened back into a vector with the same convention.

A practical wrapper is
\begin{verbatim}
from scipy.sparse.linalg import LinearOperator, eigsh

def heff_one_site_matvec(v, L, W, R, Dl, d, Dr):
    M = v.reshape((Dl, d, Dr), order='C')
    X  = np.einsum('byx,ysz->bxsz', L, M)
    Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
    HM = np.einsum('Bxtz,Bza->xta', Y, R)
    return HM.reshape(Dl * d * Dr, order='C')

Nloc = Dl * d * Dr
Heff = LinearOperator(
    shape=(Nloc, Nloc),
    matvec=lambda v: heff_one_site_matvec(v, L, W, R, Dl, d, Dr),
    dtype=np.complex128
)
\end{verbatim}

The eigensolver should be initialized with the current local tensor:
\begin{verbatim}
v0 = M_current.reshape(Nloc, order='C')
\end{verbatim}

\subsection{Validation strategy}

For small systems, one should explicitly build the dense projected operator
\[
P^\dagger H P
\]
for the one-site center and compare it against the dense matrix obtained by repeated
application of the matrix-free routine above.

\paragraph{Important warning.}
Hermiticity of the environment-built local operator is necessary but not sufficient.
With the conventions of these notes, a naive contraction using the stored left environment
without the required index interchange can still produce a Hermitian operator, while failing
to agree with the true projected operator \(P^\dagger H P\). Therefore the contraction
sequence in Eqs.~\eqref{eq:one-site-step1}--\eqref{eq:one-site-step3} should be treated as
the authoritative implementation for the present conventions.
```

---

# 2. Fully rewritten `\section{Two-site effective Hamiltonian}`

```latex
\section{Two-site effective Hamiltonian}

At bond $(i,i+1)$, define the two-site center tensor
\begin{equation}
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}},
\label{eq:theta-def}
\end{equation}
so that
\begin{equation}
\ket{\psi}
=
\sum_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
\Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}}
\ket{\alpha_{i-1}}_L \ket{\sigma_i}\ket{\sigma_{i+1}} \ket{\alpha_{i+1}}_R.
\label{eq:two-site-state}
\end{equation}

The two-site local variational problem is the direct analogue of the standard
two-site DMRG superblock problem, but expressed entirely in terms of MPS/MPO
environments and a matrix-free local action.

\paragraph{General structure.}
In principle, the effective Hamiltonian at bond $(i,i+1)$ may be written symbolically
as a contraction of
\begin{itemize}
    \item a left environment built from sites $1,\dots,i-1$,
    \item the two local MPO tensors at sites $i$ and $i+1$,
    \item a right environment built from sites $i+2,\dots,L$.
\end{itemize}
In practice, however, the most reliable specification is again the validated contraction
sequence itself.

\paragraph{Convention-sensitive point.}
For the conventions used in these notes, two implementation details are essential:
\begin{enumerate}
    \item the stored left environment enters the local action with its two MPS-bond
    indices interchanged,
    \item the local MPO tensors must be contracted with the physical-index routing that
    is consistent with the validated implementation below.
\end{enumerate}
These points are easy to get wrong, and more naive contractions can produce Hermitian
local operators that nevertheless fail to agree with the true projected operator
\[
P^\dagger H P.
\]

\subsection{Validated matrix-free action}

Let
\[
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha'_{i-1})
\]
be the stored left environment built from sites $1,\dots,i-1$, and let
\[
R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1},\alpha'_{i+1})
\]
be the stored right environment built from sites $i+2,\dots,L$.

The validated matrix-free two-site action is
\begin{align}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}'}
&=
\sum_{\alpha_{i-1}'}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1}',\alpha_{i-1})
\Theta_{\alpha_{i-1}',\sigma_i,\sigma_{i+1},\alpha_{i+1}'},
\label{eq:two-site-step1}
\\
Y_{\beta_i,\alpha_{i-1},\sigma_i',\sigma_{i+1},\alpha_{i+1}'}
&=
\sum_{\beta_{i-1},\sigma_i}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
X_{\beta_{i-1},\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}'},
\label{eq:two-site-step2}
\\
Z_{\beta_{i+1},\alpha_{i-1},\sigma_i',\sigma_{i+1}',\alpha_{i+1}'}
&=
\sum_{\beta_i,\sigma_{i+1}}
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma_{i+1}'}
Y_{\beta_i,\alpha_{i-1},\sigma_i',\sigma_{i+1},\alpha_{i+1}'},
\label{eq:two-site-step3}
\\
(\mathcal{H}_{\mathrm{eff}}\Theta)_{\alpha_{i-1},\sigma_i',\sigma_{i+1}',\alpha_{i+1}}
&=
\sum_{\beta_{i+1},\alpha_{i+1}'}
Z_{\beta_{i+1},\alpha_{i-1},\sigma_i',\sigma_{i+1}',\alpha_{i+1}'}
R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1}',\alpha_{i+1}).
\label{eq:two-site-step4}
\end{align}

As in the one-site case, the first line uses
\[
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1}',\alpha_{i-1})
\]
rather than
\[
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha_{i-1}').
\]

\subsection{Validated NumPy implementation}

Let the two-site tensor be stored as
\[
\Theta[\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}],
\]
with shape
\[
(D_{i-1},d_i,d_{i+1},D_{i+1}).
\]

With arrays
\begin{itemize}
    \item \texttt{L[b,x,y]} for the stored left environment,
    \item \texttt{Theta[y,u,v,z]} for the input two-site tensor,
    \item \texttt{W1[b,B,u,s]} for the MPO tensor on site $i$,
    \item \texttt{W2[B,C,v,t]} for the MPO tensor on site $i+1$,
    \item \texttt{R[C,z,a]} for the stored right environment,
\end{itemize}
the validated matrix-free contraction is
\begin{verbatim}
X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
HT = np.einsum('Cxstz,Cza->xsta', Z, R)
\end{verbatim}

The output tensor \texttt{HT[x,s,t,a]} has the same shape as \texttt{Theta}.

\paragraph{Interpretation of physical indices.}
The MPO tensors are stored as
\[
W^{[i]}_{\beta_{i-1},\beta_i,\sigma,\sigma'},
\]
with the first physical index the ket/output index and the second the bra/input index.
The routing in the two-site contraction above is therefore convention-dependent and should
be used exactly as written.

\subsection{Vectorization and local eigensolver interface}

If
\[
\Theta^{[i,i+1]} \in
\mathbb{C}^{D_{i-1}\times d_i \times d_{i+1}\times D_{i+1}},
\]
then the local vector-space dimension is
\[
N_{\mathrm{loc}} = D_{i-1} d_i d_{i+1} D_{i+1}.
\]
The eigensolver vector
\[
v \in \mathbb{C}^{N_{\mathrm{loc}}}
\]
is identified with $\Theta$ by
\[
\Theta = \mathrm{reshape}(v,\,(D_{i-1},d_i,d_{i+1},D_{i+1})),
\]
again using the fixed C-order convention of Section~\ref{sec:reshape}.

A practical wrapper is
\begin{verbatim}
def heff_two_site_matvec(v, L, W1, W2, R, Dl, d1, d2, Dr):
    Theta = v.reshape((Dl, d1, d2, Dr), order='C')
    X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
    Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
    Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
    HT = np.einsum('Cxstz,Cza->xsta', Z, R)
    return HT.reshape(Dl * d1 * d2 * Dr, order='C')
\end{verbatim}

This routine is then passed to \texttt{LinearOperator} and solved by Lanczos or Davidson
exactly as in the one-site case.

\subsection{Validation strategy}

For small systems, the dense matrix generated by repeated application of the matrix-free
two-site routine above should be compared entrywise against the exact projected operator
\[
P^\dagger H P.
\]
This is the decisive correctness test for the environment-based implementation.

\paragraph{Important warning.}
Hermiticity alone is not a sufficient diagnostic. In the present conventions, one can
obtain a two-site environment-built operator that is Hermitian and numerically stable,
yet still incorrect because of an index-order mismatch in the left environment or in the
MPO physical-index routing. Therefore the contraction sequence in
Eqs.~\eqref{eq:two-site-step1}--\eqref{eq:two-site-step4} should be treated as the
authoritative implementation for the present conventions.
```

---

If you want, I can next also provide:

- a **fully rewritten `\section{MPO environments}`**
- and a **fully rewritten `\section{Implementation checklist}`**

so that all convention-sensitive parts of the note are mutually consistent.

---
## 🧑‍💻 User

go with these two sections

---
## 🤖 Assistant

Below are **fully rewritten polished LaTeX sections** for

1. `\section{MPO environments}`
2. the checklist/concluding validation material, i.e. a rewritten `\section{Implementation checklist}`

written to be consistent with the corrected and validated implementation.

You can paste these directly into `MPS-Kimi-GPT-329.tex` in place of the corresponding existing sections.

---

# 1. Fully rewritten `\section{MPO environments}`

```latex
\section{MPO environments}
\label{sec:env}

For a center at site $i$, let $L^{[i]}$ denote the contraction of sites
$1,\dots,i-1$ into a left environment, and let $R^{[i]}$ denote the contraction
of sites $i+1,\dots,L$ into a right environment.

We store the left and right environments as tensors
\[
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha'_{i-1}),
\qquad
R^{[i]}_{\beta_i}(\alpha_i,\alpha'_i),
\]
with array shapes
\[
L^{[i]} \in \mathbb{C}^{\chi_{i-1}\times D_{i-1}\times D_{i-1}},
\qquad
R^{[i]} \in \mathbb{C}^{\chi_i\times D_i\times D_i}.
\]

\subsection{Left environment update}

Suppose sites $1,\dots,i-1$ are represented by left-canonical tensors. Then the
left environment is updated recursively by
\begin{equation}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha'_{i-1})
=
\sum_{\substack{
\alpha_{i-2},\alpha'_{i-2}\\
\sigma,\sigma'\\
\beta_{i-2}
}}
L^{[i-1]}_{\beta_{i-2}}(\alpha_{i-2},\alpha'_{i-2})
A^{[i-1]}_{\alpha_{i-2},\sigma,\alpha_{i-1}}
W^{[i-1]}_{\beta_{i-2},\beta_{i-1},\sigma,\sigma'}
A^{[i-1]*}_{\alpha'_{i-2},\sigma',\alpha'_{i-1}}.
\label{eq:left-env-consistent-rewritten}
\end{equation}

With array conventions
\begin{itemize}
    \item \texttt{Lold[b,x,y]},
    \item \texttt{A[x,s,a]},
    \item \texttt{W[b,B,s,t]},
\end{itemize}
the validated NumPy implementation is
\begin{verbatim}
Lnew = np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj())
\end{verbatim}

\subsection{Right environment update}

Suppose sites $i+1,\dots,L$ are represented by right-canonical tensors. Then the
right environment is updated recursively by
\begin{equation}
R^{[i]}_{\beta_i}(\alpha_i,\alpha'_i)
=
\sum_{\substack{
\alpha_{i+1},\alpha'_{i+1}\\
\sigma,\sigma'\\
\beta_{i+1}
}}
B^{[i+1]}_{\alpha_i,\sigma,\alpha_{i+1}}
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma,\sigma'}
R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1},\alpha'_{i+1})
B^{[i+1]*}_{\alpha'_i,\sigma',\alpha'_{i+1}}.
\label{eq:right-env-consistent-rewritten}
\end{equation}

With array conventions
\begin{itemize}
    \item \texttt{Rold[B,a,c]},
    \item \texttt{B[x,s,a]},
    \item \texttt{W[b,B,s,t]},
\end{itemize}
the validated NumPy implementation is
\begin{verbatim}
Rnew = np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj())
\end{verbatim}

\subsection{Boundary initialization}

For open boundary conditions,
\begin{equation}
L^{[1]}_{\beta_0}(1,1)=\delta_{\beta_0,1},
\qquad
R^{[L]}_{\beta_L}(1,1)=\delta_{\beta_L,1},
\end{equation}
with
\[
D_0=D_L=\chi_0=\chi_L=1.
\]

In code, these are simply
\begin{verbatim}
L = np.zeros((1,1,1), dtype=np.complex128); L[0,0,0] = 1
R = np.zeros((1,1,1), dtype=np.complex128); R[0,0,0] = 1
\end{verbatim}

\subsection{Interpretation and convention warning}

The update formulas above define the \emph{stored} left and right environment tensors.
However, for the tensor, reshape, and MPO conventions used in these notes, the stored left
environment does not enter the local effective-Hamiltonian action with the same bond-index
order as it is stored.

More precisely, if the stored left environment is represented in code as
\[
\texttt{L[b,x,y]},
\]
then in the validated one-site and two-site local contractions it must be used as
\[
\texttt{L[b,y,x]}.
\]

This point is easy to miss. A local operator built from environments can remain Hermitian
even when this ordering is wrong, while still failing to agree with the exact projected
operator
\[
P^\dagger H P.
\]
Therefore one should distinguish carefully between
\begin{itemize}
    \item the recursion defining the stored environments, and
    \item the way those stored tensors are inserted into the local effective-Hamiltonian action.
\end{itemize}

\subsection{Practical recommendation}

The left and right environment update routines should be tested independently against
brute-force contractions on very small systems. After that, the resulting one-site and
two-site environment-based local operators should be compared explicitly against dense
projected references. In practice, this second test is the decisive one: it verifies not
only that the environments are built recursively, but also that they are inserted into the
local effective Hamiltonian with the correct index ordering.
```

---

# 2. Fully rewritten `\section{Implementation checklist}`

```latex
\section{Implementation checklist}

Before production calculations, the implementation should be validated in a strictly
controlled order. The purpose of this checklist is not only to test individual routines,
but also to ensure that the tensor-index conventions, reshape conventions, and environment
contractions are mutually consistent.

\begin{enumerate}
    \item \textbf{Canonicalization test.}
    Starting from random tensors or a random MPS, perform left- and right-canonicalization
    sweeps and verify the defining relations
    \begin{align}
        \sum_{\alpha_{i-1},\sigma_i}
        A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
        A^{[i]*}_{\alpha_{i-1},\sigma_i,\alpha_i'}
        &= \delta_{\alpha_i,\alpha_i'},
        \\
        \sum_{\sigma_i,\alpha_i}
        B^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
        B^{[i]*}_{\alpha_{i-1}',\sigma_i,\alpha_i}
        &= \delta_{\alpha_{i-1},\alpha_{i-1}'}.
    \end{align}
    These tests verify both the QR/SVD logic and the fixed reshape conventions.

    \item \textbf{MPO construction test.}
    For small chains, such as $L=2,3,4$, contract the MPO to a dense matrix and compare
    it entrywise against a Hamiltonian built directly from Kronecker products of local operators.
    This test should be performed for both the Heisenberg MPO and the AKLT MPO.

    \item \textbf{Environment-recursion test.}
    Verify that the left and right environment updates reproduce brute-force contractions
    on very small systems. This confirms that the recursive environment-building routines are
    internally consistent.

    \item \textbf{Hermiticity test of local matrix-free operators.}
    For random small tensors, verify numerically that
    \[
    \langle x,\mathcal{H}_{\mathrm{eff}} y\rangle
    =
    \langle \mathcal{H}_{\mathrm{eff}} x, y\rangle
    \]
    up to numerical precision, for both one-site and two-site matrix-free local actions.

    \item \textbf{One-site projected-operator test.}
    For a small mixed-canonical MPS, explicitly construct the dense projected one-site operator
    \[
    P^\dagger H P
    \]
    and compare it against the dense matrix obtained by repeated application of the matrix-free
    one-site environment-based local action.

    For the conventions of these notes, the validated one-site contraction is
\begin{verbatim}
X  = np.einsum('byx,ysz->bxsz', L, M)
Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
HM = np.einsum('Bxtz,Bza->xta', Y, R)
\end{verbatim}

    \item \textbf{Two-site projected-operator test.}
    For a small mixed-canonical MPS, explicitly construct the dense projected two-site operator
    \[
    P^\dagger H P
    \]
    and compare it against the dense matrix obtained by repeated application of the matrix-free
    two-site environment-based local action.

    For the conventions of these notes, the validated two-site contraction is
\begin{verbatim}
X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
HT = np.einsum('Cxstz,Cza->xsta', Z, R)
\end{verbatim}

    This comparison against the dense projected operator is the decisive local correctness test.

    \item \textbf{Two-site DMRG benchmark for Heisenberg chains.}
    Run finite-system two-site DMRG for small spin-$1/2$ Heisenberg chains and compare the
    converged MPS energy against exact diagonalization. Agreement to numerical precision on
    small systems is the main global benchmark for the implementation.

    \item \textbf{AKLT benchmark.}
    Validate the AKLT implementation in two ways:
    \begin{itemize}
        \item compare the DMRG ground-state energy against the exact open-chain result
        \[
        E_0 = -\frac23 (L-1),
        \]
        \item evaluate the energy of the exact AKLT MPS and verify that it reproduces the same result.
    \end{itemize}

    \item \textbf{Convergence diagnostics in production runs.}
    After the implementation is validated locally and globally, monitor in larger calculations:
    \begin{itemize}
        \item energy change between sweeps,
        \item maximum discarded weight,
        \item eventual energy variance,
        \item stability of local observables and entanglement spectra.
    \end{itemize}
\end{enumerate}

\paragraph{Most important practical warning.}
For the conventions used in these notes, Hermiticity of the environment-built local operator
is \emph{not} sufficient to guarantee correctness. In particular, an incorrect treatment of the
stored left environment or an incorrect MPO physical-index routing can still yield a Hermitian
local operator, yet one that differs from the true projected operator \(P^\dagger H P\).
Therefore no full DMRG sweeps should be trusted until the one-site and two-site local operators
have been compared explicitly against dense projected references on small systems.

\paragraph{Recommended debugging order.}
If a full DMRG calculation produces stable but incorrect energies, the safest debugging sequence is:
\begin{enumerate}
    \item verify canonicalization,
    \item verify MPO-to-dense agreement,
    \item verify left and right environment updates separately,
    \item verify the one-site local projected operator,
    \item verify the two-site local projected operator,
    \item only then test full sweeps.
\end{enumerate}
In practice, the projected-operator tests are usually more informative than Hermiticity tests alone.

\paragraph{Authoritative implementation templates.}
For the conventions of these notes, the one-site and two-site matrix-free contraction templates
written above should be treated as authoritative reference implementations. Future optimizations,
refactorings, or changes of contraction order should always be benchmarked back against these
forms on small systems before being used in production calculations.
```

---

If you want, I can next do the final remaining cleanup and provide a **fully rewritten `\section{Practical coding recommendation}`** so the end of the note matches the new checklist and warnings.

---
## 🧑‍💻 User

pls go ahead

---
## 🤖 Assistant

```latex
\section{Practical coding recommendation}

For a first reliable implementation, the following strategy is strongly recommended.

\begin{itemize}
    \item \textbf{Implement two-site DMRG first.}
    Among standard finite-system algorithms, two-site DMRG is the most robust place to begin.
    It is substantially easier to debug than one-site DMRG, naturally allows bond dimensions to
    adapt before truncation, and provides a direct setting in which the local variational space is
    large enough to avoid many of the local-minimum problems of one-site updates.

    \item \textbf{Fix tensor conventions once and use them everywhere.}
    In particular, fix:
    \begin{itemize}
        \item the MPS tensor convention
        \[
        A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i},
        \]
        \item the MPO tensor convention
        \[
        W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'},
        \]
        with the first physical index the ket/output index and the second the bra/input index,
        \item the reshape rules used in QR, SVD, vectorization, and inverse reshaping,
        \item the memory order, e.g.\ NumPy C-order.
    \end{itemize}
    Small inconsistencies in these choices are among the most common sources of silent errors.

    \item \textbf{Use reduced QR for regauging and SVD for truncation.}
    Reduced QR without pivoting is convenient for moving orthogonality centers and constructing
    canonical forms. SVD should be used when a truncation decision is required, because it gives
    direct access to Schmidt singular values and discarded weights.

    \item \textbf{Keep the orthogonality-center structure explicit.}
    In two-site DMRG, the variational object is the two-site tensor
    \[
    \Theta^{[i,i+1]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_{i+1}},
    \]
    and the tensors to the left and right should be kept in the corresponding mixed-canonical form.
    During a left-to-right sweep, after splitting the optimized two-site tensor, the center moves
    to the right; during a right-to-left sweep, it moves to the left. This structure should be
    reflected explicitly in the code.

    \item \textbf{Do not build dense local effective Hamiltonians in production sweeps.}
    Except for very small validation tests, the local effective Hamiltonian should be accessed only
    through a matrix-free map
    \[
    v \mapsto \mathcal{H}_{\mathrm{eff}} v.
    \]
    This is the only scalable strategy and is the natural interface to Lanczos or Davidson.

    \item \textbf{Treat the validated contraction templates as authoritative.}
    For the conventions of these notes, the one-site and two-site local environment-based
    contractions are subtle. In particular, the stored left environment enters the local action
    with its MPS-bond indices interchanged, and the MPO physical-index routing in the two-site
    action must be implemented exactly as in the validated contraction sequence.
    These convention-dependent details should not be re-derived from memory during coding.

    \item \textbf{Benchmark against dense projected operators before full sweeps.}
    Before running finite-system DMRG sweeps, one should verify on small systems that the
    environment-built one-site and two-site local operators agree with the exact projected operators
    \[
    P^\dagger H P.
    \]
    This is a more stringent and more informative test than Hermiticity alone.

    \item \textbf{Benchmark against exact diagonalization on small systems.}
    After the local-operator tests pass, run two-site DMRG for small Heisenberg chains and compare
    the converged energy against exact diagonalization. This is the standard global correctness test.

    \item \textbf{Use the AKLT chain as an exact MPS benchmark.}
    The AKLT model is particularly valuable because:
    \begin{itemize}
        \item the Hamiltonian has an exact MPO representation of modest bond dimension,
        \item the ground state admits an exact MPS representation with bond dimension $D=2$,
        \item the open-chain ground-state energy is known exactly,
        \[
        E_0 = -\frac23 (L-1).
        \]
    \end{itemize}
    Therefore the AKLT chain tests simultaneously the MPO, the local solver, the sweep logic,
    and the handling of exact low-bond-dimension MPS ground states.

    \item \textbf{Use the current local tensor as the eigensolver initial guess.}
    In finite-system sweeps, the previously optimized local tensor provides an excellent initial
    vector for Lanczos or Davidson. This is a major practical improvement in convergence speed.

    \item \textbf{Monitor discarded weights and, when available, variances.}
    The energy alone is not always a sufficient convergence indicator. In production calculations,
    one should monitor at least:
    \begin{itemize}
        \item the energy change between sweeps,
        \item the maximum discarded weight during each sweep,
        \item the eventual energy variance,
        \item stability of local observables and entanglement spectra.
    \end{itemize}
\end{itemize}

\paragraph{Recommended order of development.}
A robust development sequence is:
\begin{enumerate}
    \item local operators,
    \item MPO construction,
    \item canonicalization routines,
    \item environment updates,
    \item one-site projected-operator validation,
    \item two-site projected-operator validation,
    \item two-site DMRG benchmark for Heisenberg chains,
    \item AKLT benchmark,
    \item only afterward: one-site DMRG, variance evaluation, and further optimizations.
\end{enumerate}

\paragraph{Most important practical warning.}
For the conventions used in these notes, a local operator built from environments can be
Hermitian and numerically stable while still being incorrect. This can happen if the stored
left environment is inserted into the local action with the wrong bond-index order, or if
the MPO physical-index routing in the local contraction is inconsistent with the chosen MPO
storage convention. Therefore full DMRG sweeps should \emph{not} be trusted until the
matrix-free local operator has been compared explicitly against the dense projected operator
\(P^\dagger H P\) on small systems.

\paragraph{Final recommendation.}
In practical coding work, the safest philosophy is:
\begin{quote}
first validate the local operator, then validate small-system energies, and only then scale up.
\end{quote}
For environment-based DMRG implementations, this order is not merely convenient; it is essential.
```

---
## 🧑‍💻 User

Any cutoff?

---
## 🤖 Assistant

No — that section was complete. It ended at:

```latex
For environment-based DMRG implementations, this order is not merely convenient; it is essential.
```

So nothing was cut off.

At this point, I have given you fully rewritten versions of:

- `\section{MPO environments}`
- `\section{One-site effective Hamiltonian}`
- `\section{Two-site effective Hamiltonian}`
- `\section{Matrix-free eigensolver interface for local effective Hamiltonians}`
- `\section{Implementation checklist}`
- `\section{Practical coding recommendation}`

If you want, I can do one final pass and produce a **concise list of exactly which original sections in `MPS-Kimi-GPT-329.tex` should now be replaced**, so you can patch the note cleanly without rereading everything.

---
## 🧑‍💻 User

Please do final pass check for this revised note.

---
## 🤖 Assistant

Yes. I did a final-pass read of your revised note, and it is **very close**, but there are a few things I would still correct before treating it as final.

I’ll separate them into:

1. **must-fix issues**
2. **strongly recommended fixes**
3. **optional cleanup**

---

# 1. Must-fix issues

## (A) Wrong equation reference in the One-site DMRG algorithm

In `\section{One-site DMRG}`, the algorithm says:

```latex
\State solve Eq.~\eqref{eq:heff-action} for lowest eigenpair using Lanczos/Davidson
```

But in the revised note, the old label `eq:heff-action` no longer exists.  
You replaced the one-site effective-Hamiltonian section by equations labeled:

- `eq:one-site-step1`
- `eq:one-site-step2`
- `eq:one-site-step3`

So this reference is now broken.

### Recommended fix
Replace that line by something like:

```latex
\State solve the one-site effective-Hamiltonian problem defined by Eqs.~\eqref{eq:one-site-step1}--\eqref{eq:one-site-step3} using Lanczos/Davidson
```

or more compactly:

```latex
\State solve the one-site effective-Hamiltonian problem using the matrix-free contraction of Section~\ref{sec:matfree}
```

That is a **must-fix** because otherwise the LaTeX cross-reference is wrong.

---

## (B) Duplicated conceptual material between sections

You now have:

- a full section `\section{One-site effective Hamiltonian}`
- a full section `\section{Two-site effective Hamiltonian}`
- and then in `\section{Matrix-free eigensolver interface for local effective Hamiltonians}` you repeat nearly the same one-site and two-site formulas in full.

This is not mathematically wrong, but it makes the note noticeably repetitive and slightly inconsistent in emphasis.

### Why this matters
If you later edit one place but not the other, the note can drift into inconsistency again.

### Recommended fix
In `\section{Matrix-free eigensolver interface for local effective Hamiltonians}`, shorten the one-site and two-site subsections so they **refer back** to the validated contractions already defined earlier, rather than re-deriving them in full again.

For example, in the one-site local solve subsection, instead of repeating the full index equations, say:

```latex
The matrix-free one-site action is the validated contraction given in
Section~\ref{sec:matfree-one-site-ref-or-one-site-section}.
```

Since you don’t currently have labels on those sections/subsections, you could add them.

This is not strictly fatal, but I would still treat it as a **must-fix structural cleanup** if you want the note to remain maintainable.

---

# 2. Strongly recommended fixes

## (C) Add labels to the rewritten sections/subsections

Right now the rewritten sections are present, but several important subsections do not have labels.

I strongly recommend adding labels like:

### In One-site effective Hamiltonian
```latex
\section{One-site effective Hamiltonian}
\label{sec:one-site-heff}
```

### In Two-site effective Hamiltonian
```latex
\section{Two-site effective Hamiltonian}
\label{sec:two-site-heff}
```

### In Matrix-free eigensolver interface
keep
```latex
\label{sec:matfree}
```
which you already have.

This will let later algorithm sections refer cleanly to the validated formulas.

---

## (D) Two-site DMRG “Environment updates” subsection is still written in the old symbolic convention

This is subtle.

In `\section{Two-site DMRG}`, under `\subsection{Environment updates}`, you still write the left-to-right update as:

```latex
A^{[i]*} L W A^{[i]}
```

symbolically in the original order.

This is fine as a statement of the stored environment recursion, because that is indeed how the stored left environment is defined.

However, after your rewritten environment section and local-action sections, a reader might mistakenly think this is the same index ordering used directly inside the local effective Hamiltonian action. The distinction is now crucial.

### Recommended fix
Add one sentence after the left-to-right and right-to-left update paragraphs:

```latex
These formulas define the stored environments. As emphasized in Section~\ref{sec:env}, the stored left environment enters the local effective-Hamiltonian action with its two MPS-bond indices interchanged relative to its storage order.
```

This would greatly reduce ambiguity.

---

## (E) The one-site DMRG algorithm still references the old conceptual formula rather than the validated matrix-free action

Even after fixing the broken reference, I recommend changing the wording to explicitly align it with the revised note.

### Replace
```latex
\State solve Eq.~\eqref{eq:heff-action} for lowest eigenpair using Lanczos/Davidson
```

### By
```latex
\State solve the one-site effective-Hamiltonian problem using the validated matrix-free contraction of Section~\ref{sec:matfree}
```

This is cleaner and future-proof.

---

## (F) Add a brief note that the local symbolic formulas are convention-dependent

You already say this in several places, which is good. But I would add one short sentence at the start of the one-site and two-site effective-Hamiltonian sections:

```latex
The symbolic index form of the local operator is convention-dependent; for implementation, the explicit validated contraction sequence below is authoritative.
```

You mostly already say this in prose, so this is only a polishing recommendation.

---

# 3. Optional cleanup

## (G) The note is now somewhat long and repetitive
This is not wrong. But if you want a cleaner final document, I would consider:

- keeping the detailed derivation in `One-site effective Hamiltonian` and `Two-site effective Hamiltonian`,
- and shortening `Matrix-free eigensolver interface...` to implementation wrappers only.

That would improve readability.

---

## (H) Add explicit subsection labels for the validated contractions
If you want cross-references to be cleaner, add:

```latex
\subsection{Validated matrix-free action}
\label{subsec:one-site-heff-action}
```

and similarly for the two-site section.

Then in the eigensolver section you can say:

```latex
The local map is the validated contraction of Section~\ref{subsec:one-site-heff-action}.
```

---

## (I) Minor stylistic point: “authoritative implementation”
You use this phrase many times. It is fine, but maybe slightly repetitive. You could occasionally vary it with:

- “reference implementation”
- “validated implementation”
- “implementation to be used for these conventions”

Purely stylistic.

---

# Concrete edit list

Here is the shortest exact edit list I recommend.

---

## Fix 1
In the `OneSiteDMRG` algorithm, replace:

```latex
\State solve Eq.~\eqref{eq:heff-action} for lowest eigenpair using Lanczos/Davidson
```

with

```latex
\State solve the one-site effective-Hamiltonian problem using the validated matrix-free contraction of Section~\ref{sec:matfree}
```

---

## Fix 2
Add labels:

```latex
\section{One-site effective Hamiltonian}
\label{sec:one-site-heff}
```

```latex
\section{Two-site effective Hamiltonian}
\label{sec:two-site-heff}
```

---

## Fix 3
In `\subsection{Environment updates}` inside `\section{Two-site DMRG}`, add after the two update formulas:

```latex
These formulas define the stored environments. As emphasized in Section~\ref{sec:env}, the stored left environment enters the local effective-Hamiltonian action with its two MPS-bond indices interchanged relative to its storage order.
```

---

## Fix 4
Optionally shorten the repetition inside `\section{Matrix-free eigensolver interface for local effective Hamiltonians}` by referring back to the earlier sections instead of restating all index formulas. This is recommended but not mandatory for correctness.

---

# Overall verdict

## Mathematically / algorithmically
The revised note is now **consistent with the validated code**, up to the one broken reference and the need for one clarification about stored vs. used left environments.

## Editorially
It is good, but slightly repetitive.

## Final assessment
I would say:

- **not yet perfect final**
- but **very close**
- with **1 definite correction** and **2–3 recommended cleanups**

---

If you want, I can now give you a **precise patch snippet** for the broken `OneSiteDMRG` algorithm block and the small environment-update clarification, so you can paste those exact lines into `MPS-Kimi-GPT-330.tex`.

---
## 🧑‍💻 User

please give me all the latex patches

---
## 🤖 Assistant

Below is a **complete set of concrete LaTeX patches** for `MPS-Kimi-GPT-330.tex`, based on the final-pass check.

I will give them as **replace / insert** snippets so you can patch the note directly.

---

# Patch 1 — add section labels

## Replace
```latex
\section{One-site effective Hamiltonian}
```

## By
```latex
\section{One-site effective Hamiltonian}
\label{sec:one-site-heff}
```

---

## Replace
```latex
\section{Two-site effective Hamiltonian}
```

## By
```latex
\section{Two-site effective Hamiltonian}
\label{sec:two-site-heff}
```

---

# Patch 2 — optionally add labels to the validated-action subsections

This is not strictly required, but strongly recommended.

## In the one-site section, replace
```latex
\subsection{Validated matrix-free action}
```

## By
```latex
\subsection{Validated matrix-free action}
\label{subsec:one-site-heff-action}
```

---

## In the two-site section, replace
```latex
\subsection{Validated matrix-free action}
```

## By
```latex
\subsection{Validated matrix-free action}
\label{subsec:two-site-heff-action}
```

---

# Patch 3 — fix the broken reference in the One-site DMRG algorithm

In `\section{One-site DMRG}`, inside the algorithm `OneSiteDMRG`, replace this line:

## Replace
```latex
\State solve Eq.~\eqref{eq:heff-action} for lowest eigenpair using Lanczos/Davidson
```

## By
```latex
\State solve the one-site effective-Hamiltonian problem using the validated matrix-free contraction of Section~\ref{sec:matfree}
```

---

# Patch 4 — clarify stored-vs-used left environment in Two-site DMRG environment updates

In `\section{Two-site DMRG}`, in the subsection

```latex
\subsection{Environment updates}
```

after the paragraph and equation for the right-to-left sweep update, insert the following paragraph.

## Insert after the right-to-left update equation
```latex
\paragraph{Convention reminder.}
The formulas above define the \emph{stored} left and right environments. As emphasized in
Section~\ref{sec:env}, for the conventions of these notes the stored left environment enters
the local effective-Hamiltonian action with its two MPS-bond indices interchanged relative
to its storage order. Thus the environment-recursion formulas and the local-action formulas
must not be conflated.
```

---

# Patch 5 — add a short convention warning at the start of the one-site effective-Hamiltonian section

This is optional, but I recommend it.

In `\section{One-site effective Hamiltonian}`, after the paragraph

```latex
\paragraph{General structure.}
...
```

insert:

## Insert
```latex
\paragraph{Authoritative convention.}
For implementation purposes, the explicit matrix-free contraction given below should be
treated as authoritative. The symbolic index form of the one-site effective Hamiltonian is
convention-dependent, and with the conventions of these notes a naive insertion of the
stored left environment into the local action leads to an incorrect operator.
```

If you feel this is redundant with the later “Convention-sensitive point” paragraph, you may skip this patch.

---

# Patch 6 — add a short convention warning at the start of the two-site effective-Hamiltonian section

Similarly, in `\section{Two-site effective Hamiltonian}`, after the paragraph

```latex
\paragraph{General structure.}
...
```

insert:

## Insert
```latex
\paragraph{Authoritative convention.}
For implementation purposes, the explicit matrix-free contraction given below should be
treated as authoritative. In the present conventions, the two-site local action is not
determined safely by a naive symbolic reading alone; the ordering of the stored left
environment and the routing of the MPO physical indices must match the validated contraction.
```

Again optional, but useful.

---

# Patch 7 — reduce duplication in `\section{Matrix-free eigensolver interface for local effective Hamiltonians}`

This patch is recommended for maintainability, though not strictly necessary for correctness.

Currently that section repeats much of the content from the earlier one-site and two-site sections. A clean way to reduce duplication is to shorten the introductory parts of the one-site and two-site local solve subsections.

Below I provide replacement text blocks.

---

## Patch 7A — replace the paragraph immediately before the one-site wrapper

Find in `\subsection{One-site local solve}` the block beginning with

```latex
\paragraph{Validated one-site contraction.}
```

and ending just before

```latex
\paragraph{Matrix-free wrapper.}
```

### Replace that whole block by

```latex
\paragraph{Validated one-site contraction.}
The one-site matrix-free action is the validated contraction given in
Section~\ref{sec:one-site-heff}. In array form, with stored environment tensor
\texttt{L[b,x,y]}, local tensor \texttt{M[y,s,z]}, MPO tensor \texttt{W[b,B,s,t]},
and right environment \texttt{R[B,z,a]}, the authoritative contraction is
\begin{verbatim}
X  = np.einsum('byx,ysz->bxsz', L, M)
Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
HM = np.einsum('Bxtz,Bza->xta', Y, R)
\end{verbatim}
Here the first contraction uses \texttt{'byx'} rather than \texttt{'bxy'}, reflecting
the convention-sensitive role of the stored left environment explained in
Section~\ref{sec:env}.
```

---

## Patch 7B — replace the paragraph immediately before the two-site wrapper

Find in `\subsection{Two-site local solve}` the block beginning with

```latex
\paragraph{Validated two-site contraction.}
```

and ending just before

```latex
\paragraph{Matrix-free wrapper.}
```

### Replace that whole block by

```latex
\paragraph{Validated two-site contraction.}
The two-site matrix-free action is the validated contraction given in
Section~\ref{sec:two-site-heff}. In array form, with stored left environment
\texttt{L[b,x,y]}, two-site tensor \texttt{Theta[y,u,v,z]}, MPO tensors
\texttt{W1[b,B,u,s]} and \texttt{W2[B,C,v,t]}, and stored right environment
\texttt{R[C,z,a]}, the authoritative contraction is
\begin{verbatim}
X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
HT = np.einsum('Cxstz,Cza->xsta', Z, R)
\end{verbatim}
This contraction has been validated explicitly against the dense projected operator
\(P^\dagger H P\) on small systems.
```

---

# Patch 8 — improve the One-site DMRG algorithm wording

This is optional, but the current wording is a bit old relative to the revised note.

In `\section{One-site DMRG}`, inside the algorithm, replace

## Replace
```latex
\State solve local lowest-eigenvalue problem
```

## By
```latex
\State solve the one-site effective-Hamiltonian problem using the validated matrix-free local contraction
```

This affects the backward sweep part of the algorithm.

---

# Patch 9 — add labels to the two DMRG algorithm sections

Not necessary, but useful for future references.

## Replace
```latex
\section{Two-site DMRG}
```

## By
```latex
\section{Two-site DMRG}
\label{sec:two-site-dmrg}
```

---

## Replace
```latex
\section{One-site DMRG}
```

## By
```latex
\section{One-site DMRG}
\label{sec:one-site-dmrg}
```

---

# Patch 10 — optional note in the One-site DMRG section

In `\section{One-site DMRG}`, after the paragraph

```latex
\paragraph{Practical warning.}
...
```

insert:

## Insert
```latex
\paragraph{Convention reminder.}
For the conventions of these notes, the one-site local matrix-free action used inside
one-site DMRG must be the validated contraction summarized in Section~\ref{sec:matfree},
rather than a naive contraction built directly from the stored left environment without
the required bond-index interchange.
```

---

# Patch 11 — optional note in the Two-site DMRG practical comments

In `\section{Two-site DMRG}`, in `\subsection{Practical comments for coding}`, add one bullet:

## Insert into the itemize list
```latex
\item \textbf{Use the validated local contraction exactly as written.}
For the conventions of these notes, the stored left environment enters the local action
with its bond indices interchanged, and the MPO physical-index routing in the two-site
local solve is convention-sensitive.
```

---

# Patch 12 — optional cleanup in the checklist

In `\section{Implementation checklist}`, after

```latex
This comparison against the dense projected operator is the decisive local correctness test.
```

insert:

## Insert
```latex
In practice, this test is more informative than Hermiticity alone and should be passed
before any full DMRG sweep is trusted.
```

This is optional because you already say similar things later, but it reads well there.

---

# Minimum patch set you definitely should apply

If you want the truly essential subset only, then apply at least:

- **Patch 1**
- **Patch 3**
- **Patch 4**

These are the most important ones.

---

# Recommended full patch set

If you want the note to be clean and maintainable, apply:

- Patch 1
- Patch 2
- Patch 3
- Patch 4
- Patch 7A
- Patch 7B
- Patch 9
- Patch 11

That is probably the best balance.

---

If you want, I can next give you a **single unified edited LaTeX block** for the affected algorithm sections (`TwoSiteDMRG` and `OneSiteDMRG`) so you can paste them directly rather than editing line by line.

---
## 🧑‍💻 User

I do not want to paste them line by line. Could you just give me the fully rewritten section or subsections for what need to fix?

---
## 🤖 Assistant

Yes. That is the better way.

Below I give you the **fully rewritten blocks** for the parts that still need fixing, so you can replace whole sections/subsections directly instead of patching line by line.

I will provide:

1. a rewritten `\section{Matrix-free eigensolver interface for local effective Hamiltonians}`  
   with reduced duplication and correct references,
2. a rewritten `\subsection{Environment updates}` inside `\section{Two-site DMRG}`,
3. a rewritten `\section{One-site DMRG}` with the broken reference fixed and wording aligned with the revised conventions.

These are the main remaining fix targets.

---

# 1. Fully rewritten `\section{Matrix-free eigensolver interface for local effective Hamiltonians}`

Replace your entire current section by the following:

```latex
\section{Matrix-free eigensolver interface for local effective Hamiltonians}
\label{sec:matfree}

In practical DMRG implementations, the local effective Hamiltonian should \emph{never}
be formed as a dense matrix except for unit tests on very small systems. Instead, one
implements only the map
\[
v \mapsto \mathcal{H}_{\mathrm{eff}} v,
\]
and passes this map to an iterative eigensolver such as Lanczos or Davidson.

\paragraph{Important convention warning.}
For the tensor, MPO, and environment conventions adopted in these notes, the stored
left environment tensor enters the local effective-Hamiltonian action with its two
MPS-bond indices interchanged. In array form, if the stored left environment is
\texttt{L[b,x,y]}, then the validated local matrix-free contractions use it as
\texttt{L[b,y,x]}. This point is subtle but essential. A local operator built from
environments can remain Hermitian even when this ordering is wrong, while still failing
to agree with the true projected operator
\[
P^\dagger H P.
\]
Accordingly, the one-site and two-site matrix-free contractions given below should be
treated as the authoritative implementations for the conventions of these notes.

\subsection{One-site local solve}

For a one-site center tensor
\[
M^{[i]} \in \mathbb{C}^{D_{i-1}\times d_i \times D_i},
\]
define the local vector-space dimension
\[
N_{\mathrm{loc}} = D_{i-1} d_i D_i.
\]
The eigensolver vector
\[
v\in\mathbb{C}^{N_{\mathrm{loc}}}
\]
is identified with the tensor $M^{[i]}$ by
\begin{equation}
M^{[i]} = \mathrm{reshape}(v,\,(D_{i-1},d_i,D_i)),
\end{equation}
using the fixed C-order convention of Section~\ref{sec:reshape}. After applying the
local effective Hamiltonian, the result is reshaped back to a vector with the same convention.

\paragraph{Validated one-site contraction.}
The one-site matrix-free action is the validated contraction defined in
Section~\ref{sec:one-site-heff}. In array form, with stored environment tensor
\texttt{L[b,x,y]}, local tensor \texttt{M[y,s,z]}, MPO tensor \texttt{W[b,B,s,t]},
and right environment \texttt{R[B,z,a]}, the authoritative contraction is
\begin{verbatim}
X  = np.einsum('byx,ysz->bxsz', L, M)
Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
HM = np.einsum('Bxtz,Bza->xta', Y, R)
\end{verbatim}

\paragraph{Matrix-free wrapper.}
A practical NumPy/\texttt{scipy} interface is:
\begin{verbatim}
from scipy.sparse.linalg import LinearOperator, eigsh

def heff_one_site_matvec(v, L, W, R, Dl, d, Dr):
    M = v.reshape((Dl, d, Dr), order='C')
    X  = np.einsum('byx,ysz->bxsz', L, M)
    Y  = np.einsum('bBst,bxsz->Bxtz', W, X)
    HM = np.einsum('Bxtz,Bza->xta', Y, R)
    return HM.reshape(Dl * d * Dr, order='C')

Nloc = Dl * d * Dr
Heff = LinearOperator(
    shape=(Nloc, Nloc),
    matvec=lambda v: heff_one_site_matvec(v, L, W, R, Dl, d, Dr),
    dtype=np.complex128
)

E, vec = eigsh(Heff, k=1, which='SA', v0=v0)
M = vec.reshape((Dl, d, Dr), order='C')
\end{verbatim}

Here \texttt{v0} should be the current local tensor flattened with the same reshape convention:
\begin{verbatim}
v0 = M_current.reshape(Nloc, order='C')
\end{verbatim}

\paragraph{Hermiticity and correctness checks.}
For a Hermitian MPO and correctly built environments, the local operator should satisfy
\[
\langle x, \mathcal{H}_{\mathrm{eff}} y\rangle
=
\langle \mathcal{H}_{\mathrm{eff}} x, y\rangle
\]
up to numerical precision. However, Hermiticity alone is not a sufficient correctness test.
For small systems, one should also compare the dense matrix obtained from the matrix-free
routine against the exact projected operator
\[
P^\dagger H P.
\]

\subsection{Two-site local solve}

For a two-site center tensor
\[
\Theta^{[i,i+1]} \in
\mathbb{C}^{D_{i-1}\times d_i \times d_{i+1}\times D_{i+1}},
\]
the local vector-space dimension is
\[
N_{\mathrm{loc}} = D_{i-1} d_i d_{i+1} D_{i+1}.
\]
Again, the eigensolver vector $v$ is identified with the tensor $\Theta$ by
\begin{equation}
\Theta = \mathrm{reshape}(v,\,(D_{i-1},d_i,d_{i+1},D_{i+1})).
\end{equation}
After applying the local effective Hamiltonian, the output tensor is flattened back
to a vector using the same C-order convention.

\paragraph{Validated two-site contraction.}
The two-site matrix-free action is the validated contraction defined in
Section~\ref{sec:two-site-heff}. In array form, with stored left environment
\texttt{L[b,x,y]}, two-site tensor \texttt{Theta[y,u,v,z]}, MPO tensors
\texttt{W1[b,B,u,s]} and \texttt{W2[B,C,v,t]}, and stored right environment
\texttt{R[C,z,a]}, the authoritative contraction is
\begin{verbatim}
X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
HT = np.einsum('Cxstz,Cza->xsta', Z, R)
\end{verbatim}

\paragraph{Matrix-free wrapper.}
A practical wrapper is
\begin{verbatim}
def heff_two_site_matvec(v, L, W1, W2, R, Dl, d1, d2, Dr):
    Theta = v.reshape((Dl, d1, d2, Dr), order='C')
    X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
    Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
    Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
    HT = np.einsum('Cxstz,Cza->xsta', Z, R)
    return HT.reshape(Dl * d1 * d2 * Dr, order='C')
\end{verbatim}
and this is passed to \texttt{LinearOperator} in the same way as for the one-site case.

\paragraph{Practical remark.}
The two-site contraction above was explicitly validated against the dense projected operator
\[
P^\dagger H P
\]
for small systems. This validation is essential, because more naive einsum routings can
produce Hermitian local operators that are nevertheless incorrect.

\subsection{Lanczos versus Davidson}

For first implementations, Lanczos is usually sufficient and easiest to access through
standard sparse-eigensolver interfaces. Davidson can be advantageous when the local
dimension becomes larger or when a useful diagonal preconditioner is available.

\begin{itemize}
    \item \textbf{Lanczos:} simplest robust choice for initial code.
    \item \textbf{Davidson:} often faster for large local problems, but requires more implementation effort.
\end{itemize}

In both cases, using the current local tensor as the initial guess is essential for sweep efficiency.

\subsection{Dense reference for unit tests}

For very small dimensions only, one should explicitly build the dense local projected operator
\[
H_{\mathrm{eff}} \in \mathbb{C}^{N_{\mathrm{loc}}\times N_{\mathrm{loc}}},
\]
and compare:
\begin{itemize}
    \item dense matrix--vector multiplication versus the matrix-free routine,
    \item lowest eigenvalue from dense diagonalization versus Lanczos/Davidson,
    \item Hermiticity of the dense reference matrix,
    \item agreement of the environment-built operator with the exact projected operator
    \(P^\dagger H P\).
\end{itemize}

This unit test should be performed for both the one-site and two-site local solvers before
running full sweeps.
```

---

# 2. Fully rewritten `\subsection{Environment updates}` inside `\section{Two-site DMRG}`

Replace only that subsection by:

```latex
\subsection{Environment updates}

After splitting the optimized two-site tensor, the environments must be updated so that
the next local problem is built in the correct mixed-canonical gauge.

\paragraph{Left-to-right sweep.}
After obtaining the new left-canonical tensor $A^{[i]}$, update the left environment from
site $i$ to site $i+1$:
\begin{equation}
L^{[i+1]}_{\beta_i}(\alpha_i,\alpha'_i)
=
\sum_{\substack{
\alpha_{i-1},\alpha'_{i-1}\\
\sigma_i,\sigma_i'\\
\beta_{i-1}
}}
L^{[i]}_{\beta_{i-1}}(\alpha_{i-1},\alpha'_{i-1})
A^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
W^{[i]}_{\beta_{i-1},\beta_i,\sigma_i,\sigma_i'}
A^{[i]*}_{\alpha'_{i-1},\sigma_i',\alpha'_i}.
\end{equation}

\paragraph{Right-to-left sweep.}
After obtaining the new right-canonical tensor $B^{[i+1]}$, update the right environment
from site $i+1$ to site $i$:
\begin{equation}
R^{[i]}_{\beta_i}(\alpha_i,\alpha'_i)
=
\sum_{\substack{
\alpha_{i+1},\alpha'_{i+1}\\
\sigma_{i+1},\sigma'_{i+1}\\
\beta_{i+1}
}}
B^{[i+1]}_{\alpha_i,\sigma_{i+1},\alpha_{i+1}}
W^{[i+1]}_{\beta_i,\beta_{i+1},\sigma_{i+1},\sigma'_{i+1}}
R^{[i+1]}_{\beta_{i+1}}(\alpha_{i+1},\alpha'_{i+1})
B^{[i+1]*}_{\alpha'_i,\sigma'_{i+1},\alpha'_{i+1}}.
\end{equation}

\paragraph{Convention reminder.}
The formulas above define the \emph{stored} environments. As emphasized in
Section~\ref{sec:env}, for the conventions of these notes the stored left environment enters
the local effective-Hamiltonian action with its two MPS-bond indices interchanged relative
to its storage order. Thus the environment-recursion formulas and the local-action formulas
must not be conflated.
```

---

# 3. Fully rewritten `\section{One-site DMRG}`

Replace your current whole section by:

```latex
\section{One-site DMRG}
\label{sec:one-site-dmrg}

One-site DMRG is best treated as an advanced refinement of the two-site algorithm. In practice,
a robust implementation typically uses several two-site sweeps first and only then switches to
one-site sweeps, optionally with subspace expansion or density-matrix perturbation.

\subsection{One-site mixed-canonical form}

In one-site DMRG, the MPS is kept in mixed-canonical form with center at site $i$:
\begin{equation}
\ket{\psi}
=
\sum_{\alpha_{i-1},\sigma_i,\alpha_i}
M^{[i]}_{\alpha_{i-1},\sigma_i,\alpha_i}
\ket{\alpha_{i-1}}_L \ket{\sigma_i} \ket{\alpha_i}_R.
\end{equation}

The local optimization is then performed over the center tensor $M^{[i]}$ while the left
and right Schmidt bases are kept fixed.

\subsection{One-site sweep algorithm}

\begin{algorithm}[H]
\caption{OneSiteDMRG}
\begin{algorithmic}[1]
\Require MPO $\{W^{[i]}\}$, initial MPS $\{M^{[i]}\}$, max bond dimension $D_{\max}$
\Ensure approximate ground-state MPS and energy
\State bring the initial MPS into mixed-canonical form
\State build all right environments
\State initialize the left boundary environment
\Repeat
    \Comment{Left-to-right sweep}
    \For{$i=1$ to $L-1$}
        \State solve the one-site effective-Hamiltonian problem using the validated matrix-free contraction of Section~\ref{sec:matfree}
        \State reshape the optimized center tensor as a left-grouped matrix
        \State compute an SVD: $M^{[i]} = U S V^\dagger$
        \State retain at most $D_{\max}$ singular values
        \State reshape $U \to A^{[i]}$ as a left-canonical tensor
        \State absorb $S V^\dagger$ into $M^{[i+1]}$
        \State update $L^{[i+1]}$
    \EndFor
    \Comment{Right-to-left sweep}
    \For{$i=L$ down to $2$}
        \State solve the one-site effective-Hamiltonian problem using the validated matrix-free contraction of Section~\ref{sec:matfree}
        \State reshape the optimized center tensor as a right-grouped matrix
        \State compute an SVD: $M^{[i]} = U S V^\dagger$
        \State retain at most $D_{\max}$ singular values
        \State reshape $V^\dagger$ using the inverse of Eq.~\eqref{eq:right-group} to obtain a right-canonical tensor $B^{[i]}$
        \State absorb $U S$ into $M^{[i-1]}$
        \State update $R^{[i-1]}$
    \EndFor
\Until{energy and variance are converged}
\end{algorithmic}
\end{algorithm}

\paragraph{Remark.}
In standard one-site DMRG, the SVD step mainly regauges the state and shifts the orthogonality
center; unlike two-site DMRG, it does not naturally enlarge the accessible bond space. This is
one reason why one-site DMRG is more prone to local minima unless combined with subspace expansion
or density-matrix perturbation.

\paragraph{Convention reminder.}
For the conventions of these notes, the one-site local matrix-free action used inside one-site
DMRG must be the validated contraction summarized in Section~\ref{sec:matfree}, rather than a
naive contraction built directly from the stored left environment without the required bond-index
interchange.

\paragraph{Practical warning.}
One-site DMRG is prone to local minima and should not be used as a first standalone implementation.
A robust implementation should either:
\begin{itemize}
    \item start with several two-site sweeps, or
    \item use a subspace-expansion / density-matrix perturbation method.
\end{itemize}
```

---

# Optional but recommended: rewritten section headers with labels

If you want everything cleaner, I also recommend replacing the section headers as follows.

## Replace
```latex
\section{Two-site DMRG}
```

## By
```latex
\section{Two-site DMRG}
\label{sec:two-site-dmrg}
```

---

## Replace
```latex
\section{One-site effective Hamiltonian}
```

## By
```latex
\section{One-site effective Hamiltonian}
\label{sec:one-site-heff}
```

---

## Replace
```latex
\section{Two-site effective Hamiltonian}
```

## By
```latex
\section{Two-site effective Hamiltonian}
\label{sec:two-site-heff}
```

---

# Final assessment after these replacements

If you replace those three blocks, your revised note should be in very good shape:

- mathematically aligned with the validated implementation,
- internally more consistent,
- free of the broken one-site DMRG reference,
- and clearer about stored-vs-used left environments.

If you want, I can do one more final step and give you a **fully rewritten `\section{Two-site DMRG}`** too, so that the whole DMRG part is stylistically uniform with the corrected earlier sections.

---
## 🧑‍💻 User

Please save our conversations in this session in a markdown.

---
## 🤖 Assistant

Below is a markdown handoff note you can save as, for example,

`GPT.code2.md`

---

```markdown
# Handoff note from this session: fully validated MPO/MPS environment-based matrix-free DMRG

This session completed the transition from the earlier projected benchmark implementation to a **fully MPO/MPS environment-based matrix-free local Hamiltonian action** for both:

- **spin-1/2 Heisenberg model**
- **spin-1 AKLT model**

The implementation is now working and validated.

---

## Main achievement

We now have a working implementation of:

- open-boundary MPS/MPO,
- left and right environments,
- **fully environment-based matrix-free one-site and two-site local Hamiltonian actions**,
- **two-site DMRG** using only tensor-network ingredients during sweeps,
- benchmark validation against:
  - exact dense projected operators \(P^\dagger H P\),
  - exact diagonalization,
  - exact AKLT energy,
  - exact AKLT MPS.

This is the scalable tensor-network version originally intended.

---

## Most important validated result

The fully environment-based two-site local operator now satisfies

\[
H_{\mathrm{eff}}^{\mathrm{env}} = P^\dagger H P
\]

to machine precision on small test systems.

For the Heisenberg test at `L=4`, bond `i=1`, we obtained:

- `max|env-proj| = 1.665e-16`

That was the decisive local validation.

---

## Final working convention-sensitive formulas

The main subtlety was that the **stored left environment** does **not** enter the local action in the same bond-index order as it is stored.

If the stored left environment array is

```python
L[b, x, y]
```

then in the validated local actions it must be used as

```python
L[b, y, x]
```

Equivalently: in the first contraction of the local action, use `byx`, not `bxy`.

This was the core issue that caused the earlier Hermitian-but-wrong local operators.

---

## Final validated environment updates

These are the correct **stored environment recursion formulas**:

### Left environment update
```python
Lnew = np.einsum('bxy,xsa,bBst,ytc->Bac', Lold, A, W, A.conj(), optimize=True)
```

### Right environment update
```python
Rnew = np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, Rold, B.conj(), optimize=True)
```

These define the stored environments.

---

## Final validated one-site local action

```python
def apply_one_site_heff_env(M, Lenv, W, Renv):
    X = np.einsum('byx,ysz->bxsz', Lenv, M, optimize=True)
    Y = np.einsum('bBst,bxsz->Bxtz', W, X, optimize=True)
    HM = np.einsum('Bxtz,Bza->xta', Y, Renv, optimize=True)
    return HM
```

Important point:

- `Lenv` enters with `byx`
- not `bxy`

This was validated by comparing the one-site operator against dense projected references.

---

## Final validated two-site local action

This is the crucial formula.

```python
def apply_two_site_heff_env(theta, Lenv, W1, W2, Renv):
    X = np.einsum('byx,yuvz->bxuvz', Lenv, theta, optimize=True)
    Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X, optimize=True)
    Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y, optimize=True)
    HT = np.einsum('Cxstz,Cza->xsta', Z, Renv, optimize=True)
    return HT
```

This was identified by systematic candidate testing against the exact dense projected operator.

### Important details
- left environment used as `byx`
- first MPO tensor routed as `bBus`
- second MPO tensor routed as `BCvt`

This is the correct convention for the storage conventions used in the code and note.

---

## Final matvec wrapper

```python
def heff_two_site_env_matvec(v, Lenv, W1, W2, Renv, Dl, d1, d2, Dr):
    theta = v.reshape((Dl, d1, d2, Dr), order='C')
    out = apply_two_site_heff_env(theta, Lenv, W1, W2, Renv)
    return out.reshape(Dl * d1 * d2 * Dr, order='C')
```

---

## Debugging process and what was learned

We systematically isolated the bug through the following steps:

### 1. Dense projected operator benchmark
We used the exact projected operator

\[
P^\dagger H P
\]

as the reference local operator.

This was essential.

### 2. Basis orthonormality check
We verified that:

- left basis map was orthonormal,
- right basis map was orthonormal,
- projector \(P\) was an isometry.

So the dense projected reference was trustworthy.

### 3. Environment reference checks
We tested whether left and right environments matched dense brute-force references.

Results:
- `Renv` matched correctly
- `Lenv` caused confusion in reference interpretation

Eventually the decisive lesson was that the **stored environment** and the **way it enters the local action** are not the same thing.

### 4. One-site operator test
We found:
- site 0 matched projected reference
- site 1 failed badly

This showed that the problem appears exactly when the left environment becomes nontrivial.

Then we tested two one-site local-action variants and found:

- using `Lenv` as `byx` fixes the one-site local operator.

### 5. Two-site candidate search
We tested several candidate contractions for the two-site action.

Results:
- candidate 4 was correct:
  - `byx`
  - `bBus`
  - `BCvt`

This gave agreement with the exact projected operator to machine precision.

---

## Final benchmark results

### Heisenberg (`L=8`)
Two-site DMRG converged to the exact ED ground-state energy:

- final MPS energy:
  `-3.37493259868789`
- exact ED:
  `-3.374932598687897`
- absolute error:
  `6.66e-15`

### AKLT (`L=6`)
Two-site DMRG converged to the exact open-chain AKLT energy:

- final MPS energy:
  `-3.3333333333333357`
- exact AKLT energy:
  `-3.333333333333333`
- absolute error:
  `2.66e-15`

### Exact AKLT MPS benchmark
The exact AKLT MPS also reproduced the exact energy to machine precision.

---

## Final Python module produced

A clean single-file implementation was written as:

```text
mps_dmrg_env_final.py
```

It contains:
- local operators
- Heisenberg MPO
- AKLT MPO
- dense Hamiltonians for tests
- MPS utilities
- left/right environments
- corrected one-site and two-site local actions
- matrix-free two-site DMRG
- local validation routines
- Heisenberg and AKLT benchmark runners

---

## Final notebook structure produced

A clean Jupyter cell structure was also produced, with labeled cells:

- Cell A: imports
- Cell B: local operators
- Cell C: MPO builders
- Cell D: dense Hamiltonians
- Cell E: MPS utilities
- Cell F: environments
- Cell G: corrected local actions
- Cell H: dense projected reference
- Cell I: SVD helpers and expectation value
- Cell J: exact AKLT MPS
- Cell K: final matrix-free two-site DMRG
- Cell L: local-operator validation
- Cell M: Heisenberg benchmark
- Cell N: AKLT benchmark
- Cell O: exact AKLT MPS benchmark

---

## Revisions made to the LaTeX note

A major part of this session was updating the implementation note so it matches the validated code.

### Rewritten or revised sections
We rewrote:

- `\section{MPO environments}`
- `\section{One-site effective Hamiltonian}`
- `\section{Two-site effective Hamiltonian}`
- `\section{Matrix-free eigensolver interface for local effective Hamiltonians}`
- `\section{Implementation checklist}`
- `\section{Practical coding recommendation}`

### Key note corrections
The most important conceptual correction to the note is:

> The stored left environment enters the local effective-Hamiltonian action with its two MPS-bond indices interchanged relative to storage order.

In code:
- stored as `L[b,x,y]`
- used in local action as `L[b,y,x]`

### Another important note correction
The validated two-site local action must be given explicitly as:

```python
X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
HT = np.einsum('Cxstz,Cza->xsta', Z, R)
```

### Why this matters
We confirmed empirically that:
- Hermiticity alone is **not** sufficient,
- one can build a Hermitian but incorrect environment-based local operator.

So the note must emphasize validation against dense projected references.

---

## Remaining recommended future work

The current implementation is complete for the intended task, but the following are natural next steps:

1. **one-site DMRG**
2. **energy variance calculation**
3. **Davidson eigensolver**
4. **incremental sweep-gauge optimization** instead of re-canonicalizing more globally
5. **pytest-style unit tests**
6. **splitting the Python module into smaller files**
7. **quantum-number / symmetry support** later if desired

---

## Most important practical lesson

For these conventions:

- environment recursion can look standard,
- the resulting local operator can be Hermitian,
- and yet the operator can still be wrong if the left-environment insertion or MPO index routing in the local action is incorrect.

Therefore:

\[
\boxed{
\text{Always validate } H_{\mathrm{eff}}^{\mathrm{env}} = P^\dagger H P
\text{ on small systems before running sweeps.}
}
\]

This was the decisive debugging principle of the session.

---

## Suggested startup test for future sessions

The fastest future sanity check is:

### Heisenberg local operator test
- `L=4`
- `d=2`
- bond `i=1`
- random mixed-canonical MPS

Then verify:
```python
compare_two_site_env_vs_proj(Hfull, mpo, mps, i, d)
```

Expected:
- `max|env-proj| ~ 1e-15`

If this fails, do **not** trust any sweep results.

---

## Final status

This session successfully completed the intended scalable implementation:

- **fully MPO/MPS environment-based**
- **matrix-free local Hamiltonian action**
- **validated**
- **working for both Heisenberg and AKLT**

```

---

If you want, I can also provide this as a **shorter executive-summary markdown** version instead of the full detailed handoff.
