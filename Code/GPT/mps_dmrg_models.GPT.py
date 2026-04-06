import os
import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.sparse.linalg import LinearOperator, eigsh


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def exact_ground_energy_dense(H):
    return np.linalg.eigvalsh(H)[0].real


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
# Dense Hamiltonians
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
# MPO to dense
# ============================================================

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


def test_mpo_heisenberg():
    for L in [2, 3, 4]:
        H1 = mpo_to_dense(heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.2))
        H2 = dense_heisenberg(L, J=1.0, Jz=1.0, h=0.2)
        err = np.max(np.abs(H1 - H2))
        print(f"Heisenberg MPO test L={L}: max|diff| = {err:.3e}")


def test_mpo_aklt():
    for L in [2, 3, 4]:
        H1 = mpo_to_dense(aklt_mpo(L))
        H2 = dense_aklt(L)
        err = np.max(np.abs(H1 - H2))
        print(f"AKLT MPO test L={L}: max|diff| = {err:.3e}")


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
    mps[0] = mps[0] / nrm
    return mps


def mps_bond_dims(mps):
    return [A.shape[2] for A in mps[:-1]]


# ============================================================
# Canonicalization
# ============================================================

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


# ============================================================
# Two-site tensor utilities
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


# ============================================================
# Basis maps for projected local Hilbert space
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
    L = len(mps)
    if i == L - 2:
        return np.array([[1.0]], dtype=np.complex128)

    A = mps[i + 2]
    X = A
    for k in range(i + 3, L):
        X = np.einsum('asd,dtu->astu', X, mps[k], optimize=True)
        X = X.reshape((X.shape[0], -1, X.shape[-1]), order='C')
    return X[:, :, 0]


# ============================================================
# Dense projected local Hamiltonian
# ============================================================

def dense_projected_two_site_heff_from_full(Hfull, mps, i, d):
    UL = left_basis_map(mps, i)   # (dimL, Dl)
    UR = right_basis_map(mps, i)  # (Dr, dimR)

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
# Matrix-free projected local action
# ============================================================

def local_theta_to_full_state(theta, mps, i, d):
    UL = left_basis_map(mps, i)
    UR = right_basis_map(mps, i)

    Dl, d1, d2, Dr = theta.shape
    assert d1 == d and d2 == d

    dimL = UL.shape[0]
    dimR = UR.shape[1]

    psi = np.zeros(dimL * d * d * dimR, dtype=np.complex128)

    for a in range(Dl):
        for s in range(d):
            for t in range(d):
                for c in range(Dr):
                    amp = theta[a, s, t, c]
                    if abs(amp) < 1e-15:
                        continue
                    mid = np.zeros(d * d, dtype=np.complex128)
                    mid[s * d + t] = 1.0
                    psi += amp * np.kron(UL[:, a], np.kron(mid, UR[c, :]))

    return psi


def full_state_to_local_theta(psi_full, mps, i, d):
    UL = left_basis_map(mps, i)
    UR = right_basis_map(mps, i)

    Dl = UL.shape[1]
    Dr = UR.shape[0]
    dimL = UL.shape[0]
    dimR = UR.shape[1]

    psi4 = psi_full.reshape((dimL, d, d, dimR), order='C')
    theta = np.zeros((Dl, d, d, Dr), dtype=np.complex128)

    for a in range(Dl):
        for c in range(Dr):
            theta[a, :, :, c] = np.einsum(
                'L,LstR,R->st',
                UL[:, a].conj(),
                psi4,
                UR[c, :].conj(),
                optimize=True
            )

    return theta


def heff_two_site_projected_matvec(v, Hfull, mps, i, d):
    Dl = mps[i].shape[0]
    Dr = mps[i + 1].shape[2]

    theta = v.reshape((Dl, d, d, Dr), order='C')
    psi_full = local_theta_to_full_state(theta, mps, i, d)
    Hpsi = Hfull @ psi_full
    theta_out = full_state_to_local_theta(Hpsi, mps, i, d)
    return theta_out.reshape(-1, order='C')


def compare_projected_matvec_to_dense_heff():
    L = 4
    d = 2
    Hfull = dense_heisenberg(L)
    mps = right_canonicalize(random_mps(L, d=d, Dmax=3, seed=7))
    i = 1

    Dl = mps[i].shape[0]
    Dr = mps[i + 1].shape[2]
    N = Dl * d * d * Dr

    Heff_true = dense_projected_two_site_heff_from_full(Hfull, mps, i, d)

    Hmv = np.zeros_like(Heff_true)
    for j in range(N):
        e = np.zeros(N, dtype=np.complex128)
        e[j] = 1.0
        Hmv[:, j] = heff_two_site_projected_matvec(e, Hfull, mps, i, d)

    print("max|Hmv - Heff_true| =", np.max(np.abs(Hmv - Heff_true)))
    print("Hermiticity err      =", np.max(np.abs(Hmv - Hmv.conj().T)))


# ============================================================
# Dense-reference two-site DMRG
# ============================================================

def two_site_dmrg_dense_reference(
    Hfull,
    mpo,
    d,
    Dmax=32,
    nsweeps=6,
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

        for i in range(L - 1):
            theta0 = form_theta(mps[i], mps[i + 1])
            Dl, d1, d2, Dr = theta0.shape

            Heff = dense_projected_two_site_heff_from_full(Hfull, mps, i, d)
            Heff = 0.5 * (Heff + Heff.conj().T)

            vals, vecs = scipy_eigh(Heff)
            theta = vecs[:, 0].reshape((Dl, d1, d2, Dr), order='C')

            Anew, Bnew, S, disc = split_theta_left_to_right(theta, Dmax, svd_cutoff)
            mps[i] = Anew
            mps[i + 1] = Bnew
            discards_this_sweep.append(disc)

        mps = left_canonicalize(mps)
        mps = normalize_mps_by_state(mps)

        for i in range(L - 2, -1, -1):
            theta0 = form_theta(mps[i], mps[i + 1])
            Dl, d1, d2, Dr = theta0.shape

            Heff = dense_projected_two_site_heff_from_full(Hfull, mps, i, d)
            Heff = 0.5 * (Heff + Heff.conj().T)

            vals, vecs = scipy_eigh(Heff)
            theta = vecs[:, 0].reshape((Dl, d1, d2, Dr), order='C')

            Anew, Bnew, S, disc = split_theta_right_to_left(theta, Dmax, svd_cutoff)
            mps[i] = Anew
            mps[i + 1] = Bnew
            discards_this_sweep.append(disc)

        mps = right_canonicalize(mps)
        mps = normalize_mps_by_state(mps)

        psi = mps_to_state(mps)
        E = (np.vdot(psi, Hfull @ psi) / np.vdot(psi, psi)).real
        max_disc = max(discards_this_sweep) if discards_this_sweep else 0.0

        sweep_energies.append(E)
        sweep_discards.append(max_disc)

        if verbose:
            print(f"Sweep {sweep+1:2d}: E = {E:.12f}, max discarded weight = {max_disc:.3e}")

    return mps, np.array(sweep_energies), np.array(sweep_discards)


# ============================================================
# Projected matrix-free two-site DMRG
# ============================================================

def two_site_dmrg_matrix_free_projected(
    Hfull,
    mpo,
    d,
    Dmax=32,
    nsweeps=6,
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

        for i in range(L - 1):
            theta0 = form_theta(mps[i], mps[i + 1])
            Dl, d1, d2, Dr = theta0.shape
            Nloc = Dl * d1 * d2 * Dr

            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v, Hfull=Hfull, mps=mps, i=i, d=d:
                    heff_two_site_projected_matvec(v, Hfull, mps, i, d),
                dtype=np.complex128
            )

            v0 = theta0.reshape(-1, order='C')
            vals, vecs = eigsh(
                Heff, k=1, which='SA',
                v0=v0, tol=lanczos_tol, maxiter=lanczos_maxiter
            )
            theta = vecs[:, 0].reshape((Dl, d, d, Dr), order='C')

            Anew, Bnew, S, disc = split_theta_left_to_right(theta, Dmax, svd_cutoff)
            mps[i] = Anew
            mps[i + 1] = Bnew
            discards_this_sweep.append(disc)

        mps = left_canonicalize(mps)
        mps = normalize_mps_by_state(mps)

        for i in range(L - 2, -1, -1):
            theta0 = form_theta(mps[i], mps[i + 1])
            Dl, d1, d2, Dr = theta0.shape
            Nloc = Dl * d1 * d2 * Dr

            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v, Hfull=Hfull, mps=mps, i=i, d=d:
                    heff_two_site_projected_matvec(v, Hfull, mps, i, d),
                dtype=np.complex128
            )

            v0 = theta0.reshape(-1, order='C')
            vals, vecs = eigsh(
                Heff, k=1, which='SA',
                v0=v0, tol=lanczos_tol, maxiter=lanczos_maxiter
            )
            theta = vecs[:, 0].reshape((Dl, d, d, Dr), order='C')

            Anew, Bnew, S, disc = split_theta_right_to_left(theta, Dmax, svd_cutoff)
            mps[i] = Anew
            mps[i + 1] = Bnew
            discards_this_sweep.append(disc)

        mps = right_canonicalize(mps)
        mps = normalize_mps_by_state(mps)

        psi = mps_to_state(mps)
        E = (np.vdot(psi, Hfull @ psi) / np.vdot(psi, psi)).real
        max_disc = max(discards_this_sweep) if discards_this_sweep else 0.0

        sweep_energies.append(E)
        sweep_discards.append(max_disc)

        if verbose:
            print(f"Sweep {sweep+1:2d}: E = {E:.12f}, max discarded weight = {max_disc:.3e}")

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