# mps_dmrg.py

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import svd, eigh, LinAlgError
from scipy.sparse.linalg import LinearOperator, eigsh


# ============================================================
# General utilities
# ============================================================

FIGDIR_DEFAULT = "figureAKLT"


def ensure_figdir(figdir=FIGDIR_DEFAULT):
    os.makedirs(figdir, exist_ok=True)
    return figdir


def savefig(name, figdir=FIGDIR_DEFAULT):
    ensure_figdir(figdir)
    path = os.path.join(figdir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    print("saved:", path)


def robust_svd(M):
    try:
        U, s, Vh = svd(M, full_matrices=False)
    except LinAlgError:
        U, s, Vh = svd(M, full_matrices=False, lapack_driver="gesvd")
    idx = np.argsort(s)[::-1]
    return U[:, idx], s[idx], Vh[idx, :]


def truncate_svd(U, s, Vh, Dmax, eps=1e-12, eps_abs=1e-14):
    if len(s) == 0:
        raise ValueError("Empty singular-value list.")
    if s[0] < eps_abs:
        D = 1
    else:
        D = np.sum(s > max(eps * s[0], eps_abs))
        D = max(1, min(Dmax, D))
    discarded = np.sum(s[D:] ** 2).real
    return U[:, :D], s[:D], Vh[:D, :], discarded


# ============================================================
# Local operators
# ============================================================

def spin_half_ops(dtype=np.complex128):
    I = np.eye(2, dtype=dtype)
    Sp = np.array([[0, 1], [0, 0]], dtype=dtype)
    Sm = np.array([[0, 0], [1, 0]], dtype=dtype)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=dtype)
    Sx = 0.5 * (Sp + Sm)
    Sy = -0.5j * (Sp - Sm)
    return {"I": I, "Sp": Sp, "Sm": Sm, "Sz": Sz, "Sx": Sx, "Sy": Sy}


def spin_one_ops(dtype=np.complex128):
    I = np.eye(3, dtype=dtype)
    Sx = (1 / np.sqrt(2)) * np.array([[0, 1, 0],
                                      [1, 0, 1],
                                      [0, 1, 0]], dtype=dtype)
    Sy = (1j / np.sqrt(2)) * np.array([[0, -1, 0],
                                       [1,  0, -1],
                                       [0,  1, 0]], dtype=dtype)
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, -1]], dtype=dtype)
    return {"I": I, "Sx": Sx, "Sy": Sy, "Sz": Sz}


# ============================================================
# MPO class
# ============================================================

class MPO:
    def __init__(self, Wlist, b_left, b_right, model_name="unknown"):
        self.W = Wlist
        self.L = len(Wlist)
        self.b_left = b_left
        self.b_right = b_right
        self.model_name = model_name
        self.d = Wlist[0].shape[2]

    @staticmethod
    def heisenberg(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
        ops = spin_half_ops(dtype=dtype)
        I, Sp, Sm, Sz = ops["I"], ops["Sp"], ops["Sm"], ops["Sz"]

        Dw, d = 5, 2
        Wbulk = np.zeros((Dw, Dw, d, d), dtype=dtype)

        Wbulk[0, 0] = I
        Wbulk[1, 0] = Sp
        Wbulk[2, 0] = Sm
        Wbulk[3, 0] = Sz
        Wbulk[4, 0] = -h * Sz
        Wbulk[4, 1] = (J / 2.0) * Sm
        Wbulk[4, 2] = (J / 2.0) * Sp
        Wbulk[4, 3] = Jz * Sz
        Wbulk[4, 4] = I

        Wleft = np.zeros((1, Dw, d, d), dtype=dtype)
        Wright = np.zeros((Dw, 1, d, d), dtype=dtype)
        Wleft[0] = Wbulk[4]
        Wright[:, 0] = Wbulk[:, 0]

        Wlist = [Wleft] + [Wbulk.copy() for _ in range(1, L - 1)] + [Wright]
        return MPO(Wlist, 4, 0, model_name="Heisenberg")

    @staticmethod
    def aklt(L, dtype=np.complex128):
        ops = spin_one_ops(dtype=dtype)
        I, Sx, Sy, Sz = ops["I"], ops["Sx"], ops["Sy"], ops["Sz"]

        Svec = [Sx, Sy, Sz]
        Qvec = [Sa @ Sb for Sa in Svec for Sb in Svec]

        Dw, d = 14, 3
        Wbulk = np.zeros((Dw, Dw, d, d), dtype=dtype)

        Wbulk[0, 0] = I
        for a in range(3):
            Wbulk[1 + a, 0] = Svec[a]
        for q in range(9):
            Wbulk[4 + q, 0] = Qvec[q]
        for a in range(3):
            Wbulk[13, 1 + a] = Svec[a]
        for q in range(9):
            Wbulk[13, 4 + q] = (1 / 3.0) * Qvec[q]
        Wbulk[13, 13] = I

        Wleft = np.zeros((1, Dw, d, d), dtype=dtype)
        Wright = np.zeros((Dw, 1, d, d), dtype=dtype)
        Wleft[0] = Wbulk[13]
        Wright[:, 0] = Wbulk[:, 0]

        Wlist = [Wleft] + [Wbulk.copy() for _ in range(1, L - 1)] + [Wright]
        return MPO(Wlist, 13, 0, model_name="AKLT")

    def expectation(self, mps):
        env = np.array([[[1.0 + 0j]]], dtype=np.complex128)
        for A, W in zip(mps.tensors, self.W):
            env = np.einsum("bxy,xsX,bBst,ytY->BXY", env, A.conj(), W, A, optimize=True)
        return env[0, 0, 0]


# ============================================================
# MPS class
# ============================================================

class MPS:
    def __init__(self, tensors):
        self.tensors = [A.astype(np.complex128).copy() for A in tensors]
        self.L = len(tensors)
        self.d = tensors[0].shape[1]

    @staticmethod
    def random(L, d, Dmax, seed=1234, dtype=np.complex128):
        rng = np.random.default_rng(seed)

        dims = [1]
        for _ in range(1, L):
            dims.append(min(Dmax, dims[-1] * d))
        dims.append(1)
        for i in range(L - 1, 0, -1):
            dims[i] = min(dims[i], dims[i + 1] * d)

        tensors = []
        for i in range(L):
            Dl, Dr = dims[i], dims[i + 1]
            A = rng.normal(size=(Dl, d, Dr)) + 1j * rng.normal(size=(Dl, d, Dr))
            A = A.astype(dtype)
            A /= np.linalg.norm(A)
            tensors.append(A)
        return MPS(tensors)

    @staticmethod
    def aklt_exact(L, dtype=np.complex128):
        Aplus = np.array([[0, np.sqrt(2 / 3)],
                          [0, 0]], dtype=dtype)
        A0 = np.array([[-1 / np.sqrt(3), 0],
                       [0,  1 / np.sqrt(3)]], dtype=dtype)
        Aminus = np.array([[0, 0],
                           [-np.sqrt(2 / 3), 0]], dtype=dtype)

        Abulk = np.stack([Aplus, A0, Aminus], axis=1)  # (2,3,2)
        vL = np.array([[1.0, 0.0]], dtype=dtype)
        vR = np.array([[1.0], [0.0]], dtype=dtype)

        tensors = []
        for i in range(L):
            if i == 0:
                T = np.tensordot(vL, Abulk, axes=([1], [0]))
            elif i == L - 1:
                T = np.tensordot(Abulk, vR, axes=([2], [0]))
            else:
                T = Abulk.copy()
            tensors.append(T.astype(dtype))
        return MPS(tensors)

    def copy(self):
        return MPS([A.copy() for A in self.tensors])

    def overlap(self, other):
        env = np.array([[1.0 + 0j]], dtype=np.complex128)
        for A, B in zip(self.tensors, other.tensors):
            env = np.einsum("xy,xsX,ysY->XY", env, A.conj(), B, optimize=True)
        return env[0, 0]

    def normalize(self):
        nrm = np.sqrt(np.abs(self.overlap(self)))
        self.tensors[0] /= nrm
        return self

    def right_canonicalize(self, Dmax=None, eps=1e-12):
        out = [A.copy() for A in self.tensors]
        G = np.array([[1.0 + 0j]], dtype=np.complex128)

        for i in range(len(out) - 1, 0, -1):
            A = np.tensordot(out[i], G, axes=([2], [0]))
            Dl, d, Dr = A.shape
            M = A.reshape(Dl, d * Dr, order="C")
            U, s, Vh = robust_svd(M)
            if Dmax is not None:
                U, s, Vh, _ = truncate_svd(U, s, Vh, Dmax=Dmax, eps=eps)
            out[i] = Vh.reshape(len(s), d, Dr, order="C")
            G = U @ np.diag(s)

        out[0] = np.tensordot(out[0], G, axes=([2], [0]))
        self.tensors = out
        return self

    def left_canonicalize(self):
        out = [A.copy() for A in self.tensors]
        C = np.array([[1.0 + 0j]], dtype=np.complex128)

        for i in range(len(out) - 1):
            A = np.tensordot(C, out[i], axes=([1], [0]))
            Dl, d, Dr = A.shape
            M = A.reshape(Dl * d, Dr, order="C")
            Q, R = np.linalg.qr(M, mode="reduced")
            out[i] = Q.reshape(Dl, d, Q.shape[1], order="C")
            C = R

        out[-1] = np.tensordot(C, out[-1], axes=([1], [0]))
        self.tensors = out
        return self

    def to_statevector(self):
        psi = self.tensors[0][0, :, :]
        for i in range(1, len(self.tensors)):
            psi = np.tensordot(psi, self.tensors[i], axes=([-1], [0]))
        psi = psi[..., 0]
        return np.asarray(psi, dtype=np.complex128).reshape(-1, order="C")

    def bond_dims(self):
        return [A.shape[2] for A in self.tensors[:-1]]


# ============================================================
# DMRG engine
# ============================================================

class DMRGEngine:
    def __init__(self, mpo):
        self.mpo = mpo

    # ----------------------------
    # Environment utilities
    # ----------------------------
    def update_left_env(self, Lold, A, W):
        return np.einsum("bxy,ytY,bBst,xsX->BXY", Lold, A, W, A.conj(), optimize=True)

    def update_right_env(self, Rold, A, W):
        return np.einsum("BXY,ytY,bBst,xsX->bxy", Rold, A, W, A.conj(), optimize=True)

    def build_left_envs(self, mps):
        L_env = [None] * mps.L
        L_env[0] = np.zeros((self.mpo.W[0].shape[1], 1, 1), dtype=np.complex128)
        L_env[0][self.mpo.b_left, 0, 0] = 1.0
        for i in range(mps.L - 1):
            L_env[i + 1] = self.update_left_env(L_env[i], mps.tensors[i], self.mpo.W[i])
        return L_env

    def build_right_envs(self, mps):
        R_env = [None] * mps.L
        R_env[-1] = np.zeros((self.mpo.W[-1].shape[0], 1, 1), dtype=np.complex128)
        R_env[-1][self.mpo.b_right, 0, 0] = 1.0
        for i in range(mps.L - 1, 0, -1):
            R_env[i - 1] = self.update_right_env(R_env[i], mps.tensors[i], self.mpo.W[i])
        return R_env

    # ----------------------------
    # Single-site local solver
    # ----------------------------
    def apply_local(self, v, Lenv, W, Renv, shape):
        Dl, d, Dr = shape
        V = v.reshape((Dl, d, Dr), order="C")
        T1 = np.einsum("bxy,ytY->bxtY", Lenv, V, optimize=True)
        T2 = np.einsum("bxtY,bBst->BxsY", T1, W, optimize=True)
        Vnew = np.einsum("BxsY,BXY->xsX", T2, Renv, optimize=True)
        return Vnew.reshape(-1, order="C")

    def dense_local_hamiltonian(self, Lenv, W, Renv, shape):
        dim = np.prod(shape)
        H = np.zeros((dim, dim), dtype=np.complex128)
        for j in range(dim):
            e = np.zeros(dim, dtype=np.complex128)
            e[j] = 1.0
            H[:, j] = self.apply_local(e, Lenv, W, Renv, shape)
        return 0.5 * (H + H.conj().T)

    def solve_local(self, Lenv, W, Renv, Minit, tol=1e-10, maxiter=300, dense_cutoff=64):
        shape = Minit.shape
        dim = np.prod(shape)

        if dim <= dense_cutoff:
            H = self.dense_local_hamiltonian(Lenv, W, Renv, shape)
            vals, vecs = eigh(H)
            return vals[0].real, vecs[:, 0].reshape(shape, order="C")

        def matvec(x):
            return self.apply_local(x, Lenv, W, Renv, shape)

        Heff = LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec, dtype=np.complex128)
        v0 = Minit.reshape(-1, order="C").copy()

        try:
            vals, vecs = eigsh(Heff, k=1, which="SA", v0=v0, tol=tol, maxiter=maxiter)
            return vals[0].real, vecs[:, 0].reshape(shape, order="C")
        except Exception:
            H = self.dense_local_hamiltonian(Lenv, W, Renv, shape)
            vals, vecs = eigh(H)
            return vals[0].real, vecs[:, 0].reshape(shape, order="C")

    def move_center_right(self, M, Mnext, Dmax, eps=1e-12):
        Dl, d, Dr = M.shape
        Mmat = M.reshape(Dl * d, Dr, order="C")
        U, s, Vh = robust_svd(Mmat)
        U, s, Vh, discarded = truncate_svd(U, s, Vh, Dmax=Dmax, eps=eps)
        Aleft = U.reshape(Dl, d, len(s), order="C")
        gauge = np.diag(s) @ Vh
        new_next = np.tensordot(gauge, Mnext, axes=([1], [0]))
        return Aleft, new_next, discarded, s

    def move_center_left(self, Mprev, M, Dmax, eps=1e-12):
        Dl, d, Dr = M.shape
        Mmat = M.reshape(Dl, d * Dr, order="C")
        U, s, Vh = robust_svd(Mmat)
        U, s, Vh, discarded = truncate_svd(U, s, Vh, Dmax=Dmax, eps=eps)
        Bright = Vh.reshape(len(s), d, Dr, order="C")
        gauge = U @ np.diag(s)
        new_prev = np.tensordot(Mprev, gauge, axes=([2], [0]))
        return new_prev, Bright, discarded, s

    def single_site(self, mps, Dmax, nsweeps=10, eig_tol=1e-10, svd_eps=1e-12,
                    energy_tol=1e-9, min_sweeps=2, verbose=True):
        mps = mps.copy().right_canonicalize(Dmax=Dmax, eps=svd_eps).normalize()
        energies, trunc_hist, entropy_hist = [], [], []

        for sweep in range(nsweeps):
            R_env = self.build_right_envs(mps)

            Lenv = np.zeros((self.mpo.W[0].shape[1], 1, 1), dtype=np.complex128)
            Lenv[self.mpo.b_left, 0, 0] = 1.0

            last_E = None
            sweep_disc, sweep_S = [], []

            # right sweep
            for i in range(mps.L - 1):
                E, Mopt = self.solve_local(Lenv, self.mpo.W[i], R_env[i], mps.tensors[i], tol=eig_tol)
                last_E = E
                Aleft, new_next, disc, s = self.move_center_right(Mopt, mps.tensors[i + 1], Dmax=Dmax, eps=svd_eps)
                mps.tensors[i], mps.tensors[i + 1] = Aleft, new_next
                Lenv = self.update_left_env(Lenv, mps.tensors[i], self.mpo.W[i])
                sweep_disc.append(disc)

                p = np.abs(s) ** 2
                p = p / np.sum(p)
                p = p[p > 1e-15]
                sweep_S.append(float(-(p * np.log(p)).sum().real))

            E, Mopt = self.solve_local(Lenv, self.mpo.W[-1], R_env[-1], mps.tensors[-1], tol=eig_tol)
            mps.tensors[-1] = Mopt
            last_E = E

            # left sweep
            L_env = self.build_left_envs(mps)
            Renv = np.zeros((self.mpo.W[-1].shape[0], 1, 1), dtype=np.complex128)
            Renv[self.mpo.b_right, 0, 0] = 1.0

            for i in range(mps.L - 1, 0, -1):
                E, Mopt = self.solve_local(L_env[i], self.mpo.W[i], Renv, mps.tensors[i], tol=eig_tol)
                last_E = E
                new_prev, Bright, disc, s = self.move_center_left(mps.tensors[i - 1], Mopt, Dmax=Dmax, eps=svd_eps)
                mps.tensors[i - 1], mps.tensors[i] = new_prev, Bright
                Renv = self.update_right_env(Renv, mps.tensors[i], self.mpo.W[i])
                sweep_disc.append(disc)

                p = np.abs(s) ** 2
                p = p / np.sum(p)
                p = p[p > 1e-15]
                sweep_S.append(float(-(p * np.log(p)).sum().real))

            E, Mopt = self.solve_local(L_env[0], self.mpo.W[0], Renv, mps.tensors[0], tol=eig_tol)
            mps.tensors[0] = Mopt
            last_E = E

            mps.normalize()
            energies.append(float(last_E))
            trunc_hist.append(float(np.sum(sweep_disc).real))
            entropy_hist.append(float(max(sweep_S) if len(sweep_S) else 0.0))

            dE = abs(energies[-1] - energies[-2]) if len(energies) >= 2 else np.nan
            if verbose:
                print(f"sweep {sweep+1:2d}: E = {energies[-1]: .12f}, dE = {dE:.3e}, trunc_sum = {trunc_hist[-1]:.3e}, Smax = {entropy_hist[-1]:.6f}")

            if (sweep + 1) >= min_sweeps and len(energies) >= 2 and abs(energies[-1] - energies[-2]) < energy_tol:
                if verbose:
                    print(f"Converged early: |dE| < {energy_tol}")
                break

        return mps, np.array(energies), np.array(trunc_hist), np.array(entropy_hist)

    # ----------------------------
    # Two-site local solver
    # ----------------------------
    def merge_two_sites(self, A, B):
        return np.tensordot(A, B, axes=([2], [0]))

    def apply_two_site(self, v, Lenv, W1, W2, Renv, shape):
        Dl, d1, d2, Dr = shape
        Theta = v.reshape((Dl, d1, d2, Dr), order="C")
        T1 = np.einsum("bxy,ytuY->bxtuY", Lenv, Theta, optimize=True)
        T2 = np.einsum("bxtuY,bcst->cxsuY", T1, W1, optimize=True)
        T3 = np.einsum("cxsuY,cBvu->BxsvY", T2, W2, optimize=True)
        out = np.einsum("BxsvY,BXY->xsvX", T3, Renv, optimize=True)
        return out.reshape(-1, order="C")

    def dense_two_site_hamiltonian(self, Lenv, W1, W2, Renv, shape):
        dim = np.prod(shape)
        H = np.zeros((dim, dim), dtype=np.complex128)
        for j in range(dim):
            e = np.zeros(dim, dtype=np.complex128)
            e[j] = 1.0
            H[:, j] = self.apply_two_site(e, Lenv, W1, W2, Renv, shape)
        return 0.5 * (H + H.conj().T)

    def solve_two_site(self, Lenv, W1, W2, Renv, Theta_init, tol=1e-10, maxiter=400, dense_cutoff=128):
        shape = Theta_init.shape
        dim = np.prod(shape)

        if dim <= dense_cutoff:
            H = self.dense_two_site_hamiltonian(Lenv, W1, W2, Renv, shape)
            vals, vecs = eigh(H)
            return vals[0].real, vecs[:, 0].reshape(shape, order="C")

        def matvec(x):
            return self.apply_two_site(x, Lenv, W1, W2, Renv, shape)

        Heff = LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec, dtype=np.complex128)
        v0 = Theta_init.reshape(-1, order="C").copy()

        try:
            vals, vecs = eigsh(Heff, k=1, which="SA", v0=v0, tol=tol, maxiter=maxiter)
            return vals[0].real, vecs[:, 0].reshape(shape, order="C")
        except Exception:
            H = self.dense_two_site_hamiltonian(Lenv, W1, W2, Renv, shape)
            vals, vecs = eigh(H)
            return vals[0].real, vecs[:, 0].reshape(shape, order="C")

    def split_two_site_right(self, Theta, Dmax, eps=1e-12):
        Dl, d1, d2, Dr = Theta.shape
        M = Theta.reshape(Dl * d1, d2 * Dr, order="C")
        U, s, Vh = robust_svd(M)
        U, s, Vh, discarded = truncate_svd(U, s, Vh, Dmax=Dmax, eps=eps)
        Aleft = U.reshape(Dl, d1, len(s), order="C")
        Bright = (np.diag(s) @ Vh).reshape(len(s), d2, Dr, order="C")
        return Aleft, Bright, discarded

    def split_two_site_left(self, Theta, Dmax, eps=1e-12):
        Dl, d1, d2, Dr = Theta.shape
        M = Theta.reshape(Dl * d1, d2 * Dr, order="C")
        U, s, Vh = robust_svd(M)
        U, s, Vh, discarded = truncate_svd(U, s, Vh, Dmax=Dmax, eps=eps)
        Aleft = (U @ np.diag(s)).reshape(Dl, d1, len(s), order="C")
        Bright = Vh.reshape(len(s), d2, Dr, order="C")
        return Aleft, Bright, discarded

    def two_site(self, mps, Dmax, nsweeps=4, eig_tol=1e-10, svd_eps=1e-12,
                 energy_tol=None, min_sweeps=2, verbose=True):
        mps = mps.copy().right_canonicalize(Dmax=Dmax, eps=svd_eps).normalize()
        energies, trunc_hist = [], []

        for sweep in range(nsweeps):
            R_env = self.build_right_envs(mps)

            Lenv = np.zeros((self.mpo.W[0].shape[1], 1, 1), dtype=np.complex128)
            Lenv[self.mpo.b_left, 0, 0] = 1.0

            last_E = None
            sweep_disc = []

            # right sweep
            for i in range(mps.L - 1):
                Theta0 = self.merge_two_sites(mps.tensors[i], mps.tensors[i + 1])
                E, Theta_opt = self.solve_two_site(Lenv, self.mpo.W[i], self.mpo.W[i + 1], R_env[i + 1], Theta0, tol=eig_tol)
                last_E = E
                Aleft, Bright, disc = self.split_two_site_right(Theta_opt, Dmax=Dmax, eps=svd_eps)
                mps.tensors[i], mps.tensors[i + 1] = Aleft, Bright
                Lenv = self.update_left_env(Lenv, mps.tensors[i], self.mpo.W[i])
                sweep_disc.append(disc)

            # left sweep
            L_env = self.build_left_envs(mps)
            Renv = np.zeros((self.mpo.W[-1].shape[0], 1, 1), dtype=np.complex128)
            Renv[self.mpo.b_right, 0, 0] = 1.0

            for i in range(mps.L - 2, -1, -1):
                Theta0 = self.merge_two_sites(mps.tensors[i], mps.tensors[i + 1])
                E, Theta_opt = self.solve_two_site(L_env[i], self.mpo.W[i], self.mpo.W[i + 1], Renv, Theta0, tol=eig_tol)
                last_E = E
                Aleft, Bright, disc = self.split_two_site_left(Theta_opt, Dmax=Dmax, eps=svd_eps)
                mps.tensors[i], mps.tensors[i + 1] = Aleft, Bright
                Renv = self.update_right_env(Renv, mps.tensors[i + 1], self.mpo.W[i + 1])
                sweep_disc.append(disc)

            mps.normalize()
            energies.append(float(last_E))
            trunc_hist.append(float(np.sum(sweep_disc).real))

            dE = abs(energies[-1] - energies[-2]) if len(energies) >= 2 else np.nan
            maxbond = max(A.shape[2] for A in mps.tensors[:-1])
            if verbose:
                print(f"sweep {sweep+1:2d}: E = {energies[-1]: .12f}, dE = {dE:.3e}, trunc_sum = {trunc_hist[-1]:.3e}, maxbond = {maxbond}")

            if energy_tol is not None and (sweep + 1) >= min_sweeps and len(energies) >= 2:
                if abs(energies[-1] - energies[-2]) < energy_tol:
                    if verbose:
                        print(f"Converged early: |dE| < {energy_tol}")
                    break

        return mps, np.array(energies), np.array(trunc_hist)


# ============================================================
# Observables / diagnostics
# ============================================================

def exact_bond_entropies_from_state(mps):
    psi = mps.to_statevector()
    psi = psi / np.linalg.norm(psi)
    L, d = mps.L, mps.d
    out = []
    for cut in range(1, L):
        M = psi.reshape((d**cut, d**(L-cut)), order="C")
        s = np.linalg.svd(M, compute_uv=False)
        p = np.abs(s) ** 2
        p = p / np.sum(p)
        p = p[p > 1e-15]
        out.append(float(-(p * np.log(p)).sum().real))
    return np.array(out)


def left_norm_envs(mps):
    envs = [None] * mps.L
    env = np.array([[1.0 + 0j]], dtype=np.complex128)
    envs[0] = env
    for i in range(mps.L - 1):
        A = mps.tensors[i]
        env = np.einsum("ab,asA,bsB->AB", env, A.conj(), A, optimize=True)
        envs[i + 1] = env
    return envs


def right_norm_envs(mps):
    envs = [None] * mps.L
    env = np.array([[1.0 + 0j]], dtype=np.complex128)
    envs[-1] = env
    for i in range(mps.L - 1, 0, -1):
        A = mps.tensors[i]
        env = np.einsum("AB,asA,bsB->ab", env, A.conj(), A, optimize=True)
        envs[i - 1] = env
    return envs


def one_site_expectation(mps, op, site):
    Lenvs = left_norm_envs(mps)
    Renvs = right_norm_envs(mps)
    A = mps.tensors[site]
    return np.einsum("ab,asA,st,btB,AB->", Lenvs[site], A.conj(), op, A, Renvs[site], optimize=True)


def two_point_expectation(mps, op_i, i, op_j, j):
    assert i <= j
    Lenvs = left_norm_envs(mps)
    Renvs = right_norm_envs(mps)
    env = Lenvs[i]
    A = mps.tensors[i]
    env = np.einsum("ab,asA,st,btB->AB", env, A.conj(), op_i, A, optimize=True)
    for site in range(i + 1, j):
        A = mps.tensors[site]
        env = np.einsum("ab,asA,bsB->AB", env, A.conj(), A, optimize=True)
    A = mps.tensors[j]
    return np.einsum("ab,asA,st,btB,AB->", env, A.conj(), op_j, A, Renvs[j], optimize=True)


def connected_two_point(mps, op, i, j):
    return two_point_expectation(mps, op, i, op, j) - one_site_expectation(mps, op, i) * one_site_expectation(mps, op, j)


def string_correlator(mps, op_left, string_op, op_right, i, j):
    env = np.array([[1.0 + 0j]], dtype=np.complex128)
    for site, A in enumerate(mps.tensors):
        if site == i:
            env = np.einsum("xy,xsX,st,ytY->XY", env, A.conj(), op_left, A, optimize=True)
        elif i < site < j:
            env = np.einsum("xy,xsX,st,ytY->XY", env, A.conj(), string_op, A, optimize=True)
        elif site == j:
            env = np.einsum("xy,xsX,st,ytY->XY", env, A.conj(), op_right, A, optimize=True)
        else:
            env = np.einsum("xy,xsX,ysY->XY", env, A.conj(), A, optimize=True)
    return env[0, 0]


# ============================================================
# Exact small-system Hamiltonians / variance
# ============================================================

def kron_n(op_list):
    out = op_list[0]
    for op in op_list[1:]:
        out = np.kron(out, op)
    return out


def heisenberg_exact_hamiltonian(L, J=1.0, Jz=1.0, h=0.0, dtype=np.complex128):
    ops = spin_half_ops(dtype=dtype)
    I, Sp, Sm, Sz = ops["I"], ops["Sp"], ops["Sm"], ops["Sz"]

    H = np.zeros((2**L, 2**L), dtype=dtype)
    for i in range(L - 1):
        term_pm, term_mp, term_zz = [], [], []
        for j in range(L):
            if j == i:
                term_pm.append(Sp); term_mp.append(Sm); term_zz.append(Sz)
            elif j == i + 1:
                term_pm.append(Sm); term_mp.append(Sp); term_zz.append(Sz)
            else:
                term_pm.append(I); term_mp.append(I); term_zz.append(I)

        H += (J / 2.0) * kron_n(term_pm)
        H += (J / 2.0) * kron_n(term_mp)
        H += Jz * kron_n(term_zz)

    if abs(h) > 0:
        for i in range(L):
            term = [I] * L
            term[i] = Sz
            H += -h * kron_n(term)

    return H


def aklt_exact_hamiltonian(L, dtype=np.complex128):
    ops = spin_one_ops(dtype=dtype)
    I, Sx, Sy, Sz = ops["I"], ops["Sx"], ops["Sy"], ops["Sz"]

    SS = np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz)
    h2 = SS + (1 / 3.0) * (SS @ SS)

    H = None
    for i in range(L - 1):
        left = [I] * i
        right = [I] * (L - i - 2)
        HL = kron_n(left) if len(left) else np.array([[1.0]], dtype=dtype)
        HR = kron_n(right) if len(right) else np.array([[1.0]], dtype=dtype)
        term = np.kron(np.kron(HL, h2), HR)
        H = term if H is None else H + term
    return H


def variance_from_statevector(mps, H):
    psi = mps.to_statevector()
    psi = psi / np.linalg.norm(psi)
    E = np.vdot(psi, H @ psi).real
    Hpsi = H @ psi
    H2 = np.vdot(Hpsi, Hpsi).real
    return E, H2, H2 - E**2