# Zero-Shot-DMRG-Claude#3-Pass

## 🧑‍💻 User


This is a review article on MPS-based DMRG. Could you generate python codes according to it, by using MPS/MPO finite-DMRG implementation. Please build Python codes from scracth, adhere strictly with the review article, and use only numpy + scipy + matplotlib. Please compute both S=1/2 Heisenberg model and AKLT model using two-site DMRG algorithm, and make sure that a "scalable" matrix-free approach with LANCZOS is adapted. I work with Jupyter and prefer to fewer cells. Please show all figures and save them in directory "figureAKLT". Begin with a quick test on small systems, then go to larger systems.


---
## 🤖 Assistant




Here is a comprehensive Python implementation of two-site finite DMRG using MPS/MPO formalism with a matrix-free Lanczos eigensolver, applied to both the S=1/2 Heisenberg model and the AKLT model. The code follows the Schollwöck review article closely.

```python
# ============================================================
# Two-site finite DMRG for Heisenberg S=1/2 and AKLT S=1 models
# Following Schollwöck, Ann. Phys. 326, 96 (2011)
# Uses: numpy, scipy, matplotlib only
# Matrix-free Lanczos eigensolver for scalability
# ============================================================

import numpy as np
from scipy.linalg import svd as scipy_svd, eigh_tridiagonal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time, warnings
warnings.filterwarnings("ignore")

os.makedirs("figureAKLT", exist_ok=True)

# ----------------------------------------------------------------
# 1. LANCZOS EIGENSOLVER (matrix-free)
# ----------------------------------------------------------------
def lanczos_ground(matvec, v0, k=30, tol=1e-12):
    """
    Matrix-free Lanczos for the lowest eigenvalue/eigenvector.
    matvec: function  v -> H*v
    v0: initial vector (will be normalized)
    k: max Lanczos iterations
    tol: convergence tolerance on eigenvalue
    """
    n = v0.shape[0]
    k = min(k, n)
    v = v0 / np.linalg.norm(v0)
    alpha_list = []
    beta_list = []
    V = np.zeros((n, k+1), dtype=v0.dtype)
    V[:, 0] = v
    w = matvec(v)
    alpha = np.real(np.dot(np.conj(v), w))
    alpha_list.append(alpha)
    w = w - alpha * v
    e_old = 0.0
    for j in range(1, k):
        beta = np.linalg.norm(w)
        if beta < 1e-15:
            k = j
            break
        beta_list.append(beta)
        v_old = V[:, j-1]
        v = w / beta
        # full reorthogonalization
        for jj in range(j):
            v -= np.dot(np.conj(V[:, jj]), v) * V[:, jj]
        v /= np.linalg.norm(v)
        V[:, j] = v
        w = matvec(v)
        alpha = np.real(np.dot(np.conj(v), w))
        alpha_list.append(alpha)
        w = w - alpha * v - beta * v_old
        # check convergence
        evals, evecs = eigh_tridiagonal(
            np.array(alpha_list, dtype=np.float64),
            np.array(beta_list, dtype=np.float64)
        )
        e0 = evals[0]
        if j > 2 and abs(e0 - e_old) < tol:
            k = j + 1
            break
        e_old = e0
    alphas = np.array(alpha_list, dtype=np.float64)
    betas = np.array(beta_list, dtype=np.float64)
    evals, evecs = eigh_tridiagonal(alphas, betas)
    e0 = evals[0]
    gs = V[:, :len(alphas)] @ evecs[:, 0]
    gs /= np.linalg.norm(gs)
    return e0, gs

# ----------------------------------------------------------------
# 2. MPO CONSTRUCTION (Schollwöck Sec. 6.1)
# ----------------------------------------------------------------

def build_mpo_heisenberg(L, J=1.0, Jz=1.0, h=0.0):
    """
    Build MPO for the S=1/2 Heisenberg chain (OBC):
    H = sum_i [ (J/2)(S+_i S-_{i+1} + S-_i S+_{i+1}) + Jz Sz_i Sz_{i+1} ] - h sum_i Sz_i
    MPO bond dimension DW = 5 for bulk.
    Operator-valued matrices following Eq.(182)-(183).
    """
    d = 2
    Id = np.eye(d)
    Sp = np.array([[0, 1], [0, 0]], dtype=float)
    Sm = np.array([[0, 0], [1, 0]], dtype=float)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=float)
    DW = 5
    # bulk W[i] for 0 < i < L-1
    # W is (DW, DW, d, d)  meaning W[bL, bR, sigma, sigma']
    def make_bulk():
        W = np.zeros((DW, DW, d, d))
        W[0, 0] = Id
        W[1, 0] = Sp
        W[2, 0] = Sm
        W[3, 0] = Sz
        W[4, 0] = -h * Sz
        W[4, 1] = (J/2.0) * Sm
        W[4, 2] = (J/2.0) * Sp
        W[4, 3] = Jz * Sz
        W[4, 4] = Id
        return W

    def make_first():
        # row vector (1, DW, d, d)
        W = np.zeros((1, DW, d, d))
        W[0, 0] = -h * Sz
        W[0, 1] = (J/2.0) * Sm
        W[0, 2] = (J/2.0) * Sp
        W[0, 3] = Jz * Sz
        W[0, 4] = Id
        return W

    def make_last():
        # column vector (DW, 1, d, d)
        W = np.zeros((DW, 1, d, d))
        W[0, 0] = Id
        W[1, 0] = Sp
        W[2, 0] = Sm
        W[3, 0] = Sz
        W[4, 0] = -h * Sz
        return W

    MPO = []
    for i in range(L):
        if i == 0:
            MPO.append(make_first())
        elif i == L - 1:
            MPO.append(make_last())
        else:
            MPO.append(make_bulk())
    return MPO


def build_mpo_aklt(L):
    """
    Build MPO for the AKLT Hamiltonian (S=1, OBC):
    H = sum_i [ S_i . S_{i+1} + (1/3)(S_i . S_{i+1})^2 ]
    
    We expand (S.S)^2 in terms of local operators.
    For S=1: S.S = Sx Sx + Sy Sy + Sz Sz = (1/2)(S+ S- + S- S+) + Sz Sz
    (S.S)^2 can be expanded; we use the known result:
    H_AKLT = sum_i [ S_i.S_{i+1} + 1/3 (S_i.S_{i+1})^2 ]
           = sum_i [ 1/2 P2(i,i+1) - 1/6 ]  (projector onto S_tot=2)
    But for MPO we directly build the two-site interaction and construct MPO.
    
    Strategy: build h_{i,i+1} as a d^2 x d^2 matrix, then construct MPO.
    For the MPO, we express h = sum_k A_k otimes B_k and build the W-matrix.
    """
    d = 3  # S=1
    # S=1 spin operators in |+1>, |0>, |-1> basis
    Sp = np.array([[0, np.sqrt(2), 0],
                   [0, 0, np.sqrt(2)],
                   [0, 0, 0]], dtype=float)
    Sm = Sp.T.copy()
    Sz = np.diag([1.0, 0.0, -1.0])
    Id = np.eye(d)
    
    # S.S as a 9x9 matrix
    SdotS = (np.kron(Sp, Sm) + np.kron(Sm, Sp)) / 2.0 + np.kron(Sz, Sz)
    h_bond = SdotS + (1.0/3.0) * SdotS @ SdotS
    
    # SVD decomposition of h_bond to get A_k x B_k form
    # h_bond is (d^2, d^2), reshape to (d*d, d*d) — it already is
    U, s, Vt = np.linalg.svd(h_bond, full_matrices=False)
    # keep nonzero
    tol_svd = 1e-12
    keep = s > tol_svd
    nterms = np.sum(keep)
    U = U[:, keep]
    s = s[keep]
    Vt = Vt[keep, :]
    
    # A_k[sigma, sigma'] = U[(sigma, sigma'), k] * sqrt(s[k])
    # B_k[sigma, sigma'] = sqrt(s[k]) * Vt[k, (sigma, sigma')]
    sqrts = np.sqrt(s)
    As = []
    Bs = []
    for k in range(nterms):
        Ak = (U[:, k] * sqrts[k]).reshape(d, d)
        Bk = (sqrts[k] * Vt[k, :]).reshape(d, d)
        As.append(Ak)
        Bs.append(Bk)
    
    DW = nterms + 2  # Id at beginning and end
    
    def make_bulk():
        W = np.zeros((DW, DW, d, d))
        W[0, 0] = Id
        for k in range(nterms):
            W[k+1, 0] = As[k]
            W[DW-1, k+1] = Bs[k]
        W[DW-1, DW-1] = Id
        return W
    
    def make_first():
        W = np.zeros((1, DW, d, d))
        for k in range(nterms):
            W[0, k+1] = Bs[k]
        W[0, DW-1] = Id
        # Wait — for the first site, we need the "row vector" form.
        # Following Schollwöck Eq.(183): W[1] = [ -hSz  (J/2)S-  (J/2)S+  Jz Sz  I ]
        # For AKLT: W[1] = [ 0...  B_1 ... B_n  I ]  but with A_k on position 0 not needed
        # Actually: first row = [h_local, B_1, ..., B_n, I]
        # There's no single-site term in AKLT, so:
        W = np.zeros((1, DW, d, d))
        for k in range(nterms):
            W[0, k+1] = Bs[k]
        W[0, DW-1] = Id
        return W
    
    def make_last():
        W = np.zeros((DW, 1, d, d))
        W[0, 0] = Id
        for k in range(nterms):
            W[k+1, 0] = As[k]
        # No single site term at the end either for AKLT
        return W
    
    MPO = []
    for i in range(L):
        if i == 0:
            MPO.append(make_first())
        elif i == L - 1:
            MPO.append(make_last())
        else:
            MPO.append(make_bulk())
    return MPO


# ----------------------------------------------------------------
# 3. MPS INITIALIZATION
# ----------------------------------------------------------------

def init_mps_random(L, d, D):
    """Initialize a random right-canonical MPS."""
    tensors = []
    dims = [1]
    for i in range(L):
        dl = dims[-1]
        dr = min(dl * d, D) if i < L - 1 else 1
        # for right half, dimensions shrink
        max_dr = 1
        for j in range(L - 1, i, -1):
            max_dr = min(max_dr * d, D)
        dr = min(min(dl * d, D), max_dr) if i < L - 1 else 1
        dims.append(dr)
    # Now create random tensors and right-normalize
    tensors_raw = []
    for i in range(L):
        dl = dims[i]
        dr = dims[i + 1]
        M = np.random.randn(d, dl, dr)
        tensors_raw.append(M)
    # Right-normalize from right to left
    tensors = [None] * L
    for i in range(L - 1, 0, -1):
        M = tensors_raw[i]  # (d, dl, dr)
        d_loc, dl, dr = M.shape
        M_mat = M.reshape(d_loc * dl, dr) if i == L - 1 else M.reshape(d_loc * dl, dr)
        # Actually reshape as (dl, d*dr) for right normalization
        M_r = M.transpose(1, 0, 2).reshape(dl, d_loc * dr)  # (dl, d*dr)
        U_m, S_m, Vt_m = np.linalg.svd(M_r, full_matrices=False)
        # B = Vt_m reshaped to (d, k, dr)
        k = S_m.shape[0]
        B = Vt_m.reshape(k, d_loc, dr).transpose(1, 0, 2)  # (d, k, dr)
        tensors[i] = B
        # Absorb U*S into left neighbor
        US = U_m @ np.diag(S_m)  # (dl, k)
        M_left = tensors_raw[i - 1]  # (d, dl_left, dl)
        d_l, dl_left, _ = M_left.shape
        new_left = np.tensordot(M_left, US, axes=([2], [0]))  # (d, dl_left, k)
        tensors_raw[i - 1] = new_left
    # Site 0
    M = tensors_raw[0]
    norm = np.linalg.norm(M)
    tensors[0] = M / norm
    return tensors


# ----------------------------------------------------------------
# 4. ENVIRONMENT TENSORS L and R (Schollwöck Sec. 6.2)
# ----------------------------------------------------------------

def compute_L(A, W, L_prev):
    """
    Update left environment.
    A: MPS tensor at site i, shape (d, chi_l, chi_r)
    W: MPO tensor at site i, shape (Dw_l, Dw_r, d, d)
    L_prev: left environment, shape (chi_l, Dw_l, chi_l')
    Returns L_new: shape (chi_r, Dw_r, chi_r')
    """
    # L_prev[a, b, a'] * A[s, a, c] -> tmp[b, a', s, c

---
## 🧑‍💻 User


Just continue from where is broken

---
## 🤖 Assistant




```python
def compute_L(A, W, L_prev):
    """
    Update left environment.
    A: MPS tensor at site i, shape (d, chi_l, chi_r)
    W: MPO tensor at site i, shape (Dw_l, Dw_r, d, d)
    L_prev: left environment, shape (chi_l, Dw_l, chi_l')
    Returns L_new: shape (chi_r, Dw_r, chi_r')
    """
    # Step 1: contract L_prev with A  (sum over chi_l = a)
    # L_prev[a, b, a'] * A[s, a, c] -> [b, a', s, c]
    tmp = np.tensordot(L_prev, A, axes=([0], [1]))  # (Dw_l, chi_l', d, chi_r)
    # Step 2: contract with W  (sum over Dw_l=b, d_in=s -> d_out=s')
    # tmp[b, a', s, c] * W[b, b', s, s'] -> [a', c, b', s']
    tmp2 = np.tensordot(tmp, W, axes=([0, 2], [0, 2]))  # (chi_l', chi_r, Dw_r, d)
    # Step 3: contract with A* (sum over s'=d_out, chi_l'=a')
    # tmp2[a', c, b', s'] * A*[s', a', c'] -> [c, b', c']
    Aconj = np.conj(A)
    L_new = np.tensordot(tmp2, Aconj, axes=([0, 3], [1, 0]))  # (chi_r, Dw_r, chi_r')
    return L_new


def compute_R(B, W, R_next):
    """
    Update right environment.
    B: MPS tensor at site i, shape (d, chi_l, chi_r)
    W: MPO tensor at site i, shape (Dw_l, Dw_r, d, d)
    R_next: right environment, shape (chi_r, Dw_r, chi_r')
    Returns R_new: shape (chi_l, Dw_l, chi_l')
    """
    # Step 1: contract R_next with B (sum over chi_r)
    # R_next[a, b, a'] * B[s, c, a] -> [b, a', s, c]
    tmp = np.tensordot(R_next, B, axes=([0], [2]))  # (Dw_r, chi_r', d, chi_l)
    # Step 2: contract with W (sum over Dw_r=b, d=s)
    # tmp[b, a', s, c] * W[b', b, s', s] -> [a', c, b', s']
    tmp2 = np.tensordot(tmp, W, axes=([0, 2], [1, 3]))  # (chi_r', chi_l, Dw_l, d)
    # Step 3: contract with B* (sum over s', chi_r'=a')
    Bconj = np.conj(B)
    R_new = np.tensordot(tmp2, Bconj, axes=([0, 3], [2, 0]))  # (chi_l, Dw_l, chi_l')
    return R_new


# ----------------------------------------------------------------
# 5. EFFECTIVE HAMILTONIAN (matrix-free matvec) for two-site DMRG
# ----------------------------------------------------------------

def matvec_twosite(theta_vec, L_env, R_env, W1, W2, d, chi_l, chi_r):
    """
    Apply effective Hamiltonian to two-site wavefunction theta.
    theta: shape (d, chi_l, d, chi_r) flattened
    L_env: (chi_l, Dw_l, chi_l')
    R_env: (chi_r, Dw_r, chi_r')
    W1: (Dw_l, Dw_m, d, d) MPO left site
    W2: (Dw_m, Dw_r, d, d) MPO right site
    Returns H*theta flattened
    """
    theta = theta_vec.reshape(d, chi_l, d, chi_r)
    # Contract L_env with theta: sum over chi_l
    # L[a, bL, a'] * theta[s1, a, s2, c] -> [bL, a', s1, s2, c]
    tmp = np.tensordot(L_env, theta, axes=([0], [1]))  # (Dw_l, chi_l', d, d, chi_r)
    # Contract with W1: sum over bL, s1
    # tmp[bL, a', s1, s2, c] * W1[bL, bM, s1', s1] -> [a', s2, c, bM, s1']
    tmp2 = np.tensordot(tmp, W1, axes=([0, 2], [0, 3]))  # (chi_l', d, chi_r, Dw_m, d)
    # Contract with W2: sum over bM, s2
    # tmp2[a', s2, c, bM, s1'] * W2[bM, bR, s2', s2] -> [a', c, s1', bR, s2']
    tmp3 = np.tensordot(tmp2, W2, axes=([1, 3], [3, 0]))  # (chi_l', chi_r, d, Dw_r, d)
    # Contract with R_env: sum over c=chi_r, bR=Dw_r
    # tmp3[a', c, s1', bR, s2'] * R[c, bR, c'] -> [a', s1', s2', c']
    tmp4 = np.tensordot(tmp3, R_env, axes=([1, 3], [0, 1]))  # (chi_l', d, d, chi_r')
    # Rearrange to (d, chi_l', d, chi_r') = (s1', a', s2', c')
    result = tmp4.transpose(1, 0, 2, 3)  # (d, chi_l', d, chi_r')
    return result.ravel()


# ----------------------------------------------------------------
# 6. TWO-SITE FINITE DMRG  (Schollwöck Sec. 6.3-6.4)
# ----------------------------------------------------------------

def two_site_dmrg(MPO, L, d, D, num_sweeps=10, verbose=True):
    """
    Two-site finite DMRG.
    MPO: list of L tensors, each (Dw_l, Dw_r, d, d)
    L: number of sites
    d: local Hilbert space dimension
    D: max bond dimension
    Returns: energy, MPS tensors list, energy_history
    """
    # Initialize random right-canonical MPS
    mps = init_mps_random(L, d, D)
    
    # Build all right environments R[i] from right
    # R[i] is the environment to the right of site i (includes sites i+1 ... L-1)
    # R[L-1] = trivial (1, 1, 1) since nothing to the right
    Dw_list = [MPO[i].shape[1] for i in range(L)]
    
    R_envs = [None] * L
    # R for rightmost site
    chi_r_last = mps[L-1].shape[2]  # should be 1
    Dw_r_last = MPO[L-1].shape[1]  # should be 1
    R_envs[L-1] = np.ones((chi_r_last, Dw_r_last, chi_r_last))
    
    for i in range(L-1, 0, -1):
        R_envs[i-1] = compute_R(mps[i], MPO[i], R_envs[i])
    
    # L_env for leftmost site
    chi_l_first = mps[0].shape[1]  # should be 1
    Dw_l_first = MPO[0].shape[0]  # should be 1
    L_envs = [None] * L
    L_envs[0] = np.ones((chi_l_first, Dw_l_first, chi_l_first))
    
    energy_history = []
    
    for sweep in range(num_sweeps):
        # ---- Right sweep: sites 0,1 -> 1,2 -> ... -> L-2, L-1 ----
        for i in range(L - 1):
            j = i + 1
            d1 = mps[i].shape[0]
            chi_l = mps[i].shape[1]
            d2 = mps[j].shape[0]
            chi_r = mps[j].shape[2]
            
            # Form initial two-site tensor theta
            # theta[s1, a, s2, c] = sum_b mps[i][s1, a, b] * mps[j][s2, b, c]
            theta = np.tensordot(mps[i], mps[j], axes=([2], [1]))  # (d, chi_l, d, chi_r)
            theta_vec = theta.ravel()
            
            # Effective Hamiltonian matvec
            Le = L_envs[i]
            Re = R_envs[j]
            W1 = MPO[i]
            W2 = MPO[j]
            
            def mv(v):
                return matvec_twosite(v, Le, Re, W1, W2, d, chi_l, chi_r)
            
            # Lanczos
            E, theta_gs = lanczos_ground(mv, theta_vec, k=min(30, len(theta_vec)))
            
            # Reshape and SVD
            theta_mat = theta_gs.reshape(d * chi_l, d * chi_r)
            U, S, Vt = np.linalg.svd(theta_mat, full_matrices=False)
            
            # Truncate
            D_keep = min(D, len(S))
            U = U[:, :D_keep]
            S = S[:D_keep]
            Vt = Vt[:D_keep, :]
            
            # Normalize singular values
            norm_S = np.linalg.norm(S)
            if norm_S > 0:
                S /= norm_S
            
            # New A tensor for site i (left-normalized)
            A = U.reshape(d, chi_l, D_keep)
            mps[i] = A
            
            # New M tensor for site j
            SV = np.diag(S) @ Vt  # (D_keep, d*chi_r)
            B = SV.reshape(D_keep, d, chi_r).transpose(1, 0, 2)  # (d, D_keep, chi_r)
            mps[j] = B
            
            # Update left environment for site j
            if j < L:
                L_envs[j] = compute_L(mps[i], MPO[i], L_envs[i])
        
        energy_history.append(E)
        
        # ---- Left sweep: sites L-1,L-2 -> L-2,L-3 -> ... -> 1,0 ----
        for i in range(L - 1, 0, -1):
            j = i - 1
            d1 = mps[j].shape[0]
            chi_l = mps[j].shape[1]
            d2 = mps[i].shape[0]
            chi_r = mps[i].shape[2]
            
            theta = np.tensordot(mps[j], mps[i], axes=([2], [1]))  # (d, chi_l, d, chi_r)
            theta_vec = theta.ravel()
            
            Le = L_envs[j]
            Re = R_envs[i]
            W1 = MPO[j]
            W2 = MPO[i]
            
            def mv(v):
                return matvec_twosite(v, Le, Re, W1, W2, d, chi_l, chi_r)
            
            E, theta_gs = lanczos_ground(mv, theta_vec, k=min(30, len(theta_vec)))
            
            theta_mat = theta_gs.reshape(d * chi_l, d * chi_r)
            U, S, Vt = np.linalg.svd(theta_mat, full_matrices=False)
            
            D_keep = min(D, len(S))
            U = U[:, :D_keep]
            S = S[:D_keep]
            Vt = Vt[:D_keep, :]
            
            norm_S = np.linalg.norm(S)
            if norm_S > 0:
                S /= norm_S
            
            # New B tensor for site i (right-normalized)
            B = Vt.reshape(D_keep, d, chi_r).transpose(1, 0, 2)  # (d, D_keep, chi_r)
            mps[i] = B
            
            # New A tensor for site j
            US = U @ np.diag(S)  # (d*chi_l, D_keep)
            A = US.reshape(d, chi_l, D_keep)
            mps[j] = A
            
            # Update right environment for site j
            if j >= 0:
                R_envs[j] = compute_R(mps[i], MPO[i], R_envs[i])
        
        energy_history.append(E)
        
        if verbose:
            print(f"  Sweep {sweep+1}/{num_sweeps}, E = {E:.12f}")
    
    return E, mps, energy_history


# ----------------------------------------------------------------
# 7. MEASUREMENT FUNCTIONS
# ----------------------------------------------------------------

def measure_local(mps, op, site):
    """Measure <psi| op_site |psi> for a normalized MPS in mixed-canonical form.
    Here we just do the full contraction."""
    L = len(mps)
    d = mps[0].shape[0]
    # Build transfer from left
    chi = mps[0].shape[1]  # should be 1
    C = np.eye(chi)  # (chi, chi)
    for i in range(L):
        M = mps[i]  # (d, chi_l, chi_r)
        d_i, chi_l, chi_r = M.shape
        if i == site:
            # C[a, a'] * M[s, a, b] * op[s, s'] * M*[s', a', b'] -> C_new[b, b']
            # Step: tmp = C @ M for each s
            C_new = np.zeros((chi_r, chi_r), dtype=complex)
            for s in range(d_i):
                for sp in range(d_i):
                    C_new += op[s, sp] * (np.conj(M[sp]).T @ C @ M[s])
            C = C_new
        else:
            C_new = np.zeros((chi_r, chi_r), dtype=complex)
            for s in range(d_i):
                C_new += np.conj(M[s]).T @ C @ M[s]
            C = C_new
    return np.real(C.item())


def measure_correlator(mps, opA, siteA, opB, siteB):
    """Measure <psi| opA_siteA * op

---
## 🧑‍💻 User


continue with from where is broken. Do not give me figure part unless request.

---
## 🤖 Assistant




```python
def measure_correlator(mps, opA, siteA, opB, siteB):
    """Measure <psi| opA_siteA * opB_siteB |psi> for a normalized MPS."""
    L = len(mps)
    d = mps[0].shape[0]
    chi = mps[0].shape[1]
    C = np.eye(chi)
    for i in range(L):
        M = mps[i]
        d_i, chi_l, chi_r = M.shape
        C_new = np.zeros((chi_r, chi_r), dtype=complex)
        if i == siteA or i == siteB:
            op = opA if i == siteA else opB
            for s in range(d_i):
                for sp in range(d_i):
                    if abs(op[s, sp]) > 1e-15:
                        C_new += op[s, sp] * (np.conj(M[sp]).T @ C @ M[s])
        else:
            for s in range(d_i):
                C_new += np.conj(M[s]).T @ C @ M[s]
        C = C_new
    return np.real(C.item())


def measure_bond_energy(mps, MPO):
    """Measure <psi|H|psi> using MPO contraction."""
    L = len(mps)
    chi = mps[0].shape[1]
    Dw = MPO[0].shape[0]
    # F[a, b, a'] starting as (1, 1, 1) dummy
    F = np.ones((chi, Dw, chi))
    for i in range(L):
        F = compute_L(mps[i], MPO[i], F)
    return np.real(F.item())


def compute_entanglement_entropy(mps, bond):
    """
    Compute von Neumann entanglement entropy at a given bond.
    We left-canonicalize up to 'bond', then SVD to get singular values.
    For simplicity, we contract the two-site tensor at (bond, bond+1) and SVD.
    """
    L = len(mps)
    # Make a copy and left-canonicalize up to bond
    tensors = [m.copy() for m in mps]
    for i in range(bond):
        M = tensors[i]
        d_i, chi_l, chi_r = M.shape
        M_mat = M.reshape(d_i * chi_l, chi_r)
        U, S, Vt = np.linalg.svd(M_mat, full_matrices=False)
        tensors[i] = U.reshape(d_i, chi_l, -1)
        SV = np.diag(S) @ Vt
        # Absorb into next
        M_next = tensors[i + 1]
        d_next, chi_l_next, chi_r_next = M_next.shape
        M_next_mat = M_next.reshape(d_next, chi_l_next, chi_r_next)
        new_next = np.tensordot(SV, M_next_mat, axes=([1], [1]))  # (k, d, chi_r)
        tensors[i + 1] = new_next.transpose(1, 0, 2)
    
    # Now SVD at bond
    M = tensors[bond]
    d_i, chi_l, chi_r = M.shape
    M_mat = M.reshape(d_i * chi_l, chi_r)
    U, S, Vt = np.linalg.svd(M_mat, full_matrices=False)
    S = S[S > 1e-16]
    S2 = S ** 2
    S2 = S2 / np.sum(S2)
    entropy = -np.sum(S2 * np.log2(S2 + 1e-30))
    return entropy


def compute_Sz_profile(mps, S_val=0.5):
    """Compute <Sz_i> for each site."""
    L = len(mps)
    if S_val == 0.5:
        d = 2
        Sz = np.array([[0.5, 0], [0, -0.5]])
    else:
        d = int(2 * S_val + 1)
        Sz = np.diag(np.arange(S_val, -S_val - 0.5, -1.0)[:d])
    result = []
    for i in range(L):
        val = measure_local(mps, Sz, i)
        result.append(val)
    return np.array(result)


# ----------------------------------------------------------------
# 8. EXACT DIAGONALIZATION FOR SMALL SYSTEMS (for benchmarking)
# ----------------------------------------------------------------

def exact_diag_heisenberg(L, J=1.0, Jz=1.0, h=0.0):
    """Exact diag for S=1/2 Heisenberg chain (OBC), returns ground state energy."""
    d = 2
    Sp = np.array([[0, 1], [0, 0]], dtype=float)
    Sm = np.array([[0, 0], [1, 0]], dtype=float)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=float)
    Id = np.eye(d)
    
    def kron_op(op, site, L):
        ops = [Id] * L
        ops[site] = op
        result = ops[0]
        for k in range(1, L):
            result = np.kron(result, ops[k])
        return result
    
    dim = d ** L
    H = np.zeros((dim, dim))
    for i in range(L - 1):
        H += (J / 2.0) * (kron_op(Sp, i, L) @ kron_op(Sm, i + 1, L) +
                           kron_op(Sm, i, L) @ kron_op(Sp, i + 1, L))
        H += Jz * kron_op(Sz, i, L) @ kron_op(Sz, i + 1, L)
    for i in range(L):
        H -= h * kron_op(Sz, i, L)
    
    from scipy.linalg import eigvalsh
    evals = eigvalsh(H)
    return evals[0]


def exact_diag_aklt(L):
    """Exact diag for AKLT chain (S=1, OBC), returns ground state energy."""
    d = 3
    Sp = np.array([[0, np.sqrt(2), 0],
                   [0, 0, np.sqrt(2)],
                   [0, 0, 0]], dtype=float)
    Sm = Sp.T.copy()
    Sz = np.diag([1.0, 0.0, -1.0])
    Id = np.eye(d)
    
    SdotS_2site = (np.kron(Sp, Sm) + np.kron(Sm, Sp)) / 2.0 + np.kron(Sz, Sz)
    h_aklt_2site = SdotS_2site + (1.0 / 3.0) * SdotS_2site @ SdotS_2site
    
    dim = d ** L
    H = np.zeros((dim, dim))
    for i in range(L - 1):
        # embed h_aklt_2site on sites i, i+1
        if i == 0:
            h_full = np.kron(h_aklt_2site, np.eye(d ** (L - 2)))
        elif i == L - 2:
            h_full = np.kron(np.eye(d ** (L - 2)), h_aklt_2site)
        else:
            h_full = np.kron(np.kron(np.eye(d ** i), h_aklt_2site), np.eye(d ** (L - 2 - i)))
        H += h_full
    
    from scipy.linalg import eigvalsh
    evals = eigvalsh(H)
    return evals[0]


# ================================================================
# 9. RUN: QUICK TESTS ON SMALL SYSTEMS
# ================================================================
print("=" * 70)
print("   TWO-SITE DMRG WITH MPS/MPO AND MATRIX-FREE LANCZOS")
print("   Following Schollwöck, Ann. Phys. 326, 96 (2011)")
print("=" * 70)

# ---- 9a. S=1/2 Heisenberg, small systems ----
print("\n" + "=" * 70)
print("  QUICK TEST: S=1/2 HEISENBERG MODEL (small systems)")
print("=" * 70)

heisenberg_test_results = []
for L_test in [6, 8, 10]:
    D_test = min(32, 2 ** (L_test // 2))
    MPO_h = build_mpo_heisenberg(L_test, J=1.0, Jz=1.0, h=0.0)
    E_dmrg, mps_h, ehist = two_site_dmrg(MPO_h, L_test, d=2, D=D_test, num_sweeps=8, verbose=False)
    E_exact = exact_diag_heisenberg(L_test) if L_test <= 14 else None
    E_per_site = E_dmrg / L_test
    err = abs(E_dmrg - E_exact) if E_exact is not None else None
    heisenberg_test_results.append((L_test, D_test, E_dmrg, E_exact, err))
    if E_exact is not None:
        print(f"  L={L_test:3d}, D={D_test:3d}, E_DMRG={E_dmrg:14.10f}, "
              f"E_exact={E_exact:14.10f}, |err|={err:.2e}")
    else:
        print(f"  L={L_test:3d}, D={D_test:3d}, E_DMRG={E_dmrg:14.10f}")

# ---- 9b. AKLT model, small systems ----
print("\n" + "=" * 70)
print("  QUICK TEST: AKLT MODEL S=1 (small systems)")
print("=" * 70)

aklt_test_results = []
for L_test in [4, 6, 8]:
    D_test = min(32, 3 ** (L_test // 2))
    MPO_a = build_mpo_aklt(L_test)
    E_dmrg, mps_a, ehist = two_site_dmrg(MPO_a, L_test, d=3, D=D_test, num_sweeps=10, verbose=False)
    E_exact = exact_diag_aklt(L_test) if L_test <= 10 else None
    err = abs(E_dmrg - E_exact) if E_exact is not None else None
    aklt_test_results.append((L_test, D_test, E_dmrg, E_exact, err))
    if E_exact is not None:
        print(f"  L={L_test:3d}, D={D_test:3d}, E_DMRG={E_dmrg:14.10f}, "
              f"E_exact={E_exact:14.10f}, |err|={err:.2e}")
    else:
        print(f"  L={L_test:3d}, D={D_test:3d}, E_DMRG={E_dmrg:14.10f}")


# ================================================================
# 10. LARGER SYSTEMS
# ================================================================

# ---- 10a. S=1/2 Heisenberg, larger systems ----
print("\n" + "=" * 70)
print("  LARGER SYSTEM: S=1/2 HEISENBERG MODEL")
print("=" * 70)

L_large_h = 64
D_large_h = 64
num_sweeps_h = 12

print(f"  Running L={L_large_h}, D={D_large_h}, sweeps={num_sweeps_h}...")
t0 = time.time()
MPO_h_large = build_mpo_heisenberg(L_large_h, J=1.0, Jz=1.0, h=0.0)
E_h_large, mps_h_large, ehist_h_large = two_site_dmrg(
    MPO_h_large, L_large_h, d=2, D=D_large_h, num_sweeps=num_sweeps_h, verbose=True)
t_h = time.time() - t0
E_per_site_h = E_h_large / L_large_h
# Bethe ansatz: E/L -> 1/4 - ln(2) = -0.443147... for L->inf
E_bethe = 0.25 - np.log(2)
print(f"  E_DMRG = {E_h_large:.10f},  E/L = {E_per_site_h:.10f}")
print(f"  Bethe ansatz E/L (L->inf) = {E_bethe:.10f}")
print(f"  Time: {t_h:.1f} s")

# Measure Sz profile
Sz_profile_h = compute_Sz_profile(mps_h_large, S_val=0.5)

# Measure entanglement entropy
ee_h = []
for bond in range(1, L_large_h):
    ee_h.append(compute_entanglement_entropy(mps_h_large, bond))
ee_h = np.array(ee_h)

# Measure <Sz_i Sz_j> correlator from center
Sz_op = np.array([[0.5, 0], [0, -0.5]])
center = L_large_h // 2
corr_zz_h = []
dists_h = []
for r in range(1, L_large_h // 2):
    j = center + r
    if j < L_large_h:
        c = measure_correlator(mps_h_large, Sz_op, center, Sz_op, j)
        sz_c = measure_local(mps_h_large, Sz_op, center)
        sz_j = measure_local(mps_h_large, Sz_op, j)
        corr_zz_h.append(c - sz_c * sz_j)
        dists_h.append(r)
dists_h = np.array(dists_h)
corr_zz_h = np.array(corr_zz_h)

# Convergence with D
print("\n  Convergence with bond dimension D (L=32 Heisenberg):")
L_conv = 32
D_vals = [8, 16, 32, 48, 64]
E_vs_D_h = []
for Dv in D_vals:
    MPO_tmp = build_mpo_heisenberg(L_conv)
    E_tmp, _, _ = two_site_dmrg(MPO_tmp, L_conv, d=2, D=Dv, num_sweeps=8, verbose=False)
    E_vs_D_h.append(E_tmp)
    print(f"    D={Dv:3d}, E={E_tmp:.10f}, E/L={E_tmp/L_conv:.10f}")
E_vs_D_h = np.array(E_vs_D_h)


# ---- 10b. AKLT model, larger systems ----
print("\n" + "=" * 70)
print

---
## 🧑‍💻 User


======================================================================
   TWO-SITE DMRG WITH MPS/MPO AND MATRIX-FREE LANCZOS
   Following Schollwöck, Ann. Phys. 326, 96 (2011)
======================================================================

======================================================================
  QUICK TEST: S=1/2 HEISENBERG MODEL (small systems)
======================================================================
  L=  6, D=  8, E_DMRG= -2.4910703637, E_exact= -2.4935771339, |err|=2.51e-03
  L=  8, D= 16, E_DMRG= -3.3694601349, E_exact= -3.3749325987, |err|=5.47e-03
  L= 10, D= 32, E_DMRG= -4.2503720731, E_exact= -4.2580352073, |err|=7.66e-03

======================================================================
  QUICK TEST: AKLT MODEL S=1 (small systems)
======================================================================
  L=  4, D=  9, E_DMRG= -3.6225336154, E_exact= -2.0000000000, |err|=1.62e+00
  L=  6, D= 27, E_DMRG= -5.6047641052, E_exact= -3.3333333333, |err|=2.27e+00
  L=  8, D= 32, E_DMRG= -7.5959911260, E_exact= -4.6666666667, |err|=2.93e+00

======================================================================
  LARGER SYSTEM: S=1/2 HEISENBERG MODEL
======================================================================
  Running L=64, D=64, sweeps=12...
  Sweep 1/12, E = -28.058642624651
  Sweep 2/12, E = -28.054912633721
  Sweep 3/12, E = -28.053626179493
  Sweep 4/12, E = -28.055273370634
  Sweep 5/12, E = -28.055458063222

---
## 🤖 Assistant




I can see several issues in the output:

1. **Heisenberg model**: Energies are not converging to exact values (errors ~10⁻³ even for small systems)
2. **AKLT model**: Energies are completely wrong (much lower than exact), indicating the MPO construction is incorrect

Let me fix the key bugs and continue from where the code broke. The main issues are:

1. **MPO construction for AKLT**: The first/last site W-matrices had wrong operator placement
2. **Lanczos initial vector**: Poor starting guess
3. **Environment contraction order**: Index contractions were inconsistent

```python
# ================================================================
# COMPLETE FIXED IMPLEMENTATION - Two-site finite DMRG
# S=1/2 Heisenberg + AKLT models
# ================================================================

import numpy as np
from scipy.linalg import svd as scipy_svd, eigh_tridiagonal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time, warnings
warnings.filterwarnings("ignore")

os.makedirs("figureAKLT", exist_ok=True)

# ----------------------------------------------------------------
# 1. LANCZOS EIGENSOLVER (matrix-free, with reorthogonalization)
# ----------------------------------------------------------------
def lanczos_ground(matvec, v0, k=40, tol=1e-12):
    n = v0.shape[0]
    k = min(k, n)
    v = v0.copy().astype(np.float64)
    nrm = np.linalg.norm(v)
    if nrm < 1e-14:
        v = np.random.randn(n)
        nrm = np.linalg.norm(v)
    v /= nrm
    alphas = []
    betas = []
    V = np.zeros((n, k), dtype=np.float64)
    V[:, 0] = v
    w = matvec(v)
    alpha = np.dot(v, w)
    alphas.append(alpha)
    w = w - alpha * v
    e_old = alpha
    for j in range(1, k):
        beta = np.linalg.norm(w)
        if beta < 1e-14:
            break
        betas.append(beta)
        v_new = w / beta
        # Full reorthogonalization
        for jj in range(j + 1):
            if jj < V.shape[1]:
                v_new -= np.dot(V[:, jj], v_new) * V[:, jj]
        nrm2 = np.linalg.norm(v_new)
        if nrm2 < 1e-14:
            break
        v_new /= nrm2
        V[:, j] = v_new
        w = matvec(v_new)
        alpha = np.dot(v_new, w)
        alphas.append(alpha)
        w = w - alpha * v_new - beta * V[:, j - 1]
        # Check convergence
        if len(betas) > 0:
            evals, evecs = eigh_tridiagonal(
                np.array(alphas), np.array(betas))
            e0 = evals[0]
            if j > 3 and abs(e0 - e_old) < tol:
                break
            e_old = e0
    alphas = np.array(alphas)
    betas = np.array(betas)
    if len(betas) == 0:
        return alphas[0], V[:, 0]
    evals, evecs = eigh_tridiagonal(alphas, betas)
    e0 = evals[0]
    gs = V[:, :len(alphas)] @ evecs[:, 0]
    gs /= np.linalg.norm(gs)
    return e0, gs

# ----------------------------------------------------------------
# 2. MPO CONSTRUCTION
# ----------------------------------------------------------------
def build_mpo_heisenberg(L, J=1.0, Jz=1.0, h=0.0):
    """
    MPO for S=1/2 Heisenberg chain (OBC):
    H = sum_i (J/2)(S+S- + S-S+) + Jz Sz Sz - h sum Sz
    W-matrix follows Schollwöck Eq.(182)-(183).
    W[bL, bR, sigma, sigma'] with DW=5.
    Convention: row=bL (left bond), col=bR (right bond)
    """
    d = 2
    Id = np.eye(d)
    Sp = np.array([[0, 1], [0, 0]], dtype=float)
    Sm = np.array([[0, 0], [1, 0]], dtype=float)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=float)
    DW = 5
    
    def make_bulk():
        W = np.zeros((DW, DW, d, d))
        W[0, 0] = Id          # 1->1: I
        W[1, 0] = Sp          # 2->1: S+
        W[2, 0] = Sm          # 3->1: S-
        W[3, 0] = Sz          # 4->1: Sz
        W[4, 0] = -h * Sz     # 5->1: -hSz
        W[4, 1] = (J/2.) * Sm # 5->2: (J/2)S-
        W[4, 2] = (J/2.) * Sp # 5->3: (J/2)S+
        W[4, 3] = Jz * Sz     # 5->4: Jz*Sz
        W[4, 4] = Id          # 5->5: I
        return W

    def make_first():
        W = np.zeros((1, DW, d, d))
        W[0, 0] = -h * Sz
        W[0, 1] = (J/2.) * Sm
        W[0, 2] = (J/2.) * Sp
        W[0, 3] = Jz * Sz
        W[0, 4] = Id
        return W

    def make_last():
        W = np.zeros((DW, 1, d, d))
        W[0, 0] = Id
        W[1, 0] = Sp
        W[2, 0] = Sm
        W[3, 0] = Sz
        W[4, 0] = -h * Sz
        return W

    MPO = []
    for i in range(L):
        if i == 0:
            MPO.append(make_first())
        elif i == L - 1:
            MPO.append(make_last())
        else:
            MPO.append(make_bulk())
    return MPO


def build_mpo_aklt(L):
    """
    MPO for AKLT Hamiltonian (S=1, OBC):
    H = sum_i [ S_i . S_{i+1} + (1/3)(S_i . S_{i+1})^2 ]
    
    We build the two-site bond operator, SVD it to get A_k x B_k terms,
    then construct the MPO W-matrices.
    """
    d = 3
    Sp = np.array([[0, np.sqrt(2), 0],
                   [0, 0, np.sqrt(2)],
                   [0, 0, 0]], dtype=float)
    Sm = Sp.T.copy()
    Sz = np.diag([1.0, 0.0, -1.0])
    Id = np.eye(d)
    
    # Build two-site bond operator
    SdotS = (np.kron(Sp, Sm) + np.kron(Sm, Sp)) / 2.0 + np.kron(Sz, Sz)
    h_bond = SdotS + (1.0/3.0) * SdotS @ SdotS
    
    # SVD to decompose into sum of tensor products
    U, s, Vt = np.linalg.svd(h_bond, full_matrices=False)
    keep = s > 1e-12
    nterms = int(np.sum(keep))
    U = U[:, keep]
    s = s[keep]
    Vt = Vt[keep, :]
    
    sqrts = np.sqrt(s)
    # A_k acts on left site, B_k on right site
    # h_bond = sum_k A_k tensor B_k
    # where A_k[s,s'] = U[(s*d+s'), k] * sqrt(s_k)
    # and   B_k[s,s'] = sqrt(s_k) * Vt[k, (s*d+s')]
    # Note: h_bond[(s1,s2),(s1',s2')] = sum_k A_k[s1,s1'] * B_k[s2,s2']
    
    Ak_list = []
    Bk_list = []
    for k in range(nterms):
        Ak = (U[:, k] * sqrts[k]).reshape(d, d)  # (s1, s1')
        Bk = (sqrts[k] * Vt[k, :]).reshape(d, d)  # (s2, s2')
        Ak_list.append(Ak)
        Bk_list.append(Bk)
    
    DW = nterms + 2  # state 0=Id (start), states 1..nterms=intermediate, state DW-1=Id (end)
    
    # Bulk W-matrix: Schollwöck convention
    # Row 0 (state "not started"): W[0,0]=Id, W[k+1,0]=A_k
    # Row DW-1 (state "completed"): W[DW-1,k+1]=B_k, W[DW-1,DW-1]=Id
    # All other entries zero
    def make_bulk():
        W = np.zeros((DW, DW, d, d))
        W[0, 0] = Id
        for k in range(nterms):
            W[k+1, 0] = Ak_list[k]     # deposit A_k
            W[DW-1, k+1] = Bk_list[k]  # complete with B_k
        W[DW-1, DW-1] = Id
        return W
    
    def make_first():
        # Row vector: (1, DW)
        W = np.zeros((1, DW, d, d))
        for k in range(nterms):
            W[0, k+1] = Ak_list[k]
        W[0, DW-1] = Id
        return W
    
    def make_last():
        # Column vector: (DW, 1)
        W = np.zeros((DW, 1, d, d))
        W[0, 0] = Id
        for k in range(nterms):
            W[k+1, 0] = Bk_list[k]
        return W
    
    MPO = []
    for i in range(L):
        if i == 0:
            MPO.append(make_first())
        elif i == L - 1:
            MPO.append(make_last())
        else:
            MPO.append(make_bulk())
    return MPO


# ----------------------------------------------------------------
# 3. MPS INITIALIZATION (right-canonical random)
# ----------------------------------------------------------------
def init_mps_random(L, d, D):
    """Initialize a random right-canonical MPS with correct bond dimensions."""
    # Compute bond dimensions
    bond_dims = [1]
    for i in range(L):
        bond_dims.append(min(bond_dims[-1] * d, D))
    # Also limit from right
    bond_dims_r = [1]
    for i in range(L - 1, -1, -1):
        bond_dims_r.append(min(bond_dims_r[-1] * d, D))
    bond_dims_r.reverse()
    for i in range(L + 1):
        bond_dims[i] = min(bond_dims[i], bond_dims_r[i])
    
    tensors = []
    for i in range(L):
        dl = bond_dims[i]
        dr = bond_dims[i + 1]
        M = np.random.randn(d, dl, dr)
        tensors.append(M)
    
    # Right-normalize from right to left
    for i in range(L - 1, 0, -1):
        M = tensors[i]  # (d, chi_l, chi_r)
        d_i, chi_l, chi_r = M.shape
        # Reshape: group (d, chi_r) as columns -> (chi_l, d*chi_r)
        # Then QR from right: M^T = Q R -> M = R^T Q^T
        M_mat = M.transpose(1, 0, 2).reshape(chi_l, d_i * chi_r)
        # SVD: M_mat = U S Vt
        U, S, Vt = np.linalg.svd(M_mat, full_matrices=False)
        k = len(S)
        # B = Vt reshaped: (k, d*chi_r) -> (k, d, chi_r) -> (d, k, chi_r)
        B = Vt.reshape(k, d_i, chi_r).transpose(1, 0, 2)
        tensors[i] = B
        # Absorb U*S into left neighbor
        US = U @ np.diag(S)  # (chi_l, k)
        M_left = tensors[i - 1]  # (d, chi_ll, chi_l)
        d_l, chi_ll, _ = M_left.shape
        # new_left[s, a, b] = sum_c M_left[s, a, c] * US[c, b]
        new_left = np.tensordot(M_left, US, axes=([2], [0]))
        tensors[i - 1] = new_left
    
    # Normalize site 0
    M = tensors[0]
    nrm = np.linalg.norm(M)
    if nrm > 1e-15:
        tensors[0] = M / nrm
    return tensors


# ----------------------------------------------------------------
# 4. ENVIRONMENT TENSORS L and R
# ----------------------------------------------------------------
def compute_L_env(A, W, L_prev):
    """
    Update left environment tensor.
    A: (d, chi_l, chi_r)
    W: (Dw_l, Dw_r, d, d)  -- W[bL, bR, s_out, s_in]
    L_prev: (chi_l, Dw_l, chi_l)
    Returns: (chi_r, Dw_r, chi_r)
    
    F[a', b', a'_conj] = sum_{a, b, s, s', a_conj}
        A*[s', a_conj, a'_conj] * W[b, b', s', s] * L[a, b, a_conj] * A[s, a, a']
    
    Efficient order:
    1) contract L with A over a:  tmp1[b, a_conj, s, a'] = L[a, b, a_conj] * A[s, a, a']
    2) contract tmp1 with W over b, s:  tmp2[a_conj, a', b', s'] = tmp1[b, a_conj, s, a'] * W[b, b', s', s]
    3) contract tmp2 with A* over

---
## 🧑‍💻 User


continue from where is broken

---
## 🤖 Assistant




```python
def compute_L_env(A, W, L_prev):
    """
    Update left environment tensor.
    A: (d, chi_l, chi_r)
    W: (Dw_l, Dw_r, d, d)  -- W[bL, bR, s_phys, s_phys']
    L_prev: (chi_l, Dw_l, chi_l_conj)
    Returns: (chi_r, Dw_r, chi_r_conj)
    
    Optimal contraction order (Schollwöck Eq. 195):
    1) tmp1[b, ac, s, ar] = L[al, b, ac] * A[s, al, ar]
    2) tmp2[ac, ar, b', s'] = tmp1[b, ac, s, ar] * W[b, b', s', s]
    3) F[ar, b', arc] = tmp2[ac, ar, b', s'] * A*[s', ac, arc]
    """
    # 1) Contract L_prev with A over chi_l index
    # L_prev: (chi_l, Dw_l, chi_l_c) ; A: (d, chi_l, chi_r)
    tmp1 = np.tensordot(L_prev, A, axes=([0], [1]))
    # tmp1: (Dw_l, chi_l_c, d, chi_r)
    
    # 2) Contract with W over (Dw_l, d_in)
    # W: (Dw_l, Dw_r, d_out, d_in)
    # tmp1: (Dw_l, chi_l_c, d, chi_r) -- here d is the "ket" physical index (s)
    # We need to contract Dw_l and the physical "ket" index
    # tmp1 indices: (b, ac, s, ar); W indices: (b, b', s', s)
    # contract over b(axis0 of tmp1, axis0 of W) and s(axis2 of tmp1, axis3 of W)
    tmp2 = np.tensordot(tmp1, W, axes=([0, 2], [0, 3]))
    # tmp2: (chi_l_c, chi_r, Dw_r, d_out)
    
    # 3) Contract with A* over (chi_l_c, d_out)
    Ac = np.conj(A)
    # Ac: (d, chi_l_c, chi_r_c)
    # tmp2: (chi_l_c, chi_r, Dw_r, d_out)
    # contract chi_l_c (axis0 of tmp2, axis1 of Ac) and d_out (axis3 of tmp2, axis0 of Ac)
    F = np.tensordot(tmp2, Ac, axes=([0, 3], [1, 0]))
    # F: (chi_r, Dw_r, chi_r_c)
    return F


def compute_R_env(B, W, R_next):
    """
    Update right environment tensor.
    B: (d, chi_l, chi_r)
    W: (Dw_l, Dw_r, d, d)
    R_next: (chi_r, Dw_r, chi_r_conj)
    Returns: (chi_l, Dw_l, chi_l_conj)
    
    Optimal contraction order:
    1) tmp1[b, ac, s, al] = R[ar, b, ac] * B[s, al, ar]
    2) tmp2[ac, al, b', s'] = tmp1[b, ac, s, al] * W[b', b, s', s]
    3) G[al, b', alc] = tmp2[ac, al, b', s'] * B*[s', alc, ac]
    """
    # 1) Contract R_next with B over chi_r
    # R_next: (chi_r, Dw_r, chi_r_c); B: (d, chi_l, chi_r)
    tmp1 = np.tensordot(R_next, B, axes=([0], [2]))
    # tmp1: (Dw_r, chi_r_c, d, chi_l)
    
    # 2) Contract with W over (Dw_r, d_ket)
    # W: (Dw_l, Dw_r, d_out, d_in)
    # contract Dw_r (axis0 of tmp1, axis1 of W) and d (axis2 of tmp1, axis3 of W)
    tmp2 = np.tensordot(tmp1, W, axes=([0, 2], [1, 3]))
    # tmp2: (chi_r_c, chi_l, Dw_l, d_out)
    
    # 3) Contract with B* over (chi_r_c, d_out)
    Bc = np.conj(B)
    # Bc: (d, chi_l_c, chi_r_c)
    # contract chi_r_c (axis0 of tmp2, axis2 of Bc) and d_out (axis3 of tmp2, axis0 of Bc)
    G = np.tensordot(tmp2, Bc, axes=([0, 3], [2, 0]))
    # G: (chi_l, Dw_l, chi_l_c)
    return G


# ----------------------------------------------------------------
# 5. TWO-SITE EFFECTIVE HAMILTONIAN (matrix-free matvec)
# ----------------------------------------------------------------
def apply_Heff_twosite(theta_vec, L_env, R_env, W1, W2, shape):
    """
    Apply effective two-site Hamiltonian to theta.
    theta shape: (d1, chi_l, d2, chi_r)
    L_env: (chi_l, Dw_l, chi_l_c)
    R_env: (chi_r, Dw_r, chi_r_c)
    W1: (Dw_l, Dw_m, d, d)   (left site MPO)
    W2: (Dw_m, Dw_r, d, d)   (right site MPO)
    
    Contraction (Schollwöck Eq. 199, inside out):
    1) tmp1[bL, ac, s1, s2, ar] = L[al, bL, ac] * theta[s1, al, s2, ar]
    2) tmp2[ac, s2, ar, bM, s1'] = tmp1[bL, ac, s1, s2, ar] * W1[bL, bM, s1', s1]
    3) tmp3[ac, ar, bR, s1', s2'] = tmp2[ac, s2, ar, bM, s1'] * W2[bM, bR, s2', s2]
    4) result[s1', ac2, s2', ar2] = tmp3[ac, ar, bR, s1', s2'] * R[ar, bR, ac2]
       but we need to be careful: ac->ac2 maps chi_l_c, ar->ar2 maps chi_r_c
    """
    d1, chi_l, d2, chi_r = shape
    theta = theta_vec.reshape(shape)
    
    # 1) L_env[al, bL, ac] * theta[s1, al, s2, ar] -> contract al
    tmp1 = np.tensordot(L_env, theta, axes=([0], [1]))
    # tmp1: (Dw_l, chi_lc, d1, d2, chi_r)
    
    # 2) contract with W1 over (Dw_l=bL, d1=s1_in)
    # W1: (Dw_l, Dw_m, d, d) = (bL, bM, s1_out, s1_in)
    tmp2 = np.tensordot(tmp1, W1, axes=([0, 2], [0, 3]))
    # tmp2: (chi_lc, d2, chi_r, Dw_m, d1_out)
    
    # 3) contract with W2 over (d2=s2_in, Dw_m=bM)
    # W2: (Dw_m, Dw_r, d, d) = (bM, bR, s2_out, s2_in)
    # tmp2 axes: 1=d2(s2_in), 3=Dw_m(bM)
    tmp3 = np.tensordot(tmp2, W2, axes=([1, 3], [3, 0]))
    # tmp3: (chi_lc, chi_r, d1_out, Dw_r, d2_out)
    
    # 4) contract with R_env over (chi_r, Dw_r)
    # R_env: (chi_r, Dw_r, chi_rc)
    tmp4 = np.tensordot(tmp3, R_env, axes=([1, 3], [0, 1]))
    # tmp4: (chi_lc, d1_out, d2_out, chi_rc)
    
    # Rearrange to (d1_out, chi_lc, d2_out, chi_rc)
    result = tmp4.transpose(1, 0, 2, 3)
    return result.ravel()


# ----------------------------------------------------------------
# 6. TWO-SITE FINITE DMRG
# ----------------------------------------------------------------
def two_site_dmrg(MPO, L, d, D, num_sweeps=10, verbose=True):
    """
    Two-site finite DMRG algorithm.
    Returns: (energy, mps_tensors, energy_history)
    """
    mps = init_mps_random(L, d, D)
    
    # Build right environments from R
    R_envs = [None] * L
    chi_last = mps[L-1].shape[2]
    Dw_last = MPO[L-1].shape[1]
    R_envs[L-1] = np.ones((chi_last, Dw_last, chi_last))
    for i in range(L - 1, 0, -1):
        R_envs[i-1] = compute_R_env(mps[i], MPO[i], R_envs[i])
    
    # Initialize left environment
    L_envs = [None] * L
    chi_first = mps[0].shape[1]
    Dw_first = MPO[0].shape[0]
    L_envs[0] = np.ones((chi_first, Dw_first, chi_first))
    
    energy_history = []
    E = 0.0
    
    for sweep in range(num_sweeps):
        # --- Right sweep: i = 0..L-2 ---
        for i in range(L - 1):
            j = i + 1
            chi_l = mps[i].shape[1]
            chi_r = mps[j].shape[2]
            d1 = mps[i].shape[0]
            d2 = mps[j].shape[0]
            
            # Two-site theta
            theta = np.tensordot(mps[i], mps[j], axes=([2], [1]))
            # theta: (d1, chi_l, d2, chi_r)
            theta_shape = (d1, chi_l, d2, chi_r)
            theta_vec = theta.ravel()
            
            Le = L_envs[i]
            Re = R_envs[j]
            W1 = MPO[i]
            W2 = MPO[j]
            
            def mv(v, _Le=Le, _Re=Re, _W1=W1, _W2=W2, _s=theta_shape):
                return apply_Heff_twosite(v, _Le, _Re, _W1, _W2, _s)
            
            E, theta_gs = lanczos_ground(mv, theta_vec, k=min(40, len(theta_vec)))
            
            # SVD and truncate
            theta_mat = theta_gs.reshape(d1 * chi_l, d2 * chi_r)
            U, S, Vt = np.linalg.svd(theta_mat, full_matrices=False)
            
            D_keep = min(D, np.sum(S > 1e-14))
            D_keep = max(D_keep, 1)
            D_keep = min(D_keep, D)
            
            U = U[:, :D_keep]
            S = S[:D_keep]
            Vt = Vt[:D_keep, :]
            
            # Normalize
            S /= np.linalg.norm(S)
            
            # A = U reshaped
            mps[i] = U.reshape(d1, chi_l, D_keep)
            # B = S Vt reshaped
            SV = np.diag(S) @ Vt
            mps[j] = SV.reshape(D_keep, d2, chi_r).transpose(1, 0, 2)
            
            # Update L environment
            if j < L:
                L_envs[j] = compute_L_env(mps[i], MPO[i], L_envs[i])
        
        # --- Left sweep: i = L-1 down to 1 ---
        for i in range(L - 1, 0, -1):
            j = i - 1
            chi_l = mps[j].shape[1]
            chi_r = mps[i].shape[2]
            d1 = mps[j].shape[0]
            d2 = mps[i].shape[0]
            
            theta = np.tensordot(mps[j], mps[i], axes=([2], [1]))
            theta_shape = (d1, chi_l, d2, chi_r)
            theta_vec = theta.ravel()
            
            Le = L_envs[j]
            Re = R_envs[i]
            W1 = MPO[j]
            W2 = MPO[i]
            
            def mv(v, _Le=Le, _Re=Re, _W1=W1, _W2=W2, _s=theta_shape):
                return apply_Heff_twosite(v, _Le, _Re, _W1, _W2, _s)
            
            E, theta_gs = lanczos_ground(mv, theta_vec, k=min(40, len(theta_vec)))
            
            theta_mat = theta_gs.reshape(d1 * chi_l, d2 * chi_r)
            U, S, Vt = np.linalg.svd(theta_mat, full_matrices=False)
            
            D_keep = min(D, np.sum(S > 1e-14))
            D_keep = max(D_keep, 1)
            D_keep = min(D_keep, D)
            
            U = U[:, :D_keep]
            S = S[:D_keep]
            Vt = Vt[:D_keep, :]
            
            S /= np.linalg.norm(S)
            
            # B = Vt reshaped (right-normalized)
            mps[i] = Vt.reshape(D_keep, d2, chi_r).transpose(1, 0, 2)
            # A = U S reshaped
            US = U @ np.diag(S)
            mps[j] = US.reshape(d1, chi_l, D_keep)
            
            # Update R environment
            if j >= 0:
                R_envs[j]

---
## 🧑‍💻 User


continue

---
## 🤖 Assistant




```python
            # Update R environment
            if j >= 0:
                R_envs[j] = compute_R_env(mps[i], MPO[i], R_envs[i])
        
        energy_history.append(E)
        if verbose:
            print(f"  Sweep {sweep+1}/{num_sweeps}, E = {E:.12f}")
    
    return E, mps, energy_history


# ----------------------------------------------------------------
# 7. MEASUREMENT FUNCTIONS
# ----------------------------------------------------------------
def measure_energy_mpo(mps, MPO):
    """Compute <psi|H|psi> via full MPO contraction."""
    L = len(mps)
    chi0 = mps[0].shape[1]
    Dw0 = MPO[0].shape[0]
    F = np.ones((chi0, Dw0, chi0))
    for i in range(L):
        F = compute_L_env(mps[i], MPO[i], F)
    return np.real(F.ravel()[0])


def measure_local_op(mps, op, site):
    """Measure <psi| op_{site} |psi> by contracting transfer matrices."""
    L = len(mps)
    d = mps[0].shape[0]
    chi = mps[0].shape[1]
    C = np.eye(chi, dtype=np.float64)
    for i in range(L):
        M = mps[i]
        d_i, cl, cr = M.shape
        C_new = np.zeros((cr, cr), dtype=np.float64)
        if i == site:
            for s in range(d_i):
                for sp in range(d_i):
                    if abs(op[s, sp]) > 1e-15:
                        C_new += op[s, sp] * (M[sp].conj().T @ C @ M[s])
        else:
            for s in range(d_i):
                C_new += M[s].conj().T @ C @ M[s]
        C = C_new
    return np.real(C.item())


def measure_two_point(mps, opA, sA, opB, sB):
    """Measure <psi| opA_{sA} opB_{sB} |psi>."""
    L = len(mps)
    d = mps[0].shape[0]
    chi = mps[0].shape[1]
    C = np.eye(chi, dtype=np.float64)
    for i in range(L):
        M = mps[i]
        d_i, cl, cr = M.shape
        C_new = np.zeros((cr, cr), dtype=np.float64)
        if i == sA:
            op = opA
        elif i == sB:
            op = opB
        else:
            op = None
        if op is not None:
            for s in range(d_i):
                for sp in range(d_i):
                    if abs(op[s, sp]) > 1e-15:
                        C_new += op[s, sp] * (M[sp].conj().T @ C @ M[s])
        else:
            for s in range(d_i):
                C_new += M[s].conj().T @ C @ M[s]
        C = C_new
    return np.real(C.item())


def compute_entanglement_entropy(mps, bond):
    """Compute von Neumann entanglement entropy at given bond by left-canonicalizing."""
    L = len(mps)
    tensors = [m.copy() for m in mps]
    for i in range(bond):
        M = tensors[i]
        d_i, cl, cr = M.shape
        M_mat = M.reshape(d_i * cl, cr)
        U, S, Vt = np.linalg.svd(M_mat, full_matrices=False)
        tensors[i] = U.reshape(d_i, cl, -1)
        SV = np.diag(S) @ Vt
        Mn = tensors[i + 1]
        dn, cln, crn = Mn.shape
        new_next = np.tensordot(SV, Mn, axes=([1], [1]))  # (k, d, cr)
        tensors[i + 1] = new_next.transpose(1, 0, 2)
    # SVD at bond
    M = tensors[bond]
    d_i, cl, cr = M.shape
    M_mat = M.reshape(d_i * cl, cr)
    _, S, _ = np.linalg.svd(M_mat, full_matrices=False)
    S = S[S > 1e-16]
    S2 = S ** 2
    S2 /= np.sum(S2)
    return -np.sum(S2 * np.log2(S2 + 1e-30))


# ----------------------------------------------------------------
# 8. EXACT DIAGONALIZATION (small systems, for benchmarking)
# ----------------------------------------------------------------
def exact_gs_energy(L, model='heisenberg', **kwargs):
    """Exact ground state energy by full diagonalization."""
    if model == 'heisenberg':
        d = 2
        Sp = np.array([[0,1],[0,0]], dtype=float)
        Sm = np.array([[0,0],[1,0]], dtype=float)
        Sz = np.diag([0.5, -0.5])
        Id = np.eye(d)
        J = kwargs.get('J', 1.0)
        Jz = kwargs.get('Jz', 1.0)
        h = kwargs.get('h', 0.0)
        dim = d**L
        H = np.zeros((dim, dim))
        for i in range(L-1):
            def op_at(op, site):
                parts = [Id]*L; parts[site] = op
                r = parts[0]
                for p in parts[1:]: r = np.kron(r, p)
                return r
            H += (J/2.)*(op_at(Sp,i)@op_at(Sm,i+1) + op_at(Sm,i)@op_at(Sp,i+1))
            H += Jz * op_at(Sz,i)@op_at(Sz,i+1)
        for i in range(L):
            def op_at(op, site):
                parts = [Id]*L; parts[site] = op
                r = parts[0]
                for p in parts[1:]: r = np.kron(r, p)
                return r
            H -= h * op_at(Sz, i)
    elif model == 'aklt':
        d = 3
        Sp = np.array([[0,np.sqrt(2),0],[0,0,np.sqrt(2)],[0,0,0]], dtype=float)
        Sm = Sp.T.copy()
        Sz = np.diag([1.,0.,-1.])
        Id = np.eye(d)
        SdS = (np.kron(Sp,Sm)+np.kron(Sm,Sp))/2. + np.kron(Sz,Sz)
        h2 = SdS + (1./3.)*SdS@SdS
        dim = d**L
        H = np.zeros((dim, dim))
        for i in range(L-1):
            pre = np.eye(d**i) if i > 0 else np.array([[1.]])
            post = np.eye(d**(L-2-i)) if i < L-2 else np.array([[1.]])
            H += np.kron(np.kron(pre, h2), post)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    from scipy.linalg import eigvalsh
    return eigvalsh(H)[0]


# ================================================================
# 9. RUN TESTS
# ================================================================
print("=" * 70)
print("  TWO-SITE DMRG: MPS/MPO + MATRIX-FREE LANCZOS")
print("  Following Schollwöck, Ann. Phys. 326 (2011) 96-192")
print("=" * 70)

# ---- 9a. Heisenberg S=1/2 quick test ----
print("\n" + "="*70)
print("  QUICK TEST: S=1/2 HEISENBERG (small systems)")
print("="*70)

for L_test in [6, 8, 10]:
    D_test = 2**(L_test//2)
    MPO_h = build_mpo_heisenberg(L_test)
    E_dmrg, _, _ = two_site_dmrg(MPO_h, L_test, d=2, D=D_test,
                                  num_sweeps=10, verbose=False)
    E_ex = exact_gs_energy(L_test, model='heisenberg')
    err = abs(E_dmrg - E_ex)
    print(f"  L={L_test:2d}  D={D_test:3d}  E_DMRG={E_dmrg:16.12f}  "
          f"E_exact={E_ex:16.12f}  |err|={err:.2e}")

# ---- 9b. AKLT S=1 quick test ----
print("\n" + "="*70)
print("  QUICK TEST: AKLT MODEL S=1 (small systems)")
print("="*70)

for L_test in [4, 6, 8]:
    D_test = min(3**(L_test//2), 81)
    MPO_a = build_mpo_aklt(L_test)
    E_dmrg, _, _ = two_site_dmrg(MPO_a, L_test, d=3, D=D_test,
                                  num_sweeps=12, verbose=False)
    E_ex = exact_gs_energy(L_test, model='aklt')
    err = abs(E_dmrg - E_ex)
    print(f"  L={L_test:2d}  D={D_test:3d}  E_DMRG={E_dmrg:16.12f}  "
          f"E_exact={E_ex:16.12f}  |err|={err:.2e}")


# ================================================================
# 10. LARGER SYSTEMS
# ================================================================

# ---- 10a. Heisenberg ----
print("\n" + "="*70)
print("  LARGER SYSTEM: S=1/2 HEISENBERG")
print("="*70)

L_h = 48
D_h = 64
nsweeps_h = 10
print(f"  L={L_h}, D={D_h}, sweeps={nsweeps_h}")
t0 = time.time()
MPO_hL = build_mpo_heisenberg(L_h)
E_hL, mps_hL, ehist_hL = two_site_dmrg(MPO_hL, L_h, d=2, D=D_h,
                                         num_sweeps=nsweeps_h, verbose=True)
t_h = time.time() - t0
E_bethe = 0.25 - np.log(2)
print(f"  Final E = {E_hL:.12f},  E/L = {E_hL/L_h:.12f}")
print(f"  Bethe E/L (L->inf) = {E_bethe:.12f}")
print(f"  Time: {t_h:.1f}s")

# Measure Sz profile
Sz2 = np.diag([0.5, -0.5])
sz_prof_h = [measure_local_op(mps_hL, Sz2, i) for i in range(L_h)]

# Entanglement entropy profile
ee_h = [compute_entanglement_entropy(mps_hL, b) for b in range(1, L_h)]

# Sz-Sz connected correlator from center
center_h = L_h // 2
corr_h = []
dist_h = []
for r in range(1, L_h//2 - 1):
    j = center_h + r
    if j < L_h:
        c = measure_two_point(mps_hL, Sz2, center_h, Sz2, j)
        s1 = measure_local_op(mps_hL, Sz2, center_h)
        s2 = measure_local_op(mps_hL, Sz2, j)
        corr_h.append(c - s1*s2)
        dist_h.append(r)

# Convergence in D
print("\n  Convergence in D (L=32 Heisenberg):")
L_cv = 32
D_list = [8, 16, 32, 48, 64]
E_cv_h = []
for Dv in D_list:
    mpo_cv = build_mpo_heisenberg(L_cv)
    E_cv, _, _ = two_site_dmrg(mpo_cv, L_cv, d=2, D=Dv, num_sweeps=8, verbose=False)
    E_cv_h.append(E_cv)
    print(f"    D={Dv:3d}  E={E_cv:.10f}  E/L={E_cv/L_cv:.10f}")


# ---- 10b. AKLT ----
print("\n" + "="*70)
print("  LARGER SYSTEM: AKLT MODEL S=1")
print("="*70)

L_a = 48
D_a = 32
nsweeps_a = 12
print(f"  L={L_a}, D={D_a}, sweeps={nsweeps_a}")
t0 = time.time()
MPO_aL = build_mpo_aklt(L_a)
E_aL, mps_aL, ehist_aL = two_site_dmrg(MPO_aL, L_a, d=3, D=D_a,
                                         num_sweeps=nsweeps_a, verbose=True)
t_a = time.time() - t0
# AKLT exact: E/bond = -2/3 for bulk, so E_bulk ~ -(2/3)(L-1)
E_aklt_approx = -(2./3.)*(L_a - 1)
print(f"  Final E = {E_aL:.12f},  E/bond = {E_aL/(L_a-1):.12f}")
print(f"  AKLT exact E/bond = {-2./3.:.12f}")
print(f"  Time: {t_a:.1f}s")

# Sz profile for AKLT
Sz1 = np.diag([1.0, 0.0, -1.0])
sz_prof_a = [measure_local_op(mps_aL, Sz1, i) for i in range(L_a)]

# Entanglement entropy for AKLT
ee_a = [compute_entanglement_entropy(mps_aL, b) for b in range(1, L_a)]

# Sz-Sz correlator for AKLT (should decay as (-1/3)^|i-j|)
center_a = L_a // 2
corr_a = []
dist_a = []
for r in range(1, L_a//2 - 1):
    j = center_a + r
    if j < L_a:
        c = measure_two_point(mps_aL, Sz1, center_a, Sz1, j)
        s1 = measure_local_op(mps_aL, Sz1, center_a)
        s2 = measure_local_op(mps_aL,

---
## 🧑‍💻 User


continue

---
## 🤖 Assistant




```python
        s2 = measure_local_op(mps_aL, Sz1, j)
        corr_a.append(c - s1*s2)
        dist_a.append(r)

# String order parameter for AKLT: <Sz_i exp(i*pi*sum_{i<k<j} Sz_k) Sz_j>
# We compute this by sequential transfer matrix approach
def measure_string_order(mps, site_i, site_j):
    """Measure string order <Sz_i * prod_{i<k<j} exp(i*pi*Sz_k) * Sz_j>."""
    L = len(mps)
    d = mps[0].shape[0]
    Sz = np.diag([1.0, 0.0, -1.0])
    exp_ipi_Sz = np.diag([np.exp(1j*np.pi*1.0), np.exp(1j*np.pi*0.0), np.exp(1j*np.pi*(-1.0))])
    # = diag(-1, 1, -1)
    Id = np.eye(d)
    chi = mps[0].shape[1]
    C = np.eye(chi, dtype=np.complex128)
    for k in range(L):
        M = mps[k]
        d_k, cl, cr = M.shape
        if k == site_i or k == site_j:
            op = Sz
        elif site_i < k < site_j:
            op = exp_ipi_Sz
        else:
            op = Id
        C_new = np.zeros((cr, cr), dtype=np.complex128)
        for s in range(d_k):
            for sp in range(d_k):
                if abs(op[s, sp]) > 1e-15:
                    C_new += op[s, sp] * (M[sp].conj().T @ C @ M[s])
        C = C_new
    return np.real(C.item())

string_a = []
dist_string = []
for r in range(2, L_a//2 - 1):
    j = center_a + r
    if j < L_a:
        so = measure_string_order(mps_aL, center_a, j)
        string_a.append(so)
        dist_string.append(r)

# Convergence in D for AKLT
print("\n  Convergence in D (L=24 AKLT):")
L_cv_a = 24
D_list_a = [4, 8, 16, 24, 32]
E_cv_a = []
for Dv in D_list_a:
    mpo_cv = build_mpo_aklt(L_cv_a)
    Ecv, _, _ = two_site_dmrg(mpo_cv, L_cv_a, d=3, D=Dv, num_sweeps=10, verbose=False)
    E_cv_a.append(Ecv)
    print(f"    D={Dv:3d}  E={Ecv:.10f}  E/bond={Ecv/(L_cv_a-1):.10f}")


# ================================================================
# 11. FIGURES
# ================================================================
print("\n" + "="*70)
print("  GENERATING FIGURES -> figureAKLT/")
print("="*70)

# --- Figure 1: Energy convergence per sweep (both models) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
sweeps_h = np.arange(1, len(ehist_hL)+1)
ax.plot(sweeps_h, ehist_hL, 'bo-', markersize=5)
ax.axhline(E_bethe * L_h, color='r', ls='--', label=f'Bethe x L ({E_bethe*L_h:.4f})')
ax.set_xlabel('Sweep')
ax.set_ylabel('Ground state energy')
ax.set_title(f'Heisenberg S=1/2, L={L_h}, D={D_h}')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
sweeps_a = np.arange(1, len(ehist_aL)+1)
ax.plot(sweeps_a, ehist_aL, 'rs-', markersize=5)
ax.axhline(E_aklt_approx, color='b', ls='--', label=f'Exact bulk ({E_aklt_approx:.4f})')
ax.set_xlabel('Sweep')
ax.set_ylabel('Ground state energy')
ax.set_title(f'AKLT S=1, L={L_a}, D={D_a}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figureAKLT/fig1_energy_convergence.pdf", dpi=150)
plt.savefig("figureAKLT/fig1_energy_convergence.png", dpi=150)
plt.show()
print("  Saved fig1_energy_convergence")

# --- Figure 2: Entanglement entropy profiles ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
bonds_h = np.arange(1, L_h)
ax.plot(bonds_h, ee_h, 'b.-')
ax.set_xlabel('Bond position')
ax.set_ylabel('Entanglement entropy $S$ (bits)')
ax.set_title(f'Heisenberg S=1/2, L={L_h}, D={D_h}')
ax.grid(True, alpha=0.3)

ax = axes[1]
bonds_a = np.arange(1, L_a)
ax.plot(bonds_a, ee_a, 'r.-')
ax.set_xlabel('Bond position')
ax.set_ylabel('Entanglement entropy $S$ (bits)')
ax.set_title(f'AKLT S=1, L={L_a}, D={D_a}')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figureAKLT/fig2_entanglement_entropy.pdf", dpi=150)
plt.savefig("figureAKLT/fig2_entanglement_entropy.png", dpi=150)
plt.show()
print("  Saved fig2_entanglement_entropy")

# --- Figure 3: Sz profiles ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(range(L_h), sz_prof_h, 'b.-')
ax.set_xlabel('Site $i$')
ax.set_ylabel(r'$\langle S^z_i \rangle$')
ax.set_title(f'Heisenberg S=1/2, L={L_h}')
ax.axhline(0, color='k', ls=':', alpha=0.5)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(range(L_a), sz_prof_a, 'r.-')
ax.set_xlabel('Site $i$')
ax.set_ylabel(r'$\langle S^z_i \rangle$')
ax.set_title(f'AKLT S=1, L={L_a}')
ax.axhline(0, color='k', ls=':', alpha=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figureAKLT/fig3_Sz_profile.pdf", dpi=150)
plt.savefig("figureAKLT/fig3_Sz_profile.png", dpi=150)
plt.show()
print("  Saved fig3_Sz_profile")

# --- Figure 4: Spin-spin correlations ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
dist_h = np.array(dist_h)
corr_h = np.array(corr_h)
ax.semilogy(dist_h, np.abs(corr_h), 'bo-', markersize=4, label=r'$|\langle S^z S^z\rangle_c|$')
# For Heisenberg, correlations decay as ~(-1)^r / r (power law at criticality)
ax.set_xlabel('Distance $r$')
ax.set_ylabel(r'$|C^{zz}(r)|$')
ax.set_title(f'Heisenberg S=1/2, L={L_h}')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
dist_a = np.array(dist_a)
corr_a = np.array(corr_a)
ax.semilogy(dist_a, np.abs(corr_a), 'ro-', markersize=4, label=r'$|\langle S^z S^z\rangle_c|$ (DMRG)')
# AKLT exact: |C| ~ (4/9)*(1/3)^r
r_fit = np.arange(1, max(dist_a)+1)
aklt_exact_corr = (4.0/9.0) * (1.0/3.0)**r_fit
ax.semilogy(r_fit, aklt_exact_corr, 'k--', label=r'$(4/9)(1/3)^r$ exact')
ax.set_xlabel('Distance $r$')
ax.set_ylabel(r'$|C^{zz}(r)|$')
ax.set_title(f'AKLT S=1, L={L_a}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figureAKLT/fig4_correlations.pdf", dpi=150)
plt.savefig("figureAKLT/fig4_correlations.png", dpi=150)
plt.show()
print("  Saved fig4_correlations")

# --- Figure 5: AKLT string order parameter ---
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
dist_string = np.array(dist_string)
string_a = np.array(string_a)
ax.plot(dist_string, string_a, 'rs-', markersize=5, label='DMRG string order')
ax.axhline(-4.0/9.0, color='k', ls='--', label=r'Exact $-4/9$')
ax.set_xlabel('Distance $|i - j|$')
ax.set_ylabel(r'$\langle S^z_i \, e^{i\pi \sum_k S^z_k} \, S^z_j \rangle$')
ax.set_title(f'AKLT String Order Parameter, L={L_a}, D={D_a}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figureAKLT/fig5_string_order.pdf", dpi=150)
plt.savefig("figureAKLT/fig5_string_order.png", dpi=150)
plt.show()
print("  Saved fig5_string_order")

# --- Figure 6: Convergence with bond dimension D ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(D_list, np.array(E_cv_h)/L_cv, 'bo-', markersize=6)
ax.axhline(E_bethe, color='r', ls='--', label=f'Bethe E/L={E_bethe:.6f}')
ax.set_xlabel('Bond dimension $D$')
ax.set_ylabel('$E/L$')
ax.set_title(f'Heisenberg S=1/2 convergence, L={L_cv}')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(D_list_a, np.array(E_cv_a)/(L_cv_a - 1), 'rs-', markersize=6)
ax.axhline(-2./3., color='b', ls='--', label='Exact $E$/bond = $-2/3$')
ax.set_xlabel('Bond dimension $D$')
ax.set_ylabel('$E$/bond')
ax.set_title(f'AKLT S=1 convergence, L={L_cv_a}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figureAKLT/fig6_convergence_D.pdf", dpi=150)
plt.savefig("figureAKLT/fig6_convergence_D.png", dpi=150)
plt.show()
print("  Saved fig6_convergence_D")

# --- Figure 7: AKLT entanglement entropy vs exact ---
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
bonds_a = np.arange(1, L_a)
ax.plot(bonds_a, ee_a, 'r.-', label='DMRG')
# AKLT bulk entanglement entropy = log2(4) = 2 bits per bond for PBC
# For OBC the bulk value is log2(2) = 1 bit (Schmidt rank 2 in bulk)
ax.axhline(np.log2(4), color='b', ls=':', alpha=0.5, label=r'$\log_2 4 = 2$ (PBC bulk)')
ax.axhline(1.0, color='g', ls='--', alpha=0.7, label=r'$\log_2 2 = 1$ (OBC bulk, D=2)')
ax.set_xlabel('Bond position')
ax.set_ylabel('Entanglement entropy (bits)')
ax.set_title(f'AKLT Entanglement Entropy, L={L_a}, D={D_a}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figureAKLT/fig7_aklt_entropy.pdf", dpi=150)
plt.savefig("figureAKLT/fig7_aklt_entropy.png", dpi=150)
plt.show()
print("  Saved fig7_aklt_entropy")

# --- Figure 8: Summary comparison table as text figure ---
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.axis('off')
table_data = [
    ['Model', 'L', 'D', 'E_DMRG', 'E/site or E/bond', 'Exact ref'],
    ['Heisenberg', str(L_h), str(D_h), f'{E_hL:.8f}',
     f'{E_hL/L_h:.8f}', f'{E_bethe:.8f} (Bethe)'],
    ['AKLT', str(L_a), str(D_a), f'{E_aL:.8f}',
     f'{E_aL/(L_a-1):.8f}', f'{-2./3.:.8f} (exact)'],
]
table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.6)
for (row, col), cell in table.get_celld().items

---
## 🧑‍💻 User


======================================================================
  TWO-SITE DMRG: MPS/MPO + MATRIX-FREE LANCZOS
  Following Schollwöck, Ann. Phys. 326 (2011) 96-192
======================================================================

======================================================================
  QUICK TEST: S=1/2 HEISENBERG (small systems)
======================================================================
  L= 6  D=  8  E_DMRG= -2.493577133888  E_exact= -2.493577133888  |err|=3.11e-15
  L= 8  D= 16  E_DMRG= -3.374932598688  E_exact= -3.374932598688  |err|=8.88e-16
  L=10  D= 32  E_DMRG= -4.258035207283  E_exact= -4.258035207283  |err|=3.55e-15

======================================================================
  QUICK TEST: AKLT MODEL S=1 (small systems)
======================================================================
  L= 4  D=  9  E_DMRG= -1.800086609435  E_exact= -2.000000000000  |err|=2.00e-01
  L= 6  D= 27  E_DMRG= -3.779289669559  E_exact= -3.333333333333  |err|=4.46e-01
  L= 8  D= 81  E_DMRG= -5.778894899766  E_exact= -4.666666666667  |err|=1.11e+00

======================================================================
  LARGER SYSTEM: S=1/2 HEISENBERG
======================================================================
  L=48, D=64, sweeps=10

---
## 🤖 Assistant


The Heisenberg model now works perfectly (machine precision errors). The AKLT model has an issue - the energies are **too low** (more negative than exact), which means the MPO is wrong. The problem is in the AKLT MPO construction: the SVD decomposition of the bond operator assigns operators to the wrong sites in the first/last W-matrices.

Let me provide the corrected `build_mpo_aklt` function and the rest of the code:

```python
# ================================================================
# FIXED AKLT MPO + complete remaining code
# ================================================================

def build_mpo_aklt(L):
    """
    MPO for AKLT Hamiltonian (S=1, OBC):
    H = sum_i [ S_i . S_{i+1} + (1/3)(S_i . S_{i+1})^2 ]
    
    We decompose h_bond = sum_k L_k x R_k via SVD,
    then build the standard W-matrix (Schollwöck Eq. 182 pattern):
    
    Bulk W:
        [ I    0    0  ]
        [ R_1   0    0  ]
        [ ...   0    0  ]
        [ R_n   0    0  ]
        [ 0   L_1 ... L_n   I ]
    
    First site (row): [ 0  L_1 ... L_n  I ]
    Last site (col):  [ I  R_1 ... R_n  0 ]^T
    """
    d = 3
    Sp = np.array([[0, np.sqrt(2), 0],
                   [0, 0, np.sqrt(2)],
                   [0, 0, 0]], dtype=float)
    Sm = Sp.T.copy()
    Sz = np.diag([1.0, 0.0, -1.0])
    Id = np.eye(d)
    
    # Two-site bond Hamiltonian in basis |s1 s2><s1' s2'|
    SdotS = (np.kron(Sp, Sm) + np.kron(Sm, Sp)) / 2.0 + np.kron(Sz, Sz)
    h_bond = SdotS + (1.0/3.0) * SdotS @ SdotS
    
    # Verify: for AKLT ground state, E/bond = -2/3
    # h_bond should have eigenvalues: 0 (x5, S_tot=2 projected out) and -2/3 (x4, from S=0,1)
    # Actually H_AKLT = P_2(S=2) * 2/3 + const, let's check eigenvalues
    evals_check = np.linalg.eigvalsh(h_bond)
    print(f"    h_bond eigenvalues: {np.sort(evals_check)}")
    
    # SVD: h_bond = sum_k sqrt(s_k) U_k x sqrt(s_k) V_k
    # h_bond[s1*d+s2, s1'*d+s2'] with row=(s1,s2) as LEFT pair, col=(s1',s2') as RIGHT pair
    # But we need: h_bond = sum_k Left_k[s1,s1'] x Right_k[s2,s2']
    # So reshape h_bond as (s1,s1') x (s2,s2') = (d^2) x (d^2)
    # h_bond currently indexed as (s1*d+s2, s1p*d+s2p)
    # We need to transpose to (s1, s1p, s2, s2p) then reshape to (s1*s1p, s2*s2p)
    
    h_4d = h_bond.reshape(d, d, d, d)  # (s1, s2, s1', s2')
    h_reorder = h_4d.transpose(0, 2, 1, 3)  # (s1, s1', s2, s2')
    h_mat = h_reorder.reshape(d*d, d*d)  # (s1*s1', s2*s2')
    
    U, s, Vt = np.linalg.svd(h_mat, full_matrices=False)
    keep = s > 1e-12
    nterms = int(np.sum(keep))
    U = U[:, keep]
    s = s[keep]
    Vt = Vt[keep, :]
    
    sqrts = np.sqrt(s)
    
    # L_k[s1, s1'] = U[s1*d+s1', k] * sqrt(s_k) -- acts on LEFT site
    # R_k[s2, s2'] = sqrt(s_k) * Vt[k, s2*d+s2'] -- acts on RIGHT site
    Lk_list = []
    Rk_list = []
    for k in range(nterms):
        Lk = (U[:, k] * sqrts[k]).reshape(d, d)
        Rk = (sqrts[k] * Vt[k, :]).reshape(d, d)
        Lk_list.append(Lk)
        Rk_list.append(Rk)
    
    # Verify decomposition: h_bond should equal sum_k kron(Lk, Rk) after reorder
    h_check = np.zeros((d**2, d**2))
    for k in range(nterms):
        h_check += np.kron(Lk_list[k], Rk_list[k])
    # h_check is in (s1,s2)x(s1',s2') ordering with the transpose convention
    # Convert back: (s1,s1',s2,s2') -> (s1,s2,s1',s2')
    h_check_4d = h_check.reshape(d,d,d,d).transpose(0,2,1,3).reshape(d**2, d**2)
    decomp_err = np.linalg.norm(h_check_4d - h_bond)
    print(f"    SVD decomposition error: {decomp_err:.2e}")
    
    DW = nterms + 2
    
    # Bulk W[bL, bR, s, s']:
    # State 0: "not yet started" -> pass identity
    # States 1..nterms: "R_k deposited, waiting for L_k"
    # State DW-1: "completed" -> pass identity
    #
    # Transitions:
    #   0 -> 0:      I         (nothing happened yet)
    #   0 -> k:      R_k       (right part deposited)
    #   k -> DW-1:   L_k       (left part completes the term)
    #   DW-1 -> DW-1: I        (term completed, pass through)
    
    def make_bulk():
        W = np.zeros((DW, DW, d, d))
        W[0, 0] = Id
        for k in range(nterms):
            W[k+1, 0] = Rk_list[k]      # 0 -> k+1: deposit R_k
            W[DW-1, k+1] = Lk_list[k]   # k+1 -> DW-1: complete with L_k
        W[DW-1, DW-1] = Id
        return W
    
    def make_first():
        # First site = row vector (1, DW, d, d)
        # Only outgoing states: start completed terms or start intermediate
        # From the "left boundary" (state DW-1 equivalent):
        # We can deposit L_k (going to state k+1 waiting for R_k on next site)
        # Or just pass I (going to state DW-1 = completed, no contribution)
        # Wait -- the convention from Schollwöck Eq.(183):
        # W[1] = [ h_local, (J/2)S-, (J/2)S+, Jz Sz, I ]
        # = [ completed_local, complete_interaction_terms, I ]
        # For us: no local term, complete terms have L_k, and I passes through
        # 
        # W_first[0, k+1] = L_k   (deposit left operator, waiting for R_k)
        # W_first[0, DW-1] = I    (pass through)
        # W_first[0, 0] = 0       (no single-site term to add)
        W = np.zeros((1, DW, d, d))
        for k in range(nterms):
            W[0, k+1] = Lk_list[k]
        W[0, DW-1] = Id
        return W
    
    def make_last():
        # Last site = column vector (DW, 1, d, d)
        # W_last[0, 0] = I        (nothing happened: trivial, contributes I)
        # W_last[k+1, 0] = R_k    (complete term started earlier)
        # W_last[DW-1, 0] = 0     (already completed: no local term)
        W = np.zeros((DW, 1, d, d))
        W[0, 0] = Id
        for k in range(nterms):
            W[k+1, 0] = Rk_list[k]
        return W
    
    MPO = []
    for i in range(L):
        if i == 0:
            MPO.append(make_first())
        elif i == L - 1:
            MPO.append(make_last())
        else:
            MPO.append(make_bulk())
    return MPO


# ================================================================
# RE-RUN AKLT TESTS WITH FIXED MPO
# ================================================================
print("\n" + "="*70)
print("  RE-TEST: AKLT MODEL S=1 (fixed MPO)")
print("="*70)

for L_test in [4, 6, 8]:
    D_test = min(3**(L_test//2), 81)
    MPO_a = build_mpo_aklt(L_test)
    E_dmrg, _, _ = two_site_dmrg(MPO_a, L_test, d=3, D=D_test,
                                  num_sweeps=12, verbose=False)
    E_ex = exact_gs_energy(L_test, model='aklt')
    err = abs(E_dmrg - E_ex)
    print(f"  L={L_test:2d}  D={D_test:3d}  E_DMRG={E_dmrg:16.12f}  "
          f"E_exact={E_ex:16.12f}  |err|={err:.2e}")


# ================================================================
# LARGER AKLT
# ================================================================
print("\n" + "="*70)
print("  LARGER SYSTEM: AKLT MODEL S=1")
print("="*70)

L_a = 48
D_a = 32
nsweeps_a = 12
print(f"  L={L_a}, D={D_a}, sweeps={nsweeps_a}")
t0 = time.time()
MPO_aL = build_mpo_aklt(L_a)
E_aL, mps_aL, ehist_aL = two_site_dmrg(MPO_aL, L_a, d=3, D=D_a,
                                         num_sweeps=nsweeps_a, verbose=True)
t_a = time.time() - t0
E_aklt_exact_bond = -2.0/3.0
print(f"  Final E = {E_aL:.12f},  E/bond = {E_aL/(L_a-1):.12f}")
print(f"  AKLT exact E/bond = {E_aklt_exact_bond:.12f}")
print(f"  Time: {t_a:.1f}s")

# ---- Measurements ----
Sz1 = np.diag([1.0, 0.0, -1.0])

# Sz profile
sz_prof_a = [measure_local_op(mps_aL, Sz1, i) for i in range(L_a)]

# Entanglement entropy
ee_a = [compute_entanglement_entropy(mps_aL, b) for b in range(1, L_a)]

# Sz-Sz correlator
center_a = L_a // 2
corr_a = []
dist_a = []
for r in range(1, L_a//2 - 1):
    j = center_a + r
    if j < L_a:
        c = measure_two_point(mps_aL, Sz1, center_a, Sz1, j)
        s1 = measure_local_op(mps_aL, Sz1, center_a)
        s2 = measure_local_op(mps_aL, Sz1, j)
        corr_a.append(c - s1*s2)
        dist_a.append(r)

# String order parameter
def measure_string_order(mps, site_i, site_j):
    """<Sz_i * prod_{i<k<j} exp(i*pi*Sz_k) * Sz_j>"""
    L = len(mps)
    d = mps[0].shape[0]
    Sz = np.diag([1.0, 0.0, -1.0])
    eipSz = np.diag([np.exp(1j*np.pi*1.0), np.exp(1j*np.pi*0.0),
                     np.exp(1j*np.pi*(-1.0))])  # diag(-1, 1, -1)
    Id = np.eye(d)
    chi = mps[0].shape[1]
    C = np.eye(chi, dtype=np.complex128)
    for k in range(L):
        M = mps[k]
        d_k, cl, cr = M.shape
        if k == site_i or k == site_j:
            op = Sz
        elif site_i < k < site_j:
            op = eipSz
        else:
            op = Id
        C_new = np.zeros((cr, cr), dtype=np.complex128)
        for s in range(d_k):
            for sp in range(d_k):
                if abs(op[s, sp]) > 1e-15:
                    C_new += op[s, sp] * (M[sp].conj().T @ C @ M[s])
        C = C_new
    return np.real(C.item())

string_a = []
dist_string = []
for r in range(2, L_a//2 - 1):
    j = center_a + r
    if j < L_a:
        string_a.append(measure_string_order(mps_aL, center_a, j))
        dist_string.append(r)

# Convergence in D
print("\n  Convergence in D (L=24 AKLT):")
L_cv_a = 24
D_list_a = [4, 8, 16, 24, 32]
E_cv_a = []
for Dv in D_list_a:
    mpo_cv = build_mpo_aklt(L_cv_a)
    Ecv, _, _ = two_site_dmrg(mpo_cv, L_cv_a, d=3, D=Dv,
                               num_sweeps=10, verbose=False)
    E_cv_a.append(Ecv)
    print(f"    D={Dv:3d}  E={Ecv:.10f}  E/bond={Ecv/(L_cv_a-1):.10f}")


---
## 🧑‍💻 User


continue

---
## 🤖 Assistant




```python
# ================================================================
# FIGURES (all saved to figureAKLT/)
# ================================================================
print("\n" + "="*70)
print("  GENERATING FIGURES -> figureAKLT/")
print("="*70)

# We need the Heisenberg data too. If not already computed, compute it.
# (Assuming the Heisenberg large run already completed above.)
# If variables are missing, recompute quickly:
try:
    _ = E_hL
except NameError:
    print("  Computing Heisenberg L=48 D=64 ...")
    L_h = 48; D_h = 64; nsweeps_h = 10
    MPO_hL = build_mpo_heisenberg(L_h)
    E_hL, mps_hL, ehist_hL = two_site_dmrg(MPO_hL, L_h, d=2, D=D_h,
                                             num_sweeps=nsweeps_h, verbose=True)
    Sz2 = np.diag([0.5, -0.5])
    sz_prof_h = [measure_local_op(mps_hL, Sz2, i) for i in range(L_h)]
    ee_h = [compute_entanglement_entropy(mps_hL, b) for b in range(1, L_h)]
    center_h = L_h // 2
    corr_h = []; dist_h = []
    for r in range(1, L_h//2 - 1):
        j = center_h + r
        if j < L_h:
            c = measure_two_point(mps_hL, Sz2, center_h, Sz2, j)
            s1v = measure_local_op(mps_hL, Sz2, center_h)
            s2v = measure_local_op(mps_hL, Sz2, j)
            corr_h.append(c - s1v*s2v); dist_h.append(r)
    L_cv = 32; D_list = [8, 16, 32, 48, 64]; E_cv_h = []
    for Dv in D_list:
        mpo_cv = build_mpo_heisenberg(L_cv)
        Ecv, _, _ = two_site_dmrg(mpo_cv, L_cv, d=2, D=Dv, num_sweeps=8, verbose=False)
        E_cv_h.append(Ecv)

E_bethe = 0.25 - np.log(2)

# ---- Figure 1: Energy convergence per sweep ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
sw_h = np.arange(1, len(ehist_hL)+1)
ax.plot(sw_h, ehist_hL, 'bo-', ms=4, lw=1.5)
ax.axhline(E_bethe * L_h, color='r', ls='--', lw=1.2,
           label=f'Bethe$\\times L$ = {E_bethe*L_h:.4f}')
ax.set_xlabel('Sweep', fontsize=12)
ax.set_ylabel('Energy $E$', fontsize=12)
ax.set_title(f'Heisenberg $S=1/2$, $L={L_h}$, $D={D_h}$', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

ax = axes[1]
sw_a = np.arange(1, len(ehist_aL)+1)
ax.plot(sw_a, ehist_aL, 'rs-', ms=4, lw=1.5)
E_aklt_ref = -(2./3.)*(L_a - 1)
ax.axhline(E_aklt_ref, color='b', ls='--', lw=1.2,
           label=f'Exact bulk = {E_aklt_ref:.4f}')
ax.set_xlabel('Sweep', fontsize=12)
ax.set_ylabel('Energy $E$', fontsize=12)
ax.set_title(f'AKLT $S=1$, $L={L_a}$, $D={D_a}$', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figureAKLT/fig1_energy_convergence.pdf", dpi=150)
plt.savefig("figureAKLT/fig1_energy_convergence.png", dpi=150)
plt.show(); print("  Saved fig1_energy_convergence")

# ---- Figure 2: Entanglement entropy ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(np.arange(1, L_h), ee_h, 'b.-', ms=3)
ax.set_xlabel('Bond position', fontsize=12)
ax.set_ylabel('Entanglement entropy $S$ (bits)', fontsize=12)
ax.set_title(f'Heisenberg $S=1/2$, $L={L_h}$, $D={D_h}$', fontsize=13)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(np.arange(1, L_a), ee_a, 'r.-', ms=3)
ax.axhline(1.0, color='g', ls='--', lw=1.2, alpha=0.7,
           label=r'$\log_2 2=1$ (AKLT exact, OBC bulk)')
ax.set_xlabel('Bond position', fontsize=12)
ax.set_ylabel('Entanglement entropy $S$ (bits)', fontsize=12)
ax.set_title(f'AKLT $S=1$, $L={L_a}$, $D={D_a}$', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figureAKLT/fig2_entanglement_entropy.pdf", dpi=150)
plt.savefig("figureAKLT/fig2_entanglement_entropy.png", dpi=150)
plt.show(); print("  Saved fig2_entanglement_entropy")

# ---- Figure 3: <Sz> profiles ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(range(L_h), sz_prof_h, 'b.-', ms=3)
ax.axhline(0, color='k', ls=':', alpha=0.4)
ax.set_xlabel('Site $i$', fontsize=12)
ax.set_ylabel(r'$\langle S^z_i \rangle$', fontsize=12)
ax.set_title(f'Heisenberg $S=1/2$, $L={L_h}$', fontsize=13)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(range(L_a), sz_prof_a, 'r.-', ms=3)
ax.axhline(0, color='k', ls=':', alpha=0.4)
ax.set_xlabel('Site $i$', fontsize=12)
ax.set_ylabel(r'$\langle S^z_i \rangle$', fontsize=12)
ax.set_title(f'AKLT $S=1$, $L={L_a}$', fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figureAKLT/fig3_Sz_profile.pdf", dpi=150)
plt.savefig("figureAKLT/fig3_Sz_profile.png", dpi=150)
plt.show(); print("  Saved fig3_Sz_profile")

# ---- Figure 4: Spin-spin correlations ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
d_h = np.array(dist_h); c_h = np.array(corr_h)
ax.semilogy(d_h, np.abs(c_h), 'bo-', ms=4, label=r'$|C^{zz}(r)|$ DMRG')
ax.set_xlabel('Distance $r$', fontsize=12)
ax.set_ylabel(r'$|\langle S^z_i S^z_j \rangle_c|$', fontsize=12)
ax.set_title(f'Heisenberg $S=1/2$, $L={L_h}$', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

ax = axes[1]
d_a = np.array(dist_a); c_a = np.array(corr_a)
ax.semilogy(d_a, np.abs(c_a), 'ro-', ms=4, label=r'$|C^{zz}(r)|$ DMRG')
r_ref = np.arange(1, max(d_a)+1)
corr_exact = (4./9.) * (1./3.)**r_ref
ax.semilogy(r_ref, corr_exact, 'k--', lw=1.5, label=r'$(4/9)(1/3)^r$ exact')
ax.set_xlabel('Distance $r$', fontsize=12)
ax.set_ylabel(r'$|\langle S^z_i S^z_j \rangle_c|$', fontsize=12)
ax.set_title(f'AKLT $S=1$, $L={L_a}$', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figureAKLT/fig4_correlations.pdf", dpi=150)
plt.savefig("figureAKLT/fig4_correlations.png", dpi=150)
plt.show(); print("  Saved fig4_correlations")

# ---- Figure 5: AKLT string order parameter ----
fig, ax = plt.subplots(figsize=(8, 5))
ds = np.array(dist_string); sa = np.array(string_a)
ax.plot(ds, sa, 'rs-', ms=5, label='DMRG')
ax.axhline(-4./9., color='k', ls='--', lw=1.5, label=r'Exact $-4/9 \approx -0.4444$')
ax.set_xlabel('Distance $|i-j|$', fontsize=12)
ax.set_ylabel(r'$\langle S^z_i \, e^{i\pi\sum_k S^z_k} \, S^z_j \rangle$', fontsize=12)
ax.set_title(f'AKLT String Order, $L={L_a}$, $D={D_a}$', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figureAKLT/fig5_string_order.pdf", dpi=150)
plt.savefig("figureAKLT/fig5_string_order.png", dpi=150)
plt.show(); print("  Saved fig5_string_order")

# ---- Figure 6: Convergence with D ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(D_list, np.array(E_cv_h)/L_cv, 'bo-', ms=6, lw=1.5)
ax.axhline(E_bethe, color='r', ls='--', lw=1.2, label=f'Bethe $E/L={E_bethe:.6f}$')
ax.set_xlabel('Bond dimension $D$', fontsize=12)
ax.set_ylabel('$E/L$', fontsize=12)
ax.set_title(f'Heisenberg convergence, $L={L_cv}$', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(D_list_a, np.array(E_cv_a)/(L_cv_a-1), 'rs-', ms=6, lw=1.5)
ax.axhline(-2./3., color='b', ls='--', lw=1.2, label='Exact $E$/bond $= -2/3$')
ax.set_xlabel('Bond dimension $D$', fontsize=12)
ax.set_ylabel('$E$/bond', fontsize=12)
ax.set_title(f'AKLT convergence, $L={L_cv_a}$', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figureAKLT/fig6_convergence_D.pdf", dpi=150)
plt.savefig("figureAKLT/fig6_convergence_D.png", dpi=150)
plt.show(); print("  Saved fig6_convergence_D")

# ---- Figure 7: AKLT entanglement vs exact ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.arange(1, L_a), ee_a, 'r.-', ms=4, label='DMRG')
ax.axhline(1.0, color='g', ls='--', lw=1.5,
           label=r'AKLT exact OBC bulk $= \log_2 2 = 1$')
ax.set_xlabel('Bond position', fontsize=12)
ax.set_ylabel('Entanglement entropy (bits)', fontsize=12)
ax.set_title(f'AKLT Entanglement, $L={L_a}$, $D={D_a}$', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figureAKLT/fig7_aklt_entropy_detail.pdf", dpi=150)
plt.savefig("figureAKLT/fig7_aklt_entropy_detail.png", dpi=150)
plt.show(); print("  Saved fig7_aklt_entropy_detail")

# ---- Figure 8: Summary table ----
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.axis('off')
col_labels = ['Model', '$L$', '$D$', '$E_{\\rm DMRG}$',
              '$E/L$ or $E$/bond', 'Exact ref.']
table_data = [
    ['Heisenberg $S{=}1/2$', f'{L_h}', f'{D_h}',
     f'{E_hL:.8f}', f'{E_hL/L_h:.8f}', f'{E_bethe:.8f} (Bethe)'],
    ['AKLT $S{=}1$', f'{L_a}', f'{D_a}',
     f'{E_aL:.8f}', f'{E_aL/(L_a-1):.8f}', f'{-2./3.:.8f} (exact)'],
]
tbl = ax.table(cellText=table_data, colLabels=col_labels,
               loc

---
## 🧑‍💻 User


======================================================================
  RE-TEST: AKLT MODEL S=1 (fixed MPO)
======================================================================
    h_bond eigenvalues: [-0.66666667 -0.66666667 -0.66666667 -0.66666667  1.33333333  1.33333333
  1.33333333  1.33333333  1.33333333]
    SVD decomposition error: 2.21e+00
  L= 4  D=  9  E_DMRG= -2.000000000000  E_exact= -2.000000000000  |err|=7.11e-15
    h_bond eigenvalues: [-0.66666667 -0.66666667 -0.66666667 -0.66666667  1.33333333  1.33333333
  1.33333333  1.33333333  1.33333333]
    SVD decomposition error: 2.21e+00
  L= 6  D= 27  E_DMRG= -3.333333333333  E_exact= -3.333333333333  |err|=2.22e-14
    h_bond eigenvalues: [-0.66666667 -0.66666667 -0.66666667 -0.66666667  1.33333333  1.33333333
  1.33333333  1.33333333  1.33333333]
    SVD decomposition error: 2.21e+00
  L= 8  D= 81  E_DMRG= -4.666666666667  E_exact= -4.666666666667  |err|=1.87e-14

======================================================================
  LARGER SYSTEM: AKLT MODEL S=1
======================================================================
  L=48, D=32, sweeps=12
    h_bond eigenvalues: [-0.66666667 -0.66666667 -0.66666667 -0.66666667  1.33333333  1.33333333
  1.33333333  1.33333333  1.33333333]
    SVD decomposition error: 2.21e+00
  Sweep 1/12, E = -31.333333333333
  Sweep 2/12, E = -31.333333333333
  Sweep 3/12, E = -31.333333333333
  Sweep 4/12, E = -31.333333333333
  Sweep 5/12, E = -31.333333333333
  Sweep 6/12, E = -31.333333333333
  Sweep 7/12, E = -31.333333333333
  Sweep 8/12, E = -31.333333333333
  Sweep 9/12, E = -31.333333333333
  Sweep 10/12, E = -31.333333333333
  Sweep 11/12, E = -31.333333333333
  Sweep 12/12, E = -31.333333333333
  Final E = -31.333333333333,  E/bond = -0.666666666667
  AKLT exact E/bond = -0.666666666667
  Time: 6.1s

  Convergence in D (L=24 AKLT):
    h_bond eigenvalues: [-0.66666667 -0.66666667 -0.66666667 -0.66666667  1.33333333  1.33333333
  1.33333333  1.33333333  1.33333333]
    SVD decomposition error: 2.21e+00
    D=  4  E=-15.3333333333  E/bond=-0.6666666667
    h_bond eigenvalues: [-0.66666667 -0.66666667 -0.66666667 -0.66666667  1.33333333  1.33333333
  1.33333333  1.33333333  1.33333333]
    SVD decomposition error: 2.21e+00
    D=  8  E=-15.3333333333  E/bond=-0.6666666667
    h_bond eigenvalues: [-0.66666667 -0.66666667 -0.66666667 -0.66666667  1.33333333  1.33333333
  1.33333333  1.33333333  1.33333333]
    SVD decomposition error: 2.21e+00
    D= 16  E=-15.3333333333  E/bond=-0.6666666667
    h_bond eigenvalues: [-0.66666667 -0.66666667 -0.66666667 -0.66666667  1.33333333  1.33333333
  1.33333333  1.33333333  1.33333333]
    SVD decomposition error: 2.21e+00
    D= 24  E=-15.3333333333  E/bond=-0.6666666667
    h_bond eigenvalues: [-0.66666667 -0.66666667 -0.66666667 -0.66666667  1.33333333  1.33333333
  1.33333333  1.33333333  1.33333333]
    SVD decomposition error: 2.21e+00
    D= 32  E=-15.3333333333  E/bond=-0.6666666667

---
## 🤖 Assistant


Both models now work perfectly — machine precision for all AKLT tests and exact E/bond = -2/3! The SVD decomposition error of 2.21 is just because the reordering check had a bug in the verification (not in the actual MPO). Let me clean that up and provide the figure code to finish:

```python
# ================================================================
# CLEAN UP: Remove diagnostic prints from build_mpo_aklt,
# then generate all figures
# ================================================================

# Patch: remove print statements from build_mpo_aklt for production
def build_mpo_aklt_clean(L):
    """AKLT MPO without diagnostic prints."""
    d = 3
    Sp = np.array([[0, np.sqrt(2), 0],[0, 0, np.sqrt(2)],[0, 0, 0]], dtype=float)
    Sm = Sp.T.copy()
    Sz = np.diag([1.0, 0.0, -1.0])
    Id = np.eye(d)
    SdotS = (np.kron(Sp, Sm) + np.kron(Sm, Sp)) / 2.0 + np.kron(Sz, Sz)
    h_bond = SdotS + (1.0/3.0) * SdotS @ SdotS
    h_4d = h_bond.reshape(d, d, d, d)
    h_reorder = h_4d.transpose(0, 2, 1, 3)
    h_mat = h_reorder.reshape(d*d, d*d)
    U, s, Vt = np.linalg.svd(h_mat, full_matrices=False)
    keep = s > 1e-12
    nterms = int(np.sum(keep))
    U = U[:, keep]; s = s[keep]; Vt = Vt[keep, :]
    sqrts = np.sqrt(s)
    Lk_list = [(U[:, k] * sqrts[k]).reshape(d, d) for k in range(nterms)]
    Rk_list = [(sqrts[k] * Vt[k, :]).reshape(d, d) for k in range(nterms)]
    DW = nterms + 2
    def make_bulk():
        W = np.zeros((DW, DW, d, d))
        W[0, 0] = Id
        for k in range(nterms):
            W[k+1, 0] = Rk_list[k]
            W[DW-1, k+1] = Lk_list[k]
        W[DW-1, DW-1] = Id
        return W
    def make_first():
        W = np.zeros((1, DW, d, d))
        for k in range(nterms):
            W[0, k+1] = Lk_list[k]
        W[0, DW-1] = Id
        return W
    def make_last():
        W = np.zeros((DW, 1, d, d))
        W[0, 0] = Id
        for k in range(nterms):
            W[k+1, 0] = Rk_list[k]
        return W
    MPO = []
    for i in range(L):
        if i == 0: MPO.append(make_first())
        elif i == L-1: MPO.append(make_last())
        else: MPO.append(make_bulk())
    return MPO

# Replace for all subsequent calls
build_mpo_aklt = build_mpo_aklt_clean

# ================================================================
# GENERATE ALL FIGURES
# ================================================================
print("\n" + "="*70)
print("  GENERATING ALL FIGURES -> figureAKLT/")
print("="*70)

# Ensure all data available; recompute if needed
try:
    _ = ehist_hL, ehist_aL
except NameError:
    # Heisenberg large
    L_h = 48; D_h = 64
    MPO_hL = build_mpo_heisenberg(L_h)
    E_hL, mps_hL, ehist_hL = two_site_dmrg(MPO_hL, L_h, 2, D_h, 10, verbose=True)
    Sz2 = np.diag([0.5, -0.5])
    sz_prof_h = [measure_local_op(mps_hL, Sz2, i) for i in range(L_h)]
    ee_h = [compute_entanglement_entropy(mps_hL, b) for b in range(1, L_h)]
    center_h = L_h//2; corr_h=[]; dist_h=[]
    for r in range(1, L_h//2-1):
        j=center_h+r
        if j<L_h:
            c=measure_two_point(mps_hL,Sz2,center_h,Sz2,j)
            s1v=measure_local_op(mps_hL,Sz2,center_h)
            s2v=measure_local_op(mps_hL,Sz2,j)
            corr_h.append(c-s1v*s2v); dist_h.append(r)
    L_cv=32; D_list=[8,16,32,48,64]; E_cv_h=[]
    for Dv in D_list:
        m=build_mpo_heisenberg(L_cv)
        e,_,_=two_site_dmrg(m,L_cv,2,Dv,8,verbose=False)
        E_cv_h.append(e)
    # AKLT large
    L_a=48; D_a=32
    MPO_aL=build_mpo_aklt(L_a)
    E_aL,mps_aL,ehist_aL=two_site_dmrg(MPO_aL,L_a,3,D_a,12,verbose=True)
    Sz1=np.diag([1.,0.,-1.])
    sz_prof_a=[measure_local_op(mps_aL,Sz1,i) for i in range(L_a)]
    ee_a=[compute_entanglement_entropy(mps_aL,b) for b in range(1,L_a)]
    center_a=L_a//2; corr_a=[]; dist_a=[]
    for r in range(1,L_a//2-1):
        j=center_a+r
        if j<L_a:
            c=measure_two_point(mps_aL,Sz1,center_a,Sz1,j)
            s1v=measure_local_op(mps_aL,Sz1,center_a)
            s2v=measure_local_op(mps_aL,Sz1,j)
            corr_a.append(c-s1v*s2v); dist_a.append(r)
    string_a=[]; dist_string=[]
    for r in range(2,L_a//2-1):
        j=center_a+r
        if j<L_a:
            string_a.append(measure_string_order(mps_aL,center_a,j))
            dist_string.append(r)
    L_cv_a=24; D_list_a=[4,8,16,24,32]; E_cv_a=[]
    for Dv in D_list_a:
        m=build_mpo_aklt(L_cv_a)
        e,_,_=two_site_dmrg(m,L_cv_a,3,Dv,10,verbose=False)
        E_cv_a.append(e)

E_bethe = 0.25 - np.log(2)

# Convert to arrays
dist_h = np.array(dist_h); corr_h = np.array(corr_h)
dist_a = np.array(dist_a); corr_a = np.array(corr_a)
dist_string = np.array(dist_string); string_a = np.array(string_a)

# ---- Fig 1: Energy convergence ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.plot(np.arange(1, len(ehist_hL)+1), ehist_hL, 'bo-', ms=4)
ax.axhline(E_bethe*L_h, color='r', ls='--', label=f'Bethe $\\times L$ = {E_bethe*L_h:.4f}')
ax.set_xlabel('Sweep'); ax.set_ylabel('Energy')
ax.set_title(f'Heisenberg $S=1/2$, $L={L_h}$, $D={D_h}$')
ax.legend(); ax.grid(True, alpha=0.3)
ax = axes[1]
ax.plot(np.arange(1, len(ehist_aL)+1), ehist_aL, 'rs-', ms=4)
ax.axhline(-(2./3.)*(L_a-1), color='b', ls='--', label=f'Exact = {-(2./3.)*(L_a-1):.4f}')
ax.set_xlabel('Sweep'); ax.set_ylabel('Energy')
ax.set_title(f'AKLT $S=1$, $L={L_a}$, $D={D_a}$')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figureAKLT/fig1_energy_convergence.png", dpi=150)
plt.savefig("figureAKLT/fig1_energy_convergence.pdf", dpi=150)
plt.show(); print("  Saved fig1")

# ---- Fig 2: Entanglement entropy ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.plot(np.arange(1, L_h), ee_h, 'b.-', ms=3)
ax.set_xlabel('Bond'); ax.set_ylabel('$S_{\\rm vN}$ (bits)')
ax.set_title(f'Heisenberg $L={L_h}$, $D={D_h}$')
ax.grid(True, alpha=0.3)
ax = axes[1]
ax.plot(np.arange(1, L_a), ee_a, 'r.-', ms=3)
ax.axhline(1.0, color='g', ls='--', label=r'AKLT exact (OBC bulk) $=1$')
ax.set_xlabel('Bond'); ax.set_ylabel('$S_{\\rm vN}$ (bits)')
ax.set_title(f'AKLT $L={L_a}$, $D={D_a}$')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figureAKLT/fig2_entanglement.png", dpi=150)
plt.savefig("figureAKLT/fig2_entanglement.pdf", dpi=150)
plt.show(); print("  Saved fig2")

# ---- Fig 3: <Sz> profiles ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(range(L_h), sz_prof_h, 'b.-', ms=3)
axes[0].axhline(0, color='k', ls=':', alpha=0.4)
axes[0].set_xlabel('Site'); axes[0].set_ylabel(r'$\langle S^z_i\rangle$')
axes[0].set_title(f'Heisenberg $L={L_h}$')
axes[0].grid(True, alpha=0.3)
axes[1].plot(range(L_a), sz_prof_a, 'r.-', ms=3)
axes[1].axhline(0, color='k', ls=':', alpha=0.4)
axes[1].set_xlabel('Site'); axes[1].set_ylabel(r'$\langle S^z_i\rangle$')
axes[1].set_title(f'AKLT $L={L_a}$')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figureAKLT/fig3_Sz_profile.png", dpi=150)
plt.savefig("figureAKLT/fig3_Sz_profile.pdf", dpi=150)
plt.show(); print("  Saved fig3")

# ---- Fig 4: Correlations ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
mask_h = np.abs(corr_h) > 1e-16
if np.any(mask_h):
    ax.semilogy(dist_h[mask_h], np.abs(corr_h[mask_h]), 'bo-', ms=4,
                label=r'$|C^{zz}(r)|$')
ax.set_xlabel('$r$'); ax.set_ylabel(r'$|C^{zz}|$')
ax.set_title(f'Heisenberg $L={L_h}$')
ax.legend(); ax.grid(True, alpha=0.3)
ax = axes[1]
mask_a = np.abs(corr_a) > 1e-16
if np.any(mask_a):
    ax.semilogy(dist_a[mask_a], np.abs(corr_a[mask_a]), 'ro-', ms=4, label='DMRG')
r_ref = np.arange(1, 20)
ax.semilogy(r_ref, (4./9.)*(1./3.)**r_ref, 'k--', label=r'$(4/9)(1/3)^r$')
ax.set_xlabel('$r$'); ax.set_ylabel(r'$|C^{zz}|$')
ax.set_title(f'AKLT $L={L_a}$')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figureAKLT/fig4_correlations.png", dpi=150)
plt.savefig("figureAKLT/fig4_correlations.pdf", dpi=150)
plt.show(); print("  Saved fig4")

# ---- Fig 5: String order ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(dist_string, string_a, 'rs-', ms=5, label='DMRG')
ax.axhline(-4./9., color='k', ls='--', lw=1.5, label=r'Exact $-4/9$')
ax.set_xlabel('$|i-j|$'); ax.set_ylabel('String order')
ax.set_title(f'AKLT String Order, $L={L_a}$, $D={D_a}$')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figureAKLT/fig5_string_order.png", dpi=150)
plt.savefig("figureAKLT/fig5_string_order.pdf", dpi=150)
plt.show(); print("  Saved fig5")

# ---- Fig 6: D convergence ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.plot(D_list, np.array(E_cv_h)/L_cv, 'bo-', ms=6)

---
## 🧑‍💻 User


How did you get these codes?

---
## 🤖 Assistant


Great question! Let me explain my reasoning process step by step, connecting each part to the Schollwöck review article:

## How I Derived the Code from the Review Article

### 1. **MPS Representation (Section 4.1)**

The review introduces MPS as:
$$c_{\sigma_1,...,\sigma_L} = A^{\sigma_1} A^{\sigma_2} \cdots A^{\sigma_L}$$

I represented each site tensor as a 3-index array `M[sigma, chi_left, chi_right]`, which stores $A^{\sigma_\ell}_{a_{\ell-1}, a_\ell}$. The left/right normalization conditions (Eqs. 38, 45) guided the `init_mps_random` function — I right-normalize from the last site using SVD, which the review explains in Section 4.4.

### 2. **MPO for Hamiltonians (Section 6.1)**

The review gives the explicit operator-valued W-matrix for the Heisenberg model in Eq. (182):

$$\hat{W}^{[i]} = \begin{pmatrix} \hat{I} & 0 & 0 & 0 & 0 \\ \hat{S}^+ & 0 & 0 & 0 & 0 \\ \hat{S}^- & 0 & 0 & 0 & 0 \\ \hat{S}^z & 0 & 0 & 0 & 0 \\ -h\hat{S}^z & \frac{J}{2}\hat{S}^- & \frac{J}{2}\hat{S}^+ & J^z\hat{S}^z & \hat{I} \end{pmatrix}$$

I directly translated this into `build_mpo_heisenberg`, with first/last sites from Eq. (183). The MPO tensor has 4 indices: `W[bL, bR, sigma, sigma_prime]`.

For **AKLT**, the review doesn't give an explicit MPO, but Section 6.1 explains the general strategy: decompose the bond Hamiltonian into tensor products via SVD. The key insight I applied:

- Build $h_{i,i+1} = \mathbf{S}_i \cdot \mathbf{S}_{i+1} + \frac{1}{3}(\mathbf{S}_i \cdot \mathbf{S}_{i+1})^2$
- **Critical step**: reshape as $(s_1, s_1') \times (s_2, s_2')$ before SVD, giving $h = \sum_k L_k \otimes R_k$
- Build the W-matrix following the same pattern as Heisenberg

The initial bug was that I SVD'd in the wrong index ordering — `(s1,s2) × (s1',s2')` instead of `(s1,s1') × (s2,s2')`. The `transpose(0,2,1,3)` fix corrected this.

### 3. **Environment Tensors (Section 6.2)**

The review defines L and R tensors in Eqs. (190-191) and shows the iterative update in Eq. (194-195) with the graphical network in Fig. 39. My `compute_L_env` and `compute_R_env` follow the optimal bracketing from Eq. (195):

```
F[i] = sum_sigma A[sigma]† * (sum_{sigma',b} W[sigma,sigma'] * (sum_a' F[i-1] * A[sigma']))
```

Three sequential contractions at cost $O(D^3 D_W d)$ each, rather than building the full $D^2 \times D^2$ matrix.

### 4. **Two-Site Effective Hamiltonian (Section 6.3-6.4)**

The review's Eq. (207)-(209) gives the eigenvalue problem. For two-site DMRG, the wavefunction is $\Psi^{\sigma_\ell \sigma_{\ell+1}}_{a_{\ell-1}, a_{\ell+1}}$ (Eq. 217, Fig. 33). My `apply_Heff_twosite` contracts L, W1, W2, R with theta following Eq. (199):

$$H|\psi\rangle = \sum_{b_{\ell-1}} L^{a_{\ell-1},a'_{\ell-1}}_{b_{\ell-1}} \sum_{b_\ell, \sigma'_\ell} W^{\sigma_\ell,\sigma'_\ell}_{b_{\ell-1},b_\ell} \left(\sum_{a'_\ell} R^{a_\ell,a'_\ell}_{b_\ell} \Psi^{\sigma'_\ell}_{a'_{\ell-1},a'_\ell}\right)$$

This is the **matrix-free** approach — I never build H explicitly, only apply it via `matvec`.

### 5. **Lanczos Eigensolver**

The review mentions Lanczos in Section 2.2 and 6.3 as the iterative sparse eigensolver. I implemented standard Lanczos with full reorthogonalization:
- Tridiagonalize via the Lanczos recursion
- Solve the tridiagonal eigenvalue problem with `eigh_tridiagonal`
- Reconstruct the ground state eigenvector

### 6. **SVD Truncation and Sweep Structure (Section 6.4)**

After solving the eigenvalue problem, the review explains (Eq. 219):
- Reshape $\Psi^{\sigma_\ell \sigma_{\ell+1}}_{a_{\ell-1}, a_{\ell+1}}$ to matrix $(a_{\ell-1}\sigma_\ell) \times (\sigma_{\ell+1} a_{\ell+1})$
- SVD and keep D largest singular values
- Right sweep: left site becomes A (left-normalized from U), right site gets $SV^\dagger$
- Left sweep: right site becomes B (right-normalized from $V^\dagger$), left site gets $US$

The sweep pattern follows Section 6.3's algorithm description exactly.

### 7. **Measurements**

- **Local operators**: Section 4.2, Eq. (95) — contract transfer matrices site by site
- **Entanglement entropy**: Section 4.1.1, Eq. (25) — left-canonicalize, SVD at the bond, $S = -\sum s_a^2 \log_2 s_a^2$
- **String order**: Based on the AKLT discussion in Section 4.1.5, using sequential transfer matrix contraction with $e^{i\pi S^z}$ inserted between sites

### Summary of Debugging Process

The main bug was in the AKLT MPO: when decomposing $h_{i,i+1}$ via SVD, the index ordering matters critically. The bond operator acts as $h[(s_1 s_2),(s_1' s_2')]$, but we need the factorization as $\sum_k L_k[s_1,s_1'] \otimes R_k[s_2,s_2']$. This requires reshaping to $(s_1 s_1') \times (s_2 s_2')$ before SVD — exactly the `transpose(0,2,1,3)` that fixed the issue.
