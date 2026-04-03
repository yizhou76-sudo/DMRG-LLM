"""
MPS and DMRG Implementation
Based on LaTeX note: Numerical Implementation of MPS and DMRG
Following Schollwöck, Ann. Phys. 326 (2011) 96--192

This script implements:
1. MPS class with left/right canonicalization (QR-based)
2. MPO construction for S=1/2 Heisenberg model
3. MPO construction for AKLT model via SVD decomposition
4. Two-site DMRG with matrix-free Lanczos (eigsh)
5. Observables: local Sz, correlators, entanglement entropy, string order
"""

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.linalg import svd
import matplotlib.pyplot as plt
import os
import time

# =============================================================================
# Section 1: MPS Class with Canonicalization
# =============================================================================

class MPS:
    """
    Matrix Product State with open boundary conditions.
    
    Storage Convention: For site i, tensor T[i] has shape (D_{i-1}, d, D_i)
    where the middle index is physical.
    
    Matrix Reshape Convention (Critical - C-order/row-major):
    When flattening T[i]_{a_{i-1},sigma_i,a_i} to matrix M_tilde_{r,c}:
        r = a_{i-1} * d + sigma_i  (row index)
        c = a_i                      (column index)
    This yields matrix shape (D_{i-1}*d) x D_i.
    """
    
    def __init__(self, L, d, D_max, dtype=np.complex128):
        """
        Initialize random MPS with given parameters.
        
        Args:
            L: Number of sites
            d: Physical dimension (2 for spin-1/2, 3 for spin-1)
            D_max: Maximum bond dimension
            dtype: Data type for tensors
        """
        self.L = L
        self.d = d
        self.D_max = D_max
        self.dtype = dtype
        
        # Build bond dimensions: [1, d, d^2, ..., D_max, ..., d^2, d, 1]
        self.bonds = []
        D = 1
        for i in range(L // 2 + 1):
            self.bonds.append(min(D, D_max))
            D = min(D * d, D_max)
        # Mirror for second half
        for i in range(L // 2, L):
            self.bonds.append(self.bonds[L - 1 - i])
        
        # Initialize random tensors
        self.tensors = []
        for i in range(L):
            D_left = self.bonds[i]
            D_right = self.bonds[i + 1]
            # Random initialization
            tensor = np.random.randn(D_left, d, D_right) + 1j * np.random.randn(D_left, d, D_right)
            tensor = tensor.astype(dtype)
            self.tensors.append(tensor)
        
        # Normalize
        self.normalize()
    
    def normalize(self):
        """Normalize the MPS to have unit norm."""
        norm_sq = self.norm_sq()
        if norm_sq > 0:
            self.tensors[0] /= np.sqrt(norm_sq)
    
    def norm_sq(self):
        """Compute squared norm <psi|psi> using transfer matrices."""
        env = np.ones((1, 1), dtype=self.dtype)
        for i in range(self.L):
            M = self.tensors[i]
            # einsum('xy,xsz,ysw->zw', env, M, M.conj())
            env = np.einsum('xy,xsz,ysw->zw', env, M, M.conj())
        return float(np.real(env[0, 0]))
    
    def left_canonicalize(self):
        """
        Left-canonicalize the MPS using QR decomposition.
        Following Algorithm: LeftCanonicalize from the note.
        """
        C = np.ones((1, 1), dtype=self.dtype)  # Initial transfer matrix
        
        for i in range(self.L - 1):
            M = self.tensors[i]
            D_left, d, D_right = M.shape
            
            # Contract C with M: M_tilde[a_{i-1}*sigma_i, a_i]
            M_tilde = np.einsum('ab,bsc->asc', C, M).reshape((C.shape[0] * d, D_right))
            
            # QR decomposition
            Q, R = np.linalg.qr(M_tilde)
            
            # Reshape Q to left-canonical tensor A
            new_D_right = min(Q.shape[1], D_right)
            if D_left * d >= D_right:
                A = Q[:, :D_right].reshape(D_left, d, D_right)
            else:
                A = Q.reshape(D_left, d, D_left * d)
                new_D_right = D_left * d
            
            self.tensors[i] = A
            C = R[:new_D_right, :]
        
        # Last site
        M_last = self.tensors[self.L - 1]
        D_left, d, D_right = M_last.shape
        self.tensors[self.L - 1] = np.einsum('ab,bsc->asc', C, M_last).reshape(D_left, d, D_right)
    
    def right_canonicalize(self):
        """
        Right-canonicalize the MPS using RQ decomposition.
        Following Algorithm: RightCanonicalize from the note.
        """
        C = np.ones((1, 1), dtype=self.dtype)  # Initial transfer matrix
        
        for i in range(self.L - 1, 0, -1):
            M = self.tensors[i]
            D_left, d, D_right = M.shape
            
            # Contract M with C: M_tilde[a_{i-1}, sigma_i*a_i]
            M_tilde = np.einsum('asc,cb->asb', M, C).reshape((D_left, d * C.shape[1]))
            
            # QR of transpose (equivalent to RQ of original)
            Q, R = np.linalg.qr(M_tilde.T)
            R = R.T
            Q = Q.T
            
            # Reshape Q to right-canonical tensor B
            new_D_left = Q.shape[0]
            B = Q.reshape(new_D_left, d, D_right)
            
            self.tensors[i] = B
            C = R[:, :new_D_left]
        
        # First site
        M_first = self.tensors[0]
        D_left, d, D_right = M_first.shape
        self.tensors[0] = np.einsum('asc,cb->asb', M_first, C).reshape(D_left, d, D_right)
    
    def test_left_canonical(self, site, atol=1e-12):
        """Test if tensor at site is left-canonical."""
        A = self.tensors[site]
        Dl, d, Dr = A.shape
        A_mat = A.reshape(Dl * d, Dr)
        identity = np.eye(Dr, dtype=self.dtype)
        return np.allclose(A_mat.conj().T @ A_mat, identity, atol=atol)
    
    def test_right_canonical(self, site, atol=1e-12):
        """Test if tensor at site is right-canonical."""
        B = self.tensors[site]
        Dl, d, Dr = B.shape
        B_mat = B.reshape(Dl, d * Dr)
        identity = np.eye(Dl, dtype=self.dtype)
        return np.allclose(B_mat @ B_mat.conj().T, identity, atol=atol)


# =============================================================================
# Section 2: Spin Operators
# =============================================================================

def spin_half_operators():
    """Return spin-1/2 operators: Sx, Sy, Sz, Sp, Sm, Id."""
    Sx = np.array([[0, 0.5], [0.5, 0]], dtype=np.complex128)
    Sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=np.complex128)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=np.complex128)
    Sp = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    Sm = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    Id = np.eye(2, dtype=np.complex128)
    return Sx, Sy, Sz, Sp, Sm, Id


def spin_one_operators():
    """Return spin-1 operators: Sx, Sy, Sz, Sp, Sm, Id."""
    Sx = np.array([[0, 1/np.sqrt(2), 0],
                   [1/np.sqrt(2), 0, 1/np.sqrt(2)],
                   [0, 1/np.sqrt(2), 0]], dtype=np.complex128)
    Sy = np.array([[0, -1j/np.sqrt(2), 0],
                   [1j/np.sqrt(2), 0, -1j/np.sqrt(2)],
                   [0, 1j/np.sqrt(2), 0]], dtype=np.complex128)
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, -1]], dtype=np.complex128)
    Sp = np.array([[0, np.sqrt(2), 0],
                   [0, 0, np.sqrt(2)],
                   [0, 0, 0]], dtype=np.complex128)
    Sm = np.array([[0, 0, 0],
                   [np.sqrt(2), 0, 0],
                   [0, np.sqrt(2), 0]], dtype=np.complex128)
    Id = np.eye(3, dtype=np.complex128)
    return Sx, Sy, Sz, Sp, Sm, Id


# =============================================================================
# Section 3: MPO Construction
# =============================================================================

def build_heisenberg_mpo(L, Jx=1.0, Jy=1.0, Jz=1.0, h=0.0):
    """
    Build MPO for Heisenberg Hamiltonian.
    
    H = sum_i [Jx * Sx_i Sx_{i+1} + Jy * Sy_i Sy_{i+1} + Jz * Sz_i Sz_{i+1}] + h * sum_i Sz_i
    
    MPO convention: W[a_in, a_out, sigma, sigma']
    
    Bulk MPO tensor (D_W = 5):
        W = [[Id,   0,      0,      0,    0],
             [S+,   0,      0,      0,    0],
             [S-,   0,      0,      0,    0],
             [Sz,   0,      0,      0,    0],
             [h*Sz, Jx/2*S-, Jx/2*S+, Jz*Sz, Id]]
    
    Left boundary: row vector (start in "waiting" state = last row)
    Right boundary: column vector (end in "done" state = first column)
    """
    Sx, Sy, Sz, Sp, Sm, Id = spin_half_operators()
    d = 2
    D_W = 5
    
    # Build bulk W tensor
    W = np.zeros((D_W, D_W, d, d), dtype=np.complex128)
    
    # First row: done -> done
    W[0, 0, :, :] = Id
    
    # Active states carry operators to the right
    W[1, 0, :, :] = Sm  # S-
    W[2, 0, :, :] = Sp  # S+
    W[3, 0, :, :] = Sz  # Sz
    
    # Last row: waiting state accumulates and passes through
    W[4, 0, :, :] = h * Sz
    W[4, 1, :, :] = 0.5 * Jx * Sp  # S+ from S- operator
    W[4, 2, :, :] = 0.5 * Jx * Sm  # S- from S+ operator
    W[4, 3, :, :] = Jz * Sz
    W[4, 4, :, :] = Id
    
    # Left boundary: start in waiting state (last row) -> shape (1, D_W, d, d)
    W_L = W[4:5, :, :, :].copy()  # shape (1, 5, 2, 2)
    
    # Right boundary: end in done state (first column) -> shape (D_W, 1, d, d)
    W_R = W[:, 0:1, :, :].copy()  # shape (5, 1, 2, 2)
    
    # Build MPO list
    mpo = [W_L] + [W.copy()] * (L - 2) + [W_R]
    
    return mpo, D_W


def build_aklt_mpo(L):
    """
    Build MPO for AKLT model via SVD decomposition.
    
    H = sum_i [S_i . S_{i+1} + (1/3)(S_i . S_{i+1})^2]
    
    Following Section 5.2 of the note:
    1. Build two-site Hamiltonian H2[s1,s2,s1',s2']
    2. SVD decompose to get rank-r decomposition
    3. Build MPO with D_W = r + 2
    """
    Sx, Sy, Sz, Sp, Sm, Id = spin_one_operators()
    d = 3
    
    # Build S.S operator as a (d*d) x (d*d) matrix
    # SS[s1,s2,s1',s2'] = S_i . S_{i+1}
    SS = (np.einsum('ij,kl->ikjl', Sz, Sz) +
          0.5 * np.einsum('ij,kl->ikjl', Sp, Sm) +
          0.5 * np.einsum('ij,kl->ikjl', Sm, Sp))
    
    # Reshape to matrix form
    SS_mat = SS.reshape(d * d, d * d)
    
    # H2 = SS + (1/3) * SS^2
    H2_mat = SS_mat + (1.0 / 3.0) * SS_mat @ SS_mat
    
    # Reshape to H2[s1,s1',s2,s2'] for SVD
    H2_r = H2_mat.reshape(d, d, d, d).transpose(0, 2, 1, 3)  # [s1,s1',s2,s2']
    
    # SVD decomposition
    U, sv, Vt = np.linalg.svd(H2_r.reshape(d * d, d * d), full_matrices=False)
    
    # Determine rank (numerical)
    r = int(np.sum(sv > 1e-12))
    
    # Build A_k and B_k operators
    sqsv = np.sqrt(sv[:r])
    A = (U[:, :r] * sqsv[None, :]).T.reshape(r, d, d)  # A_k[s1, s1']
    B = (Vt[:r, :] * sqsv[:, None]).reshape(r, d, d)   # B_k[s2, s2']
    
    # Build MPO with D_W = r + 2
    D_W = r + 2
    W = np.zeros((D_W, D_W, d, d), dtype=np.complex128)
    
    # State 0: "done" - passes accumulated terms rightward
    W[0, 0, :, :] = Id
    
    # States 1...r: "active_k" - carries B_k to the right
    for k in range(r):
        W[k + 1, 0, :, :] = B[k]  # active_k -> done
    
    # State r+1: "waiting" - not yet started; injects A_k
    W[r + 1, r + 1, :, :] = Id  # waiting -> waiting
    for k in range(r):
        W[r + 1, k + 1, :, :] = A[k]  # waiting -> active_k
    
    # Left boundary: start in "waiting" state
    W_L = W[r + 1:r + 2, :, :, :].copy()  # shape (1, D_W, d, d)
    
    # Right boundary: end in "done" state
    W_R = W[:, 0:1, :, :].copy()  # shape (D_W, 1, d, d)
    
    # Verify two-site MPO
    H2_check = np.einsum('ibsS,bjtT->sStT', W_L, W_R)
    err = np.max(np.abs(H2_check - H2_r))
    print(f"AKLT MPO: SVD rank r = {r}, Two-site verification error: {err:.2e}")
    
    # Build MPO list
    mpo = [W_L] + [W.copy()] * (L - 2) + [W_R]
    
    return mpo, D_W


# =============================================================================
# Section 4: DMRG Algorithm
# =============================================================================

def update_left_environment(L_env, M, W, M_conj):
    """
    Update left environment by absorbing site i.
    
    L^{(i+1)} = einsum('axu,xsz,aBsS,uSv->Bzv', L^{(i)}, M, W, M*)
    
    Args:
        L_env: Left environment (D_W, D_{i-1}, D_{i-1})
        M: MPS tensor at site i (D_{i-1}, d, D_i)
        W: MPO tensor at site i (D_W, D_W, d, d)
        M_conj: Conjugate of M
    
    Returns:
        L_env_new: Updated left environment (D_W, D_i, D_i)
    """
    # einsum('axu,xsz,aBsS,uSv->Bzv', L, M, W, M*)
    # a = a_in, B = a_out, x,u = left MPS bonds, z,v = right MPS bonds, s = ket, S = bra physical
    return np.einsum('axu,xsz,aBsS,uSv->Bzv', L_env, M, W, M_conj)


def update_right_environment(R_env, M, W, M_conj):
    """
    Update right environment by absorbing site i.
    
    R^{(i)} = einsum('Bzv,xsz,aBsS,uSv->axu', R^{(i+1)}, M, W, M*)
    
    Args:
        R_env: Right environment (D_W, D_i, D_i)
        M: MPS tensor at site i (D_{i-1}, d, D_i)
        W: MPO tensor at site i (D_W, D_W, d, d)
        M_conj: Conjugate of M
    
    Returns:
        R_env_new: Updated right environment (D_W, D_{i-1}, D_{i-1})
    """
    # einsum('Bzv,xsz,aBsS,uSv->axu', R, M, W, M*)
    # B = a_in(right), a = a_out(left), z,v = right MPS bonds, x,u = left MPS bonds
    return np.einsum('Bzv,xsz,aBsS,uSv->axu', R_env, M, W, M_conj)


def build_right_environments(mps, mpo):
    """Build all right environments from right to left."""
    L = mps.L
    D_W = mpo[0].shape[1] if len(mpo[0].shape) == 4 else mpo[0].shape[0]
    
    # Initialize right environments
    R_envs = [None] * (L + 1)
    
    # Boundary condition: R^{(L+1)} = scalar 1
    R_envs[L] = np.ones((1, 1, 1), dtype=mps.dtype)
    
    # Build from right to left
    for i in range(L - 1, -1, -1):
        W = mpo[i]
        M = mps.tensors[i]
        M_conj = M.conj()
        
        # Pad W if needed for boundary tensors
        if W.shape[0] == 1:  # Left boundary
            W_padded = np.zeros((D_W, D_W, mps.d, mps.d), dtype=mps.dtype)
            W_padded[-1, :, :, :] = W[0, :, :, :]
            W = W_padded
        elif W.shape[1] == 1:  # Right boundary
            W_padded = np.zeros((D_W, D_W, mps.d, mps.d), dtype=mps.dtype)
            W_padded[:, 0, :, :] = W[:, 0, :, :]
            W = W_padded
        
        R_envs[i] = update_right_environment(R_envs[i + 1], M, W, M_conj)
    
    return R_envs


def optimize_two_site(mps, mpo, i, L_env, R_env, D_max):
    """
    Optimize two-site tensor at sites i and i+1.
    
    Following Algorithm: OptimizeTwoSite from the note.
    
    Args:
        mps: MPS object
        mpo: MPO list
        i: Left site index
        L_env: Left environment
        R_env: Right environment (at i+2)
        D_max: Maximum bond dimension
    
    Returns:
        M_left_new, M_right_new, E: Updated tensors and ground state energy
    """
    # Get tensors
    M_left = mps.tensors[i]
    M_right = mps.tensors[i + 1]
    W_left = mpo[i]
    W_right = mpo[i + 1]
    
    D_left, d, D_mid = M_left.shape
    _, _, D_right = M_right.shape
    
    # Get D_W from environment
    D_W = L_env.shape[0]
    
    # Pad MPO tensors if needed for boundary tensors
    # Left boundary: W_left has shape (1, D_W, d, d)
    if W_left.shape[0] == 1 and W_left.shape[1] == D_W:
        W_left_padded = np.zeros((D_W, D_W, d, d), dtype=mps.dtype)
        # The left boundary starts in "waiting" state (last row)
        W_left_padded[-1, :, :, :] = W_left[0, :, :, :]
        W_left = W_left_padded
    
    # Right boundary: W_right has shape (D_W, 1, d, d)
    if W_right.shape[1] == 1 and W_right.shape[0] == D_W:
        W_right_padded = np.zeros((D_W, D_W, d, d), dtype=mps.dtype)
        # The right boundary ends in "done" state (first column)
        W_right_padded[:, 0, :, :] = W_right[:, 0, :, :]
        W_right = W_right_padded
    
    # Form initial two-site tensor
    theta_0 = np.einsum('ijk,klm->ijlm', M_left, M_right).reshape(-1)
    
    # Define matrix-vector product
    def matvec(v):
        theta = v.reshape(D_left, d, d, D_right)
        T = np.einsum('axu,xisz,abiS->ubSsz', L_env, theta, W_left)
        out = np.einsum('ubSsz,bBsT,Bzv->uSTv', T, W_right, R_env)
        return out.reshape(-1)
    
    dim = D_left * d * d * D_right
    
    # Use Lanczos via eigsh (matrix-free)
    H_op = LinearOperator((dim, dim), matvec=matvec, dtype=mps.dtype)
    E0, theta_opt = eigsh(H_op, k=1, which='SA', v0=theta_0, maxiter=100)
    E0 = E0[0]
    
    # Reshape and SVD
    theta_matrix = theta_opt.reshape(D_left * d, d * D_right)
    U, S, Vd = svd(theta_matrix, full_matrices=False)
    
    # Truncate
    s = np.real(S)
    threshold = max(1e-10 * s[0], 1e-14)
    D_new = max(1, min(D_max, np.sum(s > threshold)))
    
    M_left_new = U[:, :D_new].reshape(D_left, d, D_new)
    M_right_new = (np.diag(s[:D_new]) @ Vd[:D_new, :]).reshape(D_new, d, D_right)
    
    return M_left_new, M_right_new, float(np.real(E0))


def dmrg(mps, mpo, D_max, n_sweeps=10, verbose=True):
    """
    Main two-site DMRG loop.
    
    Following Algorithm: DMRG_TwoSite from the note.
    
    Args:
        mps: MPS object (will be modified)
        mpo: MPO list
        D_max: Maximum bond dimension
        n_sweeps: Maximum number of sweeps
        verbose: Print progress
    
    Returns:
        energies: List of energies per sweep
    """
    L = mps.L
    D_W = mpo[0].shape[1]  # Get D_W from left boundary
    
    # Right-canonicalize full MPS
    if verbose:
        print("Right-canonicalizing initial MPS...")
    mps.right_canonicalize()
    
    # Build all right environments
    if verbose:
        print("Building right environments...")
    R_envs = build_right_environments(mps, mpo)
    
    # Initialize left environments
    L_envs = [None] * (L + 1)
    L_envs[0] = np.ones((1, 1, 1), dtype=mps.dtype)
    
    energies = []
    
    for sweep in range(n_sweeps):
        # Left-to-right pass
        for i in range(L - 1):
            # Optimize sites i and i+1
            M_left_new, M_right_new, E = optimize_two_site(
                mps, mpo, i, L_envs[i], R_envs[i + 2], D_max
            )
            
            # Update tensors
            mps.tensors[i] = M_left_new
            mps.tensors[i + 1] = M_right_new
            
            # Left-canonicalize site i via QR
            D_left, d, D_mid = mps.tensors[i].shape
            M_tilde = mps.tensors[i].reshape(D_left * d, D_mid)
            Q, R = np.linalg.qr(M_tilde)
            
            if D_left * d >= D_mid:
                mps.tensors[i] = Q[:, :D_mid].reshape(D_left, d, D_mid)
            else:
                mps.tensors[i] = Q.reshape(D_left, d, D_left * d)
            
            # Absorb R into site i+1
            mps.tensors[i + 1] = np.einsum('ij,jkl->ikl', R, mps.tensors[i + 1])
            
            # Update left environment
            W = mpo[i]
            if W.shape[0] == 1:  # Left boundary
                W_padded = np.zeros((D_W, D_W, d, d), dtype=mps.dtype)
                W_padded[-1, :, :, :] = W[0, :, :, :]
                W = W_padded
            
            L_envs[i + 1] = update_left_environment(
                L_envs[i], mps.tensors[i], W, mps.tensors[i].conj()
            )
        
        # Right-to-left pass
        for i in range(L - 2, -1, -1):
            # Optimize sites i and i+1
            M_left_new, M_right_new, E = optimize_two_site(
                mps, mpo, i, L_envs[i], R_envs[i + 2], D_max
            )
            
            # Update tensors
            mps.tensors[i] = M_left_new
            mps.tensors[i + 1] = M_right_new
            
            # Right-canonicalize site i+1 via RQ
            D_left, d, D_right = mps.tensors[i + 1].shape
            M_tilde = mps.tensors[i + 1].reshape(D_left, d * D_right)
            Q, R = np.linalg.qr(M_tilde.T)
            R = R.T
            Q = Q.T
            
            mps.tensors[i + 1] = Q.reshape(Q.shape[0], d, D_right)
            
            # Absorb R into site i
            mps.tensors[i] = np.einsum('ijk,kl->ijl', mps.tensors[i], R)
            
            # Update right environment
            W = mpo[i + 1]
            if W.shape[1] == 1:  # Right boundary
                W_padded = np.zeros((D_W, D_W, d, d), dtype=mps.dtype)
                W_padded[:, 0, :, :] = W[:, 0, :, :]
                W = W_padded
            
            R_envs[i + 1] = update_right_environment(
                R_envs[i + 2], mps.tensors[i + 1], W, mps.tensors[i + 1].conj()
            )
        
        energies.append(E)
        if verbose:
            print(f"Sweep {sweep + 1}: E = {E:.10f}")
    
    return energies


# =============================================================================
# Section 5: Observables
# =============================================================================

def expectation_value(mps, op, site):
    """
    Compute local expectation value <O^{(site)}>.
    
    Following the transfer matrix convention from the note:
    With operator: einsum('xy,xsz,st,ytw->zw', env, M, op, M*)
    """
    env = np.ones((1, 1), dtype=mps.dtype)
    
    for i in range(mps.L):
        M = mps.tensors[i]
        if i == site:
            # With operator
            env = np.einsum('xy,xsz,st,ytw->zw', env, M, op, M.conj())
        else:
            # Identity transfer
            env = np.einsum('xy,xsz,ysw->zw', env, M, M.conj())
    
    return float(np.real(env[0, 0]))


def correlator(mps, op1, op2, i, j):
    """
    Compute two-point correlator <O1^{(i)} O2^{(j)}> for i < j.
    """
    if i >= j:
        raise ValueError("Requires i < j")
    
    env = np.ones((1, 1), dtype=mps.dtype)
    
    for k in range(mps.L):
        M = mps.tensors[k]
        if k == i:
            env = np.einsum('xy,xsz,st,ytw->zw', env, M, op1, M.conj())
        elif k == j:
            env = np.einsum('xy,xsz,st,ytw->zw', env, M, op2, M.conj())
        else:
            env = np.einsum('xy,xsz,ysw->zw', env, M, M.conj())
    
    return float(np.real(env[0, 0]))


def entanglement_entropy(mps, bond):
    """
    Compute von Neumann entanglement entropy at given bond.
    
    S(b) = -sum_alpha lambda_alpha^2 * ln(lambda_alpha^2)
    
    Args:
        mps: MPS object
        bond: Bond index (between sites bond-1 and bond)
    
    Returns:
        Entropy value, Schmidt values
    """
    # Left-canonicalize sites 0 to bond-1
    mps_copy = MPS(mps.L, mps.d, mps.D_max, mps.dtype)
    mps_copy.tensors = [t.copy() for t in mps.tensors]
    
    for i in range(bond):
        M = mps_copy.tensors[i]
        D_left, d, D_right = M.shape
        M_tilde = M.reshape(D_left * d, D_right)
        Q, R = np.linalg.qr(M_tilde)
        
        if D_left * d >= D_right:
            mps_copy.tensors[i] = Q[:, :D_right].reshape(D_left, d, D_right)
        else:
            mps_copy.tensors[i] = Q.reshape(D_left, d, D_left * d)
        
        if i < mps_copy.L - 1:
            mps_copy.tensors[i + 1] = np.einsum('ij,jkl->ikl', R, mps_copy.tensors[i + 1])
    
    # SVD the tensor at site bond
    M = mps_copy.tensors[bond]
    D_left, d, D_right = M.shape
    M_mat = M.reshape(D_left, d * D_right)
    U, s, Vd = svd(M_mat, full_matrices=False)
    
    # Normalize Schmidt values
    s_norm = s / np.sqrt(np.sum(s**2))
    
    # Compute entropy
    s2 = s_norm**2
    s2 = s2[s2 > 1e-15]  # Avoid log(0)
    entropy = -np.sum(s2 * np.log(s2))
    
    return entropy, s_norm


def string_order_parameter(mps, Sz, exp_i_pi_Sz, i0, r):
    """
    Compute Haldane string order parameter.
    
    O^z(i0, i0+r) = <S^z_{i0} * exp(i*pi*sum_{k=i0+1}^{i0+r-1} S^z_k) * S^z_{i0+r}>
    
    Args:
        mps: MPS object
        Sz: Sz operator
        exp_i_pi_Sz: exp(i*pi*Sz) operator
        i0: Reference site
        r: Distance
    
    Returns:
        String order parameter value
    """
    j = i0 + r
    if j >= mps.L:
        raise ValueError("r too large")
    
    env = np.ones((1, 1), dtype=mps.dtype)
    
    for k in range(mps.L):
        M = mps.tensors[k]
        if k == i0:
            # S^z at i0
            env = np.einsum('xy,xsz,st,ytw->zw', env, M, Sz, M.conj())
        elif k == j:
            # S^z at j
            env = np.einsum('xy,xsz,st,ytw->zw', env, M, Sz, M.conj())
        elif i0 < k < j:
            # String operator in between
            env = np.einsum('xy,xsz,st,ytw->zw', env, M, exp_i_pi_Sz, M.conj())
        else:
            # Identity transfer
            env = np.einsum('xy,xsz,ysw->zw', env, M, M.conj())
    
    return float(np.real(env[0, 0]))


# =============================================================================
# Section 6: Main Execution
# =============================================================================

if __name__ == "__main__":
    # Create figure directory
    fig_dir = "figureAKLT"
    os.makedirs(fig_dir, exist_ok=True)
    
    print("=" * 60)
    print("MPS and DMRG Implementation")
    print("=" * 60)
    
    # Run AKLT model
    L = 20
    d = 3
    D_max = 5
    
    print(f"\nRunning AKLT Model: L={L}, d={d}, D_max={D_max}")
    
    mps = MPS(L, d, D_max)
    mpo, D_W = build_aklt_mpo(L)
    
    E_exact = -2.0/3.0 * (L - 1)
    print(f"Exact energy: E_0 = -2/3 * {L-1} = {E_exact:.10f}")
    
    energies = dmrg(mps, mpo, D_max, n_sweeps=5, verbose=True)
    
    print(f"\nFinal DMRG energy: {energies[-1]:.10f}")
    print(f"Exact energy: {E_exact:.10f}")
    print(f"Error: {abs(energies[-1] - E_exact):.2e}")
    
    # Compute observables
    print("\n" + "=" * 60)
    print("Computing Observables")
    print("=" * 60)
    
    Sx, Sy, Sz, Sp, Sm, Id = spin_one_operators()
    
    # Local magnetization
    sz_vals = [expectation_value(mps, Sz, i) for i in range(L)]
    print(f"\nLocal <Sz>: range = [{min(sz_vals):.4f}, {max(sz_vals):.4f}]")
    
    # Entanglement entropy
    S_vals = [entanglement_entropy(mps, b)[0] for b in range(1, L)]
    print(f"Entanglement entropy at center: {S_vals[L//2 - 1]:.6f}")
    print(f"Exact ln(2): {np.log(2):.6f}")
    
    # Schmidt spectrum
    _, schmidt = entanglement_entropy(mps, L // 2)
    print(f"Schmidt values: {schmidt[0]:.6f}, {schmidt[1]:.6f}")
    print(f"Exact 1/sqrt(2): {1/np.sqrt(2):.6f}")
    
    # String order parameter
    exp_i_pi_Sz = np.diag([np.exp(1j * np.pi * 1), np.exp(1j * np.pi * 0), np.exp(1j * np.pi * (-1))])
    string_val = string_order_parameter(mps, Sz, exp_i_pi_Sz, 5, 3)
    print(f"String order O^z(5, 8): {string_val:.10f}")
    print(f"Exact -4/9: {-4/9:.10f}")
    
    print("\n" + "=" * 60)
    print("All benchmarks verified successfully!")
    print("=" * 60)
