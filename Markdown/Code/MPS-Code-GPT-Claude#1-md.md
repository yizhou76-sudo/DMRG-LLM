# MPS-Code-GPT-Claude#1-md

## 🧑‍💻 User

This is a latex note on MPS and DMRG. Could you generate python codes according to it? Please compute both S=1/2 Heisenberg model and AKLT model, and make sure that a "scalable" matrix-free approach with LANCZOS is adapted. I work with Jupyter. Please show all figures and save them in directory "figureAKLT". Please stay strictly with the latex note implementation. Separte figures from others in two cells. No figure is needed at this stage.

---
## 🤖 Assistant

I'll implement the MPS and DMRG code according to the LaTeX notes, computing both S=1/2 Heisenberg and AKLT models with matrix-free Lanczos approach.

```python
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
import os

# Create output directory
os.makedirs("figureAKLT", exist_ok=True)

# ============================================================================
# SPIN OPERATORS
# ============================================================================

def spin_half_operators():
    """Spin-1/2 operators"""
    Sp = np.array([[0, 1], [0, 0]], dtype=complex)
    Sm = np.array([[0, 0], [1, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    Id = np.eye(2, dtype=complex)
    return Sp, Sm, Sz, Id

def spin_one_operators():
    """Spin-1 operators"""
    Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=complex)
    Sp = np.sqrt(2) * np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=complex)
    Sm = np.sqrt(2) * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=complex)
    Id = np.eye(3, dtype=complex)
    return Sp, Sm, Sz, Id

# ============================================================================
# MPO CONSTRUCTION
# ============================================================================

def heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0):
    """Heisenberg XXZ MPO with field"""
    Sp, Sm, Sz, Id = spin_half_operators()
    d = 2
    chi = 5
    
    W = []
    
    # Left boundary
    W_left = np.zeros((1, chi, d, d), dtype=complex)
    W_left[0, 0] = -h * Sz
    W_left[0, 1] = J/2 * Sm
    W_left[0, 2] = J/2 * Sp
    W_left[0, 3] = Jz * Sz
    W_left[0, 4] = Id
    W.append(W_left)
    
    # Bulk
    for _ in range(L-2):
        W_bulk = np.zeros((chi, chi, d, d), dtype=complex)
        W_bulk[0, 0] = Id
        W_bulk[1, 0] = Sp
        W_bulk[2, 0] = Sm
        W_bulk[3, 0] = Sz
        W_bulk[4, 0] = -h * Sz
        W_bulk[4, 1] = J/2 * Sm
        W_bulk[4, 2] = J/2 * Sp
        W_bulk[4, 3] = Jz * Sz
        W_bulk[4, 4] = Id
        W.append(W_bulk)
    
    # Right boundary
    W_right = np.zeros((chi, 1, d, d), dtype=complex)
    W_right[0, 0] = Id
    W_right[1, 0] = Sp
    W_right[2, 0] = Sm
    W_right[3, 0] = Sz
    W_right[4, 0] = -h * Sz
    W.append(W_right)
    
    return W

def aklt_mpo(L):
    """AKLT MPO for spin-1 chain"""
    Sp, Sm, Sz, Id = spin_one_operators()
    d = 3
    
    # Define O_a and bar_O_a
    O1 = Sp / np.sqrt(2)
    O2 = Sm / np.sqrt(2)
    O3 = Sz
    
    # Products O_a O_b (9 operators)
    OO = [O1 @ O1, O1 @ O2, O1 @ O3, O2 @ O1, O2 @ O2, O2 @ O3, O3 @ O1, O3 @ O2, O3 @ O3]
    
    chi = 14
    W = []
    
    # Left boundary
    W_left = np.zeros((1, chi, d, d), dtype=complex)
    W_left[0, 0] = Id
    W_left[0, 1] = O1
    W_left[0, 2] = O2
    W_left[0, 3] = O3
    for k in range(9):
        W_left[0, 4+k] = OO[k] / 3
    W.append(W_left)
    
    # Bulk
    for _ in range(L-2):
        W_bulk = np.zeros((chi, chi, d, d), dtype=complex)
        W_bulk[0, 0] = Id
        W_bulk[0, 1] = O1
        W_bulk[0, 2] = O2
        W_bulk[0, 3] = O3
        for k in range(9):
            W_bulk[0, 4+k] = OO[k] / 3
        W_bulk[1, 13] = O1
        W_bulk[2, 13] = O2
        W_bulk[3, 13] = O3
        for k in range(9):
            W_bulk[4+k, 13] = OO[k]
        W_bulk[13, 13] = Id
        W.append(W_bulk)
    
    # Right boundary
    W_right = np.zeros((chi, 1, d, d), dtype=complex)
    W_right[0, 0] = 0
    W_right[1, 0] = O1
    W_right[2, 0] = O2
    W_right[3, 0] = O3
    for k in range(9):
        W_right[4+k, 0] = OO[k]
    W_right[13, 0] = Id
    W.append(W_right)
    
    return W

# ============================================================================
# CANONICALIZATION
# ============================================================================

def left_canonicalize_site(M):
    """QR canonicalization for one site"""
    Dl, d, Dr = M.shape
    M_mat = M.reshape(Dl * d, Dr, order='C')
    Q, R = np.linalg.qr(M_mat, mode='reduced')
    A = Q.reshape(Dl, d, -1, order='C')
    return A, 

---
## 🧑‍💻 User

your output is cut off, pls continue from the cutoff

---
## 🤖 Assistant

```python
def left_canonicalize_site(M):
    """QR canonicalization for one site"""
    Dl, d, Dr = M.shape
    M_mat = M.reshape(Dl * d, Dr, order='C')
    Q, R = np.linalg.qr(M_mat, mode='reduced')
    A = Q.reshape(Dl, d, -1, order='C')
    return A, R

def right_canonicalize_site(M):
    """QR canonicalization for one site (right)"""
    Dl, d, Dr = M.shape
    M_mat = M.reshape(Dl, d * Dr, order='C')
    Q, R = np.linalg.qr(M_mat.T.conj(), mode='reduced')
    B = Q.T.conj().reshape(-1, d, Dr, order='C')
    return B, R.T.conj()

def mixed_canonical(Ms, center):
    """Bring MPS to mixed canonical form with center at site"""
    L = len(Ms)
    Ms = [M.copy() for M in Ms]
    
    # Left sweep
    for i in range(center):
        Ms[i], R = left_canonicalize_site(Ms[i])
        Ms[i+1] = np.einsum('ij,jkl->ikl', R, Ms[i+1])
    
    # Right sweep
    for i in range(L-1, center, -1):
        Ms[i], R = right_canonicalize_site(Ms[i])
        Ms[i-1] = np.einsum('ijk,kl->ijl', Ms[i-1], R)
    
    return Ms

# ============================================================================
# ENVIRONMENTS
# ============================================================================

def init_left_env(chi):
    L = np.zeros((1, 1, 1), dtype=complex)
    L[0, 0, 0] = 1.0
    return L

def init_right_env(chi):
    R = np.zeros((1, 1, 1), dtype=complex)
    R[0, 0, 0] = 1.0
    return R

def update_left_env(L, A, W):
    """Update left environment"""
    return np.einsum('bxy,xsa,bBst,ytc->Bac', L, A, W, A.conj())

def update_right_env(R, B, W):
    """Update right environment"""
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, R, B.conj())

def build_right_envs(Ms, Ws):
    """Build all right environments"""
    L = len(Ms)
    Rs = [None] * (L + 1)
    Rs[L] = init_right_env(Ws[-1].shape[1])
    for i in range(L-1, 0, -1):
        Rs[i] = update_right_env(Rs[i+1], Ms[i], Ws[i])
    return Rs

# ============================================================================
# EFFECTIVE HAMILTONIANS
# ============================================================================

def heff_two_site_matvec(v, L, W1, W2, R, Dl, d1, d2, Dr):
    """Two-site effective Hamiltonian matvec"""
    Theta = v.reshape((Dl, d1, d2, Dr), order='C')
    X = np.einsum('byx,yuvz->bxuvz', L, Theta)
    Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
    Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
    HT = np.einsum('Cxstz,Cza->xsta', Z, R)
    return HT.reshape(Dl * d1 * d2 * Dr, order='C')

# ============================================================================
# TWO-SITE DMRG
# ============================================================================

def two_site_dmrg(Ws, Ms, Dmax, num_sweeps=10):
    """Two-site DMRG"""
    L = len(Ms)
    d = Ms[0].shape[1]
    energies = []
    
    Ms = mixed_canonical(Ms, 0)
    Rs = build_right_envs(Ms, Ws)
    Ls = [None] * (L + 1)
    Ls[0] = init_left_env(Ws[0].shape[0])
    
    for sweep in range(num_sweeps):
        # Left-to-right
        for i in range(L-1):
            # Form two-site tensor
            Theta = np.einsum('ijk,klm->ijlm', Ms[i], Ms[i+1])
            Dl, d1, d2, Dr = Theta.shape
            
            # Solve eigenvalue problem
            Nloc = Dl * d1 * d2 * Dr
            v0 = Theta.reshape(Nloc, order='C')
            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v: heff_two_site_matvec(v, Ls[i], Ws[i], Ws[i+1], Rs[i+2], Dl, d1, d2, Dr),
                dtype=complex
            )
            E, vec = eigsh(Heff, k=1, which='SA', v0=v0, tol=1e-10)
            Theta = vec[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            
            # SVD split
            Theta_mat = Theta.reshape(Dl * d1, d2 * Dr, order='C')
            U, S, Vh = np.linalg.svd(Theta_mat, full_matrices=False)
            
            # Truncate
            Dnew = min(Dmax, len(S))
            U, S, Vh = U[:, :Dnew], S[:Dnew], Vh[:Dnew, :]
            
            # Update tensors
            Ms[i] = U.reshape(Dl, d1, Dnew, order='C')
            Ms[i+1] = (n

---
## 🧑‍💻 User

cut off, pls continue

---
## 🤖 Assistant

```python
            Ms[i+1] = (np.diag(S) @ Vh).reshape(Dnew, d2, Dr, order='C')
            
            # Update left environment
            if i < L - 2:
                Ls[i+1] = update_left_env(Ls[i], Ms[i], Ws[i])
        
        energies.append(E[0].real)
        
        # Right-to-left
        for i in range(L-2, -1, -1):
            # Form two-site tensor
            Theta = np.einsum('ijk,klm->ijlm', Ms[i], Ms[i+1])
            Dl, d1, d2, Dr = Theta.shape
            
            # Solve eigenvalue problem
            Nloc = Dl * d1 * d2 * Dr
            v0 = Theta.reshape(Nloc, order='C')
            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v: heff_two_site_matvec(v, Ls[i], Ws[i], Ws[i+1], Rs[i+2], Dl, d1, d2, Dr),
                dtype=complex
            )
            E, vec = eigsh(Heff, k=1, which='SA', v0=v0, tol=1e-10)
            Theta = vec[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            
            # SVD split
            Theta_mat = Theta.reshape(Dl * d1, d2 * Dr, order='C')
            U, S, Vh = np.linalg.svd(Theta_mat, full_matrices=False)
            
            # Truncate
            Dnew = min(Dmax, len(S))
            U, S, Vh = U[:, :Dnew], S[:Dnew], Vh[:Dnew, :]
            
            # Update tensors
            Ms[i] = (U @ np.diag(S)).reshape(Dl, d1, Dnew, order='C')
            Ms[i+1] = Vh.reshape(Dnew, d2, Dr, order='C')
            
            # Update right environment
            if i > 0:
                Rs[i+1] = update_right_env(Rs[i+2], Ms[i+1], Ws[i+1])
        
        energies.append(E[0].real)
        
        if sweep > 0 and abs(energies[-1] - energies[-3]) < 1e-10:
            break
    
    return Ms, energies

# ============================================================================
# INITIAL MPS
# ============================================================================

def random_mps(L, d, D):
    """Random MPS"""
    Ms = []
    Ms.append(np.random.randn(1, d, min(D, d)) + 1j * np.random.randn(1, d, min(D, d)))
    for i in range(1, L-1):
        Dl = Ms[-1].shape[2]
        Dr = min(D, d**(min(i+1, L-i-1)))
        Ms.append(np.random.randn(Dl, d, Dr) + 1j * np.random.randn(Dl, d, Dr))
    Dl = Ms[-1].shape[2]
    Ms.append(np.random.randn(Dl, d, 1) + 1j * np.random.randn(Dl, d, 1))
    return Ms

def aklt_exact_mps(L):
    """Exact AKLT MPS"""
    Ap = np.array([[0, np.sqrt(2/3)], [0, 0]], dtype=complex)
    A0 = np.array([[-1/np.sqrt(3), 0], [0, 1/np.sqrt(3)]], dtype=complex)
    Am = np.array([[0, 0], [-np.sqrt(2/3), 0]], dtype=complex)
    
    Ms = []
    for i in range(L):
        if i == 0:
            M = np.zeros((1, 3, 2), dtype=complex)
        elif i == L - 1:
            M = np.zeros((2, 3, 1), dtype=complex)
        else:
            M = np.zeros((2, 3, 2), dtype=complex)
        
        M[..., 0, :] = Ap if i == 0 else (Ap if i < L - 1 else Ap[:, 0:1])
        M[..., 1, :] = A0 if i == 0 else (A0 if i < L - 1 else A0[:, 0:1])
        M[..., 2, :] = Am if i == 0 else (Am if i < L - 1 else Am[:, 0:1])
        Ms.append(M)
    
    return Ms

# ============================================================================
# MAIN COMPUTATIONS
# ============================================================================

print("=" * 60)
print("S=1/2 HEISENBERG MODEL")
print("=" * 60)

for L in [10, 20, 30]:
    print(f"\nL = {L}")
    Ws = heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0)
    Ms = random_mps(L, 2, 20)
    Ms, energies = two_site_dmrg(Ws, Ms, Dmax=50, num_sweeps=10)
    E_per_bond = energies[-1] / (L - 1)
    print(f"  Energy: {energies[-1]:.10f}")
    print(f"  Energy per bond: {E_per_bond:.10f}")

print("\n" + "=" * 60)
print("AKLT MODEL")
print("=" * 60)

for L in [10, 20, 30]:
    print(f"\nL = {L}")
    Ws = aklt_mpo(L)
    Ms = random_mps(L, 3, 10)
    Ms, energies = two_site_dmrg(Ws, Ms, Dmax=20, num_sweeps=10)
    E_exact = -2/3 * (L - 1)
    print(f"  Energy (DMRG): {energies[-1]:.10f}")
    print(f"  Energy (exact): {E_exact:.10f}")
    print(f"  Error: {abs(energies

---
## 🧑‍💻 User

cut off, continue

---
## 🤖 Assistant

```python
    print(f"  Error: {abs(energies[-1] - E_exact):.2e}")

# Test exact AKLT MPS
print("\n" + "=" * 60)
print("AKLT EXACT MPS VERIFICATION")
print("=" * 60)

for L in [10, 20, 30]:
    print(f"\nL = {L}")
    Ws = aklt_mpo(L)
    Ms = aklt_exact_mps(L)
    
    # Compute energy by contracting <psi|H|psi>
    Ms = mixed_canonical(Ms, 0)
    Rs = build_right_envs(Ms, Ws)
    L_env = init_left_env(Ws[0].shape[0])
    
    # Contract through the chain
    for i in range(L):
        # Contract with bra
        temp = np.einsum('bxy,xsa->bsay', L_env, Ms[i])
        # Contract with MPO
        temp = np.einsum('bsay,bBst->Btay', temp, Ws[i])
        # Contract with ket
        L_env = np.einsum('Btay,taz->Byz', temp, Ms[i].conj())
    
    E_mps = L_env[0, 0, 0].real
    E_exact = -2/3 * (L - 1)
    print(f"  Energy (exact MPS): {E_mps:.10f}")
    print(f"  Energy (exact): {E_exact:.10f}")
    print(f"  Error: {abs(E_mps - E_exact):.2e}")

print("\n" + "=" * 60)
print("COMPUTATION COMPLETE")
print("=" * 60)
```

---
## 🧑‍💻 User

============================================================
S=1/2 HEISENBERG MODEL
============================================================

L = 10
  Energy: -4.2580352073
  Energy per bond: -0.4731150230

L = 20
  Energy: -8.6824733344
  Energy per bond: -0.4569722808

L = 30

---
## 🤖 Assistant

The code is running correctly! The Heisenberg results look good (energy per bond converging to the expected value around -0.443 for the infinite chain). Let me wait for L=30 to complete, or if you'd like, I can provide a separate cell for plotting results once all computations finish.

Would you like me to:
1. Wait for the current computation to complete
2. Provide plotting code in a separate cell now
3. Optimize the code for faster convergence

Which would you prefer?

---
## 🧑‍💻 User

too slow, 14 mins witout completion. reduce the system size for faster test

---
## 🤖 Assistant

```python
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
import os

# Create output directory
os.makedirs("figureAKLT", exist_ok=True)

# ============================================================================
# SPIN OPERATORS
# ============================================================================

def spin_half_operators():
    """Spin-1/2 operators"""
    Sp = np.array([[0, 1], [0, 0]], dtype=complex)
    Sm = np.array([[0, 0], [1, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    Id = np.eye(2, dtype=complex)
    return Sp, Sm, Sz, Id

def spin_one_operators():
    """Spin-1 operators"""
    Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=complex)
    Sp = np.sqrt(2) * np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=complex)
    Sm = np.sqrt(2) * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=complex)
    Id = np.eye(3, dtype=complex)
    return Sp, Sm, Sz, Id

# ============================================================================
# MPO CONSTRUCTION
# ============================================================================

def heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0):
    """Heisenberg XXZ MPO with field"""
    Sp, Sm, Sz, Id = spin_half_operators()
    d = 2
    chi = 5
    
    W = []
    
    # Left boundary
    W_left = np.zeros((1, chi, d, d), dtype=complex)
    W_left[0, 0] = -h * Sz
    W_left[0, 1] = J/2 * Sm
    W_left[0, 2] = J/2 * Sp
    W_left[0, 3] = Jz * Sz
    W_left[0, 4] = Id
    W.append(W_left)
    
    # Bulk
    for _ in range(L-2):
        W_bulk = np.zeros((chi, chi, d, d), dtype=complex)
        W_bulk[0, 0] = Id
        W_bulk[1, 0] = Sp
        W_bulk[2, 0] = Sm
        W_bulk[3, 0] = Sz
        W_bulk[4, 0] = -h * Sz
        W_bulk[4, 1] = J/2 * Sm
        W_bulk[4, 2] = J/2 * Sp
        W_bulk[4, 3] = Jz * Sz
        W_bulk[4, 4] = Id
        W.append(W_bulk)
    
    # Right boundary
    W_right = np.zeros((chi, 1, d, d), dtype=complex)
    W_right[0, 0] = Id
    W_right[1, 0] = Sp
    W_right[2, 0] = Sm
    W_right[3, 0] = Sz
    W_right[4, 0] = -h * Sz
    W.append(W_right)
    
    return W

def aklt_mpo(L):
    """AKLT MPO for spin-1 chain"""
    Sp, Sm, Sz, Id = spin_one_operators()
    d = 3
    
    # Define O_a and bar_O_a
    O1 = Sp / np.sqrt(2)
    O2 = Sm / np.sqrt(2)
    O3 = Sz
    
    # Products O_a O_b (9 operators)
    OO = [O1 @ O1, O1 @ O2, O1 @ O3, O2 @ O1, O2 @ O2, O2 @ O3, O3 @ O1, O3 @ O2, O3 @ O3]
    
    chi = 14
    W = []
    
    # Left boundary
    W_left = np.zeros((1, chi, d, d), dtype=complex)
    W_left[0, 0] = Id
    W_left[0, 1] = O1
    W_left[0, 2] = O2
    W_left[0, 3] = O3
    for k in range(9):
        W_left[0, 4+k] = OO[k] / 3
    W.append(W_left)
    
    # Bulk
    for _ in range(L-2):
        W_bulk = np.zeros((chi, chi, d, d), dtype=complex)
        W_bulk[0, 0] = Id
        W_bulk[0, 1] = O1
        W_bulk[0, 2] = O2
        W_bulk[0, 3] = O3
        for k in range(9):
            W_bulk[0, 4+k] = OO[k] / 3
        W_bulk[1, 13] = O1
        W_bulk[2, 13] = O2
        W_bulk[3, 13] = O3
        for k in range(9):
            W_bulk[4+k, 13] = OO[k]
        W_bulk[13, 13] = Id
        W.append(W_bulk)
    
    # Right boundary
    W_right = np.zeros((chi, 1, d, d), dtype=complex)
    W_right[0, 0] = 0
    W_right[1, 0] = O1
    W_right[2, 0] = O2
    W_right[3, 0] = O3
    for k in range(9):
        W_right[4+k, 0] = OO[k]
    W_right[13, 0] = Id
    W.append(W_right)
    
    return W

# ============================================================================
# CANONICALIZATION
# ============================================================================

def left_canonicalize_site(M):
    """QR canonicalization for one site"""
    Dl, d, Dr = M.shape
    M_mat = M.reshape(Dl * d, Dr, order='C')
    Q, R = np.linalg.qr(M_mat, mode='reduced')
    A = Q.reshape(Dl, d, -1, order='C')
    return A, R

def right_canonicalize_site(M):
    """QR canonicalization for one site (right)"""
    Dl, d, Dr = M.shape
    M_mat = M.reshape(Dl, d * Dr, order='

---
## 🧑‍💻 User

cutoff, pls continue

---
## 🤖 Assistant

```python
    M_mat = M.reshape(Dl, d * Dr, order='C')
    Q, R = np.linalg.qr(M_mat.T.conj(), mode='reduced')
    B = Q.T.conj().reshape(-1, d, Dr, order='C')
    return B, R.T.conj()

def mixed_canonical(Ms, center):
    """Bring MPS to mixed canonical form with center at site"""
    L = len(Ms)
    Ms = [M.copy() for M in Ms]
    
    for i in range(center):
        Ms[i], R = left_canonicalize_site(Ms[i])
        Ms[i+1] = np.einsum('ij,jkl->ikl', R, Ms[i+1])
    
    for i in range(L-1, center, -1):
        Ms[i], R = right_canonicalize_site(Ms[i])
        Ms[i-1] = np.einsum('ijk,kl->ijl', Ms[i-1], R)
    
    return Ms

# ============================================================================
# ENVIRONMENTS
# ============================================================================

def init_left_env(chi):
    L = np.zeros((1, 1, 1), dtype=complex)
    L[0, 0, 0] = 1.0
    return L

def init_right_env(chi):
    R = np.zeros((1, 1, 1), dtype=complex)
    R[0, 0, 0] = 1.0
    return R

def update_left_env(L, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', L, A, W, A.conj())

def update_right_env(R, B, W):
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, R, B.conj())

def build_right_envs(Ms, Ws):
    L = len(Ms)
    Rs = [None] * (L + 1)
    Rs[L] = init_right_env(Ws[-1].shape[1])
    for i in range(L-1, 0, -1):
        Rs[i] = update_right_env(Rs[i+1], Ms[i], Ws[i])
    return Rs

# ============================================================================
# EFFECTIVE HAMILTONIANS
# ============================================================================

def heff_two_site_matvec(v, L, W1, W2, R, Dl, d1, d2, Dr):
    Theta = v.reshape((Dl, d1, d2, Dr), order='C')
    X = np.einsum('byx,yuvz->bxuvz', L, Theta)
    Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
    Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
    HT = np.einsum('Cxstz,Cza->xsta', Z, R)
    return HT.reshape(Dl * d1 * d2 * Dr, order='C')

# ============================================================================
# TWO-SITE DMRG
# ============================================================================

def two_site_dmrg(Ws, Ms, Dmax, num_sweeps=5):
    L = len(Ms)
    energies = []
    
    Ms = mixed_canonical(Ms, 0)
    Rs = build_right_envs(Ms, Ws)
    Ls = [None] * (L + 1)
    Ls[0] = init_left_env(Ws[0].shape[0])
    
    for sweep in range(num_sweeps):
        for i in range(L-1):
            Theta = np.einsum('ijk,klm->ijlm', Ms[i], Ms[i+1])
            Dl, d1, d2, Dr = Theta.shape
            Nloc = Dl * d1 * d2 * Dr
            v0 = Theta.reshape(Nloc, order='C')
            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v: heff_two_site_matvec(v, Ls[i], Ws[i], Ws[i+1], Rs[i+2], Dl, d1, d2, Dr),
                dtype=complex
            )
            E, vec = eigsh(Heff, k=1, which='SA', v0=v0, tol=1e-8, maxiter=100)
            Theta = vec[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            
            Theta_mat = Theta.reshape(Dl * d1, d2 * Dr, order='C')
            U, S, Vh = np.linalg.svd(Theta_mat, full_matrices=False)
            Dnew = min(Dmax, len(S))
            U, S, Vh = U[:, :Dnew], S[:Dnew], Vh[:Dnew, :]
            
            Ms[i] = U.reshape(Dl, d1, Dnew, order='C')
            Ms[i+1] = (np.diag(S) @ Vh).reshape(Dnew, d2, Dr, order='C')
            
            if i < L - 2:
                Ls[i+1] = update_left_env(Ls[i], Ms[i], Ws[i])
        
        energies.append(E[0].real)
        
        for i in range(L-2, -1, -1):
            Theta = np.einsum('ijk,klm->ijlm', Ms[i], Ms[i+1])
            Dl, d1, d2, Dr = Theta.shape
            Nloc = Dl * d1 * d2 * Dr
            v0 = Theta.reshape(Nloc, order='C')
            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v: heff_two_site_matvec(v, Ls[i], Ws[i], Ws[i+1], Rs[i+2], Dl, d1, d2, Dr),
                dtype=complex
            )
            E, vec = eigsh(Heff, k=1, which='SA', v0=v0, tol=1e-8, maxiter=100)
            Theta = vec[:, 0].reshape((Dl, d1, d2,

---
## 🧑‍💻 User

cut off, continue

---
## 🤖 Assistant

```python
            Theta = vec[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            
            Theta_mat = Theta.reshape(Dl * d1, d2 * Dr, order='C')
            U, S, Vh = np.linalg.svd(Theta_mat, full_matrices=False)
            Dnew = min(Dmax, len(S))
            U, S, Vh = U[:, :Dnew], S[:Dnew], Vh[:Dnew, :]
            
            Ms[i] = (U @ np.diag(S)).reshape(Dl, d1, Dnew, order='C')
            Ms[i+1] = Vh.reshape(Dnew, d2, Dr, order='C')
            
            if i > 0:
                Rs[i+1] = update_right_env(Rs[i+2], Ms[i+1], Ws[i+1])
        
        energies.append(E[0].real)
        if sweep > 0 and abs(energies[-1] - energies[-3]) < 1e-9:
            break
    
    return Ms, energies

# ============================================================================
# INITIAL MPS
# ============================================================================

def random_mps(L, d, D):
    Ms = []
    Ms.append(np.random.randn(1, d, min(D, d)) + 1j * np.random.randn(1, d, min(D, d)))
    for i in range(1, L-1):
        Dl = Ms[-1].shape[2]
        Dr = min(D, d**(min(i+1, L-i-1)))
        Ms.append(np.random.randn(Dl, d, Dr) + 1j * np.random.randn(Dl, d, Dr))
    Dl = Ms[-1].shape[2]
    Ms.append(np.random.randn(Dl, d, 1) + 1j * np.random.randn(Dl, d, 1))
    return Ms

def aklt_exact_mps(L):
    Ap = np.array([[0, np.sqrt(2/3)], [0, 0]], dtype=complex)
    A0 = np.array([[-1/np.sqrt(3), 0], [0, 1/np.sqrt(3)]], dtype=complex)
    Am = np.array([[0, 0], [-np.sqrt(2/3), 0]], dtype=complex)
    
    Ms = []
    for i in range(L):
        if i == 0:
            M = np.zeros((1, 3, 2), dtype=complex)
            M[0, 0, :] = Ap[0, :]
            M[0, 1, :] = A0[0, :]
            M[0, 2, :] = Am[0, :]
        elif i == L - 1:
            M = np.zeros((2, 3, 1), dtype=complex)
            M[:, 0, 0] = Ap[:, 0]
            M[:, 1, 0] = A0[:, 0]
            M[:, 2, 0] = Am[:, 0]
        else:
            M = np.zeros((2, 3, 2), dtype=complex)
            M[:, 0, :] = Ap
            M[:, 1, :] = A0
            M[:, 2, :] = Am
        Ms.append(M)
    return Ms

# ============================================================================
# MAIN COMPUTATIONS
# ============================================================================

print("=" * 60)
print("S=1/2 HEISENBERG MODEL")
print("=" * 60)

for L in [6, 8, 10]:
    print(f"\nL = {L}")
    Ws = heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0)
    Ms = random_mps(L, 2, 20)
    Ms, energies = two_site_dmrg(Ws, Ms, Dmax=30, num_sweeps=5)
    E_per_bond = energies[-1] / (L - 1)
    print(f"  Energy: {energies[-1]:.10f}")
    print(f"  Energy per bond: {E_per_bond:.10f}")

print("\n" + "=" * 60)
print("AKLT MODEL")
print("=" * 60)

for L in [6, 8, 10]:
    print(f"\nL = {L}")
    Ws = aklt_mpo(L)
    Ms = random_mps(L, 3, 10)
    Ms, energies = two_site_dmrg(Ws, Ms, Dmax=15, num_sweeps=5)
    E_exact = -2/3 * (L - 1)
    print(f"  Energy (DMRG): {energies[-1]:.10f}")
    print(f"  Energy (exact): {E_exact:.10f}")
    print(f"  Error: {abs(energies[-1] - E_exact):.2e}")

print("\n" + "=" * 60)
print("AKLT EXACT MPS VERIFICATION")
print("=" * 60)

for L in [6, 8, 10]:
    print(f"\nL = {L}")
    Ws = aklt_mpo(L)
    Ms = aklt_exact_mps(L)
    Ms = mixed_canonical(Ms, 0)
    Rs = build_right_envs(Ms, Ws)
    L_env = init_left_env(Ws[0].shape[0])
    
    for i in range(L):
        temp = np.einsum('bxy,xsa->bsay', L_env, Ms[i])
        temp = np.einsum('bsay,bBst->Btay', temp, Ws[i])
        L_env = np.einsum('Btay,taz->Byz', temp, Ms[i].conj())
    
    E_mps = L_env[0, 0, 0].real
    E_exact = -2/3 * (L - 1)
    print(f"  Energy (exact MPS): {E_mps:.10f}")
    print(f"  Energy (exact): {E_exact:.10f}")
    print(f"  Error: {abs(E_mps - E_exact):.2e}")

print("\n" + "=" * 60)
print("COMPUTATION COMPLETE")
print("=" * 60)
```

---
## 🧑‍💻 User

============================================================
S=1/2 HEISENBERG MODEL
============================================================

L = 6
  Energy: -2.4935771339
  Energy per bond: -0.4987154268

L = 8
  Energy: -3.3749325987
  Energy per bond: -0.4821332284

L = 10
  Energy: -4.2580352073
  Energy per bond: -0.4731150230

============================================================
AKLT MODEL
============================================================

L = 6
  Energy (DMRG): -3.3333333333
  Energy (exact): -3.3333333333
  Error: 1.42e-14

L = 8---------------------------------------------------------------------------
ArpackNoConvergence                       Traceback (most recent call last)
Cell In[3], line 340
    338 Ws = aklt_mpo(L)
    339 Ms = random_mps(L, 3, 10)
--> 340 Ms, energies = two_site_dmrg(Ws, Ms, Dmax=15, num_sweeps=5)
    341 E_exact = -2/3 * (L - 1)
    342 print(f"  Energy (DMRG): {energies[-1]:.10f}")

Cell In[3], line 229, in two_site_dmrg(Ws, Ms, Dmax, num_sweeps)
    223 v0 = Theta.reshape(Nloc, order='C')
    224 Heff = LinearOperator(
    225     shape=(Nloc, Nloc),
    226     matvec=lambda v: heff_two_site_matvec(v, Ls[i], Ws[i], Ws[i+1], Rs[i+2], Dl, d1, d2, Dr),
    227     dtype=complex
    228 )
--> 229 E, vec = eigsh(Heff, k=1, which='SA', v0=v0, tol=1e-8, maxiter=100)
    230 Theta = vec[:, 0].reshape((Dl, d1, d2, Dr), order='C')
    232 Theta_mat = Theta.reshape(Dl * d1, d2 * Dr, order='C')

File c:\Users\yizho\anaconda3\envs\science\Lib\site-packages\scipy\sparse\linalg\_eigen\arpack\arpack.py:1657, in eigsh(A, k, M, sigma, which, v0, ncv, maxiter, tol, return_eigenvectors, Minv, OPinv, mode, rng)
   1655 elif which == 'SA':
   1656     which = 'SR'
-> 1657 ret = eigs(A, k, M=M, sigma=sigma, which=which, v0=v0,
   1658            ncv=ncv, maxiter=maxiter, tol=tol,
   1659            return_eigenvectors=return_eigenvectors, Minv=Minv,
   1660            OPinv=OPinv)
   1662 if return_eigenvectors:
   1663     return ret[0].real, ret[1]

File c:\Users\yizho\anaconda3\envs\science\Lib\site-packages\scipy\sparse\linalg\_eigen\arpack\arpack.py:1428, in eigs(A, k, M, sigma, which, v0, ncv, maxiter, tol, return_eigenvectors, Minv, OPinv, OPpart, rng)
   1423 params = _UnsymmetricArpackParams(n, k, A.dtype.char, matvec, mode,
   1424                                   M_matvec, Minv_matvec, sigma,
   1425                                   ncv, v0, maxiter, which, tol, rng)
   1427 while not params.converged:
-> 1428     params.iterate()
   1430 return params.extract(return_eigenvectors)

File c:\Users\yizho\anaconda3\envs\science\Lib\site-packages\scipy\sparse\linalg\_eigen\arpack\arpack.py:830, in _UnsymmetricArpackParams.iterate(self)
    828     pass
    829 elif self.arpack_dict['info'] == 1:
--> 830     self._raise_no_convergence()
    831 else:
    832     raise ArpackError(info=self.arpack_dict['info'],
    833                       infodict=self.iterate_infodict)

File c:\Users\yizho\anaconda3\envs\science\Lib\site-packages\scipy\sparse\linalg\_eigen\arpack\arpack.py:421, in _ArpackParams._raise_no_convergence(self)
    419     vec = np.zeros((self.n, 0))
    420     k_ok = 0
--> 421 raise ArpackNoConvergence(f"No convergence ({num_iter} iterations, "
    422                           f"{k_ok}/{self.k} eigenvectors converged)",
    423                           ev, vec)

ArpackNoConvergence: ARPACK error -1: No convergence (101 iterations, 0/1 eigenvectors converged)

---
## 🤖 Assistant

The Lanczos solver isn't converging for larger AKLT systems. Let me fix this by increasing maxiter and using better initial guess:

```python
def two_site_dmrg(Ws, Ms, Dmax, num_sweeps=5):
    L = len(Ms)
    energies = []
    
    Ms = mixed_canonical(Ms, 0)
    Rs = build_right_envs(Ms, Ws)
    Ls = [None] * (L + 1)
    Ls[0] = init_left_env(Ws[0].shape[0])
    
    for sweep in range(num_sweeps):
        for i in range(L-1):
            Theta = np.einsum('ijk,klm->ijlm', Ms[i], Ms[i+1])
            Dl, d1, d2, Dr = Theta.shape
            Nloc = Dl * d1 * d2 * Dr
            v0 = Theta.reshape(Nloc, order='C')
            v0 = v0 / np.linalg.norm(v0)
            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v: heff_two_site_matvec(v, Ls[i], Ws[i], Ws[i+1], Rs[i+2], Dl, d1, d2, Dr),
                dtype=complex
            )
            E, vec = eigsh(Heff, k=1, which='SA', v0=v0, tol=1e-6, maxiter=500)
            Theta = vec[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            
            Theta_mat = Theta.reshape(Dl * d1, d2 * Dr, order='C')
            U, S, Vh = np.linalg.svd(Theta_mat, full_matrices=False)
            Dnew = min(Dmax, len(S))
            U, S, Vh = U[:, :Dnew], S[:Dnew], Vh[:Dnew, :]
            
            Ms[i] = U.reshape(Dl, d1, Dnew, order='C')
            Ms[i+1] = (np.diag(S) @ Vh).reshape(Dnew, d2, Dr, order='C')
            
            if i < L - 2:
                Ls[i+1] = update_left_env(Ls[i], Ms[i], Ws[i])
        
        energies.append(E[0].real)
        
        for i in range(L-2, -1, -1):
            Theta = np.einsum('ijk,klm->ijlm', Ms[i], Ms[i+1])
            Dl, d1, d2, Dr = Theta.shape
            Nloc = Dl * d1 * d2 * Dr
            v0 = Theta.reshape(Nloc, order='C')
            v0 = v0 / np.linalg.norm(v0)
            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v: heff_two_site_matvec(v, Ls[i], Ws[i], Ws[i+1], Rs[i+2], Dl, d1, d2, Dr),
                dtype=complex
            )
            E, vec = eigsh(Heff, k=1, which='SA', v0=v0, tol=1e-6, maxiter=500)
            Theta = vec[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            
            Theta_mat = Theta.reshape(Dl * d1, d2 * Dr, order='C')
            U, S, Vh = np.linalg.svd(Theta_mat, full_matrices=False)
            Dnew = min(Dmax, len(S))
            U, S, Vh = U[:, :Dnew], S[:Dnew], Vh[:Dnew, :]
            
            Ms[i] = (U @ np.diag(S)).reshape(Dl, d1, Dnew, order='C')
            Ms[i+1] = Vh.reshape(Dnew, d2, Dr, order='C')
            
            if i > 0:
                Rs[i+1] = update_right_env(Rs[i+2], Ms[i+1], Ws[i+1])
        
        energies.append(E[0].real)
        if sweep > 0 and abs(energies[-1] - energies[-3]) < 1e-9:
            break
    
    return Ms, energies

print("=" * 60)
print("AKLT MODEL (RETRY)")
print("=" * 60)

for L in [6, 8, 10]:
    print(f"\nL = {L}")
    Ws = aklt_mpo(L)
    Ms = random_mps(L, 3, 10)
    Ms, energies = two_site_dmrg(Ws, Ms, Dmax=15, num_sweeps=5)
    E_exact = -2/3 * (L - 1)
    print(f"  Energy (DMRG): {energies[-1]:.10f}")
    print(f"  Energy (exact): {E_exact:.10f}")
    print(f"  Error: {abs(energies[-1] - E_exact):.2e}")

print("\n" + "=" * 60)
print("AKLT EXACT MPS VERIFICATION")
print("=" * 60)

for L in [6, 8, 10]:
    print(f"\nL = {L}")
    Ws = aklt_mpo(L)
    Ms = aklt_exact_mps(L)
    Ms = mixed_canonical(Ms, 0)
    Rs = build_right_envs(Ms, Ws)
    L_env = init_left_env(Ws[0].shape[0])
    
    for i in range(L):
        temp = np.einsum('bxy,xsa->bsay', L_env, Ms[i])
        temp = np.einsum('bsay,bBst->Btay', temp, Ws[i])
        L_env = np.einsum('Btay,taz->Byz', temp, Ms[i].conj())
    
    E_mps = L_env[0, 0, 0].real
    E_exact = -2/3 * (L - 1)
    print(f"  Energy (exact MPS): {E_mps:.10f}")
    print(f"  Energy (exact): {E_exact:.10f}")
    print(f"  Error: {abs(E_mps - E_exact):.2e}")

print("\n" + "=" * 60)
print("COMPUTATION COMPLETE")
print("=" * 60)
```

---
## 🧑‍💻 User

============================================================
AKLT MODEL (RETRY)
============================================================

L = 6
  Energy (DMRG): -3.3333333333
  Energy (exact): -3.3333333333
  Error: 1.33e-15

L = 8
  Energy (DMRG): -4.6666666667
  Energy (exact): -4.6666666667
  Error: 1.07e-14

L = 10
  Energy (DMRG): -6.0000000000
  Energy (exact): -6.0000000000
  Error: 4.44e-15

============================================================
AKLT EXACT MPS VERIFICATION
============================================================

L = 6---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[5], line 306
    304     temp = np.einsum('bxy,xsa->bsay', L_env, Ms[i])
    305     temp = np.einsum('bsay,bBst->Btay', temp, Ws[i])
--> 306     L_env = np.einsum('Btay,taz->Byz', temp, Ms[i].conj())
    308 E_mps = L_env[0, 0, 0].real
    309 E_exact = -2/3 * (L - 1)

File c:\Users\yizho\anaconda3\envs\science\Lib\site-packages\numpy\_core\einsumfunc.py:1423, in einsum(out, optimize, *operands, **kwargs)
   1421     if specified_out:
   1422         kwargs['out'] = out
-> 1423     return c_einsum(*operands, **kwargs)
   1425 # Check the kwargs to avoid a more cryptic error later, without having to
   1426 # repeat default values here
   1427 valid_einsum_kwargs = ['dtype', 'order', 'casting']

ValueError: operands could not be broadcast together with remapped shapes [original->remapped]: (14,3,2,1)->(14,1,newaxis,2,3) (1,3,2)->(2,3,1)

---
## 🤖 Assistant

The energy contraction has a shape mismatch. Let me fix the exact MPS energy calculation:

```python
print("\n" + "=" * 60)
print("AKLT EXACT MPS VERIFICATION")
print("=" * 60)

for L in [6, 8, 10]:
    print(f"\nL = {L}")
    Ws = aklt_mpo(L)
    Ms = aklt_exact_mps(L)
    Ms = mixed_canonical(Ms, 0)
    Rs = build_right_envs(Ms, Ws)
    L_env = init_left_env(Ws[0].shape[0])
    
    for i in range(L):
        temp = np.einsum('bxy,xsa->bsay', L_env, Ms[i])
        temp = np.einsum('bsay,bBst->Btay', temp, Ws[i])
        L_env = np.einsum('Btay,tac->Byc', temp, Ms[i].conj())
    
    E_mps = L_env[0, 0, 0].real
    E_exact = -2/3 * (L - 1)
    print(f"  Energy (exact MPS): {E_mps:.10f}")
    print(f"  Energy (exact): {E_exact:.10f}")
    print(f"  Error: {abs(E_mps - E_exact):.2e}")

print("\n" + "=" * 60)
print("COMPUTATION COMPLETE")
print("=" * 60)
```

---
## 🧑‍💻 User

============================================================
AKLT MODEL (RETRY)
============================================================

L = 6
  Energy (DMRG): -3.3333333333
  Energy (exact): -3.3333333333
  Error: 8.88e-16

L = 8
  Energy (DMRG): -4.6666666667
  Energy (exact): -4.6666666667
  Error: 3.38e-14

L = 10
  Energy (DMRG): -6.0000000000
  Energy (exact): -6.0000000000
  Error: 1.60e-14

============================================================
AKLT EXACT MPS VERIFICATION
============================================================

L = 6---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[6], line 306
    304     temp = np.einsum('bxy,xsa->bsay', L_env, Ms[i])
    305     temp = np.einsum('bsay,bBst->Btay', temp, Ws[i])
--> 306     L_env = np.einsum('Btay,tac->Byc', temp, Ms[i].conj())
    308 E_mps = L_env[0, 0, 0].real
    309 E_exact = -2/3 * (L - 1)

File c:\Users\yizho\anaconda3\envs\science\Lib\site-packages\numpy\_core\einsumfunc.py:1423, in einsum(out, optimize, *operands, **kwargs)
   1421     if specified_out:
   1422         kwargs['out'] = out
-> 1423     return c_einsum(*operands, **kwargs)
   1425 # Check the kwargs to avoid a more cryptic error later, without having to
   1426 # repeat default values here
   1427 valid_einsum_kwargs = ['dtype', 'order', 'casting']

ValueError: operands could not be broadcast together with remapped shapes [original->remapped]: (14,3,2,1)->(14,1,newaxis,2,3) (1,3,2)->(2,3,1)

---
## 🤖 Assistant

The issue is that boundary MPS tensors have different shapes. Let me debug and fix:

```python
print("=" * 60)
print("AKLT EXACT MPS VERIFICATION")
print("=" * 60)

def compute_mps_energy(Ms, Ws):
    """Compute <psi|H|psi> using left environment contraction"""
    L = len(Ms)
    Ms_c = mixed_canonical(Ms, 0)
    
    # Compute norm to normalize
    # Build transfer matrix for norm
    norm_env = np.ones((1, 1), dtype=complex)
    for i in range(L):
        norm_env = np.einsum('xy,xsa,ysa->...', norm_env, Ms_c[i], Ms_c[i].conj())
    
    # Debug: print shapes
    # for i in range(L):
    #     print(f"  site {i}: MPS shape {Ms_c[i].shape}, MPO shape {Ws[i].shape}")
    
    # Build <psi|H|psi> using stored environment convention
    # L_env has shape (chi, D, D)
    L_env = init_left_env(Ws[0].shape[0])
    
    for i in range(L):
        A = Ms_c[i]       # (Dl, d, Dr)
        W = Ws[i]          # (chi_l, chi_r, d, d)
        Ac = A.conj()      # (Dl, d, Dr)
        
        # Step 1: contract L_env with A
        # L_env[b, x, y] * A[x, s, a] -> [b, y, s, a]
        t1 = np.einsum('bxy,xsa->bysa', L_env, A)
        # Step 2: contract with W
        # t1[b, y, s, a] * W[b, B, s, t] -> [B, y, t, a]
        t2 = np.einsum('bysa,bBst->Byta', t1, W)
        # Step 3: contract with A*
        # t2[B, y, t, a] * Ac[y, t, c] -> [B, a, c]
        L_env = np.einsum('Byta,ytc->Bac', t2, Ac)
    
    E = L_env[0, 0, 0].real
    norm = norm_env.real if np.isscalar(norm_env) else norm_env.real.item()
    return E / norm

for L in [6, 8, 10]:
    print(f"\nL = {L}")
    Ws = aklt_mpo(L)
    Ms = aklt_exact_mps(L)
    
    # Print shapes for debugging
    for i in range(L):
        print(f"  site {i}: MPS {Ms[i].shape}, MPO {Ws[i].shape}")
    
    E_mps = compute_mps_energy(Ms, Ws)
    E_exact = -2/3 * (L - 1)
    print(f"  Energy (exact MPS): {E_mps:.10f}")
    print(f"  Energy (exact): {E_exact:.10f}")
    print(f"  Error: {abs(E_mps - E_exact):.2e}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
```

---
## 🧑‍💻 User

============================================================
AKLT MODEL (RETRY)
============================================================

L = 6
  Energy (DMRG): -3.3333333333
  Energy (exact): -3.3333333333
  Error: 3.11e-15

L = 8
  Energy (DMRG): -4.6666666667
  Energy (exact): -4.6666666667
  Error: 1.78e-15

L = 10
  Energy (DMRG): -6.0000000000
  Energy (exact): -6.0000000000
  Error: 3.91e-14
============================================================
AKLT EXACT MPS VERIFICATION
============================================================

L = 6
  site 0: MPS (1, 3, 2), MPO (1, 14, 3, 3)
  site 1: MPS (2, 3, 2), MPO (14, 14, 3, 3)
  site 2: MPS (2, 3, 2), MPO (14, 14, 3, 3)
  site 3: MPS (2, 3, 2), MPO (14, 14, 3, 3)
  site 4: MPS (2, 3, 2), MPO (14, 14, 3, 3)
  site 5: MPS (2, 3, 1), MPO (14, 1, 3, 3)

---
## 🤖 Assistant

Good, shapes look correct now. The norm calculation had a bug. Let me fix:

```python
print("=" * 60)
print("AKLT EXACT MPS VERIFICATION")
print("=" * 60)

def compute_mps_energy(Ms, Ws):
    """Compute <psi|H|psi> / <psi|psi> using left environment contraction"""
    L = len(Ms)
    Ms_c = mixed_canonical(Ms, 0)
    
    # Compute norm <psi|psi>
    norm_env = np.ones((1, 1), dtype=complex)
    for i in range(L):
        A = Ms_c[i]
        # norm_env[x, y] * A[x, s, a] * A*[y, s, c] -> [a, c]
        norm_env = np.einsum('xy,xsa,ysc->ac', norm_env, A, A.conj())
    norm = norm_env[0, 0].real
    
    # Compute <psi|H|psi>
    # L_env has shape (chi, D_bra, D_ket)
    L_env = init_left_env(Ws[0].shape[0])
    
    for i in range(L):
        A = Ms_c[i]
        W = Ws[i]
        Ac = A.conj()
        
        # Use validated environment update formula
        # L_env[b, x, y] A[x, s, a] W[b, B, s, t] Ac[y, t, c] -> L_new[B, a, c]
        L_env = np.einsum('bxy,xsa,bBst,ytc->Bac', L_env, A, W, Ac)
    
    E = L_env[0, 0, 0].real
    return E / norm

for L in [6, 8, 10]:
    print(f"\nL = {L}")
    Ws = aklt_mpo(L)
    Ms = aklt_exact_mps(L)
    
    E_mps = compute_mps_energy(Ms, Ws)
    E_exact = -2/3 * (L - 1)
    print(f"  Energy (exact MPS): {E_mps:.10f}")
    print(f"  Energy (exact): {E_exact:.10f}")
    print(f"  Error: {abs(E_mps - E_exact):.2e}")

# Also run Heisenberg for completeness
print("\n" + "=" * 60)
print("S=1/2 HEISENBERG MODEL")
print("=" * 60)

heisenberg_results = {}
for L in [6, 8, 10, 12]:
    print(f"\nL = {L}")
    Ws = heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0)
    Ms = random_mps(L, 2, 20)
    Ms, energies = two_site_dmrg(Ws, Ms, Dmax=30, num_sweeps=5)
    heisenberg_results[L] = energies
    print(f"  Energy: {energies[-1]:.10f}")
    print(f"  Energy/bond: {energies[-1]/(L-1):.10f}")

print("\n" + "=" * 60)
print("AKLT MODEL DMRG")
print("=" * 60)

aklt_results = {}
for L in [6, 8, 10, 12]:
    print(f"\nL = {L}")
    Ws = aklt_mpo(L)
    Ms = random_mps(L, 3, 10)
    Ms, energies = two_site_dmrg(Ws, Ms, Dmax=15, num_sweeps=5)
    aklt_results[L] = energies
    E_exact = -2/3 * (L - 1)
    print(f"  Energy (DMRG): {energies[-1]:.10f}")
    print(f"  Energy (exact): {E_exact:.10f}")
    print(f"  Error: {abs(energies[-1] - E_exact):.2e}")

print("\n" + "=" * 60)
print("ALL COMPUTATIONS COMPLETE")
print("=" * 60)
```

---
## 🧑‍💻 User

============================================================
AKLT MODEL (RETRY)
============================================================

L = 6
  Energy (DMRG): -3.3333333333
  Energy (exact): -3.3333333333
  Error: 1.15e-14

L = 8
  Energy (DMRG): -4.6666666667
  Energy (exact): -4.6666666667
  Error: 8.88e-16

L = 10
  Energy (DMRG): -6.0000000000
  Energy (exact): -6.0000000000
  Error: 3.55e-15
============================================================
AKLT EXACT MPS VERIFICATION
============================================================

L = 6
  Energy (exact MPS): -0.3926940639
  Energy (exact): -3.3333333333
  Error: 2.94e+00

L = 8
  Energy (exact MPS): -0.5219953266
  Energy (exact): -4.6666666667
  Error: 4.14e+00

L = 10
  Energy (exact MPS): -0.6671634208
  Energy (exact): -6.0000000000
  Error: 5.33e+00

============================================================
S=1/2 HEISENBERG MODEL
============================================================

L = 6
  Energy: -2.4935771339
  Energy/bond: -0.4987154268

L = 8
  Energy: -3.3749325987
  Energy/bond: -0.4821332284

L = 10
  Energy: -4.2580352073
  Energy/bond: -0.4731150230

L = 12
  Energy: -5.1420906328
  Energy/bond: -0.4674627848

============================================================
AKLT MODEL DMRG
============================================================

L = 6
  Energy (DMRG): -3.3333333333
  Energy (exact): -3.3333333333
  Error: 1.07e-14

L = 8
  Energy (DMRG): -4.6666666667
  Energy (exact): -4.6666666667
  Error: 1.95e-14

L = 10
  Energy (DMRG): -6.0000000000
  Energy (exact): -6.0000000000
  Error: 1.15e-14

L = 12
  Energy (DMRG): -7.3333333333
  Energy (exact): -7.3333333333
  Error: 1.33e-14

============================================================
ALL COMPUTATIONS COMPLETE
============================================================

---
## 🤖 Assistant

DMRG results are perfect! The exact MPS energy evaluation has a normalization issue - the AKLT MPS with open boundary conditions and the boundary truncation isn't normalized. Let me fix:

```python
print("=" * 60)
print("AKLT EXACT MPS VERIFICATION (FIXED)")
print("=" * 60)

def compute_mps_energy_v2(Ms, Ws):
    """Compute <psi|H|psi> / <psi|psi> properly"""
    L = len(Ms)
    # Do NOT re-canonicalize: use raw tensors to preserve exact AKLT form
    
    # Compute <psi|psi>
    norm_env = np.ones((1, 1), dtype=complex)
    for i in range(L):
        A = Ms[i]
        norm_env = np.einsum('xy,xsa,ysc->ac', norm_env, A, A.conj())
    norm = norm_env[0, 0].real
    
    # Compute <psi|H|psi>
    L_env = init_left_env(Ws[0].shape[0])
    for i in range(L):
        A = Ms[i]
        Ac = A.conj()
        W = Ws[i]
        L_env = np.einsum('bxy,xsa,bBst,ytc->Bac', L_env, A, W, Ac)
    
    E = L_env[0, 0, 0].real
    return E, norm, E / norm

for L in [4, 6, 8, 10, 12]:
    print(f"\nL = {L}")
    Ws = aklt_mpo(L)
    Ms = aklt_exact_mps(L)
    
    E_raw, norm, E_normalized = compute_mps_energy_v2(Ms, Ws)
    E_exact = -2/3 * (L - 1)
    print(f"  <psi|psi> = {norm:.10f}")
    print(f"  <psi|H|psi> = {E_raw:.10f}")
    print(f"  E = <H>/<psi|psi> = {E_normalized:.10f}")
    print(f"  E_exact = {E_exact:.10f}")
    print(f"  Error: {abs(E_normalized - E_exact):.2e}")

# Verify right-canonical property of bulk AKLT tensors
print("\n" + "=" * 60)
print("AKLT BULK TENSOR RIGHT-CANONICAL CHECK")
print("=" * 60)

Ap = np.array([[0, np.sqrt(2/3)], [0, 0]], dtype=complex)
A0 = np.array([[-1/np.sqrt(3), 0], [0, 1/np.sqrt(3)]], dtype=complex)
Am = np.array([[0, 0], [-np.sqrt(2/3), 0]], dtype=complex)

# sum_sigma A^sigma (A^sigma)^dagger should be identity for right-canonical
check = Ap @ Ap.conj().T + A0 @ A0.conj().T + Am @ Am.conj().T
print("sum_sigma A^sigma (A^sigma)^dag =")
print(check)
print(f"Is identity: {np.allclose(check, np.eye(2))}")

# For D=2 bulk, the MPS tensor is (2,3,2)
M_bulk = np.zeros((2, 3, 2), dtype=complex)
M_bulk[:, 0, :] = Ap
M_bulk[:, 1, :] = A0
M_bulk[:, 2, :] = Am

# Right-canonical check: sum over sigma,alpha_right
rc = np.einsum('isa,jsa->ij', M_bulk, M_bulk.conj())
print(f"\nRight-canonical check (bulk):")
print(rc)
print(f"Is identity: {np.allclose(rc, np.eye(2))}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
```

---
## 🧑‍💻 User

============================================================
AKLT MODEL (RETRY)
============================================================

L = 6
  Energy (DMRG): -3.3333333333
  Energy (exact): -3.3333333333
  Error: 4.44e-16

L = 8
  Energy (DMRG): -4.6666666667
  Energy (exact): -4.6666666667
  Error: 1.42e-14

L = 10
  Energy (DMRG): -6.0000000000
  Energy (exact): -6.0000000000
  Error: 3.57e-13
============================================================
AKLT EXACT MPS VERIFICATION (FIXED)
============================================================

L = 4
  <psi|psi> = 0.5061728395
  <psi|H|psi> = -0.1728395062
  E = <H>/<psi|psi> = -0.3414634146
  E_exact = -2.0000000000
  Error: 1.66e+00

L = 6
  <psi|psi> = 0.5006858711
  <psi|H|psi> = -0.1966163695
  E = <H>/<psi|psi> = -0.3926940639
  E_exact = -3.3333333333
  Error: 2.94e+00

L = 8
  <psi|psi> = 0.5000762079
  <psi|H|psi> = -0.2610374435
  E = <H>/<psi|psi> = -0.5219953266
  E_exact = -4.6666666667
  Error: 4.14e+00

L = 10
  <psi|psi> = 0.5000084675
  <psi|H|psi> = -0.3335873597
  E = <H>/<psi|psi> = -0.6671634208
  E_exact = -6.0000000000
  Error: 5.33e+00

L = 12
  <psi|psi> = 0.5000009408
  <psi|H|psi> = -0.4074419048
  E = <H>/<psi|psi> = -0.8148822763
  E_exact = -7.3333333333
  Error: 6.52e+00

============================================================
AKLT BULK TENSOR RIGHT-CANONICAL CHECK
============================================================
sum_sigma A^sigma (A^sigma)^dag =
[[1.+0.j 0.+0.j]
 [0.+0.j 1.+0.j]]
Is identity: True

Right-canonical check (bulk):
[[1.+0.j 0.+0.j]
 [0.+0.j 1.+0.j]]
Is identity: True

============================================================
VERIFICATION COMPLETE
============================================================

---
## 🤖 Assistant

The bulk tensors are correctly right-canonical, but the boundary truncation is wrong. For OBC with `D=2` bulk, the boundary site tensors need to keep both rows/columns. Let me fix `aklt_exact_mps`:

```python
print("=" * 60)
print("DEBUGGING AKLT EXACT MPS BOUNDARY")
print("=" * 60)

# The bulk AKLT matrices are 2x2
Ap = np.array([[0, np.sqrt(2/3)], [0, 0]], dtype=complex)
A0 = np.array([[-1/np.sqrt(3), 0], [0, 1/np.sqrt(3)]], dtype=complex)
Am = np.array([[0, 0], [-np.sqrt(2/3), 0]], dtype=complex)

# For OBC with D=2 bond dimension, the boundary sites should
# still have bond dimension 2 at the open end.
# Site 0: shape (1, 3, 2) -- pick one row of the 2x2 matrices
# Site L-1: shape (2, 3, 1) -- pick one column of the 2x2 matrices
# But which row/column? We need ALL of them to span the 4-fold degenerate ground space.

# The 4 ground states correspond to choosing boundary row/column indices.
# For a single ground state, pick e.g. row 0 at left, column 0 at right.

# Let's test: use FULL bond dimension at boundaries
# i.e., boundary bond dim = 2, not 1

def aklt_exact_mps_v2(L):
    """Exact AKLT MPS with D=2 everywhere including boundaries"""
    Ap = np.array([[0, np.sqrt(2/3)], [0, 0]], dtype=complex)
    A0 = np.array([[-1/np.sqrt(3), 0], [0, 1/np.sqrt(3)]], dtype=complex)
    Am = np.array([[0, 0], [-np.sqrt(2/3), 0]], dtype=complex)
    
    Ms = []
    for i in range(L):
        M = np.zeros((2, 3, 2), dtype=complex)
        M[:, 0, :] = Ap
        M[:, 1, :] = A0
        M[:, 2, :] = Am
        Ms.append(M)
    
    # For OBC, we need D_0 = D_L = 1
    # Choose specific boundary vectors to project into ground space
    # The 4 ground states are labeled by (left_row, right_col)
    # Pick all 4 and verify each has E = -2/3*(L-1)
    
    results = []
    for row in range(2):
        for col in range(2):
            Ms_obc = []
            # Left boundary: pick row
            M0 = np.zeros((1, 3, 2), dtype=complex)
            M0[0, :, :] = Ms[0][row, :, :]
            Ms_obc.append(M0)
            
            # Bulk
            for i in range(1, L-1):
                Ms_obc.append(Ms[i].copy())
            
            # Right boundary: pick column
            ML = np.zeros((2, 3, 1), dtype=complex)
            ML[:, :, 0] = Ms[-1][:, :, col]
            Ms_obc.append(ML)
            
            results.append((row, col, Ms_obc))
    
    return results

for L in [4, 6, 8, 10]:
    print(f"\nL = {L}")
    Ws = aklt_mpo(L)
    E_exact = -2/3 * (L - 1)
    
    all_states = aklt_exact_mps_v2(L)
    for row, col, Ms in all_states:
        # Compute <psi|psi>
        norm_env = np.ones((1, 1), dtype=complex)
        for i in range(L):
            norm_env = np.einsum('xy,xsa,ysc->ac', norm_env, Ms[i], Ms[i].conj())
        norm = norm_env[0, 0].real
        
        # Compute <psi|H|psi>
        L_env = init_left_env(Ws[0].shape[0])
        for i in range(L):
            L_env = np.einsum('bxy,xsa,bBst,ytc->Bac', L_env, Ms[i], Ws[i], Ms[i].conj())
        E_raw = L_env[0, 0, 0].real
        
        E_norm = E_raw / norm
        print(f"  (row={row},col={col}): norm={norm:.6f}, E={E_norm:.10f}, exact={E_exact:.10f}, err={abs(E_norm-E_exact):.2e}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
```

---
## 🧑‍💻 User

============================================================
DEBUGGING AKLT EXACT MPS BOUNDARY
============================================================

L = 4
  (row=0,col=0): norm=0.506173, E=-0.3414634146, exact=-2.0000000000, err=1.66e+00
  (row=0,col=1): norm=0.493827, E=-0.1000000000, exact=-2.0000000000, err=1.90e+00
  (row=1,col=0): norm=0.493827, E=-0.1000000000, exact=-2.0000000000, err=1.90e+00
  (row=1,col=1): norm=0.506173, E=-0.3414634146, exact=-2.0000000000, err=1.66e+00

L = 6
  (row=0,col=0): norm=0.500686, E=-0.3926940639, exact=-3.3333333333, err=2.94e+00
  (row=0,col=1): norm=0.499314, E=-0.3479853480, exact=-3.3333333333, err=2.99e+00
  (row=1,col=0): norm=0.499314, E=-0.3479853480, exact=-3.3333333333, err=2.99e+00
  (row=1,col=1): norm=0.500686, E=-0.3926940639, exact=-3.3333333333, err=2.94e+00

L = 8
  (row=0,col=0): norm=0.500076, E=-0.5219953266, exact=-4.6666666667, err=4.14e+00
  (row=0,col=1): norm=0.499924, E=-0.5150406504, exact=-4.6666666667, err=4.15e+00
  (row=1,col=0): norm=0.499924, E=-0.5150406504, exact=-4.6666666667, err=4.15e+00
  (row=1,col=1): norm=0.500076, E=-0.5219953266, exact=-4.6666666667, err=4.14e+00

L = 10
  (row=0,col=0): norm=0.500008, E=-0.6671634208, exact=-6.0000000000, err=5.33e+00
  (row=0,col=1): norm=0.499992, E=-0.6661698957, exact=-6.0000000000, err=5.33e+00
  (row=1,col=0): norm=0.499992, E=-0.6661698957, exact=-6.0000000000, err=5.33e+00
  (row=1,col=1): norm=0.500008, E=-0.6671634208, exact=-6.0000000000, err=5.33e+00

============================================================
DONE
============================================================

---
## 🤖 Assistant

The energies are way off - the issue must be in the AKLT MPO `bar_O` operators. Looking at the LaTeX notes carefully, `bar_O_1 = S^-/sqrt(2)`, `bar_O_2 = S^+/sqrt(2)`, `bar_O_3 = S^z`. But in the right boundary and the finishing channels, we need the `bar_O` products, not `O` products. Let me fix:

```python
print("=" * 60)
print("DEBUGGING AKLT MPO")
print("=" * 60)

# First verify MPO against direct Hamiltonian for small L
def aklt_hamiltonian_direct(L):
    """Build AKLT Hamiltonian directly from Kronecker products"""
    Sp, Sm, Sz, Id = spin_one_operators()
    d = 3
    H = np.zeros((d**L, d**L), dtype=complex)
    
    for i in range(L-1):
        # Build S_i . S_{i+1}
        # = 0.5*(Sp_i Sm_{i+1} + Sm_i Sp_{i+1}) + Sz_i Sz_{i+1}
        def site_op(op, site):
            ops = [np.eye(d, dtype=complex)] * L
            ops[site] = op
            result = ops[0]
            for k in range(1, L):
                result = np.kron(result, ops[k])
            return result
        
        SdotS = (0.5 * (site_op(Sp, i) @ site_op(Sm, i+1) +
                         site_op(Sm, i) @ site_op(Sp, i+1)) +
                 site_op(Sz, i) @ site_op(Sz, i+1))
        
        H += SdotS + (1.0/3.0) * SdotS @ SdotS
    
    return H

def mpo_to_dense(Ws):
    """Contract MPO to dense matrix"""
    L = len(Ws)
    d = Ws[0].shape[2]
    
    # Start with first site
    # W has shape (chi_l, chi_r, d, d')
    T = Ws[0]  # (1, chi, d, d')
    
    for i in range(1, L):
        # T has shape (1, chi_i, d^i, d^i) effectively
        # Contract bond index
        T = np.einsum('...bst,bBuv->...Bsutv', T, Ws[i])
        # Reshape physical indices
        sh = T.shape
        # T: (1, chi_{i+1}, d, d, ..., d, d, d, d)
        # We need to group ket indices together and bra indices together
        # Current: ...s,t,...u,v -> ...su, tv
        ndim = len(sh)
        # The physical indices are at positions -4,-3,-2,-1
        # Reshape: merge ket indices (s,u) and bra indices (t,v)
        chi_l = sh[0]
        chi_r = sh[1]
        n_phys = ndim - 2
        # Collect all dims
        phys_dims = sh[2:]
        # phys_dims = (d_ket_old, d_bra_old, d_ket_new, d_bra_new)
        d_ket = 1
        d_bra = 1
        for k in range(0, len(phys_dims), 2):
            d_ket *= phys_dims[k]
        for k in range(1, len(phys_dims), 2):
            d_bra *= phys_dims[k]
        
        # Need to reorder: group all kets then all bras
        # Current order after einsum: chi_l, chi_r, s, t, u, v
        # Want: chi_l, chi_r, s, u, t, v  -> then reshape
        T = T.reshape(chi_l, chi_r, -1)  # This won't work properly
    
    # Let me use a simpler approach
    return None

def mpo_to_dense_v2(Ws):
    """Contract MPO to dense matrix - simple approach"""
    L = len(Ws)
    d = Ws[0].shape[2]
    D = d ** L
    
    H = np.zeros((D, D), dtype=complex)
    
    for ket in range(D):
        for bra in range(D):
            # Decode indices
            ket_indices = []
            bra_indices = []
            k, b = ket, bra
            for i in range(L-1, -1, -1):
                ket_indices.insert(0, k % d)
                bra_indices.insert(0, b % d)
                k //= d
                b //= d
            
            # Contract MPO bond indices
            # Start with shape (chi,)
            vec = None
            for i in range(L):
                s = ket_indices[i]
                t = bra_indices[i]
                W_slice = Ws[i][:, :, s, t]  # (chi_l, chi_r)
                if vec is None:
                    vec = W_slice[0, :]  # first site: chi_l = 1
                else:
                    vec = vec @ W_slice
            
            H[ket, bra] = vec[0]  # last site: chi_r = 1
    
    return H

# Test for L=2,3,4
for L in [2, 3, 4]:
    print(f"\nL = {L}")
    Ws = aklt_mpo(L)
    H_mpo = mpo_to_dense_v2(Ws)
    H_direct = aklt_hamiltonian_direct(L)
    
    diff = np.max(np.abs(H_mpo - H_direct))
    print(f"  Max |H_mpo - H_direct| = {diff:.2e}")
    
    # Check eigenvalues
    evals_mpo = np.sort(np.linalg.eigvalsh(H_mpo))
    evals_direct = np.sort(np.linalg.eigvalsh(H_direct))
    print(f"  GS energy (MPO):    {evals_mpo[0]:.10f}")
    print(f"  GS energy (direct): {evals_direct[0]:.10f}")
    print(f"  Exact:              {-2/3*(L-1):.10f}")
    
    # Test exact AKLT state
    d = 3
    Ms = aklt_exact_mps_v2(L)[0][2]  # (row=0, col=0)
    psi = Ms[0]  # (1, d, D_r)
    for i in range(1, L):
        psi = np.einsum('...a,asb->...sb', psi, Ms[i])
    psi = psi.reshape(-1)
    psi = psi / np.linalg.norm(psi)
    
    E_test = (psi.conj() @ H_direct @ psi).real
    print(f"  <AKLT|H|AKLT>:      {E_test:.10f}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
```

---
## 🧑‍💻 User

============================================================
DEBUGGING AKLT MPO
============================================================

L = 2
  Max |H_mpo - H_direct| = 1.33e+00
  GS energy (MPO):    -0.6666666667
  GS energy (direct): -0.6666666667
  Exact:              -0.6666666667
  <AKLT|H|AKLT>:      -0.6666666667

L = 3
  Max |H_mpo - H_direct| = 1.33e+00
  GS energy (MPO):    -1.3333333333
  GS energy (direct): -1.3333333333
  Exact:              -1.3333333333
  <AKLT|H|AKLT>:      -1.3333333333

L = 4
  Max |H_mpo - H_direct| = 1.33e+00
  GS energy (MPO):    -2.0000000000
  GS energy (direct): -2.0000000000
  Exact:              -2.0000000000
  <AKLT|H|AKLT>:      -2.0000000000

============================================================
DONE
============================================================

---
## 🤖 Assistant

The MPO eigenvalues are correct but there's a constant offset `max diff = 1.33 = 4/3`. The AKLT Hamiltonian has the identity term from `(S·S)²` expansion. The MPO differs from direct by a constant per bond. But since eigenvalues match, the issue is only in the `<psi|H|psi>` contraction via environments. Let me trace the bug:

```python
print("=" * 60)
print("TRACING THE BUG: MPO vs DIRECT")
print("=" * 60)

L = 3
Ws = aklt_mpo(L)
H_mpo = mpo_to_dense_v2(Ws)
H_direct = aklt_hamiltonian_direct(L)

diff = H_mpo - H_direct
print("Difference H_mpo - H_direct (diagonal):")
print(np.diag(diff).real)
print(f"\nDiff is proportional to identity? {np.allclose(diff, diff[0,0]*np.eye(diff.shape[0]))}")

# The MPO and direct have the same eigenvalues but differ by a constant
# This means the environment contraction should still give correct energy
# The bug must be in the environment-based energy evaluation

# Let's test environment contraction on L=2 with exact AKLT
print("\n" + "=" * 60)
print("TESTING ENVIRONMENT CONTRACTION L=2")
print("=" * 60)

L = 2
Ws = aklt_mpo(L)

# Build exact AKLT MPS for L=2
Ap = np.array([[0, np.sqrt(2/3)], [0, 0]], dtype=complex)
A0 = np.array([[-1/np.sqrt(3), 0], [0, 1/np.sqrt(3)]], dtype=complex)
Am = np.array([[0, 0], [-np.sqrt(2/3), 0]], dtype=complex)

# Use row=0 boundary on left, col=0 on right
M0 = np.zeros((1, 3, 2), dtype=complex)
M0[0, 0, :] = Ap[0, :]
M0[0, 1, :] = A0[0, :]
M0[0, 2, :] = Am[0, :]

M1 = np.zeros((2, 3, 1), dtype=complex)
M1[:, 0, 0] = Ap[:, 0]
M1[:, 1, 0] = A0[:, 0]
M1[:, 2, 0] = Am[:, 0]

Ms = [M0, M1]
print(f"M0 shape: {M0.shape}")
print(f"M1 shape: {M1.shape}")

# Contract to wavefunction
psi = np.einsum('ijk,klm->ijlm', M0, M1).reshape(-1)
print(f"psi = {psi}")
print(f"norm = {np.linalg.norm(psi)**2}")

# Direct energy
H_direct_2 = aklt_hamiltonian_direct(2)
E_direct = (psi.conj() @ H_direct_2 @ psi).real
E_mpo_dense = (psi.conj() @ mpo_to_dense_v2(Ws) @ psi).real
print(f"E via direct H: {E_direct}")
print(f"E via MPO dense: {E_mpo_dense}")

# Now via environment contraction
L_env = np.zeros((1, 1, 1), dtype=complex)
L_env[0, 0, 0] = 1.0

print(f"\nStep-by-step environment contraction:")
print(f"L_env initial: {L_env.shape}")

# Site 0
L_env_1 = np.einsum('bxy,xsa,bBst,ytc->Bac', L_env, Ms[0], Ws[0], Ms[0].conj())
print(f"After site 0: L_env shape {L_env_1.shape}")
print(f"L_env_1 = \n{L_env_1[:, :, :].real}")

# Site 1
L_env_2 = np.einsum('bxy,xsa,bBst,ytc->Bac', L_env_1, Ms[1], Ws[1], Ms[1].conj())
print(f"After site 1: L_env shape {L_env_2.shape}")
print(f"E from env contraction = {L_env_2[0,0,0].real}")

# Compare: what should it be?
print(f"\nExpected E (unnormalized) = norm * E = {np.linalg.norm(psi)**2 * E_direct}")
print(f"Got E from env = {L_env_2[0,0,0].real}")

# The issue might be in the einsum contraction order
# Let's try the update formula from the LaTeX notes exactly:
# L_new[B,a,c] = sum L[b,x,y] A[x,s,a] W[b,B,s,t] A*[y,t,c]
# Note: W has indices (beta_l, beta_r, sigma_ket, sigma_bra)
# The contraction sums over b,x,y,s,t

# Let's do it step by step manually for site 0
A = Ms[0]      # (1, 3, 2)
W = Ws[0]      # (1, 14, 3, 3)
Ac = A.conj()  # (1, 3, 2)

# Method 1: direct einsum (current)
L1_m1 = np.einsum('bxy,xsa,bBst,ytc->Bac', L_env, A, W, Ac)

# Method 2: step by step per the notes
# Step 1: contract L with A
t1 = np.einsum('bxy,xsa->bysa', L_env, A)       # (1,1,3,2)
# Step 2: contract with W
t2 = np.einsum('bysa,bBst->Byat', t1, W)         # (14,1,2,3)
# Step 3: contract with A*
L1_m2 = np.einsum('Byat,ytc->Bac', t2, Ac)       # (14,2,2)

print(f"\nMethod 1 == Method 2: {np.allclose(L1_m1, L1_m2)}")

# Method 3: match the notes EXACTLY
# L[b,x,y] A[x,s,alpha] W[b,B,s,s'] A*[y,s',alpha']
# indices: b=mpo_left, B=mpo_right, x=bra_left, y=ket_left, 
#          s=phys_ket, s'=phys_bra, alpha=bra_right, alpha'=ket_right
# Wait - let me re-read the notes carefully

# From the notes eq for left env update:
# L^[i]_beta(alpha, alpha') = 
#   sum L^[i-1]_beta'(alpha'', alpha''') 
#       A_[alpha'', sigma, alpha]
#       W_[beta', beta, sigma, sigma']  
#       A*_[alpha''', sigma', alpha']

# So in the left env L[b, x, y]:
#   x is the "unprimed" bra index
#   y is the "primed" ket index (conjugate side)

# But when we USE L in the effective Hamiltonian, the notes say
# we must swap: L[b, y, x] instead of L[b, x, y]

# So the recursion builds L[b, alpha, alpha'] where alpha goes with A and alpha' with A*
# And the usage swaps them.

# The recursion is CORRECT as written.
# The issue must be elsewhere.

# Let's verify by brute force:
# Contract everything manually
print("\n--- Brute force check ---")
E_bf = 0.0
for s0 in range(3):
    for s1 in range(3):
        for t0 in range(3):
            for t1 in range(3):
                # <s0 s1| H |t0 t1> via MPO
                h_elem = 0.0
                for b in range(14):
                    h_elem += Ws[0][0, b, s0, t0] * Ws[1][b, 0, s1, t1]
                
                # psi coefficients
                for a in range(2):
                    c_ket = Ms[0][0, t0, a] * Ms[1][a, t1, 0]
                    c_bra = (Ms[0][0, s0, a] * Ms[1][a, s1, 0]).conj()
                    E_bf += c_bra * h_elem * c_ket

print(f"Brute force E = {E_bf.real}")
print(f"Env contraction E = {L_env_2[0,0,0].real}")
print(f"Direct H E (unnorm) = {(psi.conj() @ H_mpo_dense @ psi).real if False else 'skip'}")

# Check with MPO dense
H_mpo_2 = mpo_to_dense_v2(Ws)
E_mpo_unnorm = (psi.conj() @ H_mpo_2 @ psi).real
print(f"MPO dense E (unnorm) = {E_mpo_unnorm}")

print(f"\nBrute force matches env? {np.isclose(E_bf.real, L_env_2[0,0,0].real)}")
print(f"Brute force matches MPO dense? {np.isclose(E_bf.real, E_mpo_unnorm)}")
```

---
## 🧑‍💻 User

============================================================
TRACING THE BUG: MPO vs DIRECT
============================================================
Difference H_mpo - H_direct (diagonal):
[ 6.66666667e-01  3.33333333e-01 -2.22044605e-16 -2.22044605e-16
 -2.22044605e-16 -2.22044605e-16 -6.66666667e-01 -3.33333333e-01
 -2.22044605e-16  3.33333333e-01 -2.22044605e-16 -3.33333333e-01
 -2.22044605e-16 -4.44089210e-16 -2.22044605e-16 -3.33333333e-01
 -2.22044605e-16  3.33333333e-01 -2.22044605e-16 -3.33333333e-01
 -6.66666667e-01 -2.22044605e-16 -2.22044605e-16 -2.22044605e-16
 -2.22044605e-16  3.33333333e-01  6.66666667e-01]

Diff is proportional to identity? False

============================================================
TESTING ENVIRONMENT CONTRACTION L=2
============================================================
M0 shape: (1, 3, 2)
M1 shape: (2, 3, 1)
psi = [ 0.        +0.j  0.        +0.j -0.66666667+0.j  0.        +0.j
  0.33333333+0.j  0.        +0.j  0.        +0.j  0.        +0.j
  0.        +0.j]
norm = 0.5555555555555556
E via direct H: -0.3703703703703704
E via MPO dense: -0.22222222222222215

Step-by-step environment contraction:
L_env initial: (1, 1, 1)
After site 0: L_env shape (14, 2, 2)
L_env_1 = 
[[[ 0.33333333  0.        ]
  [ 0.          0.66666667]]

 [[ 0.          0.        ]
  [-0.47140452  0.        ]]

 [[ 0.         -0.47140452]
  [ 0.          0.        ]]

 [[ 0.          0.        ]
  [ 0.          0.66666667]]

 [[ 0.          0.        ]
  [ 0.          0.        ]]

 [[ 0.11111111  0.        ]
  [ 0.          0.22222222]]

 [[ 0.          0.        ]
  [ 0.          0.        ]]

 [[ 0.11111111  0.        ]
  [ 0.          0.        ]]

 [[ 0.          0.        ]
  [ 0.          0.        ]]

 [[ 0.         -0.15713484]
  [ 0.          0.        ]]

 [[ 0.          0.        ]
  [-0.15713484  0.        ]]

 [[ 0.          0.        ]
  [ 0.          0.        ]]

 [[ 0.          0.        ]
  [ 0.          0.22222222]]

 [[ 0.          0.        ]
  [ 0.          0.        ]]]
After site 1: L_env shape (1, 1, 1)
E from env contraction = -0.2222222222222222

Expected E (unnormalized) = norm * E = -0.20576131687242802
Got E from env = -0.2222222222222222

Method 1 == Method 2: True

--- Brute force check ---
Brute force E = -0.22222222222222215
Env contraction E = -0.2222222222222222
Direct H E (unnorm) = skip
MPO dense E (unnorm) = -0.22222222222222215

Brute force matches env? True
Brute force matches MPO dense? True

---
## 🤖 Assistant

Now I see the issue clearly. The environment contraction is correct and matches the MPO. But the MPO itself differs from the direct Hamiltonian - the difference is NOT proportional to identity, so it's a real bug in the MPO construction. The `bar_O` operators in the right boundary and finishing channels need to be different from `O`. Let me fix:

```python
print("=" * 60)
print("FIXING AKLT MPO: bar_O vs O")
print("=" * 60)

# From the LaTeX notes:
# O_1 = S+/sqrt(2),  O_2 = S-/sqrt(2),  O_3 = Sz
# bar_O_1 = S-/sqrt(2), bar_O_2 = S+/sqrt(2), bar_O_3 = Sz
#
# In the MPO:
# - channels 0->a (a=1,2,3): use O_a on the LEFT site
# - channels a->13: use bar_O_a on the RIGHT site  <--- THIS IS THE KEY
# - channels 0->(a,b): use (1/3) O_a O_b on the LEFT
# - channels (a,b)->13: use bar_O_a bar_O_b on the RIGHT <--- AND THIS

Sp, Sm, Sz, Id = spin_one_operators()
d = 3

# O operators (for "start" channels)
O = [Sp / np.sqrt(2), Sm / np.sqrt(2), Sz]

# bar_O operators (for "finish" channels) -- DIFFERENT from O!
Obar = [Sm / np.sqrt(2), Sp / np.sqrt(2), Sz]

# Products
OO = []
ObarObar = []
for a in range(3):
    for b in range(3):
        OO.append(O[a] @ O[b])
        ObarObar.append(Obar[a] @ Obar[b])

def aklt_mpo_fixed(L):
    """Corrected AKLT MPO with proper bar_O operators"""
    chi = 14
    W = []
    
    # Left boundary: shape (1, 14, d, d)
    # Row 0 of the bulk MPO
    W_left = np.zeros((1, chi, d, d), dtype=complex)
    W_left[0, 0] = Id                       # 0 -> 0: identity
    for a in range(3):
        W_left[0, 1+a] = O[a]               # 0 -> a: start bilinear
    for k in range(9):
        W_left[0, 4+k] = OO[k] / 3.0        # 0 -> (a,b): start biquadratic
    # W_left[0, 13] = 0  (no direct identity propagation from start)
    W.append(W_left)
    
    # Bulk: shape (14, 14, d, d)
    for _ in range(L - 2):
        W_bulk = np.zeros((chi, chi, d, d), dtype=complex)
        # Row 0: start channels
        W_bulk[0, 0] = Id
        for a in range(3):
            W_bulk[0, 1+a] = O[a]
        for k in range(9):
            W_bulk[0, 4+k] = OO[k] / 3.0
        # Rows 1,2,3 -> 13: finish bilinear with bar_O
        for a in range(3):
            W_bulk[1+a, 13] = Obar[a]
        # Rows 4..12 -> 13: finish biquadratic with bar_O bar_O
        for k in range(9):
            W_bulk[4+k, 13] = ObarObar[k]
        # Row 13 -> 13: identity propagation
        W_bulk[13, 13] = Id
        W.append(W_bulk)
    
    # Right boundary: shape (14, 1, d, d)
    # Column 0 of the bulk MPO (the "terminal" column)
    W_right = np.zeros((chi, 1, d, d), dtype=complex)
    # W_right[0, 0] = 0  (no term starts and ends here)
    for a in range(3):
        W_right[1+a, 0] = Obar[a]            # finish bilinear
    for k in range(9):
        W_right[4+k, 0] = ObarObar[k]        # finish biquadratic
    W_right[13, 0] = Id                       # identity propagation
    W.append(W_right)
    
    return W

# Verify for L=2,3,4
for L in [2, 3, 4]:
    print(f"\nL = {L}")
    Ws_fixed = aklt_mpo_fixed(L)
    H_mpo = mpo_to_dense_v2(Ws_fixed)
    H_direct = aklt_hamiltonian_direct(L)
    
    diff = np.max(np.abs(H_mpo - H_direct))
    print(f"  Max |H_mpo - H_direct| = {diff:.2e}")
    
    evals_mpo = np.sort(np.linalg.eigvalsh(H_mpo))
    evals_direct = np.sort(np.linalg.eigvalsh(H_direct))
    print(f"  GS energy (MPO):    {evals_mpo[0]:.10f}")
    print(f"  GS energy (direct): {evals_direct[0]:.10f}")
    print(f"  Exact:              {-2/3*(L-1):.10f}")

# Now verify exact MPS energy with fixed MPO
print("\n" + "=" * 60)
print("EXACT MPS ENERGY WITH FIXED MPO")
print("=" * 60)

for L in [4, 6, 8, 10]:
    Ws = aklt_mpo_fixed(L)
    E_exact = -2/3 * (L - 1)
    
    all_states = aklt_exact_mps_v2(L)
    row, col, Ms = all_states[0]  # first ground state
    
    # Compute norm
    norm_env = np.ones((1, 1), dtype=complex)
    for i in range(L):
        norm_env = np.einsum('xy,xsa,ysc->ac', norm_env, Ms[i], Ms[i].conj())
    norm = norm_env[0, 0].real
    
    # Compute <psi|H|psi> via environments
    L_env = np.zeros((1, 1, 1), dtype=complex)
    L_env[0, 0, 0] = 1.0
    for i in range(L):
        L_env = np.einsum('bxy,xsa,bBst,ytc->Bac', L_env, Ms[i], Ws[i], Ms[i].conj())
    E_raw = L_env[0, 0, 0].real
    
    E_norm = E_raw / norm
    print(f"L={L}: E={E_norm:.10f}, exact={E_exact:.10f}, err={abs(E_norm-E_exact):.2e}")

# Now run DMRG with fixed MPO
print("\n" + "=" * 60)
print("DMRG WITH FIXED AKLT MPO")
print("=" * 60)

for L in [6, 8, 10, 12]:
    print(f"\nL = {L}")
    Ws = aklt_mpo_fixed(L)
    Ms = random_mps(L, 3, 10)
    Ms, energies = two_site_dmrg(Ws, Ms, Dmax=15, num_sweeps=5)
    E_exact = -2/3 * (L - 1)
    print(f"  Energy (DMRG):  {energies[-1]:.10f}")
    print(f"  Energy (exact): {E_exact:.10f}")
    print(f"  Error: {abs(energies[-1] - E_exact):.2e}")

print("\n" + "=" * 60)
print("ALL CHECKS COMPLETE")
print("=" * 60)
```

---
## 🧑‍💻 User

============================================================
FIXING AKLT MPO: bar_O vs O
============================================================

L = 2
  Max |H_mpo - H_direct| = 2.22e-16
  GS energy (MPO):    -0.6666666667
  GS energy (direct): -0.6666666667
  Exact:              -0.6666666667

L = 3
  Max |H_mpo - H_direct| = 4.44e-16
  GS energy (MPO):    -1.3333333333
  GS energy (direct): -1.3333333333
  Exact:              -1.3333333333

L = 4
  Max |H_mpo - H_direct| = 6.66e-16
  GS energy (MPO):    -2.0000000000
  GS energy (direct): -2.0000000000
  Exact:              -2.0000000000

============================================================
EXACT MPS ENERGY WITH FIXED MPO
============================================================
L=4: E=-2.0000000000, exact=-2.0000000000, err=4.44e-16
L=6: E=-3.3333333333, exact=-3.3333333333, err=8.88e-16
L=8: E=-4.6666666667, exact=-4.6666666667, err=8.88e-16
L=10: E=-6.0000000000, exact=-6.0000000000, err=0.00e+00

============================================================
DMRG WITH FIXED AKLT MPO
============================================================

L = 6
  Energy (DMRG):  -3.3333333333
  Energy (exact): -3.3333333333
  Error: 4.44e-16

L = 8
  Energy (DMRG):  -4.6666666667
  Energy (exact): -4.6666666667
  Error: 7.99e-15

L = 10
  Energy (DMRG):  -6.0000000000
  Energy (exact): -6.0000000000
  Error: 1.15e-14

L = 12
  Energy (DMRG):  -7.3333333333
  Energy (exact): -7.3333333333
  Error: 5.33e-15

============================================================
ALL CHECKS COMPLETE
============================================================

---
## 🤖 Assistant

Everything works perfectly now. All errors are at machine precision. Now let me create the complete figure cells. First cell for Heisenberg figures, second for AKLT figures:

```python
# ============================================================================
# CELL 1: HEISENBERG FIGURES
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("figureAKLT", exist_ok=True)

# Run Heisenberg for multiple system sizes
print("Running Heisenberg DMRG for figures...")
heisenberg_data = {}
for L in [6, 8, 10, 12, 14]:
    print(f"  L = {L}...", end=" ")
    Ws = heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0)
    Ms = random_mps(L, 2, 20)
    Ms, energies = two_site_dmrg(Ws, Ms, Dmax=30, num_sweeps=6)
    heisenberg_data[L] = energies
    print(f"E = {energies[-1]:.10f}")

# Figure 1: Energy convergence per sweep (Heisenberg)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for L, energies in heisenberg_data.items():
    half_sweeps = np.arange(1, len(energies) + 1)
    ax.plot(half_sweeps, energies, 'o-', label=f'L={L}', markersize=4)
ax.set_xlabel('Half-sweep', fontsize=13)
ax.set_ylabel('Energy', fontsize=13)
ax.set_title('S=1/2 Heisenberg: Energy vs Sweep', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Figure 2: Energy per bond vs 1/L
ax = axes[1]
Ls = sorted(heisenberg_data.keys())
E_per_bond = [heisenberg_data[L][-1] / (L - 1) for L in Ls]
inv_L = [1.0 / L for L in Ls]
ax.plot(inv_L, E_per_bond, 'rs-', markersize=8, linewidth=2)
ax.set_xlabel('1/L', fontsize=13)
ax.set_ylabel('E / (L-1)', fontsize=13)
ax.set_title('S=1/2 Heisenberg: Energy per Bond', fontsize=14)
ax.axhline(y=-0.4432, color='k', linestyle='--', alpha=0.5, label='Bethe ansatz (∞)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figureAKLT/heisenberg_energy.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figureAKLT/heisenberg_energy.png")

# Figure 3: Heisenberg convergence (energy change)
fig, ax = plt.subplots(figsize=(8, 5))
for L, energies in heisenberg_data.items():
    dE = [abs(energies[i] - energies[i-1]) for i in range(1, len(energies))]
    ax.semilogy(range(2, len(energies) + 1), dE, 'o-', label=f'L={L}', markersize=4)
ax.set_xlabel('Half-sweep', fontsize=13)
ax.set_ylabel('|ΔE|', fontsize=13)
ax.set_title('S=1/2 Heisenberg: Energy Convergence', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figureAKLT/heisenberg_convergence.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figureAKLT/heisenberg_convergence.png")
```

```python
# ============================================================================
# CELL 2: AKLT FIGURES
# ============================================================================

print("Running AKLT DMRG for figures...")
aklt_data = {}
aklt_errors = {}
for L in [6, 8, 10, 12, 14]:
    print(f"  L = {L}...", end=" ")
    Ws = aklt_mpo_fixed(L)
    Ms = random_mps(L, 3, 10)
    Ms, energies = two_site_dmrg(Ws, Ms, Dmax=15, num_sweeps=6)
    aklt_data[L] = energies
    E_exact = -2/3 * (L - 1)
    aklt_errors[L] = abs(energies[-1] - E_exact)
    print(f"E = {energies[-1]:.10f}, exact = {E_exact:.10f}, err = {aklt_errors[L]:.2e}")

# Also compute exact MPS energies
aklt_exact_errors = {}
for L in [4, 6, 8, 10, 12, 14]:
    Ws = aklt_mpo_fixed(L)
    all_states = aklt_exact_mps_v2(L)
    _, _, Ms_ex = all_states[0]
    
    norm_env = np.ones((1, 1), dtype=complex)
    for i in range(L):
        norm_env = np.einsum('xy,xsa,ysc->ac', norm_env, Ms_ex[i], Ms_ex[i].conj())
    norm = norm_env[0, 0].real
    
    L_env = np.zeros((1, 1, 1), dtype=complex)
    L_env[0, 0, 0] = 1.0
    for i in range(L):
        L_env = np.einsum('bxy,xsa,bBst,ytc->Bac', L_env, Ms_ex[i], Ws[i], Ms_ex[i].conj())
    E_mps = L_env[0, 0, 0].real / norm
    aklt_exact_errors[L] = abs(E_mps - (-2/3 * (L - 1)))

# Figure 4: AKLT energy convergence per sweep
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for L, energies in aklt_data.items():
    half_sweeps = np.arange(1, len(energies) + 1)
    E_exact = -2/3 * (L - 1)
    ax.plot(half_sweeps, np.array(energies) - E_exact, 'o-', label=f'L={L}', markersize=4)
ax.set_xlabel('Half-sweep', fontsize=13)
ax.set_ylabel('E - E_exact', fontsize=13)
ax.set_title('AKLT: Energy Error vs Sweep', fontsize=14)
ax.set_yscale('symlog', linthresh=1e-14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Figure 5: AKLT DMRG error vs L
ax = axes[1]
Ls_dmrg = sorted(aklt_errors.keys())
errs_dmrg = [aklt_errors[L] for L in Ls_dmrg]
ax.semilogy(Ls_dmrg, errs_dmrg, 'rs-', markersize=8, linewidth=2, label='DMRG')

Ls_exact = sorted(aklt_exact_errors.keys())
errs_exact = [max(aklt_exact_errors[L], 1e-16) for L in Ls_exact]
ax.semilogy(Ls_exact, errs_exact, 'bo-', markersize=8, linewidth=2, label='Exact MPS')

ax.set_xlabel('L', fontsize=13)
ax.set_ylabel('|E - E_exact|', fontsize=13)
ax.set_title('AKLT: Energy Error vs System Size', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figureAKLT/aklt_energy.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figureAKLT/aklt_energy.png")

# Figure 6: AKLT convergence (energy change)
fig, ax = plt.subplots(figsize=(8, 5))
for L, energies in aklt_data.items():
    dE = [abs(energies[i] - energies[i-1]) for i in range(1, len(energies))]
    if len(dE) > 0:
        dE_plot = [max(x, 1e-16) for x in dE]
        ax.semilogy(range(2, len(energies) + 1), dE_plot, 'o-', label=f'L={L}', markersize=4)
ax.set_xlabel('Half-sweep', fontsize=13)
ax.set_ylabel('|ΔE|', fontsize=13)
ax.set_title('AKLT: Energy Convergence per Sweep', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figureAKLT/aklt_convergence.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figureAKLT/aklt_convergence.png")

# Summary table
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Model':<12} {'L':>4} {'E_DMRG':>18} {'E_exact':>18} {'Error':>12}")
print("-" * 66)
for L in sorted(heisenberg_data.keys()):
    E = heisenberg_data[L][-1]
    print(f"{'Heisenberg':<12} {L:>4} {E:>18.10f} {'---':>18} {'---':>12}")
for L in sorted(aklt_data.keys()):
    E = aklt_data[L][-1]
    E_ex = -2/3 * (L - 1)
    err = abs(E - E_ex)
    print(f"{'AKLT':<12} {L:>4} {E:>18.10f} {E_ex:>18.10f} {err:>12.2e}")
```

---
## 🧑‍💻 User

Thank you. I would like to make sure one- or two-site DMRG is used?

---
## 🤖 Assistant

Looking at the code, **two-site DMRG** is used, exactly as recommended by the LaTeX notes:

> *"This should be the first DMRG implementation."*

The `two_site_dmrg` function implements Algorithm 8 from the notes:

1. **Left-to-right sweep**: optimizes two-site tensor `Θ[i,i+1]`, SVD splits as `A[i]` (left-canonical) + `M[i+1]` (center)
2. **Right-to-left sweep**: optimizes two-site tensor `Θ[i,i+1]`, SVD splits as `M[i]` (center) + `B[i+1]` (right-canonical)

The matrix-free Lanczos solver uses the **two-site** effective Hamiltonian matvec:

```python
def heff_two_site_matvec(v, L, W1, W2, R, Dl, d1, d2, Dr):
    # validated two-site contraction from the notes
    X  = np.einsum('byx,yuvz->bxuvz', L, Theta)
    Y  = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
    Z  = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
    HT = np.einsum('Cxstz,Cza->xsta', Z, R)
```

**One-site DMRG is not implemented**, which is consistent with the notes' recommendation:

> *"One-site DMRG is prone to local minima and should not be used as a first standalone implementation."*

Would you like me to also implement one-site DMRG (Algorithm 7 from the notes) for comparison?

---
## 🧑‍💻 User

Yes, pls

---
## 🤖 Assistant



```python
# ============================================================================
# COMPLETE IMPLEMENTATION: ONE-SITE + TWO-SITE DMRG
# ============================================================================

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
import os

os.makedirs("figureAKLT", exist_ok=True)

# ============================================================================
# SPIN OPERATORS
# ============================================================================

def spin_half_operators():
    Sp = np.array([[0, 1], [0, 0]], dtype=complex)
    Sm = np.array([[0, 0], [1, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    Id = np.eye(2, dtype=complex)
    return Sp, Sm, Sz, Id

def spin_one_operators():
    Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=complex)
    Sp = np.sqrt(2) * np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=complex)
    Sm = np.sqrt(2) * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=complex)
    Id = np.eye(3, dtype=complex)
    return Sp, Sm, Sz, Id

# ============================================================================
# MPO CONSTRUCTION
# ============================================================================

def heisenberg_mpo(L, J=1.0, Jz=1.0, h=0.0):
    Sp, Sm, Sz, Id = spin_half_operators()
    d = 2
    chi = 5
    W = []
    
    W_left = np.zeros((1, chi, d, d), dtype=complex)
    W_left[0, 0] = -h * Sz
    W_left[0, 1] = J/2 * Sm
    W_left[0, 2] = J/2 * Sp
    W_left[0, 3] = Jz * Sz
    W_left[0, 4] = Id
    W.append(W_left)
    
    for _ in range(L - 2):
        W_bulk = np.zeros((chi, chi, d, d), dtype=complex)
        W_bulk[0, 0] = Id
        W_bulk[1, 0] = Sp
        W_bulk[2, 0] = Sm
        W_bulk[3, 0] = Sz
        W_bulk[4, 0] = -h * Sz
        W_bulk[4, 1] = J/2 * Sm
        W_bulk[4, 2] = J/2 * Sp
        W_bulk[4, 3] = Jz * Sz
        W_bulk[4, 4] = Id
        W.append(W_bulk)
    
    W_right = np.zeros((chi, 1, d, d), dtype=complex)
    W_right[0, 0] = Id
    W_right[1, 0] = Sp
    W_right[2, 0] = Sm
    W_right[3, 0] = Sz
    W_right[4, 0] = -h * Sz
    W.append(W_right)
    
    return W

def aklt_mpo(L):
    Sp, Sm, Sz, Id = spin_one_operators()
    d = 3
    O = [Sp / np.sqrt(2), Sm / np.sqrt(2), Sz]
    Obar = [Sm / np.sqrt(2), Sp / np.sqrt(2), Sz]
    OO = [O[a] @ O[b] for a in range(3) for b in range(3)]
    ObarObar = [Obar[a] @ Obar[b] for a in range(3) for b in range(3)]
    
    chi = 14
    W = []
    
    W_left = np.zeros((1, chi, d, d), dtype=complex)
    W_left[0, 0] = Id
    for a in range(3):
        W_left[0, 1 + a] = O[a]
    for k in range(9):
        W_left[0, 4 + k] = OO[k] / 3.0
    W.append(W_left)
    
    for _ in range(L - 2):
        W_bulk = np.zeros((chi, chi, d, d), dtype=complex)
        W_bulk[0, 0] = Id
        for a in range(3):
            W_bulk[0, 1 + a] = O[a]
        for k in range(9):
            W_bulk[0, 4 + k] = OO[k] / 3.0
        for a in range(3):
            W_bulk[1 + a, 13] = Obar[a]
        for k in range(9):
            W_bulk[4 + k, 13] = ObarObar[k]
        W_bulk[13, 13] = Id
        W.append(W_bulk)
    
    W_right = np.zeros((chi, 1, d, d), dtype=complex)
    for a in range(3):
        W_right[1 + a, 0] = Obar[a]
    for k in range(9):
        W_right[4 + k, 0] = ObarObar[k]
    W_right[13, 0] = Id
    W.append(W_right)
    
    return W

# ============================================================================
# CANONICALIZATION
# ============================================================================

def left_canonicalize_site(M):
    Dl, d, Dr = M.shape
    Q, R = np.linalg.qr(M.reshape(Dl * d, Dr, order='C'), mode='reduced')
    A = Q.reshape(Dl, d, -1, order='C')
    return A, R

def right_canonicalize_site(M):
    Dl, d, Dr = M.shape
    Q, R = np.linalg.qr(M.reshape(Dl, d * Dr, order='C').T.conj(), mode='reduced')
    B = Q.T.conj().reshape(-1, d, Dr, order='C')
    return B, R.T.conj()

def mixed_canonical(Ms, center):
    L = len(Ms)
    Ms = [M.copy() for M in Ms]
    for i in range(center):
        Ms[i], R = left_canonicalize_site(Ms[i])
        Ms[i + 1] = np.einsum('ij,jkl->ikl', R, Ms[i + 1])
    for i in range(L - 1, center, -1):
        Ms[i], R = right_canonicalize_site(Ms[i])
        Ms[i - 1] = np.einsum('ijk,kl->ijl', Ms[i - 1], R)
    return Ms

# ============================================================================
# ENVIRONMENTS
# ============================================================================

def update_left_env(L_env, A, W):
    return np.einsum('bxy,xsa,bBst,ytc->Bac', L_env, A, W, A.conj())

def update_right_env(R_env, B, W):
    return np.einsum('xsa,bBst,Bac,ytc->bxy', B, W, R_env, B.conj())

def build_right_envs(Ms, Ws):
    L = len(Ms)
    Rs = [None] * (L + 1)
    Rs[L] = np.zeros((1, 1, 1), dtype=complex)
    Rs[L][0, 0, 0] = 1.0
    for i in range(L - 1, 0, -1):
        Rs[i] = update_right_env(Rs[i + 1], Ms[i], Ws[i])
    return Rs

# ============================================================================
# EFFECTIVE HAMILTONIANS (MATRIX-FREE)
# ============================================================================

def heff_one_site_matvec(v, L_env, W, R_env, Dl, d, Dr):
    """Validated one-site matvec from the notes"""
    M = v.reshape((Dl, d, Dr), order='C')
    X = np.einsum('byx,ysz->bxsz', L_env, M)
    Y = np.einsum('bBst,bxsz->Bxtz', W, X)
    HM = np.einsum('Bxtz,Bza->xta', Y, R_env)
    return HM.reshape(Dl * d * Dr, order='C')

def heff_two_site_matvec(v, L_env, W1, W2, R_env, Dl, d1, d2, Dr):
    """Validated two-site matvec from the notes"""
    Theta = v.reshape((Dl, d1, d2, Dr), order='C')
    X = np.einsum('byx,yuvz->bxuvz', L_env, Theta)
    Y = np.einsum('bBus,bxuvz->Bxsvz', W1, X)
    Z = np.einsum('BCvt,Bxsvz->Cxstz', W2, Y)
    HT = np.einsum('Cxstz,Cza->xsta', Z, R_env)
    return HT.reshape(Dl * d1 * d2 * Dr, order='C')

# ============================================================================
# TWO-SITE DMRG (Algorithm 8 from notes)
# ============================================================================

def two_site_dmrg(Ws, Ms, Dmax, num_sweeps=5):
    L = len(Ms)
    energies = []
    
    Ms = mixed_canonical(Ms, 0)
    Rs = build_right_envs(Ms, Ws)
    Ls = [None] * (L + 1)
    Ls[0] = np.zeros((1, 1, 1), dtype=complex)
    Ls[0][0, 0, 0] = 1.0
    
    for sweep in range(num_sweeps):
        # Left-to-right
        for i in range(L - 1):
            Theta = np.einsum('ijk,klm->ijlm', Ms[i], Ms[i + 1])
            Dl, d1, d2, Dr = Theta.shape
            Nloc = Dl * d1 * d2 * Dr
            v0 = Theta.reshape(Nloc, order='C')
            v0 = v0 / np.linalg.norm(v0)
            
            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v: heff_two_site_matvec(
                    v, Ls[i], Ws[i], Ws[i + 1], Rs[i + 2], Dl, d1, d2, Dr),
                dtype=complex)
            E, vec = eigsh(Heff, k=1, which='SA', v0=v0, tol=1e-6, maxiter=500)
            Theta = vec[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            
            U, S, Vh = np.linalg.svd(
                Theta.reshape(Dl * d1, d2 * Dr, order='C'), full_matrices=False)
            Dnew = min(Dmax, len(S))
            U, S, Vh = U[:, :Dnew], S[:Dnew], Vh[:Dnew, :]
            
            Ms[i] = U.reshape(Dl, d1, Dnew, order='C')
            Ms[i + 1] = (np.diag(S) @ Vh).reshape(Dnew, d2, Dr, order='C')
            
            if i < L - 2:
                Ls[i + 1] = update_left_env(Ls[i], Ms[i], Ws[i])
        
        energies.append(E[0].real)
        
        # Right-to-left
        for i in range(L - 2, -1, -1):
            Theta = np.einsum('ijk,klm->ijlm', Ms[i], Ms[i + 1])
            Dl, d1, d2, Dr = Theta.shape
            Nloc = Dl * d1 * d2 * Dr
            v0 = Theta.reshape(Nloc, order='C')
            v0 = v0 / np.linalg.norm(v0)
            
            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v: heff_two_site_matvec(
                    v, Ls[i], Ws[i], Ws[i + 1], Rs[i + 2], Dl, d1, d2, Dr),
                dtype=complex)
            E, vec = eigsh(Heff, k=1, which='SA', v0=v0, tol=1e-6, maxiter=500)
            Theta = vec[:, 0].reshape((Dl, d1, d2, Dr), order='C')
            
            U, S, Vh = np.linalg.svd(
                Theta.reshape(Dl * d1, d2 * Dr, order='C'), full_matrices=False)
            Dnew = min(Dmax, len(S))
            U, S, Vh = U[:, :Dnew], S[:Dnew], Vh[:Dnew, :]
            
            Ms[i] = (U @ np.diag(S)).reshape(Dl, d1, Dnew, order='C')
            Ms[i + 1] = Vh.reshape(Dnew, d2, Dr, order='C')
            
            if i > 0:
                Rs[i + 1] = update_right_env(Rs[i + 2], Ms[i + 1], Ws[i + 1])
        
        energies.append(E[0].real)
        if sweep > 0 and abs(energies[-1] - energies[-3]) < 1e-12:
            break
    
    return Ms, energies

# ============================================================================
# ONE-SITE DMRG (Algorithm 7 from notes)
# ============================================================================

def one_site_dmrg(Ws, Ms, Dmax, num_sweeps=5):
    """One-site DMRG following Algorithm 7 of the notes.
    
    Left-to-right: optimize M[i], SVD -> A[i] (left-canonical),
                   absorb SV† into M[i+1], update L[i+1].
    Right-to-left: optimize M[i], SVD -> B[i] (right-canonical),
                   absorb US into M[i-1], update R[i-1].
    """
    L = len(Ms)
    energies = []
    
    # Start in mixed-canonical form with center at site 0
    Ms = mixed_canonical(Ms, 0)
    Rs = build_right_envs(Ms, Ws)
    Ls = [None] * (L + 1)
    Ls[0] = np.zeros((1, 1, 1), dtype=complex)
    Ls[0][0, 0, 0] = 1.0
    
    for sweep in range(num_sweeps):
        # ---- Left-to-right sweep: sites 0 .. L-2 ----
        for i in range(L - 1):
            Dl, d, Dr = Ms[i].shape
            Nloc = Dl * d * Dr
            v0 = Ms[i

---
## 🧑‍💻 User

cut off, pls continue from the cutoff

---
## 🤖 Assistant

```python
            v0 = Ms[i].reshape(Nloc, order='C')
            v0 = v0 / np.linalg.norm(v0)
            
            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v: heff_one_site_matvec(
                    v, Ls[i], Ws[i], Rs[i + 1], Dl, d, Dr),
                dtype=complex)
            E, vec = eigsh(Heff, k=1, which='SA', v0=v0, tol=1e-6, maxiter=500)
            M_opt = vec[:, 0].reshape((Dl, d, Dr), order='C')
            
            # SVD split: left-grouped matrix (Dl*d, Dr)
            M_mat = M_opt.reshape(Dl * d, Dr, order='C')
            U, S, Vh = np.linalg.svd(M_mat, full_matrices=False)
            Dnew = min(Dmax, len(S))
            U, S, Vh = U[:, :Dnew], S[:Dnew], Vh[:Dnew, :]
            
            # A[i] is left-canonical
            Ms[i] = U.reshape(Dl, d, Dnew, order='C')
            # Absorb S V† into M[i+1]
            SVh = np.diag(S) @ Vh
            Ms[i + 1] = np.einsum('ij,jkl->ikl', SVh, Ms[i + 1])
            
            # Update left environment
            Ls[i + 1] = update_left_env(Ls[i], Ms[i], Ws[i])
        
        energies.append(E[0].real)
        
        # ---- Right-to-left sweep: sites L-1 .. 1 ----
        for i in range(L - 1, 0, -1):
            Dl, d, Dr = Ms[i].shape
            Nloc = Dl * d * Dr
            v0 = Ms[i].reshape(Nloc, order='C')
            v0 = v0 / np.linalg.norm(v0)
            
            Heff = LinearOperator(
                shape=(Nloc, Nloc),
                matvec=lambda v: heff_one_site_matvec(
                    v, Ls[i], Ws[i], Rs[i + 1], Dl, d, Dr),
                dtype=complex)
            E, vec = eigsh(Heff, k=1, which='SA', v0=v0, tol=1e-6, maxiter=500)
            M_opt = vec[:, 0].reshape((Dl, d, Dr), order='C')
            
            # SVD split: right-grouped matrix (Dl, d*Dr)
            M_mat = M_opt.reshape(Dl, d * Dr, order='C')
            U, S, Vh = np.linalg.svd(M_mat, full_matrices=False)
            Dnew = min(Dmax, len(S))
            U, S, Vh = U[:, :Dnew], S[:Dnew], Vh[:Dnew, :]
            
            # B[i] is right-canonical (reshape V†)
            Ms[i] = Vh.reshape(Dnew, d, Dr, order='C')
            # Absorb U S into M[i-1]
            US = U @ np.diag(S)
            Ms[i - 1] = np.einsum('ijk,kl->ijl', Ms[i - 1], US)
            
            # Update right environment
            Rs[i] = update_right_env(Rs[i + 1], Ms[i], Ws[i])
        
        energies.append(E[0].real)
        if sweep > 0 and abs(energies[-1] - energies[-3]) < 1e-12:
            break
    
    return Ms, energies

# ============================================================================
# INITIAL MPS
# ============================================================================

def random_mps(L, d, D):
    Ms = []
    Ms.append(np.random.randn(1, d, min(D, d)) + 1j * np.random.randn(1, d, min(D, d)))
    for i in range(1, L - 1):
        Dl = Ms[-1].shape[2]
        Dr = min(D, d ** min(i + 1, L - i - 1))
        Ms.append(np.random.randn(Dl, d, Dr) + 1j * np.random.randn(Dl, d, Dr))
    Dl = Ms[-1].shape[2]
    Ms.append(np.random.randn(Dl, d, 1) + 1j * np.random.randn(Dl, d, 1))
    return Ms

def aklt_exact_mps(L):
    Ap = np.array([[0, np.sqrt(2/3)], [0, 0]], dtype=complex)
    A0 = np.array([[-1/np.sqrt(3), 0], [0, 1/np.sqrt(3)]], dtype=complex)
    Am = np.array([[0, 0], [-np.sqrt(2/3), 0]], dtype=complex)
    Ms = []
    for i in range(L):
        if i == 0:
            M = np.zeros((1, 3, 2), dtype=complex)
            M[0, 0, :] = Ap[0, :]
            M[0, 1, :] = A0[0, :]
            M[0, 2, :] = Am[0, :]
        elif i == L - 1:
            M = np.zeros((2, 3, 1), dtype=complex)
            M[:, 0, 0] = Ap[:, 0]
            M[:, 1, 0] = A0[:, 0]
            M[:, 2, 0] = Am[:, 0]
        else:
            M = np.zeros((2, 3, 2), dtype=complex)
            M[:, 0, :] = Ap
            M[:, 1, :] = A0
            M[:, 2, :] = Am
        Ms.append(M)
    return Ms

# ============================================================================
# RUN BOTH ALGORITHMS
# ============================================================================

np.random.seed(42)

print("=" * 70)
print("COMPARISON: ONE-SITE vs TWO-SITE DMRG")
print("=" * 70)

print("\n--- S=1/2 Heisenberg ---")
print(f"{'L':>4} {'E (2-site)':>18} {'E (1-site)':>18} {'diff':>12}")
print("-" * 56)

heisenberg_1s = {}
heisenberg_2s = {}

for L in [6, 8, 10, 12]:
    Ws = heisenberg_mpo(L)
    
    np.random.seed(42)
    Ms = random_mps(L, 2, 20)
    Ms_2s, en_2s = two_site_dmrg(Ws, Ms, Dmax=30, num_sweeps=6)
    heisenberg_2s[L] = en_2s
    
    # For one-site: start from the two-site result for fair comparison
    # (notes warn one-site is prone to local minima from random start)
    Ms_1s, en_1s = one_site_dmrg(Ws, Ms_2s, Dmax=30, num_sweeps=6)
    heisenberg_1s[L] = en_1s
    
    diff = abs(en_2s[-1] - en_1s[-1])
    print(f"{L:>4} {en_2s[-1]:>18.10f} {en_1s[-1]:>18.10f} {diff:>12.2e}")

print("\n--- AKLT ---")
print(f"{'L':>4} {'E (2-site)':>18} {'E (1-site)':>18} {'E_exact':>18} {'err(2s)':>10} {'err(1s)':>10}")
print("-" * 82)

aklt_1s = {}
aklt_2s = {}

for L in [6, 8, 10, 12]:
    Ws = aklt_mpo(L)
    E_exact = -2/3 * (L - 1)
    
    np.random.seed(42)
    Ms = random_mps(L, 3, 10)
    Ms_2s, en_2s = two_site_dmrg(Ws, Ms, Dmax=15, num_sweeps=6)
    aklt_2s[L] = en_2s
    
    Ms_1s, en_1s = one_site_dmrg(Ws, Ms_2s, Dmax=15, num_sweeps=6)
    aklt_1s[L] = en_1s
    
    err_2s = abs(en_2s[-1] - E_exact)
    err_1s = abs(en_1s[-1] - E_exact)
    print(f"{L:>4} {en_2s[-1]:>18.10f} {en_1s[-1]:>18.10f} {E_exact:>18.10f} {err_2s:>10.2e} {err_1s:>10.2e}")

# Verify exact AKLT MPS
print("\n--- AKLT Exact MPS Verification ---")
for L in [6, 8, 10, 12]:
    Ws = aklt_mpo(L)
    Ms = aklt_exact_mps(L)
    
    norm_env = np.ones((1, 1), dtype=complex)
    for i in range(L):
        norm_env = np.einsum('xy,xsa,ysc->ac', norm_env, Ms[i], Ms[i].conj())
    norm = norm_env[0, 0].real
    
    L_env = np.zeros((1, 1, 1), dtype=complex)
    L_env[0, 0, 0] = 1.0
    for i in range(L):
        L_env = np.einsum('bxy,xsa,bBst,ytc->Bac', L_env, Ms[i], Ws[i], Ms[i].conj())
    E_mps = L_env[0, 0, 0].real / norm
    E_exact = -2/3 * (L - 1)
    print(f"  L={L}: E_mps={E_mps:.10f}, exact={E_exact:.10f}, err={abs(E_mps-E_exact):.2e}")

print("\n" + "=" * 70)
print("COMPUTATION COMPLETE")
print("=" * 70)
```

---
## 🧑‍💻 User

======================================================================
COMPARISON: ONE-SITE vs TWO-SITE DMRG
======================================================================

--- S=1/2 Heisenberg ---
   L         E (2-site)         E (1-site)         diff
--------------------------------------------------------
   6      -2.4935771339      -2.4935771339     2.22e-15
   8      -3.3749325987      -3.3749325987     9.33e-15
  10      -4.2580352073      -4.2580352073     4.80e-14
  12      -5.1420906328      -5.1420906328     2.84e-14

--- AKLT ---
   L         E (2-site)         E (1-site)            E_exact    err(2s)    err(1s)
----------------------------------------------------------------------------------
   6      -3.3333333333      -3.3333333333      -3.3333333333   1.33e-15   2.22e-15
   8      -4.6666666667      -4.6666666667      -4.6666666667   8.88e-16   5.33e-15
  10      -6.0000000000      -6.0000000000      -6.0000000000   0.00e+00   1.78e-15
  12      -7.3333333333      -7.3333333333      -7.3333333333   3.55e-15   8.88e-16

--- AKLT Exact MPS Verification ---
  L=6: E_mps=-3.3333333333, exact=-3.3333333333, err=8.88e-16
  L=8: E_mps=-4.6666666667, exact=-4.6666666667, err=8.88e-16
  L=10: E_mps=-6.0000000000, exact=-6.0000000000, err=0.00e+00
  L=12: E_mps=-7.3333333333, exact=-7.3333333333, err=0.00e+00

======================================================================
COMPUTATION COMPLETE
======================================================================

---
## 🤖 Assistant

All results are perfect at machine precision. Now let me create the figure cells:

```python
# ============================================================================
# CELL: HEISENBERG FIGURES
# ============================================================================
import matplotlib.pyplot as plt

print("Generating Heisenberg figures...")

# Figure 1: Energy convergence per sweep
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for L in sorted(heisenberg_2s.keys()):
    en = heisenberg_2s[L]
    ax.plot(range(1, len(en) + 1), en, 'o-', label=f'2-site L={L}', markersize=4)
for L in sorted(heisenberg_1s.keys()):
    en = heisenberg_1s[L]
    ax.plot(range(1, len(en) + 1), en, 's--', label=f'1-site L={L}', markersize=3, alpha=0.7)
ax.set_xlabel('Half-sweep', fontsize=13)
ax.set_ylabel('Energy', fontsize=13)
ax.set_title('S=1/2 Heisenberg: Energy vs Sweep', fontsize=14)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Figure 2: Energy per bond vs 1/L
ax = axes[1]
Ls = sorted(heisenberg_2s.keys())
E_per_bond_2s = [heisenberg_2s[L][-1] / (L - 1) for L in Ls]
E_per_bond_1s = [heisenberg_1s[L][-1] / (L - 1) for L in Ls]
inv_L = [1.0 / L for L in Ls]
ax.plot(inv_L, E_per_bond_2s, 'rs-', markersize=8, linewidth=2, label='2-site DMRG')
ax.plot(inv_L, E_per_bond_1s, 'b^--', markersize=8, linewidth=2, label='1-site DMRG')
ax.axhline(y=-0.4432, color='k', linestyle=':', alpha=0.5, label='Bethe ansatz (L→∞)')
ax.set_xlabel('1/L', fontsize=13)
ax.set_ylabel('E / (L−1)', fontsize=13)
ax.set_title('S=1/2 Heisenberg: Energy per Bond', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figureAKLT/heisenberg_energy.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figureAKLT/heisenberg_energy.png")

# Figure 3: Convergence |dE|
fig, ax = plt.subplots(figsize=(8, 5))
for L in sorted(heisenberg_2s.keys()):
    en = heisenberg_2s[L]
    dE = [abs(en[i] - en[i - 1]) for i in range(1, len(en))]
    dE = [max(x, 1e-16) for x in dE]
    ax.semilogy(range(2, len(en) + 1), dE, 'o-', label=f'2-site L={L}', markersize=4)
for L in sorted(heisenberg_1s.keys()):
    en = heisenberg_1s[L]
    dE = [abs(en[i] - en[i - 1]) for i in range(1, len(en))]
    dE = [max(x, 1e-16) for x in dE]
    ax.semilogy(range(2, len(en) + 1), dE, 's--', label=f'1-site L={L}', markersize=3, alpha=0.7)
ax.set_xlabel('Half-sweep', fontsize=13)
ax.set_ylabel('|ΔE|', fontsize=13)
ax.set_title('S=1/2 Heisenberg: Energy Convergence', fontsize=14)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figureAKLT/heisenberg_convergence.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figureAKLT/heisenberg_convergence.png")
```

```python
# ============================================================================
# CELL: AKLT FIGURES
# ============================================================================

print("Generating AKLT figures...")

# Figure 4: AKLT energy error vs sweep
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for L in sorted(aklt_2s.keys()):
    en = aklt_2s[L]
    E_exact = -2/3 * (L - 1)
    err = [abs(e - E_exact) for e in en]
    err = [max(x, 1e-16) for x in err]
    ax.semilogy(range(1, len(en) + 1), err, 'o-', label=f'2-site L={L}', markersize=4)
for L in sorted(aklt_1s.keys()):
    en = aklt_1s[L]
    E_exact = -2/3 * (L - 1)
    err = [abs(e - E_exact) for e in en]
    err = [max(x, 1e-16) for x in err]
    ax.semilogy(range(1, len(en) + 1), err, 's--', label=f'1-site L={L}', markersize=3, alpha=0.7)
ax.set_xlabel('Half-sweep', fontsize=13)
ax.set_ylabel('|E − E_exact|', fontsize=13)
ax.set_title('AKLT: Energy Error vs Sweep', fontsize=14)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Figure 5: Final error vs L
ax = axes[1]
Ls = sorted(aklt_2s.keys())
err_2s = [max(abs(aklt_2s[L][-1] - (-2/3*(L-1))), 1e-16) for L in Ls]
err_1s = [max(abs(aklt_1s[L][-1] - (-2/3*(L-1))), 1e-16) for L in Ls]
ax.semilogy(Ls, err_2s, 'rs-', markersize=8, linewidth=2, label='2-site DMRG')
ax.semilogy(Ls, err_1s, 'b^--', markersize=8, linewidth=2, label='1-site DMRG')
ax.set_xlabel('L', fontsize=13)
ax.set_ylabel('|E − E_exact|', fontsize=13)
ax.set_title('AKLT: Final Energy Error vs System Size', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figureAKLT/aklt_energy.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figureAKLT/aklt_energy.png")

# Figure 6: AKLT convergence |dE|
fig, ax = plt.subplots(figsize=(8, 5))
for L in sorted(aklt_2s.keys()):
    en = aklt_2s[L]
    dE = [abs(en[i] - en[i - 1]) for i in range(1, len(en))]
    dE = [max(x, 1e-16) for x in dE]
    ax.semilogy(range(2, len(en) + 1), dE, 'o-', label=f'2-site L={L}', markersize=4)
for L in sorted(aklt_1s.keys()):
    en = aklt_1s[L]
    dE = [abs(en[i] - en[i - 1]) for i in range(1, len(en))]
    dE = [max(x, 1e-16) for x in dE]
    ax.semilogy(range(2, len(en) + 1), dE, 's--', label=f'1-site L={L}', markersize=3, alpha=0.7)
ax.set_xlabel('Half-sweep', fontsize=13)
ax.set_ylabel('|ΔE|', fontsize=13)
ax.set_title('AKLT: Energy Convergence per Sweep', fontsize=14)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figureAKLT/aklt_convergence.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figureAKLT/aklt_convergence.png")

# Summary
print("\n" + "=" * 70)
print("SUMMARY TABLE: ONE-SITE vs TWO-SITE DMRG")
print("=" * 70)
print(f"{'Model':<12} {'L':>4} {'E(2-site)':>18} {'E(1-site)':>18} {'E_exact':>18} {'|diff|':>10}")
print("-" * 84)
for L in sorted(heisenberg_2s.keys()):
    e2 = heisenberg_2s[L][-1]
    e1 = heisenberg_1s[L][-1]
    print(f"{'Heisenberg':<12} {L:>4} {e2:>18.10f} {e1:>18.10f} {'---':>18} {abs(e2-e1):>10.2e}")
for L in sorted(aklt_2s.keys()):
    e2 = aklt_2s[L][-1]
    e1 = aklt_1s[L][-1]
    ex = -2/3 * (L - 1)
    print(f"{'AKLT':<12} {L:>4} {e2:>18.10f} {e1:>18.10f} {ex:>18.10f} {abs(e2-e1):>10.2e}")

print("\nAll figures saved to figureAKLT/")
```

---
## 🧑‍💻 User

Saved: figureAKLT/aklt_convergence.png

======================================================================
SUMMARY TABLE: ONE-SITE vs TWO-SITE DMRG
======================================================================
Model           L          E(2-site)          E(1-site)            E_exact     |diff|
------------------------------------------------------------------------------------
Heisenberg      6      -2.4935771339      -2.4935771339                ---   2.22e-15
Heisenberg      8      -3.3749325987      -3.3749325987                ---   9.33e-15
Heisenberg     10      -4.2580352073      -4.2580352073                ---   4.80e-14
Heisenberg     12      -5.1420906328      -5.1420906328                ---   2.84e-14
AKLT            6      -3.3333333333      -3.3333333333      -3.3333333333   8.88e-16
AKLT            8      -4.6666666667      -4.6666666667      -4.6666666667   6.22e-15
AKLT           10      -6.0000000000      -6.0000000000      -6.0000000000   1.78e-15
AKLT           12      -7.3333333333      -7.3333333333      -7.3333333333   4.44e-15

All figures saved to figureAKLT/

---
## 🤖 Assistant

Everything is working perfectly. Let me provide a clean summary of what was implemented and verified:

## Summary

### Algorithms Implemented
1. **Two-site DMRG** (Algorithm 8 from notes) — primary algorithm
2. **One-site DMRG** (Algorithm 7 from notes) — initialized from two-site result

### Models
| Model | Spin | Local dim | MPO bond dim | Key result |
|-------|------|-----------|--------------|------------|
| Heisenberg | S=1/2 | d=2 | χ=5 | E/bond → -0.4432 (Bethe ansatz) |
| AKLT | S=1 | d=3 | χ=14 | E = -2(L-1)/3 exact to ~10⁻¹⁵ |

### Key implementation details (strictly following notes)
- **Matrix-free Lanczos** via `scipy.sparse.linalg.eigsh` with `LinearOperator`
- **Left environment swap**: `L[b,y,x]` not `L[b,x,y]` in matvec (critical convention)
- **Validated einsum contractions** exactly as in the notes
- **AKLT MPO**: uses distinct `Ō_a` (bar operators) for finishing channels vs `O_a` for starting channels
- **Exact AKLT MPS** verified with proper normalization

### Figures saved to `figureAKLT/`
| File | Content |
|------|---------|
| `heisenberg_energy.png` | Energy vs sweep + energy per bond vs 1/L |
| `heisenberg_convergence.png` | |ΔE| convergence per half-sweep |
| `aklt_energy.png` | Energy error vs sweep + final error vs L |
| `aklt_convergence.png` | |ΔE| convergence per half-sweep |

Both one-site and two-site DMRG agree to machine precision (~10⁻¹⁴) for all system sizes tested.
