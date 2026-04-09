## 🤖 Assistant

Numerical implementation of matrix product states and DMRG, based on the review article "The density-matrix renormalization group in the age of matrix product states" by U. Schollwöck, Annals of Physics 326, pages 96 through 192, 2011.

These implementation notes summarize the practical coding structure of matrix product states, matrix product operators, and finite-system DMRG algorithms for one-dimensional systems with open boundary conditions. The presentation is based on Schollwöck's review article, but rewritten with explicit implementation conventions suitable for numerical work.

Throughout, assume open boundary conditions, dense tensors without explicit quantum-number block sparsity, finite-system algorithms, and NumPy-like row-major array storage.

The tensor conventions are as follows. The MPS tensor on site i is the tensor A with three indices: alpha at bond i minus 1, sigma at site i, and alpha at bond i. Its shape is D at bond i minus 1, local physical dimension d at site i, and D at bond i. The MPO tensor on site i is the tensor W with four indices: beta at MPO bond i minus 1, beta at MPO bond i, sigma at site i, and sigma-prime at site i. Its shape is chi at bond i minus 1, chi at bond i, d at site i, and d at site i. The operator convention is that the local operator-valued MPO tensor is the sum over sigma and sigma-prime of the component W with indices beta at bond i minus 1, beta at bond i, sigma, and sigma-prime, multiplying the ket sigma and bra sigma-prime. Thus the first physical index is the ket index and the second physical index is the bra index.

For open boundary conditions, the MPS boundary bond dimensions are D at bond zero equal to one and D at bond L equal to one. The MPO boundary bond dimensions are chi at bond zero equal to one and chi at bond L equal to one.

Unless stated otherwise, all tensors should be stored as complex128.

For numerical stability and reproducibility, reshape conventions must be fixed once and used everywhere.

For a rank-three MPS tensor T on site i, with indices alpha at bond i minus 1, sigma at site i, and alpha at bond i, and with storage shape D left, d, and D right, define the left-grouped matrix by combining the left bond index alpha at bond i minus 1 and the physical index sigma at site i into one row index, while keeping alpha at bond i as the column index. This left-grouped matrix has size D at bond i minus 1 times d at site i by D at bond i.

Also define the right-grouped matrix by keeping alpha at bond i minus 1 as the row index and combining the physical index sigma at site i with alpha at bond i into one column index. This right-grouped matrix has size D at bond i minus 1 by d at site i times D at bond i.

In row-major C-order storage, the flattening used for the left-grouped matrix is this: the row index is alpha at bond i minus 1 times d at site i plus sigma at site i, and the column index is alpha at bond i.

In NumPy, the left-grouped matrix is obtained by reshaping T into shape D left times d by D right, using C order. The right-grouped matrix is obtained by reshaping T into shape D left by d times D right, using C order.

Typical inverse reshapes are these. Reshape U into A with shape D left, d, and D new, using C order. Reshape V dagger into B with shape D new, d, and D right, using C order.

All QR and SVD steps must be consistent with these conventions, and the inverse reshape must use the same memory order and grouped-index convention.

For a fixed physical index sigma, one may view the MPS tensor A at site i and physical index sigma as a matrix from the left bond to the right bond.

Using index notation avoids ambiguity. A tensor A at site i is left-canonical if the sum over alpha at bond i minus 1 and sigma at site i of A with indices alpha at bond i minus 1, sigma at site i, alpha at bond i, times the complex conjugate of A with indices alpha at bond i minus 1, sigma at site i, alpha-prime at bond i, equals the Kronecker delta between alpha at bond i and alpha-prime at bond i. Equivalently, if A at site i is reshaped as the left-grouped matrix, then the Hermitian conjugate of that matrix times the matrix itself is the identity matrix of size D at bond i.

A tensor B at site i is right-canonical if the sum over sigma at site i and alpha at bond i of B with indices alpha at bond i minus 1, sigma at site i, alpha at bond i, times the complex conjugate of B with indices alpha-prime at bond i minus 1, sigma at site i, alpha at bond i, equals the Kronecker delta between alpha at bond i minus 1 and alpha-prime at bond i minus 1. Equivalently, if B at site i is reshaped as the right-grouped matrix, then that matrix times its Hermitian conjugate is the identity matrix of size D at bond i minus 1.

A mixed-canonical state with orthogonality center at bond i may be written as a sum over alpha at bond i of the Schmidt singular value s at bond i and index alpha at bond i, multiplying the left Schmidt state alpha at bond i and the right Schmidt state alpha at bond i. The Schmidt singular values are nonnegative, and the left and right Schmidt states are orthonormal. For a normalized state, the sum over alpha at bond i of the squared Schmidt singular values is one.

For a left sweep by reduced QR, start with a general tensor M at site i. Reshape it as the matrix with combined row index alpha at bond i minus 1 and sigma at site i, and column index alpha at bond i. Perform a reduced QR factorization of this matrix into Q times R, where Q has size D at bond i minus 1 times d at site i by k, R has size k by D at bond i, and k is the minimum of D at bond i minus 1 times d at site i and D at bond i. Reshape Q into a left-canonical tensor A at site i with indices alpha at bond i minus 1, sigma at site i, and new alpha at bond i. The new bond dimension D tilde at bond i is k. Then absorb R into the next tensor by replacing M at site i plus 1, with indices new alpha at bond i, sigma at site i plus 1, and alpha at bond i plus 1, by the sum over old alpha at bond i of R with indices new alpha at bond i and old alpha at bond i, times M at site i plus 1 with indices old alpha at bond i, sigma at site i plus 1, and alpha at bond i plus 1.

The left-canonicalization by QR algorithm is as follows.

- Input the MPS tensors M at sites one through L.
- Output left-canonical tensors A at sites one through L minus one and the final center tensor C at site L.
- For i from one to L minus one:
  - Reshape M at site i into a matrix with shape D at bond i minus 1 times d at site i by D at bond i.
  - Compute the reduced QR factorization of this reshaped matrix into Q times R.
  - Reshape Q into A at site i with shape D at bond i minus 1, d at site i, and new D at bond i.
  - Absorb R into M at site i plus one using the absorption rule just described.
- Set the final center tensor C at site L equal to M at site L.

For canonicalization, use reduced QR without pivoting. Pivoted QR changes the bond basis by a permutation and is inconvenient unless the permutation is explicitly propagated into neighboring tensors.

For a right sweep by reduced QR, to right-canonicalize site i, reshape M at site i as a matrix with row index alpha at bond i minus 1 and combined column index sigma at site i with alpha at bond i. This matrix has size D at bond i minus 1 by d at site i times D at bond i. Compute a reduced QR factorization of the Hermitian transpose of this matrix. The Hermitian transpose is Q times R, with Q of size d at site i times D at bond i by k, R of size k by D at bond i minus 1, and k equal to the minimum of d at site i times D at bond i and D at bond i minus 1. Then M at site i equals R dagger times Q dagger. Reshape Q dagger into a right-canonical tensor B at site i with indices new alpha at bond i minus 1, sigma at site i, and alpha at bond i. The new bond dimension D tilde at bond i minus 1 is k. Absorb R dagger into site i minus 1 by replacing M at site i minus 1, with indices alpha at bond i minus 2, sigma at site i minus 1, and new alpha at bond i minus 1, by the sum over old alpha at bond i minus 1 of M at site i minus 1 with indices alpha at bond i minus 2, sigma at site i minus 1, and old alpha at bond i minus 1, times R dagger with indices old alpha at bond i minus 1 and new alpha at bond i minus 1.

For SVD truncation, start with a complex matrix X of size m by n. Compute the reduced SVD of X as U times S times V dagger, where U has size m by r, S is the diagonal matrix with singular values s one through s r, V dagger has size r by n, and the singular values are ordered from largest to smallest and are nonnegative. Truncate to bond dimension D by keeping the D largest singular values, so that X is approximated by U restricted to D columns, S restricted to D values, and V dagger restricted to D rows.

The discarded weight eta is the sum over j greater than D of s j squared. If the state is normalized, eta is the squared two-norm of the discarded Schmidt tail.

The SVD truncation algorithm is as follows.

- Input a matrix X, a maximum bond dimension D max, and a tolerance epsilon.
- Output the truncated U, S, and V dagger, together with the discarded weight eta.
- Compute the reduced SVD of X.
- Let D be the minimum of D max and the number of singular values greater than epsilon.
- If D equals zero, set D to one.
- Keep the first D singular values and vectors.
- Set eta to the sum of the squared discarded singular values with indices larger than D.

In production codes, the retained bond dimension is often chosen using both a hard cap D max and a discarded-weight tolerance, rather than only a singular-value threshold. A robust rule is to retain the smallest D less than or equal to D max such that the discarded weight is below a prescribed tolerance, while always keeping at least one singular value.

For the spin-one-half nearest-neighbor XXZ chain with field, the Hamiltonian is the sum from site one to site L minus one of the following bond term: J over two times S plus on site i times S minus on site i plus one, plus S minus on site i times S plus on site i plus one, plus J z times S z on site i times S z on site i plus one. Then subtract h times the sum from site one to site L of S z on site i.

A standard open-boundary MPO for this Hamiltonian has bond dimension chi equal to five. The bulk operator-valued MPO is a five by five lower-triangular operator-valued matrix. Its first row is identity, zero, zero, zero, zero. Its second row is S plus, zero, zero, zero, zero. Its third row is S minus, zero, zero, zero, zero. Its fourth row is S z, zero, zero, zero, zero. Its fifth row is minus h times S z, J over two times S minus, J over two times S plus, J z times S z, and identity.

This bulk MPO is an operator-valued matrix. The tensor components W with indices beta left, beta right, sigma, and sigma-prime are obtained by expanding each operator entry in the local basis, using the matrix element of operator O between bra sigma and ket sigma-prime.

For the local spin-one-half basis consisting of spin up and spin down, use the following matrices. S plus is the two by two matrix with entries zero, one in the first row and zero, zero in the second row. S minus is the two by two matrix with zero, zero in the first row and one, zero in the second row. S z is one half times the diagonal matrix with entries one and minus one.

For open boundary conditions, choose the left boundary MPO as the row vector with entries minus h times S z, J over two times S minus, J over two times S plus, J z times S z, and identity. Choose the right boundary MPO as the column vector with entries identity, S plus, S minus, S z, and minus h times S z. Use the bulk MPO on all intermediate sites, with the chosen lower-triangular convention.

In code, this means that the left boundary tensor has shape one, five, d, d and is the row vector selecting the bottom row of the bulk operator-valued MPO. The right boundary tensor has shape five, one, d, d and is the column vector selecting the leftmost column of the bulk operator-valued MPO.

The recommended unit test is: for L equal to two, three, and four, contract the MPO to a dense Hamiltonian and compare entrywise against the Hamiltonian constructed directly by Kronecker products of the local operators.

For the spin-one AKLT chain, the Hamiltonian is the sum from site one to site L minus one of the following bond term: the dot product of spin vector S at site i and spin vector S at site i plus one, plus one third times the square of that dot product. This is the Hamiltonian whose exact ground state is the AKLT MPS given later in these notes.

In the local spin-one basis consisting of plus one, zero, and minus one, use the standard spin-one matrices. S z is the three by three diagonal matrix with diagonal entries one, zero, and minus one. S plus is square root of two times the three by three matrix with first row zero, one, zero; second row zero, zero, one; and third row zero, zero, zero. S minus is square root of two times the three by three matrix with first row zero, zero, zero; second row one, zero, zero; and third row zero, one, zero.

Define the bilinear bond operator X on sites i and i plus one as the spin dot product between the two sites. It equals one half times S plus on site i times S minus on site i plus one plus S minus on site i times S plus on site i plus one, plus S z on site i times S z on site i plus one. Then the AKLT local bond Hamiltonian is X plus one third times X squared, and the total AKLT Hamiltonian is the sum of this local bond Hamiltonian from i equal to one to L minus one.

Introduce the operator lists as follows. O one is one over square root of two times S plus, O two is one over square root of two times S minus, and O three is S z. The barred operators are bar O one equal to one over square root of two times S minus, bar O two equal to one over square root of two times S plus, and bar O three equal to S z. Then the bilinear bond operator X is the sum over a from one to three of O a acting on site i times bar O a acting on site i plus one. Therefore X squared is the double sum over a and b from one to three of O a O b acting on site i times bar O a bar O b acting on site i plus one. Substituting this into the AKLT bond term gives the bond Hamiltonian as the sum over a from one to three of O a on site i times bar O a on site i plus one, plus one third times the double sum over a and b from one to three of O a O b on site i times bar O a bar O b on site i plus one.

This factorized bond expression is an exact nearest-neighbor MPO. A convenient MPO has bond dimension chi equal to one plus three plus nine plus one, which is fourteen. This slightly larger bond dimension is chosen because it leads to a completely explicit implementation without block-index ambiguities.

For implementation, order the AKLT MPO bond index as follows. Bond index beta equal to zero is the start channel. Bond indices beta equal to one, two, and three correspond to a equal to one, two, and three. Bond indices beta equal to four through twelve correspond to ordered pairs a, b in the sequence one one, one two, one three, two one, two two, two three, three one, three two, three three, taken in lexicographic order. Bond index beta equal to thirteen is the terminal or identity-propagation channel. This convention is used consistently in the bulk tensor and in both boundary tensors.

Define the row of O operators as O one, O two, O three. Define the column of barred O operators as bar O one, bar O two, bar O three. Define the row of second-order O operators as O one O one, O one O two, O one O three, O two O one, O two O two, O two O three, O three O one, O three O two, and O three O three. Define the column of second-order barred O operators as bar O one bar O one, bar O one bar O two, bar O one bar O three, bar O two bar O one, bar O two bar O two, bar O two bar O three, bar O three bar O one, bar O three bar O two, and bar O three bar O three. The ordering of the nine entries in these second-order lists is the same lexicographic ordering justdescribed.

The AKLT bulk operator-valued MPO tensor is a block matrix with block sizes one, three, nine, and one. In block form, its first block row is identity, the row of O operators, one third times the row of second-order O operators, and zero. Its second block row is zero, zero, zero, and the column of barred O operators. Its third block row is zero, zero, zero, and the column of second-order barred O operators. Its fourth block row is zero, zero, zero, and identity.

In words, channel zero to a with a equal to one, two, or three starts the bilinear term via O a. Channel zero to the pair a, b starts the biquadratic term via one third times O a O b. Channel a to thirteen finishes the bilinear term via bar O a. Channel pair a, b to thirteen finishes the biquadratic term via bar O a bar O b. Channel thirteen to thirteen propagates the identity after a term has been completed.

With this index convention, contracting the MPO generates exactly the sum from site one to site L minus one of the AKLT bond term, namely the spin dot product plus one third times the square of the spin dot product.

For open boundary conditions, use the AKLT left boundary tensor as the row block vector identity, the row of O operators, one third times the row of second-order O operators, and zero. Use the AKLT right boundary tensor as the column block vector zero, the column of barred O operators, the column of second-order barred O operators, and identity.

Thus, in the tensor convention of these notes, the left boundary tensor has shape one, fourteen, three, three. Each bulk tensor has shape fourteen, fourteen, three, three. The right boundary tensor has shape fourteen, one, three, three. The MPO physical indices are ordered as W at site i with indices beta at bond i minus one, beta at bond i, sigma, and sigma-prime, with sigma the ket index and sigma-prime the bra index, exactly as in the Heisenberg MPO section.

As in the Heisenberg MPO section, the operator-valued entries are converted into numerical MPO tensor components by taking the local matrix element between bra sigma and ket sigma-prime of the operator-valued entry with MPO bond indices beta left and beta right. Here sigma and sigma-prime belong to the set plus one, zero, minus one, and the local physical dimension is three.

A transparent implementation strategy is:

- Construct the local spin-one matrices S plus, S minus, and S z.
- Define O a and bar O a.
- Build the nine operators O a O b.
- Build the nine operators bar O a bar O b.
- Assemble the operator-valued boundary and bulk MPOs from the AKLT bulk block form and the AKLT boundary forms just described.
- Expand each operator entry into local matrix elements.

The recommended unit tests for the AKLT MPO are, for L equal to two, three, and four:

- Contract the MPO to a dense matrix.
- Compare against the Hamiltonian built directly from the AKLT Hamiltonian.
- Verify that the AKLT MPS given later in these notes has energy minus two thirds times L minus one for open chains, up to numerical precision.

For MPO environments, for a center at site i, let L at site i denote the contraction of sites one through i minus one into a left environment, and let R at site i denote the contraction of sites i plus one through L into a right environment.

Store the left and right environments as tensors. The left environment L at site i has one MPO bond index beta at bond i minus one and two MPS bond indices alpha at bond i minus one and alpha-prime at bond i minus one. Its array shape is chi at bond i minus one, D at bond i minus one, and D at bond i minus one. The right environment R at site i has one MPO bond index beta at bond i and two MPS bond indices alpha at bond i and alpha-prime at bond i. Its array shape is chi at bond i, D at bond i, and D at bond i.

For the left environment update, suppose sites one through i minus one are represented by left-canonical tensors. Then the left environment is updated recursively as follows. The new left environment at site i, with MPO index beta at bond i minus one and MPS indices alpha at bond i minus one and alpha-prime at bond i minus one, is the sum over alpha and alpha-prime at bond i minus two, over physical indices sigma and sigma-prime, and over beta at bond i minus two, of the old left environment at site i minus one, times A at site i minus one, times W at site i minus one, times the complex conjugate of A at site i minus one. The indices are routed so that the old environment uses beta at bond i minus two and the two old MPS bonds, A connects old alpha, sigma, and new alpha, W connects old beta, new beta, sigma, and sigma-prime, and the conjugate A connects old alpha-prime, sigma-prime, and new alpha-prime.

With array conventions Lold with indices b, x, y; A with indices x, s, a; and W with indices b, B, s, t, the validated NumPy implementation is the contraction of Lold, A, W, and the conjugate of A over shared indices b, x, y, s, and t, producing Lnew with indices B, a, c. In einsum form this is bxy, xsa, bBst, ytc producing Bac.

For the right environment update, suppose sites i plus one through L are represented by right-canonical tensors. Then the right environment is updated recursively as follows. The new right environment at site i, with MPO index beta at bond i and MPS indices alpha at bond i and alpha-prime at bond i, is the sum over alpha and alpha-prime at bond i plus one, over physical indices sigma and sigma-prime, and over beta at bond i plus one, of B at site i plus one, times W at site i plus one, times the old right environment at site i plus one, times the complex conjugate of B at site i plus one. The indices are routed so that B connects alpha at bond i, sigma, and alpha at bond i plus one; W connects beta at bond i, beta at bond i plus one, sigma, and sigma-prime; Rold uses beta at bond i plus one and the two MPS bonds at bond i plus one; and the conjugate B connects alpha-prime at bond i, sigma-prime, and alpha-prime at bond i plus one.

With array conventions Rold with indices B, a, c; B tensor with indices x, s, a; and W with indices b, B, s, t, the validated NumPy implementation is the contraction of B, W, Rold, and the conjugate of B over shared indices B, a, c, s, and t, producing Rnew with indices b, x, y. In einsum form this is xsa, bBst, Bac, ytc producing bxy.

For open boundary conditions, initialize the left boundary environment at site one so that the only element is one at beta zero equal to the only MPO bond value and MPS indices one, one. Similarly initialize the right boundary environment at site L so that the only element is one. The dimensions are D zero, D L, chi zero, and chi L all equal to one. In code, these are simply arrays of shape one, one, one with complex128 dtype, with the only element set to one.

The update formulas above define the stored left and right environment tensors. However, for the tensor, reshape, and MPO conventions used in these notes, the stored left environment does not enter the local effective-Hamiltonian action with the same bond-index order as it is stored. More precisely, if the stored left environment is represented in code as L with indices b, x, y, then in the validated one-site and two-site local contractions it must be used as L with indices b, y, x.

This point is easy to miss. A local operator built from environments can remain Hermitian even when this ordering is wrong, while still failing to agree with the exact projected operator P dagger H P. Therefore one should distinguish carefully between the recursion defining the stored environments and the way those stored tensors are inserted into the local effective-Hamiltonian action.

The practical recommendation is that the left and right environment update routines should be tested independently against brute-force contractions on very small systems. After that, the resulting one-site and two-site environment-based local operators should be compared explicitly against dense projected references. In practice, this second test is the decisive one: it verifies not only that the environments are built recursively, but also that they are inserted into the local effective Hamiltonian with the correct index ordering.

For the one-site effective Hamiltonian, let the center tensor at site i be M with indices alpha at bond i minus one, sigma at site i, and alpha at bond i. The local variational problem is the effective Hamiltonian acting on vector v equals energy E times vector v, where v denotes the vectorized form of the center tensor.

The one-site effective Hamiltonian is obtained by contracting the full MPO with all MPS tensors outside site i, leaving only the local tensor M at site i open. In principle one may write this object symbolically as a contraction of a left environment, one local MPO tensor, and a right environment. In practice, however, the safest way to specify the operator, and the one that is directly useful for coding, is through its validated matrix-free contraction sequence.

For the conventions used in these notes, the stored left environment tensor does not enter the local action with the same MPS-bond index order as it is stored. If the stored array is L with indices b, x, y, then the validated local contraction uses it as L with indices b, y, x. Equivalently, the left environment enters the one-site effective Hamiltonian with its two MPS-bond indices interchanged relative to the most naive symbolic reading.

Let the stored left environment at site i have MPO index beta at bond i minus one and MPS bond indices alpha at bond i minus one and alpha-prime at bond i minus one. Let the stored right environment at site i have MPO index beta at bond i and MPS bond indices alpha at bond i and alpha-prime at bond i. The validated one-site matrix-free action proceeds in three contractions.

First, contract the stored left environment with the center tensor M, but use the left environment with its two MPS-bond indices interchanged. The intermediate X has indices beta at bond i minus one, alpha at bond i minus one, sigma at site i, and alpha-prime at bond i. It is obtained by summing over alpha-prime at bond i minus one of the stored left environment with MPS indices alpha-prime then alpha, times M with alpha-prime, sigma, and alpha-prime at the right bond.

Second, contract the MPO tensor with X. The intermediate Y has indices beta at bond i, alpha at bond i minus one, sigma-prime at site i, and alpha-prime at bond i. It is obtained by summing over beta at bond i minus one and sigma at site i of W at site i with indices beta left, beta right, sigma-prime, and sigma, times X with beta left, alpha left, sigma, and right bond alpha-prime.

Third, contract Y with the stored right environment. The output effective-Hamiltonian action on M has indices alpha at bond i minus one, sigma-prime at site i, and alpha at bond i. It is obtained by summing over beta at bond i and alpha-prime at bond i of Y times the right environment with MPS indices alpha-prime and alpha.

This is the convention-correct form for the one-site effective Hamiltonian in the present notes. In particular, the first step involves the stored left environment with MPS indices alpha-prime first and alpha second, not alpha first and alpha-prime second.

With arrays L with indices b, x, y for the stored left environment, M with indices y, s, z for the local center tensor, W with indices b, B, s, t for the MPO tensor, and R with indices B, z, a for the stored right environment, the validated implementation is as follows. Contract L and M using L as b, y, x and M as y, s, z, producing X with indices b, x, s, z. Then contract W and X over b and s, producing Y with indices B, x, t, z. Then contract Y and R over B and z, producing HM with indices x, t, a. In einsum form, these are byx with ysz producing bxsz, then bBst with bxsz producing Bxtz, then Bxtz with Bza producing xta.

Here HM has the same shape as M. The use of the swapped left-environment ordering b, y, x rather than b, x, y in the first contraction is essential.

The MPO tensor is stored with indices beta at bond i minus one, beta at bond i, sigma, and sigma-prime, with the first physical index sigma the ket or output index and the second physical index sigma-prime the bra or input index. The contraction in the second step of the one-site action is the validated routing consistent with this convention and with the environment storage described above.

If M at site i has shape D at bond i minus one, d at site i, and D at bond i, then the local vector-space dimension is D at bond i minus one times d at site i times D at bond i. The eigensolver vector v has that length and is identified with the tensor M by reshaping v into shape D left, d, and D right, using the fixed C-order convention. After applying the local effective Hamiltonian, the output tensor is flattened back into a vector with the same convention.

A practical wrapper imports LinearOperator and eigsh from scipy sparse linear algebra. The matvec routine takes v, L, W, R, D left, d, and D right. It reshapes v into M with shape D left, d, D right using C order, performs the validated one-site contraction sequence just described, and returns HM flattened into length D left times d times D right using C order. Then set the local dimension to D left times d times D right, define a LinearOperator with shape local dimension by local dimension, the matvec just described, and dtype complex128. The eigensolver should be initialized with the current local tensor flattened using the same C-order convention.

For small systems, one should explicitly build the dense projected operator P dagger H P for the one-site center and compare it against the dense matrix obtained by repeated application of the matrix-free routine above.

Hermiticity of the environment-built local operator is necessary but not sufficient. With the conventions of these notes, a naive contraction using the stored left environment without the required index interchange can still produce a Hermitian operator, while failing to agree with the true projected operator P dagger H P. Therefore the contraction sequence described above should be treated as the authoritative implementation for the present conventions.

For the two-site effective Hamiltonian, at bond i, i plus one, define the two-site center tensor Theta with indices alpha at bond i minus one, sigma at site i, sigma at site i plus one, and alpha at bond i plus one. The state is written as the sum over these four indices of Theta times the left Schmidt state alpha at bond i minus one, the local ket sigma at site i, the local ket sigma at site i plus one, and the right Schmidt state alpha at bond i plus one.

The two-site local variational problem is the direct analogue of the standard two-site DMRG superblock problem, but expressed entirely in terms of MPS and MPO environments and a matrix-free local action.

In principle, the effective Hamiltonian at bond i, i plus one may be written symbolically as a contraction of a left environment built from sites one through i minus one, the two local MPO tensors at sites i and i plus one, and a right environment built from sites i plus two through L. In practice, however, the most reliable specification is again the validated contraction sequence itself.

For the conventions used in these notes, two implementation details are essential. First, the stored left environment enters the local action with its two MPS-bond indices interchanged. Second, the local MPO tensors must be contracted with the physical-index routing that is consistent with the validated implementation below. These points are easy to get wrong, and more naive contractions can produce Hermitian local operators that nevertheless fail to agree with the true projected operator P dagger H P.

Let the stored left environment at site i be built from sites one through i minus one, with MPO index beta at bond i minus one and MPS indices alpha and alpha-prime at bond i minus one. Let the stored right environment at site i plus one be built from sites i plus two through L, with MPO index beta at bond i plus one and MPS indices alpha and alpha-prime at bond i plus one.

The validated matrix-free two-site action proceeds as follows.

First, contract the stored left environment with Theta, again using the left environment with its two MPS-bond indices interchanged. The intermediate X has indices beta at bond i minus one, alpha at bond i minus one, sigma at site i, sigma at site i plus one, and alpha-prime at bond i plus one. It is obtained by summing over alpha-prime at bond i minus one of the stored left environment with MPS indices alpha-prime then alpha, times Theta with alpha-prime, sigma at site i, sigma at site i plus one, and right bond alpha-prime.

Second, contract the site i MPO tensor with X. The intermediate Y has indices beta at bond i, alpha at bond i minus one, sigma-prime at site i, sigma at site i plus one, and alpha-prime at bond i plus one. It is obtained by summing over beta at bond i minus one and sigma at site i of W at site i with indices beta at bond i minus one, beta at bond i, sigma at site i, and sigma-prime at site i, times X with beta at bond i minus one, alpha at bond i minus one, sigma at site i, sigma at site i plus one, and alpha-prime at bond i plus one.

Third, contract the site i plus one MPO tensor with Y. The intermediate Z has indices beta at bond i plus one, alpha at bond i minus one, sigma-prime at site i, sigma-prime at site i plus one, and alpha-prime at bond i plus one. It is obtained by summing over beta at bond i and sigma at site i plus one of W at site i plus one with indices beta at bond i, beta at bond i plus one, sigma at site i plus one, and sigma-prime at site i plus one, times Y.

Fourth, contract Z with the stored right environment. The output effective-Hamiltonian action on Theta has indices alpha at bond i minus one, sigma-prime at site i, sigma-prime at site i plus one, and alpha at bond i plus one. It is obtained by summing over beta at bond i plus one and alpha-prime at bond i plus one of Z times the right environment with MPS indices alpha-prime and alpha.

As in the one-site case, the first line uses the stored left environment with MPS indices alpha-prime first and alpha second, rather than alpha first and alpha-prime second.

Let the two-site tensor be stored as Theta with indices alpha at bond i minus one, sigma at site i, sigma at site i plus one, and alpha at bond i plus one, with shape D at bond i minus one, d at site i, d at site i plus one, and D at bond i plus one.

With arrays L with indices b, x, y for the stored left environment, Theta with indices y, u, v, z for the input two-site tensor, W1 with indices b, B, u, s for the MPO tensor on site i, W2 with indices B, C, v, t for the MPO tensor on site i plus one, and R with indices C, z, a for the stored right environment, the validated matrix-free contraction is as follows. Contract L, used as b, y, x, with Theta over y, producing X with indices b, x, u, v, z. Contract W1 and X over b and u, producing Y with indices B, x, s, v, z. Contract W2 and Y over B and v, producing Z with indices C, x, s, t, z. Contract Z and R over C and z, producing HT with indices x, s, t, a. In einsum form, these are byx with yuvz producing bxuvz, then bBus with bxuvz producing Bxsvz, then BCvt with Bxsvz producing Cxstz, then Cxstz with Cza producing xsta.

The output tensor HT with indices x, s, t, a has the same shape as Theta.

The MPO tensors are stored with indices beta at bond i minus one, beta at bond i, sigma, and sigma-prime, with the first physical index the ket or output index and the second the bra or input index. The routing in the two-site contraction above is therefore convention-dependent and should be used exactly as written.

If Theta has shape D at bond i minus one, d at site i, d at site i plus one, and D at bond i plus one, then the local vector-space dimension is D at bond i minus one times d at site i times d at site i plus one times D at bond i plus one. The eigensolver vector v has this length and is identified with Theta by reshaping v into that four-index shape, again using the fixed C-order convention.

A practical wrapper takes v, L, W1, W2, R, D left, d one, d two, and D right. It reshapes v into Theta with shape D left, d one, d two, D right using C order, performs the validated two-site contraction sequence just described, and returns HT flattened into length D left times d one times d two times D right using C order. This routine is then passed to LinearOperator and solved by Lanczos or Davidson exactly as in the one-site case.

For small systems, the dense matrix generated by repeated application of the matrix-free two-site routine above should be compared entrywise against the exact projected operator P dagger H P. This is the decisive correctness test for the environment-based implementation.

Hermiticity alone is not a sufficient diagnostic. In the present conventions, one can obtain a two-site environment-built operator that is Hermitian and numerically stable, yet still incorrect because of an index-order mismatch in the left environment or in the MPO physical-index routing. Therefore the two-site contraction sequence described above should be treated as the authoritative implementation for the present conventions.

In practical DMRG implementations, the local effective Hamiltonian should never be formed as a dense matrix except for unit tests on very small systems. Instead, one implements only the map from v to the effective Hamiltonian applied to v, and passes this map to an iterative eigensolver such as Lanczos or Davidson.

For the tensor, MPO, and environment conventions adopted in these notes, the stored left environment tensor enters the local effective-Hamiltonian action with its two MPS-bond indices interchanged. In array form, if the stored left environment is L with indices b, x, y, then the validated local matrix-free contractions use it as L with indices b, y, x. This point is subtle but essential. A local operator built from environments can remain Hermitian even when this ordering is wrong, while still failing to agree with the true projected operator P dagger H P. Accordingly, the one-site and two-site matrix-free contractions given here should be treated as the authoritative implementations for the conventions of these notes.

For a one-site center tensor M at site i with shape D at bond i minus one, d at site i, and D at bond i, define the local vector-space dimension as the product of those three dimensions. The eigensolver vector v is identified with M by reshaping v into that shape using the fixed C-order convention. After applying the local effective Hamiltonian, the result is reshaped back to a vector with the same convention.

The authoritative one-site matrix-free contraction, in array form, uses stored left environment L with indices b, x, y; local tensor M with indices y, s, z; MPO tensor W with indices b, B, s, t; and right environment R with indices B, z, a. Contract L as b, y, x with M to get X with indices b, x, s, z. Contract W with X to get Y with indices B, x, t, z. Contract Y with R to get HM with indices x, t, a. In einsum form, this is byx with ysz producing bxsz, then bBst with bxsz producing Bxtz, then Bxtz with Bza producing xta.

A practical NumPy and scipy interface imports LinearOperator and eigsh. The one-site matvec reshapes the input vector into M, performs the validated contractions, and returns the flattened output. The LinearOperator has shape local dimension by local dimension and dtype complex128. The eigensolver call can use eigsh with one eigenpair, asking for the algebraically smallest eigenvalue, and the current local tensor flattened in C order should be supplied as v0. The resulting eigenvector is reshaped back into the local tensor shape using C order.

For a Hermitian MPO and correctly built environments, the local operator should satisfy the usual Hermiticity inner-product relation: inner product of x with the effective Hamiltonian applied to y equals inner product of the effective Hamiltonian applied to x with y, up to numerical precision. However, Hermiticity alone is not a sufficient correctness test. For small systems, one should also compare the dense matrix obtained from the matrix-free routine against the exact projected operator P dagger H P.

For a two-site center tensor Theta with shape D at bond i minus one, d at site i, d at site i plus one, and D at bond i plus one, the local vector-space dimension is the product of those four dimensions. Again, the eigensolver vector v is identified with Theta by reshaping v into that shape. After applying the local effective Hamiltonian, the output tensor is flattened back to a vector using the same C-order convention.

The authoritative two-site matrix-free contraction, in array form, uses stored left environment L with indices b, x, y; two-site tensor Theta with indices y, u, v, z; MPO tensors W1 with indices b, B, u, s and W2 with indices B, C, v, t; and stored right environment R with indices C, z, a. Contract L as b, y, x with Theta to get X with indices b, x, u, v, z. Contract W1 with X to get Y with indices B, x, s, v, z. Contract W2 with Y to get Z with indices C, x, s, t, z. Contract Z with R to get HT with indices x, s, t, a. In einsum form, this is byx with yuvz producing bxuvz, then bBus with bxuvz producing Bxsvz, then BCvt with Bxsvz producing Cxstz, then Cxstz with Cza producing xsta.

A practical two-site wrapper reshapes the input vector into Theta, performs these contractions, and returns HT flattened in C order. This is passed to LinearOperator in the same way as for the one-site case.

The two-site contraction above was explicitly validated against the dense projected operator P dagger H P for small systems. This validation is essential, because more naive einsum routings can produce Hermitian local operators that are nevertheless incorrect.

For first implementations, Lanczos is usually sufficient and easiest to access through standard sparse-eigensolver interfaces. Davidson can be advantageous when the local dimension becomes larger or when a useful diagonal preconditioner is available. Lanczos is the simplest robust choice for initial code. Davidson is often faster for large local problems, but requires more implementation effort. In both cases, using the current local tensor as the initial guess is essential for sweep efficiency.

For very small dimensions only, one should explicitly build the dense local projected operator with size local dimension by local dimension. Compare dense matrix-vector multiplication versus the matrix-free routine, the lowest eigenvalue from dense diagonalization versus Lanczos or Davidson, Hermiticity of the dense reference matrix, and agreement of the environment-built operator with the exact projected operator P dagger H P. This unit test should be performed for both the one-site and two-site local solvers before running full sweeps.

Two-site DMRG is the most robust finite-system ground-state algorithm and is recommended for a first implementation. It avoids many local-minimum problems of one-site DMRG and dynamically adjusts the bond dimension before truncation.

At bond i, i plus one, define left-orthonormal Schmidt states on sites one through i minus one and right-orthonormal Schmidt states on sites i plus two through L. The variational two-site center tensor is Theta with indices alpha at bond i minus one, sigma at site i, sigma at site i plus one, and alpha at bond i plus one. The state is written as the sum over these indices of Theta times the left Schmidt state, the two local physical states, and the right Schmidt state.

This form is the direct analogue of the usual two-site DMRG superblock state and should be treated as the primary variational object. It is not necessary to define Theta through pre-existing tensors A at site i and B at site i plus one.

Given a matrix-free routine for mapping Theta to the effective Hamiltonian applied to Theta, solve for the lowest eigenpair using Lanczos or Davidson. The local vector-space dimension is D at bond i minus one times d at site i times d at site i plus one times D at bond i plus one. For spin-one-half chains with uniform physical dimension two, this local dimension is four times D at bond i minus one times D at bond i plus one.

The eigensolver should accept an initial guess. A good choice is the current two-site center tensor before optimization, flattened with the same C-order convention used everywhere else. This substantially accelerates convergence once sweeps are underway.

After optimization, form the two-site center tensor Theta with indices alpha, sigma, sigma-prime, beta by contracting M at site i, with indices alpha, sigma, gamma, with M at site i plus 1, with indices gamma, sigma-prime, beta, over the shared bond index gamma.

After optimization, reshape the two-site tensor as a matrix by combining alpha at bond i minus one with sigma at site i into the row index, and combining sigma at site i plus one with alpha at bond i plus one into the column index. This matrix has size D at bond i minus one times d at site i by d at site i plus one times D at bond i plus one. Compute its reduced SVD as U times S times V dagger.

Let the full bond dimension be the rank of Theta, and let the new bond dimension be the minimum of D max and that full rank, or more generally choose the new bond dimension from a singular-value threshold or discarded-weight tolerance. Keep only the largest new-D singular values. The discarded weight at bond i is the sum over singular values with index greater than the new bond dimension of the squared singular values.

For a left-to-right sweep, set A at site i by reshaping U into shape D at bond i minus one, d at site i, and new D. Set the new center tensor M at site i plus one by reshaping S times V dagger into shape new D, d at site i plus one, and D at bond i plus one. Then A at site i is left-canonical, and the orthogonality center has moved to site i plus one.

For a right-to-left sweep, instead set the new center tensor M at site i by reshaping U times S into shape D at bond i minus one, d at site i, and new D. Set B at site i plus one by reshaping V dagger into shape new D, d at site i plus one, and D at bond i plus one. Then B at site i plus one is right-canonical, and the orthogonality center moves to site i.

The reshapes in these updates are only correct if the two-site matrix reshape uses exactly the same flattening convention as the code.

After splitting the optimized two-site tensor, the environments must be updated so that the next local problem is built in the correct mixed-canonical gauge.

In a left-to-right sweep, after obtaining the new left-canonical tensor A at site i, update the left environment from site i to site i plus one. The new left environment has MPO index beta at bond i and MPS indices alpha and alpha-prime at bond i. It is obtained by summing over alpha and alpha-prime at bond i minus one, over physical indices sigma and sigma-prime at site i, and over beta at bond i minus one, of the old left environment at site i, times A at site i, times W at site i, times the complex conjugate of A at site i.

In a right-to-left sweep, after obtaining the new right-canonical tensor B at site i plus one, update the right environment from site i plus one to site i. The new right environment has MPO index beta at bond i and MPS indices alpha and alpha-prime at bond i. It is obtained by summing over alpha and alpha-prime at bond i plus one, over physical indices sigma and sigma-prime at site i plus one, and over beta at bond i plus one, of B at site i plus one, times W at site i plus one, times the old right environment at site i plus one, times the complex conjugate of B at site i plus one.

These formulas define the stored environments. As emphasized earlier, for the conventions of these notes the stored left environment enters the local effective-Hamiltonian action with its two MPS-bond indices interchanged relative to its storage order. Thus the environment-recursion formulas and the local-action formulas must not be conflated.

The full two-site DMRG algorithm is as follows.

- Input the MPO tensors W, the initial MPS tensors M, and the maximum bond dimension D max.
- Output the approximate ground-state MPS and energy.
- Bring the initial MPS into mixed-canonical form.
- Build all right environments.
- Initialize the left boundary environment.
- Repeat until the energy, variance, and discarded-weight criteria are converged:
  - Perform a left-to-right sweep:
    - For i from one to L minus one:
      - Form the current two-site center tensor Theta at sites i and i plus one.
      - Solve the lowest-eigenvalue problem for the effective Hamiltonian acting on Theta.
      - Reshape Theta as the two-site matrix with left physical index grouped with the left bond and right physical index grouped with the right bond.
      - Compute the reduced SVD of Theta.
      - Truncate to new bond dimension no larger than D max.
      - Set A at site i by reshaping U into shape D at bond i minus one, d at site i, and new D.
      - Set M at site i plus one by reshaping S times V dagger into shape new D, d at site i plus one, and D at bond i plus one.
      - Record the discarded weight at bond i.
      - Update the left environment at site i plus one.
  - Perform a right-to-left sweep:
    - For i from L minus one down to one:
      - Form the current two-site center tensor Theta at sites i and i plus one.
      - Solve the lowest-eigenvalue problem.
      - Reshape Theta as the same two-site matrix.
      - Compute the reduced SVD of Theta.
      - Truncate to new bond dimension no larger than D max.
      - Set M at site i by reshaping U times S into shape D at bond i minus one, d at site i, and new D.
      - Set B at site i plus one by reshaping V dagger into shape new D, d at site i plus one, and D at bond i plus one.
      - Record the discarded weight at bond i.
      - Update the right environment at site i.

Practical comments for coding are these. Two-site DMRG should be the first DMRG implementation, because it is much easier to debug than one-site DMRG. Keep the orthogonality center explicit: during a left-to-right sweep, after the split the center lives on site i plus one as M at site i plus one; during a right-to-left sweep, it lives on site i as M at site i. Do not compare tensors directly across sweeps without gauge fixing. Always unit test the local effective-Hamiltonian matvec against a dense reference for small dimensions. Monitor the maximum discarded weight during each sweep, defined as the maximum over bonds of the discarded weight at that bond. Benchmark first against exact diagonalization for small Heisenberg chains before attempting larger systems.

One-site DMRG is best treated as an advanced refinement of the two-site algorithm. In practice, a robust implementation typically uses several two-site sweeps first and only then switches to one-site sweeps, optionally with subspace expansion or density-matrix perturbation.

In one-site DMRG, the MPS is kept in mixed-canonical form with center at site i. The state is written as a sum over alpha at bond i minus one, sigma at site i, and alpha at bond i of the center tensor M times the left Schmidt state, the local physical state, and the right Schmidt state. The local optimization is then performed over the center tensor M at site i while the left and right Schmidt bases are kept fixed.

The one-site DMRG algorithm is as follows.

- Input the MPO tensors W, the initial MPS tensors M, and the maximum bond dimension D max.
- Output the approximate ground-state MPS and energy.
- Bring the initial MPS into mixed-canonical form.
- Build all right environments.
- Initialize the left boundary environment.
- Repeat until the energy and variance are converged:
  - Perform a left-to-right sweep:
    - For i from one to L minus one:
      - Solve the one-site effective-Hamiltonian problem using the validated matrix-free contraction described earlier.
      - Reshape the optimized center tensor as a left-grouped matrix.
      - Compute an SVD of that matrix.
      - Retain at most D max singular values.
      - Reshape U into A at site i as a left-canonical tensor.
      - Absorb S times V dagger into M at site i plus one.
      - Update the left environment at site i plus one.
  - Perform a right-to-left sweep:
    - For i from L down to two:
      - Solve the one-site effective-Hamiltonian problem using the validated matrix-free contraction described earlier.
      - Reshape the optimized center tensor as a right-grouped matrix.
      - Compute an SVD of that matrix.
      - Retain at most D max singular values.
      - Reshape V dagger using the inverse of the right-grouped convention to obtain a right-canonical tensor B at site i.
      - Absorb U times S into M at site i minus one.
      - Update the right environment at site i minus one.

In standard one-site DMRG, the SVD step mainly regauges the state and shifts the orthogonality center. Unlike two-site DMRG, it does not naturally enlarge the accessible bond space. This is one reason why one-site DMRG is more prone to local minima unless combined with subspace expansion or density-matrix perturbation.

For the conventions of these notes, the one-site local matrix-free action used inside one-site DMRG must be the validated contraction summarized earlier, rather than a naive contraction built directly from the stored left environment without the required bond-index interchange.

One-site DMRG is prone to local minima and should not be used as a first standalone implementation. A robust implementation should either start with several two-site sweeps or use a subspace-expansion or density-matrix perturbation method.

An exact MPS representation of the AKLT state is, up to gauge, given by three two by two matrices. The matrix A plus has first row zero and square root of two thirds, and second row zero and zero. The matrix A zero is diagonal with entries minus one over square root of three and plus one over square root of three. The matrix A minus has first row zero and zero, and second row minus square root of two thirds and zero.

For this gauge, the sum over physical index sigma in the set plus, zero, minus of A sigma times A sigma dagger is the two by two identity. Therefore these tensors are right-canonical in the right-canonical convention used in these notes.

This state is an excellent benchmark. The AKLT ground state admits an exact MPS representation with bond dimension two. For open boundary conditions, the AKLT Hamiltonian has a fourfold degenerate ground-state manifold associated with spin-one-half edge modes, and every ground state in that manifold has exact energy minus two thirds times L minus one.

Reliable convergence diagnostics for DMRG should include energy change smaller than a chosen energy tolerance, the energy variance, the maximum discarded weight during a sweep, and stability of the entanglement spectrum or local observables. The energy variance is the expectation value of H squared minus the square of the expectation value of H.

For small systems, the variance can be benchmarked against dense linear algebra. In production calculations, it can be obtained either from an MPO representation of H squared or by first computing H applied to psi as an MPS and then evaluating its norm. One should keep in mind that an MPO for H squared can have a significantly larger bond dimension than the MPO for H.

Direct tensor-by-tensor comparison is not gauge-invariant and should not be used as a primary convergence criterion unless the gauge is fixed identically before comparison.

Before production calculations, the implementation should be validated in a strictly controlled order. The purpose of this checklist is not only to test individual routines, but also to ensure that the tensor-index conventions, reshape conventions, and environment contractions are mutually consistent.

First, run a canonicalization test. Starting from random tensors or a random MPS, perform left- and right-canonicalization sweeps and verify the defining relations. For left-canonical tensors, the sum over alpha at bond i minus one and sigma at site i of A times complex conjugate A must give the Kronecker delta on the right bond. For right-canonical tensors, the sum over sigma at site i and alpha at bond i of B times complex conjugate B must give the Kronecker delta on the left bond. These tests verify both the QR or SVD logic and the fixed reshape conventions.

Second, run an MPO construction test. For small chains such as L equal to two, three, and four, contract the MPO to a dense matrix and compare it entrywise against a Hamiltonian built directly from Kronecker products of local operators. This test should be performed for both the Heisenberg MPO and the AKLT MPO.

Third, run an environment-recursion test. Verify that the left and right environment updates reproduce brute-force contractions on very small systems. This confirms that the recursive environment-building routines are internally consistent.

Fourth, run a Hermiticity test of local matrix-free operators. For random small tensors, verify numerically that the inner product of x with the effective Hamiltonian applied to y equals the inner product of the effective Hamiltonian applied to x with y, up to numerical precision, for both one-site and two-site matrix-free local actions.

Fifth, run a one-site projected-operator test. For a small mixed-canonical MPS, explicitly construct the dense projected one-site operator P dagger H P and compare it against the dense matrix obtained by repeated application of the matrix-free one-site environment-based local action. For the conventions of these notes, the validated one-site contraction is: contract L as b, y, x with M as y, s, z to produce X with indices b, x, s, z; contract W as b, B, s, t with X to produce Y with indices B, x, t, z; and contract Y with R as B, z, a to produce HM with indices x, t, a.

Sixth, run a two-site projected-operator test. For a small mixed-canonical MPS, explicitly construct the dense projected two-site operator P dagger H P and compare it against the dense matrix obtained by repeated application of the matrix-free two-site environment-based local action. For the conventions of these notes, the validated two-site contraction is: contract L as b, y, x with Theta as y, u, v, z to produce X with indices b, x, u, v, z; contract W1 as b, B, u, s with X to produce Y with indices B, x, s, v, z; contract W2 as B, C, v, t with Y to produce Z with indices C, x, s, t, z; and contract Z with R as C, z, a to produce HT with indices x, s, t, a. This comparison against the dense projected operator is the decisive local correctness test.

Seventh, run a two-site DMRG benchmark for Heisenberg chains. Run finite-system two-site DMRG for small spin-one-half Heisenberg chains and compare the converged MPS energy against exact diagonalization. Agreement to numerical precision on small systems is the main global benchmark for the implementation.

Eighth, run the AKLT benchmark. Validate the AKLT implementation in two ways: compare the DMRG ground-state energy against the exact open-chain result minus two thirds times L minus one, and evaluate the energy of the exact AKLT MPS and verify that it reproduces the same result.

Ninth, in production runs, monitor convergence diagnostics. After the implementation is validated locally and globally, monitor the energy change between sweeps, the maximum discarded weight, the eventual energy variance, and stability of local observables and entanglement spectra.

The most important practical warning is that, for the conventions used in these notes, Hermiticity of the environment-built local operator is not sufficient to guarantee correctness. In particular, an incorrect treatment of the stored left environment or an incorrect MPO physical-index routing can still yield a Hermitian local operator, yet one that differs from the true projected operator P dagger H P. Therefore no full DMRG sweeps should be trusted until the one-site and two-site local operators have been compared explicitly against dense projected references on small systems.

If a full DMRG calculation produces stable but incorrect energies, the safest debugging sequence is: verify canonicalization; verify MPO-to-dense agreement; verify left and right environment updates separately; verify the one-site local projected operator; verify the two-site local projected operator; and only then test full sweeps. In practice, the projected-operator tests are usually more informative than Hermiticity tests alone.

For the conventions of these notes, the one-site and two-site matrix-free contraction templates written above should be treated as authoritative reference implementations. Future optimizations, refactorings, or changes of contraction order should always be benchmarked back against these forms on small systems before being used in production calculations.

For a first reliable implementation, the following strategy is strongly recommended.

Implement two-site DMRG first. Among standard finite-system algorithms, two-site DMRG is the most robust place to begin. It is substantially easier to debug than one-site DMRG, naturally allows bond dimensions to adapt before truncation, and provides a direct setting in which the local variational space is large enough to avoid many of the local-minimum problems of one-site updates.

Fix tensor conventions once and use them everywhere. In particular, fix the MPS tensor convention with indices alpha at bond i minus one, sigma at site i, and alpha at bond i. Fix the MPO tensor convention with indices beta at bond i minus one, beta at bond i, sigma at site i, and sigma-prime at site i, with the first physical index the ket or output index and the second the bra or input index. Fix the reshape rules used in QR, SVD, vectorization, and inverse reshaping. Fix the memory order, for example NumPy C order. Small inconsistencies in these choices are among the most common sources of silent errors.

Use reduced QR for regauging and SVD for truncation. Reduced QR without pivoting is convenient for moving orthogonality centers and constructing canonical forms. SVD should be used when a truncation decision is required, because it gives direct access to Schmidt singular values and discarded weights.

Keep the orthogonality-center structure explicit. In two-site DMRG, the variational object is the two-site tensor Theta with indices alpha at bond i minus one, sigma at site i, sigma at site i plus one, and alpha at bond i plus one, and the tensors to the left and right should be kept in the corresponding mixed-canonical form. During a left-to-right sweep, after splitting the optimized two-site tensor, the center moves to the right; during a right-to-left sweep, it moves to the left. This structure should be reflected explicitly in the code.

Do not build dense local effective Hamiltonians in production sweeps. Except for very small validation tests, the local effective Hamiltonian should be accessed only through a matrix-free map from v to the effective Hamiltonian applied to v. This is the only scalable strategy and is the natural interface to Lanczos or Davidson.

Treat the validated contraction templates as authoritative. For the conventions of these notes, the one-site and two-site local environment-based contractions are subtle. In particular, the stored left environment enters the local action with its MPS-bond indices interchanged, and the MPO physical-index routing in the two-site action must be implemented exactly as in the validated contraction sequence. These convention-dependent details should not be re-derived from memory during coding.

Benchmark against dense projected operators before full sweeps. Before running finite-system DMRG sweeps, one should verify on small systems that the environment-built one-site and two-site local operators agree with the exact projected operators P dagger H P. This is a more stringent and more informative test than Hermiticity alone.

Benchmark against exact diagonalization on small systems. After the local-operator tests pass, run two-site DMRG for small Heisenberg chains and compare the converged energy against exact diagonalization. This is the standard global correctness test.

Use the AKLT chain as an exact MPS benchmark. The AKLT model is particularly valuable because the Hamiltonian has an exact MPO representation of modest bond dimension, the ground state admits an exact MPS representation with bond dimension two, and the open-chain ground-state energy is known exactly as minus two thirds times L minus one. Therefore the AKLT chain tests simultaneously the MPO, the local solver, the sweep logic, and the handling of exact low-bond-dimension MPS ground states.

Use the current local tensor as the eigensolver initial guess. In finite-system sweeps, the previously optimized local tensor provides an excellent initial vector for Lanczos or Davidson. This is a major practical improvement in convergence speed.

Monitor discarded weights and, when available, variances. The energy alone is not always a sufficient convergence indicator. In production calculations, one should monitor at least the energy change between sweeps, the maximum discarded weight during each sweep, the eventual energy variance, and stability of local observables and entanglement spectra.

A robust order of development is: local operators; MPO construction; canonicalization routines; environment updates; one-site projected-operator validation; two-site projected-operator validation; two-site DMRG benchmark for Heisenberg chains; AKLT benchmark; and only afterward one-site DMRG, variance evaluation, and further optimizations.

The most important practical warning is that, for the conventions used in these notes, a local operator built from environments can be Hermitian and numerically stable while still being incorrect. This can happen if the stored left environment is inserted into the local action with the wrong bond-index order, or if the MPO physical-index routing in the local contraction is inconsistent with the chosen MPO storage convention. Therefore full DMRG sweeps should not be trusted until the matrix-free local operator has been compared explicitly against the dense projected operator P dagger H P on small systems.

The final recommendation is this: first validate the local operator, then validate small-system energies, and only then scale up. For environment-based DMRG implementations, this order is not merely convenient; it is essential.
