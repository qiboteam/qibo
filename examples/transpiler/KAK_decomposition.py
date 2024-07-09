import numpy as np


def orth_decomp_of_unitary(X):
  """Decomposes unitary X = Ql @ exp(i Theta) @ Qr' 
  into orthogonal matrices Ql, Qr and angles Theta. 

  """

  # one can also directly decompose XX^T and infer Ql from Qr. This might save about 3 eigh / qrs 
  # I will here try to implement something where I understand the proof, why it works. 
  # The benefit here is that we reliably work with real orthogonal matrices all the time. 
  Xr = X.real
  Xi = X.imag
  _, Ql = np.linalg.eigh(Xr@Xi.transpose())
  _, Qr = np.linalg.eigh(Xr.transpose()@Xi)

  Dr = Ql.transpose() @ Xr @ Qr 
  # if Xi injective this is diagonal, if not there is an arbitrary SO(dim ker(xi)) too much
  # fixing the kernels, I don't know if there is a smarted way to do this! 
  if not np.allclose(Dr, np.diag(np.diag(Dr))):
    Q, R = np.linalg.qr(Dr)
    Dr = np.diag(R).copy()
    Ql = Ql @ Q
  else:
    Dr = np.diag(Dr).copy()


  Di = Ql.transpose() @ Xi @ Qr
  # if Xr injective this is diagonal, if not there is an arbitrary SO(dim ker(Xr)) too much
  if not np.allclose(Di, np.diag(np.diag(Di))):
    Q, R = np.linalg.qr(Di.T)
    Di = np.diag(R).copy()
    Qr = Qr @ Q

  else:
    Di = np.diag(Di).copy()

  # ensure Ql, Qr are in SO
  if np.linalg.det(Ql) < 0: 
    Ql[:,0] = -Ql[:,0]
    Dr[0] = -Dr[0]
    Di[0] = -Di[0]

  if np.linalg.det(Qr) < 0: 
    Qr[:,0] = -Qr[:,0]
    Dr[0] = -Dr[0]
    Di[0] = -Di[0]

  Theta = np.angle(Dr + 1j * Di)

  return Theta, Qr, Ql


def unit_kronecker_rank_approx(X, verbose=True): 
  """Approximates a n^2 x m^2 matrix X  as A otimes B 
  """
  # better check for square dimensions 

  n, m = np.sqrt(X.shape).astype(int)

  Y = X.reshape(n, n, m, m).transpose((0,2,1,3)).reshape(n * m, n * m)
  U, S, Vh = np.linalg.svd(Y)

  if verbose:
    print('Truncating the spectrum', S)

  U = U @ np.sqrt(np.diag(S))
  Vh = np.sqrt(np.diag(S)) @ Vh
  A = U[:,0].reshape(n, m)
  B = Vh[0,:].reshape(n,m)
  return A, B


MAGIC_BASIS = np.array([[1,  0,  0, 1j],
                        [0, 1j,  1,  0],
                        [0, 1j, -1,  0],
                        [1,  0,  0, -1j]]) / np.sqrt(2)

HADAMARD = np.array([[1,  1, -1,  1],
                     [1,  1,  1, -1],
                     [1, -1, -1, -1],
                     [1, -1,  1,  1]]) / 2 


def KAK_decomp(U):
  """ Calculates the KAK decomposition of U in U(4).
      U = np.kron(A0, A1) @ heisenberg_unitary(K) @ np.kron(B0.conj().T, B1.conj().T) 
  """

  Theta, Qr, Ql = orth_decomp_of_unitary(MAGIC_BASIS.conj().T @ U @ MAGIC_BASIS)
  A0, A1 = unit_kronecker_rank_approx(MAGIC_BASIS @ Ql @ MAGIC_BASIS.conj().T)
  B0, B1 = unit_kronecker_rank_approx(MAGIC_BASIS @ Qr @ MAGIC_BASIS.conj().T)
  K = HADAMARD.T @ Theta / 2

  print('Theta', Theta)
  print('K', K)

  # Make A0, A1, B0, B1 special
  def make_special(X):
    return np.sqrt(np.linalg.det(X).conj()) * X

  A0, A1, B0, B1 = map(make_special, (A0, A1, B0, B1))

  return A0, A1, K, B0, B1

#----
# Tests
#----

# some convenient functions and constants
def haar_rand_unitary(dim, real=False):
    """Returns a random unitary dim x dim matrix drawn from the Haar measure.

    Args:
        dim (int): the dimension of the unitary
        real (boolean): If true returns an orthogonal matrix.

    Returns:
        numpy.array: random unitary matrix
    """

    # Draw a random Gaussian
    gaussianMatrix = np.random.randn(dim, dim)
    if not real:  # Add a random imaginary part
        gaussianMatrix = gaussianMatrix + 1j * np.random.randn(dim, dim)

    # project to Haar random unitary
    Q, R = np.linalg.qr(gaussianMatrix)
    D = np.diag(R)
    D = D / np.abs(D)
    R = np.diag(D)
    Q = Q.dot(R)

    return Q


def is_unitary(X, special=False, eps=1E-8): 
    is_special = np.abs(np.linalg.det(X) - 1) < 1E-8 if special else True
    return is_special and np.linalg.norm(X @ X.conj().T - np.eye(X.shape[0])) < eps

PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1, 0], [0, -1]])

PXX = np.kron(PAULI_X, PAULI_X)
PYY = np.kron(PAULI_Y, PAULI_Y)
PZZ = np.kron(PAULI_Z, PAULI_Z)

def heisenberg_unitary(k): 
  from scipy.linalg import expm 
  H = k[0] * np.eye(4) + k[1] * PXX + k[2] * PYY + k[3] * PZZ
  return expm(1j * H)


# test low_kronecker_rank_approx

n, m = 2, 3

A = np.random.randn(n, m)
B = np.random.randn(n, m)

C = np.kron(A, B)

Arec, Brec = unit_kronecker_rank_approx(C)

assert np.linalg.norm(C - np.kron(Arec, Brec)) < 1E-8 


# test orth_decomp_of_unitary
dim = 10

X = haar_rand_unitary(dim)

def test_orth_decomp_of_unitary(X):

  Theta, Qr, Ql = orth_decomp_of_unitary(X)
  assert np.linalg.norm(Ql @ np.diag(np.exp(1j*Theta)) @ Qr.T - X) < 1E-8

test_orth_decomp_of_unitary(X)

# test KAK_decomp(U)

def test_KAK_decomposition(U):
    A0, A1, K, B0, B1 = KAK_decomp(U)

    # Correctly decomposed
    assert np.linalg.norm(np.kron(A0, A1) @ heisenberg_unitary(K) @ np.kron(B0.conj().T, B1.conj().T) - U) < 1E-8

    # A0, A1, B0, B1 are in SU(2)
    assert is_unitary(A0, special=True)
    assert is_unitary(A1, special=True)
    assert is_unitary(B0, special=True)
    assert is_unitary(B1, special=True)

# test KAK_decomp of random unitary

test_KAK_decomposition(haar_rand_unitary(4))

# test on Marek's example
from scipy.linalg import expm

H = .1 * (PXX + PYY + .5 * (np.kron(PAULI_Z, np.eye(2)) + np.kron(np.eye(2), PAULI_Z)))
U = expm(1j * H)
UM = MAGIC_BASIS.conj().T @ U @ MAGIC_BASIS
assert is_unitary(U)
assert is_unitary(UM)

test_orth_decomp_of_unitary(UM)

test_KAK_decomposition(U)

H2 = .1 * (PXX + PZZ+ PYY+(np.kron(PAULI_Z, np.eye(2)) + np.kron(np.eye(2), PAULI_Z)))
U2 = expm(1j * H2)
UM2 = MAGIC_BASIS.conj().T @ U2 @ MAGIC_BASIS
test_orth_decomp_of_unitary(UM2)

test_KAK_decomposition(expm(1j * H2))

# test Heisenberg type Hamiltonian

K = [.1, .3, .6, .2]
test_KAK_decomposition(heisenberg_unitary(K))




