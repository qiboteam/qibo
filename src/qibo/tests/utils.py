import numpy as np
from scipy import sparse


def random_complex(shape, dtype=None):
    x = np.random.random(shape) + 1j * np.random.random(shape)
    if dtype is None:
        return x
    return x.astype(dtype)


def random_hermitian(nqubits):
    shape = 2 * (2 ** nqubits,)
    m = random_complex(shape)
    return m + m.T.conj()


def random_state(nqubits):
    """Generates a random normalized state vector as numpy array."""
    initial_state = random_complex(2 ** nqubits)
    return initial_state / np.sqrt((np.abs(initial_state) ** 2).sum())


def random_density_matrix(nqubits):
    """Generates a random normalized density matrix."""
    rho = random_hermitian(nqubits)
    # Normalize
    ids = np.arange(2 ** nqubits)
    rho[ids, ids] = rho[ids, ids] / np.trace(rho)
    return rho


def random_sparse_matrix(backend, n, sparse_type=None):
    if backend.name == "tensorflow":
        nonzero = int(0.1 * n * n)
        indices = np.random.randint(0, n, size=(nonzero, 2))
        data = np.random.random(nonzero) + 1j * np.random.random(nonzero)
        data = backend.cast(data)
        return backend.tf.sparse.SparseTensor(indices, data, (n, n))
    else:
        re = sparse.rand(n, n, format=sparse_type)
        im = sparse.rand(n, n, format=sparse_type)
        return re + 1j * im
