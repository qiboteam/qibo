import numpy as np
from scipy import sparse


def random_complex(shape, dtype=None):
    x = np.random.random(shape) + 1j * np.random.random(shape)
    if dtype is None:
        return x
    return x.astype(dtype)


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
