import numpy as np
from scipy import sparse


def random_sparse_matrix(backend, n, sparse_type=None):
    if backend.platform == "tensorflow":
        nonzero = int(0.1 * n * n)
        indices = np.random.randint(0, n, size=(nonzero, 2))
        data = np.random.random(nonzero) + 1j * np.random.random(nonzero)
        data = backend.cast(data)

        return backend.tf.sparse.SparseTensor(indices, data, (n, n))

    re = sparse.rand(n, n, format=sparse_type)
    im = sparse.rand(n, n, format=sparse_type)
    return re + 1j * im
