import numpy as np
from scipy import sparse


def random_sparse_matrix(backend, n, sparse_type=None):
    if backend.platform == "tensorflow":
        nonzero = int(0.1 * n * n)
        indices = np.random.randint(0, n, size=(nonzero, 2))
        data = np.random.random(nonzero) + 1j * np.random.random(nonzero)
        data = backend.cast(data)

        return backend.tf.sparse.SparseTensor(indices, data, (n, n))

    n_tries = 0
    while n_tries < 100:
        re = sparse.rand(n, n, format=sparse_type, dtype=complex)
        im = sparse.rand(n, n, format=sparse_type, dtype=complex)
        try:
            matrix = re + 1j * im
            break
        except ValueError:
            n_tries += 1
            continue
    if n_tries == 100:
        raise RuntimeError("Maximum number of tries reached for random sparse matrix.")
    return matrix


def fig2array(fig):
    """Convert matplotlib image into numpy array."""
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    return data


def match_figure_image(fig, arr_path):
    """Check whether the two image arrays match."""
    return np.all(fig2array(fig) == np.load(arr_path))
