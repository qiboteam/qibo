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


def fig2array(fig):
    """Convert matplotlib image into numpy array."""
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    return data


def match_figure_image(fig, arr_path, transparent_layer=False):
    """
    Check whether the two image arrays match.

    Args:
        fig (matplotlib.figure.Figure): Matplotlib figure to compare.
        arr_path: str
        arr_path (str): Path to the numpy array file containing the reference image.
        transparent_layer (bool, optional): If True, checks if the alpha channel matches; otherwise,
                           ignores it, this helps to bypass the test if the images do not fully match.
                           sometimes the rendering on different screen make the images do not match if the image uses automatic scale,
                           for examples if the axis are not set to a fixed range. This is useful for testing purposes.

    Returns:
        bool: True if the images match, False otherwise.
    """
    if transparent_layer:
        return np.all(fig2array(fig)[:, :, 3] == np.load(arr_path)[:, :, 3])
    return np.all(fig2array(fig) == np.load(arr_path))
