import os
import tempfile

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


def match_figure_image(fig, arr_path: str):
    """Check whether the two image arrays match.

    Args:
        fig (:class:`matplotlib.figure.Figure`): Figure to compare.
        arr_path (str): Path to the ``numpy`` array file containing the reference image.

    Returns:
        bool: ``True`` if the images match, ``False`` otherwise.
    """
    return np.all(fig2array(fig) == np.load(arr_path))


def fig2png(figure):
    """Save a matplotlib figure to a temporary PNG file.
    Args:
        figure (:class:`matplotlib.figure.Figure`): The figure to save.
    Returns:
        str: The path to the temporary PNG file if successful, otherwise None.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        # Save the figure as a PNG file in the temporary file
        temp_file_path = temp_file.name
        figure.savefig(temp_file_path)
        if os.path.exists(temp_file_path):
            return temp_file_path
        return None
