import math

import numpy as np

from qibo import gates
from qibo.models import Circuit


def pad_input(X):
    """Add 0s if X log2(X.dim) != round int.

    Args:
        X (:class:`numpy.ndarray`): Input data.

    Returns:
        :class:`numpy.ndarray`: Padded X.
    """
    num_features = len(X)
    if not float(np.log2(num_features)).is_integer():
        size_needed = pow(2, math.ceil(math.log(num_features) / math.log(2)))
        X = np.pad(X, (0, size_needed - num_features), "constant")
    return X


def DistCalc(a, b, nshots=10000):
    """Distance calculation using destructive interference.

    Args:
        a (:class:`numpy.ndarray`): first point - shape = (latent space dimension,)
        b (:class:`numpy.ndarray`): second point - shape = (latent space dimension,)
        nshots (int, optional): number of shots for executing a quantum circuit to
            get frequencies. Defaults to :math:`10^{4}`.

    Returns:
        Tuple(float, :class:`qibo.models.Circuit`): (distance, quantum circuit).

    """
    norm = np.linalg.norm(a - b)
    a_norm = a / norm
    b_norm = b / norm

    a_norm = pad_input(a_norm)
    b_norm = pad_input(b_norm)

    amplitudes = np.concatenate((a_norm, b_norm))
    n_qubits = int(np.log2(len(amplitudes)))

    # QIBO
    qc = Circuit(n_qubits)
    qc.add(gates.H(0))
    qc.add(gates.M(0))

    result = qc.execute(initial_state=amplitudes, nshots=nshots)

    counts = result.frequencies(binary=True)
    distance = norm * math.sqrt(2) * math.sqrt(counts["1"] / nshots)

    return distance, qc
