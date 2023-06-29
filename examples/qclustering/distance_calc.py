import numpy as np
import math
import tensorflow as tf
from qibo.models import Circuit
from qibo import gates
import util as u

def pad_input(X):
    """Adds 0s if X log2(X.dim) != round int.

    Parameters
    ----------
    X : :class:`numpy.ndarray`
        Input data

    Returns
    -------
    :class:`numpy.ndarray`
        Padded X
    """
    num_features = len(X)
    if not float(np.log2(num_features)).is_integer():
        size_needed = pow(2, math.ceil(math.log(num_features) / math.log(2)))
        X = np.pad(X, (0, size_needed - num_features), "constant")
    return X


def DistCalc(a, b, shots_n=10000):
    """Distance calculation using destructive interference.

    Parameters
    ----------
    a : :class:`numpy.ndarray`
        First point - shape = (latent space dimension,)
    b : :class:`numpy.ndarray`
        First point - shape = (latent space dimension,)
    device_name : str
        Name of device for executing a simulation of quantum circuit.
    shots_n : int
        Number of shots for executing a quantum circuit - to get frequencies.

    Returns
    -------
    (float, :class:`qibo.models.Circuit`)
        (distance, quantum circuit)

    """
    num_features = len(a)
    norm = u.calc_norm(a, b)
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
    
    result = qc.execute(initial_state=amplitudes, nshots=shots_n)
    
    counts = result.frequencies(binary=True)
    distance = norm * math.sqrt(2) * math.sqrt((counts["1"] / shots_n))
    
    return distance, qc
