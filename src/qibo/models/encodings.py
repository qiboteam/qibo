"""Module with functions that encode classical data into quantum circuits."""

import math

import numpy as np

from qibo import gates
from qibo.backends import GlobalBackend
from qibo.config import raise_error
from qibo.models.circuit import Circuit


def unary_encoder(data):
    """Creates circuit that performs the unary encoding of ``data``.

    Given a classical ``data`` array :math:`\\mathbf{x} \\in \\mathbb{R}^{d}` such that

    .. math::
        \\mathbf{x} = (x_{1}, x_{2}, \\dots, x_{d}) \\, ,

    this function generate the circuit that prepares the quantum state

    .. math::
        \\frac{1}{\\|\\mathbf{x}\\|} \\, \\sum_{k=1}^{d} \\, x_{k} \\, \\ket{k} \\, .

    Here, :math:`\\ket{k}` is a unary representation of the number :math:`1` through
    :math:`d`.

    Args:
        data (ndarray): array of data to be loaded.

    Returns:
        :class:`qibo.models.circuit.Circuit`: circuit that loads ``data`` in unary representation.

    References:
        1. S. Johri *et al.*, *Nearest Centroid Classiﬁcation on a Trapped Ion Quantum Computer*.
        `arXiv:2012.04145v2 [quant-ph] <https://arxiv.org/abs/2012.04145>`_.
    """
    if len(data.shape) != 1:
        raise_error(
            TypeError,
            f"``data`` must be a 1-dimensional array, but it has dimensions {data.shape}.",
        )

    nqubits = len(data)
    j_max = int(nqubits / 2)

    # generating list of indexes representing the RBS connections
    pairs_rbs = [[(0, int(nqubits / 2))]]
    indexes = list(np.array(pairs_rbs).flatten())
    for depth in range(2, int(math.log2(nqubits)) + 1):
        pairs_rbs_per_depth = [
            [(index, index + int(nqubits / 2**depth)) for index in indexes]
        ]
        pairs_rbs += pairs_rbs_per_depth
        indexes = list(np.array(pairs_rbs_per_depth).flatten())

    # creating circuit with all RBS initialised with 0.0 phase
    circuit = Circuit(nqubits)
    circuit.add(gates.X(0))
    for row in pairs_rbs:
        for pair in row:
            circuit.add(gates.RBS(*pair, 0.0, trainable=True))

    # calculating phases and setting circuit parameters
    r_array = np.zeros(nqubits - 1, dtype=float)
    phases = np.zeros(nqubits - 1, dtype=float)
    for j in range(1, j_max + 1):
        r_array[j_max + j - 2] = math.sqrt(data[2 * j - 1] ** 2 + data[2 * j - 2] ** 2)
        theta = math.acos(data[2 * j - 2] / r_array[j_max + j - 2])
        if data[2 * j - 1] < 0.0:
            theta = 2 * math.pi - theta
        phases[j_max + j - 2] = theta

    for j in range(j_max - 1, 0, -1):
        r_array[j - 1] = math.sqrt(r_array[2 * j] ** 2 + r_array[2 * j - 1] ** 2)
        phases[j - 1] = math.acos(r_array[2 * j - 1] / r_array[j - 1])

    circuit.set_parameters(phases)

    return circuit
