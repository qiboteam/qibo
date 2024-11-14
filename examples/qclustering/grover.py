import math

import numpy as np

from qibo import gates
from qibo.models import Circuit


def iam_operator(n):
    """Construct quantum circuit for 'inversion around mean' operator.

    Args:
        n (int): number of qubits.

    Returns:
        :class:`qibo.models.Circuit`: quantum circuit.
    """
    qc = Circuit(n)

    for qubit in range(n):
        qc.add(gates.H(qubit))  # apply H-gate
        qc.add(gates.X(qubit))  # apply X-gate

    # apply H-gate to last qubit
    qc.add(gates.H(n - 1))
    # apply multi-controlled toffoli gate to last qubit controlled by the ones before it (less significant)
    qc.add(gates.X(n - 1).controlled_by(*list(range(0, n - 1))))
    # apply H-gate to last qubit
    qc.add(gates.H(n - 1))

    for qubit in range(n):
        qc.add(gates.X(qubit))
        qc.add(gates.H(qubit))

    return qc


def grover_qc(qc, n, oracle, n_indices_flip):
    """Construct quantum circuit for Grover's algorithm.

    Args:
        qc (:class:`qibo.models.Circuit`): initial quantum circuit to build up on.
        n (int): number of qubits.
        oracle (:class:`qibo.models.Circuit`): quantum circuit representing an Oracle operator.
        n_indices_flip (int): number of indices which have been selected by Oracle operator.

    Returns:
        :class:`qibo.models.Circuit`: quantum circuit.

    """
    if n_indices_flip:
        r_i = int(math.floor(np.pi / 4 * np.sqrt(2**n / n_indices_flip)))
    else:
        r_i = 0

    for _ in range(r_i):
        qc.add(oracle.on_qubits(*(list(range(n)))))  # apply oracle
        qc_diff = iam_operator(n)
        qc.add(qc_diff.on_qubits(*(list(range(n)))))  # apply inversion around mean

    qc.add(gates.M(*list(range(n))))

    return qc
