# -*- coding: utf-8 -*-
import numpy as np
from qibo import matrices, K
from qibo.config import BACKEND_NAME, DTYPES
if BACKEND_NAME == "tensorflow":
    from qibo.tensorflow.hamiltonians import TensorflowHamiltonian as Hamiltonian
else: # pragma: no cover
    raise NotImplementedError("Only Tensorflow backend is implemented.")


def _multikron(matrix_list):
    """Calculates Kronecker product of a list of matrices.

    Args:
        matrices (list): List of matrices as ``np.ndarray``s.

    Returns:
        ``np.ndarray`` of the Kronecker product of all ``matrices``.
    """
    h = 1
    for m in matrix_list:
        h = np.kron(h, m)
    return h


def _build_spin_model(nqubits, matrix, condition):
    """Helper method for building nearest-neighbor spin model Hamiltonians."""
    h = sum(_multikron((matrix if condition(i, j) else matrices.I
                        for j in range(nqubits)))
            for i in range(nqubits))
    return h


def XXZ(nqubits, delta=0.5):
    """Heisenberg XXZ model with periodic boundary conditions.

    .. math::
        H = \\sum _{i=0}^N \\left ( X_iX_{i + 1} + Y_iY_{i + 1} + \\delta Z_iZ_{i + 1} \\right ).

    Args:
        nqubits (int): number of quantum bits.
        delta (float): coefficient for the Z component (default 0.5).

    Example:
        ::

            from qibo.hamiltonians import XXZ
            h = XXZ(3) # initialized XXZ model with 3 qubits
    """
    condition = lambda i, j: i in {j % nqubits, (j+1) % nqubits}
    hx = _build_spin_model(nqubits, matrices.X, condition)
    hy = _build_spin_model(nqubits, matrices.Y, condition)
    hz = _build_spin_model(nqubits, matrices.Z, condition)
    matrix = hx + hy + delta * hz
    return Hamiltonian(nqubits, matrix)


def _OneBodyPauli(nqubits, matrix):
    """Helper method for constracting non-interacting X, Y, Z Hamiltonians."""
    condition = lambda i, j: i == j % nqubits
    ham = _build_spin_model(nqubits, matrix, condition)
    ham = K.cast(-ham, dtype=DTYPES.get('DTYPECPX'))
    return Hamiltonian(nqubits, ham)


def X(nqubits):
    """Non-interacting pauli-X Hamiltonian.

    .. math::
        H = - \\sum _{i=0}^N X_i.

    Args:
        nqubits (int): number of quantum bits.
    """
    return _OneBodyPauli(nqubits, matrices.X)


def Y(nqubits):
    """Non-interacting pauli-X Hamiltonian.

    .. math::
        H = - \\sum _{i=0}^N Y_i.

    Args:
        nqubits (int): number of quantum bits.
    """
    return _OneBodyPauli(nqubits, matrices.Y)


def Z(nqubits):
    """Non-interacting pauli-X Hamiltonian.

    .. math::
        H = - \\sum _{i=0}^N Z_i.

    Args:
        nqubits (int): number of quantum bits.
    """
    return _OneBodyPauli(nqubits, matrices.Z)


def TFIM(nqubits, h=0.0):
    """Transverse field Ising model with periodic boundary conditions.

    .. math::
        H = - \\sum _{i=0}^N \\left ( Z_i Z_{i + 1} + h X_i \\right ).

    Args:
        nqubits (int): number of quantum bits.
        h (float): value of the transverse field.
    """
    condition = lambda i, j: i in {j % nqubits, (j+1) % nqubits}
    ham = _build_spin_model(nqubits, matrices.Z, condition)
    if h != 0:
        condition = lambda i, j: i == j % nqubits
        ham += _build_spin_model(nqubits, matrices.X, condition)
    ham = K.cast(-ham, dtype=DTYPES.get('DTYPECPX'))
    return Hamiltonian(nqubits, ham)
