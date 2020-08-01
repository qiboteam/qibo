# -*- coding: utf-8 -*-
import numpy as np
from qibo import matrices, K
from qibo.config import DTYPES
from abc import ABCMeta


NUMERIC_TYPES = (np.int, np.float, np.complex,
                 np.int32, np.int64, np.float32,
                 np.float64, np.complex64, np.complex128)


class Hamiltonian(object):
    """This class implements the abstract Hamiltonian operator.

    Args:
        nqubits (int): number of quantum bits.
    """
    __metaclass__ = ABCMeta

    def __init__(self, nqubits, hamiltonian):
        if not isinstance(nqubits, int):
            raise RuntimeError(f'nqubits must be an integer')
        self.nqubits = nqubits
        self.hamiltonian = hamiltonian
        self._eigenvalues = None
        self._eigenvectors = None

    def eigenvalues(self):
        """Computes the eigenvalues for the Hamiltonian."""
        if self._eigenvalues is None:
            self._eigenvalues = K.linalg.eigvalsh(self.hamiltonian)
        return self._eigenvalues

    def eigenvectors(self):
        """Computes the eigenvectors for the Hamiltonian."""
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = K.linalg.eigh(
                self.hamiltonian)
        return self._eigenvectors

    def expectation(self, state):
        """Computes the real expectation value for a given state.
        Args:
            state (array): the expectation state.
        """
        a = K.math.conj(state)
        b = K.tensordot(self.hamiltonian, state, axes=1)
        n = K.math.real(K.reduce_sum(a*b))
        return n

    def __add__(self, o):
        """Add operator."""
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise RuntimeError('Only hamiltonians with the same '
                                   'number of qubits can be added.')
            new_hamiltonian = self.hamiltonian + o.hamiltonian
            return self.__class__(self.nqubits, new_hamiltonian)
        elif isinstance(o, NUMERIC_TYPES):
            new_hamiltonian = self.hamiltonian + o * \
                K.eye(2 ** self.nqubits, dtype=self.hamiltonian.dtype)
            return self.__class__(self.nqubits, new_hamiltonian)
        else:
            raise NotImplementedError(f'Hamiltonian addition to {type(o)} '
                                      'not implemented.')

    def __radd__(self, o): # pragma: no cover
        """Right operator addition."""
        return self.__add__(o)

    def __sub__(self, o):
        """Subtraction operator."""
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise RuntimeError('Only hamiltonians with the same '
                                   'number of qubits can be added.')
            new_hamiltonian = self.hamiltonian - o.hamiltonian
            return self.__class__(self.nqubits, new_hamiltonian)
        elif isinstance(o, NUMERIC_TYPES):
            new_hamiltonian = self.hamiltonian - o * \
                K.eye(2 ** self.nqubits, dtype=self.hamiltonian.dtype)
            return self.__class__(self.nqubits, new_hamiltonian)
        else:
            raise NotImplementedError(f'Hamiltonian subtraction to {type(o)} '
                                      'not implemented.')

    def __rsub__(self, o):
        """Right subtraction operator."""
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise RuntimeError('Only hamiltonians with the same '
                                   'number of qubits can be added.')
            new_hamiltonian = o.hamiltonian - self.hamiltonian
            return self.__class__(self.nqubits, new_hamiltonian)
        elif isinstance(o, NUMERIC_TYPES):
            new_hamiltonian = o * \
                K.eye(2 ** self.nqubits, dtype=self.hamiltonian.dtype) - \
                self.hamiltonian
            return self.__class__(self.nqubits, new_hamiltonian)
        else:
            raise NotImplementedError(f'Hamiltonian subtraction to {type(o)} '
                                      'not implemented.')

    def __mul__(self, o):
        """Multiplication to scalar operator."""
        if isinstance(o, NUMERIC_TYPES):
            new_hamiltonian = self.hamiltonian * o
            r = self.__class__(self.nqubits, new_hamiltonian)
            if self._eigenvalues is not None:
                if o.real >= 0:
                    r._eigenvalues = o * self._eigenvalues
                else:
                    r._eigenvalues = o * self._eigenvalues[::-1]
            if self._eigenvectors is not None:
                if o.real > 0:
                    r._eigenvectors = self._eigenvectors
                elif o.real == 0:
                    r._eigenvectors = K.eye(
                        self._eigenvectors.shape[0], dtype=self.hamiltonian.dtype)
            return r
        else:
            raise NotImplementedError(f'Hamiltonian multiplication to {type(o)} '
                                      'not implemented.')

    def __rmul__(self, o):
        """Right scalar multiplication."""
        return self.__mul__(o)


def _multikron(matrices):
    """Calculates Kronecker product of a list of matrices.

    Args:
        matrices (list): List of matrices as ``np.ndarray``s.

    Returns:
        ``np.ndarray`` of the Kronecker product of all ``matrices``.
    """
    h = 1
    for m in matrices:
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
    condition = lambda i, j: i == j % nqubits or i == (j+1) % nqubits
    hx = _build_spin_model(nqubits, matrices.X, condition)
    hy = _build_spin_model(nqubits, matrices.Y, condition)
    hz = _build_spin_model(nqubits, matrices.Z, condition)
    hamiltonian = hx + hy + delta * hz
    return Hamiltonian(nqubits, hamiltonian)


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
    condition = lambda i, j: i == j % nqubits or i == (j+1) % nqubits
    ham = _build_spin_model(nqubits, matrices.Z, condition)
    if h != 0:
        condition = lambda i, j: i == j % nqubits
        ham += _build_spin_model(nqubits, matrices.X, condition)
    ham = K.cast(-ham, dtype=DTYPES.get('DTYPECPX'))
    return Hamiltonian(nqubits, ham)
