# -*- coding: utf-8 -*-
import numpy as np
from qibo import matrices, K
from qibo.config import DTYPES, raise_error
from abc import ABCMeta


NUMERIC_TYPES = (np.int, np.float, np.complex,
                 np.int32, np.int64, np.float32,
                 np.float64, np.complex64, np.complex128)


class Hamiltonian(object):
    """This class implements the abstract Hamiltonian operator.

    Args:
        nqubits (int): number of quantum bits.
        matrix (np.ndarray): Matrix representation of the Hamiltonian in the
            computational basis as an array of shape
            ``(2 ** nqubits, 2 ** nqubits)``.
    """
    __metaclass__ = ABCMeta

    def __init__(self, nqubits, matrix):
        if not isinstance(nqubits, int):
            raise_error(RuntimeError, f'nqubits must be an integer')
        shape = tuple(matrix.shape)
        if shape != 2 * (2 ** nqubits,):
            raise_error(ValueError, f"The Hamiltonian is defined for {nqubits} qubits "
                                     "while the given matrix has shape {shape}.")

        self.nqubits = nqubits
        self.matrix = matrix
        self._eigenvalues = None
        self._eigenvectors = None

    def eigenvalues(self):
        """Computes the eigenvalues for the Hamiltonian."""
        if self._eigenvalues is None:
            self._eigenvalues = K.linalg.eigvalsh(self.matrix)
        return self._eigenvalues

    def eigenvectors(self):
        """Computes a tensor with the eigenvectors for the Hamiltonian."""
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = K.linalg.eigh(
                self.matrix)
        return self._eigenvectors

    def exp(self, a):
        """Computes a tensor corresponding to exp(a * H).

        Args:
            a (complex): Complex number to multiply Hamiltonian before
                exponentiation.
        """
        if self._eigenvectors is None:
            return K.linalg.expm(a * self.matrix)
        expd = K.linalg.diag(K.exp(a * self._eigenvalues))
        ud = K.transpose(K.math.conj(self._eigenvectors))
        return K.matmul(self._eigenvectors, K.matmul(expd, ud))

    def expectation(self, state, normalize=False):
        """Computes the real expectation value for a given state.

        Args:
            state (array): the expectation state.
            normalize (bool): If ``True`` the expectation value is divided
                with the state's norm squared.

        Returns:
            Real number corresponding to the expectation value.
        """
        statec = K.math.conj(state)
        hstate = self @ state
        ev = K.math.real(K.reduce_sum(statec * hstate))
        if normalize:
            norm = K.reduce_sum(K.square(K.abs(state)))
            return ev / norm
        return ev

    def __add__(self, o):
        """Add operator."""
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise_error(RuntimeError, 'Only hamiltonians with the same '
                                          'number of qubits can be added.')
            new_matrix = self.matrix + o.matrix
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, NUMERIC_TYPES):
            new_matrix = (self.matrix +
                          o * K.eye(2 ** self.nqubits, dtype=self.matrix.dtype))
            return self.__class__(self.nqubits, new_matrix)
        else:
            raise_error(NotImplementedError, f'Hamiltonian addition to {type(o)} '
                                              'not implemented.')

    def __radd__(self, o):
        """Right operator addition."""
        return self.__add__(o)

    def __sub__(self, o):
        """Subtraction operator."""
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise_error(RuntimeError, 'Only hamiltonians with the same '
                                          'number of qubits can be added.')
            new_matrix = self.matrix - o.matrix
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, NUMERIC_TYPES):
            new_matrix = (self.matrix -
                          o * K.eye(2 ** self.nqubits, dtype=self.matrix.dtype))
            return self.__class__(self.nqubits, new_matrix)
        else:
            raise_error(NotImplementedError, f'Hamiltonian subtraction to {type(o)} '
                                              'not implemented.')

    def __rsub__(self, o):
        """Right subtraction operator."""
        if isinstance(o, self.__class__): # pragma: no cover
            # impractical case because it will be handled by `__sub__`
            if self.nqubits != o.nqubits:
                raise_error(RuntimeError, 'Only hamiltonians with the same '
                                          'number of qubits can be added.')
            new_matrix = o.matrix - self.matrix
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, NUMERIC_TYPES):
            new_matrix = (o * K.eye(2 ** self.nqubits, dtype=self.matrix.dtype)
                          - self.matrix)
            return self.__class__(self.nqubits, new_matrix)
        else:
            raise_error(NotImplementedError, f'Hamiltonian subtraction to {type(o)} '
                                              'not implemented.')

    def __mul__(self, o):
        """Multiplication to scalar operator."""
        if isinstance(o, NUMERIC_TYPES):
            new_matrix = self.matrix * o
            r = self.__class__(self.nqubits, new_matrix)
            if self._eigenvalues is not None:
                if o.real >= 0:
                    r._eigenvalues = o * self._eigenvalues
                else:
                    r._eigenvalues = o * self._eigenvalues[::-1]
            if self._eigenvectors is not None:
                if o.real > 0:
                    r._eigenvectors = self._eigenvectors
                elif o == 0:
                    r._eigenvectors = K.eye(
                        self._eigenvectors.shape[0], dtype=self.matrix.dtype)
            return r
        elif isinstance(o, (np.ndarray, K.Tensor)):
            new_matrix = self.matrix * K.cast(o, dtype=self.matrix.dtype)
            return self.__class__(self.nqubits, new_matrix)
        else:
            raise_error(NotImplementedError, f'Hamiltonian multiplication to {type(o)} '
                                              'not implemented.')

    def __rmul__(self, o):
        """Right scalar multiplication."""
        return self.__mul__(o)

    def __matmul__(self, o):
        """Matrix multiplication with other Hamiltonians or state vectors."""
        if isinstance(o, self.__class__):
            new_matrix = K.matmul(self.matrix, o.matrix)
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, (K.Tensor, np.ndarray)):
            rank = len(tuple(o.shape))
            if rank == 1: # vector
                return K.matmul(self.matrix, o[:, K.newaxis])[:, 0]
            elif rank == 2: # matrix
                return K.matmul(self.matrix, o)
            else:
                raise_error(ValueError, f'Cannot multiply Hamiltonian with '
                                         'rank-{rank} tensor.')
        else:
            raise_error(NotImplementedError, f'Hamiltonian matrix multiplication to '
                                              '{type(o)} not implemented.')


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
