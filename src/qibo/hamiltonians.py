# -*- coding: utf-8 -*-
import numpy as np
from qibo import matrices, K
from qibo.config import DTYPES
from abc import ABCMeta, abstractmethod


NUMERIC_TYPES = (np.int, np.float, np.complex,
                 np.int32, np.int64, np.float32,
                 np.float64, np.complex64, np.complex128)


def isclassinstance(o, w):
    """Check if objects are from the same base class."""
    return (isinstance(o, w.__class__) or
            issubclass(o.__class__, w.__class__) or
            issubclass(w.__class__, o.__class__))


class Hamiltonian(object):
    """This class implements the abstract Hamiltonian operator.

    Args:
        nqubits (int): number of quantum bits.
    """
    __metaclass__ = ABCMeta

    def __init__(self, nqubits):
        if not isinstance(nqubits, int):
            raise RuntimeError(f'nqubits must be an integer')
        self.hamiltonian = None
        self.nqubits = nqubits
        self._eigenvalues = None
        self._eigenvectors = None

    @abstractmethod
    def _build(self, *args, **kwargs):
        """Implements the Hamiltonian construction."""
        pass

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
        if isclassinstance(o, self):
            if self.nqubits != o.nqubits:
                raise RuntimeError('Only hamiltonians with the same '
                                   'number of qubits can be added.')
            r = self.__class__(nqubits=self.nqubits)
            r.hamiltonian = self.hamiltonian + o.hamiltonian
            return r
        elif isinstance(o, NUMERIC_TYPES):
            r = self.__class__(nqubits=self.nqubits)
            r.hamiltonian = self.hamiltonian + o * \
                K.eye(2 ** self.nqubits, dtype=self.hamiltonian.dtype)
            return r
        else:
            raise NotImplementedError(f'Hamiltonian addition to {type(o)} '
                                      'not implemented.')

    def __radd__(self, o):
        """Right operator addition."""
        return self.__add__(o)

    def __sub__(self, o):
        """Subtraction operator."""
        if isclassinstance(o, self):
            if self.nqubits != o.nqubits:
                raise RuntimeError('Only hamiltonians with the same '
                                   'number of qubits can be added.')
            r = self.__class__(nqubits=self.nqubits)
            r.hamiltonian = self.hamiltonian - o.hamiltonian
            return r
        elif isinstance(o, NUMERIC_TYPES):
            r = self.__class__(nqubits=self.nqubits)
            r.hamiltonian = self.hamiltonian - o * \
                K.eye(2 ** self.nqubits, dtype=self.hamiltonian.dtype)
            return r
        else:
            raise NotImplementedError(f'Hamiltonian subtraction to {type(o)} '
                                      'not implemented.')

    def __rsub__(self, o):
        """Right subtraction operator."""
        if isclassinstance(o, self):
            if self.nqubits != o.nqubits:
                raise RuntimeError('Only hamiltonians with the same '
                                   'number of qubits can be added.')
            r = self.__class__(nqubits=self.nqubits)
            r.hamiltonian = o.hamiltonian - self.hamiltonian
            return r
        elif isinstance(o, NUMERIC_TYPES):
            r = self.__class__(nqubits=self.nqubits)
            r.hamiltonian = o * \
                K.eye(2 ** self.nqubits, dtype=self.hamiltonian.dtype) - \
                self.hamiltonian
            return r
        else:
            raise NotImplementedError(f'Hamiltonian subtraction to {type(o)} '
                                      'not implemented.')

    def __mul__(self, o):
        """Multiplication to scalar operator."""
        if isinstance(o, NUMERIC_TYPES):
            r = self.__class__(nqubits=self.nqubits)
            r.hamiltonian = self.hamiltonian * o
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


class SpinModel(Hamiltonian):

    @staticmethod
    def multikron(matrices):
        h = 1
        for m in matrices:
            h = np.kron(h, m)
        return h

    def _build(self, matrix, condition):
        h = sum(self.multikron((matrix if condition(i, j) else matrices.I
                                for j in range(self.nqubits)))
                for i in range(self.nqubits))
        return h


class Ising(SpinModel):

    def __init__(self, nqubits, h=0.0):
        super(Ising, self).__init__(nqubits)
        condition = lambda i, j: i == j % nqubits or i == (j+1) % nqubits
        hzz = self._build(matrices.Z, condition)
        condition = lambda i, j: i == j % nqubits
        hx = self._build(matrices.X, condition)
        self.hamiltonian = K.cast(-(hzz + h * hx), dtype=DTYPES.get('DTYPECPX'))


class OneBodyPauli(SpinModel):

    def __init__(self, nqubits, p=(1.0, 0.0, 0.0)):
        super(OneBodyPauli, self).__init__(nqubits)
        condition = lambda i, j: i == j % nqubits
        ms = (matrices.X, matrices.Y, matrices.Z)
        h = sum(pi * self._build(m, condition) for pi, m in zip(p, ms))
        self.hamiltonian = K.cast(-h, dtype=DTYPES.get('DTYPECPX'))


class XXZ(SpinModel):
    """Implements the Heisenberg XXZ model.

    Uses Pauli matrices to build the final Hamiltonian matrix as:

    .. math::
        H = H_x + H_y + \\delta \cdot H_z.

    Args:
        nqubits (int): number of quantum bits.
        delta (float): coefficient for the Z component (default 0.5).

    Example:
        ::

            from qibo.hamiltonian import XXZ
            h = XXZ(3) # initialized XXZ model with 3 qubits
    """

    def __init__(self, nqubits, delta=0.5, **kwargs):
        """Initialize XXZ model."""
        super(XXZ, self).__init__(nqubits)
        condition = lambda i, j: i == j % nqubits or i == (j+1) % nqubits
        hx = self._build(matrices.X, condition)
        hy = self._build(matrices.Y, condition)
        hz = self._build(matrices.Z, condition)
        self.hamiltonian = hx + hy + delta * hz
