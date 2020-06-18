# -*- coding: utf-8 -*-
import numpy as np
from qibo.config import matrices, K
from abc import ABCMeta, abstractmethod


class Hamiltonian(object):
    """This class implements the abstract Hamiltonian operator.

    Args:
        nqubits (int): number of quantum bits.
    """
    __metaclass__ = ABCMeta

    def __init__(self, nqubits):
        self.hamiltonian = None
        self.nqubits = nqubits
        self._eigenvalues = None

    @abstractmethod
    def _build(self, *args, **kwargs):
        """Implements the Hamiltonian construction."""
        pass

    def eigenvalues(self):
        """Computes the minimum eigenvalue for the Hamiltonian."""
        if self._eigenvalues is None:
            self._eigenvalues = K.linalg.eigvalsh(self.hamiltonian)
        return self._eigenvalues

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
            r = self.__class__(nqubits=self.nqubits)
            r.hamiltonian = self.hamiltonian + o.hamiltonian
            return r
        else:
            raise NotImplementedError(f'Hamiltonian addition to {type(o)} '
                                       'not implemented.')

    def __radd__(self, o):
        """Right scalar addition."""
        return self.__add__(o)

    def __sub__(self, o):
        """Subtraction operator."""
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise RuntimeError('Only hamiltonians with the same '
                                   'number of qubits can be added.')
            r = self.__class__(nqubits=self.nqubits)
            r.hamiltonian = self.hamiltonian - o.hamiltonian
            return r
        else:
            raise NotImplementedError(f'Hamiltonian subtraction to {type(o)} '
                                       'not implemented.')

    def __rsub__(self, o):
        """Right scalar addition."""
        return self.__sub__(o)

    def __mul__(self, o):
        """Multiplication to scalar operator."""
        if isinstance(o, (np.float, np.int, np.complex)):
            r = self.__class__(nqubits=self.nqubits)
            r.hamiltonian = self.hamiltonian * o
            return r
        else:
            raise NotImplementedError(f'Hamiltonian multiplication to {type(o)} '
                                       'not implemented.')

    def __rmul__(self, o):
        """Right scalar multiplication."""
        return self.__mul__(o)


class XXZ(Hamiltonian):
    """This class implements the Heisenberg XXZ model.
    The mode uses the Pauli matrices and build the final
    Hamiltonian:

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
    def __init__(self, delta=0.5, **kwargs):
        """Initialize XXZ model."""
        Hamiltonian.__init__(self, **kwargs)
        hx = self._build(matrices._npX())
        hy = self._build(matrices._npY())
        hz = self._build(matrices._npZ())
        self.hamiltonian = hx + hy + delta * hz

    def _build(self, *args, **kwargs):
        """Builds the Heisenber model for a given operator sigma"""
        hamiltonian = 0
        eye = matrices._npI()
        n = self.nqubits
        for i in range(n):
            h = 1
            for j in range(n):
                if i == j % n or i == (j+1) % n:
                    h = np.kron(args[0], h)
                else:
                    h = np.kron(eye, h)
            hamiltonian += h
        return hamiltonian
