# -*- coding: utf-8 -*-
import numpy as np
from qibo.config import matrices
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
        self._min_eigenvalue = None

    @abstractmethod
    def _build(self, sigma):
        """Implements the Hamiltonian construction."""
        pass

    def min_eigenvalue(self):
        """Computes the minimum eigenvalue for the Hamiltonian."""
        if self._min_eigenvalue is None:
            self._min_eigenvalue = np.min(
                np.linalg.eigvalsh(self.hamiltonian)
            )
        return self._min_eigenvalue

    def expectation(self, state):
        """Computes the real expectation value for a given state.
        Args:
            state (array): the expectation state.
        """
        n = np.dot(np.conj(state).T, np.dot(self.hamiltonian, state))
        return np.real(n)


class XXZ(Hamiltonian):
    """This class implements the Heisenberg XXZ model.
    The mode uses the Pauli matrices and build the final
    Hamiltonian: H = Hx + Hy + delta * Hz.

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

    def _build(self, sigma):
        """Builds the Heisenber model for a given operator sigma"""
        hamiltonian = 0
        eye = matrices._npI()
        n = self.nqubits
        for i in range(n):
            h = 1
            for j in range(n):
                if i == j % n or i == (j+1) % n:
                    h = np.kron(sigma, h)
                else:
                    h = np.kron(eye, h)
            hamiltonian += h
        return hamiltonian
