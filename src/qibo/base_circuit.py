# -*- coding: utf-8 -*-
# @authors: S. Carrazza and A. Garcia
from abc import ABCMeta, abstractmethod


class BaseCircuit(object):
    """This class implements the circuit object which holds all gates.

    Args:
        nqubits (int): number of quantum bits.

    Example:
        ::

            from qibo.models import Circuit
            c = Circuit(3) # initialized circuit with 3 qubits
    """

    __metaclass__ = ABCMeta

    def __init__(self, nqubits):
        """Initialize properties."""
        self.nqubits = nqubits
        self.queue = []

    def __add__(self, c0):
        """Add circuits.

        Args:
            c0 (qibo.models.Circuit): the circuit to be added.
        Return:
            qibo.models.Circuit: a new circuit.
        """
        if self.nqubits != c0.size():
            raise TypeError("Circuits of different size")
        newcircuit = Circuit(self.nqubits)
        newgates = self.queue + c0.gates
        for gate in newgates:
            newcircuit.add(gate)
        return newcircuit

    def add(self, gate):
        """Add a gate to a given queue.

        Args:
            gate (qibo.gates): the specific gate (see :ref:`Gates`).
        """
        self.queue.append(gate)

    @property
    def size(self):
        """
        Return:
            number of qubits in the circuit
        """
        return self.nqubits

    @property
    def depth(self):
        """
        Return:
            number of gates/operations in the circuit
        """
        return len(self.queue)

    @abstractmethod
    def CNOT(self, **args):
        """The Controlled-NOT gate."""
        pass

    @abstractmethod
    def H(self, **args):
        """The Hadamard gate."""
        pass

    @abstractmethod
    def X(self, **args):
        """The Pauli X gate."""
        pass

    @abstractmethod
    def Y(self, **args):
        """The Pauli Y gate."""
        pass

    @abstractmethod
    def Z(self, **args):
        """The Pauli Z gate."""
        pass

    @abstractmethod
    def Barrier(self, **args):
        """The barrier gate."""
        pass

    @abstractmethod
    def S(self, **args):
        """The swap gate."""
        pass

    @abstractmethod
    def T(self, **args):
        """The Toffoli gate."""
        pass

    @abstractmethod
    def Iden(self, **args):
        """The identity gate."""
        pass

    @abstractmethod
    def RX(self, **args):
        """The rotation around X-axis gate."""
        pass

    @abstractmethod
    def RY(self, **args):
        """The rotation around Y-axis gate."""
        pass

    @abstractmethod
    def RZ(self, **args):
        """The rotation around Z-axis gate."""
        pass

    @abstractmethod
    def CRZ(self, **args):
        """The controlled rotation around Z-axis gate."""
        pass

    @abstractmethod
    def MX(self, **args):
        """The measure gate X."""
        pass

    @abstractmethod
    def MY(self, **args):
        """The measure gate Y."""
        pass

    @abstractmethod
    def MZ(self, **args):
        """The measure gate Z."""
        pass

    @abstractmethod
    def Flatten(self, **args):
        """Set wave function coefficients"""
        pass

    @abstractmethod
    def execute(self, model):
        """Executes the circuit on a given backend.

        Args:
            model: (qibo.models.Circuit): The circuit to be executed.
        Returns:
            The final wave function.
        """
        pass
