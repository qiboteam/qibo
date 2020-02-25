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
        gate.nqubits = self.nqubits
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
    def execute(self, model):
        """Executes the circuit on a given backend.

        Args:
            model: (qibo.models.Circuit): The circuit to be executed.
        Returns:
            The final wave function.
        """
        pass