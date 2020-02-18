# -*- coding: utf-8 -*-
# @authors: S. Carrazza and A. Garcia
import os


class Circuit(object):
    """This class implements the circuit object which holds all gates.

    Args:
        nqubits (int): number of quantum bits.

    Example:
        ::

            from qibo.models import Circuit
            c = Circuit(3) # initialized circuit with 3 qubits
    """

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
        newgates = self.queue + c0.gates()
        for gate in newgates:
            newcircuit.add(gate)
        return newcircuit

    def add(self, gate):
        """Add a gate to a given queue.

        Args:
            gate (qibo.gates): the specific gate (see :ref:`Gates`).
        """
        self.queue.append(gate)

    def gates(self):
        """
        Return:
            queue of sequential operations in the circuit
        """
        return self.queue

    def size(self):
        """
        Return:
            number of qubits in the circuit
        """
        return self.nqubits

    def depth(self):
        """
        Return:
            number of gates/operations in the circuit
        """
        return len(self.queue)
