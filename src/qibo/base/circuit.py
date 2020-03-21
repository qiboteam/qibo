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
        # Flag to keep track if the circuit was executed
        # We do not allow adding gates in an executed circuit
        self.is_executed = False

        self.measure_sets = []
        self.measure_qubits = set()

    def __add__(self, circuit):
        """Add circuits.

        Args:
            circuit: Circuit to be added to the current one.
        Return:
            The resulting circuit from the addition.
        """
        return BaseCircuit._circuit_addition(self, circuit)

    @classmethod
    def _circuit_addition(cls, c1, c2):
        if c1.nqubits != c2.nqubits:
            raise ValueError("Cannot add circuits with different number of "
                             "qubits. The first has {} qubits while the "
                             "second has {}".format(c1.nqubits, c2.nqubits))
        newcircuit = cls(c1.nqubits)
        for gate in c1.queue:
            newcircuit.add(gate)
        for gate in c2.queue:
            newcircuit.add(gate)
        return newcircuit

    def add(self, gate):
        """Add a gate to a given queue.

        Args:
            gate (qibo.gates): the specific gate (see :ref:`Gates`).
        """
        if self.is_executed:
            raise RuntimeError("Cannot add gates to a circuit after it is "
                               "executed.")

        # Set number of qubits in gate
        if gate._nqubits is None:
            gate.nqubits = self.nqubits
        elif gate.nqubits != self.nqubits:
            raise ValueError("Attempting to add gate with {} total qubits to "
                             "a circuit with {} qubits."
                             "".format(gate.nqubits, self.nqubits))

        # Check if any of the qubits that the gate acts on is already measured
        # Currently we do not allow measured qubits to be reused
        for qubit in gate.qubits:
            if qubit in self.measure_qubits:
                raise ValueError("Cannot reuse qubit {} because it is already "
                                 "measured".format(qubit))

        if gate.name == "measure":
            self.measure_sets.append(gate.target_qubits)
            self.measure_qubits |= gate.target_qubits
        else:
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
    def execute(self):
        """Executes the circuit on a given backend.

        Args:
            model: (qibo.models.Circuit): The circuit to be executed.
        Returns:
            The final wave function.
        """
        raise NotImplementedError
