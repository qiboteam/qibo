# -*- coding: utf-8 -*-
# @authors: S. Carrazza and A. Garcia
from abc import abstractmethod
from typing import Tuple


class Gate(object):
    """The base class for gate implementation.

    **Properties:**
        * name: the gate string name.
        * target_qubits: tuple with indices of target qubits.
        * control_qubits: tuple with indices of control qubits.
        * qubits: tuple with all qubits (control + target) that the gate acts.
        * nqubits: total number of qubits in the circuit/state the gate acts.
            This is set automatically when the gate is added to a circuit or
            when it is called on a state.
        * nstates: 2 ** nqubits.
    """

    def __init__(self):
        self.name = None
        self.parameters = []

        self.target_qubits = tuple()
        self.control_qubits = tuple()

        self._nqubits = None
        self._nstates = None

    @property
    def qubits(self) -> Tuple[int]:
        return self.control_qubits + self.target_qubits

    @property
    def nqubits(self) -> int:
        if self._nqubits is None:
            raise ValueError("Accessing number of qubits for gate {} but "
                             "this is not yet set.".format(self))
        return self._nqubits

    @property
    def nstates(self) -> int:
        if self._nstates is None:
            raise ValueError("Accessing number of qubits for gate {} but "
                             "this is not yet set.".format(self))
        return self._nstates

    @nqubits.setter
    def nqubits(self, n: int):
        if self._nqubits is not None and self._nqubits != n:
            raise ValueError("Setting number of qubits as {} while it is "
                             "already set as {}.".format(self._nqubits))
        self._nqubits = n
        self._nstates = 2**n

    def controlled_by(self, *q) -> "Gate":
        if self.control_qubits:
            raise ValueError("Cannot use `controlled_by` method on gate {} "
                             "because it is already controlled by {}."
                             "".format(self, self.control_qubits))
        self.control_qubits = tuple(q)
        return self

    @abstractmethod
    def __call__(self, state):
        """Implements the `Gate` on a given state."""
        pass


class H(Gate):
    """The Hadamard gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(H, self).__init__()
        self.name = "H"
        self.target_qubits = (q,)


class X(Gate):
    """The Pauli X gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(X, self).__init__()
        self.name = "X"
        self.target_qubits = (q,)


class Y(Gate):
    """The Pauli Y gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(Y, self).__init__()
        self.name = "Y"
        self.target_qubits = (q,)


class Z(Gate):
    """The Pauli Z gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(Z, self).__init__()
        self.name = "Z"
        self.target_qubits = (q,)


class Barrier(Gate):
    """The barrier gate."""

    def __init__(self, q):
        super(Barrier, self).__init__()
        self.name = "barrier"
        self.target_qubits = (q,)


class Iden(Gate):
    """The identity gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(Iden, self).__init__()
        self.name = "Iden"
        self.target_qubits = (q,)


class MX(Gate):
    """The Measure X gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(MX, self).__init__()
        self.name = "MX"
        self.target_qubits = (q,)


class MY(Gate):
    """The Measure Y gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(MY, self).__init__()
        self.name = "MY"
        self.target_qubits = (q,)


class MZ(Gate):
    """The Measure Z gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(MZ, self).__init__()
        self.name = "measure"
        self.target_qubits = (q,)


class RX(Gate):
    """Rotation X-axis.

    [[g·c, -i·g·s], [-i·g·s, g·c]]
    where c = cos(π theta / 2), s = sin(π theta / 2), g = exp(i π theta / 2).

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
    """

    def __init__(self, q, theta):
        super(RX, self).__init__()
        self.name = "RX"
        self.target_qubits = (q,)


class RY(Gate):
    """Rotation Y-axis defined as:

    [[g·c, -g·s], [g·s, g·c]]
    where c = cos(π theta / 2), s = sin(π theta / 2), g = exp(i π theta / 2).

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
    """

    def __init__(self, q, theta):
        super(RY, self).__init__()
        self.name = "RY"
        self.target_qubits = (q,)


class RZ(Gate):
    """Rotation Z-axis.

    Convention is [[1, 0], [0, exp(i π theta)]].

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
    """

    def __init__(self, q, theta):
        super(RZ, self).__init__()
        self.name = "RZ"
        self.target_qubits = (q,)


class CNOT(Gate):
    """The Controlled-NOT gate.

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
    """

    def __init__(self, q0, q1):
        super(CNOT, self).__init__()
        self.name = "CNOT"
        self.control_qubits = (q0,)
        self.target_qubits = (q1,)


class SWAP(Gate):
    """The swap gate.

    Args:
        q0, q1 (ints): id numbers of the qubits to be swapped.
    """

    def __init__(self, q0, q1):
        super(SWAP, self).__init__()
        self.name = "SWAP"
        self.target_qubits = (q0, q1)


class CRZ(Gate):
    """Controlled Rotation Z-axis.

    Convention is the same as RZ.

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): the rotation angle.
    """

    def __init__(self, q0, q1, theta):
        super(CRZ, self).__init__()
        self.name = "CRZ"
        self.control_qubits = (q0,)
        self.target_qubits = (q1,)


class Toffoli(Gate):
    """The Toffoli gate.

    Args:
        q0 (int): the first control qubit id number.
        q1 (int): the second control qubit id number.
        q2 (int): the target qubit id number.
    """

    def __init__(self, q0, q1, q2):
        super(Toffoli, self).__init__()
        self.name = "Toffoli"
        self.control_qubits = (q0, q1)
        self.target_qubits = (q2,)


class Flatten(Gate):
    """Custom coefficients

    Args:
        coeff (list): list of coefficients for the wave function.
    """

    def __init__(self, coefficients):
        super(Flatten, self).__init__()
        self.name = "Flatten"
        self.coefficients = coefficients