# -*- coding: utf-8 -*-
# @authors: S. Carrazza and A. Garcia
from abc import abstractmethod


class Gate(object):
    """The base class for gate implementation

    **Properties:**
        * **name** *(str)* - the gate string name
    """

    def __init__(self):
        self.name = None
        self.nqubits = None

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
        self.qubits = [q]


class X(Gate):
    """The Pauli X gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(X, self).__init__()
        self.name = "X"
        self.qubits = [q]


class Y(Gate):
    """The Pauli Y gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(Y, self).__init__()
        self.name = "Y"
        self.qubits = [q]


class Z(Gate):
    """The Pauli Z gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(Z, self).__init__()
        self.name = "Z"
        self.qubits = [q]


class Barrier(Gate):
    """The barrier gate."""

    def __init__(self, q):
        super(Barrier, self).__init__()
        self.name = "barrier"
        self.qubits = [q]


class Iden(Gate):
    """The identity gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(Iden, self).__init__()
        self.name = "Iden"
        self.qubits = [q]


class M(Gate):
    """The Measure Z gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, *q):
        super(M, self).__init__()
        self.name = "measure"
        self.qubits = set(q)

    # TODO: Override `controlled_by` method and raise NotImplementedError
    # We will not allow controlled measurements for simplicity (for now).


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
        self.qubits = [q]
        self.theta = theta


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
        self.qubits = [q]
        self.theta = theta


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
        self.qubits = [q]
        self.theta = theta


class CNOT(Gate):
    """The Controlled-NOT gate.

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
    """

    def __init__(self, q0, q1):
        super(CNOT, self).__init__()
        self.name = "CNOT"
        self.qubits = [q0, q1]


class SWAP(Gate):
    """The swap gate.

    Args:
        q0, q1 (ints): id numbers of the qubits to be swapped.
    """

    def __init__(self, q0, q1):
        super(SWAP, self).__init__()
        self.name = "SWAP"
        self.qubits = [q0, q1]


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
        self.qubits = [q0, q1]
        self.theta = theta


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
        self.qubits = [q0, q1, q2]


class Flatten(Gate):
    """Custom coefficients

    Args:
        coeff (list): list of coefficients for the wave function.
    """

    def __init__(self, coefficients):
        super(Flatten, self).__init__()
        self.name = "Flatten"
        self.coefficients = coefficients
