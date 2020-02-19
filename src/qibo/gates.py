# -*- coding: utf-8 -*-
# @authors: S. Carrazza and A. Garcia


class Gate(object):
    """The base class for gate implementation

    **Properties:**

        * **args** *(dict)* - a dictionnary which holds the gate init arguments
        * **name** *(str)* - the gate string name
    """

    def __init__(self):
        self.args = None
        self.name = None


class CNOT(Gate):
    """The Controlled-NOT gate.

    Args:
        q0 (int): the first qubit id number.
        q1 (int): the second qubit id number.
    """

    def __init__(self, q0, q1):
        super(CNOT, self).__init__()
        self.args = {"id0": q0, "id1": q1}
        self.name = "CNOT"


class H(Gate):
    """The Hadamard gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(H, self).__init__()
        self.args = {"id": q}
        self.name = "H"


class X(Gate):
    """The Pauli X gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(X, self).__init__()
        self.args = {"id": q}
        self.name = "X"


class Y(Gate):
    """The Pauli Y gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(Y, self).__init__()
        self.args = {"id": q}
        self.name = "Y"


class Z(Gate):
    """The Pauli Z gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(Z, self).__init__()
        self.args = {"id": q}
        self.name = "Z"


class Barrier(Gate):
    """The barrier gate."""

    def __init__(self, q):
        super(Barrier, self).__init__()
        self.args = {"id": q}
        self.name = "barrier"


class S(Gate):
    """The swap gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(S, self).__init__()
        self.args = {"id": q}
        self.name = "S"


class T(Gate):
    """The Toffoli gate.

    Args:
        q (int): the qubit id number
    """

    def __init__(self, q):
        super(T, self).__init__()
        self.args = {"id": q}
        self.name = "T"


class Iden(Gate):
    """The identity gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(Iden, self).__init__()
        self.args = {"id": q}
        self.name = "Iden"


class MX(Gate):
    """The Measure X gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(MX, self).__init__()
        self.args = {"id": q}
        self.name = "MX"


class MY(Gate):
    """The Measure Y gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(MY, self).__init__()
        self.args = {"id": q}
        self.name = "MY"


class MZ(Gate):
    """The Measure Z gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(MZ, self).__init__()
        self.args = {"id": q}
        self.name = "measure"


class RX(Gate):
    """Rotation X-axis.

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
    """

    def __init__(self, q, theta):
        super(RX, self).__init__()
        self.args = {"id": q, "theta": theta}
        self.name = "RX"


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
        self.args = {"id": q, "theta": theta}
        self.name = "RY"


class RZ(Gate):
    """Rotation Z-axis.

    Convention is [[1, 0], [0, exp(i π theta)]].

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
    """

    def __init__(self, q, theta):
        super(RZ, self).__init__()
        self.args = {"id": q, "theta": theta}
        self.name = "RZ"


class Flatten(Gate):
    """Custom coefficients

    Args:
        coeff (list): list of coefficients for the wave function.
    """

    def __init__(self, coefficients):
        super(Flatten, self).__init__()
        self.args = {"coefficients": coefficients}
        self.name = "Flatten"