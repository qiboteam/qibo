# -*- coding: utf-8 -*-
# @authors: S. Carrazza and A. Garcia
from typing import Optional, Sequence, Set, Tuple


class Gate(object):
    """The base class for gate implementation.

    All gates should inherit this class.

    Attributes:
        name: Name of the gate.
        target_qubits: Tuple with ids of target qubits.
    """

    def __init__(self):
        self.name = None
        self.is_controlled_by = False
        self.parameters = []

        self.target_qubits = tuple()
        self._control_qubits = tuple()

        self._nqubits = None
        self._nstates = None

    @property
    def control_qubits(self) -> Tuple[int]:
        """Tuple with ids of control qubits sorted in increasing order."""
        return self._control_qubits

    @control_qubits.setter
    def control_qubits(self, q: Sequence[int]):
        """Sets control qubits sorted."""
        self._control_qubits = tuple(sorted(q))

    @property
    def qubits(self) -> Tuple[int]:
        """Tuple with ids of all qubits (control and target) that the gate acts."""
        return self.control_qubits + self.target_qubits

    @property
    def nqubits(self) -> int:
        """Number of qubits in the circuit that the gate is part of.

        This is set automatically when the gate is added on a circuit or
        when the gate is called on a state. The user should not set this.
        """
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
        """Sets the total number of qubits that this gate acts on.

        This setter is used by `circuit.add` if the gate is added in a circuit
        or during `__call__` if the gate is called directly on a state.
        The user is not supposed to set `nqubits` by hand.
        """
        if self._nqubits is not None:
            raise ValueError("The number of qubits for this gates is already "
                             "set to {}.".format(self._nqubits))
        self._nqubits = n
        self._nstates = 2**n

    def controlled_by(self, *q: int) -> "Gate":
        """Controls the gate on (arbitrarily many) qubits.

        Args:
            *q (int): Ids of the qubits that the gate will be controlled on.

        Returns:
            A :class:`qibo.base.gates.Gate` object in with the corresponding gate being
            controlled in the given qubits.
        """
        if self.control_qubits:
            raise ValueError("Cannot use `controlled_by` method on gate {} "
                             "because it is already controlled by {}."
                             "".format(self, self.control_qubits))
        if self._nqubits is not None:
            raise RuntimeError("Cannot use controlled_by on a gate that is "
                               "part of a Circuit or has been called on a "
                               "state.")
        self.is_controlled_by = True
        self.control_qubits = q
        return self

    def __call__(self, state):
        """Acts with the gate on a given state vector:

        Args:
            state: Input state vector.
                The type and shape of this depend on the backend.

        Returns:
            The state vector after the action of the gate.
        """
        raise NotImplementedError


class H(Gate):
    """The Hadamard gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(H, self).__init__()
        self.name = "h"
        self.target_qubits = (q,)


class X(Gate):
    """The Pauli X gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(X, self).__init__()
        self.name = "x"
        self.target_qubits = (q,)


class Y(Gate):
    """The Pauli Y gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(Y, self).__init__()
        self.name = "y"
        self.target_qubits = (q,)


class Z(Gate):
    """The Pauli Z gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(Z, self).__init__()
        self.name = "z"
        self.target_qubits = (q,)


class M(Gate):
    """The Measure Z gate.

    Args:
        *q (int): id numbers of the qubits to measure.
            Order does not matter as measurement results will follow increasing
            order in ids.
        register_name: Optional name of the register to distinguish it from
            other registers when used in circuits.
    """

    def __init__(self, *q, register_name: Optional[str] = None):
        super(M, self).__init__()
        self.name = "measure"
        self.target_qubits = set(q)
        self.register_name = register_name

        self.is_executed = False
        self._unmeasured_qubits = None # Set

    def _add(self, qubits: Set[int]):
        """Adds target qubits to a measurement gate.

        This method is only used for creating the global measurement gate used
        by the `models.Circuit`.
        The user is not supposed to use this method and a `ValueError` is
        raised if he does so.

        Args:
            qubits: Set of qubit ids to be added to the measurement's qubits.
        """
        if self.is_executed:
            raise RuntimeError("Cannot add qubits to a measurement gate that "
                               "was executed.")
        self.target_qubits |= qubits
        if self._unmeasured_qubits is not None:
            self._unmeasured_qubits -= qubits

    @property
    def qubits(self) -> Tuple[int]:
        """Tuple with ids of measured qubits sorted in increasing order."""
        return tuple(sorted(self.target_qubits))

    @property
    def unmeasured_qubits(self) -> Tuple[int]:
        """Tuple with ids of unmeasured qubits sorted in increasing order.

        This is useful when tracing out unmeasured qubits to calculate
        probabilities.
        """
        if self._nqubits is None:
            raise ValueError("Cannot calculate set of unmeasured qubits if "
                             "the number of qubits in the circuit is unknown.")
        if self._unmeasured_qubits is None:
            self._unmeasured_qubits = set(i for i in range(self.nqubits)
                                          if i not in self.target_qubits)
        return tuple(sorted(self._unmeasured_qubits))

    def controlled_by(self, *q):
        """"""
        raise NotImplementedError("Measurement gates cannot be controlled.")


class RX(Gate):
    """Rotation around the X-axis of the Bloch sphere.

    Corresponds to the following unitary matrix

    .. math::
        e^{i \\pi \\theta / 2}\\begin{pmatrix}
        \\cos \\left (\\frac{\\pi }{2} \\theta \\right ) &
        -i\\sin \\left (\\frac{\\pi }{2} \\theta \\right ) \\\\
        -i\\sin \\left (\\frac{\\pi }{2} \\theta \\right ) &
        \\cos \\left (\\frac{\\pi }{2} \\theta \\right ) \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
    """

    def __init__(self, q, theta):
        super(RX, self).__init__()
        self.name = "rx"
        self.target_qubits = (q,)
        self.theta = theta


class RY(Gate):
    """Rotation around the Y-axis of the Bloch sphere.

    Corresponds to the following unitary matrix

    .. math::
        e^{i \\pi \\theta / 2}\\begin{pmatrix}
        \\cos \\left (\\frac{\\pi }{2} \\theta \\right ) &
        -\\sin \\left (\\frac{\\pi }{2} \\theta \\right ) \\\\
        \\sin \\left (\\frac{\\pi }{2} \\theta \\right ) &
        \\cos \\left (\\frac{\\pi }{2} \\theta \\right ) \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
    """

    def __init__(self, q, theta):
        super(RY, self).__init__()
        self.name = "ry"
        self.target_qubits = (q,)
        self.theta = theta


class RZ(Gate):
    """Rotation around the X-axis of the Bloch sphere.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 \\\\
        0 & e^{i \\pi \\theta} \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
    """

    def __init__(self, q, theta):
        super(RZ, self).__init__()
        self.name = "rz"
        self.target_qubits = (q,)
        self.theta = theta


class CNOT(Gate):
    """The Controlled-NOT gate.

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
    """

    def __init__(self, q0, q1):
        super(CNOT, self).__init__()
        self.name = "cx"
        self.control_qubits = (q0,)
        self.target_qubits = (q1,)


class CRZ(Gate):
    """Controlled rotation around the Z-axis of the Bloch sphere.

    The convention for the unitary matrix is the same as in `RZ`.

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): the rotation angle.
    """

    def __init__(self, q0, q1, theta):
        super(CRZ, self).__init__()
        self.name = "crz"
        self.control_qubits = (q0,)
        self.target_qubits = (q1,)
        self.theta = theta


class SWAP(Gate):
    """The swap gate.

    Args:
        q0 (int): the first qubit to be swapped id number.
        q1 (int): the second qubit to be swapped id number.
    """

    def __init__(self, q0, q1):
        super(SWAP, self).__init__()
        self.name = "swap"
        self.target_qubits = (q0, q1)


class TOFFOLI(Gate):
    """The Toffoli gate.

    Args:
        q0 (int): the first control qubit id number.
        q1 (int): the second control qubit id number.
        q2 (int): the target qubit id number.
    """

    def __init__(self, q0, q1, q2):
        super(TOFFOLI, self).__init__()
        self.name = "ccx"
        self.control_qubits = (q0, q1)
        self.target_qubits = (q2,)


class Unitary(Gate):
    """Arbitrary unitary gate.

    Args:
        unitary: Unitary matrix as a tensor supported by the backend.
            Note that there is no check that the matrix passed is actually
            unitary. This allows the user to create non-unitary gates.
        *q (int): Qubit id numbers that the gate acts on.
        name (str): Optional name for the gate.
    """

    def __init__(self, unitary, *q, name: Optional[str] = None):
        super(Unitary, self).__init__()
        self.name = "Unitary" if name is None else name
        self.unitary = unitary
        self.target_qubits = tuple(q)


class Flatten(Gate):
    """Passes an arbitrary state vector in the circuit.

    Args:
        coefficients (list): list of the target state vector components.
            This can also be a tensor supported by the backend.
    """

    def __init__(self, coefficients):
        super(Flatten, self).__init__()
        self.name = "Flatten"
        self.coefficients = coefficients
        import math
        self.target_qubits = tuple(range(int(math.log2(len(coefficients)))))
