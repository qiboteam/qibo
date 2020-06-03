# -*- coding: utf-8 -*-
# @authors: S. Carrazza and A. Garcia
import sys
from typing import List, Optional, Sequence, Tuple

QASM_GATES = {"h": "H", "x": "X", "y": "Y", "z": "Z",
              "rx": "RX", "ry": "RY", "rz": "RZ",
              "cx": "CNOT", "swap": "SWAP",
              "crz": "CZPow", "ccx": "TOFFOLI"}
PARAMETRIZED_GATES = {"rx", "ry", "rz", "crz"}


class Gate(object):
    """The base class for gate implementation.

    All gates should inherit this class.

    Attributes:
        name: Name of the gate.
        target_qubits: Tuple with ids of target qubits.
    """

    def __init__(self):
        self.name = None
        self.is_channel = False
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
        if q:
            self.is_controlled_by = True
            self.control_qubits = q
        return self

    def __call__(self, state, is_density_matrix):
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
    _MODULE = sys.modules[__name__]

    def __init__(self, q):
        super(X, self).__init__()
        self.name = "x"
        self.target_qubits = (q,)

    def controlled_by(self, *q):
        """Fall back to CNOT and Toffoli if controls are one or two."""
        if len(q) == 1:
            gate = getattr(self._MODULE, "CNOT")(q[0], self.target_qubits[0])
        elif len(q) == 2:
            gate = getattr(self._MODULE, "TOFFOLI")(q[0], q[1], self.target_qubits[0])
        else:
            gate = super(X, self).controlled_by(*q)
        return gate


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

    _MODULE = sys.modules[__name__]

    def __init__(self, q):
        super(Z, self).__init__()
        self.name = "z"
        self.target_qubits = (q,)

    def controlled_by(self, *q):
        """Fall back to CZ if control is one."""
        if len(q) == 1:
            gate = getattr(self._MODULE, "CZ")(q[0], self.target_qubits[0])
        else:
            gate = super(X, self).controlled_by(*q)
        return gate


class M(Gate):
    """The Measure Z gate.

    Args:
        *q (int): id numbers of the qubits to measure.
            It is possible to measure multiple qubits using ``gates.M(0, 1, 2, ...)``.
            If the qubits to measure are held in an iterable (eg. list) the ``*``
            operator can be used, for example ``gates.M(*[0, 1, 4])`` or ``gates.M(*range(5))``.
        register_name: Optional name of the register to distinguish it from
            other registers when used in circuits.
    """

    def __init__(self, *q, register_name: Optional[str] = None):
        super(M, self).__init__()
        self.name = "measure"
        self.target_qubits = tuple(q)
        self._control_qubits = tuple()
        self.register_name = register_name

        self._unmeasured_qubits = None # Tuple
        self._reduced_target_qubits = None # List

    def _add(self, qubits: Tuple[int]):
        """Adds target qubits to a measurement gate.

        This method is only used for creating the global measurement gate used
        by the `models.Circuit`.
        The user is not supposed to use this method and a `ValueError` is
        raised if he does so.

        Args:
            qubits: Tuple of qubit ids to be added to the measurement's qubits.
        """
        if self._unmeasured_qubits is not None:
            raise RuntimeError("Cannot add qubits to a measurement gate that "
                               "was executed.")
        self.target_qubits += qubits

    def _set_unmeasured_qubits(self):
        if self._nqubits is None:
            raise ValueError("Cannot calculate set of unmeasured qubits if "
                             "the number of qubits in the circuit is unknown.")
        if self._unmeasured_qubits is not None:
            raise RuntimeError("Cannot recalculate unmeasured qubits.")
        target_qubits = set(self.target_qubits)
        unmeasured_qubits = []
        reduced_target_qubits = dict()
        for i in range(self.nqubits):
            if i in target_qubits:
                reduced_target_qubits[i] = i - len(unmeasured_qubits)
            else:
                unmeasured_qubits.append(i)

        self._unmeasured_qubits = tuple(unmeasured_qubits)
        self._reduced_target_qubits = list(reduced_target_qubits[i]
                                           for i in self.target_qubits)

    @property
    def unmeasured_qubits(self) -> Tuple[int]:
        """Tuple with ids of unmeasured qubits sorted in increasing order.

        This is useful when tracing out unmeasured qubits to calculate
        probabilities.
        """
        if self._unmeasured_qubits is None:
            self._set_unmeasured_qubits()
        return self._unmeasured_qubits

    @property
    def reduced_target_qubits(self) -> List[int]:
        if self._unmeasured_qubits is None:
            self._set_unmeasured_qubits()
        return self._reduced_target_qubits

    def controlled_by(self, *q):
        """"""
        raise NotImplementedError("Measurement gates cannot be controlled.")


class RX(Gate):
    """Rotation around the X-axis of the Bloch sphere.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        \\cos \\frac{\\theta }{2}  &
        -i\\sin \\frac{\\theta }{2} \\\\
        -i\\sin \\frac{\\theta }{2}  &
        \\cos \\frac{\\theta }{2} \\\\
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
        \\begin{pmatrix}
        \\cos \\frac{\\theta }{2}  &
        -\\sin \\frac{\\theta }{2} \\\\
        \\sin \\frac{\\theta }{2}  &
        \\cos \\frac{\\theta }{2} \\\\
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
        e^{-i \\theta / 2} & 0 \\\\
        0 & e^{i \\theta / 2} \\\\
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


class CZ(Gate):
    """The Controlled-Phase gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & -1 \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
    """

    def __init__(self, q0, q1):
        super(CZ, self).__init__()
        self.name = "cz"
        self.control_qubits = (q0,)
        self.target_qubits = (q1,)


class CZPow(Gate):
    """Controlled rotation around the Z-axis of the Bloch sphere.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & e^{i \\theta } \\\\
        \\end{pmatrix}

    Note that this differs from the :class:`qibo.base.gates.RZ` gate.

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): the rotation angle.
    """

    def __init__(self, q0, q1, theta):
        super(CZPow, self).__init__()
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


class fSim(Gate):
    """The fSim gate defined in `arXiv:2001.08343 <https://arxiv.org/abs/2001.08343>`_.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & \\cos \\theta & -i\\sin \\theta & 0 \\\\
        0 & -i\\sin \\theta & \\cos \\theta & 0 \\\\
        0 & 0 & 0 & e^{-i \\phi } \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the first qubit to be swapped id number.
        q1 (int): the second qubit to be swapped id number.
        theta (float): Angle for the one-qubit rotation.
        phi (float): Angle for the |11> phase.
    """
    # TODO: Check how this works with QASM.

    def __init__(self, q0, q1, theta, phi):
        super(fSim, self).__init__()
        self.name = "fsim"
        self.target_qubits = (q0, q1)
        self.theta = theta
        self.phi = phi


class GeneralizedfSim(Gate):
    """The fSim gate with a general rotation.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & R_{00} & R_{01} & 0 \\\\
        0 & R_{10} & R_{11} & 0 \\\\
        0 & 0 & 0 & e^{-i \\phi } \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the first qubit to be swapped id number.
        q1 (int): the second qubit to be swapped id number.
        unitary (np.ndarray): Unitary that corresponds to the one-qubit rotation.
        phi (float): Angle for the |11> phase.
    """

    def __init__(self, q0, q1, unitary, phi):
        super(GeneralizedfSim, self).__init__()
        self.name = "generalizedfsim"
        self.target_qubits = (q0, q1)
        self.unitary = unitary
        self.phi = phi


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


class VariationalLayer(Gate):
    """Layer of one-qubit parametrized gates followed by two-qubit entangling gates.

    Args:
        TODO
    """

    def __init__(self, qubit_pairs, one_qubit_gate, two_qubit_gate, thetas,
                 name: Optional[str] = None):
        super(VariationalLayer, self).__init__()
        self.name = "VariationalLayer" if name is None else name

        if len(thetas) != 2 * len(qubit_pairs):
            raise ValueError("Cannot initialize variational layer with {} "
                             "qubit pairs and {} variational parameters."
                             "".format(len(qubit_pairs), len(thetas)))
        self.thetas = thetas
        self.qubit_pairs = qubit_pairs
        self.target_qubits = tuple(q for p in qubit_pairs for q in p)
        if set(self.thetas.keys()) != set(self.target_qubits):
            raise ValueError("Keys of theta parameters do not agree with given "
                             "qubit pairs.")

        self.one_qubit_gate = one_qubit_gate
        self.two_qubit_gate = two_qubit_gate


class NoiseChannel(Gate):
    """Probabilistic noise channel.

    Implements the following evolution

    .. math::
        \\rho \\rightarrow (1 - p_x - p_y - p_z) \\rho + p_x X\\rho X + p_y Y\\rho Y + p_z Z\\rho Z

    which can be used to simulate phase flip and bit flip errors.

    Args:
        q (int): Qubit id that the noise acts on.
        px (float): Bit flip (X) error probability.
        py (float): Y-error probability.
        pz (float): Phase flip (Z) error probability.
    """

    def __init__(self, q, px=0, py=0, pz=0):
        super(NoiseChannel, self).__init__()
        self.name = "NoiseChannel"
        self.is_channel = True
        self.target_qubits = (q,)
        self.p = (px, py, pz)
        self.total_p = sum(self.p)

    def controlled_by(self, *q):
        """"""
        raise ValueError("Noise channel cannot be controlled on qubits.")


class GeneralChannel(Gate):
    """General channel defined by arbitrary Krauss operators.

    Implements the following evolution

    .. math::
        \\rho \\rightarrow \\sum _k A_k \\rho A_k^\\dagger

    where A are arbitrary Krauss operators given by the user. Note that the
    Krauss operators set should be trace preserving, however this is not checked here.
    For more information on channels and Krauss operators please check
    `J. Preskill's notes <http://www.theory.caltech.edu/people/preskill/ph219/chap3_15.pdf>`_.

    Args:
        A (list): List of Krauss operators as pairs ``(qubits, Ak)`` where
          qubits are the qubit ids that ``Ak`` acts on and ``Ak`` is the
          corresponding matrix.

    Example:
        ::

            from qibo.models import Circuit
            from qibo import gates
            # initialize circuit with 3 qubits
            c = Circuit(3)
            # define a sqrt(0.4) * X gate
            a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
            # define a sqrt(0.6) * CNOT gate
            a2 = np.sqrt(0.6) * np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                          [0, 0, 0, 1], [0, 0, 1, 0]])
            # define the channel rho -> 0.4 X{1} rho X{1} + 0.6 CNOT{0, 2} rho CNOT{0, 2}
            channel = gates.GeneralChannel([((1,), a1), ((0, 2), a2)])
            # add the channel to the circuit
            c.add(channel)
    """

    def __init__(self, A):
        super(GeneralChannel, self).__init__()
        self.name = "GeneralChannel"
        self.is_channel = True
        self.target_qubits = tuple(sorted(set(
          q for qubits, _ in A for q in qubits)))

        # Check that given operators have the proper shape
        for qubits, matrix in A:
            rank = 2 ** len(qubits)
            shape = tuple(matrix.shape)
            if shape != (rank, rank):
                raise ValueError("Invalid Krauss operator shape {} for acting "
                                 "on {} qubits.".format(shape, len(qubits)))

    def controlled_by(self, *q):
        """"""
        raise ValueError("Channel cannot be controlled on qubits.")


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
