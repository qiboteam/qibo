# -*- coding: utf-8 -*-
# @authors: S. Carrazza and A. Garcia
from qibo import config
from qibo.config import raise_error
from typing import Dict, List, Optional, Sequence, Tuple

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

    import sys
    module = sys.modules[__name__]

    def __init__(self):
        self.name = None
        self.is_channel = False
        self.is_controlled_by = False
        self.is_special_gate = False
        # special gates are ``CallbackGate`` and ``Flatten``

        # args for creating gate
        self.init_args = []
        self.init_kwargs = {}

        self._target_qubits = tuple()
        self._control_qubits = set()
        self.qubits_tensor = None

        self._unitary = None
        self._nqubits = None
        self._nstates = None
        self.qubits_tensor = None

        # Cast gate matrices to the proper device
        self.device = config.get_device()
        # Reference to copies of this gate that are casted in devices when
        # a distributed circuit is used
        self.device_gates = set()

        config.ALLOW_SWITCHERS = False

    @property
    def target_qubits(self) -> Tuple[int]:
        """Tuple with ids of target qubits."""
        return self._target_qubits

    @property
    def control_qubits(self) -> Tuple[int]:
        """Tuple with ids of control qubits sorted in increasing order."""
        return tuple(sorted(self._control_qubits))

    @property
    def qubits(self) -> Tuple[int]:
        """Tuple with ids of all qubits (control and target) that the gate acts."""
        return self.control_qubits + self.target_qubits

    def _set_target_qubits(self, qubits: Sequence[int]):
        """Helper method for setting target qubits."""
        self._target_qubits = tuple(qubits)
        if len(self._target_qubits) != len(set(qubits)):
            repeated = self._find_repeated(qubits)
            raise_error(ValueError, "Target qubit {} was given twice for gate {}."
                                    "".format(repeated, self.name))

    def _set_control_qubits(self, qubits: Sequence[int]):
        """Helper method for setting control qubits."""
        self._control_qubits = set(qubits)
        if len(self._control_qubits) != len(qubits):
            repeated = self._find_repeated(qubits)
            raise_error(ValueError, "Control qubit {} was given twice for gate {}."
                                    "".format(repeated, self.name))

    @target_qubits.setter
    def target_qubits(self, qubits: Sequence[int]):
        """Sets control qubits tuple."""
        self._set_target_qubits(qubits)
        self._check_control_target_overlap()

    @control_qubits.setter
    def control_qubits(self, qubits: Sequence[int]):
        """Sets control qubits set."""
        self._set_control_qubits(qubits)
        self._check_control_target_overlap()

    def set_targets_and_controls(self, target_qubits: Sequence[int],
                                 control_qubits: Sequence[int]):
        """Sets target and control qubits simultaneously.

        This is used for the reduced qubit updates in the distributed circuits
        because using the individual setters may raise errors due to temporary
        overlap of control and target qubits.
        """
        self._set_target_qubits(target_qubits)
        self._set_control_qubits(control_qubits)
        self._check_control_target_overlap()

    @staticmethod
    def _find_repeated(qubits: Sequence[int]) -> int:
        """Finds the first qubit id that is repeated in a sequence of qubit ids."""
        temp_set = set()
        for qubit in qubits:
            if qubit in temp_set:
                return qubit
            temp_set.add(qubit)

    def _check_control_target_overlap(self):
        """Checks that there are no qubits that are both target and controls."""
        common = set(self._target_qubits) & self._control_qubits
        if common:
            raise_error(ValueError, "{} qubits are both targets and controls for "
                                    "gate {}.".format(common, self.name))

    @property
    def nqubits(self) -> int:
        """Number of qubits in the circuit that the gate is part of.

        This is set automatically when the gate is added on a circuit or
        when the gate is called on a state. The user should not set this.
        """
        if self._nqubits is None:
            raise_error(ValueError, "Accessing number of qubits for gate {} but "
                                    "this is not yet set.".format(self))
        return self._nqubits

    @property
    def nstates(self) -> int:
        if self._nstates is None:
            raise_error(ValueError, "Accessing number of qubits for gate {} but "
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
            raise_error(RuntimeError, "The number of qubits for this gates is already "
                                      "set to {}.".format(self._nqubits))
        self._nqubits = n
        self._nstates = 2**n
        self._calculate_qubits_tensor()
        self._prepare()

    @property
    def unitary(self):
        """Unitary matrix corresponding to the gate's action on a state vector.

        This matrix is not necessarily used by ``__call__`` when applying the
        gate to a state vector.
        """
        if len(self.qubits) > 2:
            raise_error(NotImplementedError, "Cannot calculate unitary matrix for "
                                             "gates that target more than two qubits.")
        if self._unitary is None:
            self._unitary = self.construct_unitary()
            if self.is_controlled_by:
                self._unitary = self.control_unitary(self._unitary)
        return self._unitary

    def construct_unitary(self): # pragma: no cover
        """Constructs the gate's unitary matrix.

        Args:
            *args: Variational parameters for parametrized gates.

        Returns:
            Unitary matrix as an array or tensor supported by the backend.
        """
        # abstract method
        return raise_error(NotImplementedError)

    @staticmethod
    def control_unitary(unitary): # pragma: no cover
        """Controls unitary matrix on one qubit.

        Helper method for ``construct_unitary`` for gates where ``controlled_by``
        has been used.
        """
        # abstract method
        raise_error(NotImplementedError)

    def __matmul__(self, other: "Gate") -> "Gate":
        """Gate multiplication."""
        if self.qubits != other.qubits:
            raise_error(NotImplementedError, "Cannot multiply gates that target "
                                             "different qubits.")
        if self.__class__.__name__ == other.__class__.__name__:
            square_identity = {"H", "X", "Y", "Z", "CNOT", "CZ", "SWAP"}
            if self.__class__.__name__ in square_identity:
                from qibo import gates
                return gates.I(*self.qubits)
        return None

    def __rmatmul__(self, other: "TensorflowGate") -> "TensorflowGate": # pragma: no cover
        # abstract method
        return self.__matmul__(other)

    def _calculate_qubits_tensor(self):
        """Calculates ``qubits`` tensor required for applying gates using custom operators."""
        pass

    def _prepare(self): # pragma: no cover
        """Prepares the gate for application to state vectors.

        Called automatically by the ``nqubits`` setter.
        Calculates the ``matrix`` required to apply the gate to state vectors.
        This is not necessarily the same as the unitary matrix of the gate.
        """
        # abstract method
        pass

    def commutes(self, gate: "Gate") -> bool:
        """Checks if two gates commute.

        Args:
            gate: Gate to check if it commutes with the current gate.

        Returns:
            ``True`` if the gates commute, otherwise ``False``.
        """
        if self.is_special_gate or gate.is_special_gate:
            return False
        t1 = set(self.target_qubits)
        t2 = set(gate.target_qubits)
        a = self.__class__ == gate.__class__ and t1 == t2
        b = not (t1 & set(gate.qubits) or t2 & set(self.qubits))
        return a or b

    def controlled_by(self, *qubits: int) -> "Gate":
        """Controls the gate on (arbitrarily many) qubits.

        Args:
            *qubits (int): Ids of the qubits that the gate will be controlled on.

        Returns:
            A :class:`qibo.base.gates.Gate` object in with the corresponding gate being
            controlled in the given qubits.
        """
        if self.control_qubits:
            raise_error(RuntimeError, "Cannot use `controlled_by` method on gate {} "
                                      "because it is already controlled by {}."
                                      "".format(self, self.control_qubits))
        if self._nqubits is not None:
            raise_error(RuntimeError, "Cannot use controlled_by on a gate that is "
                                      "part of a Circuit or has been called on a "
                                      "state.")
        if qubits:
            self.is_controlled_by = True
            self.control_qubits = qubits
        return self

    def decompose(self, *free) -> List["Gate"]:
        """Decomposes multi-control gates to gates supported by OpenQASM.

        Decompositions are based on `arXiv:9503016 <https://arxiv.org/abs/quant-ph/9503016>`_.

        Args:
            free: Ids of free qubits to use for the gate decomposition.

        Returns:
            List with gates that have the same effect as applying the original gate.
        """
        # FIXME: Implement this method for all gates not supported by OpenQASM.
        # If the method is not implemented this returns a deep copy of the
        # original gate
        return [self.__class__(*self.init_args, **self.init_kwargs)]

    def __call__(self, state, is_density_matrix): # pragma: no cover
        """Acts with the gate on a given state vector:

        Args:
            state: Input state vector.
                The type and shape of this depend on the backend.

        Returns:
            The state vector after the action of the gate.
        """
        # abstract method
        raise_error(NotImplementedError)


class H(Gate):
    """The Hadamard gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(H, self).__init__()
        self.name = "h"
        self.target_qubits = (q,)
        self.init_args = [q]


class X(Gate):
    """The Pauli X gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(X, self).__init__()
        self.name = "x"
        self.target_qubits = (q,)
        self.init_args = [q]

    def controlled_by(self, *q):
        """Fall back to CNOT and Toffoli if there is one or two controls."""
        if len(q) == 1:
            gate = getattr(self.module, "CNOT")(q[0], self.target_qubits[0])
        elif len(q) == 2:
            gate = getattr(self.module, "TOFFOLI")(q[0], q[1], self.target_qubits[0])
        else:
            gate = super(X, self).controlled_by(*q)
        return gate

    def decompose(self, *free: int, use_toffolis: bool = True) -> List[Gate]:
        """Decomposes multi-control ``X`` gate to one-qubit, ``CNOT`` and ``TOFFOLI`` gates.

        Args:
            free: Ids of free qubits to use for the gate decomposition.
            use_toffolis: If ``True`` the decomposition contains only ``TOFFOLI`` gates.
                If ``False`` a congruent representation is used for ``TOFFOLI`` gates.
                See :class:`qibo.base.gates.TOFFOLI` for more details on this representation.

        Returns:
            List with one-qubit, ``CNOT`` and ``TOFFOLI`` gates that have the
            same effect as applying the original multi-control gate.
        """
        if set(free) & set(self.qubits):
            raise_error(ValueError, "Cannot decompose multi-control X gate if free "
                                    "qubits coincide with target or controls.")
        if self._nqubits is not None:
            for q in free:
                if q >= self.nqubits:
                    raise_error(ValueError, "Gate acts on {} qubits but {} was given "
                                            "as free qubit.".format(self.nqubits, q))

        controls = self.control_qubits
        target = self.target_qubits[0]
        m = len(controls)
        if m < 3:
            return [self.__class__(target).controlled_by(*controls)]

        decomp_gates = []
        n = m + 1 + len(free)
        TOFFOLI = self.module.TOFFOLI
        if (n >= 2 * m - 1) and (m >= 3):
            gates1 = [TOFFOLI(controls[m - 2 - i],
                              free[m - 4 - i],
                              free[m - 3 - i]
                              ).congruent(use_toffolis=use_toffolis)
                      for i in range(m - 3)]
            gates2 = TOFFOLI(controls[0], controls[1], free[0]
                             ).congruent(use_toffolis=use_toffolis)
            first_toffoli = TOFFOLI(controls[m - 1], free[m - 3], target)

            decomp_gates.append(first_toffoli)
            for gates in gates1:
                decomp_gates.extend(gates)
            decomp_gates.extend(gates2)
            for gates in gates1[::-1]:
                decomp_gates.extend(gates)

        elif len(free) >= 1:
            m1 = n // 2
            free1 = controls[m1:] + (target,) + tuple(free[1:])
            x1 = self.__class__(free[0]).controlled_by(*controls[:m1])
            part1 = x1.decompose(*free1, use_toffolis=use_toffolis)

            free2 = controls[:m1] + tuple(free[1:])
            controls2 = controls[m1:] + (free[0],)
            x2 = self.__class__(target).controlled_by(*controls2)
            part2 = x2.decompose(*free2, use_toffolis=use_toffolis)

            decomp_gates = [*part1, *part2]

        else: # pragma: no cover
            # impractical case
            raise_error(NotImplementedError, "X decomposition not implemented "
                                             "for zero free qubits.")

        decomp_gates.extend(decomp_gates)
        return decomp_gates


class Y(Gate):
    """The Pauli Y gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(Y, self).__init__()
        self.name = "y"
        self.target_qubits = (q,)
        self.init_args = [q]


class Z(Gate):
    """The Pauli Z gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(Z, self).__init__()
        self.name = "z"
        self.target_qubits = (q,)
        self.init_args = [q]

    def controlled_by(self, *q):
        """Fall back to CZ if there is only one control."""
        if len(q) == 1:
            gate = getattr(self.module, "CZ")(q[0], self.target_qubits[0])
        else:
            gate = super(Z, self).controlled_by(*q)
        return gate


class I(Gate):
    """The identity gate.

    Args:
        *q (int): the qubit id numbers.
    """

    def __init__(self, *q):
        super(I, self).__init__()
        self.name = "identity"
        self.target_qubits = tuple(q)
        self.init_args = q


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
        self.target_qubits = q
        self.register_name = register_name

        self.init_args = q
        self.init_kwargs = {"register_name": register_name}

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
            raise_error(RuntimeError, "Cannot add qubits to a measurement gate that "
                                      "was executed.")
        self.target_qubits += qubits

    def _set_unmeasured_qubits(self):
        if self._nqubits is None:
            raise_error(RuntimeError, "Cannot calculate set of unmeasured "
                                      "qubits if the number of qubits in the "
                                      "circuit is unknown.")
        if self._unmeasured_qubits is not None:
            raise_error(RuntimeError, "Cannot recalculate unmeasured qubits.")
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
        raise_error(NotImplementedError, "Measurement gates cannot be controlled.")

    @property
    def unitary(self):
        raise_error(ValueError, "Measurements cannot be represented as unitary "
                                "matrices.")


class ParametrizedGate(Gate):
    """Base class for parametrized gates.

    Implements the basic functionality of parameter setters and getters.
    """

    def __init__(self):
        super(ParametrizedGate, self).__init__()
        self._theta = None
        self.nparams = 1

    @property
    def parameter(self):
        return self._theta

    def _reprepare(self):
        if self.device_gates:
            for gate in self.device_gates:
                gate.parameter = self.parameter
        else:
            self._prepare()

    @parameter.setter
    def parameter(self, x):
        self._unitary = None
        self._theta = x
        self._reprepare()


class RX(ParametrizedGate):
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
        self.parameter = theta

        self.init_args = [q]
        self.init_kwargs = {"theta": theta}


class RY(ParametrizedGate):
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
        self.parameter = theta

        self.init_args = [q]
        self.init_kwargs = {"theta": theta}


class RZ(ParametrizedGate):
    """Rotation around the Z-axis of the Bloch sphere.

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
        self.parameter = theta

        self.init_args = [q]
        self.init_kwargs = {"theta": theta}


class ZPow(ParametrizedGate):
    """Equivalent to :class:`qibo.base.gates.RZ` with a different global phase.


    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 \\\\
        0 & e^{i \\theta} \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
    """

    def __init__(self, q, theta):
        super(ZPow, self).__init__()
        self.name = "rz"
        self.target_qubits = (q,)
        self.parameter = theta

        self.init_args = [q]
        self.init_kwargs = {"theta": theta}

    def controlled_by(self, *q):
        """Fall back to CZPow if there is only one control."""
        if len(q) == 1:
            gate = getattr(self.module, "CZPow")(q[0], self.target_qubits[0],
                                                 theta=self.parameter)
        else:
            gate = super(ZPow, self).controlled_by(*q)
        return gate


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
        self.init_args = [q0, q1]

    def decompose(self, *free, use_toffolis: bool = True) -> List[Gate]:
        q0, q1 = self.control_qubits[0], self.target_qubits[0]
        return [self.__class__(q0, q1)]


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
        self.init_args = [q0, q1]


class CZPow(ParametrizedGate):
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
        self.parameter = theta

        self.init_args = [q0, q1]
        self.init_kwargs = {"theta": theta}


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
        self.init_args = [q0, q1]


class fSim(ParametrizedGate):
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
        self._phi = None
        self.nparams = 2
        self.parameter = theta, phi

        self.init_args = [q0, q1]
        self.init_kwargs = {"theta": theta, "phi": phi}

    @property
    def parameter(self):
        return self._theta, self._phi

    @parameter.setter
    def parameter(self, x):
        self._unitary = None
        self._theta, self._phi = x
        self._reprepare()


class GeneralizedfSim(ParametrizedGate):
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

        self._phi = None
        self.__unitary = None
        self.nparams = 2
        self.parameter = unitary, phi

        self.init_args = [q0, q1]
        self.init_kwargs = {"unitary": unitary, "phi": phi}

    @property
    def parameter(self):
        return self.__unitary, self._phi

    @parameter.setter
    def parameter(self, x):
        shape = tuple(x[0].shape)
        if shape != (2, 2):
            raise_error(ValueError, "Invalid shape {} of rotation for generalized "
                                    "fSim gate".format(shape))
        self._unitary = None
        self.__unitary, self._phi = x
        self._reprepare()


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
        self.init_args = [q0, q1, q2]

    @property
    def unitary(self):
        if self._unitary is None:
            self._unitary = self.construct_unitary()
        return self._unitary

    def decompose(self, *free, use_toffolis: bool = True) -> List[Gate]:
        c0, c1 = self.control_qubits
        t = self.target_qubits[0]
        return [self.__class__(c0, c1, t)]

    def congruent(self, use_toffolis: bool = True) -> List[Gate]:
        """Congruent representation of ``TOFFOLI`` gate.

        This is a helper method for the decomposition of multi-control ``X`` gates.
        The congruent representation is based on Sec. 6.2 of
        `arXiv:9503016 <https://arxiv.org/abs/quant-ph/9503016>`_.
        The sequence of the gates produced here has the same effect as ``TOFFOLI``
        with the phase of the |101> state reversed.

        Args:
            use_toffolis: If ``True`` a single ``TOFFOLI`` gate is returned.
                If ``False`` the congruent representation is returned.

        Returns:
            List with ``RY`` and ``CNOT`` gates that have the same effect as
            applying the original ``TOFFOLI`` gate.
        """
        if use_toffolis:
            return self.decompose()

        import importlib
        import numpy as np
        control0, control1 = self.control_qubits
        target = self.target_qubits[0]
        RY = self.module.RY
        CNOT = self.module.CNOT
        return [RY(target, -np.pi / 4), CNOT(control1, target),
                RY(target, -np.pi / 4), CNOT(control0, target),
                RY(target, np.pi / 4), CNOT(control1, target),
                RY(target, np.pi / 4)]


class Unitary(ParametrizedGate):
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
        self.target_qubits = tuple(q)

        self.__unitary = None
        self.parameter = unitary
        self.nparams = int(tuple(unitary.shape)[0]) ** 2

        self.init_args = [unitary] + list(q)
        self.init_kwargs = {"name": name}

    @property
    def rank(self) -> int:
        return len(self.target_qubits)

    @property
    def parameter(self):
        return self.__unitary

    @parameter.setter
    def parameter(self, x):
        shape = tuple(x.shape)
        true_shape = (2 ** self.rank, 2 ** self.rank)
        if shape == true_shape:
            self.__unitary = x
        elif shape == (2 ** (2 * self.rank),):
            self.__unitary = x.reshape(true_shape)
        else:
            raise_error(ValueError, "Invalid shape {} of unitary matrix acting on "
                                    "{} target qubits.".format(shape, self.rank))
        self._unitary = None
        self._reprepare()

    @property
    def unitary(self):
        return self._unitary


class VariationalLayer(ParametrizedGate):
    """Layer of one-qubit parametrized gates followed by two-qubit entangling gates.

    Performance is optimized by fusing the variational one-qubit gates with the
    two-qubit entangling gates that follow them and applying a single layer of
    two-qubit gates as 4x4 matrices.

    Args:
        qubits (list): List of one-qubit gate target qubit IDs.
        pairs (list): List of pairs of qubit IDs on which the two qubit gate act.
        one_qubit_gate: Type of one qubit gate to use as the variational gate.
        two_qubit_gate: Type of two qubit gate to use as entangling gate.
        params (list): Variational parameters of one-qubit gates as a list that
            has the same length as ``qubits``. These gates act before the layer
            of entangling gates.
        params2 (list): Variational parameters of one-qubit gates as a list that
            has the same length as ``qubits``. These gates act after the layer
            of entangling gates.
        name (str): Optional name for the gate.
            If ``None`` the name ``"VariationalLayer"`` will be used.

    Example:
        ::

            import numpy as np
            from qibo.models import Circuit
            from qibo import gates
            # generate an array of variational parameters for 8 qubits
            theta = 2 * np.pi * np.random.random(8)

            # define qubit pairs that two qubit gates will act
            pairs = [(i, i + 1) for i in range(0, 7, 2)]
            # map variational parameters to qubit IDs
            theta_map = {i: th for i, th in enumerate(theta}
            # define a circuit of 8 qubits and add the variational layer
            c = Circuit(8)
            c.add(gates.VariationalLayer(pairs, gates.RY, gates.CZ, theta_map))
            # this will create an optimized version of the following circuit
            c2 = Circuit(8)
            c.add((gates.RY(i, th) for i, th in enumerate(theta)))
            c.add((gates.CZ(i, i + 1) for i in range(7)))
    """

    def __init__(self, qubits: List[int], pairs: List[Tuple[int, int]],
                 one_qubit_gate, two_qubit_gate,
                 params: List[float], params2: Optional[List[float]] = None,
                 name: Optional[str] = None):
        super(VariationalLayer, self).__init__()
        self.init_args = [qubits, pairs, one_qubit_gate, two_qubit_gate]
        self.init_kwargs = {"params": params, "params2": params2, "name": name}
        self.name = "VariationalLayer" if name is None else name

        self.target_qubits = tuple(qubits)
        self.params = self._create_params_dict(params)
        self._parameters = list(params)
        if params2 is None:
            self.params2 = {}
        else:
            self.params2 = self._create_params_dict(params2)
            self._parameters.extend(params2)
        self.nparams = len(self.params) + len(self.params2)

        self.pairs = pairs
        targets = set(self.target_qubits)
        two_qubit_targets = set(q for p in pairs for q in p)
        additional_targets = targets - two_qubit_targets
        if not additional_targets:
            self.additional_target = None
        elif len(additional_targets) == 1:
            self.additional_target = additional_targets.pop()
        else:
            raise_error(ValueError, "Variational layer can have at most one additional "
                                    "target for one qubit gates but has {}."
                                    "".format(additional_targets))

        self.one_qubit_gate = one_qubit_gate
        self.two_qubit_gate = two_qubit_gate

        self.unitaries = []
        self.additional_unitary = None

    def _create_params_dict(self, params: List[float]) -> Dict[int, float]:
        if len(self.target_qubits) != len(params):
            raise_error(ValueError, "VariationalLayer has {} target qubits but {} "
                                    "parameters were given."
                                    "".format(len(self.target_qubits), len(params)))
        return {q: p for q, p in zip(self.target_qubits, params)}

    def _calculate_unitaries(self): # pragma: no cover
        # abstract method
        return raise_error(NotImplementedError)

    @property
    def parameter(self) -> List[float]:
        return self._parameters

    @parameter.setter
    def parameter(self, x):
        if self.params2:
            n = len(x) // 2
            self.params = self._create_params_dict(x[:n])
            self.params2 = self._create_params_dict(x[n:])
        else:
            self.params = self._create_params_dict(x)
        self._parameters = x

        matrices, additional_matrix = self._calculate_unitaries()
        for unitary, matrix in zip(self.unitaries, matrices):
            unitary.parameter = matrix
        if additional_matrix is not None:
            self.additional_unitary.parameter = additional_matrix

    @property
    def unitary(self):
        raise_error(ValueError, "Unitary property does not exist for the "
                                "``VariationalLayer``.")


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

        self.init_args = [q]
        self.init_kwargs = {"px": px, "py": py, "pz": pz}

    @property
    def unitary(self): # pragma: no cover
        # future TODO
        raise_error(NotImplementedError, "Unitary property not implemented for "
                                         "channels.")

    def controlled_by(self, *q):
        """"""
        raise_error(ValueError, "Noise channel cannot be controlled on qubits.")


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
        self.init_args = [A]

        # Check that given operators have the proper shape
        for qubits, matrix in A:
            rank = 2 ** len(qubits)
            shape = tuple(matrix.shape)
            if shape != (rank, rank):
                raise_error(ValueError, "Invalid Krauss operator shape {} for "
                                        " acting on {} qubits."
                                        "".format(shape, len(qubits)))

    @property
    def unitary(self): # pragma: no cover
        # future TODO
        raise_error(NotImplementedError, "Unitary property not implemented for "
                                         "channels.")

    def controlled_by(self, *q):
        """"""
        raise_error(ValueError, "Channel cannot be controlled on qubits.")


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
        self.init_args = [coefficients]
        self.is_special_gate = True


class CallbackGate(Gate):
    """Calculates a :class:`qibo.tensorflow.callbacks.Callback` at a specific point in the circuit.

    This gate performs the callback calulation without affecting the state vector.

    Args:
        callback (:class:`qibo.tensorflow.callbacks.Callback`): Callback object to calculate.
    """

    def __init__(self, callback: "Callback"):
        super(CallbackGate, self).__init__()
        self.name = callback.__class__.__name__
        self.callback = callback
        self.init_args = [callback]
        self.is_special_gate = True

    @Gate.nqubits.setter
    def nqubits(self, n: int):
        Gate.nqubits.fset(self, n) # pylint: disable=no-member
        self.callback.nqubits = n
