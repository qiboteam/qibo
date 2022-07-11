import math
from qibo.config import raise_error
from typing import Dict, List, Optional, Tuple
from qibo.gates.abstract import Gate, ParametrizedGate


QASM_GATES = {"h": "H", "x": "X", "y": "Y", "z": "Z",
              "rx": "RX", "ry": "RY", "rz": "RZ",
              "u1": "U1", "u2": "U2", "u3": "U3",
              "cx": "CNOT", "swap": "SWAP", "fswap": "FSWAP", "cz": "CZ",
              "crx": "CRX", "cry": "CRY", "crz": "CRZ",
              "cu1": "CU1", "cu3": "CU3",
              "ccx": "TOFFOLI", "id": "I", "s": "S",
              "sdg": "SDG", "t": "T", "tdg": "TDG"}
PARAMETRIZED_GATES = {"rx", "ry", "rz", "u1", "u2", "u3",
                      "crx", "cry", "crz", "cu1", "cu3"}


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

    @Gate.check_controls
    def controlled_by(self, *q):
        """Fall back to CNOT and Toffoli if there is one or two controls."""
        if len(q) == 1:
            gate = CNOT(q[0], self.target_qubits[0])
        elif len(q) == 2:
            gate = TOFFOLI(q[0], q[1], self.target_qubits[0])
        else:
            gate = super(X, self).controlled_by(*q)
        return gate

    def decompose(self, *free, use_toffolis=True):
        """Decomposes multi-control ``X`` gate to one-qubit, ``CNOT`` and ``TOFFOLI`` gates.

        Args:
            free: Ids of free qubits to use for the gate decomposition.
            use_toffolis: If ``True`` the decomposition contains only ``TOFFOLI`` gates.
                If ``False`` a congruent representation is used for ``TOFFOLI`` gates.
                See :class:`qibo.gates.TOFFOLI` for more details on this representation.

        Returns:
            List with one-qubit, ``CNOT`` and ``TOFFOLI`` gates that have the
            same effect as applying the original multi-control gate.
        """
        if set(free) & set(self.qubits):
            raise_error(ValueError, "Cannot decompose multi-control X gate if free "
                                    "qubits coincide with target or controls.")

        controls = self.control_qubits
        target = self.target_qubits[0]
        m = len(controls)
        if m < 3:
            return [self.__class__(target).controlled_by(*controls)]

        decomp_gates = []
        n = m + 1 + len(free)
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

    @Gate.check_controls
    def controlled_by(self, *q):
        """Fall back to CZ if there is only one control."""
        if len(q) == 1:
            gate = CZ(q[0], self.target_qubits[0])
        else:
            gate = super(Z, self).controlled_by(*q)
        return gate


class S(Gate):
    """The S gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 \\\\
        0 & i \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super().__init__()
        self.name = "s"
        self.target_qubits = (q,)
        self.init_args = [q]

    def _dagger(self):
        return SDG(*self.init_args)


class SDG(Gate):
    """The conjugate transpose of the S gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 \\\\
        0 & -i \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super().__init__()
        self.name = "sdg"
        self.target_qubits = (q,)
        self.init_args = [q]

    def _dagger(self):
        return S(*self.init_args)


class T(Gate):
    """The T gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 \\\\
        0 & e^{i \\pi / 4} \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super().__init__()
        self.name = "t"
        self.target_qubits = (q,)
        self.init_args = [q]

    def _dagger(self):
        return TDG(*self.init_args)


class TDG(Gate):
    """The conjugate transpose of the T gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 \\\\
        0 & e^{-i \\pi / 4} \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super().__init__()
        self.name = "tdg"
        self.target_qubits = (q,)
        self.init_args = [q]

    def _dagger(self):
        return T(*self.init_args)


class I(ParametrizedGate):
    """The identity gate.

    Args:
        *q (int): the qubit id numbers.
    """

    def __init__(self, *q):
        super(I, self).__init__()
        self.name = "id"
        self.target_qubits = tuple(q)
        self.init_args = q
        # save the number of target qubits as parameter
        # for proper identity matrix construction
        self.parameters = 2 ** len(self.target_qubits)


class Align(ParametrizedGate):

    def __init__(self, *q):
        super(Align, self).__init__()
        self.name = "align"
        self.target_qubits = tuple(q)
        self.init_args = q
        # save the number of target qubits as parameter
        # for proper identity matrix construction
        self.parameters = 2 ** len(self.target_qubits)


class _Rn_(ParametrizedGate):
    """Abstract class for defining the RX, RY and RZ rotations.

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.AbstractCircuit.set_parameters`
            (default is ``True``).
    """

    def __init__(self, q, theta, trainable=True):
        super().__init__(trainable)
        self.name = None
        self._controlled_gate = None
        self.target_qubits = (q,)

        self.parameters = theta
        self.init_args = [q]
        self.init_kwargs = {"theta": theta, "trainable": trainable}

    def _dagger(self) -> "Gate":
        """"""
        return self.__class__(self.target_qubits[0], -self.parameters[0]) # pylint: disable=E1130

    @Gate.check_controls
    def controlled_by(self, *q):
        """Fall back to CRn if there is only one control."""
        if len(q) == 1:
            gate = self._controlled_gate( # pylint: disable=E1102
                q[0], self.target_qubits[0], **self.init_kwargs)
        else:
            gate = super().controlled_by(*q)
        return gate


class RX(_Rn_):
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
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.AbstractCircuit.set_parameters`
            (default is ``True``).
    """

    def __init__(self, q, theta, trainable=True):
        super().__init__(q, theta, trainable)
        self.name = "rx"
        self._controlled_gate = CRX


class RY(_Rn_):
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
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`
            (default is ``True``).
    """

    def __init__(self, q, theta, trainable=True):
        super().__init__(q, theta, trainable)
        self.name = "ry"
        self._controlled_gate = CRY


class RZ(_Rn_):
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
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`
            (default is ``True``).
    """

    def __init__(self, q, theta, trainable=True):
        super().__init__(q, theta, trainable)
        self.name = "rz"
        self._controlled_gate = CRZ


class _Un_(ParametrizedGate):
    """Abstract class for defining the U1, U2 and U3 gates.

    Args:
        q (int): the qubit id number.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`
            (default is ``True``).
    """

    def __init__(self, q, trainable=True):
        super().__init__(trainable)
        self.name = None
        self._controlled_gate = None
        self.nparams = 0
        self.target_qubits = (q,)
        self.init_args = [q]
        self.init_kwargs = {"trainable": trainable}

    @Gate.check_controls
    def controlled_by(self, *q):
        """Fall back to CUn if there is only one control."""
        if len(q) == 1:
            gate = self._controlled_gate( # pylint: disable=E1102
              q[0], self.target_qubits[0], **self.init_kwargs)
        else:
            gate = super().controlled_by(*q)
        return gate


class U1(_Un_):
    """First general unitary gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 \\\\
        0 & e^{i \\theta} \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.AbstractCircuit.set_parameters`
            (default is ``True``).
    """

    def __init__(self, q, theta, trainable=True):
        super().__init__(q, trainable=trainable)
        self.name = "u1"
        self._controlled_gate = CU1
        self.nparams = 1
        self.parameters = theta
        self.init_kwargs = {"theta": theta, "trainable": trainable}

    def _dagger(self) -> "Gate":
        theta = - self.parameters[0]
        return self.__class__(self.target_qubits[0], theta) # pylint: disable=E1130


class U2(_Un_):
    """Second general unitary gate.

    Corresponds to the following unitary matrix

    .. math::
        \\frac{1}{\\sqrt{2}}
        \\begin{pmatrix}
        e^{-i(\\phi + \\lambda )/2} & -e^{-i(\\phi - \\lambda )/2} \\\\
        e^{i(\\phi - \\lambda )/2} & e^{i (\\phi + \\lambda )/2} \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
        phi (float): first rotation angle.
        lamb (float): second rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`
            (default is ``True``).
    """

    def __init__(self, q, phi, lam, trainable=True):
        super().__init__(q, trainable=trainable)
        self.name = "u2"
        self._controlled_gate = CU2
        self.nparams = 2
        self._phi, self._lam = None, None
        self.init_kwargs = {"phi": phi, "lam": lam, "trainable": trainable}
        self.parameter_names = ["phi", "lam"]
        self.parameters = phi, lam

    def _dagger(self) -> "Gate":
        """"""
        phi, lam = self.parameters
        phi, lam = math.pi - lam, - math.pi - phi
        return self.__class__(self.target_qubits[0], phi, lam)


class U3(_Un_):
    """Third general unitary gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        e^{-i(\\phi + \\lambda )/2}\\cos\\left (\\frac{\\theta }{2}\\right ) & -e^{-i(\\phi - \\lambda )/2}\\sin\\left (\\frac{\\theta }{2}\\right ) \\\\
        e^{i(\\phi - \\lambda )/2}\\sin\\left (\\frac{\\theta }{2}\\right ) & e^{i (\\phi + \\lambda )/2}\\cos\\left (\\frac{\\theta }{2}\\right ) \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
        theta (float): first rotation angle.
        phi (float): second rotation angle.
        lamb (float): third rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`
            (default is ``True``).
    """

    def __init__(self, q, theta, phi, lam, trainable=True):
        super().__init__(q, trainable=trainable)
        self.name = "u3"
        self._controlled_gate = CU3
        self.nparams = 3
        self._theta, self._phi, self._lam = None, None, None
        self.init_kwargs = {"theta": theta, "phi": phi, "lam": lam,
                            "trainable": trainable}
        self.parameter_names = ["theta", "phi", "lam"]
        self.parameters = theta, phi, lam

    def _dagger(self) -> "Gate":
        """"""
        theta, lam, phi = tuple(-x for x in self.parameters) # pylint: disable=E1130
        return self.__class__(self.target_qubits[0], theta, phi, lam)


class CNOT(Gate):
    """The Controlled-NOT gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        0 & 0 & 1 & 0 \\\\
        \\end{pmatrix}

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


class _CRn_(ParametrizedGate):
    """Abstract method for defining the CRX, CRY and CRZ gates.

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): the rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`
            (default is ``True``).
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super(_CRn_, self).__init__(trainable)
        self.name = None
        self.control_qubits = (q0,)
        self.target_qubits = (q1,)
        self.parameters = theta

        self.init_args = [q0, q1]
        self.init_kwargs = {"theta": theta, "trainable": trainable}

    def _dagger(self) -> "Gate":
        """"""
        q0 = self.control_qubits[0]
        q1 = self.target_qubits[0]
        theta = - self.parameters[0]
        return self.__class__(q0, q1, theta) # pylint: disable=E1130


class CRX(_CRn_):
    """Controlled rotation around the X-axis for the Bloch sphere.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & \\cos \\frac{\\theta }{2}  & -i\\sin \\frac{\\theta }{2} \\\\
        0 & 0 & -i\\sin \\frac{\\theta }{2}  & \\cos \\frac{\\theta }{2} \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): the rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`
            (default is ``True``).
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super().__init__(q0, q1, theta, trainable)
        self.name = "crx"


class CRY(_CRn_):
    """Controlled rotation around the Y-axis for the Bloch sphere.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & \\cos \\frac{\\theta }{2}  & -\\sin \\frac{\\theta }{2} \\\\
        0 & 0 & \\sin \\frac{\\theta }{2}  & \\cos \\frac{\\theta }{2} \\\\
        \\end{pmatrix}

    Note that this differs from the :class:`qibo.gates.RZ` gate.

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): the rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`
            (default is ``True``).
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super().__init__(q0, q1, theta, trainable)
        self.name = "cry"


class CRZ(_CRn_):
    """Controlled rotation around the Z-axis for the Bloch sphere.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & e^{-i \\theta / 2} & 0 \\\\
        0 & 0 & 0 & e^{i \\theta / 2} \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): the rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`
            (default is ``True``).
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super().__init__(q0, q1, theta, trainable)
        self.name = "crz"


class _CUn_(ParametrizedGate):
    """Abstract method for defining the CU1, CU2 and CU3 gates.

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`
            (default is ``True``).
    """

    def __init__(self, q0, q1, trainable=True):
        super(_CUn_, self).__init__(trainable)
        self.name = None
        self.nparams = 0
        self.control_qubits = (q0,)
        self.target_qubits = (q1,)
        self.init_args = [q0, q1]
        self.init_kwargs = {"trainable": trainable}


class CU1(_CUn_):
    """Controlled first general unitary gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & e^{i \\theta } \\\\
        \\end{pmatrix}

    Note that this differs from the :class:`qibo.gates.CRZ` gate.

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): the rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`
            (default is ``True``).
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super().__init__(q0, q1, trainable=trainable)
        self.name = "cu1"
        self.nparams = 1
        self.parameters = theta
        self.init_kwargs = {"theta": theta, "trainable": trainable}

    def _dagger(self) -> "Gate":
        """"""
        q0 = self.control_qubits[0]
        q1 = self.target_qubits[0]
        theta = - self.parameters[0]
        return self.__class__(q0, q1, theta) # pylint: disable=E1130


class CU2(_CUn_):
    """Controlled second general unitary gate.

    Corresponds to the following unitary matrix

    .. math::
        \\frac{1}{\\sqrt{2}}
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & e^{-i(\\phi + \\lambda )/2} & -e^{-i(\\phi - \\lambda )/2} \\\\
        0 & 0 & e^{i(\\phi - \\lambda )/2} & e^{i (\\phi + \\lambda )/2} \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        phi (float): first rotation angle.
        lamb (float): second rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`
            (default is ``True``).
    """

    def __init__(self, q0, q1, phi, lam, trainable=True):
        super().__init__(q0, q1, trainable=trainable)
        self.name = "cu2"
        self.nparams = 2
        self.init_kwargs = {"phi": phi, "lam": lam, "trainable": trainable}

        self.parameter_names = ["phi", "lam"]
        self.parameters = phi, lam

    def _dagger(self) -> "Gate":
        """"""
        q0 = self.control_qubits[0]
        q1 = self.target_qubits[0]
        phi, lam = self.parameters
        phi, lam = math.pi - lam, - math.pi - phi
        return self.__class__(q0, q1, phi, lam)


class CU3(_CUn_):
    """Controlled third general unitary gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & e^{-i(\\phi + \\lambda )/2}\\cos\\left (\\frac{\\theta }{2}\\right ) & -e^{-i(\\phi - \\lambda )/2}\\sin\\left (\\frac{\\theta }{2}\\right ) \\\\
        0 & 0 & e^{i(\\phi - \\lambda )/2}\\sin\\left (\\frac{\\theta }{2}\\right ) & e^{i (\\phi + \\lambda )/2}\\cos\\left (\\frac{\\theta }{2}\\right ) \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): first rotation angle.
        phi (float): second rotation angle.
        lamb (float): third rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`
            (default is ``True``).
    """

    def __init__(self, q0, q1, theta, phi, lam, trainable=True):
        super(CU3, self).__init__(q0, q1, trainable=trainable)
        self.name = "cu3"
        self.nparams = 3
        self._theta, self._phi, self._lam = None, None, None
        self.init_kwargs = {"theta": theta, "phi": phi, "lam": lam,
                            "trainable": trainable}
        self.parameter_names = ["theta", "phi", "lam"]
        self.parameters = theta, phi, lam

    def _dagger(self) -> "Gate":
        """"""
        q0 = self.control_qubits[0]
        q1 = self.target_qubits[0]
        theta, lam, phi = tuple(-x for x in self.parameters) # pylint: disable=E1130
        return self.__class__(q0, q1, theta, phi, lam)


class SWAP(Gate):
    """The swap gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the first qubit to be swapped id number.
        q1 (int): the second qubit to be swapped id number.
    """

    def __init__(self, q0, q1):
        super(SWAP, self).__init__()
        self.name = "swap"
        self.target_qubits = (q0, q1)
        self.init_args = [q0, q1]


class FSWAP(Gate):
    """The fermionic swap gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & -1 \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the first qubit to be f-swapped id number.
        q1 (int): the second qubit to be f-swapped id number.
    """

    def __init__(self, q0, q1):
        super(FSWAP, self).__init__()
        self.name = "fswap"
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
        phi (float): Angle for the ``|11>`` phase.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`
            (default is ``True``).
    """
    # TODO: Check how this works with QASM.

    def __init__(self, q0, q1, theta, phi, trainable=True):
        super(fSim, self).__init__(trainable)
        self.name = "fsim"
        self.target_qubits = (q0, q1)

        self.parameter_names = ["theta", "phi"]
        self.parameters = theta, phi
        self.nparams = 2

        self.init_args = [q0, q1]
        self.init_kwargs = {"theta": theta, "phi": phi, "trainable": trainable}

    def _dagger(self) -> "Gate":
        """"""
        q0, q1 = self.target_qubits
        params = (-x for x in self.parameters) # pylint: disable=E1130
        return self.__class__(q0, q1, *params)


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
        phi (float): Angle for the ``|11>`` phase.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`
            (default is ``True``).
    """

    def __init__(self, q0, q1, unitary, phi, trainable=True):
        super(GeneralizedfSim, self).__init__(trainable)
        self.name = "generalizedfsim"
        self.target_qubits = (q0, q1)

        self.parameter_names = ["u", "phi"]
        self.parameters = unitary, phi
        self.nparams = 5

        self.init_args = [q0, q1]
        self.init_kwargs = {"unitary": unitary, "phi": phi,
                            "trainable": trainable}

    def _dagger(self):
        import numpy as np
        q0, q1 = self.target_qubits
        u, phi = self.parameters
        init_kwargs = dict(self.init_kwargs)
        init_kwargs["unitary"] = np.conj(np.transpose(u))
        init_kwargs["phi"] = -phi
        return self.__class__(q0, q1, **init_kwargs)

    @Gate.parameters.setter
    def parameters(self, x):
        shape = tuple(x[0].shape)
        if shape != (2, 2):
            raise_error(ValueError, "Invalid rotation shape {} for generalized "
                                    "fSim gate".format(shape))
        ParametrizedGate.parameters.fset(self, x) # pylint: disable=no-member


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
        with the phase of the ``|101>`` state reversed.

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
        control0, control1 = self.control_qubits
        target = self.target_qubits[0]
        return [RY(target, -math.pi / 4), CNOT(control1, target),
                RY(target, -math.pi / 4), CNOT(control0, target),
                RY(target, math.pi / 4), CNOT(control1, target),
                RY(target, math.pi / 4)]


class Unitary(ParametrizedGate):
    """Arbitrary unitary gate.

    Args:
        unitary: Unitary matrix as a tensor supported by the backend.
            Note that there is no check that the matrix passed is actually
            unitary. This allows the user to create non-unitary gates.
        *q (int): Qubit id numbers that the gate acts on.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`
            (default is ``True``).
        name (str): Optional name for the gate.
    """

    def __init__(self, unitary, *q, trainable=True, name=None):
        super(Unitary, self).__init__(trainable)
        self.name = "Unitary" if name is None else name
        self.target_qubits = tuple(q)

        # TODO: Check that given ``unitary`` has proper shape?
        self.parameter_names = "u"
        self._parameters = (unitary,)
        self.nparams = 4 ** len(self.target_qubits)

        self.init_args = [unitary] + list(q)
        self.init_kwargs = {"name": name, "trainable": trainable}

    @Gate.parameters.setter
    def parameters(self, x):
        import numpy as np
        shape = self.parameters[0].shape
        self._parameters = (np.reshape(x, shape),)
        for gate in self.device_gates:  # pragma: no cover
            gate.parameters = x

    def on_qubits(self, qubit_map):
        args = [self.init_args[0]]
        args.extend((qubit_map.get(i) for i in self.target_qubits))
        gate = self.__class__(*args, **self.init_kwargs)
        if self.is_controlled_by:
            controls = (qubit_map.get(i) for i in self.control_qubits)
            gate = gate.controlled_by(*controls)
        return gate

    def _dagger(self):
        import numpy as np
        ud = np.conj(np.transpose(self.parameters[0]))
        return self.__class__(ud, *self.target_qubits, **self.init_kwargs)
