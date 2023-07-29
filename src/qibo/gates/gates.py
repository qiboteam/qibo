import math
from typing import List

import numpy as np

from qibo.config import PRECISION_TOL, raise_error
from qibo.gates.abstract import Gate, ParametrizedGate


class H(Gate):
    """The Hadamard gate.

    Corresponds to the following unitary matrix

    .. math::
        \\frac{1}{\\sqrt{2}} \\, \\begin{pmatrix}
        1 & 1 \\\\
        1 & -1 \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super().__init__()
        self.name = "h"
        self.draw_label = "H"
        self.target_qubits = (q,)
        self.init_args = [q]
        self.clifford = True
        self.unitary = True

    @property
    def qasm_label(self):
        return "h"


class X(Gate):
    """The Pauli-:math:`X` gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        0 & 1 \\\\
        1 & 0 \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super().__init__()
        self.name = "x"
        self.draw_label = "X"
        self.target_qubits = (q,)
        self.init_args = [q]
        self.clifford = True
        self.unitary = True

    @property
    def qasm_label(self):
        return "x"

    @Gate.check_controls
    def controlled_by(self, *q):
        """Fall back to CNOT and Toffoli if there is one or two controls."""
        if len(q) == 1:
            gate = CNOT(q[0], self.target_qubits[0])
        elif len(q) == 2:
            gate = TOFFOLI(q[0], q[1], self.target_qubits[0])
        else:
            gate = super().controlled_by(*q)
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
            raise_error(
                ValueError,
                "Cannot decompose multi-control X gate if free "
                "qubits coincide with target or controls.",
            )

        controls = self.control_qubits
        target = self.target_qubits[0]
        m = len(controls)
        if m < 3:
            return [self.__class__(target).controlled_by(*controls)]

        decomp_gates = []
        n = m + 1 + len(free)
        if (n >= 2 * m - 1) and (m >= 3):
            gates1 = [
                TOFFOLI(
                    controls[m - 2 - i], free[m - 4 - i], free[m - 3 - i]
                ).congruent(use_toffolis=use_toffolis)
                for i in range(m - 3)
            ]
            gates2 = TOFFOLI(controls[0], controls[1], free[0]).congruent(
                use_toffolis=use_toffolis
            )
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

        else:  # pragma: no cover
            # impractical case
            raise_error(
                NotImplementedError,
                "X decomposition not implemented for zero free qubits.",
            )

        decomp_gates.extend(decomp_gates)
        return decomp_gates

    def basis_rotation(self):
        return H(self.target_qubits[0])


class Y(Gate):
    """The Pauli-:math:`Y` gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        0 & -i \\\\
        i & 0 \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super().__init__()
        self.name = "y"
        self.draw_label = "Y"
        self.target_qubits = (q,)
        self.init_args = [q]
        self.clifford = True
        self.unitary = True

    @property
    def qasm_label(self):
        return "y"

    def basis_rotation(self):
        from qibo import matrices  # pylint: disable=C0415

        matrix = (matrices.Y + matrices.Z) / math.sqrt(2)
        return Unitary(matrix, self.target_qubits[0], trainable=False)


class Z(Gate):
    """The Pauli-:math:`Z` gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 \\\\
        0 & -1 \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super().__init__()
        self.name = "z"
        self.draw_label = "Z"
        self.target_qubits = (q,)
        self.init_args = [q]
        self.clifford = True
        self.unitary = True

    @property
    def qasm_label(self):
        return "z"

    @Gate.check_controls
    def controlled_by(self, *q):
        """Fall back to CZ if there is only one control."""
        if len(q) == 1:
            gate = CZ(q[0], self.target_qubits[0])
        else:
            gate = super().controlled_by(*q)
        return gate

    def basis_rotation(self):
        return None


class SX(Gate):
    """The :math:`\\sqrt{X}` gate.

    Corresponds to the following unitary matrix

    .. math::
        \\frac{1}{2} \\, \\begin{pmatrix}
        1 + i & 1 - i \\\\
        1 - i & 1 + i \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super().__init__()
        self.name = "sx"
        self.draw_label = "SX"
        self.target_qubits = (q,)
        self.init_args = [q]
        self.clifford = True
        self.unitary = True

    @property
    def qasm_label(self):
        return "sx"

    def decompose(self):
        """A global phase difference exists between the definitions of
        :math:`\\sqrt{X}` and :math:`\\text{RX}(\\pi / 2)`, with :math:`\\text{RX}`
        being the :class:`qibo.gates.RX` gate. More precisely,
        :math:`\\sqrt{X} = e^{i \\pi / 4} \\, \\text{RX}(\\pi / 2)`.
        """
        return [RX(self.init_args[0], np.pi / 2, trainable=False)]

    def _dagger(self):
        """"""
        return SXDG(self.init_args[0])


class SXDG(Gate):
    """The conjugate transpose of the :math:`\\sqrt{X}` gate.

    Corresponds to the following unitary matrix

    .. math::
        \\frac{1}{2} \\, \\begin{pmatrix}
        1 - i & 1 + i \\\\
        1 + i & 1 - i \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super().__init__()
        self.name = "sxdg"
        self.draw_label = "SXDG"
        self.target_qubits = (q,)
        self.init_args = [q]
        self.clifford = True
        self.unitary = True

    @property
    def qasm_label(self):
        return "sxdg"

    def decompose(self):
        """A global phase difference exists between the definitions of
        :math:`\\sqrt{X}` and :math:`\\text{RX}(\\pi / 2)`, with :math:`\\text{RX}`
        being the :class:`qibo.gates.RX` gate. More precisely,
        :math:`(\\sqrt{X})^{\\dagger} = e^{-i \\pi / 4} \\, \\text{RX}(-\\pi / 2)`.
        """
        return [RX(self.init_args[0], -np.pi / 2, trainable=False)]

    def _dagger(self):
        """"""
        return SX(self.init_args[0])


class S(Gate):
    """The :math:`S` gate.

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
        self.draw_label = "S"
        self.target_qubits = (q,)
        self.init_args = [q]
        self.clifford = True
        self.unitary = True

    @property
    def qasm_label(self):
        return "s"

    def _dagger(self):
        return SDG(*self.init_args)


class SDG(Gate):
    """The conjugate transpose of the :math:`S` gate.

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
        self.draw_label = "SDG"
        self.target_qubits = (q,)
        self.init_args = [q]
        self.clifford = True
        self.unitary = True

    @property
    def qasm_label(self):
        return "sdg"

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
        self.draw_label = "T"
        self.target_qubits = (q,)
        self.init_args = [q]
        self.unitary = True

    @property
    def qasm_label(self):
        return "t"

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
        self.draw_label = "TDG"
        self.target_qubits = (q,)
        self.init_args = [q]
        self.unitary = True

    @property
    def qasm_label(self):
        return "tdg"

    def _dagger(self):
        return T(*self.init_args)


class I(Gate):
    """The identity gate.

    Args:
        *q (int): the qubit id numbers.
    """

    def __init__(self, *q):
        super().__init__()
        self.name = "id"
        self.draw_label = "I"
        self.target_qubits = tuple(q)
        self.init_args = q
        self.clifford = True
        self.unitary = True

    @property
    def qasm_label(self):
        return "id"


class Align(Gate):
    """Aligns proceeding qubit operations and (optionally) waits ``delay`` amount of time.

    Args:
        *q (int): The qubit ID numbers.
        delay (int, optional): The time (in ns) for which to delay circuit execution on the specified qubits.
            Defaults to ``0`` (zero).
    """

    def __init__(self, *q, delay: int = 0):
        if not isinstance(delay, int):
            raise_error(
                TypeError, f"delay must be type int, but it is type {type(delay)}."
            )
        if delay < 0.0:
            raise_error(ValueError, "Delay must not be negative.")

        super().__init__()
        self.name = "align"
        self.delay = delay
        self.draw_label = f"A({delay})"
        self.init_kwargs = {"delay": delay}
        self.target_qubits = tuple(q)


class _Rn_(ParametrizedGate):
    """Abstract class for defining the RX, RY and RZ rotations.

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.AbstractCircuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q, theta, trainable=True):
        super().__init__(trainable)
        self.name = None
        self._controlled_gate = None
        self.target_qubits = (q,)
        self.unitary = True

        self.parameters = theta
        self.init_args = [q]
        self.init_kwargs = {"theta": theta, "trainable": trainable}

    def _dagger(self) -> "Gate":
        """"""
        return self.__class__(
            self.target_qubits[0], -self.parameters[0]
        )  # pylint: disable=E1130

    @Gate.check_controls
    def controlled_by(self, *q):
        """Fall back to CRn if there is only one control."""
        if len(q) == 1:
            gate = self._controlled_gate(  # pylint: disable=E1102
                q[0], self.target_qubits[0], **self.init_kwargs
            )
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
            :meth:`qibo.models.circuit.AbstractCircuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q, theta, trainable=True):
        super().__init__(q, theta, trainable)
        self.name = "rx"
        self.draw_label = "RX"
        self._controlled_gate = CRX

    @property
    def qasm_label(self):
        return "rx"

    def generator_eigenvalue(self):
        return 0.5


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
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q, theta, trainable=True):
        super().__init__(q, theta, trainable)
        self.name = "ry"
        self.draw_label = "RY"
        self._controlled_gate = CRY

    @property
    def qasm_label(self):
        return "ry"

    def generator_eigenvalue(self):
        return 0.5


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
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q, theta, trainable=True):
        super().__init__(q, theta, trainable)
        self.name = "rz"
        self.draw_label = "RZ"
        self._controlled_gate = CRZ

    @property
    def qasm_label(self):
        return "rz"

    def generator_eigenvalue(self):
        return 0.5


class GPI(ParametrizedGate):
    """The GPI gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        0 & e^{- i \\phi} \\\\
        e^{i \\phi} & 0 \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
        phi (float): phase.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.AbstractCircuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q, phi, trainable=True):
        super().__init__(trainable)
        self.name = "gpi"
        self.draw_label = "GPI"
        self.target_qubits = (q,)
        self.unitary = True

        self.parameter_names = "phi"
        self.parameters = phi
        self.nparams = 1

        self.init_args = [q]
        self.init_kwargs = {"phi": phi, "trainable": trainable}


class GPI2(ParametrizedGate):
    """The GPI2 gate.

    Corresponds to the following unitary matrix

    .. math::
        \\frac{1}{\\sqrt{2}} \\, \\begin{pmatrix}
        1 & -i e^{- i \\phi} \\\\
        -i e^{i \\phi} & 1 \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
        phi (float): phase.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.AbstractCircuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q, phi, trainable=True):
        super().__init__(trainable)
        self.name = "gpi2"
        self.draw_label = "GPI2"
        self.target_qubits = (q,)
        self.unitary = True

        self.parameter_names = "phi"
        self.parameters = phi
        self.nparams = 1

        self.init_args = [q]
        self.init_kwargs = {"phi": phi, "trainable": trainable}

    def _dagger(self) -> "Gate":
        """"""
        return self.__class__(self.target_qubits[0], self.parameters[0] + math.pi)


class _Un_(ParametrizedGate):
    """Abstract class for defining the U1, U2 and U3 gates.

    Args:
        q (int): the qubit id number.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q, trainable=True):
        super().__init__(trainable)
        self.name = None
        self._controlled_gate = None
        self.nparams = 0
        self.target_qubits = (q,)
        self.init_args = [q]
        self.unitary = True

        self.init_kwargs = {"trainable": trainable}

    @Gate.check_controls
    def controlled_by(self, *q):
        """Fall back to CUn if there is only one control."""
        if len(q) == 1:
            gate = self._controlled_gate(  # pylint: disable=E1102
                q[0], self.target_qubits[0], **self.init_kwargs
            )
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
            :meth:`qibo.models.circuit.AbstractCircuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q, theta, trainable=True):
        super().__init__(q, trainable=trainable)
        self.name = "u1"
        self.draw_label = "U1"
        self._controlled_gate = CU1

        self.nparams = 1
        self.parameters = theta
        self.init_kwargs = {"theta": theta, "trainable": trainable}

    @property
    def qasm_label(self):
        return "u1"

    def _dagger(self) -> "Gate":
        theta = -self.parameters[0]
        return self.__class__(self.target_qubits[0], theta)  # pylint: disable=E1130


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
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q, phi, lam, trainable=True):
        super().__init__(q, trainable=trainable)
        self.name = "u2"
        self.draw_label = "U2"
        self._controlled_gate = CU2
        self.nparams = 2
        self._phi, self._lam = None, None
        self.init_kwargs = {"phi": phi, "lam": lam, "trainable": trainable}
        self.parameter_names = ["phi", "lam"]
        self.parameters = phi, lam

    @property
    def qasm_label(self):
        return "u2"

    def _dagger(self) -> "Gate":
        """"""
        phi, lam = self.parameters
        phi, lam = math.pi - lam, -math.pi - phi
        return self.__class__(self.target_qubits[0], phi, lam)


class U3(_Un_):
    """Third general unitary gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        e^{-i(\\phi + \\lambda )/2}\\cos\\left (\\frac{\\theta }{2}\\right ) &
            -e^{-i(\\phi - \\lambda )/2}\\sin\\left (\\frac{\\theta }{2}\\right ) \\\\
        e^{i(\\phi - \\lambda )/2}\\sin\\left (\\frac{\\theta }{2}\\right ) &
            e^{i (\\phi + \\lambda )/2}\\cos\\left (\\frac{\\theta }{2}\\right ) \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
        theta (float): first rotation angle.
        phi (float): second rotation angle.
        lamb (float): third rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q, theta, phi, lam, trainable=True):
        super().__init__(q, trainable=trainable)
        self.name = "u3"
        self.draw_label = "U3"
        self._controlled_gate = CU3
        self.nparams = 3
        self._theta, self._phi, self._lam = None, None, None
        self.init_kwargs = {
            "theta": theta,
            "phi": phi,
            "lam": lam,
            "trainable": trainable,
        }
        self.parameter_names = ["theta", "phi", "lam"]
        self.parameters = theta, phi, lam

    @property
    def qasm_label(self):
        return "u3"

    def _dagger(self) -> "Gate":
        """"""
        theta, lam, phi = tuple(-x for x in self.parameters)  # pylint: disable=E1130
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
        super().__init__()
        self.name = "cx"
        self.draw_label = "X"
        self.control_qubits = (q0,)
        self.target_qubits = (q1,)
        self.init_args = [q0, q1]
        self.clifford = True
        self.unitary = True

    @property
    def qasm_label(self):
        return "cx"

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
        super().__init__()
        self.name = "cz"
        self.draw_label = "Z"
        self.control_qubits = (q0,)
        self.target_qubits = (q1,)
        self.init_args = [q0, q1]
        self.clifford = True
        self.unitary = True

    @property
    def qasm_label(self):
        return "cz"


class CSX(Gate):
    """The Controlled-:math:`\\sqrt{X}` gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & e^{i\\pi/4} & e^{-i\\pi/4} \\\\
        0 & 0 & e^{-i\\pi/4} & e^{i\\pi/4} \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
    """

    def __init__(self, q0, q1):
        super().__init__()
        self.name = "csx"
        self.draw_label = "CSX"
        self.control_qubits = (q0,)
        self.target_qubits = (q1,)
        self.init_args = [q0, q1]
        self.clifford = True
        self.unitary = True

    @property
    def qasm_label(self):
        return "csx"

    def decompose(self, *free, use_toffolis: bool = True) -> List[Gate]:
        """"""
        q0, q1 = self.init_args
        return [H(q1), CU1(q0, q1, np.pi / 2), H(q1)]

    def _dagger(self):
        """"""
        return CSXDG(*self.init_args)


class CSXDG(Gate):
    """The transpose conjugate of the Controlled-:math:`\\sqrt{X}` gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & e^{-i\\pi/4} & e^{i\\pi/4} \\\\
        0 & 0 & e^{i\\pi/4} & e^{-i\\pi/4} \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
    """

    def __init__(self, q0, q1):
        super().__init__()
        self.name = "csxdg"
        self.draw_label = "CSXDG"
        self.control_qubits = (q0,)
        self.target_qubits = (q1,)
        self.init_args = [q0, q1]
        self.clifford = True
        self.unitary = True

    @property
    def qasm_label(self):
        return "csxdg"

    def decompose(self, *free, use_toffolis: bool = True) -> List[Gate]:
        """"""
        q0, q1 = self.init_args
        return [H(q1), CU1(q0, q1, -np.pi / 2), H(q1)]

    def _dagger(self):
        """"""
        return CSX(*self.init_args)


class _CRn_(ParametrizedGate):
    """Abstract method for defining the CRX, CRY and CRZ gates.

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): the rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super().__init__(trainable)
        self.name = None
        self.control_qubits = (q0,)
        self.target_qubits = (q1,)
        self.parameters = theta
        self.unitary = True

        self.init_args = [q0, q1]
        self.init_kwargs = {"theta": theta, "trainable": trainable}

    def _dagger(self) -> "Gate":
        """"""
        q0 = self.control_qubits[0]
        q1 = self.target_qubits[0]
        theta = -self.parameters[0]
        return self.__class__(q0, q1, theta)  # pylint: disable=E1130


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
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super().__init__(q0, q1, theta, trainable)
        self.name = "crx"
        self.draw_label = "RX"

    @property
    def qasm_label(self):
        return "crx"


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
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super().__init__(q0, q1, theta, trainable)
        self.name = "cry"
        self.draw_label = "RY"

    @property
    def qasm_label(self):
        return "cry"


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
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super().__init__(q0, q1, theta, trainable)
        self.name = "crz"
        self.draw_label = "RZ"

    @property
    def qasm_label(self):
        return "crz"


class _CUn_(ParametrizedGate):
    """Abstract method for defining the CU1, CU2 and CU3 gates.

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q0, q1, trainable=True):
        super().__init__(trainable)
        self.name = None
        self.nparams = 0
        self.control_qubits = (q0,)
        self.target_qubits = (q1,)
        self.init_args = [q0, q1]
        self.unitary = True
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
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super().__init__(q0, q1, trainable=trainable)
        self.name = "cu1"
        self.draw_label = "U1"
        self.nparams = 1
        self.parameters = theta
        self.init_kwargs = {"theta": theta, "trainable": trainable}

    @property
    def qasm_label(self):
        return "cu1"

    def _dagger(self) -> "Gate":
        """"""
        q0 = self.control_qubits[0]
        q1 = self.target_qubits[0]
        theta = -self.parameters[0]
        return self.__class__(q0, q1, theta)  # pylint: disable=E1130


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
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q0, q1, phi, lam, trainable=True):
        super().__init__(q0, q1, trainable=trainable)
        self.name = "cu2"
        self.draw_label = "U2"
        self.nparams = 2
        self.init_kwargs = {"phi": phi, "lam": lam, "trainable": trainable}

        self.parameter_names = ["phi", "lam"]
        self.parameters = phi, lam

    def _dagger(self) -> "Gate":
        """"""
        q0 = self.control_qubits[0]
        q1 = self.target_qubits[0]
        phi, lam = self.parameters
        phi, lam = math.pi - lam, -math.pi - phi
        return self.__class__(q0, q1, phi, lam)


class CU3(_CUn_):
    """Controlled third general unitary gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & e^{-i(\\phi + \\lambda )/2}\\cos\\left (\\frac{\\theta }{2}\\right ) &
            -e^{-i(\\phi - \\lambda )/2}\\sin\\left (\\frac{\\theta }{2}\\right ) \\\\
        0 & 0 & e^{i(\\phi - \\lambda )/2}\\sin\\left (\\frac{\\theta }{2}\\right ) &
            e^{i (\\phi + \\lambda )/2}\\cos\\left (\\frac{\\theta }{2}\\right ) \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): first rotation angle.
        phi (float): second rotation angle.
        lamb (float): third rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q0, q1, theta, phi, lam, trainable=True):
        super().__init__(q0, q1, trainable=trainable)
        self.name = "cu3"
        self.draw_label = "U3"
        self.nparams = 3
        self._theta, self._phi, self._lam = None, None, None
        self.init_kwargs = {
            "theta": theta,
            "phi": phi,
            "lam": lam,
            "trainable": trainable,
        }
        self.parameter_names = ["theta", "phi", "lam"]
        self.parameters = theta, phi, lam

    @property
    def qasm_label(self):
        return "cu3"

    def _dagger(self) -> "Gate":
        """"""
        q0 = self.control_qubits[0]
        q1 = self.target_qubits[0]
        theta, lam, phi = tuple(-x for x in self.parameters)  # pylint: disable=E1130
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
        super().__init__()
        self.name = "swap"
        self.draw_label = "x"
        self.target_qubits = (q0, q1)
        self.init_args = [q0, q1]
        self.clifford = True
        self.unitary = True

    @property
    def qasm_label(self):
        return "swap"


class iSWAP(Gate):
    """The iswap gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & i & 0 \\\\
        0 & i & 0 & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the first qubit to be swapped id number.
        q1 (int): the second qubit to be swapped id number.
    """

    def __init__(self, q0, q1):
        super().__init__()
        self.name = "iswap"
        self.draw_label = "i"
        self.target_qubits = (q0, q1)
        self.init_args = [q0, q1]
        self.clifford = True
        self.unitary = True

    @property
    def qasm_label(self):
        return "iswap"


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
        super().__init__()
        self.name = "fswap"
        self.draw_label = "fx"
        self.target_qubits = (q0, q1)
        self.init_args = [q0, q1]
        self.clifford = True
        self.unitary = True

    @property
    def qasm_label(self):
        return "fswap"


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
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    # TODO: Check how this works with QASM.

    def __init__(self, q0, q1, theta, phi, trainable=True):
        super().__init__(trainable)
        self.name = "fsim"
        self.draw_label = "f"
        self.target_qubits = (q0, q1)
        self.unitary = True

        self.parameter_names = ["theta", "phi"]
        self.parameters = theta, phi
        self.nparams = 2

        self.init_args = [q0, q1]
        self.init_kwargs = {"theta": theta, "phi": phi, "trainable": trainable}

    def _dagger(self) -> "Gate":
        """"""
        q0, q1 = self.target_qubits
        params = (-x for x in self.parameters)  # pylint: disable=E1130
        return self.__class__(q0, q1, *params)


class SYC(Gate):
    """The Sycamore gate, defined in the Supplementary Information
    of `Quantum supremacy using a programmable superconducting processor
    <https://www.nature.com/articles/s41586-019-1666-5>`_.

    Corresponding to the following unitary matrix

    .. math::
        \\text{fSim}(\\pi / 2, \\, \\pi / 6) = \\beging{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 0 & -i & 0 \\\\
            0 & -i & 0 & 0 \\\\
            0 & 0 & 0 & e^{-i \\pi / 6} \\\\
        \\end{pmatrix} \\, ,

    where :math:`\\text{fSim}` is the :class:`qibo.gates.fSim` gate.

    Args:
        q0 (int): the first qubit to be swapped id number.
        q1 (int): the second qubit to be swapped id number.
    """

    def __init__(self, q0, q1):
        super().__init__()
        self.name = "syc"
        self.draw_label = "SYC"
        self.target_qubits = (q0, q1)
        self.init_args = [q0, q1]
        self.unitary = True

    def _dagger(self) -> "Gate":
        """"""
        return fSim(*self.target_qubits, -np.pi / 2, -np.pi / 6)


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
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q0, q1, unitary, phi, trainable=True):
        super().__init__(trainable)
        self.name = "generalizedfsim"
        self.draw_label = "gf"
        self.target_qubits = (q0, q1)
        self.unitary = True

        self.parameter_names = ["unitary", "phi"]
        self.parameters = unitary, phi
        self.nparams = 5

        self.init_args = [q0, q1]
        self.init_kwargs = {"unitary": unitary, "phi": phi, "trainable": trainable}

    def _dagger(self):
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
            raise_error(
                ValueError,
                "Invalid rotation shape {} for generalized " "fSim gate".format(shape),
            )
        ParametrizedGate.parameters.fset(self, x)  # pylint: disable=no-member


class _Rnn_(ParametrizedGate):
    """Abstract class for defining the RXX, RYY, RZZ, and RZX rotations.

    Args:
        q0 (int): the first entangled qubit id number.
        q1 (int): the second entangled qubit id number.
        theta (float): the rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.AbstractCircuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super().__init__(trainable)
        self.name = None
        self._controlled_gate = None
        self.target_qubits = (q0, q1)
        self.unitary = True

        self.parameters = theta
        self.init_args = [q0, q1]
        self.init_kwargs = {"theta": theta, "trainable": trainable}

    def _dagger(self) -> "Gate":
        """"""
        q0, q1 = self.target_qubits
        return self.__class__(q0, q1, -self.parameters[0])  # pylint: disable=E1130


class RXX(_Rnn_):
    """Parametric 2-qubit XX interaction, or rotation about XX-axis.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        \\cos \\frac{\\theta }{2} & 0 & 0 & -i\\sin \\frac{\\theta }{2} \\\\
        0 & \\cos \\frac{\\theta }{2} & -i\\sin \\frac{\\theta }{2} & 0 \\\\
        0 & -i\\sin \\frac{\\theta }{2} & \\cos \\frac{\\theta }{2} & 0 \\\\
        -i\\sin \\frac{\\theta }{2} & 0 & 0 & \\cos \\frac{\\theta }{2} \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the first entangled qubit id number.
        q1 (int): the second entangled qubit id number.
        theta (float): the rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.AbstractCircuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super().__init__(q0, q1, theta, trainable)
        self.name = "rxx"
        self.draw_label = "RXX"

    @property
    def qasm_label(self):
        return "rxx"


class RYY(_Rnn_):
    """Parametric 2-qubit YY interaction, or rotation about YY-axis.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        \\cos \\frac{\\theta }{2} & 0 & 0 & i\\sin \\frac{\\theta }{2} \\\\
        0 & \\cos \\frac{\\theta }{2} & -i\\sin \\frac{\\theta }{2} & 0 \\\\
        0 & -i\\sin \\frac{\\theta }{2} & \\cos \\frac{\\theta }{2} & 0 \\\\
        i\\sin \\frac{\\theta }{2} & 0 & 0 & \\cos \\frac{\\theta }{2} \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the first entangled qubit id number.
        q1 (int): the second entangled qubit id number.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super().__init__(q0, q1, theta, trainable)
        self.name = "ryy"
        self.draw_label = "RYY"

    @property
    def qasm_label(self):
        return "ryy"


class RZZ(_Rnn_):
    """Parametric 2-qubit ZZ interaction, or rotation about ZZ-axis.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        e^{-i \\theta / 2} & 0 & 0 & 0 \\\\
        0 & e^{i \\theta / 2} & 0 & 0 \\\\
        0 & 0 & e^{i \\theta / 2} & 0 \\\\
        0 & 0 & 0 & e^{-i \\theta / 2} \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the first entangled qubit id number.
        q1 (int): the second entangled qubit id number.
        theta (float): the rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super().__init__(q0, q1, theta, trainable)
        self.name = "rzz"
        self.draw_label = "RZZ"

    @property
    def qasm_label(self):
        return "rzz"


class RZX(_Rnn_):
    """Parametric 2-qubit ZX interaction, or rotation about ZX-axis.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
            \\text{RX}(\\theta) & 0 \\\\
            0 & \\text{RX}(-\\theta) \\\\
        \\end{pmatrix} =
        \\begin{pmatrix}
            \\cos{\\frac{\\theta}{2}} & -i \\sin{\\frac{\\theta}{2}} & 0 & 0 \\\\
            -i \\sin{\\frac{\\theta}{2}} & \\cos{\\frac{\\theta}{2}} & 0 & 0 \\\\
            0 & 0 & \\cos{\\frac{\\theta}{2}} & i \\sin{\\frac{\\theta}{2}} \\\\
            0 & 0 & i \\sin{\\frac{\\theta}{2}} & \\cos{\\frac{\\theta}{2}} \\\\
        \\end{pmatrix} \\, ,

    where :math:`\\text{RX}` is the :class:`qibo.gates.RX` gate.

    Args:
        q0 (int): the first entangled qubit id number.
        q1 (int): the second entangled qubit id number.
        theta (float): the rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super().__init__(q0, q1, theta, trainable)
        self.name = "rzx"
        self.draw_label = "RZX"

    def decompose(self, *free, use_toffolis: bool = True) -> List[Gate]:
        """"""
        q0, q1 = self.target_qubits
        theta = self.init_kwargs["theta"]
        return [H(q1), CNOT(q0, q1), RZ(q1, theta), CNOT(q0, q1), H(q1)]


class RXY(_Rnn_):
    """Parametric 2-qubit :math:`XX + YY` interaction, or rotation about :math:`XX + YY`-axis.

    Corresponds to the following unitary matrix

    .. math::
        \\exp\\left(-i \\frac{\\theta}{4}(XX + YY)\\right) =
        \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & \\cos{\\frac{\\theta}{2}} & -i \\sin{\\frac{\\theta}{2}} & 0 \\\\
            0 & -i \\sin{\\frac{\\theta}{2}} & \\cos{\\frac{\\theta}{2}} & 0 \\\\
            0 & 0 & 0 & 1 \\\\
        \\end{pmatrix} \\, ,

    Args:
        q0 (int): the first entangled qubit id number.
        q1 (int): the second entangled qubit id number.
        theta (float): the rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super().__init__(q0, q1, theta, trainable)
        self.name = "rxy"
        self.draw_label = "RXY"

    def decompose(self, *free, use_toffolis: bool = True) -> List[Gate]:
        """This decomposition has a global phase difference with respect to the
        original gate due to a phase difference in :math:`\\left(\\sqrt{X}\\right)^{\\dagger}`.
        """
        q0, q1 = self.target_qubits
        theta = self.init_kwargs["theta"]
        return [
            RZ(q1, -np.pi / 2),
            S(q0),
            SX(q1),
            RZ(q1, np.pi / 2),
            CNOT(q1, q0),
            RY(q0, -theta / 2),
            RY(q1, -theta / 2),
            CNOT(q1, q0),
            SDG(q0),
            RZ(q1, -np.pi / 2),
            SX(q1).dagger(),
            RZ(q1, np.pi / 2),
        ]


class MS(ParametrizedGate):
    """The Mølmer–Sørensen (MS) gate is a two-qubit gate native to trapped ions.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        \\cos(\\theta / 2) & 0 & 0 & -i e^{-i( \\phi_0 +  \\phi_1)} \\sin(\\theta / 2) \\\\
        0 & \\cos(\\theta / 2) & -i e^{-i( \\phi_0 -  \\phi_1)} \\sin(\\theta / 2) & 0 \\\\
        0 & -i e^{i( \\phi_0 -  \\phi_1)} \\sin(\\theta / 2) & \\cos(\\theta / 2) & 0 \\\\
        -i e^{i( \\phi_0 +  \\phi_1)} \\sin(\\theta / 2) & 0 & 0 & \\cos(\\theta / 2) \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the first qubit to be swapped id number.
        q1 (int): the second qubit to be swapped id number.
        phi0 (float): first qubit's phase.
        phi1 (float): second qubit's phase
        theta (float, optional): arbitrary angle in the interval
            :math:`0 \\leq \\theta \\leq \\pi /2`.  If :math:`\\theta \\rightarrow \\pi / 2`,
            the fully-entangling MS gate is defined. Defaults to :math:`\\pi / 2`.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
    """

    # TODO: Check how this works with QASM.

    def __init__(self, q0, q1, phi0, phi1, theta: float = math.pi / 2, trainable=True):
        super().__init__(trainable)
        self.name = "ms"
        self.draw_label = "MS"
        self.target_qubits = (q0, q1)
        self.unitary = True

        if theta < 0.0 or theta > math.pi / 2:
            raise_error(
                ValueError,
                f"Theta is defined in the interval 0 <= theta <= pi/2, but it is {theta}.",
            )

        self.parameter_names = ["phi0", "phi1", "theta"]
        self.parameters = phi0, phi1, theta
        self.nparams = 3

        self.init_args = [q0, q1]
        self.init_kwargs = {
            "phi0": phi0,
            "phi1": phi1,
            "theta": theta,
            "trainable": trainable,
        }

    def _dagger(self) -> "Gate":
        """"""
        q0, q1 = self.target_qubits
        phi0, phi1, theta = self.parameters
        return self.__class__(q0, q1, phi0 + math.pi, phi1, theta)


class GIVENS(ParametrizedGate):
    """The Givens gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & \\cos(\\theta) & -\\sin(\\theta) & 0 \\\\
            0 & \\sin(\\theta) & \\cos(\\theta) & 0 \\\\
            0 & 0 & 0 & 1 \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the first qubit id number.
        q1 (int): the second qubit id number.
        theta (float): the rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.AbstractCircuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super().__init__(trainable)
        self.name = "g"
        self.draw_label = "G"
        self.target_qubits = (q0, q1)
        self.unitary = True

        self.parameter_names = "theta"
        self.parameters = theta
        self.nparams = 1

        self.init_args = [q0, q1]
        self.init_kwargs = {"theta": theta, "trainable": trainable}

    def _dagger(self) -> "Gate":
        """"""
        return self.__class__(*self.target_qubits, -self.parameters[0])

    def decompose(self, *free, use_toffolis: bool = True) -> List[Gate]:
        """Decomposition of Givens gate according to
        `ArXiv:2106.13839 <https://arxiv.org/abs/2106.13839>`_.
        """
        q0, q1 = self.target_qubits
        theta = self.init_kwargs["theta"]
        return [
            CNOT(q0, q1),
            RY(q0, theta),
            CNOT(q1, q0),
            RY(q0, -theta),
            CNOT(q1, q0),
            CNOT(q0, q1),
        ]


class RBS(ParametrizedGate):
    """The Reconfigurable Beam Splitter gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & \\cos(\\theta) & \\sin(\\theta) & 0 \\\\
            0 & -\\sin(\\theta) & \\cos(\\theta) & 0 \\\\
            0 & 0 & 0 & 1 \\\\
        \\end{pmatrix}

    Note that, in our implementation, :math:`\\text{RBS}(\\theta) = \\text{Givens}(-\\theta)`,
    where :math:`\\text{Givens}` is the :class:`qibo.gates.GIVENS` gate.
    However, we point out that this definition is not unique.

    Args:
        q0 (int): the first qubit id number.
        q1 (int): the second qubit id number.
        theta (float): the rotation angle.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.AbstractCircuit.set_parameters`.
            Defaults to ``True``.
    """

    def __init__(self, q0, q1, theta, trainable=True):
        super().__init__(trainable)
        self.name = "rbs"
        self.draw_label = "RBS"
        self.target_qubits = (q0, q1)
        self.unitary = True

        self.parameter_names = "theta"
        self.parameters = theta
        self.nparams = 1

        self.init_args = [q0, q1]
        self.init_kwargs = {"theta": theta, "trainable": trainable}

    def _dagger(self) -> "Gate":
        """"""
        return self.__class__(*self.target_qubits, -self.parameters[0])

    def decompose(self, *free, use_toffolis: bool = True) -> List[Gate]:
        """Decomposition of RBS gate according to
        `ArXiv:2109.09685 <https://arxiv.org/abs/2109.09685>`_.
        """
        q0, q1 = self.target_qubits
        theta = self.init_kwargs["theta"]
        return [
            H(q0),
            CNOT(q0, q1),
            H(q1),
            RY(q0, theta),
            RY(q1, -theta),
            H(q1),
            CNOT(q0, q1),
            H(q0),
        ]


class ECR(Gate):
    """THe Echo Cross-Resonance gate.

    Corresponds ot the following matrix

    .. math::
        \\frac{1}{\\sqrt{2}} \\left( X \\, I - Y \\, X \\right) =
        \\frac{1}{\\sqrt{2}} \\, \\begin{pmatrix}
            0 & 0 & 1 & i \\\\
            0 & 0 & i & 1 \\\\
            1 & -i & 0 & 0 \\\\
            -i & 1 & 0 & 0 \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the first qubit id number.
        q1 (int): the second qubit id number.
    """

    def __init__(self, q0, q1):
        super().__init__()
        self.name = "ecr"
        self.draw_label = "ECR"
        self.target_qubits = (q0, q1)
        self.init_args = [q0, q1]
        self.unitary = True

    def decompose(self, *free, use_toffolis: bool = True) -> List[Gate]:
        """"""
        q0, q1 = self.target_qubits
        return [RZX(q0, q1, np.pi / 4), X(q0), RZX(q0, q1, -np.pi / 4)]


class TOFFOLI(Gate):
    """The Toffoli gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\\\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\\\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\\\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\\\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\\\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the first control qubit id number.
        q1 (int): the second control qubit id number.
        q2 (int): the target qubit id number.
    """

    def __init__(self, q0, q1, q2):
        super().__init__()
        self.name = "ccx"
        self.draw_label = "X"
        self.control_qubits = (q0, q1)
        self.target_qubits = (q2,)
        self.init_args = [q0, q1, q2]
        self.clifford = True
        self.unitary = True

    @property
    def qasm_label(self):
        return "ccx"

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

        control0, control1 = self.control_qubits
        target = self.target_qubits[0]
        return [
            RY(target, -math.pi / 4),
            CNOT(control1, target),
            RY(target, -math.pi / 4),
            CNOT(control0, target),
            RY(target, math.pi / 4),
            CNOT(control1, target),
            RY(target, math.pi / 4),
        ]


class DEUTSCH(ParametrizedGate):
    """The Deutsch gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\\\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\\\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\\\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\\\
            0 & 0 & 0 & 0 & 0 & 0 & i \\cos{\\theta} & \\sin{\\theta} \\\\
            0 & 0 & 0 & 0 & 0 & 0 & \\sin{\\theta} & i \\cos{\\theta} \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the first control qubit id number.
        q1 (int): the second control qubit id number.
        q2 (int): the target qubit id number.
    """

    def __init__(self, q0, q1, q2, theta, trainable=True):
        super().__init__(trainable)
        self.name = "deutsch"
        self.draw_label = "DE"
        self.control_qubits = (q0, q1)
        self.target_qubits = (q2,)
        self.unitary = True

        self.parameter_names = "theta"
        self.parameters = theta
        self.nparams = 1

        self.init_args = [q0, q1, q2]
        self.init_kwargs = {"theta": theta, "trainable": trainable}


class Unitary(ParametrizedGate):
    """Arbitrary unitary gate.

    Args:
        unitary: Unitary matrix as a tensor supported by the backend.
        *q (int): Qubit id numbers that the gate acts on.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``True``.
        name (str): Optional name for the gate.
        check_unitary (bool): if ``True``, checks if ``unitary`` is an unitary operator.
            If ``False``, check is not performed and ``unitary`` attribute
            defaults to ``False``. Note that, even when the check is performed,
            there is no enforcement. This allows the user to create
            non-unitary gates. Default is ``True``.
    """

    def __init__(
        self, unitary, *q, trainable=True, name: str = None, check_unitary: bool = True
    ):
        super().__init__(trainable)
        self.name = "Unitary" if name is None else name
        self.draw_label = "U"
        self.target_qubits = tuple(q)

        # TODO: Check that given ``unitary`` has proper shape?
        self.parameter_names = "u"
        self._parameters = (unitary,)
        self.nparams = 4 ** len(self.target_qubits)

        self.init_args = [unitary] + list(q)
        self.init_kwargs = {
            "name": name,
            "check_unitary": check_unitary,
            "trainable": trainable,
        }

        # checking unitarity without invoking any backend
        if check_unitary:
            product = np.transpose(np.conj(unitary)) @ unitary
            sums = all(np.abs(1 - np.sum(product, axis=1)) < PRECISION_TOL)
            diagonal = all(np.abs(1 - np.diag(product)) < PRECISION_TOL)

            self.unitary = True if sums and diagonal else False
            del sums, diagonal, product

    @Gate.parameters.setter
    def parameters(self, x):
        shape = self.parameters[0].shape
        self._parameters = (np.reshape(x, shape),)
        for gate in self.device_gates:  # pragma: no cover
            gate.parameters = x

    def on_qubits(self, qubit_map):
        args = [self.init_args[0]]
        args.extend(qubit_map.get(i) for i in self.target_qubits)
        gate = self.__class__(*args, **self.init_kwargs)
        if self.is_controlled_by:
            controls = (qubit_map.get(i) for i in self.control_qubits)
            gate = gate.controlled_by(*controls)
        gate.parameters = self.parameters
        return gate

    def _dagger(self):
        ud = np.conj(np.transpose(self.parameters[0]))
        return self.__class__(ud, *self.target_qubits, **self.init_kwargs)
