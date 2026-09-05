"""Decomposition of multi-controlled single-qubit gates.

Implements the techniques from Vale et al. (2023),
`arXiv:2302.06377 <https://arxiv.org/abs/2302.06377>`_.
"""

import math
from typing import List, Sequence, Tuple

import numpy as np

from qibo import gates
from qibo.backends import NumpyBackend
from qibo.gates.abstract import Gate
from qibo.transpiler.unitary_decompositions import u3_decomposition

_SINGLE_QUBIT_GATE_NAMES = frozenset(
    {
        "h",
        "x",
        "y",
        "z",
        "s",
        "sdg",
        "t",
        "tdg",
        "sx",
        "sxdg",
        "rx",
        "ry",
        "rz",
        "u1",
        "u2",
        "u3",
        "gpi2",
        "prx",
        "unitary",
    }
)


def _isclose(value: complex, reference: float = 0.0, tol: float = 1e-12) -> bool:
    return abs(value - reference) < tol


def _is_su2(unitary: np.ndarray, tol: float = 1e-10) -> bool:
    return _isclose(np.linalg.det(unitary), 1.0, tol=tol)


def _controlled_unitary_gates(
    unitary: np.ndarray, control: int, target: int
) -> List[Gate]:
    """Decompose a controlled single-qubit unitary into elementary gates."""
    from qibo.transpiler.decompositions import (  # pylint: disable=C0415
        standard_decompositions,
    )
    from qibo.transpiler.unitary_decompositions import (  # pylint: disable=C0415
        two_qubit_decomposition,
    )

    phase_factor = np.conj(np.linalg.det(unitary) ** (-0.5))
    su2 = unitary / phase_factor
    theta, phi, lam = u3_decomposition(su2, NumpyBackend())
    decomp = list(standard_decompositions(gates.CU3(control, target, theta, phi, lam)))

    alpha = np.angle(phase_factor)
    if not _isclose(alpha):
        phase_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, phase_factor, 0.0],
                [0.0, 0.0, 0.0, phase_factor],
            ],
            dtype=complex,
        )
        decomp.extend(
            two_qubit_decomposition(
                control, target, phase_matrix, backend=NumpyBackend()
            )
        )
    return decomp


def _c1c2(
    unitary: np.ndarray,
    n_qubits: int,
    qubit_map: Sequence[int],
    first: bool = True,
    step: int = 1,
) -> List[Gate]:
    """Recursive building block of the Ldmcu decomposition."""
    from collections import namedtuple

    pairs = namedtuple("pairs", ["control", "target"])
    start = 0 if step == 1 else 1
    reverse = step == 1

    qubit_pairs = [
        pairs(control, target)
        for target in range(n_qubits)
        for control in range(start, target)
    ]
    qubit_pairs.sort(key=lambda pair: pair.control + pair.target, reverse=reverse)

    decomp: List[Gate] = []
    for pair in qubit_pairs:
        exponent = pair.target - pair.control
        if pair.control == 0:
            exponent -= 1
        param = 2**exponent
        signal = -1 if (pair.control == 0 and not first) else 1
        signal *= step

        control = qubit_map[pair.control]
        target = qubit_map[pair.target]

        if pair.target == n_qubits - 1 and first:
            coef = param
            root_param = 1 / abs(coef)
            values, vectors = np.linalg.eig(unitary)
            root_gate = (
                np.power(values[0] + 0j, root_param)
                * vectors[:, [0]]
                @ vectors[:, [0]].conj().T
                + np.power(values[1] + 0j, root_param)
                * vectors[:, [1]]
                @ vectors[:, [1]].conj().T
            )
            if signal < 0:
                root_gate = np.linalg.inv(root_gate)
            decomp.extend(_controlled_unitary_gates(root_gate, control, target))
        else:
            decomp.append(gates.CRX(control, target, signal * math.pi / param))

    return decomp


def _decompose_multi_controlled_u2(
    unitary: np.ndarray,
    controls: Sequence[int],
    target: int,
) -> List[Gate]:
    """Decompose a multi-controlled U(2) gate (Ldmcu algorithm)."""
    qubit_map = list(controls) + [target]
    n_qubits = len(qubit_map)
    decomp: List[Gate] = []
    decomp.extend(_c1c2(unitary, n_qubits, qubit_map, True, 1))
    decomp.extend(_c1c2(unitary, n_qubits, qubit_map, True, -1))
    decomp.extend(_c1c2(unitary, n_qubits - 1, qubit_map, False, 1))
    decomp.extend(_c1c2(unitary, n_qubits - 1, qubit_map, False, -1))
    return decomp


def _single_qubit_matrix(gate: Gate) -> np.ndarray:
    """Return the single-qubit unitary matrix of ``gate``."""
    matrix = gate.matrix(NumpyBackend())
    return np.asarray(matrix, dtype=complex)


def _matrix_to_single_qubit_gates(matrix: np.ndarray, qubit: int) -> List[Gate]:
    """Decompose a single-qubit unitary into ``RZ``, ``SX``, and ``RY`` gates."""
    theta, phi, lam = u3_decomposition(matrix, NumpyBackend())
    return gates.U3(qubit, theta, phi, lam).decompose()


def _toffoli_gate(
    control0: int, control1: int, target: int, cancel: str | None = None
) -> List[Gate]:
    """Congruent Toffoli decomposition with optional gate cancellation."""
    decomp = []
    if cancel != "left":
        decomp.extend(
            [
                gates.RY(target, -math.pi / 4),
                gates.CNOT(control0, target),
                gates.RY(target, -math.pi / 4),
            ]
        )
    decomp.append(gates.CNOT(control1, target))
    if cancel != "right":
        decomp.extend(
            [
                gates.RY(target, math.pi / 4),
                gates.CNOT(control0, target),
                gates.RY(target, math.pi / 4),
            ]
        )
    return decomp


def _toffoli_multi_target(
    control0: int,
    control1: int,
    targets: Sequence[int],
    side: str | None = None,
) -> List[Gate]:
    """Multi-target Toffoli helper from Iten et al."""
    target_list = list(targets)
    decomp: List[Gate] = []

    if side in ("l", None):
        for index in range(len(target_list) - 1):
            decomp.append(
                gates.CNOT(
                    target_list[len(target_list) - index - 2],
                    target_list[len(target_list) - index - 1],
                )
            )
        decomp.extend(_toffoli_gate(control0, control1, target_list[0]))
        if side == "l":
            return decomp

    if side == "r":
        decomp.extend(_toffoli_gate(control0, control1, target_list[0]))
        for index in range(len(target_list) - 1):
            decomp.append(gates.CNOT(target_list[index + 1], target_list[index + 2]))
        return decomp

    for index in range(len(target_list) - 1):
        decomp.append(gates.CNOT(target_list[index + 1], target_list[index + 2]))
    return decomp


def _three_control_x(
    control0: int, control1: int, control2: int, target: int, ancilla: int
) -> List[Gate]:
    """Decompose a three-controlled ``X`` gate using one ancilla qubit."""
    return list(
        gates.X(target)
        .controlled_by(control0, control1, control2)
        .decompose(ancilla, use_toffolis=False)
    )


def _append_action_circuit(
    decomp: List[Gate],
    controls: Sequence[int],
    ancilla: Sequence[int],
    targets: Sequence[int],
    targets_aux: Sequence[int],
    num_controls: int,
    num_ancilla: int,
    side: str,
    relative_phase: bool,
    iteration: int,
) -> None:
    """Append the action part of ``McxVchainDirty``."""
    for index in range(num_controls):
        if index < num_controls - 2:
            if targets_aux[index] not in targets or relative_phase:
                control_1 = controls[num_controls - index - 1]
                control_2 = ancilla[num_ancilla - index - 1]
                cancel = (
                    "left"
                    if relative_phase
                    and targets_aux[index] in targets
                    and iteration == 1
                    else "right"
                )
                decomp.extend(
                    _toffoli_gate(control_1, control_2, targets_aux[index], cancel)
                )
            else:
                decomp.extend(
                    _toffoli_multi_target(
                        controls[num_controls - index - 1],
                        ancilla[num_ancilla - index - 1],
                        targets,
                        side,
                    )
                )
        else:
            decomp.extend(
                _toffoli_gate(
                    controls[num_controls - index - 2],
                    controls[num_controls - index - 1],
                    targets_aux[index],
                )
            )
            break


def _mcx_vchain_dirty(
    qubits: Sequence[int],
    num_controls: int,
    action_only: bool = False,
    relative_phase: bool = False,
) -> List[Gate]:
    """Multi-controlled X using dirty ancillas (Iten et al., Lemma 8)."""
    controls = list(qubits[:num_controls])
    target = qubits[-1]
    num_ancilla = max(num_controls - 2, 0)
    ancilla = list(qubits[num_controls : num_controls + num_ancilla])
    targets = (target,)
    targets_aux = (target,) + tuple(reversed(ancilla))

    if num_controls == 1:
        return [gates.CNOT(controls[0], target)]

    if num_controls == 2:
        return [gates.TOFFOLI(controls[0], controls[1], target)]

    if not relative_phase and num_controls == 3 and len(targets) == 1:
        return _three_control_x(
            controls[0], controls[1], controls[2], target, ancilla[0]
        )

    decomp: List[Gate] = []
    for iteration in range(2):
        side = "l" if iteration == 0 else "r"
        _append_action_circuit(
            decomp,
            controls,
            ancilla,
            targets,
            targets_aux,
            num_controls,
            num_ancilla,
            side,
            relative_phase,
            iteration,
        )

        if iteration == 0:
            for index in range(max(num_ancilla - 1, 0)):
                decomp.extend(
                    _toffoli_gate(
                        controls[2 + index],
                        ancilla[index],
                        ancilla[index + 1],
                        "left",
                    )
                )

        if action_only and iteration == 1:
            decomp.extend(
                _toffoli_multi_target(controls[-1], ancilla[-1], targets, "r")
            )
            break

    return decomp


def _get_x_z(su2: np.ndarray) -> Tuple[complex, complex]:
    """Extract parameters for Lemma 1 of Vale et al. (2023)."""
    is_secondary_diag_real = _isclose(su2[0, 1].imag) and _isclose(su2[1, 0].imag)
    if is_secondary_diag_real:
        return su2[0, 1], su2[1, 1]
    return -su2[0, 1].real, su2[1, 1] - su2[0, 1].imag * 1.0j


def _compute_gate_a(x_value: complex, z_value: complex) -> np.ndarray:
    """Compute the ``A`` operator from Lemma 1 of Vale et al. (2023)."""
    if x_value == 0:
        alpha = (z_value + 0j) ** 0.25
        beta = 0.0
    else:
        alpha_r = np.sqrt((np.sqrt((z_value.real + 1.0) / 2.0) + 1.0) / 2.0)
        alpha_i = z_value.imag / (
            2.0
            * np.sqrt(
                (z_value.real + 1.0) * (np.sqrt((z_value.real + 1.0) / 2.0) + 1.0)
            )
        )
        alpha = alpha_r + 1.0j * alpha_i
        beta = x_value / (
            2.0
            * np.sqrt(
                (z_value.real + 1.0) * (np.sqrt((z_value.real + 1.0) / 2.0) + 1.0)
            )
        )
    return np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]])


def _linear_depth_mcv(
    su2_unitary: np.ndarray,
    controls: Sequence[int],
    target: int,
) -> List[Gate]:
    """Theorem 1 of Vale et al. (2023) for real-diagonal multi-controlled SU(2)."""
    x_value, z_value = _get_x_z(su2_unitary)
    gate_a = _compute_gate_a(x_value, z_value)

    num_ctrl = len(controls)
    k_1 = int(math.ceil(num_ctrl / 2.0))
    k_2 = int(math.floor(num_ctrl / 2.0))

    controls = list(controls)
    decomp: List[Gate] = []

    qubits_1 = controls[:k_1] + controls[k_1 : 2 * k_1 - 2] + [target]
    decomp.extend(_mcx_vchain_dirty(qubits_1, k_1))
    decomp.extend(_matrix_to_single_qubit_gates(gate_a, target))

    qubits_2 = controls[k_1:] + controls[k_1 - k_2 + 2 : k_1] + [target]
    decomp.extend(
        [gate.dagger() for gate in reversed(_mcx_vchain_dirty(qubits_2, k_2))]
    )
    decomp.extend(
        [
            gate.dagger()
            for gate in reversed(_matrix_to_single_qubit_gates(gate_a, target))
        ]
    )

    decomp.extend(_mcx_vchain_dirty(qubits_1, k_1))
    decomp.extend(_matrix_to_single_qubit_gates(gate_a, target))
    decomp.extend(_mcx_vchain_dirty(qubits_2, k_2))
    decomp.extend(
        [
            gate.dagger()
            for gate in reversed(_matrix_to_single_qubit_gates(gate_a, target))
        ]
    )

    return decomp


def _has_real_diagonal(unitary: np.ndarray) -> bool:
    """Return ``True`` if at least one diagonal of ``unitary`` is real-valued."""
    is_main_diag_real = _isclose(unitary[0, 0].imag) and _isclose(unitary[1, 1].imag)
    is_secondary_diag_real = _isclose(unitary[0, 1].imag) and _isclose(
        unitary[1, 0].imag
    )
    return is_main_diag_real or is_secondary_diag_real


def _decompose_multi_controlled_su2_real(
    unitary: np.ndarray,
    controls: Sequence[int],
    target: int,
) -> List[Gate]:
    """Decompose a multi-controlled SU(2) gate with a real-valued diagonal.

    Implements Theorem 2 of Vale et al. (2023).
    """
    decomp: List[Gate] = []
    is_secondary_diag_real = _isclose(unitary[0, 1].imag) and _isclose(
        unitary[1, 0].imag
    )

    if not is_secondary_diag_real:
        decomp.append(gates.H(target))

    decomp.extend(_linear_depth_mcv(unitary, controls, target))

    if not is_secondary_diag_real:
        decomp.append(gates.H(target))

    return decomp


def _decompose_multi_controlled_su2(
    unitary: np.ndarray,
    controls: Sequence[int],
    target: int,
) -> List[Gate]:
    """Decompose a generic multi-controlled SU(2) gate.

    Implements Theorem 1 of Vale et al. (2023).
    """
    is_main_diag_real = _isclose(unitary[0, 0].imag) and _isclose(unitary[1, 1].imag)
    is_secondary_diag_real = _isclose(unitary[0, 1].imag) and _isclose(
        unitary[1, 0].imag
    )

    if is_main_diag_real and is_secondary_diag_real:
        return _decompose_multi_controlled_su2_real(unitary, controls, target)

    if is_main_diag_real or is_secondary_diag_real:
        return _decompose_multi_controlled_su2_real(unitary, controls, target)

    eig_vals, eig_vecs = np.linalg.eig(unitary)
    decomp = _matrix_to_single_qubit_gates(np.linalg.inv(eig_vecs), target)
    decomp.extend(_linear_depth_mcv(np.diag(eig_vals), controls, target))
    decomp.extend(_matrix_to_single_qubit_gates(eig_vecs, target))
    return decomp


def _decompose_single_control(gate: Gate, method: str = "standard") -> List[Gate]:
    """Decompose a singly-controlled single-qubit gate using standard rules."""
    control = gate.control_qubits[0]
    target = gate.target_qubits[0]
    controlled_classes = {
        "rx": gates.CRX,
        "ry": gates.CRY,
        "rz": gates.CRZ,
        "u1": gates.CU1,
        "u2": gates.CU2,
        "u3": gates.CU3,
    }

    if gate.name in controlled_classes:
        controlled = controlled_classes[gate.name](control, target, *gate.parameters)
        from qibo.transpiler.decompositions import (  # pylint: disable=C0415
            standard_decompositions,
        )

        if method == "clifford_plus_t":
            try:
                from qibo.transpiler.decompositions import (  # pylint: disable=C0415
                    clifford_plus_t,
                )

                return clifford_plus_t(controlled)
            except (ImportError, NameError):
                pass
        return standard_decompositions(controlled)

    unitary = _single_qubit_matrix(gate)
    if not _is_su2(unitary):
        return _controlled_unitary_gates(unitary, control, target)

    if _has_real_diagonal(unitary):
        return _decompose_multi_controlled_su2_real(unitary, (control,), target)
    return _decompose_multi_controlled_su2(unitary, (control,), target)


def decompose_multi_controlled_single_qubit(
    gate: Gate,
    *free: int,
    method: str = "standard",
    **kwargs,
) -> List[Gate]:
    """Decompose a multi-controlled single-qubit gate into elementary gates.

    Args:
        gate: Gate with ``is_controlled_by=True`` acting on a single target qubit.
        free: Unused. Kept for API compatibility with :meth:`qibo.gates.Gate.decompose`.
        method: Decomposition method passed to nested decompositions.

    Returns:
        List of elementary gates equivalent to ``gate``.
    """
    del free, kwargs

    if not gate.is_controlled_by or len(gate.target_qubits) != 1:
        return [gate]

    controls = gate.control_qubits
    target = gate.target_qubits[0]
    num_controls = len(controls)

    if num_controls == 0:
        return gate.decompose(method=method)

    if num_controls == 1:
        return _decompose_single_control(gate, method=method)

    unitary = _single_qubit_matrix(gate)

    if not _is_su2(unitary):
        return _decompose_multi_controlled_u2(unitary, controls, target)

    if _has_real_diagonal(unitary):
        return _decompose_multi_controlled_su2_real(unitary, controls, target)
    return _decompose_multi_controlled_su2(unitary, controls, target)


def expand_controlled_single_qubit_gates(
    gate_list: List[Gate],
    *free: int,
    method: str = "standard",
    **kwargs,
) -> List[Gate]:
    """Expand any remaining multi-controlled single-qubit gates in ``gate_list``."""
    expanded = list(gate_list)
    while True:
        updated: List[Gate] = []
        changed = False
        for gate in expanded:
            if (
                gate.is_controlled_by
                and len(gate.target_qubits) == 1
                and gate.name in _SINGLE_QUBIT_GATE_NAMES
                and len(gate.control_qubits) >= 2
            ):
                updated.extend(
                    decompose_multi_controlled_single_qubit(
                        gate, *free, method=method, **kwargs
                    )
                )
                changed = True
            else:
                updated.append(gate)
        expanded = updated
        if not changed:
            break
    return expanded
