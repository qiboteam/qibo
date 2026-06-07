import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.transpiler.multi_controlled_su2 import decompose_multi_controlled_single_qubit


def assert_unitaries_equal_up_to_global_phase(reference, decomposed, backend):
    reference_unitary = reference.unitary(backend)
    decomposed_unitary = decomposed.unitary(backend)
    phase = np.angle(np.vdot(reference_unitary.flatten(), decomposed_unitary.flatten()))
    backend.assert_allclose(
        reference_unitary,
        decomposed_unitary * np.exp(-1j * phase),
        atol=1e-6,
    )


def assert_no_multi_controlled_single_qubit_gates(circuit):
    for gate in circuit.queue:
        if gate.is_controlled_by and len(gate.target_qubits) == 1:
            pytest.fail(
                f"Found undecomposed multi-controlled gate {gate.name} "
                f"on qubits {gate.qubits}"
            )


@pytest.mark.parametrize(
    ("gate_factory", "nqubits"),
    [
        (lambda: gates.RY(3, 0.2).controlled_by(0, 1), 4),
        (lambda: gates.RY(4, 0.3).controlled_by(0, 1, 2), 5),
        (lambda: gates.RY(5, 0.15).controlled_by(0, 1, 2, 3), 6),
        (lambda: gates.RX(4, 0.3).controlled_by(0, 1, 2), 5),
        (lambda: gates.H(4).controlled_by(0, 1, 2), 5),
        (lambda: gates.H(3).controlled_by(0, 1), 4),
        (lambda: gates.H(1).controlled_by(0), 2),
        (lambda: gates.U3(4, 0.2, 0.1, 0.3).controlled_by(0, 1, 2), 5),
    ],
)
def test_multi_controlled_single_qubit_decomposition(backend, gate_factory, nqubits):
    gate = gate_factory()
    reference = Circuit(nqubits)
    reference.add(gate)

    decomposed = Circuit(nqubits)
    for decomposed_gate in decompose_multi_controlled_single_qubit(gate):
        decomposed.add(decomposed_gate)

    assert_unitaries_equal_up_to_global_phase(reference, decomposed, backend)
    assert_no_multi_controlled_single_qubit_gates(decomposed)


def test_controlled_givens_issue_example(backend):
    """Regression test for https://github.com/qiboteam/qibo/issues/1843."""
    theta = 0.1
    reference = Circuit(4)
    reference.add(gates.GIVENS(2, 3, theta).controlled_by(0, 1))

    decomposed = reference.decompose()

    assert_unitaries_equal_up_to_global_phase(reference, decomposed, backend)
    assert_no_multi_controlled_single_qubit_gates(decomposed)
    assert all(not gate.is_controlled_by for gate in decomposed.queue)
