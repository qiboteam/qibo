import networkx as nx
import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.transpiler.optimizer import (
    InverseCancellation,
    Preprocessing,
    Rearrange,
    RotationGateFusion,
    U3GateFusion,
)


def test_preprocessing_error(star_connectivity):
    circuit = Circuit(7)
    preprocesser = Preprocessing(connectivity=star_connectivity())
    with pytest.raises(ValueError):
        new_circuit = preprocesser(circuit=circuit)

    circuit = Circuit(5, wire_names=[0, 1, 2, "q3", "q4"])
    with pytest.raises(ValueError):
        new_circuit = preprocesser(circuit=circuit)


def test_preprocessing_same(star_connectivity):
    circuit = Circuit(5)
    circuit.add(gates.CNOT(0, 1))
    preprocesser = Preprocessing(connectivity=star_connectivity())
    new_circuit = preprocesser(circuit=circuit)
    assert new_circuit.ngates == 1


def test_preprocessing_add(star_connectivity):
    circuit = Circuit(3)
    circuit.add(gates.CNOT(0, 1))
    preprocesser = Preprocessing(connectivity=star_connectivity())
    new_circuit = preprocesser(circuit=circuit)
    assert new_circuit.ngates == 1
    assert new_circuit.nqubits == 5


def test_fusion():
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.Z(0))
    circuit.add(gates.Y(0))
    circuit.add(gates.X(1))
    fusion = Rearrange(max_qubits=1)
    fused_circ = fusion(circuit)
    assert isinstance(fused_circ.queue[0], gates.Unitary)


def test_inversecancellation_no_gates(backend):
    circuit = Circuit(1)
    target = circuit.unitary(backend)

    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_inversecancellation_fusedgates(backend):
    circuit = Circuit(1)
    circuit.add(gates.H(0))
    circuit.add(gates.X(0))
    fused_circuit = circuit.fuse()
    target = fused_circuit.unitary(backend)

    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=fused_circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_inversecancellation_gates_x(backend):
    circuit = Circuit(1)
    circuit.add(gates.X(0))
    circuit.add(gates.X(0))
    target = circuit.unitary(backend)

    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_inversecancellation_gates_y(backend):
    circuit = Circuit(1)
    circuit.add(gates.Y(0))
    circuit.add(gates.Y(0))
    target = circuit.unitary(backend)

    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_inversecancellation_gates_z(backend):
    circuit = Circuit(1)
    circuit.add(gates.Z(0))
    circuit.add(gates.Z(0))
    target = circuit.unitary(backend)

    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_inversecancellation_gates_h(backend):
    circuit = Circuit(1)
    circuit.add(gates.H(0))
    circuit.add(gates.H(0))
    target = circuit.unitary(backend)

    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_inversecancellation_gates_cnot(backend):
    circuit = Circuit(2)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(0, 1))
    target = circuit.unitary(backend)

    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_inversecancellation_gates_cnot_x_q1(backend):
    circuit = Circuit(2)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.X(1))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.X(1))
    target = circuit.unitary(backend)

    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_inversecancellation_gates_cnot_x_q0(backend):
    circuit = Circuit(2)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.X(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.X(0))
    target = circuit.unitary(backend)

    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_inversecancellation_gates_cz(backend):
    circuit = Circuit(2)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.CZ(0, 1))
    target = circuit.unitary(backend)

    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_inversecancellation_gates_cz_x_q0(backend):
    circuit = Circuit(2)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.X(0))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.X(0))
    target = circuit.unitary(backend)

    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_inversecancellation_gates_cz_x_q1(backend):
    circuit = Circuit(2)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.X(1))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.X(1))
    target = circuit.unitary(backend)

    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_inversecancellation_gates_swap(backend):
    circuit = Circuit(2)
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.SWAP(0, 1))
    target = circuit.unitary(backend)

    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_inversecancellation_gates_swap_x_q0(backend):
    circuit = Circuit(2)
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.X(0))
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.X(0))
    target = circuit.unitary(backend)

    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_inversecancellation_gates_swap_x_q1(backend):
    circuit = Circuit(2)
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.X(1))
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.X(1))
    target = circuit.unitary(backend)

    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_inversecancellation_gates_various(backend):
    circuit = Circuit(3)
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.TOFFOLI(0, 1, 2))
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.X(1))
    target = circuit.unitary(backend)

    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_rotationgatefusion_no_gates(backend):
    circuit = Circuit(1)
    target = circuit.unitary(backend)

    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_rotationgatefusion_fusedgates(backend):
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.X(1))
    fused_circuit = circuit.fuse()
    target = fused_circuit.unitary(backend)

    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=fused_circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_rotationgatefusion_rx_rx(backend):
    circuit = Circuit(1)
    circuit.add(gates.RX(0, 3.15))
    circuit.add(gates.RX(0, 3.15))
    target = circuit.unitary(backend)

    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_rotationgatefusion_ry(backend):
    circuit = Circuit(1)
    circuit.add(gates.RY(0, 3.15))
    target = circuit.unitary(backend)

    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_rotationgatefusion_ry_ry(backend):
    circuit = Circuit(1)
    circuit.add(gates.RY(0, 3.15))
    circuit.add(gates.RY(0, 3.15))
    target = circuit.unitary(backend)

    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_rotationgatefusion_rz_rz(backend):
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, 3.15))
    circuit.add(gates.RZ(0, 3.15))
    target = circuit.unitary(backend)

    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_rotationgatefusion_rx_ry(backend):
    circuit = Circuit(1)
    circuit.add(gates.RX(0, 3.15))
    circuit.add(gates.RY(0, 3.15))
    target = circuit.unitary(backend)

    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_rotationgatefusion_rxs_cnot(backend):
    circuit = Circuit(2)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RX(0, 3.15))
    circuit.add(gates.CNOT(1, 0))
    circuit.add(gates.RX(0, 3.15))
    target = circuit.unitary(backend)

    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_rotationgatefusion_rxs_swaps(backend):
    circuit = Circuit(2)
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.RX(0, 3.15))
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.RX(0, 3.15))
    target = circuit.unitary(backend)

    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_rotationgatefusion_rxs_toffoli(backend):
    circuit = Circuit(3)
    circuit.add(gates.RX(0, 3.15))
    circuit.add(gates.TOFFOLI(0, 1, 2))
    circuit.add(gates.RX(0, 3.15))
    target = circuit.unitary(backend)

    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_rotationgatefusion_rx_toffoli(backend):
    circuit = Circuit(3)
    circuit.add(gates.TOFFOLI(0, 1, 2))
    circuit.add(gates.RX(0, 3.15))
    circuit.add(gates.RX(0, 3.15))
    target = circuit.unitary(backend)

    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_u3gatefusion_no_gates(backend):
    circuit = Circuit(1)
    target = circuit.unitary(backend)

    u3_gate_fusion = U3GateFusion()
    new_circuit = u3_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_u3gatefusion_fusedgates(backend):
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.H(1))
    fused_circuit = circuit.fuse()
    target = fused_circuit.unitary(backend)

    u3_gate_fusion = U3GateFusion()
    new_circuit = u3_gate_fusion(circuit=fused_circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_u3gatefusion_one_u3(backend):
    circuit = Circuit(1)
    circuit.add(gates.U3(0, 0.5, 0.2, 0.3))
    target = circuit.unitary(backend)

    u3_gate_fusion = U3GateFusion()
    new_circuit = u3_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_u3gatefusion_two_u3(backend):
    circuit = Circuit(1)
    circuit.add(gates.U3(0, 0.5, 0.2, 0.3))
    circuit.add(gates.U3(0, 1.0, 0.4, 0.6))
    target = circuit.unitary(backend)

    u3_gate_fusion = U3GateFusion()
    new_circuit = u3_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_u3gatefusion_three_u3(backend):
    circuit = Circuit(1)
    circuit.add(gates.U3(0, 0.5, 0.2, 0.3))
    circuit.add(gates.U3(0, 1.0, 0.4, 0.6))
    circuit.add(gates.U3(0, 1.0, 0.4, 0.6))
    target = circuit.unitary(backend)

    u3_gate_fusion = U3GateFusion()
    new_circuit = u3_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_u3gatefusion_two_u3_diff(backend):
    circuit = Circuit(2)
    circuit.add(gates.U3(0, 0.5, 0.2, 0.3))
    circuit.add(gates.U3(1, 0.5, 0.2, 0.3))
    target = circuit.unitary(backend)

    u3_gate_fusion = U3GateFusion()
    new_circuit = u3_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_u3gatefusion_two_u3_diff_cnot(backend):
    circuit = Circuit(2)
    circuit.add(gates.U3(0, 0.5, 0.2, 0.3))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.U3(1, 0.5, 0.2, 0.3))
    target = circuit.unitary(backend)

    u3_gate_fusion = U3GateFusion()
    new_circuit = u3_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_u3gatefusion_two_u3_cnot(backend):
    circuit = Circuit(2)
    circuit.add(gates.U3(0, 0.5, 0.2, 0.3))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.U3(0, 0.5, 0.2, 0.3))
    target = circuit.unitary(backend)

    u3_gate_fusion = U3GateFusion()
    new_circuit = u3_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)


def test_u3gatefusion_u3_various_gates(backend):
    circuit = Circuit(2)
    circuit.add(gates.RX(0, 0.5))
    circuit.add(gates.U3(0, 0.5, 0.2, 0.3))
    circuit.add(gates.U3(0, 0.5, 0.2, 0.3))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.U3(0, 0.5, 0.2, 0.3))
    circuit.add(gates.U3(1, 0.5, 0.2, 0.3))
    target = circuit.unitary(backend)

    u3_gate_fusion = U3GateFusion()
    new_circuit = u3_gate_fusion(circuit=circuit)
    unitary = new_circuit.unitary(backend)

    backend.assert_allclose(unitary, target)
