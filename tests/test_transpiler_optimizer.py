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
    circ = Circuit(7)
    preprocesser = Preprocessing(connectivity=star_connectivity())
    with pytest.raises(ValueError):
        new_circuit = preprocesser(circuit=circ)

    circ = Circuit(5, wire_names=[0, 1, 2, "q3", "q4"])
    with pytest.raises(ValueError):
        new_circuit = preprocesser(circuit=circ)


def test_preprocessing_same(star_connectivity):
    circ = Circuit(5)
    circ.add(gates.CNOT(0, 1))
    preprocesser = Preprocessing(connectivity=star_connectivity())
    new_circuit = preprocesser(circuit=circ)
    assert new_circuit.ngates == 1


def test_preprocessing_add(star_connectivity):
    circ = Circuit(3)
    circ.add(gates.CNOT(0, 1))
    preprocesser = Preprocessing(connectivity=star_connectivity())
    new_circuit = preprocesser(circuit=circ)
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


def test_inversecancellation_no_gates():
    circ = Circuit(1)
    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circ)
    assert new_circuit.ngates == 0
    assert new_circuit.nqubits == 1


def test_inversecancellation_gatesX():
    circ = Circuit(1)
    circ.add(gates.X(0))
    circ.add(gates.X(0))
    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circ)
    assert new_circuit.ngates == 0
    assert new_circuit.nqubits == 1


def test_inversecancellation_gatesY():
    circ = Circuit(1)
    circ.add(gates.Y(0))
    circ.add(gates.Y(0))
    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circ)
    assert new_circuit.ngates == 0
    assert new_circuit.nqubits == 1


def test_inversecancellation_gatesZ():
    circ = Circuit(1)
    circ.add(gates.Z(0))
    circ.add(gates.Z(0))
    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circ)
    assert new_circuit.ngates == 0
    assert new_circuit.nqubits == 1


def test_inversecancellation_gatesH():
    circ = Circuit(1)
    circ.add(gates.H(0))
    circ.add(gates.H(0))
    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circ)
    assert new_circuit.ngates == 0
    assert new_circuit.nqubits == 1


def test_inversecancellation_gatesCNOT():
    circ = Circuit(2)
    circ.add(gates.CNOT(0, 1))
    circ.add(gates.CNOT(0, 1))
    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circ)
    assert new_circuit.ngates == 0
    assert new_circuit.nqubits == 2


def test_inversecancellation_gatesCNOTXq1():
    circ = Circuit(2)
    circ.add(gates.CNOT(0, 1))
    circ.add(gates.X(1))
    circ.add(gates.CNOT(0, 1))
    circ.add(gates.X(1))
    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circ)
    assert new_circuit.ngates == 4
    assert new_circuit.nqubits == 2


def test_inversecancellation_gatesCNOTXq0():
    circ = Circuit(2)
    circ.add(gates.CNOT(0, 1))
    circ.add(gates.X(0))
    circ.add(gates.CNOT(0, 1))
    circ.add(gates.X(0))
    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circ)
    assert new_circuit.ngates == 4
    assert new_circuit.nqubits == 2


def test_inversecancellation_gatesCZ():
    circ = Circuit(2)
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(0, 1))
    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circ)
    assert new_circuit.ngates == 0
    assert new_circuit.nqubits == 2


def test_inversecancellation_gatesCZXq0():
    circ = Circuit(2)
    circ.add(gates.CZ(0, 1))
    circ.add(gates.X(0))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.X(0))
    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circ)
    assert new_circuit.ngates == 4
    assert new_circuit.nqubits == 2


def test_inversecancellation_gatesCZXq1():
    circ = Circuit(2)
    circ.add(gates.CZ(0, 1))
    circ.add(gates.X(1))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.X(1))
    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circ)
    assert new_circuit.ngates == 4
    assert new_circuit.nqubits == 2


def test_inversecancellation_gatesSWAP():
    circ = Circuit(2)
    circ.add(gates.SWAP(0, 1))
    circ.add(gates.SWAP(0, 1))
    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circ)
    assert new_circuit.ngates == 0
    assert new_circuit.nqubits == 2


def test_inversecancellation_gatesSWAPXq0():
    circ = Circuit(2)
    circ.add(gates.SWAP(0, 1))
    circ.add(gates.X(0))
    circ.add(gates.SWAP(0, 1))
    circ.add(gates.X(0))
    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circ)
    assert new_circuit.ngates == 4
    assert new_circuit.nqubits == 2


def test_inversecancellation_gatesSWAPXq1():
    circ = Circuit(2)
    circ.add(gates.SWAP(0, 1))
    circ.add(gates.X(1))
    circ.add(gates.SWAP(0, 1))
    circ.add(gates.X(1))
    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circ)
    assert new_circuit.ngates == 4
    assert new_circuit.nqubits == 2


def test_inversecancellation_gatesVarious():
    circ = Circuit(3)
    circ.add(gates.SWAP(0, 1))
    circ.add(gates.TOFFOLI(0, 1, 2))
    circ.add(gates.SWAP(0, 1))
    circ.add(gates.X(1))
    inverse_cancellation = InverseCancellation()
    new_circuit = inverse_cancellation(circuit=circ)
    assert new_circuit.ngates == 4
    assert new_circuit.nqubits == 3


def test_rotationgatefusion_RXRX():
    circ = Circuit(1)
    circ.add(gates.RX(0, 3.15))
    circ.add(gates.RX(0, 3.15))
    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circ)
    assert new_circuit.ngates == 1
    assert new_circuit.nqubits == 1
    assert new_circuit.get_parameters() == [(6.30,)]


def test_rotationgatefusion_RY():
    circ = Circuit(1)
    circ.add(gates.RY(0, 3.15))
    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circ)
    assert new_circuit.ngates == 1
    assert new_circuit.nqubits == 1
    assert new_circuit.get_parameters() == [(3.15,)]


def test_rotationgatefusion_RYRY():
    circ = Circuit(1)
    circ.add(gates.RY(0, 3.15))
    circ.add(gates.RY(0, 3.15))
    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circ)
    assert new_circuit.ngates == 1
    assert new_circuit.nqubits == 1
    assert new_circuit.get_parameters() == [(6.30,)]


def test_rotationgatefusion_RZRZ():
    circ = Circuit(1)
    circ.add(gates.RZ(0, 3.15))
    circ.add(gates.RZ(0, 3.15))
    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circ)
    assert new_circuit.ngates == 1
    assert new_circuit.nqubits == 1
    assert new_circuit.get_parameters() == [(6.30,)]


def test_rotationgatefusion_RXRY():
    circ = Circuit(1)
    circ.add(gates.RX(0, 3.15))
    circ.add(gates.RY(0, 3.15))
    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circ)
    assert new_circuit.ngates == 2
    assert new_circuit.nqubits == 1
    assert new_circuit.get_parameters() == [(3.15,), (3.15,)]


def test_rotationgatefusion_RXsCNOT():
    circ = Circuit(2)
    circ.add(gates.CNOT(0, 1))
    circ.add(gates.RX(0, 3.15))
    circ.add(gates.CNOT(1, 0))
    circ.add(gates.RX(0, 3.15))
    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circ)
    assert new_circuit.ngates == 4
    assert new_circuit.nqubits == 2
    assert new_circuit.get_parameters() == [(3.15,), (3.15,)]


def test_rotationgatefusion_RXsSWAPs():
    circ = Circuit(2)
    circ.add(gates.SWAP(0, 1))
    circ.add(gates.RX(0, 3.15))
    circ.add(gates.SWAP(0, 1))
    circ.add(gates.RX(0, 3.15))
    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circ)
    assert new_circuit.ngates == 4
    assert new_circuit.nqubits == 2
    assert new_circuit.get_parameters() == [(3.15,), (3.15,)]


def test_rotationgatefusion_RXsTOFFOLI():
    circ = Circuit(3)
    circ.add(gates.RX(0, 3.15))
    circ.add(gates.TOFFOLI(0, 1, 2))
    circ.add(gates.RX(0, 3.15))
    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circ)
    assert new_circuit.ngates == 3
    assert new_circuit.nqubits == 3
    assert new_circuit.get_parameters() == [(3.15,), (3.15,)]


def test_rotationgatefusion_1RXTOFFOLI():
    circ = Circuit(3)
    circ.add(gates.TOFFOLI(0, 1, 2))
    circ.add(gates.RX(0, 3.15))
    circ.add(gates.RX(0, 3.15))
    rotation_gate_fusion = RotationGateFusion()
    new_circuit = rotation_gate_fusion(circuit=circ)
    assert new_circuit.ngates == 2
    assert new_circuit.nqubits == 3
    assert new_circuit.get_parameters() == [(6.30,)]


def test_u3gatefusion_1U3():
    circ = Circuit(1)
    circ.add(gates.U3(0, 0.5, 0.2, 0.3))
    u3_gate_fusion = U3GateFusion()
    new_circuit = u3_gate_fusion(circuit=circ)
    assert new_circuit.ngates == 1
    assert new_circuit.nqubits == 1
    assert new_circuit.get_parameters() == [(0.5, 0.2, 0.3)]


def test_u3gatefusion_2U3():
    circ = Circuit(1)
    circ.add(gates.U3(0, 0.5, 0.2, 0.3))
    circ.add(gates.U3(0, 1.0, 0.4, 0.6))
    u3_gate_fusion = U3GateFusion()
    new_circuit = u3_gate_fusion(circuit=circ)
    assert new_circuit.ngates == 1
    assert new_circuit.nqubits == 1
    assert new_circuit.get_parameters() == [
        (1.3764832408758527, 0.7581212965158095, 0.9626552310874916)
    ]


def test_u3gatefusion_2U3():
    circ = Circuit(1)
    circ.add(gates.U3(0, 0.5, 0.2, 0.3))
    circ.add(gates.U3(0, 1.0, 0.4, 0.6))
    circ.add(gates.U3(0, 1.0, 0.4, 0.6))
    u3_gate_fusion = U3GateFusion()
    new_circuit = u3_gate_fusion(circuit=circ)
    assert new_circuit.ngates == 1
    assert new_circuit.nqubits == 1


def test_u3gatefusion_2U3dif():
    circ = Circuit(2)
    circ.add(gates.U3(0, 0.5, 0.2, 0.3))
    circ.add(gates.U3(1, 0.5, 0.2, 0.3))
    u3_gate_fusion = U3GateFusion()
    new_circuit = u3_gate_fusion(circuit=circ)
    assert new_circuit.ngates == 2
    assert new_circuit.nqubits == 2
    assert new_circuit.get_parameters() == [(0.5, 0.2, 0.3), (0.5, 0.2, 0.3)]


def test_u3gatefusion_2U3diffCNOT():
    circ = Circuit(2)
    circ.add(gates.U3(0, 0.5, 0.2, 0.3))
    circ.add(gates.CNOT(0, 1))
    circ.add(gates.U3(1, 0.5, 0.2, 0.3))
    u3_gate_fusion = U3GateFusion()
    new_circuit = u3_gate_fusion(circuit=circ)
    assert new_circuit.ngates == 3
    assert new_circuit.nqubits == 2
    assert new_circuit.get_parameters() == [(0.5, 0.2, 0.3), (0.5, 0.2, 0.3)]


def test_u3gatefusion_2U3CNOT():
    circ = Circuit(2)
    circ.add(gates.U3(0, 0.5, 0.2, 0.3))
    circ.add(gates.CNOT(0, 1))
    circ.add(gates.U3(0, 0.5, 0.2, 0.3))
    u3_gate_fusion = U3GateFusion()
    new_circuit = u3_gate_fusion(circuit=circ)
    assert new_circuit.ngates == 3
    assert new_circuit.nqubits == 2
    assert new_circuit.get_parameters() == [(0.5, 0.2, 0.3), (0.5, 0.2, 0.3)]


def test_u3gatefusion_U3VariousGates():
    circ = Circuit(2)
    circ.add(gates.RX(0, 0.5))
    circ.add(gates.U3(0, 0.5, 0.2, 0.3))
    circ.add(gates.U3(0, 0.5, 0.2, 0.3))
    circ.add(gates.CNOT(0, 1))
    circ.add(gates.U3(0, 0.5, 0.2, 0.3))
    circ.add(gates.U3(1, 0.5, 0.2, 0.3))
    u3_gate_fusion = U3GateFusion()
    new_circuit = u3_gate_fusion(circuit=circ)
    assert new_circuit.ngates == 5
    assert new_circuit.nqubits == 2
