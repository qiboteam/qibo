"""
Testing that Tensorflow gates' action agrees with Cirq.
"""
import numpy as np
import cirq
import pytest
from qibo.models import Circuit
from qibo.tensorflow import cgates as custom_gates
from qibo.tensorflow import gates as native_gates

_GATES = [custom_gates, native_gates]
_BACKENDS = [(custom_gates, None), (native_gates, "DefaultEinsum"),
             (native_gates, "MatmulEinsum")]


def random_initial_state(nqubits, dtype=np.complex128):
    """Generates a random normalized state vector."""
    x = np.random.random(2 ** nqubits) + 1j * np.random.random(2 ** nqubits)
    return (x / np.sqrt((np.abs(x) ** 2).sum())).astype(dtype)


def assert_gates_equivalent(qibo_gates, cirq_gates, nqubits, atol=1e-7):
    """Asserts that QIBO and Cirq gates have equivalent action on a random state.

    Args:
        qibo_gates: QIBO gate or list of QIBO gates.
        cirq_gates: List of tuples (cirq gate, target qubit IDs).
        nqubits: Total number of qubits in the circuit.
        atol: Absolute tolerance in state vector comparsion.
    """
    initial_state = random_initial_state(nqubits)

    c = Circuit(nqubits)
    c.add(qibo_gates)
    final_state = c(np.copy(initial_state)).numpy()

    c = cirq.Circuit()
    q = [cirq.LineQubit(i) for i in range(nqubits)]
    # apply identity gates to all qubits so that they become part of the circuit
    c.append([cirq.I(qi) for qi in q])
    for gate, targets in cirq_gates:
        c.append(gate(*[q[i] for i in targets]))
    result = cirq.Simulator().simulate(c, initial_state=np.copy(initial_state))
    target_state = result.final_state

    np.testing.assert_allclose(target_state, final_state, atol=atol)


@pytest.mark.parametrize("gates", _GATES)
@pytest.mark.parametrize("gate_name", ["H", "X", "Y", "Z"])
def test_one_qubit_gates(gates, gate_name):
    """Check simple one-qubit gates."""
    qibo_gate = getattr(gates, gate_name)(0)
    cirq_gate = [(getattr(cirq, gate_name), (0,))]
    assert_gates_equivalent(qibo_gate, cirq_gate, 1)


@pytest.mark.parametrize("gates", _GATES)
@pytest.mark.parametrize("gate_name", ["RX", "RY", "RZ"])
def test_one_qubit_parametrized_gates(gates, gate_name):
    """Check parametrized one-qubit rotations."""
    theta = 0.1234
    qibo_gate = getattr(gates, gate_name)(0, theta)
    cirq_gate = [(getattr(cirq, gate_name.lower())(theta), (0,))]
    assert_gates_equivalent(qibo_gate, cirq_gate, 1)


@pytest.mark.parametrize("gates", _GATES)
@pytest.mark.parametrize("gate_name", ["CNOT", "SWAP"])
def test_two_qubit_gates(gates, gate_name):
    """Check two-qubit gates."""
    # TODO: Add CZ gate when it is merged
    qibo_gate = getattr(gates, gate_name)(0, 1)
    cirq_gate = [(getattr(cirq, gate_name), (0, 1))]
    assert_gates_equivalent(qibo_gate, cirq_gate, 2)


@pytest.mark.parametrize("gates", _GATES)
def test_two_qubit_parametrized_gates(gates):
    """Check ``CZPow`` and ``fSim`` gate."""
    theta = 0.1234
    phi = 0.4321

    qibo_gate = gates.CZPow(0, 1, np.pi * theta)
    cirq_gate = [(cirq.CZPowGate(exponent=theta), (0, 1))]
    assert_gates_equivalent(qibo_gate, cirq_gate, 2)

    qibo_gate = gates.fSim(0, 1, theta, phi)
    cirq_gate = [(cirq.FSimGate(theta=theta, phi=phi), (0, 1))]
    assert_gates_equivalent(qibo_gate, cirq_gate, 2)


@pytest.mark.parametrize("gates", _GATES)
@pytest.mark.parametrize(("gate_name", "nqubits"),
                         [("H", 3), ("Z", 4), ("Y", 5),
                          ("X", 6), ("H", 7)])
def test_one_qubit_gates_controlled_by(gates, gate_name, nqubits):
    """Check one-qubit gates controlled on arbitrary number of qubits."""
    all_qubits = np.arange(nqubits)
    for _ in range(5):
        # Generate control and target qubits randomly
        np.random.shuffle(all_qubits)
        nactive = np.random.randint(2, nqubits)
        activeq = list(all_qubits[:nactive])

        qibo_gate = getattr(gates, gate_name)(activeq[-1]).controlled_by(*activeq[:-1])
        cirq_gate = [(getattr(cirq, gate_name).controlled(nactive - 1), activeq)]
        assert_gates_equivalent(qibo_gate, cirq_gate, nqubits)


@pytest.mark.parametrize("gates", _GATES)
@pytest.mark.parametrize(("gate_name", "nqubits"),
                         [("RZ", 4), ("RY", 5), ("RX", 8)])
def test_one_qubit_rotations_controlled_by(gates, gate_name, nqubits):
    """Check one-qubit rotations controlled on arbitrary number of qubits."""
    all_qubits = np.arange(nqubits)
    for _ in range(5):
        # Generate control and target qubits randomly
        np.random.shuffle(all_qubits)
        nactive = np.random.randint(2, nqubits)
        activeq = list(all_qubits[:nactive])

        theta = np.random.random()
        qibo_gate = getattr(gates, gate_name)(activeq[-1], theta).controlled_by(*activeq[:-1])
        cirq_gate = [(getattr(cirq, gate_name)(theta).controlled(nactive - 1), activeq)]
        assert_gates_equivalent(qibo_gate, cirq_gate, nqubits)


@pytest.mark.parametrize("gates", _GATES)
@pytest.mark.parametrize("nqubits", [4, 5, 6, 7, 8])
def test_two_qubit_gates_controlled_by(gates, nqubits):
    """Check ``SWAP`` and ``fSim`` gates controlled on arbitrary number of qubits."""
    all_qubits = np.arange(nqubits)
    for _ in range(5):
        np.random.shuffle(all_qubits)
        nactive = np.random.randint(3, nqubits)
        activeq = list(all_qubits[:nactive])

        qibo_gate = gates.SWAP(*activeq[-2:]).controlled_by(*activeq[:-2])
        cirq_gate = [(cirq.SWAP.controlled(nactive - 2), activeq)]
        assert_gates_equivalent(qibo_gate, cirq_gate, nqubits)

        theta = np.random.random()
        phi = np.random.random()
        qibo_gate = gates.fSim(*activeq[-2:], theta, phi).controlled_by(*activeq[:-2])
        cirq_gate = [(cirq.FSimGate(theta, phi).controlled(nactive - 2), activeq)]
        assert_gates_equivalent(qibo_gate, cirq_gate, nqubits)
