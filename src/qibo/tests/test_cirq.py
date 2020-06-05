"""
Testing that Tensorflow gates' action agrees with Cirq.
"""
import numpy as np
import cirq
import pytest
from qibo.models import Circuit
from qibo.tensorflow import cgates as custom_gates
from qibo.tensorflow import gates as native_gates

_ATOL = 1e-6
_GATES = [custom_gates, native_gates]
_BACKENDS = [(custom_gates, None), (native_gates, "DefaultEinsum"),
             (native_gates, "MatmulEinsum")]


def random_initial_state(nqubits, dtype=np.complex128):
    x = np.random.random(2 ** nqubits) + 1j * np.random.random(2 ** nqubits)
    return (x / np.sqrt((np.abs(x) ** 2).sum())).astype(dtype)


class CirqCircuit:

    def __init__(self, nqubits):
        self.program = cirq.Circuit()
        self.qubits = [cirq.LineQubit(i) for i in range(nqubits)]
        self.simulator = cirq.Simulator()

    def add(self, gate, qubit_ids):
        targets = [self.qubits[i] for i in qubit_ids]
        self.program.append(gate(*targets))

    def __call__(self, initial_state):
        result = self.simulator.simulate(self.program,
                                         initial_state=initial_state)
        return result.final_state


def assert_gates_equivalent(qibo_gates, cirq_gates, nqubits, atol=_ATOL):
    initial_state = random_initial_state(nqubits)

    c = Circuit(nqubits)
    c.add(qibo_gates)
    final_state = c(np.copy(initial_state)).numpy()

    c = CirqCircuit(nqubits)
    for gate, targets in cirq_gates:
        c.add(gate, targets)
    target_state = c(np.copy(initial_state))

    np.testing.assert_allclose(target_state, final_state, atol=atol)


@pytest.mark.parametrize("gates", _GATES)
@pytest.mark.parametrize("gate_name", ["H", "X", "Y", "Z"])
def test_one_qubit_gates(gates, gate_name):
    qibo_gate = getattr(gates, gate_name)(0)
    cirq_gate = [(getattr(cirq, gate_name), (0,))]
    assert_gates_equivalent(qibo_gate, cirq_gate, 1)


@pytest.mark.parametrize("gates", _GATES)
@pytest.mark.parametrize("gate_name", ["RX", "RY", "RZ"])
def test_one_qubit_parametrized_gates(gates, gate_name):
    theta = 0.1234
    qibo_gate = getattr(gates, gate_name)(0, theta)
    cirq_gate = [(getattr(cirq, gate_name.lower())(theta), (0,))]
    assert_gates_equivalent(qibo_gate, cirq_gate, 1)


@pytest.mark.parametrize("gates", _GATES)
@pytest.mark.parametrize("gate_name", ["CNOT", "SWAP"])
def test_two_qubit_gates(gates, gate_name):
    qibo_gate = getattr(gates, gate_name)(0, 1)
    cirq_gate = [(getattr(cirq, gate_name), (0, 1))]
    assert_gates_equivalent(qibo_gate, cirq_gate, 2)


@pytest.mark.parametrize("gates", _GATES)
def test_two_qubit_parametrized_gates(gates):
    theta = 0.1234
    phi = 0.4321

    qibo_gate = gates.CZPow(0, 1, np.pi * theta)
    cirq_gate = [(cirq.CZPowGate(exponent=theta), (0, 1))]
    assert_gates_equivalent(qibo_gate, cirq_gate, 2)

    qibo_gate = gates.fSim(0, 1, theta, phi)
    cirq_gate = [(cirq.FSimGate(theta=theta, phi=phi), (0, 1))]
    assert_gates_equivalent(qibo_gate, cirq_gate, 2)
