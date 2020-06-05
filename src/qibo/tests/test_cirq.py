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


def random_unitary_matrix(nqubits, dtype=np.complex128):
    """Generates a random unitary matrix of shape (2^nqubits, 2^nqubits)."""
    from scipy.linalg import expm
    shape = 2 * (2 ** nqubits,)
    m = np.random.random(shape) + 1j * np.random.random(shape)
    return expm(1j * (m + m.conj().T))


def random_active_qubits(nqubits, ntargets=None, nactive=None):
    """Generates random list of target and control qubits."""
    all_qubits = np.arange(nqubits)
    np.random.shuffle(all_qubits)
    if nactive is None:
        nactive = np.random.randint(ntargets + 1, nqubits)
    return list(all_qubits[:nactive])


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

    for i, (x, y) in enumerate(zip(final_state, target_state)):
        if np.abs(x - y) > 1e-6:
            print(i, x, y)

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
def test_unitary_matrix_gate(gates):
    """Check arbitrary unitary gate."""
    matrix = random_unitary_matrix(1)
    qibo_gate = gates.Unitary(matrix, 0)
    cirq_gate = [(cirq.MatrixGate(matrix), (0,))]
    assert_gates_equivalent(qibo_gate, cirq_gate, 1)

    matrix = random_unitary_matrix(2)
    qibo_gate = gates.Unitary(matrix, 0, 1)
    cirq_gate = [(cirq.MatrixGate(matrix), (0, 1))]
    assert_gates_equivalent(qibo_gate, cirq_gate, 2)


@pytest.mark.parametrize("gates", _GATES)
@pytest.mark.parametrize(("gate_name", "nqubits"),
                         [("H", 3), ("Z", 4), ("Y", 5),
                          ("X", 6), ("H", 7)])
def test_one_qubit_gates_controlled_by(gates, gate_name, nqubits):
    """Check one-qubit gates controlled on arbitrary number of qubits."""
    all_qubits = np.arange(nqubits)
    for _ in range(5):
        activeq = random_active_qubits(nqubits, 1)
        qibo_gate = getattr(gates, gate_name)(activeq[-1]).controlled_by(*activeq[:-1])
        cirq_gate = [(getattr(cirq, gate_name).controlled(len(activeq) - 1), activeq)]
        assert_gates_equivalent(qibo_gate, cirq_gate, nqubits)


@pytest.mark.skip("Cirq probably changes angle conventions when using ``controlled``.")
@pytest.mark.parametrize("gates", _GATES)
@pytest.mark.parametrize(("gate_name", "nqubits"),
                         [("RZ", 4), ("RY", 5), ("RX", 8)])
def test_one_qubit_rotations_controlled_by(gates, gate_name, nqubits):
    """Check one-qubit rotations controlled on arbitrary number of qubits."""
    all_qubits = np.arange(nqubits)
    for _ in range(5):
        activeq = random_active_qubits(nqubits, 1)
        theta = np.random.random()
        qibo_gate = getattr(gates, gate_name)(activeq[-1], np.pi * theta).controlled_by(*activeq[:-1])
        cirq_gate = [(getattr(cirq, gate_name.lower())(theta).controlled(len(activeq) - 1), activeq)]
        assert_gates_equivalent(qibo_gate, cirq_gate, nqubits)


@pytest.mark.skip
@pytest.mark.parametrize("gates", _GATES)
@pytest.mark.parametrize("nqubits", [4, 5, 8, 9, 12, 15, 17])
def test_two_qubit_gates_controlled_by(gates, nqubits):
    """Check ``SWAP`` and ``fSim`` gates controlled on arbitrary number of qubits."""
    all_qubits = np.arange(nqubits)
    for _ in range(5):
        activeq = random_active_qubits(nqubits, 2)
        qibo_gate = gates.SWAP(*activeq[-2:]).controlled_by(*activeq[:-2])
        cirq_gate = [(cirq.SWAP.controlled(len(activeq) - 2), activeq)]
        assert_gates_equivalent(qibo_gate, cirq_gate, nqubits)

        theta = np.random.random()
        phi = np.random.random()
        qibo_gate = gates.fSim(*activeq[-2:], theta, phi).controlled_by(*activeq[:-2])
        cirq_gate = [(cirq.FSimGate(theta, phi).controlled(len(activeq) - 2), activeq)]
        assert_gates_equivalent(qibo_gate, cirq_gate, nqubits)


@pytest.mark.skip
@pytest.mark.parametrize("gates", _GATES)
#@pytest.mark.parametrize("nqubits", [5, 6, 7, 11, 13, 14])
@pytest.mark.parametrize("nqubits", [5])
@pytest.mark.parametrize("ntargets", [1, 2])
def test_unitary_matrix_gate(gates, nqubits, ntargets):
    """Check arbitrary unitary gate controlled on arbitrary number of qubits."""
    all_qubits = np.arange(nqubits)
    for _ in range(5):
        activeq = random_active_qubits(nqubits, nactive=5)
        matrix = random_unitary_matrix(ntargets)
        qibo_gate = gates.Unitary(matrix, *activeq[-ntargets:]).controlled_by(*activeq[:-ntargets])
        cirq_gate = [(cirq.MatrixGate(matrix).controlled(len(activeq) - ntargets), activeq)]
        assert_gates_equivalent(qibo_gate, cirq_gate, nqubits)
