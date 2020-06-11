"""
Testing that Tensorflow gates' action agrees with Cirq.
"""
import copy
import numpy as np
import cirq
import pytest
from qibo import models
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


def random_active_qubits(nqubits, nmin=None, nactive=None):
    """Generates random list of target and control qubits."""
    all_qubits = np.arange(nqubits)
    np.random.shuffle(all_qubits)
    if nactive is None:
        nactive = np.random.randint(nmin + 1, nqubits)
    return list(all_qubits[:nactive])


def execute_cirq(cirq_gates, nqubits, initial_state=None) -> np.ndarray:
    """Executes a Cirq circuit with the given list of gates."""
    c = cirq.Circuit()
    q = [cirq.LineQubit(i) for i in range(nqubits)]
    # apply identity gates to all qubits so that they become part of the circuit
    c.append([cirq.I(qi) for qi in q])
    for gate, targets in cirq_gates:
        c.append(gate(*[q[i] for i in targets]))
    result = cirq.Simulator().simulate(c, initial_state=initial_state)
    return result.final_state


def assert_gates_equivalent(qibo_gate, cirq_gates, nqubits,
                            ndevices=None, atol=1e-7):
    """Asserts that QIBO and Cirq gates have equivalent action on a random state.

    Args:
        qibo_gate: QIBO gate.
        cirq_gates: List of tuples (cirq gate, target qubit IDs).
        nqubits: Total number of qubits in the circuit.
        atol: Absolute tolerance in state vector comparsion.
    """
    initial_state = random_initial_state(nqubits)
    target_state = execute_cirq(cirq_gates, nqubits, np.copy(initial_state))

    if ndevices is None:
        accelerators = None
    else:
        if isinstance(qibo_gate, native_gates.TensorflowGate):
            pytest.skip("Distributed circuit does not support native gates.")
        accelerators = {"/GPU:0": ndevices}

    c = models.Circuit(nqubits, accelerators)
    c.add(qibo_gate)
    final_state = c(np.copy(initial_state)).numpy()
    np.testing.assert_allclose(target_state, final_state, atol=atol)


@pytest.mark.parametrize(("gates", "backend"), _BACKENDS)
@pytest.mark.parametrize(("gate_name", "nqubits", "ndevices"),
                         [("H", 3, None), ("H", 3, 2),
                          ("X", 2, None), ("X", 2, 2),
                          ("Y", 1, None), ("Z", 1, None)])
def test_one_qubit_gates(gates, backend, gate_name, nqubits, ndevices):
    """Check simple one-qubit gates."""
    targets = random_active_qubits(nqubits, nactive=1)
    qibo_gate = getattr(gates, gate_name)(*targets).with_backend(backend)
    cirq_gate = [(getattr(cirq, gate_name), targets)]
    assert_gates_equivalent(qibo_gate, cirq_gate, nqubits, ndevices)


@pytest.mark.parametrize(("gates", "backend"), _BACKENDS)
@pytest.mark.parametrize(("gate_name", "nqubits", "ndevices"),
                         [("RX", 3, None), ("RX", 3, 4),
                          ("RY", 2, None), ("RY", 2, 2),
                          ("RZ", 1, None)])
def test_one_qubit_parametrized_gates(gates, backend, gate_name, nqubits, ndevices):
    """Check parametrized one-qubit rotations."""
    theta = 0.1234
    targets = random_active_qubits(nqubits, nactive=1)
    qibo_gate = getattr(gates, gate_name)(*targets, theta).with_backend(backend)
    cirq_gate = [(getattr(cirq, gate_name.lower())(theta), targets)]
    assert_gates_equivalent(qibo_gate, cirq_gate, nqubits, ndevices)


@pytest.mark.parametrize(("gates", "backend"), _BACKENDS)
@pytest.mark.parametrize("gate_name", ["CNOT", "SWAP", "CZ"])
@pytest.mark.parametrize("nqubits", [3, 4, 5])
@pytest.mark.parametrize("ndevices", [None, 2])
def test_two_qubit_gates(gates, backend, gate_name, nqubits, ndevices):
    """Check two-qubit gates."""
    targets = random_active_qubits(nqubits, nactive=2)
    qibo_gate = getattr(gates, gate_name)(*targets).with_backend(backend)
    cirq_gate = [(getattr(cirq, gate_name), targets)]
    assert_gates_equivalent(qibo_gate, cirq_gate, nqubits, ndevices)


@pytest.mark.parametrize(("gates", "backend"), _BACKENDS)
@pytest.mark.parametrize(("nqubits", "ndevices"),
                         [(2, None), (6, None), (6, 2),
                          (7, None), (7, 4)])
def test_two_qubit_parametrized_gates(gates, backend, nqubits, ndevices):
    """Check ``CZPow`` and ``fSim`` gate."""
    theta = 0.1234
    phi = 0.4321

    targets = random_active_qubits(nqubits, nactive=2)
    qibo_gate = gates.CZPow(*targets, np.pi * theta).with_backend(backend)
    cirq_gate = [(cirq.CZPowGate(exponent=theta), targets)]
    assert_gates_equivalent(qibo_gate, cirq_gate, nqubits)

    targets = random_active_qubits(nqubits, nactive=2)
    qibo_gate = gates.fSim(*targets, theta, phi).with_backend(backend)
    cirq_gate = [(cirq.FSimGate(theta=theta, phi=phi), targets)]
    assert_gates_equivalent(qibo_gate, cirq_gate, nqubits, ndevices)


@pytest.mark.parametrize(("gates", "backend"), _BACKENDS)
@pytest.mark.parametrize(("nqubits", "ndevices"),
                         [(5, None), (6, None), (9, None),
                          (5, 2), (6, 4), (9, 8)])
def test_unitary_matrix_gate(gates, backend, nqubits, ndevices):
    """Check arbitrary unitary gate."""
    matrix = random_unitary_matrix(1)
    targets = random_active_qubits(nqubits, nactive=1)
    qibo_gate = gates.Unitary(matrix, *targets).with_backend(backend)
    cirq_gate = [(cirq.MatrixGate(matrix), targets)]
    assert_gates_equivalent(qibo_gate, cirq_gate, nqubits)

    for _ in range(10):
        matrix = random_unitary_matrix(2)
        targets = random_active_qubits(nqubits, nactive=2)
        qibo_gate = gates.Unitary(matrix, *targets).with_backend(backend)
        cirq_gate = [(cirq.MatrixGate(matrix), targets)]
        assert_gates_equivalent(qibo_gate, cirq_gate, nqubits, ndevices)


@pytest.mark.parametrize("gates", _GATES)
@pytest.mark.parametrize(("gate_name", "nqubits", "ndevices"),
                         [("H", 3, None), ("Z", 4, None), ("Y", 5, 4),
                          ("X", 6, None), ("H", 7, 2), ("Z", 8, 8),
                          ("Y", 12, 16)])
def test_one_qubit_gates_controlled_by(gates, gate_name, nqubits, ndevices):
    """Check one-qubit gates controlled on arbitrary number of qubits."""
    all_qubits = np.arange(nqubits)
    for _ in range(5):
        activeq = random_active_qubits(nqubits, nmin=1)
        qibo_gate = getattr(gates, gate_name)(activeq[-1]).controlled_by(*activeq[:-1])
        cirq_gate = [(getattr(cirq, gate_name).controlled(len(activeq) - 1), activeq)]
        assert_gates_equivalent(qibo_gate, cirq_gate, nqubits, ndevices)


@pytest.mark.parametrize("gates", _GATES)
@pytest.mark.parametrize(("nqubits", "ndevices"),
                         [(4, None), (5, None), (8, None),
                          (12, None), (15, None), (17, None),
                          (6, 2), (9, 2), (11, 4), (13, 8), (14, 16)])
def test_two_qubit_gates_controlled_by(gates, nqubits, ndevices):
    """Check ``SWAP`` and ``fSim`` gates controlled on arbitrary number of qubits."""
    all_qubits = np.arange(nqubits)
    for _ in range(5):
        activeq = random_active_qubits(nqubits, nmin=2)
        qibo_gate = gates.SWAP(*activeq[-2:]).controlled_by(*activeq[:-2])
        cirq_gate = [(cirq.SWAP.controlled(len(activeq) - 2), activeq)]
        assert_gates_equivalent(qibo_gate, cirq_gate, nqubits, ndevices)

        theta = np.random.random()
        phi = np.random.random()
        qibo_gate = gates.fSim(*activeq[-2:], theta, phi).controlled_by(*activeq[:-2])
        cirq_gate = [(cirq.FSimGate(theta, phi).controlled(len(activeq) - 2), activeq)]
        assert_gates_equivalent(qibo_gate, cirq_gate, nqubits, ndevices)


@pytest.mark.parametrize("gates", _GATES)
@pytest.mark.parametrize("nqubits", [5, 12, 13, 14])
@pytest.mark.parametrize("ntargets", [1, 2])
@pytest.mark.parametrize("ndevices", [None, 1, 2])
def test_unitary_matrix_gate_controlled_by(gates, nqubits, ntargets, ndevices):
    """Check arbitrary unitary gate controlled on arbitrary number of qubits."""
    all_qubits = np.arange(nqubits)
    for _ in range(10):
        activeq = random_active_qubits(nqubits, nactive=5)
        matrix = random_unitary_matrix(ntargets)
        qibo_gate = gates.Unitary(matrix, *activeq[-ntargets:]).controlled_by(*activeq[:-ntargets])
        cirq_gate = [(cirq.MatrixGate(matrix).controlled(len(activeq) - ntargets), activeq)]
        assert_gates_equivalent(qibo_gate, cirq_gate, nqubits, ndevices)


@pytest.mark.parametrize("gates", _GATES)
@pytest.mark.parametrize("nqubits", [5, 6, 7, 11, 12])
def test_qft(gates, nqubits):
    initial_state = random_initial_state(nqubits)
    c = models.QFT(nqubits, gates=gates)
    final_state = c(np.copy(initial_state)).numpy()
    cirq_gates = [(cirq.QFT, list(range(nqubits)))]
    target_state = execute_cirq(cirq_gates, nqubits, np.copy(initial_state))
    np.testing.assert_allclose(target_state, final_state, atol=1e-6)
