"""Test that Qibo gate execution agrees with Cirq."""
import numpy as np
import cirq
import pytest
import qibo
from qibo import models, gates


def random_state(nqubits):
    """Generates a random normalized state of shape (2^nqubits,)."""
    n = 2 ** nqubits
    state = np.random.random(n) + 1j * np.random.random(n)
    return state / np.sqrt((np.abs(state) ** 2).sum())


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


def execute_cirq(cirq_gates, nqubits, initial_state=None):
    """Executes a Cirq circuit with the given list of gates."""
    c = cirq.Circuit()
    q = [cirq.LineQubit(i) for i in range(nqubits)]
    # apply identity gates to all qubits so that they become part of the circuit
    c.append([cirq.I(qi) for qi in q])
    for gate, targets in cirq_gates:
        c.append(gate(*[q[i] for i in targets]))
    result = cirq.Simulator().simulate(c, initial_state=initial_state) # pylint: disable=no-member
    depth = len(cirq.Circuit(c.all_operations()))
    return result.final_state_vector, depth - 1


def assert_gates_equivalent(qibo_gate, cirq_gates, nqubits,
                            ndevices=None, atol=1e-7):
    """Asserts that QIBO and Cirq gates have equivalent action on a random state.

    Args:
        qibo_gate: QIBO gate.
        cirq_gates: List of tuples (cirq gate, target qubit IDs).
        nqubits: Total number of qubits in the circuit.
        atol: Absolute tolerance in state vector comparsion.
    """
    initial_state = random_state(nqubits)
    target_state, target_depth = execute_cirq(cirq_gates, nqubits,
                                              np.copy(initial_state))
    backend = qibo.get_backend()
    if ndevices is None or "numpy" in backend:
        accelerators = None
    else:
        accelerators = {"/GPU:0": ndevices}

    if backend != "custom" and accelerators:
        with pytest.raises(NotImplementedError):
            c = models.Circuit(nqubits, accelerators)
            c.add(qibo_gate)
    else:
        c = models.Circuit(nqubits, accelerators)
        c.add(qibo_gate)
        final_state = c(np.copy(initial_state))
        assert c.depth == target_depth
        np.testing.assert_allclose(final_state, target_state, atol=atol)


@pytest.mark.parametrize(("gate_name", "nqubits", "ndevices"),
                         [("H", 3, None), ("H", 3, 2),
                          ("X", 2, None), ("X", 2, 2),
                          ("Y", 1, None), ("Z", 1, None)])
def test_one_qubit_gates(backend, gate_name, nqubits, ndevices):
    """Check simple one-qubit gates."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    targets = random_active_qubits(nqubits, nactive=1)
    qibo_gate = getattr(gates, gate_name)(*targets)
    cirq_gate = [(getattr(cirq, gate_name), targets)]
    assert_gates_equivalent(qibo_gate, cirq_gate, nqubits, ndevices)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("gate_name", "nqubits", "ndevices"),
                         [("RX", 3, None), ("RX", 3, 4),
                          ("RY", 2, None), ("RY", 2, 2),
                          ("RZ", 1, None)])
def test_one_qubit_parametrized_gates(backend, gate_name, nqubits, ndevices):
    """Check parametrized one-qubit rotations."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    targets = random_active_qubits(nqubits, nactive=1)
    qibo_gate = getattr(gates, gate_name)(*targets, theta)
    cirq_gate = [(getattr(cirq, gate_name.lower())(theta), targets)]
    assert_gates_equivalent(qibo_gate, cirq_gate, nqubits, ndevices)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("nqubits", "ndevices"),
                         [(2, None), (3, 4), (2, 2)])
def test_u1_gate(backend, nqubits, ndevices):
    """Check U1 gate."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    targets = random_active_qubits(nqubits, nactive=1)
    qibo_gate = gates.U1(*targets, theta)
    cirq_gate = [(cirq.ZPowGate(exponent=theta / np.pi), targets)]
    assert_gates_equivalent(qibo_gate, cirq_gate, nqubits, ndevices)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("gate_name", ["CNOT", "SWAP", "CZ"])
@pytest.mark.parametrize("nqubits", [3, 4, 5])
@pytest.mark.parametrize("ndevices", [None, 2])
def test_two_qubit_gates(backend, gate_name, nqubits, ndevices):
    """Check two-qubit gates."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    targets = random_active_qubits(nqubits, nactive=2)
    qibo_gate = getattr(gates, gate_name)(*targets)
    cirq_gate = [(getattr(cirq, gate_name), targets)]
    assert_gates_equivalent(qibo_gate, cirq_gate, nqubits, ndevices)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("nqubits", "ndevices"),
                         [(2, None), (6, None), (6, 2),
                          (7, None), (7, 4)])
def test_two_qubit_parametrized_gates(backend, nqubits, ndevices):
    """Check ``CU1`` and ``fSim`` gate."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    phi = 0.4321

    targets = random_active_qubits(nqubits, nactive=2)
    qibo_gate = gates.CU1(*targets, np.pi * theta)
    cirq_gate = [(cirq.CZPowGate(exponent=theta), targets)]
    assert_gates_equivalent(qibo_gate, cirq_gate, nqubits)

    targets = random_active_qubits(nqubits, nactive=2)
    qibo_gate = gates.fSim(*targets, theta, phi)
    cirq_gate = [(cirq.FSimGate(theta=theta, phi=phi), targets)]
    assert_gates_equivalent(qibo_gate, cirq_gate, nqubits, ndevices)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("nqubits", "ndevices"),
                         [(5, None), (6, None), (9, None),
                          (5, 2), (6, 4), (9, 8)])
def test_unitary_matrix_gate(backend, nqubits, ndevices):
    """Check arbitrary unitary gate."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    matrix = random_unitary_matrix(1)
    targets = random_active_qubits(nqubits, nactive=1)
    qibo_gate = gates.Unitary(matrix, *targets)
    cirq_gate = [(cirq.MatrixGate(matrix), targets)]
    assert_gates_equivalent(qibo_gate, cirq_gate, nqubits)

    for _ in range(10):
        matrix = random_unitary_matrix(2)
        targets = random_active_qubits(nqubits, nactive=2)
        qibo_gate = gates.Unitary(matrix, *targets)
        cirq_gate = [(cirq.MatrixGate(matrix), targets)]
        assert_gates_equivalent(qibo_gate, cirq_gate, nqubits, ndevices)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("gate_name", "nqubits", "ndevices"),
                         [("H", 3, None), ("Z", 4, None), ("Y", 5, 4),
                          ("X", 6, None), ("H", 7, 2), ("Z", 8, 8),
                          ("Y", 12, 16)])
def test_one_qubit_gates_controlled_by(backend, gate_name, nqubits, ndevices):
    """Check one-qubit gates controlled on arbitrary number of qubits."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    all_qubits = np.arange(nqubits)
    for _ in range(5):
        activeq = random_active_qubits(nqubits, nmin=1)
        qibo_gate = getattr(gates, gate_name)(activeq[-1]).controlled_by(*activeq[:-1])
        cirq_gate = [(getattr(cirq, gate_name).controlled(len(activeq) - 1), activeq)]
        assert_gates_equivalent(qibo_gate, cirq_gate, nqubits, ndevices)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("nqubits", "ndevices"),
                         [(4, None), (5, None), (8, None),
                          (12, None), (15, None), (17, None),
                          (6, 2), (9, 2), (11, 4), (13, 8), (14, 16)])
def test_two_qubit_gates_controlled_by(backend, nqubits, ndevices):
    """Check ``SWAP`` and ``fSim`` gates controlled on arbitrary number of qubits."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
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
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("nqubits", [5, 12, 13, 14])
@pytest.mark.parametrize("ntargets", [1, 2])
@pytest.mark.parametrize("ndevices", [None, 2, 8])
def test_unitary_matrix_gate_controlled_by(backend, nqubits, ntargets, ndevices):
    """Check arbitrary unitary gate controlled on arbitrary number of qubits."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    all_qubits = np.arange(nqubits)
    for _ in range(10):
        activeq = random_active_qubits(nqubits, nactive=5)
        matrix = random_unitary_matrix(ntargets)
        qibo_gate = gates.Unitary(matrix, *activeq[-ntargets:]).controlled_by(*activeq[:-ntargets])
        cirq_gate = [(cirq.MatrixGate(matrix).controlled(len(activeq) - ntargets), activeq)]
        assert_gates_equivalent(qibo_gate, cirq_gate, nqubits, ndevices)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("nqubits", [5, 6, 7, 11, 12])
def test_qft(backend, accelerators, nqubits):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = models.QFT(nqubits, accelerators=accelerators)
    initial_state = random_state(nqubits)
    final_state = c(np.copy(initial_state))
    cirq_gates = [(cirq.qft, list(range(nqubits)))]
    target_state, _ = execute_cirq(cirq_gates, nqubits, np.copy(initial_state))
    np.testing.assert_allclose(target_state, final_state, atol=1e-6)
    qibo.set_backend(original_backend)
