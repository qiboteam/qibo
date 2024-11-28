import numpy as np
import pytest

from qibo import Circuit, gates


@pytest.mark.parametrize("nqubits", [2, 3])
def test_fused_gate_construct_unitary(backend, nqubits):
    gate = gates.FusedGate(0, 1)
    gate.append(gates.H(0))
    gate.append(gates.H(1))
    gate.append(gates.CZ(0, 1))
    hmatrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    czmatrix = np.diag([1, 1, 1, -1])
    target_matrix = czmatrix @ np.kron(hmatrix, hmatrix)
    if nqubits > 2:
        gate.append(gates.TOFFOLI(0, 1, 2))
        toffoli = np.eye(8)
        toffoli[-2:, -2:] = np.array([[0, 1], [1, 0]])
        target_matrix = toffoli @ np.kron(target_matrix, np.eye(2))
    backend.assert_allclose(gate.matrix(backend), target_matrix)


def test_single_fusion_gate():
    """Check circuit fusion that creates a single ``FusedGate``."""
    queue = [gates.H(0), gates.X(1), gates.CZ(0, 1)]
    c = Circuit(2)
    c.add(queue)
    c = c.fuse()
    assert len(c.queue) == 1
    fgate = c.queue[0]
    assert fgate.gates[0] == queue[1]
    assert fgate.gates[1] == queue[0]
    assert fgate.gates[2] == queue[2]


def test_two_fusion_gate():
    """Check fusion that creates two ``FusedGate``s."""
    queue = [
        gates.X(0),
        gates.H(1),
        gates.RX(2, theta=0.1234).controlled_by(1),
        gates.H(2),
        gates.Y(1),
        gates.H(0),
    ]
    c = Circuit(3)
    c.add(queue)
    c = c.fuse()
    assert len(c.queue) == 2
    fgate1, fgate2 = c.queue
    assert fgate2.gates[0] == queue[0]
    assert fgate2.gates[1] == queue[-1]
    assert fgate1.gates == [queue[1], queue[2], queue[4], queue[3]]


def test_fusedgate_matrix_calculation(backend):
    queue = [gates.H(0), gates.H(1), gates.CNOT(0, 1), gates.X(0), gates.X(1)]
    circuit = Circuit(2)
    circuit.add(queue)
    circuit = circuit.fuse()
    assert len(circuit.queue) == 1
    fused_gate = circuit.queue[0]

    x = np.array([[0, 1], [1, 0]])
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    target_matrix = np.kron(x, x) @ cnot @ np.kron(h, h)
    fused_matrix = fused_gate.matrix(backend)
    backend.assert_allclose(fused_matrix, target_matrix)


def test_fuse_circuit_two_qubit_gates(backend):
    """Check circuit fusion in circuit with two-qubit gates only."""
    c = Circuit(2)
    c.add(gates.CNOT(0, 1))
    c.add(gates.RX(0, theta=0.1234).controlled_by(1))
    c.add(gates.SWAP(0, 1))
    c.add(gates.fSim(1, 0, theta=0.1234, phi=0.324))
    c.add(gates.RY(1, theta=0.1234).controlled_by(0))
    fused_c = c.fuse()
    backend.assert_circuitclose(fused_c, c)


@pytest.mark.parametrize("max_qubits", [2, 3, 4])
def test_fuse_circuit_three_qubit_gate(backend, max_qubits):
    """Check circuit fusion in circuit with three-qubit gate."""
    c = Circuit(4)
    c.add(gates.H(i) for i in range(4))
    c.add(gates.CZ(0, 1))
    c.add(gates.CZ(2, 3))
    c.add(gates.TOFFOLI(0, 1, 2))
    c.add(gates.SWAP(1, 2))
    c.add(gates.H(i) for i in range(4))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CZ(2, 3))
    fused_c = c.fuse(max_qubits=max_qubits)
    backend.assert_circuitclose(fused_c, c, atol=1e-12)


@pytest.mark.parametrize("nqubits", [4, 5, 10, 11])
@pytest.mark.parametrize("nlayers", [1, 2])
@pytest.mark.parametrize("max_qubits", [2, 3, 4])
def test_variational_layer_fusion(backend, nqubits, nlayers, max_qubits):
    """Check fused variational layer execution."""
    theta = 2 * np.pi * np.random.random((2 * nlayers * nqubits,))
    theta_iter = iter(theta)

    c = Circuit(nqubits)
    for _ in range(nlayers):
        c.add(gates.RY(i, next(theta_iter)) for i in range(nqubits))
        c.add(gates.CZ(i, i + 1) for i in range(0, nqubits - 1, 2))
        c.add(gates.RY(i, next(theta_iter)) for i in range(nqubits))
        c.add(gates.CZ(i, i + 1) for i in range(1, nqubits - 1, 2))
        c.add(gates.CZ(0, nqubits - 1))

    fused_c = c.fuse(max_qubits=max_qubits)
    backend.assert_circuitclose(fused_c, c)


@pytest.mark.parametrize("nqubits", [4, 5])
@pytest.mark.parametrize("ngates", [10, 20])
@pytest.mark.parametrize("max_qubits", [2, 3, 4])
def test_random_circuit_fusion(backend, nqubits, ngates, max_qubits):
    """Check gate fusion in randomly generated circuits."""
    one_qubit_gates = [gates.RX, gates.RY, gates.RZ]
    two_qubit_gates = [gates.CNOT, gates.CZ, gates.SWAP]
    thetas = np.pi * np.random.random((ngates,))
    c = Circuit(nqubits)
    for i in range(ngates):
        gate = one_qubit_gates[int(np.random.randint(0, 3))]
        q0 = np.random.randint(0, nqubits)
        c.add(gate(q0, thetas[i]))
        gate = two_qubit_gates[int(np.random.randint(0, 3))]
        q0, q1 = np.random.randint(0, nqubits, (2,))
        while q0 == q1:
            q0, q1 = np.random.randint(0, nqubits, (2,))
        c.add(gate(q0, q1))
    fused_c = c.fuse(max_qubits=max_qubits)
    backend.assert_circuitclose(fused_c, c, atol=1e-7)


def test_controlled_by_gates_fusion(backend):
    """Check circuit fusion that contains ``controlled_by`` gates."""
    c = Circuit(4)
    c.add(gates.H(i) for i in range(4))
    c.add(gates.RX(1, theta=0.1234).controlled_by(0))
    c.add(gates.RX(3, theta=0.4321).controlled_by(2))
    c.add(gates.RY(i, theta=0.5678) for i in range(4))
    c.add(gates.RX(1, theta=0.1234).controlled_by(0))
    c.add(gates.RX(3, theta=0.4321).controlled_by(2))
    fused_c = c.fuse()
    backend.assert_circuitclose(fused_c, c)


def test_callbacks_fusion(backend):
    """Check entropy calculation in fused circuit."""
    from qibo import callbacks

    entropy = callbacks.EntanglementEntropy([0])
    c = Circuit(5)
    c.add(gates.H(0))
    c.add(gates.X(1))
    c.add(gates.CallbackGate(entropy))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))
    fused_c = c.fuse()
    backend.assert_circuitclose(fused_c, c)
    target_entropy = [0.0, 1.0, 0.0, 1.0]
    final_entropy = [backend.to_numpy(x) for x in entropy[:]]
    backend.assert_allclose(final_entropy, target_entropy, atol=1e-7)


def test_set_parameters_fusion(backend):
    """Check gate fusion when ``circuit.set_parameters`` is used."""
    c = Circuit(2)
    c.add(gates.RX(0, theta=0.1234))
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.CNOT(0, 1))
    c.add(gates.RY(0, theta=0.1234))
    c.add(gates.RY(1, theta=0.1234))
    fused_c = c.fuse()
    backend.assert_circuitclose(fused_c, c)

    c.set_parameters(4 * [0.4321])
    fused_c.set_parameters(4 * [0.4321])
    backend.assert_circuitclose(fused_c, c)


@pytest.mark.parametrize("max_qubits", [1, 2, 3])
def test_fusion_with_measurements(backend, max_qubits):
    c = Circuit(3, density_matrix=True)
    c.add(gates.X(i) for i in range(3))
    c.add(gates.M(0))
    c.add(gates.CNOT(0, 1))
    c.add(gates.H(2))
    c.add(gates.M(0, 1, 2))
    fused_c = c.fuse(max_qubits=max_qubits)
    assert fused_c.measurements == c.measurements
    backend.assert_circuitclose(fused_c, c)


def test_add_fused_gate(backend):
    """Check adding fused gate to a circuit."""
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.CNOT(0, 1))
    fused_c = c.fuse()
    fgate = fused_c.queue[0]
    assert isinstance(fgate, gates.FusedGate)

    new_c = Circuit(2)
    new_c.add(fgate)
    assert c.depth == 2
    assert c.ngates == 3
    assert new_c.depth == 1
    assert new_c.ngates == 1
    backend.assert_circuitclose(fused_c, c)
    backend.assert_circuitclose(new_c, c)


def test_fused_gate_draw():
    ref = (
        "q0: ─[─H─U1───]─U1─U1─U1─────────────────────────────────────x───\n"
        "q1: ─[───o──H─]─|──|──|──[─U1───]─U1─U1──────────────────────|─x─\n"
        "q2: ────────────o──|──|──[─o──H─]─|──|──[─U1───]─U1──────────|─|─\n"
        "q3: ───────────────o──|───────────o──|──[─o──H─]─|──[─U1───]─|─x─\n"
        "q4: ──────────────────o──────────────o───────────o──[─o──H─]─x───"
    )
    circuit = Circuit(5, wire_names=["q0", "q1", "q2", "q3", "q4"])
    for i1 in range(5):
        circuit.add(gates.H(i1))
        for i2 in range(i1 + 1, 5):
            circuit.add(gates.CU1(i2, i1, theta=0))
    circuit.add(gates.SWAP(0, 4))
    circuit.add(gates.SWAP(1, 3))
    circuit = circuit.fuse()
    assert str(circuit) == ref
