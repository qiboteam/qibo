"""Test how features defined in :class:`qibo.models.circuit.Circuit` work during circuit execution."""
import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit


def test_circuit_unitary(backend):
    from qibo import matrices
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.CNOT(0, 1))
    c.add(gates.X(0))
    c.add(gates.Y(1))
    final_matrix = c.unitary(backend)
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    target_matrix = np.kron(matrices.X, matrices.Y) @ cnot @ np.kron(h, h)
    backend.assert_allclose(final_matrix, target_matrix)


def test_circuit_unitary_bigger(backend):
    from qibo import matrices
    c = Circuit(4)
    c.add(gates.H(i) for i in range(4))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CZ(1, 2))
    c.add(gates.CNOT(0, 3))
    final_matrix = c.unitary(backend)
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    h = np.kron(np.kron(h, h), np.kron(h, h))
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    cz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
    m1 = np.kron(cnot, np.eye(4))
    m2 = np.kron(np.kron(np.eye(2), cz), np.eye(2))
    m3 = np.kron(cnot, np.eye(4)).reshape(8 * (2,))
    m3 = np.transpose(m3, [0, 2, 3, 1, 4, 6, 7, 5]).reshape((16, 16))
    target_matrix = m3 @ m2 @ m1 @ h
    backend.assert_allclose(final_matrix, target_matrix)


@pytest.mark.parametrize("compile", [False, True])
def test_circuit_vs_gate_execution(backend, compile):
    """Check consistency between executing circuit and stand alone gates."""
    nqubits = 2
    theta = 0.1234
    target_c = Circuit(nqubits)
    target_c.add(gates.X(0))
    target_c.add(gates.X(1))
    target_c.add(gates.CU1(0, 1, theta))
    target_result = backend.execute_circuit(target_c)

    # custom circuit
    def custom_circuit(state, theta):
        state = backend.apply_gate(gates.X(0), state, nqubits)
        state = backend.apply_gate(gates.X(1), state, nqubits)
        state = backend.apply_gate(gates.CU1(0, 1, theta), state, nqubits)
        return state

    initial_state = backend.zero_state(nqubits)
    if compile:
        c = backend.compile(custom_circuit)
    else:
        c = custom_circuit

    result = c(initial_state, theta)
    backend.assert_allclose(result, target_result)


def test_circuit_addition_execution(backend, accelerators):
    c1 = Circuit(4, accelerators)
    c1.add(gates.H(0))
    c1.add(gates.H(1))
    c1.add(gates.H(2))
    c2 = Circuit(4, accelerators)
    c2.add(gates.CNOT(0, 1))
    c2.add(gates.CZ(2, 3))
    c3 = c1 + c2

    c = Circuit(4, accelerators)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.H(2))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CZ(2, 3))
    backend.assert_circuitclose(c3, c)


@pytest.mark.parametrize("deep", [False, True])
def test_copied_circuit_execution(backend, accelerators, deep):
    """Check that circuit copy execution is equivalent to original circuit."""
    theta = 0.1234
    c1 = Circuit(4, accelerators)
    c1.add([gates.X(0), gates.X(1), gates.CU1(0, 1, theta)])
    c1.add([gates.H(2), gates.H(3), gates.CU1(2, 3, theta)])
    if not deep and accelerators is not None:  # pragma: no cover
        with pytest.raises(ValueError):
            c2 = c1.copy(deep)
    else:
        c2 = c1.copy(deep)
        backend.assert_circuitclose(c2, c1)


@pytest.mark.parametrize("fuse", [False, True])
def test_inverse_circuit_execution(backend, accelerators, fuse):
    c = Circuit(4, accelerators)
    c.add(gates.RX(0, theta=0.1))
    c.add(gates.U2(1, phi=0.2, lam=0.3))
    c.add(gates.U3(2, theta=0.1, phi=0.3, lam=0.2))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CZ(1, 2))
    c.add(gates.fSim(0, 2, theta=0.1, phi=0.3))
    c.add(gates.CU2(0, 1, phi=0.1, lam=0.1))
    if fuse:
        if accelerators:  # pragma: no cover
            with pytest.raises(NotImplementedError):
                c = c.fuse()
        else:
            c = c.fuse()
    invc = c.invert()
    target_state = np.ones(2 ** 4) / 4
    final_state = backend.execute_circuit(c, initial_state=np.copy(target_state))
    final_state = backend.execute_circuit(invc, initial_state=final_state)
    backend.assert_allclose(final_state, target_state)


def test_circuit_invert_and_addition_execution(backend, accelerators):
    subroutine = Circuit(6)
    subroutine.add([gates.RX(i, theta=0.1) for i in range(5)])
    subroutine.add([gates.CZ(i, i + 1) for i in range(0, 5, 2)])
    middle = Circuit(6)
    middle.add([gates.CU2(i, i + 1, phi=0.1, lam=0.2) for i in range(0, 5, 2)])
    circuit = subroutine + middle + subroutine.invert()

    c = Circuit(6)
    c.add([gates.RX(i, theta=0.1) for i in range(5)])
    c.add([gates.CZ(i, i + 1) for i in range(0, 5, 2)])
    c.add([gates.CU2(i, i + 1, phi=0.1, lam=0.2) for i in range(0, 5, 2)])
    c.add([gates.CZ(i, i + 1) for i in range(0, 5, 2)])
    c.add([gates.RX(i, theta=-0.1) for i in range(5)])

    assert c.depth == circuit.depth
    backend.assert_circuitclose(circuit, c)


@pytest.mark.parametrize("distribute_small", [False, True])
def test_circuit_on_qubits_execution(backend, accelerators, distribute_small):
    if distribute_small:
        smallc = Circuit(3, accelerators=accelerators)
    else:
        smallc = Circuit(3)
    smallc.add((gates.RX(i, theta=i + 0.1) for i in range(3)))
    smallc.add((gates.CNOT(0, 1), gates.CZ(1, 2)))

    largec = Circuit(6, accelerators=accelerators)
    largec.add((gates.RY(i, theta=i + 0.2) for i in range(0, 6, 2)))
    largec.add(smallc.on_qubits(1, 3, 5))

    targetc = Circuit(6)
    targetc.add((gates.RY(i, theta=i + 0.2) for i in range(0, 6, 2)))
    targetc.add((gates.RX(i, theta=i // 2 + 0.1) for i in range(1, 6, 2)))
    targetc.add((gates.CNOT(1, 3), gates.CZ(3, 5)))
    assert largec.depth == targetc.depth
    backend.assert_circuitclose(largec, targetc)


@pytest.mark.parametrize("distribute_small", [False, True])
def test_circuit_on_qubits_double_execution(backend, accelerators, distribute_small):
    if distribute_small:
        smallc = Circuit(3, accelerators=accelerators)
    else:
        smallc = Circuit(3)
    smallc.add((gates.RX(i, theta=i + 0.1) for i in range(3)))
    smallc.add((gates.CNOT(0, 1), gates.CZ(1, 2)))
    # execute the small circuit before adding it to the large one
    _ = backend.execute_circuit(smallc)

    largec = Circuit(6, accelerators=accelerators)
    largec.add((gates.RY(i, theta=i + 0.2) for i in range(0, 6, 2)))
    if distribute_small and accelerators is not None:  # pragma: no cover
        with pytest.raises(RuntimeError):
            largec.add(smallc.on_qubits(1, 3, 5))
    else:
        largec.add(smallc.on_qubits(1, 3, 5))
        targetc = Circuit(6)
        targetc.add((gates.RY(i, theta=i + 0.2) for i in range(0, 6, 2)))
        targetc.add((gates.RX(i, theta=i // 2 + 0.1) for i in range(1, 6, 2)))
        targetc.add((gates.CNOT(1, 3), gates.CZ(3, 5)))
        assert largec.depth == targetc.depth
        backend.assert_circuitclose(largec, targetc)


def test_circuit_on_qubits_controlled_by_execution(backend, accelerators):
    smallc = Circuit(3)
    smallc.add(gates.RX(0, theta=0.1).controlled_by(1, 2))
    smallc.add(gates.RY(1, theta=0.2).controlled_by(0))
    smallc.add(gates.RX(2, theta=0.3).controlled_by(1, 0))
    smallc.add(gates.RZ(1, theta=0.4).controlled_by(0, 2))

    largec = Circuit(6, accelerators=accelerators)
    largec.add(gates.H(i) for i in range(6))
    largec.add(smallc.on_qubits(1, 4, 3))

    targetc = Circuit(6)
    targetc.add(gates.H(i) for i in range(6))
    targetc.add(gates.RX(1, theta=0.1).controlled_by(3, 4))
    targetc.add(gates.RY(4, theta=0.2).controlled_by(1))
    targetc.add(gates.RX(3, theta=0.3).controlled_by(1, 4))
    targetc.add(gates.RZ(4, theta=0.4).controlled_by(1, 3))

    assert largec.depth == targetc.depth
    backend.assert_circuitclose(largec, targetc)


@pytest.mark.parametrize("controlled", [False, True])
def test_circuit_on_qubits_with_unitary_execution(backend, accelerators, controlled):
    unitaries = np.random.random((2, 2, 2))
    smallc = Circuit(2)
    if controlled:
        smallc.add(gates.Unitary(unitaries[0], 0).controlled_by(1))
        smallc.add(gates.Unitary(unitaries[1], 1).controlled_by(0))
    else:
        smallc.add(gates.Unitary(unitaries[0], 0))
        smallc.add(gates.Unitary(unitaries[1], 1))
    smallc.add(gates.CNOT(0, 1))

    largec = Circuit(4, accelerators=accelerators)
    largec.add(gates.RY(0, theta=0.1))
    largec.add(gates.RY(1, theta=0.2))
    largec.add(gates.RY(2, theta=0.3))
    largec.add(gates.RY(3, theta=0.2))
    largec.add(smallc.on_qubits(3, 0))

    targetc = Circuit(4)
    targetc.add(gates.RY(0, theta=0.1))
    targetc.add(gates.RY(1, theta=0.2))
    targetc.add(gates.RY(2, theta=0.3))
    targetc.add(gates.RY(3, theta=0.2))
    if controlled:
        targetc.add(gates.Unitary(unitaries[0], 3).controlled_by(0))
        targetc.add(gates.Unitary(unitaries[1], 0).controlled_by(3))
    else:
        targetc.add(gates.Unitary(unitaries[0], 3))
        targetc.add(gates.Unitary(unitaries[1], 0))
    targetc.add(gates.CNOT(3, 0))
    assert largec.depth == targetc.depth
    backend.assert_circuitclose(largec, targetc)


def test_circuit_decompose_execution(backend):
    c = Circuit(6)
    c.add(gates.RX(0, 0.1234))
    c.add(gates.RY(1, 0.4321))
    c.add((gates.H(i) for i in range(2, 6)))
    c.add(gates.CNOT(0, 1))
    c.add(gates.X(3).controlled_by(0, 1, 2, 4))
    decomp_c = c.decompose(5)
    backend.assert_circuitclose(c, decomp_c, atol=1e-6)


def test_repeated_execute_pauli_noise_channel(backend):
    thetas = np.random.random(4)
    backend.set_seed(1234)
    c = Circuit(4)
    c.add((gates.RY(i, t) for i, t in enumerate(thetas)))
    c.add((gates.PauliNoiseChannel(i, px=0.1, py=0.2, pz=0.3)
          for i in range(4)))
    final_state = backend.execute_circuit(c, nshots=20)

    backend.set_seed(1234)
    target_state = []
    for _ in range(20):
        noiseless_c = Circuit(4)
        noiseless_c.add((gates.RY(i, t) for i, t in enumerate(thetas)))
        for i in range(4):
            if backend.np.random.random() < 0.1:
                noiseless_c.add(gates.X(i))
            if backend.np.random.random() < 0.2:
                noiseless_c.add(gates.Y(i))
            if backend.np.random.random() < 0.3:
                noiseless_c.add(gates.Z(i))
        result = backend.execute_circuit(noiseless_c)
        target_state.append(result.state(numpy=True))
    final_state = [backend.to_numpy(x) for x in final_state]
    target_state = np.stack(target_state)
    backend.assert_allclose(final_state, target_state)


def test_repeated_execute_with_noise(backend):
    thetas = np.random.random(4)
    c = Circuit(4)
    c.add((gates.RY(i, t) for i, t in enumerate(thetas)))
    noisy_c = c.with_noise((0.2, 0.0, 0.1))
    backend.set_seed(1234)
    final_state = backend.execute_circuit(noisy_c, nshots=20)

    backend.set_seed(1234)
    target_state = []
    for _ in range(20):
        noiseless_c = Circuit(4)
        for i, t in enumerate(thetas):
            noiseless_c.add(gates.RY(i, theta=t))
            if backend.np.random.random() < 0.2:
                noiseless_c.add(gates.X(i))
            if backend.np.random.random() < 0.1:
                noiseless_c.add(gates.Z(i))
        result = backend.execute_circuit(noiseless_c)
        target_state.append(result.state(numpy=True))
    target_state = np.stack(target_state)
    final_state = [backend.to_numpy(x) for x in final_state]
    backend.assert_allclose(final_state, target_state)
