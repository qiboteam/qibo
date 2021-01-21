"""Test how features defined in :class:`qibo.abstractions.circuit.AbstractCircuit` work during circuit execution."""
import numpy as np
import pytest
import qibo
from qibo import gates
from qibo.models import Circuit


def test_circuit_vs_gate_execution(backend):
    """Check consistency between executing circuit and stand alone gates."""
    from qibo import K
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.CU1(0, 1, theta))
    r1 = c.execute()

    # custom circuit
    def custom_circuit(initial_state, theta):
        l1 = gates.X(0)(initial_state)
        l2 = gates.X(1)(l1)
        o = gates.CU1(0, 1, theta)(l2)
        return o

    init2 = c.get_initial_state()
    init3 = c.get_initial_state()
    if backend != "custom":
        init2 = K.reshape(init2, (2, 2))
        init3 = K.reshape(init3, (2, 2))

    r2 = K.reshape(custom_circuit(init2, theta), (4,))
    np.testing.assert_allclose(r1, r2)
    compiled_custom_circuit = K.compile(custom_circuit)
    if backend == "custom":
        with pytest.raises(NotImplementedError):
            r3 = compiled_custom_circuit(init3, theta)
    else:
        r3 = K.reshape(compiled_custom_circuit(init3, theta), (4,))
        np.testing.assert_allclose(r2, r3)
    qibo.set_backend(original_backend)


def test_circuit_addition_execution(backend, accelerators):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
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
    np.testing.assert_allclose(c3(), c())
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("deep", [False, True])
def test_copied_circuit_execution(backend, accelerators, deep):
    """Check that circuit copy execution is equivalent to original circuit."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234

    c1 = Circuit(4, accelerators)
    c1.add([gates.X(0), gates.X(1), gates.CU1(0, 1, theta)])
    c1.add([gates.H(2), gates.H(3), gates.CU1(2, 3, theta)])
    if not deep and accelerators is not None:
        with pytest.raises(ValueError):
            c2 = c1.copy(deep)
    else:
        c2 = c1.copy(deep)
        np.testing.assert_allclose(c2(), c1())
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("fuse", [False, True])
def test_inverse_circuit_execution(backend, accelerators, fuse):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    c = Circuit(4, accelerators)
    c.add(gates.RX(0, theta=0.1))
    c.add(gates.U2(1, phi=0.2, lam=0.3))
    c.add(gates.U3(2, theta=0.1, phi=0.3, lam=0.2))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CZ(1, 2))
    c.add(gates.fSim(0, 2, theta=0.1, phi=0.3))
    c.add(gates.CU2(0, 1, phi=0.1, lam=0.1))
    if fuse:
        c = c.fuse()
    invc = c.invert()
    target_state = np.ones(2 ** 4) / 4
    final_state = invc(c(np.copy(target_state)))
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_circuit_invert_and_addition_execution(backend, accelerators):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
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
    np.testing.assert_allclose(circuit(), c())
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("distribute_small", [False, True])
def test_circuit_on_qubits_execution(backend, accelerators, distribute_small):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

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
    np.testing.assert_allclose(largec(), targetc())
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("distribute_small", [False, True])
def test_circuit_on_qubits_double_execution(backend, accelerators, distribute_small):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    if distribute_small:
        smallc = Circuit(3, accelerators=accelerators)
    else:
        smallc = Circuit(3)
    smallc.add((gates.RX(i, theta=i + 0.1) for i in range(3)))
    smallc.add((gates.CNOT(0, 1), gates.CZ(1, 2)))
    # execute the small circuit before adding it to the large one
    _ = smallc()

    largec = Circuit(6, accelerators=accelerators)
    largec.add((gates.RY(i, theta=i + 0.2) for i in range(0, 6, 2)))
    if distribute_small and accelerators is not None:
        with pytest.raises(RuntimeError):
            largec.add(smallc.on_qubits(1, 3, 5))
    else:
        largec.add(smallc.on_qubits(1, 3, 5))
        targetc = Circuit(6)
        targetc.add((gates.RY(i, theta=i + 0.2) for i in range(0, 6, 2)))
        targetc.add((gates.RX(i, theta=i // 2 + 0.1) for i in range(1, 6, 2)))
        targetc.add((gates.CNOT(1, 3), gates.CZ(3, 5)))
        assert largec.depth == targetc.depth
        np.testing.assert_allclose(largec(), targetc())
    qibo.set_backend(original_backend)


def test_circuit_on_qubits_with_unitary_execution(backend, accelerators):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    unitaries = np.random.random((2, 2, 2))
    smallc = Circuit(2)
    smallc.add((gates.Unitary(u, i) for i, u in enumerate(unitaries)))
    smallc.add(gates.CNOT(0, 1))

    largec = Circuit(4, accelerators=accelerators)
    largec.add(gates.RY(1, theta=0.1))
    largec.add(gates.RY(2, theta=0.2))
    largec.add(smallc.on_qubits(0, 3))

    targetc = Circuit(4)
    targetc.add(gates.RY(1, theta=0.1))
    targetc.add(gates.RY(2, theta=0.2))
    targetc.add(gates.Unitary(unitaries[0], 0))
    targetc.add(gates.Unitary(unitaries[1], 3))
    targetc.add(gates.CNOT(0, 3))
    assert largec.depth == targetc.depth
    np.testing.assert_allclose(largec(), targetc())
    qibo.set_backend(original_backend)


def test_circuit_on_qubits_with_varlayer_execution(backend, accelerators):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    thetas = np.random.random([2, 4])
    smallc = Circuit(4)
    smallc.add(gates.VariationalLayer(range(4), [(0, 1), (2, 3)],
                                      gates.RX, gates.CNOT,
                                      thetas[0]))

    largec = Circuit(8, accelerators=accelerators)
    largec.add(smallc.on_qubits(*range(0, 8, 2)))
    largec.add(gates.VariationalLayer(range(1, 8, 2), [(1, 3), (5, 7)],
                                      gates.RY, gates.CZ,
                                      thetas[1]))

    targetc = Circuit(8)
    targetc.add(gates.VariationalLayer(range(0, 8, 2), [(0, 2), (4, 6)],
                                       gates.RX, gates.CNOT,
                                       thetas[0]))
    targetc.add(gates.VariationalLayer(range(1, 8, 2), [(1, 3), (5, 7)],
                                       gates.RY, gates.CZ,
                                       thetas[1]))
    assert largec.depth == targetc.depth
    np.testing.assert_allclose(largec(), targetc())
    qibo.set_backend(original_backend)


def test_repeated_execute_pauli_noise_channel(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    thetas = np.random.random(4)
    c = Circuit(4)
    c.add((gates.RY(i, t) for i, t in enumerate(thetas)))

    c.add((gates.PauliNoiseChannel(i, px=0.1, py=0.2, pz=0.3, seed=1234)
          for i in range(4)))
    final_state = c(nshots=20)

    np.random.seed(1234)
    target_state = []
    for _ in range(20):
        noiseless_c = Circuit(4)
        noiseless_c.add((gates.RY(i, t) for i, t in enumerate(thetas)))
        for i in range(4):
            if np.random.random() < 0.1:
                noiseless_c.add(gates.X(i))
            if np.random.random() < 0.2:
                noiseless_c.add(gates.Y(i))
            if np.random.random() < 0.3:
                noiseless_c.add(gates.Z(i))
        target_state.append(noiseless_c())
    target_state = np.stack(target_state)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_repeated_execute_with_noise(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    thetas = np.random.random(4)
    c = Circuit(4)
    c.add((gates.RY(i, t) for i, t in enumerate(thetas)))
    noisy_c = c.with_noise((0.2, 0.0, 0.1))
    np.random.seed(1234)
    final_state = noisy_c(nshots=20)

    np.random.seed(1234)
    target_state = []
    for _ in range(20):
        noiseless_c = Circuit(4)
        for i, t in enumerate(thetas):
            noiseless_c.add(gates.RY(i, theta=t))
            if np.random.random() < 0.2:
                noiseless_c.add(gates.X(i))
            if np.random.random() < 0.1:
                noiseless_c.add(gates.Z(i))
        target_state.append(noiseless_c())
    target_state = np.stack(target_state)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)
