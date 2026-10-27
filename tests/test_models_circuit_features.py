"""Test how features defined in :class:`qibo.models.circuit.Circuit` work during circuit execution."""

import sys
from collections import Counter

import numpy as np
import pytest

from qibo import Circuit, gates, matrices
from qibo.noise import NoiseModel, PauliError
from qibo.quantum_info import random_unitary


def test_circuit_unitary(backend):
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.X(0))
    circuit.add(gates.Y(1))
    final_matrix = circuit.unitary(backend)
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    target_matrix = np.kron(matrices.X, matrices.Y) @ cnot @ np.kron(h, h)
    backend.assert_allclose(final_matrix, target_matrix)


@pytest.mark.parametrize("with_measurement", [False, True])
def test_circuit_unitary_bigger(backend, with_measurement):
    circuit = Circuit(4)
    circuit.add(gates.H(i) for i in range(4))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.CNOT(0, 3))
    if with_measurement:
        circuit.add(gates.M(*range(4)))
    final_matrix = circuit.unitary(backend)
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


def test_circuit_unitary_and_inverse_with_noise_channel(backend):
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.DepolarizingChannel([0, 1], 0.2))
    with pytest.raises(NotImplementedError):
        circuit.unitary(backend)
    with pytest.raises(NotImplementedError):
        circuit.invert()


def test_circuit_unitary_non_trivial(backend):
    target = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=complex,
    )
    target = backend.cast(target, dtype=target.dtype)
    nqubits = 3
    circuit = Circuit(nqubits)
    circuit.add(gates.SWAP(0, 2).controlled_by(1))
    unitary = circuit.unitary(backend)
    backend.assert_allclose(unitary, target)


@pytest.mark.parametrize("compile_flag", [False, True])
def test_circuit_vs_gate_execution(backend, compile_flag):
    """Check consistency between executing circuit and stand alone gates."""
    nqubits = 2
    theta = 0.1234
    target_circuit = Circuit(nqubits)
    target_circuit.add(gates.X(0))
    target_circuit.add(gates.X(1))
    target_circuit.add(gates.CU1(0, 1, theta))
    target_result = backend.execute_circuit(target_circuit)._state

    # custom circuit
    def custom_circuit(state, theta):
        state = backend.apply_gate(gates.X(0), state, nqubits)
        state = backend.apply_gate(gates.X(1), state, nqubits)
        state = backend.apply_gate(gates.CU1(0, 1, theta), state, nqubits)
        return state

    initial_state = backend.zero_state(nqubits)
    if compile_flag:
        if backend.platform != "tensorflow":
            pytest.skip("``compile`` only defined for ``TensorFlowBackend``.")
        circuit = backend.compile(custom_circuit)
    else:
        circuit = custom_circuit

    result = circuit(initial_state, theta)
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

    circuit = Circuit(4, accelerators)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.H(2))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CZ(2, 3))
    backend.assert_circuitclose(c3, circuit)


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
    circuit = Circuit(4, accelerators)
    circuit.add(gates.RX(0, theta=0.1))
    circuit.add(gates.U2(1, phi=0.2, lam=0.3))
    circuit.add(gates.U3(2, theta=0.1, phi=0.3, lam=0.2))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.fSim(0, 2, theta=0.1, phi=0.3))
    circuit.add(gates.CU2(0, 1, phi=0.1, lam=0.1))
    if fuse:
        if accelerators:  # pragma: no cover
            with pytest.raises(NotImplementedError):
                circuit = circuit.fuse()
        else:
            circuit = circuit.fuse()
    inv_circuit = circuit.invert()
    target_state = backend.plus_state(4)
    final_state = backend.execute_circuit(
        circuit,
        initial_state=backend.cast(target_state, dtype=target_state.dtype, copy=True),
    )._state
    final_state = backend.execute_circuit(inv_circuit, initial_state=final_state)._state
    backend.assert_allclose(final_state, target_state, atol=1e-6)


def test_circuit_invert_and_addition_execution(backend):
    subroutine = Circuit(6)
    subroutine.add([gates.RX(i, theta=0.1) for i in range(5)])
    subroutine.add([gates.CZ(i, i + 1) for i in range(0, 5, 2)])
    middle = Circuit(6)
    middle.add([gates.CU2(i, i + 1, phi=0.1, lam=0.2) for i in range(0, 5, 2)])
    circuit_add = subroutine + middle + subroutine.invert()

    circuit = Circuit(6)
    circuit.add([gates.RX(i, theta=0.1) for i in range(5)])
    circuit.add([gates.CZ(i, i + 1) for i in range(0, 5, 2)])
    circuit.add([gates.CU2(i, i + 1, phi=0.1, lam=0.2) for i in range(0, 5, 2)])
    circuit.add([gates.CZ(i, i + 1) for i in range(0, 5, 2)])
    circuit.add([gates.RX(i, theta=-0.1) for i in range(5)])

    assert circuit_add.depth == circuit.depth
    backend.assert_circuitclose(circuit_add, circuit)


@pytest.mark.parametrize("distribute_small", [False, True])
def test_circuit_on_qubits_execution(backend, accelerators, distribute_small):
    if distribute_small:
        small_circuit = Circuit(3, accelerators=accelerators)
    else:
        small_circuit = Circuit(3)
    small_circuit.add(gates.RX(i, theta=i + 0.1) for i in range(3))
    small_circuit.add((gates.CNOT(0, 1), gates.CZ(1, 2)))

    large_circuit = Circuit(6, accelerators=accelerators)
    large_circuit.add(gates.RY(i, theta=i + 0.2) for i in range(0, 6, 2))
    large_circuit.add(small_circuit.on_qubits(1, 3, 5))

    target_circuit = Circuit(6)
    target_circuit.add(gates.RY(i, theta=i + 0.2) for i in range(0, 6, 2))
    target_circuit.add(gates.RX(i, theta=i // 2 + 0.1) for i in range(1, 6, 2))
    target_circuit.add((gates.CNOT(1, 3), gates.CZ(3, 5)))
    assert large_circuit.depth == target_circuit.depth
    backend.assert_circuitclose(large_circuit, target_circuit)


@pytest.mark.parametrize("distribute_small", [False, True])
def test_circuit_on_qubits_double_execution(backend, accelerators, distribute_small):
    if distribute_small:
        small_circuit = Circuit(3, accelerators=accelerators)
    else:
        small_circuit = Circuit(3)
    small_circuit.add(gates.RX(i, theta=i + 0.1) for i in range(3))
    small_circuit.add((gates.CNOT(0, 1), gates.CZ(1, 2)))
    # execute the small circuit before adding it to the large one
    _ = backend.execute_circuit(small_circuit)

    large_circuit = Circuit(6, accelerators=accelerators)
    large_circuit.add(gates.RY(i, theta=i + 0.2) for i in range(0, 6, 2))
    if distribute_small and accelerators is not None:  # pragma: no cover
        with pytest.raises(RuntimeError):
            large_circuit.add(small_circuit.on_qubits(1, 3, 5))
    else:
        large_circuit.add(small_circuit.on_qubits(1, 3, 5))
        target_circuit = Circuit(6)
        target_circuit.add(gates.RY(i, theta=i + 0.2) for i in range(0, 6, 2))
        target_circuit.add(gates.RX(i, theta=i // 2 + 0.1) for i in range(1, 6, 2))
        target_circuit.add((gates.CNOT(1, 3), gates.CZ(3, 5)))
        assert large_circuit.depth == target_circuit.depth
        backend.assert_circuitclose(large_circuit, target_circuit)


def test_circuit_on_qubits_controlled_by_execution(backend, accelerators):
    small_circuit = Circuit(3)
    small_circuit.add(gates.RX(0, theta=0.1).controlled_by(1, 2))
    small_circuit.add(gates.RY(1, theta=0.2).controlled_by(0))
    small_circuit.add(gates.RX(2, theta=0.3).controlled_by(1, 0))
    small_circuit.add(gates.RZ(1, theta=0.4).controlled_by(0, 2))

    large_circuit = Circuit(6, accelerators=accelerators)
    large_circuit.add(gates.H(i) for i in range(6))
    large_circuit.add(small_circuit.on_qubits(1, 4, 3))

    target_circuit = Circuit(6)
    target_circuit.add(gates.H(i) for i in range(6))
    target_circuit.add(gates.RX(1, theta=0.1).controlled_by(3, 4))
    target_circuit.add(gates.RY(4, theta=0.2).controlled_by(1))
    target_circuit.add(gates.RX(3, theta=0.3).controlled_by(1, 4))
    target_circuit.add(gates.RZ(4, theta=0.4).controlled_by(1, 3))

    assert large_circuit.depth == target_circuit.depth
    backend.assert_circuitclose(large_circuit, target_circuit)


@pytest.mark.parametrize("controls", [(0,), (0, 1), (3, 1)])
@pytest.mark.parametrize(
    "build_gate",
    [
        lambda q, u: gates.H(q[0]),
        lambda q, u: gates.SWAP(q[0], q[1]),
        lambda q, u: gates.RX(q[0], theta=0.3),
        lambda q, u: gates.U3(q[0], theta=0.3, phi=0.2, lam=0.1),
        lambda q, u: gates.Unitary(u, q[0]),
        lambda q, u: gates.GlobalPhase(0.5),
        lambda q, u: gates.CNOT(q[0], q[1]),
        lambda q, u: gates.CY(q[0], q[1]),
        lambda q, u: gates.CZ(q[0], q[1]),
        lambda q, u: gates.CH(q[0], q[1]),
        lambda q, u: gates.CSX(q[0], q[1]),
        lambda q, u: gates.CSXDG(q[0], q[1]),
        lambda q, u: gates.CRX(q[0], q[1], theta=0.3),
        lambda q, u: gates.CRY(q[0], q[1], theta=0.3),
        lambda q, u: gates.CRZ(q[0], q[1], theta=0.3),
        lambda q, u: gates.CU1(q[0], q[1], theta=0.3),
        lambda q, u: gates.CU2(q[0], q[1], phi=0.3, lam=0.2),
        lambda q, u: gates.CU3(q[0], q[1], theta=0.3, phi=0.2, lam=0.1),
        lambda q, u: gates.TOFFOLI(q[0], q[1], q[2]),
        lambda q, u: gates.CCZ(q[0], q[1], q[2]),
        lambda q, u: gates.RX(q[2], theta=0.5).controlled_by(q[0], q[1]),
        lambda q, u: gates.X(q[3]).controlled_by(q[0], q[1], q[2]),
        lambda q, u: gates.Unitary(u, q[2]).controlled_by(q[0], q[1]),
    ],
)
def test_circuit_controlled_by(backend, build_gate, controls):
    nqubits = 6
    free = [q for q in range(nqubits) if q not in controls]

    circuit = Circuit(nqubits)
    circuit.add(build_gate(free, random_unitary(2, backend=backend)))
    controlled_circuit = circuit.controlled_by(*controls)
    assert controlled_circuit.nqubits == circuit.nqubits
    assert controlled_circuit.ngates == circuit.ngates

    # the original unitary acts trivially on the control qubits, hence it commutes with the
    # projector on the subspace where all controls are in the state 1
    target = circuit.unitary(backend)
    projector = backend.identity(1)
    for qubit in range(nqubits):
        factor = (
            backend.cast([[0, 0], [0, 1]]) if qubit in controls else backend.identity(2)
        )
        projector = backend.kron(projector, factor)
    expected = (
        backend.identity(2**nqubits) - projector + backend.matmul(target, projector)
    )
    backend.assert_allclose(controlled_circuit.unitary(backend), expected, atol=1e-10)


def test_circuit_controlled_by_gates():
    circuit = Circuit(4)
    circuit.add(gates.I(1))
    circuit.add(gates.Barrier(1, 2))
    circuit.add(gates.Align(2))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.RX(2, theta=0.2).controlled_by(1, 3))
    circuit.add(gates.RY(1, theta=0.1))
    controlled_circuit = circuit.controlled_by(0)

    # identity gates are not controlled
    for index in range(3):
        assert controlled_circuit.queue[index].control_qubits == ()
    # gates controlled by default are rebuilt from their uncontrolled counterpart
    assert isinstance(controlled_circuit.queue[3], gates.TOFFOLI)
    assert controlled_circuit.queue[3].control_qubits == (0, 1)
    assert controlled_circuit.queue[4].control_qubits == (0, 1, 3)
    assert controlled_circuit.queue[5].control_qubits == (0,)

    # the original circuit is left untouched
    assert isinstance(circuit.queue[3], gates.CNOT)
    assert circuit.queue[4].control_qubits == (1, 3)
    assert circuit.queue[5].control_qubits == ()


def test_circuit_controlled_by_parameters(backend):
    circuit = Circuit(3)
    circuit.add(gates.CRX(1, 2, theta=0.1))
    circuit.add(gates.RY(1, theta=0.2))
    circuit.set_parameters([0.7, 0.4])
    controlled_circuit = circuit.controlled_by(0)

    target = Circuit(3)
    target.add(gates.RX(2, theta=0.7).controlled_by(0, 1))
    target.add(gates.RY(1, theta=0.4).controlled_by(0))
    backend.assert_circuitclose(controlled_circuit, target)


def test_circuit_controlled_by_on_qubits(backend):
    # subroutine with qubit 0 left idle to be used as control
    subroutine = Circuit(3)
    subroutine.add(gates.RX(1, theta=0.3))
    subroutine.add(gates.CNOT(1, 2))
    controlled_subroutine = subroutine.controlled_by(0).on_qubits(3, 0, 1)

    circuit = Circuit(4)
    circuit.add(gates.H(i) for i in range(4))
    circuit.add(controlled_subroutine)

    target = Circuit(4)
    target.add(gates.H(i) for i in range(4))
    target.add(gates.RX(0, theta=0.3).controlled_by(3))
    target.add(gates.TOFFOLI(3, 0, 1))
    backend.assert_circuitclose(circuit, target)


def test_circuit_controlled_by_without_controls(backend):
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    controlled_circuit = circuit.controlled_by()
    assert controlled_circuit is not circuit
    backend.assert_circuitclose(controlled_circuit, circuit)


def test_circuit_controlled_by_density_matrix(backend):
    circuit = Circuit(3, density_matrix=True)
    circuit.add(gates.H(1))
    circuit.add(gates.CNOT(1, 2))
    controlled_circuit = circuit.controlled_by(0)
    assert controlled_circuit.density_matrix

    # control qubit in the state 1 leaves the action of the circuit unchanged
    target = Circuit(3, density_matrix=True)
    target.add(gates.X(0))
    target.add(circuit.on_qubits(0, 1, 2))
    final = Circuit(3, density_matrix=True)
    final.add(gates.X(0))
    final.add(controlled_circuit.on_qubits(0, 1, 2))
    backend.assert_circuitclose(final, target)


def test_circuit_controlled_by_errors():
    circuit = Circuit(3)
    circuit.add(gates.H(1))
    for qubit in (3, -1):
        with pytest.raises(ValueError):
            circuit.controlled_by(qubit)
    # control qubit overlaps with a target qubit
    with pytest.raises(ValueError):
        circuit.controlled_by(1)

    for gate in (gates.DEUTSCH(1, 2, 3, theta=0.1), gates.FanOut(1, 2, 3)):
        circuit = Circuit(4)
        circuit.add(gate)
        with pytest.raises(NotImplementedError):
            circuit.controlled_by(0)

    circuit = Circuit(3)
    circuit.add(gates.M(1))
    with pytest.raises(NotImplementedError):
        circuit.controlled_by(0)


@pytest.mark.parametrize("controlled", [False, True])
def test_circuit_on_qubits_with_unitary_execution(backend, accelerators, controlled):
    unitaries = np.random.random((2, 2, 2))
    small_circuit = Circuit(2)
    if controlled:
        small_circuit.add(gates.Unitary(unitaries[0], 0).controlled_by(1))
        small_circuit.add(gates.Unitary(unitaries[1], 1).controlled_by(0))
    else:
        small_circuit.add(gates.Unitary(unitaries[0], 0))
        small_circuit.add(gates.Unitary(unitaries[1], 1))
    small_circuit.add(gates.CNOT(0, 1))

    large_circuit = Circuit(4, accelerators=accelerators)
    large_circuit.add(gates.RY(0, theta=0.1))
    large_circuit.add(gates.RY(1, theta=0.2))
    large_circuit.add(gates.RY(2, theta=0.3))
    large_circuit.add(gates.RY(3, theta=0.2))
    large_circuit.add(small_circuit.on_qubits(3, 0))

    target_circuit = Circuit(4)
    target_circuit.add(gates.RY(0, theta=0.1))
    target_circuit.add(gates.RY(1, theta=0.2))
    target_circuit.add(gates.RY(2, theta=0.3))
    target_circuit.add(gates.RY(3, theta=0.2))
    if controlled:
        target_circuit.add(gates.Unitary(unitaries[0], 3).controlled_by(0))
        target_circuit.add(gates.Unitary(unitaries[1], 0).controlled_by(3))
    else:
        target_circuit.add(gates.Unitary(unitaries[0], 3))
        target_circuit.add(gates.Unitary(unitaries[1], 0))
    target_circuit.add(gates.CNOT(3, 0))
    assert large_circuit.depth == target_circuit.depth
    backend.assert_circuitclose(large_circuit, target_circuit)


def test_circuit_decompose_execution(backend):
    circuit = Circuit(6)
    circuit.add(gates.RX(0, 0.1234))
    circuit.add(gates.RY(1, 0.4321))
    circuit.add(gates.H(i) for i in range(2, 6))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.X(3).controlled_by(0, 1, 2, 4))
    decomp_circuit = circuit.decompose(5)
    backend.assert_circuitclose(circuit, decomp_circuit, atol=1e-6)


def test_repeated_execute_pauli_noise_channel(backend):
    thetas = np.random.random(4)
    backend.set_seed(1234)
    circuit = Circuit(4)
    circuit.add((gates.RY(i, t) for i, t in enumerate(thetas)))
    circuit.add(
        gates.PauliNoiseChannel(i, list(zip(["X", "Y", "Z"], [0.1, 0.2, 0.3])))
        for i in range(4)
    )
    with pytest.raises(RuntimeError) as excinfo:
        backend.execute_circuit(circuit, nshots=20)
    assert (
        str(excinfo.value)
        == "Attempting to perform noisy simulation with `density_matrix=False` and no Measurement gate in the Circuit. If you wish to retrieve the statistics of the outcomes please include measurements in the circuit, otherwise set `density_matrix=True` to recover the final state."
    )


def test_repeated_execute_with_pauli_noise(backend):
    thetas = np.random.random(4)
    circuit = Circuit(4)
    circuit.add((gates.RY(i, t) for i, t in enumerate(thetas)))
    noisy_circuit = circuit.with_pauli_noise(list(zip(["X", "Z"], [0.2, 0.1])))
    backend.set_seed(1234)
    with pytest.raises(RuntimeError) as excinfo:
        backend.execute_circuit(noisy_circuit, nshots=20)
    assert (
        str(excinfo.value)
        == "Attempting to perform noisy simulation with `density_matrix=False` and no Measurement gate in the Circuit. If you wish to retrieve the statistics of the outcomes please include measurements in the circuit, otherwise set `density_matrix=True` to recover the final state."
    )


@pytest.mark.skipif(sys.platform == "darwin", reason="Mac tests")
@pytest.mark.parametrize("nqubits", [1, 2])
def test_repeated_execute_probs_and_freqs(backend, nqubits):
    circuit = Circuit(nqubits)
    circuit.add(gates.X(q) for q in range(nqubits))
    circuit.add(gates.M(q) for q in range(nqubits))

    noise_map = list(zip(["X", "Y", "Z"], [0.1, 0.1, 0.1]))
    noise_map = PauliError(noise_map)
    noise = NoiseModel()
    noise.add(noise_map, gates.X)
    noisy_circuit = noise.apply(circuit)
    backend.set_seed(1234)
    result = backend.execute_circuit_repeated(noisy_circuit, nshots=1024)

    # Tensorflow seems to yield different results with same seed
    if backend.__class__.__name__ == "TensorflowBackend":
        test_frequencies = (
            Counter({"1": 844, "0": 180})
            if nqubits == 1
            else Counter({"11": 674, "10": 155, "01": 154, "00": 41})
        )
    elif backend.__class__.__name__ == "PyTorchBackend":
        test_frequencies = (
            Counter({"1": 817, "0": 207})
            if nqubits == 1
            else Counter({"11": 664, "01": 162, "10": 166, "00": 32})
        )
    elif backend.__class__.__name__ == "CuQuantumBackend":
        test_frequencies = (
            Counter({"1": 788, "0": 236})
            if nqubits == 1
            else Counter({"11": 637, "10": 156, "01": 183, "00": 48})
        )
    else:
        # CupyBackend's random_choice now matches numpy's exactly (once
        # probabilities are correctly normalized), so it falls in here too.
        test_frequencies = (
            Counter({"1": 790, "0": 234})
            if nqubits == 1
            else Counter({"11": 618, "10": 169, "01": 185, "00": 52})
        )
    for key in dict(test_frequencies):
        backend.assert_allclose(result.frequencies()[key], test_frequencies[key])
