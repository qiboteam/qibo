import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.noise import (
    AmplitudeDampingError,
    CustomError,
    DepolarizingError,
    IBMQNoiseModel,
    KrausError,
    NoiseModel,
    PauliError,
    PhaseDampingError,
    ReadoutError,
    ResetError,
    ThermalRelaxationError,
    UnitaryError,
)
from qibo.quantum_info import (
    random_clifford,
    random_density_matrix,
    random_statevector,
    random_stochastic_matrix,
)


@pytest.mark.parametrize("density_matrix", [True])
@pytest.mark.parametrize("nshots", [None, 10, 100])
def test_kraus_error(backend, density_matrix, nshots):
    k1 = np.sqrt(0.4) * np.array([[1, 0], [0, 1]])
    k2 = np.sqrt(0.6) * np.array([[1, 0], [0, 1]])

    with pytest.raises(ValueError):
        KrausError([k1, np.array([1])])

    kraus_error = KrausError([k1, k2])

    noise = NoiseModel()
    noise.add(kraus_error, gates.X, 1)
    noise.add(kraus_error, gates.CNOT)
    noise.add(kraus_error, gates.Z, (0, 1))

    circuit = Circuit(3, density_matrix=density_matrix)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.Z(1))
    circuit.add(gates.X(1))
    circuit.add(gates.X(2))
    circuit.add(gates.Z(2))
    circuit.add(gates.M(0, 1, 2))

    target_circuit = Circuit(3, density_matrix=density_matrix)
    target_circuit.add(gates.CNOT(0, 1))
    target_circuit.add(gates.KrausChannel(0, [k1, k2]))
    target_circuit.add(gates.KrausChannel(1, [k1, k2]))
    target_circuit.add(gates.Z(1))
    target_circuit.add(gates.KrausChannel(1, [k1, k2]))
    target_circuit.add(gates.X(1))
    target_circuit.add(gates.KrausChannel(1, [k1, k2]))
    target_circuit.add(gates.X(2))
    target_circuit.add(gates.Z(2))
    target_circuit.add(gates.M(0, 1, 2))

    initial_psi = (
        random_density_matrix(2**3, backend=backend)
        if density_matrix
        else random_statevector(2**3, backend=backend)
    )
    backend.set_seed(123)
    final_state = backend.execute_circuit(
        noise.apply(circuit), initial_state=backend.np.copy(initial_psi), nshots=nshots
    )
    final_state_samples = final_state.samples() if nshots else None
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(
        target_circuit, initial_state=backend.np.copy(initial_psi), nshots=nshots
    )
    target_final_state_samples = target_final_state.samples() if nshots else None

    if nshots is None:
        backend.assert_allclose(final_state._state, target_final_state._state)
    else:
        backend.assert_allclose(final_state_samples, target_final_state_samples)


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("nshots", [10, 100])
def test_unitary_error(backend, density_matrix, nshots):
    u1 = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0], [0, 0, 1.0, 0]])
    u2 = np.array([[0, 1.0, 0, 0], [1.0, 0, 0, 0], [0, 0, 0, 1.0], [0, 0, 1.0, 0]])
    qubits = (0, 1)
    p1, p2 = (0.3, 0.7)

    with pytest.raises(ValueError):
        UnitaryError([p1, p2], [u1, np.array([[1, 0], [0, 1]])])

    unitary_error = UnitaryError([p1, p2], [u1, u2])
    unitary_error_1q = UnitaryError([0.1], [np.eye(2)])

    noise = NoiseModel()
    noise.add(unitary_error_1q, qubits=[0])
    noise.add(unitary_error, gates.CNOT)

    circuit = Circuit(3, density_matrix=density_matrix)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.Z(1))
    circuit.add(gates.X(1))
    circuit.add(gates.X(2))
    circuit.add(gates.Z(2))
    circuit.add(gates.M(0, 1, 2))

    target_circuit = Circuit(3, density_matrix=density_matrix)
    target_circuit.add(gates.CNOT(0, 1))
    target_circuit.add(gates.UnitaryChannel(qubits, [(p1, u1), (p2, u2)]))
    target_circuit.add(gates.UnitaryChannel(0, [(0.1, np.eye(2))]))
    target_circuit.add(gates.Z(1))
    target_circuit.add(gates.X(1))
    target_circuit.add(gates.X(2))
    target_circuit.add(gates.Z(2))
    target_circuit.add(gates.M(0, 1, 2))

    initial_psi = (
        random_density_matrix(2**3, backend=backend)
        if density_matrix
        else random_statevector(2**3, backend=backend)
    )
    backend.set_seed(123)
    final_state = backend.execute_circuit(
        noise.apply(circuit), initial_state=backend.np.copy(initial_psi), nshots=nshots
    )
    final_state_samples = final_state.samples() if nshots else None
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(
        target_circuit, initial_state=backend.np.copy(initial_psi), nshots=nshots
    )
    target_final_state_samples = target_final_state.samples() if nshots else None

    if nshots is None:
        backend.assert_allclose(final_state._state, target_final_state._state)
    else:
        backend.assert_allclose(final_state_samples, target_final_state_samples)


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("nshots", [10, 100])
def test_pauli_error(backend, density_matrix, nshots):
    list_paulis = ["X", "Y", "Z"]
    probabilities = np.array([0, 0.2, 0.3])
    zipped = list(zip(list_paulis, probabilities))

    pauli = PauliError(zipped)
    noise = NoiseModel()
    noise.add(pauli, gates.X, 1)
    noise.add(pauli, gates.CNOT)
    noise.add(pauli, gates.Z, (0, 1))

    circuit = Circuit(3, density_matrix=density_matrix)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.Z(1))
    circuit.add(gates.X(1))
    circuit.add(gates.X(2))
    circuit.add(gates.Z(2))
    circuit.add(gates.M(0, 1, 2))

    target_circuit = Circuit(3, density_matrix=density_matrix)
    target_circuit.add(gates.CNOT(0, 1))
    target_circuit.add(gates.PauliNoiseChannel(0, zipped))
    target_circuit.add(gates.PauliNoiseChannel(1, zipped))
    target_circuit.add(gates.Z(1))
    target_circuit.add(gates.PauliNoiseChannel(1, zipped))
    target_circuit.add(gates.X(1))
    target_circuit.add(gates.PauliNoiseChannel(1, zipped))
    target_circuit.add(gates.X(2))
    target_circuit.add(gates.Z(2))
    target_circuit.add(gates.M(0, 1, 2))

    initial_psi = (
        random_density_matrix(2**3, backend=backend)
        if density_matrix
        else random_statevector(2**3, backend=backend)
    )
    backend.set_seed(123)
    final_state = backend.execute_circuit(
        noise.apply(circuit), initial_state=backend.np.copy(initial_psi), nshots=nshots
    )
    final_state_samples = final_state.samples() if nshots else None
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(
        target_circuit, initial_state=backend.np.copy(initial_psi), nshots=nshots
    )
    target_final_state_samples = target_final_state.samples() if nshots else None

    if nshots is None:
        backend.assert_allclose(final_state._state, target_final_state._state)
    else:
        backend.assert_allclose(final_state_samples, target_final_state_samples)


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("nshots", [10, 100])
def test_depolarizing_error(backend, density_matrix, nshots):
    depol = DepolarizingError(0.3)
    noise = NoiseModel()
    noise.add(depol, gates.X, 1)
    noise.add(depol, gates.CNOT)
    noise.add(depol, gates.Z, (0, 1))

    circuit = Circuit(3, density_matrix=density_matrix)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.Z(1))
    circuit.add(gates.X(1))
    circuit.add(gates.X(2))
    circuit.add(gates.Z(2))
    circuit.add(gates.M(0, 1, 2))

    target_circuit = Circuit(3, density_matrix=density_matrix)
    target_circuit.add(gates.CNOT(0, 1))
    target_circuit.add(gates.DepolarizingChannel((0, 1), 0.3))
    target_circuit.add(gates.Z(1))
    target_circuit.add(gates.DepolarizingChannel((1,), 0.3))
    target_circuit.add(gates.X(1))
    target_circuit.add(gates.DepolarizingChannel((1,), 0.3))
    target_circuit.add(gates.X(2))
    target_circuit.add(gates.Z(2))
    target_circuit.add(gates.M(0, 1, 2))

    initial_psi = (
        random_density_matrix(2**3, backend=backend)
        if density_matrix
        else random_statevector(2**3, backend=backend)
    )
    backend.set_seed(123)
    final_state = backend.execute_circuit(
        noise.apply(circuit),
        initial_state=backend.cast(initial_psi, copy=True, dtype=initial_psi.dtype),
        nshots=nshots,
    )
    final_state_samples = final_state.samples() if nshots else None
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(
        target_circuit, initial_state=backend.np.copy(initial_psi), nshots=nshots
    )
    target_final_state_samples = target_final_state.samples() if nshots else None

    if nshots is None:
        backend.assert_allclose(final_state._state, target_final_state._state)
    else:
        backend.assert_allclose(final_state_samples, target_final_state_samples)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_thermal_error(backend, density_matrix):
    if not density_matrix:
        pytest.skip("Thermal Relaxation error is not implemented for state vectors.")
    thermal = ThermalRelaxationError(2, 1, 0.3)
    noise = NoiseModel()
    noise.add(thermal, gates.X, 1)
    noise.add(thermal, gates.CNOT)
    noise.add(thermal, gates.Z, (0, 1))

    circuit = Circuit(3, density_matrix=density_matrix)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.Z(1))
    circuit.add(gates.X(1))
    circuit.add(gates.X(2))
    circuit.add(gates.Z(2))

    target_circuit = Circuit(3, density_matrix=density_matrix)
    target_circuit.add(gates.CNOT(0, 1))
    target_circuit.add(gates.ThermalRelaxationChannel(0, [2, 1, 0.3]))
    target_circuit.add(gates.ThermalRelaxationChannel(1, [2, 1, 0.3]))
    target_circuit.add(gates.Z(1))
    target_circuit.add(gates.ThermalRelaxationChannel(1, [2, 1, 0.3]))
    target_circuit.add(gates.X(1))
    target_circuit.add(gates.ThermalRelaxationChannel(1, [2, 1, 0.3]))
    target_circuit.add(gates.X(2))
    target_circuit.add(gates.Z(2))

    initial_psi = (
        random_density_matrix(2**3, backend=backend)
        if density_matrix
        else random_statevector(2**3, backend=backend)
    )
    backend.set_seed(123)
    final_state = backend.execute_circuit(
        noise.apply(circuit), backend.np.copy(initial_psi)
    )._state
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(
        target_circuit, backend.np.copy(initial_psi)
    )._state

    backend.assert_allclose(final_state, target_final_state)


@pytest.mark.parametrize("density_matrix", [True])
@pytest.mark.parametrize("nshots", [None, 10, 100])
def test_amplitude_damping_error(backend, density_matrix, nshots):
    damping = AmplitudeDampingError(0.3)
    noise = NoiseModel()
    noise.add(damping, gates.X, 1)
    noise.add(damping, gates.CNOT)
    noise.add(damping, gates.Z, (0, 1))

    circuit = Circuit(3, density_matrix=density_matrix)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.Z(1))
    circuit.add(gates.X(1))
    circuit.add(gates.X(2))
    circuit.add(gates.Z(2))
    circuit.add(gates.M(0, 1, 2))

    target_circuit = Circuit(3, density_matrix=density_matrix)
    target_circuit.add(gates.CNOT(0, 1))
    target_circuit.add(gates.AmplitudeDampingChannel(0, 0.3))
    target_circuit.add(gates.AmplitudeDampingChannel(1, 0.3))
    target_circuit.add(gates.Z(1))
    target_circuit.add(gates.AmplitudeDampingChannel(1, 0.3))
    target_circuit.add(gates.X(1))
    target_circuit.add(gates.AmplitudeDampingChannel(1, 0.3))
    target_circuit.add(gates.X(2))
    target_circuit.add(gates.Z(2))
    target_circuit.add(gates.M(0, 1, 2))

    initial_psi = (
        random_density_matrix(2**3, backend=backend)
        if density_matrix
        else random_statevector(2**3, backend=backend)
    )
    backend.set_seed(123)
    final_state = backend.execute_circuit(
        noise.apply(circuit),
        initial_state=backend.cast(initial_psi, copy=True, dtype=initial_psi.dtype),
        nshots=nshots,
    )
    final_state_samples = final_state.samples() if nshots else None
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(
        target_circuit, initial_state=backend.np.copy(initial_psi), nshots=nshots
    )
    target_final_state_samples = target_final_state.samples() if nshots else None

    if nshots is None:
        backend.assert_allclose(final_state._state, target_final_state._state)
    else:
        backend.assert_allclose(final_state_samples, target_final_state_samples)


@pytest.mark.parametrize("density_matrix", [True])
@pytest.mark.parametrize("nshots", [None, 10, 100])
def test_phase_damping_error(backend, density_matrix, nshots):
    damping = PhaseDampingError(0.3)
    noise = NoiseModel()
    noise.add(damping, gates.X, 1)
    noise.add(damping, gates.CNOT)
    noise.add(damping, gates.Z, (0, 1))

    circuit = Circuit(3, density_matrix=density_matrix)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.Z(1))
    circuit.add(gates.X(1))
    circuit.add(gates.X(2))
    circuit.add(gates.Z(2))
    circuit.add(gates.M(0, 1, 2))

    target_circuit = Circuit(3, density_matrix=density_matrix)
    target_circuit.add(gates.CNOT(0, 1))
    target_circuit.add(gates.PhaseDampingChannel(0, 0.3))
    target_circuit.add(gates.PhaseDampingChannel(1, 0.3))
    target_circuit.add(gates.Z(1))
    target_circuit.add(gates.PhaseDampingChannel(1, 0.3))
    target_circuit.add(gates.X(1))
    target_circuit.add(gates.PhaseDampingChannel(1, 0.3))
    target_circuit.add(gates.X(2))
    target_circuit.add(gates.Z(2))
    target_circuit.add(gates.M(0, 1, 2))

    initial_psi = (
        random_density_matrix(2**3, backend=backend)
        if density_matrix
        else random_statevector(2**3, backend=backend)
    )
    backend.set_seed(123)
    final_state = backend.execute_circuit(
        noise.apply(circuit),
        initial_state=backend.cast(initial_psi, copy=True, dtype=initial_psi.dtype),
        nshots=nshots,
    )
    final_state_samples = final_state.samples() if nshots else None
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(
        target_circuit, initial_state=backend.np.copy(initial_psi), nshots=nshots
    )
    target_final_state_samples = target_final_state.samples() if nshots else None

    if nshots is None:
        backend.assert_allclose(final_state._state, target_final_state._state)
    else:
        backend.assert_allclose(final_state_samples, target_final_state_samples)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_readout_error(backend, density_matrix):
    if not density_matrix:
        pytest.skip("Readout error is not implemented for state vectors.")

    nqubits = 1
    d = 2**nqubits

    state = random_density_matrix(d, seed=1, backend=backend)
    P = random_stochastic_matrix(d, seed=1, backend=backend)

    readout = ReadoutError(P)
    noise = NoiseModel()
    noise.add(readout, gates.M, qubits=0)

    circuit = Circuit(nqubits, density_matrix=density_matrix)
    circuit.add(gates.M(0))
    final_state = backend.execute_circuit(
        noise.apply(circuit), initial_state=backend.np.copy(state)
    )

    target_state = gates.ReadoutErrorChannel(0, P).apply_density_matrix(
        backend, backend.np.copy(state), nqubits
    )

    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_reset_error(backend, density_matrix):
    if not density_matrix:
        pytest.skip("Reset error is not implemented for state vectors.")
    reset = ResetError(0.8, 0.2)
    noise = NoiseModel()
    noise.add(reset, gates.X, 1)
    noise.add(reset, gates.CNOT)
    noise.add(reset, gates.Z, (0, 1))

    circuit = Circuit(3, density_matrix=density_matrix)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.Z(1))

    target_circuit = Circuit(3, density_matrix=density_matrix)
    target_circuit.add(gates.CNOT(0, 1))
    target_circuit.add(gates.ResetChannel(0, [0.8, 0.2]))
    target_circuit.add(gates.ResetChannel(1, [0.8, 0.2]))
    target_circuit.add(gates.Z(1))
    target_circuit.add(gates.ResetChannel(1, [0.8, 0.2]))

    initial_psi = (
        random_density_matrix(2**3, backend=backend)
        if density_matrix
        else random_statevector(2**3, backend=backend)
    )
    backend.set_seed(123)
    final_state = backend.execute_circuit(
        noise.apply(circuit), backend.np.copy(initial_psi)
    )._state
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(
        target_circuit, backend.np.copy(initial_psi)
    )._state

    backend.assert_allclose(final_state, target_final_state)


@pytest.mark.parametrize("density_matrix", [True])
@pytest.mark.parametrize("nshots", [None, 10, 100])
def test_custom_error(backend, density_matrix, nshots):
    a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
    a2 = np.sqrt(0.6) * np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    )
    error_channel = gates.KrausChannel([(1,), (0, 2)], [a1, a2])
    custom_error = CustomError(error_channel)

    noise = NoiseModel()
    noise.add(custom_error, gates.X, 1)
    noise.add(custom_error, gates.CNOT)
    noise.add(custom_error, gates.Z, (0, 1))

    circuit = Circuit(3, density_matrix=density_matrix)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.Z(1))
    circuit.add(gates.X(1))
    circuit.add(gates.X(2))
    circuit.add(gates.Z(2))
    circuit.add(gates.M(0, 1, 2))

    target_circuit = Circuit(3, density_matrix=density_matrix)
    target_circuit.add(gates.CNOT(0, 1))
    target_circuit.add(error_channel)
    target_circuit.add(gates.Z(1))
    target_circuit.add(error_channel)
    target_circuit.add(gates.X(1))
    target_circuit.add(error_channel)
    target_circuit.add(gates.X(2))
    target_circuit.add(gates.Z(2))
    target_circuit.add(gates.M(0, 1, 2))

    initial_psi = (
        random_density_matrix(2**3, backend=backend)
        if density_matrix
        else random_statevector(2**3, backend=backend)
    )
    backend.set_seed(123)
    final_state = backend.execute_circuit(
        noise.apply(circuit), initial_state=backend.np.copy(initial_psi), nshots=nshots
    )
    final_state_samples = final_state.samples() if nshots else None
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(
        target_circuit, initial_state=backend.np.copy(initial_psi), nshots=nshots
    )
    target_final_state_samples = target_final_state.samples() if nshots else None

    if nshots is None:
        backend.assert_allclose(final_state._state, target_final_state._state)
    else:
        backend.assert_allclose(final_state_samples, target_final_state_samples)


@pytest.mark.parametrize("density_matrix", [True])
def test_add_condition(backend, density_matrix):
    def condition_pi_2(gate):
        return np.pi / 2 in gate.parameters

    def condition_3_pi_2(gate):
        return 3 * np.pi / 2 in gate.parameters

    reset = ResetError(0.8, 0.2)
    thermal = ThermalRelaxationError(2, 1, 0.3)
    noise = NoiseModel()
    noise.add(reset, gates.RX, conditions=condition_pi_2)
    noise.add(thermal, gates.RX, conditions=condition_3_pi_2)

    with pytest.raises(TypeError):
        noise.add(reset, gates.RX, conditions=2)
    with pytest.raises(TypeError):
        noise.add(reset, gates.RX, conditions=[condition_pi_2, 2])

    circuit = Circuit(3, density_matrix=density_matrix)
    circuit.add(gates.RX(0, np.pi))
    circuit.add(gates.RX(0, np.pi / 2))
    circuit.add(gates.RX(0, np.pi / 3))
    circuit.add(gates.RX(1, np.pi))
    circuit.add(gates.RX(1, 3 * np.pi / 2))
    circuit.add(gates.RX(1, 2 * np.pi / 3))

    target_circuit = Circuit(3, density_matrix=density_matrix)
    target_circuit.add(gates.RX(0, np.pi))
    target_circuit.add(gates.RX(0, np.pi / 2))
    target_circuit.add(gates.ResetChannel(0, [0.8, 0.2]))
    target_circuit.add(gates.RX(0, np.pi / 3))
    target_circuit.add(gates.RX(1, np.pi))
    target_circuit.add(gates.RX(1, 3 * np.pi / 2))
    target_circuit.add(gates.ThermalRelaxationChannel(1, [2, 1, 0.3]))
    target_circuit.add(gates.RX(1, 2 * np.pi / 3))

    initial_psi = (
        random_density_matrix(2**3, backend=backend)
        if density_matrix
        else random_statevector(2**3, backend=backend)
    )
    backend.set_seed(123)
    final_state = backend.execute_circuit(
        noise.apply(circuit), backend.np.copy(initial_psi)
    )._state
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(
        target_circuit, backend.np.copy(initial_psi)
    )._state

    backend.assert_allclose(final_state, target_final_state)


@pytest.mark.parametrize("density_matrix", [True])
def test_gate_independent_noise(backend, density_matrix):
    pauli = PauliError(list(zip(["Y", "Z"], [0.2, 0.3])))
    depol = DepolarizingError(0.3)
    noise = NoiseModel()
    noise.add(pauli)
    noise.add(depol, qubits=0)

    circuit = Circuit(3, density_matrix=density_matrix)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.X(1))
    circuit.add(gates.Z(2))
    circuit.add(gates.M(0, 1, 2))

    target_circuit = Circuit(3, density_matrix=density_matrix)
    target_circuit.add(gates.CNOT(0, 1))
    target_circuit.add(gates.PauliNoiseChannel(0, list(zip(["Y", "Z"], [0.2, 0.3]))))
    target_circuit.add(gates.PauliNoiseChannel(1, list(zip(["Y", "Z"], [0.2, 0.3]))))
    target_circuit.add(gates.DepolarizingChannel((0,), 0.3))
    target_circuit.add(gates.X(1))
    target_circuit.add(gates.PauliNoiseChannel(1, list(zip(["Y", "Z"], [0.2, 0.3]))))
    target_circuit.add(gates.Z(2))
    target_circuit.add(gates.PauliNoiseChannel(2, list(zip(["Y", "Z"], [0.2, 0.3]))))
    target_circuit.add(gates.M(0, 1, 2))

    initial_psi = (
        random_density_matrix(2**3, backend=backend)
        if density_matrix
        else random_statevector(2**3, backend=backend)
    )
    backend.set_seed(123)
    final_state = backend.execute_circuit(
        noise.apply(circuit), initial_state=backend.np.copy(initial_psi)
    )._state
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(
        target_circuit, initial_state=backend.np.copy(initial_psi)
    )._state

    backend.assert_allclose(final_state, target_final_state)


class _Conditions:
    def __init__(self, qubits=None):
        self.qubits = qubits

    def condition_single(self, gate):
        return len(gate.qubits) == 1

    def condition_two(self, gate):
        return len(gate.qubits) == 2

    def condition_qubits(self, gate):
        return gate.qubits == self.qubits


@pytest.mark.parametrize(
    "readout_one_qubit", [0.01, {"0": 0.01, "1": [0.001], "4": (0.02, 0.03)}]
)
@pytest.mark.parametrize("excited_population", [0.0, 0.1])
@pytest.mark.parametrize("gate_times", [(0.1, 0.2)])
@pytest.mark.parametrize(
    "t1, t2", [(0.1, 0.01), ({"1": 0.1, "2": 0.05}, {"1": 0.01, "2": 0.001})]
)
@pytest.mark.parametrize(
    "depolarizing_two_qubit", [0.01, {"0-1": 0.01, "1-3": 0.02, "4-5": 0.05}]
)
@pytest.mark.parametrize(
    "depolarizing_one_qubit", [0.01, {"0": 0.01, "1": 0.02, "4": 0.05}]
)
@pytest.mark.parametrize("nqubits", [5])
def test_ibmq_noise(
    backend,
    nqubits,
    depolarizing_one_qubit,
    depolarizing_two_qubit,
    t1,
    t2,
    gate_times,
    excited_population,
    readout_one_qubit,
):
    ## Since the IBMQNoiseModel inherits the NoiseModel class,
    ## we will test only what is different

    circuit = random_clifford(nqubits, density_matrix=True, backend=backend)
    circuit.add(gates.M(qubit) for qubit in range(nqubits))

    parameters = {
        "t1": t1,
        "t2": t2,
        "depolarizing_one_qubit": depolarizing_one_qubit,
        "depolarizing_two_qubit": depolarizing_two_qubit,
        "excited_population": excited_population,
        "readout_one_qubit": readout_one_qubit,
        "gate_times": gate_times,
    }

    noise_model = IBMQNoiseModel()
    noise_model.from_dict(parameters)
    noisy_circuit = noise_model.apply(circuit)

    noise_model_target = NoiseModel()
    if isinstance(depolarizing_one_qubit, float):
        noise_model_target.add(
            DepolarizingError(depolarizing_one_qubit),
            conditions=_Conditions().condition_single,
        )

    if isinstance(depolarizing_one_qubit, dict):
        for qubit_key, lamb in depolarizing_one_qubit.items():
            noise_model_target.add(
                DepolarizingError(lamb),
                qubits=int(qubit_key),
                conditions=_Conditions().condition_single,
            )

    if isinstance(depolarizing_two_qubit, (float, int)):
        noise_model_target.add(
            DepolarizingError(depolarizing_two_qubit),
            conditions=_Conditions().condition_two,
        )

    if isinstance(depolarizing_two_qubit, dict):
        for key, lamb in depolarizing_two_qubit.items():
            qubits = key.replace(" ", "").split("-")
            qubits = tuple(map(int, qubits))
            noise_model_target.add(
                DepolarizingError(lamb),
                qubits=qubits,
                conditions=[
                    _Conditions().condition_two,
                    _Conditions(qubits).condition_qubits,
                ],
            )

    if isinstance(t1, float):
        noise_model_target.add(
            ThermalRelaxationError(t1, t2, gate_times[0], excited_population),
            conditions=_Conditions().condition_single,
        )
        noise_model_target.add(
            ThermalRelaxationError(t1, t2, gate_times[1], excited_population),
            conditions=_Conditions().condition_two,
        )
    else:
        for qubit in t1.keys():
            noise_model_target.add(
                ThermalRelaxationError(
                    t1[qubit], t2[qubit], gate_times[0], excited_population
                ),
                qubits=int(qubit),
                conditions=_Conditions().condition_single,
            )
            noise_model_target.add(
                ThermalRelaxationError(
                    t1[qubit], t2[qubit], gate_times[1], excited_population
                ),
                qubits=int(qubit),
                conditions=_Conditions().condition_two,
            )

    if isinstance(readout_one_qubit, float):
        probabilities = [
            [1 - readout_one_qubit, readout_one_qubit],
            [readout_one_qubit, 1 - readout_one_qubit],
        ]
        noise_model_target.add(ReadoutError(probabilities), gate=gates.M)
    else:
        for qubit, probs in readout_one_qubit.items():
            if isinstance(probs, (int, float)):
                probs = (probs, probs)
            elif isinstance(probs, (tuple, list)) and len(probs) == 1:
                probs *= 2

            probabilities = [[1 - probs[0], probs[0]], [probs[1], 1 - probs[1]]]
            noise_model_target.add(
                ReadoutError(probabilities),
                gate=gates.M,
                qubits=int(qubit),
            )

    noisy_circuit_target = noise_model_target.apply(circuit)

    assert str(noisy_circuit) == str(noisy_circuit_target)

    backend.set_seed(2024)
    state = backend.execute_circuit(noisy_circuit, nshots=10)

    backend.set_seed(2024)
    state_target = backend.execute_circuit(noisy_circuit_target, nshots=10)

    backend.assert_allclose(state, state_target)
