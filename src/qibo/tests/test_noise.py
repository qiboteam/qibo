import numpy as np
import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.noise import *
from qibo.tests.utils import random_density_matrix, random_state


@pytest.mark.parametrize("density_matrix", [True])
@pytest.mark.parametrize("nshots", [None, 10, 100])
def test_custom_error(backend, density_matrix, nshots):
    a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
    a2 = np.sqrt(0.6) * np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    )
    error_channel = gates.KrausChannel([([1], a1), ([0, 2], a2)])
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

    initial_psi = random_density_matrix(3) if density_matrix else random_state(3)
    backend.set_seed(123)
    final_state = backend.execute_circuit(
        noise.apply(circuit), initial_state=np.copy(initial_psi), nshots=nshots
    )
    final_state_samples = final_state.samples() if nshots else None
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(
        target_circuit, initial_state=np.copy(initial_psi), nshots=nshots
    )
    target_final_state_samples = target_final_state.samples() if nshots else None

    if nshots is None:
        backend.assert_allclose(final_state, target_final_state)
    else:
        backend.assert_allclose(final_state_samples, target_final_state_samples)


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("nshots", [None, 10, 100])
def test_pauli_error(backend, density_matrix, nshots):
    pauli = PauliError(0, 0.2, 0.3)
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
    target_circuit.add(gates.PauliNoiseChannel(0, 0, 0.2, 0.3))
    target_circuit.add(gates.PauliNoiseChannel(1, 0, 0.2, 0.3))
    target_circuit.add(gates.Z(1))
    target_circuit.add(gates.PauliNoiseChannel(1, 0, 0.2, 0.3))
    target_circuit.add(gates.X(1))
    target_circuit.add(gates.PauliNoiseChannel(1, 0, 0.2, 0.3))
    target_circuit.add(gates.X(2))
    target_circuit.add(gates.Z(2))
    target_circuit.add(gates.M(0, 1, 2))

    initial_psi = random_density_matrix(3) if density_matrix else random_state(3)
    backend.set_seed(123)
    final_state = backend.execute_circuit(
        noise.apply(circuit), initial_state=np.copy(initial_psi), nshots=nshots
    )
    final_state_samples = final_state.samples() if nshots else None
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(
        target_circuit, initial_state=np.copy(initial_psi), nshots=nshots
    )
    target_final_state_samples = target_final_state.samples() if nshots else None

    if nshots is None:
        backend.assert_allclose(final_state, target_final_state)
    else:
        backend.assert_allclose(final_state_samples, target_final_state_samples)


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("nshots", [None, 10, 100])
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

    initial_psi = random_density_matrix(3) if density_matrix else random_state(3)
    backend.set_seed(123)
    final_state = backend.execute_circuit(
        noise.apply(circuit), initial_state=np.copy(initial_psi), nshots=nshots
    )
    final_state_samples = final_state.samples() if nshots else None
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(
        target_circuit, initial_state=np.copy(initial_psi), nshots=nshots
    )
    target_final_state_samples = target_final_state.samples() if nshots else None

    if nshots is None:
        backend.assert_allclose(final_state, target_final_state)
    else:
        backend.assert_allclose(final_state_samples, target_final_state_samples)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_thermal_error(backend, density_matrix):
    if not density_matrix:
        pytest.skip("Reset error is not implemented for state vectors.")
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
    target_circuit.add(gates.ThermalRelaxationChannel(0, 2, 1, 0.3))
    target_circuit.add(gates.ThermalRelaxationChannel(1, 2, 1, 0.3))
    target_circuit.add(gates.Z(1))
    target_circuit.add(gates.ThermalRelaxationChannel(1, 2, 1, 0.3))
    target_circuit.add(gates.X(1))
    target_circuit.add(gates.ThermalRelaxationChannel(1, 2, 1, 0.3))
    target_circuit.add(gates.X(2))
    target_circuit.add(gates.Z(2))

    initial_psi = random_density_matrix(3) if density_matrix else random_state(3)
    backend.set_seed(123)
    final_state = backend.execute_circuit(noise.apply(circuit), np.copy(initial_psi))
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(target_circuit, np.copy(initial_psi))

    backend.assert_allclose(final_state, target_final_state)


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
    target_circuit.add(gates.ResetChannel(0, 0.8, 0.2))
    target_circuit.add(gates.ResetChannel(1, 0.8, 0.2))
    target_circuit.add(gates.Z(1))
    target_circuit.add(gates.ResetChannel(1, 0.8, 0.2))

    initial_psi = random_density_matrix(3) if density_matrix else random_state(3)
    backend.set_seed(123)
    final_state = backend.execute_circuit(noise.apply(circuit), np.copy(initial_psi))
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(target_circuit, np.copy(initial_psi))

    backend.assert_allclose(final_state, target_final_state)


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("nshots", [None, 10, 100])
def test_unitary_error(backend, density_matrix, nshots):
    u1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    u2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    probabilities = [0.3, 0.7]

    with pytest.raises(ValueError):
        UnitaryError(probabilities, [u1, np.array([[1, 0], [0, 1]])])

    unitary_error = UnitaryError(probabilities, [u1, u2])

    noise = NoiseModel()
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
    target_circuit.add(
        gates.UnitaryChannel(probabilities, [([0, 1], u1), ([0, 1], u2)])
    )
    target_circuit.add(gates.Z(1))
    target_circuit.add(gates.X(1))
    target_circuit.add(gates.X(2))
    target_circuit.add(gates.Z(2))
    target_circuit.add(gates.M(0, 1, 2))

    initial_psi = random_density_matrix(3) if density_matrix else random_state(3)
    backend.set_seed(123)
    final_state = backend.execute_circuit(
        noise.apply(circuit), initial_state=np.copy(initial_psi), nshots=nshots
    )
    final_state_samples = final_state.samples() if nshots else None
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(
        target_circuit, initial_state=np.copy(initial_psi), nshots=nshots
    )
    target_final_state_samples = target_final_state.samples() if nshots else None

    if nshots is None:
        backend.assert_allclose(final_state, target_final_state)
    else:
        backend.assert_allclose(final_state_samples, target_final_state_samples)


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
    target_circuit.add(gates.KrausChannel([([0], k1), ([0], k2)]))
    target_circuit.add(gates.KrausChannel([([1], k1), ([1], k2)]))
    target_circuit.add(gates.Z(1))
    target_circuit.add(gates.KrausChannel([([1], k1), ([1], k2)]))
    target_circuit.add(gates.X(1))
    target_circuit.add(gates.KrausChannel([([1], k1), ([1], k2)]))
    target_circuit.add(gates.X(2))
    target_circuit.add(gates.Z(2))
    target_circuit.add(gates.M(0, 1, 2))

    initial_psi = random_density_matrix(3) if density_matrix else random_state(3)
    backend.set_seed(123)
    final_state = backend.execute_circuit(
        noise.apply(circuit), initial_state=np.copy(initial_psi), nshots=nshots
    )
    final_state_samples = final_state.samples() if nshots else None
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(
        target_circuit, initial_state=np.copy(initial_psi), nshots=nshots
    )
    target_final_state_samples = target_final_state.samples() if nshots else None

    if nshots is None:
        backend.assert_allclose(final_state, target_final_state)
    else:
        backend.assert_allclose(final_state_samples, target_final_state_samples)
