import numpy as np
import pytest

from qibo import gates, noise_model
from qibo.models import Circuit
from qibo.noise import *
from qibo.tests.utils import random_density_matrix, random_state


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
def test_noise_model(backend, density_matrix, nshots):
    if not density_matrix:
        pytest.skip("Thermal Relaxation Error is not implemented for state vectors.")
    circuit = Circuit(3, density_matrix=density_matrix)
    circuit.add(
        [
            gates.H(0),
            gates.X(0),
            gates.CNOT(2, 1),
            gates.Z(2),
            gates.Z(0),
            gates.H(1),
            gates.H(2),
            gates.CNOT(2, 1),
            gates.M(0, 1),
            gates.M(2),
        ]
    )

    params = {
        "t1": (1.0, 1.1, 1.2),
        "t2": (0.7, 0.8, 0.9),
        "gate time": (1.5, 1.6),
        "excited population": 0,
        "depolarizing error": (0.5, 0.6),
        "bitflips error": (0.01, 0.02, 0.015),
    }

    target_circuit = Circuit(3, density_matrix=density_matrix)
    target_circuit.add(gates.H(0))
    target_circuit.add(gates.DepolarizingChannel((0,), 0.5))
    target_circuit.add(gates.ThermalRelaxationChannel(0, 1.0, 0.7, 1.5, 0))
    target_circuit.add(gates.CNOT(2, 1))
    target_circuit.add(gates.DepolarizingChannel((1, 2), 0.6))
    target_circuit.add(gates.ThermalRelaxationChannel(1, 1.1, 0.8, 1.6, 0))
    target_circuit.add(gates.ThermalRelaxationChannel(2, 1.2, 0.9, 1.6, 0))
    target_circuit.add(gates.X(0))
    target_circuit.add(gates.DepolarizingChannel((0,), 0.5))
    target_circuit.add(gates.ThermalRelaxationChannel(0, 1.0, 0.7, 1.5, 0))
    target_circuit.add(gates.H(1))
    target_circuit.add(gates.DepolarizingChannel((1,), 0.5))
    target_circuit.add(gates.ThermalRelaxationChannel(1, 1.1, 0.8, 1.5, 0))
    target_circuit.add(gates.Z(2))
    target_circuit.add(gates.DepolarizingChannel((2,), 0.5))
    target_circuit.add(gates.ThermalRelaxationChannel(2, 1.2, 0.9, 1.5, 0))
    target_circuit.add(gates.Z(0))
    target_circuit.add(gates.DepolarizingChannel((0,), 0.5))
    target_circuit.add(gates.ThermalRelaxationChannel(0, 1.0, 0.7, 1.5, 0))
    target_circuit.add(gates.H(2))
    target_circuit.add(gates.DepolarizingChannel((2,), 0.5))
    target_circuit.add(gates.ThermalRelaxationChannel(2, 1.2, 0.9, 1.5, 0))
    target_circuit.add(gates.ThermalRelaxationChannel(1, 1.1, 0.8, 1.5, 0))  # dt
    target_circuit.add(gates.CNOT(2, 1))
    target_circuit.add(gates.DepolarizingChannel((1, 2), 0.6))
    target_circuit.add(gates.ThermalRelaxationChannel(1, 1.1, 0.8, 1.6, 0))
    target_circuit.add(gates.ThermalRelaxationChannel(2, 1.2, 0.9, 1.6, 0))
    target_circuit.add(gates.ThermalRelaxationChannel(0, 1.0, 0.7, 1.7, 0))  # dt
    target_circuit.add(gates.PauliNoiseChannel(0, px=0.01))
    target_circuit.add(gates.PauliNoiseChannel(1, px=0.02))
    target_circuit.add(gates.PauliNoiseChannel(2, px=0.015))
    target_circuit.add(gates.M(0, 1))
    target_circuit.add(gates.M(2))

    initial_psi = random_density_matrix(3) if density_matrix else random_state(3)
    backend.set_seed(123)
    final_state = backend.execute_circuit(
        noise_model.noise_model(circuit, params),
        initial_state=np.copy(initial_psi),
        nshots=nshots,
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
