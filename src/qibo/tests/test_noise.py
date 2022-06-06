import numpy as np
import pytest
from qibo import gates
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
    noise.add(pauli, gates.Z, (0,1))

    circuit = Circuit(3, density_matrix=density_matrix)
    circuit.add(gates.CNOT(0,1))
    circuit.add(gates.Z(1))
    circuit.add(gates.X(1))
    circuit.add(gates.X(2))
    circuit.add(gates.Z(2))
    circuit.add(gates.M(0, 1, 2))

    target_circuit = Circuit(3, density_matrix=density_matrix)
    target_circuit.add(gates.CNOT(0,1))
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
    final_state = backend.execute_circuit(noise.apply(circuit), initial_state=np.copy(initial_psi), nshots=nshots)
    final_state_samples = final_state.samples() if nshots else None
    backend.set_seed(123)
    target_final_state = backend.execute_circuit(target_circuit, initial_state=np.copy(initial_psi), nshots=nshots)
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
    noise.add(thermal, gates.Z, (0,1))

    circuit = Circuit(3, density_matrix=density_matrix)
    circuit.add(gates.CNOT(0,1))
    circuit.add(gates.Z(1))
    circuit.add(gates.X(1))
    circuit.add(gates.X(2))
    circuit.add(gates.Z(2))

    target_circuit = Circuit(3, density_matrix=density_matrix)
    target_circuit.add(gates.CNOT(0,1))
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
    noise.add(reset, gates.Z, (0,1))

    circuit = Circuit(3, density_matrix=density_matrix)
    circuit.add(gates.CNOT(0,1))
    circuit.add(gates.Z(1))

    target_circuit = Circuit(3, density_matrix=density_matrix)
    target_circuit.add(gates.CNOT(0,1))
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
