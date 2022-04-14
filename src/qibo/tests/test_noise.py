import numpy as np
import pytest
from qibo import K, gates
from qibo.models import Circuit
from qibo.noise import *
from qibo.tests.utils import random_density_matrix, random_state

@pytest.mark.parametrize("density_matrix", [False, True])
def test_pauli_error(backend, density_matrix):

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

    initial_psi = random_density_matrix(3) if density_matrix else random_state(3)
    np.random.seed(123)
    K.set_seed(123)
    final_state = noise.apply(circuit)(initial_state=np.copy(initial_psi))
    np.random.seed(123)
    K.set_seed(123)
    target_final_state = target_circuit(initial_state=np.copy(initial_psi))

    K.assert_allclose(final_state, target_final_state)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_thermal_error(backend, density_matrix):

    thermal = ThermalRelaxationError(1, 1, 0.3)
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
    target_circuit.add(gates.ThermalRelaxationChannel(0, 1, 1, 0.3))
    target_circuit.add(gates.ThermalRelaxationChannel(1, 1, 1, 0.3))
    target_circuit.add(gates.Z(1))
    target_circuit.add(gates.ThermalRelaxationChannel(1, 1, 1, 0.3))
    target_circuit.add(gates.X(1))
    target_circuit.add(gates.ThermalRelaxationChannel(1, 1, 1, 0.3))
    target_circuit.add(gates.X(2))
    target_circuit.add(gates.Z(2))

    initial_psi = random_density_matrix(3) if density_matrix else random_state(3)
    np.random.seed(123)
    K.set_seed(123)
    final_state = noise.apply(circuit)(initial_state=np.copy(initial_psi))
    np.random.seed(123)
    K.set_seed(123)
    target_final_state = target_circuit(initial_state=np.copy(initial_psi))

    K.assert_allclose(final_state, target_final_state)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_reset_error(backend, density_matrix):

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
    np.random.seed(123)
    K.set_seed(123)
    final_state = noise.apply(circuit)(initial_state=np.copy(initial_psi))
    np.random.seed(123)
    K.set_seed(123)
    target_final_state = target_circuit(initial_state=np.copy(initial_psi))

    K.assert_allclose(final_state, target_final_state)


