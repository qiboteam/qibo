import numpy as np
import pytest
from qibo import K, gates
from qibo.models import Circuit
from qibo.noise import *
from qibo.tests.utils import random_density_matrix


def test_pauli_error(backend):

    pauli = PauliError(0, 0.2, 0.3)

    noise = NoiseModel()
    noise.add(pauli, "x", 1)
    noise.add(pauli, "cx")
    noise.add(pauli, "z", (0,1))

    circuit = Circuit(3)
    circuit.add(gates.CNOT(0,1))
    circuit.add(gates.Z(1))
    circuit.add(gates.X(1))
    circuit.add(gates.X(2))
    circuit.add(gates.Z(2))

    target_circuit = Circuit(3)
    target_circuit.add(gates.PauliNoiseChannel(0, 0, 0.2, 0.3))
    target_circuit.add(gates.PauliNoiseChannel(1, 0, 0.2, 0.3))
    target_circuit.add(gates.CNOT(0,1))
    target_circuit.add(gates.PauliNoiseChannel(1, 0, 0.2, 0.3))
    target_circuit.add(gates.Z(1))
    target_circuit.add(gates.PauliNoiseChannel(1, 0, 0.2, 0.3))
    target_circuit.add(gates.X(1))
    target_circuit.add(gates.X(2))
    target_circuit.add(gates.Z(2))

    np.random.seed(123)
    K.set_seed(123)

    final_state = noise.apply(circuit)()
    target_final_state = target_circuit()

    K.assert_allclose(final_state, target_final_state)

def test_thermal_error(backend):

    thermal = ThermalRelaxationError(1, 1, 0.3)

    noise = NoiseModel()
    noise.add(thermal, "x", 1)
    noise.add(thermal, "cx")
    noise.add(thermal, "z", (0,1))

    circuit = Circuit(3)
    circuit.add(gates.CNOT(0,1))
    circuit.add(gates.Z(1))
    circuit.add(gates.X(1))
    circuit.add(gates.X(2))
    circuit.add(gates.Z(2))

    target_circuit = Circuit(3)
    target_circuit.add(gates.ThermalRelaxationChannel(0, 1, 0.1, 0.2, 0.3))
    target_circuit.add(gates.ThermalRelaxationChannel(1, 1, 0.1, 0.2, 0.3))
    target_circuit.add(gates.CNOT(0,1))
    target_circuit.add(gates.ThermalRelaxationChannel(1, 1, 0.1, 0.2, 0.3))
    target_circuit.add(gates.Z(1))
    target_circuit.add(gates.ThermalRelaxationChannel(1, 1, 0.1, 0.2, 0.3))
    target_circuit.add(gates.X(1))
    target_circuit.add(gates.X(2))
    target_circuit.add(gates.Z(2))

    np.random.seed(123)
    K.set_seed(123)

    final_state = noise.apply(circuit)()
    target_final_state = target_circuit()

    K.assert_allclose(final_state, target_final_state)

def test_reset_error(backend):

    reset = ResetError(0.8, 0.2)

    noise = NoiseModel()
    noise.add(reset, "x", 1)
    noise.add(reset, "cx")
    noise.add(reset, "z", (0,1))

    circuit = Circuit(3)
    circuit.add(gates.CNOT(0,1))
    circuit.add(gates.Z(1))

    target_circuit = Circuit(3)
    target_circuit.add(gates.ResetChannel(0, 0.5, 0.2))
    target_circuit.add(gates.ResetChannel(1, 0.5, 0.2))
    target_circuit.add(gates.CNOT(0,1))
    target_circuit.add(gates.ResetChannel(1, 0.5, 0.2))
    target_circuit.add(gates.Z(1))


    np.random.seed(123)
    K.set_seed(123)

    final_state = noise.apply(circuit)()
    target_final_state = target_circuit()

    K.assert_allclose(final_state, target_final_state)


