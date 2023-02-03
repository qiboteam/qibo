from collections import Counter

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


@pytest.mark.parametrize("nshots", [100, 1000, 20000])
@pytest.mark.parametrize("idle_qubits", [True, False])
def test_noisy_circuit(backend, nshots, idle_qubits):
    if nshots != 20000:

        circuit = Circuit(3, density_matrix=True)
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
                gates.H(0),
                gates.X(0),
                gates.M(0, 1),
                gates.M(2),
            ]
        )

        params = {
            "t1": (1.0, 1.1, 1.2),
            "t2": (0.7, 0.8, 0.9),
            "gate_time": (1.5, 1.6),
            "excited_population": 0,
            "depolarizing_error": (0.5, 0.6),
            "bitflips_error": ([0.01, 0.02, 0.03], [0.02, 0.03, 0.04]),
            "idle_qubits": idle_qubits,
        }

        noise_model = NoiseModel()
        noise_model.composite(params)
        noisy_circ = noise_model.apply(circuit)

        backend.set_seed(123)
        final_samples = backend.execute_circuit(noisy_circ, nshots=nshots).samples()

        target_circuit = Circuit(3, density_matrix=True)
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
        target_circuit.add(gates.H(0))
        target_circuit.add(gates.DepolarizingChannel((0,), 0.5))
        target_circuit.add(gates.ThermalRelaxationChannel(0, 1.0, 0.7, 1.5, 0))
        if idle_qubits == True:
            target_circuit.add(
                gates.ThermalRelaxationChannel(1, 1.1, 0.8, 1.5, 0)
            )  # dt
        target_circuit.add(gates.CNOT(2, 1))
        target_circuit.add(gates.DepolarizingChannel((1, 2), 0.6))
        target_circuit.add(gates.ThermalRelaxationChannel(1, 1.1, 0.8, 1.6, 0))
        target_circuit.add(gates.ThermalRelaxationChannel(2, 1.2, 0.9, 1.6, 0))
        target_circuit.add(gates.X(0))
        target_circuit.add(gates.DepolarizingChannel((0,), 0.5))
        target_circuit.add(gates.ThermalRelaxationChannel(0, 1.0, 0.7, 1.5, 0))
        if idle_qubits == True:
            target_circuit.add(
                gates.ThermalRelaxationChannel(1, 1.1, 0.8, 1.3, 0)
            )  # dt
        target_circuit.add(gates.M(0, 1))
        target_circuit.add(gates.M(2))

        backend.set_seed(123)
        target_state = backend.execute_circuit(target_circuit, nshots=nshots)
        target_samples = target_state.apply_bitflips(
            p0=[0.01, 0.02, 0.03], p1=[0.02, 0.03, 0.04]
        )

        backend.assert_allclose(final_samples, target_samples)

    else:

        circuit = Circuit(3, density_matrix=True)

        circuit.add(
            [
                gates.CNOT(0, 1),
                gates.RZ(2, -np.pi / 2),
                gates.RZ(0, -np.pi / 2),
                gates.RZ(1, np.pi / 2),
                gates.RX(2, np.pi / 2),
                gates.RX(0, np.pi / 2),
                gates.RX(1, np.pi / 2),
                gates.RZ(2, -np.pi / 2),
                gates.RZ(0, np.pi / 2),
                gates.RZ(1, np.pi / 2),
                gates.CNOT(2, 1),
                gates.CNOT(0, 1),
                gates.CNOT(0, 1),
                gates.RZ(2, -np.pi / 2),
                gates.RZ(0, -np.pi / 2),
                gates.RZ(1, np.pi / 2),
                gates.RX(2, np.pi / 2),
                gates.RX(0, np.pi / 2),
                gates.RX(1, np.pi / 2),
                gates.RZ(2, -np.pi / 2),
                gates.RZ(0, np.pi / 2),
                gates.RZ(1, np.pi / 2),
                gates.CNOT(2, 1),
                gates.CNOT(0, 1),
                gates.CNOT(0, 1),
                gates.RZ(2, -np.pi / 2),
                gates.RZ(0, -np.pi / 2),
                gates.RZ(1, np.pi / 2),
                gates.RX(2, np.pi / 2),
                gates.RX(0, np.pi / 2),
                gates.RX(1, np.pi / 2),
                gates.RZ(2, -np.pi / 2),
                gates.RZ(0, np.pi / 2),
                gates.RZ(1, np.pi / 2),
                gates.CNOT(2, 1),
                gates.CNOT(0, 1),
                gates.CNOT(0, 1),
                gates.RZ(2, -np.pi / 2),
                gates.RZ(0, -np.pi / 2),
                gates.RZ(1, np.pi / 2),
                gates.RX(2, np.pi / 2),
                gates.RX(0, np.pi / 2),
                gates.RX(1, np.pi / 2),
                gates.RZ(2, -np.pi / 2),
                gates.RZ(0, np.pi / 2),
                gates.RZ(1, np.pi / 2),
                gates.CNOT(2, 1),
                gates.CNOT(0, 1),
                gates.CNOT(0, 1),
                gates.RZ(2, -np.pi / 2),
                gates.RZ(0, -np.pi / 2),
                gates.RZ(1, np.pi / 2),
                gates.RX(2, np.pi / 2),
                gates.RX(0, np.pi / 2),
                gates.RX(1, np.pi / 2),
                gates.RZ(2, -np.pi / 2),
                gates.RZ(0, np.pi / 2),
                gates.RZ(1, np.pi / 2),
                gates.CNOT(2, 1),
                gates.CNOT(0, 1),
                gates.M(0),
                gates.M(1),
                gates.M(2),
            ]
        )
        result = circuit(nshots=2000)
        f = {0: 2908, 1: 2504, 2: 2064, 3: 2851, 4: 2273, 5: 2670, 6: 2170, 7: 2560}
        counts = Counter({k: v for k, v in f.items()})
        result._frequencies = counts
        result.frequencies()

        params = {"idle_qubits": True}
        bounds = [
            True,
            [
                [50e-6] * 6 + [50e-9] * 2 + [1e-5] + [1e-3] * 7,
                [500e-6] * 6 + [300e-9] * 2 + [0.2] * 2 + [0.3] * 6,
            ],
        ]
        for bound in bounds:
            noise_model = CompositeNoiseModel(params)
            noise_model.fit(result, bounds=bound, backend=backend)
            backend.assert_allclose(
                noise_model.hellinger, 1, rtol=noise_model.hellinger0["shot_error"]
            )
