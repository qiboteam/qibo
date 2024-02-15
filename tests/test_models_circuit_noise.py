"""Test :class:`qibo.models.circuit.Circuit` for density matrix and noise simulation."""

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.config import PRECISION_TOL
from qibo.quantum_info import (
    random_clifford,
    random_density_matrix,
    random_statevector,
    random_unitary,
)


def test_pauli_noise_channel(backend):
    from qibo import matrices

    c = Circuit(2, density_matrix=True)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.PauliNoiseChannel(0, list(zip(["X", "Z"], [0.5, 0.3]))))
    c.add(gates.PauliNoiseChannel(1, list(zip(["Y", "Z"], [0.1, 0.3]))))
    final_rho = backend.execute_circuit(c)._state

    psi = np.ones(4) / 2
    rho = np.outer(psi, psi.conj())
    m1 = np.kron(matrices.X, matrices.I)
    m2 = np.kron(matrices.Z, matrices.I)

    rho = 0.2 * rho + 0.5 * m1.dot(rho.dot(m1)) + 0.3 * m2.dot(rho.dot(m2))
    m1 = np.kron(matrices.I, matrices.Y)
    m2 = np.kron(matrices.I, matrices.Z)
    rho = 0.6 * rho + 0.1 * m1.dot(rho.dot(m1)) + 0.3 * m2.dot(rho.dot(m2))
    backend.assert_allclose(final_rho, rho)


def test_noisy_circuit_reexecution(backend):
    c = Circuit(2, density_matrix=True)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.PauliNoiseChannel(0, [("X", 0.5)]))
    c.add(gates.PauliNoiseChannel(1, [("Z", 0.3)]))
    final_rho = backend.execute_circuit(c).state()
    final_rho2 = backend.execute_circuit(c).state()
    backend.assert_allclose(final_rho, final_rho2)


def test_circuit_with_pauli_noise_gates():
    c = Circuit(2, density_matrix=True)
    c.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])
    noisy_c = c.with_pauli_noise(list(zip(["X", "Y", "Z"], [0.1, 0.2, 0.3])))
    assert noisy_c.depth == 4
    assert noisy_c.ngates == 7
    for i in [1, 3, 5, 6]:
        assert noisy_c.queue[i].__class__.__name__ == "PauliNoiseChannel"


def test_circuit_with_pauli_noise_execution(backend):
    c = Circuit(2, density_matrix=True)
    c.add([gates.H(0), gates.H(1)])
    noisy_c = c.with_pauli_noise(list(zip(["X", "Y", "Z"], [0.1, 0.2, 0.3])))
    final_state = backend.execute_circuit(noisy_c).state()

    target_c = Circuit(2, density_matrix=True)
    target_c.add(gates.H(0))
    target_c.add(
        gates.PauliNoiseChannel(0, list(zip(["X", "Y", "Z"], [0.1, 0.2, 0.3])))
    )
    target_c.add(gates.H(1))
    target_c.add(
        gates.PauliNoiseChannel(1, list(zip(["X", "Y", "Z"], [0.1, 0.2, 0.3])))
    )
    target_state = backend.execute_circuit(target_c).state()
    backend.assert_allclose(final_state, target_state)


def test_circuit_with_pauli_noise_measurements(backend):
    c = Circuit(2, density_matrix=True)
    c.add([gates.H(0), gates.H(1)])
    c.add(gates.M(0))
    noisy_c = c.with_pauli_noise(list(zip(["X", "Y", "Z"], [0.1, 0.1, 0.1])))
    final_state = backend.execute_circuit(noisy_c).state()

    target_c = Circuit(2, density_matrix=True)
    target_c.add(gates.H(0))
    target_c.add(
        gates.PauliNoiseChannel(0, list(zip(["X", "Y", "Z"], [0.1, 0.1, 0.1])))
    )
    target_c.add(gates.H(1))
    target_c.add(
        gates.PauliNoiseChannel(1, list(zip(["X", "Y", "Z"], [0.1, 0.1, 0.1])))
    )
    target_state = backend.execute_circuit(target_c).state()
    backend.assert_allclose(final_state, target_state)


def test_circuit_with_pauli_noise_noise_map(backend):
    noise_map = {
        0: list(zip(["X", "Y", "Z"], [0.1, 0.2, 0.1])),
        1: list(zip(["X", "Y", "Z"], [0.2, 0.3, 0.0])),
        2: list(zip(["X", "Y", "Z"], [0.0, 0.0, 0.0])),
    }

    c = Circuit(3, density_matrix=True)
    c.add([gates.H(0), gates.H(1), gates.X(2)])
    c.add(gates.M(2))
    noisy_c = c.with_pauli_noise(noise_map)
    final_state = backend.execute_circuit(noisy_c).state()

    target_c = Circuit(3, density_matrix=True)
    target_c.add(gates.H(0))
    target_c.add(
        gates.PauliNoiseChannel(0, list(zip(["X", "Y", "Z"], [0.1, 0.2, 0.1])))
    )
    target_c.add(gates.H(1))
    target_c.add(
        gates.PauliNoiseChannel(1, list(zip(["X", "Y", "Z"], [0.2, 0.3, 0.0])))
    )
    target_c.add(gates.X(2))
    target_state = backend.execute_circuit(target_c).state()
    backend.assert_allclose(final_state, target_state)


def test_circuit_with_pauli_noise_errors():
    c = Circuit(2, density_matrix=True)
    c.add([gates.H(0), gates.H(1), gates.PauliNoiseChannel(0, [("X", 0.2)])])
    with pytest.raises(ValueError):
        noisy_c = c.with_pauli_noise(list(zip(["X", "Y"], [0.2, 0.3])))
    c = Circuit(2, density_matrix=True)
    c.add([gates.H(0), gates.H(1)])
    with pytest.raises(ValueError):
        noisy_c = c.with_pauli_noise({0: list(zip(["X", "Y", "Z"], [0.2, 0.3, 0.1]))})
    with pytest.raises(TypeError):
        noisy_c = c.with_pauli_noise({0, 1})


def test_density_matrix_circuit_measurement(backend):
    """Check measurement gate on density matrices using circuit."""
    from .test_measurements import assert_register_result, assert_result

    state = np.zeros(16)
    state[0] = 1
    init_rho = np.outer(state, state.conj())

    c = Circuit(4, density_matrix=True)
    c.add(gates.X(1))
    c.add(gates.X(3))
    c.add(gates.M(0, 1, register_name="A"))
    c.add(gates.M(3, 2, register_name="B"))
    result = backend.execute_circuit(c, init_rho, nshots=100)

    target_binary_samples = np.zeros((100, 4))
    target_binary_samples[:, 1] = 1
    target_binary_samples[:, 2] = 1
    assert_result(
        backend,
        result,
        decimal_samples=6 * np.ones((100,)),
        binary_samples=target_binary_samples,
        decimal_frequencies={6: 100},
        binary_frequencies={"0110": 100},
    )

    target = {}
    target["decimal_samples"] = {"A": np.ones((100,)), "B": 2 * np.ones((100,))}
    target["binary_samples"] = {"A": np.zeros((100, 2)), "B": np.zeros((100, 2))}
    target["binary_samples"]["A"][:, 1] = 1
    target["binary_samples"]["B"][:, 0] = 1
    target["decimal_frequencies"] = {"A": {1: 100}, "B": {2: 100}}
    target["binary_frequencies"] = {"A": {"01": 100}, "B": {"10": 100}}
    assert_register_result(backend, result, **target)


def test_circuit_add_sampling(backend):
    """Check measurements when simulating added circuits with noise"""
    # Create random noisy circuit and add noiseless inverted circuit
    gates_set = [gates.X, gates.Y, gates.Z, gates.H, gates.S, gates.SDG, gates.I]
    circ = Circuit(1)
    circ_no_noise = Circuit(1)

    for _ in range(10):
        new_gate = np.random.choice(gates_set)(0)
        circ.add(gates.PauliNoiseChannel(0, [("Z", 0.01)]))
        circ.add(new_gate)
        circ_no_noise.add(new_gate)

    circ.add(gates.PauliNoiseChannel(0, [("Z", 0.01)]))
    circ += circ_no_noise.invert()
    measurement = circ.add(gates.M(0))

    # Sampling using 10 shots
    np.random.seed(123)
    backend.set_seed(123)
    samples = backend.execute_circuit(circ, nshots=10).samples()

    # Sampling using 1 shot in for loop
    target_samples = []
    backend.set_seed(123)
    np.random.seed(123)
    for _ in range(10):
        measurement.reset()
        result = backend.execute_circuit(circ, nshots=1)
        target_samples.append(result.samples())

    target_samples = np.stack(target_samples)

    backend.assert_allclose(samples, target_samples[:, 0])


@pytest.mark.parametrize("nqubits", [2, 4, 6])
def test_probabilities_repeated_execution(backend, nqubits):
    probabilities = list(np.random.rand(nqubits + 1)) + [1.0]
    probabilities /= np.sum(probabilities)

    unitaries = [random_unitary(2**1, backend=backend) for _ in range(nqubits)]
    unitaries += [random_unitary(2**nqubits, backend=backend)]

    qubits_list = [(q,) for q in range(nqubits)]
    qubits_list += [tuple(q for q in range(nqubits))]

    circuit = random_clifford(nqubits, return_circuit=True, backend=backend)
    circuit.add(gates.UnitaryChannel(qubits_list, list(zip(probabilities, unitaries))))
    circuit.add(gates.M(*range(nqubits)))

    circuit_density_matrix = circuit.copy(deep=True)
    circuit_density_matrix.density_matrix = True

    state = random_density_matrix(2**nqubits, backend=backend)

    # set has_collapse=True just to trigger the repeated execution
    # with density_matrix=True
    circuit.has_collapse = True
    # if we don't set density_matrix=True a MeasurementOutcomes object
    # is returned, which doesn't have any probabilities() method.
    circuit.density_matrix = True

    result = backend.execute_circuit_repeated(
        circuit, initial_state=state, nshots=int(1e2)
    )
    result = result.probabilities()

    result_density_matrix = backend.execute_circuit(
        circuit_density_matrix,
        initial_state=state,
        nshots=int(1e2),
    )
    result_density_matrix = result_density_matrix.probabilities()

    backend.assert_allclose(result, result_density_matrix, rtol=2e-2, atol=5e-3)
