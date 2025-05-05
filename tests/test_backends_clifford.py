"""Tests for Clifford backend."""

from itertools import product

import numpy as np
import pytest

from qibo import Circuit, gates, get_backend, set_backend
from qibo.backends import CliffordBackend, NumpyBackend, clifford
from qibo.backends.clifford import _get_engine_name
from qibo.noise import DepolarizingError, NoiseModel, PauliError
from qibo.quantum_info.random_ensembles import random_clifford

numpy_bkd = NumpyBackend()


def construct_clifford_backend(backend):
    if backend.__class__.__name__ in (
        "TensorflowBackend",
        "PyTorchBackend",
        "CuQuantumBackend",
    ):
        with pytest.raises(NotImplementedError):
            clifford_backend = CliffordBackend(backend.name)
        pytest.skip("Clifford backend not defined for this engine.")

    return CliffordBackend(_get_engine_name(backend))


def test_set_backend(backend):
    clifford_bkd = construct_clifford_backend(backend)
    platform = _get_engine_name(backend)
    set_backend("clifford", platform=platform)
    assert isinstance(get_backend(), CliffordBackend)
    global_platform = get_backend().platform
    assert global_platform == platform


def test_global_backend(backend):
    construct_clifford_backend(backend)
    set_backend(backend.name, platform=backend.platform)
    clifford_bkd = CliffordBackend()
    target = get_backend().name if backend.name == "numpy" else get_backend().platform
    assert clifford_bkd.platform == target


THETAS_1Q = [
    th + 2 * i * np.pi for i in range(2) for th in [0, np.pi / 2, np.pi, 3 * np.pi / 2]
]

AXES = ["RX", "RY", "RZ"]


@pytest.mark.parametrize("axis,theta", list(product(AXES, THETAS_1Q)))
def test_rotations_1q(backend, theta, axis):
    clifford_bkd = construct_clifford_backend(backend)
    c = Circuit(3, density_matrix=True)
    qubits = np.random.randint(3, size=2)
    H_qubits = np.random.choice(range(3), size=2, replace=False)
    c.add(gates.H(q) for q in H_qubits)
    c.add(getattr(gates, axis)(qubits[0], theta=theta))
    c.add(getattr(gates, axis)(qubits[1], theta=theta))
    clifford_state = clifford_bkd.execute_circuit(c).state()
    numpy_state = numpy_bkd.execute_circuit(c).state()
    numpy_state = backend.cast(numpy_state)
    backend.assert_allclose(clifford_state, numpy_state, atol=1e-8)


THETAS_2Q = [i * np.pi for i in range(4)]


@pytest.mark.parametrize("axis,theta", list(product(AXES, THETAS_2Q)))
def test_rotations_2q(backend, theta, axis):
    clifford_bkd = construct_clifford_backend(backend)
    c = Circuit(3, density_matrix=True)
    qubits_0 = np.random.choice(range(3), size=2, replace=False)
    qubits_1 = np.random.choice(range(3), size=2, replace=False)
    H_qubits = np.random.choice(range(3), size=2, replace=False)
    c.add(gates.H(q) for q in H_qubits)
    c.add(getattr(gates, f"C{axis}")(*qubits_0, theta=theta))
    c.add(getattr(gates, f"C{axis}")(*qubits_1, theta=theta))
    clifford_state = clifford_bkd.execute_circuit(c).state()
    numpy_state = numpy_bkd.execute_circuit(c).state()
    numpy_state = backend.cast(numpy_state)
    backend.assert_allclose(clifford_state, numpy_state, atol=1e-8)


SINGLE_QUBIT_CLIFFORDS = ["I", "H", "S", "Z", "X", "Y", "SX", "SDG", "SXDG"]


@pytest.mark.parametrize("gate", SINGLE_QUBIT_CLIFFORDS)
def test_single_qubit_gates(backend, gate):
    clifford_bkd = construct_clifford_backend(backend)
    c = Circuit(3, density_matrix=True)
    qubits = np.random.randint(3, size=2)
    H_qubits = np.random.choice(range(3), size=2, replace=False)
    c.add(gates.H(q) for q in H_qubits)
    c.add(getattr(gates, gate)(qubits[0]))
    c.add(getattr(gates, gate)(qubits[1]))
    clifford_state = clifford_bkd.execute_circuit(c).state()
    numpy_state = numpy_bkd.execute_circuit(c).state()
    numpy_state = backend.cast(numpy_state)
    backend.assert_allclose(clifford_state, numpy_state, atol=1e-8)


TWO_QUBITS_CLIFFORDS = ["CNOT", "CZ", "CY", "SWAP", "iSWAP", "FSWAP", "ECR"]


@pytest.mark.parametrize("gate", TWO_QUBITS_CLIFFORDS)
def test_two_qubits_gates(backend, gate):
    clifford_bkd = construct_clifford_backend(backend)
    c = Circuit(5, density_matrix=True)
    qubits = np.random.choice(range(5), size=4, replace=False).reshape(2, 2)
    H_qubits = np.random.choice(range(5), size=3, replace=False)
    c.add(gates.H(q) for q in H_qubits)
    c.add(getattr(gates, gate)(*qubits[0]))
    c.add(getattr(gates, gate)(*qubits[1]))
    clifford_state = clifford_bkd.execute_circuit(c).state()
    numpy_state = numpy_bkd.execute_circuit(c).state()
    numpy_state = backend.cast(numpy_state)
    backend.assert_allclose(clifford_state, numpy_state, atol=1e-8)


@pytest.mark.parametrize("prob_qubits", [1, 2])
@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize(
    "seed",
    [
        15,
        25,
    ],
)
def test_random_clifford_circuit(backend, prob_qubits, binary, seed):
    np.random.seed(seed)
    numpy_bkd.set_seed(seed)
    backend.set_seed(seed)
    nqubits, nshots = 3, 1000
    clifford_bkd = construct_clifford_backend(backend)
    clifford_bkd.set_seed(seed)

    circuit = random_clifford(nqubits, seed=seed, density_matrix=True, backend=backend)
    circuit_copy = circuit.copy()
    MEASURED_QUBITS = np.random.choice(list(range(nqubits)), size=2, replace=False)
    prob_qubits = np.random.choice(MEASURED_QUBITS, size=prob_qubits, replace=False)
    circuit.add(gates.M(*MEASURED_QUBITS))
    circuit_copy.add(gates.M(*MEASURED_QUBITS))
    numpy_result = numpy_bkd.execute_circuit(circuit, nshots=nshots)
    clifford_result = clifford_bkd.execute_circuit(circuit_copy, nshots=nshots)
    backend.assert_allclose(backend.cast(numpy_result.state()), clifford_result.state())
    clifford_prob = clifford_result.probabilities(prob_qubits)
    numpy_prob = backend.cast(numpy_result.probabilities(prob_qubits))
    backend.assert_allclose(
        numpy_prob,
        clifford_prob,
        atol=1e-1,
    )

    numpy_freq = numpy_result.frequencies(binary)
    clifford_freq = clifford_result.frequencies(binary)
    assert set(numpy_freq.keys()) == set(clifford_freq.keys())
    clifford_freq = {state: clifford_freq[state] for state in numpy_freq.keys()}

    for np_count, clif_count in zip(numpy_freq.values(), clifford_freq.values()):
        backend.assert_allclose(np_count / nshots, clif_count / nshots, atol=1e-1)


@pytest.mark.parametrize("seed", [2024])
def test_collapsing_measurements(backend, seed):
    clifford_bkd = construct_clifford_backend(backend)
    np.random.seed(seed)
    numpy_bkd.set_seed(seed)
    backend.set_seed(seed)
    clifford_bkd.set_seed(seed)
    gate_queue = random_clifford(
        3, density_matrix=False, seed=seed, backend=backend
    ).queue
    local_state = np.random.default_rng(seed)
    measured_qubits = local_state.choice(range(3), size=2, replace=False)
    c1 = Circuit(3)
    c2 = Circuit(3, density_matrix=True)
    for i, g in enumerate(gate_queue):
        if i == int(len(gate_queue) / 2):
            c1.add(gates.M(*measured_qubits))
            c2.add(gates.M(*measured_qubits))
        else:
            c1.add(g)
            c2.add(g)
    c1.add(gates.M(*range(3)))
    c2.add(gates.M(*range(3)))

    clifford_res = clifford_bkd.execute_circuit(c1, nshots=1000)

    numpy_res = numpy_bkd.execute_circuit(c2, nshots=1000)

    backend.assert_allclose(
        clifford_res.probabilities(measured_qubits),
        backend.cast(numpy_res.probabilities(measured_qubits)),
        atol=1e-1,
    )


def test_non_clifford_error(backend):
    clifford_bkd = construct_clifford_backend(backend)
    c = Circuit(1)
    c.add(gates.T(0))
    with pytest.raises(RuntimeError) as excinfo:
        clifford_bkd.execute_circuit(c)
        assert str(excinfo.value) == "Circuit contains non-Clifford gates."


def test_initial_state(backend):
    clifford_bkd = construct_clifford_backend(backend)
    state = random_clifford(3, backend=numpy_bkd)
    tmp = clifford_bkd.execute_circuit(state)
    initial_symplectic_matrix = tmp.symplectic_matrix
    initial_state = numpy_bkd.execute_circuit(state).state()
    initial_state = np.outer(initial_state, np.transpose(np.conj(initial_state)))
    c = random_clifford(3, density_matrix=True, backend=backend)
    numpy_state = numpy_bkd.execute_circuit(c, initial_state=initial_state).state()
    clifford_state = clifford_bkd.execute_circuit(
        c, initial_state=initial_symplectic_matrix
    ).state()
    backend.assert_allclose(numpy_state, clifford_state)


@pytest.mark.parametrize("seed", [10])
def test_bitflip_noise(backend, seed):
    backend.set_seed(seed)
    clifford_bkd = construct_clifford_backend(backend)
    circuit = random_clifford(5, seed=seed, backend=backend)
    circuit_copy = circuit.copy()
    qubits = backend.np.random.choice(range(3), size=2, replace=False)
    circuit.add(gates.M(*qubits, p0=0.1, p1=0.5))
    circuit_copy.add(gates.M(*qubits, p0=0.1, p1=0.5))
    numpy_res = numpy_bkd.execute_circuit(circuit_copy)
    clifford_res = clifford_bkd.execute_circuit(circuit)
    backend.assert_allclose(
        clifford_res.probabilities(qubits), numpy_res.probabilities(qubits), atol=1e-1
    )


@pytest.mark.parametrize("seed", [2024])
def test_noise_channels(backend, seed):
    numpy_bkd.set_seed(seed)
    backend.set_seed(seed)

    clifford_bkd = construct_clifford_backend(backend)
    clifford_bkd.set_seed(seed)

    nqubits = 3

    circuit = random_clifford(nqubits, density_matrix=False, seed=seed, backend=backend)

    noise = NoiseModel()
    noisy_gates = [circuit.queue[0]]
    noise.add(PauliError([("X", 0.3)]), gates.H)
    noise.add(DepolarizingError(0.3), noisy_gates[0].__class__)

    circuit_copy = circuit.copy()
    non_noisy = circuit.copy()
    non_noisy.add(gates.M(*range(nqubits)))
    circuit.add(gates.M(*range(nqubits)))
    circuit_copy.add(gates.M(*range(nqubits)))

    circuit = noise.apply(circuit)
    circuit_copy = noise.apply(circuit_copy)

    numpy_result = numpy_bkd.execute_circuit(circuit)
    clifford_result = clifford_bkd.execute_circuit(circuit_copy)
    backend.assert_allclose(
        backend.cast(numpy_result.probabilities()),
        clifford_result.probabilities(),
        atol=1e-1,
    )


def test_stim(backend):
    clifford_bkd = construct_clifford_backend(backend)
    clifford_stim = CliffordBackend(engine="stim")

    nqubits = 3
    circuit = random_clifford(nqubits, backend=backend)

    result_qibo = clifford_bkd.execute_circuit(circuit)
    result_stim = clifford_stim.execute_circuit(circuit)

    backend.assert_allclose(
        result_stim.symplectic_matrix, result_qibo.symplectic_matrix
    )
