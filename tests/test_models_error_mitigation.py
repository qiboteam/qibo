import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.backends import construct_backend
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models.error_mitigation import (
    CDR,
    ICS,
    ZNE,
    get_expectation_val_with_readout_mitigation,
    get_response_matrix,
    sample_clifford_training_circuit,
    sample_training_circuit_cdr,
    vnCDR,
)
from qibo.noise import DepolarizingError, NoiseModel, ReadoutError
from qibo.quantum_info import random_stochastic_matrix
from qibo.symbols import Z


def get_noise_model(error, gate, resp_matrix=[False, None]):
    noise = NoiseModel()
    noise.add(error, gate)
    if resp_matrix[0]:
        noise.add(ReadoutError(probabilities=resp_matrix[1]), gate=gates.M)

    return noise


def get_circuit(nqubits, nmeas=None):
    if nmeas is None:
        nmeas = nqubits
    # Define the circuit
    hz = 0.5
    hx = 0.5
    dt = 0.25
    c = Circuit(nqubits, density_matrix=True)
    c.add(gates.RZ(q, theta=-2 * hz * dt - np.pi / 2) for q in range(nqubits))
    c.add(gates.RX(q, theta=np.pi / 2) for q in range(nqubits))
    c.add(gates.RZ(q, theta=-2 * hx * dt + np.pi) for q in range(nqubits))
    c.add(gates.RX(q, theta=np.pi / 2) for q in range(nqubits))
    c.add(gates.RZ(q, theta=-np.pi / 2) for q in range(nqubits))
    c.add(gates.CNOT(q, q + 1) for q in range(0, nqubits - 1, 2))
    c.add(gates.RZ(q + 1, theta=-2 * dt) for q in range(0, nqubits - 1, 2))
    c.add(gates.CNOT(q, q + 1) for q in range(0, nqubits - 1, 2))
    c.add(gates.CNOT(q, q + 1) for q in range(1, nqubits, 2))
    c.add(gates.RZ(q + 1, theta=-2 * dt) for q in range(1, nqubits, 2))
    c.add(gates.CNOT(q, q + 1) for q in range(1, nqubits, 2))
    c.add(gates.M(*range(nmeas)))

    return c


backend = construct_backend("numpy")
# # Generate random response matrices
resp_matrix_1q = random_stochastic_matrix(
    2, diagonally_dominant=True, seed=2, backend=backend
)
resp_matrix_2q = random_stochastic_matrix(
    4, diagonally_dominant=True, seed=2, backend=backend
)


@pytest.mark.parametrize(
    "nqubits,noise,insertion_gate,readout",
    [
        (3, get_noise_model(DepolarizingError(0.1), gates.CNOT), "CNOT", None),
        (
            3,
            get_noise_model(DepolarizingError(0.1), gates.CNOT),
            "CNOT",
            {"response_matrix": resp_matrix_2q, "ibu_iters": None},
        ),
        (
            3,
            get_noise_model(DepolarizingError(0.1), gates.CNOT),
            "CNOT",
            {"ncircuits": 2},
        ),
        (1, get_noise_model(DepolarizingError(0.1), gates.RX), "RX", None),
        (
            1,
            get_noise_model(DepolarizingError(0.3), gates.RX),
            "RX",
            {"response_matrix": resp_matrix_1q, "ibu_iters": None},
        ),
        (
            1,
            get_noise_model(DepolarizingError(0.3), gates.RX),
            "RX",
            {"response_matrix": resp_matrix_1q, "ibu_iters": 10},
        ),
        (1, get_noise_model(DepolarizingError(0.1), gates.RX), "RX", {"ncircuits": 2}),
    ],
)
@pytest.mark.parametrize("solve", [False, True])
@pytest.mark.parametrize("GUF", [False, True])
def test_zne(backend, nqubits, noise, solve, GUF, insertion_gate, readout):
    """Test that ZNE reduces the noise."""
    if backend.platform == "tensorflow":
        backend.tf.config.threading.set_inter_op_parallelism_threads = 1
        backend.tf.config.threading.set_intra_op_parallelism_threads = 1
    else:
        backend.set_threads(1)

    if nqubits == 1:
        nmeas = 1
    else:
        nmeas = nqubits - 1
    # Define the circuit
    c = get_circuit(nqubits, nmeas)
    # Define the observable
    obs = np.prod([Z(i) for i in range(nmeas)])
    obs_exact = SymbolicHamiltonian(obs, nqubits=nqubits, backend=backend)
    obs = SymbolicHamiltonian(obs, backend=backend)
    # Noise-free expected value
    exact = obs_exact.expectation(backend.execute_circuit(c).state())
    # Noisy expected value without mitigation
    state = backend.execute_circuit(noise.apply(c), nshots=10000)
    noisy = state.expectation_from_samples(obs)
    # Mitigated expected value
    estimate = ZNE(
        circuit=c,
        observable=obs,
        noise_levels=np.array(range(4)),
        noise_model=noise,
        nshots=10000,
        solve_for_gammas=solve,
        global_unitary_folding=GUF,
        insertion_gate=insertion_gate,
        readout=readout,
        backend=backend,
    )

    assert backend.np.abs(exact - estimate) <= backend.np.abs(exact - noisy)


@pytest.mark.parametrize("nqubits", [3])
@pytest.mark.parametrize("full_output", [False, True])
@pytest.mark.parametrize(
    "noise,readout",
    [
        (get_noise_model(DepolarizingError(0.1), gates.CNOT), None),
        (
            get_noise_model(DepolarizingError(0.1), gates.CNOT, [True, resp_matrix_2q]),
            {"response_matrix": resp_matrix_2q, "ibu_iters": None},
        ),
        (
            get_noise_model(DepolarizingError(0.1), gates.CNOT, [True, resp_matrix_2q]),
            {"response_matrix": resp_matrix_2q, "ibu_iters": 10},
        ),
        (
            get_noise_model(DepolarizingError(0.1), gates.CNOT, [True, resp_matrix_2q]),
            {"ncircuits": 2},
        ),
    ],
)
def test_cdr(backend, nqubits, noise, full_output, readout):
    """Test that CDR reduces the noise."""
    if backend.platform == "tensorflow":
        backend.tf.config.threading.set_inter_op_parallelism_threads = 1
        backend.tf.config.threading.set_intra_op_parallelism_threads = 1
    else:
        backend.set_threads(1)

    nmeas = 1 if nqubits == 1 else nqubits - 1
    # Define the circuit
    c = get_circuit(nqubits, nmeas)
    # Define the observable
    obs = np.prod([Z(i) for i in range(nmeas)])
    obs_exact = SymbolicHamiltonian(obs, nqubits=nqubits, backend=backend)
    obs = SymbolicHamiltonian(obs, backend=backend)
    # Noise-free expected value
    exact = obs_exact.expectation(backend.execute_circuit(c).state())
    # Noisy expected value without mitigation
    state = backend.execute_circuit(noise.apply(c), nshots=10000)
    noisy = state.expectation_from_samples(obs)
    # Mitigated expected value
    estimate = CDR(
        circuit=c,
        observable=obs,
        noise_model=noise,
        nshots=10000,
        n_training_samples=20,
        full_output=full_output,
        readout=readout,
        backend=backend,
    )
    if full_output:
        estimate = estimate[0]

    assert backend.np.abs(exact - estimate) <= backend.np.abs(exact - noisy)


@pytest.mark.parametrize("nqubits", [3])
def test_sample_training_circuit(nqubits):
    # Define the circuit
    hz = -2
    hx = 1
    dt = np.pi / 4
    c = Circuit(nqubits, density_matrix=True)
    c.add(gates.RZ(q, theta=-2 * hz * dt - np.pi / 2) for q in range(nqubits))
    c.add(gates.RX(q, theta=np.pi / 2) for q in range(nqubits))
    c.add(gates.RZ(q, theta=-2 * hx * dt + np.pi) for q in range(nqubits))
    c.add(gates.RX(q, theta=np.pi / 2) for q in range(nqubits))
    c.add(gates.RZ(q, theta=-np.pi / 2) for q in range(nqubits))
    c.add(gates.CNOT(q, q + 1) for q in range(0, nqubits - 1, 2))
    c.add(gates.RZ(q + 1, theta=-2 * dt) for q in range(0, nqubits - 1, 2))
    c.add(gates.CNOT(q, q + 1) for q in range(0, nqubits - 1, 2))
    c.add(gates.CNOT(q, q + 1) for q in range(1, nqubits, 2))
    c.add(gates.RZ(q + 1, theta=-2 * dt) for q in range(1, nqubits, 2))
    c.add(gates.CNOT(q, q + 1) for q in range(1, nqubits, 2))
    c.add(gates.M(q) for q in range(nqubits))

    with pytest.raises(ValueError):
        sample_training_circuit_cdr(c)
    with pytest.raises(ValueError):
        sample_clifford_training_circuit(c)


@pytest.mark.parametrize(
    "nqubits,noise,insertion_gate,readout",
    [
        (1, get_noise_model(DepolarizingError(0.1), gates.RX), "RX", None),
        (
            1,
            get_noise_model(DepolarizingError(0.1), gates.RX, [True, resp_matrix_1q]),
            "RX",
            {"response_matrix": resp_matrix_1q, "ibu_iters": 10},
        ),
        (
            1,
            get_noise_model(DepolarizingError(0.1), gates.RX, [True, resp_matrix_1q]),
            "RX",
            {"ncircuits": 2},
        ),
    ],
)
@pytest.mark.parametrize("full_output", [False, True])
def test_vncdr(backend, nqubits, noise, full_output, insertion_gate, readout):
    """Test that vnCDR reduces the noise."""
    if backend.platform == "tensorflow":
        backend.tf.config.threading.set_inter_op_parallelism_threads = 1
        backend.tf.config.threading.set_intra_op_parallelism_threads = 1
    else:
        backend.set_threads(1)
    # Define the circuit
    c = get_circuit(nqubits)
    # Define the observable
    obs = np.prod([Z(i) for i in range(nqubits)])
    obs = SymbolicHamiltonian(obs, backend=backend)
    # Noise-free expected value
    exact = obs.expectation(backend.execute_circuit(c).state())
    # Noisy expected value without mitigation
    state = backend.execute_circuit(noise.apply(c), nshots=10000)
    noisy = state.expectation_from_samples(obs)
    # Mitigated expected value
    estimate = vnCDR(
        circuit=c,
        observable=obs,
        backend=backend,
        noise_levels=range(3),
        noise_model=noise,
        nshots=10000,
        n_training_samples=20,
        insertion_gate=insertion_gate,
        full_output=full_output,
        readout=readout,
    )
    if full_output:
        estimate = estimate[0]

    assert backend.np.abs(exact - estimate) <= backend.np.abs(exact - noisy)


@pytest.mark.parametrize("nqubits,nmeas", [(3, 2)])
@pytest.mark.parametrize("method", ["response_matrix", "randomized"])
@pytest.mark.parametrize("ibu_iters", [None, 10])
def test_readout_mitigation(backend, nqubits, nmeas, method, ibu_iters):
    if backend.platform == "tensorflow":
        backend.tf.config.threading.set_inter_op_parallelism_threads = 1
        backend.tf.config.threading.set_intra_op_parallelism_threads = 1
    else:
        backend.set_threads(1)
    nshots = 10000
    p = random_stochastic_matrix(2**nmeas, diagonally_dominant=True, seed=5)
    noise = NoiseModel()
    noise.add(ReadoutError(probabilities=p), gate=gates.M)
    if method == "response_matrix":
        response = get_response_matrix(
            nmeas, None, noise, nshots=nshots, backend=backend
        )
        readout = {"response_matrix": response, "ibu_iters": ibu_iters}
    elif method == "randomized":
        readout = {"ncircuits": 10}
    # Define the observable
    obs = np.prod([Z(i) for i in range(nmeas)])
    obs = SymbolicHamiltonian(obs, backend=backend)
    # get noise free expected val
    c = get_circuit(nqubits, nmeas)
    true_state = backend.execute_circuit(c, nshots=nshots)
    true_val = true_state.expectation_from_samples(obs)
    # get noisy expected val
    state = backend.execute_circuit(noise.apply(c), nshots=nshots)
    noisy_val = state.expectation_from_samples(obs)

    mit_val = get_expectation_val_with_readout_mitigation(
        c, obs, noise, nshots, readout, backend=backend
    )

    assert backend.np.abs(true_val - mit_val) <= backend.np.abs(true_val - noisy_val)


@pytest.mark.parametrize("nqubits", [3])
@pytest.mark.parametrize("full_output", [False, True])
@pytest.mark.parametrize(
    "noise,readout",
    [
        (get_noise_model(DepolarizingError(0.1), gates.CNOT), None),
        (
            get_noise_model(DepolarizingError(0.1), gates.CNOT, [True, resp_matrix_2q]),
            {"response_matrix": resp_matrix_2q, "ibu_iters": None},
        ),
        (
            get_noise_model(DepolarizingError(0.1), gates.CNOT, [True, resp_matrix_2q]),
            {"response_matrix": resp_matrix_2q, "ibu_iters": 10},
        ),
        (
            get_noise_model(DepolarizingError(0.1), gates.CNOT, [True, resp_matrix_2q]),
            {"ncircuits": 2},
        ),
    ],
)
def test_ics(backend, nqubits, noise, full_output, readout):
    if backend.platform == "tensorflow":
        backend.tf.config.threading.set_inter_op_parallelism_threads = 1
        backend.tf.config.threading.set_intra_op_parallelism_threads = 1
    else:
        backend.set_threads(1)
    """Test that ICS reduces the noise."""
    # Define the circuit
    c = get_circuit(nqubits, nqubits - 1)
    # Define the observable
    obs = np.prod([Z(i, backend=backend) for i in range(nqubits - 1)])
    obs_exact = SymbolicHamiltonian(obs, nqubits=nqubits, backend=backend)
    obs = SymbolicHamiltonian(obs, backend=backend)
    # Noise-free expected value
    exact = obs_exact.expectation(backend.execute_circuit(c).state())
    # Noisy expected value without mitigation
    state = backend.execute_circuit(noise.apply(c), nshots=10000)
    noisy = state.expectation_from_samples(obs)
    # Mitigated expected value
    estimate = ICS(
        circuit=c,
        observable=obs,
        noise_model=noise,
        nshots=10000,
        n_training_samples=20,
        full_output=full_output,
        readout=readout,
        backend=backend,
    )
    if full_output:
        estimate = estimate[0]

    assert backend.np.abs(exact - estimate) <= backend.np.abs(exact - noisy)
