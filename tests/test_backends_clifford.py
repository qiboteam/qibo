from itertools import product

import numpy as np
import pytest

from qibo import Circuit, gates, set_backend
from qibo.backends import (
    CliffordBackend,
    GlobalBackend,
    NumpyBackend,
    TensorflowBackend,
)
from qibo.quantum_info.random_ensembles import random_clifford

numpy_bkd = NumpyBackend()


def construct_clifford_backend(backend):
    if isinstance(backend, TensorflowBackend):
        with pytest.raises(NotImplementedError) as excinfo:
            clifford_backend = CliffordBackend(backend)
            assert (
                str(excinfo.value)
                == "TensorflowBackend for Clifford Simulation is not supported yet."
            )
    else:
        return CliffordBackend(backend)


THETAS_1Q = [
    th + 2 * i * np.pi for i in range(2) for th in [0, np.pi / 2, np.pi, 3 * np.pi / 2]
]

AXES = ["RX", "RY", "RZ"]


@pytest.mark.parametrize("axis,theta", list(product(AXES, THETAS_1Q)))
def test_rotations_1q(backend, theta, axis):
    clifford_bkd = construct_clifford_backend(backend)
    if not clifford_bkd:
        return
    c = Circuit(3, density_matrix=True)
    qubits = np.random.randint(3, size=2)
    H_qubits = np.random.choice(range(3), size=2, replace=False)
    for q in H_qubits:
        c.add(gates.H(q))
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
    if not clifford_bkd:
        return
    c = Circuit(3, density_matrix=True)
    qubits_0 = np.random.choice(range(3), size=2, replace=False)
    qubits_1 = np.random.choice(range(3), size=2, replace=False)
    H_qubits = np.random.choice(range(3), size=2, replace=False)
    for q in H_qubits:
        c.add(gates.H(q))
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
    if not clifford_bkd:
        return
    c = Circuit(3, density_matrix=True)
    qubits = np.random.randint(3, size=2)
    H_qubits = np.random.choice(range(3), size=2, replace=False)
    for q in H_qubits:
        c.add(gates.H(q))
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
    if not clifford_bkd:
        return
    c = Circuit(5, density_matrix=True)
    qubits = np.random.choice(range(5), size=4, replace=False).reshape(2, 2)
    H_qubits = np.random.choice(range(5), size=3, replace=False)
    for q in H_qubits:
        c.add(gates.H(q))
    c.add(getattr(gates, gate)(*qubits[0]))
    c.add(getattr(gates, gate)(*qubits[1]))
    clifford_state = clifford_bkd.execute_circuit(c).state()
    numpy_state = numpy_bkd.execute_circuit(c).state()
    numpy_state = backend.cast(numpy_state)
    backend.assert_allclose(clifford_state, numpy_state, atol=1e-8)


MEASURED_QUBITS = sorted(np.random.choice(range(5), size=3, replace=False))


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize(
    "prob_qubits",
    [
        range(5),
        np.random.choice(MEASURED_QUBITS, size=2, replace=False),
        [0],
        [1],
        [2],
        [3],
        [4],
    ],
)
def test_random_clifford_circuit(backend, prob_qubits, binary):
    clifford_bkd = construct_clifford_backend(backend)
    if not clifford_bkd:
        return
    c = random_clifford(5, backend=backend)
    c.density_matrix = True
    c_copy = c.copy()
    c.add(gates.M(*MEASURED_QUBITS))
    c_copy.add(gates.M(*MEASURED_QUBITS))
    numpy_result = numpy_bkd.execute_circuit(c, nshots=1000)
    clifford_result = clifford_bkd.execute_circuit(c_copy, nshots=1000)

    backend.assert_allclose(backend.cast(numpy_result.state()), clifford_result.state())

    if not set(prob_qubits).issubset(set(MEASURED_QUBITS)):
        with pytest.raises(RuntimeError) as excinfo:
            numpy_bkd.assert_allclose(
                numpy_result.probabilities(prob_qubits),
                clifford_result.probabilities(prob_qubits),
            )
            assert (
                str(excinfo.value)
                == f"Asking probabilities for qubits {prob_qubits}, but only qubits {MEASURED_QUBITS} were measured."
            )
    else:
        backend.assert_allclose(
            backend.cast(numpy_result.probabilities(prob_qubits)),
            clifford_result.probabilities(prob_qubits),
            atol=1e-1,
        )

        numpy_freq = numpy_result.frequencies(binary)
        clifford_freq = clifford_result.frequencies(binary)
        print(numpy_freq)
        print(clifford_freq)
        clifford_freq = {
            state: clifford_freq[state] for state in numpy_freq.keys()
        }
        print(clifford_freq.values())
        assert (
            np.sum(np.abs(
                np.array(list(numpy_freq.values()))
                - np.array(list(clifford_freq.values()))
            ))
            < 200
        )


def test_collapsing_measurements(backend):
    clifford_bkd = construct_clifford_backend(backend)
    if not clifford_bkd:
        return
    gate_queue = random_clifford(3, density_matrix=True, backend=backend).queue
    measured_qubits = np.random.choice(range(3), size=2, replace=False)
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
        clifford_res.probabilities(), backend.cast(numpy_res.probabilities()), atol=1e-1
    )


def test_non_clifford_error(backend):
    clifford_bkd = construct_clifford_backend(backend)
    if not clifford_bkd:
        return
    c = Circuit(1)
    c.add(gates.T(0))
    with pytest.raises(RuntimeError) as excinfo:
        clifford_bkd.execute_circuit(c)
        assert str(excinfo.value) == "Circuit contains non-Clifford gates."


def test_initial_state(backend):
    clifford_bkd = construct_clifford_backend(backend)
    if not clifford_bkd:
        return
    state = random_clifford(3, backend=numpy_bkd)
    tmp = clifford_bkd.execute_circuit(state)
    initial_symplectic_matrix = tmp.symplectic_matrix
    initial_state = numpy_bkd.execute_circuit(state).state()
    initial_state = np.outer(initial_state, np.transpose(np.conj(initial_state)))
    print(type(initial_state))
    c = random_clifford(3, density_matrix=True, backend=backend)
    numpy_state = numpy_bkd.execute_circuit(c, initial_state=initial_state).state()
    clifford_state = clifford_bkd.execute_circuit(
        c, initial_state=initial_symplectic_matrix
    ).state()
    backend.assert_allclose(numpy_state, clifford_state)


def test_bitflip_noise(backend):
    clifford_bkd = construct_clifford_backend(backend)
    if not clifford_bkd:
        return
    c = random_clifford(5, backend=backend)
    c_copy = c.copy()
    qubits = np.random.choice(range(3), size=2, replace=False)
    c.add(gates.M(*qubits, p0=0.1, p1=0.5))
    c_copy.add(gates.M(*qubits, p0=0.1, p1=0.5))
    numpy_res = numpy_bkd.execute_circuit(c_copy)
    clifford_res = clifford_bkd.execute_circuit(c)
    backend.assert_allclose(
        clifford_res.probabilities(qubits), numpy_res.probabilities(qubits), atol=1e-1
    )


def test_set_backend(backend):
    clifford_bkd = construct_clifford_backend(backend)
    if not clifford_bkd:
        return
    set_backend("clifford")
    assert isinstance(GlobalBackend(), type(clifford_bkd))
