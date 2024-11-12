from functools import reduce
from re import finditer

import numpy as np
import pytest

from qibo import Circuit, gates, matrices
from qibo.config import PRECISION_TOL
from qibo.quantum_info.metrics import fidelity
from qibo.quantum_info.random_ensembles import random_clifford
from qibo.quantum_info.utils import (
    haar_integral,
    hadamard_transform,
    hamming_distance,
    hamming_weight,
    hellinger_distance,
    hellinger_fidelity,
    hellinger_shot_error,
    pqc_integral,
    total_variation_distance,
)


@pytest.mark.parametrize("bitstring", range(2**5))
@pytest.mark.parametrize(
    "kind",
    [
        str,
        list,
        tuple,
        lambda b: np.array(list(b)),
        lambda b: int(b, 2),
        lambda b: list(map(int, b)),
    ],
)
def test_hamming_weight(bitstring, kind):
    with pytest.raises(TypeError):
        test = hamming_weight("0101", return_indexes="True")
    with pytest.raises(TypeError):
        test = hamming_weight(2.3)

    bitstring = f"{bitstring:b}"
    weight_test = len(bitstring.replace("0", ""))
    indexes_test = [item.start() for item in finditer("1", bitstring)]

    weight = hamming_weight(kind(bitstring), False)
    indexes = hamming_weight(kind(bitstring), True)

    assert weight == weight_test
    assert indexes == indexes_test


bitstring_1, bitstring_2 = "11111", "10101"


@pytest.mark.parametrize(
    ["bitstring_1", "bitstring_2"],
    [
        [bitstring_1, bitstring_2],
        [int(bitstring_1, 2), int(bitstring_2, 2)],
        [list(bitstring_1), list(bitstring_2)],
        [tuple(bitstring_1), tuple(bitstring_2)],
    ],
)
def test_hamming_distance(bitstring_1, bitstring_2):
    with pytest.raises(TypeError):
        test = hamming_distance("0101", "1010", return_indexes="True")
    with pytest.raises(TypeError):
        test = hamming_distance(2.3, "1010")
    with pytest.raises(TypeError):
        test = hamming_distance("1010", 2.3)

    if isinstance(bitstring_1, int):
        bitstring_1, bitstring_2 = f"{bitstring_1:b}", f"{bitstring_2:b}"

    distance = hamming_distance(bitstring_1, bitstring_2)
    indexes = hamming_distance(bitstring_1, bitstring_2, return_indexes=True)

    assert distance == 2
    assert indexes == [1, 3]


@pytest.mark.parametrize("is_matrix", [False, True])
@pytest.mark.parametrize("implementation", ["fast", "regular"])
@pytest.mark.parametrize("nqubits", [3, 4, 5])
def test_hadamard_transform(backend, nqubits, implementation, is_matrix):
    with pytest.raises(TypeError):
        test = np.random.rand(2, 2, 2)
        test = backend.cast(test, dtype=test.dtype)
        test = hadamard_transform(test, implementation=implementation, backend=backend)
    with pytest.raises(TypeError):
        test = np.random.rand(2, 3)
        test = backend.cast(test, dtype=test.dtype)
        test = hadamard_transform(test, implementation=implementation, backend=backend)
    with pytest.raises(TypeError):
        test = np.random.rand(3, 3)
        test = backend.cast(test, dtype=test.dtype)
        test = hadamard_transform(test, implementation=implementation, backend=backend)
    with pytest.raises(TypeError):
        test = np.random.rand(2**nqubits)
        test = backend.cast(test, dtype=test.dtype)
        test = hadamard_transform(test, implementation=True, backend=backend)
    with pytest.raises(ValueError):
        test = np.random.rand(2**nqubits)
        test = backend.cast(test, dtype=test.dtype)
        test = hadamard_transform(test, implementation="fas", backend=backend)

    dim = 2**nqubits

    state = np.random.rand(dim, dim) if is_matrix else np.random.rand(dim)
    state = backend.cast(state, dtype=state.dtype)

    hadamards = np.real(reduce(np.kron, [matrices.H] * nqubits))
    hadamards /= 2 ** (nqubits / 2)
    hadamards = backend.cast(hadamards, dtype=hadamards.dtype)

    test_transformed = hadamards @ state
    if is_matrix:
        test_transformed = test_transformed @ hadamards

    transformed = hadamard_transform(
        state, implementation=implementation, backend=backend
    )

    backend.assert_allclose(transformed, test_transformed, atol=PRECISION_TOL)


@pytest.mark.parametrize("kind", [None, list])
@pytest.mark.parametrize("validate", [False, True])
def test_hellinger(backend, validate, kind):
    with pytest.raises(TypeError):
        prob = np.random.rand(1, 2)
        prob_q = np.random.rand(1, 5)
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = hellinger_distance(prob, prob_q, backend=backend)
    with pytest.raises(TypeError):
        prob = np.random.rand(1, 2)[0]
        prob_q = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = hellinger_distance(prob, prob_q, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([-1, 2.0])
        prob_q = np.random.rand(1, 5)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = hellinger_distance(prob, prob_q, validate=True, backend=backend)
    with pytest.raises(ValueError):
        prob = np.random.rand(1, 2)[0]
        prob_q = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = hellinger_distance(prob, prob_q, validate=True, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.random.rand(1, 2)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = hellinger_distance(prob, prob_q, validate=True, backend=backend)

    prob_p = np.random.rand(10)
    prob_q = np.random.rand(10)
    prob_p /= np.sum(prob_p)
    prob_q /= np.sum(prob_q)
    prob_p = backend.cast(prob_p, dtype=prob_p.dtype)
    prob_q = backend.cast(prob_q, dtype=prob_q.dtype)

    target = float(
        backend.calculate_vector_norm(backend.np.sqrt(prob_p) - backend.np.sqrt(prob_q))
        / np.sqrt(2)
    )

    prob_p = (
        kind(prob_p) if kind is not None else backend.cast(prob_p, dtype=prob_p.dtype)
    )
    prob_q = (
        kind(prob_q) if kind is not None else backend.cast(prob_q, dtype=prob_q.dtype)
    )

    distance = hellinger_distance(prob_p, prob_q, validate=validate, backend=backend)
    fidelity = hellinger_fidelity(prob_p, prob_q, validate=validate, backend=backend)

    assert distance == target
    assert fidelity == (1 - target**2) ** 2


@pytest.mark.parametrize("kind", [None, list])
@pytest.mark.parametrize("validate", [False, True])
def test_hellinger_shot_error(backend, validate, kind):
    nqubits, nshots = 5, 1000

    circuit = random_clifford(nqubits, seed=1, backend=backend)
    circuit.add(gates.M(qubit) for qubit in range(nqubits))

    circuit_2 = random_clifford(nqubits, seed=2, backend=backend)
    circuit_2.add(gates.M(qubit) for qubit in range(nqubits))

    prob_dist_p = backend.execute_circuit(circuit, nshots=nshots).probabilities()
    prob_dist_q = backend.execute_circuit(circuit_2, nshots=nshots).probabilities()

    if kind is not None:
        prob_dist_p = kind(prob_dist_p)
        prob_dist_q = kind(prob_dist_q)

    hellinger_error = hellinger_shot_error(
        prob_dist_p, prob_dist_q, nshots, validate=validate, backend=backend
    )
    hellinger_fid = hellinger_fidelity(
        prob_dist_p, prob_dist_q, validate=validate, backend=backend
    )

    assert 2 * hellinger_error < hellinger_fid


@pytest.mark.parametrize("kind", [None, list])
@pytest.mark.parametrize("validate", [False, True])
def test_total_variation_distance(backend, validate, kind):
    with pytest.raises(ValueError):
        prob = np.array([-1, 2.0])
        prob_q = np.random.rand(1, 5)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = total_variation_distance(prob, prob_q, validate=True, backend=backend)
    with pytest.raises(ValueError):
        prob = np.random.rand(1, 2)[0]
        prob_q = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = total_variation_distance(prob, prob_q, validate=True, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.random.rand(1, 2)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = total_variation_distance(prob, prob_q, validate=True, backend=backend)

    prob_p = np.random.rand(10)
    prob_q = np.random.rand(10)
    prob_p /= np.sum(prob_p)
    prob_q /= np.sum(prob_q)
    prob_p = backend.cast(prob_p, dtype=prob_p.dtype)
    prob_q = backend.cast(prob_q, dtype=prob_q.dtype)

    target = float(backend.calculate_vector_norm(prob_p - prob_q, order=1) / 2)

    prob_p = (
        kind(prob_p) if kind is not None else backend.cast(prob_p, dtype=prob_p.dtype)
    )
    prob_q = (
        kind(prob_q) if kind is not None else backend.cast(prob_q, dtype=prob_q.dtype)
    )

    tvd = total_variation_distance(prob_p, prob_q, validate, backend)
    distance = hellinger_distance(prob_p, prob_q, validate, backend)

    assert tvd == target
    assert tvd <= np.sqrt(2) * distance
    assert tvd >= distance**2


def test_haar_integral_errors(backend):
    with pytest.raises(TypeError):
        nqubits, power_t, samples = 0.5, 2, 10
        test = haar_integral(nqubits, power_t, samples, backend=backend)
    with pytest.raises(TypeError):
        nqubits, power_t, samples = 2, 0.5, 10
        test = haar_integral(nqubits, power_t, samples, backend=backend)
    with pytest.raises(TypeError):
        nqubits, power_t, samples = 2, 1, 1.2
        test = haar_integral(nqubits, power_t, samples=samples, backend=backend)


@pytest.mark.parametrize("power_t", [1, 2])
@pytest.mark.parametrize("nqubits", [2, 3])
def test_haar_integral(backend, nqubits, power_t):
    samples = int(1e3)

    haar_int_exact = haar_integral(nqubits, power_t, samples=None, backend=backend)

    haar_int_sampled = haar_integral(nqubits, power_t, samples=samples, backend=backend)

    backend.assert_allclose(haar_int_sampled, haar_int_exact, atol=1e-1)


def test_pqc_integral(backend):
    with pytest.raises(TypeError):
        power_t, samples = 0.5, 10
        circuit = Circuit(2)
        test = pqc_integral(circuit, power_t, samples, backend=backend)
    with pytest.raises(TypeError):
        power_t, samples = 2, 0.5
        circuit = Circuit(2)
        test = pqc_integral(circuit, power_t, samples, backend=backend)

    circuit = Circuit(2)
    power_t, samples = 1, 100

    pqc_int = pqc_integral(circuit, power_t, samples, backend=backend)

    fid = fidelity(pqc_int, pqc_int, backend=backend)

    backend.assert_allclose(fid, 1.0, atol=PRECISION_TOL)
