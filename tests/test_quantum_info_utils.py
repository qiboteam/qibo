from functools import reduce
from re import finditer

import numpy as np
import pytest

from qibo import Circuit, matrices
from qibo.config import PRECISION_TOL
from qibo.quantum_info.metrics import fidelity
from qibo.quantum_info.utils import (
    haar_integral,
    hadamard_transform,
    hamming_weight,
    hellinger_distance,
    hellinger_fidelity,
    pqc_integral,
    shannon_entropy,
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

    bitstring = f"{bitstring:b}"
    weight_test = len(bitstring.replace("0", ""))
    indexes_test = [item.start() for item in finditer("1", bitstring)]

    weight = hamming_weight(kind(bitstring), False)
    indexes = hamming_weight(kind(bitstring), True)

    assert weight == weight_test
    assert indexes == indexes_test


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


def test_shannon_entropy_errors(backend):
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, -2, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([[1.0], [0.0]])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, -1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.1, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([0.5, 0.4999999])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, backend=backend)


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_shannon_entropy(backend, base):
    prob_array = [1.0, 0.0]
    result = shannon_entropy(prob_array, base, backend=backend)
    backend.assert_allclose(result, 0.0)

    if base == 2:
        prob_array = np.array([0.5, 0.5])
        prob_array = backend.cast(prob_array, dtype=prob_array.dtype)
        result = shannon_entropy(prob_array, base, backend=backend)
        backend.assert_allclose(result, 1.0)


def test_hellinger(backend):
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

    prob = [1.0, 0.0]
    prob_q = [1.0, 0.0]
    backend.assert_allclose(hellinger_distance(prob, prob_q, backend=backend), 0.0)
    backend.assert_allclose(hellinger_fidelity(prob, prob_q, backend=backend), 1.0)


def test_haar_integral_errors(backend):
    with pytest.raises(TypeError):
        nqubits, power_t, samples = 0.5, 2, 10
        test = haar_integral(nqubits, power_t, samples, backend=backend)
    with pytest.raises(TypeError):
        nqubits, power_t, samples = 2, 0.5, 10
        test = haar_integral(nqubits, power_t, samples, backend=backend)
    with pytest.raises(TypeError):
        nqubits, power_t, exact, samples = 2, 1, False, 1.2
        test = haar_integral(
            nqubits, power_t, exact=exact, samples=samples, backend=backend
        )
    with pytest.raises(TypeError):
        nqubits, power_t, samples = 2, 2, 0.5
        test = haar_integral(nqubits, power_t, samples, backend=backend)


@pytest.mark.parametrize("power_t", [1, 2, 3])
@pytest.mark.parametrize("nqubits", [2, 3])
def test_haar_integral(backend, nqubits, power_t):
    samples = int(1e4)

    haar_int_exact = haar_integral(
        nqubits, power_t, exact=True, samples=samples, backend=backend
    )

    haar_int_sampled = haar_integral(
        nqubits, power_t, exact=False, samples=samples, backend=backend
    )

    backend.assert_allclose(haar_int_sampled, haar_int_exact, atol=1e-2)


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

    fid = fidelity(pqc_int, pqc_int)

    backend.assert_allclose(fid, 1.0, atol=PRECISION_TOL)
