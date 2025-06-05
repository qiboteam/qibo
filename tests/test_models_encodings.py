"""Tests for qibo.models.encodings"""

import math
from functools import reduce
from itertools import combinations

import networkx as nx
import numpy as np
import pytest
from scipy.optimize import curve_fit
from scipy.special import binom

from qibo import Circuit, gates
from qibo.models.encodings import (
    _ehrlich_algorithm,
    _get_next_bistring,
    binary_encoder,
    comp_basis_encoder,
    dicke_state,
    entangling_layer,
    ghz_state,
    graph_state,
    hamming_weight_encoder,
    phase_encoder,
    sparse_encoder,
    unary_encoder,
    unary_encoder_random_gaussian,
)
from qibo.quantum_info.random_ensembles import random_statevector


def _gaussian(x, a, b, c):
    """Gaussian used in the `unary_encoder_random_gaussian test"""
    return np.exp(a * x**2 + b * x + c)


@pytest.mark.parametrize(
    "basis_element", [5, "101", ["1", "0", "1"], [1, 0, 1], ("1", "0", "1"), (1, 0, 1)]
)
def test_comp_basis_encoder(backend, basis_element):
    with pytest.raises(TypeError):
        circuit = comp_basis_encoder(2.3)
    with pytest.raises(ValueError):
        circuit = comp_basis_encoder("0b001")
    with pytest.raises(ValueError):
        circuit = comp_basis_encoder("001", nqubits=2)
    with pytest.raises(TypeError):
        circuit = comp_basis_encoder("001", nqubits=3.1)
    with pytest.raises(ValueError):
        circuit = comp_basis_encoder(3)

    zero = np.array([1, 0], dtype=complex)
    one = np.array([0, 1], dtype=complex)
    target = np.kron(one, np.kron(zero, one))
    target = backend.cast(target, dtype=target.dtype)

    state = (
        comp_basis_encoder(basis_element, nqubits=3)
        if isinstance(basis_element, int)
        else comp_basis_encoder(basis_element)
    )

    state = backend.execute_circuit(state).state()

    backend.assert_allclose(state, target)


@pytest.mark.parametrize("kind", [None, list])
@pytest.mark.parametrize("rotation", ["RX", "RY", "RZ"])
def test_phase_encoder(backend, rotation, kind):
    sampler = np.random.default_rng(1)

    nqubits = 3

    with pytest.raises(TypeError):
        data = sampler.random(nqubits)
        data = backend.cast(data, dtype=data.dtype)
        phase_encoder(data, rotation=True)
    with pytest.raises(ValueError):
        data = sampler.random(nqubits)
        data = backend.cast(data, dtype=data.dtype)
        phase_encoder(data, rotation="rzz")

    phases = backend.np.random.rand(nqubits)

    gate = getattr(gates, rotation)
    target = reduce(
        backend.np.kron,
        [gate(qubit, phase).matrix(backend) for qubit, phase in enumerate(phases)],
    )[:, 0]

    if kind is not None:
        phases = kind(phases)

    state = phase_encoder(phases, rotation=rotation, backend=backend)
    state.draw()
    state = backend.execute_circuit(state).state()

    backend.assert_allclose(state, target)


@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("parametrization", ["hopf", "hyperspherical"])
@pytest.mark.parametrize("nqubits", [3, 4, 5])
def test_binary_encoder(backend, nqubits, parametrization, complex_data):
    if parametrization == "hopf" and complex_data:
        pytest.skip(
            "``binary_encoder`` in Hopf coordinates not implemented for complex data."
        )

    with pytest.raises(ValueError):
        dims = 5
        test = np.random.rand(dims)
        test = backend.cast(test, dtype=test.dtype)
        test = binary_encoder(test)

    dims = 2**nqubits

    target = random_statevector(dims, seed=10, backend=backend)
    if not complex_data:
        target = backend.np.real(target)
        target /= backend.np.linalg.norm(target)

    circuit = binary_encoder(target, parametrization=parametrization, backend=backend)
    state = backend.execute_circuit(circuit).state()

    backend.assert_allclose(state, target, atol=1e-10, rtol=1e-4)


@pytest.mark.parametrize("kind", [None, list])
@pytest.mark.parametrize("architecture", ["tree", "diagonal"])
@pytest.mark.parametrize("nqubits", [8])
def test_unary_encoder(backend, nqubits, architecture, kind):
    sampler = np.random.default_rng(1)

    with pytest.raises(TypeError):
        data = sampler.random(nqubits)
        data = backend.cast(data, dtype=data.dtype)
        unary_encoder(data, architecture=True)
    with pytest.raises(ValueError):
        data = sampler.random(nqubits)
        data = backend.cast(data, dtype=data.dtype)
        unary_encoder(data, architecture="semi-diagonal")
    if architecture == "tree":
        with pytest.raises(ValueError):
            data = sampler.random(nqubits + 1)
            data = backend.cast(data, dtype=data.dtype)
            unary_encoder(data, architecture=architecture)

    # sampling random data in interval [-1, 1]
    sampler = np.random.default_rng(1)
    data = 2 * sampler.random(nqubits) - 1
    data = kind(data) if kind is not None else backend.cast(data, dtype=data.dtype)

    circuit = unary_encoder(data, architecture=architecture, backend=backend)
    state = backend.execute_circuit(circuit).state()
    indexes = np.flatnonzero(backend.to_numpy(state))
    state = backend.np.real(state[indexes])

    backend.assert_allclose(
        state,
        backend.cast(data, dtype=np.float64) / backend.calculate_vector_norm(data, 2),
        rtol=1e-5,
    )


@pytest.mark.parametrize("seed", [None, 10, np.random.default_rng(10)])
@pytest.mark.parametrize("nqubits", [8])
def test_unary_encoder_random_gaussian(backend, nqubits, seed):
    """Tests if encoded vector are random variables sampled from
    Gaussian distribution with 0.0 mean and variance close to the norm
    of the random Gaussian vector that was encoded."""
    with pytest.raises(TypeError):
        unary_encoder_random_gaussian("1", seed=seed)
    with pytest.raises(ValueError):
        unary_encoder_random_gaussian(-1, seed=seed)
    with pytest.raises(ValueError):
        unary_encoder_random_gaussian(3, seed=seed)
    with pytest.raises(TypeError):
        unary_encoder_random_gaussian(nqubits, architecture=True, seed=seed)
    with pytest.raises(NotImplementedError):
        unary_encoder_random_gaussian(nqubits, architecture="diagonal", seed=seed)
    with pytest.raises(TypeError):
        unary_encoder_random_gaussian(nqubits, seed="seed")

    samples = int(1e2)

    local_state = np.random.default_rng(seed) if seed in [None, 10] else seed

    amplitudes = []
    for _ in range(samples):
        circuit = unary_encoder_random_gaussian(
            nqubits, seed=local_state, backend=backend
        )
        state = backend.execute_circuit(circuit).state()
        indexes = np.flatnonzero(backend.to_numpy(state))
        state = np.real(state[indexes])
        amplitudes += [float(elem) for elem in list(state)]

    y, x = np.histogram(amplitudes, bins=50, density=True)
    x = (x[:-1] + x[1:]) / 2

    params, _ = curve_fit(_gaussian, x, y)

    stddev = np.sqrt(-1 / (2 * params[0]))
    mean = stddev**2 * params[1]

    theoretical_norm = (
        math.sqrt(2) * math.gamma((nqubits + 1) / 2) / math.gamma(nqubits / 2)
    )
    theoretical_norm = 1.0 / theoretical_norm

    backend.assert_allclose(0.0, mean, atol=1e-1)
    backend.assert_allclose(stddev, theoretical_norm, atol=1e-1)


@pytest.mark.parametrize("seed", [10])
@pytest.mark.parametrize("optimize_controls", [False, True])
@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("full_hwp", [False, True])
@pytest.mark.parametrize("weight", [1, 2, 3])
@pytest.mark.parametrize("nqubits", [4, 5, 6])
def test_hamming_weight_encoder(
    backend,
    nqubits,
    weight,
    full_hwp,
    complex_data,
    optimize_controls,
    seed,
):
    n_choose_k = int(binom(nqubits, weight))
    dims = 2**nqubits
    dtype = complex if complex_data else float

    initial_string = np.array([1] * weight + [0] * (nqubits - weight))
    indices = _ehrlich_algorithm(initial_string, False)
    indices = [int(string, 2) for string in indices]
    indices_lex = np.sort(np.copy(indices))

    rng = np.random.default_rng(seed)
    data = rng.random(n_choose_k)
    if complex_data:
        data = data.astype(complex) + 1j * rng.random(n_choose_k)
    data /= np.linalg.norm(data)

    target = np.zeros(dims, dtype=dtype)
    target[indices_lex] = data
    target = backend.cast(target, dtype=target.dtype)

    data = backend.cast(data, dtype=data.dtype)

    circuit = hamming_weight_encoder(
        data,
        nqubits=nqubits,
        weight=weight,
        full_hwp=full_hwp,
        optimize_controls=optimize_controls,
        backend=backend,
    )
    if full_hwp:
        circuit.queue = [
            gates.X(nqubits - 1 - qubit) for qubit in range(weight)
        ] + circuit.queue
    state = backend.execute_circuit(circuit).state()

    backend.assert_allclose(state, target, atol=1e-7)


@pytest.mark.parametrize("seed", [10])
@pytest.mark.parametrize("zip_input", [False, True])
@pytest.mark.parametrize("integers", [False, True])
@pytest.mark.parametrize("nqubits", [4, 7])
def test_sparse_encoder(backend, nqubits, integers, zip_input, seed):
    dims = 2**nqubits
    sparsity = nqubits

    data = random_statevector(sparsity, seed=10, backend=backend)
    np.random.seed(seed)
    indices = np.random.choice(range(dims), size=sparsity, replace=False)
    indices = backend.cast(indices, dtype=int)

    target = backend.cast(backend.np.zeros(dims))
    target[indices] = data

    if not integers:
        indices = [f"{elem:0{nqubits}b}" for elem in indices]

    data = zip(indices, data)
    if not zip_input:
        data = list(data)

    if integers and not zip_input:
        with pytest.raises(ValueError):
            circuit = sparse_encoder(data, nqubits=None)

    _nqubits = nqubits if integers else None
    circuit = sparse_encoder(data, _nqubits, backend=backend)
    state = backend.execute_circuit(circuit).state()

    backend.assert_allclose(state, target)


def test_entangling_layer_errors():
    with pytest.raises(TypeError):
        entangling_layer(10.5)
    with pytest.raises(ValueError):
        entangling_layer(-4)
    with pytest.raises(TypeError):
        entangling_layer(10, architecture=True)
    with pytest.raises(NotImplementedError):
        entangling_layer(10, architecture="qibo")
    with pytest.raises(TypeError):
        entangling_layer(10, closed_boundary="True")
    with pytest.raises(NotImplementedError):
        entangling_layer(10, entangling_gate=gates.GeneralizedfSim)
    with pytest.raises(NotImplementedError):
        entangling_layer(10, entangling_gate=gates.TOFFOLI)
    with pytest.raises(ValueError):
        entangling_layer(7, architecture="x")


@pytest.mark.parametrize("closed_boundary", [False, True])
@pytest.mark.parametrize("entangling_gate", ["CNOT", gates.CZ, gates.RBS])
@pytest.mark.parametrize(
    "architecture",
    [
        "diagonal",
        "even_layer",
        "next_nearest",
        "odd_layer",
        "pyramid",
        "shifted",
        "v",
        "x",
    ],
)
@pytest.mark.parametrize("nqubits", [4, 6])
def test_entangling_layer(nqubits, architecture, entangling_gate, closed_boundary):
    target_circuit = Circuit(nqubits)
    if architecture in ["next_nearest", "pyramid", "v", "x"]:
        if architecture == "next_nearest":
            target_circuit.add(
                _helper_entangling_test(entangling_gate, qubit, qubit + 2)
                for qubit in range(nqubits - 2)
            )
            if closed_boundary:
                target_circuit.add(
                    _helper_entangling_test(entangling_gate, nqubits - 1, 0)
                )
        elif architecture == "pyramid":
            for end in range(nqubits - 1, 1, -1):
                target_circuit.add(
                    _helper_entangling_test(entangling_gate, qubit)
                    for qubit in range(end)
                )
        elif architecture == "v":
            target_circuit.add(
                _helper_entangling_test(entangling_gate, qubit, qubit + 1)
                for qubit in range(nqubits - 1)
            )
            target_circuit.add(
                _helper_entangling_test(entangling_gate, qubit - 1, qubit)
                for qubit in range(nqubits - 2, 1, -1)
            )
    else:
        if architecture == "diagonal":
            target_circuit.add(
                _helper_entangling_test(entangling_gate, qubit)
                for qubit in range(nqubits - 1)
            )
        elif architecture == "even_layer":
            target_circuit.add(
                _helper_entangling_test(entangling_gate, qubit)
                for qubit in range(0, nqubits - 1, 2)
            )
        elif architecture == "odd_layer":
            target_circuit.add(
                _helper_entangling_test(entangling_gate, qubit)
                for qubit in range(1, nqubits - 1, 2)
            )
        else:
            target_circuit.add(
                _helper_entangling_test(entangling_gate, qubit)
                for qubit in range(0, nqubits - 1, 2)
            )
            target_circuit.add(
                _helper_entangling_test(entangling_gate, qubit)
                for qubit in range(1, nqubits - 1, 2)
            )

        if closed_boundary:
            target_circuit.add(_helper_entangling_test(entangling_gate, nqubits - 1, 0))

    circuit = entangling_layer(nqubits, architecture, entangling_gate, closed_boundary)

    for gate, target in zip(circuit.queue, target_circuit.queue):
        assert gate.__class__.__name__ == target.__class__.__name__
        assert gate.qubits == target.qubits
        assert gate.target_qubits == target.target_qubits
        assert gate.control_qubits == target.control_qubits
        assert gate.parameters == target.parameters


def _helper_entangling_test(gate, qubit_0, qubit_1=None):
    """Creates two-qubit gate with of without parameters."""
    if qubit_1 is None:
        qubit_1 = qubit_0 + 1

    if callable(gate) and gate.__name__ == "RBS":
        return gate(qubit_0, qubit_1, 0.0)

    if gate == "CNOT":
        gate = gates.CNOT

    return gate(qubit_0, qubit_1)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_circuit_kwargs(backend, density_matrix):
    test = comp_basis_encoder(5, 7, density_matrix=density_matrix)
    assert test.density_matrix is density_matrix

    test = entangling_layer(5, density_matrix=density_matrix)
    assert test.density_matrix is density_matrix

    data = backend.cast(np.random.rand(5), dtype=backend.np.float64)
    test = phase_encoder(data, density_matrix=density_matrix, backend=backend)
    assert test.density_matrix is density_matrix

    test = unary_encoder(
        data, "diagonal", density_matrix=density_matrix, backend=backend
    )
    assert test.density_matrix is density_matrix

    test = unary_encoder_random_gaussian(
        4, density_matrix=density_matrix, backend=backend
    )
    assert test.density_matrix is density_matrix


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("nqubits", [1, 2, 3, 4])
def test_ghz_circuit(backend, nqubits, density_matrix):
    if nqubits < 2:
        with pytest.raises(ValueError):
            GHZ_circ = ghz_state(nqubits, density_matrix=density_matrix)
    else:
        target = np.zeros(2**nqubits, dtype=complex)
        target[0] = 1 / np.sqrt(2)
        target[2**nqubits - 1] = 1 / np.sqrt(2)
        target = backend.cast(target, dtype=target.dtype)

        GHZ_circ = ghz_state(nqubits, density_matrix=density_matrix)
        state = backend.execute_circuit(GHZ_circ).state()

        if density_matrix:
            target = backend.np.outer(target, backend.np.conj(target.T))

        backend.assert_allclose(state, target)


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize(
    "nqubits, weight",
    [
        (2, -1),
        (2, 1),
        (3, 1),
        (3, 2),
        (4, 2),
        (4, 1),
    ],
)
def test_dicke_state(backend, nqubits, weight, density_matrix):
    if weight < 0 or weight > nqubits:
        with pytest.raises(ValueError):
            dicke_circ = dicke_state(nqubits, weight, density_matrix=density_matrix)
    else:
        # Build expected Dicke state vector
        target = np.zeros(2**nqubits, dtype=complex)
        bitstrings = combinations(range(nqubits), weight)
        for positions in bitstrings:
            index = sum(1 << (nqubits - 1 - p) for p in positions)
            target[index] = 1.0
        target /= np.sqrt(np.count_nonzero(target))

        target = backend.cast(target, dtype=target.dtype)

        dicke_circ = dicke_state(nqubits, weight, density_matrix=density_matrix)
        result = backend.execute_circuit(dicke_circ)
        state = result.state()

        if density_matrix:
            target = backend.np.outer(target, backend.np.conj(target.T))

        backend.assert_allclose(state, target)


@pytest.mark.parametrize(
    "matrix, expects_error, circuit1, circuit2",
    [
        # Test Case 1: 3-qubit graph
        ([[0, 1, 0], [1, 0, 1], [0, 1, 0]], False, True, False),
        # Test Case 2: 5-qubit  graph
        (
            [
                [0, 1, 0, 0, 1],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1],
                [1, 0, 0, 1, 0],
            ],
            False,
            False,
            True,
        ),
        # Test Case 3: Non-symmetric matrix (expected error)
        (
            [[0, 1, 0], [0, 0, 1], [0, 1, 0]],
            True,
            False,
            False,
        ),  # matrix[1,0] != matrix[0,1]
        # Test Case 4: Non-symmetric matrix (expected error)
        ([[0, 1], [0, 0]], True, False, False),  # matrix[1,0] != matrix[0,1]
    ],
)
def test_graph_state(backend, matrix, expects_error, circuit1, circuit2):

    if expects_error:
        # We expect a ValueError for non-symmetric matrices
        with pytest.raises(ValueError):
            graph_state(matrix, backend=backend)

    else:
        if circuit1:
            nqubits = 3
            target = Circuit(nqubits)
            target.add(gates.H(qubit) for qubit in range(nqubits))
            target.add(gates.CZ(0, 1))
            target.add(gates.CZ(1, 2))

        if circuit2:
            nqubits = 5
            target = Circuit(nqubits)
            target.add(gates.H(qubit) for qubit in range(nqubits))
            target.add(gates.CZ(0, 1))
            target.add(gates.CZ(0, 4))
            target.add(gates.CZ(1, 2))
            target.add(gates.CZ(2, 3))
            target.add(gates.CZ(3, 4))

        circuit = graph_state(matrix)
        backend.assert_circuitclose(circuit, target)
