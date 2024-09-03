"""Tests for qibo.models.encodings"""

import math
from itertools import product

import numpy as np
import pytest
from scipy.optimize import curve_fit

from qibo import Circuit, gates
from qibo.models.encodings import (
    comp_basis_encoder,
    entangling_layer,
    phase_encoder,
    unary_encoder,
    unary_encoder_random_gaussian,
)


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
    dims = 2**nqubits

    with pytest.raises(TypeError):
        data = sampler.random((nqubits, nqubits))
        data = backend.cast(data, dtype=data.dtype)
        phase_encoder(data, rotation=rotation)
    with pytest.raises(TypeError):
        data = sampler.random(nqubits)
        data = backend.cast(data, dtype=data.dtype)
        phase_encoder(data, rotation=True)
    with pytest.raises(ValueError):
        data = sampler.random(nqubits)
        data = backend.cast(data, dtype=data.dtype)
        phase_encoder(data, rotation="rzz")

    phases = np.random.rand(nqubits)

    if rotation in ["RX", "RY"]:
        functions = list(product([np.cos, np.sin], repeat=nqubits))
        target = []
        for row in functions:
            elem = 1.0
            for phase, func in zip(phases, row):
                elem *= func(phase / 2)
                if rotation == "RX" and func.__name__ == "sin":
                    elem *= -1.0j
            target.append(elem)
    else:
        target = [np.exp(-0.5j * sum(phases))] + [0.0] * (dims - 1)

    target = np.array(target, dtype=complex)
    target = backend.cast(target, dtype=target.dtype)

    if kind is not None:
        phases = kind(phases)

    state = phase_encoder(phases, rotation=rotation)
    state = backend.execute_circuit(state).state()

    backend.assert_allclose(state, target)


@pytest.mark.parametrize("kind", [None, list])
@pytest.mark.parametrize("architecture", ["tree", "diagonal"])
@pytest.mark.parametrize("nqubits", [8])
def test_unary_encoder(backend, nqubits, architecture, kind):
    sampler = np.random.default_rng(1)

    with pytest.raises(TypeError):
        data = sampler.random((nqubits, nqubits))
        data = backend.cast(data, dtype=data.dtype)
        unary_encoder(data, architecture=architecture)
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

    circuit = unary_encoder(data, architecture=architecture)
    state = backend.execute_circuit(circuit).state()
    indexes = np.flatnonzero(backend.to_numpy(state))
    state = backend.np.real(state[indexes])

    backend.assert_allclose(state, backend.cast(data) / backend.calculate_norm(data, 2))


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
        circuit = unary_encoder_random_gaussian(nqubits, seed=local_state)
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


@pytest.mark.parametrize("closed_boundary", [False, True])
@pytest.mark.parametrize("entangling_gate", ["CNOT", gates.CZ, gates.RBS])
@pytest.mark.parametrize(
    "architecture", ["diagonal", "shifted", "even-layer", "odd-layer"]
)
@pytest.mark.parametrize("nqubits", [4, 9])
def test_entangling_layer(nqubits, architecture, entangling_gate, closed_boundary):
    target_circuit = Circuit(nqubits)
    if architecture == "diagonal":
        target_circuit.add(
            _helper_entangling_test(entangling_gate, qubit)
            for qubit in range(nqubits - 1)
        )
    elif architecture == "even-layer":
        target_circuit.add(
            _helper_entangling_test(entangling_gate, qubit)
            for qubit in range(0, nqubits - 1, 2)
        )
    elif architecture == "odd-layer":
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
def test_circuit_kwargs(density_matrix):
    test = comp_basis_encoder(5, 7, density_matrix=density_matrix)
    assert test.density_matrix is density_matrix

    test = entangling_layer(5, density_matrix=density_matrix)
    assert test.density_matrix is density_matrix

    data = np.random.rand(5)
    test = phase_encoder(data, density_matrix=density_matrix)
    assert test.density_matrix is density_matrix

    test = unary_encoder(data, "diagonal", density_matrix=density_matrix)
    assert test.density_matrix is density_matrix

    test = unary_encoder_random_gaussian(4, density_matrix=density_matrix)
    assert test.density_matrix is density_matrix
