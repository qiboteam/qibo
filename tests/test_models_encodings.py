"""Tests for qibo.models.encodings"""
import math

import numpy as np
import pytest
from scipy.optimize import curve_fit

from qibo.models.encodings import (
    comp_basis_encoder,
    unary_encoder,
    unary_encoder_random_gaussian,
)


def gaussian(x, a, b, c):
    """Gaussian used in the `unary_encoder_random_gaussian test"""
    return np.exp(a * x**2 + b * x + c)


@pytest.mark.parametrize("nqubits", [3])
@pytest.mark.parametrize(
    "basis_element", [5, "101", ["1", "0", "1"], [1, 0, 1], ("1", "0", "1"), (1, 0, 1)]
)
def test_comp_basis_encoder(backend, basis_element, nqubits):
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

    state = comp_basis_encoder(basis_element, nqubits)
    state = backend.execute_circuit(state).state()

    backend.assert_allclose(state, target)


@pytest.mark.parametrize("architecture", ["tree", "diagonal"])
@pytest.mark.parametrize("nqubits", [8])
def test_unary_encoder(backend, nqubits, architecture):
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
    data = backend.cast(data, dtype=data.dtype)

    circuit = unary_encoder(data, architecture=architecture)
    state = backend.execute_circuit(circuit).state()
    indexes = np.flatnonzero(state)
    state = np.real(state[indexes])

    backend.assert_allclose(state, data / backend.calculate_norm(data, order=2))


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
        indexes = np.flatnonzero(state)
        state = np.real(state[indexes])
        amplitudes += [float(elem) for elem in list(state)]

    y, x = np.histogram(amplitudes, bins=50, density=True)
    x = (x[:-1] + x[1:]) / 2

    params, _ = curve_fit(gaussian, x, y)

    stddev = np.sqrt(-1 / (2 * params[0]))
    mean = stddev**2 * params[1]

    theoretical_norm = (
        math.sqrt(2) * math.gamma((nqubits + 1) / 2) / math.gamma(nqubits / 2)
    )
    theoretical_norm = 1.0 / theoretical_norm

    backend.assert_allclose(0.0, mean, atol=1e-1)
    backend.assert_allclose(stddev, theoretical_norm, atol=1e-1)
