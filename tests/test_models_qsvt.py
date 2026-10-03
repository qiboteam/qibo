"""Tests for quantum singular value transformation in `qibo/models/qsvt.py`."""

import math

import pytest
from numpy.typing import ArrayLike

from qibo import Circuit, gates
from qibo.backends import Backend
from qibo.models.qsvt import qsvt_circuit, qsvt_phases
from qibo.quantum_info import random_unitary


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 7, 25])
@pytest.mark.parametrize("nancillas,nsystem", [(1, 1), (1, 2), (2, 1)])
def test_qsvt_circuit(backend, nancillas, nsystem, degree):
    """The block of the circuit is the transformation of the singular values."""
    nqubits = nancillas + nsystem
    matrix = random_unitary(2**nqubits, seed=nqubits + degree, backend=backend)
    block_encoding = Circuit(nqubits)
    block_encoding.add(gates.Unitary(matrix, *range(nqubits)))

    coefficients = _coefficients(degree, seed=degree, backend=backend)
    phases = qsvt_phases(coefficients, backend=backend)
    circuit = qsvt_circuit(block_encoding, phases, nancillas)

    dim = 2**nsystem
    target = _transform(matrix[:dim, :dim], coefficients, backend)

    assert circuit.nqubits == nqubits + 1
    backend.assert_allclose(
        circuit.unitary(backend)[:dim, :dim], target, atol=1e-8, rtol=0.0
    )


@pytest.mark.parametrize("degree", [3, 5, 7])
def test_qsvt_circuit_amplification(backend, degree):
    """Chebyshev polynomials amplify the amplitude ``cos(pi / degree)`` to one."""
    block_encoding = Circuit(2)
    block_encoding.add(gates.RY(0, 2 * math.pi / degree))
    block_encoding.add(gates.H(1))

    coefficients = backend.zeros(degree + 1, dtype="float64")
    coefficients[degree] = 1.0
    circuit = qsvt_circuit(
        block_encoding, qsvt_phases(coefficients, backend=backend), nancillas=1
    )

    plus = Circuit(1)
    plus.add(gates.H(0))

    # T_d(cos(pi / d)) = -1 for odd d, so that the state is that of -|+>
    backend.assert_allclose(
        circuit(backend=backend).state()[:2],
        -plus(backend=backend).state(),
        atol=1e-6,
        rtol=0.0,
    )


def test_qsvt_circuit_errors(backend):
    block_encoding = Circuit(2)
    block_encoding.add(gates.H(1))
    phases = [0.1, 0.2, 0.3]

    with pytest.raises(ValueError):
        qsvt_circuit(block_encoding, [0.1], nancillas=1)
    with pytest.raises(ValueError):
        qsvt_circuit(block_encoding, phases, nancillas=0)
    with pytest.raises(ValueError):
        qsvt_circuit(block_encoding, phases, nancillas=3)


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 10, 25, 60])
def test_qsvt_phases(backend, degree):
    """The phases reproduce the polynomial for every singular value."""
    coefficients = _coefficients(degree, seed=degree + 10, backend=backend)
    phases = qsvt_phases(coefficients, backend=backend)

    xs = backend.cast(-1.0 + 2.0 * backend.arange(101) / 100, dtype="float64")

    assert len(phases) == degree + 1
    backend.assert_allclose(
        _polynomial(phases, xs, backend),
        _chebyshev(coefficients, xs, backend),
        atol=1e-10,
        rtol=0.0,
    )


@pytest.mark.parametrize("degree", [2, 3, 30, 60])
def test_qsvt_phases_chebyshev(backend, degree):
    """Polynomials that reach one are implemented with a small relative error."""
    coefficients = backend.zeros(degree + 1, dtype="float64")
    coefficients[degree] = 1.0
    phases = qsvt_phases(coefficients, backend=backend)

    xs = backend.cast(-1.0 + 2.0 * backend.arange(201) / 200, dtype="float64")

    backend.assert_allclose(
        _polynomial(phases, xs, backend),
        _chebyshev(coefficients, xs, backend),
        atol=1e-5,
        rtol=0.0,
    )


def test_qsvt_phases_errors(backend):
    with pytest.raises(ValueError):
        qsvt_phases([1.0], backend=backend)
    with pytest.raises(ValueError):
        qsvt_phases([0.0, 0.0, 0.0], backend=backend)
    with pytest.raises(ValueError):
        qsvt_phases([0.5, 0.0, 0.0, 0.0], backend=backend)
    # mixed parity
    with pytest.raises(ValueError):
        qsvt_phases([0.0, 0.3, 0.2], backend=backend)


@pytest.mark.parametrize("parity", [0, 1])
def test_qsvt_phases_interpolation(backend, parity):
    """Chebyshev series of smooth functions, with round-off in their coefficients."""
    functions = {
        0: lambda x: 0.9 * backend.cos(20 * x),
        1: lambda x: 0.9 * backend.sin(20 * x),
    }
    coefficients = _interpolate(functions[parity], 60, backend)
    phases = qsvt_phases(coefficients, backend=backend)

    xs = backend.cast(-1.0 + 2.0 * backend.arange(201) / 200, dtype="float64")

    assert len(phases) < 61
    backend.assert_allclose(
        _polynomial(phases, xs, backend), functions[parity](xs), atol=1e-8, rtol=0.0
    )


def test_qsvt_phases_normalization(backend):
    """The polynomial is divided by its maximum if this is larger than one."""
    coefficients = 4 * _coefficients(5, seed=3, backend=backend)
    xs = backend.cast(-1.0 + 2.0 * backend.arange(2001) / 2000, dtype="float64")
    maximum = backend.max(backend.abs(_chebyshev(coefficients, xs, backend)))

    phases = qsvt_phases(coefficients, backend=backend)

    assert maximum > 2.0
    backend.assert_allclose(
        _polynomial(phases, xs, backend),
        _chebyshev(coefficients, xs, backend) / maximum,
        atol=1e-6,
        rtol=0.0,
    )


def _chebyshev(coefficients: ArrayLike, xs: ArrayLike, backend: Backend) -> ArrayLike:
    """Chebyshev series evaluated at ``xs``."""
    previous, current = 0 * xs + 1.0, xs
    total = coefficients[0] * previous + coefficients[1] * current
    for coefficient in coefficients[2:]:
        previous, current = current, 2 * xs * current - previous
        total = total + coefficient * current

    return total


def _coefficients(degree: int, seed: int, backend: Backend) -> ArrayLike:
    """Random Chebyshev coefficients of parity ``degree % 2``, with maximum ``0.9``."""
    coefficients = backend.random_normal(
        0.0, 1.0, size=degree + 1, seed=seed, dtype="float64"
    )
    orders = backend.arange(degree + 1)
    coefficients = backend.where(orders % 2 == degree % 2, coefficients, 0.0)
    # the highest order is not negligible
    coefficients[degree] = 1.0 + backend.abs(coefficients[degree])

    xs = backend.cast(-1.0 + 2.0 * backend.arange(2001) / 2000, dtype="float64")
    maximum = backend.max(backend.abs(_chebyshev(coefficients, xs, backend)))

    return 0.9 * coefficients / maximum


def _interpolate(function, degree: int, backend: Backend) -> ArrayLike:
    """Chebyshev coefficients of the interpolant of ``function`` at the Chebyshev nodes."""
    nnodes = degree + 1
    thetas = math.pi * (backend.arange(nnodes) + 0.5) / nnodes
    matrix = backend.cos(backend.outer(backend.arange(nnodes), thetas))
    coefficients = 2 * backend.matmul(matrix, function(backend.cos(thetas))) / nnodes
    coefficients[0] = coefficients[0] / 2

    return coefficients


def _polynomial(phases: ArrayLike, xs: ArrayLike, backend: Backend) -> ArrayLike:
    """Real part of the polynomial that the phases implement, as a function of ``x``."""
    reflections = backend.zeros((len(xs), 2, 2), dtype="complex128")
    reflections[:, 0, 0] = xs
    reflections[:, 0, 1] = backend.sqrt(1 - xs**2)
    reflections[:, 1, 0] = backend.sqrt(1 - xs**2)
    reflections[:, 1, 1] = -xs

    matrix = backend.zeros((len(xs), 2, 2), dtype="complex128")
    matrix[:, 0, 0] = 1.0
    matrix[:, 1, 1] = 1.0
    for phase in phases[:-1]:
        rotation = backend.cast(
            [[backend.exp(1j * phase), 0], [0, backend.exp(-1j * phase)]],
            dtype="complex128",
        )
        matrix = backend.matmul(reflections, backend.matmul(rotation, matrix))

    return backend.real(backend.exp(1j * phases[-1]) * matrix[:, 0, 0])


def _transform(
    matrix: ArrayLike, coefficients: ArrayLike, backend: Backend
) -> ArrayLike:
    """Transformation of the singular values of ``matrix`` by a Chebyshev series."""
    left, singular_values, right = backend.singular_value_decomposition(matrix)
    values = _chebyshev(coefficients, singular_values, backend)
    if (len(coefficients) - 1) % 2 == 1:
        return backend.matmul(left * values, right)

    return backend.matmul(backend.dagger(right) * values, right)
