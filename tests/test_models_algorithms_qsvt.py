"""Tests for quantum singular value transformation in `qibo/models/algorithms/qsvt.py`."""

import math
import warnings

import pytest
from numpy.typing import ArrayLike

from qibo import Circuit, gates, hamiltonians, models
from qibo.backends import Backend
from qibo.hamiltonians import Hamiltonian
from qibo.models.algorithms import qsvt as qsvt_module
from qibo.models.algorithms.qsvt import qsvt_circuit, qsvt_function, qsvt_phases
from qibo.quantum_info.random_ensembles import random_hermitian, random_unitary


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


def test_qsvt_exports():
    """The QSVT functions are available from ``qibo.models``."""
    assert models.qsvt_circuit is qsvt_circuit
    assert models.qsvt_function is qsvt_function
    assert models.qsvt_phases is qsvt_phases


@pytest.mark.parametrize(
    "kind",
    [
        "evolution",
        "gibbs",
        "even",
        "odd",
        "imaginary",
        "constant and odd",
        "constant and even",
        "complex",
        "negative constant",
    ],
)
@pytest.mark.parametrize("nqubits", [1, 2])
def test_qsvt_function(backend, nqubits, kind):
    """The first block of the circuit is the function of the matrix, times the constant."""
    matrix = random_hermitian(2**nqubits, seed=3, backend=backend)
    function, nextra = _functions(backend)[kind]

    circuit, constant = qsvt_function(matrix, function, backend=backend)

    block = circuit.unitary(backend)[: 2**nqubits, : 2**nqubits]

    # the qubits of the linear combination, the signal qubit, the auxiliary qubit of the block
    # encoding, and the qubits of the matrix
    assert circuit.nqubits == nqubits + nextra
    assert constant > 0
    backend.assert_allclose(
        block, constant * _apply(function, matrix, backend), atol=1e-5
    )


@pytest.mark.parametrize("kind", ["odd", "even", "complex"])
def test_qsvt_function_coefficients(backend, kind):
    """The coefficients of a polynomial in the matrix divided by its norm are applied."""
    matrix = random_hermitian(4, seed=3, backend=backend)
    alpha = backend.max(backend.abs(backend.eigh(matrix)[0]))

    # Chebyshev polynomials T_3 and T_2 and a complex combination of them, with the peak one
    coefficients = {
        "odd": [0.0, 0.0, 0.0, 1.0],
        "even": [0.0, 0.0, 1.0],
        "complex": [0.0, 0.0, 0.5, 0.5j],
    }[kind]
    polynomial = {
        "odd": lambda x: 4 * x**3 - 3 * x,
        "even": lambda x: 2 * x**2 - 1,
        "complex": lambda x: 0.5 * (2 * x**2 - 1) + 0.5j * (4 * x**3 - 3 * x),
    }[kind]

    circuit, constant = qsvt_function(matrix, coefficients, backend=backend)

    block = circuit.unitary(backend)[:4, :4]
    expected = _apply(lambda x: polynomial(x / alpha), matrix, backend)

    backend.assert_allclose(block, constant * expected, atol=1e-5)


def test_qsvt_function_degree(backend):
    """The degree of the interpolation can be given, and the negligible terms are removed."""
    matrix = random_hermitian(4, seed=3, backend=backend)
    function = lambda x: backend.exp(-0.9j * x)

    adaptive, constant = qsvt_function(matrix, function, backend=backend)
    given, given_constant = qsvt_function(matrix, function, 40, backend=backend)

    # the series of the exponential is truncated by the tolerance, and it is the same one
    assert len(given.queue) == len(adaptive.queue)
    backend.assert_allclose(given_constant, constant, atol=1e-12)
    backend.assert_allclose(
        given.unitary(backend)[:4, :4], adaptive.unitary(backend)[:4, :4], atol=1e-8
    )

    # a tighter tolerance needs a higher degree, and it is more precise
    precise, _ = qsvt_function(matrix, function, epsilon=1e-10, backend=backend)
    assert len(precise.queue) > len(adaptive.queue)


def test_qsvt_function_density_matrix(backend):
    """Keyword arguments are passed to the circuit."""
    matrix = random_hermitian(2, seed=3, backend=backend)
    circuit, _ = qsvt_function(
        matrix, lambda x: backend.cos(x), density_matrix=True, backend=backend
    )

    assert circuit.density_matrix
    assert isinstance(circuit, Circuit)


def test_qsvt_function_errors(backend):
    matrix = random_hermitian(2, seed=3, backend=backend)
    function = lambda x: backend.cos(x)

    for epsilon in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError, match="epsilon must be in"):
            qsvt_function(matrix, function, epsilon=epsilon, backend=backend)
    for degree in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="degree must be a positive integer"):
            qsvt_function(matrix, function, degree, backend=backend)
    for wrong in ([1.0, 0.5], [[1.0]], [[1.0, 0.0, 0.0]], [[1.0] * 3] * 3):
        with pytest.raises(ValueError, match="matrix must be a square matrix"):
            qsvt_function(wrong, function, backend=backend)
    with pytest.raises(ValueError, match="matrix must be Hermitian"):
        qsvt_function([[1.0, 0.5], [0.0, 1.0]], function, backend=backend)
    with pytest.raises(ValueError, match="matrix must not be zero"):
        qsvt_function([[0.0, 0.0], [0.0, 0.0]], function, backend=backend)

    # a step has coefficients that decrease too slowly
    with pytest.raises(ValueError, match="is not approximated"):
        qsvt_function(matrix, lambda x: backend.sign(x), backend=backend)
    with pytest.raises(ValueError, match="function must not be zero"):
        qsvt_function(matrix, lambda x: 0 * x, backend=backend)
    with pytest.raises(ValueError, match="function must not be zero"):
        qsvt_function(matrix, [0.0, 0.0, 0.0], backend=backend)
    for coefficients in ([1.0], [[1.0, 0.0], [0.0, 1.0]]):
        with pytest.raises(ValueError, match="must be a vector of at least two"):
            qsvt_function(matrix, coefficients, backend=backend)


def test_qsvt_function_hamiltonians(backend):
    """The matrix can be a Hamiltonian."""
    hamiltonian = hamiltonians.TFIM(2, h=0.5, dense=False, backend=backend)
    function = lambda x: backend.exp(-0.6j * x)

    for matrix in (
        hamiltonian,
        Hamiltonian(2, hamiltonian.matrix, backend=backend),
        hamiltonian.matrix,
    ):
        circuit, constant = qsvt_function(matrix, function, backend=backend)

        backend.assert_allclose(
            circuit.unitary(backend)[:4, :4],
            constant * _apply(function, hamiltonian.matrix, backend),
            atol=1e-5,
        )


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


def test_qsvt_phases_interior_maximum(backend):
    """A polynomial that touches one away from the endpoints does not give warnings."""
    # odd polynomial that approximates the inverse, with a maximum above two inside of [-1, 1]
    degree = 23
    coefficients = _interpolate(lambda x: (1 - (1 - x**2) ** 12) / x, degree, backend)
    xs = backend.cast(-1.0 + 2.0 * backend.arange(20001) / 20000, dtype="float64")
    maximum = backend.max(backend.abs(_chebyshev(coefficients, xs, backend)))

    # the maximum is estimated on a coarser grid than the one of the factorization, which
    # can exceed it between the nodes, and then 1 - P^2 is negative for the smallest margins
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        phases = qsvt_phases(coefficients, backend=backend)

    assert maximum > 2.0
    backend.assert_allclose(
        _polynomial(phases, xs, backend),
        _chebyshev(coefficients, xs, backend) / maximum,
        atol=1e-5,
        rtol=0.0,
    )


def test_qsvt_phases_margin(backend, monkeypatch):
    """The margin is increased if the factorization has not converged."""
    calls = []
    factorize = qsvt_module._spectral_factor

    def not_converged(residual, degree, backend=None):
        factor, converged = factorize(residual, degree, backend)
        calls.append(converged)

        # the first factorization is rejected
        return factor, converged and len(calls) > 1

    monkeypatch.setattr(qsvt_module, "_spectral_factor", not_converged)

    # T_3 touches one at the endpoints, and its first factorization converges
    phases = qsvt_phases([0.0, 0.0, 0.0, 1.0], backend=backend)

    xs = backend.cast(-1.0 + 2.0 * backend.arange(2001) / 2000, dtype="float64")
    assert calls == [True, True]
    backend.assert_allclose(
        _polynomial(phases, xs, backend), 4 * xs**3 - 3 * xs, atol=1e-6, rtol=0.0
    )


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


def _apply(function, matrix, backend):
    eigenvalues, eigenvectors = backend.eigh(matrix)

    return backend.matmul(
        eigenvectors * backend.cast(function(eigenvalues), dtype=backend.complex128),
        backend.conj(backend.transpose(eigenvectors)),
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


def _functions(backend):
    return {
        "evolution": (lambda x: backend.exp(-0.9j * x), 3),
        "gibbs": (lambda x: backend.exp(-0.7 * x), 3),
        "even": (lambda x: backend.cos(x), 2),
        "odd": (lambda x: backend.sin(0.8 * x), 2),
        "imaginary": (lambda x: 0.2j * x**3, 2),
        "constant and odd": (lambda x: 0.5 + 0.3 * x, 3),
        "constant and even": (lambda x: 0.3 + 0.2 * x**2, 2),
        "complex": (lambda x: (1 + 1j) * backend.exp(0.5 * x), 4),
        "negative constant": (lambda x: -0.5 + 0 * x, 2),
    }


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
