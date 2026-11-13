"""Tests for the QSVT matrix functions in `qibo/models/algorithms/qsvt_function.py`."""

import pytest

from qibo import Circuit, hamiltonians
from qibo.hamiltonians import Hamiltonian
from qibo.models import qsvt_function
from qibo.quantum_info import random_hermitian


def _apply(function, matrix, backend):
    eigenvalues, eigenvectors = backend.eigh(matrix)

    return backend.matmul(
        eigenvectors * backend.cast(function(eigenvalues), dtype=backend.complex128),
        backend.conj(backend.transpose(eigenvectors)),
    )


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
