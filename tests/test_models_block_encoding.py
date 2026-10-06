"""Tests for block encodings in `qibo/models/block_encoding.py`."""

import pytest
from numpy.typing import ArrayLike

from qibo import models
from qibo.backends import Backend
from qibo.models.block_encoding import block_encoding_circuit
from qibo.models.qsvt import qsvt_circuit, qsvt_phases


@pytest.mark.parametrize("norm", [0.4, 2.5])
@pytest.mark.parametrize("dim", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("method", ["dense", "lcu"])
def test_block_encoding_circuit(backend, method, dim, norm):
    """The circuit is unitary and has the normalized matrix as its block."""
    matrix = _matrix(dim, seed=dim, norm=norm, backend=backend)
    circuit, alpha, nauxiliary = block_encoding_circuit(
        matrix, method=method, backend=backend
    )

    nsystem = max((dim - 1).bit_length(), 1)
    unitary = circuit.unitary(backend)
    block = unitary[: 2**nsystem, : 2**nsystem]

    assert circuit.nqubits == nauxiliary + nsystem
    if method == "dense":
        assert nauxiliary == 1
        assert alpha == pytest.approx(max(norm, 1.0), rel=1e-12)
    else:
        assert alpha >= norm * (1 - 1e-12)
    backend.assert_allclose(
        backend.matmul(unitary, backend.dagger(unitary)),
        backend.identity(2 ** (nauxiliary + nsystem), dtype="complex128"),
        atol=1e-10,
    )
    backend.assert_allclose(block[:dim, :dim], matrix / alpha, atol=1e-10)
    # the zero-padding is encoded as zeros
    backend.assert_allclose(block[dim:, :], 0.0, atol=1e-10)
    backend.assert_allclose(block[:, dim:], 0.0, atol=1e-10)


def test_block_encoding_circuit_alpha(backend):
    """A normalization larger than the norm rescales the encoded block."""
    matrix = _matrix(4, seed=1, norm=1.7, backend=backend)
    circuit, alpha, nauxiliary = block_encoding_circuit(
        matrix, alpha=3.4, backend=backend
    )

    assert alpha == 3.4
    assert nauxiliary == 1
    backend.assert_allclose(circuit.unitary(backend)[:4, :4], matrix / 3.4, atol=1e-10)


def test_block_encoding_circuit_errors(backend):
    matrix = _matrix(4, seed=1, norm=1.7, backend=backend)

    with pytest.raises(ValueError):
        block_encoding_circuit(matrix, method="fourier", backend=backend)
    with pytest.raises(ValueError):
        block_encoding_circuit(matrix[:, :3], backend=backend)
    with pytest.raises(ValueError):
        block_encoding_circuit(matrix[0], backend=backend)
    with pytest.raises(ValueError):
        block_encoding_circuit(matrix, alpha=1.0, backend=backend)
    with pytest.raises(ValueError):
        block_encoding_circuit(matrix, alpha=0.0, backend=backend)
    with pytest.raises(ValueError):
        block_encoding_circuit(matrix, method="lcu", alpha=2.0, backend=backend)
    with pytest.raises(ValueError):
        block_encoding_circuit(0.0 * matrix, method="lcu", backend=backend)


def test_block_encoding_circuit_exports():
    """The function is available from ``qibo.models``."""
    assert models.block_encoding_circuit is block_encoding_circuit


def test_block_encoding_circuit_pauli_strings(backend):
    """The LCU of a matrix with known Pauli coefficients."""
    pauli_x, pauli_y, pauli_z = (
        backend.matrices.X,
        backend.matrices.Y,
        backend.matrices.Z,
    )
    identity = backend.matrices.I()
    matrix = (
        0.5 * backend.kron(pauli_x, pauli_z)
        + 0.25j * backend.kron(pauli_y, identity)
        - 0.25 * backend.kron(identity, identity)
    )

    circuit, alpha, nauxiliary = block_encoding_circuit(
        matrix, method="lcu", backend=backend
    )

    assert alpha == pytest.approx(1.0, rel=1e-12)
    assert nauxiliary == 2
    backend.assert_allclose(circuit.unitary(backend)[:4, :4], matrix, atol=1e-10)


@pytest.mark.parametrize("degree", [2, 3, 4, 5])
@pytest.mark.parametrize("method", ["dense", "lcu"])
def test_block_encoding_circuit_qsvt(backend, method, degree):
    """The circuit is a valid input of QSVT, with the singular values of ``A / alpha``."""
    matrix = _matrix(2, seed=degree, norm=1.7, backend=backend)
    alpha = 2.0 if method == "dense" else None
    block_encoding, alpha, nauxiliary = block_encoding_circuit(
        matrix, method=method, alpha=alpha, backend=backend
    )

    phases = qsvt_phases([0.0] * degree + [1.0], backend=backend)
    circuit = qsvt_circuit(block_encoding, phases, nancillas=nauxiliary)

    # Chebyshev polynomial T_degree(x) = cos(degree * arccos(x))
    left, singular_values, right = backend.singular_value_decomposition(matrix / alpha)
    values = backend.cos(degree * backend.arccos(singular_values))
    if degree % 2 == 1:
        target = backend.matmul(left * values, right)
    else:
        target = backend.matmul(backend.dagger(right) * values, right)

    backend.assert_allclose(circuit.unitary(backend)[:2, :2], target, atol=1e-8)


def _matrix(dim: int, seed: int, norm: float, backend: Backend) -> ArrayLike:
    """Random complex matrix with the given spectral norm."""
    matrix = backend.random_normal(
        0.0, 1.0, size=(dim, dim), seed=seed, dtype="float64"
    ) + 1j * backend.random_normal(
        0.0, 1.0, size=(dim, dim), seed=seed + 100, dtype="float64"
    )

    return norm * matrix / backend.matrix_norm(matrix, order=2)
