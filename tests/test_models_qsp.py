"""Tests for quantum signal processing in `qibo/models/qsp.py`."""

import math

import pytest
from numpy.typing import ArrayLike

from qibo import Circuit, gates
from qibo.backends import Backend
from qibo.models.qsp import (
    _fourier_qsp_analytic_extension_coefficients,
    _fourier_qsp_bounded_error_coefficients,
    _qsp_hamiltonian_simulation_phases,
    qsp_circuit,
    qsp_phases,
)


@pytest.mark.parametrize(
    "name,epsilon",
    [("cosine", 1e-6), ("exponential", 1e-8), ("gaussian", 1e-4), ("decay", 1e-2)],
)
def test_fourier_qsp_analytic_extension_coefficients(backend, name, epsilon):
    """The series approximates the function with error ``epsilon`` in [-1, 1]."""
    functions = {
        "cosine": lambda x: 0.9 * backend.cos(2 * x),
        "exponential": lambda x: backend.exp(-1j * x),
        "gaussian": lambda x: 0.5 * backend.exp(-((x + 0.2) ** 2)),
        # bounded by one in [-1, 1], but larger than one right outside of it
        "decay": lambda x: backend.exp(-(x + 1)),
    }
    function = functions[name]

    coefficients, alpha, time = _fourier_qsp_analytic_extension_coefficients(
        function, epsilon, backend=backend
    )

    half = (len(coefficients) - 1) // 2
    frequencies = backend.arange(-half, half + 1)
    lambdas = 2 * backend.arange(101) / 100 - 1
    series = backend.matmul(
        backend.exp(1j * backend.outer(lambdas * time, frequencies)), coefficients
    )

    xs = math.pi * (2 * backend.arange(401) / 400 - 1)
    maximum = backend.max(
        backend.abs(
            backend.matmul(
                backend.exp(1j * backend.outer(xs, frequencies)), coefficients
            )
        )
    )

    assert time == 1.0
    backend.assert_allclose(series, function(lambdas), atol=epsilon, rtol=0.0)
    assert maximum <= 1 + epsilon
    # the modulus can slightly exceed one near the cut-off of the step, by at most epsilon
    assert 1 / (1 + epsilon) <= alpha <= 1.0
    assert alpha * maximum <= 1 + 1e-8


def test_fourier_qsp_analytic_extension_coefficients_errors(backend):
    with pytest.raises(ValueError):
        _fourier_qsp_analytic_extension_coefficients(backend.cos, 0.0, backend=backend)
    with pytest.raises(ValueError):
        _fourier_qsp_analytic_extension_coefficients(
            lambda x: 2 * backend.cos(x), 1e-3, backend=backend
        )
    # larger than one immediately outside of [-1, 1]
    with pytest.raises(ValueError):
        _fourier_qsp_analytic_extension_coefficients(
            lambda x: backend.where(backend.abs(x) > 1.0, 2.0, 0.5),
            1e-3,
            backend=backend,
        )
    # the step that cuts the function off would need more than 2^24 samples
    with pytest.raises(RuntimeError):
        _fourier_qsp_analytic_extension_coefficients(
            lambda x: backend.exp(-(x + 1)), 1e-7, backend=backend
        )
    # a jump inside of [-1, 1] prevents the Fourier series from converging
    with pytest.raises(RuntimeError):
        _fourier_qsp_analytic_extension_coefficients(
            lambda x: 0.5 * backend.where(x > 0, 1.0, -1.0), 1e-6, backend=backend
        )


@pytest.mark.parametrize("delta,epsilon", [(0.6, 1e-2), (0.5, 1e-3)])
def test_fourier_qsp_bounded_error_coefficients(backend, delta, epsilon):
    """Lemma 3: the series approximates ``alpha * f`` with error ``epsilon``."""
    beta = 1.0
    taylor = _exponential_taylor(beta, 16)

    coefficients, alpha, time = _fourier_qsp_bounded_error_coefficients(
        taylor, delta, epsilon, backend=backend
    )

    half = (len(coefficients) - 1) // 2
    frequencies = backend.arange(-half, half + 1)
    lambdas = 2 * backend.arange(101) / 100 - 1
    series = backend.matmul(
        backend.exp(1j * backend.outer(lambdas * time, frequencies)), coefficients
    )
    ratio = 1 - 2 * delta / math.pi
    norm = backend.sum(
        backend.abs(backend.cast(taylor, dtype="float64")) / ratio ** backend.arange(17)
    )
    expected_half = 2 * int(
        backend.ceil(backend.log(4 * alpha * norm / epsilon) / (2 * delta / math.pi))
    )

    assert time == pytest.approx(math.pi / 2 - delta)
    assert alpha == pytest.approx(1 / norm)
    assert half == expected_half
    backend.assert_allclose(
        series, alpha * backend.exp(-beta * (lambdas + 1)), atol=epsilon, rtol=0.0
    )
    assert backend.sum(backend.abs(coefficients)) <= 1.0

    # the coefficients are achievable, i.e. they can be turned into phases
    phases = qsp_phases(coefficients, method="fourier", backend=backend)
    xs = math.pi * (2 * backend.arange(21) / 20 - 1)
    target = backend.matmul(
        backend.exp(1j * backend.outer(xs, frequencies)), coefficients
    )
    amplitudes = backend.cast(
        [_fourier_rotations(phases, x, backend)[0, 0] for x in xs], dtype="complex128"
    )

    backend.assert_allclose(amplitudes, target, atol=1e-8, rtol=0.0)


def test_fourier_qsp_bounded_error_coefficients_errors(backend):
    with pytest.raises(ValueError):
        _fourier_qsp_bounded_error_coefficients([1.0, 0.1], 0.0, 1e-2, backend=backend)
    with pytest.raises(ValueError):
        _fourier_qsp_bounded_error_coefficients([1.0, 0.1], 2.0, 1e-2, backend=backend)
    with pytest.raises(ValueError):
        _fourier_qsp_bounded_error_coefficients([1.0, 0.1], 0.5, 1.5, backend=backend)


def test_fourier_qsp_imaginary_time_evolution(backend):
    """Coefficients, phases and circuit block-encode ``alpha * exp(-beta (H + 1))``."""
    beta, delta, epsilon = 1.0, 0.6, 1e-2
    coefficients, alpha, time = _fourier_qsp_bounded_error_coefficients(
        _exponential_taylor(beta, 16), delta, epsilon, backend=backend
    )

    hamiltonian = _hamiltonian(2, seed=3, norm=0.95, backend=backend)
    evolution = Circuit(2)
    evolution.add(gates.Unitary(backend.expm(-1j * time * hamiltonian), 0, 1))

    phases = qsp_phases(coefficients, method="fourier", backend=backend)
    circuit = qsp_circuit(evolution, phases, method="fourier")
    unitary = backend.reshape(circuit.unitary(backend), (2, 4, 2, 4))

    eigenvalues, eigenvectors = backend.eigh(hamiltonian)
    target = alpha * backend.matmul(
        eigenvectors * backend.exp(-beta * (eigenvalues + 1)),
        backend.dagger(eigenvectors),
    )

    assert backend.matrix_norm(unitary[0, :, 0, :] - target, order=2) < epsilon


@pytest.mark.parametrize(
    "extraction,tau",
    [("phase_sums", 0.5), ("phase_sums", 2.0), ("layer_stripping", 5.0)],
)
def test_qsp_circuit(backend, extraction, tau):
    """The ancilla-projected circuit must be the exact time evolution."""
    hamiltonian, walk = _walk(nqubits=2, seed=1, backend=backend)

    phases = _qsp_hamiltonian_simulation_phases(
        tau, 1e-8, extraction=extraction, backend=backend
    )
    circuit = qsp_circuit(walk, phases)
    unitary = backend.reshape(circuit.unitary(backend), (2, 2, 4, 2, 2, 4))

    # project both the QSP ancilla and the quantum walk ancilla onto |0>
    block = unitary[0, 0, :, 0, 0, :]
    target = backend.expm(-1j * tau * hamiltonian)

    assert circuit.nqubits == walk.nqubits + 1
    assert backend.matrix_norm(block - target, order=2) < 1e-6


def test_qsp_circuit_errors():
    walk = Circuit(1)
    walk.add(gates.RX(0, 0.3))

    with pytest.raises(ValueError):
        qsp_circuit(walk, [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        qsp_circuit(walk, [])
    with pytest.raises(ValueError):
        qsp_circuit(walk, [[0.0] * 4] * 4, method="fourier")
    with pytest.raises(ValueError):
        qsp_circuit(walk, [[0.0] * 4], method="fourier")
    with pytest.raises(ValueError):
        qsp_circuit(walk, [[0.0] * 3] * 3, method="fourier")
    with pytest.raises(ValueError):
        qsp_circuit(walk, [0.1, 0.2], method="unknown")


@pytest.mark.parametrize("degree", [2, 4, 8])
def test_qsp_circuit_fourier(backend, degree):
    """Theorem 2: the ancilla block is the operator Fourier series."""
    coefficients = _coefficients(degree, seed=degree, backend=backend)
    time = 0.8

    hamiltonian = _hamiltonian(2, seed=1, norm=1.0, backend=backend)
    evolution = Circuit(2)
    evolution.add(gates.Unitary(backend.expm(-1j * time * hamiltonian), 0, 1))

    phases = qsp_phases(coefficients, method="fourier", backend=backend)
    circuit = qsp_circuit(evolution, phases, method="fourier")
    unitary = backend.reshape(circuit.unitary(backend), (2, 4, 2, 4))

    eigenvalues, eigenvectors = backend.eigh(hamiltonian)
    frequencies = backend.arange(-degree // 2, degree // 2 + 1)
    series = backend.matmul(
        backend.exp(1j * backend.outer(eigenvalues * time, frequencies)),
        coefficients,
    )
    target = backend.matmul(eigenvectors * series, backend.dagger(eigenvectors))

    assert circuit.nqubits == 3
    assert backend.matrix_norm(unitary[0, :, 0, :] - target, order=2) < 1e-8


@pytest.mark.parametrize("epsilon", [1e-4, 1e-8])
@pytest.mark.parametrize(
    "extraction,tau",
    [
        ("phase_sums", 0.5),
        ("phase_sums", 1.0),
        ("layer_stripping", 0.5),
        ("layer_stripping", 3.0),
        ("layer_stripping", 8.0),
    ],
)
def test_qsp_hamiltonian_simulation_phases(backend, extraction, tau, epsilon):
    """Theorem 2: the trace distance is at most eight times the error."""
    phases = _qsp_hamiltonian_simulation_phases(
        tau, epsilon, extraction=extraction, backend=backend
    )

    thetas = math.pi * (2 * backend.arange(41) / 40 - 1)
    amplitudes = backend.cast(
        [_plus_amplitude(phases, theta, backend) for theta in thetas],
        dtype="complex128",
    )

    assert len(phases) % 2 == 0
    backend.assert_allclose(
        amplitudes,
        backend.exp(-1j * tau * backend.sin(thetas)),
        atol=8 * epsilon,
        rtol=0.0,
    )


def test_qsp_hamiltonian_simulation_phases_errors(backend):
    with pytest.raises(ValueError):
        _qsp_hamiltonian_simulation_phases(1.0, 0.0, backend=backend)


@pytest.mark.parametrize("extraction", ["phase_sums", "layer_stripping"])
def test_qsp_phases(backend, extraction):
    """Arbitrary target with ``A(0) = 1`` and polynomial degree ``N = 4``."""
    cosine = backend.cast([0.5, 0.3, 0.2], dtype="float64")
    sine = backend.cast([0.2, -0.1], dtype="float64")

    phases = qsp_phases(cosine, sine, extraction=extraction, backend=backend)

    thetas = math.pi * (2 * backend.arange(41) / 40 - 1)
    angles = backend.outer(thetas, backend.arange(1, 3))
    target = (
        cosine[0]
        + backend.matmul(backend.cos(angles), cosine[1:])
        + 1j * backend.matmul(backend.sin(angles), sine)
    )
    amplitudes = backend.cast(
        [_plus_amplitude(phases, theta, backend) for theta in thetas],
        dtype="complex128",
    )

    assert len(phases) == 4
    backend.assert_allclose(amplitudes, target, atol=1e-5, rtol=0.0)


def test_qsp_phases_errors(backend):
    with pytest.raises(ValueError):
        qsp_phases([1.0, 0.0], [], backend=backend)
    with pytest.raises(ValueError):
        qsp_phases([1.0, 0.0, 0.0], [0.1], backend=backend)
    with pytest.raises(ValueError):
        qsp_phases([1.0, 0.1], [0.1], extraction="unknown", backend=backend)
    with pytest.raises(ValueError):
        qsp_phases([1.0, 0.1], backend=backend)
    with pytest.raises(ValueError):
        qsp_phases([1.0, 0.1], [0.1], method="unknown", backend=backend)
    with pytest.raises(ValueError):
        qsp_phases([0.1, 0.5, 0.1], [0.1], method="fourier", backend=backend)
    with pytest.raises(ValueError):
        qsp_phases([1.0], method="fourier", backend=backend)
    with pytest.raises(ValueError):
        qsp_phases([0.1, 0.5, 0.1, 0.2], method="fourier", backend=backend)


@pytest.mark.parametrize("degree", [2, 4, 10, 30, 200])
def test_qsp_phases_fourier(backend, degree):
    coefficients = _coefficients(degree, seed=degree, backend=backend)

    phases = qsp_phases(coefficients, method="fourier", backend=backend)

    xs = math.pi * (2 * backend.arange(41) / 40 - 1)
    frequencies = backend.arange(-degree // 2, degree // 2 + 1)
    target = backend.matmul(
        backend.exp(1j * backend.outer(xs, frequencies)), coefficients
    )
    amplitudes = backend.cast(
        [_fourier_rotations(phases, x, backend)[0, 0] for x in xs], dtype="complex128"
    )

    assert phases.shape == (degree + 1, 4)
    backend.assert_allclose(amplitudes, target, atol=1e-8, rtol=0.0)


def test_qsp_phases_fourier_normalization(backend):
    """Coefficients that break normalization are divided by the maximum modulus."""
    coefficients = 3 * _coefficients(4, seed=0, backend=backend)

    phases = qsp_phases(coefficients, method="fourier", backend=backend)

    xs = math.pi * (2 * backend.arange(41) / 40 - 1)
    target = backend.matmul(
        backend.exp(1j * backend.outer(xs, backend.arange(-2, 3))), coefficients
    )
    target = target / backend.max(backend.abs(target))
    amplitudes = backend.cast(
        [_fourier_rotations(phases, x, backend)[0, 0] for x in xs], dtype="complex128"
    )

    backend.assert_allclose(amplitudes, target, atol=1e-2, rtol=0.0)


def test_qsp_phases_fourier_unimodular(backend):
    """The series ``exp(i x)`` has unit modulus for all ``x``."""
    phases = qsp_phases([0.0, 0.0, 1.0], method="fourier", backend=backend)

    xs = math.pi * (2 * backend.arange(21) / 20 - 1)
    amplitudes = backend.cast(
        [_fourier_rotations(phases, x, backend)[0, 0] for x in xs], dtype="complex128"
    )

    backend.assert_allclose(amplitudes, backend.exp(1j * xs), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("extraction", ["phase_sums", "layer_stripping"])
def test_qsp_phases_unimodular(backend, extraction):
    """The target ``cos(theta) + i sin(theta)`` has unit modulus for all ``theta``."""
    phases = qsp_phases([0.0, 1.0], [1.0], extraction=extraction, backend=backend)

    thetas = math.pi * (2 * backend.arange(21) / 20 - 1)
    amplitudes = backend.cast(
        [_plus_amplitude(phases, theta, backend) for theta in thetas],
        dtype="complex128",
    )

    backend.assert_allclose(amplitudes, backend.exp(1j * thetas), atol=1e-8, rtol=0.0)


def _coefficients(degree: int, seed: int, backend: Backend) -> ArrayLike:
    """Random Fourier coefficients with maximum modulus ``0.9``."""
    coefficients = backend.random_normal(
        0.0, 1.0, size=degree + 1, seed=seed, dtype="float64"
    ) + 1j * backend.random_normal(
        0.0, 1.0, size=degree + 1, seed=seed + 100, dtype="float64"
    )
    xs = 2 * math.pi * backend.arange(2000) / 2000
    frequencies = backend.arange(-degree // 2, degree // 2 + 1)
    modulus = backend.max(
        backend.abs(
            backend.matmul(
                backend.exp(1j * backend.outer(xs, frequencies)), coefficients
            )
        )
    )

    return 0.9 * coefficients / modulus


def _exponential_taylor(beta: float, order: int) -> list[float]:
    """Taylor coefficients of ``exp(-beta (x + 1))``, up to the given order."""
    coefficients = [math.exp(-beta)]
    for k in range(1, order + 1):
        coefficients.append(coefficients[-1] * (-beta) / k)

    return coefficients


def _fourier_rotations(phases: ArrayLike, x: float, backend: Backend) -> ArrayLike:
    """Product of the single-qubit gates :math:`R_{q}(x) \\cdots R_{0}(x)`."""
    matrix = backend.matrices.I()
    for index, (zeta, eta, varphi, kappa) in enumerate(phases):
        omega = 0.0 if index == 0 else (-1) ** (index + 1) / 2
        for factor in (
            backend.expm(-1j * kappa * backend.matrices.Y),
            backend.expm(1j * omega * x * backend.matrices.Z),
            backend.expm(0.5j * (zeta - eta) * backend.matrices.Z),
            backend.expm(-1j * varphi * backend.matrices.Y),
            backend.expm(0.5j * (zeta + eta) * backend.matrices.Z),
        ):
            matrix = backend.matmul(factor, matrix)

    return matrix


def _hamiltonian(nqubits: int, seed: int, norm: float, backend: Backend) -> ArrayLike:
    """Random Hermitian matrix with the given spectral norm."""
    dim = 2**nqubits
    hamiltonian = backend.random_normal(
        0.0, 1.0, size=(dim, dim), seed=seed, dtype="float64"
    ) + 1j * backend.random_normal(
        0.0, 1.0, size=(dim, dim), seed=seed + 100, dtype="float64"
    )
    hamiltonian = (hamiltonian + backend.dagger(hamiltonian)) / 2

    return norm * hamiltonian / backend.matrix_norm(hamiltonian, order=2)


def _plus_amplitude(phases: ArrayLike, theta: float, backend: Backend) -> complex:
    """Matrix element :math:`\\langle + | R_{\\phi_{N}} \\cdots R_{\\phi_{1}} | + \\rangle`."""
    plus = backend.cast([1.0, 1.0], dtype="complex128") / backend.sqrt(2.0)
    matrix = plus
    for phase in phases:
        axis = (
            backend.cos(phase) * backend.matrices.X
            + backend.sin(phase) * backend.matrices.Y
        )
        matrix = backend.matmul(backend.expm(-0.5j * theta * axis), matrix)

    return backend.matmul(backend.conj(plus), matrix)


def _walk(nqubits: int, seed: int, backend: Backend) -> tuple[ArrayLike, Circuit]:
    """Qubitized walk with :math:`\\sin(\\theta_{\\lambda}) = \\lambda`."""
    dim = 2**nqubits
    hamiltonian = _hamiltonian(nqubits, seed, norm=1 / 1.1, backend=backend)

    matrix = 1j * backend.kron(backend.matrices.I(), hamiltonian) - backend.kron(
        backend.matrices.Y,
        backend.matrix_sqrt(
            backend.identity(dim, dtype="complex128")
            - backend.matmul(hamiltonian, hamiltonian)
        ),
    )

    walk = Circuit(nqubits + 1)
    walk.add(gates.Unitary(matrix, *range(nqubits + 1)))

    return hamiltonian, walk
