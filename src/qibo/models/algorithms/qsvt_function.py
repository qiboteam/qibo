import math
from collections.abc import Callable
from numbers import Integral

from numpy.typing import ArrayLike

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.models.algorithms.block_encoding import block_encoding_circuit
from qibo.models.algorithms.lcu import lcu_circuit
from qibo.models.algorithms.qsvt import qsvt_circuit, qsvt_phases
from qibo.models.circuit import Circuit


def qsvt_function(
    matrix: ArrayLike | Hamiltonian | SymbolicHamiltonian,
    function: Callable | ArrayLike,
    degree: int | None = None,
    epsilon: float = 1e-6,
    backend: Backend = None,
    **kwargs,
) -> tuple[Circuit, float]:
    """Creates the circuit that applies a function of a Hermitian matrix with QSVT.

    Let :math:`A` be a Hermitian matrix of size :math:`2^{n} \\times 2^{n}`, with eigenvalues
    :math:`\\lambda` and :math:`\\alpha = \\max |\\lambda|`, and let :math:`f` be a function of
    the eigenvalues. The matrix is block-encoded with
    :func:`qibo.models.block_encoding_circuit`, and :math:`f(\\alpha x)` is approximated in
    :math:`x \\in [-1, 1]` by its Chebyshev series, which is truncated where its coefficients are
    below ``epsilon`` times the largest one. A quantum singular value transformation (QSVT) of
    the block encoding applies a real polynomial of a definite parity, which is bounded by one
    (Corollary 11 of Ref. [1]), and for a Hermitian matrix it is the same polynomial of the
    eigenvalues. Hence, the real and the imaginary parts of the even and of the odd terms of the
    series are four polynomials, which are divided by their maximum :math:`s_{j}` and applied by
    :func:`qibo.models.qsvt_circuit`, with the phases of :func:`qibo.models.qsvt_phases`. The terms
    that are not zero are added with a linear combination of unitaries,
    :func:`qibo.models.lcu_circuit`, whose coefficients are :math:`s_{j}` times :math:`1` or
    :math:`i`. If the function has only one of them, there is no linear combination.

    The circuit acts on the qubits of the linear combination, if there is one, followed by the
    signal qubit of the QSVT, the auxiliary qubit of the block encoding, and the :math:`n` qubits
    of :math:`A`. If all of the qubits but the last :math:`n` start and end in :math:`\\ket{0}`,
    the circuit applies :math:`C f(A)` to the last :math:`n` qubits, where :math:`C` is the
    constant that it returns, which is :math:`1 / \\sum_{j} s_{j}`. It is the first
    :math:`2^{n} \\times 2^{n}` block of its unitary. The matrix is block-encoded exactly, which
    is inefficient in the number of qubits :math:`n`, and it is only meant for small systems.

    Example:
        Evolve :math:`\\ket{0}` with :math:`H = X` during the time :math:`t = 0.7`, which is
        :math:`e^{-i H t} \\ket{0} = \\cos(t) \\ket{0} - i \\sin(t) \\ket{1}`.

        .. testcode::

            from qibo.backends import NumpyBackend
            from qibo.models import qsvt_function

            backend = NumpyBackend()

            matrix = backend.matrices.X
            time = 0.7

            circuit, constant = qsvt_function(matrix, lambda x: backend.exp(-1j * time * x))

            # the first two amplitudes are those of the qubit when the others are zero
            state = circuit().state()[:2]
            probabilities = backend.abs(state) ** 2

            print(backend.round(probabilities / backend.sum(probabilities), 3))

        .. testoutput::

            [0.585 0.415]

    Args:
        matrix (ArrayLike or :class:`qibo.hamiltonians.abstract.AbstractHamiltonian`):
            Hermitian matrix :math:`A`, or Hamiltonian with its matrix. Its size is
            :math:`2^{n} \\times 2^{n}`, with :math:`n \\geq 1`.
        function (Callable or ArrayLike): the function :math:`f`, which is called with the array
            of the values :math:`\\alpha x` at the Chebyshev nodes and returns the array of its
            values, which can be complex. Alternatively, the Chebyshev coefficients
            :math:`(c_{0}, \\dots, c_{d})` of the polynomial :math:`P(x)`, so that
            :math:`f(\\lambda) = P(\\lambda / \\alpha)`. The coefficients below :math:`10^{-12}`
            times the largest one are taken as zero.
        degree (int, optional): degree of the Chebyshev interpolation of a function. If ``None``,
            it is doubled from :math:`8` to :math:`1024` until the last quarter of the coefficients
            is below ``epsilon`` times the largest one, and a ``ValueError`` is raised if it never
            is. It is not used for the coefficients. Defaults to ``None``.
        epsilon (float, optional): tolerance of the coefficients of a function, relative to the
            largest one, in :math:`(0, 1)`. Defaults to :math:`10^{-6}`, which is the precision
            of the phases of :func:`qibo.models.qsvt_phases`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        tuple: Circuit and the constant :math:`C`.

    References:
        1. A. Gilyén, Y. Su, G. H. Low, and N. Wiebe, *Quantum singular value
        transformation and beyond: exponential improvements for quantum matrix
        arithmetics*, `Proceedings of the 51st Annual ACM SIGACT Symposium on Theory
        of Computing (STOC '19), 193-204 (2019)
        <https://doi.org/10.1145/3313276.3316366>`_.
    """
    backend = _check_backend(backend)

    if not 0 < epsilon < 1:
        raise_error(ValueError, f"epsilon must be in (0, 1), but it is {epsilon}.")
    if degree is not None and (
        isinstance(degree, bool) or not isinstance(degree, Integral) or degree < 1
    ):
        raise_error(
            ValueError, f"degree must be a positive integer, but it is {degree}."
        )

    matrix, _ = _hermitian_matrix(matrix, backend)
    alpha = float(backend.max(backend.abs(backend.eigh(matrix)[0])))
    if alpha == 0.0:
        raise_error(ValueError, "matrix must not be zero.")

    if callable(function):
        threshold = epsilon
        for trial in [degree] if degree is not None else [2**k for k in range(3, 11)]:
            coefficients = _chebyshev_series(function, alpha, trial, backend)
            magnitudes = backend.abs(coefficients)
            if degree is not None or backend.max(
                magnitudes[3 * (trial + 1) // 4 :]
            ) <= epsilon * backend.max(magnitudes):
                break
        else:
            raise_error(
                ValueError,
                f"function is not approximated with a tolerance of {epsilon} by "
                + "a polynomial of degree 1024.",
            )
    else:
        threshold = 1e-12
        coefficients = backend.cast(function, dtype=backend.complex128)
        if len(coefficients.shape) != 1 or coefficients.shape[0] < 2:
            raise_error(
                ValueError,
                "the coefficients of function must be a vector of at least two entries.",
            )

    largest = backend.max(backend.abs(coefficients))
    if largest == 0.0:
        raise_error(ValueError, "function must not be zero.")
    significant = backend.nonzero(backend.abs(coefficients) > threshold * largest)[0]
    coefficients = coefficients[: int(significant[-1]) + 1]

    block, _, nauxiliary = block_encoding_circuit(
        matrix, method="dense", alpha=alpha, backend=backend
    )

    # polynomials of a definite parity, from the real and the imaginary parts of the series
    unitaries, weights = [], []
    for parity in (0, 1):
        mask = backend.cast(
            [float(order % 2 == parity) for order in range(coefficients.shape[0])],
            dtype=backend.float64,
        )
        for phase, part in ((1.0, backend.real), (1j, backend.imag)):
            polynomial = part(coefficients) * mask
            if backend.max(backend.abs(polynomial)) <= threshold * largest:
                continue

            polynomial = polynomial[: int(backend.nonzero(polynomial)[0][-1]) + 1]
            if polynomial.shape[0] == 1:
                # a constant is the identity
                unitaries.append(Circuit(block.nqubits + 1))
                weights.append(float(polynomial[0]) * phase)
            else:
                peak = _peak(polynomial, backend)
                phases = qsvt_phases(polynomial / peak, backend=backend)
                unitaries.append(qsvt_circuit(block, phases, nauxiliary))
                weights.append(peak * phase)

    angle = 0.0
    if len(unitaries) == 1:
        transformation, constant = unitaries[0], 1 / abs(weights[0])
        angle = math.atan2(weights[0].imag, weights[0].real)
    else:
        transformation, normalization, _ = lcu_circuit(
            unitaries, weights, backend=backend
        )
        constant = 1 / normalization

    circuit = Circuit(transformation.nqubits, **kwargs)
    circuit.add(transformation.on_qubits(*range(transformation.nqubits)))
    if angle != 0.0:
        circuit.add(gates.GlobalPhase(angle))

    return circuit, float(constant)


def _chebyshev_series(
    function: Callable, alpha: float, degree: int, backend: Backend = None
) -> ArrayLike:
    """Computes the Chebyshev coefficients of :math:`f(\\alpha x)` by interpolation.

    Args:
        function (Callable): function that is called with the array of the values
            :math:`\\alpha x` at the Chebyshev nodes.
        alpha (float): scale :math:`\\alpha` of the variable.
        degree (int): degree of the interpolation.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the
            calculation. If ``None``, it uses the current backend. Defaults to ``None``.

    Returns:
        ArrayLike: complex Chebyshev coefficients :math:`(c_{0}, \\dots, c_{d})`.
    """
    backend = _check_backend(backend)

    angles = (
        math.pi
        * (backend.arange(degree + 1, dtype=backend.float64) + 0.5)
        / (degree + 1)
    )
    values = backend.cast(
        function(alpha * backend.cos(angles)), dtype=backend.complex128
    )

    orders = backend.reshape(backend.arange(degree + 1, dtype=backend.float64), (-1, 1))
    weights = backend.cast([0.5] + degree * [1.0], dtype=backend.float64) * (
        2 / (degree + 1)
    )

    return weights * backend.matmul(
        backend.cast(backend.cos(orders * angles), dtype=backend.complex128), values
    )


def _hermitian_matrix(
    matrix: ArrayLike | Hamiltonian | SymbolicHamiltonian, backend: Backend = None
) -> tuple[ArrayLike, int]:
    """Checks that a matrix is Hermitian, and that its size is a power of two.

    Args:
        matrix (ArrayLike or :class:`qibo.hamiltonians.abstract.AbstractHamiltonian`):
            matrix, or Hamiltonian with its matrix.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the
            calculation. If ``None``, it uses the current backend. Defaults to ``None``.

    Returns:
        tuple: complex matrix and the number :math:`n` of qubits of its size :math:`2^{n}`.
    """
    backend = _check_backend(backend)

    if isinstance(matrix, (Hamiltonian, SymbolicHamiltonian)):
        matrix = matrix.matrix
    matrix = backend.cast(matrix, dtype=backend.complex128)

    shape = tuple(matrix.shape)
    if (
        len(shape) != 2
        or shape[0] != shape[1]
        or shape[0] < 2
        or shape[0] & (shape[0] - 1) != 0
    ):
        raise_error(
            ValueError,
            f"matrix must be a square matrix of size 2^n with n >= 1, but its shape is {shape}.",
        )

    hermitian = backend.matrix_norm(
        matrix - backend.conj(backend.transpose(matrix)), order="fro"
    )
    if hermitian > 1e-8 * backend.matrix_norm(matrix, order="fro"):
        raise_error(ValueError, "matrix must be Hermitian.")

    return matrix, shape[0].bit_length() - 1


def _peak(polynomial: ArrayLike, backend: Backend = None) -> float:
    """Computes the maximum of the absolute value of a Chebyshev series in :math:`[-1, 1]`.

    Args:
        polynomial (ArrayLike): real Chebyshev coefficients :math:`(c_{0}, \\dots, c_{d})`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the
            calculation. If ``None``, it uses the current backend. Defaults to ``None``.

    Returns:
        float: maximum in a grid of :math:`64 (d + 1)` Chebyshev points.
    """
    backend = _check_backend(backend)

    degree = polynomial.shape[0] - 1
    nsamples = 64 * (degree + 1)
    angles = backend.reshape(
        math.pi * backend.arange(nsamples, dtype=backend.float64) / nsamples, (-1, 1)
    )
    values = backend.matmul(
        backend.cos(angles * backend.arange(degree + 1, dtype=backend.float64)),
        polynomial,
    )

    return float(backend.max(backend.abs(values)))
