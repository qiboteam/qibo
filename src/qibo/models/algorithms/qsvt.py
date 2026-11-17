import math
from collections.abc import Callable
from numbers import Integral

from numpy.typing import ArrayLike

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.models.algorithms._signal_processing import _spectral_factor
from qibo.models.algorithms.block_encoding import block_encoding_circuit
from qibo.models.algorithms.lcu import lcu_circuit
from qibo.models.circuit import Circuit


def qsvt_circuit(block_encoding: Circuit, phases: ArrayLike, nancillas: int) -> Circuit:
    """Creates a quantum singular value transformation (QSVT) circuit.

    Let :math:`U` be the unitary of ``block_encoding``, which acts on :math:`a`
    auxiliary qubits and :math:`n` system qubits, and let :math:`A = (\\bra{0}^{\\otimes a}
    \\otimes \\mathbb{1}) \\, U \\, (\\ket{0}^{\\otimes a} \\otimes \\mathbb{1})` be the
    matrix that it encodes, with singular value decomposition :math:`A = \\sum_{i}
    \\varsigma_{i} \\ket{w_{i}}\\!\\bra{v_{i}}`. The block of the circuit unitary in
    which the signal qubit and the auxiliary qubits start and end in :math:`\\ket{0}`
    applies a polynomial :math:`P` of degree :math:`d` to the singular values,

    .. math::
        P^{(SV)}(A) = \\begin{cases} \\sum_{i} P(\\varsigma_{i}) \\,
        \\ket{w_{i}}\\!\\bra{v_{i}} & d \\text{ odd,} \\\\ \\sum_{i} P(\\varsigma_{i}) \\,
        \\ket{v_{i}}\\!\\bra{v_{i}} & d \\text{ even.} \\end{cases}

    The circuit queries :math:`U` and :math:`U^{\\dagger}` alternately :math:`d` times
    (Theorem 2 of Ref. [1]) and uses a single extra qubit, the signal qubit, which
    controls the sign of the phases of the rotations that are controlled by the
    projector on the auxiliary qubits in :math:`|0\\rangle`.

    Example:
        Apply the cubic Chebyshev polynomial :math:`T_{3}(x) = 4 x^{3} - 3 x` to the
        singular value :math:`1/2` of a block-encoding of :math:`\\ket{+} / 2`.
        Since :math:`T_{3}(1/2) = -1`, the amplitudes with the signal qubit and the
        auxiliary qubit in :math:`\\ket{0}` are those of :math:`-\\ket{+}`.

        .. testcode::

            import math

            from qibo import Circuit, gates, get_backend
            from qibo.models import qsvt_circuit, qsvt_phases

            backend = get_backend()

            # prepares |+> with amplitude 1/2 when the auxiliary qubit is in |0>
            block_encoding = Circuit(2)
            block_encoding.add(gates.RY(0, 2 * math.pi / 3))
            block_encoding.add(gates.H(1))

            # Chebyshev series of the polynomial, i.e. only the coefficient of T_3
            phases = qsvt_phases([0.0, 0.0, 0.0, 1.0])
            circuit = qsvt_circuit(block_encoding, phases, nancillas=1)

            target = Circuit(1)
            target.add(gates.H(0))

            state = circuit().state()[:2]
            print(backend.allclose(state, -target().state(), atol=1e-6))

        .. testoutput::

            True

    Args:
        block_encoding (:class:`qibo.models.circuit.Circuit`): circuit that
            implements :math:`U`. Its first ``nancillas`` qubits are the auxiliary qubits.
        phases (ArrayLike): phases of the polynomial, *e.g.* the output of
            :func:`qibo.models.qsvt_phases`.
        nancillas (int): number :math:`a` of auxiliary qubits of ``block_encoding``.

    Returns:
        :class:`qibo.models.circuit.Circuit`: circuit with the signal qubit at position
        :math:`0`, followed by the qubits of ``block_encoding`` in the same order.

    References:
        1. A. Gilyén, Y. Su, G. H. Low, and N. Wiebe, *Quantum singular value
        transformation and beyond: exponential improvements for quantum matrix
        arithmetics*, `Proceedings of the 51st Annual ACM SIGACT Symposium on Theory
        of Computing (STOC '19), 193-204 (2019)
        <https://doi.org/10.1145/3313276.3316366>`_.
    """
    phases = [float(phase) for phase in phases]
    nqubits = block_encoding.nqubits

    if len(phases) < 2:
        raise_error(ValueError, "``phases`` must have at least two elements.")

    if not 0 < nancillas <= nqubits:
        raise_error(
            ValueError,
            f"``nancillas`` must be between 1 and {nqubits}, but is {nancillas}.",
        )

    qubits = list(range(1, nqubits + 1))
    ancillas = qubits[:nancillas]
    inverse = block_encoding.invert()

    circuit = Circuit(nqubits + 1)
    circuit.add(gates.H(0))
    for index, phase in enumerate(phases[:-1]):
        # rotation exp(i * phase * (2 Pi - 1)), with Pi the projector on the ancillas
        # in |0>, that is controlled by the sign of the signal qubit (Fig. 3b of Ref. [1])
        circuit.add(gates.X(qubit) for qubit in ancillas)
        circuit.add(gates.X(0).controlled_by(*ancillas))
        circuit.add(gates.RZ(0, 2 * phase))
        circuit.add(gates.X(0).controlled_by(*ancillas))
        circuit.add(gates.X(qubit) for qubit in ancillas)

        oracle = block_encoding if index % 2 == 0 else inverse
        circuit.add(oracle.on_qubits(*qubits))
    circuit.add(gates.RZ(0, -2 * phases[-1]))
    circuit.add(gates.H(0))

    return circuit


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
    :math:`i`. If the function has only one of them, there is no linear combination. A part with
    only a constant term is applied as a multiple of the identity, whose sign is a global phase if
    there is no other part.

    The circuit acts on the qubits of the linear combination, if there is one, followed by the
    signal qubit of the QSVT, the auxiliary qubit of the block encoding, and the :math:`n` qubits
    of :math:`A`. If all of the qubits but the last :math:`n` start and end in :math:`\\ket{0}`,
    the circuit applies :math:`C f(A)` to the last :math:`n` qubits, where :math:`C` is the
    constant that it returns, which is :math:`1 / \\sum_{j} s_{j}`, up to a relative error of
    about :math:`10^{-6}`, the precision of the phases. It is the first
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
            is. If given, it is used as is, without checking that the coefficients converge. It is
            not used for the coefficients. Defaults to ``None``.
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


def qsvt_phases(coefficients: ArrayLike, backend: Backend = None) -> ArrayLike:
    """Computes the phases of a quantum singular value transformation (QSVT) circuit.

    Given the Chebyshev series :math:`P(x) = \\sum_{k = 0}^{d} c_{k} T_{k}(x)` of a real
    polynomial of degree :math:`d \\ge 1` with the parity of :math:`d` and
    :math:`|P(x)| \\le 1` for :math:`x \\in [-1, 1]`, it finds the phases for which
    :func:`qibo.models.qsvt_circuit` applies :math:`P` to the singular values
    (Corollary 11 of Ref. [1]). The polynomial is completed into a unitary matrix whose
    entries are trigonometric polynomials, using the spectral factorization of
    :math:`1 - P^{2}` that :func:`qibo.models.qsp_phases` uses for the Fourier-based
    QSP, and the phases are then extracted one at a time as in Ref. [2].

    The coefficients are divided by its (grid-estimated) :math:`\\max_{x} |P(x)|`
    if it is larger than one. If :math:`|P|` reaches one, the polynomial is implemented
    with a relative error of about :math:`10^{-6}` for degrees up to :math:`100` and
    :math:`10^{-5}` up to :math:`300`, and the error is much smaller for polynomials that
    stay away from one.

    Args:
        coefficients (ArrayLike): Chebyshev coefficients :math:`(c_{0}, \\dots, c_{d})`.
            The degree is that of the last non-zero coefficient, and the coefficients
            of the opposite parity must vanish. Those with modulus below :math:`10^{-12}`
            times the largest one are taken as zero.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ArrayLike: :math:`d + 1` phases. The first :math:`d` are those of the rotations
        controlled by the projector, in the order that they are applied, and the last
        one is the final rotation of the signal qubit.

    References:
        1. A. Gilyén, Y. Su, G. H. Low, and N. Wiebe, *Quantum singular value
        transformation and beyond: exponential improvements for quantum matrix
        arithmetics*, `Proceedings of the 51st Annual ACM SIGACT Symposium on Theory
        of Computing (STOC '19), 193-204 (2019)
        <https://doi.org/10.1145/3313276.3316366>`_.
        2. J. Haah, *Product decomposition of periodic functions in quantum signal
        processing*, `Quantum 3, 190 (2019)
        <https://doi.org/10.22331/q-2019-10-07-190>`_.
    """
    backend = _check_backend(backend)

    coefficients = backend.cast(coefficients, dtype="float64")

    if len(coefficients) < 2:
        raise_error(
            ValueError,
            "``coefficients`` must define a polynomial of degree at least one.",
        )

    # coefficients at the level of the round-off error are set to zero
    coefficients = backend.where(
        backend.abs(coefficients) > 1e-12 * backend.max(backend.abs(coefficients)),
        coefficients,
        0.0,
    )
    nonzero = backend.flatnonzero(coefficients)
    degree = int(nonzero[-1]) if len(nonzero) > 0 else 0
    coefficients = coefficients[: degree + 1]

    if degree < 1:
        raise_error(
            ValueError,
            "``coefficients`` must define a polynomial of degree at least one.",
        )

    if backend.any(coefficients[1 - degree % 2 :: 2] != 0.0):
        raise_error(
            ValueError, "``coefficients`` must define a polynomial of definite parity."
        )

    def evaluate(nsamples: int) -> ArrayLike:
        # the series at theta_j = pi * j / nsamples, with x = cos(theta)
        padded = backend.zeros(2 * nsamples, dtype="complex128")
        padded[: degree + 1] = coefficients

        return backend.real(backend.ifft(padded) * 2 * nsamples)[:nsamples]

    # Normalize the polynomial so that its modulus is bounded by one.
    # 128 is a heuristic for resolving the peak of the Chebyshev series on the nbase grid.
    # It works well for the tested degrees. It could be turn into a hyperparameter
    nbase = 2 ** max(12, int(backend.ceil(backend.log2(128 * (degree + 1)))))
    peak = float(backend.max(backend.abs(evaluate(nbase))))
    scale = 1.0 / max(peak, 1.0)

    # Complementary polynomials: 1 - P^2 = B^2 + D^2, with B and D series of cosines and
    # sines, respectively, so that P + i B and D are the entries of an SU(2) matrix.
    # The margin, which is increased until the factorization converges, keeps 1 - P^2
    # positive when |P| touches one.
    for margin in 10.0 ** backend.arange(-14, -1):
        minimum = 1 - (scale * (1 - margin) * peak) ** 2
        # the grid has to resolve the near-zeros of 1 - P^2, at a distance ~ 1 / d
        nsamples = max(
            nbase,
            2 ** int(backend.ceil(backend.log2(16 * degree / backend.sqrt(minimum)))),
        )
        if nsamples > 2**21:
            continue

        values = scale * (1 - margin) * evaluate(nsamples)
        residual = 1 - values**2
        # the maximum is estimated on a coarser grid, and the polynomial can exceed it between
        # the nodes of that one, where 1 - P^2 is negative and has no factor
        if backend.min(residual) <= 0:
            continue
        factor, converged = _spectral_factor(residual, degree, backend)
        if converged:
            break
    else:  # pragma: no cover
        raise_error(RuntimeError, "Complementary polynomials not found.")

    # Coefficients of the matrix P + i B + i D Y in powers of w = e^{i theta}: the
    # coefficient of w^{-d + 2j} is ``matrix[j]``.
    factor = backend.real(factor)
    orders = backend.arange(degree % 2, degree + 1, 2)
    laurent_a = backend.zeros(degree + 1, dtype="complex128")
    laurent_a[(degree + orders) // 2] += scale * (1 - margin) * coefficients[orders] / 2
    laurent_a[(degree - orders) // 2] += scale * (1 - margin) * coefficients[orders] / 2
    laurent_b = (factor + backend.flip(factor)) / 2
    laurent_d = (factor - backend.flip(factor)) / 2j

    identity = backend.matrices.I()
    matrix = (
        laurent_a[:, None, None] * identity
        + 1j * laurent_b[:, None, None] * backend.matrices.Z
        + 1j * laurent_d[:, None, None] * backend.matrices.Y
    )

    # Layer stripping of matrix = exp(i psi_d Z) W exp(i psi_{d-1} Z) ... W exp(i psi_0 Z),
    # where W = exp(i theta X) = w |+><+| + w^{-1} |-><-|. The highest-degree coefficient
    # has rank one with range exp(i psi Z) |+>, which gives the leftmost phase.
    plus = (identity + backend.matrices.X) / 2
    minus = (identity - backend.matrices.X) / 2
    angles = backend.zeros(degree + 1, dtype="float64")
    for step in range(degree, 0, -1):
        left = backend.singular_value_decomposition(matrix[step])[0][:, 0]
        angles[step] = (backend.angle(left[0]) - backend.angle(left[1])) / 2

        rotation = backend.cast(
            [[backend.exp(-1j * angles[step]), 0], [0, backend.exp(1j * angles[step])]],
            dtype="complex128",
        )
        rotated = backend.matmul(rotation, matrix)
        matrix = backend.matmul(plus, rotated[1 : step + 1]) + backend.matmul(
            minus, rotated[:step]
        )
    angles[0] = backend.angle(matrix[0][0, 0])

    # Change of convention from the products of W to those of the reflections in the
    # circuit, R(x) = [[x, sqrt(1 - x^2)], [sqrt(1 - x^2), -x]] (Corollary 5 of Ref. [1]).
    phases = backend.zeros(degree + 1, dtype="float64")
    phases[:degree] = angles[:degree] - math.pi / 2
    phases[degree] = angles[degree] + degree * math.pi / 2

    return phases


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
    # the first coefficient has half of the weight of the others
    weights = [0.5]
    weights.extend(degree * [1.0])
    weights = backend.cast(weights, dtype=backend.float64) * (2 / (degree + 1))

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

    The maxima of a grid of :math:`64 (d + 1)` Chebyshev points, which it underestimates by up to
    :math:`10^{-3}`, are refined with Newton iterations on the derivative of the series.

    Args:
        polynomial (ArrayLike): real Chebyshev coefficients :math:`(c_{0}, \\dots, c_{d})`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the
            calculation. If ``None``, it uses the current backend. Defaults to ``None``.

    Returns:
        float: maximum of the series.
    """
    backend = _check_backend(backend)

    degree = polynomial.shape[0] - 1
    nsamples = 64 * (degree + 1)
    orders = backend.arange(degree + 1, dtype=backend.float64)
    angles = math.pi * backend.arange(nsamples, dtype=backend.float64) / nsamples
    magnitudes = backend.abs(
        backend.matmul(
            backend.cos(backend.reshape(angles, (-1, 1)) * orders), polynomial
        )
    )
    peak = float(backend.max(magnitudes))

    # the grid points that are close to the maximum are the starting points of the iterations
    nodes = angles[magnitudes >= (1 - 1e-3) * peak]
    for _ in range(5):
        grid = backend.reshape(nodes, (-1, 1)) * orders
        cosines = backend.cos(grid)
        peak = max(
            peak, float(backend.max(backend.abs(backend.matmul(cosines, polynomial))))
        )

        first = -backend.matmul(backend.sin(grid), orders * polynomial)
        second = -backend.matmul(cosines, orders**2 * polynomial)
        nodes = nodes - first / backend.where(second == 0.0, 1.0, second)

    return peak
