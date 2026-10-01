import math

from numpy.typing import ArrayLike

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.models.circuit import Circuit
from qibo.models.qsp import _spectral_factor


def qsvt_circuit(block_encoding: Circuit, phases: ArrayLike, nancillas: int) -> Circuit:
    """Creates a quantum singular value transformation (QSVT) circuit.

    Let :math:`U` be the unitary of ``block_encoding``, which acts on :math:`a`
    ancilla qubits and :math:`n` system qubits, and let :math:`A = (\\langle 0|^{\\otimes a}
    \\otimes \\mathbb{1}) \\, U \\, (|0\\rangle^{\\otimes a} \\otimes \\mathbb{1})` be the
    matrix that it encodes, with singular value decomposition :math:`A = \\sum_{i}
    \\varsigma_{i} |w_{i}\\rangle\\langle v_{i}|`. The block of the circuit unitary in
    which the signal qubit and the ancillas start and end in :math:`|0\\rangle`
    applies a polynomial :math:`P` of degree :math:`d` to the singular values,

    .. math::
        P^{(SV)}(A) = \\begin{cases} \\sum_{i} P(\\varsigma_{i}) \\,
        |w_{i}\\rangle\\langle v_{i}| & d \\text{ odd,} \\\\ \\sum_{i} P(\\varsigma_{i}) \\,
        |v_{i}\\rangle\\langle v_{i}| & d \\text{ even.} \\end{cases}

    The circuit queries :math:`U` and :math:`U^{\\dagger}` alternately :math:`d` times
    (Theorem 2 of Ref. [1]) and uses a single extra qubit, the signal qubit, which
    controls the sign of the phases of the rotations that are controlled by the
    projector on the ancillas in :math:`|0\\rangle`.

    Args:
        block_encoding (:class:`qibo.models.circuit.Circuit`): circuit that
            implements :math:`U`. Its first ``nancillas`` qubits are the ancillas.
        phases (ArrayLike): phases of the polynomial, e.g. the output of
            :func:`qibo.models.qsvt.qsvt_phases`.
        nancillas (int): number :math:`a` of ancilla qubits of ``block_encoding``.

    Returns:
        :class:`qibo.models.circuit.Circuit`: circuit with the signal qubit at position
        :math:`0`, followed by the qubits of ``block_encoding`` in the same order.

    Example:
        Apply the cubic Chebyshev polynomial :math:`T_{3}(x) = 4 x^{3} - 3 x` to the
        singular value :math:`1/2` of a block-encoding of :math:`|+\\rangle / 2`.
        Since :math:`T_{3}(1/2) = -1`, the amplitudes with the signal qubit and the
        ancilla in :math:`|0\\rangle` are those of :math:`-|+\\rangle`.

        .. testcode::

            import math

            from qibo import Circuit, gates, get_backend
            from qibo.models.qsvt import qsvt_circuit, qsvt_phases

            backend = get_backend()

            # prepares |+> with amplitude 1/2 when the ancilla is in |0>
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


def qsvt_phases(coefficients: ArrayLike, backend: Backend = None) -> ArrayLike:
    """Computes the phases of a quantum singular value transformation (QSVT) circuit.

    Given the Chebyshev series :math:`P(x) = \\sum_{k = 0}^{d} c_{k} T_{k}(x)` of a real
    polynomial of degree :math:`d \\ge 1` with the parity of :math:`d` and
    :math:`|P(x)| \\le 1` for :math:`x \\in [-1, 1]`, it finds the phases for which
    :func:`qibo.models.qsvt.qsvt_circuit` applies :math:`P` to the singular values
    (Corollary 11 of Ref. [1]). The polynomial is completed into a unitary matrix whose
    entries are trigonometric polynomials, using the spectral factorization of
    :math:`1 - P^{2}` that :func:`qibo.models.qsp.qsp_phases` uses for the Fourier-based
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
        factor, converged = _spectral_factor(1 - values**2, degree, backend)
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
