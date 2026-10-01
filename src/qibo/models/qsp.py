import math
import sys
from collections.abc import Callable

from numpy.polynomial import Chebyshev, Polynomial
from numpy.typing import ArrayLike
from scipy.special import comb, erf, gammaln, jv
from scipy.stats import binom

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.models.circuit import Circuit


def qsp_circuit(
    oracle: Circuit, phases: ArrayLike, method: str | None = None
) -> Circuit:
    """Creates a quantum signal processing (QSP) circuit.

    The circuit interleaves queries to ``oracle`` with rotations of an ancilla
    qubit, placed at qubit :math:`0`. The remaining qubits are those of ``oracle``.

    - ``method=None``: QSP of Ref. [1]. ``oracle`` implements a unitary
      :math:`W`, which is queried :math:`N` times. The block of the circuit unitary
      with the ancilla starting and ending in :math:`|0\\rangle` applies
      :math:`A(\\theta) + i C(\\theta)` to each eigenphase :math:`\\theta` of
      :math:`W`, where :math:`A` and :math:`C` are the trigonometric polynomials
      encoded in ``phases``.
    - ``method="fourier"``: Fourier-based QSP of Ref. [2]. ``oracle`` implements
      :math:`e^{-i t H}` for a Hermitian operator :math:`H` and a time :math:`t`,
      and is queried :math:`q` times under control. The same block of the circuit
      unitary is :math:`\\tilde{g}(H t)`, where :math:`\\tilde{g}` is the Fourier
      series encoded in ``phases``.

    Args:
        oracle (:class:`qibo.models.circuit.Circuit`): circuit on :math:`n` qubits
            that implements :math:`W` or :math:`e^{-i t H}`, depending on
            ``method``. It has to be exact, including its global phase, since it
            is applied under control.
        phases (ArrayLike): output of :func:`qibo.models.qsp.qsp_phases` for the
            same ``method``.
        method (str, optional): ``None`` for QSP of Ref. [1] or ``"fourier"`` for
            Fourier-based QSP of Ref. [2]. Defaults to ``None``.

    Returns:
        :class:`qibo.models.circuit.Circuit`: circuit on :math:`n + 1` qubits,
        with the ancilla at qubit :math:`0`.

    Example:
        Apply :math:`(1 + W) / 2`, with :math:`W = X`, to :math:`|0\\rangle`. The
        amplitudes with the ancilla in :math:`|0\\rangle` are those of
        :math:`|+\\rangle / \\sqrt{2}`.

        .. testcode::

            from qibo import Circuit, gates, get_backend
            from qibo.models.qsp import qsp_circuit, qsp_phases

            backend = get_backend()

            walk = Circuit(1)
            walk.add(gates.X(0))

            # cosine and sine series of (1 + exp(i theta)) / 2
            phases = qsp_phases([0.5, 0.5], [0.5])
            circuit = qsp_circuit(walk, phases)

            target = Circuit(1)
            target.add(gates.H(0))

            state = circuit().state()[:2]
            print(backend.allclose(state, target().state() / backend.sqrt(2), atol=1e-4))

        .. testoutput::

            True

        Invert a rotation with the Fourier series :math:`\\tilde{g}(x) = e^{i x}`.
        The amplitudes with the ancilla in :math:`|0\\rangle` are those of the
        inverse rotation applied to :math:`|0\\rangle`.

        .. testcode::

            from qibo import Circuit, gates, get_backend
            from qibo.models.qsp import qsp_circuit, qsp_phases

            backend = get_backend()

            evolution = Circuit(1)
            evolution.add(gates.RX(0, 0.6))

            # coefficients of exp(-i x), 1, and exp(i x)
            phases = qsp_phases([0.0, 0.0, 1.0], method="fourier")
            circuit = qsp_circuit(evolution, phases, method="fourier")

            target = Circuit(1)
            target.add(gates.RX(0, -0.6))

            state = circuit().state()[:2]
            print(backend.allclose(state, target().state(), atol=1e-6))

        .. testoutput::

            True

    References:
        1. G. H. Low and I. L. Chuang, *Optimal Hamiltonian Simulation by Quantum
        Signal Processing*, `Phys. Rev. Lett. 118, 010501 (2017)
        <https://doi.org/10.1103/PhysRevLett.118.010501>`_.
        2. T. L. Silva, L. Borges, and L. Aolita, *Fourier-based quantum signal
        processing*, `arXiv:2206.02826 <https://arxiv.org/abs/2206.02826>`_.
    """
    if method not in (None, "fourier"):
        raise_error(ValueError, f"Unknown ``method`` {method}. Use None or 'fourier'.")

    if method == "fourier":
        return _fourier_qsp_circuit(oracle, phases)

    return _qsp_circuit(oracle, phases)


def qsp_phases(
    coefficients: ArrayLike,
    sine_coefficients: ArrayLike | None = None,
    method: str | None = None,
    extraction: str = "phase_sums",
    backend: Backend = None,
) -> ArrayLike:
    """Computes the phases of a quantum signal processing (QSP) circuit.

    - ``method=None``: QSP of Ref. [1]. Given the series
      :math:`A(\\theta) = \\sum_{k = 0}^{N/2} a_{k} \\cos(k \\theta)` and
      :math:`C(\\theta) = \\sum_{k = 1}^{N/2} c_{k} \\sin(k \\theta)`, with even
      :math:`N`, it finds :math:`N` phases such that
      :math:`\\langle + | V(\\theta) | + \\rangle = A(\\theta) + i C(\\theta)`. Here,
      :math:`V(\\theta)` is the product of the :math:`N` single-qubit rotations of
      angle :math:`\\theta` about the axes of the :math:`xy`-plane given by the
      phases, and :math:`|+\\rangle` is the :math:`+1` eigenstate of the Pauli-
      :math:`X` operator. Since :math:`V(0)` is the identity, the target has to
      satisfy :math:`A(0) = 1`. It is rescaled if :math:`|A + i C| > 1`.
    - ``method="fourier"``: Fourier-based QSP of Ref. [2]. Given the coefficients
      :math:`c_{m}` of the Fourier series
      :math:`\\tilde{g}(x) = \\sum_{m = -q/2}^{q/2} c_{m} \\, e^{i m x}`, with even
      :math:`q`, it finds the angles of :math:`q + 1` single-qubit gates whose
      product has :math:`\\tilde{g}(x)` as its upper-left matrix element. The series
      is divided by :math:`\\max_{x} |\\tilde{g}(x)|` if it is larger than one.

    Args:
        coefficients (ArrayLike): if ``method=None``, cosine coefficients
            :math:`(a_{0}, \\dots, a_{N/2})`. If ``method="fourier"``, Fourier
            coefficients :math:`(c_{-q/2}, \\dots, c_{q/2})`.
        sine_coefficients (ArrayLike, optional): sine coefficients
            :math:`(c_{1}, \\dots, c_{N/2})`. Required if ``method=None`` and not
            used otherwise. Defaults to ``None``.
        method (str, optional): ``None`` for QSP of Ref. [1] or ``"fourier"`` for
            Fourier-based QSP of Ref. [2]. Defaults to ``None``.
        extraction (str, optional): how the phases are extracted if
            ``method=None``. ``"phase_sums"`` follows Ref. [3] and loses precision
            for :math:`N \\gtrsim 30`, while ``"layer_stripping"`` removes one
            rotation at a time and stays accurate for larger :math:`N`.
            Defaults to ``"phase_sums"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ArrayLike: phases for :func:`qibo.models.qsp.qsp_circuit` with the same
        ``method``. An array of length :math:`N` if ``method=None``, or an array
        of shape :math:`(q + 1, 4)` if ``method="fourier"``.

    References:
        1. G. H. Low and I. L. Chuang, *Optimal Hamiltonian Simulation by Quantum
        Signal Processing*, `Phys. Rev. Lett. 118, 010501 (2017)
        <https://doi.org/10.1103/PhysRevLett.118.010501>`_.
        2. T. L. Silva, L. Borges, and L. Aolita, *Fourier-based quantum signal
        processing*, `arXiv:2206.02826 <https://arxiv.org/abs/2206.02826>`_.
        3. G. H. Low, T. J. Yoder, and I. L. Chuang, *The methodology of resonant
        equiangular composite quantum gates*, `Phys. Rev. X 6, 041067 (2016)
        <https://doi.org/10.1103/PhysRevX.6.041067>`_.
    """
    backend = _check_backend(backend)

    if method not in (None, "fourier"):
        raise_error(ValueError, f"Unknown ``method`` {method}. Use None or 'fourier'.")

    if method == "fourier":
        if sine_coefficients is not None:
            raise_error(
                ValueError, "``sine_coefficients`` is not used if ``method='fourier'``."
            )

        return _fourier_qsp_phases(coefficients, backend=backend)

    if sine_coefficients is None:
        raise_error(ValueError, "``sine_coefficients`` is required if ``method=None``.")

    return _qsp_phases(
        coefficients, sine_coefficients, extraction=extraction, backend=backend
    )


def _fourier_qsp_analytic_extension_coefficients(
    function: Callable[[ArrayLike], ArrayLike],
    epsilon: float,
    backend: Backend = None,
) -> tuple[ArrayLike, float, float]:
    """Computes Fourier coefficients using an analytic extension.

    Follows Sec. III D of Ref. [1]. The coefficients are those of ``function``
    multiplied by a smooth step of width :math:`2 \\chi`, with
    :math:`1 < \\chi < \\pi`, and the order :math:`q` is the smallest even one with
    error below ``epsilon`` in :math:`[-1, 1]`.

    Args:
        function (Callable): function :math:`f` with :math:`|f| \\le 1` in
            :math:`[-1, 1]`. It takes and returns backend arrays.
        epsilon (float): target error, between :math:`0` and :math:`1`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        tuple: coefficients :math:`(c_{-q/2}, \\dots, c_{q/2})`, subnormalization
        :math:`\\alpha`, and evolution time :math:`t = 1`, such that the Fourier
        series at :math:`\\lambda t` approximates :math:`\\alpha f(\\lambda)` in
        :math:`[-1, 1]`.

    References:
        1. T. L. Silva, L. Borges, and L. Aolita, *Fourier-based quantum signal
        processing*, `arXiv:2206.02826 <https://arxiv.org/abs/2206.02826>`_.
    """
    backend = _check_backend(backend)

    if not 0.0 < epsilon < 1.0:
        raise_error(ValueError, f"``epsilon`` must be in (0, 1), but is {epsilon}.")

    def modulus(radius: float) -> float:
        # maximum of |f| in [-radius, radius]
        nodes = radius * (2 * backend.arange(2**14) / (2**14 - 1) - 1)

        return backend.max(backend.abs(function(nodes)))

    limit = 1.0 + epsilon / 3
    upper = (1.0 + math.pi) / 2
    if modulus(1.0) > limit:
        raise_error(ValueError, "``function`` must be bounded by one in [-1, 1].")

    # largest chi with max |f| in [-chi, chi] below 1 + epsilon / 3, by bisection
    if modulus(upper) <= limit:
        chi = upper
    else:
        low, high = 1.0, upper
        for _ in range(60):
            middle = (low + high) / 2
            low, high = (middle, high) if modulus(middle) <= limit else (low, middle)
        chi = low

    if chi - 1.0 < 1e-9:
        raise_error(ValueError, "``function`` grows too fast right outside [-1, 1].")

    scale = float(backend.sqrt(backend.log(3 / (2 * epsilon))))
    steepness = max(scale / (chi - 1.0), scale / (math.pi - chi))

    # the erf step has a spectrum that decays as exp(-m^2 / (4 L^2))
    nsamples = max(2 ** int(backend.ceil(backend.log2(64 * steepness))), 2**12)
    if nsamples > 2**24:
        raise_error(
            RuntimeError,
            f"``epsilon`` = {epsilon} requires more than 2^24 samples.",
        )

    xs = 2 * math.pi * backend.arange(nsamples) / nsamples
    xs = backend.mod(xs + math.pi, 2 * math.pi) - math.pi  # x in [-pi, pi)
    target = backend.cast(function(xs), dtype="complex128")
    step = (
        erf(backend.to_numpy(steepness * (xs + chi)))
        - erf(backend.to_numpy(steepness * (xs - chi)))
    ) / 2
    smooth = target * backend.cast(step, dtype="float64")
    # coefficient of e^{imx} at index m
    spectrum = backend.fft(smooth) / nsamples

    inside = backend.abs(xs) <= 1.0

    def truncated_series(half: int) -> ArrayLike:
        # series with |m| <= half, evaluated on the grid with the inverse FFT
        frequencies = backend.mod(backend.arange(-half, half + 1), nsamples)
        truncated = backend.zeros(nsamples, dtype="complex128")
        truncated[frequencies] = spectrum[frequencies]

        return backend.ifft(truncated) * nsamples

    # smallest order q = 2 * half with error below epsilon, by binary search
    low, high = 1, nsamples // 4
    error = backend.max(backend.abs(truncated_series(high) - target)[inside])
    if error >= epsilon:
        raise_error(RuntimeError, "Fourier series did not reach the target error.")
    while low < high:
        middle = (low + high) // 2
        error = backend.max(backend.abs(truncated_series(middle) - target)[inside])
        low, high = (low, middle) if error < epsilon else (middle + 1, high)
    half = low

    coefficients = spectrum[backend.mod(backend.arange(-half, half + 1), nsamples)]
    alpha = 1.0 / max(1.0, float(backend.max(backend.abs(truncated_series(half)))))

    return coefficients, float(alpha), 1.0


def _fourier_qsp_bounded_error_coefficients(
    power_series: ArrayLike,
    delta: float,
    epsilon: float,
    backend: Backend = None,
) -> tuple[ArrayLike, float, float]:
    """Computes Fourier coefficients with bounded error from a power series.

    Follows Sec. III C of Ref. [1], based on Lemma 37 of Ref. [2]. The power
    series in :math:`\\lambda` is converted into a Fourier series in
    :math:`e^{i x}`, with :math:`x = \\lambda t`, through the series of
    :math:`\\arcsin`. The order :math:`q` is an upper bound, and the highest
    frequencies may have negligible coefficients.

    Args:
        power_series (ArrayLike): coefficients :math:`(a_{0}, \\dots, a_{K})` of
            an approximation of the target function in :math:`[-1, 1]`.
        delta (float): distance :math:`\\delta`, between :math:`0` and
            :math:`\\pi/2`, to the boundary of the period, with
            :math:`t = \\pi/2 - \\delta`.
        epsilon (float): target error, between :math:`0` and :math:`1`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        tuple: coefficients :math:`(c_{-q/2}, \\dots, c_{q/2})`, subnormalization
        :math:`\\alpha`, and evolution time :math:`t`, such that the Fourier
        series at :math:`\\lambda t` approximates :math:`\\alpha f(\\lambda)` in
        :math:`[-1, 1]`.

    References:
        1. T. L. Silva, L. Borges, and L. Aolita, *Fourier-based quantum signal
        processing*, `arXiv:2206.02826 <https://arxiv.org/abs/2206.02826>`_.
        2. J. van Apeldoorn, A. Gilyén, S. Gribling, and R. de Wolf, *Quantum
        SDP-Solvers: Better upper and lower bounds*, `Quantum 4, 230 (2020)
        <https://doi.org/10.22331/q-2020-02-14-230>`_.
    """
    backend = _check_backend(backend)

    if not 0.0 < delta < math.pi / 2:
        raise_error(ValueError, f"``delta`` must be in (0, pi/2), but is {delta}.")
    if not 0.0 < epsilon < 1.0:
        raise_error(ValueError, f"``epsilon`` must be in (0, 1), but is {epsilon}.")

    power_series = backend.cast(power_series, dtype="complex128")

    ratio = 1.0 - 2 * delta / math.pi  # (2 / pi) * (pi / 2 - delta)
    orders = backend.arange(len(power_series))
    alpha = min(
        1.0, float(1.0 / backend.sum(backend.abs(power_series) / ratio**orders))
    )
    scaled = alpha * power_series / ratio**orders  # coefficients d_l
    norm = max(float(backend.sum(backend.abs(scaled))), sys.float_info.min)

    # highest frequency M = q / 2 (at least one), and highest power L of sin(x)
    logarithm = float(backend.log(4 * norm / epsilon))
    half = max(2 * int(backend.ceil(logarithm / (2 * delta / math.pi))), 1)
    npowers = max(int(backend.ceil(logarithm / backend.log(1 / backend.cos(delta)))), 1)

    # power series of (2 arcsin(z) / pi)^k in z = sin(x), up to z^L
    odd = backend.arange(0, (npowers - 1) // 2 + 1)
    log_gamma = gammaln(backend.to_numpy(2 * odd + 1)) - 2 * gammaln(
        backend.to_numpy(odd + 1)
    )
    arcsine = backend.zeros(npowers + 1, dtype="float64")
    arcsine[2 * odd + 1] = (
        backend.exp(backend.cast(log_gamma, dtype="float64") - odd * backend.log(4.0))
        / (2 * odd + 1)
        * 2
        / math.pi
    )

    # weights of the powers of z = sin(x) in sum_l d_l (2 x / pi)^l
    weights = backend.zeros(npowers + 1, dtype="complex128")
    monomial = backend.zeros(npowers + 1, dtype="float64")
    monomial[0] = 1.0
    for coefficient in scaled:
        weights = weights + coefficient * monomial
        monomial = backend.convolve(monomial, arcsine)[: npowers + 1]

    # sin(x)^l = i^{-l} sum_j binom(l, j) 2^{-l} (-1)^{l - j} e^{i (2j - l) x}
    coefficients = backend.zeros(2 * half + 1, dtype="complex128")
    for power in backend.nonzero(weights)[0]:
        power = int(power)
        lowest = max(0, -((half - power) // 2))
        highest = min(power, (power + half) // 2)
        js = backend.arange(lowest, highest + 1)
        probabilities = backend.cast(
            binom.pmf(backend.to_numpy(js), power, 0.5), dtype="float64"
        )
        coefficients[2 * js - power + half] += (
            weights[power]
            * (1j) ** (-power)
            * (1 - 2 * backend.mod(power - js, 2))
            * probabilities
        )

    return coefficients, float(alpha), math.pi / 2 - delta


def _fourier_qsp_circuit(evolution: Circuit, phases: ArrayLike) -> Circuit:
    """Creates the Fourier-based QSP circuit of Ref. [1].

    Args:
        evolution (:class:`qibo.models.circuit.Circuit`): circuit that implements
            :math:`e^{-i t H}` on :math:`n` qubits.
        phases (ArrayLike): array of shape :math:`(q + 1, 4)`, with even
            :math:`q`, of pulse angles.

    Returns:
        :class:`qibo.models.circuit.Circuit`: circuit on :math:`n + 1` qubits,
        with the ancilla at qubit :math:`0`.

    References:
        1. T. L. Silva, L. Borges, and L. Aolita, *Fourier-based quantum signal
        processing*, `arXiv:2206.02826 <https://arxiv.org/abs/2206.02826>`_.
    """
    phases = [[float(angle) for angle in pulse] for pulse in phases]

    if (
        any(len(pulse) != 4 for pulse in phases)
        or len(phases) < 3
        or len(phases) % 2 == 0
    ):
        raise_error(
            ValueError,
            "``phases`` must have shape (q + 1, 4) with even q >= 2, "
            + f"but has {len(phases)} rows.",
        )

    qubits = list(range(1, evolution.nqubits + 1))
    inverse = evolution.invert()

    circuit = Circuit(evolution.nqubits + 1)
    for index, (zeta, eta, varphi, kappa) in enumerate(phases):
        # Pulse 0 is the input unitary W_in without oracle. Odd pulses query O
        # and even pulses query O^†, so that their global phases cancel.
        circuit.add(gates.RY(0, 2 * kappa))
        if index > 0:
            oracle = evolution if index % 2 == 1 else inverse
            circuit.add(gate.controlled_by(0) for gate in oracle.on_qubits(*qubits))
        circuit.add(gates.RZ(0, -(zeta - eta)))
        circuit.add(gates.RY(0, 2 * varphi))
        circuit.add(gates.RZ(0, -(zeta + eta)))

    return circuit


def _fourier_qsp_phases(coefficients: ArrayLike, backend: Backend = None) -> ArrayLike:
    """Computes the pulse angles of the Fourier-based QSP circuit of Ref. [1].

    The complementary Fourier series is computed as the spectral factor of
    :math:`1 - |\\tilde{g}|^{2}` with the cepstrum, and the pulses are then
    removed one at a time from the special unitary matrix that has
    :math:`\\tilde{g}` as its upper-left element.

    Args:
        coefficients (ArrayLike): Fourier coefficients
            :math:`(c_{-q/2}, \\dots, c_{q/2})`, with even :math:`q \\ge 2`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ArrayLike: array of shape :math:`(q + 1, 4)` whose :math:`k`-th row contains
        the angles :math:`(\\zeta_{k}, \\eta_{k}, \\varphi_{k}, \\kappa_{k})`.

    References:
        1. T. L. Silva, L. Borges, and L. Aolita, *Fourier-based quantum signal
        processing*, `arXiv:2206.02826 <https://arxiv.org/abs/2206.02826>`_.
    """
    backend = _check_backend(backend)

    coefficients = backend.cast(coefficients, dtype="complex128")

    degree = len(coefficients) - 1  # q in the paper
    if degree < 2 or degree % 2 != 0:
        raise_error(
            ValueError,
            "``coefficients`` must have odd length q + 1 with q >= 2, "
            + f"but has length {len(coefficients)}.",
        )

    half = degree // 2
    frequencies = backend.arange(-half, half + 1)

    # The series is evaluated on a dense grid of the unit circle, z = e^{ix}.
    nsamples = 2 ** max(12, int(backend.ceil(backend.log2(128 * (degree + 1)))))
    indices = backend.mod(frequencies, nsamples)
    spectrum = backend.zeros(nsamples, dtype="complex128")

    # Normalize the series so that its modulus is bounded by one.
    spectrum[indices] = coefficients
    series = backend.ifft(spectrum) * nsamples
    modulus = float(backend.max(backend.abs(series)))
    coefficients = coefficients / max(modulus, 1.0)

    # Complementary series: the Laurent polynomial 1 - |g|^2 = |h|^2 on the unit
    # circle, where h is the factor with all its roots outside of the unit disk
    # (the one selected in Lemma 4 of Ref. [1]). Instead of computing the roots of
    # the polynomial, which is ill conditioned for large q, h is obtained from the
    # cepstrum (Fourier series of the logarithm) of 1 - |g|^2: since
    # log(h) = (1/2) c_0 + sum_{n > 0} c_n z^n, with c_n the cepstrum, h is
    # analytic in the unit disk and has no roots inside. A margin, which is
    # increased until the cepstrum decays fast, keeps 1 - |g|^2 away from zero
    # when |g| touches one.
    for margin in 10.0 ** backend.arange(-14, -1):
        spectrum = backend.zeros(nsamples, dtype="complex128")
        spectrum[indices] = coefficients * (1 - margin)
        residual = 1 - backend.abs(backend.ifft(spectrum) * nsamples) ** 2
        complement, converged = _spectral_factor(residual, degree, backend)
        if converged:
            break
    else:  # pragma: no cover
        raise_error(RuntimeError, "Complementary Fourier series not found.")

    coefficients = coefficients * (1 - margin)

    # Coefficients of the SU(2) matrix [[g, h], [-h^*, g^*]] for each frequency, in
    # powers of w = e^{ix/2}: the coefficient of w^{-q + 2j} is ``matrix[j]``.
    matrix = backend.zeros((degree + 1, 2, 2), dtype="complex128")
    matrix[:, 0, 0] = coefficients
    matrix[:, 0, 1] = complement
    matrix[:, 1, 0] = -backend.conj(backend.flip(complement))
    matrix[:, 1, 1] = backend.conj(backend.flip(coefficients))

    kappa = math.pi / 4
    kappa_inverse = backend.cast(
        [
            [backend.cos(kappa), backend.sin(kappa)],
            [-backend.sin(kappa), backend.cos(kappa)],
        ],
        dtype="complex128",
    )

    angles = backend.zeros((degree + 1, 4), dtype="float64")
    for step in range(degree, 0, -1):
        # The iterate of pulse ``step`` is diag(w^{sign}, w^{-sign}). Its inverse
        # raises the frequencies of one row and lowers those of the other. Degree
        # reduction requires that the raised row has no w^{step} term and that the
        # lowered row has no w^{-step} term, which fixes the rows of the inverse of
        # the pulse up to a phase that is set by ``eta = zeta``.
        sign = 1 if step % 2 == 1 else -1
        highest = matrix[step] if sign == -1 else matrix[0]
        # row vector that satisfies ``null_vector @ highest = 0``
        null_vector = backend.conj(
            backend.singular_value_decomposition(highest)[0][:, 1]
        )

        zeta = (backend.angle(null_vector[1]) - backend.angle(null_vector[0])) / 2
        varphi = backend.arctan2(
            backend.abs(null_vector[1]), backend.abs(null_vector[0])
        )
        pulse_inverse = backend.cast(
            [
                [
                    backend.cos(varphi) * backend.exp(-1j * zeta),
                    backend.sin(varphi) * backend.exp(1j * zeta),
                ],
                [
                    -backend.sin(varphi) * backend.exp(-1j * zeta),
                    backend.cos(varphi) * backend.exp(1j * zeta),
                ],
            ],
            dtype="complex128",
        )

        rotated = backend.matmul(pulse_inverse, matrix)
        lowered = backend.zeros((step, 2, 2), dtype="complex128")
        lowered[:, 0] = rotated[:step, 0] if sign == -1 else rotated[1:, 0]
        lowered[:, 1] = rotated[1:, 1] if sign == -1 else rotated[:step, 1]
        matrix = backend.matmul(kappa_inverse, lowered)

        angles[step] = backend.cast([zeta, zeta, varphi, kappa], dtype="float64")

    # The constant matrix that is left is the first pulse, without iterate.
    constant = matrix[0]
    angles[0] = backend.cast(
        [
            backend.angle(constant[0, 0]),
            backend.angle(-constant[0, 1]),
            backend.arctan2(backend.abs(constant[0, 1]), backend.abs(constant[0, 0])),
            0.0,
        ],
        dtype="float64",
    )

    return angles


def _qsp_circuit(walk: Circuit, phases: ArrayLike) -> Circuit:
    """Creates the QSP circuit of Ref. [1].

    Args:
        walk (:class:`qibo.models.circuit.Circuit`): circuit that implements the
            unitary :math:`W` on :math:`n` qubits.
        phases (ArrayLike): even number :math:`N` of phases.

    Returns:
        :class:`qibo.models.circuit.Circuit`: circuit on :math:`n + 1` qubits,
        with the ancilla at qubit :math:`0`.

    References:
        1. G. H. Low and I. L. Chuang, *Optimal Hamiltonian Simulation by Quantum
        Signal Processing*, `Phys. Rev. Lett. 118, 010501 (2017)
        <https://doi.org/10.1103/PhysRevLett.118.010501>`_.
    """
    phases = [float(phase) for phase in phases]

    if len(phases) == 0 or len(phases) % 2 != 0:
        raise_error(
            ValueError,
            f"``phases`` must have a positive even length, but has {len(phases)}.",
        )

    nqubits = walk.nqubits
    qubits = list(range(1, nqubits + 1))

    walk_inverse = walk.invert()

    circuit = Circuit(nqubits + 1)
    circuit.add(gates.H(0))
    for index, phase in enumerate(phases):
        # Even (zero-based) positions use U_{phi}, odd positions use U_{phi + pi}^†.
        # Both act as R_{phi} on the eigenstates up to opposite global phases.
        inverse = index % 2 == 0
        angle = phase + math.pi if inverse else phase
        controlled_walk = walk_inverse if inverse else walk

        circuit.add(gates.RZ(0, -angle))
        circuit.add(gates.H(0))
        circuit.add(
            gate.controlled_by(0) for gate in controlled_walk.on_qubits(*qubits)
        )
        circuit.add(gates.H(0))
        circuit.add(gates.RZ(0, angle))
    circuit.add(gates.H(0))

    return circuit


def _qsp_hamiltonian_simulation_phases(
    tau: float,
    epsilon: float,
    extraction: str = "phase_sums",
    backend: Backend = None,
) -> ArrayLike:
    """Computes the QSP phases that approximate :math:`e^{-i \\tau \\sin(\\theta)}`.

    The target is approximated by the Jacobi-Anger expansion, truncated at the
    smallest even degree with error below ``epsilon`` (Ref. [1]).

    Args:
        tau (float): simulation length :math:`\\tau`.
        epsilon (float): target error.
        extraction (str, optional): how the phases are extracted, see
            :func:`qibo.models.qsp.qsp_phases`. Defaults to ``"phase_sums"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ArrayLike: phases for :func:`qibo.models.qsp.qsp_circuit`.

    References:
        1. G. H. Low and I. L. Chuang, *Optimal Hamiltonian Simulation by Quantum
        Signal Processing*, `Phys. Rev. Lett. 118, 010501 (2017)
        <https://doi.org/10.1103/PhysRevLett.118.010501>`_.
    """
    backend = _check_backend(backend)

    if epsilon <= 0.0:
        raise_error(ValueError, f"``epsilon`` must be positive, but is {epsilon}.")

    # the Bessel functions decay super-exponentially, so a finite window of
    # terms is enough to bound the tail of the series
    degree = 1
    while True:
        window = backend.to_numpy(backend.arange(degree + 1, degree + 65))
        tail = 2 * backend.sum(
            backend.abs(backend.cast(jv(window, tau), dtype="float64"))
        )
        if tail <= epsilon:
            break
        degree += 1

    orders = backend.arange(degree + 1)
    bessel = backend.cast(jv(backend.to_numpy(orders), tau), dtype="float64")

    cosine = backend.where(backend.mod(orders, 2) == 0, 2 * bessel, 0.0)
    cosine[0] = bessel[0]
    sine = backend.where(backend.mod(orders, 2) == 1, -2 * bessel, 0.0)[1:]

    return _qsp_phases(cosine, sine, extraction=extraction, backend=backend)


def _qsp_phases(
    cosine_coefficients: ArrayLike,
    sine_coefficients: ArrayLike,
    extraction: str = "phase_sums",
    backend: Backend = None,
) -> ArrayLike:
    """Computes the phases of the QSP circuit of Ref. [1].

    The complementary polynomials are computed with the Fejér-Riesz
    factorization, and the phases are extracted as in
    :func:`qibo.models.qsp.qsp_phases`.

    Args:
        cosine_coefficients (ArrayLike): cosine coefficients
            :math:`(a_{0}, \\dots, a_{N/2})`.
        sine_coefficients (ArrayLike): sine coefficients
            :math:`(c_{1}, \\dots, c_{N/2})`.
        extraction (str, optional): ``"phase_sums"`` or ``"layer_stripping"``.
            Defaults to ``"phase_sums"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ArrayLike: array of :math:`N` phases.

    References:
        1. G. H. Low and I. L. Chuang, *Optimal Hamiltonian Simulation by Quantum
        Signal Processing*, `Phys. Rev. Lett. 118, 010501 (2017)
        <https://doi.org/10.1103/PhysRevLett.118.010501>`_.
    """
    backend = _check_backend(backend)

    if extraction not in ("phase_sums", "layer_stripping"):
        raise_error(ValueError, f"Unknown ``extraction`` {extraction}.")

    cosine = backend.cast(cosine_coefficients, dtype="float64")
    sine = backend.cast(sine_coefficients, dtype="float64")

    half = len(sine)
    if half == 0 or len(cosine) != half + 1:
        raise_error(
            ValueError,
            "``cosine_coefficients`` must have one more element than "
            + "``sine_coefficients``, which cannot be empty.",
        )

    degree = 2 * half  # N in the paper
    nsamples = 8 * degree + 8
    thetas = 2 * math.pi * backend.arange(nsamples) / nsamples
    orders = backend.arange(1, half + 1)

    # Step 1: rescale the target so that A^2 + C^2 < 1 everywhere. The margin
    # is increased until the Fejer-Riesz roots are safely away from the unit circle.
    angles = backend.outer(thetas, orders)
    modulus = backend.max(
        backend.abs(
            cosine[0]
            + backend.matmul(backend.cos(angles), cosine[1:])
            + 1j * backend.matmul(backend.sin(angles), sine)
        )
    )
    cosine, sine = cosine / max(modulus, 1.0), sine / max(modulus, 1.0)

    for margin in 10.0 ** backend.arange(-14, -3):
        # Laurent coefficients in z = e^{i theta}, for exponents -N/2, ..., N/2.
        laurent_a = backend.zeros(degree + 1, dtype="complex128")
        laurent_c = backend.zeros(degree + 1, dtype="complex128")
        laurent_a[half] = cosine[0]
        laurent_a[half + orders] = cosine[1:] / 2
        laurent_a[half - orders] = cosine[1:] / 2
        laurent_c[half + orders] = sine / 2j
        laurent_c[half - orders] = -sine / 2j
        laurent_a, laurent_c = (1 - margin) * laurent_a, (1 - margin) * laurent_c

        # Step 2: Fejer-Riesz factorization 1 - A^2 - C^2 = |Q(z)|^2 with real Q.
        laurent_f = -backend.convolve(laurent_a, laurent_a) - backend.convolve(
            laurent_c, laurent_c
        )
        laurent_f[degree] += 1.0
        roots = backend.cast(backend.roots(backend.real(laurent_f)), dtype="complex128")
        distance = backend.min(backend.abs(backend.abs(roots) - 1.0))
        if distance > 1e-6 and backend.sum(backend.abs(roots) < 1.0) == degree:
            break
    else:  # pragma: no cover
        raise_error(RuntimeError, "Fejer-Riesz factorization failed.")

    roots = roots[backend.abs(roots) < 1.0]
    reference = backend.exp(0.5j)
    exponents = backend.arange(2 * degree, -1, -1)
    target = backend.real(
        backend.sum(backend.real(laurent_f) * reference**exponents) / reference**degree
    )
    factor = backend.sqrt(target) / backend.abs(backend.prod(reference - roots))
    poly_q = backend.flip(
        factor * backend.real(backend.poly(roots))
    )  # ascending powers

    laurent_b = (poly_q + backend.flip(poly_q)) / 2
    laurent_d = (poly_q - backend.flip(poly_q)) / 2j

    # Rotate (A, B) by delta so that the new A equals one at theta = 0.
    delta = backend.arctan2(
        backend.real(backend.sum(laurent_b)), backend.real(backend.sum(laurent_a))
    )
    laurent_a, laurent_b = (
        laurent_a * backend.cos(delta) + laurent_b * backend.sin(delta),
        laurent_b * backend.cos(delta) - laurent_a * backend.sin(delta),
    )

    if extraction == "phase_sums":
        # Step 3: Lemma 1 of Ref. [2]. The Fourier series of A, B, C, and D are
        # written as polynomials in x = cos(theta / 2) and y = sin(theta / 2), such
        # that V = A + i B Z + i x C X + i x D Y, with A, B even in x and C, D odd
        # in y. Then, V = sum_j (-i)^j y^j x^(N-j) Phi_j, where the phase sums
        # Phi_j define the phases uniquely (Eqs. 3, 8, 9, and 10 of Ref. [2]).
        to_x = Polynomial([-1.0, 0.0, 2.0])  # cos(theta) = 2 x^2 - 1
        to_y = Polynomial([1.0, 0.0, -2.0])  # cos(theta) = 1 - 2 y^2, sin(k theta)
        # = x * 2 y * U_{k-1}(cos(theta)), with U_{k-1} = T_k' / k

        poly_a, poly_b = (
            backend.cast(
                Chebyshev(
                    backend.to_numpy(
                        backend.concatenate(
                            (
                                backend.real(laurent[half : half + 1]),
                                2 * backend.real(laurent[half + orders]),
                            )
                        )
                    )
                )
                .convert(kind=Polynomial)(to_x)
                .coef,
                dtype="float64",
            )
            for laurent in (laurent_a, laurent_b)
        )
        poly_c, poly_d = (
            backend.cast(
                (
                    Polynomial([0.0, 1.0])
                    * Chebyshev(
                        backend.to_numpy(
                            backend.concatenate(
                                (
                                    backend.zeros(1, dtype="float64"),
                                    4
                                    * backend.real(1j * laurent[half + orders])
                                    / orders,
                                )
                            )
                        )
                    )
                    .deriv()
                    .convert(kind=Polynomial)(to_y)
                ).coef,
                dtype="float64",
            )
            for laurent in (laurent_c, laurent_d)
        )
        poly_a, poly_b, poly_c, poly_d = (
            backend.concatenate(
                (coef, backend.zeros(degree + 1 - len(coef), dtype="float64"))
            )
            for coef in (poly_a, poly_b, poly_c, poly_d)
        )

        sums = backend.zeros(degree + 1, dtype="complex128")
        for j in range(degree + 1):
            if j % 2 == 0:
                n = backend.arange(0, degree + 1, 2)
                weights = backend.cast(
                    comb(backend.to_numpy((degree - n) // 2), j // 2), dtype="float64"
                )
                sums[j] = (1j) ** j * backend.sum(
                    (poly_a[n] + 1j * poly_b[n]) * weights
                )
            else:
                n = backend.arange(1, degree + 1, 2)
                weights = backend.cast(
                    comb(
                        backend.to_numpy((degree - n - 1) // 2),
                        backend.to_numpy((j - n) // 2),
                    ),
                    dtype="float64",
                )
                sums[j] = (1j) ** j * backend.sum(
                    (1j * poly_c[n] - poly_d[n]) * weights
                )

        phases = backend.zeros(degree, dtype="float64")
        for step in range(degree, 0, -1):
            odd = backend.sum(sums[1 : step + 1 : 2])
            even = backend.sum(sums[0 : step + 1 : 2])
            phases[step - 1] = backend.angle(odd / even)

            lower = backend.zeros(step, dtype="complex128")
            for j in range(step):
                signs = backend.where(
                    (j + backend.arange(j + 1)) % 2 == 1,
                    -backend.exp(-1j * phases[step - 1] * (-1) ** j),
                    1.0,
                )
                lower[j] = backend.sum(sums[: j + 1] * signs)
            sums = lower
    else:
        # Step 3: layer stripping. ``coefficients[j]`` multiplies w^{2j - d}, where
        # w = e^{i theta / 2} and d is the current degree. Each rotation is
        # R_phi = w P_minus + w^{-1} P_plus, where P_plus and P_minus project onto
        # the eigenspaces of cos(phi) X + sin(phi) Y, and the highest-degree
        # coefficient of V must be supported on the range of P_minus.
        paulis = backend.cast(
            [
                [[1, 0], [0, 1]],
                [[1, 0], [0, -1]],
                [[0, 1], [1, 0]],
                [[0, -1j], [1j, 0]],
            ],
            dtype="complex128",
        )
        coefficients = (
            laurent_a[:, None, None] * paulis[0]
            + 1j * laurent_b[:, None, None] * paulis[1]
            + 1j * laurent_c[:, None, None] * paulis[2]
            + 1j * laurent_d[:, None, None] * paulis[3]
        )
        phases = backend.zeros(degree, dtype="float64")
        for step in range(degree, 0, -1):
            top = coefficients[step]
            norms = backend.sum(backend.abs(top), axis=0)
            column = top[:, int(backend.argsort(norms)[-1])]
            phases[step - 1] = backend.angle(-column[1] / column[0])

            axis = backend.cast(
                [
                    [0, backend.exp(-1j * phases[step - 1])],
                    [backend.exp(1j * phases[step - 1]), 0],
                ],
                dtype="complex128",
            )
            coefficients = backend.matmul(
                (paulis[0] - axis) / 2, coefficients[1 : step + 1]
            ) + backend.matmul((paulis[0] + axis) / 2, coefficients[:step])

    return phases


def _spectral_factor(
    residual: ArrayLike, degree: int, backend: Backend = None
) -> tuple[ArrayLike, bool]:
    """Computes the spectral factor of a positive trigonometric polynomial.

    The cepstrum, i.e. the Fourier series of the logarithm, of the polynomial
    :math:`F(x) = |h(e^{i x})|^{2}` gives the factor :math:`h(z) = \\sum_{n = 0}^{q}
    h_{n} z^{n}` that is analytic and has no roots in the unit disk, which is stable
    for large degrees. A positive polynomial with real coefficients has a real factor.

    Args:
        residual (ArrayLike): values of :math:`F` on :math:`N` equispaced points of
            :math:`[0, 2 \\pi)`, with :math:`N` a power of two.
        degree (int): degree :math:`q` of the factor.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        tuple: coefficients :math:`(h_{0}, \\dots, h_{q})` and whether the factor
        has converged, i.e. its coefficients beyond :math:`q` are negligible.
    """
    backend = _check_backend(backend)

    nsamples = len(residual)
    cepstrum = backend.fft(backend.log(residual)) / nsamples
    causal = backend.zeros(nsamples, dtype="complex128")
    causal[0] = cepstrum[0] / 2
    causal[1 : nsamples // 2] = cepstrum[1 : nsamples // 2]
    causal[nsamples // 2] = cepstrum[nsamples // 2] / 2
    series = backend.fft(backend.exp(backend.ifft(causal) * nsamples)) / nsamples
    converged = backend.max(backend.abs(series[degree + 1 : nsamples // 2])) < 1e-12

    return series[: degree + 1], bool(converged)
