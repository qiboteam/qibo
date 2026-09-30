import math
from collections.abc import Callable

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from scipy.special import comb, erf, gammaln, jv
from scipy.stats import binom

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.models.circuit import Circuit


def fourier_qsp_analytic_extension_coefficients(
    function: Callable,
    epsilon: float,
    backend: Backend = None,
):
    r"""Computes a Fourier series for a function using an analytic extension.

    Implements Sec. III D of Ref. [1]. Let :math:`f` be a function, analytic on
    :math:`[-\pi, \pi]`, with :math:`|f(\lambda)| \le 1` for
    :math:`\lambda \in [-1, 1]`, which is the range of the eigenvalues of the
    Hermitian operator :math:`H` to be transformed (the norm of :math:`H` is at
    most one). Its periodic extension is not smooth in general, which would slow
    down the convergence of the Fourier series (Gibbs phenomenon). Instead, the
    series is computed for the analytic function

    .. math::
        g(\lambda) = f(\lambda) \, \frac{\mathrm{erf}[L (\lambda + \chi)] -
        \mathrm{erf}[L (\lambda - \chi)]}{2} \, ,

    where :math:`\mathrm{erf}` is the error function, that is a smoothed
    version of :math:`f(\lambda)` times a step function of width
    :math:`2 \chi`, with :math:`1 < \chi < \pi`. The step goes to zero at the
    boundaries of the period, so that the Fourier coefficients converge
    exponentially fast. The parameters are chosen as follows:

    - :math:`\chi` is such that :math:`\max_{|\lambda| \le \chi} |f(\lambda)| =
      1 + \epsilon / 3` (Eq. 15 of Ref. [1]), bounded by :math:`(1 + \pi) / 2`.
    - :math:`L = \sqrt{\ln(3 / (2 \epsilon))} / (\chi - 1)` (Eq. 13 of Ref. [1]),
      which guarantees :math:`|f - g| < \epsilon / 3` in :math:`[-1, 1]`. If
      :math:`\chi` is close to :math:`\pi`, :math:`L` is increased so that
      the step also decays at the boundaries of the period.

    The coefficients are the discrete Fourier transform of :math:`g`, and the
    truncation order :math:`q` is the smallest even integer for which the
    truncated series approximates :math:`f` with error below :math:`\epsilon`
    in :math:`[-1, 1]`, found by binary search as in Ref. [1]. The filter is
    steep when :math:`f` exceeds one right outside of :math:`[-1, 1]` (such as
    :math:`e^{-\beta (\lambda + 1)}`), so the number of samples needed
    grows as :math:`\mathcal{O}(1 / \epsilon)`.

    Args:
        function (Callable): function :math:`f`. It has to accept an array of
            real numbers and return an array of (possibly complex) numbers.
        epsilon (float): target error, between :math:`0` and :math:`1`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        tuple: coefficients :math:`(c_{-q/2}, \dots, c_{q/2})` of the Fourier
        series :math:`\tilde{g}_{q}`, subnormalization
        :math:`\alpha = 1 / \max(1, \max_{x} |\tilde{g}_{q}(x)|)`, and evolution
        time :math:`t = 1`. Then, :math:`\alpha \, \tilde{g}_{q}(\lambda t)`
        approximates :math:`\alpha f(\lambda)` with error :math:`\epsilon`, and
        the coefficients can be given to
        :func:`qibo.models.qsp.fourier_qsp_phases` (which rescales them by
        :math:`\alpha`) with an oracle for :math:`e^{-i t H}`.

    References:
        1. T. L. Silva, L. Borges, and L. Aolita, *Fourier-based quantum signal
        processing*, `arXiv:2206.02826 <https://arxiv.org/abs/2206.02826>`_.
    """
    backend = _check_backend(backend)

    if not 0.0 < epsilon < 1.0:
        raise_error(ValueError, f"``epsilon`` must be in (0, 1), but is {epsilon}.")

    def modulus(radius):
        # maximum of |f| in [-radius, radius]
        return np.abs(function(np.linspace(-radius, radius, 2**14))).max()

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

    scale = math.sqrt(math.log(3 / (2 * epsilon)))
    steepness = max(scale / (chi - 1.0), scale / (math.pi - chi))

    # the erf step has a spectrum that decays as exp(-m^2 / (4 L^2))
    nsamples = 2 ** math.ceil(math.log2(64 * steepness))
    nsamples = max(nsamples, 2**12)
    if nsamples > 2**24:
        raise_error(
            RuntimeError,
            f"``epsilon`` = {epsilon} requires more than 2^24 samples.",
        )

    xs = 2 * math.pi * np.arange(nsamples) / nsamples
    xs = (xs + math.pi) % (2 * math.pi) - math.pi  # x in [-pi, pi)
    target = np.asarray(function(xs), dtype=complex)
    smooth = target * (erf(steepness * (xs + chi)) - erf(steepness * (xs - chi))) / 2
    spectrum = np.fft.fft(smooth) / nsamples  # coefficient of e^{imx} at index m

    inside = np.abs(xs) <= 1.0

    def truncated_series(half):
        # series with |m| <= half, evaluated on the grid with the inverse FFT
        frequencies = np.arange(-half, half + 1) % nsamples
        truncated = np.zeros(nsamples, dtype=complex)
        truncated[frequencies] = spectrum[frequencies]
        return np.fft.ifft(truncated) * nsamples

    # smallest order q = 2 * half with error below epsilon, by binary search
    low, high = 1, nsamples // 4
    error = np.abs(truncated_series(high) - target)[inside].max()
    if error >= epsilon:
        raise_error(RuntimeError, "Fourier series did not reach the target error.")
    while low < high:
        middle = (low + high) // 2
        error = np.abs(truncated_series(middle) - target)[inside].max()
        low, high = (low, middle) if error < epsilon else (middle + 1, high)
    half = low

    coefficients = spectrum[np.arange(-half, half + 1) % nsamples]
    alpha = 1.0 / max(1.0, np.abs(truncated_series(half)).max())

    return (
        backend.cast(coefficients, dtype=coefficients.dtype),
        float(alpha),
        1.0,
    )


def fourier_qsp_bounded_error_coefficients(
    power_series,
    delta: float,
    epsilon: float,
    backend: Backend = None,
):
    r"""Computes a Fourier series with bounded error from a power series.

    Implements Sec. III C of Ref. [1], which is based on Lemma 37 of Ref. [2].
    Let

    .. math::
        \tilde{f}(\lambda) = \sum_{l = 0}^{K} a_{l} \, \lambda^{l}

    be a polynomial approximation of the target function :math:`f` on
    :math:`[-1, 1]`, e.g. its truncated Taylor series, with
    :math:`|f - \tilde{f}| \le \epsilon / (4 \alpha)` (see below for
    :math:`\alpha`). Given :math:`\delta \in (0, \pi/2)`, this function computes
    the coefficients :math:`c_{m}` of the Fourier series

    .. math::
        \tilde{g}_{q}(x) = \sum_{m = -q/2}^{q/2} c_{m} \, e^{i m x} \, ,
        \quad q = 2 \left\lceil \frac{\pi}{2 \delta}
        \ln\left( \frac{4 \|d\|_{1}}{\epsilon} \right) \right\rceil \, ,

    such that :math:`|\tilde{g}_{q}(\lambda t) - \alpha \tilde{f}(\lambda)| \le
    \epsilon` for all :math:`\lambda \in [-1, 1]`, with evolution time
    :math:`t = \pi/2 - \delta`. Here, :math:`\|d\|_{1}` is the sum of the absolute
    values of :math:`d_{l} = \alpha \, a_{l} / (1 - 2 \delta / \pi)^{l}`.
    The series is built in two steps:

    1. Since :math:`x = (2/\pi) \arcsin(\sin(\pi x / 2))`, each monomial
       :math:`(2 x / \pi)^{l}` is written as a power series in
       :math:`\sin(x)`, which converges on :math:`|x| \le \pi/2 - \delta`. The
       series is truncated when its tail is below :math:`\epsilon / 4`.
    2. Each power :math:`\sin^{k}(x)` is expanded in exponentials
       :math:`e^{i m x}` with binomial coefficients, keeping :math:`|m| \le q / 2`.

    The sum of the absolute values of the coefficients does not increase in
    the process, :math:`\|c\|_{1} \le \|d\|_{1}`. Therefore, :math:`|\tilde{g}_{q}|
    \le 1` for all :math:`x` if :math:`\|d\|_{1} \le 1`, which is guaranteed by
    the subnormalization

    .. math::
        \alpha = \min\left(1, \Big[ \sum_{l = 0}^{K} |a_{l}| \,
        (1 - 2 \delta / \pi)^{-l} \Big]^{-1}\right)

    (Eq. 10 of Ref. [1]). The order :math:`q` is an upper bound that grows as
    :math:`1 / \delta`, and the number of operations grows as
    :math:`\mathcal{O}(1 / \delta^{3})`. In practice, the error is well below
    :math:`\epsilon` and the highest frequencies have negligible coefficients.

    Args:
        power_series (ArrayLike): coefficients :math:`(a_{0}, \dots, a_{K})`.
        delta (float): distance :math:`\delta`, between :math:`0` and
            :math:`\pi / 2`, between the interval of convergence and the boundary
            of the period. It sets the trade-off between the order :math:`q`
            and the subnormalization :math:`\alpha`.
        epsilon (float): target error, between :math:`0` and :math:`1`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        tuple: coefficients :math:`(c_{-q/2}, \dots, c_{q/2})`, subnormalization
        :math:`\alpha`, and evolution time :math:`t = \pi/2 - \delta`. The
        coefficients can be given to
        :func:`qibo.models.qsp.fourier_qsp_phases`, and the oracle has to
        implement :math:`e^{-i t H}` for the returned :math:`t`. The resulting
        block-encoding approximates :math:`\alpha f[H]`.

    References:
        1. T. L. Silva, L. Borges, and L. Aolita, *Fourier-based quantum signal
        processing*, `arXiv:2206.02826 <https://arxiv.org/abs/2206.02826>`_.
        2. J. van Apeldoorn, A. Gilyén, S. Gribling, and R. de Wolf, *Quantum
        SDP-Solvers: Better upper and lower bounds*, `Quantum 4, 230 (2020)
        <https://doi.org/10.22331/q-2020-02-14-230>`_,
        `arXiv:1705.01843 <https://arxiv.org/abs/1705.01843>`_.
    """
    backend = _check_backend(backend)

    if not 0.0 < delta < math.pi / 2:
        raise_error(ValueError, f"``delta`` must be in (0, pi/2), but is {delta}.")
    if not 0.0 < epsilon < 1.0:
        raise_error(ValueError, f"``epsilon`` must be in (0, 1), but is {epsilon}.")

    power_series = np.asarray(backend.to_numpy(power_series), dtype=complex)

    ratio = 1.0 - 2 * delta / math.pi  # (2 / pi) * (pi / 2 - delta)
    orders = np.arange(len(power_series))
    alpha = min(1.0, 1.0 / np.sum(np.abs(power_series) / ratio**orders))
    scaled = alpha * power_series / ratio**orders  # coefficients d_l
    norm = max(np.abs(scaled).sum(), np.finfo(float).tiny)

    # highest frequency M = q / 2 (at least one), and highest power L of sin(x)
    logarithm = math.log(4 * norm / epsilon)
    half = max(2 * math.ceil(logarithm / (2 * delta / math.pi)), 1)
    npowers = max(math.ceil(logarithm / math.log(1 / math.cos(delta))), 1)

    # power series of (2 arcsin(z) / pi)^k in z = sin(x), up to z^L
    odd = np.arange(0, (npowers - 1) // 2 + 1)
    arcsine = np.zeros(npowers + 1)
    arcsine[1::2][: len(odd)] = (
        np.exp(gammaln(2 * odd + 1) - 2 * gammaln(odd + 1) - odd * math.log(4))
        / (2 * odd + 1)
        * 2
        / math.pi
    )

    # weights of the powers of z = sin(x) in sum_l d_l (2 x / pi)^l
    weights = np.zeros(npowers + 1, dtype=complex)
    monomial = np.zeros(npowers + 1)
    monomial[0] = 1.0
    for coefficient in scaled:
        weights += coefficient * monomial
        monomial = np.convolve(monomial, arcsine)[: npowers + 1]

    # sin(x)^l = i^{-l} sum_j binom(l, j) 2^{-l} (-1)^{l - j} e^{i (2j - l) x}
    coefficients = np.zeros(2 * half + 1, dtype=complex)
    for power in np.nonzero(weights)[0]:
        lowest = max(0, math.ceil((power - half) / 2))
        highest = min(power, (power + half) // 2)
        js = np.arange(lowest, highest + 1)
        coefficients[2 * js - power + half] += (
            weights[power]
            * (1j) ** (-power)
            * (-1.0) ** (power - js)
            * binom.pmf(js, power, 0.5)
        )

    return (
        backend.cast(coefficients, dtype=coefficients.dtype),
        float(alpha),
        math.pi / 2 - delta,
    )


def fourier_qsp_circuit(evolution: Circuit, phases) -> Circuit:
    r"""Creates the Fourier-based quantum signal processing (QSP) circuit of Fig. 1
    in Ref. [1].

    Let :math:`H` be a Hermitian operator on :math:`n` qubits and let ``evolution``
    be the circuit that implements the time evolution :math:`e^{-i t H}`, for a
    fixed time :math:`t`. The circuit returned by this function calls the
    real-time evolution oracle

    .. math::
        O = \mathbb{1} \otimes |0\rangle\langle 0| + e^{-i t H} \otimes
        |1\rangle\langle 1|

    (or its inverse :math:`O^{\dagger}`) once per pulse, controlled by a single
    ancilla qubit, and interleaves it with single-qubit rotations of the ancilla,
    :math:`q` times in total. If :math:`\tilde{g}_{q}(x) = \sum_{m = -q/2}^{q/2}
    c_{m} \, e^{i m x}` is the Fourier series that ``phases`` was computed from
    (see :func:`qibo.models.qsp.fourier_qsp_phases`), then, for
    :math:`|0\rangle` being the ground state of the ancilla,

    .. math::
        \langle 0 | V_{\Phi} | 0 \rangle = \sum_{\lambda} \tilde{g}_{q}(\lambda t)
        \, |\lambda\rangle\langle\lambda| \, ,

    where :math:`V_{\Phi}` is the unitary implemented by the circuit and
    :math:`\lambda` and :math:`|\lambda\rangle` are the eigenvalues and eigenvectors
    of :math:`H`. That is, :math:`V_{\Phi}` is a block-encoding of the operator
    Fourier series :math:`\tilde{g}_{q}(H t)`, which is obtained after
    post-selecting the ancilla on :math:`|0\rangle`. Unlike the QSP circuit
    of :func:`qibo.models.qsp.qsp_circuit`, no qubitization of :math:`H` is
    required.

    Args:
        evolution (:class:`qibo.models.circuit.Circuit`): circuit that implements the
            unitary :math:`e^{-i t H}` on :math:`n` qubits. The circuit has to be
            exact, including its global phase, since it is applied under control.
        phases (ArrayLike): pulse angles :math:`\Phi`, as an array of shape
            :math:`(q + 1, 4)` with :math:`q` even. Row :math:`k` contains the
            angles :math:`(\zeta_{k}, \eta_{k}, \varphi_{k}, \kappa_{k})` of the
            :math:`k`-th pulse, e.g. the output of
            :func:`qibo.models.qsp.fourier_qsp_phases`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: circuit on :math:`n + 1` qubits.
        Qubit :math:`0` is the ancilla and the remaining qubits are those of
        ``evolution``, in the same order. It queries ``evolution`` (or its
        inverse) :math:`q` times.

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


def fourier_qsp_phases(coefficients, backend: Backend = None):
    r"""Computes the pulse angles that synthesize a Fourier series with QSP.

    Given the (complex) coefficients :math:`c_{m}` of the Fourier series

    .. math::
        \tilde{g}_{q}(x) = \sum_{m = -q/2}^{q/2} c_{m} \, e^{i m x} \, ,

    where :math:`q` is even, this function computes the angles
    :math:`\Phi = (\xi_{0}, \dots, \xi_{q})`, with
    :math:`\xi_{k} = (\zeta_{k}, \eta_{k}, \varphi_{k}, \kappa_{k})`, such that
    the product of single-qubit gates

    .. math::
        \mathcal{R}(x, \Phi) = R_{q}(x) \cdots R_{1}(x) \, R_{0}(x) \, ,
        \quad R_{k}(x) = e^{i (\zeta_{k} + \eta_{k}) Z / 2} \,
        e^{-i \varphi_{k} Y} \, e^{i (\zeta_{k} - \eta_{k}) Z / 2} \,
        e^{i \omega_{k} x Z} \, e^{-i \kappa_{k} Y} \, ,

    with :math:`\omega_{0} = 0`, :math:`\omega_{k} = 1/2` for odd :math:`k`, and
    :math:`\omega_{k} = -1/2` for even :math:`k > 0`, has
    :math:`\tilde{g}_{q}(x)` as its upper-left matrix element for all
    :math:`x \in [-\pi, \pi]`. Here, :math:`Y` and :math:`Z` are Pauli operators
    (Theorem 1 of Ref. [1]). Such an
    element can be obtained if and only if :math:`|\tilde{g}_{q}(x)| \le 1` for
    all :math:`x`. If this is not the case for the given coefficients, they are
    divided by :math:`\max_{x} |\tilde{g}_{q}(x)|`.

    The angles are computed in two steps, both taking polynomial time in
    :math:`q`:

    1. The complementary Fourier series :math:`\tilde{h}_{q}` of the same order,
       with :math:`|\tilde{g}_{q}|^{2} + |\tilde{h}_{q}|^{2} = 1`, is the
       spectral factor of the Laurent polynomial :math:`1 - |\tilde{g}_{q}|^{2}`
       whose roots lie outside of the unit disk (Lemma 4 of Ref. [1]). It is
       computed with the fast Fourier transform from the cepstrum of the
       polynomial, which gives the same factor as the root-based construction
       of Ref. [1] but stays accurate for large :math:`q`. This defines the special unitary matrix
       :math:`[[\tilde{g}_{q}, \tilde{h}_{q}], [-\tilde{h}_{q}^{*}, \tilde{g}_{q}^{*}]]`,
       where :math:`*` is complex conjugation.
    2. Gates are removed one by one from the left of this matrix, from
       :math:`R_{q}` to :math:`R_{1}`, by choosing :math:`\zeta_{k} = \eta_{k}`
       and :math:`\varphi_{k}` such that the highest and lowest frequencies
       vanish, with :math:`\kappa_{k} = \pi/4`. The remaining constant matrix
       gives :math:`\xi_{0}` (Theorem 1 of Ref. [1]).

    The angles are accurate to about :math:`10^{-13}` for the orders tested
    (up to :math:`q \sim 200`). If :math:`|\tilde{g}_{q}|` reaches one, the
    series is slightly rescaled to keep the complementary series well defined.

    Args:
        coefficients (ArrayLike): coefficients
            :math:`(c_{-q/2}, \dots, c_{q/2})` of the Fourier series, which is a
            sequence of odd length :math:`q + 1 \ge 3`, e.g. the output of
            :func:`qibo.models.qsp.fourier_qsp_bounded_error_coefficients`
            or :func:`qibo.models.qsp.fourier_qsp_analytic_extension_coefficients`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ArrayLike: array of shape :math:`(q + 1, 4)` whose :math:`k`-th row
        contains :math:`(\zeta_{k}, \eta_{k}, \varphi_{k}, \kappa_{k})`.

    References:
        1. T. L. Silva, L. Borges, and L. Aolita, *Fourier-based quantum signal
        processing*, `arXiv:2206.02826 <https://arxiv.org/abs/2206.02826>`_.
    """
    backend = _check_backend(backend)

    coefficients = np.asarray(backend.to_numpy(coefficients), dtype=complex)

    degree = len(coefficients) - 1  # q in the paper
    if degree < 2 or degree % 2 != 0:
        raise_error(
            ValueError,
            "``coefficients`` must have odd length q + 1 with q >= 2, "
            + f"but has length {len(coefficients)}.",
        )

    half = degree // 2
    frequencies = np.arange(-half, half + 1)

    # The series is evaluated on a dense grid of the unit circle, z = e^{ix}.
    nsamples = 2 ** max(12, math.ceil(math.log2(128 * (degree + 1))))
    spectrum = np.zeros(nsamples, dtype=complex)

    # Normalize the series so that its modulus is bounded by one.
    spectrum[frequencies % nsamples] = coefficients
    modulus = np.abs(np.fft.ifft(spectrum) * nsamples).max()
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
    for margin in 10.0 ** np.arange(-14, -1):
        spectrum[:] = 0.0
        spectrum[frequencies % nsamples] = coefficients * (1 - margin)
        residual = 1 - np.abs(np.fft.ifft(spectrum) * nsamples) ** 2
        if residual.min() <= 0.0:
            continue

        cepstrum = np.fft.fft(np.log(residual)) / nsamples
        causal = np.zeros(nsamples, dtype=complex)
        causal[0] = cepstrum[0] / 2
        causal[1 : nsamples // 2] = cepstrum[1 : nsamples // 2]
        causal[nsamples // 2] = cepstrum[nsamples // 2] / 2
        series = np.fft.fft(np.exp(nsamples * np.fft.ifft(causal))) / nsamples
        if np.abs(series[degree + 1 : nsamples // 2]).max() < 1e-12:
            break
    else:
        raise_error(RuntimeError, "Complementary Fourier series not found.")

    coefficients = coefficients * (1 - margin)
    complement = series[: degree + 1]  # coefficients of z^n, n = 0, ..., q

    # Coefficients of the SU(2) matrix [[g, h], [-h^*, g^*]] for each frequency, in
    # powers of w = e^{ix/2}: the coefficient of w^{-q + 2j} is ``matrix[j]``.
    matrix = np.zeros((degree + 1, 2, 2), dtype=complex)
    matrix[:, 0, 0] = coefficients
    matrix[:, 0, 1] = complement
    matrix[:, 1, 0] = -np.conj(complement[::-1])
    matrix[:, 1, 1] = np.conj(coefficients[::-1])

    kappa = math.pi / 4
    kappa_inverse = np.array(
        [[math.cos(kappa), math.sin(kappa)], [-math.sin(kappa), math.cos(kappa)]]
    )

    angles = np.zeros((degree + 1, 4))
    for step in range(degree, 0, -1):
        # The iterate of pulse ``step`` is diag(w^{sign}, w^{-sign}). Its inverse
        # raises the frequencies of one row and lowers those of the other. Degree
        # reduction requires that the raised row has no w^{step} term and that the
        # lowered row has no w^{-step} term, which fixes the rows of the inverse of
        # the pulse up to a phase that is set by ``eta = zeta``.
        sign = 1 if step % 2 == 1 else -1
        highest = matrix[step] if sign == -1 else matrix[0]
        null_vector = np.conj(np.linalg.svd(highest)[0][:, 1])  # u @ highest = 0

        zeta = (np.angle(null_vector[1]) - np.angle(null_vector[0])) / 2
        varphi = math.atan2(np.abs(null_vector[1]), np.abs(null_vector[0]))
        pulse_inverse = np.array(
            [
                [
                    math.cos(varphi) * np.exp(-1j * zeta),
                    math.sin(varphi) * np.exp(1j * zeta),
                ],
                [
                    -math.sin(varphi) * np.exp(-1j * zeta),
                    math.cos(varphi) * np.exp(1j * zeta),
                ],
            ]
        )

        rotated = pulse_inverse @ matrix
        matrix = np.stack(
            (
                rotated[:step, 0] if sign == -1 else rotated[1:, 0],
                rotated[1:, 1] if sign == -1 else rotated[:step, 1],
            ),
            axis=1,
        )
        matrix = kappa_inverse @ matrix

        angles[step] = [zeta, zeta, varphi, kappa]

    # The constant matrix that is left is the first pulse, without iterate.
    constant = matrix[0]
    angles[0] = [
        np.angle(constant[0, 0]),
        np.angle(-constant[0, 1]),
        math.atan2(np.abs(constant[0, 1]), np.abs(constant[0, 0])),
        0.0,
    ]

    return backend.cast(angles, dtype=angles.dtype)


def qsp_circuit(walk: Circuit, phases) -> Circuit:
    """Creates the quantum signal processing (QSP) circuit of Fig. 1 in Ref. [1].

    Let :math:`W` be the unitary implemented by ``walk``, with eigenstates
    :math:`W \\, |u_{\\lambda}\\rangle = e^{i \\theta_{\\lambda}} \\, |u_{\\lambda}\\rangle`.
    Each signal unitary

    .. math::
        U_{\\phi} = e^{-i \\phi Z / 2} \\, H \\,
        (|0\\rangle\\langle 0| \\otimes \\mathbb{1} + |1\\rangle\\langle 1| \\otimes W)
        \\, H \\, e^{i \\phi Z / 2}

    acts on a single ancilla qubit, placed at position :math:`0`, and reduces on
    :math:`|u_{\\lambda}\\rangle` to a single-qubit rotation
    :math:`R_{\\phi}(\\theta_{\\lambda})` up to the global phase
    :math:`e^{i \\theta_{\\lambda} / 2}`. Here, :math:`Z` is the Pauli-:math:`Z`
    operator and :math:`H` is the Hadamard gate. Alternating :math:`U_{\\phi}`
    and :math:`U_{\\phi + \\pi}^{\\dagger}` cancels this global phase, which is
    possible because the number of phases is even.

    The circuit is preceded and followed by a Hadamard gate on the ancilla. Thus,
    the probability amplitude of measuring the ancilla in :math:`|0\\rangle`
    transforms the eigenphases of :math:`W` according to the polynomial encoded
    in ``phases`` (see :func:`qibo.models.qsp.qsp_phases`).

    For Hamiltonian simulation, :math:`W` is a quantum walk built from the
    Hamiltonian, e.g. the qubitization walk of Ref. [2].

    Args:
        walk (:class:`qibo.models.circuit.Circuit`): circuit that implements the
            unitary :math:`W` on :math:`n` qubits.
        phases (ArrayLike): even-length sequence of phases
            :math:`(\\phi_{1}, \\dots, \\phi_{N})`, e.g. the output of
            :func:`qibo.models.qsp.qsp_phases`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: circuit on :math:`n + 1` qubits.
        Qubit :math:`0` is the ancilla and the remaining qubits are those of
        ``walk``, in the same order. The circuit calls ``walk`` (or its inverse)
        once per phase.

    References:
        1. G. H. Low and I. L. Chuang, *Optimal Hamiltonian Simulation by Quantum
        Signal Processing*, `Phys. Rev. Lett. 118, 010501 (2017)
        <https://doi.org/10.1103/PhysRevLett.118.010501>`_,
        `arXiv:1606.02685 <https://arxiv.org/abs/1606.02685>`_.
        2. G. H. Low and I. L. Chuang, *Hamiltonian Simulation by Qubitization*,
        `Quantum 3, 163 (2019) <https://doi.org/10.22331/q-2019-07-12-163>`_,
        `arXiv:1610.06546 <https://arxiv.org/abs/1610.06546>`_.
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


def qsp_hamiltonian_simulation_phases(
    tau: float,
    epsilon: float,
    method: str = "phase_sums",
    backend: Backend = None,
):
    """Computes the QSP phases for Hamiltonian simulation.

    Following Ref. [1], the target function
    :math:`h(\\theta) = -\\tau \\sin(\\theta)` is approximated by the truncated
    Jacobi-Anger expansion

    .. math::
        e^{-i \\tau \\sin(\\theta)} \\approx
        \\sum_{k = 0}^{N / 2} \\alpha_{k} \\cos(k \\theta)
        + i \\sum_{k = 1}^{N / 2} \\beta_{k} \\sin(k \\theta) \\, ,

    where :math:`\\alpha_{0} = J_{0}(\\tau)`, :math:`\\alpha_{k} = 2 J_{k}(\\tau)`
    for even :math:`k > 0` and :math:`\\beta_{k} = -2 J_{k}(\\tau)` for odd
    :math:`k`, with :math:`J_{k}` being the Bessel function of the first kind
    (all other coefficients vanish). The smallest even degree :math:`N` for
    which the truncation error is at most ``epsilon`` is selected.
    The phases are computed with :func:`qibo.models.qsp.qsp_phases` (Ref. [2]).
    Applied to a quantum walk :math:`W` with eigenphases
    :math:`\\sin(\\theta_{\\lambda}) = \\lambda / \\alpha`, where :math:`\\lambda`
    are the eigenvalues of a Hamiltonian and :math:`\\alpha` is a
    normalization factor, the resulting QSP circuit approximates
    :math:`e^{-i t H}` with :math:`\\tau = \\alpha t`.

    Args:
        tau (float): simulation length :math:`\\tau = t \\, \\alpha`, with
            :math:`t` being the evolution time.
        epsilon (float): target error of the Jacobi-Anger truncation.
        method (str, optional): method to compute the phases, see
            :func:`qibo.models.qsp.qsp_phases`. Use ``"layer_stripping"`` for
            degrees above about :math:`26`. Defaults to ``"phase_sums"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ArrayLike: phases :math:`(\\phi_{1}, \\dots, \\phi_{N})` to be given to
        :func:`qibo.models.qsp.qsp_circuit`.

    References:
        1. G. H. Low and I. L. Chuang, *Optimal Hamiltonian Simulation by Quantum
        Signal Processing*, `Phys. Rev. Lett. 118, 010501 (2017)
        <https://doi.org/10.1103/PhysRevLett.118.010501>`_,
        `arXiv:1606.02685 <https://arxiv.org/abs/1606.02685>`_.
        2. G. H. Low, T. J. Yoder, and I. L. Chuang, *The methodology of resonant
        equiangular composite quantum gates*, `Phys. Rev. X 6, 041067 (2016)
        <https://doi.org/10.1103/PhysRevX.6.041067>`_,
        `arXiv:1603.03996 <https://arxiv.org/abs/1603.03996>`_.
    """
    backend = _check_backend(backend)

    if epsilon <= 0.0:
        raise_error(ValueError, f"``epsilon`` must be positive, but is {epsilon}.")

    # the Bessel functions decay super-exponentially, so a finite window of
    # terms is enough to bound the tail of the series
    degree = 1
    while True:
        tail = 2 * np.abs(jv(np.arange(degree + 1, degree + 65), tau)).sum()
        if tail <= epsilon:
            break
        degree += 1

    orders = np.arange(degree + 1)
    bessel = jv(orders, tau)

    cosine = np.where(orders % 2 == 0, 2 * bessel, 0.0)
    cosine[0] = bessel[0]
    sine = np.where(orders % 2 == 1, -2 * bessel, 0.0)[1:]

    return qsp_phases(cosine, sine, method=method, backend=backend)


def qsp_phases(
    cosine_coefficients,
    sine_coefficients,
    method: str = "phase_sums",
    backend: Backend = None,
):
    r"""Computes the QSP phases that approximate a target Fourier series.

    Given the real coefficients :math:`a_{k}` and :math:`c_{k}` of

    .. math::
        A(\theta) = \sum_{k = 0}^{N / 2} a_{k} \cos(k \theta) \, , \quad
        C(\theta) = \sum_{k = 1}^{N / 2} c_{k} \sin(k \theta) \, ,

    where :math:`N` is even, this function returns phases
    :math:`(\phi_{1}, \dots, \phi_{N})` such that the product of single-qubit
    rotations

    .. math::
        V(\theta) = R_{\phi_{N}}(\theta) \cdots R_{\phi_{1}}(\theta) \, ,
        \quad R_{\phi}(\theta) = e^{-i (\theta / 2)
        (X \cos(\phi) + Y \sin(\phi))} \, ,

    satisfies :math:`\langle + | V(\theta) | + \rangle = A(\theta) + i C(\theta)`
    up to a small rescaling of the target that guarantees
    :math:`A^{2} + C^{2} \le 1`. Here, :math:`X` and :math:`Y` are Pauli
    operators and :math:`|+\rangle` is the :math:`+1` eigenstate of :math:`X`.
    When :math:`A + i C` approximates a unimodular function
    :math:`e^{i h(\theta)}` with error :math:`\epsilon`, the resulting
    unitary approximates it with error at most :math:`8 \epsilon` (Theorem 2
    of Ref. [1]).

    The phases are obtained in three steps:

    1. :math:`A` and :math:`C` are divided by :math:`1 + \epsilon`, with
       :math:`\epsilon` being the amount by which their modulus exceeds one.
       This makes :math:`1 - A^{2} - C^{2}` strictly positive.
    2. The trigonometric polynomials :math:`B` and :math:`D` with
       :math:`A^{2} + B^{2} + C^{2} + D^{2} = 1` are found through the
       Fejér-Riesz factorization of :math:`1 - A^{2} - C^{2}`, which is
       equivalent to the polynomial sum of squares of Ref. [2]. Then, :math:`A` and
       :math:`B` are rotated by the angle :math:`\delta`, with
       :math:`\cos(\delta) = A(0)`, so that :math:`A(0) = 1`.
    3. The phases are extracted from :math:`V = A \mathbb{1} + i B Z + i C X + i D Y`
       following Lemma 1 of Ref. [2] (``method="phase_sums"``). The
       :math:`N + 1` phase sums :math:`\Phi_{j}` are the coefficients of :math:`V`
       in the homogeneous basis :math:`\cos^{N - j}(\theta / 2)
       \sin^{j}(\theta / 2)`, and the phases follow from
       :math:`e^{i \phi_{N}} = \sum_{j \, \mathrm{odd}} \Phi_{j} /
       \sum_{j \, \mathrm{even}} \Phi_{j}`, after which :math:`\Phi` is reduced
       to the sums of a product of :math:`N - 1` rotations, recursively.
       Since it works in a monomial basis, this method loses precision as
       :math:`N` grows (about :math:`10^{-9}` error at :math:`N = 18` and
       :math:`10^{-7}` at :math:`N = 26`, and it fails for :math:`N \gtrsim 30`
       in double precision). ``method="layer_stripping"`` instead removes the
       rotations one at a time from the highest-degree term of the Laurent
       expansion of :math:`V` in :math:`e^{i \theta / 2}`, which stays accurate
       for larger :math:`N`.

    Args:
        cosine_coefficients (ArrayLike): coefficients :math:`(a_{0}, \dots, a_{N/2})`.
        sine_coefficients (ArrayLike): coefficients :math:`(c_{1}, \dots, c_{N/2})`.
        method (str, optional): either ``"phase_sums"`` (Lemma 1 of Ref. [2]) or
            ``"layer_stripping"``. Defaults to ``"phase_sums"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ArrayLike: phases :math:`(\phi_{1}, \dots, \phi_{N})`.

    References:
        1. G. H. Low and I. L. Chuang, *Optimal Hamiltonian Simulation by Quantum
        Signal Processing*, `Phys. Rev. Lett. 118, 010501 (2017)
        <https://doi.org/10.1103/PhysRevLett.118.010501>`_,
        `arXiv:1606.02685 <https://arxiv.org/abs/1606.02685>`_.
        2. G. H. Low, T. J. Yoder, and I. L. Chuang, *The methodology of resonant
        equiangular composite quantum gates*, `Phys. Rev. X 6, 041067 (2016)
        <https://doi.org/10.1103/PhysRevX.6.041067>`_,
        `arXiv:1603.03996 <https://arxiv.org/abs/1603.03996>`_.
    """
    backend = _check_backend(backend)

    if method not in ("phase_sums", "layer_stripping"):
        raise_error(ValueError, f"Unknown ``method`` {method}.")

    cosine = np.asarray(backend.to_numpy(cosine_coefficients), dtype=float)
    sine = np.asarray(backend.to_numpy(sine_coefficients), dtype=float)

    half = len(sine)
    if half == 0 or len(cosine) != half + 1:
        raise_error(
            ValueError,
            "``cosine_coefficients`` must have one more element than "
            + "``sine_coefficients``, which cannot be empty.",
        )

    degree = 2 * half  # N in the paper
    thetas = np.linspace(0.0, 2 * math.pi, 8 * degree + 8, endpoint=False)
    orders = np.arange(1, half + 1)

    # Step 1: rescale the target so that A^2 + C^2 < 1 everywhere. The margin
    # is increased until the Fejer-Riesz roots are safely away from the unit circle.
    modulus = np.abs(
        cosine[0]
        + np.cos(np.outer(thetas, orders)) @ cosine[1:]
        + 1j * (np.sin(np.outer(thetas, orders)) @ sine)
    ).max()
    cosine, sine = cosine / max(modulus, 1.0), sine / max(modulus, 1.0)

    for margin in 10.0 ** np.arange(-14, -3):
        # Laurent coefficients in z = e^{i theta}, for exponents -N/2, ..., N/2.
        laurent_a = np.zeros(degree + 1, dtype=complex)
        laurent_c = np.zeros(degree + 1, dtype=complex)
        laurent_a[half] = cosine[0]
        laurent_a[half + orders] = cosine[1:] / 2
        laurent_a[half - orders] = cosine[1:] / 2
        laurent_c[half + orders] = sine / 2j
        laurent_c[half - orders] = -sine / 2j
        laurent_a, laurent_c = (1 - margin) * laurent_a, (1 - margin) * laurent_c

        # Step 2: Fejer-Riesz factorization 1 - A^2 - C^2 = |Q(z)|^2 with real Q.
        laurent_f = -np.convolve(laurent_a, laurent_a) - np.convolve(
            laurent_c, laurent_c
        )
        laurent_f[degree] += 1.0
        roots = np.roots(laurent_f.real)
        distance = np.abs(np.abs(roots) - 1.0).min()
        if distance > 1e-6 and np.sum(np.abs(roots) < 1.0) == degree:
            break
    else:
        raise_error(RuntimeError, "Fejer-Riesz factorization failed.")

    roots = roots[np.abs(roots) < 1.0]
    reference = np.exp(0.5j)
    target = (np.polyval(laurent_f.real, reference) / reference**degree).real
    factor = math.sqrt(target) / np.abs(np.prod(reference - roots))
    poly_q = (factor * np.poly(roots).real)[::-1]  # ascending powers of z

    laurent_b = (poly_q + poly_q[::-1]) / 2
    laurent_d = (poly_q - poly_q[::-1]) / 2j

    # Rotate (A, B) by delta so that the new A equals one at theta = 0.
    delta = math.atan2(laurent_b.sum().real, laurent_a.sum().real)
    laurent_a, laurent_b = (
        laurent_a * math.cos(delta) + laurent_b * math.sin(delta),
        laurent_b * math.cos(delta) - laurent_a * math.sin(delta),
    )

    if method == "phase_sums":
        # Step 3: Lemma 1 of Ref. [2]. The Fourier series of A, B, C, and D are
        # written as polynomials in x = cos(theta / 2) and y = sin(theta / 2), such
        # that V = A + i B Z + i x C X + i x D Y, with A, B even in x and C, D odd
        # in y. Then, V = sum_j (-i)^j y^j x^(N-j) Phi_j, where the phase sums
        # Phi_j define the phases uniquely (Eqs. 3, 8, 9, and 10 of Ref. [2]).
        to_x = Polynomial([-1.0, 0.0, 2.0])  # cos(theta) = 2 x^2 - 1
        to_y = Polynomial([1.0, 0.0, -2.0])  # cos(theta) = 1 - 2 y^2, sin(k theta)
        # = x * 2 y * U_{k-1}(cos(theta)), with U_{k-1} = T_k' / k

        poly_a, poly_b = (
            Chebyshev(np.r_[laurent[half].real, 2 * laurent[half + orders].real])
            .convert(kind=Polynomial)(to_x)
            .coef
            for laurent in (laurent_a, laurent_b)
        )
        poly_c, poly_d = (
            (
                Polynomial([0.0, 1.0])
                * Chebyshev(np.r_[0.0, 4 * (1j * laurent[half + orders]).real / orders])
                .deriv()
                .convert(kind=Polynomial)(to_y)
            ).coef
            for laurent in (laurent_c, laurent_d)
        )
        poly_a, poly_b, poly_c, poly_d = (
            np.pad(coef, (0, degree + 1 - len(coef)))
            for coef in (poly_a, poly_b, poly_c, poly_d)
        )

        sums = np.zeros(degree + 1, dtype=complex)
        for j in range(degree + 1):
            if j % 2 == 0:
                n = np.arange(0, degree + 1, 2)
                weights = comb((degree - n) // 2, j // 2)
                sums[j] = (1j) ** j * np.sum((poly_a[n] + 1j * poly_b[n]) * weights)
            else:
                n = np.arange(1, degree + 1, 2)
                weights = comb((degree - n - 1) // 2, (j - n) // 2)
                sums[j] = (1j) ** j * np.sum((1j * poly_c[n] - poly_d[n]) * weights)

        phases = np.zeros(degree)
        for step in range(degree, 0, -1):
            odd, even = sums[1 : step + 1 : 2].sum(), sums[0 : step + 1 : 2].sum()
            phases[step - 1] = np.angle(odd / even)

            lower = np.zeros(step, dtype=complex)
            for j in range(step):
                signs = np.where(
                    (j + np.arange(j + 1)) % 2 == 1,
                    -np.exp(-1j * phases[step - 1] * (-1) ** j),
                    1.0,
                )
                lower[j] = np.sum(sums[: j + 1] * signs)
            sums = lower
    else:
        # Step 3: layer stripping. ``coefficients[j]`` multiplies w^{2j - d}, where
        # w = e^{i theta / 2} and d is the current degree. Each rotation is
        # R_phi = w P_minus + w^{-1} P_plus, where P_plus and P_minus project onto
        # the eigenspaces of cos(phi) X + sin(phi) Y, and the highest-degree
        # coefficient of V must be supported on the range of P_minus.
        paulis = np.array(
            [
                [[1, 0], [0, 1]],
                [[1, 0], [0, -1]],
                [[0, 1], [1, 0]],
                [[0, -1j], [1j, 0]],
            ]
        )
        coefficients = (
            laurent_a[:, None, None] * paulis[0]
            + 1j * laurent_b[:, None, None] * paulis[1]
            + 1j * laurent_c[:, None, None] * paulis[2]
            + 1j * laurent_d[:, None, None] * paulis[3]
        )
        phases = np.zeros(degree)
        for step in range(degree, 0, -1):
            top = coefficients[step]
            column = top[:, np.argmax(np.abs(top).sum(0))]
            phases[step - 1] = np.angle(-column[1] / column[0])

            axis = np.array(
                [
                    [0, np.exp(-1j * phases[step - 1])],
                    [np.exp(1j * phases[step - 1]), 0],
                ]
            )
            coefficients = (paulis[0] - axis) / 2 @ coefficients[1 : step + 1] + (
                paulis[0] + axis
            ) / 2 @ coefficients[:step]

    return backend.cast(phases, dtype=phases.dtype)