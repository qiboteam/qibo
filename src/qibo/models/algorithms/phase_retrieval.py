import math
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike

from qibo.backends import Backend, _check_backend, _check_backend_and_local_state
from qibo.config import raise_error


def hybrid_input_output(
    magnitudes: ArrayLike,
    phases: ArrayLike | None = None,
    beta: float = 0.9,
    niterations: int = 5000,
    initial: ArrayLike | None = None,
    seed: int | np.random.Generator | None = None,
    backend: Backend | None = None,
) -> tuple[ArrayLike, ArrayLike]:
    """Retrieves the phases of a two-dimensional time series with Fienup's hybrid input-output.

    Quantum phase estimation without controlled unitaries only has access to the absolute values
    of the time series :math:`f(t) = \\langle \\psi | e^{i H t} | \\psi \\rangle`, with the circuit
    :math:`U_{\\psi}^{\\dagger} \\, e^{i H t} \\, U_{\\psi}`, where :math:`U_{\\psi}` prepares
    :math:`|\\psi\\rangle` from the all-zero state, and the probability of measuring the
    all-zero state is :math:`|f(t)|^{2}`. The problem is embedded in two dimensions [1] with a
    Hamiltonian :math:`H_{D}` that commutes with :math:`H`, and the signal

    .. math::
        f(t, z) = \\langle \\psi | e^{i H t} \\, e^{i H_{D} z} | \\psi \\rangle \\, .

    It is sampled at :math:`t_{j} = j \\, \\Delta t` and :math:`z_{l} = l \\, \\Delta z`, where
    :math:`j` and :math:`l` are integers that take the values :math:`-(N - 1) / 2` to
    :math:`(N - 1) / 2` and :math:`-(M - 1) / 2` to :math:`(M - 1) / 2`, and the spectrum is its
    discrete Fourier transform :math:`F[k, m] = \\sum_{j, l} f[j, l] \\,
    e^{-2 \\pi i (k j / N + m l / M)}`. A two-dimensional signal is almost always determined by
    its absolute values up to a global phase, a translation and a reflection with complex
    conjugation. The spectrum is real because :math:`f(-t, -z) = f(t, z)^{*}`, and it is
    positive if the signal is multiplied by a window whose Fourier transform is positive and
    :math:`[H, H_{D}] = 0`. This function multiplies the absolute values by the triangular window
    :math:`(1 - 2 |j| / N)(1 - 2 |l| / M)` of Ref. [1], and then it iterates the algorithm
    with the constraints that the spectrum is real and positive:

    1. :math:`f_{i} = \\mathrm{DFT}^{-1}(F_{i})`;
    2. :math:`\\tilde{f}_{i} = |f| \\, e^{i \\arg(f_{i})}`, which has the measured absolute
       values;
    3. :math:`\\tilde{F}_{i} = \\mathrm{Re}(\\mathrm{DFT}(\\tilde{f}_{i}))`;
    4. :math:`F_{i + 1}[k, m] = \\tilde{F}_{i}[k, m]` if :math:`\\tilde{F}_{i}[k, m] \\geq 0`, and
       :math:`F_{i + 1}[k, m] = F_{i}[k, m] - \\beta \\, \\tilde{F}_{i}[k, m]` otherwise.

    The spectrum :math:`F[k, 0]` of the time series of :math:`H` alone is the discrete
    Fourier transform of :math:`f[j, 0]`. The result depends on the initial spectrum, so the
    algorithm can be restarted with other seeds if it stagnates.

    The samples of the circuits are only needed for :math:`j \\geq 0`, because the absolute
    values are the same for :math:`(-j, -l)` and :math:`(j, l)`.

    Example:
        .. testcode::

            import numpy as np
            from qibo.models import hybrid_input_output

            # signal with three peaks, whose frequencies are the energies of H and of H_D
            nsamples = 31
            weights, energies, charges = [0.5, 0.3, 0.2], [0.5, 1.0, 1.6], [0.2, 0.8, 1.4]
            times = np.fft.fftfreq(nsamples, 1 / nsamples)
            signal = sum(
                weight
                * np.exp(1j * energy * times[:, None])
                * np.exp(1j * charge * times[None, :])
                for weight, energy, charge in zip(weights, energies, charges)
            )

            # absolute values, and phases of the signal at the time zero, which is only
            # evolved with H_D
            spectrum, estimate = hybrid_input_output(
                np.abs(signal), phases=np.angle(signal[0]), niterations=2000, seed=0
            )
            print(spectrum.shape, np.allclose(np.angle(estimate[0]), np.angle(signal[0])))

        .. testoutput::

            (31, 31) True

    Args:
        magnitudes (ArrayLike): absolute values :math:`|f(t_{j}, z_{l})|` of the signal, as a
            matrix of shape :math:`(N, M)` with the order of the discrete Fourier transform,
            *i.e.* the index :math:`j` is the one of the time :math:`t_{j}` for
            :math:`j \\leq N / 2`, and of the time :math:`t_{j - N}` otherwise, and the same
            for :math:`l`.
        phases (ArrayLike, optional): phases of :math:`f(0, z_{l})`, for :math:`l = 0, \\dots,
        M - 1`, which only involve the time evolution of :math:`H_{D}`, and can be computed
        classically if :math:`|\\psi\\rangle` is easy to simulate. If ``None``, they are not
        used. Defaults to ``None``.
        beta (float, optional): parameter :math:`\\beta \\in [0, 1]` of the relaxed positivity
            constraint. Defaults to :math:`0.9`.
        niterations (int, optional): number of iterations. Defaults to :math:`5000`.
        initial (ArrayLike, optional): real initial spectrum :math:`F_{1}` of shape
            :math:`(N, M)`. If ``None``, its entries are drawn uniformly at random, with the
            mean :math:`f(0, 0)`. Defaults to ``None``.
        seed (int or :class:`numpy.random.Generator`, optional): seed or generator used to draw
            the initial spectrum. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the
            calculation. If ``None``, it uses the current backend. Defaults to ``None``.

    Returns:
        tuple: the real spectrum :math:`\\tilde{F}[k, m]` and the signal :math:`\\tilde{f}[j, l]`
        of the last iteration, which includes the window and are recovered up to the trivial
        ambiguities.

    References:
        1. L. Clinton, T. S. Cubitt, R. Garcia-Patron, A. Montanaro, S. Stanisic, and
        M. Stroeks, *Quantum phase estimation without controlled unitaries*,
        `PRX Quantum 7, 010345 (2026) <https://doi.org/10.1103/7qcr-znl2>`_.

        2. J. R. Fienup, *Phase retrieval algorithms: a comparison*,
        `Appl. Opt. 21, 2758 (1982) <https://doi.org/10.1364/AO.21.002758>`_.
    """
    backend, local_state = _check_backend_and_local_state(seed, backend)

    magnitudes = backend.cast(magnitudes, dtype=backend.float64)
    if len(magnitudes.shape) != 2:
        raise_error(
            ValueError,
            f"magnitudes must be a matrix, but its shape is {tuple(magnitudes.shape)}.",
        )
    nrows, ncols = magnitudes.shape

    if phases is not None:
        phases = backend.cast(phases, dtype=backend.float64)
        if tuple(phases.shape) != (ncols,):
            raise_error(
                ValueError,
                f"phases must have shape ({ncols},), but its shape is {tuple(phases.shape)}.",
            )

    if isinstance(beta, bool) or not isinstance(beta, Real) or not 0 <= beta <= 1:
        raise_error(
            ValueError, f"beta must be in the interval [0, 1], but it is {beta}."
        )

    if (
        isinstance(niterations, bool)
        or not isinstance(niterations, Integral)
        or niterations < 1
    ):
        raise_error(
            ValueError,
            f"niterations must be a positive integer, but it is {niterations}.",
        )

    if initial is None:
        spectrum = (
            2
            * magnitudes[0, 0]
            * backend.random_uniform(size=(nrows, ncols), seed=local_state)
        )
    else:
        spectrum = backend.cast(initial, dtype=backend.float64)
        if tuple(spectrum.shape) != (nrows, ncols):
            raise_error(
                ValueError,
                f"initial must have shape {(nrows, ncols)}, "
                + f"but its shape is {tuple(spectrum.shape)}.",
            )

    # triangular window, in terms of the signed index of the discrete times
    rows = backend.cast(backend.arange(nrows), dtype=backend.float64)
    cols = backend.cast(backend.arange(ncols), dtype=backend.float64)
    rows = backend.where(rows <= nrows - rows, rows, nrows - rows)
    cols = backend.where(cols <= ncols - cols, cols, ncols - cols)
    magnitudes = magnitudes * backend.outer(1 - 2 * rows / nrows, 1 - 2 * cols / ncols)

    for _ in range(niterations):
        signal = backend.ifft(backend.ifft(spectrum, axis=0), axis=1)
        modulus = backend.abs(signal)
        phase = backend.where(
            modulus > 0, signal / backend.where(modulus > 0, modulus, 1.0), 1.0
        )
        if phases is not None:
            phase[0] = backend.exp(1j * phases)
        estimate = magnitudes * phase
        transformed = backend.real(backend.fft(backend.fft(estimate, axis=0), axis=1))
        spectrum = backend.where(
            transformed >= 0, transformed, spectrum - beta * transformed
        )

    return transformed, estimate


def vectorial_phase_retrieval(
    target: ArrayLike,
    references: ArrayLike,
    support: int,
    backend: Backend | None = None,
) -> tuple[ArrayLike, ArrayLike]:
    """Retrieves the phases of a time series with vectorial phase retrieval.

    Quantum phase estimation without controlled unitaries only has access to the absolute values
    of the time series :math:`f_{1}[j] = \\langle \\phi | e^{i j \\Delta t H} | \\phi \\rangle`,
    with the circuit :math:`U_{\\phi}^{\\dagger} \\, e^{i j \\Delta t H} \\, U_{\\phi}`, where
    :math:`U_{\\phi}` prepares :math:`|\\phi\\rangle` from the all-zero state, and the
    probability of measuring the all-zero state is :math:`|f_{1}[j]|^{2}`. The phases are
    recovered with the vectorial phase retrieval of Ref. [2], generalized to :math:`R` reference
    states :math:`|\\psi_{r}\\rangle` that are orthogonal to :math:`|\\phi\\rangle` [1]. For each
    of them, the absolute values of the signals

    .. math::
        \\begin{aligned}
        f_{2}^{(r)}[j] &= \\langle \\phi | e^{i j \\Delta t H} | \\psi_{r} \\rangle \\, , \\\\
        f_{3}^{(r)}[j] &= f_{1}[j] + f_{2}^{(r)}[j] \\, , \\\\
        f_{4}^{(r)}[j] &= f_{1}[j] + i f_{2}^{(r)}[j]
        \\end{aligned}

    are measured, which need the states :math:`|\\psi_{r}\\rangle`,
    :math:`(|\\phi\\rangle + |\\psi_{r}\\rangle) / \\sqrt{2}` and
    :math:`(|\\phi\\rangle + i |\\psi_{r}\\rangle) / \\sqrt{2}`. The phases of
    :math:`f_{1}` and :math:`f_{2}^{(r)}` are related by
    :math:`e^{i \\theta_{1}[j]} = e^{i \\theta_{2}^{(r)}[j]} G_{r}[j]`, with

    .. math::
        G_{r}[j] = \\frac{|f_{3}^{(r)}[j]|^{2} + i |f_{4}^{(r)}[j]|^{2}
        - (1 + i) (|f_{1}[j]|^{2} + |f_{2}^{(r)}[j]|^{2})}{2 |f_{1}[j]| \\, |f_{2}^{(r)}[j]|}

    on the unit circle. The estimate :math:`y` of the phases minimizes the cost

    .. math::
        \\sum_{k = s}^{N - 1} \\sum_{r = 0}^{R} \\left| \\sum_{j = 0}^{N - 1} |f_{r}[j]| \\,
        y_{r, j} \\, e^{-2 \\pi i j k / N} \\right|^{2}
        + \\sum_{r = 1}^{R} \\sum_{j = 0}^{N - 1} \\left| y_{0, j} - y_{r, j} \\,
        \\frac{G_{r}[j]}{|G_{r}[j]|} \\right|^{2} \\, ,

    where :math:`f_{0} = f_{1}` and :math:`f_{r} = f_{2}^{(r)}`. The first term is the
    support constraint, which requires the discrete Fourier transforms of the signals to vanish
    for the frequencies :math:`k \\geq s`, *i.e.* the energies of :math:`H` have to be in the
    interval :math:`[0, \\pi / \\Delta t)`, and the second one is the interference constraint.
    The cost is relaxed from phases to vectors of norm :math:`\\sqrt{(R + 1) N}`, so
    :math:`y` is the eigenvector of the smallest eigenvalue of the matrix of its quadratic form.
    Then, :math:`f_{1}[j] \\approx |f_{1}[j]| \\, y_{0, j}`. In the absence of noise, the solution
    is unique up to a global phase if the signals have no common roots of their
    :math:`z`-transforms, and :math:`s` is the exact size of their support. With noise,
    a drop of the smallest eigenvalue as a function of :math:`s`, with a large gap to the
    second one, signals a good estimate, and a larger :math:`R` improves it [1]. The
    eigenvalue problem has size :math:`(R + 1) N`, and so it costs :math:`O(R^{3} N^{3})`.

    If :math:`|f_{1}[j]| \\, |f_{2}^{(r)}[j]| = 0`, which is always the case for :math:`j = 0`
    as the states are orthogonal, the phase :math:`G_{r}[j] / |G_{r}[j]|` is not defined, and it
    is set to :math:`1`, which does not affect the signal because :math:`f_{2}^{(r)}[j] = 0`.
    The global phase of the result is chosen such that :math:`f_{1}[0]`, which is one, is
    real.

    Example:
        .. testcode::

            import numpy as np
            from qibo.models import vectorial_phase_retrieval

            # two signals whose spectra only have the first 5 out of 16 frequencies
            nsamples, support = 16, 5
            rng = np.random.default_rng(0)
            spectra = np.zeros((2, nsamples), dtype=complex)
            spectra[:, :support] = rng.normal(size=(2, support)) + 1j * rng.normal(
                size=(2, support)
            )
            f1, f2 = np.fft.ifft(spectra)
            f1, f2 = f1 / (f1[0] / abs(f1[0])), f2 / (f1[0] / abs(f1[0]))

            # probabilities that the circuits measure, which are the squared absolute values of
            # f1, f2 and f1 + f2 and f1 + i f2 divided by 2, as their states are normalized
            target = abs(f1) ** 2
            references = np.array(
                [[abs(f2) ** 2, abs(f1 + f2) ** 2 / 2, abs(f1 + 1j * f2) ** 2 / 2]]
            )

            estimate, _ = vectorial_phase_retrieval(target, references, support)
            print(np.allclose(estimate, f1))

        .. testoutput::

            True

    Args:
        target (ArrayLike): probabilities :math:`|f_{1}[j]|^{2}` of measuring the all-zero state
            after the evolution of :math:`|\\phi\\rangle`, at :math:`j = 0, \\dots, N - 1`.
        references (ArrayLike): array of shape :math:`(R, 3, N)`. For each state
            :math:`|\\psi_{r}\\rangle`, the probabilities of measuring :math:`|\\phi\\rangle`
            after the evolution of :math:`|\\psi_{r}\\rangle`, of
            :math:`(|\\phi\\rangle + |\\psi_{r}\\rangle) / \\sqrt{2}` and of
            :math:`(|\\phi\\rangle + i |\\psi_{r}\\rangle) / \\sqrt{2}`, in this order.
        support (int): number :math:`s` of the first frequencies that are allowed to be nonzero,
            between :math:`0` and :math:`N - 1`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the
            calculation. If ``None``, it uses the current backend. Defaults to ``None``.

    Returns:
        tuple: the estimate of :math:`f_{1}[j]`, and the two smallest eigenvalues of the matrix
        of the cost.

    References:
        1. L. Clinton, T. S. Cubitt, R. Garcia-Patron, A. Montanaro, S. Stanisic, and
        M. Stroeks, *Quantum phase estimation without controlled unitaries*,
        `PRX Quantum 7, 010345 (2026) <https://doi.org/10.1103/7qcr-znl2>`_.

        2. O. Raz, N. Dudovich, and B. Nadler, *Vectorial phase retrieval of 1-D signals*,
        `IEEE Trans. Signal Process. 61, 1632 (2013)
        <https://doi.org/10.1109/TSP.2013.2243441>`_.
    """
    backend = _check_backend(backend)

    target = backend.cast(target, dtype=backend.float64)
    references = backend.cast(references, dtype=backend.float64)
    if len(target.shape) != 1:
        raise_error(
            ValueError,
            f"target must be a vector, but its shape is {tuple(target.shape)}.",
        )
    nsamples = target.shape[0]
    if (
        len(references.shape) != 3
        or references.shape[0] < 1
        or tuple(references.shape[1:]) != (3, nsamples)
    ):
        raise_error(
            ValueError,
            f"references must have shape (R, 3, {nsamples}) with R >= 1, "
            + f"but its shape is {tuple(references.shape)}.",
        )
    nreferences = references.shape[0]

    if (
        isinstance(support, bool)
        or not isinstance(support, Integral)
        or not 0 <= support < nsamples
    ):
        raise_error(
            ValueError,
            f"support must be an integer between 0 and {nsamples - 1}, "
            + f"but it is {support}.",
        )

    amplitudes = backend.sqrt(backend.vstack((target[None], references[:, 0])))

    # relative phases of the target and the reference signals, where the sums have the
    # squared absolute values of f1 + f2 and f1 + i f2, as the states are normalized
    squares = amplitudes[0] ** 2 + amplitudes[1:] ** 2
    numerator = 2 * references[:, 1] + 1j * 2 * references[:, 2] - (1 + 1j) * squares
    denominator = 2 * amplitudes[0] * amplitudes[1:]
    ratio = numerator / backend.where(denominator > 0, denominator, 1.0)
    modulus = backend.abs(ratio)
    relative = backend.where(
        modulus > 0, ratio / backend.where(modulus > 0, modulus, 1.0), 1.0
    )

    # discrete Fourier transform of the frequencies where the support constraint applies
    times = backend.cast(backend.arange(nsamples), dtype=backend.float64)
    frequencies = times[support:]
    kernel = backend.exp(-2j * math.pi * backend.outer(frequencies, times) / nsamples)

    # quadratic form of the cost, which has blocks for the target and for each reference
    dimension = (nreferences + 1) * nsamples
    identity = backend.identity(nsamples, dtype=backend.complex128)
    matrix = backend.zeros((dimension, dimension), dtype=backend.complex128)
    for index in range(nreferences + 1):
        block = slice(index * nsamples, (index + 1) * nsamples)
        weighted = kernel * amplitudes[index]
        matrix[block, block] = backend.matmul(
            backend.conj(backend.transpose(weighted)), weighted
        )
        if index == 0:
            matrix[block, block] += nreferences * identity
            continue

        matrix[block, block] += identity
        coupling = -backend.diag(relative[index - 1])
        matrix[:nsamples, block] = coupling
        matrix[block, :nsamples] = backend.conj(backend.transpose(coupling))

    eigenvalues, eigenvectors = backend.eigh(matrix)

    phases = math.sqrt(dimension) * eigenvectors[:nsamples, 0]
    anchor = backend.abs(phases[0])
    phases = phases * backend.where(
        anchor > 0,
        backend.conj(phases[0]) / backend.where(anchor > 0, anchor, 1.0),
        1.0,
    )

    return amplitudes[0] * phases, eigenvalues[:2]
