import math
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike

from qibo import gates
from qibo.backends import Backend, _check_backend, _check_backend_and_local_state
from qibo.config import raise_error
from qibo.gates.abstract import Gate
from qibo.measurements import MeasurementResult
from qibo.models.algorithms.qft import QFT
from qibo.models.circuit import Circuit


def IQPE(
    unitary: Circuit | Gate | ArrayLike,
    nbits: int,
    **kwargs,
) -> tuple[Circuit, list[MeasurementResult]]:
    """Creates a circuit that implements the Iterative Quantum Phase Estimation.

    Given a unitary :math:`U` and one of its eigenstates :math:`|\\psi\\rangle`, such that
    :math:`U|\\psi\\rangle = e^{2 \\pi i \\varphi}|\\psi\\rangle`, the circuit estimates
    :math:`\\varphi \\in [0, \\, 1)` with :math:`t` bits of precision, where :math:`t` is
    ``nbits``, as :func:`qibo.models.QPE` does. However, a single counting qubit is
    measured :math:`t` times, instead of using :math:`t` counting qubits and the inverse
    :class:`qibo.models.QFT`. The circuit acts on :math:`1 + n` qubits, with :math:`n` being
    the number of qubits of :math:`U`. The first qubit is the counting qubit and the last
    :math:`n` qubits are the target qubits.

    The bits of :math:`\\varphi` are obtained one at a time, starting from the least significant
    one. In each step :math:`k = 1, \\dots, t`:

    1. a Hadamard gate is applied to the counting qubit;
    2. :math:`U^{2^{t - k}}` is applied to the target qubits, controlled by the counting qubit;
    3. a phase gate is applied to the counting qubit, with the angle
       :math:`-2 \\pi \\sum_{j=1}^{k-1} m_{j} \\, 2^{-(k - j + 1)}`, where :math:`m_{j}` is the
       outcome of the step :math:`j`, which removes the contribution of the bits that were
       already measured;
    4. a Hadamard gate is applied to the counting qubit, which is measured with
       ``collapse=True``, and the outcome :math:`m_{k}` is the bit :math:`b_{t - k + 1}` of
       :math:`\\varphi = 0.b_{1} \\cdots b_{t}`;
    5. if it is not the last step, the counting qubit is reset to :math:`|0\\rangle` with an
       :math:`X` rotation conditioned on :math:`m_{k}`.

    The circuit finishes with a measurement of the counting qubit that is not collapsed, so that
    it can be simulated with state vectors. The target qubits have to be prepared in
    :math:`|\\psi\\rangle` before the circuit is applied, *e.g.* by adding the circuit that
    prepares it to a larger circuit before the one returned by this function.

    The outcomes of the measurements are not merged into the result of the execution of the
    circuit. They are returned as a list of :class:`qibo.measurements.MeasurementResult`, which
    is ordered from the most to the least significant bit, *i.e.* in the reverse order of the
    measurements. Hence, they can be joined with
    :meth:`qibo.measurements.MeasurementResult.join` and the estimate of :math:`\\varphi` is
    obtained with :func:`qibo.quantum_info.estimate_phase`.

    .. note::
        As in :func:`qibo.models.QPE`, :math:`U` is applied :math:`2^{t} - 1` times in
        total. The circuit has fewer qubits, but when it is simulated, each shot has to be
        simulated separately because of the measurements with collapse.

    Example:
        .. testcode::

            import math

            from qibo import Circuit, gates
            from qibo.measurements import MeasurementResult
            from qibo.models import IQPE
            from qibo.quantum_info import estimate_phase

            # unitary with eigenstate |1> and eigenvalue exp(2 pi i phase)
            phase = 5 / 16
            unitary = Circuit(1)
            unitary.add(gates.U1(0, theta=2 * math.pi * phase))

            # phase estimation with 4 bits
            iqpe, results = IQPE(unitary, 4)

            # prepare the eigenstate in the target qubit and apply the phase estimation
            circuit = Circuit(iqpe.nqubits)
            circuit.add(gates.X(1))
            circuit.add(iqpe)
            circuit(nshots=100)

            print(estimate_phase(MeasurementResult.join(*results)))

        .. testoutput::

            0.3125

    Args:
        unitary (:class:`qibo.models.circuit.Circuit` or :class:`qibo.gates.Gate` or ndarray):
            unitary :math:`U`, given as a circuit, as a gate, *e.g.* :class:`qibo.gates.Unitary`,
            or as a matrix. See :func:`qibo.models.QPE` for the details.
        nbits (int): number of bits :math:`t` of the phase to be estimated, which sets the
            resolution :math:`2^{-t}` and the number of measurements.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        tuple: the circuit that implements the Iterative Quantum Phase Estimation and the
        list of :class:`qibo.measurements.MeasurementResult` with the outcomes of the
        measurements of the counting qubit, from the most to the least significant bit.

    References:
        1. A. Yu. Kitaev, *Quantum measurements and the Abelian Stabilizer Problem*,
        `arXiv:quant-ph/9511026 <https://arxiv.org/abs/quant-ph/9511026>`_.

        2. R. B. Griffiths and C.-S. Niu, *Semiclassical Fourier Transform for Quantum
        Computation*, `Phys. Rev. Lett. 76, 3228 (1996)
        <https://arxiv.org/abs/quant-ph/9511007>`_.
    """
    unitary, nbits = _check_inputs(unitary, nbits, "nbits")

    target_qubits = list(range(1, unitary.nqubits + 1))

    # the control is placed right after the qubits of the unitary,
    # and it is mapped to the counting qubit when the circuit is added
    controlled_unitary = unitary.controlled_by(unitary.nqubits)

    circuit = Circuit(unitary.nqubits + 1, **kwargs)
    results = []
    for step in range(1, nbits + 1):
        circuit.add(gates.H(0))
        for _ in range(2 ** (nbits - step)):
            circuit.add(controlled_unitary.on_qubits(*target_qubits, 0))

        if step > 1:
            # removes the contribution of the bits that were already measured
            correction = sum(
                results[previous].symbols[0] / 2 ** (step - previous)
                for previous in range(step - 1)
            )
            circuit.add(gates.U1(0, theta=-2 * math.pi * correction))

        circuit.add(gates.H(0))
        results.append(circuit.add(gates.M(0, collapse=True)))

        if step < nbits:
            # resets the counting qubit to the state 0 if the outcome was 1
            circuit.add(gates.RX(0, theta=math.pi * results[-1].symbols[0]))

    circuit.add(gates.M(0))

    return circuit, results[::-1]


def QPE(
    unitary: Circuit | Gate | ArrayLike,
    nqubits_counting: int,
    measure: bool = True,
    **kwargs,
) -> Circuit:
    """Creates a circuit that implements the canonical Quantum Phase Estimation.

    Given a unitary :math:`U` and one of its eigenstates :math:`|\\psi\\rangle`, such that
    :math:`U|\\psi\\rangle = e^{2 \\pi i \\varphi}|\\psi\\rangle`, the circuit estimates
    :math:`\\varphi \\in [0, \\, 1)` with :math:`t` bits of precision, where :math:`t`
    is the number of counting qubits. It acts on :math:`t + n` qubits, with :math:`n`
    being the number of qubits of :math:`U`, and it is composed of the following steps:

    1. a Hadamard gate on each of the :math:`t` counting qubits, which are the first qubits of
       the circuit;
    2. :math:`U^{2^{t - 1 - j}}`, controlled by the counting qubit :math:`j`, acting on the
       last :math:`n` qubits of the circuit, the target qubits;
    3. the inverse :class:`qibo.models.QFT` on the counting qubits;
    4. the measurement of the counting qubits, if ``measure=True``.

    The target qubits have to be prepared in :math:`|\\psi\\rangle` before the circuit is applied,
    *e.g.* by adding the circuit that prepares it to a larger circuit before the one returned by
    this function. At the end, the counting qubits hold a bitstring :math:`b_{1} \\cdots b_{t}`,
    with :math:`b_{1}` being the outcome of the first qubit, such that :math:`\\varphi` is
    approximated by :math:`0.b_{1} \\cdots b_{t}` in binary, which is exact if :math:`\\varphi`
    is an integer multiple of :math:`2^{-t}`. The conversion from the bitstring into the phase
    can be done with :func:`qibo.quantum_info.bitstring_to_phase`, and the phase of the most
    probable outcome, *i.e.* the estimate of :math:`\\varphi`, with
    :func:`qibo.quantum_info.estimate_phase`.

    .. note::
        The power :math:`U^{2^{k}}` is implemented by applying the controlled :math:`U` a number
        :math:`2^{k}` of times. Hence, :math:`U` is applied :math:`2^{t} - 1` times in total.

    Args:
        unitary (:class:`qibo.models.circuit.Circuit` or :class:`qibo.gates.Gate` or ndarray):
            unitary :math:`U`, given as a circuit, as a gate, *e.g.* :class:`qibo.gates.Unitary`,
            or as a matrix. A matrix must be unitary and of dimension :math:`2^{n} \\times 2^{n}`
            with :math:`n > 0`, and it is converted into a :class:`qibo.gates.Unitary` acting
            on :math:`n` qubits. The qubits of a gate are relabeled, in increasing order, as the
            qubits of a circuit with the same number of qubits as the gate. All of the gates
            must be supported by :meth:`qibo.models.circuit.Circuit.controlled_by`.
        nqubits_counting (int): number of counting qubits :math:`t`. The phase is estimated with
            a resolution of :math:`2^{-t}`.
        measure (bool, optional): If ``True``, adds a measurement of the counting qubits at
            the end of the circuit. Defaults to ``True``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`. For details, see the documentation
            of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: implementation of the Quantum Phase Estimation.

    Example:
        .. testcode::

            import math

            from qibo import Circuit, gates
            from qibo.models import QPE
            from qibo.quantum_info import estimate_phase

            # unitary with eigenstate |1> and eigenvalue exp(2 pi i phase)
            phase = 5 / 16
            unitary = Circuit(1)
            unitary.add(gates.U1(0, theta=2 * math.pi * phase))

            # phase estimation with 4 counting qubits
            nqubits_counting = 4
            qpe = QPE(unitary, nqubits_counting)

            # prepare the eigenstate in the target qubit and apply the phase estimation
            circuit = Circuit(qpe.nqubits)
            circuit.add(gates.X(nqubits_counting))
            circuit.add(qpe)

            result = circuit(nshots=100)
            print(estimate_phase(result))

        .. testoutput::

            0.3125

    References:
        1. R. Cleve, A. Ekert, C. Macchiavello, and M. Mosca, *Quantum algorithms revisited*,
        `Proc. R. Soc. Lond. A 454, 339 (1998) <https://arxiv.org/abs/quant-ph/9708016>`_.

        2. M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*,
        Cambridge University Press (2010), Sec. 5.2.
    """
    unitary, nqubits_counting = _check_inputs(
        unitary, nqubits_counting, "nqubits_counting"
    )

    counting_qubits = list(range(nqubits_counting))
    target_qubits = list(range(nqubits_counting, nqubits_counting + unitary.nqubits))

    # the control is placed right after the qubits of the unitary,
    # and it is mapped to each counting qubit when the circuit is added
    controlled_unitary = unitary.controlled_by(unitary.nqubits)

    circuit = Circuit(nqubits_counting + unitary.nqubits, **kwargs)
    circuit.add(gates.H(qubit) for qubit in counting_qubits)
    for qubit in counting_qubits:
        for _ in range(2 ** (nqubits_counting - 1 - qubit)):
            circuit.add(controlled_unitary.on_qubits(*target_qubits, qubit))
    circuit.add(QFT(nqubits_counting).invert().on_qubits(*counting_qubits))

    if measure:
        circuit.add(gates.M(*counting_qubits))

    return circuit


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
    of the time series :math:`f(t) = \\bra{\\psi} e^{i t H} \\ket{\\psi}`, with the circuit
    :math:`U_{\\psi}^{\\dagger} \\, e^{i t H} \\, U_{\\psi}`, where :math:`U_{\\psi}` prepares
    :math:`\\ket{\\psi}` from the all-zero state, and the probability of measuring the
    all-zero state is :math:`|f(t)|^{2}`. The problem is embedded in two dimensions [1] with a
    Hamiltonian :math:`H_{D}` that commutes with :math:`H`, and the signal

    .. math::
        f(t, z) = \\bra{\\psi} e^{i t H} \\, e^{i z H_{D}} \\ket{\\psi} \\, .

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
       values, after replacing the phases of :math:`f_{i}(0, z_{l})` by ``phases``, if given;
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

            from qibo import get_backend
            from qibo.models import hybrid_input_output

            backend = get_backend()

            # signal with three peaks, whose frequencies are the energies of H and of H_D
            nsamples = 31
            weights, energies, charges = [0.5, 0.3, 0.2], [0.5, 1.0, 1.6], [0.2, 0.8, 1.4]
            indices = backend.cast(backend.arange(nsamples), dtype=backend.float64)
            times = backend.where(indices <= nsamples - indices, indices, indices - nsamples)
            signal = sum(
                weight
                * backend.exp(1j * energy * times)[:, None]
                * backend.exp(1j * charge * times)[None, :]
                for weight, energy, charge in zip(weights, energies, charges)
            )

            # absolute values, and phases of the signal at the time zero, which is only
            # evolved with H_D
            spectrum, estimate = hybrid_input_output(
                backend.abs(signal),
                phases=backend.angle(signal[0]),
                niterations=2000,
                seed=0,
            )
            print(
                spectrum.shape,
                backend.allclose(backend.angle(estimate[0]), backend.angle(signal[0])),
            )

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
        classically if :math:`\\ket{\\psi}` is easy to simulate. If ``None``, they are not
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
    of the time series :math:`f_{1}[j] = \\bra{\\phi} e^{i j \\Delta t H} \\ket{\\phi}`,
    with the circuit :math:`U_{\\phi}^{\\dagger} \\, e^{i j \\Delta t H} \\, U_{\\phi}`, where
    :math:`U_{\\phi}` prepares :math:`\\ket{\\phi}` from the all-zero state, and the
    probability of measuring the all-zero state is :math:`|f_{1}[j]|^{2}`. The phases are
    recovered with the vectorial phase retrieval of Ref. [2], generalized to :math:`R` reference
    states :math:`\\ket{\\psi_{r}}` that are orthogonal to :math:`\\ket{\\phi}` [1]. For each
    of them, the absolute values of the signals

    .. math::
        \\begin{aligned}
        f_{2}^{(r)}[j] &= \\bra{\\phi} e^{i j \\Delta t H} \\ket{\\psi_{r}} \\, , \\\\
        f_{3}^{(r)}[j] &= f_{1}[j] + f_{2}^{(r)}[j] \\, , \\\\
        f_{4}^{(r)}[j] &= f_{1}[j] + i f_{2}^{(r)}[j]
        \\end{aligned}

    are measured, which need the states :math:`\\ket{\\psi_{r}}`,
    :math:`(\\ket{\\phi} + \\ket{\\psi_{r}}) / \\sqrt{2}` and
    :math:`(\\ket{\\phi} + i \\ket{\\psi_{r}}) / \\sqrt{2}`. The phases of
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
    as the states are orthogonal, or if the numerator of :math:`G_{r}[j]` vanishes, the phase
    :math:`G_{r}[j] / |G_{r}[j]|` is not defined, and it is set to :math:`1`. For the first
    case this does not affect the signal, because :math:`f_{2}^{(r)}[j] = 0`, and so any phase
    satisfies the interference constraint.
    The global phase of the result is chosen such that :math:`f_{1}[0]`, which is one, is
    real.

    Example:
        .. testcode::

            from qibo import get_backend
            from qibo.models import vectorial_phase_retrieval

            backend = get_backend()

            # two signals whose spectra only have the first 5 out of 16 frequencies
            nsamples, support = 16, 5
            spectra = backend.concatenate(
                (
                    backend.cast(
                        backend.random_normal(0.0, 1.0, size=(2, support), seed=0)
                        + 1j * backend.random_normal(0.0, 1.0, size=(2, support), seed=1),
                        dtype=backend.complex128,
                    ),
                    backend.zeros((2, nsamples - support), dtype=backend.complex128),
                ),
                axis=1,
            )
            f1, f2 = backend.ifft(spectra, axis=1)
            phase = f1[0] / backend.abs(f1[0])
            f1, f2 = f1 / phase, f2 / phase

            # probabilities that the circuits measure, which are the squared absolute values of
            # f1, f2 and f1 + f2 and f1 + i f2 divided by 2, as their states are normalized
            target = backend.abs(f1) ** 2
            references = backend.cast(
                [
                    [
                        backend.abs(f2) ** 2,
                        backend.abs(f1 + f2) ** 2 / 2,
                        backend.abs(f1 + 1j * f2) ** 2 / 2,
                    ]
                ],
                dtype=backend.float64,
            )

            estimate, _ = vectorial_phase_retrieval(target, references, support)
            print(backend.allclose(estimate, f1))

        .. testoutput::

            True

    Args:
        target (ArrayLike): probabilities :math:`|f_{1}[j]|^{2}` of measuring the all-zero state
            after the evolution of :math:`\\ket{\\phi}`, at :math:`j = 0, \\dots, N - 1`.
        references (ArrayLike): array of shape :math:`(R, 3, N)`. For each state
            :math:`\\ket{\\psi_{r}}`, the probabilities of measuring :math:`\\ket{\\phi}`
            after the evolution of :math:`\\ket{\\psi_{r}}`, of
            :math:`(\\ket{\\phi} + \\ket{\\psi_{r}}) / \\sqrt{2}` and of
            :math:`(\\ket{\\phi} + i \\ket{\\psi_{r}}) / \\sqrt{2}`, in this order.
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
        denominator * modulus > 0,
        ratio / backend.where(modulus > 0, modulus, 1.0),
        1.0,
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


def _check_inputs(
    unitary: Circuit | Gate | ArrayLike, nbits: int, name: str
) -> tuple[Circuit, int]:
    """Checks the inputs of the phase estimation circuits and converts them if needed.

    Args:
        unitary (:class:`qibo.models.circuit.Circuit` or :class:`qibo.gates.Gate` or ndarray):
            unitary, given as a circuit, as a gate or as a matrix.
        nbits (int): number of counting qubits or of bits of the phase to be estimated.
        name (str): name of ``nbits`` in the error messages.

    Returns:
        tuple: circuit that implements the unitary, and ``nbits`` as an ``int``.
    """
    if not isinstance(unitary, (Circuit, Gate)):
        shape = getattr(unitary, "shape", None)
        if shape is None:
            raise_error(
                TypeError,
                "unitary must be a Circuit, a Gate or a matrix, "
                + f"but it is type {type(unitary)}.",
            )

        if (
            len(shape) != 2
            or shape[0] != shape[1]
            or shape[0] < 2
            or shape[0] & (shape[0] - 1)
        ):
            raise_error(
                ValueError,
                "unitary must be a square matrix of dimension 2^n, with n > 0, "
                + f"but it has shape {tuple(shape)}.",
            )

        unitary = gates.Unitary(unitary, *range(int(math.log2(shape[0]))))
        if not unitary.unitary:
            raise_error(ValueError, "unitary must be a unitary matrix.")

    if isinstance(unitary, Gate):
        qubits = sorted(unitary.qubits)
        if len(qubits) == 0:
            raise_error(
                ValueError, "unitary must be a gate that acts on at least one qubit."
            )

        gate = unitary.on_qubits({qubit: index for index, qubit in enumerate(qubits)})
        unitary = Circuit(len(qubits))
        unitary.add(gate)

    if isinstance(nbits, bool) or not isinstance(nbits, Integral) or nbits < 1:
        raise_error(
            ValueError,
            f"{name} must be a positive integer, but it is {nbits}.",
        )

    return unitary, int(nbits)
