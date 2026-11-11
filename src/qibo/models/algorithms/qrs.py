import math
from collections.abc import Sequence
from numbers import Integral

from numpy.typing import ArrayLike

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.models.algorithms.qaa import QAA
from qibo.models.circuit import Circuit


def QRS(
    algorithm: Circuit,
    target: ArrayLike,
    source: ArrayLike | None = None,
    qubits: Sequence[int] | None = None,
    probability: float = 1.0,
    exact: bool = False,
    measure: bool = True,
    backend: Backend | None = None,
    **kwargs,
) -> Circuit:
    """Creates a circuit that implements Quantum Rejection Sampling.

    Let the circuit :math:`\\mathcal{A}` prepare the state
    :math:`|\\pi\\rangle = \\sum_{k} \\sqrt{p_{k}} \\, |\\xi_{k}\\rangle |k\\rangle` from the
    all-zero state, where :math:`|k\\rangle` is the basis state of the index ``qubits``,
    :math:`p_{k}` is its source probability, and :math:`|\\xi_{k}\\rangle` are states of the
    other qubits that can be unknown. Given a target probability :math:`q_{k}`, the circuit
    prepares a state :math:`|\\tilde{\\sigma}\\rangle` whose squared overlap with
    :math:`|\\sigma\\rangle = \\sum_{k} \\sqrt{q_{k}} \\, |\\xi_{k}\\rangle |k\\rangle` is at
    least ``probability``, which is :math:`1` by default, in which case
    :math:`|\\tilde{\\sigma}\\rangle = |\\sigma\\rangle`. It acts on :math:`n + 1` qubits, where
    :math:`n` is the number of qubits of :math:`\\mathcal{A}`, and it is composed of the
    following steps:

    1. :math:`\\mathcal{A}`, and a rotation :math:`R_{y}` of the last qubit, the flag, which
       is controlled by the state :math:`|k\\rangle` of the index qubits and takes the flag
       from :math:`|0\\rangle` to :math:`\\sqrt{1 - \\sin^{2}\\theta_{k}} \\, |0\\rangle +
       \\sin\\theta_{k} \\, |1\\rangle`, with
       :math:`\\sin\\theta_{k} = \\epsilon_{k} / \\sqrt{p_{k}}`.
       The amplitudes :math:`\\epsilon_{k} = \\min\\{\\sqrt{p_{k}}, \\gamma \\sqrt{q_{k}}\\}` fill
       the "tanks" of height :math:`\\sqrt{p_{k}}` up to a "water level" :math:`\\gamma`, which
       is chosen such that the squared overlap of :math:`\\sum_{k} \\epsilon_{k} |\\xi_{k}\\rangle
       |k\\rangle` with :math:`|\\sigma\\rangle` is ``probability``. For ``probability=1``,
       :math:`\\epsilon_{k} \\propto \\sqrt{q_{k}}` with :math:`\\gamma = \\min_{k} \\sqrt{p_{k} /
       q_{k}}`. The probability of the flag being in the state :math:`|1\\rangle` is then
       :math:`\\sum_{k} \\epsilon_{k}^{2}`, and the state of the other qubits is
       :math:`|\\tilde{\\sigma}\\rangle`;
    2. the amplification of the flag being in the state :math:`|1\\rangle` with
       :func:`qibo.models.QAA`, which, if ``exact=True``, is done with certainty, by scaling
       all of :math:`\\epsilon_{k}` by the same factor, so that :math:`\\mathcal{A}` and its
       inverse are used :math:`O(1 / \\|\\epsilon\\|_{2})` times.

    Hence, the other qubits are in the state :math:`|\\tilde{\\sigma}\\rangle` if the flag is in
    the state :math:`|1\\rangle`, which has probability :math:`1` if ``exact=True``. The rotations
    are controlled by each of the :math:`2^{|\\mathrm{qubits}|}` basis states of the index
    qubits, so the circuit is only efficient for a small number of them.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import QRS

            # uniform distribution on 2 qubits, and a target distribution that is 1 for |11>
            algorithm = Circuit(2)
            algorithm.add(gates.H(qubit) for qubit in range(2))
            circuit = QRS(algorithm, [0, 0, 0, 1], exact=True)

            # the outcome is |11>, and the flag is |1>
            result = circuit(nshots=100)
            print(list(result.frequencies()))

        .. testoutput::

            ['111']

    Args:
        algorithm (:class:`qibo.models.circuit.Circuit`): circuit :math:`\\mathcal{A}` without
            measurements.
        target (ArrayLike): :math:`1`-dimensional array with the non-negative target
            probabilities :math:`q_{k}` of the :math:`2^{|\\mathrm{qubits}|}` basis states of the
            index qubits. It does not need to be normalized.
        source (ArrayLike, optional): :math:`1`-dimensional array with the source probabilities
            :math:`p_{k}` of the same basis states, where the ones smaller than
            :math:`10^{-12}` are considered to be zero. If ``None``, they are calculated from
            the simulation of ``algorithm``. Defaults to ``None``.
        qubits (Sequence[int], optional): index qubits of ``algorithm``, with the first one
            being the most significant bit of :math:`k`. If ``None``, they are all of the
            qubits of ``algorithm``. Defaults to ``None``.
        probability (float, optional): minimum squared overlap of the prepared state with
            :math:`|\\sigma\\rangle`, which is in the interval :math:`(0, 1]`. A lower value
            needs fewer applications of :math:`\\mathcal{A}`. It cannot be larger than
            :math:`\\sum_{k: p_{k} > 0} q_{k}`, which is :math:`1` if :math:`q_{k}` is zero
            wherever :math:`p_{k}` is zero. Defaults to :math:`1`.
        exact (bool, optional): If ``True``, the flag is found in the state :math:`|1\\rangle`
            with certainty. Defaults to ``False``.
        measure (bool, optional): If ``True``, adds a measurement of the qubits of
            ``algorithm`` and of the flag at the end of the circuit. Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the
            calculation of the rotations. If ``None``, it uses the current backend.
            Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: implementation of Quantum Rejection Sampling.

    References:
        1. M. Ozols, M. Roetteler, and J. Roland, *Quantum rejection sampling*,
        `ACM Trans. Comput. Theory 5, 11 (2013) <https://doi.org/10.1145/2493252.2493256>`_.
    """
    backend = _check_backend(backend)

    if not isinstance(algorithm, Circuit):
        raise_error(
            TypeError, f"algorithm must be a Circuit, but it is type {type(algorithm)}."
        )

    nqubits = algorithm.nqubits
    index = list(range(nqubits)) if qubits is None else list(qubits)
    if (
        len(index) == 0
        or len(set(index)) != len(index)
        or any(
            not isinstance(qubit, Integral) or not 0 <= qubit < nqubits
            for qubit in index
        )
    ):
        raise_error(
            ValueError,
            f"qubits must be distinct integers in [0, {nqubits}), but they are {index}.",
        )

    if not 0 < probability <= 1:
        raise_error(
            ValueError, f"probability must be in (0, 1], but it is {probability}."
        )

    if source is None:
        source = backend.execute_circuit(algorithm).probabilities(index)

    source = backend.cast(source, dtype=backend.float64)
    target = backend.cast(target, dtype=backend.float64)

    for name, array in (("source", source), ("target", target)):
        if (
            array.shape != (2 ** len(index),)
            or backend.any(array < 0)
            or backend.sum(array) <= 0
        ):
            raise_error(
                ValueError,
                f"{name} must be a non-negative non-zero array with {2 ** len(index)} entries.",
            )

    # amplitudes of the source and the target, which are in the same basis states
    # source probabilities that are only numerical noise are zero
    source = source / backend.sum(source)
    pi = [
        float(amplitude) if float(probability_) > 1e-12 else 0.0
        for amplitude, probability_ in zip(backend.sqrt(source), source)
    ]
    sigma = [
        float(amplitude) for amplitude in backend.sqrt(target / backend.sum(target))
    ]

    # the largest and smallest squared overlaps that are possible
    maximum = sum(s**2 for p, s in zip(pi, sigma) if p > 0)
    minimum = sum(p * s for p, s in zip(pi, sigma)) ** 2
    if probability > maximum * (1 + 1e-12):
        raise_error(
            ValueError,
            f"probability cannot be larger than {maximum}, which is the weight of the target "
            + "where the source is not zero.",
        )

    # the water level is the smallest one, which is the most efficient, if the probability
    # is the maximum, and the largest one if the probability is not above the minimum
    ratios = [p / s for p, s in zip(pi, sigma) if p > 0 and s > 0]
    low, high = min(ratios), max(ratios)
    if probability >= maximum * (1 - 1e-12):
        high = low
    elif probability > minimum:
        for _ in range(100):
            level = (low + high) / 2
            epsilon = [min(p, level * s) for p, s in zip(pi, sigma)]
            overlap = sum(s * e for s, e in zip(sigma, epsilon)) ** 2 / sum(
                e**2 for e in epsilon
            )
            low, high = (level, high) if overlap > probability else (low, level)
        low = high
    else:
        low = high
    epsilon = [min(p, low * s) for p, s in zip(pi, sigma)]

    # probability of the flag being in the state 1, and its amplification, which is exact if
    # all of the amplitudes are scaled to make it a number of iterations
    acceptance = sum(e**2 for e in epsilon)
    niterations = None
    if exact:
        theta = float(backend.arcsin(backend.sqrt(min(acceptance, 1.0))))
        niterations = math.ceil(math.pi / (4 * theta) - 1 / 2 - 1e-12)
        scaling = math.sin(math.pi / (2 * (2 * niterations + 1))) / math.sin(theta)
        epsilon = [scaling * e for e in epsilon]

    flag = nqubits
    amplitude = Circuit(nqubits + 1)
    amplitude.add(algorithm.on_qubits(*range(nqubits)))
    for state, (p, e) in enumerate(zip(pi, epsilon)):
        angle = 2 * float(backend.arcsin(min(e / p, 1.0))) if p > 0 else 0.0
        if angle == 0:
            continue

        bits = format(state, f"0{len(index)}b")
        flipped = [qubit for qubit, bit in zip(index, bits) if bit == "0"]
        amplitude.add(gates.X(qubit) for qubit in flipped)
        amplitude.add(gates.RY(flag, theta=angle).controlled_by(*index))
        amplitude.add(gates.X(qubit) for qubit in flipped)

    # the flag being in the state 1 is the good state
    oracle = Circuit(nqubits + 1)
    oracle.add(gates.Z(flag))

    return QAA(
        amplitude,
        oracle,
        niterations=niterations,
        probability=None if exact else min(acceptance, 1.0),
        measure=measure,
        backend=backend,
        **kwargs,
    )
