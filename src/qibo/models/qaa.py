import math
from collections.abc import Callable
from numbers import Integral, Real

import numpy as np

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.models.circuit import Circuit


def amplification_operator(algorithm: Circuit, oracle: Circuit, **kwargs) -> Circuit:
    """Creates the circuit of the amplitude amplification operator.

    Given a circuit :math:`\\mathcal{A}`, which prepares the state
    :math:`|\\Psi\\rangle = \\mathcal{A}|0\\rangle` from the all-zero state :math:`|0\\rangle`,
    and a Boolean function :math:`\\chi`, which marks the "good" computational basis states,
    the amplification operator is

    .. math::
        \\mathbf{Q} = -\\mathcal{A} \\, \\mathbf{S}_{0} \\, \\mathcal{A}^{-1} \\,
        \\mathbf{S}_{\\chi}.

    Here, :math:`\\mathbf{S}_{\\chi}` changes the sign of the amplitude of every good state and
    :math:`\\mathbf{S}_{0}` changes the sign of the amplitude of the state :math:`|0\\rangle`
    only. The operator :math:`\\mathbf{Q}` rotates :math:`|\\Psi\\rangle`, in the plane spanned by
    its good and bad components, by an angle :math:`2\\theta_{a}`, where
    :math:`a = \\sin^{2}(\\theta_{a})` is the probability of measuring a good state in
    :math:`|\\Psi\\rangle`. For :math:`\\mathcal{A}` equal to Hadamard gates on every qubit,
    :math:`\\mathbf{Q}` is the iterate of Grover's search algorithm.

    The operator is applied to the circuit as :math:`\\mathbf{S}_{\\chi}`,
    :math:`\\mathcal{A}^{-1}`, :math:`\\mathbf{S}_{0}`, :math:`\\mathcal{A}` and a global phase of
    :math:`\\pi`, which accounts for the minus sign and matters if the circuit is controlled.

    Args:
        algorithm (:class:`qibo.models.circuit.Circuit`): circuit :math:`\\mathcal{A}`. It cannot
            contain measurements.
        oracle (:class:`qibo.models.circuit.Circuit`): circuit :math:`\\mathbf{S}_{\\chi}`, a
            diagonal operator that multiplies the good states by :math:`-1`, *e.g.* a
            multi-controlled :class:`qibo.gates.Z` gate. It must act on the same number of qubits
            as ``algorithm``, and it cannot contain measurements.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: circuit that implements :math:`\\mathbf{Q}`, with
        the same number of qubits as ``algorithm``.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import amplification_operator

            # Hadamard gates on 2 qubits, and the oracle that marks the state |11>
            algorithm = Circuit(2)
            algorithm.add(gates.H(qubit) for qubit in range(2))
            oracle = Circuit(2)
            oracle.add(gates.CZ(0, 1))

            # for 2 qubits and 1 good state, one application of Q finds the good state
            circuit = Circuit(2)
            circuit.add(algorithm)
            circuit.add(amplification_operator(algorithm, oracle))
            circuit.add(gates.M(0, 1))

            result = circuit(nshots=10)
            print(result.frequencies())

        .. testoutput::

            Counter({'11': 10})

    References:
        1. G. Brassard, P. Høyer, M. Mosca, and A. Tapp, *Quantum Amplitude Amplification and
        Estimation*, `Contemporary Mathematics 305, 53 (2002)
        <https://doi.org/10.1090/conm/305/05215>`_.
    """
    nqubits = _check_circuits(algorithm=algorithm, oracle=oracle)
    qubits = list(range(nqubits))

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(oracle)
    circuit.add(algorithm.invert())

    # S_0 is a multi-controlled Z gate between X gates, so that it acts on the zero state
    circuit.add(gates.X(qubit) for qubit in qubits)
    circuit.add(gates.Z(qubits[-1]).controlled_by(*qubits[:-1]))
    circuit.add(gates.X(qubit) for qubit in qubits)

    circuit.add(algorithm)
    circuit.add(gates.GlobalPhase(math.pi))

    return circuit


def FPQAA(
    algorithm: Circuit,
    oracle: Circuit,
    delta: float,
    niterations: int | None = None,
    probability: float | None = None,
    measure: bool = True,
    backend: Backend = None,
    **kwargs,
) -> Circuit:
    """Creates a circuit that implements fixed-point Quantum Amplitude Amplification (FPQAA).

    Let :math:`\\mathcal{A}` be the circuit ``algorithm``, :math:`\\chi` a Boolean function that
    marks "good" computational basis states, and :math:`a` the probability of measuring a good
    state in :math:`|\\Psi\\rangle = \\mathcal{A}|0\\rangle`, where :math:`|0\\rangle` is the
    all-zero state. The success probability of :func:`qibo.models.qaa.QAA` oscillates with the
    number of iterations, so too many iterations decrease it, and the right number depends on
    :math:`a`. In contrast, this circuit, which follows Yoder, Low and Chuang, only needs a lower
    bound for :math:`a`, and more iterations never push its success probability below the bound.

    The circuit applies :math:`\\mathcal{A}` and then :math:`l` generalized Grover iterates,

    .. math::
        \\mathcal{S}_{L} = G(\\alpha_{l}, \\beta_{l}) \\cdots G(\\alpha_{1}, \\beta_{1}),
        \\qquad
        G(\\alpha, \\beta) = -\\mathbf{S}_{\\Psi}(\\alpha) \\, \\mathbf{S}_{\\chi}(\\beta),

    with :math:`L = 2l + 1`. Here, :math:`\\mathbf{S}_{\\chi}(\\beta)` multiplies the amplitude of
    every good state by :math:`e^{i\\beta}`, and :math:`\\mathbf{S}_{\\Psi}(\\alpha) =
    \\mathcal{A} \\, \\mathbf{S}_{0}(\\alpha) \\, \\mathcal{A}^{-1}` multiplies the amplitude of
    :math:`|\\Psi\\rangle` by :math:`e^{-i\\alpha}`, where :math:`\\mathbf{S}_{0}(\\alpha)` does it
    for :math:`|0\\rangle`. The phases are

    .. math::
        \\alpha_{j} = -\\beta_{l - j + 1} = 2 \\cot^{-1}\\left(\\tan(2 \\pi j / L)
        \\sqrt{1 - \\gamma^{2}}\\right),
        \\qquad
        \\gamma^{-1} = T_{1 / L}(1 / \\delta),

    where :math:`T_{n}(x) = \\cos(n \\cos^{-1}(x))` is the Chebyshev polynomial of the first kind
    of degree :math:`n`, which is also defined for :math:`x > 1` and non-integer :math:`n`. The
    probability of measuring a good state is then

    .. math::
        P_{L} = 1 - \\delta^{2} \\, T_{L}\\left(T_{1 / L}(1 / \\delta) \\sqrt{1 - a}\\right)^{2}
        \\geq 1 - \\delta^{2} \\quad \\text{for every} \\quad
        a \\geq 1 - T_{1 / L}(1 / \\delta)^{-2}.

    The lower bound on :math:`a` decreases as :math:`L` grows, and it is about
    :math:`(\\log(2 / \\delta) / L)^{2}` for large :math:`L` and small :math:`\\delta`. Thus, the
    oracle is used :math:`2l = L - 1` times, which is in :math:`O(\\log(2 / \\delta) / \\sqrt{a})`
    for a suitable :math:`L`, as in :func:`qibo.models.qaa.QAA`. For :math:`\\delta = 1` the circuit
    is Grover's one, *i.e.* the one of :func:`qibo.models.qaa.QAA` with ``niterations=l``, and for
    :math:`\\delta = 0` and :math:`l = 1` it is the :math:`\\pi / 3` fixed-point search of
    Grover, whose probability is :math:`1 - (1 - a)^{L}`.

    The circuit acts on the qubits of ``algorithm`` and on one auxiliary qubit, which is the last
    qubit, starts in the state :math:`|0\\rangle` and returns to it. It is used to apply the phase
    :math:`e^{i\\beta}` to the good states: the oracle is controlled by the auxiliary qubit and
    surrounded by Hadamard gates, which flips the auxiliary qubit if and only if the state is
    good, then a phase gate acts on it and the flip is undone. Hence, all of the gates of the
    oracle must be supported by :meth:`qibo.models.circuit.Circuit.controlled_by`. The measurement,
    if ``measure=True``, is applied to the qubits of ``algorithm`` only. The number of iterations
    :math:`l` is determined in one of the following ways:

    * ``niterations`` is given, and :math:`l` is its value;
    * ``probability`` is given, which is a lower bound :math:`a_{0}` of :math:`a`, and :math:`l` is
      the smallest number of iterations for which :math:`P_{L} \\geq 1 - \\delta^{2}` for every
      :math:`a \\geq a_{0}`, *i.e.*
      :math:`L \\geq \\cosh^{-1}(1 / \\delta) / \\tanh^{-1}(\\sqrt{a_{0}})`. Note that the bound
      on the failure probability is trivial for :math:`\\delta = 1`, and then :math:`l = 0`.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import FPQAA

            # Hadamard gates on 3 qubits, and the oracle that marks the state |111>
            nqubits = 3
            algorithm = Circuit(nqubits)
            algorithm.add(gates.H(qubit) for qubit in range(nqubits))
            oracle = Circuit(nqubits)
            oracle.add(gates.CCZ(0, 1, 2))

            # the failure probability is at most 0.1^2 if the probability of |111>, which is
            # 1 / 8, is not smaller than 1 / 16
            circuit = FPQAA(algorithm, oracle, delta=0.1, probability=1 / 16)

            result = circuit(nshots=1000)
            print(result.frequencies().most_common(1)[0][0])

        .. testoutput::

            111

    Args:
        algorithm (:class:`qibo.models.circuit.Circuit`): circuit :math:`\\mathcal{A}` without
            measurements.
        oracle (:class:`qibo.models.circuit.Circuit`): circuit :math:`\\mathbf{S}_{\\chi}` that
            multiplies the good states by :math:`-1`, without measurements and with the same
            number of qubits as ``algorithm``. See :func:`qibo.models.qaa.amplification_operator`.
        delta (float): bound :math:`\\delta \\in [0, 1]` of the failure probability, which is
            at most :math:`\\delta^{2}` over the range of :math:`a` above.
        niterations (int, optional): number :math:`l \\geq 0` of generalized Grover iterates.
            Exactly one of ``niterations`` and ``probability`` has to be given.
            Defaults to ``None``.
        probability (float, optional): lower bound :math:`a_{0} \\in (0, 1]` of the probability of
            measuring a good state in :math:`\\mathcal{A}|0\\rangle`, from which the number of
            iterations is computed. It requires :math:`\\delta > 0`. Defaults to ``None``.
        measure (bool, optional): If ``True``, adds a measurement of the qubits of
            ``algorithm`` at the end of the circuit. Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the
            calculation of the number of iterations and of the phases. If ``None``, it
            uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: implementation of fixed-point Quantum Amplitude
        Amplification, with one more qubit than ``algorithm``.

    References:
        1. T. J. Yoder, G. H. Low, and I. L. Chuang, *Fixed-Point Quantum Search with an Optimal
        Number of Queries*, `Phys. Rev. Lett. 113, 210501 (2014)
        <https://doi.org/10.1103/PhysRevLett.113.210501>`_.
    """
    backend = _check_backend(backend)

    nqubits = _check_circuits(algorithm, oracle)

    if isinstance(delta, bool) or not isinstance(delta, Real) or not 0 <= delta <= 1:
        raise_error(
            ValueError, f"delta must be in the interval [0, 1], but it is {delta}."
        )

    if (niterations is None) == (probability is None):
        raise_error(
            ValueError,
            "Exactly one of niterations and probability must be given, "
            + f"but niterations is {niterations} and probability is {probability}.",
        )

    if niterations is not None and (
        isinstance(niterations, bool)
        or not isinstance(niterations, Integral)
        or niterations < 0
    ):
        raise_error(
            ValueError,
            f"niterations must be a non-negative integer, but it is {niterations}.",
        )

    if probability is not None and (
        isinstance(probability, bool)
        or not isinstance(probability, Real)
        or not 0 < probability <= 1
        or delta == 0
    ):
        raise_error(
            ValueError,
            "probability must be in the interval (0, 1], and delta must be positive, "
            + f"but they are {probability} and {delta}.",
        )

    qubits = list(range(nqubits))
    auxiliary = nqubits

    if probability is not None:
        # L is the smallest odd integer such that the fixed-point bound holds for probability
        width = (
            math.inf
            if probability == 1
            else float(backend.arctanh(backend.sqrt(probability)))
        )
        niterations = max(
            int(backend.ceil((backend.arccosh(1 / delta) / width - 1) / 2)), 0
        )

    length = 2 * niterations + 1
    gamma = (
        0.0
        if delta == 0
        else float(1 / backend.cosh(backend.arccosh(1 / delta) / length))
    )
    # the phases are 2 cot^{-1}(x), which for x in the real line is 2 arctan2(1, x)
    alphas = [
        float(
            2
            * backend.arctan2(
                1.0, backend.tan(2 * math.pi * j / length) * backend.sqrt(1 - gamma**2)
            )
        )
        for j in range(1, niterations + 1)
    ]

    # the oracle flips the auxiliary qubit if and only if the state is good
    flip = Circuit(nqubits + 1, **kwargs)
    flip.add(gates.H(auxiliary))
    flip.add(oracle.controlled_by(auxiliary))
    flip.add(gates.H(auxiliary))

    inverse = algorithm.invert()

    circuit = Circuit(nqubits + 1, **kwargs)
    circuit.add(algorithm.on_qubits(*qubits))
    for alpha, beta in zip(alphas, [-alpha for alpha in reversed(alphas)]):
        # S_chi(beta), which multiplies the good states by exp(i beta)
        circuit.add(flip)
        circuit.add(gates.U1(auxiliary, theta=beta))
        circuit.add(flip)

        # S_Psi(alpha) = A S_0(alpha) A^{-1}, where S_0(alpha) multiplies |0> by exp(-i alpha)
        circuit.add(inverse.on_qubits(*qubits))
        circuit.add(gates.X(qubit) for qubit in qubits)
        circuit.add(gates.U1(qubits[-1], theta=-alpha).controlled_by(*qubits[:-1]))
        circuit.add(gates.X(qubit) for qubit in qubits)
        circuit.add(algorithm.on_qubits(*qubits))

        circuit.add(gates.GlobalPhase(math.pi))

    if measure:
        circuit.add(gates.M(*qubits))

    return circuit


def OAA(
    unitary: Circuit,
    nauxiliary: int,
    niterations: int | None = None,
    probability: float | None = None,
    exact: bool = False,
    **kwargs,
) -> Circuit:
    """Create a circuit that implements Oblivious Amplitude Amplification (OAA).

    Let :math:`U` be the circuit ``unitary``, which acts on :math:`\\mu` auxiliary qubits, the
    first ones of the circuit, and on :math:`n` system qubits, the last ones. Let
    :math:`|0^{\\mu}\\rangle` be the all-zero state of the auxiliary qubits and
    :math:`\\Pi = |0^{\\mu}\\rangle\\langle 0^{\\mu}| \\otimes \\mathbb{1}` the projector onto it.
    The circuit applies a unitary :math:`V` on the system qubits with amplitude
    :math:`\\sin(\\theta)`, for every state :math:`|\\psi\\rangle` of the system qubits, *i.e.*

    .. math::
        U |0^{\\mu}\\rangle |\\psi\\rangle = \\sin(\\theta) |0^{\\mu}\\rangle V |\\psi\\rangle
        + \\cos(\\theta) |\\Phi^{\\perp}\\rangle,

    where :math:`|\\Phi^{\\perp}\\rangle`, which depends on :math:`|\\psi\\rangle`, satisfies
    :math:`\\Pi |\\Phi^{\\perp}\\rangle = 0`. This is the case of a block encoding of
    :math:`\\sin(\\theta) V`, and the probability :math:`p = \\sin^{2}(\\theta)` of measuring the
    auxiliary qubits in the state :math:`|0^{\\mu}\\rangle` is the same for every
    :math:`|\\psi\\rangle`. The circuit then boosts this amplitude with the operator

    .. math::
        \\mathbf{S} = -U \\mathbf{R} U^{\\dagger} \\mathbf{R}, \\qquad
        \\mathbf{R} = 2 \\Pi - \\mathbb{1},

    which satisfies, for any integer :math:`l`,

    .. math::
        \\mathbf{S}^{l} U |0^{\\mu}\\rangle |\\psi\\rangle
        = \\sin((2l + 1)\\theta) |0^{\\mu}\\rangle V |\\psi\\rangle
        + \\cos((2l + 1)\\theta) |\\Phi^{\\perp}\\rangle.

    In contrast with :func:`qibo.models.qaa.amplification_operator`, :math:`\\mathbf{S}` does not
    reflect about the state :math:`U|0^{\\mu}\\rangle|\\psi\\rangle`, which is unknown since
    :math:`|\\psi\\rangle` is not. Thus, the circuit amplifies the implementation of :math:`V`
    without knowing, preparing or measuring the input state, so it can be a part of a larger
    circuit. It is applied to the circuit as :math:`U`, followed by :math:`l` times a sign flip of
    :math:`|0^{\\mu}\\rangle`, :math:`U^{\\dagger}`, the sign flip, :math:`U` and a global phase of
    :math:`\\pi`, since :math:`\\mathbf{R}` is minus the sign flip. The number :math:`l` of
    iterations is determined in one of the following ways:

    * ``niterations`` is given, and :math:`l` is its value;
    * ``probability`` is given, which is :math:`p`, and
      :math:`l = \\lfloor \\pi / (4 \\theta) \\rfloor`. The probability of measuring the auxiliary
      qubits in the state :math:`|0^{\\mu}\\rangle` is then at least :math:`\\max(p, 1 - p)`;
    * ``probability`` is given and ``exact=True``. Then :math:`V` is implemented exactly, *i.e.*
      the final state is :math:`|0^{\\mu}\\rangle V|\\psi\\rangle`. The circuit uses
      :math:`l = \\lceil \\pi / (4 \\theta) - 1 / 2 \\rceil` iterations, which is a fraction of an
      iteration too many. To compensate it, the amplitude is lowered to
      :math:`\\sin(\\pi / (4l + 2))` with an extra auxiliary qubit, which is the last qubit of
      the circuit, and rotated by a :math:`R_{y}` gate. The extra qubit is also required to be in
      the state :math:`|0\\rangle` at the end, where it is found with certainty. For
      :math:`p = 1 / 4`, one iteration is enough, so :math:`U` is used three times.

    Example:
        .. testcode::

            import math

            from qibo import Circuit, gates
            from qibo.models import OAA

            # U acts on 1 auxiliary qubit and 1 system qubit. If the auxiliary qubit is found in
            # the state 0, which has probability sin(theta)^2 = 1 / 4, V is the X gate
            theta = math.pi / 6
            unitary = Circuit(2)
            unitary.add(gates.X(1))
            unitary.add(gates.RY(0, theta=math.pi - 2 * theta))
            unitary.add(gates.H(1).controlled_by(0))

            # one iteration implements V with certainty, so the system qubit, in the state 0,
            # is flipped without measuring the auxiliary qubit before
            circuit = Circuit(2)
            circuit.add(OAA(unitary, 1, probability=1 / 4))
            circuit.add(gates.M(0, 1))

            result = circuit(nshots=100)
            print(result.frequencies())

        .. testoutput::

            Counter({'01': 100})

    Args:
        unitary (:class:`qibo.models.circuit.Circuit`): circuit :math:`U` without measurements.
        nauxiliary (int): number :math:`\\mu \\geq 1` of auxiliary qubits, which are the first
            qubits of ``unitary``.
        niterations (int, optional): number :math:`l \\geq 0` of applications of
            :math:`\\mathbf{S}`. Exactly one of ``niterations`` and ``probability`` has to be
            given. Defaults to ``None``.
        probability (float, optional): probability :math:`p \\in (0, 1]` of measuring the
            auxiliary qubits in the state :math:`|0^{\\mu}\\rangle` after :math:`U`, from which the
            number of iterations is computed. Defaults to ``None``.
        exact (bool, optional): If ``True``, :math:`V` is implemented exactly, which requires
            ``probability``. Defaults to ``False``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: implementation of Oblivious Amplitude
        Amplification, with one more qubit than ``unitary`` if ``exact=True``.

    References:
        1. D. W. Berry, A. M. Childs, R. Cleve, R. Kothari, and R. D. Somma, *Exponential
        improvement in precision for simulating sparse Hamiltonians*, `Forum Math. Sigma 5, e8
        (2017) <https://doi.org/10.1017/fms.2017.2>`_.
    """
    nqubits = _check_circuits(unitary=unitary)

    if (
        isinstance(nauxiliary, bool)
        or not isinstance(nauxiliary, Integral)
        or not 1 <= nauxiliary <= nqubits
    ):
        raise_error(
            ValueError,
            f"nauxiliary must be an integer between 1 and {nqubits}, "
            + f"but it is {nauxiliary}.",
        )

    niterations, probability = _check_iterations(niterations, probability, exact)

    auxiliary = list(range(nauxiliary))

    circuit = Circuit(nqubits + 1 if exact else nqubits, **kwargs)
    circuit.add(unitary.on_qubits(*range(nqubits)))
    if exact:
        # the extra qubit lowers the amplitude to sin(pi / (4l + 2)), for which l iterations
        # implement V with certainty, and it has to be found in the state 0
        reduced = math.sin(math.pi / (4 * niterations + 2))
        rotation = 2 * math.acos(min(reduced / math.sqrt(probability), 1.0))
        circuit.add(gates.RY(nqubits, theta=rotation))
        auxiliary.append(nqubits)

    # sign flip of the zero state of the auxiliary qubits, which is minus R
    flip = Circuit(circuit.nqubits, **kwargs)
    flip.add(gates.X(qubit) for qubit in auxiliary)
    flip.add(gates.Z(auxiliary[-1]).controlled_by(*auxiliary[:-1]))
    flip.add(gates.X(qubit) for qubit in auxiliary)

    # S = -U R U^{-1} R = -U (-flip) U^{-1} (-flip), where the signs of R cancel
    operator = Circuit(circuit.nqubits, **kwargs)
    operator.add(flip)
    operator.add(circuit.invert())
    operator.add(flip)
    operator.add(circuit)
    operator.add(gates.GlobalPhase(math.pi))

    for _ in range(niterations):
        circuit.add(operator)

    return circuit


def QAA(
    algorithm: Circuit,
    oracle: Circuit,
    niterations: int | None = None,
    probability: float | None = None,
    exact: bool = False,
    measure: bool = True,
    backend: Backend = None,
    **kwargs,
) -> Circuit:
    """Creates a circuit that implements Quantum Amplitude Amplification (QAA).

    Given a circuit :math:`\\mathcal{A}` and a Boolean function :math:`\\chi` that marks "good"
    computational basis states, let :math:`a` be the probability of measuring a good state in
    :math:`\\mathcal{A}|0\\rangle`, where :math:`|0\\rangle` is the all-zero state. The circuit
    applies :math:`\\mathcal{A}` and then :math:`m` times the operator :math:`\\mathbf{Q}` of
    :func:`qibo.models.qaa.amplification_operator`, such that the probability of measuring a good
    state becomes :math:`\\sin^{2}((2m + 1)\\theta_{a})`, with :math:`a = \\sin^{2}(\\theta_{a})`
    and :math:`0 \\leq \\theta_{a} \\leq \\pi / 2`. The number :math:`m` of iterations is
    determined in one of the following ways:

    * ``niterations`` is given, and :math:`m` is its value;
    * ``probability`` is given, which is :math:`a`, and
      :math:`m = \\lfloor \\pi / (4 \\theta_{a}) \\rfloor`. The probability of measuring a
      good state is then at least :math:`\\max(a, 1 - a)`, and :math:`m` is in
      :math:`O(1 / \\sqrt{a})`, whereas :math:`O(1 / a)` repetitions of
      :math:`\\mathcal{A}` are needed without amplification;
    * ``probability`` is given and ``exact=True``. Then the good states are found with
      certainty. The circuit uses :math:`m = \\lceil \\pi / (4 \\theta_{a}) - 1 / 2 \\rceil`
      iterations, which is a fraction of an iteration too many. To compensate it, the success
      probability of :math:`\\mathcal{A}` is lowered to :math:`\\sin^{2}(\\pi / (4m + 2))` with an
      extra auxiliary qubit, which is the last qubit of the circuit. It is rotated by
      :math:`R_{y}` gate and a state is considered good only if :math:`\\chi` marks it and the
      auxiliary qubit is in the state :math:`|1\\rangle`. The oracle is controlled by the
      auxiliary qubit, so all of its gates must be supported by
      :meth:`qibo.models.circuit.Circuit.controlled_by`.

    The circuit acts on the qubits of ``algorithm``, plus the auxiliary qubit if ``exact=True``,
    and the measurement, if ``measure=True``, is applied to the qubits of ``algorithm`` only. If
    the probability :math:`a` is unknown, use :func:`qibo.models.qaa.QSearch`.

    Args:
        algorithm (:class:`qibo.models.circuit.Circuit`): circuit :math:`\\mathcal{A}` without
            measurements.
        oracle (:class:`qibo.models.circuit.Circuit`): circuit :math:`\\mathbf{S}_{\\chi}` that
            multiplies the good states by :math:`-1`, without measurements and with the same
            number of qubits as ``algorithm``. See :func:`qibo.models.qaa.amplification_operator`.
        niterations (int, optional): number :math:`m \\geq 0` of applications of
            :math:`\\mathbf{Q}`. Exactly one of ``niterations`` and ``probability`` has to be
            given. Defaults to ``None``.
        probability (float, optional): probability :math:`a \\in (0, 1]` of measuring a good state
            in :math:`\\mathcal{A}|0\\rangle`, from which the number of iterations is computed.
            Defaults to ``None``.
        exact (bool, optional): If ``True``, the good states are found with certainty, which
            requires ``probability``. Defaults to ``False``.
        measure (bool, optional): If ``True``, adds a measurement of the qubits of
            ``algorithm`` at the end of the circuit. Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the
            calculation of the number of iterations and of the rotation angle. If ``None``, it
            uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: implementation of Quantum Amplitude Amplification.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import QAA

            # Hadamard gates on 3 qubits, and the oracle that marks the state |111>
            nqubits = 3
            algorithm = Circuit(nqubits)
            algorithm.add(gates.H(qubit) for qubit in range(nqubits))
            oracle = Circuit(nqubits)
            oracle.add(gates.CCZ(0, 1, 2))

            # the initial probability of measuring |111> is 1 / 8
            circuit = QAA(algorithm, oracle, probability=1 / 8)

            result = circuit(nshots=1000)
            print(result.frequencies().most_common(1)[0][0])

        .. testoutput::

            111

    References:
        1. G. Brassard, P. Høyer, M. Mosca, and A. Tapp, *Quantum Amplitude Amplification and
        Estimation*, `Contemporary Mathematics 305, 53 (2002)
        <https://doi.org/10.1090/conm/305/05215>`_.
    """
    backend = _check_backend(backend)

    nqubits = _check_circuits(algorithm, oracle)

    if (niterations is None) == (probability is None):
        raise_error(
            ValueError,
            "Exactly one of niterations and probability must be given, "
            + f"but niterations is {niterations} and probability is {probability}.",
        )

    if exact and probability is None:
        raise_error(ValueError, "exact=True requires probability.")

    if niterations is not None and (
        isinstance(niterations, bool)
        or not isinstance(niterations, Integral)
        or niterations < 0
    ):
        raise_error(
            ValueError,
            f"niterations must be a non-negative integer, but it is {niterations}.",
        )

    if probability is not None and (
        isinstance(probability, bool)
        or not isinstance(probability, Real)
        or not 0 < probability <= 1
    ):
        raise_error(
            ValueError,
            f"probability must be in the interval (0, 1], but it is {probability}.",
        )

    qubits = list(range(nqubits))

    if probability is not None:
        angle = float(backend.arcsin(backend.sqrt(probability)))
        niterations = (
            int(backend.ceil(max(math.pi / (4 * angle) - 1 / 2, 0)))
            if exact
            else int(backend.floor(math.pi / (4 * angle)))
        )

    circuit = Circuit(nqubits + 1 if exact else nqubits, **kwargs)
    circuit.add(algorithm.on_qubits(*qubits))
    if exact:
        # the auxiliary qubit lowers the success probability to sin^2(pi / (4m + 2)),
        # for which m iterations are enough to find a good state with certainty
        reduced = backend.sin(math.pi / (4 * niterations + 2)) ** 2
        rotation = float(
            2 * backend.arcsin(backend.sqrt(min(reduced / probability, 1.0)))
        )
        circuit.add(gates.RY(nqubits, theta=rotation))
        oracle = oracle.controlled_by(nqubits)

    operator = amplification_operator(circuit, oracle, **kwargs)
    for _ in range(niterations):
        circuit.add(operator)

    if measure:
        circuit.add(gates.M(*qubits))

    return circuit


def QSearch(
    algorithm: Circuit,
    oracle: Circuit,
    check: Callable[[str], bool],
    growth: float = 6 / 5,
    max_rounds: int = 30,
    seed: int | np.random.Generator | None = None,
    backend: Backend = None,
) -> tuple[str | None, int]:
    """Finds a good state with Quantum Amplitude Amplification when its probability is unknown.

    Let :math:`a` be the unknown probability of measuring a good state in
    :math:`\\mathcal{A}|0\\rangle`, where :math:`\\mathcal{A}` is the circuit ``algorithm`` and
    :math:`|0\\rangle` is the all-zero state. In each round :math:`\\ell = 1, 2, \\dots`, the
    algorithm:

    1. sets :math:`M = \\lceil c^{\\ell} \\rceil`, where :math:`c` is ``growth``;
    2. measures :math:`\\mathcal{A}|0\\rangle`, and stops if the outcome is good;
    3. draws an integer :math:`j` uniformly at random between :math:`1` and :math:`M`, measures
       :math:`\\mathbf{Q}^{j}\\mathcal{A}|0\\rangle`, with :math:`\\mathbf{Q}` the operator of
       :func:`qibo.models.qaa.amplification_operator`, and stops if the outcome is good.

    A good outcome is detected with the classical function ``check``. If :math:`a > 0`, the
    expected number of applications of :math:`\\mathcal{A}` and :math:`\\mathcal{A}^{-1}` is in
    :math:`\\Theta(1 / \\sqrt{a})`, and otherwise the search never finishes, which is why the
    number of rounds is limited by ``max_rounds``.

    Args:
        algorithm (:class:`qibo.models.circuit.Circuit`): circuit :math:`\\mathcal{A}` without
            measurements.
        oracle (:class:`qibo.models.circuit.Circuit`): circuit :math:`\\mathbf{S}_{\\chi}` that
            multiplies the good states by :math:`-1`, without measurements and with the same
            number of qubits as ``algorithm``. See :func:`qibo.models.qaa.amplification_operator`.
        check (Callable): function that receives a measured bitstring, *e.g.* ``"101"``, and
            returns ``True`` if the state is good.
        growth (float, optional): factor :math:`c` by which the maximum number of iterations
            grows in each round, which has to be in the interval :math:`(1, 2)`. Defaults to
            :math:`6 / 5`.
        max_rounds (int, optional): maximum number of rounds. Defaults to :math:`30`.
        seed (int or :class:`numpy.random.Generator`, optional): seed or generator used to draw
            :math:`j`. The measurements are sampled by the backend, so they are reproducible only
            if the seed of the backend is set, *e.g.* with
            :meth:`qibo.backends.abstract.Backend.set_seed`. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the
            execution of the circuits. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        tuple: the good bitstring, or ``None`` if none was found in ``max_rounds`` rounds, and the
        number of applications of :math:`\\mathcal{A}` and :math:`\\mathcal{A}^{-1}` used.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import QSearch

            # Hadamard gates on 4 qubits, and the oracle that marks the state |1011>
            nqubits = 4
            algorithm = Circuit(nqubits)
            algorithm.add(gates.H(qubit) for qubit in range(nqubits))
            oracle = Circuit(nqubits)
            oracle.add(gates.X(2))
            oracle.add(gates.Z(3).controlled_by(0, 1, 2))
            oracle.add(gates.X(2))

            bitstring, napplications = QSearch(
                algorithm, oracle, check=lambda bitstring: bitstring == "1011", seed=0
            )
            print(bitstring)

        .. testoutput::

            1011

    References:
        1. G. Brassard, P. Høyer, M. Mosca, and A. Tapp, *Quantum Amplitude Amplification and
        Estimation*, `Contemporary Mathematics 305, 53 (2002)
        <https://doi.org/10.1090/conm/305/05215>`_.
    """
    backend = _check_backend(backend)

    _check_circuits(algorithm=algorithm, oracle=oracle)

    if not callable(check):
        raise_error(TypeError, f"check must be callable, but it is {type(check)}.")

    if isinstance(growth, bool) or not isinstance(growth, Real) or not 1 < growth < 2:
        raise_error(
            ValueError, f"growth must be in the interval (1, 2), but it is {growth}."
        )

    if (
        isinstance(max_rounds, bool)
        or not isinstance(max_rounds, Integral)
        or max_rounds < 1
    ):
        raise_error(
            ValueError,
            f"max_rounds must be a positive integer, but it is {max_rounds}.",
        )

    generator = np.random.default_rng(seed)

    napplications = 0
    for step in range(1, max_rounds + 1):
        maximum = int(backend.ceil(growth**step))
        for niterations in (0, int(generator.integers(1, maximum + 1))):
            circuit = QAA(algorithm, oracle, niterations=niterations, backend=backend)
            outcome = backend.execute_circuit(circuit, nshots=1).frequencies()
            bitstring = next(iter(outcome))

            napplications += 2 * niterations + 1
            if check(bitstring):
                return bitstring, napplications

    return None, napplications


def _check_circuits(**circuits: Circuit) -> int:
    """Checks the circuits of the amplitude amplification.

    Args:
        circuits (dict): circuits to be checked, with their names as keys. They cannot contain
            measurements, and they must have the same number of qubits.

    Returns:
        int: number of qubits of the circuits.
    """
    for name, circuit in circuits.items():
        if not isinstance(circuit, Circuit):
            raise_error(
                TypeError,
                f"{name} must be a Circuit, but it is type {type(circuit)}.",
            )

        if len(circuit.measurements) > 0:
            raise_error(ValueError, f"{name} cannot contain measurements.")

    nqubits = {circuit.nqubits for circuit in circuits.values()}
    if len(nqubits) > 1:
        raise_error(
            ValueError,
            f"{' and '.join(circuits)} must have the same number of qubits, "
            + f"but they have {' and '.join(str(c.nqubits) for c in circuits.values())}.",
        )

    return nqubits.pop()


def _check_iterations(
    niterations: int | None, probability: float | None, exact: bool
) -> tuple[int, float | None]:
    """Checks the arguments that determine the number of iterations of the amplification.

    Args:
        niterations (int, optional): number of iterations, or ``None``.
        probability (float, optional): probability of measuring a good state, or ``None``.
        exact (bool): whether the good state has to be found with certainty.

    Returns:
        tuple: the number of iterations, which is computed from ``probability`` if
        ``niterations`` is ``None``, and ``probability``.
    """
    if (niterations is None) == (probability is None):
        raise_error(
            ValueError,
            "Exactly one of niterations and probability must be given, "
            + f"but niterations is {niterations} and probability is {probability}.",
        )

    if exact and probability is None:
        raise_error(ValueError, "exact=True requires probability.")

    if niterations is not None and (
        isinstance(niterations, bool)
        or not isinstance(niterations, Integral)
        or niterations < 0
    ):
        raise_error(
            ValueError,
            f"niterations must be a non-negative integer, but it is {niterations}.",
        )

    if probability is not None and (
        isinstance(probability, bool)
        or not isinstance(probability, Real)
        or not 0 < probability <= 1
    ):
        raise_error(
            ValueError,
            f"probability must be in the interval (0, 1], but it is {probability}.",
        )

    if probability is None:
        return int(niterations), None

    angle = math.asin(math.sqrt(probability))
    if exact:
        return math.ceil(max(math.pi / (4 * angle) - 1 / 2, 0)), float(probability)

    return math.floor(math.pi / (4 * angle)), float(probability)
