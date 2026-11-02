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
    nqubits = _check_circuits(algorithm, oracle)
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


def QAA(
    algorithm: Circuit,
    oracle: Circuit,
    niterations: int | None = None,
    probability: float | None = None,
    exact: bool = False,
    measure: bool = True,
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
        angle = math.asin(math.sqrt(probability))
        niterations = (
            math.ceil(max(math.pi / (4 * angle) - 1 / 2, 0))
            if exact
            else math.floor(math.pi / (4 * angle))
        )

    circuit = Circuit(nqubits + 1 if exact else nqubits, **kwargs)
    circuit.add(algorithm.on_qubits(*qubits))
    if exact:
        # the auxiliary qubit lowers the success probability to sin^2(pi / (4m + 2)),
        # for which m iterations are enough to find a good state with certainty
        reduced = math.sin(math.pi / (4 * niterations + 2)) ** 2
        rotation = 2 * math.asin(math.sqrt(min(reduced / probability, 1.0)))
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

    _check_circuits(algorithm, oracle)

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
        maximum = math.ceil(growth**step)
        for niterations in (0, int(generator.integers(1, maximum + 1))):
            circuit = QAA(algorithm, oracle, niterations=niterations)
            outcome = backend.execute_circuit(circuit, nshots=1).frequencies()
            bitstring = next(iter(outcome))

            napplications += 2 * niterations + 1
            if check(bitstring):
                return bitstring, napplications

    return None, napplications


def _check_circuits(algorithm: Circuit, oracle: Circuit) -> int:
    """Checks the circuits of the amplitude amplification.

    Args:
        algorithm (:class:`qibo.models.circuit.Circuit`): circuit that prepares the state to be
            amplified.
        oracle (:class:`qibo.models.circuit.Circuit`): circuit that multiplies the good states
            by :math:`-1`.

    Returns:
        int: number of qubits of the circuits.
    """
    for name, circuit in (("algorithm", algorithm), ("oracle", oracle)):
        if not isinstance(circuit, Circuit):
            raise_error(
                TypeError,
                f"{name} must be a Circuit, but it is type {type(circuit)}.",
            )

        if len(circuit.measurements) > 0:
            raise_error(ValueError, f"{name} cannot contain measurements.")

    if algorithm.nqubits != oracle.nqubits:
        raise_error(
            ValueError,
            "algorithm and oracle must have the same number of qubits, "
            + f"but they have {algorithm.nqubits} and {oracle.nqubits}.",
        )

    return algorithm.nqubits
