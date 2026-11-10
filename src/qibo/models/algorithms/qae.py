from qibo.models.algorithms.qaa import amplification_operator
from qibo.models.algorithms.qpe import QPE
from qibo.models.circuit import Circuit


def QAE(
    algorithm: Circuit,
    oracle: Circuit,
    nqubits_counting: int,
    measure: bool = True,
    **kwargs,
) -> Circuit:
    """Creates a circuit that implements the canonical Quantum Amplitude Estimation.

    Given a circuit :math:`\\mathcal{A}`, which prepares the state
    :math:`|\\Psi\\rangle = \\mathcal{A}|0\\rangle` from the all-zero state :math:`|0\\rangle`, and
    an oracle that marks the "good" computational basis states, the circuit estimates the
    probability :math:`a = \\sin^{2}(\\theta_{a})` of measuring a good state in
    :math:`|\\Psi\\rangle`, where :math:`\\theta_{a} \\in [0, \\, \\pi / 2]`. It acts on
    :math:`t + n` qubits, with :math:`t` being the number of counting qubits and :math:`n` the
    number of qubits of :math:`\\mathcal{A}`, and it is composed of the following steps:

    1. :math:`\\mathcal{A}` on the last :math:`n` qubits of the circuit;
    2. the :func:`qibo.models.QPE` of the amplification operator :math:`\\mathbf{Q}` of
       :func:`qibo.models.amplification_operator`, with the :math:`t` first qubits as
       counting qubits.

    The state :math:`|\\Psi\\rangle` is a superposition of two eigenvectors of :math:`\\mathbf{Q}`,
    with eigenvalues :math:`e^{\\pm 2 i \\theta_{a}}`. Hence, if :math:`y` is the integer
    represented by the bitstring of the counting qubits, and :math:`M = 2^{t}`, then
    :math:`y / M` approximates :math:`\\theta_{a} / \\pi` or :math:`1 - \\theta_{a} / \\pi`, and
    the estimate of :math:`a` is :math:`\\tilde{a} = \\sin^{2}(\\pi y / M)`. The phase
    :math:`y / M` of the most probable outcome is given by
    :func:`qibo.quantum_info.estimate_phase`. With probability of at least
    :math:`8 / \\pi^{2}`,

    .. math::
        |\\tilde{a} - a| \\leq \\frac{2 \\pi \\sqrt{a (1 - a)}}{M} + \\frac{\\pi^{2}}{M^{2}} \\, .

    .. note::
        :math:`\\mathbf{Q}` is applied :math:`2^{t} - 1` times in total.

    Example:
        .. testcode::

            import math

            from qibo import Circuit, gates
            from qibo.models import QAE
            from qibo.quantum_info import estimate_phase

            # state cos(theta) |0> + sin(theta) |1> with theta = pi / 8, and good state |1>
            theta = math.pi / 8
            algorithm = Circuit(1)
            algorithm.add(gates.RY(0, theta=2 * theta))
            oracle = Circuit(1)
            oracle.add(gates.Z(0))

            # amplitude estimation with 3 counting qubits
            circuit = QAE(algorithm, oracle, 3)
            result = circuit(nshots=100)

            print(round(math.sin(math.pi * estimate_phase(result)) ** 2, 4))

        .. testoutput::

            0.1464

    Args:
        algorithm (:class:`qibo.models.circuit.Circuit`): circuit :math:`\\mathcal{A}`. It cannot
            contain measurements.
        oracle (:class:`qibo.models.circuit.Circuit`): circuit that multiplies the good states
            by :math:`-1`, *e.g.* a multi-controlled :class:`qibo.gates.Z` gate. It must act on
            the same number of qubits as ``algorithm``, and it cannot contain measurements.
        nqubits_counting (int): number of counting qubits :math:`t`, which sets the resolution
            :math:`2^{-t}` of the estimate of :math:`\\theta_{a} / \\pi`.
        measure (bool, optional): If ``True``, adds a measurement of the counting qubits at
            the end of the circuit. Defaults to ``True``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: implementation of the Quantum Amplitude
        Estimation.

    References:
        1. G. Brassard, P. Høyer, M. Mosca, and A. Tapp, *Quantum Amplitude Amplification and
        Estimation*, `Contemporary Mathematics 305, 53 (2002)
        <https://doi.org/10.1090/conm/305/05215>`_.

        2. R. Cleve, A. Ekert, C. Macchiavello, and M. Mosca, *Quantum algorithms revisited*,
        `Proc. R. Soc. Lond. A 454, 339 (1998) <https://doi.org/10.1098/rspa.1998.0164>`_.
    """
    operator = amplification_operator(algorithm, oracle)

    circuit = Circuit(nqubits_counting + algorithm.nqubits, **kwargs)
    circuit.add(
        algorithm.on_qubits(
            *range(nqubits_counting, nqubits_counting + algorithm.nqubits)
        )
    )
    circuit.add(QPE(operator, nqubits_counting, measure=measure, **kwargs))

    return circuit
