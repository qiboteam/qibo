from qibo import gates
from qibo.config import raise_error
from qibo.models.circuit import Circuit
from qibo.models.qae import QAE


def Counting(
    oracle: Circuit, nqubits_counting: int, measure: bool = True, **kwargs
) -> Circuit:
    """Creates a circuit that implements the Quantum Counting algorithm.

    Given a Boolean function :math:`\\chi` over the :math:`N = 2^{n}` computational basis states
    of :math:`n` qubits, the circuit estimates the number :math:`M` of states that
    :math:`\\chi` marks with :math:`1`. It is the :func:`qibo.models.qae.QAE` of the uniform
    superposition of the :math:`N` states, *i.e.* with Hadamard gates on every qubit as the
    circuit that prepares the state, in which the probability of measuring a marked state is
    :math:`M / N = \\sin^{2}(\\theta)`. The circuit acts on :math:`t + n` qubits, with :math:`t`
    being the number of counting qubits. If :math:`y` is the integer represented by the bitstring
    of the counting qubits, and :math:`L = 2^{t}`, then the estimate of :math:`M` is

    .. math::
        \\tilde{M} = N \\sin^{2}\\left( \\frac{\\pi y}{L} \\right) \\, .

    The phase :math:`y / L` of the most probable outcome is given by
    :func:`qibo.quantum_info.estimate_phase`. With probability of at least :math:`8 / \\pi^{2}`,

    .. math::
        |\\tilde{M} - M| \\leq \\frac{2 \\pi \\sqrt{M (N - M)}}{L} + \\frac{\\pi^{2} N}{L^{2}} \\, .

    For :math:`M = 0`, the estimate is exact, which decides if there are any marked states.

    Example:
        .. testcode::

            import math

            from qibo import Circuit, gates
            from qibo.models import Counting
            from qibo.quantum_info import estimate_phase

            # oracle that marks the states 10 and 11 of 2 qubits
            oracle = Circuit(2)
            oracle.add(gates.Z(0))

            # counting with 3 counting qubits
            circuit = Counting(oracle, 3)
            result = circuit(nshots=100)

            print(round(4 * math.sin(math.pi * estimate_phase(result)) ** 2))

        .. testoutput::

            2

    Args:
        oracle (:class:`qibo.models.circuit.Circuit`): circuit that multiplies the marked states
            by :math:`-1`, *e.g.* a multi-controlled :class:`qibo.gates.Z` gate. It cannot
            contain measurements.
        nqubits_counting (int): number of counting qubits :math:`t`, which sets the resolution
            :math:`2^{-t}` of the estimate of :math:`\\theta / \\pi`.
        measure (bool, optional): If ``True``, adds a measurement of the counting qubits at
            the end of the circuit. Defaults to ``True``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: implementation of the Quantum Counting.

    References:
        1. G. Brassard, P. Høyer, and A. Tapp, *Quantum Counting*,
        `Automata, Languages and Programming, LNCS 1443, 820 (1998)
        <https://doi.org/10.1007/BFb0055105>`_.

        2. G. Brassard, P. Høyer, M. Mosca, and A. Tapp, *Quantum Amplitude Amplification and
        Estimation*, `Contemporary Mathematics 305, 53 (2002)
        <https://doi.org/10.1090/conm/305/05215>`_.
    """
    if not isinstance(oracle, Circuit):
        raise_error(
            TypeError, f"oracle must be a Circuit, but it is type {type(oracle)}."
        )

    algorithm = Circuit(oracle.nqubits)
    algorithm.add(gates.H(qubit) for qubit in range(oracle.nqubits))

    return QAE(algorithm, oracle, nqubits_counting, measure=measure, **kwargs)
