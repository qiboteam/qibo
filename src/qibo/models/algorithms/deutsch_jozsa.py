from qibo import gates
from qibo.config import raise_error
from qibo.models.circuit import Circuit


def DeutschJozsa(oracle: Circuit, measure: bool = True, **kwargs) -> Circuit:
    """Creates a circuit that implements the Deutsch-Jozsa algorithm.

    Given a Boolean function :math:`f : \\{0, \\, 1\\}^{n} \\to \\{0, \\, 1\\}` that is promised
    to be either constant or balanced, *i.e.* equal to :math:`1` for exactly half of the inputs,
    the algorithm decides which one it is with a single query to the oracle
    :math:`U_{f} \\ket{x} \\ket{y} = \\ket{x} \\ket{y \\oplus f(x)}`, where :math:`\\oplus` is the
    sum modulo :math:`2`. The circuit acts on :math:`n + 1` qubits, with the first :math:`n` being
    the input qubits and the last one the output qubit of the oracle, and it is composed of the
    following steps:

    1. a :class:`qibo.gates.X` gate on the output qubit, and a Hadamard gate on every qubit;
    2. the oracle :math:`U_{f}`;
    3. a Hadamard gate on each of the input qubits;
    4. the measurement of the input qubits, if ``measure=True``.

    The output qubit is in the state :math:`\\ket{-}`, so that the oracle only multiplies
    :math:`\\ket{x}` by :math:`(-1)^{f(x)}`. Then, the probability of measuring the input qubits in
    the state :math:`\\ket{0}^{\\otimes \\, n}` is :math:`1` if :math:`f` is constant, and :math:`0`
    if it is balanced.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import DeutschJozsa

            # oracle of the balanced function that is equal to the first input bit
            oracle = Circuit(4)
            oracle.add(gates.CNOT(0, 3))

            circuit = DeutschJozsa(oracle)
            result = circuit(nshots=10)

            # the outcome is not 000, so the function is balanced
            print(result.frequencies())

        .. testoutput::

            Counter({'100': 10})

    Args:
        oracle (:class:`qibo.models.circuit.Circuit`): circuit that implements :math:`U_{f}`,
            which has :math:`n + 1` qubits, with the output qubit being the last one. It
            cannot contain measurements.
        measure (bool, optional): If ``True``, adds a measurement of the input qubits at
            the end of the circuit. Defaults to ``True``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: implementation of the Deutsch-Jozsa algorithm.

    References:
        1. D. Deutsch and R. Jozsa, *Rapid solution of problems by quantum computation*,
        `Proc. R. Soc. Lond. A 439, 553 (1992) <https://doi.org/10.1098/rspa.1992.0167>`_.

        2. R. Cleve, A. Ekert, C. Macchiavello, and M. Mosca, *Quantum algorithms revisited*,
        `Proc. R. Soc. Lond. A 454, 339 (1998) <https://doi.org/10.1098/rspa.1998.0164>`_.
    """
    if not isinstance(oracle, Circuit):
        raise_error(
            TypeError, f"oracle must be a Circuit, but it is type {type(oracle)}."
        )

    if oracle.nqubits < 2:
        raise_error(
            ValueError,
            f"oracle must have at least 2 qubits, but it has {oracle.nqubits}.",
        )

    if len(oracle.measurements) > 0:
        raise_error(ValueError, "oracle cannot contain measurements.")

    nqubits = oracle.nqubits
    input_qubits = list(range(nqubits - 1))

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(gates.X(nqubits - 1))
    circuit.add(gates.H(qubit) for qubit in range(nqubits))
    circuit.add(oracle)
    circuit.add(gates.H(qubit) for qubit in input_qubits)

    if measure:
        circuit.add(gates.M(*input_qubits))

    return circuit
