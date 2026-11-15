from collections.abc import Iterable, Mapping

from qibo import gates
from qibo.config import raise_error
from qibo.models.circuit import Circuit
from qibo.result import MeasurementOutcomes


def Simon(oracle: Circuit, measure: bool = True, **kwargs) -> Circuit:
    """Creates a circuit that implements the quantum subroutine of Simon's algorithm.

    Given a function :math:`f : \\{0, \\, 1\\}^{n} \\to \\{0, \\, 1\\}^{n}` that is promised
    to satisfy :math:`f(x) = f(x')` if and only if :math:`x' \\in \\{x, \\, x \\oplus s\\}`
    for a hidden string :math:`s \\in \\{0, \\, 1\\}^{n}`, where :math:`\\oplus` is the bitwise sum
    modulo :math:`2`, the circuit samples bitstrings :math:`y` that satisfy
    :math:`y \\cdot s = 0 \\pmod 2`. The oracle is :math:`U_{f} \\ket{x} \\ket{z} =
    \\ket{x} \\ket{z \\oplus f(x)}`. The circuit acts on :math:`2n` qubits, with the first
    :math:`n` being the input qubits and the last :math:`n` the output qubits of the oracle, and
    it is composed of the following steps:

    1. a Hadamard gate on each of the input qubits;
    2. the oracle :math:`U_{f}`;
    3. a Hadamard gate on each of the input qubits;
    4. the measurement of the input qubits, if ``measure=True``.

    The measured bitstrings are uniformly distributed over the :math:`2^{n - 1}` strings
    orthogonal to :math:`s` if :math:`f` is two-to-one, and over all the :math:`2^{n}` strings
    if :math:`f` is one-to-one, with the qubit :math:`k` of the bitstring being the bit
    :math:`y_{k}`. Therefore, :math:`s` is found, with high probability, by collecting
    :math:`O(n)` samples and solving the linear system :math:`y \\cdot s = 0` over
    :math:`\\mathbb{F}_{2}`, which is done by :func:`qibo.models.simon_secret_string`.
    A nonzero solution is only a candidate: if :math:`f` is one-to-one, :math:`n - 1` linearly
    independent bitstrings also yield a nonzero solution. It is verified by checking whether
    :math:`f(0 \\dots 0) = f(s)`, with two evaluations of the function.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import Simon, simon_secret_string

            # oracle of f(x) = (x_0 XOR x_1, x_2, 0), with the hidden string s = 110
            oracle = Circuit(6)
            oracle.add(gates.CNOT(0, 3))
            oracle.add(gates.CNOT(1, 3))
            oracle.add(gates.CNOT(2, 4))

            circuit = Simon(oracle)
            result = circuit(nshots=100)

            print(simon_secret_string(result, 3))

        .. testoutput::

            110

    Args:
        oracle (:class:`qibo.models.circuit.Circuit`): circuit that implements :math:`U_{f}`,
            which has :math:`2n` qubits, with the output qubits being the last :math:`n` ones.
            It cannot contain measurements.
        measure (bool, optional): If ``True``, adds a measurement of the input qubits at
            the end of the circuit. Defaults to ``True``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: implementation of the quantum subroutine of
        Simon's algorithm.

    References:
        1. D. R. Simon, *On the power of quantum computation*,
        `SIAM J. Comput. 26, 1474 (1997) <https://doi.org/10.1137/S0097539796298637>`_.

        2. R. Cleve, A. Ekert, C. Macchiavello, and M. Mosca, *Quantum algorithms revisited*,
        `Proc. R. Soc. Lond. A 454, 339 (1998) <https://doi.org/10.1098/rspa.1998.0164>`_.
    """
    if not isinstance(oracle, Circuit):
        raise_error(
            TypeError, f"oracle must be a Circuit, but it is type {type(oracle)}."
        )

    if oracle.nqubits < 2 or oracle.nqubits % 2 != 0:
        raise_error(
            ValueError,
            "oracle must have an even number of qubits, equal to or larger than 2, "
            f"but it has {oracle.nqubits}.",
        )

    if len(oracle.measurements) > 0:
        raise_error(ValueError, "oracle cannot contain measurements.")

    nqubits = oracle.nqubits
    input_qubits = list(range(nqubits // 2))

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(gates.H(qubit) for qubit in input_qubits)
    circuit.add(oracle)
    circuit.add(gates.H(qubit) for qubit in input_qubits)

    if measure:
        circuit.add(gates.M(*input_qubits))

    return circuit


def simon_secret_string(
    outcomes: MeasurementOutcomes | Mapping | Iterable, nbits: int | None = None
) -> str:
    """Finds the hidden string of Simon's algorithm from the measured bitstrings.

    Each measured bitstring :math:`y` of the circuit created by :func:`qibo.models.Simon`
    satisfies :math:`y \\cdot s = 0 \\pmod 2`. This function solves such linear system over
    the finite field :math:`\\mathbb{F}_{2}` with Gaussian elimination. If the equations
    only have the trivial solution, then :math:`n` linearly independent bitstrings were found,
    and the function is one-to-one, with :math:`s = 0 \\dots 0`. Otherwise, the nonzero solution
    is returned, which is the hidden string if the function is two-to-one. However, if the
    function is one-to-one and only :math:`n - 1` linearly independent bitstrings were found,
    it is a spurious string, and it must be verified by checking whether
    :math:`f(0 \\dots 0) = f(s)`. If there are fewer than :math:`n - 1` linearly independent
    bitstrings, the solution is not unique, and an error is raised.

    Args:
        outcomes (:class:`qibo.result.MeasurementOutcomes` or dict or list): measured
            bitstrings. It can be the :class:`qibo.result.MeasurementOutcomes` (or any object
            with a ``frequencies`` method, *e.g.* :class:`qibo.measurements.MeasurementResult`)
            returned by the circuit, a dictionary of frequencies (*e.g.* ``{"01": 3, "10": 5}``),
            or an iterable of bitstrings (strings of ``"0"`` and ``"1"``, or sequences of bits).
        nbits (int, optional): number of bits :math:`n` of the hidden string. If ``None``, it
            is inferred from the length of the bitstrings. Defaults to ``None``.

    Returns:
        str: hidden string, in which the character :math:`k` is the bit :math:`s_{k}`.
    """
    if hasattr(outcomes, "frequencies"):
        outcomes = outcomes.frequencies()
    if isinstance(outcomes, Mapping):
        outcomes = list(outcomes)
    elif isinstance(outcomes, str):
        raise_error(TypeError, "outcomes must be a collection of bitstrings.")

    strings = []
    for outcome in outcomes:
        if isinstance(outcome, str):
            bits = outcome
        else:
            bits = "".join(str(int(bit)) for bit in outcome)
        if bits == "" or set(bits) - {"0", "1"}:
            raise_error(ValueError, f"Invalid bitstring {outcome!r}.")
        strings.append(bits)

    if len(strings) == 0:
        raise_error(ValueError, "outcomes cannot be empty.")

    if nbits is None:
        nbits = len(strings[0])
    if nbits < 1:
        raise_error(ValueError, f"nbits must be positive, but it is {nbits}.")
    if any(len(bits) != nbits for bits in strings):
        raise_error(
            ValueError, f"All the bitstrings must have {nbits} bits, the hidden string."
        )

    # Gaussian elimination over GF(2), with the bitstrings stored as integers in which
    # the bit ``k`` is the character ``k``
    pivots = {}  # pivot column -> row, in reduced form
    for bits in strings:
        row = sum(1 << k for k, bit in enumerate(bits) if bit == "1")
        for column, pivot_row in pivots.items():
            if (row >> column) & 1:
                row ^= pivot_row
        if row == 0:
            continue
        column = (row & -row).bit_length() - 1
        for other, pivot_row in pivots.items():
            if (pivot_row >> column) & 1:
                pivots[other] = pivot_row ^ row
        pivots[column] = row

    # basis of the nonzero solutions, with one string per free column of the reduced form
    solutions = []
    for free in (k for k in range(nbits) if k not in pivots):
        solution = 1 << free
        for column, row in pivots.items():
            if (row >> free) & 1:
                solution |= 1 << column
        solutions.append("".join(str((solution >> k) & 1) for k in range(nbits)))

    if len(solutions) > 1:
        raise_error(
            ValueError,
            "The hidden string is not unique: the bitstrings have rank "
            f"{nbits - len(solutions)}, but at least {nbits - 1} are needed.",
        )

    return solutions[0] if solutions else "0" * nbits
