import cmath
import math

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import log, raise_error
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.models.algorithms.lcu import lcu_circuit, lcu_select
from qibo.models.algorithms.qaa import _check_circuits
from qibo.models.circuit import Circuit


def hadamard_test(
    state_1: Circuit,
    state_2: Circuit,
    observable: (
        Circuit
        | Hamiltonian
        | SymbolicHamiltonian
        | list[tuple[str, complex]]
        | tuple[tuple[str, complex], ...]
        | None
    ) = None,
    imaginary: bool = False,
    measure: bool = True,
    method: str = "block_encoding",
    backend: Backend = None,
    **kwargs,
) -> tuple[Circuit, float] | tuple[list[Circuit], list[float]]:
    """Creates the circuits that implement the Hadamard test of a transition amplitude.

    Let :math:`\\ket{\\psi_{1}}` and :math:`\\ket{\\psi_{2}}` be the states of :math:`n` qubits
    that ``state_1`` and ``state_2`` prepare from the all-zero state. The circuits estimate the
    real or the imaginary part of :math:`\\bra{\\psi_{2}} O \\ket{\\psi_{1}}`, which for
    ``observable=None`` is the overlap of the states, *i.e.* :math:`O` is the identity. The
    operator :math:`O` is given by ``observable``, which is either

    - a circuit that implements a unitary :math:`O`;
    - a list or a tuple of tuples ``(pauli, coefficient)``, with the Pauli string ``pauli`` and
      its complex coefficient :math:`c_{k}`, so that :math:`O = \\sum_{k} c_{k} P_{k}`. The
      string has :math:`n` letters from ``"IXYZ"``, and its :math:`j`-th letter is the Pauli
      matrix applied to the :math:`j`-th qubit of the states, *e.g.* ``"XIZ"`` is
      :math:`X_{0} Z_{2}`;
    - a :class:`qibo.hamiltonians.Hamiltonian` or a
      :class:`qibo.hamiltonians.SymbolicHamiltonian`, which is also written as a sum of Pauli
      strings. A symbolic Hamiltonian made of Pauli symbols gives its strings directly.
      Otherwise, they are computed from the matrix of the Hamiltonian with
      :func:`qibo.quantum_info.pauli_basis`, which is inefficient in the number of qubits, and a
      warning is logged.

    A sum of Pauli strings is estimated with one of the following ``method``:

    - ``"block_encoding"``: one circuit block-encodes :math:`O / \\alpha`, with
      :math:`\\alpha = \\sum_{k} |c_{k}|`, with :func:`qibo.models.lcu_circuit`, using auxiliary
      qubits. It is the unitary of the Hadamard test.
    - ``"per_pauli"``: one circuit for each Pauli string, which is the unitary of its Hadamard
      test, with no auxiliary qubits. The phase of :math:`c_{k}` is absorbed in the circuit.

    Each circuit acts on one qubit for the test, followed by the auxiliary qubits of the block
    encoding, if there are any, and the :math:`n` qubits of the states. It is composed of the
    following steps:

    1. a Hadamard gate on the first qubit;
    2. the oracle of :func:`qibo.models.lcu_select`, which applies the unitary :math:`U` after
       ``state_1`` if the first qubit is in :math:`\\ket{0}`, and ``state_2`` if it is in
       :math:`\\ket{1}`. It is the circuit of ``observable`` if it is a circuit, the block
       encoding for ``"block_encoding"``, and the Pauli string :math:`P_{k}` for ``"per_pauli"``;
    3. a :class:`qibo.gates.U1` gate on the first qubit, with the angle
       :math:`\\theta = \\pi / 2 - \\arg c_{k}` if ``imaginary=True``, and
       :math:`\\theta = - \\arg c_{k}` otherwise, where :math:`\\arg c_{k}` is the phase of the
       coefficient of the string, which is zero if there is no coefficient or the method is
       ``"block_encoding"``. It is skipped if :math:`\\theta = 0`;
    4. a Hadamard gate on the first qubit;
    5. the measurement of the first qubit, if ``measure=True``.

    This is the :func:`qibo.models.lcu_circuit` of the sum of the two unitaries, with the
    Hadamard gates as the :math:`\\mathrm{PREPARE}` oracles. Therefore, the probability of
    measuring the first qubit in :math:`\\ket{0}` of the :math:`k`-th circuit is

    .. math::
        P_{k}(0) = \\frac{1}{2} \\left( 1 + \\frac{x_{k}}{w_{k}} \\right), \\quad x_{k} =
        \\begin{cases}
        \\mathrm{Re} \\, \\bra{\\psi_{2}} O_{k} \\ket{\\psi_{1}} & \\text{if }
        \\texttt{imaginary=False} \\\\
        \\mathrm{Im} \\, \\bra{\\psi_{2}} O_{k} \\ket{\\psi_{1}} & \\text{if }
        \\texttt{imaginary=True}
        \\end{cases}

    where :math:`O_{k} = O` and :math:`w_{k} = 1` for a circuit or ``None``,
    :math:`O_{k} = O` and :math:`w_{k} = \\alpha` for ``"block_encoding"``, and
    :math:`O_{k} = c_{k} P_{k}` and :math:`w_{k} = |c_{k}|` for ``"per_pauli"``. The estimate is
    recombined from the outcomes as :math:`x = \\sum_{k} w_{k} (2 P_{k}(0) - 1)`. Both parts
    are obtained with two sets of circuits.

    Example:
        Estimate the overlap of :math:`\\ket{\\psi_{1}} = \\ket{+}` and
        :math:`\\ket{\\psi_{2}} = (\\ket{0} + i \\ket{1}) / \\sqrt{2}`, which is
        :math:`\\braket{\\psi_{2} | \\psi_{1}} = (1 - i) / 2`.

        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import hadamard_test

            state_1 = Circuit(1)
            state_1.add(gates.H(0))
            state_2 = Circuit(1)
            state_2.add(gates.H(0))
            state_2.add(gates.S(0))

            for imaginary in (False, True):
                circuit, alpha = hadamard_test(state_1, state_2, imaginary=imaginary)
                probability = circuit().probabilities(qubits=[0])[0]

                print(imaginary, f"{alpha * (2 * probability - 1):.2f}")

        .. testoutput::

            False 0.50
            True -0.50

        Estimate the expectation value of :math:`2 X + Z` in the state :math:`\\ket{+}`, which is
        :math:`2`, with a list of Pauli strings, and with a Hamiltonian.

        .. testcode::

            from qibo.hamiltonians import SymbolicHamiltonian
            from qibo.symbols import X, Z

            observables = [
                [("X", 2.0), ("Z", 1.0)],
                SymbolicHamiltonian(2 * X(0) + Z(0)),
            ]

            for observable in observables:
                circuit, alpha = hadamard_test(state_1, state_1, observable)
                probability = circuit().probabilities(qubits=[0])[0]

                print(f"{alpha * (2 * probability - 1):.2f}")

        .. testoutput::

            2.00
            2.00

        The same, with one circuit for each Pauli string.

        .. testcode::

            circuits, weights = hadamard_test(
                state_1, state_1, [("X", 2.0), ("Z", 1.0)], method="per_pauli"
            )
            probabilities = [circuit().probabilities(qubits=[0])[0] for circuit in circuits]

            print(f"{sum(w * (2 * p - 1) for w, p in zip(weights, probabilities)):.2f}")

        .. testoutput::

            2.00

    Args:
        state_1 (:class:`qibo.models.circuit.Circuit`): circuit that prepares
            :math:`\\ket{\\psi_{1}}`. It cannot contain measurements.
        state_2 (:class:`qibo.models.circuit.Circuit`): circuit that prepares
            :math:`\\ket{\\psi_{2}}`. It has the same number of qubits as ``state_1``, and it
            cannot contain measurements.
        observable (:class:`qibo.models.circuit.Circuit`, list, tuple or
            :class:`qibo.hamiltonians.abstract.AbstractHamiltonian`, optional): circuit that
            implements the unitary :math:`O`, list or tuple of tuples ``(pauli, coefficient)`` of
            its Pauli decomposition, or Hamiltonian that is :math:`O`, on the same number of
            qubits as ``state_1``. A circuit cannot contain measurements, and it has to be exact,
            including its global phase, since it is applied under control. If ``None``, it is the
            identity. Defaults to ``None``.
        imaginary (bool, optional): If ``True``, estimates the imaginary part of
            :math:`\\bra{\\psi_{2}} O \\ket{\\psi_{1}}`, and the real part otherwise. Defaults to
            ``False``.
        measure (bool, optional): If ``True``, adds a measurement of the first qubit at the end
            of each circuit. Defaults to ``True``.
        method (str, optional): either ``"block_encoding"``, with one circuit, or ``"per_pauli"``,
            with one circuit for each Pauli string. It only changes the output if ``observable``
            is a circuit or ``None``, for which ``"per_pauli"`` returns one circuit.
            Defaults to ``"block_encoding"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        tuple: for ``"block_encoding"``, the circuit of the Hadamard test and the weight
        :math:`w`, which is :math:`\\alpha`, or :math:`1` if ``observable`` is a circuit or
        ``None``. For ``"per_pauli"``, the list of circuits and the list of their weights
        :math:`w_{k}`, which are the :math:`|c_{k}|`, or :math:`[1]` if ``observable`` is a
        circuit or ``None``.

    References:
        1. D. Aharonov, V. Jones, and Z. Landau, *A polynomial quantum algorithm for
        approximating the Jones polynomial*, `Algorithmica 55, 395-421 (2009)
        <https://doi.org/10.1007/s00453-008-9168-0>`_.

        2. R. Cleve, A. Ekert, C. Macchiavello, and M. Mosca, *Quantum algorithms revisited*,
        `Proc. R. Soc. Lond. A 454, 339 (1998) <https://doi.org/10.1098/rspa.1998.0164>`_.
    """
    backend = _check_backend(backend)

    if method not in ("block_encoding", "per_pauli"):
        raise_error(
            ValueError,
            f"``method`` must be 'block_encoding' or 'per_pauli', but it is '{method}'.",
        )

    circuits = {"state_1": state_1, "state_2": state_2}
    is_hamiltonian = isinstance(observable, (Hamiltonian, SymbolicHamiltonian))
    if isinstance(observable, Circuit):
        circuits["observable"] = observable
    elif observable is not None and not (
        is_hamiltonian or isinstance(observable, (list, tuple))
    ):
        raise_error(
            TypeError,
            "observable must be a Circuit, a list, a tuple, a Hamiltonian or a "
            + f"SymbolicHamiltonian, but it is type {type(observable)}.",
        )
    nqubits = _check_circuits(**circuits)

    if is_hamiltonian and observable.nqubits != nqubits:
        raise_error(
            ValueError,
            f"observable must have {nqubits} qubits, but it has {observable.nqubits}.",
        )

    terms = None
    if isinstance(observable, (list, tuple)):
        for term in observable:
            if not (
                isinstance(term, (list, tuple))
                and len(term) == 2
                and isinstance(term[0], str)
            ):
                raise_error(
                    TypeError,
                    "the elements of observable must be tuples of a Pauli string and "
                    + f"a coefficient, but one is {term}.",
                )
            if len(term[0]) != nqubits or not set(term[0]) <= set("IXYZ"):
                raise_error(
                    ValueError,
                    f"the Pauli string '{term[0]}' must have {nqubits} letters "
                    + "among 'I', 'X', 'Y' and 'Z'.",
                )
        terms = list(observable)
    elif isinstance(observable, SymbolicHamiltonian) and all(
        set(name) <= set("XYZ") for name in observable.simple_terms[1]
    ):
        coefficients, names, qubits = observable.simple_terms
        terms = []
        for coefficient, name, targets in zip(coefficients, names, qubits):
            pauli = ["I"] * nqubits
            for letter, qubit in zip(name, targets):
                pauli[qubit] = letter
            terms.append(("".join(pauli), coefficient))
        if observable.constant != 0.0:
            terms.append(("I" * nqubits, observable.constant))
    elif is_hamiltonian:
        from qibo.quantum_info import pauli_basis  # circular import

        log.warning(
            "Calculating the Pauli decomposition of the matrix of a Hamiltonian. "
            "This operation is inefficient in the number of qubits."
        )
        # coefficients of the Pauli strings, whose index in base four has the Pauli
        # matrix I, X, Y, Z of the first qubit as its most significant digit
        basis = pauli_basis(nqubits, vectorize=True, order="row", backend=backend)
        matrix = backend.cast(observable.matrix, dtype="complex128")
        coefficients = backend.matmul(
            backend.conj(basis), backend.reshape(matrix, (-1,))
        ) / (2**nqubits)
        magnitudes = backend.abs(coefficients)
        terms = [
            (
                "".join(
                    "IXYZ"[(int(index) // 4 ** (nqubits - 1 - qubit)) % 4]
                    for qubit in range(nqubits)
                ),
                complex(coefficients[index]),
            )
            for index in backend.nonzero(magnitudes > 1e-12 * backend.max(magnitudes))[
                0
            ]
        ]

    if terms is not None and len(terms) == 0:
        raise_error(ValueError, "observable must have at least one Pauli string.")

    operators, weights, phases, nauxiliary = [observable], [1.0], [0.0], 0
    if terms is not None:
        unitaries = []
        for pauli, _ in terms:
            unitary = Circuit(nqubits)
            unitary.add(
                getattr(gates, letter)(qubit)
                for qubit, letter in enumerate(pauli)
                if letter != "I"
            )
            unitaries.append(unitary)

        if method == "block_encoding":
            block, alpha, nauxiliary = lcu_circuit(
                unitaries, [coefficient for _, coefficient in terms], backend=backend
            )
            operators, weights = [block], [float(alpha)]
        else:
            coefficients = [complex(coefficient) for _, coefficient in terms]
            operators = unitaries
            weights = [abs(coefficient) for coefficient in coefficients]
            phases = [-cmath.phase(coefficient) for coefficient in coefficients]

    system = range(nauxiliary, nauxiliary + nqubits)
    circuits = []
    for operator, phase in zip(operators, phases):
        first = Circuit(nauxiliary + nqubits)
        first.add(state_1.on_qubits(*system))
        if operator is not None:
            first.add(operator.on_qubits(*range(nauxiliary + nqubits)))
        second = Circuit(nauxiliary + nqubits)
        second.add(state_2.on_qubits(*system))
        select = lcu_select([first, second], 1, backend=backend)

        circuit = Circuit(nauxiliary + nqubits + 1, **kwargs)
        circuit.add(gates.H(0))
        circuit.add(select.on_qubits(*range(circuit.nqubits)))
        phase += math.pi / 2 * int(imaginary)
        if phase != 0.0:
            circuit.add(gates.U1(0, phase))
        circuit.add(gates.H(0))

        if measure:
            circuit.add(gates.M(0))

        circuits.append(circuit)

    if method == "block_encoding":
        return circuits[0], weights[0]

    return circuits, weights


def swap_test(
    state_1: Circuit, state_2: Circuit, measure: bool = True, **kwargs
) -> Circuit:
    """Creates a circuit that implements the swap test of two states.

    Let :math:`\\ket{\\psi_{1}}` and :math:`\\ket{\\psi_{2}}` be the states of :math:`n` qubits
    that ``state_1`` and ``state_2`` prepare from the all-zero state. The circuit estimates the
    squared overlap :math:`|\\braket{\\psi_{2} | \\psi_{1}}|^{2}`. It acts on one auxiliary
    qubit, followed by the :math:`n` qubits of :math:`\\ket{\\psi_{1}}` and the :math:`n` qubits
    of :math:`\\ket{\\psi_{2}}`, and it is composed of the following steps:

    1. a Hadamard gate on the auxiliary qubit, and the preparation of the states;
    2. a :class:`qibo.gates.SWAP` gate between each qubit of :math:`\\ket{\\psi_{1}}` and the
       corresponding one of :math:`\\ket{\\psi_{2}}`, under the control of the auxiliary qubit;
    3. a Hadamard gate on the auxiliary qubit;
    4. the measurement of the auxiliary qubit, if ``measure=True``.

    The probability of measuring the auxiliary qubit in :math:`\\ket{0}` is

    .. math::
        P(0) = \\frac{1}{2} \\left( 1 + |\\braket{\\psi_{2} | \\psi_{1}}|^{2} \\right) \\, ,

    and :math:`|\\braket{\\psi_{2} | \\psi_{1}}|^{2} = 2 P(0) - 1`. Unlike the Hadamard test of
    :func:`qibo.models.hadamard_test`, it does not depend on the phase of the overlap, and it
    does not require the inverse of the circuits.

    Example:
        Estimate the squared overlap of :math:`\\ket{0}` and :math:`\\ket{+}`, which is
        :math:`1 / 2`.

        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import swap_test

            state_1 = Circuit(1)
            state_2 = Circuit(1)
            state_2.add(gates.H(0))

            circuit = swap_test(state_1, state_2)
            probability = circuit().probabilities(qubits=[0])[0]

            print(f"{2 * probability - 1:.2f}")

        .. testoutput::

            0.50

    Args:
        state_1 (:class:`qibo.models.circuit.Circuit`): circuit that prepares
            :math:`\\ket{\\psi_{1}}`. It cannot contain measurements.
        state_2 (:class:`qibo.models.circuit.Circuit`): circuit that prepares
            :math:`\\ket{\\psi_{2}}`. It has the same number of qubits as ``state_1``, and it
            cannot contain measurements.
        measure (bool, optional): If ``True``, adds a measurement of the auxiliary qubit at the
            end of the circuit. Defaults to ``True``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: circuit of the swap test, with the auxiliary
        qubit at the first qubit.

    References:
        1. H. Buhrman, R. Cleve, J. Watrous, and R. de Wolf, *Quantum fingerprinting*,
        `Phys. Rev. Lett. 87, 167902 (2001) <https://doi.org/10.1103/PhysRevLett.87.167902>`_.
    """
    nqubits = _check_circuits(state_1=state_1, state_2=state_2)

    circuit = Circuit(2 * nqubits + 1, **kwargs)
    circuit.add(gates.H(0))
    circuit.add(state_1.on_qubits(*range(1, nqubits + 1)))
    circuit.add(state_2.on_qubits(*range(nqubits + 1, 2 * nqubits + 1)))
    circuit.add(
        gates.SWAP(qubit, qubit + nqubits).controlled_by(0)
        for qubit in range(1, nqubits + 1)
    )
    circuit.add(gates.H(0))

    if measure:
        circuit.add(gates.M(0))

    return circuit
