from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import log, raise_error
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.models.algorithms.block_encoding import block_encoding_circuit
from qibo.models.algorithms.lcu import lcu_circuit, lcu_select
from qibo.models.algorithms.qaa import _check_circuits
from qibo.models.circuit import Circuit


def hadamard_test(
    state_1: Circuit,
    state_2: Circuit,
    observable: Circuit | Hamiltonian | SymbolicHamiltonian | None = None,
    part: str = "real",
    measure: bool = True,
    backend: Backend = None,
    **kwargs,
) -> tuple[Circuit, float]:
    """Creates a circuit that implements the Hadamard test of a transition amplitude.

    Let :math:`\\ket{\\psi_{1}}` and :math:`\\ket{\\psi_{2}}` be the states of :math:`n` qubits
    that ``state_1`` and ``state_2`` prepare from the all-zero state. The circuit estimates the
    real or the imaginary part of :math:`\\bra{\\psi_{2}} O \\ket{\\psi_{1}}`, which for
    ``observable=None`` is the overlap of the states, *i.e.* :math:`O` is the identity. The
    operator :math:`O` is given by ``observable``, which is either

    - a circuit that implements a unitary :math:`O`;
    - a :class:`qibo.hamiltonians.Hamiltonian` or a
      :class:`qibo.hamiltonians.SymbolicHamiltonian`, which is written as a sum of Pauli strings,
      :math:`O = \\sum_{i} c_{i} P_{i}`. A symbolic Hamiltonian made of Pauli symbols gives its
      strings directly. Otherwise, they are computed from the matrix of the Hamiltonian with
      :func:`qibo.quantum_info.pauli_basis`, which is inefficient in the number of qubits, and a
      warning is logged. The circuit block-encodes :math:`O / \\alpha`, with
      :math:`\\alpha = \\sum_{i} |c_{i}|`, with :func:`qibo.models.lcu_circuit`.

    The circuit acts on one qubit for the test, followed by the auxiliary qubits of the block
    encoding, if there are any, and the :math:`n` qubits of the states. It is composed of the
    following steps:

    1. a Hadamard gate on the first qubit;
    2. the oracle of :func:`qibo.models.lcu_select`, which applies :math:`O` after ``state_1``
       if the first qubit is in :math:`\\ket{0}`, and ``state_2`` if it is in :math:`\\ket{1}`;
    3. a :class:`qibo.gates.S` gate on the first qubit, if ``part="imag"``;
    4. a Hadamard gate on the first qubit;
    5. the measurement of the first qubit, if ``measure=True``.

    This is the :func:`qibo.models.lcu_circuit` of the sum of the two unitaries, with the
    Hadamard gates as the :math:`\\mathrm{PREPARE}` oracles. Therefore, the probability of
    measuring the first qubit in :math:`\\ket{0}` is

    .. math::
        P(0) = \\frac{1}{2} \\left( 1 + \\frac{x}{\\alpha} \\right), \\quad x =
        \\begin{cases}
        \\mathrm{Re} \\, \\bra{\\psi_{2}} O \\ket{\\psi_{1}} & \\text{if } \\texttt{part="real"}
        \\\\
        \\mathrm{Im} \\, \\bra{\\psi_{2}} O \\ket{\\psi_{1}} & \\text{if } \\texttt{part="imag"}
        \\end{cases}

    and :math:`x = \\alpha (2 P(0) - 1)`, where :math:`\\alpha = 1` unless ``observable`` is a
    Hamiltonian. Both parts are obtained with two circuits.

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

            for part in ("real", "imag"):
                circuit, alpha = hadamard_test(state_1, state_2, part=part)
                probability = circuit().probabilities(qubits=[0])[0]

                print(part, f"{alpha * (2 * probability - 1):.2f}")

        .. testoutput::

            real 0.50
            imag -0.50

        Estimate the expectation value of :math:`2 X + Z` in the state :math:`\\ket{+}`, which is
        :math:`2`.

        .. testcode::

            from qibo.hamiltonians import SymbolicHamiltonian
            from qibo.symbols import X, Z

            hamiltonian = SymbolicHamiltonian(2 * X(0) + Z(0))

            circuit, alpha = hadamard_test(state_1, state_1, hamiltonian)
            probability = circuit().probabilities(qubits=[0])[0]

            print(f"{alpha * (2 * probability - 1):.2f}")

        .. testoutput::

            2.00

    Args:
        state_1 (:class:`qibo.models.circuit.Circuit`): circuit that prepares
            :math:`\\ket{\\psi_{1}}`. It cannot contain measurements.
        state_2 (:class:`qibo.models.circuit.Circuit`): circuit that prepares
            :math:`\\ket{\\psi_{2}}`. It has the same number of qubits as ``state_1``, and it
            cannot contain measurements.
        observable (:class:`qibo.models.circuit.Circuit` or
            :class:`qibo.hamiltonians.abstract.AbstractHamiltonian`, optional): circuit that
            implements the unitary :math:`O`, or Hamiltonian that is :math:`O`, on the same
            number of qubits as ``state_1``. A circuit cannot contain measurements, and it has to
            be exact, including its global phase, since it is applied under control. If ``None``,
            it is the identity. Defaults to ``None``.
        part (str, optional): part of :math:`\\bra{\\psi_{2}} O \\ket{\\psi_{1}}` that is
            estimated, either ``"real"`` or ``"imag"``. Defaults to ``"real"``.
        measure (bool, optional): If ``True``, adds a measurement of the first qubit at the end
            of the circuit. Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        tuple: circuit of the Hadamard test, and the normalization :math:`\\alpha`, which is
        :math:`1` if ``observable`` is not a Hamiltonian.

    References:
        1. D. Aharonov, V. Jones, and Z. Landau, *A polynomial quantum algorithm for
        approximating the Jones polynomial*, `Algorithmica 55, 395-421 (2009)
        <https://doi.org/10.1007/s00453-008-9168-0>`_.

        2. R. Cleve, A. Ekert, C. Macchiavello, and M. Mosca, *Quantum algorithms revisited*,
        `Proc. R. Soc. Lond. A 454, 339 (1998) <https://doi.org/10.1098/rspa.1998.0164>`_.
    """
    backend = _check_backend(backend)

    circuits = {"state_1": state_1, "state_2": state_2}
    is_hamiltonian = isinstance(observable, (Hamiltonian, SymbolicHamiltonian))
    if isinstance(observable, Circuit):
        circuits["observable"] = observable
    elif observable is not None and not is_hamiltonian:
        raise_error(
            TypeError,
            "observable must be a Circuit, a Hamiltonian or a SymbolicHamiltonian, "
            + f"but it is type {type(observable)}.",
        )
    nqubits = _check_circuits(**circuits)

    if is_hamiltonian and observable.nqubits != nqubits:
        raise_error(
            ValueError,
            f"observable must have {nqubits} qubits, but it has {observable.nqubits}.",
        )

    if part not in ("real", "imag"):
        raise_error(
            ValueError, f"``part`` must be 'real' or 'imag', but it is '{part}'."
        )

    alpha, nauxiliary = 1.0, 0
    if isinstance(observable, SymbolicHamiltonian) and all(
        set(name) <= set("XYZ") for name in observable.simple_terms[1]
    ):
        coefficients, names, qubits = observable.simple_terms
        coefficients = list(coefficients)
        unitaries = []
        for name, targets in zip(names, qubits):
            unitary = Circuit(nqubits)
            unitary.add(
                getattr(gates, pauli)(qubit) for pauli, qubit in zip(name, targets)
            )
            unitaries.append(unitary)
        if observable.constant != 0.0:
            coefficients.append(observable.constant)
            unitaries.append(Circuit(nqubits))
        block, alpha, nauxiliary = lcu_circuit(unitaries, coefficients, backend=backend)
    elif is_hamiltonian:
        log.warning(
            "Calculating the Pauli decomposition of the matrix of a Hamiltonian. "
            "This operation is inefficient in the number of qubits."
        )
        block, alpha, nauxiliary = block_encoding_circuit(
            observable.matrix, method="lcu", backend=backend
        )

    system = range(nauxiliary, nauxiliary + nqubits)

    first = Circuit(nauxiliary + nqubits)
    first.add(state_1.on_qubits(*system))
    if isinstance(observable, Circuit):
        first.add(observable.on_qubits(*system))
    elif is_hamiltonian:
        first.add(block)
    second = Circuit(nauxiliary + nqubits)
    second.add(state_2.on_qubits(*system))
    select = lcu_select([first, second], 1, backend=backend)

    circuit = Circuit(nauxiliary + nqubits + 1, **kwargs)
    circuit.add(gates.H(0))
    circuit.add(select.on_qubits(*range(circuit.nqubits)))
    if part == "imag":
        circuit.add(gates.S(0))
    circuit.add(gates.H(0))

    if measure:
        circuit.add(gates.M(0))

    return circuit, float(alpha)


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
