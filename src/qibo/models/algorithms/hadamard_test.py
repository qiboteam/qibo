from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.models.algorithms.lcu import lcu_select
from qibo.models.algorithms.qaa import _check_circuits
from qibo.models.circuit import Circuit


def hadamard_test(
    phi: Circuit,
    psi: Circuit,
    observable: Circuit | None = None,
    part: str = "real",
    measure: bool = True,
    backend: Backend = None,
    **kwargs,
) -> Circuit:
    """Creates a circuit that implements the Hadamard test of a transition amplitude.

    Let :math:`\\ket{\\psi}` and :math:`\\ket{\\varphi}` be the states of :math:`n` qubits that
    ``psi`` and ``phi`` prepare from the all-zero state, and let :math:`O` be the unitary of
    ``observable``, which is the identity if ``observable=None``. The circuit estimates the real
    or the imaginary part of :math:`\\bra{\\varphi} O \\ket{\\psi}`, which for
    ``observable=None`` is the overlap of the states. It acts on one auxiliary qubit, followed by
    the :math:`n` qubits of the states, and it is composed of the following steps:

    1. a Hadamard gate on the auxiliary qubit;
    2. the oracle of :func:`qibo.models.lcu_select`, which applies :math:`O` after ``psi`` if the
       auxiliary qubit is in :math:`\\ket{0}`, and ``phi`` if it is in :math:`\\ket{1}`;
    3. a :class:`qibo.gates.S` gate on the auxiliary qubit, if ``part="imag"``;
    4. a Hadamard gate on the auxiliary qubit;
    5. the measurement of the auxiliary qubit, if ``measure=True``.

    This is the :func:`qibo.models.lcu_circuit` of the sum of the two unitaries, with the
    Hadamard gates as the :math:`\\mathrm{PREPARE}` oracles. Therefore, the probability of
    measuring the auxiliary qubit in :math:`\\ket{0}` is

    .. math::
        P(0) = \\frac{1}{2} \\left( 1 + x \\right), \\quad x =
        \\begin{cases}
        \\mathrm{Re} \\, \\bra{\\varphi} O \\ket{\\psi} & \\text{if } \\texttt{part="real"} \\\\
        \\mathrm{Im} \\, \\bra{\\varphi} O \\ket{\\psi} & \\text{if } \\texttt{part="imag"}
        \\end{cases}

    and :math:`x = 2 P(0) - 1`. Both parts are obtained with two circuits. A non-unitary
    operator is estimated by linearity from the unitaries of its decomposition, *e.g.* its Pauli
    strings.

    Example:
        Estimate the overlap of :math:`\\ket{\\psi} = \\ket{+}` and
        :math:`\\ket{\\varphi} = (\\ket{0} + i \\ket{1}) / \\sqrt{2}`, which is
        :math:`\\braket{\\varphi | \\psi} = (1 - i) / 2`.

        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import hadamard_test

            psi = Circuit(1)
            psi.add(gates.H(0))
            phi = Circuit(1)
            phi.add(gates.H(0))
            phi.add(gates.S(0))

            for part in ("real", "imag"):
                circuit = hadamard_test(phi, psi, part=part)
                probability = circuit().probabilities(qubits=[0])[0]

                print(part, f"{2 * probability - 1:.2f}")

        .. testoutput::

            real 0.50
            imag -0.50

    Args:
        phi (:class:`qibo.models.circuit.Circuit`): circuit that prepares
            :math:`\\ket{\\varphi}`. It cannot contain measurements.
        psi (:class:`qibo.models.circuit.Circuit`): circuit that prepares
            :math:`\\ket{\\psi}`. It has the same number of qubits as ``phi``, and it cannot
            contain measurements.
        observable (:class:`qibo.models.circuit.Circuit`, optional): circuit that implements
            the unitary :math:`O`, on the same number of qubits as ``phi``. It cannot contain
            measurements. It has to be exact, including its global phase, since it is applied
            under control. If ``None``, it is the identity. Defaults to ``None``.
        part (str, optional): part of :math:`\\bra{\\varphi} O \\ket{\\psi}` that is estimated,
            either ``"real"`` or ``"imag"``. Defaults to ``"real"``.
        measure (bool, optional): If ``True``, adds a measurement of the auxiliary qubit at the
            end of the circuit. Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: circuit of the Hadamard test, with the auxiliary
        qubit at the first qubit.

    References:
        1. D. Aharonov, V. Jones, and Z. Landau, *A polynomial quantum algorithm for
        approximating the Jones polynomial*, `Algorithmica 55, 395-421 (2009)
        <https://doi.org/10.1007/s00453-008-9168-0>`_.

        2. R. Cleve, A. Ekert, C. Macchiavello, and M. Mosca, *Quantum algorithms revisited*,
        `Proc. R. Soc. Lond. A 454, 339 (1998) <https://doi.org/10.1098/rspa.1998.0164>`_.
    """
    backend = _check_backend(backend)

    circuits = {"phi": phi, "psi": psi}
    if observable is not None:
        circuits["observable"] = observable
    nqubits = _check_circuits(**circuits)

    if part not in ("real", "imag"):
        raise_error(
            ValueError, f"``part`` must be 'real' or 'imag', but it is '{part}'."
        )

    ket = Circuit(nqubits)
    ket.add(psi)
    if observable is not None:
        ket.add(observable)
    select = lcu_select([ket, phi], 1, backend=backend)

    circuit = Circuit(nqubits + 1, **kwargs)
    circuit.add(gates.H(0))
    circuit.add(select.on_qubits(*range(nqubits + 1)))
    if part == "imag":
        circuit.add(gates.S(0))
    circuit.add(gates.H(0))

    if measure:
        circuit.add(gates.M(0))

    return circuit


def swap_test(phi: Circuit, psi: Circuit, measure: bool = True, **kwargs) -> Circuit:
    """Creates a circuit that implements the swap test of two states.

    Let :math:`\\ket{\\psi}` and :math:`\\ket{\\varphi}` be the states of :math:`n` qubits that
    ``psi`` and ``phi`` prepare from the all-zero state. The circuit estimates the squared
    overlap :math:`|\\braket{\\varphi | \\psi}|^{2}`. It acts on one auxiliary qubit, followed by
    the :math:`n` qubits of :math:`\\ket{\\psi}` and the :math:`n` qubits of
    :math:`\\ket{\\varphi}`, and it is composed of the following steps:

    1. a Hadamard gate on the auxiliary qubit, and the preparation of the states;
    2. a :class:`qibo.gates.SWAP` gate between each qubit of :math:`\\ket{\\psi}` and the
       corresponding one of :math:`\\ket{\\varphi}`, under the control of the auxiliary qubit;
    3. a Hadamard gate on the auxiliary qubit;
    4. the measurement of the auxiliary qubit, if ``measure=True``.

    The probability of measuring the auxiliary qubit in :math:`\\ket{0}` is

    .. math::
        P(0) = \\frac{1}{2} \\left( 1 + |\\braket{\\varphi | \\psi}|^{2} \\right) \\, ,

    and :math:`|\\braket{\\varphi | \\psi}|^{2} = 2 P(0) - 1`. Unlike the Hadamard test of
    :func:`qibo.models.hadamard_test`, it does not depend on the phase of the overlap, and it
    does not require the inverse of the circuits.

    Example:
        Estimate the squared overlap of :math:`\\ket{0}` and :math:`\\ket{+}`, which is
        :math:`1 / 2`.

        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import swap_test

            psi = Circuit(1)
            phi = Circuit(1)
            phi.add(gates.H(0))

            circuit = swap_test(phi, psi)
            probability = circuit().probabilities(qubits=[0])[0]

            print(f"{2 * probability - 1:.2f}")

        .. testoutput::

            0.50

    Args:
        phi (:class:`qibo.models.circuit.Circuit`): circuit that prepares
            :math:`\\ket{\\varphi}`. It cannot contain measurements.
        psi (:class:`qibo.models.circuit.Circuit`): circuit that prepares
            :math:`\\ket{\\psi}`. It has the same number of qubits as ``phi``, and it cannot
            contain measurements.
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
    nqubits = _check_circuits(phi=phi, psi=psi)

    circuit = Circuit(2 * nqubits + 1, **kwargs)
    circuit.add(gates.H(0))
    circuit.add(psi.on_qubits(*range(1, nqubits + 1)))
    circuit.add(phi.on_qubits(*range(nqubits + 1, 2 * nqubits + 1)))
    circuit.add(
        gates.SWAP(qubit, qubit + nqubits).controlled_by(0)
        for qubit in range(1, nqubits + 1)
    )
    circuit.add(gates.H(0))

    if measure:
        circuit.add(gates.M(0))

    return circuit
