from numpy.typing import ArrayLike

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.gates.abstract import GATES_CONTROLLED_BY_DEFAULT
from qibo.models.circuit import Circuit
from qibo.models.encodings import binary_encoder


def lcu_circuit(
    unitaries: list[Circuit], coefficients: ArrayLike, backend: Backend = None
) -> tuple[Circuit, float, int]:
    """Creates a linear combination of unitaries (LCU) circuit.

    Let :math:`U_{i}` be the unitary of the :math:`i`-th circuit of ``unitaries``,
    with :math:`i = 0, \\dots, L - 1`, let :math:`c_{i}` be its complex coefficient, and let
    :math:`\\alpha = \\sum_{i} |c_{i}|`. The circuit acts on :math:`a = \\max(1, \\lceil
    \\log_{2} L \\rceil)` auxiliary qubits, followed by the qubits of the unitaries,
    and block-encodes the operator :math:`A / \\alpha`, with
    :math:`A = \\sum_{i} c_{i} U_{i}`. That is, the block of the circuit unitary in which
    the auxiliary qubits start and end in :math:`|0\\rangle` is :math:`A / \\alpha`. The
    circuit is :math:`\\mathrm{PREPARE}_{L}^{\\dagger} \\, \\mathrm{SELECT} \\,
    \\mathrm{PREPARE}_{R}`, where :math:`\\mathrm{PREPARE}_{L}` and
    :math:`\\mathrm{PREPARE}_{R}` are created with :func:`qibo.models.lcu.lcu_prepare`, and
    :math:`\\mathrm{SELECT}` is created with :func:`qibo.models.lcu.lcu_select`.

    Args:
        unitaries (list[:class:`qibo.models.circuit.Circuit`]): circuits
            on the same number of qubits that implement :math:`U_{i}`. They have to be
            exact, including their global phase, since they are applied under control.
        coefficients (ArrayLike): one-dimensional array of the (complex) coefficients
            :math:`c_{i}`, with at least one non-zero element.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        tuple: the circuit, with the auxiliary qubits at the first qubits, the
        normalization :math:`\\alpha`, and the number :math:`a` of auxiliary qubits. The
        last two are the arguments of :func:`qibo.models.qsvt.qsvt_circuit` for which
        the circuit is a block encoding of :math:`A / \\alpha`.

    Example:
        Apply :math:`(X + Z) / 2` to :math:`|0\\rangle`, where :math:`X` and :math:`Z`
        are Pauli matrices. The amplitudes with the auxiliary qubit in :math:`|0\\rangle`
        are those of :math:`|+\\rangle / \\sqrt{2}`.

        .. testcode::

            from qibo import Circuit, gates, get_backend
            from qibo.models.lcu import lcu_circuit

            backend = get_backend()

            pauli_x = Circuit(1)
            pauli_x.add(gates.X(0))
            pauli_z = Circuit(1)
            pauli_z.add(gates.Z(0))
            circuit, alpha, nauxiliary = lcu_circuit([pauli_x, pauli_z], [0.5, 0.5])

            target = Circuit(1)
            target.add(gates.H(0))

            state = circuit().state()[:2]
            print(
                backend.allclose(state, target().state() / backend.sqrt(2)),
                alpha,
                nauxiliary,
            )

        .. testoutput::

            True 1.0 1

    References:
        1. A. M. Childs and N. Wiebe, *Hamiltonian simulation using linear
        combinations of unitary operations*, `Quantum Information & Computation 12,
        901-924 (2012) <https://doi.org/10.26421/QIC12.11-12-1>`_.
    """
    backend = _check_backend(backend)

    coefficients = backend.cast(coefficients, dtype="complex128")

    if len(coefficients.shape) != 1 or len(coefficients) != len(unitaries):
        raise_error(
            ValueError,
            "``coefficients`` must be one-dimensional and have the same length as "
            + f"``unitaries``, which is {len(unitaries)}.",
        )

    prepare_left, prepare_right, alpha, nauxiliary = lcu_prepare(
        coefficients, backend=backend
    )

    # a term with a zero coefficient is replaced by the identity, which is skipped by SELECT
    unitaries = [
        unitary if float(magnitude) > 0.0 else Circuit(unitary.nqubits)
        for unitary, magnitude in zip(unitaries, backend.abs(coefficients))
    ]
    select = lcu_select(unitaries, nauxiliary, backend=backend)

    circuit = Circuit(select.nqubits)
    circuit.add(prepare_right.on_qubits(*range(nauxiliary)))
    circuit.add(select.on_qubits(*range(select.nqubits)))
    circuit.add(prepare_left.invert().on_qubits(*range(nauxiliary)))

    return circuit, alpha, nauxiliary


def lcu_prepare(
    coefficients: ArrayLike, backend: Backend = None
) -> tuple[Circuit, Circuit, float, int]:
    """Creates the :math:`\\mathrm{PREPARE}` oracles of a linear combination of
    unitaries (LCU).

    Let :math:`c_{i}` be the :math:`i`-th (complex) element of ``coefficients``, with
    :math:`i = 0, \\dots, L - 1`, and let :math:`\\alpha = \\sum_{i} |c_{i}|`. The oracles
    act on :math:`a = \\max(1, \\lceil \\log_{2} L \\rceil)` auxiliary qubits. When applied
    to :math:`|0\\rangle`, :math:`\\mathrm{PREPARE}_{L}` and :math:`\\mathrm{PREPARE}_{R}`
    load the amplitudes

    .. math::
        \\mathrm{PREPARE}_{L} |0\\rangle = \\sum_{i} \\sqrt{\\frac{|c_{i}|}{\\alpha}}
        \\, |i\\rangle \\quad \\text{and} \\quad
        \\mathrm{PREPARE}_{R} |0\\rangle = \\sum_{i} \\sqrt{\\frac{|c_{i}|}{\\alpha}}
        \\, \\frac{c_{i}}{|c_{i}|} \\, |i\\rangle \\, ,

    respectively, with the amplitudes of :math:`|i\\rangle`, for :math:`i \\geq L`, set to
    zero. For real and non-negative coefficients, they are the same oracle. The oracles
    are built with :func:`qibo.models.encodings.binary_encoder`, with the Mottonen
    parametrization, since the resulting circuits have an exact inverse. Together with
    :func:`qibo.models.lcu.lcu_select`, they build the block encoding
    :math:`\\mathrm{PREPARE}_{L}^{\\dagger} \\, \\mathrm{SELECT} \\, \\mathrm{PREPARE}_{R}`
    of :math:`\\sum_{i} c_{i} U_{i} / \\alpha`.

    Args:
        coefficients (ArrayLike): one-dimensional array of the (complex) coefficients
            :math:`c_{i}`, with at least one non-zero element.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        tuple: the circuits of :math:`\\mathrm{PREPARE}_{L}` and
        :math:`\\mathrm{PREPARE}_{R}`, both on :math:`a` qubits, the normalization
        :math:`\\alpha`, and the number :math:`a` of auxiliary qubits.

    Example:
        Prepare the amplitudes of the coefficients :math:`3` and :math:`-1`, for which
        :math:`\\alpha = 4`.

        .. testcode::

            from qibo import get_backend
            from qibo.models.lcu import lcu_prepare

            backend = get_backend()

            prepare_left, prepare_right, alpha, nauxiliary = lcu_prepare([3.0, -1.0])

            left = backend.cast([0.75**0.5, 0.25**0.5], dtype="complex128")
            right = backend.cast([0.75**0.5, -(0.25**0.5)], dtype="complex128")

            print(
                backend.allclose(prepare_left().state(), left),
                backend.allclose(prepare_right().state(), right),
                alpha,
                nauxiliary,
            )

        .. testoutput::

            True True 4.0 1

    References:
        1. A. M. Childs and N. Wiebe, *Hamiltonian simulation using linear
        combinations of unitary operations*, `Quantum Information & Computation 12,
        901-924 (2012) <https://doi.org/10.26421/QIC12.11-12-1>`_.
    """
    backend = _check_backend(backend)

    coefficients = backend.cast(coefficients, dtype="complex128")

    if len(coefficients.shape) != 1 or len(coefficients) == 0:
        raise_error(
            ValueError, "``coefficients`` must be a non-empty one-dimensional array."
        )

    magnitudes = backend.abs(coefficients)
    alpha = float(backend.sum(magnitudes))
    if alpha == 0.0:
        raise_error(ValueError, "``coefficients`` must have a non-zero element.")

    nauxiliary = max((len(coefficients) - 1).bit_length(), 1)
    padding = backend.zeros(2**nauxiliary - len(coefficients), dtype="float64")
    amplitudes = backend.concatenate((backend.sqrt(magnitudes / alpha), padding))
    phases = coefficients / backend.where(magnitudes > 0.0, magnitudes, 1.0)
    phases = backend.concatenate((phases, padding))

    # The Mottonen parametrization is used because its circuit has an exact inverse.
    prepare_left = binary_encoder(
        nauxiliary, parametrization="mottonen", data=amplitudes, backend=backend
    )
    prepare_right = binary_encoder(
        nauxiliary,
        parametrization="mottonen",
        data=amplitudes * phases,
        backend=backend,
    )

    return prepare_left, prepare_right, alpha, nauxiliary


def lcu_select(
    unitaries: list[Circuit], nauxiliary: int | None = None, backend: Backend = None
) -> Circuit:
    """Creates the :math:`\\mathrm{SELECT}` oracle of a linear combination of unitaries
    (LCU).

    Let :math:`U_{i}` be the unitary of the :math:`i`-th circuit of ``unitaries``, with
    :math:`i = 0, \\dots, L - 1`. The oracle acts on :math:`a` auxiliary qubits, followed by
    the qubits of the unitaries, and applies :math:`U_{i}` to the latter when the auxiliary
    qubits are in the computational basis state :math:`|i\\rangle`,

    .. math::
        \\mathrm{SELECT} = \\sum_{i = 0}^{L - 1} |i\\rangle\\langle i| \\otimes U_{i}
        + \\sum_{i = L}^{2^{a} - 1} |i\\rangle\\langle i| \\otimes I \\, .

    Each unitary is applied under the control of all the auxiliary qubits, with :math:`X`
    gates that map :math:`|i\\rangle` into :math:`|1 \\dots 1\\rangle`. Gates that are
    already controlled are supported. A unitary with no gates is the identity, and is
    skipped.

    Args:
        unitaries (list[:class:`qibo.models.circuit.Circuit`]): circuits
            on the same number of qubits that implement :math:`U_{i}`. They have to be
            exact, including their global phase, since they are applied under control.
        nauxiliary (int, optional): number :math:`a` of auxiliary qubits. It has to satisfy
            :math:`2^{a} \\geq L`. If ``None``, it is :math:`\\max(1, \\lceil \\log_{2} L
            \\rceil)`. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        :class:`qibo.models.circuit.Circuit`: circuit of :math:`\\mathrm{SELECT}`, with the
        auxiliary qubits at the first qubits.

    Example:
        Apply the Pauli matrices :math:`X` and :math:`Z` to :math:`|0\\rangle`, selected
        by an auxiliary qubit in :math:`|+\\rangle`. The state is
        :math:`(|0\\rangle|1\\rangle + |1\\rangle|0\\rangle) / \\sqrt{2}`.

        .. testcode::

            from qibo import Circuit, gates, get_backend
            from qibo.models.lcu import lcu_select

            backend = get_backend()

            pauli_x = Circuit(1)
            pauli_x.add(gates.X(0))
            pauli_z = Circuit(1)
            pauli_z.add(gates.Z(0))
            select = lcu_select([pauli_x, pauli_z])

            circuit = Circuit(2)
            circuit.add(gates.H(0))
            circuit.add(select.on_qubits(0, 1))

            target = backend.cast([0.0, 1.0, 1.0, 0.0], dtype="complex128")

            print(backend.allclose(circuit().state(), target / backend.sqrt(2)))

        .. testoutput::

            True

    References:
        1. A. M. Childs and N. Wiebe, *Hamiltonian simulation using linear
        combinations of unitary operations*, `Quantum Information & Computation 12,
        901-924 (2012) <https://doi.org/10.26421/QIC12.11-12-1>`_.
    """
    backend = _check_backend(backend)

    if len(unitaries) == 0:
        raise_error(ValueError, "``unitaries`` must have at least one element.")

    nsystem = unitaries[0].nqubits
    if any(unitary.nqubits != nsystem for unitary in unitaries):
        raise_error(ValueError, "``unitaries`` must act on the same number of qubits.")

    minimum = max((len(unitaries) - 1).bit_length(), 1)
    if nauxiliary is None:
        nauxiliary = minimum
    elif nauxiliary < minimum:
        raise_error(
            ValueError,
            f"``nauxiliary`` must be at least {minimum} for {len(unitaries)} unitaries.",
        )

    auxiliary = list(range(nauxiliary))
    system = list(range(nauxiliary, nauxiliary + nsystem))

    circuit = Circuit(nauxiliary + nsystem)
    for position, unitary in enumerate(unitaries):
        if len(unitary.queue) == 0:
            continue

        # X gates turn the auxiliary state |position> into |1...1>
        zeros = [
            qubit
            for qubit, bit in enumerate(format(position, f"0{nauxiliary}b"))
            if bit == "0"
        ]
        circuit.add(gates.X(qubit) for qubit in zeros)
        for gate in unitary.on_qubits(*system):
            controls = list(auxiliary)
            if gate.control_qubits:
                # a gate that is already controlled is rebuilt from its target matrix
                controls.extend(gate.control_qubits)
                matrix = gate.matrix(backend)
                if gate.name in GATES_CONTROLLED_BY_DEFAULT:
                    dim = 2 ** len(gate.target_qubits)
                    matrix = matrix[-dim:, -dim:]
                gate = gates.Unitary(matrix, *gate.target_qubits)
            circuit.add(gate.controlled_by(*controls))
        circuit.add(gates.X(qubit) for qubit in zeros)

    return circuit
