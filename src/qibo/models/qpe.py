import math
from numbers import Integral

from numpy.typing import ArrayLike

from qibo import gates
from qibo.config import raise_error
from qibo.gates.abstract import Gate
from qibo.measurements import MeasurementResult
from qibo.models.circuit import Circuit
from qibo.models.qft import QFT


def IQPE(
    unitary: Circuit | Gate | ArrayLike,
    nbits: int,
    **kwargs,
) -> tuple[Circuit, list[MeasurementResult]]:
    """Creates a circuit that implements the Iterative Quantum Phase Estimation.

    Given a unitary :math:`U` and one of its eigenstates :math:`|\\psi\\rangle`, such that
    :math:`U|\\psi\\rangle = e^{2 \\pi i \\varphi}|\\psi\\rangle`, the circuit estimates
    :math:`\\varphi \\in [0, \\, 1)` with :math:`t` bits of precision, where :math:`t` is
    ``nbits``, as :func:`qibo.models.qpe.QPE` does. However, a single counting qubit is
    measured :math:`t` times, instead of using :math:`t` counting qubits and the inverse
    :class:`qibo.models.qft.QFT`. The circuit acts on :math:`1 + n` qubits, with :math:`n` being
    the number of qubits of :math:`U`. The first qubit is the counting qubit and the last
    :math:`n` qubits are the target qubits.

    The bits of :math:`\\varphi` are obtained one at a time, starting from the least significant
    one. In each step :math:`k = 1, \\dots, t`:

    1. a Hadamard gate is applied to the counting qubit;
    2. :math:`U^{2^{t - k}}` is applied to the target qubits, controlled by the counting qubit;
    3. a phase gate is applied to the counting qubit, with the angle
       :math:`-2 \\pi \\sum_{j=1}^{k-1} m_{j} \\, 2^{-(k - j + 1)}`, where :math:`m_{j}` is the
       outcome of the step :math:`j`, which removes the contribution of the bits that were
       already measured;
    4. a Hadamard gate is applied to the counting qubit, which is measured with
       ``collapse=True``, and the outcome :math:`m_{k}` is the bit :math:`b_{t - k + 1}` of
       :math:`\\varphi = 0.b_{1} \\cdots b_{t}`;
    5. if it is not the last step, the counting qubit is reset to :math:`|0\\rangle` with an
       :math:`X` rotation conditioned on :math:`m_{k}`.

    The circuit finishes with a measurement of the counting qubit that is not collapsed, so that
    it can be simulated with state vectors. The target qubits have to be prepared in
    :math:`|\\psi\\rangle` before the circuit is applied, *e.g.* by adding the circuit that
    prepares it to a larger circuit before the one returned by this function.

    The outcomes of the measurements are not merged into the result of the execution of the
    circuit. They are returned as a list of :class:`qibo.measurements.MeasurementResult`, which
    is ordered from the most to the least significant bit, *i.e.* in the reverse order of the
    measurements. Hence, they can be joined with
    :meth:`qibo.measurements.MeasurementResult.join` and the estimate of :math:`\\varphi` is
    obtained with :func:`qibo.quantum_info.estimate_phase`.

    .. note::
        As in :func:`qibo.models.qpe.QPE`, :math:`U` is applied :math:`2^{t} - 1` times in
        total. The circuit has fewer qubits, but when it is simulated, each shot has to be
        simulated separately because of the measurements with collapse.

    Args:
        unitary (:class:`qibo.models.circuit.Circuit` or :class:`qibo.gates.Gate` or ndarray):
            unitary :math:`U`, given as a circuit, as a gate, *e.g.* :class:`qibo.gates.Unitary`,
            or as a matrix. See :func:`qibo.models.qpe.QPE` for the details.
        nbits (int): number of bits :math:`t` of the phase to be estimated, which sets the
            resolution :math:`2^{-t}` and the number of measurements.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        tuple: the circuit that implements the Iterative Quantum Phase Estimation and the
        list of :class:`qibo.measurements.MeasurementResult` with the outcomes of the
        measurements of the counting qubit, from the most to the least significant bit.

    Example:
        .. testcode::

            import math

            from qibo import Circuit, gates
            from qibo.measurements import MeasurementResult
            from qibo.models import IQPE
            from qibo.quantum_info import estimate_phase

            # unitary with eigenstate |1> and eigenvalue exp(2 pi i phase)
            phase = 5 / 16
            unitary = Circuit(1)
            unitary.add(gates.U1(0, theta=2 * math.pi * phase))

            # phase estimation with 4 bits
            iqpe, results = IQPE(unitary, 4)

            # prepare the eigenstate in the target qubit and apply the phase estimation
            circuit = Circuit(iqpe.nqubits)
            circuit.add(gates.X(1))
            circuit.add(iqpe)
            circuit(nshots=100)

            print(estimate_phase(MeasurementResult.join(*results)))

        .. testoutput::

            0.3125

    References:
        1. A. Yu. Kitaev, *Quantum measurements and the Abelian Stabilizer Problem*,
        `arXiv:quant-ph/9511026 <https://arxiv.org/abs/quant-ph/9511026>`_.

        2. R. B. Griffiths and C.-S. Niu, *Semiclassical Fourier Transform for Quantum
        Computation*, `Phys. Rev. Lett. 76, 3228 (1996)
        <https://arxiv.org/abs/quant-ph/9511007>`_.
    """
    unitary, nbits = _check_inputs(unitary, nbits, "nbits")

    target_qubits = list(range(1, unitary.nqubits + 1))

    # the control is placed right after the qubits of the unitary,
    # and it is mapped to the counting qubit when the circuit is added
    controlled_unitary = unitary.controlled_by(unitary.nqubits)

    circuit = Circuit(unitary.nqubits + 1, **kwargs)
    results = []
    for step in range(1, nbits + 1):
        circuit.add(gates.H(0))
        for _ in range(2 ** (nbits - step)):
            circuit.add(controlled_unitary.on_qubits(*target_qubits, 0))

        if step > 1:
            # removes the contribution of the bits that were already measured
            correction = sum(
                results[previous].symbols[0] / 2 ** (step - previous)
                for previous in range(step - 1)
            )
            circuit.add(gates.U1(0, theta=-2 * math.pi * correction))

        circuit.add(gates.H(0))
        results.append(circuit.add(gates.M(0, collapse=True)))

        if step < nbits:
            # resets the counting qubit to the state 0 if the outcome was 1
            circuit.add(gates.RX(0, theta=math.pi * results[-1].symbols[0]))

    circuit.add(gates.M(0))

    return circuit, results[::-1]


def QPE(
    unitary: Circuit | Gate | ArrayLike,
    nqubits_counting: int,
    measure: bool = True,
    **kwargs,
) -> Circuit:
    """Creates a circuit that implements the canonical Quantum Phase Estimation.

    Given a unitary :math:`U` and one of its eigenstates :math:`|\\psi\\rangle`, such that
    :math:`U|\\psi\\rangle = e^{2 \\pi i \\varphi}|\\psi\\rangle`, the circuit estimates
    :math:`\\varphi \\in [0, \\, 1)` with :math:`t` bits of precision, where :math:`t`
    is the number of counting qubits. It acts on :math:`t + n` qubits, with :math:`n`
    being the number of qubits of :math:`U`, and it is composed of the following steps:

    1. a Hadamard gate on each of the :math:`t` counting qubits, which are the first qubits of
       the circuit;
    2. :math:`U^{2^{t - 1 - j}}`, controlled by the counting qubit :math:`j`, acting on the
       last :math:`n` qubits of the circuit, the target qubits;
    3. the inverse :class:`qibo.models.qft.QFT` on the counting qubits;
    4. the measurement of the counting qubits, if ``measure=True``.

    The target qubits have to be prepared in :math:`|\\psi\\rangle` before the circuit is applied,
    *e.g.* by adding the circuit that prepares it to a larger circuit before the one returned by
    this function. At the end, the counting qubits hold a bitstring :math:`b_{1} \\cdots b_{t}`,
    with :math:`b_{1}` being the outcome of the first qubit, such that :math:`\\varphi` is
    approximated by :math:`0.b_{1} \\cdots b_{t}` in binary, which is exact if :math:`\\varphi`
    is an integer multiple of :math:`2^{-t}`. The conversion from the bitstring into the phase
    can be done with :func:`qibo.quantum_info.bitstring_to_phase`, and the phase of the most
    probable outcome, *i.e.* the estimate of :math:`\\varphi`, with
    :func:`qibo.quantum_info.estimate_phase`.

    .. note::
        The power :math:`U^{2^{k}}` is implemented by applying the controlled :math:`U` a number
        :math:`2^{k}` of times. Hence, :math:`U` is applied :math:`2^{t} - 1` times in total.

    Args:
        unitary (:class:`qibo.models.circuit.Circuit` or :class:`qibo.gates.Gate` or ndarray):
            unitary :math:`U`, given as a circuit, as a gate, *e.g.* :class:`qibo.gates.Unitary`,
            or as a matrix. A matrix must be unitary and of dimension :math:`2^{n} \\times 2^{n}`
            with :math:`n > 0`, and it is converted into a :class:`qibo.gates.Unitary` acting
            on :math:`n` qubits. The qubits of a gate are relabeled, in increasing order, as the
            qubits of a circuit with the same number of qubits as the gate. All of the gates
            must be supported by :meth:`qibo.models.circuit.Circuit.controlled_by`.
        nqubits_counting (int): number of counting qubits :math:`t`. The phase is estimated with
            a resolution of :math:`2^{-t}`.
        measure (bool, optional): If ``True``, adds a measurement of the counting qubits at
            the end of the circuit. Defaults to ``True``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`. For details, see the documentation
            of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: implementation of the Quantum Phase Estimation.

    Example:
        .. testcode::

            import math

            from qibo import Circuit, gates
            from qibo.models import QPE
            from qibo.quantum_info import estimate_phase

            # unitary with eigenstate |1> and eigenvalue exp(2 pi i phase)
            phase = 5 / 16
            unitary = Circuit(1)
            unitary.add(gates.U1(0, theta=2 * math.pi * phase))

            # phase estimation with 4 counting qubits
            nqubits_counting = 4
            qpe = QPE(unitary, nqubits_counting)

            # prepare the eigenstate in the target qubit and apply the phase estimation
            circuit = Circuit(qpe.nqubits)
            circuit.add(gates.X(nqubits_counting))
            circuit.add(qpe)

            result = circuit(nshots=100)
            print(estimate_phase(result))

        .. testoutput::

            0.3125

    References:
        1. R. Cleve, A. Ekert, C. Macchiavello, and M. Mosca, *Quantum algorithms revisited*,
        `Proc. R. Soc. Lond. A 454, 339 (1998) <https://arxiv.org/abs/quant-ph/9708016>`_.

        2. M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*,
        Cambridge University Press (2010), Sec. 5.2.
    """
    unitary, nqubits_counting = _check_inputs(
        unitary, nqubits_counting, "nqubits_counting"
    )

    counting_qubits = list(range(nqubits_counting))
    target_qubits = list(range(nqubits_counting, nqubits_counting + unitary.nqubits))

    # the control is placed right after the qubits of the unitary,
    # and it is mapped to each counting qubit when the circuit is added
    controlled_unitary = unitary.controlled_by(unitary.nqubits)

    circuit = Circuit(nqubits_counting + unitary.nqubits, **kwargs)
    circuit.add(gates.H(qubit) for qubit in counting_qubits)
    for qubit in counting_qubits:
        for _ in range(2 ** (nqubits_counting - 1 - qubit)):
            circuit.add(controlled_unitary.on_qubits(*target_qubits, qubit))
    circuit.add(QFT(nqubits_counting).invert().on_qubits(*counting_qubits))

    if measure:
        circuit.add(gates.M(*counting_qubits))

    return circuit


def _check_inputs(
    unitary: Circuit | Gate | ArrayLike, nbits: int, name: str
) -> tuple[Circuit, int]:
    """Checks the inputs of the phase estimation circuits and converts them if needed.

    Args:
        unitary (:class:`qibo.models.circuit.Circuit` or :class:`qibo.gates.Gate` or ndarray):
            unitary, given as a circuit, as a gate or as a matrix.
        nbits (int): number of counting qubits or of bits of the phase to be estimated.
        name (str): name of ``nbits`` in the error messages.

    Returns:
        tuple: circuit that implements the unitary, and ``nbits`` as an ``int``.
    """
    if not isinstance(unitary, (Circuit, Gate)):
        shape = getattr(unitary, "shape", None)
        if shape is None:
            raise_error(
                TypeError,
                "unitary must be a Circuit, a Gate or a matrix, "
                + f"but it is type {type(unitary)}.",
            )

        if (
            len(shape) != 2
            or shape[0] != shape[1]
            or shape[0] < 2
            or shape[0] & (shape[0] - 1)
        ):
            raise_error(
                ValueError,
                "unitary must be a square matrix of dimension 2^n, with n > 0, "
                + f"but it has shape {tuple(shape)}.",
            )

        unitary = gates.Unitary(unitary, *range(int(math.log2(shape[0]))))
        if not unitary.unitary:
            raise_error(ValueError, "unitary must be a unitary matrix.")

    if isinstance(unitary, Gate):
        qubits = sorted(unitary.qubits)
        if len(qubits) == 0:
            raise_error(
                ValueError, "unitary must be a gate that acts on at least one qubit."
            )

        gate = unitary.on_qubits({qubit: index for index, qubit in enumerate(qubits)})
        unitary = Circuit(len(qubits))
        unitary.add(gate)

    if isinstance(nbits, bool) or not isinstance(nbits, Integral) or nbits < 1:
        raise_error(
            ValueError,
            f"{name} must be a positive integer, but it is {nbits}.",
        )

    return unitary, int(nbits)
