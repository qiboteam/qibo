import math

from numpy.typing import ArrayLike

from qibo import gates
from qibo.config import raise_error
from qibo.gates.abstract import Gate
from qibo.models.circuit import Circuit
from qibo.models.qft import QFT


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
    can be done with :func:`qibo.quantum_info.bitstring_to_phase`.

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
            from qibo.quantum_info import bitstring_to_phase

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

            frequencies = circuit(nshots=100).frequencies()
            print(bitstring_to_phase(frequencies.most_common(1)[0][0]))

        .. testoutput::

            0.3125

    References:
        1. R. Cleve, A. Ekert, C. Macchiavello, and M. Mosca, *Quantum algorithms revisited*,
        `Proc. R. Soc. Lond. A 454, 339 (1998) <https://arxiv.org/abs/quant-ph/9708016>`_.

        2. M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*,
        Cambridge University Press (2010), Sec. 5.2.
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

    if not isinstance(nqubits_counting, int) or nqubits_counting < 1:
        raise_error(
            ValueError,
            f"nqubits_counting must be a positive integer, but it is {nqubits_counting}.",
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
