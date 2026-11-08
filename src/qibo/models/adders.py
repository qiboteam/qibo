from numbers import Integral

from qibo import gates
from qibo.config import raise_error
from qibo.models.circuit import Circuit


def VBE(nbits: int, **kwargs) -> Circuit:
    """Creates a circuit that implements the Vedral-Barenco-Ekert (VBE) plain adder.

    Given two :math:`n`-bit non-negative integers :math:`a` and :math:`b`, where :math:`n` is
    ``nbits``, the circuit performs

    .. math::
        \\ket{a} \\, \\ket{b} \\, \\ket{0} \\mapsto \\ket{a} \\, \\ket{a + b} \\, \\ket{0} \\, .

    It acts on :math:`3n + 1` qubits, which are split into three registers:

    1. the first :math:`n` qubits hold :math:`a`, which is left unchanged;
    2. the next :math:`n + 1` qubits hold :math:`b`, which is replaced by :math:`a + b`. The last
       qubit of this register is the overflow bit, and it must be in the state :math:`\\ket{0}`
       at the beginning;
    3. the last :math:`n` qubits are auxiliary qubits that store the carries. They must be in
       the state :math:`\\ket{0}` at the beginning, and they are returned to it at the end.

    In every register, the :math:`k`-th qubit holds the bit :math:`2^{k}`, *i.e.* the least
    significant bit comes first. The circuit is composed of :math:`n` blocks that compute the
    carries, followed by blocks that compute the sum bits while the carries are uncomputed, in
    reverse order. It has :math:`4n - 2` :class:`qibo.gates.TOFFOLI` gates and :math:`4n`
    :class:`qibo.gates.CNOT` gates.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import VBE

            nbits = 2
            adder = VBE(nbits)

            # a = 3 in the qubits 0 and 1, and b = 1 in the qubits 2, 3 and 4
            circuit = Circuit(adder.nqubits)
            circuit.add(gates.X(0))
            circuit.add(gates.X(1))
            circuit.add(gates.X(2))
            circuit.add(adder)
            circuit.add(gates.M(*range(nbits, 2 * nbits + 1)))

            result = circuit(nshots=1)
            bitstring = next(iter(result.frequencies(binary=True)))

            # the least significant bit comes first
            print(int(bitstring[::-1], 2))

        .. testoutput::

            4

    Args:
        nbits (int): number of bits :math:`n` of each of the integers to be added.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: implementation of the VBE plain adder.

    References:
        1. V. Vedral, A. Barenco, and A. Ekert, *Quantum networks for elementary arithmetic
        operations*, `Phys. Rev. A 54, 147 (1996) <https://doi.org/10.1103/PhysRevA.54.147>`_.
    """
    if isinstance(nbits, bool) or not isinstance(nbits, Integral) or nbits < 1:
        raise_error(ValueError, f"nbits must be a positive integer, but it is {nbits}.")

    nbits = int(nbits)
    register_a = list(range(nbits))
    register_b = list(range(nbits, 2 * nbits + 1))
    register_carry = list(range(2 * nbits + 1, 3 * nbits + 1))

    # the carry of the last bit is stored in the overflow qubit of the register of b
    carries = register_carry + register_b[-1:]

    # CARRY block acts on the qubits (c_i, a_i, b_i, c_{i + 1})
    carry = Circuit(4)
    carry.add(gates.TOFFOLI(1, 2, 3))
    carry.add(gates.CNOT(1, 2))
    carry.add(gates.TOFFOLI(0, 2, 3))

    # SUM block acts on the qubits (c_i, a_i, b_i)
    summation = Circuit(3)
    summation.add(gates.CNOT(1, 2))
    summation.add(gates.CNOT(0, 2))

    circuit = Circuit(3 * nbits + 1, **kwargs)
    for k in range(nbits):
        circuit.add(
            carry.on_qubits(carries[k], register_a[k], register_b[k], carries[k + 1])
        )

    # the last carry is not uncomputed, so the CNOT of its block has to be undone
    circuit.add(gates.CNOT(register_a[-1], register_b[nbits - 1]))
    circuit.add(
        summation.on_qubits(carries[nbits - 1], register_a[-1], register_b[nbits - 1])
    )

    for k in reversed(range(nbits - 1)):
        circuit.add(
            carry.invert().on_qubits(
                carries[k], register_a[k], register_b[k], carries[k + 1]
            )
        )
        circuit.add(summation.on_qubits(carries[k], register_a[k], register_b[k]))

    return circuit
