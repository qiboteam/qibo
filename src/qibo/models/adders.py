import math
from numbers import Integral

from qibo import gates
from qibo.config import raise_error
from qibo.models.circuit import Circuit
from qibo.models.qft import QFT


def Adder(nbits: int, method: str = "Cuccaro", **kwargs) -> Circuit:
    """Creates a circuit that adds two integers with the adder chosen by ``method``.

    Given two :math:`n`-bit non-negative integers :math:`a` and :math:`b`, where :math:`n` is
    ``nbits``, all of the adders perform

    .. math::
        \\ket{a} \\, \\ket{b} \\, \\ket{0} \\mapsto \\ket{a} \\, \\ket{a + b \\bmod 2^{n + 1}}
            \\, \\ket{0} \\, ,

    where the last register are the auxiliary qubits, if the adder has any. The first :math:`n`
    qubits hold :math:`a`, and the next :math:`n + 1` qubits hold :math:`b`, with the overflow
    bit being the last one. In every register, the least significant bit comes first. The
    adders differ in the number of auxiliary qubits and of gates:

    * ``"Cuccaro"``: :func:`qibo.models.adders.Cuccaro`, ripple-carry adder with one auxiliary
      qubit;
    * ``"Draper"``: :func:`qibo.models.adders.Draper`, adder in the Fourier basis, without
      auxiliary qubits;
    * ``"TTK"``: :func:`qibo.models.adders.TTK`, ripple-carry adder of
      :class:`qibo.gates.TOFFOLI` and :class:`qibo.gates.CNOT` gates, without auxiliary qubits;
    * ``"VBE"``: :func:`qibo.models.adders.VBE`, ripple-carry adder with :math:`n` auxiliary
      qubits.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import Adder

            nbits = 2
            adder = Adder(nbits, method="TTK")

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
        method (str, optional): adder to be used. Either ``"Cuccaro"``, ``"Draper"``, ``"TTK"``
            or ``"VBE"``. Defaults to ``"Cuccaro"``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: implementation of the chosen adder.

    References:
        1. S. A. Cuccaro, T. G. Draper, S. A. Kutin, and D. P. Moulton, *A new quantum
        ripple-carry addition circuit*,
        `arXiv:quant-ph/0410184 <https://arxiv.org/abs/quant-ph/0410184>`_.

        2. T. G. Draper, *Addition on a Quantum Computer*,
        `arXiv:quant-ph/0008033 <https://arxiv.org/abs/quant-ph/0008033>`_.

        3. Y. Takahashi, S. Tani, and N. Kunihiro, *Quantum addition circuits and unbounded
        fan-out*, `Quantum Inf. Comput. 10, 872 (2010) <https://doi.org/10.26421/QIC10.9-10-12>`_.

        4. V. Vedral, A. Barenco, and A. Ekert, *Quantum networks for elementary arithmetic
        operations*, `Phys. Rev. A 54, 147 (1996) <https://doi.org/10.1103/PhysRevA.54.147>`_.
    """
    methods = {"Cuccaro": Cuccaro, "Draper": Draper, "TTK": TTK, "VBE": VBE}

    if method not in tuple(methods):
        raise_error(
            ValueError,
            f"method must be one of {tuple(methods)}, but it is {method}.",
        )

    return methods[method](nbits, **kwargs)


def Cuccaro(nbits: int, **kwargs) -> Circuit:
    """Creates a circuit that implements the Cuccaro-Draper-Kutin-Moulton ripple-carry adder.

    Given two :math:`n`-bit non-negative integers :math:`a` and :math:`b`, where :math:`n` is
    ``nbits``, the circuit performs

    .. math::
        \\ket{a} \\, \\ket{b} \\, \\ket{0} \\mapsto \\ket{a} \\, \\ket{a + b \\bmod 2^{n + 1}}
            \\, \\ket{0} \\, .

    It acts on :math:`2n + 2` qubits, which are split into three registers:

    1. the first :math:`n` qubits hold :math:`a`, which is left unchanged;
    2. the next :math:`n + 1` qubits hold :math:`b`, which is replaced by :math:`a + b`. The last
       qubit of this register is the overflow bit, which is in the state :math:`\\ket{0}` at the
       beginning if :math:`b < 2^{n}`, so that the sum is exact;
    3. the last qubit is a single auxiliary qubit, the carry-in. It must be in the state
       :math:`\\ket{0}` at the beginning, and it is returned to it at the end.

    In every register, the :math:`k`-th qubit holds the bit :math:`2^{k}`, *i.e.* the least
    significant bit comes first. The circuit is composed of :math:`n` majority (MAJ) blocks,
    which compute the carries in place, a :class:`qibo.gates.CNOT` that copies the last carry
    into the overflow qubit, and :math:`n` unmajority-and-add (UMA) blocks, which uncompute the
    carries and write the sum bits, in reverse order. It has :math:`2n`
    :class:`qibo.gates.TOFFOLI` gates and :math:`4n + 1` :class:`qibo.gates.CNOT` gates.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import Cuccaro

            nbits = 2
            adder = Cuccaro(nbits)

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
        :class:`qibo.models.circuit.Circuit`: implementation of the Cuccaro ripple-carry adder.

    References:
        1. S. A. Cuccaro, T. G. Draper, S. A. Kutin, and D. P. Moulton, *A new quantum
        ripple-carry addition circuit*,
        `arXiv:quant-ph/0410184 <https://arxiv.org/abs/quant-ph/0410184>`_.
    """
    nbits = _check_nbits(nbits)

    register_a = list(range(nbits))
    register_b = list(range(nbits, 2 * nbits + 1))
    carry_in = 2 * nbits + 1

    # the carry c_{i + 1} is stored in the qubit of a_i, and the carry c_0 in the carry-in qubit
    carries = [carry_in] + register_a[:-1]

    # MAJ block acts on the qubits (c_i, b_i, a_i)
    majority = Circuit(3)
    majority.add(gates.CNOT(2, 1))
    majority.add(gates.CNOT(2, 0))
    majority.add(gates.TOFFOLI(0, 1, 2))

    # UMA block acts on the qubits (c_i, b_i, a_i)
    unmajority = Circuit(3)
    unmajority.add(gates.TOFFOLI(0, 1, 2))
    unmajority.add(gates.CNOT(2, 0))
    unmajority.add(gates.CNOT(0, 1))

    circuit = Circuit(2 * nbits + 2, **kwargs)
    for k in range(nbits):
        circuit.add(majority.on_qubits(carries[k], register_b[k], register_a[k]))

    # the last carry is added to the overflow qubit
    circuit.add(gates.CNOT(register_a[-1], register_b[-1]))

    for k in reversed(range(nbits)):
        circuit.add(unmajority.on_qubits(carries[k], register_b[k], register_a[k]))

    return circuit


def Draper(nbits: int, **kwargs) -> Circuit:
    """Creates a circuit that implements the Draper quantum Fourier transform (QFT) adder.

    Given two :math:`n`-bit non-negative integers :math:`a` and :math:`b`, where :math:`n` is
    ``nbits``, the circuit performs

    .. math::
        \\ket{a} \\, \\ket{b} \\mapsto \\ket{a} \\, \\ket{a + b \\bmod 2^{n + 1}} \\, .

    It acts on :math:`2n + 1` qubits, and it does not need auxiliary qubits. The first :math:`n`
    qubits hold :math:`a`, which is left unchanged, and the last :math:`n + 1` qubits hold
    :math:`b`, which is replaced by :math:`a + b`. The last qubit of the second register is the
    overflow bit, which is in the state :math:`\\ket{0}` at the beginning if :math:`b < 2^{n}`,
    so that the sum is exact.

    In every register, the :math:`k`-th qubit holds the bit :math:`2^{k}`, *i.e.* the least
    significant bit comes first. The circuit is composed of the following steps:

    1. the :class:`qibo.models.qft.QFT` on the register of :math:`b`, without swaps;
    2. :class:`qibo.gates.CU1` gates, controlled by the qubits of :math:`a`, which add :math:`a`
       to :math:`b` in the Fourier basis, by rotating the phases of the qubits of :math:`b`;
    3. the inverse :class:`qibo.models.qft.QFT` on the register of :math:`b`, without swaps.

    The smallest rotation angle is :math:`2 \\pi / 2^{n + 1}`.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import Draper

            nbits = 2
            adder = Draper(nbits)

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
        :class:`qibo.models.circuit.Circuit`: implementation of the Draper adder.

    References:
        1. T. G. Draper, *Addition on a Quantum Computer*,
        `arXiv:quant-ph/0008033 <https://arxiv.org/abs/quant-ph/0008033>`_.
    """
    nbits = _check_nbits(nbits)

    register_a = list(range(nbits))
    register_b = list(range(nbits, 2 * nbits + 1))

    # the QFT receives the qubits of b from the most to the least significant one
    qft = QFT(nbits + 1, with_swaps=False)

    circuit = Circuit(2 * nbits + 1, **kwargs)
    circuit.add(qft.on_qubits(*register_b[::-1]))

    # without swaps, the qubit of b_j has a phase proportional to b / 2^{j + 1}, and the
    # rotations that are multiples of 2 pi, i.e. the ones with k > j, are left out
    for j, target in enumerate(register_b):
        for k in range(min(j + 1, nbits)):
            theta = 2 * math.pi / 2 ** (j + 1 - k)
            circuit.add(gates.CU1(register_a[k], target, theta))

    circuit.add(qft.invert().on_qubits(*register_b[::-1]))

    return circuit


def TTK(nbits: int, **kwargs) -> Circuit:
    """Creates a circuit that implements the Takahashi-Tani-Kunihiro (TTK) adder.

    Given two :math:`n`-bit non-negative integers :math:`a` and :math:`b`, where :math:`n` is
    ``nbits``, the circuit performs

    .. math::
        \\ket{a} \\, \\ket{b} \\mapsto \\ket{a} \\, \\ket{a + b \\bmod 2^{n + 1}} \\, .

    It acts on :math:`2n + 1` qubits, and it does not need auxiliary qubits. The first :math:`n`
    qubits hold :math:`a`, which is left unchanged, and the last :math:`n + 1` qubits hold
    :math:`b`, which is replaced by :math:`a + b`. The last qubit of the second register is the
    overflow bit, which is in the state :math:`\\ket{0}` at the beginning if :math:`b < 2^{n}`,
    so that the sum is exact.

    In every register, the :math:`k`-th qubit holds the bit :math:`2^{k}`, *i.e.* the least
    significant bit comes first. The circuit only has :class:`qibo.gates.TOFFOLI` and
    :class:`qibo.gates.CNOT` gates. The carries are computed in place, using the qubits of
    :math:`a`, and then uncomputed while the sum bits are written in :math:`b`. For
    :math:`n \\geq 2`, it has :math:`2n - 1` :class:`qibo.gates.TOFFOLI` gates and :math:`5n - 5`
    :class:`qibo.gates.CNOT` gates. For :math:`n = 1`, it is a half adder, with one of each.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import TTK

            nbits = 2
            adder = TTK(nbits)

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
        :class:`qibo.models.circuit.Circuit`: implementation of the TTK adder.

    References:
        1. Y. Takahashi, S. Tani, and N. Kunihiro, *Quantum addition circuits and unbounded
        fan-out*, `Quantum Inf. Comput. 10, 872 (2010) <https://doi.org/10.26421/QIC10.9-10-12>`_.
    """
    nbits = _check_nbits(nbits)

    register_a = list(range(nbits))
    register_b = list(range(nbits, 2 * nbits + 1))
    overflow = register_b[-1]

    circuit = Circuit(2 * nbits + 1, **kwargs)

    # for a single bit, there is no carry to be propagated
    if nbits == 1:
        circuit.add(gates.TOFFOLI(register_a[0], register_b[0], overflow))
        circuit.add(gates.CNOT(register_a[0], register_b[0]))
        return circuit

    # the qubits of a hold the carries, which are computed with the qubits of b in the state
    # b_k + a_k, and they are uncomputed at the end
    circuit.add(gates.CNOT(register_a[k], register_b[k]) for k in range(1, nbits))
    circuit.add(gates.CNOT(register_a[-1], overflow))
    circuit.add(
        gates.CNOT(register_a[k], register_a[k + 1]) for k in range(nbits - 2, 0, -1)
    )
    circuit.add(
        gates.TOFFOLI(register_a[k], register_b[k], register_a[k + 1])
        for k in range(nbits - 1)
    )
    circuit.add(gates.TOFFOLI(register_a[-1], register_b[nbits - 1], overflow))

    for k in range(nbits - 1, 0, -1):
        circuit.add(gates.CNOT(register_a[k], register_b[k]))
        circuit.add(gates.TOFFOLI(register_a[k - 1], register_b[k - 1], register_a[k]))

    circuit.add(
        gates.CNOT(register_a[k], register_a[k + 1]) for k in range(1, nbits - 1)
    )
    circuit.add(gates.CNOT(register_a[k], register_b[k]) for k in range(nbits))

    return circuit


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
    nbits = _check_nbits(nbits)

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


def _check_nbits(nbits: int) -> int:
    """Checks the number of bits of the integers to be added.

    Args:
        nbits (int): number of bits of each of the integers to be added.

    Returns:
        int: ``nbits`` as an ``int``.
    """
    if isinstance(nbits, bool) or not isinstance(nbits, Integral) or nbits < 1:
        raise_error(ValueError, f"nbits must be a positive integer, but it is {nbits}.")

    return int(nbits)
