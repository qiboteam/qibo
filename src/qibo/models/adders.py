import math

from qibo import gates
from qibo.config import raise_error
from qibo.models.circuit import Circuit
from qibo.models.qft import QFT


def Adder(nbits: int, method: str = "Cuccaro", **kwargs) -> Circuit:
    """Creates a circuit that adds two integers with the adder chosen by ``method``.

    Given two :math:`n`-bit non-negative integers :math:`a` and :math:`b`, where :math:`n` is
    ``nbits``, all of the adders perform

    .. math::
        \\ket{a} \\, \\ket{b} \\, \\ket{0} \\mapsto \\ket{a} \\, \\ket{a + b} \\, \\ket{0} \\, ,

    where the last register are the auxiliary qubits, if the adder has any. The first :math:`n`
    qubits hold :math:`a`, and the next :math:`n + 1` qubits hold :math:`b`, with the overflow
    bit being the last one, which must be in the state :math:`\\ket{0}` at the beginning. In
    every register, the least significant bit comes first. The adders differ in the number of
    auxiliary qubits and of gates:

    * ``"Cuccaro"``: :func:`qibo.models.adders.Cuccaro`, ripple-carry adder with one auxiliary
      qubit;
    * ``"DKRS"``: :func:`qibo.models.adders.DKRS`, carry-lookahead adder with logarithmic depth
      and a linear number of auxiliary qubits;
    * ``"Draper"``: :func:`qibo.models.adders.Draper`, adder in the Fourier basis, without
      auxiliary qubits;
    * ``"Gidney"``: :func:`qibo.models.adders.Gidney`, ripple-carry adder that uses measurements
      to uncompute the carries, with :math:`n - 1` auxiliary qubits;
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
        method (str, optional): adder to be used. Either ``"Cuccaro"``, ``"DKRS"``,
            ``"Draper"``, ``"Gidney"``, ``"TTK"`` or ``"VBE"``. Defaults to ``"Cuccaro"``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: implementation of the chosen adder.

    References:
        1. S. A. Cuccaro, T. G. Draper, S. A. Kutin, and D. P. Moulton, *A new quantum
        ripple-carry addition circuit*,
        `arXiv:quant-ph/0410184 <https://arxiv.org/abs/quant-ph/0410184>`_.

        2. T. G. Draper, S. A. Kutin, E. M. Rains, and K. M. Svore, *A logarithmic-depth quantum
        carry-lookahead adder*,
        `Quantum Inf. Comput. 6, 351 (2006) <https://doi.org/10.26421/QIC6.4-5-4>`_.

        3. T. G. Draper, *Addition on a Quantum Computer*,
        `arXiv:quant-ph/0008033 <https://arxiv.org/abs/quant-ph/0008033>`_.

        4. C. Gidney, *Halving the cost of quantum addition*,
        `Quantum 2, 74 (2018) <https://doi.org/10.22331/q-2018-06-18-74>`_.

        5. Y. Takahashi, S. Tani, and N. Kunihiro, *Quantum addition circuits and unbounded
        fan-out*, `Quantum Inf. Comput. 10, 872 (2010) <https://doi.org/10.26421/QIC10.9-10-12>`_.

        6. V. Vedral, A. Barenco, and A. Ekert, *Quantum networks for elementary arithmetic
        operations*, `Phys. Rev. A 54, 147 (1996) <https://doi.org/10.1103/PhysRevA.54.147>`_.
    """
    methods = {
        "Cuccaro": Cuccaro,
        "DKRS": DKRS,
        "Draper": Draper,
        "Gidney": Gidney,
        "TTK": TTK,
        "VBE": VBE,
    }

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
    carries = [carry_in]
    carries.extend(register_a[:-1])

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


def DKRS(nbits: int, **kwargs) -> Circuit:
    """Creates a circuit that implements the Draper-Kutin-Rains-Svore (DKRS) carry-lookahead adder.

    Given two :math:`n`-bit non-negative integers :math:`a` and :math:`b`, where :math:`n` is
    ``nbits``, the circuit performs

    .. math::
        \\ket{a} \\, \\ket{b} \\, \\ket{0} \\mapsto \\ket{a} \\, \\ket{a + b} \\, \\ket{0} \\, .

    It acts on :math:`3n + 2 + m` qubits, with :math:`m = n - w(n) - \\lfloor \\log_{2} n \\rfloor`
    and :math:`w(n)` being the number of ones in the binary representation of :math:`n`. The
    qubits are split into three registers:

    1. the first :math:`n` qubits hold :math:`a`, which is left unchanged;
    2. the next :math:`n + 1` qubits hold :math:`b`, which is replaced by :math:`a + b`. The last
       qubit of this register is the overflow bit, which must be in the state :math:`\\ket{0}`
       at the beginning;
    3. the last :math:`n + 1 + m` qubits are auxiliary qubits. They must be in the state
       :math:`\\ket{0}` at the beginning, and they are returned to it at the end.

    In every register, the :math:`k`-th qubit holds the bit :math:`2^{k}`, *i.e.* the least
    significant bit comes first. The carries are computed with a tree of
    :class:`qibo.gates.TOFFOLI` gates, so that the depth of the circuit is logarithmic in
    :math:`n`. The sum is first written in the auxiliary qubits by an out-of-place adder. Then,
    the inverse of the same adder, applied to the complement of the sum, erases :math:`b`, and
    the registers are swapped.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import DKRS

            nbits = 2
            adder = DKRS(nbits)

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
        :class:`qibo.models.circuit.Circuit`: implementation of the DKRS adder.

    References:
        1. T. G. Draper, S. A. Kutin, E. M. Rains, and K. M. Svore, *A logarithmic-depth quantum
        carry-lookahead adder*,
        `Quantum Inf. Comput. 6, 351 (2006) <https://doi.org/10.26421/QIC6.4-5-4>`_.
    """
    nbits = _check_nbits(nbits)

    # floor of the logarithm in base 2 of nbits, and number of qubits for the propagate products
    nlevels = nbits.bit_length() - 1
    nproducts = nbits - nbits.bit_count() - nlevels

    # out-of-place adder, which writes the sum of a and y in the n + 1 qubits of s. The qubits
    # are a, y, s, and the ones that hold the products of the propagate bits
    register_a = list(range(nbits))
    register_y = list(range(nbits, 2 * nbits))
    register_s = list(range(2 * nbits, 3 * nbits + 1))
    register_products = list(range(3 * nbits + 1, 3 * nbits + 1 + nproducts))

    lookahead = Circuit(3 * nbits + 1 + nproducts)

    # generate bits g_k = a_k y_k are added to s_{k + 1}, and the qubits of y hold the propagate
    # bits p_k = a_k + y_k, except for the first one, which does not have an incoming carry
    lookahead.add(
        gates.TOFFOLI(register_a[k], register_y[k], register_s[k + 1])
        for k in range(nbits)
    )
    lookahead.add(gates.CNOT(register_a[k], register_y[k]) for k in range(1, nbits))

    # products[t][m] holds the product of the propagate bits of the m-th block of 2^t bits
    products = [{m: register_y[m] for m in range(1, nbits)}]
    available = iter(register_products)
    tree = Circuit(lookahead.nqubits)
    for t in range(1, nlevels):
        products.append({m: next(available) for m in range(1, nbits // 2**t)})
        for m, target in products[t].items():
            tree.add(
                gates.TOFFOLI(
                    products[t - 1][2 * m], products[t - 1][2 * m + 1], target
                )
            )
    lookahead.add(tree)

    # the carry out of each block of 2^t bits is accumulated in s, and then the carries into
    # the remaining blocks are obtained, from the largest blocks to the smallest ones
    for t in range(1, nlevels + 1):
        for m in range(nbits // 2**t):
            lookahead.add(
                gates.TOFFOLI(
                    register_s[m * 2**t + 2 ** (t - 1)],
                    products[t - 1][2 * m + 1],
                    register_s[(m + 1) * 2**t],
                )
            )

    for t in range((2 * nbits // 3).bit_length() - 1, 0, -1):
        for m in range(1, (nbits - 2 ** (t - 1)) // 2**t + 1):
            lookahead.add(
                gates.TOFFOLI(
                    register_s[2**t * m],
                    products[t - 1][2 * m],
                    register_s[2**t * m + 2 ** (t - 1)],
                )
            )

    # the products are erased, and the sum bits are the carries plus the propagate bits
    lookahead.add(tree.invert())
    lookahead.add(gates.CNOT(register_a[0], register_s[0]))
    lookahead.add(gates.CNOT(register_y[0], register_s[0]))
    lookahead.add(gates.CNOT(register_y[k], register_s[k]) for k in range(1, nbits))
    lookahead.add(gates.CNOT(register_a[k], register_y[k]) for k in range(1, nbits))

    register_b = list(range(nbits, 2 * nbits + 1))
    register_sum = list(range(2 * nbits + 1, 3 * nbits + 2))
    register_aux = list(range(3 * nbits + 2, 3 * nbits + 2 + nproducts))

    circuit = Circuit(3 * nbits + 2 + nproducts, **kwargs)

    # sum of a and b in the auxiliary qubits
    circuit.add(
        lookahead.on_qubits(
            *register_a, *register_b[:nbits], *register_sum, *register_aux
        )
    )

    # the sum s is the sum of a and b, so the complement of b is the sum of a and the
    # complement of s. This is undone by the inverse of the adder, which erases b
    circuit.add(gates.X(qubit) for qubit in register_b[:nbits])
    circuit.add(gates.X(qubit) for qubit in register_sum[:nbits])
    circuit.add(gates.CNOT(register_sum[-1], register_b[-1]))
    circuit.add(
        lookahead.invert().on_qubits(
            *register_a, *register_sum[:nbits], *register_b, *register_aux
        )
    )
    circuit.add(gates.X(qubit) for qubit in register_sum[:nbits])

    circuit.add(gates.SWAP(*qubits) for qubits in zip(register_b, register_sum))

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


def Gidney(nbits: int, **kwargs) -> Circuit:
    """Creates a circuit that implements the Gidney adder.

    Given two :math:`n`-bit non-negative integers :math:`a` and :math:`b`, where :math:`n` is
    ``nbits``, the circuit performs

    .. math::
        \\ket{a} \\, \\ket{b} \\, \\ket{0} \\mapsto \\ket{a} \\, \\ket{a + b} \\, \\ket{0} \\, .

    It acts on :math:`3n` qubits, which are split into three registers:

    1. the first :math:`n` qubits hold :math:`a`, which is left unchanged;
    2. the next :math:`n + 1` qubits hold :math:`b`, which is replaced by :math:`a + b`. The last
       qubit of this register is the overflow bit, which must be in the state :math:`\\ket{0}`
       at the beginning;
    3. the last :math:`n - 1` qubits are auxiliary qubits that store the carries. They must be in
       the state :math:`\\ket{0}` at the beginning, and they are returned to it at the end.

    In every register, the :math:`k`-th qubit holds the bit :math:`2^{k}`, *i.e.* the least
    significant bit comes first. Each carry is computed in a qubit in the state :math:`\\ket{0}`
    with a temporary logical-AND, which has four :class:`qibo.gates.T` or
    :class:`qibo.gates.TDG` gates. Then, it is erased without them: the qubit is measured in
    the :math:`X` basis, and, if the outcome is :math:`1`, a :class:`qibo.gates.CZ` gate, which is
    applied with a :class:`qibo.gates.CU1` conditioned on the outcome, fixes the phase. Hence,
    the circuit has :math:`4n` of these gates, and :math:`n - 1` measurements with ``collapse``
    set to ``True``. Because of them, each shot has to be simulated separately, and the circuit
    cannot be inverted.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import Gidney

            nbits = 2
            adder = Gidney(nbits)

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
        :class:`qibo.models.circuit.Circuit`: implementation of the Gidney adder.

    References:
        1. C. Gidney, *Halving the cost of quantum addition*,
        `Quantum 2, 74 (2018) <https://doi.org/10.22331/q-2018-06-18-74>`_.
    """
    nbits = _check_nbits(nbits)

    register_a = list(range(nbits))
    register_b = list(range(nbits, 2 * nbits + 1))
    register_carry = list(range(2 * nbits + 1, 3 * nbits))

    # carries[k] is the qubit of the carry c_k, and the last carry is the overflow qubit
    carries = [None]
    carries.extend(register_carry)
    carries.append(register_b[-1])

    # temporary logical-AND acts on the qubits (x, y, target), and it takes the target from the
    # state 0 to the state x y
    temporary_and = Circuit(3)
    temporary_and.add(gates.H(2))
    temporary_and.add(gates.T(2))
    temporary_and.add(gates.CNOT(0, 2))
    temporary_and.add(gates.CNOT(1, 2))
    temporary_and.add(gates.CNOT(2, 0))
    temporary_and.add(gates.CNOT(2, 1))
    temporary_and.add(gates.TDG(0))
    temporary_and.add(gates.TDG(1))
    temporary_and.add(gates.T(2))
    temporary_and.add(gates.CNOT(2, 0))
    temporary_and.add(gates.CNOT(2, 1))
    temporary_and.add(gates.H(2))
    temporary_and.add(gates.S(2))

    # the carry c_{k + 1} is the majority of a_k, b_k and c_k, and it is computed as
    # c_k + (a_k + c_k) (b_k + c_k), where the first carry c_0 is zero
    circuit = Circuit(3 * nbits, **kwargs)
    for k in range(nbits):
        if k > 0:
            circuit.add(gates.CNOT(carries[k], register_a[k]))
            circuit.add(gates.CNOT(carries[k], register_b[k]))

        circuit.add(
            temporary_and.on_qubits(register_a[k], register_b[k], carries[k + 1])
        )

        if k > 0:
            circuit.add(gates.CNOT(carries[k], carries[k + 1]))

    # the carries, except for the last one, are erased by measurements, and the sum bits are
    # written in the qubits of b
    for k in reversed(range(nbits)):
        if k < nbits - 1:
            if k > 0:
                circuit.add(gates.CNOT(carries[k], carries[k + 1]))

            circuit.add(gates.H(carries[k + 1]))
            outcome = circuit.add(gates.M(carries[k + 1], collapse=True))
            phase = math.pi * outcome.symbols[0]
            circuit.add(gates.CU1(register_a[k], register_b[k], theta=phase))
            circuit.add(gates.RX(carries[k + 1], theta=phase))

        if k > 0:
            circuit.add(gates.CNOT(carries[k], register_a[k]))

        circuit.add(gates.CNOT(register_a[k], register_b[k]))

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
    carries = list(register_carry)
    carries.extend(register_b[-1:])

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
    if isinstance(nbits, bool) or not isinstance(nbits, int) or nbits < 1:
        raise_error(ValueError, f"nbits must be a positive integer, but it is {nbits}.")

    return int(nbits)
