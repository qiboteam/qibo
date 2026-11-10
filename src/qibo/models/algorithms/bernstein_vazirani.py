from qibo.models.algorithms.deutsch_jozsa import DeutschJozsa
from qibo.models.circuit import Circuit


def BernsteinVazirani(oracle: Circuit, measure: bool = True, **kwargs) -> Circuit:
    """Creates a circuit that implements the Bernstein-Vazirani algorithm.

    Given a Boolean function :math:`f(x) = s \\cdot x \\oplus c`, where
    :math:`s \\in \\{0, \\, 1\\}^{n}` is a hidden string, :math:`\\cdot` is the bitwise inner
    product modulo :math:`2`, and :math:`c \\in \\{0, \\, 1\\}`, the algorithm finds :math:`s`
    with a single query to the oracle
    :math:`U_{f} \\ket{x} \\ket{y} = \\ket{x} \\ket{y \\oplus f(x)}`, where :math:`\\oplus` is the
    sum modulo :math:`2`. The circuit acts on :math:`n + 1` qubits,
    with the first :math:`n` being the input qubits and the last one the output qubit of the
    oracle. It is the same circuit as the one of :func:`qibo.models.DeutschJozsa`:
    the output qubit is in the state :math:`\\ket{-}`, so that the oracle multiplies
    :math:`\\ket{x}` by :math:`(-1)^{s \\cdot x}`, which is undone by the Hadamard gates on the
    input qubits, that are left in the state :math:`\\ket{s}`. If ``measure=True``, the qubit
    :math:`k` of the measured bitstring is the bit :math:`s_{k}`.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import BernsteinVazirani

            # oracle of the function with the hidden string s = 101
            oracle = Circuit(4)
            oracle.add(gates.CNOT(0, 3))
            oracle.add(gates.CNOT(2, 3))

            circuit = BernsteinVazirani(oracle)
            result = circuit(nshots=10)

            print(result.frequencies())

        .. testoutput::

            Counter({'101': 10})

    Args:
        oracle (:class:`qibo.models.circuit.Circuit`): circuit that implements :math:`U_{f}`,
            which has :math:`n + 1` qubits, with the output qubit being the last one. It
            cannot contain measurements.
        measure (bool, optional): If ``True``, adds a measurement of the input qubits at
            the end of the circuit. Defaults to ``True``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: implementation of the Bernstein-Vazirani
        algorithm.

    References:
        1. E. Bernstein and U. Vazirani, *Quantum complexity theory*,
        `SIAM J. Comput. 26, 1411 (1997) <https://doi.org/10.1137/S0097539796300921>`_.

        2. R. Cleve, A. Ekert, C. Macchiavello, and M. Mosca, *Quantum algorithms revisited*,
        `Proc. R. Soc. Lond. A 454, 339 (1998) <https://doi.org/10.1098/rspa.1998.0164>`_.
    """
    return DeutschJozsa(oracle, measure=measure, **kwargs)
