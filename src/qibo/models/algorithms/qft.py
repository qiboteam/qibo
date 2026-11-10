import math

from qibo import gates
from qibo.config import raise_error
from qibo.models.circuit import Circuit


def QFT(nqubits: int, with_swaps: bool = True, accelerators=None, **kwargs) -> Circuit:
    """Creates a circuit that implements the Quantum Fourier Transform.

    The Quantum Fourier Transform (QFT) on :math:`n` qubits is the unitary operator that acts on
    the computational basis states as

    .. math::
        \\text{QFT} \\, \\ket{j} = \\frac{1}{\\sqrt{N}} \\sum_{k=0}^{N-1}
        e^{2 \\pi i j k / N} \\ket{k} \\, ,

    where :math:`N = 2^{n}`, *i.e.* it applies the discrete Fourier transform to the amplitudes
    of a state. The integer :math:`j = j_{1} 2^{n - 1} + \\cdots + j_{n} 2^{0}` is represented by
    the bitstring :math:`j_{1} \\cdots j_{n}` of the qubits, with the first qubit being the most
    significant one. The circuit is the one in Fig. 5.1 of Ref. [1]. For each qubit
    :math:`q = 0, \\dots, n - 1`, from the first to the last, it applies

    1. a Hadamard gate on the qubit :math:`q`;
    2. the controlled-:math:`R_{k}` gates, implemented with :class:`qibo.gates.CU1`, that have
       the qubit :math:`q + k - 1` as control and the qubit :math:`q` as target, for
       :math:`k = 2, \\dots, n - q`, where

       .. math::
           R_{k} = \\begin{pmatrix} 1 & 0 \\\\ 0 & e^{2 \\pi i / 2^{k}} \\end{pmatrix} \\, .

    If ``with_swaps=True``, :math:`\\lfloor n / 2 \\rfloor` :class:`qibo.gates.SWAP` gates are
    added at the end to reverse the order of the qubits. Hence, the circuit has
    :math:`n (n + 1) / 2` Hadamard and controlled phase gates, plus the swaps, which is
    :math:`\\Theta(n^{2})` gates. The inverse QFT is obtained with
    :meth:`qibo.models.circuit.Circuit.invert`.

    Example:
        .. testcode::

            import numpy as np

            from qibo.backends import NumpyBackend
            from qibo.models import QFT
            from qibo.quantum_info import random_statevector

            backend = NumpyBackend()

            nqubits = 6
            dims = 2**nqubits

            # Random normalized initial state vector
            initial_state = random_statevector(dims, backend=backend)

            circuit = QFT(nqubits)

            # Execute the circuit
            final_state = circuit(initial_state)

    Args:
        nqubits (int): number of qubits in the circuit.
        with_swaps (bool, optional): If ``True``, uses :class:`qibo.gates.SWAP` gates
            at the end of the circuit so that the qubit order in the final
            state is the same as the initial state. Otherwise, the qubits of the final state are
            in the reverse order. Defaults to ``True``.
        accelerators (dict, optional): Accelerator device dictionary in order to use a
            distributed circuit. If ``None``, a simple (non-distributed)
            circuit will be used.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.Circuit`. For details, see the documentation
            of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.Circuit`: implementation of the Quantum Fourier Transform.

    References:
        1. M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*,
        `Cambridge University Press (2010), Sec. 5.1 <https://doi.org/10.1017/CBO9780511976667>`_.
    """
    if accelerators is not None:
        if not with_swaps:
            raise_error(
                NotImplementedError,
                "Distributed QFT is only implemented with SWAPs.",
            )
        return _DistributedQFT(nqubits, accelerators, **kwargs)

    circuit = Circuit(nqubits, **kwargs)
    for i1 in range(nqubits):
        circuit.add(gates.H(i1))
        for i2 in range(i1 + 1, nqubits):
            theta = math.pi / 2 ** (i2 - i1)
            circuit.add(gates.CU1(i2, i1, theta))

    if with_swaps:
        for qubit in range(nqubits // 2):
            circuit.add(gates.SWAP(qubit, nqubits - qubit - 1))

    return circuit


def _DistributedQFT(nqubits, accelerators=None, **kwargs):
    """QFT with the order of gates optimized for reduced multi-device communication."""
    circuit = Circuit(nqubits, accelerators, **kwargs)
    icrit = nqubits // 2 + nqubits % 2
    if accelerators is not None:
        circuit.global_qubits = range(circuit.nlocal, nqubits)
        if icrit < circuit.nglobal:
            raise_error(
                NotImplementedError,
                f"Cannot implement QFT for {nqubits} qubits "
                + f"using {circuit.nglobal} global qubits.",
            )

    for i1 in range(nqubits):
        if i1 < icrit:
            i1eff = i1
        else:
            i1eff = nqubits - i1 - 1
            circuit.add(gates.SWAP(i1, i1eff))

        circuit.add(gates.H(i1eff))
        for i2 in range(i1 + 1, nqubits):
            theta = math.pi / 2 ** (i2 - i1)
            circuit.add(gates.CU1(i2, i1eff, theta))

    return circuit
