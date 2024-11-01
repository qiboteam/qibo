import math

from qibo import gates
from qibo.config import raise_error
from qibo.models.circuit import Circuit


def QFT(nqubits: int, with_swaps: bool = True, accelerators=None, **kwargs) -> Circuit:
    """Creates a circuit that implements the Quantum Fourier Transform.

    Args:
        nqubits (int): number of qubits in the circuit.
        with_swaps (bool, optional): If ``True``, uses :class:`qibo.gates.SWAP` gates
            at the end of the circuit so that the qubit order in the final
            state is the same as the initial state. Defauts to ``True``.
        accelerators (dict, optional): Accelerator device dictionary in order to use a
            distributed circuit. If ``None``, a simple (non-distributed)
            circuit will be used.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.Circuit`. For details, see the documentation
            of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.Circuit`: implementation of the Quantum Fourier Transform.

    Example:
        .. testcode::

            import numpy as np
            from qibo.models import QFT
            nqubits = 6
            circuit = QFT(nqubits)
            # Random normalized initial state vector
            init_state = np.random.random(2 ** nqubits) + 1j * np.random.random(2 ** nqubits)
            init_state = init_state / np.sqrt((np.abs(init_state)**2).sum())
            # Execute the circuit
            final_state = circuit(init_state)
    """
    if accelerators is not None:
        if not with_swaps:
            raise_error(
                NotImplementedError,
                "Distributed QFT is only implemented " "with SWAPs.",
            )
        return _DistributedQFT(nqubits, accelerators, **kwargs)

    circuit = Circuit(nqubits, **kwargs)
    for i1 in range(nqubits):
        circuit.add(gates.H(i1))
        for i2 in range(i1 + 1, nqubits):
            theta = math.pi / 2 ** (i2 - i1)
            circuit.add(gates.CU1(i2, i1, theta))

    if with_swaps:
        for i in range(nqubits // 2):
            circuit.add(gates.SWAP(i, nqubits - i - 1))

    return circuit


def _DistributedQFT(nqubits, accelerators=None, **kwargs):
    """QFT with the order of gates optimized for reduced multi-device communication."""
    circuit = Circuit(nqubits, accelerators, **kwargs)
    icrit = nqubits // 2 + nqubits % 2
    if accelerators is not None:
        circuit.global_qubits = range(circuit.nlocal, nqubits)  # pylint: disable=E1101
        if icrit < circuit.nglobal:  # pylint: disable=E1101
            raise_error(
                NotImplementedError,
                f"Cannot implement QFT for {nqubits} qubits "
                + f"using {circuit.nglobal} global qubits.",
            )  # pylint: disable=E1101

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
