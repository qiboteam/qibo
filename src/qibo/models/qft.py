import math

from qibo import gates
from qibo.config import raise_error
from qibo.models.circuit import Circuit


def QFT(nqubits, with_swaps=True, accelerators=None) -> Circuit:
    """Creates a circuit that implements the Quantum Fourier Transform.

    Args:
        nqubits (int): Number of qubits in the circuit.
        with_swaps (bool): Use SWAP gates at the end of the circuit so that the
            qubit order in the final state is the same as the initial state.
        accelerators (dict): Accelerator device dictionary in order to use a
            distributed circuit
            If ``None`` a simple (non-distributed) circuit will be used.

    Returns:
        A qibo.models.Circuit that implements the Quantum Fourier Transform.

    Example:
        .. testcode::

            import numpy as np
            from qibo.models import QFT
            nqubits = 6
            c = QFT(nqubits)
            # Random normalized initial state vector
            init_state = np.random.random(2 ** nqubits) + 1j * np.random.random(2 ** nqubits)
            init_state = init_state / np.sqrt((np.abs(init_state)**2).sum())
            # Execute the circuit
            final_state = c(init_state)
    """
    if accelerators is not None:
        if not with_swaps:
            raise_error(
                NotImplementedError,
                "Distributed QFT is only implemented " "with SWAPs.",
            )
        return _DistributedQFT(nqubits, accelerators)

    circuit = Circuit(nqubits)
    for i1 in range(nqubits):
        circuit.add(gates.H(i1))
        for i2 in range(i1 + 1, nqubits):
            theta = math.pi / 2 ** (i2 - i1)
            circuit.add(gates.CU1(i2, i1, theta))

    if with_swaps:
        for i in range(nqubits // 2):
            circuit.add(gates.SWAP(i, nqubits - i - 1))

    return circuit


def _DistributedQFT(nqubits, accelerators=None):
    """QFT with the order of gates optimized for reduced multi-device communication."""
    circuit = Circuit(nqubits, accelerators)
    icrit = nqubits // 2 + nqubits % 2
    if accelerators is not None:
        circuit.global_qubits = range(circuit.nlocal, nqubits)  # pylint: disable=E1101
        if icrit < circuit.nglobal:  # pylint: disable=E1101
            raise_error(
                NotImplementedError,
                "Cannot implement QFT for {} qubits "
                "using {} global qubits."
                "".format(nqubits, circuit.nglobal),
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
