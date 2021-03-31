import math
from qibo.config import raise_error
from qibo.core.circuit import Circuit as StateCircuit
from typing import Dict, Optional


class Circuit(StateCircuit):
    """"""

    @classmethod
    def _constructor(cls, *args, **kwargs):
        if kwargs["density_matrix"]:
            if kwargs["accelerators"] is not None:
                raise_error(NotImplementedError,
                            "Distributed circuits are not implemented for "
                            "density matrices.")
            from qibo.core.circuit import DensityMatrixCircuit as circuit_cls
            kwargs = {}
        elif kwargs["accelerators"] is None:
            circuit_cls = StateCircuit
            kwargs = {}
        else:
            try:
                from qibo.tensorflow.distcircuit import DistributedCircuit
            except ModuleNotFoundError: # pragma: no cover
                # CI installs all required libraries by default
                raise_error(ModuleNotFoundError,
                            "Cannot create distributed circuit because some "
                            "required libraries are missing.")
            circuit_cls = DistributedCircuit
            kwargs.pop("density_matrix")
        return circuit_cls, args, kwargs

    def __new__(cls, nqubits: int,
                accelerators: Optional[Dict[str, int]] = None,
                memory_device: str = "/CPU:0",
                density_matrix: bool = False):
        circuit_cls, args, kwargs = cls._constructor(
                  nqubits, accelerators=accelerators,
                  memory_device=memory_device,
                  density_matrix=density_matrix
                )
        return circuit_cls(*args, **kwargs)

    @classmethod
    def from_qasm(cls, qasm_code: str,
                  accelerators: Optional[Dict[str, int]] = None,
                  memory_device: str = "/CPU:0",
                  density_matrix: bool = False):
      circuit_cls, args, kwargs = cls._constructor(
                qasm_code, accelerators=accelerators,
                memory_device=memory_device,
                density_matrix=density_matrix
              )
      return circuit_cls.from_qasm(*args, **kwargs)


def QFT(nqubits: int, with_swaps: bool = True,
        accelerators: Optional[Dict[str, int]] = None,
        memory_device: str = "/CPU:0") -> Circuit:
    """Creates a circuit that implements the Quantum Fourier Transform.

    Args:
        nqubits (int): Number of qubits in the circuit.
        with_swaps (bool): Use SWAP gates at the end of the circuit so that the
            qubit order in the final state is the same as the initial state.
        accelerators (dict): Accelerator device dictionary in order to use a
            distributed circuit
            If ``None`` a simple (non-distributed) circuit will be used.
        memory_device (str): Device to use for memory in case a distributed circuit
            is used. Ignored for non-distributed circuits.

    Returns:
        A qibo.models.Circuit that implements the Quantum Fourier Transform.

    Example:
        ::

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
            raise_error(NotImplementedError, "Distributed QFT is only implemented "
                                             "with SWAPs.")
        return _DistributedQFT(nqubits, accelerators, memory_device)

    from qibo import gates

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


def _DistributedQFT(nqubits: int,
                    accelerators: Optional[Dict[str, int]] = None,
                    memory_device: str = "/CPU:0"):
    """QFT with the order of gates optimized for reduced multi-device communication."""
    from qibo import gates

    circuit = Circuit(nqubits, accelerators, memory_device)
    icrit = nqubits // 2 + nqubits % 2
    if accelerators is not None:
        circuit.global_qubits = range(circuit.nlocal, nqubits) # pylint: disable=E1101
        if icrit < circuit.nglobal: # pylint: disable=E1101
            raise_error(NotImplementedError, "Cannot implement QFT for {} qubits "
                                             "using {} global qubits."
                                             "".format(nqubits, circuit.nglobal)) # pylint: disable=E1101

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
