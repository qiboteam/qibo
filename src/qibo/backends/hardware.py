from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error


class IcarusQBackend(NumpyBackend):

    description = "Uses QPU controlled by the IcarusQ FPGA."

    def __init__(self):
        super().__init__()
        self.name = "icarusq"
        self.custom_gates = True
        import qiboicarusq
        from qiboicarusq import gates
        from qiboicarusq.circuit import HardwareCircuit
        self.hardware_module = qiboicarusq
        self.hardware_gates = qiboicarusq.gates
        self.hardware_circuit = qiboicarusq.circuit.HardwareCircuit

    def create_einsum_cache(self, qubits, nqubits, ncontrol=None): # pragma: no cover
        raise_error(NotImplementedError, "`create_einsum_cache` method is "
                                         "not required for hardware backends.")

    def einsum_call(self, cache, state, matrix): # pragma: no cover
        raise_error(NotImplementedError, "`einsum_call` method is not required "
                                         "for hardware backends.")
