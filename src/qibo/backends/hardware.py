from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error


class IcarusQBackend(NumpyBackend):

    description = "Uses QPU controlled by the IcarusQ FPGA."

    def __init__(self):
        super().__init__()
        import qiboicarusq
        self.name = "icarusq"
        self.custom_gates = True
        self.hardware_module = qiboicarusq

    def create_einsum_cache(self, qubits, nqubits, ncontrol=None): # pragma: no cover
        raise_error(NotImplementedError, "`create_einsum_cache` method is "
                                         "not required for hardware backends.")

    def einsum_call(self, cache, state, matrix): # pragma: no cover
        raise_error(NotImplementedError, "`einsum_call` method is not required "
                                         "for hardware backends.")
