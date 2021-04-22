from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error


class HardwareBackend(NumpyBackend):

    def create_einsum_cache(self, qubits, nqubits, ncontrol=None): # pragma: no cover
        raise_error(NotImplementedError, "`create_einsum_cache` method is "
                                         "not required for hardware backends.")

    def einsum_call(self, cache, state, matrix): # pragma: no cover
        raise_error(NotImplementedError, "`einsum_call` method is not required "
                                         "for hardware backends.")


class IcarusQBackend(HardwareBackend):

    description = "Uses QPU controlled by the IcarusQ FPGA."

    def __init__(self):
        super().__init__()
        from qibo.hardware.experiments import IcarusQ
        from qibo.hardware.scheduler import TaskScheduler
        self.name = "icarusq"
        self.custom_gates = True
        self.experiment = IcarusQ()
        self.scheduler = TaskScheduler()
