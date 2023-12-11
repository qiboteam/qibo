import dask
import dask.array as da
from dask.distributed import Client

from qibo.backends.numpy import NumpyBackend


class DarrayBackend(NumpyBackend):
    def __init__(self, client=None):
        super().__init__()
        self.name = "darray"
        self.client = client
        self.versions["dask"] = dask.__version__

    def set_device(self, device: Client):
        self.client = device

    def zero_state(self, nqubits: int):
        state = da.zeros(2**nqubits, dtype=self.dtype)
        state[0] = 1
        return state

    def zero_density_matrix(self, nqubits: int):
        state = da.zeros(2 * (2**nqubits,), dtype=self.dtype)
        state[0, 0] = 1
        return state

    def execute_circuit(self, circuit, initial_state=None, nshots=1000):
        if initial_state is not None:
            raise ValueError("Initial state not supported by darray backend")

        super().execute_circuit(circuit, nshots=nshots)
        # pylint: disable=E1101
        self.client.compute(self._final_state._state)
