from typing import Optional

import dask
import dask.array as da
from dask.distributed import Client

from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error
from qibo.result import CircuitResult, QuantumState


class DarrayBackend(NumpyBackend):
    """Lazy and distributed execution.

    This backend will not convert the result of the simulation to NumPy format, since
    this might request more memory than available.
    In order to obtain an in-memory array for the result, :meth:`to_numpy` has to be
    called explicitly.
    """

    def __init__(self, client: Optional[Client] = None):
        super().__init__()
        self.name = "darray"
        self.versions["dask"] = dask.__version__
        # acquire a default client, if no one has been passed, to avoid reinstantiating
        # it every time
        self.client = client if client is not None else Client()

    def set_device(self, device: Client):
        self.client = device

    def to_numpy(self, x):
        return self.client.compute(x)

    def zero_state(self, nqubits):
        state = da.zeros(2**nqubits, dtype=self.dtype)
        state[0] = 1
        return state

    def zero_density_matrix(self, nqubits):
        state = da.zeros(2 * (2**nqubits,), dtype=self.dtype)
        state[0, 0] = 1
        return state

    def execute_circuit(self, circuit, initial_state=None, nshots=1000):
        if isinstance(initial_state, type(circuit)):
            if not initial_state.density_matrix == circuit.density_matrix:
                raise_error(
                    ValueError,
                    f"""Cannot set circuit with density_matrix {initial_state.density_matrix} as
                      initial state for circuit with density_matrix {circuit.density_matrix}.""",
                )
            elif (
                not initial_state.accelerators == circuit.accelerators
            ):  # pragma: no cover
                raise_error(
                    ValueError,
                    f"""Cannot set circuit with accelerators {initial_state.density_matrix} as
                      initial state for circuit with accelerators {circuit.density_matrix}.""",
                )
            else:
                return self.execute_circuit(initial_state + circuit, None, nshots)

        if circuit.repeated_execution:
            if circuit.measurements or circuit.has_collapse:
                return self.execute_circuit_repeated(circuit, nshots, initial_state)
            else:
                raise RuntimeError(
                    "Attempting to perform noisy simulation with `density_matrix=False` and no Measurement gate in the Circuit. If you wish to retrieve the statistics of the outcomes please include measurements in the circuit, otherwise set `density_matrix=True` to recover the final state."
                )

        nqubits = circuit.nqubits

        if circuit.density_matrix:
            if initial_state is None:
                state = self.zero_density_matrix(nqubits)
            else:
                # cast to proper complex type
                state = self.cast(initial_state)

            state = da.array(state)
            for gate in circuit.queue:
                state = gate.apply_density_matrix(self, state, nqubits)

        else:
            if initial_state is None:
                state = self.zero_state(nqubits)
            else:
                # cast to proper complex type
                state = self.cast(initial_state)

            state = da.array(state)
            for gate in circuit.queue:
                state = gate.apply(self, state, nqubits)

        # trigger the actual computation
        state = self.client.persist(state)

        if circuit.measurements:
            circuit._final_state = CircuitResult(
                state, circuit.measurements, backend=self, nshots=nshots
            )
        else:
            circuit._final_state = QuantumState(state, backend=self)

        return circuit._final_state
