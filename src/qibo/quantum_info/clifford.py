from dataclasses import dataclass

import numpy as np

from qibo import Circuit
from qibo.backends import CliffordBackend
from qibo.result import CircuitResult


@dataclass
class Clifford:
    _input: Circuit | CircuitResult
    _backend: CliffordBackend = CliffordBackend()

    tableau: np.ndarray = None

    def __post_init__(self):
        if isinstance(self._input, CircuitResult):
            self.tableau = _input.state(numpy=True)

    @classmethod
    def run(
        cls, circuit: Circuit, initial_state: np.ndarray = None, nshots: int = 1000
    ):
        result = cls._backend.execute_circuit(_input, initial_state, nshots)
        return cls(result)

    def get_stabilizers_generators(self, return_array=False):
        return self._backend.tableau_to_generators(
            self.tableau, "stabilizers", return_array
        )

    def get_destabilizers_generators(self, return_array=False):
        return self._backend.tableau_to_generators(
            self.tableau, "destabilizers", return_array
        )
