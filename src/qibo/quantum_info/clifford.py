from dataclasses import dataclass
from itertools import permutations

import numpy as np

from qibo import Circuit
from qibo.backends import CliffordBackend
from qibo.result import CircuitResult


@dataclass
class Clifford:
    _input: Circuit | CircuitResult
    _backend: CliffordBackend = CliffordBackend()

    tableau: np.ndarray = None
    nqubits: int = None

    def __post_init__(self):
        if isinstance(self._input, CircuitResult):
            self.tableau = self._input.state(numpy=True)
            self.nqubits = int((self.tableau.shape[1] - 1) / 2)
        else:
            self.nqubits = self._input.nqubits

    @classmethod
    def run(
        cls, circuit: Circuit, initial_state: np.ndarray = None, nshots: int = 1000
    ):
        result = cls._backend.execute_circuit(_input, initial_state, nshots)
        return cls(result)

    def get_stabilizers_generators(self, return_array=False):
        generators = self._backend.tableau_to_generators(self.tableau, return_array)
        return generators[self.nqubits :]

    def get_destabilizers_generators(self, return_array=False):
        generators = self._backend.tableau_to_generators(self.tableau, return_array)
        return generators[: self.nqubits]

    def get_stabilizers(self):
        generators = self.get_stabilizers_generators(True)
        stabilizers = []
        for P1, P2 in permutations(generators, 2):
            stabilizers.append(P1 @ P2)
        return generators + stabilizers

    def get_destabilizers(self):
        generators = self.get_destabilizers_generators(True)
        destabilizers = []
        for P1, P2 in permutations(generators, 2):
            destabilizers.append(P1 @ P2)
        return generators + destabilizers

    def state(self):
        stabilizers = self.get_stabilizers()
        return np.sum(stabilizers, 0) / len(stabilizers)
