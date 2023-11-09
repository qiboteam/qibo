from dataclasses import dataclass
from itertools import product

import numpy as np

from qibo import Circuit
from qibo.backends import CliffordBackend
from qibo.result import CircuitResult


def _string_product(A, B):
    phase_A = False if A[0] == "-" else True
    phase_B = False if B[0] == "-" else True
    phase = phase_A ^ phase_B
    prod = []
    for a, b in zip(A.replace("-", ""), B.replace("-", "")):
        if a == "I":
            prod.append(b)
        elif b == "I":
            prod.append(a)
        else:
            prod.append("".join([a, b]))
    result = "-" if phase else ""
    return f"{result}({')('.join(prod)})"


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
        generators, phases = self._backend.tableau_to_generators(
            self.tableau, return_array
        )
        return generators[self.nqubits :], phases[self.nqubits :]

    def get_destabilizers_generators(self, return_array=False):
        generators, phases = self._backend.tableau_to_generators(
            self.tableau, return_array
        )
        return generators[: self.nqubits], phases[: self.nqubits]

    def _construct_operators(self, generators, phases, is_array=False):
        if is_array:
            operators = np.array(generators) * phases.reshape(-1, 1, 1)
        else:
            operators = generators.copy()
            for i in (phases == -1).nonzero()[0]:
                operators[i] = f"-{operators[i]}"
        identity = (
            np.eye(2**self.nqubits)
            if is_array
            else "".join(["I" for _ in range(self.nqubits)])
        )
        operators = [[g, identity] for g in operators]
        if is_array:
            return [g1 @ g2 for g1, g2 in product(*operators)]
        return [_string_product(g1, g2) for g1, g2 in product(*operators)]

    def get_stabilizers(self, return_array=False):
        generators, phases = self.get_stabilizers_generators(return_array)
        return self._construct_operators(generators, phases, return_array)

    def get_destabilizers(self):
        generators = self.get_destabilizers_generators(True)
        return self._construct_operators(generators, phases, return_array)

    def state(self):
        stabilizers = self.get_stabilizers(True)
        return np.sum(stabilizers, 0) / len(stabilizers)
