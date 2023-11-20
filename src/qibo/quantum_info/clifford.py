from dataclasses import dataclass
from functools import reduce
from itertools import product

import numpy as np

from qibo import Circuit
from qibo.backends import CliffordBackend
from qibo.config import raise_error
from qibo.gates import M
from qibo.measurements import frequencies_to_binary


def _string_product(operators):
    # calculate global sign
    phases = np.array(["-" in op for op in operators], dtype=bool)
    # remove the - signs
    operators = "|".join(operators).replace("-", "").split("|")
    prod = []
    for op in zip(*operators):
        tmp = "".join([o for o in op if o != "I"])
        if tmp == "":
            tmp = "I"
        prod.append(tmp)
    result = "-" if len(phases.nonzero()[0]) % 2 == 1 else ""
    return f"{result}({')('.join(prod)})"


def _list_of_matrices_product(operators):
    return reduce(np.matmul, operators)  # faster
    # return np.einsum(*[d for i, op in enumerate(operators) for d in (op, (i, i + 1))])


@dataclass
class Clifford:
    _input: Circuit | np.ndarray
    _backend: CliffordBackend = CliffordBackend()
    _samples = None

    tableau: np.ndarray = None
    measurements: list = None
    nqubits: int = None
    nshots: int = 1000

    def __post_init__(self):
        if isinstance(self._input, np.ndarray):
            self.tableau = self._input
            self.nqubits = int((self.tableau.shape[1] - 1) / 2)
            if self.measurements is None:
                self.measurements = [M(*range(self.nqubits))]
        else:
            self.nqubits = self._input.nqubits
            self.measurements = self._input.measurements
        if self.has_samples():
            self._samples = np.hstack([m.result.samples() for m in self.measurements])

    @classmethod
    def run(
        cls, circuit: Circuit, initial_state: np.ndarray = None, nshots: int = 1000
    ):
        self.nshots = nshots
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
        operators = [(g, identity) for g in operators]
        if is_array:
            return [_list_of_matrices_product(ops) for ops in product(*operators)]
        return [_string_product(ops) for ops in product(*operators)]

    def get_stabilizers(self, return_array=False):
        generators, phases = self.get_stabilizers_generators(return_array)
        return self._construct_operators(generators, phases, return_array)

    def get_destabilizers(self, return_array=False):
        generators = self.get_destabilizers_generators(return_array)
        return self._construct_operators(generators, phases, return_array)

    def state(self):
        stabilizers = self.get_stabilizers(True)
        return np.sum(stabilizers, 0) / len(stabilizers)

    def has_samples(self):
        if self.measurements:
            return (
                self.measurements[0].result.has_samples() or self._samples is not None
            )
        return False

    def samples(self, binary=True):
        measured_qubits = [q for m in self.measurements for q in m.target_qubits]
        if not self.has_samples():
            self._samples = self._backend.sample_shots(
                self.tableau, measured_qubits, self.nqubits, self.nshots
            )
        if binary:
            return self._samples
        else:
            return self._backend.samples_to_decimal(self._samples, len(measured_qubits))

    def frequencies(self, binary=True):
        measured_qubits = {q for m in self.measurements for q in m.target_qubits}
        freq = self._backend.calculate_frequencies(self.samples(False))
        if binary:
            return frequencies_to_binary(freq, len(measured_qubits))
        else:
            return freq

    def probabilities(self, qubits=None):
        measured_qubits = list({q for m in self.measurements for q in m.target_qubits})
        if qubits is not None:
            if not set(qubits).issubset(set(measured_qubits)):
                raise_error(
                    RuntimeError,
                    f"Asking probabilities for qubits {qubits}, but only qubits {measured_qubits} were measured.",
                )
        else:
            qubits = measured_qubits

        probs = [0 for _ in range(2 ** len(measured_qubits))]
        samples = self.samples(False)
        for s in samples:
            probs[s] += 1
        probs = self._backend.cast(probs) / len(samples)
        qubits = [measured_qubits.index(q) for q in qubits]
        return self._backend.calculate_probabilities(
            np.sqrt(probs), qubits, len(measured_qubits)
        )
