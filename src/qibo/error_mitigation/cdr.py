import copy
import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from qibo import Circuit, gates
from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error
from qibo.error_mitigation.abstract import DataRegressionErrorMitigation
from qibo.gates.abstract import Gate, ParametrizedGate

NPBACKEND = NumpyBackend()


@dataclass
class CDR(DataRegressionErrorMitigation):

    replacement_gates: Optional[Tuple[Gate, List[Dict]]] = None
    _targeted_gate: Gate = None

    def __post_init__(self):
        super().__post_init__()
        # setup the regression model
        self._regression_model_setup()

        # setup the replacement gates
        if self.replacement_gates is None:
            self.replacement_gates = (
                gates.RZ,
                [{"theta": n * np.pi / 2} for n in range(4)],
            )
        self._targeted_gate = self.replacement_gates[0]
        if not all(
            self._targeted_gate(0, **kwargs).clifford
            for kwargs in self.replacement_gates[1]
        ):
            raise_error(
                ValueError,
                f"The provided set of gates for replacement: {self.replacement_gates} include one or more non clifford gates.",
            )

    def _regression_model_setup(self):
        if self.regression_model is None:
            self.regression_model = lambda x, a, b: a * x + b
            if self.model_parameters is None:
                self.model_parameters = np.array([1, 0], dtype=np.float64)
        elif self.model_parameters is None:
            nparams = len(inspect.signature(self.regression_model).parameters) - 1
            self.model_parameters = np.random.randn(nparams).astype(np.float64)

    def sample_circuit(
        self, circuit: Optional[Circuit] = None, sigma: float = 0.5
    ) -> Circuit:
        circuit = self._circuit(circuit)

        gates_to_replace = []
        for i, gate in enumerate(circuit.queue):
            if isinstance(gate, self._targeted_gate):
                if not gate.clifford:
                    gates_to_replace.append((i, gate))
        if len(gates_to_replace) == 0:
            raise_error(RuntimeError, "No non-Clifford gate found, no circuit sampled.")

        replacement, distance = [], []
        for _, gate in gates_to_replace:
            rep_gates = np.array(
                [
                    self._targeted_gate(*gate.init_args, **kwargs)
                    for kwargs in self.replacement_gates[1]
                ]
            )

            replacement.append(rep_gates)

            if isinstance(gate, ParametrizedGate):
                gate = copy.deepcopy(gate)
                gate.parameters = NPBACKEND.cast(
                    [self.backend.to_numpy(p) for p in gate.parameters],
                    dtype=np.float64,
                )
            gate_matrix = gate.matrix(NPBACKEND)
            rep_gate_matrix = [rep_gate.matrix(NPBACKEND) for rep_gate in rep_gates]
            rep_gate_matrix = NPBACKEND.cast(
                rep_gate_matrix, dtype=rep_gate_matrix[0].dtype
            )
            matrix_norm = NPBACKEND.np.linalg.norm(
                gate_matrix - rep_gate_matrix, ord="fro", axis=(1, 2)
            )
            distance.append(NPBACKEND.np.real(matrix_norm))

        distance = np.vstack(distance)
        prob = np.exp(-(distance**2) / sigma**2)

        index = np.random.choice(
            range(len(gates_to_replace)),
            size=min(int(len(gates_to_replace) / 2), 50),
            replace=False,
            p=np.sum(prob, -1) / np.sum(prob),
        )

        gates_to_replace = np.array([gates_to_replace[i] for i in index])
        prob = [prob[i] for i in index]
        prob = NPBACKEND.cast(prob, dtype=prob[0].dtype)

        replacement = np.array([replacement[i] for i in index])
        replacement = [
            replacement[i][np.random.choice(range(len(p)), size=1, p=p / np.sum(p))[0]]
            for i, p in enumerate(prob)
        ]
        replacement = {i[0]: g for i, g in zip(gates_to_replace, replacement)}

        sampled_circuit = circuit.__class__(**circuit.init_kwargs)
        for i, gate in enumerate(circuit.queue):
            sampled_circuit.add(replacement.get(i, gate))

        return sampled_circuit
