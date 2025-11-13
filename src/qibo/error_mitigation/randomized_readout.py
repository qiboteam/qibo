from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Optional, Tuple

from qibo import gates
from qibo.error_mitigation.abstract import ReadoutMitigationRoutine
from qibo.models.circuit import Circuit
from qibo.quantum_info.random_ensembles import random_pauli


@dataclass
class RandomizedReadout(ReadoutMitigationRoutine):

    circuit: Optional[Circuit] = None
    n_circuits: int = 10

    def _n_circuits(self, n_circuits: Optional[int] = None) -> int:
        if n_circuits is None:
            return self.n_circuits
        return n_circuits

    def _sample_calibration_circuits(
        self,
        circuit: Optional[Circuit] = None,
        n_circuits: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[List[Circuit], List[Dict[int, int]]]:
        n_circuits = self._n_circuits(n_circuits)
        nqubits = circuit.nqubits if circuit is not None else self.nqubits
        cal_circuits, error_maps = [], []
        for k in range(n_circuits):
            # set the initial seed only at the first iteration
            seed = seed if k == 0 else None
            x_gate = random_pauli(
                nqubits, 1, subset=["I", "X"], seed=seed, return_circuit=True
            )
            error_maps.append(
                {
                    gate.qubits[0]: 1
                    for gate in x_gate.queue
                    if isinstance(gate, gates.X)
                }
            )
            cal_circuits.append(circuit + x_gate if circuit is not None else x_gate)
        return cal_circuits, error_maps

    @cached_property
    def pauli_circuits(self):
        return self._sample_calibration_circuits()

    def __call__(self, frequencies: Dict[int | str, int]) -> Dict[int | str, int]:
        pass
