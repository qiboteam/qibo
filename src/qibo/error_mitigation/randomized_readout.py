from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Optional, Tuple

from qibo import gates
from qibo.error_mitigation.abstract import ReadoutMitigationRoutine
from qibo.models import circuit
from qibo.models.circuit import Circuit
from qibo.quantum_info.random_ensembles import random_pauli


@dataclass
class RandomizedReadout(ReadoutMitigationRoutine):

    circuit: Optional[Circuit] = None
    n_circuits: int = 10

    @staticmethod
    def _sample_X_pauli(
        nqubits: int, seed: Optional[int] = None
    ) -> Tuple[Circuit, Dict[int, int]]:
        x_gate = random_pauli(
            nqubits, 1, subset=["I", "X"], return_circuit=True, seed=seed
        )
        error_map = {
            gate.qubits[0]: 1 for gate in x_gate.queue if isinstance(gate, gates.X)
        }
        return x_gate, error_map

    @cached_property
    def X_paulis(self) -> Tuple[List[Circuit], List[Dict[int, int]]]:
        x_paulis, error_maps = zip(
            *[self._sample_X_pauli(self.nqubits) for _ in range(self.n_circuits)]
        )
        return x_paulis, error_maps

    @staticmethod
    def measurements_layer(
        nqubits: int, error_map: Optional[Dict[int, int]] = None, **circ_kwargs
    ) -> Circuit:
        meas_circ = Circuit(nqubits, **circ_kwargs)
        meas_circ.add(gates.M(*range(nqubits), p0=error_map))
        return meas_circ

    @cached_property
    def calibration_circuits(self) -> Tuple[List[Circuit], List[Circuit]]:
        empty_circuits = [
            x_pauli + self.measurements_layer(self.nqubits, error_map=err_map)
            for x_pauli, err_map in self.X_paulis
        ]
        circuits = [
            self.circuit.copy(deep=True)
            + self.measurements_layer(self.nqubits, error_map=err_map)
            for _, err_map in self.X_paulis
        ]
        return empty_circuits, circuits

    def _sample_calibration_circuit(
        self,
        circuit: Optional[Circuit] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Circuit, Dict[int, int]]:
        x_pauli, error_map = self._sample_X_pauli(circuit.nqubits, seed)
        return (
            circuit.copy(deep=True)
            + x_pauli
            + self.measurements_layer(circuit.nqubits),
            error_map,
        )

    def lam(self, nshots):
        frequencies = [
            result.frequencies(binary=False)
            for result in self.backend.execute_circuits(
                sum(*self.calibration_circuits), nshots=nshots
            )
        ]

    def __call__(self, frequencies: Dict[int | str, int]) -> Dict[int | str, int]:
        pass
