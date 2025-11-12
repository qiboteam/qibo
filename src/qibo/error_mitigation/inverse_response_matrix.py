from functools import cached_property
from typing import Dict, List

from qibo import Circuit, gates
from qibo.error_mitigation.abstract import ReadoutMitigationRoutine


class InverseResponseMatrix(ReadoutMitigationRoutine):

    _circuit: Circuit = None

    @cached_property
    def response_circuits(self) -> List[Circuit]:
        circuits = []
        for i in range(2**self.nqubits):
            binary_state = format(i, f"0{self.nqubits}b")
            circuit = Circuit(self.nqubits, density_matrix=True)
            for qubit, bit in enumerate(binary_state):
                if bit == "1":
                    circuit.add(gates.X(qubit))
            circuit.add(gates.M(*range(self.nqubits)))
            circuits.append(self.noise_model.apply(circuit))
        return circuits

    @cached_property
    def response_matrix(self):
        response_matrix = self.backend.np.zeros((2**self.nqubits, 2**self.nqubits))
        circuit_results = self.backend.execute_circuits(
            self.response_circuits, nshots=self.nshots
        )
        for i, result in enumerate(circuit_results):
            frequencies = result.frequencies()
            column = self.backend.np.zeros(2**self.nqubits)
            for key, value in frequencies.items():
                column[int(key, 2)] = value / self.nshots
            response_matrix[:, i] = column
        return response_matrix

    @cached_property
    def inverse_response_matrix(self):
        return self.backend.np.linalg.inv(self.response_matrix)

    def __call__(self, frequencies: Dict[int | str, int]) -> Dict[int | str, float]:
        if self.nshots is None:
            self.nshots = sum(tuple(frequencies.values()))
        is_key_integer = isinstance(tuple(frequencies)[0], int)
        # convert keys to integer representation
        if not is_key_integer:
            frequencies = {int(key, 2): value for key, value in frequencies.items()}
        mitigated_frequencies = self.backend.np.zeros(2**self.nqubits)
        for key, value in frequencies.items():
            mitigated_frequencies[key] = value
        mitigated_frequencies = mitigated_frequencies.reshape(-1, 1)
        mitigated_frequencies = self.inverse_response_matrix @ mitigated_frequencies
        if not is_key_integer:
            return {
                f"{i:0{self.nqubits}b}": float(value[0])
                for i, value in enumerate(mitigated_frequencies)
            }
        return {i: float(value[0]) for i, value in enumerate(mitigated_frequencies)}
