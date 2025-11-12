from functools import cached_property
from typing import Dict, List

from numpy.typing import ArrayLike

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

    def _frequencies_to_array(self, frequencies: Dict[int, int | float]) -> ArrayLike:
        array = self.backend.np.zeros(2**self.nqubits)
        for key, value in frequencies.items():
            array[key] = value
        return self.backend.np.reshape(array, (-1, 1))

    def mitigate_frequencies(self, frequencies: ArrayLike) -> ArrayLike:
        return self.inverse_response_matrix @ frequencies

    def __call__(self, frequencies: Dict[int | str, int]) -> Dict[int | str, float]:
        if self.nshots is None:
            self.nshots = sum(tuple(frequencies.values()))
        is_key_integer = isinstance(tuple(frequencies)[0], int)
        # convert keys to integer representation
        if not is_key_integer:
            frequencies = self.binary_to_integer_keys(frequencies)
        frequencies = self._frequencies_to_array(frequencies)
        mitigated_frequencies = self.mitigate_frequencies(frequencies)
        mitigated_frequencies = (
            self.backend.to_numpy(mitigated_frequencies).ravel().tolist()
        )
        if not is_key_integer:
            return self.integer_to_binary_keys(mitigated_frequencies, self.nqubits)
        return {i: value for i, value in enumerate(mitigated_frequencies)}
