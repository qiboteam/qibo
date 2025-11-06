from copy import deepcopy
from functools import cached_property

from qibo import Circuit, gates
from qibo.error_mitigation.abstract import ReadoutMitigationRoutine
from qibo.measurements import MeasurementResult


class InverseResponseMatrix(ReadoutMitigationRoutine):

    _circuit: Circuit = None

    @cached_property
    def response_circuits(self):
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

    def __call__(self, measurement_result: MeasurementResult) -> MeasurementResult:
        original_frequencies = deepcopy(measurement_result.frequencies)
        self._nqubits = len(measurement_result.target_qubits)
        measurement_result.frequencies = self.frequencies_decorator(
            original_frequencies
        )
        return measurement_result

    def frequencies_decorator(self, freq_method):

        def monkey_frequencies():
            frequencies = self.backend.np.zeros(2**self.nqubits)
            for key, value in freq_method().items():
                frequencies[int(key, 2)] = value
            frequencies = frequencies.reshape(-1, 1)
            frequencies = self.inverse_response_matrix @ frequencies
            return {i: float(value[0].real) for i, value in enumerate(frequencies)}

        return monkey_frequencies
