from qibo.error_mitigation.cdr import CDR


class ICS(CDR):

    def sample_circuit(self, circuit: Optional[Circuit] = None) -> Circuit:
        sampled_circuit = super().sample_circuit(circuit)
        symplectic_matrix = self.simulation_backend.execute_circuit(
            sampled_circuit.invert(), nshots=1
        ).symplectic_matrix[:-1, :-1]
        raise NotImplementedError
