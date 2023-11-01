from qibo import __version__
from qibo.backends.abstract import Backend
from qibo.backends.clifford_operations import CliffordOperations
from qibo.backends.numpy import NumpyBackend


class CliffordBackend(NumpyBackend):
    def __init__(self):
        super().__init__()

        import numpy as np

        self.name = "clifford"
        self.clifford_operations = CliffordOperations()
        self.np = np

    def clifford_operation(self, gate):
        name = gate.__class__.__name__
        return getattr(self.clifford_operations, name)

    def apply_gate(self, gate, tableau):
        operation = gate.clifford_operation(self)
        if len(gate.control_qubits) > 0:
            return operation(
                tableau,
                self.np.array(gate.control_qubits),
                self.np.array(gate.target_qubits),
            )
        else:
            return operation(tableau, self.np.array(gate.qubits))

    def execute_circuit(self, circuit, initial_state=None):
        for gate in circuit.queue:
            if not gate.clifford and not gate.__class__.__name__ == "M":
                raise RuntimeError("The circuit contains non-Clifford gates.")

        try:
            nqubits = circuit.nqubits

            if initial_state is None:
                I = self.np.eye(nqubits)
                tableau = self.np.zeros((2 * nqubits + 1, 2 * nqubits + 1), dtype=bool)
                tableau[:nqubits, :nqubits] = I
                tableau[nqubits:-1, nqubits : 2 * nqubits] = I
            else:
                tableau = initial_state

            for gate in circuit.queue:
                tableau = self.apply_gate(gate, tableau)

            return tableau

        except self.oom_error:
            raise_error(
                RuntimeError,
                f"State does not fit in {self.device} memory."
                "Please switch the execution device to a "
                "different one using ``qibo.set_device``.",
            )
