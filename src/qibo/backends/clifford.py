from qibo import __version__
from qibo.backends.abstract import Backend
from qibo.backends.clifford_operations import CliffordOperations


class CliffordBackend(NumpyBackend):
    def __init__(self):
        super().__init__()
        self.name = "clifford"
        self.clifford_operations = CliffordOperations()

    def clifford_operation(self, gate):
        name = gate.__class__.__name__
        return getattr(self.clifford_operations, name)

    def apply_gate(self, gate, tableau):
        operation = gate.clifford(self)
        if gate.is_controlled_by:
            return operation(tableu, gate.control_qubits, gate.target_qubits)
        else:
            return operation(tableu, gate.qubits)

    def execute_circuit(self, circuit, initial_state=None):
        for gate in circuit.queue:
            if not gate.clifford:
                raise RuntimeError("The circuit contains non-Clifford gates.")

        try:
            nqubits = circuit.nqubits

            if initial_state is None:
                I = np.eye(nqubits)
                tableau = self.np.zeros((2 * nqubits, 2 * nqubits + 1), dtype=bool)
                tableau[:nqubits, :nqubits] = I
                tableau[nqubits:, nqubits : 2 * nqubits] = I
            else:
                tableau = initial_state

            for gate in circuit.queue:
                tableau = self.apply_gate(gate, tableau)

        except self.oom_error:
            raise_error(
                RuntimeError,
                f"State does not fit in {self.device} memory."
                "Please switch the execution device to a "
                "different one using ``qibo.set_device``.",
            )
