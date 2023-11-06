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

    def zero_state(self, nqubits):
        I = self.np.eye(nqubits)
        tableau = self.np.zeros((2 * nqubits + 1, 2 * nqubits + 1), dtype=bool)
        tableau[:nqubits, :nqubits] = I.copy()
        tableau[nqubits:-1, nqubits : 2 * nqubits] = I.copy()
        return tableau

    def clifford_operation(self, gate):
        name = gate.__class__.__name__
        return getattr(self.clifford_operations, name)

    def apply_gate_clifford(self, gate, tableau, nqubits, nshots):
        operation = gate.clifford_operation(self)
        if len(gate.control_qubits) > 0:
            return operation(
                tableau,
                self.np.array(gate.control_qubits),
                self.np.array(gate.target_qubits),
                nqubits,
            )
        else:
            return operation(tableau, self.np.array(gate.qubits), nqubits)

    def execute_circuit(self, circuit, initial_state=None, nshots=1000):
        for gate in circuit.queue:
            if not gate.clifford and not gate.__class__.__name__ == "M":
                raise RuntimeError("The circuit contains non-Clifford gates.")

        try:
            nqubits = circuit.nqubits

            if initial_state is None:
                state = self.zero_state(nqubits)
            else:
                state = initial_state

            for gate in circuit.queue:
                state = gate.apply_clifford(self, state, nqubits, nshots)

            return state

        except self.oom_error:
            raise_error(
                RuntimeError,
                f"State does not fit in {self.device} memory."
                "Please switch the execution device to a "
                "different one using ``qibo.set_device``.",
            )

    def sample_shots(self, state, qubits, nqubits, nshots, collapse=False):
        operation = CliffordOperations()
        if collapse:
            samples = [operation.M(state, qubits, nqubits) for _ in range(nshots - 1)]
            samples.append(operation.M(state, qubits, nqubits, collapse))
        else:
            samples = [operation.M(state, qubits, nqubits) for _ in range(nshots)]
        return self.np.asarray(samples).reshape(nshots, -1)

    def calculate_probabilities(self, state, qubits, nqubits, nshots):
        samples = self.sample_shots(state, qubits, nqubits, nshots)
        probs = samples.sum(0) / nshots
        return self._order_probabilities(probs, qubits, nqubits).ravel()
