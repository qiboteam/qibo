from qibo import gates

class PauliError():
  def __init__(self, px=0, py=0, pz=0, seed=None):
    self.options = px, py, pz
    self.channel = gates.PauliNoiseChannel


class ThermalRelaxationError():
  def __init__(self, t1, t2, time, excited_population=0, seed=None):
    self.options = t1, t2, time, excited_population
    self.channel = gates.ThermalRelaxationChannel


class ResetError():
  def __init__(self, p0, p1, seed=None):
    self.options = p0, p1
    self.channel = gates.ResetChannel


class NoiseModel():
    """Generic noise model."""

    def __init__(self):
        self.errors = {}

    def add(self, error, gatename, qubits=None):
        """Add a quantum error for a specific gate to the noise model.

            Args:
                error (`qibo.core.noise.QuantumError`): quantum error
                gatename (str): QASM representation of the gate
        """
        if isinstance(qubits, int):
            qubits = (qubits, )

        self.errors[gatename] = (error, qubits)

    def apply(self, circuit):
        """"Generate a noisy quantum circuit according to the noise model built.

            Args:
                circuit (`qibo.core.circuit.Circuit`): ideal quantum circuit

            Return:
                Circuit with noise.
        """
        circ = circuit.__class__(**circuit.init_kwargs)
        for gate in circuit.queue:
            if gate.name in self.errors:
                error, qubits = self.errors.get(gate.name)
                if qubits is None:
                    qubits = gate.qubits
                else:
                    qubits = tuple(set(gate.qubits) & set(qubits))
                for q in qubits:
                    circ.add(error.channel(q, *error.options))
            circ.add(gate)
        return circ