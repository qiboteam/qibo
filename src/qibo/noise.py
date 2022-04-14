from qibo import gates

class PauliError():
    """Quantum error associated with the `qibo.core.gates.PauliNoiseChannel`.

        Args:
            options (tuple): see :class:`qibo.core.gates.PauliNoiseChannel`
    """
    def __init__(self, px=0, py=0, pz=0, seed=None):
        self.options = px, py, pz, seed
        self.channel = gates.PauliNoiseChannel


class ThermalRelaxationError():
    """Quantum error associated with the `qibo.core.gates.ThermalRelaxationChannel`.

        Args:
            options (tuple): see :class:`qibo.core.gates.ThermalRelaxationChannel`
    """
    def __init__(self, t1, t2, time, excited_population=0, seed=None):
        self.options = t1, t2, time, excited_population, seed
        self.channel = gates.ThermalRelaxationChannel


class ResetError():
    """Quantum error associated with the `qibo.core.gates.ResetChannel`.

        Args:
            options (tuple): see :class:`qibo.core.gates.ResetChannel`
    """
    def __init__(self, p0, p1, seed=None):
        self.options = p0, p1, seed
        self.channel = gates.ResetChannel


class NoiseModel():
    """Class for the implementation of a custom noise model."""

    def __init__(self):
        self.errors = {}

    def add(self, error, gate, qubits=None):
        """Add a quantum error for a specific gate and qubit to the noise model.

            Args:
                error: quantum error to associate with the gate. Possible choices
                       are :class:`qibo.noise.PauliError`,
                       :class:`qibo.noise.ThermalRelaxationError` and
                       :class:`qibo.noise.ResetError`.
                gate (:class:`qibo.core.gates`): gate after which the noise will be added.
                qubits (tuple): qubits where the noise will be applied, if None the noise
                                will be added after every instance of the gate.
        """
        if isinstance(qubits, int):
            qubits = (qubits, )

        self.errors[gate] = (error, qubits)

    def apply(self, circuit):
        """Generate a noisy quantum circuit according to the noise model built.

            Args:
                circuit (:class:`qibo.core.circuit.Circuit`): quantum circuit

            Returns:
                A (:class:`qibo.core.circuit.Circuit`) which corresponds
                to the initial circuit with noise gates added according
                to the noise model.
        """
        circ = circuit.__class__(**circuit.init_kwargs)
        for gate in circuit.queue:
            circ.add(gate)
            if gate.__class__ in self.errors:
                error, qubits = self.errors.get(gate.__class__)
                if qubits is None:
                    qubits = gate.qubits
                else:
                    qubits = tuple(set(gate.qubits) & set(qubits))
                for q in qubits:
                    circ.add(error.channel(q, *error.options))
        return circ