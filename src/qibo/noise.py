from qibo import gates
from qibo.config import raise_error


class CustomError:
    """Quantum error associated with the :class:`qibo.gates.Channel`

    Args:
        channel (:class:`qibo.gates.Channel`): any channel

    Example:

    .. testcode::

        import numpy as np
        from qibo.gates import KrausChannel
        from qibo.noise import CustomError

        # define |0><0|
        a1 = np.array([[1, 0], [0, 0]])
        # define |0><1|
        a2 = np.array([[0, 1], [0, 0]])

        # Create an Error associated with Kraus Channel rho -> |0><0| rho |0><0| + |0><1| rho |0><1|
        error = CustomError(gates.KrausChannel([((0,), a1), ((0,), a2)]))
    """

    def __init__(self, channel):
        self.channel = channel


class PauliError:
    """Quantum error associated with the :class:`qibo.gates.PauliNoiseChannel`.

    Args:
        options (tuple): see :class:`qibo.gates.PauliNoiseChannel`
    """

    def __init__(self, px=0, py=0, pz=0):
        self.options = px, py, pz
        self.channel = gates.PauliNoiseChannel


class ThermalRelaxationError:
    """Quantum error associated with the :class:`qibo.gates.ThermalRelaxationChannel`.

    Args:
        options (tuple): see :class:`qibo.gates.ThermalRelaxationChannel`
    """

    def __init__(self, t1, t2, time, excited_population=0):
        self.options = t1, t2, time, excited_population
        self.channel = gates.ThermalRelaxationChannel


class DepolarizingError:
    """Quantum error associated with the :class:`qibo.gates.DepolarizingChannel`.

    Args:
        options (float): see :class:`qibo.gates.DepolarizingChannel`
    """

    def __init__(self, lam):
        self.options = (lam,)
        self.channel = gates.DepolarizingChannel


class ResetError:
    """Quantum error associated with the `qibo.gates.ResetChannel`.

    Args:
        options (tuple): see :class:`qibo.gates.ResetChannel`
    """

    def __init__(self, p0, p1):
        self.options = p0, p1
        self.channel = gates.ResetChannel


class KrausError:
    """Quantum error associated with the :class:`qibo.gates.KrausChannel`.

    Args:
        ops (list): List of Kraus operators as a ``np.ndarray`` or ``tf.Tensor``.
        nqubits (int): Number of qubits each that Kraus operator acts on.
    """

    def __init__(self, ops):
        self.options = ops

        shape = ops[0].shape
        if any(o.shape != shape for o in ops):
            raise_error(
                ValueError,
                "Kraus operators of different shapes." "Use qibo.noise.Error instead.",
            )

        self.rank = shape[0]

    def channel(self, qubits):
        ops = [([*qubits], o) for o in self.options]
        return gates.KrausChannel(ops)


class UnitaryError:
    """Quantum error associated with the :class:`qibo.gates.UnitaryChannel`.

    Args:
        probabilities (list): List of floats that correspond to the probability
            that each unitary Uk is applied.
        unitaries (list): List of unitary matrices as ``np.ndarray``/``tf.Tensor`` of the same shape.
            Must have the same length as the given probabilities ``p``.
    """

    def __init__(self, probabilities, unitaries):
        self.probabilities = probabilities
        self.unitaries = unitaries

        shape = unitaries[0].shape
        if any(o.shape != shape for o in unitaries):
            raise_error(
                ValueError,
                "Unitary matrices have different shapes."
                "Use qibo.noise.Error instead.",
            )

        self.rank = shape[0]

    def channel(self, qubits):
        ops = [([*qubits], u) for u in self.unitaries]
        return gates.UnitaryChannel(self.probabilities, ops)


class NoiseModel:
    """Class for the implementation of a custom noise model.

    Example:

    .. testcode::

        from qibo import models, gates
        from qibo.noise import NoiseModel, PauliError

        # Build specific noise model with 2 quantum errors:
        # - Pauli error on H only for qubit 1.
        # - Pauli error on CNOT for all the qubits.
        noise = NoiseModel()
        noise.add(PauliError(px = 0.5), gates.H, 1)
        noise.add(PauliError(py = 0.5), gates.CNOT)

        # Generate noiseless circuit.
        c = models.Circuit(2)
        c.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])

        # Apply noise to the circuit according to the noise model.
        noisy_c = noise.apply(c)
    """

    def __init__(self):
        self.errors = {}

    def add(self, error, gate, qubits=None):
        """Add a quantum error for a specific gate and qubit to the noise model.

        Args:
            error: quantum error to associate with the gate. Possible choices
                   are :class:`qibo.noise.PauliError`,
                   :class:`qibo.noise.ThermalRelaxationError`,
                   :class:`qibo.noise.DepolarizingError` and
                   :class:`qibo.noise.ResetError`.
            gate (:class:`qibo.gates.Gate`): gate after which the noise will be added.
            qubits (tuple): qubits where the noise will be applied, if None the noise
                            will be added after every instance of the gate.
        """

        if isinstance(qubits, int):
            qubits = (qubits,)

        self.errors[gate] = (error, qubits)

    def apply(self, circuit):
        """Generate a noisy quantum circuit according to the noise model built.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): quantum circuit

        Returns:
            A (:class:`qibo.models.circuit.Circuit`) which corresponds
            to the initial circuit with noise gates added according
            to the noise model.
        """
        noisy_circuit = circuit.__class__(**circuit.init_kwargs)
        for gate in circuit.queue:
            noisy_circuit.add(gate)
            if gate.__class__ in self.errors:
                error, qubits = self.errors.get(gate.__class__)
                if qubits is None:
                    qubits = gate.qubits
                else:
                    qubits = tuple(set(gate.qubits) & set(qubits))
                if isinstance(error, CustomError) and qubits:
                    noisy_circuit.add(error.channel)
                elif isinstance(error, DepolarizingError) and qubits:
                    noisy_circuit.add(error.channel(qubits, *error.options))
                elif isinstance(error, UnitaryError) or isinstance(error, KrausError):
                    if error.rank == 2:
                        for q in qubits:
                            noisy_circuit.add(error.channel([q]))
                    elif error.rank == 2 ** len(qubits):
                        noisy_circuit.add(error.channel(qubits))
                else:
                    for q in qubits:
                        noisy_circuit.add(error.channel(q, *error.options))
        return noisy_circuit
