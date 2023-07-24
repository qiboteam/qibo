import collections
from itertools import combinations
from math import log2

from qibo import gates
from qibo.config import raise_error
from qibo.noise_model import CompositeNoiseModel


class KrausError:
    """Quantum error associated with the :class:`qibo.gates.KrausChannel`.

    Args:
        ops (list): List of Kraus operators of type ``np.ndarray``
            or ``tf.Tensor`` and of the same shape.

    """

    def __init__(self, ops):
        shape = ops[0].shape
        if any(o.shape != shape for o in ops):
            raise_error(
                ValueError,
                "Kraus operators of different shapes."
                "Use qibo.noise.CustomError instead.",
            )

        self.rank = shape[0]
        self.options = ops

    def channel(self, qubits, options):
        return [
            gates.KrausChannel(q, options)
            for q in combinations(qubits, int(log2(self.rank)))
        ]


class UnitaryError:
    """Quantum error associated with the :class:`qibo.gates.UnitaryChannel`.

    Args:
        probabilities (list): List of floats that correspond to the probability
            that each unitary Uk is applied.
        unitaries (list): List of unitary matrices as ``np.ndarray``/``tf.Tensor``
            of the same shape. Must have the same length as the given
            probabilities ``p``.

    """

    def __init__(self, probabilities, unitaries):
        shape = unitaries[0].shape
        if any(o.shape != shape for o in unitaries):
            raise_error(
                ValueError,
                "Unitary matrices have different shapes."
                "Use qibo.noise.CustomError instead.",
            )
        self.rank = shape[0]
        self.options = list(zip(probabilities, unitaries))

    def channel(self, qubits, options):
        return [
            gates.UnitaryChannel(q, options)
            for q in combinations(qubits, int(log2(self.rank)))
        ]


class PauliError:
    """Quantum error associated with the :class:`qibo.gates.PauliNoiseChannel`.

    Args:
        operators (list): see :class:`qibo.gates.PauliNoiseChannel`
    """

    def __init__(self, operators):
        self.options = operators

    def channel(self, qubits, options):
        return [gates.PauliNoiseChannel(q, options) for q in qubits]


class DepolarizingError:
    """Quantum error associated with the :class:`qibo.gates.DepolarizingChannel`.

    Args:
        options (float): see :class:`qibo.gates.DepolarizingChannel`
    """

    def __init__(self, lam):
        self.options = lam
        self.channel = gates.DepolarizingChannel


class ThermalRelaxationError:
    """Quantum error associated with the :class:`qibo.gates.ThermalRelaxationChannel`.

    Args:
        options (tuple): see :class:`qibo.gates.ThermalRelaxationChannel`
    """

    def __init__(self, t1, t2, time, excited_population=0):
        self.options = [t1, t2, time, excited_population]
        self.channel = gates.ThermalRelaxationChannel


class AmplitudeDampingError:
    """Quantum error associated with the :class:`qibo.gates.AmplitudeDampingChannel`.

    Args:
        options (float): see :class:`qibo.gates.AmplitudeDampingChannel`
    """

    def __init__(self, gamma):
        self.options = gamma
        self.channel = gates.AmplitudeDampingChannel


class ReadoutError:
    """Quantum error associated with :class:'qibo.gates;ReadoutErrorChannel'.

    Args:
        options (array): see :class:'qibo.gates.ReadoutErrorChannel'
    """

    def __init__(self, probabilities):
        self.options = probabilities
        self.channel = gates.ReadoutErrorChannel


class ResetError:
    """Quantum error associated with the `qibo.gates.ResetChannel`.

    Args:
        options (tuple): see :class:`qibo.gates.ResetChannel`
    """

    def __init__(self, p0, p1):
        self.options = [p0, p1]
        self.channel = gates.ResetChannel


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

        # Create an Error associated with Kraus Channel
        # rho -> |0><0| rho |0><0| + |0><1| rho |0><1|
        error = CustomError(gates.KrausChannel((0,), [a1, a2]))

    """

    def __init__(self, channel):
        self.channel = channel


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
        noise.add(PauliError([("X", 0.5)]), gates.H, 1)
        noise.add(PauliError([("Y", 0.5)]), gates.CNOT)

        # Generate noiseless circuit.
        c = models.Circuit(2)
        c.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])

        # Apply noise to the circuit according to the noise model.
        noisy_c = noise.apply(c)

    """

    def __init__(self):
        self.errors = collections.defaultdict(list)
        self.noise_model = {}

    def add(self, error, gate=None, qubits=None, condition=None):
        """Add a quantum error for a specific gate and qubit to the noise model.

        Args:
            error: quantum error to associate with the gate. Possible choices
                are :class:`qibo.noise.PauliError`,
                :class:`qibo.noise.DepolarizingError`,
                :class:`qibo.noise.ThermalRelaxationError`,
                :class:`qibo.noise.AmplitudeDampingError`,
                :class:`qibo.noise.ReadoutError`,
                :class:`qibo.noise.ResetError`,
                :class:`qibo.noise.UnitaryError`,
                :class:`qibo.noise.KrausError` and
                :class:`qibo.noise.CustomError`.
            gate (:class:`qibo.gates.Gate`): gate after which the noise will be added.
                If ``None``, the noise will be added after each gate except
                :class:`qibo.gates.Channel` and :class:`qibo.gates.M`.
            qubits (tuple): qubits where the noise will be applied. If ``None``,
                the noise will be added after every instance of the gate.
            condition (callable, optional): function that takes :class:`qibo.gates.Gate`
                object as an input and returns ``True`` if noise should be added to it.

        Example:

        .. testcode::

            import numpy as np
            from qibo import Circuit, gates
            from qibo.noise import NoiseModel, PauliError

            # Check if a gate is RX(pi/2).
            def is_sqrt_x(gate):
                return np.pi/2 in gate.parameters

            # Build a noise model with a Pauli error on RX(pi/2) gates.
            error = PauliError(list(zip(["X", "Y", "Z"], [0.01, 0.5, 0.1])))
            noise = NoiseModel()
            noise.add(PauliError([("X", 0.5)]), gates.RX, condition=is_sqrt_x)

            # Generate a noiseless circuit.
            circuit = Circuit(1)
            circuit.add(gates.RX(0, np.pi / 2))
            circuit.add(gates.RX(0, 3 * np.pi / 2))
            circuit.add(gates.X(0))

            # Apply noise to the circuit.
            noisy_circuit = noise.apply(circuit)

        """

        if isinstance(qubits, int):
            qubits = (qubits,)

        if condition is not None and not callable(condition):
            raise TypeError(
                "condition should be callable. Got {} instead."
                "".format(type(condition))
            )
        else:
            self.errors[gate].append((condition, error, qubits))

    def composite(self, params):
        """Build a noise model to simulate the noisy behaviour of a quantum computer.

        Args:
            params (dict): contains the parameters of the channels organized as follow \n
                {'t1' : (``t1``, ``t2``,..., ``tn``),
                't2' : (``t1``, ``t2``,..., ``tn``),
                'gate time' : (``time1``, ``time2``),
                'excited population' : 0,
                'depolarizing error' : (``lambda1``, ``lambda2``),
                'bitflips error' : ([``p1``, ``p2``,..., ``pm``],[``p1``, ``p2``,..., ``pm``]),
                'idle_qubits' : True}
                where `n` is the number of qubits, and `m` the number of measurement gates.
                The first four parameters are used by the thermal relaxation error.
                The first two  elements are the tuple containing the :math:`T_1` and
                :math:`T_2` parameters; the third one is a tuple which contain the gate times,
                for single and two qubit gates; then we have the excited population parameter.
                The fifth parameter is a tuple containing the depolaraziong errors for single
                and 2 qubit gate. The sisxth parameter is a tuple containg the two arrays for
                bitflips probability errors: the first one implements 0->1 errors, the other
                one 1->0. The last parameter is a boolean variable: if ``True`` the noise
                model takes into account idle qubits.
        """

        self.noise_model = CompositeNoiseModel(params)

    def apply(self, circuit):
        """Generate a noisy quantum circuit according to the noise model built.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): quantum circuit

        Returns:
            A (:class:`qibo.models.circuit.Circuit`) which corresponds
            to the initial circuit with noise gates added according
            to the noise model.
        """

        if isinstance(self.noise_model, CompositeNoiseModel):
            self.noise_model.apply(circuit)
            noisy_circuit = self.noise_model.noisy_circuit
        else:
            noisy_circuit = circuit.__class__(**circuit.init_kwargs)
            for gate in circuit.queue:
                errors_list = (
                    self.errors[gate.__class__]
                    if (isinstance(gate, gates.Channel) or isinstance(gate, gates.M))
                    else self.errors[gate.__class__] + self.errors[None]
                )
                if all(
                    isinstance(error, ReadoutError) is False
                    for _, error, _ in errors_list
                ):
                    noisy_circuit.add(gate)
                for condition, error, qubits in errors_list:
                    if condition is None or condition(gate):
                        if qubits is None:
                            qubits = gate.qubits
                        else:
                            qubits = tuple(set(gate.qubits) & set(qubits))
                        if len(qubits) == 0:
                            continue

                        if isinstance(error, CustomError) and qubits:
                            noisy_circuit.add(error.channel)
                        elif (
                            isinstance(
                                error,
                                (
                                    ThermalRelaxationError,
                                    AmplitudeDampingError,
                                    ResetError,
                                ),
                            )
                            and qubits
                        ):
                            for q in qubits:
                                noisy_circuit.add(error.channel(q, error.options))
                        elif isinstance(error, ReadoutError) and qubits:
                            noisy_circuit.add(error.channel(qubits, error.options))
                            noisy_circuit.add(gate)
                        else:
                            noisy_circuit.add(error.channel(qubits, error.options))

        return noisy_circuit
