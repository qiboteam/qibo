"""Module defining Error channels and NoiseModel class(es)."""

import collections
from itertools import combinations
from math import log2
from typing import Optional, Union

from qibo import gates
from qibo.config import raise_error


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
        """Returns the quantum channel associated to the quantum error."""
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
        """Returns the quantum channel associated to the quantum error."""
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
        """Returns the quantum channel associated to the quantum error."""
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


class PhaseDampingError:
    """Quantum error associated with the :class:`qibo.gates.PhaseDampingChannel`.

    Args:
        options (float): see :class:`qibo.gates.PhaseDampingChannel`
    """

    def __init__(self, gamma):
        self.options = gamma
        self.channel = gates.PhaseDampingChannel


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

        from qibo import Circuit, gates
        from qibo.noise import NoiseModel, PauliError

        # Build specific noise model with 2 quantum errors:
        # - Pauli error on H only for qubit 1.
        # - Pauli error on CNOT for all the qubits.

        noise_model = NoiseModel()
        noise_model.add(PauliError([("X", 0.5)]), gates.H, 1)
        noise_model.add(PauliError([("Y", 0.5)]), gates.CNOT)

        # Generate noiseless circuit.
        circuit = Circuit(2)
        circuit.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])

        # Apply noise to the circuit according to the noise model.
        noisy_circuit = noise_model.apply(circuit)

    """

    def __init__(self):
        self.errors = collections.defaultdict(list)

    def add(
        self,
        error,
        gate: Optional[gates.Gate] = None,
        qubits: Optional[Union[int, tuple]] = None,
        conditions=None,
    ):
        """Add a quantum error for a specific gate and qubit to the noise model.

        Args:
            error: quantum error to associate with the gate. Possible choices
                are :class:`qibo.noise.PauliError`,
                :class:`qibo.noise.DepolarizingError`,
                :class:`qibo.noise.ThermalRelaxationError`,
                :class:`qibo.noise.AmplitudeDampingError`,
                :class:`qibo.noise.PhaseDampingError`,
                :class:`qibo.noise.ReadoutError`,
                :class:`qibo.noise.ResetError`,
                :class:`qibo.noise.UnitaryError`,
                :class:`qibo.noise.KrausError`, and
                :class:`qibo.noise.CustomError`.
            gate (:class:`qibo.gates.Gate`, optional): gate after which the noise will be added.
                If ``None``, the noise will be added after each gate except
                :class:`qibo.gates.Channel` and :class:`qibo.gates.M`.
            qubits (int or tuple, optional): qubits where the noise will be applied. If ``None``,
                the noise will be added after every instance of the gate.
                Defaults to ``None``.
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
            noise.add(PauliError([("X", 0.5)]), gates.RX, conditions=is_sqrt_x)

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

        if (
            conditions is not None
            and not callable(conditions)
            and not isinstance(conditions, list)
        ):
            raise_error(
                TypeError,
                "`conditions` should be  either a callable or a list of callables. "
                + f"Got {type(conditions)} instead.",
            )

        if isinstance(conditions, list) and not all(
            callable(condition) for condition in conditions
        ):
            raise_error(
                TypeError,
                "A element of `conditions` list is not a callable.",
            )

        if callable(conditions):
            conditions = [conditions]

        self.errors[gate].append((conditions, error, qubits))

    def apply(self, circuit):
        """Generate a noisy quantum circuit according to the noise model built.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): quantum circuit

        Returns:
            :class:`qibo.models.circuit.Circuit`: initial circuit with noise gates
                added according to the noise model.
        """

        noisy_circuit = circuit.__class__(**circuit.init_kwargs)
        for gate in circuit.queue:
            errors_list = (
                self.errors[gate.__class__]
                if isinstance(gate, (gates.Channel, gates.M))
                else self.errors[gate.__class__] + self.errors[None]
            )

            if all(not isinstance(error, ReadoutError) for _, error, _ in errors_list):
                noisy_circuit.add(gate)

            for conditions, error, qubits in errors_list:
                if conditions is None or all(
                    condition(gate) for condition in conditions
                ):
                    qubits = (
                        gate.qubits
                        if qubits is None
                        else tuple(set(gate.qubits) & set(qubits))
                    )

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
                                PhaseDampingError,
                                ResetError,
                            ),
                        )
                        and qubits
                    ):
                        for qubit in qubits:
                            noisy_circuit.add(error.channel(qubit, error.options))
                    elif isinstance(error, ReadoutError) and qubits:
                        noisy_circuit.add(error.channel(qubits, error.options))
                        noisy_circuit.add(gate)
                    else:
                        noisy_circuit.add(error.channel(qubits, error.options))

            if gate.name == "measure":
                readout_error_qubits = [
                    qubits
                    for _, error, qubits in errors_list
                    if isinstance(error, ReadoutError)
                ]
                if (
                    gate.qubits not in readout_error_qubits
                    and gate.register_name
                    not in noisy_circuit.measurement_tuples.keys()
                ):
                    noisy_circuit.add(gate)

        return noisy_circuit


class _Conditions:
    def __init__(self, qubits=None):
        self.qubits = qubits

    def condition_qubits(self, gate):
        return gate.qubits == self.qubits

    def condition_gate_single(self, gate):
        """Condition that had to be matched to apply noise channel to single-qubit ``gate``."""
        return len(gate.qubits) == 1

    def condition_gate_two(self, gate):
        """Condition that had to be matched to apply noise channel to two-qubit ``gate``."""
        return len(gate.qubits) == 2


class IBMQNoiseModel(NoiseModel):
    """Class for the implementation of a IBMQ noise model.

    This noise model applies a :class:`qibo.gates.DepolarizingChannel` followed by a
    :class:`qibo.gates.ThermalRelaxationChannel` after each one- or two-qubit gate in the circuit.
    It also applies single-qubit :class:`qibo.gates.ReadoutErrorChannel`
    *before* every measurement gate.


    Example:

    .. testcode::

        from qibo import Circuit, gates
        from qibo.models.encodings import phase_encoder
        from qibo.noise import DepolarizingError, ThermalRelaxationError, ReadoutError
        from qibo.noise import IBMQNoiseModel, NoiseModel

        nqubits = 4

        # creating circuit
        phases = list(range(nqubits))
        circuit = phase_encoder(phases, rotation="RY")
        circuit.add(gates.CNOT(qubit, qubit + 1) for qubit in range(nqubits - 1))
        circuit.add(gates.M(qubit) for qubit in range(1, nqubits - 1))

        # creating noise model from dictionary
        parameters = {
            "depolarizing_one_qubit" : {"0": 0.1, "2": 0.04, "3": 0.15},
            "depolarizing_two_qubit": {"0-1": 0.2},
            "t1" : {"0": 0.1, "1": 0.2, "3": 0.01},
            "t2" : {"0": 0.01, "1": 0.02, "3": 0.0001},
            "gate_times" : (0.1, 0.2),
            "excited_population" : 0.1,
            "readout_one_qubit" : {"0": (0.1, 0.1), "1": 0.1, "3": [0.1, 0.1]},
            }

        noise_model = IBMQNoiseModel()
        noise_model.from_dict(parameters)
        noisy_circuit = noise_model.apply(circuit)
    """

    def from_dict(self, parameters: dict):
        """Method used to pass noise ``parameters`` as inside dictionary.

        Args:
            parameters (dict): Contains parameters necessary to initialise
                :class:`qibo.noise.DepolarizingError`, :class:`qibo.noise.ThermalRelaxationError`,
                and :class:`qibo.noise.ReadoutError`.

                The keys and values of the dictionary parameters are defined below:

                - ``"depolarizing_one_qubit"`` (*int* or *float* or *dict*):  If ``int`` or
                    ``float``, all qubits share the same single-qubit depolarizing parameter.
                    If ``dict``, expects qubit indexes as keys and their respective
                    depolarizing parameter as values.
                    See :class:`qibo.gates.channels.DepolarizingChannel`
                    for a detailed definition of depolarizing parameter.
                - ``"depolarizing_two_qubit"`` (*int* or *float* or *dict*):  If ``int`` or
                    ``float``, all two-qubit gates share the same two-qubit depolarizing
                    parameter regardless in which pair of qubits the two-qubit gate is acting on.
                    If ``dict``, expects pair qubit indexes as keys separated by a hiphen
                    (e.g. "0-1" for gate that has "0" as control and "1" as target)
                    and their respective depolarizing parameter as values.
                    See :class:`qibo.gates.channels.DepolarizingChannel`
                    for a detailed definition of depolarizing parameter.
                - ``"t1"`` (*int* or *float* or *dict*): If ``int`` or ``float``, all qubits
                    share the same ``t1``. If ``dict``, expects qubit indexes as keys and its
                    respective ``t1`` as values.
                    See :class:`qibo.gates.channels.ThermalRelaxationChannel`
                    for a detailed definition of ``t1``.
                    Note that ``t1`` and ``t2`` must be passed with the same type.
                - ``"t2"`` (*int* or *float* or *dict*): If ``int`` or ``float``, all qubits share
                    the same ``t2``. If ``dict``, expects qubit indexes as keys and its
                    respective ``t2`` as values.
                    See :class:`qibo.gates.channels.ThermalRelaxationChannel`
                    for a detailed definition of ``t2``.
                    Note that ``t2`` and ``t1`` must be passed with the same type.
                - ``"gate_times"`` (*tuple* or *list*): pair of gate times representing
                    gate times for :class:`ThermalRelaxationError` following, respectively,
                    one- and two-qubit gates.
                - ``"excited_population"`` (*int* or *float*): See
                    :class:`ThermalRelaxationChannel`.
                - ``"readout_one_qubit"`` (*int* or *float* or *dict*): If ``int`` or ``float``,
                    :math:`p(0|1) = p(1|0)`, and all qubits share the same readout error
                    probabilities. If ``dict``, expects qubit indexes as keys and
                    values as ``tuple`` (or ``list``) in the format :math:`(p(0|1),\\,p(1|0))`.
                    If values are ``tuple`` or ``list`` of length 1 or ``float`` or ``int``,
                    then it is assumed that :math:`p(0|1) = p(1|0)`.
        """
        t_1 = parameters["t1"]
        t_2 = parameters["t2"]
        gate_time_1, gate_time_2 = parameters["gate_times"]
        excited_population = parameters["excited_population"]
        depolarizing_one_qubit = parameters["depolarizing_one_qubit"]
        depolarizing_two_qubit = parameters["depolarizing_two_qubit"]
        readout_one_qubit = parameters["readout_one_qubit"]

        if isinstance(depolarizing_one_qubit, (float, int)):
            self.add(
                DepolarizingError(depolarizing_one_qubit),
                conditions=_Conditions().condition_gate_single,
            )

        if isinstance(depolarizing_one_qubit, dict):
            for qubit_key, lamb in depolarizing_one_qubit.items():
                self.add(
                    DepolarizingError(lamb),
                    qubits=int(qubit_key),
                    conditions=_Conditions().condition_gate_single,
                )

        if isinstance(depolarizing_two_qubit, (float, int)):
            self.add(
                DepolarizingError(depolarizing_two_qubit),
                conditions=_Conditions().condition_gate_two,
            )

        if isinstance(depolarizing_two_qubit, dict):
            for key, lamb in depolarizing_two_qubit.items():
                qubits = key.replace(" ", "").split("-")
                qubits = tuple(map(int, qubits))
                self.add(
                    DepolarizingError(lamb),
                    qubits=qubits,
                    conditions=[
                        _Conditions().condition_gate_two,
                        _Conditions(qubits).condition_qubits,
                    ],
                )

        if isinstance(t_1, (float, int)) and isinstance(t_2, (float, int)):
            self.add(
                ThermalRelaxationError(t_1, t_2, gate_time_1, excited_population),
                conditions=_Conditions().condition_gate_single,
            )
            self.add(
                ThermalRelaxationError(t_1, t_2, gate_time_2, excited_population),
                conditions=_Conditions().condition_gate_two,
            )

        if isinstance(t_1, dict) and isinstance(t_2, dict):
            for qubit_key in t_1.keys():
                self.add(
                    ThermalRelaxationError(
                        t_1[qubit_key], t_2[qubit_key], gate_time_1, excited_population
                    ),
                    qubits=int(qubit_key),
                    conditions=_Conditions().condition_gate_single,
                )
                self.add(
                    ThermalRelaxationError(
                        t_1[qubit_key], t_2[qubit_key], gate_time_2, excited_population
                    ),
                    qubits=int(qubit_key),
                    conditions=_Conditions().condition_gate_two,
                )

        if isinstance(readout_one_qubit, (int, float)):
            probabilities = [
                [1 - readout_one_qubit, readout_one_qubit],
                [readout_one_qubit, 1 - readout_one_qubit],
            ]
            self.add(ReadoutError(probabilities), gate=gates.M)

        if isinstance(readout_one_qubit, dict):
            for qubit, probs in readout_one_qubit.items():
                if isinstance(probs, (int, float)):
                    probs = (probs, probs)
                elif isinstance(probs, (tuple, list)) and len(probs) == 1:
                    probs *= 2

                probabilities = [[1 - probs[0], probs[0]], [probs[1], 1 - probs[1]]]
                self.add(
                    ReadoutError(probabilities),
                    gate=gates.M,
                    qubits=int(qubit),
                )
