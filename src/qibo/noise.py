# %%
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

    # def composite(self, params):
    #     """Build a noise model to simulate the noisy behaviour of a quantum computer.

    #     Args:
    #         params (dict): contains the parameters of the channels organized as follow \n
    #             {'t1' : (``t1``, ``t2``,..., ``tn``),
    #             't2' : (``t1``, ``t2``,..., ``tn``),
    #             'gate time' : (``time1``, ``time2``),
    #             'excited population' : 0,
    #             'depolarizing error' : (``lambda1``, ``lambda2``),
    #             'bitflips error' : ([``p1``, ``p2``,..., ``pm``],[``p1``, ``p2``,..., ``pm``]),
    #             'idle_qubits' : True}
    #             where `n` is the number of qubits, and `m` the number of measurement gates.
    #             The first four parameters are used by the thermal relaxation error.
    #             The first two  elements are the tuple containing the :math:`T_1` and
    #             :math:`T_2` parameters; the third one is a tuple which contain the gate times,
    #             for single and two qubit gates; then we have the excited population parameter.
    #             The fifth parameter is a tuple containing the depolaraziong errors for single
    #             and 2 qubit gate. The sisxth parameter is a tuple containg the two arrays for
    #             bitflips probability errors: the first one implements 0->1 errors, the other
    #             one 1->0. The last parameter is a boolean variable: if ``True`` the noise
    #             model takes into account idle qubits.
    #     """

    #     self.noise_model = CompositeNoiseModel(params)

    def apply(self, circuit):
        """Generate a noisy quantum circuit according to the noise model built.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): quantum circuit

        Returns:
            (:class:`qibo.models.circuit.Circuit`): initial circuit with noise gates
                added according to the noise model.
        """

        noisy_circuit = circuit.__class__(**circuit.init_kwargs)
        for gate in circuit.queue:
            errors_list = (
                self.errors[gate.__class__]
                if (isinstance(gate, gates.Channel) or isinstance(gate, gates.M))
                else self.errors[gate.__class__] + self.errors[None]
            )
            if all(not isinstance(error, ReadoutError) for _, error, _ in errors_list):
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
                                PhaseDampingError,
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

            for x in noisy_circuit.queue:
                print(x.name, x.qubits, x.init_args)
            print()
            print()

        return noisy_circuit


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
        parameters = {
            "t1" : {"0": 0.1, "1": 0.2, "3": 0.01},
            "t2" : {"0": 0.01, "1: 0.2, "3": 0.1},
            "gate_time" : (0.1, 0.2),
            "excited_population" : 0.1,
            "depolarizing_error" : (0.1, 0.2),
            "bitflips_error" : ([0.1 for _ in range(nqubits)], [0.1 for _ in range(4)]),
            }

        noise_model = IBMQNoiseModel()
        noise_model.add(parameters)

        phases = list(range(nqubits))
        circuit = phase_encoder(phases, rotation="RY")
        circuit.add(gates.CNOT(qubit, qubit + 1) for qubit in range(nqubits - 1))
        circuit.add(gates.M(qubit) for qubit in range(1, nqubits - 1))

        noisy_circuit = noise_model.apply(circuit)

        ## Note that the implementation above is equivalent to the one below

        def _condition_single_qubit_gate(gate):
            return len(gate.qubits) == 1

        def _condition_two_qubit_gate(gate):
            return len(gate.qubits) == 2


        noise_model = NoiseModel()
        noise_model.add(
            DepolarizingError(0.1), condition=self._condition_single_qubit_gate
        )
        noise_model.add(DepolarizingError(0.2), gate=gates.CNOT)
        noise_model.add(
            ThermalRelaxationError(0.1, 0.01, 0.1, 0.1),
            qubits=0,
            condition=_condition_single_qubit_gate
        )
        noise_model.add(
            ThermalRelaxationError(0.2, 0.02, 0.1, 0.1),
            qubits=1,
            condition=_condition_single_qubit_gate
        )
        noise_model.add(
            ThermalRelaxationError(0.01, 0.001, 0.1, 0.1),
            qubits=3,
            condition=_condition_single_qubit_gate
        )
        noise_model.add(
            ThermalRelaxationError(0.1, 0.01, 0.2, 0.1),
            qubits=0,
            condition=_condition_two_qubit_gate
        )
        noise_model.add(
            ThermalRelaxationError(0.2, 0.02, 0.2, 0.1),
            qubits=1,
            condition=_condition_two_qubit_gate
        )
        noise_model.add(
            ThermalRelaxationError(0.01, 0.001, 0.2, 0.1),
            qubits=3,
            condition=_condition_two_qubit_gate
        )

        probabilities = [[0.9, 0.1], [0.1, 0.9]]
        noise_model.add(ReadoutError(probabilities), gate=gates.M)

        noisy_circuit = noise_model.apply(circuit)

    """

    def __init__(self):
        super().__init__()

    def _condition_single_qubit_gate(self, gate):
        """Condition that had to be matched to apply noise channel to single-qubit ``gate``."""
        return len(gate.qubits) == 1

    def _condition_two_qubit_gate(self, gate):
        """Condition that had to be matched to apply noise channel to two-qubit ``gate``."""
        return len(gate.qubits) == 2

    def add(self, parameters):
        self.parameters = parameters
        t_1 = self.parameters["t1"]
        t_2 = self.parameters["t2"]
        gate_time_1, gate_time_2 = self.parameters["gate_time"]
        excited_population = self.parameters["excited_population"]
        depolarizing_one_qubit = self.parameters["depolarizing_one_qubit"]
        depolarizing_two_qubit = self.parameters["depolarizing_two_qubit"]
        bitflips_01, bitflips_10 = self.parameters["bitflips_error"]

        for qubit_key, lamb in depolarizing_one_qubit.items():
            super().add(
                DepolarizingError(lamb),
                qubits=int(qubit_key),
                condition=self._condition_single_qubit_gate,
            )

        if isinstance(depolarizing_two_qubit, (float, int)):
            super().add(
                DepolarizingError(depolarizing_two_qubit),
                condition=self._condition_two_qubit_gate,
            )
        elif isinstance(depolarizing_two_qubit, dict):
            for key, lamb in depolarizing_two_qubit.items():
                qubits = key.replace(" ", "").split("-")
                qubits = tuple(map(int, qubits))
                super().add(
                    DepolarizingError(lamb),
                    qubits=qubits,
                    condition=self._condition_two_qubit_gate,
                )

        for qubit_key in t_1.keys():
            super().add(
                ThermalRelaxationError(
                    t_1[qubit_key], t_2[qubit_key], gate_time_1, excited_population
                ),
                qubits=int(qubit_key),
                condition=self._condition_single_qubit_gate,
            )
            super().add(
                ThermalRelaxationError(
                    t_1[qubit_key], t_2[qubit_key], gate_time_2, excited_population
                ),
                qubits=int(qubit_key),
                condition=self._condition_two_qubit_gate,
            )

        for qubit, (p_01, p_10) in enumerate(zip(bitflips_01, bitflips_10)):
            probabilities = [[1 - p_01, p_01], [p_10, 1 - p_10]]
            super().add(
                ReadoutError(probabilities),
                gate=gates.M,
                qubits=qubit,
            )


# %%
from qibo.models.encodings import phase_encoder

nqubits = 4
parameters = {
    "t1": {"0": 0.1, "1": 0.2, "3": 0.01},
    "t2": {"0": 0.01, "1": 0.02, "3": 0.001},
    "gate_time": (0.1, 0.2),
    "excited_population": 0.1,
    "depolarizing_one_qubit": {"0": 0.1, "1": 0.1, "3": 0.1},
    "depolarizing_two_qubit": {"0-1": 0.1},  # "1-2": 0.2, "2-3": 0.3},
    # "depolarizing_two_qubit": 0.2,
    "bitflips_error": ([0.1 for _ in range(nqubits)], [0.1 for _ in range(4)]),
}

noise_model = IBMQNoiseModel()
noise_model.add(parameters)

phases = list(range(nqubits))
circuit = phase_encoder(phases, rotation="RY")
circuit.add(gates.CNOT(qubit, qubit + 1) for qubit in range(nqubits - 1))
circuit.add(gates.M(qubit) for qubit in range(1, nqubits - 1))

noisy_circuit = noise_model.apply(circuit)
print(noisy_circuit.draw())

# %%
for gate in noisy_circuit.queue:
    print(gate.name, gate.qubits)


# %%
def _condition_single_qubit_gate(gate):
    return len(gate.qubits) == 1


def _condition_two_qubit_gate(gate):
    return len(gate.qubits) == 2


noise_model = NoiseModel()
noise_model.add(DepolarizingError(0.1), condition=_condition_single_qubit_gate)
noise_model.add(DepolarizingError(0.2), gate=gates.CNOT)
noise_model.add(
    ThermalRelaxationError(0.1, 0.01, 0.1, 0.1),
    qubits=0,
    condition=_condition_single_qubit_gate,
)
noise_model.add(
    ThermalRelaxationError(0.2, 0.02, 0.1, 0.1),
    qubits=1,
    condition=_condition_single_qubit_gate,
)
noise_model.add(
    ThermalRelaxationError(0.01, 0.001, 0.1, 0.1),
    qubits=3,
    condition=_condition_single_qubit_gate,
)
noise_model.add(
    ThermalRelaxationError(0.1, 0.01, 0.2, 0.1),
    qubits=0,
    condition=_condition_two_qubit_gate,
)
noise_model.add(
    ThermalRelaxationError(0.2, 0.02, 0.2, 0.1),
    qubits=1,
    condition=_condition_two_qubit_gate,
)
noise_model.add(
    ThermalRelaxationError(0.01, 0.001, 0.2, 0.1),
    qubits=3,
    condition=_condition_two_qubit_gate,
)

probabilities = [[0.9, 0.1], [0.1, 0.9]]
noise_model.add(ReadoutError(probabilities), gate=gates.M)

noisy_circuit = noise_model.apply(circuit)

print(noisy_circuit.draw())

# %%
"3- 4".replace(" ", "").split("-")
