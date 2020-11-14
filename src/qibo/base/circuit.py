# -*- coding: utf-8 -*-
# @authors: S. Carrazza and A. Garcia
import collections
from abc import ABCMeta, abstractmethod
from qibo.base import gates
from qibo import gates as gate_module
from qibo.config import raise_error
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
NoiseMapType = Union[Tuple[int, int, int],
                     Dict[int, Tuple[int, int, int]]]


class _ParametrizedGates(list):
    """Simple data structure for keeping track of parametrized gates.

    Useful for the ``circuit.set_parameters()`` method.
    Holds parametrized gates in a list and a set and also keeps track of the
    total number of parameters.
    """

    def __init__(self):
        super(_ParametrizedGates, self).__init__(self)
        self.set = set()
        self.nparams = 0

    def append(self, gate: gates.ParametrizedGate):
        super(_ParametrizedGates, self).append(gate)
        self.set.add(gate)
        self.nparams += gate.nparams


class _Queue(list):
    """List that holds the queue of gates of a circuit.

    In addition to the queue, it holds a list of gate moments, where each gate
    is placed in the earliest possible position depending for the qubits it acts.
    """

    def __init__(self, nqubits):
        super(_Queue, self).__init__(self)
        self.nqubits = nqubits
        self.moments = [nqubits * [None]]
        self.moment_index = nqubits * [0]

    def append(self, gate: gates.ParametrizedGate):
        super(_Queue, self).append(gate)
        # Calculate moment index for this gate
        if gate.qubits:
            idx = max(self.moment_index[q] for q in gate.qubits)
        for q in gate.qubits:
            if idx >= len(self.moments):
                # Add a moment
                self.moments.append(len(self.moments[-1]) * [None])
            self.moments[idx][q] = gate
            self.moment_index[q] = idx + 1


class BaseCircuit(object):
    """Circuit object which holds a list of gates.

    This circuit is symbolic and cannot perform calculations.
    A specific backend has to be used for performing calculations.
    All backend-based circuits should inherit ``BaseCircuit``.

    Qibo provides the following circuits:
    A state vector simulation circuit:
    :class:`qibo.tensorflow.circuit.TensorflowCircuit`,
    a density matrix simulation circuit:
    :class:`qibo.tensorflow.circuit.TensorflowDensityMatrixCircuit`
    and a circuit that distributes state vector simulation on multiple devices:
    :class:`qibo.tensorflow.distcircuit.TensorflowDistributedCircuit`.
    All circuits use Tensorflow as the computation backend.

    Args:
        nqubits (int): Total number of qubits in the circuit.
    """

    __metaclass__ = ABCMeta
    from qibo.base import fusion

    def __init__(self, nqubits):
        if not isinstance(nqubits, int):
            raise_error(RuntimeError, f'nqubits must be an integer')
        if nqubits < 1:
            raise_error(ValueError, 'nqubits must be > 0')
        self.nqubits = nqubits
        self._init_kwargs = {"nqubits": nqubits}
        self.queue = _Queue(nqubits)
        # Keep track of parametrized gates for the ``set_parameters`` method
        self.parametrized_gates = _ParametrizedGates()
        # Flag to keep track if the circuit was executed
        # We do not allow adding gates in an executed circuit
        self.is_executed = False

        self.measurement_tuples = dict()
        self.measurement_gate = None
        self.measurement_gate_result = None

        self.fusion_groups = []

        self._final_state = None
        self.density_matrix = False
        self.repeated_execution = False

    def __add__(self, circuit) -> "BaseCircuit":
        """Add circuits.

        Args:
            circuit: Circuit to be added to the current one.

        Returns:
            The resulting circuit from the addition.
        """
        return self.__class__._circuit_addition(self, circuit)

    @classmethod
    def _circuit_addition(cls, c1, c2):
        for k, kwarg1 in c1._init_kwargs.items():
            kwarg2 = c2._init_kwargs[k]
            if kwarg1 != kwarg2:
                raise_error(ValueError, "Cannot add circuits with different kwargs. "
                                        "{} is {} for first circuit and {} for the "
                                        "second.".format(k, kwarg1, kwarg2))
        newcircuit = cls(**c1._init_kwargs)
        # Add gates from `c1` to `newcircuit` (including measurements)
        for gate in c1.queue:
            newcircuit.add(gate)
        newcircuit.measurement_gate = c1.measurement_gate
        newcircuit.measurement_tuples = c1.measurement_tuples
        # Add gates from `c2` to `newcircuit` (including measurements)
        for gate in c2.queue:
            newcircuit.add(gate)
        if newcircuit.measurement_gate is None:
            newcircuit.measurement_gate = c2.measurement_gate
            newcircuit.measurement_tuples = c2.measurement_tuples
        elif c2.measurement_gate is not None:
            for k, v in c2.measurement_tuples.items():
                if k in newcircuit.measurement_tuples:
                    raise_error(KeyError, "Register name {} already exists in the "
                                          "circuit.".format(k))
                newcircuit._check_measured(v)
                newcircuit.measurement_tuples[k] = v
            newcircuit.measurement_gate._add(c2.measurement_gate)
        return newcircuit

    def on_qubits(self, *q) -> Iterable[gates.Gate]:
        """Generator of gates contained in the circuit acting on specified qubits.

        Useful for adding a circuit as a subroutine in a larger circuit.

        Args:
            q (int): Qubit ids that the gates should act.

        Example:
            ::

                from qibo import gates, models
                # create small circuit on 4 qubits
                smallc = models.Circuit(4)
                smallc.add((gates.RX(i, theta=0.1) for i in range(4)))
                smallc.add((gates.CNOT(0, 1), gates.CNOT(2, 3)))
                # create large circuit on 8 qubits
                largec = models.Circuit(8)
                largec.add((gates.RY(i, theta=0.1) for i in range(8)))
                # add the small circuit to the even qubits of the large one
                largec.add(smallc.on_qubits(*range(0, 8, 2)))
        """
        if len(q) != self.nqubits:
            raise_error(ValueError, "Cannot return gates on {} qubits because "
                                    "the circuit contains {} qubits."
                                    "".format(len(q), self.nqubits))
        for gate in self.queue:
            yield gate.on_qubits(*(q[i] for i in gate.qubits))

    def copy(self, deep: bool = False) -> "BaseCircuit":
        """Creates a copy of the current ``circuit`` as a new ``Circuit`` model.

        Args:
            deep (bool): If ``True`` copies of the  gate objects will be created
                for the new circuit. If ``False``, the same gate objects of
                ``circuit`` will be used.

        Returns:
            The copied circuit object.
        """
        import copy
        new_circuit = self.__class__(**self._init_kwargs)
        if deep:
            for gate in self.queue:
                new_gate = copy.copy(gate)
                new_circuit.queue.append(new_gate)
                if isinstance(gate, gates.ParametrizedGate):
                    new_circuit.parametrized_gates.append(new_gate)
            new_circuit.measurement_gate = copy.copy(self.measurement_gate)
            if self.fusion_groups: # pragma: no cover
                # impractical case
                raise_error(NotImplementedError, "Cannot create deep copy of fused "
                                                 "circuit.")
        else:
            new_circuit.queue = copy.copy(self.queue)
            new_circuit.parametrized_gates = list(self.parametrized_gates)
            new_circuit.measurement_gate = self.measurement_gate
            new_circuit.fusion_groups = list(self.fusion_groups)
        new_circuit.measurement_tuples = dict(self.measurement_tuples)
        return new_circuit

    def invert(self) -> "BaseCircuit":
        """Creates a new ``Circuit`` that is the inverse of the original.

        Inversion is obtained by taking the dagger of all gates in reverse order.
        If the original circuit contains measurement gates, these are included
        in the inverted circuit.

        Returns:
            The circuit inverse.
        """
        import copy
        new_circuit = self.__class__(**self._init_kwargs)
        for gate in self.queue[::-1]:
            new_circuit.add(gate.dagger())
        new_circuit.measurement_gate = copy.copy(self.measurement_gate)
        new_circuit.measurement_tuples = dict(self.measurement_tuples)
        return new_circuit

    def _fuse_copy(self) -> "BaseCircuit":
        """Helper method for ``circuit.fuse``.

        For standard (non-distributed) circuits this creates a copy of the
        circuit with deep-copying the parametrized gates only.
        For distributed circuits a fully deep copy should be created.
        """
        import copy
        new_circuit = self.__class__(**self._init_kwargs)
        for gate in self.queue:
            if isinstance(gate, gates.ParametrizedGate):
                new_gate = copy.copy(gate)
                new_circuit.queue.append(new_gate)
                new_circuit.parametrized_gates.append(new_gate)
            else:
                new_circuit.queue.append(gate)
        new_circuit.measurement_gate = copy.copy(self.measurement_gate)
        new_circuit.measurement_tuples = dict(self.measurement_tuples)
        return new_circuit

    def fuse(self) -> "BaseCircuit":
        """Creates an equivalent ``Circuit`` with gates fused up to two-qubits.

        Returns:
            The equivalent ``Circuit`` object where the gates are fused.

        Example:
            ::

                from qibo import models, gates
                c = models.Circuit(2)
                c.add([gates.H(0), gates.H(1)])
                c.add(gates.CNOT(0, 1))
                c.add([gates.Y(0), gates.Y(1)])
                # create circuit with fused gates
                fused_c = c.fuse()
                # now ``fused_c`` contains only one ``gates.Unitary`` gate
                # that is equivalent to applying the five gates of the original
                # circuit.
        """
        new_circuit = self._fuse_copy()
        new_circuit.fusion_groups = self.fusion.FusionGroup.from_queue(
            new_circuit.queue)
        new_circuit.queue = list(gate for group in new_circuit.fusion_groups
                                 for gate in group.gates)
        return new_circuit

    def _check_noise_map(self, noise_map: NoiseMapType) -> NoiseMapType:
        if isinstance(noise_map, tuple) or isinstance(noise_map, list):
            if len(noise_map) != 3:
                raise_error(ValueError, "Noise map expects three probabilities "
                                        "but received {}.".format(len(noise_map)))
            return {q: noise_map for q in range(self.nqubits)}
        elif isinstance(noise_map, dict):
            if len(noise_map) != self.nqubits:
                raise_error(ValueError, "Noise map has {} qubits while the circuit "
                                        "has {}.".format(len(noise_map), self.nqubits))
            for v in noise_map.values():
                if len(v) != 3:
                    raise_error(ValueError, "Noise map expects three probabilities "
                                            "but received {}.".format(v))
            return noise_map

        raise_error(TypeError, "Type {} of noise map is not recognized."
                               "".format(type(noise_map)))

    def decompose(self, *free: int) -> "BaseCircuit":
        """Decomposes circuit's gates to gates supported by OpenQASM.

        Args:
            free: Ids of free (work) qubits to use for gate decomposition.

        Returns:
            Circuit that contains only gates that are supported by OpenQASM
            and has the same effect as the original circuit.
        """
        # FIXME: This method is not completed until the ``decompose`` is
        # implemented for all gates not supported by OpenQASM.
        decomp_circuit = self.__class__(self.nqubits)
        for i, gate in enumerate(self.queue):
            decomp_circuit.add(gate.decompose(*free))
        decomp_circuit.measurement_tuples = dict(self.measurement_tuples)
        decomp_circuit.measurement_gate = self.measurement_gate
        return decomp_circuit

    def with_noise(self, noise_map: NoiseMapType) -> "BaseCircuit":
        """Creates a copy of the circuit with noise gates after each gate.

        If the original circuit uses state vectors then noise simulation will
        be done using sampling and repeated circuit execution.
        In order to use density matrices the original circuit should be created
        using the ``density_matrix`` flag set to ``True``.
        For more information we refer to the
        :ref:`How to perform noisy simulation? <noisy-example>` example.

        Args:
            noise_map (dict): Dictionary that maps qubit ids to noise
                probabilities (px, py, pz).
                If a tuple of probabilities (px, py, pz) is given instead of
                a dictionary, then the same probabilities will be used for all
                qubits.

        Returns:
            Circuit object that contains all the gates of the original circuit
            and additional noise channels on all qubits after every gate.

        Example:
            ::

                from qibo.models import Circuit
                from qibo import gates
                # use density matrices for noise simulation
                c = Circuit(2, density_matrix=True)
                c.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])
                noise_map = {0: (0.1, 0.0, 0.2), 1: (0.0, 0.2, 0.1)}
                noisy_c = c.with_noise(noise_map)

                # ``noisy_c`` will be equivalent to the following circuit
                c2 = Circuit(2, density_matrix=True)
                c2.add(gates.H(0))
                c2.add(gates.PauliNoiseChannel(0, 0.1, 0.0, 0.2))
                c2.add(gates.H(1))
                c2.add(gates.PauliNoiseChannel(1, 0.0, 0.2, 0.1))
                c2.add(gates.CNOT(0, 1))
                c2.add(gates.PauliNoiseChannel(0, 0.1, 0.0, 0.2))
                c2.add(gates.PauliNoiseChannel(1, 0.0, 0.2, 0.1))
        """
        noise_map = self._check_noise_map(noise_map)
        # Generate noise gates
        noise_gates = []
        for gate in self.queue:
            if isinstance(gate, gates.KrausChannel):
                raise_error(ValueError, "`.with_noise` method is not available "
                                        "for circuits that already contain "
                                        "channels.")
            noise_gates.append([])
            for q in gate.qubits:
                if q in noise_map and sum(noise_map[q]) > 0:
                    p = noise_map[q]
                    noise_gates[-1].append(gate_module.PauliNoiseChannel(
                            q, px=p[0], py=p[1], pz=p[2]))

        # Create new circuit with noise gates inside
        noisy_circuit = self.__class__(self.nqubits)
        for i, gate in enumerate(self.queue):
            # Do not use `circuit.add` here because these gates are already
            # added in the original circuit
            noisy_circuit.queue.append(gate)
            for noise_gate in noise_gates[i]:
                noisy_circuit.add(noise_gate)
        noisy_circuit.parametrized_gates = list(self.parametrized_gates)
        noisy_circuit.measurement_tuples = dict(self.measurement_tuples)
        noisy_circuit.measurement_gate = self.measurement_gate
        return noisy_circuit

    def _check_measured(self, gate_qubits: Tuple[int]):
        """Helper method for `add`.

        Checks if the qubits that a gate acts are already measured and raises
        a `NotImplementedError` if they are because currently we do not allow
        measured qubits to be reused.
        """
        for qubit in gate_qubits:
            if (self.measurement_gate is not None and
                qubit in self.measurement_gate.target_qubits):
                raise_error(ValueError, "Cannot reuse qubit {} because it is already "
                                        "measured".format(qubit))

    def add(self, gate):
        """Add a gate to a given queue.

        Args:
            gate (:class:`qibo.base.gates.Gate`): the gate object to add.
                See :ref:`Gates` for a list of available gates.
                `gate` can also be an iterable or generator of gates.
                In this case all gates in the iterable will be added in the
                circuit.
        """
        if isinstance(gate, Iterable):
            for g in gate:
                self.add(g)
        elif isinstance(gate, gates.Gate):
            self._add(gate)
        else:
            raise_error(TypeError, "Unknown gate type {}.".format(type(gate)))

    def _add(self, gate: gates.Gate):
        if gate.density_matrix and not self.density_matrix:
            raise_error(ValueError, "Cannot add {} on circuits that uses state "
                                    "vectors. Please switch to density matrix "
                                    "circuit.".format(gate.name))
        elif self.density_matrix:
            gate.density_matrix = True

        if self._final_state is not None:
            raise_error(RuntimeError, "Cannot add gates to a circuit after it is "
                                      "executed.")

        for q in gate.target_qubits:
            if q >= self.nqubits:
                raise_error(ValueError, "Attempting to add gate with target qubits {} "
                                        "on a circuit of {} qubits."
                                        "".format(gate.target_qubits, self.nqubits))

        self._set_nqubits(gate)
        self._check_measured(gate.qubits)
        if isinstance(gate, gates.M):
            self._add_measurement(gate)
        elif isinstance(gate, gates.VariationalLayer):
            self._add_layer(gate)
        elif (isinstance(gate, gates.UnitaryChannel) and
              not self.density_matrix):
            self.repeated_execution = True
            self.queue.append(gate)
        else:
            self.queue.append(gate)
        if isinstance(gate, gates.ParametrizedGate):
            self.parametrized_gates.append(gate)

    def _set_nqubits(self, gate: gates.Gate):
        """Sets the number of qubits in ``gate``.

        Helper method for ``circuit.add(gate)``.
        """
        raise_error(ValueError, "Attempting to add gate with {} total qubits "
                                "to a circuit with {} qubits."
                                "".format(gate.nqubits, self.nqubits))

    def _add_measurement(self, gate: gates.Gate):
        """Called automatically by `add` when `gate` is measurement.

        This is because measurement gates (`gates.M`) are treated differently
        than all other gates.
        The user is not supposed to use the `add_measurement` method.
        """
        # Set register's name and log the set of qubits in `self.measurement_tuples`
        name = gate.register_name
        if name is None:
            name = "register{}".format(len(self.measurement_tuples))
            gate.register_name = name
        elif name in self.measurement_tuples:
            raise_error(KeyError, "Register name {} has already been used."
                                  "".format(name))

        # Update circuit's global measurement gate
        if self.measurement_gate is None:
            self.measurement_gate = gate
            self.measurement_tuples[name] = tuple(gate.target_qubits)
        else:
            self.measurement_gate._add(gate)
            self.measurement_tuples[name] = gate.target_qubits

    def _add_layer(self, gate: gates.Gate):
        """Called automatically by `add` when `gate` is measurement."""
        for unitary in gate.unitaries:
            self._set_nqubits(unitary)
            self.queue.append(unitary)
        if gate.additional_unitary is not None:
            self._set_nqubits(gate.additional_unitary)
            self.queue.append(gate.additional_unitary)

    @property
    def size(self) -> int:
        """Total number of qubits in the circuit."""
        return self.nqubits

    @property
    def ngates(self) -> int:
        """Total number of gates/operations in the circuit."""
        return len(self.queue)

    @property
    def depth(self) -> int:
        """Circuit depth if each gate is placed at the earliest possible position."""
        return len(self.queue.moments)

    @property
    def gate_types(self) -> collections.Counter:
        """``collections.Counter`` with the number of appearances of each gate type.

        The QASM names are used as gate identifiers.
        """
        gates = collections.Counter()
        for gate in self.queue:
            gates[gate.name] += 1
        return gates

    def gates_of_type(self, gate: Union[str, type]) -> List[Tuple[int, gates.Gate]]:
        """Finds all gate objects of specific type.

        Args:
            gate (str, type): The QASM name of a gate or the corresponding gate class.

        Returns:
            List with all gates that are in the circuit and have the same type
            with the given ``gate``. The list contains tuples ``(i, g)`` where
            ``i`` is the index of the gate ``g`` in the circuit's gate queue.
        """
        if isinstance(gate, str):
            return [(i, g) for i, g in enumerate(self.queue)
                    if g.name == gate]
        if isinstance(gate, type) and issubclass(gate, gates.Gate):
            return [(i, g) for i, g in enumerate(self.queue)
                    if isinstance(g, gate)]
        raise_error(TypeError, "Gate identifier {} not recognized.".format(gate))

    def _set_parameters_list(self, parameters: List, n: int):
        """Helper method for ``set_parameters`` when a list is given.

        Also works if ``parameters`` is ``np.ndarray`` or ``tf.Tensor``.
        """
        if n == len(self.parametrized_gates):
            for i, gate in enumerate(self.parametrized_gates):
                gate.parameter = parameters[i]
        elif n == self.parametrized_gates.nparams:
            import numpy as np
            parameters = np.array(parameters)
            k = 0
            for i, gate in enumerate(self.parametrized_gates):
                if gate.nparams == 1:
                    gate.parameter = parameters[i + k]
                else:
                    gate.parameter = parameters[i + k: i + k + gate.nparams]
                k += gate.nparams - 1
        else:
            raise_error(ValueError, "Given list of parameters has length {} while "
                                    "the circuit contains {} parametrized gates."
                                    "".format(n, len(self.parametrized_gates)))

        for fusion_group in self.fusion_groups:
            fusion_group.update()

    def set_parameters(self, parameters: Union[Dict, List]):
        """Updates the parameters of the circuit's parametrized gates.

        For more information on how to use this method we refer to the
        :ref:`How to use parametrized gates?<params-examples>` example.

        Args:
            parameters: List or dictionary with the new parameter values.
                If a list is given its length and elements of this list
                should be compatible with the circuit's parametrized gates.
                If a dictionary is given its keys should be references to the
                parametrized gates.

        Example:
            ::

                from qibo.models import Circuit
                from qibo import gates
                # create a circuit with all parameters set to 0.
                c = Circuit(3, accelerators)
                c.add(gates.RX(0, theta=0))
                c.add(gates.RY(1, theta=0))
                c.add(gates.CZ(1, 2))
                c.add(gates.fSim(0, 2, theta=0, phi=0))
                c.add(gates.H(2))

                # set new values to the circuit's parameters
                params = [0.123, 0.456, (0.789, 0.321)]
                c.set_parameters(params)
        """
        if isinstance(parameters, (list, tuple)):
            self._set_parameters_list(parameters, len(parameters))
        elif isinstance(parameters, dict):
            if self.fusion_groups:
                raise_error(TypeError, "Cannot accept new parameters as dictionary "
                                       "for fused circuits. Use list, tuple or array.")
            if set(parameters.keys()) != self.parametrized_gates.set:
                raise_error(ValueError, "Dictionary with gate parameters does not "
                                        "agree with the circuit gates.")
            for gate in self.parametrized_gates:
                gate.parameter = parameters[gate]
        else:
            raise_error(TypeError, "Invalid type of parameters {}."
                                   "".format(type(parameters)))

    def get_parameters(self, format: str = "list") -> Union[List, Dict]:
        """Returns the parameters of all parametrized gates in the circuit.

        Inverse method of :meth:`qibo.base.circuit.BaseCircuit.set_parameters`.

        Args:
            format: How to return the variational parameters.
                Available formats are 'list', 'dict' and 'flatlist'.
                See :meth:`qibo.base.circuit.BaseCircuit.set_parameters` for more
                details on each format.
        """
        if format == "list":
            return [gate.parameter for gate in self.parametrized_gates]
        elif format == "dict":
            return {gate: gate.parameter for gate in self.parametrized_gates}
        elif format == "flatlist":
            import numpy as np
            from collections.abc import Iterable
            params = []
            for gate in self.parametrized_gates:
                if isinstance(gate.parameter, np.ndarray):
                    params.extend(gate.parameter.ravel())
                elif isinstance(gate.parameter, Iterable):
                    params.extend(gate.parameter)
                else:
                    params.append(gate.parameter)
            return params
        else:
            raise_error(ValueError, f"Unknown format {format} given in ``get_parameters``.")

    def summary(self) -> str:
        """Generates a summary of the circuit.

        The summary contains the circuit depths, total number of qubits and
        the all gates sorted in decreasing number of appearance.

        Example:
            ::

                from qibo.models import Circuit
                from qibo import gates
                c = Circuit(3)
                c.add(gates.H(0))
                c.add(gates.H(1))
                c.add(gates.CNOT(0, 2))
                c.add(gates.CNOT(1, 2))
                c.add(gates.H(2))
                c.add(gates.TOFFOLI(0, 1, 2))
                print(c.summary())
                # Prints
                '''
                Circuit depth = 5
                Total number of gates = 7
                Number of qubits = 3
                Most common gates:
                h: 3
                cx: 2
                ccx: 1
                '''
        """
        logs = [f"Circuit depth = {self.depth}",
                f"Total number of gates = {self.ngates}",
                f"Number of qubits = {self.nqubits}",
                "Most common gates:"]
        common_gates = self.gate_types.most_common()
        logs.extend("{}: {}".format(g, n) for g, n in common_gates)
        return "\n".join(logs)

    @property
    def final_state(self): # pragma: no cover
        """Returns the final state after full simulation of the circuit.

        If the circuit is executed more than once, only the last final state
        is returned.
        """
        # abstract method
        raise_error(NotImplementedError)

    @abstractmethod
    def execute(self, *args): # pragma: no cover
        """Executes the circuit. Exact implementation depends on the backend.

        See :meth:`qibo.tensorflow.circuit.TensorflowCircuit.execute` for more
        details.
        """
        # abstract method
        raise_error(NotImplementedError)

    def __call__(self, *args): # pragma: no cover
        """Equivalent to ``circuit.execute``."""
        # abstract method
        return self.execute(*args)

    def to_qasm(self):
        """Convert circuit to QASM.

        Args:
            filename (str): The filename where the code is saved.
        """
        from qibo import __version__
        code = [f"// Generated by QIBO {__version__}"]
        code += ["OPENQASM 2.0;"]
        code += ["include \"qelib1.inc\";"]
        code += [f"qreg q[{self.nqubits}];"]

        # Set measurements
        for reg_name, reg_qubits in self.measurement_tuples.items():
            if not reg_name.islower():
                raise_error(NameError, "OpenQASM does not support capital letters in "
                                       "register names but {} was used".format(reg_name))
            code.append(f"creg {reg_name}[{len(reg_qubits)}];")

        # Add gates
        for gate in self.queue:
            if gate.name not in gates.QASM_GATES:
                raise_error(ValueError, f"Gate {gate.name} is not supported by OpenQASM.")
            if gate.is_controlled_by:
                raise_error(ValueError, "OpenQASM does not support multi-controlled gates.")

            qubits = ",".join(f"q[{i}]" for i in gate.qubits)
            if gate.name in gates.PARAMETRIZED_GATES:
                if isinstance(gate.parameter, Iterable):
                    params = (str(x) for x in gate.parameter)
                    name = "{}({})".format(gate.name, ", ".join(params))
                else:
                    name = f"{gate.name}({gate.parameter})"
            else:
                name = gate.name
            code.append(f"{name} {qubits};")

        # Add measurements
        for reg_name, reg_qubits in self.measurement_tuples.items():
            for i, q in enumerate(reg_qubits):
                code.append(f"measure q[{q}] -> {reg_name}[{i}];")

        return "\n".join(code)

    @classmethod
    def from_qasm(cls, qasm_code: str, **kwargs) -> "BaseCircuit":
        """Constructs a circuit from QASM code.

        Args:
            qasm_code (str): String with the QASM script.

        Returns:
            A :class:`qibo.base.circuit.BaseCircuit` that contains the gates
            specified by the given QASM script.

        Example:
            ::

                from qibo import models, gates

                qasm_code = '''OPENQASM 2.0;
                include "qelib1.inc";
                qreg q[2];
                h q[0];
                h q[1];
                cx q[0],q[1];'''
                c = models.Circuit.from_qasm(qasm_code)

                # is equivalent to creating the following circuit
                c2 = models.Circuit(2)
                c2.add(gates.H(0))
                c2.add(gates.H(1))
                c2.add(gates.CNOT(0, 1))
        """
        kwargs["nqubits"], gate_list = cls._parse_qasm(qasm_code)
        circuit = cls(**kwargs)
        for gate_name, qubits, params in gate_list:
            gate = getattr(gate_module, gate_name)
            if gate_name == "M":
                circuit.add(gate(*qubits, register_name=params))
            elif params is None:
                circuit.add(gate(*qubits))
            else:
                # assume parametrized gate
                circuit.add(gate(*qubits, *params))
        return circuit

    @staticmethod
    def _parse_qasm(qasm_code: str
                    ) -> Tuple[int,
                               List[Tuple[str, List[int],
                                          Optional[Union[str, float]]]]]:
        """Extracts circuit information from QASM script.

        Helper method for ``from_qasm``.

        Args:
            qasm_code: String with the QASM code to parse.

        Returns:
            nqubits: The total number of qubits in the circuit.
            gate_list: List that specifies the gates of the circuit.
                Contains tuples of the form
                (Qibo gate name, qubit IDs, optional additional parameter).
                The additional parameter is the ``register_name`` for
                measurement gates or ``theta`` for parametrized gates.
        """
        import re
        def read_args(args):
            _args = iter(re.split("[\[\],]", args))
            for name in _args:
                if name:
                    index = next(_args)
                    if not index.isdigit():
                        raise_error(ValueError, "Invalid QASM qubit arguments:", args)
                    yield name, int(index)

        # Remove comment lines
        lines = "".join(line for line in qasm_code.split("\n")
                        if line and line[:2] != "//")
        lines = (line for line in lines.split(";") if line)

        if next(lines) != "OPENQASM 2.0":
            raise_error(ValueError, "QASM code should start with 'OPENQASM 2.0'.")

        qubits = {} # Dict[Tuple[str, int], int]: map from qubit tuple to qubit id
        cregs_size = {} # Dict[str, int]: map from `creg` name to its size
        registers = {} # Dict[str, List[int]]: map from register names to target qubit ids
        gate_list = [] # List[Tuple[str, List[int]]]: List of (gate name, list of target qubit ids)
        for line in lines:
            command, args = line.split(None, 1)
            # remove spaces
            command = command.replace(" ", "")
            args = args.replace(" ", "")

            if command == "include":
                pass

            elif command == "qreg":
                for name, nqubits in read_args(args):
                    for i in range(nqubits):
                        qubits[(name, i)] = len(qubits)

            elif command == "creg":
                for name, nqubits in read_args(args):
                    cregs_size[name] = nqubits

            elif command == "measure":
                args = args.split("->")
                if len(args) != 2:
                    raise_error(ValueError, "Invalid QASM measurement:", line)
                qubit = next(read_args(args[0]))
                if qubit not in qubits:
                    raise_error(ValueError, "Qubit {} is not defined in QASM code."
                                            "".format(qubit))

                register, idx = next(read_args(args[1]))
                if register not in cregs_size:
                    raise_error(ValueError, "Classical register name {} is not defined "
                                            "in QASM code.".format(register))
                if idx >= cregs_size[register]:
                    raise_error(ValueError, "Cannot access index {} of register {} "
                                            "with {} qubits."
                                            "".format(idx, register, cregs_size[register]))
                if register in registers:
                    if idx in registers[register]:
                        raise_error(KeyError, "Key {} of register {} has already "
                                              "been used.".format(idx, register))
                    registers[register][idx] = qubits[qubit]
                else:
                    registers[register] = {idx: qubits[qubit]}
                    gate_list.append(("M", register, None))

            else:
                pieces = [x for x in re.split("[()]", command) if x]
                if len(pieces) == 1:
                    gatename, params = pieces[0], None
                    if gatename not in gates.QASM_GATES:
                        raise_error(ValueError, "QASM command {} is not recognized."
                                                "".format(command))
                    if gatename in gates.PARAMETRIZED_GATES:
                        raise_error(ValueError, "Missing parameters for QASM "
                                                "gate {}.".format(gatename))

                elif len(pieces) == 2:
                    gatename, params = pieces
                    if gatename not in gates.PARAMETRIZED_GATES:
                        raise_error(ValueError, "Invalid QASM command {}."
                                                "".format(command))
                    params = params.replace(" ", "").split(",")
                    try:
                        for i, p in enumerate(params):
                            if 'pi' in p:
                                import numpy as np
                                s = p.replace('pi', str(np.pi)).split('*')
                                p = np.prod([float(j) for j in s], axis=0)
                            params[i] = float(p)
                    except ValueError:
                        raise_error(ValueError, "Invalid value {} for gate parameters."
                                                "".format(params))

                else:
                    raise_error(ValueError, "QASM command {} is not recognized."
                                            "".format(command))

                # Add gate to gate list
                qubit_list = []
                for qubit in read_args(args):
                    if qubit not in qubits:
                        raise_error(ValueError, "Qubit {} is not defined in QASM "
                                                "code.".format(qubit))
                    qubit_list.append(qubits[qubit])
                gate_list.append((gates.QASM_GATES[gatename],
                                  list(qubit_list),
                                  params))

        # Create measurement gate qubit lists from registers
        for i, (gatename, register, _) in enumerate(gate_list):
            if gatename == "M":
                qubit_list = registers[register]
                qubit_list = [qubit_list[k] for k in sorted(qubit_list.keys())]
                gate_list[i] = ("M", qubit_list, register)

        return len(qubits), gate_list
