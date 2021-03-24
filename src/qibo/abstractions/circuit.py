# -*- coding: utf-8 -*-
# @authors: S. Carrazza and A. Garcia
import collections
from abc import ABC, abstractmethod
from qibo.abstractions import gates
from qibo import gates as gate_module
from qibo.config import raise_error
from typing import Dict, List, Optional, Tuple, Union
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

    def append(self, gate: gates.Gate):
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


class AbstractCircuit(ABC):
    """Circuit object which holds a list of gates.

    This circuit is symbolic and cannot perform calculations.
    A specific backend has to be used for performing calculations.
    All backend-based circuits should inherit ``AbstractCircuit``.

    Qibo provides the following circuits:
    A state vector simulation circuit:
    :class:`qibo.core.circuit.Circuit`,
    a density matrix simulation circuit:
    :class:`qibo.core.circuit.DensityMatrixCircuit`
    and a circuit that distributes state vector simulation on multiple devices:
    :class:`qibo.core.distcircuit.DistributedCircuit`.
    All circuits use core as the computation backend.

    Args:
        nqubits (int): Total number of qubits in the circuit.
    """

    def __init__(self, nqubits):
        if not isinstance(nqubits, int):
            raise_error(TypeError, "Number of qubits must be an integer but "
                                   "is {}.".format(nqubits))
        if nqubits < 1:
            raise_error(ValueError, "Number of qubits must be positive but is "
                                    "{}.".format(nqubits))
        self.nqubits = nqubits
        self.init_kwargs = {"nqubits": nqubits}
        self.queue = _Queue(nqubits)
        # Keep track of parametrized gates for the ``set_parameters`` method
        self.parametrized_gates = _ParametrizedGates()
        self.trainable_gates = _ParametrizedGates()
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

        self.param_tensor_types = (None.__class__,)

    def __add__(self, circuit):
        """Add circuits.

        Args:
            circuit: Circuit to be added to the current one.

        Returns:
            The resulting circuit from the addition.
        """
        for k, kwarg1 in self.init_kwargs.items():
            kwarg2 = circuit.init_kwargs[k]
            if kwarg1 != kwarg2:
                raise_error(ValueError,
                            "Cannot add circuits with different kwargs. "
                            "{} is {} for first circuit and {} for the "
                            "second.".format(k, kwarg1, kwarg2))

        newcircuit = self.__class__(**self.init_kwargs)
        # Add gates from `self` to `newcircuit` (including measurements)
        for gate in self.queue:
            newcircuit.queue.append(gate)
            if isinstance(gate, gates.ParametrizedGate):
                newcircuit.parametrized_gates.append(gate)
                if gate.trainable:
                    newcircuit.trainable_gates.append(gate)
        newcircuit.measurement_gate = self.measurement_gate
        newcircuit.measurement_tuples = self.measurement_tuples
        # Add gates from `circuit` to `newcircuit` (including measurements)
        for gate in circuit.queue:
            newcircuit.check_measured(gate.qubits)
            newcircuit.queue.append(gate)
            if isinstance(gate, gates.ParametrizedGate):
                newcircuit.parametrized_gates.append(gate)
                if gate.trainable:
                    newcircuit.trainable_gates.append(gate)

        if newcircuit.measurement_gate is None:
            newcircuit.measurement_gate = circuit.measurement_gate
            newcircuit.measurement_tuples = circuit.measurement_tuples
        elif circuit.measurement_gate is not None:
            for k, v in circuit.measurement_tuples.items():
                if k in newcircuit.measurement_tuples:
                    raise_error(KeyError, "Register name {} already exists in "
                                          "circuit.".format(k))
                newcircuit.check_measured(v)
                newcircuit.measurement_tuples[k] = v
            newcircuit.measurement_gate.add(circuit.measurement_gate)
        return newcircuit

    def on_qubits(self, *q):
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
            yield gate.on_qubits(*q)

    def copy(self, deep: bool = False):
        """Creates a copy of the current ``circuit`` as a new ``Circuit`` model.

        Args:
            deep (bool): If ``True`` copies of the  gate objects will be created
                for the new circuit. If ``False``, the same gate objects of
                ``circuit`` will be used.

        Returns:
            The copied circuit object.
        """
        import copy
        new_circuit = self.__class__(**self.init_kwargs)
        if deep:
            for gate in self.queue:
                new_gate = copy.copy(gate)
                new_circuit.queue.append(new_gate)
                if isinstance(gate, gates.ParametrizedGate):
                    new_circuit.parametrized_gates.append(new_gate)
                    if gate.trainable:
                        new_circuit.trainable_gates.append(new_gate)
            new_circuit.measurement_gate = copy.copy(self.measurement_gate)
            if self.fusion_groups: # pragma: no cover
                # impractical case
                raise_error(NotImplementedError, "Cannot create deep copy of fused "
                                                 "circuit.")
        else:
            new_circuit.queue = copy.copy(self.queue)
            new_circuit.parametrized_gates = list(self.parametrized_gates)
            new_circuit.trainable_gates = list(self.trainable_gates)
            new_circuit.measurement_gate = self.measurement_gate
            new_circuit.fusion_groups = list(self.fusion_groups)
        new_circuit.measurement_tuples = dict(self.measurement_tuples)
        return new_circuit

    def invert(self):
        """Creates a new ``Circuit`` that is the inverse of the original.

        Inversion is obtained by taking the dagger of all gates in reverse order.
        If the original circuit contains measurement gates, these are included
        in the inverted circuit.

        Returns:
            The circuit inverse.
        """
        import copy
        new_circuit = self.__class__(**self.init_kwargs)
        for gate in self.queue[::-1]:
            new_circuit.add(gate.dagger())
        new_circuit.measurement_gate = copy.copy(self.measurement_gate)
        new_circuit.measurement_tuples = dict(self.measurement_tuples)
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

    def decompose(self, *free: int):
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
        for gate in self.queue:
            decomp_circuit.add(gate.decompose(*free))
        decomp_circuit.measurement_tuples = dict(self.measurement_tuples)
        decomp_circuit.measurement_gate = self.measurement_gate
        return decomp_circuit

    def with_noise(self, noise_map: NoiseMapType):
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
        noisy_circuit.trainable_gates = list(self.trainable_gates)
        noisy_circuit.measurement_tuples = dict(self.measurement_tuples)
        noisy_circuit.measurement_gate = self.measurement_gate
        return noisy_circuit

    def check_measured(self, gate_qubits: Tuple[int]):
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
            gate (:class:`qibo.abstractions.gates.Gate`): the gate object to add.
                See :ref:`Gates` for a list of available gates.
                `gate` can also be an iterable or generator of gates.
                In this case all gates in the iterable will be added in the
                circuit.

        Returns:
            If the circuit contains measurement gates with ``collapse=True``
            a ``sympy.Symbol`` that parametrizes the corresponding outcome.
        """
        if isinstance(gate, collections.abc.Iterable):
            for g in gate:
                self.add(g)
        elif isinstance(gate, gates.Gate):
            return self._add(gate)
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

        self.check_measured(gate.qubits)
        if isinstance(gate, gates.M) and not gate.collapse:
            self._add_measurement(gate)
        elif isinstance(gate, gates.VariationalLayer):
            self._add_layer(gate)
        else:
            self.set_nqubits(gate)
            self.queue.append(gate)
            if isinstance(gate, gates.M):
                self.repeated_execution = True
                return gate.symbol()
            if isinstance(gate, gates.UnitaryChannel):
                self.repeated_execution = not self.density_matrix
        if isinstance(gate, gates.ParametrizedGate):
            self.parametrized_gates.append(gate)
            if gate.trainable:
                self.trainable_gates.append(gate)

    def set_nqubits(self, gate: gates.Gate):
        """Sets the number of qubits and prepares all gates.

        Helper method for ``circuit.add(gate)``.
        """
        if gate.is_prepared and gate.nqubits != self.nqubits:
            raise_error(RuntimeError, "Cannot add gate {} that acts on {} "
                                      "qubits to circuit that contains {}"
                                      "qubits.".format(
                                            gate, gate.nqubits, self.nqubits))

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
            self.measurement_gate.add(gate)
            self.measurement_tuples[name] = gate.target_qubits

    @abstractmethod
    def _add_layer(self, gate: gates.Gate): # pragma: no cover
        """Called automatically by `add` when `gate` is measurement."""
        raise_error(NotImplementedError)

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
        gatecounter = collections.Counter()
        for gate in self.queue:
            gatecounter[gate.name] += 1
        return gatecounter

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
        if n == len(self.trainable_gates):
            for i, gate in enumerate(self.trainable_gates):
                gate.parameters = parameters[i]
        elif n == self.trainable_gates.nparams:
            parameters = list(parameters)
            k = 0
            for i, gate in enumerate(self.trainable_gates):
                if gate.nparams == 1:
                    gate.parameters = parameters[i + k]
                else:
                    gate.parameters = parameters[i + k: i + k + gate.nparams]
                k += gate.nparams - 1
        else:
            raise_error(ValueError, "Given list of parameters has length {} while "
                                    "the circuit contains {} parametrized gates."
                                    "".format(n, len(self.trainable_gates)))

        for fusion_group in self.fusion_groups:
            fusion_group.update()

    def set_parameters(self, parameters):
        """Updates the parameters of the circuit's parametrized gates.

        For more information on how to use this method we refer to the
        :ref:`How to use parametrized gates?<params-examples>` example.

        Args:
            parameters: Container holding the new parameter values.
                It can have one of the following types:
                List with length equal to the number of parametrized gates and
                each of its elements compatible with the corresponding gate.
                Dictionary with keys that are references to the parametrized
                gates and values that correspond to the new parameters for
                each gate.
                Flat list with length equal to the total number of free
                parameters in the circuit.
                A backend supported tensor (for example ``np.ndarray`` or
                ``tf.Tensor``) may also be given instead of a flat list.


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

                # set new values to the circuit's parameters using list
                params = [0.123, 0.456, (0.789, 0.321)]
                c.set_parameters(params)
                # or using dictionary
                params = {c.queue[0]: 0.123, c.queue[1]: 0.456
                          c.queue[3]: (0.789, 0.321)}
                c.set_parameters(params)
                # or using flat list (or an equivalent `np.array`/`tf.Tensor`)
                params = [0.123, 0.456, 0.789, 0.321]
                c.set_parameters(params)
        """
        if isinstance(parameters, (list, tuple)):
            self._set_parameters_list(parameters, len(parameters))
        elif isinstance(parameters, self.param_tensor_types):
            self._set_parameters_list(parameters, int(parameters.shape[0]))
        elif isinstance(parameters, dict):
            if self.fusion_groups:
                raise_error(TypeError, "Cannot accept new parameters as dictionary "
                                       "for fused circuits. Use list, tuple or array.")
            diff = set(parameters.keys()) - self.trainable_gates.set
            if diff:
                raise_error(KeyError, "Dictionary contains gates {} which are "
                                      "not on the list of parametrized gates "
                                      "of the circuit.".format(diff))
            for gate, params in parameters.items():
                gate.parameters = params
        else:
            raise_error(TypeError, "Invalid type of parameters {}."
                                   "".format(type(parameters)))

    def get_parameters(self, format: str = "list",
                       include_not_trainable: bool = False
                       ) -> Union[List, Dict]: # pylint: disable=W0622
        """Returns the parameters of all parametrized gates in the circuit.

        Inverse method of :meth:`qibo.abstractions.circuit.AbstractCircuit.set_parameters`.

        Args:
            format (str): How to return the variational parameters.
                Available formats are ``'list'``, ``'dict'`` and ``'flatlist'``.
                See :meth:`qibo.abstractions.circuit.AbstractCircuit.set_parameters`
                for more details on each format. Default is ``'list'``.
            include_not_trainable (bool): If ``True`` it includes the parameters
                of non-trainable parametrized gates in the returned list or
                dictionary. Default is ``False``.
        """
        if include_not_trainable:
            parametrized_gates = self.parametrized_gates
        else:
            parametrized_gates = self.trainable_gates

        if format == "list":
            params = [gate.parameters for gate in parametrized_gates]
        elif format == "dict":
            params = {gate: gate.parameters for gate in parametrized_gates}
        elif format == "flatlist":
            params = []
            for gate in parametrized_gates:
                if isinstance(gate.parameters, self.param_tensor_types):
                    def traverse(x):
                        if isinstance(x, self.param_tensor_types):
                            for v1 in x:
                                for v2 in traverse(v1):
                                    yield v2
                        else:
                            yield x
                    params.extend(traverse(gate.parameters))
                elif isinstance(gate.parameters, collections.abc.Iterable):
                    params.extend(gate.parameters)
                else:
                    params.append(gate.parameters)
        else:
            raise_error(ValueError, "Unknown format {} given in "
                                    "``get_parameters``.".format(format))
        return params

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

    @abstractmethod
    def fuse(self): # pragma: no cover
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
        raise_error(NotImplementedError)

    @property
    @abstractmethod
    def final_state(self): # pragma: no cover
        """Returns the final state after full simulation of the circuit.

        If the circuit is executed more than once, only the last final state
        is returned.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def execute(self, initial_state=None, nshots=None): # pragma: no cover
        """Executes the circuit. Exact implementation depends on the backend.

        See :meth:`qibo.core.circuit.Circuit.execute` for more
        details.
        """
        raise_error(NotImplementedError)

    def __call__(self, initial_state=None, nshots=None):
        """Equivalent to ``circuit.execute``."""
        return self.execute(initial_state=initial_state, nshots=nshots)

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
                if isinstance(gate.parameters, collections.abc.Iterable):
                    params = (str(x) for x in gate.parameters)
                    name = "{}({})".format(gate.name, ", ".join(params))
                else:
                    name = f"{gate.name}({gate.parameters})"
            else:
                name = gate.name
            code.append(f"{name} {qubits};")

        # Add measurements
        for reg_name, reg_qubits in self.measurement_tuples.items():
            for i, q in enumerate(reg_qubits):
                code.append(f"measure q[{q}] -> {reg_name}[{i}];")

        return "\n".join(code)

    @classmethod
    def from_qasm(cls, qasm_code: str, **kwargs):
        """Constructs a circuit from QASM code.

        Args:
            qasm_code (str): String with the QASM script.

        Returns:
            A :class:`qibo.abstractions.circuit.AbstractCircuit` that contains the gates
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
            _args = iter(re.split(r"[\[\],]", args))
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
                                import math
                                from operator import mul
                                from functools import reduce
                                s = p.replace('pi', str(math.pi)).split('*')
                                p = reduce(mul, [float(j) for j in s], 1)
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

    def draw(self, line_wrap=70) -> str:
        """Draw text circuit using unicode symbols.

        Args:
            line_wrap (int): maximum number of characters per line. This option
                split the circuit text diagram in chunks of line_wrap characters.

        Return:
            String containing text circuit diagram.
        """
        labels = {"h": "H", "x": "X", "y": "Y", "z": "Z",
                  "rx": "RX", "ry": "RY", "rz": "RZ",
                  "u1": "U1", "u2": "U2", "u3": "U3",
                  "cx": "X", "swap": "x", "cz": "Z",
                  "crx": "RX", "cry": "RY", "crz": "RZ",
                  "cu1": "U1", "cu3": "U3", "ccx": "X",
                  "id": "I", "measure": "M", "fsim": "f",
                  "generalizedfsim": "gf"}

        # build string representation of gates
        matrix = [[] for _ in range(self.nqubits)]
        idx = [0] * self.nqubits

        for gate in self.queue:
            if gate.name not in labels:
                raise_error(NotImplementedError, f"{gate.name} gate is not supported by `circuit.draw`")
            gate_name = labels.get(gate.name)
            targets = list(gate.target_qubits)
            controls = list(gate.control_qubits)

            # identify boundaries
            qubits = targets + controls
            qubits.sort()
            min_qubits_id = qubits[0]
            max_qubits_id = qubits[-1]

            # identify column
            col = idx[targets[0]] if not controls and len(targets) == 1 else max(idx)

            # extend matrix
            for iq in range(self.nqubits):
                matrix[iq].extend((1 + col - len(matrix[iq]))* [''])

            # fill
            for iq in range(min_qubits_id, max_qubits_id + 1):
                if iq in targets:
                    matrix[iq][col] = gate_name
                elif iq in controls:
                    matrix[iq][col] = 'o'
                else:
                    matrix[iq][col] = '|'

            # update indexes
            if not controls and len(targets) == 1:
                idx[targets[0]] += 1
            else:
                idx = [col + 1] * self.nqubits

        # Include measurement gates
        if self.measurement_gate:
            for iq in range(self.nqubits):
                matrix[iq].append('M' if iq in self.measurement_gate.target_qubits else '')

        # Add some spacers
        for col in range(len(matrix[0])):
            maxlen = max([len(matrix[l][col]) for l in range(self.nqubits)])
            for row in range(self.nqubits):
                matrix[row][col] += '─' * (1 + maxlen - len(matrix[row][col]))

        # Print to terminal
        output = ""
        for q in range(self.nqubits):
            output += f'q{q}' + ' ' * (len(str(self.nqubits))-len(str(q))) + \
                       ': ─' + ''.join(matrix[q]) + '\n'

        # line wrap
        if line_wrap:
            loutput = output.splitlines()
            def chunkstring(string, length):
                nchunks = range(0, len(string), length)
                return (string[i:length+i] for i in nchunks), len(nchunks)
            for row in range(self.nqubits):
                chunks, nchunks = chunkstring(loutput[row][3 + len(str(self.nqubits)):], line_wrap)
                if nchunks == 1:
                    loutput = None
                    break
                for i, c in enumerate(chunks):
                    loutput += ['' for _ in range(self.nqubits)]
                    suffix = f' ...\n'
                    prefix = f'q{row}' + ' ' * (len(str(self.nqubits))-len(str(row))) + ': '
                    if i == 0:
                        prefix += ' ' * 4
                    elif row == 0:
                        prefix = '\n' + prefix + '... '
                    else:
                        prefix += '... '
                    if i == nchunks-1:
                        suffix = '\n'
                    loutput[row + i * self.nqubits] = prefix + c + suffix
            if loutput is not None:
                output = ''.join(loutput)

        return output.rstrip('\n')
