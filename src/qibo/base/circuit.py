# -*- coding: utf-8 -*-
# @authors: S. Carrazza and A. Garcia
from abc import ABCMeta, abstractmethod
from qibo.base import gates
from typing import Dict, Iterable, Optional, Set, Tuple, Union
NoiseMapType = Union[Tuple[int, int, int],
                     Dict[int, Tuple[int, int, int]]]

QASM_GATES = {"h", "x", "y", "z",
              "rx", "ry", "rz",
              "cx", "swap", "crz", "ccx"}


class BaseCircuit(object):
    """Circuit object which holds a list of gates.

    This circuit is symbolic and cannot perform calculations.
    A specific backend (eg. Tensorflow) has to be used for performing
    calculations (evolving the state vector).
    All backend-based circuits should inherit `BaseCircuit`.

    Args:
        nqubits (int): Total number of qubits in the circuit.

    Example:
        ::

            from qibo.models import Circuit
            from qibo import gates
            c = Circuit(3) # initialized circuit with 3 qubits
            c.add(gates.H(0)) # added Hadamard gate on qubit 0
    """

    __metaclass__ = ABCMeta
    _PARAMETRIZED_GATES = {"rx", "ry", "rz", "crz"}

    def __init__(self, nqubits):
        self.nqubits = nqubits
        self.queue = []
        # Flag to keep track if the circuit was executed
        # We do not allow adding gates in an executed circuit
        self.is_executed = False

        self.measurement_tuples = dict()
        self.measurement_gate = None
        self.measurement_gate_result = None

        self._final_state = None
        self.using_density_matrix = False

    def __add__(self, circuit):
        """Add circuits.

        Args:
            circuit: Circuit to be added to the current one.

        Returns:
            The resulting circuit from the addition.
        """
        return BaseCircuit._circuit_addition(self, circuit)

    @classmethod
    def _circuit_addition(cls, c1, c2):
        if c1.nqubits != c2.nqubits:
            raise ValueError("Cannot add circuits with different number of "
                             "qubits. The first has {} qubits while the "
                             "second has {}".format(c1.nqubits, c2.nqubits))
        newcircuit = cls(c1.nqubits)
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
                    raise KeyError("Register name {} already exists in the "
                                   "circuit.".format(k))
                newcircuit._check_measured(v)
                newcircuit.measurement_tuples[k] = v
            newcircuit.measurement_gate._add(c2.measurement_gate.target_qubits)
        return newcircuit

    def copy(self, deep: bool = False) -> "BaseCircuit":
        """Creates a copy of the current ``circuit`` as a new ``Circuit`` model.

        Args:
            deep (bool): If True new gate objects will be created that act in the same
                qubits as in ``circuit``. Otherwise, the same gate objects of
                ``circuit`` will be used.

        Returns:
            The copied circuit object.
        """
        if deep:
            raise NotImplementedError("Deep copy is not implemented yet.")

        new_circuit = self.__class__(self.nqubits)
        new_circuit.queue = list(self.queue)
        new_circuit.measurement_tuples = dict(self.measurement_tuples)
        new_circuit.measurement_gate = self.measurement_gate
        return new_circuit

    def _check_noise_map(self, noise_map: NoiseMapType) -> NoiseMapType:
        if isinstance(noise_map, tuple) or isinstance(noise_map, list):
            if len(noise_map) != 3:
                raise ValueError("Noise map expects three probabilities "
                                 "but received {}.".format(v))
            return {q: noise_map for q in range(self.nqubits)}
        elif isinstance(noise_map, dict):
            for v in noise_map.values():
                if len(v) != 3:
                    raise ValueError("Noise map expects three probabilities "
                                     "but received {}.".format(v))
            return noise_map

        raise TypeError("Type {} of noise map is not recognized."
                        "".format(type(noise_map)))

    def with_noise(self, noise_channel_class,
                   noise_map: NoiseMapType,
                   measurement_noise: Optional[NoiseMapType] = None
                   ) -> "BaseCircuit":
        noise_map = self._check_noise_map(noise_map)
        if measurement_noise is not None:
            if self.measurement_gate is None:
                raise ValueError("Passed measurement noise but the circuit "
                                 "does not contain measurement gates.")
            measurement_noise = self._check_noise_map(measurement_noise)
            # apply measurement noise only to the qubits that are measured
            # and leave default noise to the rest
            measured_qubits = set(self.measurement_gate.target_qubits)
            measurement_noise = {q: measurement_noise[q] if q in measured_qubits
                                 else noise_map[q] for q in range(self.nqubits)}

        # Generate noise gates
        noise_gates = []
        for gate in self.queue:
            if isinstance(gate, noise_channel_class):
                raise ValueError("`.with_noise` method is not available for "
                                 "circuits that already contain noise channels.")
            noise_gates.append([noise_channel_class(q, *list(p))
                                for q, p in noise_map.items()
                                if sum(p) > 0])
        if measurement_noise is not None:
            noise_gates[-1] = [noise_channel_class(q, *list(p))
                               for q, p in measurement_noise.items()
                               if sum(p) > 0]

        # Create new circuit with noise gates inside
        noisy_circuit = self.__class__(self.nqubits)
        noisy_circuit.queue = []
        for i, gate in enumerate(self.queue):
            # Do not use `circuit.add` here because these gates are already
            # added in the original circuit
            noisy_circuit.queue.append(gate)
            for noise_gate in noise_gates[i]:
                noisy_circuit.add(noise_gate)
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
                raise ValueError("Cannot reuse qubit {} because it is already "
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
            return
        elif not isinstance(gate, gates.Gate):
            raise TypeError("Unknown gate type {}.".format(type(gate)))

        if self._final_state is not None:
            raise RuntimeError("Cannot add gates to a circuit after it is "
                               "executed.")

        # Set number of qubits in gate
        if gate._nqubits is None:
            gate.nqubits = self.nqubits
        elif gate.nqubits != self.nqubits:
            raise ValueError("Attempting to add gate with {} total qubits to "
                             "a circuit with {} qubits."
                             "".format(gate.nqubits, self.nqubits))

        self._check_measured(gate.qubits)
        if gate.name == "measure":
            self._add_measurement(gate)
        else:
            self.queue.append(gate)

    def _add_measurement(self, gate):
        """Gets called automatically by `add` when `gate` is measurement.

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
            raise KeyError("Register name {} has already been used."
                           "".format(name))

        # Update circuit's global measurement gate
        if self.measurement_gate is None:
            self.measurement_gate = gate
            self.measurement_tuples[name] = tuple(gate.target_qubits)
        else:
            self.measurement_gate._add(gate.target_qubits)
            self.measurement_tuples[name] = gate.target_qubits

    @property
    def size(self) -> int:
        """Total number of qubits in the circuit."""
        return self.nqubits

    @property
    def depth(self) -> int:
        """Total number of gates/operations in the circuit."""
        return len(self.queue)

    @property
    def final_state(self):
        """Returns the final state after full simulation of the circuit.

        If the circuit is executed more than once, only the last final state
        is returned.
        """
        raise NotImplementedError

    @abstractmethod
    def execute(self, *args):
        """Executes the circuit. Exact implementation depends on the backend."""
        raise NotImplementedError

    def __call__(self, *args):
        """Equivalent to ``circuit.execute``."""
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
                raise NameError("OpenQASM does not support capital letters in "
                                "register names but {} was used".format(reg_name))
            code.append(f"creg {reg_name}[{len(reg_qubits)}];")

        # Add gates
        for gate in self.queue:
            if gate.name not in QASM_GATES:
                raise ValueError(f"Gate {gate.name} is not supported by OpenQASM.")
            if gate.is_controlled_by:
                raise ValueError("OpenQASM does not support multi-controlled gates.")

            qubits = ",".join(f"q[{i}]" for i in gate.qubits)
            name = gate.name
            if gate.name in self._PARAMETRIZED_GATES:
                # TODO: Make sure that our parameter convention agrees with OpenQASM
                name += f"({gate.theta})"
            code.append(f"{name} {qubits};")

        # Add measurements
        for reg_name, reg_qubits in self.measurement_tuples.items():
            for i, q in enumerate(reg_qubits):
                code.append(f"measure q[{q}] -> {reg_name}[{i}];")

        return "\n".join(code)
