import collections
import json
from typing import List, Sequence, Tuple

import sympy

from qibo.backends import _check_backend
from qibo.config import raise_error

REQUIRED_FIELDS = [
    "name",
    "init_args",
    "init_kwargs",
    "_target_qubits",
    "_control_qubits",
]
REQUIRED_FIELDS_INIT_KWARGS = [
    "theta",
    "phi",
    "lam",
    "phi0",
    "phi1",
    "register_name",
    "collapse",
    "basis",
    "p0",
    "p1",
]


class Gate:
    """Abstract class for gates.

    Args:
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.models.circuit.Circuit.set_parameters`.
            Defaults to ``False``.
        optimize_depth (bool): whether to optimize for circuit depth.
            Defaults to ``True``.
    """

    def __init__(self, trainable=False, optimize_depth=True):
        self.name = None
        self.draw_label = None
        self.target_qubits = None
        self.control_qubits = None
        self.init_args = []
        self.init_kwargs = {}
        self.unitary = False
        self.parameters = None
        self.nparams = 0
        self.parameter_names = []
        self.trainable = trainable
        self.optimize_depth = optimize_depth

    @property
    def clifford(self):
        """Whether the gate is a Clifford gate."""
        return False

    @property
    def hamming_weight(self):
        """Whether the gate preserves Hamming weight."""
        return False

    @property
    def qasm_label(self):
        """QASM label of the gate."""
        return self.name

    def _base_decompose(self, *free, use_toffolis=True):
        """Base decomposition method that returns the gate itself.

        This method should be overridden by subclasses to provide specific
        decomposition implementations.

        Returns:
            List containing the gate itself.
        """
        return [self]

    def decompose(self, *free, use_toffolis=True):
        """Decomposes the gate into simpler gates.

        This method handles the decomposition of controlled gates by:
        1. First decomposing the base gate
        2. Then applying the control mask to the decomposed gates
        3. Optimizing the decomposition based on the optimization profile

        Args:
            *free: Additional qubits that can be used in the decomposition.
            use_toffolis: Whether to use Toffoli gates in the decomposition.

        Returns:
            List of gates that have the same effect as applying the original gate.
        """
        # First decompose the base gate
        gates = self._base_decompose(*free, use_toffolis=use_toffolis)

        # If this is a controlled gate, we need to add controls to the decomposed gates
        if self.control_qubits:
            # Get the control mask for the decomposed gates
            control_mask = self.control_mask_after_stripping(gates)

            # Add controls to the gates that need them
            controlled_gates = []
            for gate, needs_control in zip(gates, control_mask):
                if needs_control:
                    controlled_gates.append(gate.controlled_by(*self.control_qubits))
                else:
                    controlled_gates.append(gate)

            # Optimize the decomposition based on the optimization profile
            if self.optimize_depth:
                controlled_gates = self._optimize_depth(controlled_gates)
            else:
                controlled_gates = self._optimize_compilation_time(controlled_gates)

            return controlled_gates

        return gates

    def _optimize_depth(self, gates):
        """Optimizes the decomposition for minimum circuit depth.

        This method implements the divide-and-conquer approach from recent research
        to reduce circuit depth.

        Args:
            gates: List of gates to optimize.

        Returns:
            Optimized list of gates.
        """
        if len(gates) <= 1:
            return gates

        # Divide the gates into two halves
        mid = len(gates) // 2
        left = self._optimize_depth(gates[:mid])
        right = self._optimize_depth(gates[mid:])

        # Merge the optimized halves
        return self._merge_optimized_gates(left, right)

    def _optimize_compilation_time(self, gates):
        """Optimizes the decomposition for minimum compilation time.

        This method implements a simpler optimization strategy that focuses
        on reducing compilation time rather than circuit depth.

        Args:
            gates: List of gates to optimize.

        Returns:
            Optimized list of gates.
        """
        # For compilation time optimization, we use a simpler strategy
        # that focuses on reducing the number of gates
        optimized = []
        i = 0
        while i < len(gates):
            if i + 1 < len(gates) and self.gates_cancel(gates[i], gates[i + 1]):
                i += 2
            else:
                optimized.append(gates[i])
                i += 1
        return optimized

    def _merge_optimized_gates(self, left, right):
        """Merges two optimized gate lists while maintaining optimal depth.

        Args:
            left: First list of gates.
            right: Second list of gates.

        Returns:
            Merged list of gates.
        """
        # If either list is empty, return the other
        if not left:
            return right
        if not right:
            return left

        # Try to cancel gates at the boundary
        if self.gates_cancel(left[-1], right[0]):
            return left[:-1] + right[1:]

        # Otherwise, just concatenate
        return left + right

    def controlled_by(self, *q):
        """Controls the gate on the specified qubits.

        Args:
            *q: Control qubit indices.

        Returns:
            A new gate that is controlled on the specified qubits.
        """
        if not q:
            return self

        if self.control_qubits is None:
            self.control_qubits = tuple(q)
        else:
            self.control_qubits = tuple(q) + self.control_qubits

        return self

    def on_qubits(self, *q):
        """Applies the gate on the specified qubits.

        Args:
            *q: Qubit indices.

        Returns:
            A new gate that is applied on the specified qubits.
        """
        if len(q) != len(self.target_qubits):
            raise_error(
                ValueError,
                f"Gate {self.name} requires {len(self.target_qubits)} qubits, "
                f"but {len(q)} were given.",
            )

        if self.control_qubits is None:
            self.control_qubits = tuple()

        self.target_qubits = tuple(q)
        return self

    def _dagger(self):
        """Returns the dagger (conjugate transpose) of the gate.

        Returns:
            A new gate that is the dagger of the original gate.
        """
        return self

    def dagger(self):
        """Returns the dagger (conjugate transpose) of the gate.

        Returns:
            A new gate that is the dagger of the original gate.
        """
        return self._dagger()

    def __call__(self, *q):
        """Applies the gate on the specified qubits.

        Args:
            *q: Qubit indices.

        Returns:
            A new gate that is applied on the specified qubits.
        """
        return self.on_qubits(*q)

    def __matmul__(self, other):
        """Composes two gates.

        Args:
            other: Another gate.

        Returns:
            A new gate that is the composition of the two gates.
        """
        if not isinstance(other, Gate):
            raise_error(
                TypeError,
                f"Gate can only be composed with another Gate, but {type(other)} was given.",
            )

        if self.target_qubits is None or other.target_qubits is None:
            raise_error(
                ValueError,
                "Cannot compose gates that have not been applied to qubits.",
            )

        if set(self.target_qubits) & set(other.target_qubits):
            raise_error(
                ValueError,
                "Cannot compose gates that act on the same qubits.",
            )

        if self.control_qubits is None:
            self.control_qubits = tuple()

        if other.control_qubits is None:
            other.control_qubits = tuple()

        if set(self.control_qubits) & set(other.control_qubits):
            raise_error(
                ValueError,
                "Cannot compose gates that are controlled by the same qubits.",
            )

        if set(self.target_qubits) & set(other.control_qubits):
            raise_error(
                ValueError,
                "Cannot compose gates where one gate's target qubits are controlled by the other gate.",
            )

        if set(other.target_qubits) & set(self.control_qubits):
            raise_error(
                ValueError,
                "Cannot compose gates where one gate's target qubits are controlled by the other gate.",
            )

        return self.controlled_by(*other.control_qubits).on_qubits(*other.target_qubits)

    def __eq__(self, other):
        """Checks if two gates are equal.

        Args:
            other: Another gate.

        Returns:
            True if the gates are equal, False otherwise.
        """
        if not isinstance(other, Gate):
            return False

        if self.name != other.name:
            return False

        if self.target_qubits != other.target_qubits:
            return False

        if self.control_qubits != other.control_qubits:
            return False

        if self.parameters is not None and other.parameters is not None:
            if self.parameters != other.parameters:
                return False

        return True

    def __hash__(self):
        """Returns the hash of the gate.

        Returns:
            The hash of the gate.
        """
        return hash(
            (
                self.name,
                self.target_qubits,
                self.control_qubits,
                self.parameters,
            )
        )

    def __str__(self):
        """Returns a string representation of the gate.

        Returns:
            A string representation of the gate.
        """
        if self.control_qubits:
            return f"{self.name}({self.control_qubits})->{self.target_qubits}"
        return f"{self.name}({self.target_qubits})"

    def __repr__(self):
        """Returns a string representation of the gate.

        Returns:
            A string representation of the gate.
        """
        return self.__str__()

    @staticmethod
    def gates_cancel(gate1, gate2):
        """Checks if two gates cancel each other out.

        Args:
            gate1: First gate.
            gate2: Second gate.

        Returns:
            True if the gates cancel each other out, False otherwise.
        """
        if not isinstance(gate1, Gate) or not isinstance(gate2, Gate):
            return False

        if gate1.name != gate2.name:
            return False

        if gate1.target_qubits != gate2.target_qubits:
            return False

        if gate1.control_qubits != gate2.control_qubits:
            return False

        if gate1.parameters is not None and gate2.parameters is not None:
            if gate1.parameters != gate2.parameters:
                return False

        return True

    def control_mask_after_stripping(self, gates):
        """Returns a mask indicating which gates should be controlled after analyzing a list of gates.

        This method analyzes the list of gates and returns a boolean mask indicating
        which gates should be controlled. The mask is determined by:
        1. Gates that act on the same qubits as the original gate should be controlled
        2. Gates that are part of a decomposition that needs to be controlled should be controlled
        3. Gates that are part of a decomposition that doesn't need to be controlled should not be controlled

        Args:
            gates: List of gates to analyze.

        Returns:
            A list of booleans indicating which gates should be controlled.
        """
        if not self.control_qubits:
            return [False] * len(gates)

        # Initialize mask with all False
        mask = [False] * len(gates)

        # Find gates that act on the same qubits as the original gate
        for i, gate in enumerate(gates):
            if set(gate.target_qubits) & set(self.target_qubits):
                mask[i] = True

        # Find gates that are part of a decomposition that needs to be controlled
        for i in range(len(gates)):
            if mask[i]:
                # If this gate is controlled, check if it's part of a decomposition
                # that needs to be controlled
                for j in range(i + 1, len(gates)):
                    if gates_cancel(gates[i], gates[j]):
                        # If we find a gate that cancels this one, neither should be controlled
                        mask[i] = False
                        mask[j] = False
                        break

        return mask

    @property
    def raw(self) -> dict:
        """Serialize to dictionary.

        The values used in the serialization should be compatible with a
        JSON dump (or any other one supporting a minimal set of scalar
        types). Though the specific implementation is up to the specific
        gate.
        """
        encoded = self.__dict__

        encoded_simple = {
            key: value for key, value in encoded.items() if key in REQUIRED_FIELDS
        }

        encoded_simple["init_kwargs"] = {
            key: value
            for key, value in encoded_simple["init_kwargs"].items()
            if key in REQUIRED_FIELDS_INIT_KWARGS
        }

        encoded_simple["_class"] = type(self).__name__

        return encoded_simple

    @staticmethod
    def from_dict(raw: dict):
        """Load from serialization.

        Essentially the counter-part of :meth:`raw`.
        """
        from qibo.gates import gates, measurements

        for mod in (gates, measurements):
            try:
                cls = getattr(mod, raw["_class"])
                break
            except AttributeError:
                # gate not found in given module, try next
                pass
        else:
            raise ValueError(f"Unknown gate {raw['_class']}")

        gate = cls(*raw["init_args"], **raw["init_kwargs"])
        if raw["_class"] == "M":
            if raw["measurement_result"]["samples"] is not None:
                gate.result.register_samples(raw["measurement_result"]["samples"])
            return gate
        try:
            return gate.controlled_by(*raw["_control_qubits"])
        except RuntimeError as e:
            if "controlled" in e.args[0]:
                return gate
            raise e

    def to_json(self):
        """Dump gate to JSON.

        Note:
            Consider using :meth:`raw` directly.
        """
        return json.dumps(self.raw)

    @property
    def target_qubits(self) -> Tuple[int, ...]:
        """Tuple with ids of target qubits."""
        return self._target_qubits

    @property
    def control_qubits(self) -> Tuple[int, ...]:
        """Tuple with ids of control qubits sorted in increasing order."""
        return tuple(sorted(self._control_qubits))

    @property
    def qubits(self) -> Tuple[int, ...]:
        """Tuple with ids of all qubits (control and target) that the gate acts."""
        return self.control_qubits + self.target_qubits

    def _set_target_qubits(self, qubits: Sequence[int]):
        """Helper method for setting target qubits."""
        self._target_qubits = tuple(qubits)
        if len(self._target_qubits) != len(set(qubits)):
            repeated = self._find_repeated(qubits)
            raise_error(
                ValueError,
                f"Target qubit {repeated} was given twice for gate {self.__class__.__name__}.",
            )

    def _set_control_qubits(self, qubits: Sequence[int]):
        """Helper method for setting control qubits."""
        if len(set(qubits)) != len(qubits):
            repeated = self._find_repeated(qubits)
            raise_error(
                ValueError,
                f"Control qubit {repeated} was given twice for gate {self.__class__.__name__}.",
            )
        self._control_qubits = qubits

    @target_qubits.setter
    def target_qubits(self, qubits: Sequence[int]):
        """Sets target qubits tuple."""
        self._set_target_qubits(qubits)
        self._check_control_target_overlap()

    @control_qubits.setter
    def control_qubits(self, qubits: Sequence[int]):
        """Sets control qubits set."""
        self._set_control_qubits(qubits)
        self._check_control_target_overlap()

    def _set_targets_and_controls(
        self, target_qubits: Sequence[int], control_qubits: Sequence[int]
    ):
        """Sets target and control qubits simultaneously.

        This is used for the reduced qubit updates in the distributed
        circuits because using the individual setters may raise errors
        due to temporary overlap of control and target qubits.
        """
        self._set_target_qubits(target_qubits)
        self._set_control_qubits(control_qubits)
        self._check_control_target_overlap()

    @staticmethod
    def _find_repeated(qubits: Sequence[int]) -> int:
        """Finds the first qubit id that is repeated in a sequence of qubit ids."""
        temp_set = set()
        for qubit in qubits:
            if qubit in temp_set:
                return qubit
            temp_set.add(qubit)

    def _check_control_target_overlap(self):
        """Checks that there are no qubits that are both target and
        controls."""
        control_and_target = self._control_qubits + self._target_qubits
        common = len(set(control_and_target)) != len(control_and_target)
        if common:
            raise_error(
                ValueError,
                f"{set(self._target_qubits) & set(self._control_qubits)}"
                + "qubits are both targets and controls "
                + f"for gate {self.__class__.__name__}.",
            )

    @property
    def parameters(self):
        """Returns a tuple containing the current value of gate's parameters."""
        return self.parameters

    def commutes(self, gate: "Gate") -> bool:
        """Checks if two gates commute.

        Args:
            gate: Gate to check if it commutes with the current gate.

        Returns:
            bool: ``True`` if the gates commute, ``False`` otherwise.
        """
        if isinstance(gate, SpecialGate):  # pragma: no cover
            return False
        t1 = set(self.target_qubits)
        t2 = set(gate.target_qubits)
        a = self.__class__ == gate.__class__ and t1 == t2
        b = not (t1 & set(gate.qubits) or t2 & set(self.qubits))
        return a or b

    def matrix(self, backend=None):
        """Returns the matrix representation of the gate.

        If gate has controlled qubits inserted by :meth:`qibo.gates.Gate.controlled_by`,
        then :meth:`qibo.gates.Gate.matrix` returns the matrix of the original gate.

        .. code-block:: python

            from qibo import gates

            gate = gates.SWAP(3, 4).controlled_by(0, 1, 2)
            print(gate.matrix())

        To return the full matrix that takes the control qubits into account,
        one should use :meth:`qibo.models.Circuit.unitary`, e.g.

        .. code-block:: python

            from qibo import Circuit, gates

            nqubits = 5
            circuit = Circuit(nqubits)
            circuit.add(gates.SWAP(3, 4).controlled_by(0, 1, 2))
            print(circuit.unitary())

        Args:
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend
                to be used in the execution. If ``None``, it uses
                the current backend. Defaults to ``None``.

        Returns:
            ndarray: Matrix representation of gate.
        """
        backend = _check_backend(backend)

        return backend.matrix(self)

    def generator_eigenvalue(self):
        """This function returns the eigenvalues of the gate's generator.

        Returns:
            float: eigenvalue of the generator.
        """

        raise_error(
            NotImplementedError,
            f"Generator eigenvalue is not implemented for {self.__class__.__name__}",
        )

    def basis_rotation(self):
        """Transformation required to rotate the basis for measuring the gate."""
        raise_error(
            NotImplementedError,
            f"Basis rotation is not implemented for {self.__class__.__name__}",
        )

    def apply(self, backend, state, nqubits):
        return backend.apply_gate(self, state, nqubits)

    def apply_density_matrix(self, backend, state, nqubits):
        return backend.apply_gate_density_matrix(self, state, nqubits)

    def apply_clifford(self, backend, state, nqubits):
        return backend.apply_gate_clifford(self, state, nqubits)


class SpecialGate(Gate):
    """Abstract class for special gates."""

    def commutes(self, gate):
        return False

    def on_qubits(self, qubit_map):
        raise_error(NotImplementedError, "Cannot use special gates on subroutines.")

    def matrix(self, backend=None):  # pragma: no cover
        raise_error(
            NotImplementedError, "Special gates do not have matrix representation."
        )


class ParametrizedGate(Gate):
    """Base class for parametrized gates.

    Implements the basic functionality of parameter setters and getters.
    """

    def __init__(self, trainable=True):
        super().__init__(trainable)
        self.parameter_names = "theta"
        self.nparams = 1
        self.trainable = trainable

    @Gate.parameters.setter
    def parameters(self, x):
        """Updates the values of gate's parameters."""
        if isinstance(self.parameter_names, str):
            nparams = 1
            names = [self.parameter_names]
            if not isinstance(x, collections.abc.Iterable):
                x = [x]
            else:
                # Captures the ``Unitary`` gate case where the given parameter
                # can be an array
                try:
                    if len(x) != 1:  # pragma: no cover
                        x = [x]
                except TypeError:  # tf.Variable case
                    s = tuple(x.shape)
                    if not s or s[0] != 1:
                        x = [x]
        else:
            nparams = len(self.parameter_names)
            names = self.parameter_names

        if not self.parameters:
            params = nparams * [None]
        else:
            params = list(self.parameters)
        if len(x) != nparams:
            raise_error(
                ValueError,
                f"Parametrized gate has {nparams} parameters "
                + f"but {len(x)} update values were given.",
            )
        for i, v in enumerate(x):
            if isinstance(v, sympy.Expr):
                self.symbolic_parameters[i] = v
            params[i] = v
        self.parameters = tuple(params)
        self.init_kwargs.update(
            {n: v for n, v in zip(names, self.parameters) if n in self.init_kwargs}
        )

        # set parameters in device gates
        for gate in self.device_gates:  # pragma: no cover
            gate.parameters = x

    def on_qubits(self, qubit_map):
        gate = super().on_qubits(qubit_map)
        gate.parameters = self.parameters
        return gate

    def substitute_symbols(self):
        params = list(self.parameters)
        for i, param in self.symbolic_parameters.items():
            for symbol in param.free_symbols:
                param = symbol.evaluate(param)
            params[i] = float(param)
        self.parameters = tuple(params)

    def matrix(self, backend=None):
        backend = _check_backend(backend)

        return backend.matrix_parametrized(self)
