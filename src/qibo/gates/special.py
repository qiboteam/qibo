from qibo.config import raise_error
from qibo.gates.abstract import Gate, ParametrizedGate, SpecialGate


class Flatten(SpecialGate):
    """Passes an arbitrary state vector in the circuit.

    Args:
        coefficients (list): list of the target state vector components.
            This can also be a tensor supported by the backend.
    """

    def __init__(self, coefficients):
        super(Flatten, self).__init__()
        self.name = "Flatten"
        self.coefficients = coefficients
        self.init_args = [coefficients]


class CallbackGate(SpecialGate):
    """Calculates a :class:`qibo.core.callbacks.Callback` at a specific point in the circuit.

    This gate performs the callback calulation without affecting the state vector.

    Args:
        callback (:class:`qibo.core.callbacks.Callback`): Callback object to calculate.
    """

    def __init__(self, callback: "Callback"):
        super(CallbackGate, self).__init__()
        self.name = callback.__class__.__name__
        self.callback = callback
        self.init_args = [callback]

    @Gate.nqubits.setter
    def nqubits(self, n: int):
        Gate.nqubits.fset(self, n) # pylint: disable=no-member
        self.callback.nqubits = n


class VariationalLayer(SpecialGate, ParametrizedGate):
    """Layer of one-qubit parametrized gates followed by two-qubit entangling gates.

    Performance is optimized by fusing the variational one-qubit gates with the
    two-qubit entangling gates that follow them and applying a single layer of
    two-qubit gates as 4x4 matrices.

    Args:
        qubits (list): List of one-qubit gate target qubit IDs.
        pairs (list): List of pairs of qubit IDs on which the two qubit gate act.
        one_qubit_gate: Type of one qubit gate to use as the variational gate.
        two_qubit_gate: Type of two qubit gate to use as entangling gate.
        params (list): Variational parameters of one-qubit gates as a list that
            has the same length as ``qubits``. These gates act before the layer
            of entangling gates.
        params2 (list): Variational parameters of one-qubit gates as a list that
            has the same length as ``qubits``. These gates act after the layer
            of entangling gates.
        trainable (bool): whether gate parameters can be updated using
            :meth:`qibo.abstractions.circuit.AbstractCircuit.set_parameters`
            (default is ``True``).

    Example:
        .. testcode::

            import numpy as np
            from qibo.models import Circuit
            from qibo import gates
            # generate an array of variational parameters for 8 qubits
            theta = 2 * np.pi * np.random.random(8)
            # define qubit pairs that two qubit gates will act
            pairs = [(i, i + 1) for i in range(0, 7, 2)]
            # define a circuit of 8 qubits and add the variational layer
            c = Circuit(8)
            c.add(gates.VariationalLayer(range(8), pairs, gates.RY, gates.CZ, theta))
            # this will create an optimized version of the following circuit
            c2 = Circuit(8)
            c.add((gates.RY(i, th) for i, th in enumerate(theta)))
            c.add((gates.CZ(i, i + 1) for i in range(7)))
    """

    def __init__(self, qubits, pairs, one_qubit_gate, two_qubit_gate,
                 params, params2=None, trainable=True):
        ParametrizedGate.__init__(self, trainable)
        self.init_args = [qubits, pairs, one_qubit_gate, two_qubit_gate]
        self.init_kwargs = {"params": params, "params2": params2,
                            "trainable": trainable}
        self.name = "VariationalLayer"

        self.unitaries = []
        self.additional_unitary = None

        self.target_qubits = tuple(qubits)
        self.parameter_names = [f"theta{i}" for i, _ in enumerate(params)]
        parameter_values = list(params)
        self.params = self._create_params_dict(params)
        self.params2 = {}
        if params2 is not None:
            self.params2 = self._create_params_dict(params2)
            n = len(self.parameter_names)
            self.parameter_names.extend([f"theta{i + n}" for i, _ in enumerate(params2)])
            parameter_values.extend(params2)
        self.parameters = parameter_values
        self.nparams = len(parameter_values)

        self.pairs = pairs
        targets = set(self.target_qubits)
        two_qubit_targets = set(q for p in pairs for q in p)
        additional_targets = targets - two_qubit_targets
        if not additional_targets:
            self.additional_target = None
        elif len(additional_targets) == 1:
            self.additional_target = additional_targets.pop()
        else:
            raise_error(ValueError, "Variational layer can have at most one "
                                    "additional target for one qubit gates but "
                                    " has {}.".format(additional_targets))

        self.one_qubit_gate = one_qubit_gate
        self.two_qubit_gate = two_qubit_gate

    def _create_params_dict(self, params):
        if len(self.target_qubits) != len(params):
            raise_error(ValueError, "VariationalLayer has {} target qubits but "
                                    "{} parameters were given."
                                    "".format(len(self.target_qubits), len(params)))
        return {q: p for q, p in zip(self.target_qubits, params)}


class FusedGate(SpecialGate):
    """Collection of gates that will be fused and applied as single gate during simulation.

    This gate is constructed automatically by :meth:`qibo.core.circuit.Circuit.fuse`
    and should not be used by user.
    """

    def __init__(self, *q):
        super().__init__()
        self.name = "fused"
        self.target_qubits = tuple(sorted(q))
        self.init_args = list(q)
        self.qubit_set = set(q)
        self.gates = []
        self.marked = False
        self.fused = False

        self.left_neighbors = {}
        self.right_neighbors = {}

    @classmethod
    def from_gate(cls, gate):
        fgate = cls(*gate.qubits)
        fgate.append(gate)
        if isinstance(gate, SpecialGate):
            # special gates do not participate in fusion
            fgate.marked = True
        return fgate

    def prepend(self, gate):
        self.qubit_set = self.qubit_set | set(gate.qubits)
        self.init_args = sorted(self.qubit_set)
        self.target_qubits = tuple(self.init_args)
        if isinstance(gate, self.__class__):
            self.gates = gate.gates + self.gates
        else:
            self.gates = [gate] + self.gates

    def append(self, gate):
        self.qubit_set = self.qubit_set | set(gate.qubits)
        self.init_args = sorted(self.qubit_set)
        self.target_qubits = tuple(self.init_args)
        if isinstance(gate, self.__class__):
            self.gates.extend(gate.gates)
        else:
            self.gates.append(gate)

    def __iter__(self):
        return iter(self.gates)

    def _dagger(self):
        dagger = self.__class__(*self.init_args)
        for gate in self.gates[::-1]:
            dagger.append(gate.dagger())
        return dagger

    def can_fuse(self, gate, max_qubits):
        """Check if two gates can be fused."""
        if gate is None:
            return False
        if self.marked or gate.marked:
            # gates are already fused
            return False
        if len(self.qubit_set | gate.qubit_set) > max_qubits:
            # combined qubits are more than ``max_qubits``
            return False
        return True

    def fuse(self, gate):
        """Fuses two gates."""
        left_gates = set(self.right_neighbors.values()) - {gate}
        right_gates = set(gate.left_neighbors.values()) - {self}
        if len(left_gates) > 0 and len(right_gates) > 0:
            # abort if there are blocking gates between the two gates
            # not in the shared qubits
            return

        qubits = self.qubit_set & gate.qubit_set
        # the gate with most neighbors different than the two gates to
        # fuse will be the parent
        if len(left_gates) > len(right_gates):
            parent, child = self, gate
            between_gates = set(parent.right_neighbors.get(q) for q in qubits)
            if between_gates == {child}:
                child.marked = True
                parent.append(child)
                for q in qubits:
                    neighbor = child.right_neighbors.get(q)
                    if neighbor is not None:
                        parent.right_neighbors[q] = neighbor
                        neighbor.left_neighbors[q] = parent
                    else:
                        parent.right_neighbors.pop(q)
        else:
            parent, child = gate, self
            between_gates = set(parent.left_neighbors.get(q) for q in qubits)
            if between_gates == {child}:
                child.marked = True
                parent.prepend(child)
                for q in qubits:
                    neighbor = child.left_neighbors.get(q)
                    if neighbor is not None:
                        parent.left_neighbors[q] = neighbor
                        neighbor.right_neighbors[q] = parent

        if child.marked:
            # update the neighbors graph
            for q in child.qubit_set - qubits:
                neighbor = child.right_neighbors.get(q)
                if neighbor is not None:
                    parent.right_neighbors[q] = neighbor
                    neighbor.left_neighbors[q] = parent
                neighbor = child.left_neighbors.get(q)
                if neighbor is not None:
                    parent.left_neighbors[q] = neighbor
                    neighbor.right_neighbors[q] = parent
