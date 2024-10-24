from qibo.backends import _check_backend
from qibo.gates.abstract import SpecialGate
from qibo.gates.measurements import M


class CallbackGate(SpecialGate):
    """Calculates a :class:`qibo.callbacks.Callback` at a specific point in the circuit.

    This gate performs the callback calulation without affecting the state vector.

    Args:
        callback (:class:`qibo.callbacks.Callback`): Callback object to calculate.
    """

    def __init__(self, callback: "Callback"):
        super().__init__()
        self.name = callback.__class__.__name__
        self.draw_label = "".join([c for c in self.name if c.isupper()])
        self.callback = callback
        self.init_args = [callback]

    def apply(self, backend, state, nqubits):
        self.callback.nqubits = nqubits
        self.callback.apply(backend, state)
        return state

    def apply_density_matrix(self, backend, state, nqubits):
        self.callback.nqubits = nqubits
        self.callback.apply_density_matrix(backend, state)
        return state


class FusedGate(SpecialGate):
    """Collection of gates that will be fused and applied as single gate during simulation.
    This gate is constructed automatically by :meth:`qibo.models.circuit.Circuit.fuse`
    and should not be used by user.
    """

    def __init__(self, *q):
        super().__init__()
        self.name = "Fused Gate"
        self.draw_label = "[]"
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
        if isinstance(gate, (M, SpecialGate)):
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

    def matrix(self, backend=None):
        """Returns matrix representation of special gate.

        Args:
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the execution. If ``None``, it uses the current backend. Defaults to ``None``.

        Returns:
            ndarray: Matrix representation of special gate.
        """
        backend = _check_backend(backend)

        return backend.matrix_fused(self)

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
            between_gates = {parent.right_neighbors.get(q) for q in qubits}
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
            between_gates = {parent.left_neighbors.get(q) for q in qubits}
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

    def apply_clifford(self, backend, state, nqubits):
        for gate in self.gates:
            state = gate.apply_clifford(backend, state, nqubits)
        return state
