from typing import TYPE_CHECKING

from qibo.backends import _check_backend
from qibo.config import raise_error
from qibo.gates.abstract import SpecialGate
from qibo.gates.measurements import M

if TYPE_CHECKING:
    from qibo.callbacks import Callback


class Barrier(SpecialGate):
    """Directive that separates the operations before and after it on the given qubits.

    It does not change the quantum state (its action is the identity), so it
    never affects simulation results. Its only purpose is to mark a boundary
    that circuit-manipulation tools must respect:

    * :meth:`qibo.models.circuit.Circuit.fuse` does not fuse gates across it;
    * :meth:`qibo.gates.abstract.Gate.commutes` always returns ``False``, so
      commutation-based rewriting never moves gates through it;
    * it is drawn by :meth:`qibo.models.circuit.Circuit.draw` and exported to
      OpenQASM (Open Quantum Assembly Language) as ``barrier q[i], q[j];``.

    .. note::
        Like every :class:`qibo.gates.abstract.SpecialGate`, a barrier is
        treated as acting on *all* qubits of the circuit by
        :meth:`qibo.models.circuit.Circuit.fuse`, even when it lists only a
        subset of them. This is conservative: it can only prevent fusions,
        never allow a wrong one.

    Args:
        q (int): Indices of the qubits the barrier acts on.

    Example:

        .. testcode::

            from qibo import Circuit, gates
            from barrier import Barrier

            circuit = Circuit(3)
            circuit.add(gates.H(0))
            circuit.add(gates.H(1))
            circuit.add(Barrier(0, 1))
            circuit.add(gates.CNOT(0, 1))
            circuit.add(Barrier(*range(circuit.nqubits)))
            circuit.draw()

        .. testoutput::

            0: ─H─░─o─░─
            1: ─H─░─X─░─
            2: ───────░─
    """

    def __init__(self, *q: int):
        super().__init__()
        if len(q) == 0:
            raise_error(
                ValueError,
                "Barrier requires at least one qubit. To act on every qubit of a "
                "circuit use ``Barrier(*range(circuit.nqubits))``.",
            )
        self.name = "barrier"
        self.draw_label = "░"
        self.init_args = list(q)
        self.target_qubits = q

    @property
    def clifford(self) -> bool:
        """Barriers are the identity operation, hence trivially Clifford."""
        return True

    @property
    def qasm_label(self) -> str:
        """OpenQASM (Open Quantum Assembly Language) keyword for the barrier."""
        return "barrier"

    def apply(self, backend, state, nqubits):
        """Return ``state`` unchanged, since a barrier does not act on the state."""
        return state

    def apply_clifford(self, backend, state, nqubits):
        """Return ``state`` unchanged, since a barrier does not act on the state."""
        return state

    def on_qubits(self, qubit_map: dict):
        """Creates the same barrier acting on different qubits.

        Args:
            qubit_map (dict): Dictionary mapping original qubit indices to new ones.
                Every qubit of the barrier must appear as a key.

        Returns:
            :class:`barrier.Barrier`: Barrier acting on the mapped qubits.
        """
        missing = [q for q in self.target_qubits if q not in qubit_map]
        if len(missing) > 0:
            raise_error(
                KeyError,
                f"Qubits {missing} of the barrier are missing from ``qubit_map``.",
            )
        return self.__class__(*(qubit_map[q] for q in self.target_qubits))


class CallbackGate(SpecialGate):
    """Calculates a :class:`qibo.callbacks.Callback` at a specific point in the circuit.

    This gate performs the callback calulation without affecting the state vector.

    Args:
        callback (:class:`qibo.callbacks.Callback`): Callback object to calculate.
    """

    def __init__(self, callback: "Callback"):  # type: ignore
        super().__init__()
        self.name = callback.__class__.__name__
        self.draw_label = "".join([c for c in self.name if c.isupper()])
        self.callback = callback
        self.init_args = [callback]

    def apply(self, backend, state, nqubits):
        self.callback.nqubits = nqubits
        self.callback.apply(backend, state)
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
        # combined qubits are more than ``max_qubits``
        return len(self.qubit_set | gate.qubit_set) <= max_qubits

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


def remove_barriers(circuit: "Circuit") -> "Circuit":
    """Return a deep copy of ``circuit`` with every barrier removed.

    Use it before :mod:`qibo.transpiler`: its placer, router and unroller do not
    handle barriers, and either raise an error or treat a two-qubit barrier as a
    real interaction, inserting unnecessary ``SWAP`` gates to make its qubits
    adjacent.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): Circuit that may contain
            barriers. It is not modified.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Copy of ``circuit`` without barriers.
    """
    new_circuit = circuit.copy(deep=True)
    kept_gates = [gate for gate in new_circuit.queue if gate.name != "barrier"]
    new_circuit.queue[:] = kept_gates
    return new_circuit
