from collections.abc import Iterable
from math import prod
from typing import TYPE_CHECKING

from qibo.backends import _check_backend
from qibo.config import raise_error
from qibo.gates.abstract import Gate, ParametrizedGate, SpecialGate
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
        q (int, ...): Indices of the qubits the barrier acts on.
    Example:

        .. testcode::

            from qibo import Circuit, gates
            from qibo.gates.special import Barrier

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
            :class:`qibo.gates.Barrier`: Barrier acting on the mapped qubits.
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
        self._initialize(q)

    def _initialize(self, qubits):
        """Initialize the fusion data shared by composite gate classes."""
        self.name = "Fused Gate"
        self.draw_label = "[]"
        self.target_qubits = tuple(sorted(qubits))
        self.init_args = list(qubits)
        self.qubit_set = set(qubits)
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
        if isinstance(gate, self.__class__) and not isinstance(
            gate, ParametrizedFusedGate
        ):
            self.gates = gate.gates + self.gates
        else:
            self.gates = [gate] + self.gates

    def append(self, gate):
        self.qubit_set = self.qubit_set | set(gate.qubits)
        self.init_args = sorted(self.qubit_set)
        self.target_qubits = tuple(self.init_args)
        if isinstance(gate, self.__class__) and not isinstance(
            gate, ParametrizedFusedGate
        ):
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


class ParametrizedFusedGate(FusedGate, ParametrizedGate):
    """Composite gate whose trainable child parameters remain accessible.

    Gates are applied in the order in which they are appended. The composite's
    :attr:`parameters` concatenates the parameter tuples of its trainable
    children, including nested composites. Array-valued parameters remain arrays;
    :attr:`nparams` counts their individual entries. Non-trainable children still
    act on the state but are excluded from the composite's parameter interface.
    Setting ``trainable=False`` on the composite excludes the whole block from
    :meth:`qibo.models.Circuit.set_parameters`.

    Children are kept by reference, so updating a child's parameters also updates
    the composite. Complete the structure and choose each child's ``trainable``
    flag before adding the composite to a circuit or another parametrized
    composite. Afterwards, ``append`` and ``prepend`` are disabled to preserve
    the containing circuit's or composite's qubit and parameter bookkeeping.
    Do not mutate ``gates`` directly or change child ``trainable`` flags after
    adding the block to a circuit or another parametrized composite.

    Args:
        *q (int): Initial target qubits. Additional targets are inferred from
            appended or prepended gates.
        trainable (bool, optional): Whether the composite participates in circuit
            parameter updates. Defaults to ``True``.

    Example:
        .. testcode::

            from qibo import Circuit, gates

            block = gates.ParametrizedFusedGate(0, 1)
            block.append(gates.RX(0, theta=0.1))
            block.append(gates.CNOT(0, 1))
            block.append(gates.RY(1, theta=0.2))
            circuit = Circuit(2)
            circuit.add(block)
            circuit.set_parameters([0.3, 0.4])
            assert block.parameters == (0.3, 0.4)
    """

    def __init__(self, *q: int, trainable: bool = True):
        # ParametrizedGate initializes a fixed nparams, whereas this class
        # computes its parameter metadata from the current child gates.
        Gate.__init__(self)
        self._initialize(q)
        self.name = "Parametrized Fused Gate"
        self.trainable = trainable
        self.init_kwargs = {"trainable": trainable}
        self.unitary = True
        self._structure_frozen = False

    def _parametrized_gates(self, gates=None):
        """Collect active leaves, checking cycles and shared free parameters."""
        from qibo.gates.channels import Channel

        parametrized, seen = [], {}

        def collect(gate, ancestors, active=True):
            if isinstance(gate, FusedGate):
                if id(gate) in ancestors:
                    raise_error(ValueError, "Composite gates cannot contain cycles.")
                if isinstance(gate, ParametrizedFusedGate):
                    active = active and gate.trainable
                for child in gate.gates:
                    collect(child, ancestors | {id(gate)}, active)
            elif not isinstance(gate, Gate) or isinstance(
                gate, (SpecialGate, M, Channel)
            ):
                raise_error(
                    TypeError,
                    "ParametrizedFusedGate only supports matrix gates; "
                    "measurements, channels and special directives cannot be fused.",
                )
            elif isinstance(gate, ParametrizedGate):
                active = active and gate.trainable
                if id(gate) in seen and (active or seen[id(gate)]):
                    raise_error(
                        ValueError,
                        "A trainable gate cannot occur more than once in a "
                        "ParametrizedFusedGate. Use separate gate instances.",
                    )
                if active:
                    parametrized.append(gate)
                seen[id(gate)] = active

        for gate in self.gates if gates is None else gates:
            collect(gate, {id(self)})
        return parametrized

    @property
    def nparams(self) -> int:
        """Number of scalar parameters exposed by this composite."""
        return sum(gate.nparams for gate in self._parametrized_gates())

    @property
    def parameter_names(self) -> list[str]:
        """Child parameter names, prefixed by their position in the composite."""
        names = []
        for index, gate in enumerate(self._parametrized_gates()):
            child_names = gate.parameter_names
            if isinstance(child_names, str):
                child_names = [child_names]
            names.extend(f"{index}_{name}" for name in child_names)
        return names

    @property
    def parameters(self) -> tuple:
        """Current trainable child parameters in gate application order."""
        return tuple(
            parameter
            for gate in self._parametrized_gates()
            for parameter in gate.parameters
        )

    @parameters.setter
    def parameters(self, values):
        """Update trainable children using structured or flat parameter values."""
        if values is None or isinstance(values, (str, bytes, dict, set)):
            raise_error(TypeError, "Composite parameters must be an ordered sequence.")
        gates = self._parametrized_gates()
        current = tuple(parameter for gate in gates for parameter in gate.parameters)
        shape = getattr(values, "shape", None)
        if (
            not isinstance(values, Iterable)
            or shape == ()
            or (
                len(current) == 1
                and shape is not None
                and shape == getattr(current[0], "shape", None)
            )
        ):
            values = (values,)
        else:
            values = tuple(values)

        if len(values) == len(current):
            parameters = values
        elif len(values) == self.nparams:
            from qibo.gates.gates import _check_engine

            parameters, offset = [], 0
            for parameter in current:
                shape = tuple(getattr(parameter, "shape", ()))
                size = prod(shape)
                if shape:
                    chunk = values[offset : offset + size]
                    engine = _check_engine(chunk)
                    # Preserve tensor values and their differentiation graph.
                    if engine.__name__ == "torch":  # pragma: no cover
                        value = engine.stack(chunk).reshape(shape)
                    else:
                        value = engine.reshape(chunk, shape)
                    parameters.append(value)
                else:
                    parameters.append(values[offset])
                offset += size
        else:
            raise_error(
                ValueError,
                f"ParametrizedFusedGate has {len(current)} parameter values "
                f"({self.nparams} scalar entries), but {len(values)} were given.",
            )

        # Check array shapes before modifying any children.
        for old, new in zip(current, parameters):
            old_shape = tuple(getattr(old, "shape", ()))
            new_shape = tuple(getattr(new, "shape", ()))
            invalid_scalar = not old_shape and (
                new is None
                or isinstance(new, (str, bytes, dict, set))
                or (isinstance(new, Iterable) and getattr(new, "shape", None) != ())
            )
            if old_shape != new_shape or invalid_scalar:
                raise_error(
                    ValueError,
                    f"Composite parameter shape {new_shape} does not match "
                    f"the child parameter shape {old_shape}.",
                )
        offset = 0
        for gate in gates:
            size = len(gate.parameters)
            gate.parameters = tuple(parameters[offset : offset + size])
            offset += size
        for gate in self.device_gates:  # pragma: no cover
            gate.parameters = parameters

    def _flat_parameters(self) -> tuple:
        parameters = []
        for value in self.parameters:
            if tuple(getattr(value, "shape", ())):
                parameters.extend(value.reshape(-1))
            else:
                parameters.append(value)
        return tuple(parameters)

    def _insert(self, gate, prepend=False):
        if self._structure_frozen:
            raise_error(
                RuntimeError,
                "Cannot change a ParametrizedFusedGate's structure after adding "
                "it to a circuit or another parametrized composite.",
            )

        def flatten(child, ancestors):
            if isinstance(child, FusedGate) and not isinstance(
                child, ParametrizedFusedGate
            ):
                if id(child) in ancestors:
                    raise_error(ValueError, "Composite gates cannot contain cycles.")
                if child.control_qubits:
                    raise_error(
                        NotImplementedError,
                        "Use ParametrizedFusedGate for a controlled composite child.",
                    )
                return [
                    leaf
                    for nested in child.gates
                    for leaf in flatten(nested, ancestors | {id(child)})
                ]
            return [child]

        children = flatten(gate, {id(self)})
        candidates = children + self.gates if prepend else self.gates + children
        self._parametrized_gates(candidates)
        qubits = self.qubit_set | {q for child in children for q in child.qubits}
        if qubits & set(self.control_qubits):
            raise_error(
                ValueError,
                "Composite target qubits cannot overlap with its control qubits.",
            )
        self.target_qubits = tuple(sorted(qubits))
        self.qubit_set = qubits
        self.init_args = list(self.target_qubits)
        self.gates = candidates
        self.unitary = all(child.unitary for child in self.gates)
        for child in children:
            if isinstance(child, ParametrizedFusedGate):
                child._freeze()

    def append(self, gate):
        """Append a matrix gate or a composite, preserving parameter references."""
        self._insert(gate)

    def prepend(self, gate):
        """Prepend a matrix gate or a composite, preserving parameter references."""
        self._insert(gate, prepend=True)

    def _freeze(self):
        """Prevent structural mutations after a circuit registers the block."""
        self._parametrized_gates()
        for gate in self.gates:
            if isinstance(gate, ParametrizedFusedGate):
                gate._freeze()
        self._structure_frozen = True

    def substitute_symbols(self):
        """Resolve measurement-dependent parameters, including frozen children."""
        for gate in self.gates:
            if isinstance(gate, ParametrizedFusedGate) or gate.symbolic_parameters:
                gate.substitute_symbols()

    def matrix(self, backend=None):
        self.substitute_symbols()
        return super().matrix(backend)

    def apply(self, backend, state, nqubits):
        # Backends with specialized kernels dispatch on concrete gate classes.
        # Reuse their existing fusion path while retaining live child parameters.
        self.substitute_symbols()
        gate = FusedGate(*self.target_qubits)
        gate.gates = self.gates
        gate.controlled_by(*self.control_qubits)
        return backend.apply_gate(gate, state, nqubits)

    def _dagger(self):
        dagger = self.__class__(*self.target_qubits, trainable=self.trainable)
        for gate in reversed(self.gates):
            child = gate.dagger()
            if isinstance(gate, ParametrizedGate):
                child.trainable = gate.trainable
                child.init_kwargs["trainable"] = gate.trainable
            dagger.append(child)
        return dagger

    def on_qubits(self, qubit_map: dict):
        """Create the composite on different qubits, preserving trainable flags."""
        missing = [qubit for qubit in self.qubits if qubit not in qubit_map]
        if missing:
            raise_error(KeyError, f"Qubits {missing} are missing from qubit_map.")
        mapped = self.__class__(
            *(qubit_map[qubit] for qubit in self.target_qubits),
            trainable=self.trainable,
        )
        for gate in self.gates:
            child = gate.on_qubits(qubit_map)
            if isinstance(gate, ParametrizedGate):
                child.trainable = gate.trainable
                child.init_kwargs["trainable"] = gate.trainable
            mapped.append(child)
        if self.is_controlled_by:
            mapped.controlled_by(*(qubit_map[q] for q in self.control_qubits))
        return mapped

    def decompose(self, *free, method: str = "standard", **kwargs):
        """Decompose the children of an uncontrolled composite in order."""
        if self.control_qubits:
            raise_error(
                NotImplementedError,
                "Decomposition of controlled ParametrizedFusedGate is not supported.",
            )
        return [
            child
            for gate in self.gates
            for child in gate.decompose(*free, method=method, **kwargs)
        ]

    def gradient(self, backend=None):
        raise_error(
            NotImplementedError,
            "Differentiate the individual children of ParametrizedFusedGate.",
        )

    @property
    def raw(self) -> dict:
        """Serialize the composite, retaining child parameters and trainability."""
        raw = Gate.raw.fget(self)
        raw["init_kwargs"] = {"trainable": self.trainable}
        raw["gates"] = []
        for gate in self.gates:
            child = dict(gate.raw)
            child["init_kwargs"] = dict(gate.init_kwargs)
            if isinstance(gate, ParametrizedGate):
                child["init_kwargs"]["trainable"] = gate.trainable
                child["parameters"] = gate.parameters
            raw["gates"].append(child)
        return raw

    @classmethod
    def _from_dict(cls, raw: dict):
        gate = cls(
            *raw["init_args"], trainable=raw["init_kwargs"].get("trainable", True)
        )
        for child_raw in raw["gates"]:
            child = Gate.from_dict(child_raw)
            if isinstance(child, ParametrizedGate) and "parameters" in child_raw:
                child.parameters = child_raw["parameters"]
            gate.append(child)
        return gate.controlled_by(*raw["_control_qubits"])


def remove_barriers(circuit):
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
    kept_gates = [gate for gate in new_circuit.queue if not isinstance(gate, Barrier)]
    new_circuit.queue[:] = kept_gates
    return new_circuit
