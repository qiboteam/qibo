import networkx as nx

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import log, raise_error
from qibo.gates.abstract import SpecialGate
from qibo.models import Circuit
from qibo.transpiler.abstract import Optimizer


class Preprocessing(Optimizer):
    """Pad the circuit with unused qubits to match the number of physical qubits.

    Args:
        connectivity (:class:`networkx.Graph`): Hardware connectivity as a graph.
    """

    def __init__(self, connectivity: nx.Graph | None = None):
        self.connectivity = connectivity

    def __call__(self, circuit: Circuit) -> Circuit:
        if not all(qubit in self.connectivity.nodes for qubit in circuit.wire_names):
            default_wire_names = list(self.connectivity.nodes)[: circuit.nqubits]
            log.warning(
                f"Some wire_names in the circuit are not in the connectivity graph. Using wire name {default_wire_names}."
            )
            circuit.wire_names = default_wire_names

        physical_qubits = self.connectivity.number_of_nodes()
        logical_qubits = circuit.nqubits
        if logical_qubits > physical_qubits:
            raise_error(
                ValueError,
                f"The number of qubits in the circuit ({logical_qubits}) "
                + f"can't be greater than the number of physical qubits ({physical_qubits}).",
            )
        if logical_qubits == physical_qubits:
            return circuit

        new_wire_names = circuit.wire_names + list(
            self.connectivity.nodes - circuit.wire_names
        )

        new_circuit = Circuit(nqubits=physical_qubits, wire_names=new_wire_names)
        for gate in circuit.queue:
            new_circuit.add(gate)

        return new_circuit


class Rearrange(Optimizer):
    """Rearranges gates using ``qibo``'s fusion algorithm.
    May reduce number of :class:`qibo.gates.SWAP` when fixing for connectivity
    but this has not been tested.

    Args:
        max_qubits (int, optional): Maximum number of qubits to fuse.
            Defaults to :math:`1`.
    """

    def __init__(self, max_qubits: int = 1):
        self.max_qubits = max_qubits

    def __call__(self, circuit: Circuit, backend: Backend | None = None) -> Circuit:
        backend = _check_backend(backend)
        fused_circuit = circuit.fuse(max_qubits=self.max_qubits)
        new = circuit.__class__(nqubits=circuit.nqubits, wire_names=circuit.wire_names)
        for fgate in fused_circuit.queue:
            if isinstance(fgate, gates.FusedGate):
                new.add(gates.Unitary(fgate.matrix(backend), *fgate.qubits))
            else:
                new.add(fgate)

        return new


class InverseCancellation(Optimizer):
    """Cancels pairs of consecutive gates whose product is the identity.

    Two gates cancel when no other gate acts on their qubits in between, both act on
    the same qubits in the same order with the same control qubits, and the second gate
    undoes the first one, i.e. the product of their matrices is the identity matrix up
    to ``atol``. Cancellation is applied repeatedly, so nested pairs are removed
    completely: a Pauli-X gate, a Pauli-Y gate, a second Pauli-Y gate and a second
    Pauli-X gate applied in this order on one qubit all disappear.

    The product must equal the identity itself, not merely the identity times a global
    phase, because for gates built with ``controlled_by`` a phase on the target matrix
    is a relative phase on the control qubits. Measurements, alignments, barriers, noise
    channels and fused gates are never cancelled and stop the cancellation across
    them on their qubits. Gates are compared through their matrices at their current parameter
    values, so a cancelled pair of parametrized gates is not tracked by the returned
    circuit anymore.

    Args:
        atol (float, optional): Tolerance on the Frobenius norm (square root of the sum
            of the squared absolute values of all entries) of the difference between
            the product of the two gate matrices and the identity matrix.
            Defaults to :math:`10^{-12}`.

    Example:

        The pair of controlled-NOT (CNOT) gates cancels first, which makes the two
        Hadamard gates on the first qubit consecutive so they cancel as well. The two
        rotations around the X axis on the second qubit cancel because their angles
        add up to zero. The Hadamard gates on the second qubit stay, because the CNOT
        gate between them acts on that qubit.

        .. testcode::

            from qibo import Circuit, gates
            from qibo.transpiler import InverseCancellation

            circuit = Circuit(2)
            circuit.add(gates.H(0))
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.H(0))
            circuit.add(gates.RX(1, 0.3))
            circuit.add(gates.RX(1, -0.3))
            circuit.add(gates.X(0))
            circuit.add(gates.H(1))
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.H(1))

            circuit.draw()
            print()
            InverseCancellation()(circuit).draw()

        .. testoutput::

            0: ─H─o─o─H──X────o───
            1: ───X─X─RX─RX─H─X─H─

            0: ─X─o───
            1: ─H─X─H─
    """

    def __init__(self, atol: float = 1e-12):
        if atol < 0:
            raise_error(ValueError, f"``atol`` must be non-negative, but got {atol}.")
        self.atol = atol

    def __call__(self, circuit: Circuit, backend: Backend = None) -> Circuit:
        """Remove pairs of consecutive gates that multiply to the identity.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be optimized.
            backend (:class:`qibo.backends.abstract.Backend`, optional): Backend used to
                build the gate matrices. If ``None``, defaults to the global backend.
                Defaults to ``None``.

        Returns:
            :class:`qibo.models.circuit.Circuit`: Circuit without the cancelled pairs.
        """
        backend = _check_backend(backend)

        uncancellable = (gates.M, gates.Align, gates.Channel, SpecialGate)
        # ``kept`` holds the surviving gates in order (cancelled ones are set to
        # ``None``) and ``stacks`` holds, for each qubit, the positions in ``kept``
        # of the surviving gates acting on it. The last position is the latest gate.
        kept = []
        stacks = {qubit: [] for qubit in range(circuit.nqubits)}
        for gate in circuit.queue:
            latest = {
                stacks[qubit][-1] if stacks[qubit] else None for qubit in gate.qubits
            }
            index = latest.pop() if len(latest) == 1 else None
            partner = None if index is None else kept[index]

            cancels = (
                partner is not None
                and not isinstance(gate, uncancellable)
                and not isinstance(partner, uncancellable)
                and partner.qubits == gate.qubits
                and partner.control_qubits == gate.control_qubits
            )
            if cancels:
                first = partner.matrix(backend)
                second = gate.matrix(backend)
                cancels = first.shape == second.shape and (
                    backend.matrix_norm(
                        second @ first - backend.matrices.I(first.shape[0]),
                        order="fro",
                    )
                    <= self.atol
                )

            if cancels:
                kept[index] = None
                for qubit in gate.qubits:
                    stacks[qubit].pop()
            else:
                kept.append(gate)
                for qubit in gate.qubits:
                    stacks[qubit].append(len(kept) - 1)

        new = Circuit(**circuit.init_kwargs)
        new.add([gate for gate in kept if gate is not None])

        return new


class TGateRules(Optimizer):
    """Replaces runs of consecutive :class:`qibo.gates.gates.T` gates by shorter equivalents.

    The :math:`T` gate is the diagonal matrix :math:`\\mathrm{diag}(1, e^{i \\pi / 4})`,
    so :math:`k` consecutive :math:`T` gates on the same qubit give
    :math:`\\mathrm{diag}(1, e^{i k \\pi / 4})`. This only depends on :math:`k \bmod 8`,
    hence eight rules are enough to simplify any number of consecutive :math:`T`
    gates, using the Clifford gates :math:`S = T^2` and :math:`Z = T^4`:

    .. list-table::
        :header-rows: 1

        * - :math:`k \bmod 8`
          - Replacement
        * - 0
          - identity (no gate)
        * - 1
          - :math:`T`
        * - 2
          - :math:`S`
        * - 3
          - :math:`S T`
        * - 4
          - :math:`Z`
        * - 5
          - :math:`Z T`
        * - 6
          - :math:`S^\\dagger`
        * - 7
          - :math:`T^\\dagger`

    All rules are exact, i.e. they hold without any global phase, so they are also
    correct inside larger circuits. Only :math:`T` gates without control qubits are
    replaced. A run of :math:`T` gates on a qubit ends as soon as any other gate,
    including a measurement or a barrier, acts on that qubit; gates acting on other
    qubits do not end it.

    Example:

        Five :math:`T` gates on the first qubit become :math:`Z T`. On the second
        qubit, the controlled-NOT (CNOT) gate ends the run, so the first :math:`T`
        gate stays and the two :math:`T` gates after the CNOT become an :math:`S` gate.

        .. testcode::

            from qibo import Circuit, gates
            from qibo.transpiler import TGateRules

            circuit = Circuit(2)
            circuit.add(gates.T(0) for _ in range(5))
            circuit.add(gates.T(1))
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.T(1))
            circuit.add(gates.T(1))

            circuit.draw()
            print()
            TGateRules()(circuit).draw()

        .. testoutput::

            0: ─T─T─T─T─T─o─────
            1: ─T─────────X─T─T─

            0: ─Z─T─o───
            1: ─T───X─S─
    """

    # Gates replacing ``T ** k`` for ``k = 0, ..., 7``, in the order they are applied.
    _RULES = (
        (),
        (gates.T,),
        (gates.S,),
        (gates.S, gates.T),
        (gates.Z,),
        (gates.Z, gates.T),
        (gates.SDG,),
        (gates.TDG,),
    )

    def __call__(self, circuit: Circuit) -> Circuit:
        """Replace runs of consecutive :math:`T` gates using the rules for powers of :math:`T`.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be optimized.

        Returns:
            :class:`qibo.models.circuit.Circuit`: Circuit with the runs of :math:`T`
            gates replaced.
        """
        new = Circuit(**circuit.init_kwargs)
        # ``powers`` maps each qubit to the number of consecutive T gates pending on it.
        # The final ``None`` flushes the runs still pending at the end of the circuit.
        powers = {}
        for gate in [*circuit.queue, None]:
            if isinstance(gate, gates.T) and not gate.control_qubits:
                powers[gate.qubits[0]] = powers.get(gate.qubits[0], 0) + 1
                continue

            for qubit in list(powers) if gate is None else gate.qubits:
                new.add(rule(qubit) for rule in self._RULES[powers.pop(qubit, 0) % 8])

            if gate is not None:
                new.add(gate)

        return new
