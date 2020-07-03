from typing import List, Optional, Set, Tuple


class FusionGroup:
    """Group of one-qubit and two-qubit gates that act in two specific gates.

    These gates can be fused into a single two-qubit gate represented by a
    general 4x4 matrix.

    Attrs:
        qubit0 (int): Id of the first qubit that the ``FusionGroup`` act.
        qubit1 (int): Id of the first qubit that the ``FusionGroup`` act.
        two_qubit_gates: List of tuples (two-qubit gate, revert flag).
            If the revert flag is ``False`` the two-qubit gate is applied to
            (qubit0, qubit1) while if it is ``True`` it is applied to
            (qubit1, qubit0).
        gates0: List of lists of one-qubit gates to be applied to ``qubit0``.
            One qubit gates are split in groups according to when the two
            qubit gates are applied (see example).
            Has ``len(gates0) = len(two_qubit_gates) + 1``.
        gates1: Same as ``gates0`` but for ``qubit1``.

    Example:
        If ``gates0 = [[gates.H(0)], [gates.X(0)]]``,
        ``gates1 = [[gates.H(1)], [gates.X(1)]]`` and
        ``two_qubit_gates = [gates.CNOT(0, 1)]`` then the ``H`` gates are
        applied before the ``CNOT`` and the ``X`` gates after.
    """

    # ``FusionGroup`` cannot start with these gates because it is more
    # efficient to apply them on their own
    _efficient_gates = {"CNOT", "CZ", "SWAP", "CZPow"}

    def __init__(self):
        self.qubit0 = None
        self.qubit1 = None
        self.gates0 = [[]]
        self.gates1 = [[]]
        self.two_qubit_gates = []

        self.completed = False
        self.special_gate = None
        self._fused_gates = None

    @property
    def qubits(self) -> Set[int]:
        """Set of ids of the two qubits that the ``FusionGroup`` acts on."""
        if self.qubit0 is None:
            return {}
        if self.qubit1 is None:
            return {self.qubit0}
        return {self.qubit0, self.qubit1}

    @property
    def gates(self) -> Tuple["Gate"]:
        """Tuple with fused gates.

        These gates have equivalent action with all the original gates that
        were added in the ``FusionGroup``.
        """
        if self._fused_gates is None:
            self._fused_gates = self.calculate()
        return self._fused_gates

    def first_gate(self, i: int) -> Optional["Gate"]:
        """First one-qubit gate of the group."""
        if i < 0 or i > 1:
            raise ValueError(f"Invalid integer {i} given in FusionGroup.first_gate.")
        gates = self.gates0 if i == 0 else self.gates1
        for group in gates:
            if group:
                return group[0]
        return None

    def is_efficient(self, gate: "Gate") -> bool:
        """Checks if given two-qubit ``gate`` is efficient.

        Efficient gates are not fused if they are in the start or in the end.
        """
        return gate.__class__.__name__ in self._efficient_gates

    @classmethod
    def from_queue(cls, queue: List["Gate"]) -> List["FusionGroup"]:
        """Fuses a queue of gates by combining up to two-qubit gates.

        Args:
            queue (list): List of gates.

        Returns:
            List of ``FusionGroup`` objects that correspond to the fused gates.
        """
        group_queue = []
        remaining_queue = list(queue)
        while remaining_queue:
            gates = iter(remaining_queue)
            gate = next(gates)
            new_group = cls()
            new_group.add(gate)
            remaining_queue = []
            for gate in gates:
                if new_group.completed:
                    remaining_queue.append(gate)
                    break

                commutes = True
                for blocking_gate in remaining_queue:
                    commutes = commutes and gate.commutes(blocking_gate)
                    if not commutes:
                        break

                if commutes:
                    try:
                        new_group.add(gate)
                    except ValueError:
                        remaining_queue.append(gate)
                else:
                    remaining_queue.append(gate)

            new_group.completed = True
            remaining_queue.extend(gates)
            group_queue.append(new_group)

        return group_queue

    @property
    def module(self):
        """Module of the gates that the ``FusionGroup`` contains."""
        if self.two_qubit_gates:
            return self.two_qubit_gates[0].module
        if self.special_gate is not None:
            return self.special_gate.module
        for i in range(2):
            gate = self.first_gate(i)
            if gate is not None:
                return gate.module
        raise ValueError("Unable to find gate module.")

    def add(self, gate: "Gate"):
        """Adds a gate in the group.

        Raises:
            ValueError: If the gate cannot be added in the group (eg. because
                it acts on different qubits).
            RuntimeError: If the group is completed.
        """
        if self.completed:
            raise RuntimeError("Cannot add gates to completed FusionGroup.")

        if not gate.qubits:
            self._add_special_gate(gate)
        elif len(gate.qubits) == 1:
            self._add_one_qubit_gate(gate)
        elif len(gate.qubits) == 2:
            self._add_two_qubit_gate(gate)
        else:
            raise ValueError("Cannot add gate acting on {} qubits in fusion "
                             "group.".format(len(gate.qubits)))

    def calculate(self):
        """Calculates fused gate."""
        if not self.completed:
            raise RuntimeError("Cannot calculate fused gates for incomplete "
                               "FusionGroup.")
        if self.special_gate is None and self.qubit0 is None:
            raise RuntimeError("Cannot calculate fused gates for empty "
                               "FusionGroup.")
        return self._calculate()

    def _calculate(self): # pragma: no cover
        raise NotImplementedError

    def _add_special_gate(self, gate: "Gate"):
        """Adds ``CallbackGate`` or ``Flatten`` on ``FusionGroup``."""
        if self.qubits:
            raise ValueError("Cannot add special gate on fusion group with "
                             "qubits already set.")
        self.special_gate = gate
        self.completed = True

    def _add_one_qubit_gate(self, gate: "Gate"):
        """Adds one-qubit gate to ``FusionGroup``."""
        qubit = gate.qubits[0]
        if self.qubit0 is None or self.qubit0 == qubit:
            self.qubit0 = qubit
            self.gates0[-1].append(gate)
        elif self.qubit1 is None or self.qubit1 == qubit:
            self.qubit1 = qubit
            self.gates1[-1].append(gate)
        else:
            raise ValueError("Cannot add gate on qubit {} in fusion group of "
                             "qubits {} and {}."
                             "".format(qubit, self.qubit0, self.qubit1))

    def _add_two_qubit_gate(self, gate: "Gate"):
        """Adds two-qubit gate to ``FusionGroup``."""
        qubit0, qubit1 = gate.qubits
        if self.qubit0 is None:
            self.qubit0, self.qubit1 = qubit0, qubit1
            self.two_qubit_gates.append(gate)
            if self.is_efficient(gate):
                self.completed = True

        elif self.qubit1 is None:
            if self.is_efficient(gate):
                raise ValueError("It is not efficient to add {} in FusionGroup "
                                 "with only one qubit set.".format(gate))

            if self.qubit0 == qubit0:
                self.qubit1 = qubit1
                self.two_qubit_gates.append(gate)
            elif self.qubit0 == qubit1:
                self.qubit1 = qubit0
                self.two_qubit_gates.append(gate)
            else:
                raise ValueError("Cannot add gate on qubits {} and {} in "
                                 "fusion group of qubit {}."
                                 "".format(qubit0, qubit1, self.qubit0))

        else:
            if self.qubits != {qubit0, qubit1}:
                raise ValueError("Cannot add gate on qubits {} and {} in "
                                 "fusion group of qubits {} and {}."
                                 "".format(qubit0, qubit1, self.qubit0, self.qubit1))
            self.two_qubit_gates.append(gate)

        self.gates0.append([])
        self.gates1.append([])
