from qibo.base import gates
from typing import List, Set


class FusionGroup:

    def __init__(self):
        self.qubit0 = None
        self.qubit1 = None
        self.gates0 = [[]]
        self.gates1 = [[]]
        self.two_qubit_gates = [] # list of tuples (gate, revert flag)

        self.special_gate = None

    @property
    def qubits(self) -> Set[int]:
        if self.qubit0 is None:
            return {}
        if self.qubit1 is None:
            return {self.qubit0}
        return {self.qubit0, self.qubit1}

    def add(self, gate: gates.Gate):
        if self.special_gate is not None:
            raise ValueError("Cannot add gate on special fusion group.")
        if not gate.qubits:
            if self.qubits:
                raise ValueError("Cannot add special gate on fusion group with "
                                 "qubits already set.")
            self.special_gate = None
        elif len(gate.qubits) == 1:
            self._add_one_qubit_gate(gate)
        elif len(gate.qubits) == 2:
            self._add_two_qubit_gate(gate)
        else:
            raise ValueError("Cannot add gate acting on {} qubits in fusion "
                             "group.".format(len(gate.qubits)))

    def _add_one_qubit_gate(self, gate: gates.Gate):
        qubit = gate.qubits[0]
        if self.qubit0 is None or self.qubit0 == qubit:
            self.qubit0 = qubit
            self.gates0[-1].append(gate)
        elif self.qubit1 is None or self.qubit1 == qubit:
            self.qubit1 = qubit
            self.gates1[-1].append(gate)
        else: # pragma: no cover
            raise ValueError("Cannot add gate on qubit {} in fusion group of "
                             "qubits {} and {}."
                             "".format(qubit, self.qubit0, self.qubit1))

    def _add_two_qubit_gate(self, gate: gates.Gate):
        qubit0, qubit1 = gate.qubits
        if self.qubit0 is None:
            self.qubit0, self.qubit1 = qubit0, qubit1
            self.two_qubit_gates.append((gate, False))
        elif self.qubit1 is None:
            if self.qubit0 == qubit0:
                self.qubit1 = qubit1
                self.two_qubit_gates.append((gate, False))
            elif self.qubit0 == qubit1:
                self.qubit1 = qubit0
                self.two_qubit_gates.append((gate, True))
            else: # pragma: no cover
                raise ValueError("Cannot add gate on qubits {} and {} in "
                                 "fusion group of qubit {}."
                                 "".format(qubit0, qubit1, self.qubit0))
        else:
            if self.qubits != {qubit0, qubit1}: # pragma: no cover
                raise ValueError("Cannot add gate on qubits {} and {} in "
                                 "fusion group of qubits {} and {}."
                                 "".format(qubit0, qubit1, self.qubit0, self.qubit1))
            self.two_qubit_gates.append((gate, self.qubit1 == qubit0))

        self.gates0.append([])
        self.gates1.append([])


def fuse_queue(queue: List[gates.Gate]) -> List[FusionGroup]:
    group_queue = []
    remaining_queue = list(queue)
    while remaining_queue:
        new_remaining_queue = []
        gates = iter(remaining_queue)
        new_group = FusionGroup()
        new_group.add(next(gates))
        for gate in gates:
            commutes = True
            for blocking_gate in new_remaining_queue:
                commutes = commutes and gate.commutes(blocking_gate)
                if not commutes:
                    break

            if commutes:
                try:
                    new_group.add(gate)
                except ValueError:
                    new_remaining_queue.append(gate)
            else:
                new_remaining_queue.append(gate)

        group_queue.append(new_group)
        remaining_queue = list(new_remaining_queue)

    return group_queue
