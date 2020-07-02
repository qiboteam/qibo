import functools
import operator
import tensorflow as tf
from qibo.base import gates
from qibo.config import DTYPES
from typing import List, Optional, Set, Tuple


class FusionGroup:
    """Group of one-qubit and two-qubit gates that act in two specific gates.

    These gates can be fused into a single two-qubit gate represented by a
    general 4x4 matrix.

    Attrs:
        qubit0 (int): Id of the first qubit that the gates act.
        qubit1 (int): Id of the first qubit that the gates act.
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

    def __init__(self):
        self.qubit0 = None
        self.qubit1 = None
        self.gates0 = [[]]
        self.gates1 = [[]]
        self.two_qubit_gates = [] # list of tuples (gate, revert flag)

        self.special_gate = None
        self._fused_gates = None

    @property
    def qubits(self) -> Set[int]:
        if self.qubit0 is None:
            return {}
        if self.qubit1 is None:
            return {self.qubit0}
        return {self.qubit0, self.qubit1}

    @property
    def fused_gates(self) -> Tuple[gates.Gate]:
        if self._fused_gates is None:
            self._fused_gates = self.calculate_fused_gates()
        return self._fused_gates

    @property
    def module(self):
        if self.gates0[0]:
            return self.gates0[0][0].module
        if self.gates1[0]:
            return self.gates1[0][0].module
        if self.two_qubit_gates:
            return self.two_qubit_gates[0].module
        if self.special_gate is not None:
            return self.special_gate.module
        raise ValueError("Unable to find gate module.")

    def add(self, gate: gates.Gate):
        """Adds a gate in the group."""
        if self._fused_gates is not None:
            raise RuntimeError("Cannot add gates to ``FusionGroup`` for "
                               "which the fused gate was already calculated.")

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
        else:
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
            else:
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

    def _one_qubit_matrix(self, gate0: gates.Gate, gate1: gates.Gate) -> tf.Tensor:
        """Calculates 4x4 Kroneker product of one qubit gate unitary matrices."""
        matrix0 = gate0.construct_unitary(*gate0.unitary_params)
        matrix1 = gate1.construct_unitary(*gate1.unitary_params)
        matrix = tf.tensordot(matrix0, matrix1, axes=0)
        matrix = tf.reshape(tf.transpose(matrix, [0, 2, 1, 3]), (4, 4))
        return matrix

    def _two_qubit_matrix(self, gate: Optional[gates.Gate] = None,
                          revert: bool = False) -> tf.Tensor:
        matrix = gate.construct_unitary(*gate.unitary_params)
        if revert:
            matrix = tf.reshape(matrix, 4 * (2,))
            matrix = tf.transpose(matrix, [1, 0, 3, 2])
            matrix = tf.reshape(matrix, (4, 4))
        return matrix

    def calculate_fused_gates(self) -> Tuple[gates.Gate]:
        if self.special_gate is not None:
            assert not self.gates0[0] and not self.gates1[0]
            assert not self.two_qubit_gates
            return (self.special_gate,)

        module = self.module
        ident1 = module.I(self.qubit0)
        ident2 = module.I(self.qubit1)
        # Fuse one-qubit gates
        gates0 = (functools.reduce(operator.matmul, reversed(gates), ident1)
                  for gates in self.gates0)
        gates1 = (functools.reduce(operator.matmul, reversed(gates), ident2)
                  for gates in self.gates1)

        if not self.two_qubit_gates:
            return (functools.reduce(operator.matmul, reversed(gates0), ident1),
                    functools.reduce(operator.matmul, reversed(gates1), ident2))

        fused_matrix = self._one_qubit_matrix(next(gates0), next(gates1))
        for g0, g1, (g2, flag) in zip(gates0, gates1, self.two_qubit_gates):
            matrix = self._one_qubit_matrix(g0, g1)
            matrix2 = self._two_qubit_matrix(g2, flag)
            fused_matrix = tf.matmul(tf.matmul(matrix, matrix2), fused_matrix)
        return (module.Unitary(fused_matrix, self.qubit0, self.qubit1),)


def fuse_queue(queue: List[gates.Gate]) -> List[FusionGroup]:
    """Fuses a queue of gates by combining up to two-qubit gates.

    Args:
        queue (list): List of gates.

    Returns:
        List of ``FusionGroup`` objects that correspond to the fused gates.
    """
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
