import functools
import operator
import tensorflow as tf
from qibo.base import fusion
from typing import Tuple


class FusionGroup(fusion.FusionGroup):

    def _one_qubit_matrix(self, gate0: "Gate", gate1: "Gate") -> tf.Tensor:
        """Calculates 4x4 Kroneker product of one qubit gate unitary matrices."""
        matrix0 = gate0.construct_unitary(*gate0.unitary_params)
        matrix1 = gate1.construct_unitary(*gate1.unitary_params)
        matrix = tf.tensordot(matrix0, matrix1, axes=0)
        matrix = tf.reshape(tf.transpose(matrix, [0, 2, 1, 3]), (4, 4))
        return matrix

    def _two_qubit_matrix(self, gate: "Gate", revert: bool = False) -> tf.Tensor:
        matrix = gate.construct_unitary(*gate.unitary_params)
        if revert:
            matrix = tf.reshape(matrix, 4 * (2,))
            matrix = tf.transpose(matrix, [1, 0, 3, 2])
            matrix = tf.reshape(matrix, (4, 4))
        return matrix

    def calculate(self) -> Tuple["Gate"]:
        # Case 1: Special gate
        if self.special_gate is not None:
            assert not self.gates0[0] and not self.gates1[0]
            assert not self.two_qubit_gates
            return (self.special_gate,)

        # Case 2: Two-qubit gates only (no one-qubit gates)
        if self.first_gate(0) is None and self.first_gate(1) is None:
            assert self.two_qubit_gates
            # Case 2a: One two-qubit gate only
            if len(self.two_qubit_gates) == 1:
                return (self.two_qubit_gates[0][0],)

            # Case 2b: Two or more two-qubit gates
            module = self.module
            fused_matrix = self._two_qubit_matrix(*self.two_qubit_gates[0])
            for gate, flag in self.two_qubit_gates[1:]:
                matrix = self._two_qubit_matrix(gate, flag)
                fused_matrix = tf.matmul(matrix, fused_matrix)
            return (module.Unitary(fused_matrix, self.qubit0, self.qubit1),)

        # Case 3: One-qubit gates exist
        if not self.gates0[-1] and not self.gates1[-1]:
            self.gates0.pop()
            self.gates1.pop()

        module = self.module
        ident0 = module.I(self.qubit0)
        ident1 = module.I(self.qubit1)
        # Fuse one-qubit gates
        gates0 = (functools.reduce(operator.matmul, reversed(gates), ident0)
                  if len(gates) != 1 else gates[0] for gates in self.gates0)
        gates1 = (functools.reduce(operator.matmul, reversed(gates), ident1)
                  if len(gates) != 1 else gates[0] for gates in self.gates1)

        # Case 3a: One-qubit gates only (no two-qubit gates)
        if not self.two_qubit_gates:
            gates0 = list(gates0)[::-1]
            gates1 = list(gates1)[::-1]
            fused_gate0 = (functools.reduce(operator.matmul, gates0, ident0)
                           if len(gates0) != 1 else gates0[0])
            fused_gate1 = (functools.reduce(operator.matmul, gates1, ident1)
                           if len(gates1) != 1 else gates1[0])
            return (fused_gate0, fused_gate1)

        # Case 3b: One-qubit and two-qubit gates exist
        fused_matrix = self._one_qubit_matrix(next(gates0), next(gates1))
        for g0, g1, (g2, flag) in zip(gates0, gates1, self.two_qubit_gates):
            matrix = self._one_qubit_matrix(g0, g1)
            matrix2 = self._two_qubit_matrix(g2, flag)
            fused_matrix = tf.matmul(tf.matmul(matrix, matrix2), fused_matrix)

        if len(self.two_qubit_gates) == len(self.gates0):
            g2, flag = self.two_qubit_gates[-1]
            #if self.is_efficient(g2):
            #    fused_gate = module.Unitary(fused_matrix, self.qubit0, self.qubit1)
            #    return (fused_gate, g2)

            matrix2 = self._two_qubit_matrix(g2, flag)
            fused_matrix = tf.matmul(matrix2, fused_matrix)

        fused_gate = module.Unitary(fused_matrix, self.qubit0, self.qubit1)
        return (fused_gate,)
