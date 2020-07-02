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
        if self.special_gate is not None:
            assert not self.gates0[0] and not self.gates1[0]
            assert not self.two_qubit_gates
            return (self.special_gate,)

        module = self.module
        ident0 = module.I(self.qubit0)
        ident1 = module.I(self.qubit1)
        # Fuse one-qubit gates
        gates0 = (functools.reduce(operator.matmul, reversed(gates), ident0)
                  if len(gates) != 1 else gates[0] for gates in self.gates0)
        gates1 = (functools.reduce(operator.matmul, reversed(gates), ident1)
                  if len(gates) != 1 else gates[0] for gates in self.gates1)

        if not self.two_qubit_gates:
            gates0 = reversed(list(gates0))
            gates1 = reversed(list(gates1))
            fused_gate0 = (functools.reduce(operator.matmul, gates0, ident0)
                           if len(gates0) != 1 else gates0[0])
            fused_gate1 = (functools.reduce(operator.matmul, gates1, ident1)
                           if len(gates1) != 1 else gates1[0])
            return (fused_gate0, fused_gate1)

        fused_matrix = self._one_qubit_matrix(next(gates0), next(gates1))
        for g0, g1, (g2, flag) in zip(gates0, gates1, self.two_qubit_gates):
            matrix = self._one_qubit_matrix(g0, g1)
            matrix2 = self._two_qubit_matrix(g2, flag)
            fused_matrix = tf.matmul(tf.matmul(matrix, matrix2), fused_matrix)
        return (module.Unitary(fused_matrix, self.qubit0, self.qubit1),)
