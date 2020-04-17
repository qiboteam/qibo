# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
"""Custom implementations of ``tf.einsum`` used to apply gates to state vectors.

QIBO uses ``tf.einsum`` to apply gates to state vectors. The einsum string that
specifies the contraction indices is created and cached when a gate is created
so that it is not recalculated every time the gate is called on a state. This
functionality is implemented in :class:`qibo.tensorflow.einsum.DefaultEinsum`.

Due to an `issue <https://github.com/tensorflow/tensorflow/issues/37307>`_
with automatic differentiation and complex numbers in ``tf.einsum``, we have
implemented an alternative calculation backend based on ``tf.matmul`` in
:class:`qibo.tensorflow.einsum.MatmulEinsum`. Note that this is slower than
the default ``tf.einsum`` on GPU but slightly faster on CPU.

The user can switch the default einsum used by the gates by changing the
``einsum`` variable in `config.py`. It is recommended to use the default unless
automatic differentiation is required. For the latter case, we refer to our
examples.
"""
import tensorflow as tf
from typing import Sequence


class DefaultEinsum:
    """Einsum backend that uses Tensorflow's default ``tf.einsum``.

    This is the most efficient implementation for GPU, however its
    backpropagation is not working properly for complex numbers.
    The user should switch to :class:`qibo.tensorflow.einsum.MatmulEinsum`
    if automatic differentiation is required.
    """

    _chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __call__(self, cache: str, state: tf.Tensor, gate: tf.Tensor) -> tf.Tensor:
      return tf.einsum(cache, state, gate)

    @classmethod
    def create_cache(cls, qubits: Sequence[int], nqubits: int) -> str:
        """Creates index string for `tf.einsum`.

        Args:
            qubits (list): List with the qubit indices that the gate is applied to.
            nqubits (int): Total number of qubits in the circuit / state vector.

        Returns:
            String formated as {input state}{gate matrix}->{output state}.
        """
        if len(qubits) + nqubits > len(cls._chars):
            raise NotImplementedError("Not enough einsum characters.")

        input_state = list(cls._chars[: nqubits])
        output_state = input_state[:]
        gate_chars = list(cls._chars[nqubits : nqubits + len(qubits)])

        for i, q in enumerate(qubits):
            gate_chars.append(input_state[q])
            output_state[q] = gate_chars[i]

        input_str = "".join(input_state)
        output_str = "".join(output_state)
        gate_str = "".join(gate_chars)
        rest_str = cls._chars[nqubits + len(qubits): 2 * nqubits + len(qubits)]

        cache = {"vector": f"{input_str},{gate_str}->{output_str}",
                 "dmL": f"{input_str}{rest_str},{gate_str}->{output_str}{rest_str}",
                 "dmR": f"{rest_str}{input_str},{gate_str}->{rest_str}{output_str}"}
        return cache


class MatmulEinsum:
  """Einsum backend that uses a custom implementation based on ``tf.matmul``.

  This is more efficient than ``tf.einsum`` on CPU but slower on GPU.
  The matmul version implemented here is not the most efficient possible.
  The implementation algorithm is the following.

  Assume that we are applying
  a two qubit gate of shape (4, 4) to qubits 0 and 3 of a five qubit state
  vector of shape 5 * (2,). We perform the following steps:

  * Reshape the state to (2, 4, 2, 2)
  * Transpose to (2, 2, 4, 2) to bring the target qubits in the beginning.
  * Reshape to (4, 8).
  * Apply the gate using the matmul (4, 4) x (4, 8).
  * Reshape to the original shape 5 * (2,) and traspose so that the final
    qubit order agrees with the initial.
  """

  def __call__(self, cache, state: tf.Tensor, gate: tf.Tensor) -> tf.Tensor:
      indices, inv_indices = cache["indices"], cache["inv_indices"]
      shapes = cache["shapes"]

      state = tf.reshape(state, shapes[0])
      state = tf.transpose(state, indices)
      state = tf.reshape(state, shapes[1])

      n = len(tuple(gate.shape))
      if n > 2:
          dim = 2 ** (n // 2)
          state = tf.matmul(tf.reshape(gate, (dim, dim)), state)
      else:
          state = tf.matmul(gate, state)

      state = tf.reshape(state, shapes[2])
      state = tf.transpose(state, inv_indices)
      state = tf.reshape(state, shapes[3])
      return state

  @staticmethod
  def create_cache(qubits: Sequence[int], nqubits: int):
      """Creates indeces and shapes required for gate application with matmul.

      Args:
          qubits (tuple): Tuple with the qubit indices that the gate is applied to.
          nqubits (int): Total number of qubits in the circuit / state vector.

      Returns:
          Indices for the first transposition (before matmul) and the inverse
          transposition (after matmul) and the four reshape shapes.
      """
      ntargets = len(qubits)
      nrest = nqubits - ntargets

      last_index = 0
      target_ids = {}
      rest_ids = []
      shape = []
      for q in sorted(qubits):
          if q > last_index:
              shape.append(2 ** (q - last_index))
              rest_ids.append(len(shape) - 1)
          shape.append(2)
          target_ids[q] = len(shape) - 1
          last_index = q + 1
      if last_index < nqubits:
          shape.append(2 ** (nqubits - last_index))
          rest_ids.append(len(shape) - 1)

      ids = [target_ids[q] for q in qubits] + rest_ids
      transposed_shape = []
      inv_ids = len(ids) * [0]
      for i, r in enumerate(ids):
          inv_ids[r] = i
          transposed_shape.append(shape[r])

      cache = {}
      cache["vector"] = {"indices": ids, "inv_indices": inv_ids,
                         "shapes": (shape, (2 ** ntargets, 2 ** nrest),
                                    transposed_shape, nqubits * (2,))}
      return cache
