"""Custom implementation of `tf.einsum` using `tf.matmul`.

Temporary solution until the complex gradients of `tf.einsum` are fixed.
This approach is not optimal in terms of performance.
"""
import tensorflow as tf
from typing import Sequence


class DefaultEinsum:

    _chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __call__(self, cache, state: tf.Tensor, gate: tf.Tensor) -> tf.Tensor:
        return tf.einsum(cache, state, gate)

    @classmethod
    def create_cache(cls, qubits: Sequence[int], nqubits: int) -> str:
        """Creates index string for `tf.einsum`.

        Args:
            qubits: List with the qubit indices that the gate is applied to.

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
        gate_str = "".join(gate_chars)
        output_str = "".join(output_state)
        return "{},{}->{}".format(input_str, gate_str, output_str)


class MatmulEinsum:

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
          qubits: Tuple with the qubit indices that the gate is applied to.
          nqubits: Total number of qubits in the circuit / state vector.
      Returns:
          indices: Tuple indices for the first transposition (before matmul)
              and the inverse transposition (after matmul).
          shapes: Tuple with the four reshape shapes.
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

      cache = {"indices": ids, "inv_indices": inv_ids,
               "shapes": (shape, (2 ** ntargets, 2 ** nrest),
                          transposed_shape, nqubits * (2,))}
      return cache
