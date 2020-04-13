"""Custom implementations of `tf.einsum` used to apply gates to state vectors.

The user can switch the default einsum used by the gates by changing the
`einsum` variable in `config.py`.
"""
import tensorflow as tf
from typing import Sequence


class DefaultEinsum:
    """Einsum backend that uses Tensorflow's default `tf.einsum`.

    This is the most efficient implementation for GPU, however its
    backpropagation is not working properly for complex numbers.
    The user should switch to :class:`qibo.tensorflow.einsum.MatmulEinsum`
    if automatic differentiation is required.
    """

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
  """Einsum backend that uses a custom implementation based on `tf.matmul`.

  This is more efficient than `tf.einsum` on CPU but slower on GPU.
  The matmul version implemented here is not the most efficient possible.
  The implementation algorithm is the following. Assume that we are applying
  a two qubit gate of shape (4, 4) to qubits 0 and 3 of a five qubit state
  vector of shape 5 * (2,). We perform the following steps:
  1) Reshape the state to (2, 4, 2, 2)
  2) Transpose to (2, 2, 4, 2) to bring the target qubits in the beginning.
  3) Reshape to (4, 8).
  4) Apply the gate using the matmul (4, 4) x (4, 8).
  5) Reshape to the original shape 5 * (2,) and traspose so that the final
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
