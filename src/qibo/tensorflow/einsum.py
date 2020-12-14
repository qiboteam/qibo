# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
"""
Tensorflow gates use ``tf.einsum`` to apply gates to state vectors. The einsum string that
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
from qibo.base import cache
from qibo.config import raise_error
from typing import Dict, Optional, Sequence


class DefaultEinsum:
    """Einsum backend that uses Tensorflow's default ``tf.einsum``.

    This is the most efficient implementation for GPU, however its
    backpropagation is not working properly for complex numbers.
    The user should switch to :class:`qibo.tensorflow.einsum.MatmulEinsum`
    if automatic differentiation is required.
    """

    def __call__(self, cache: str, state: tf.Tensor, gate: tf.Tensor) -> tf.Tensor:
      return tf.einsum(cache, state, gate)

    @staticmethod
    def create_cache(qubits: Sequence[int], nqubits: int,
                     ncontrol: Optional[int] = None) -> cache.DefaultEinsumCache:
        return cache.DefaultEinsumCache(qubits, nqubits, ncontrol)


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

  def __call__(self, cache: Dict, state: tf.Tensor,
               gate: tf.Tensor) -> tf.Tensor:
      shapes = cache["shapes"]

      state = tf.reshape(state, shapes[0])
      state = tf.transpose(state, cache["ids"])
      if cache["conjugate"]:
          state = tf.reshape(tf.math.conj(state), shapes[1])
      else:
          state = tf.reshape(state, shapes[1])

      n = len(tuple(gate.shape))
      if n > 2:
          dim = 2 ** (n // 2)
          state = tf.matmul(tf.reshape(gate, (dim, dim)), state)
      else:
          state = tf.matmul(gate, state)

      state = tf.reshape(state, shapes[2])
      state = tf.transpose(state, cache["inverse_ids"])
      state = tf.reshape(state, shapes[3])
      return state

  @staticmethod
  def create_cache(qubits: Sequence[int], nqubits: int,
                   ncontrol: Optional[int] = None) -> cache.MatmulEinsumCache:
      return cache.MatmulEinsumCache(qubits, nqubits, ncontrol)
