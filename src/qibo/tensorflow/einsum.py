# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
"""
Tensorflow gates use ``einsum`` to apply gates to state vectors. The einsum string that
specifies the contraction indices is created and cached when a gate is created
so that it is not recalculated every time the gate is called on a state. This
functionality is implemented in :class:`qibo.tensorflow.einsum.DefaultEinsum`.

Due to an `issue <https://github.com/tensorflow/tensorflow/issues/37307>`_
with automatic differentiation and complex numbers in ``einsum``, we have
implemented an alternative calculation backend based on ``matmul`` in
:class:`qibo.tensorflow.einsum.MatmulEinsum`. Note that this is slower than
the default ``einsum`` on GPU but slightly faster on CPU.

The user can switch the default einsum used by the gates by changing the
``einsum`` variable in `config.py`. It is recommended to use the default unless
automatic differentiation is required. For the latter case, we refer to our
examples.
"""
from qibo import K
from qibo.base import cache
from qibo.config import raise_error
from typing import Dict, Optional, Sequence


class DefaultEinsum:
    """Gate application backend that based on default ``einsum``.

    This is the most efficient implementation for GPU, however its
    backpropagation is not working properly for complex numbers.
    The user should switch to :class:`qibo.tensorflow.einsum.MatmulEinsum`
    if automatic differentiation is required.
    """

    def __call__(self, cache: str, state: K.Tensor, gate: K.Tensor) -> K.Tensor:
      return K.einsum(cache, state, gate)

    @staticmethod
    def create_cache(qubits: Sequence[int], nqubits: int,
                     ncontrol: Optional[int] = None) -> cache.DefaultEinsumCache:
        return cache.DefaultEinsumCache(qubits, nqubits, ncontrol)


class MatmulEinsum:
  """Gate application backend based on ``matmul``.

  For Tensorflow this is more efficient than ``einsum`` on CPU but slower on GPU.
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

  def __call__(self, cache: Dict, state: K.Tensor, gate: K.Tensor) -> K.Tensor:
      shapes = cache["shapes"]

      state = K.reshape(state, shapes[0])
      state = K.transpose(state, cache["ids"])
      if cache["conjugate"]:
          state = K.reshape(K.conj(state), shapes[1])
      else:
          state = K.reshape(state, shapes[1])

      n = len(tuple(gate.shape))
      if n > 2:
          dim = 2 ** (n // 2)
          state = K.matmul(K.reshape(gate, (dim, dim)), state)
      else:
          state = K.matmul(gate, state)

      state = K.reshape(state, shapes[2])
      state = K.transpose(state, cache["inverse_ids"])
      state = K.reshape(state, shapes[3])
      return state

  @staticmethod
  def create_cache(qubits: Sequence[int], nqubits: int,
                   ncontrol: Optional[int] = None) -> cache.MatmulEinsumCache:
      return cache.MatmulEinsumCache(qubits, nqubits, ncontrol)
