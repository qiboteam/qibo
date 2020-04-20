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
from typing import Dict, List, Sequence, Set, Tuple


class BaseCache:

    def __init__(self):
        # Cache for state vectors
        self._vector = None
        # Cache for density matrices
        self._left = None
        self._right = None
        self._left0 = None
        self._right0 = None

    @property
    def vector(self):
        if self._vector is None:
            self._calculate_state_vector()
        return self._vector

    @property
    def left(self):
        if self._left is None:
            self._calculate_density_matrix()
        return self._left

    @property
    def right(self):
        if self._right is None:
            self._calculate_density_matrix()
        return self._right

    @property
    def left0(self):
        if self._left0 is None:
            self._calculate_density_matrix(is_controlled_by=True)
        return self._left0

    @property
    def right0(self):
        if self._right0 is None:
            self._calculate_density_matrix(is_controlled_by=True)
        return self._right0

    def _calculate_state_vector(self):
        raise NotImplementedError

    def _calculate_density_matrix(self, is_controlled_by: bool = False):
        raise NotImplementedError


class DefaultEinsumCache(BaseCache):

    _chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, qubits: Sequence[int], nqubits: int):
      """Creates index string for `tf.einsum`.

      Args:
          qubits (list): List with the qubit indices that the gate is applied to.
          nqubits (int): Total number of qubits in the circuit / state vector.

      Returns:
          String formated as {input state}{gate matrix}->{output state}.
      """
      super(DefaultEinsumCache, self).__init__()
      self.nqubits = nqubits

      if nqubits + len(qubits) > len(self._chars):
          raise NotImplementedError("Not enough einsum characters.")

      input_state = list(self._chars[:nqubits])
      output_state = input_state[:]
      gate_chars = list(self._chars[nqubits : nqubits + len(qubits)])

      for i, q in enumerate(qubits):
          gate_chars.append(input_state[q])
          output_state[q] = gate_chars[i]

      self.input = "".join(input_state)
      self.output = "".join(output_state)
      self.gate = "".join(gate_chars)
      self.rest = self._chars[nqubits + len(qubits):]

      self._vector = f"{self.input},{self.gate}->{self.output}"

    def _calculate_density_matrix(self, is_controlled_by: bool = False):
        if self.nqubits > len(self.rest):
            raise NotImplementedError("Not enough einsum characters.")

        rest = self.rest[:self.nqubits]

        self._left = f"{self.input}{rest},{self.gate}->{self.output}{rest}"
        self._right = f"{rest}{self.input},{self.gate}->{rest}{self.output}"

        if is_controlled_by:
            if self.nqubits + 1 > len(self.rest):
                raise NotImplementedError("Not enough einsum characters.")
            c = self.rest[self.nqubits]
            self._left0 = f"{c}{self.input}{rest},{self.gate}->{c}{self.output}{rest}"
            self._right0 = f"{c}{rest}{self.input},{self.gate}->{c}{rest}{self.output}"


class MatmulEinsumCache(BaseCache):

    def __init__(self, qubits: Sequence[int], nqubits: int):
        """Creates indeces and shapes required for gate application with matmul.

        Args:
            qubits (tuple): Tuple with the qubit indices that the gate is applied to.
            nqubits (int): Total number of qubits in the circuit / state vector.

        Returns:
            Indices for the first transposition (before matmul) and the inverse
            transposition (after matmul) and the four reshape shapes.
        """
        super(MatmulEinsumCache, self).__init__()
        self.nqubits = nqubits
        self.ntargets = len(qubits)
        self.nrest = nqubits - self.ntargets
        self.nstates = 2 ** nqubits

        last_index = 0
        target_ids, rest_ids = {}, []
        self.shape = []
        for q in sorted(qubits):
            if q > last_index:
                self.shape.append(2 ** (q - last_index))
                rest_ids.append(len(self.shape) - 1)
            self.shape.append(2)
            target_ids[q] = len(self.shape) - 1
            last_index = q + 1
        if last_index < self.nqubits:
            self.shape.append(2 ** (self.nqubits - last_index))
            rest_ids.append(len(self.shape) - 1)

        self.ids = [target_ids[q] for q in qubits] + rest_ids
        self.transposed_shape = []
        self.inverse_ids = len(self.ids) * [0]
        for i, r in enumerate(self.ids):
            self.inverse_ids[r] = i
            self.transposed_shape.append(self.shape[r])

        self.shape = tuple(self.shape)
        self.transposed_shape = tuple(self.transposed_shape)
        self._vector = {"ids": self.ids, "inverse_ids": self.inverse_ids,
                        "shapes": (self.shape,
                                   (2 ** self.ntargets, 2 ** self.nrest),
                                   self.transposed_shape,
                                   self.nqubits * (2,)),
                        "conjugate": False}

    def _calculate_density_matrix(self, is_controlled_by: bool = False):
        self._left = {"ids": self.ids + [len(self.ids)],
                      "inverse_ids": self.inverse_ids + [len(self.ids)],
                      "shapes": (self.shape + (self.nstates,),
                                 (2 ** self.ntargets, (2 ** self.nrest) * self.nstates),
                                 self.transposed_shape + (self.nstates,),
                                 2 * self.nqubits * (2,)),
                      "conjugate": False}

        self._right = dict(self._left)
        self._right["inverse_ids"] = [len(self.ids)] + self.inverse_ids
        self._right["conjugate"] = True

        if is_controlled_by:
            raise NotImplementedError



class DefaultEinsum:
    """Einsum backend that uses Tensorflow's default ``tf.einsum``.

    This is the most efficient implementation for GPU, however its
    backpropagation is not working properly for complex numbers.
    The user should switch to :class:`qibo.tensorflow.einsum.MatmulEinsum`
    if automatic differentiation is required.
    """

    _chars = DefaultEinsumCache._chars

    def __call__(self, cache: str, state: tf.Tensor, gate: tf.Tensor) -> tf.Tensor:
      return tf.einsum(cache, state, gate)

    @staticmethod
    def create_cache(qubits: Sequence[int], nqubits: int) -> DefaultEinsumCache:
        return DefaultEinsumCache(qubits, nqubits)

    @classmethod
    def partialtrace_str(cls, qubits: Set[int], nqubits: int,
                         measuring: bool = False) -> str:
        """Generates einsum strings for partial trace of density matrices.

        Helper method used when measuring or calculating entanglement entropies
        on density matrices.

        Args:
            qubits: Set of qubit ids that are traced out.
            nqubits: Total number of qubits in the state.
            measuring: If True non-traced-out indices are multiplied and the
                output has shape (nqubits - len(qubits),).
                If False the output has shape 2 * (nqubits - len(qubits),).

        Returns:
            String to use in einsum for performing partial density of a
            density matrix.
        """
        if (2 - int(measuring)) * nqubits > len(cls._chars):
            raise NotImplementedError("Not enough einsum characters.")

        left_in, right_in, left_out, right_out = [], [], [], []
        for i in range(nqubits):
            left_in.append(cls._chars[i])
            if i in qubits:
                right_in.append(cls._chars[i])
            else:
                left_out.append(cls._chars[i])
                if measuring:
                    right_in.append(cls._chars[i])
                else:
                    right_in.append(cls._chars[i + nqubits])
                    right_out.append(cls._chars[i + nqubits])

        left_in, left_out = "".join(left_in), "".join(left_out)
        right_in, right_out = "".join(right_in), "".join(right_out)
        return f"{left_in}{right_in}->{left_out}{right_out}"


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
  def create_cache(qubits: Sequence[int], nqubits: int) -> MatmulEinsumCache:
      return MatmulEinsumCache(qubits, nqubits)
