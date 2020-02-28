# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import itertools
import numpy as np
import tensorflow as tf
from qibo.base import gates as base_gates
from typing import Optional, Tuple


class TensorflowGate:
    """The base Tensorflow gate."""
    from qibo.config import DTYPECPX as dtype

    def _iterables(self, d):
      for i in range(1, len(d)):
          yield range(d[i] // (2 * d[i - 1]))
      yield range(self.nstates // d[-1])

    def _base_slicer(self, d: np.ndarray) -> np.ndarray:
        d_sorted = np.array([1] + list(np.sort(d)))
        configurations = itertools.product(*list(self._iterables(d_sorted)))
        return np.array([d_sorted.dot(c) for c in configurations])

    def _create_slicers(self) -> Tuple[Tuple[int], Tuple[int]]:
        d = 2 ** (self.nqubits - np.array(self.qubits))
        slicer = self._base_slicer(d)
        return tuple(slicer), tuple(slicer + d[0] // 2)

    def call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        """Implements the `Gate` on a given state."""
        # Set nqubits for the case the gate is called outside of the circuit
        if self._nqubits is None:
            self.nqubits = int(np.log2(state.shape[0]))

        slice0, slice1 = self._create_slicers()
        state0 = tf.gather(state, slice0)
        state1 = tf.gather(state, slice1)

        new0 = self.call_0(state0, state1)
        new1 = self.call_1(state0, state1)

        slice0 = tf.constant(slice0)[:, tf.newaxis]
        slice1 = tf.constant(slice1)[:, tf.newaxis]

        new = tf.tensor_scatter_nd_update(state, slice0, new0)
        new = tf.tensor_scatter_nd_update(new, slice1, new1)
        return new


class TensorflowControlledGate(TensorflowGate):

    def _create_slicers(self) -> Tuple[Tuple[int], Tuple[int]]:
        d = 2 ** (self.nqubits - np.array(self.qubits))
        slicer = self._base_slicer(d)
        return tuple(slicer + d[0] // 2), tuple(slicer + (d[0] + d[1]) // 2)


class H(TensorflowGate, base_gates.H):

    def __init__(self, *args):
        base_gates.H.__init__(self, *args)

    def call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return (state0 + state1) / np.sqrt(2)

    def call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return (state0 - state1) / np.sqrt(2)


class X(TensorflowGate, base_gates.X):

    def __init__(self, *args):
        base_gates.X.__init__(self, *args)

    def call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state1

    def call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state0


class Y(TensorflowGate, base_gates.Y):

    def __init__(self, *args):
        base_gates.Y.__init__(self, *args)

    def call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return -1j * state1

    def call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return 1j * state0


class Z(TensorflowGate, base_gates.Z):

    def __init__(self, *args):
        base_gates.Z.__init__(self, *args)

    def call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state0

    def call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return -state1


class Barrier(TensorflowGate, base_gates.Barrier):

    def __init__(self):
        raise NotImplementedError


class Iden(TensorflowGate, base_gates.Iden):

    def __init__(self, *args):
        base_gates.Iden.__init__(self, *args)

    def call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state0

    def call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state1


class MX(TensorflowGate, base_gates.MX):

    def __init__(self):
        raise NotImplementedError


class MY(TensorflowGate, base_gates.MY):

    def __init__(self):
        raise NotImplementedError


class MZ(TensorflowGate, base_gates.MZ):

    def __init__(self):
        raise NotImplementedError


class RX(TensorflowGate, base_gates.RX):

    def __init__(self, *args):
        base_gates.RX.__init__(self, *args)

        self.phase = tf.exp(1j * np.pi * self.theta / 2.0)
        self.cos = tf.cast(tf.math.real(self.phase), dtype=self.dtype)
        self.sin = tf.cast(tf.math.imag(self.phase), dtype=self.dtype)

    def call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return self.phase * (self.cos * state0 - 1j * self.sin * state1)

    def call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return self.phase * (- 1j * self.sin * state0 + self.cos * state1)


class RY(TensorflowGate, base_gates.RY):

    def __init__(self, *args):
        base_gates.RY.__init__(self, *args)

        self.phase = tf.exp(1j * np.pi * self.theta / 2.0)
        self.cos = tf.cast(tf.math.real(self.phase), dtype=self.dtype)
        self.sin = tf.cast(tf.math.imag(self.phase), dtype=self.dtype)

    def call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return self.phase * (self.cos * state0 - self.sin * state1)

    def call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return self.phase * (self.sin * state0 + self.cos * state1)


class RZ(TensorflowGate, base_gates.RZ):

    def __init__(self, *args):
        base_gates.RZ.__init__(self, *args)
        self.phase = tf.exp(1j * np.pi * self.theta)

    def call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state0

    def call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return self.phase * state1


class CNOT(TensorflowControlledGate, base_gates.CNOT):

    def __init__(self, *args):
        base_gates.CNOT.__init__(self, *args)

    def call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state1

    def call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state0


class SWAP(TensorflowGate, base_gates.SWAP):

    def __init__(self, *args):
        base_gates.SWAP.__init__(self, *args)

    def _create_slicers(self) -> Tuple[Tuple[int], Tuple[int]]:
        d = 2 ** (self.nqubits - np.array(self.qubits))
        slicer = self._base_slicer(d)
        return tuple(slicer + d[0] // 2), tuple(slicer + d[1] // 2)

    def call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state1

    def call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state0


class CRZ(TensorflowControlledGate, base_gates.CRZ):

    def __init__(self, *args):
        base_gates.CRZ.__init__(self, *args)
        self.phase = tf.exp(1j * np.pi * self.theta)

    def call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state0

    def call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return self.phase * state1


class Toffoli(TensorflowGate, base_gates.Toffoli):

    def __init__(self, *args):
        base_gates.Toffoli.__init__(self, *args)

    def _create_slicers(self) -> Tuple[Tuple[int], Tuple[int]]:
        d = 2 ** (self.nqubits - np.array(self.qubits))
        slicer = self._base_slicer(d)
        c = (d[0] + d[1]) // 2
        return tuple(slicer + c), tuple(slicer + c + d[2] // 2)

    def call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state1

    def call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state0


class Flatten(TensorflowGate, base_gates.Flatten):

    def __init__(self, *args):
        base_gates.Flatten.__init__(self, *args)

    def __call__(self, state):
        if len(self.coefficients) != 2 ** self.nqubits:
                raise ValueError(
                    "Circuit was created with {} qubits but the "
                    "flatten layer state has {} coefficients."
                    "".format(self.nqubits, self.coefficients)
                )

        return tf.convert_to_tensor(self.coefficients, dtype=state.dtype)
