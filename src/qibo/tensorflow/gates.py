# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import gates as base_gates
from typing import Tuple


class TensorflowGate:
    """The base Tensorflow gate."""

    def slice_generator(self, q: int, is_one: bool = False) -> int:
        s = (q + 1) * int(is_one)
        while s < self.nqubits:
            for i in range(s, s + q + 1):
                yield i
            s += q + 2

    def _call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def _call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        """Implements the `Gate` on a given state."""
        slice0 = tuple(self.slice_generator(self.qubits[0], False))
        slice1 = tuple(self.slice_generator(self.qubits[0], True))

        new0 = self._call_0(state[slice0], state[slice1])
        new1 = self._call_1(state[slice0], state[slice1])

        new = tf.tensor_scatter_nd_update(state, slice0, new0)
        new = tf.tensor_scatter_nd_update(new, slice1, new1)
        return new


class H(TensorflowGate, base_gates.H):

    def __init__(self, *args):
        base_gates.H.__init__(self, *args)

    def _call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return (state0 + state1) / np.sqrt(2)

    def _call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return (state0 - state1) / np.sqrt(2)


class X(TensorflowGate, base_gates.X):

    def __init__(self, *args):
        base_gates.X.__init__(self, *args)


class Y(TensorflowGate, base_gates.Y):

    def __init__(self, *args):
        base_gates.Y.__init__(self, *args)


class Z(TensorflowGate, base_gates.Z):

    def __init__(self, *args):
        base_gates.Z.__init__(self, *args)


class Barrier(TensorflowGate, base_gates.Barrier):

    def __init__(self):
        raise NotImplementedError


class Iden(TensorflowGate, base_gates.Iden):

    def __init__(self, *args):
        base_gates.Iden.__init__(self, *args)


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

        phase = tf.exp(1j * np.pi * self.theta / 2.0)
        cos = tf.cast(tf.math.real(phase), dtype=self.dtype)
        sin = tf.cast(tf.math.imag(phase), dtype=self.dtype)


class RY(TensorflowGate, base_gates.RY):

    def __init__(self, *args):
        base_gates.RY.__init__(self, *args)

        phase = tf.exp(1j * np.pi * self.theta / 2.0)
        cos = tf.cast(tf.math.real(phase), dtype=self.dtype)
        sin = tf.cast(tf.math.imag(phase), dtype=self.dtype)


class RZ(TensorflowGate, base_gates.RZ):

    def __init__(self, *args):
        base_gates.RZ.__init__(self, *args)

        phase = tf.exp(1j * np.pi * self.theta)
        rz = tf.eye(2, dtype=self.dtype)
        self.matrix = tf.tensor_scatter_nd_update(rz, [[1, 1]], [phase])


class CNOT(TensorflowGate, base_gates.CNOT):

    def __init__(self, *args):
        base_gates.CNOT.__init__(self, *args)


class SWAP(TensorflowGate, base_gates.SWAP):

    def __init__(self, *args):
        base_gates.SWAP.__init__(self, *args)


class CRZ(TensorflowGate, base_gates.CRZ):

    def __init__(self, *args):
        base_gates.CRZ.__init__(self, *args)

        phase = tf.exp(1j * np.pi * self.theta)
        crz = tf.eye(4, dtype=self.dtype)
        crz = tf.tensor_scatter_nd_update(crz, [[3, 3]], [phase])
        self.matrix = tf.reshape(crz, 4 * (2,))


class Toffoli(TensorflowGate, base_gates.Toffoli):

    def __init__(self, *args):
        base_gates.Toffoli.__init__(self, *args)


class Flatten(TensorflowGate, base_gates.Flatten):

    def __init__(self, *args):
        base_gates.Flatten.__init__(self, *args)

    def __call__(self, state):
        if self.nqubits is None:
            self.nqubits = len(tuple(state.shape))

        if len(self.coefficients) != 2 ** self.nqubits:
                raise ValueError(
                    "Circuit was created with {} qubits but the "
                    "flatten layer state has {} coefficients."
                    "".format(self.nqubits, self.coefficients)
                )

        _state = np.array(self.coefficients).reshape(self.nqubits * (2,))
        return tf.convert_to_tensor(_state, dtype=state.dtype)