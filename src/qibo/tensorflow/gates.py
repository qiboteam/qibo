# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import gates as base_gates
from typing import Optional, Tuple


class TensorflowGate:
    """The base Tensorflow gate."""
    from qibo.config import DTYPECPX as dtype

    def _call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def _call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        """Implements the `Gate` on a given state."""
        d = 2**(self.nqubits - self.qubits[0])
        slice = np.array([x + d * y for x in range(d // 2) for y in range(self.nstates // d)])
        slice0 = tuple(slice)
        slice1 = tuple(slice + d // 2)

        state0 = tf.gather(state, slice0)
        state1 = tf.gather(state, slice1)

        new0 = self._call_0(state0, state1)
        new1 = self._call_1(state0, state1)

        slice0 = tf.constant(slice0)[:, tf.newaxis]
        slice1 = tf.constant(slice1)[:, tf.newaxis]

        new = tf.tensor_scatter_nd_update(state, slice0, new0)
        new = tf.tensor_scatter_nd_update(new, slice1, new1)
        return new


class TensorflowControlledGate(TensorflowGate):

    def slice_generator(self, control: int, q: int, is_one: bool = False) -> int:
        q = self.nqubits - q - 1 # because we use "cirq" like order
        control = self.nqubits - control - 1
        s = (q + 1) * int(is_one)
        while s < self.nstates:
            for i in range(s, s + q + 1):
                if control is None:
                    yield i
                elif (i // 2**control) % 2 == 1:
                    yield i
            s += 2 * q + 2

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        """Implements the `Gate` on a given state."""
        #slice0 = tuple(self.slice_generator(self.qubits[0], self.qubits[1], False))
        #slice1 = tuple(self.slice_generator(self.qubits[0], self.qubits[1], True))

        dc = 2 ** (self.nqubits - self.qubits[0])
        dt = 2 ** (self.nqubits - self.qubits[1])
        dmin, dmax = min(dc, dt), min(dc, dt)
        slice = np.array([x + dmin * y + dmax * z
                          for x in range(dmin // 2)
                          for y in range(dmax // dmin)
                          for z in range(self.nstates // dmax)])
        slice0 = tuple(slice + dc // 2)
        slice1 = tuple(slice + (dc + dt) // 2)

        print(slice0)
        print(slice1)

        state0 = tf.gather(state, slice0)
        state1 = tf.gather(state, slice1)

        new0 = self._call_0(state0, state1)
        new1 = self._call_1(state0, state1)

        slice0 = tf.constant(slice0)[:, tf.newaxis]
        slice1 = tf.constant(slice1)[:, tf.newaxis]

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

    def _call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state1

    def _call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state0


class Y(TensorflowGate, base_gates.Y):

    def __init__(self, *args):
        base_gates.Y.__init__(self, *args)

    def _call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return -1j * state1

    def _call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return 1j * state0


class Z(TensorflowGate, base_gates.Z):

    def __init__(self, *args):
        base_gates.Z.__init__(self, *args)

    def _call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state0

    def _call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return -state1


class Barrier(TensorflowGate, base_gates.Barrier):

    def __init__(self):
        raise NotImplementedError


class Iden(TensorflowGate, base_gates.Iden):

    def __init__(self, *args):
        base_gates.Iden.__init__(self, *args)

    def _call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state0

    def _call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
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

    def _call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return self.phase * (self.cos * state0 - 1j * self.sin * state1)

    def _call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return self.phase * (- 1j * self.sin * state0 + self.cos * state1)


class RY(TensorflowGate, base_gates.RY):

    def __init__(self, *args):
        base_gates.RY.__init__(self, *args)

        self.phase = tf.exp(1j * np.pi * self.theta / 2.0)
        self.cos = tf.cast(tf.math.real(self.phase), dtype=self.dtype)
        self.sin = tf.cast(tf.math.imag(self.phase), dtype=self.dtype)

    def _call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return self.phase * (self.cos * state0 - self.sin * state1)

    def _call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return self.phase * (self.sin * state0 + self.cos * state1)


class RZ(TensorflowGate, base_gates.RZ):

    def __init__(self, *args):
        base_gates.RZ.__init__(self, *args)
        self.phase = tf.exp(1j * np.pi * self.theta)

    def _call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state0

    def _call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return self.phase * state1


class CNOT(TensorflowControlledGate, base_gates.CNOT):

    def __init__(self, *args):
        base_gates.CNOT.__init__(self, *args)

    def _call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state1

    def _call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state0


class SWAP(TensorflowGate, base_gates.SWAP):

    def __init__(self, *args):
        base_gates.SWAP.__init__(self, *args)

    def slice_generator(self, control: int, q: int, is_one: bool = False) -> int:
        q = self.nqubits - q - 1 # because we use "cirq" like order
        control = self.nqubits - control - 1
        s = (q + 1) * int(is_one)
        while s < self.nstates:
            for i in range(s, s + q + 1):
                if control is None:
                    yield i
                elif (i // 2**control) % 2 == 1:
                    yield i
            s += 2 * q + 2

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        d1 = 2 ** (self.nqubits - self.qubits[0] - 1)
        d2 = 2 ** (self.nqubits - self.qubits[1] - 1)
        ind01, ind10 = [], []
        for i in range(self.nstates):
            b1 = (i // d1) % 2
            b2 = (i // d2) % 2
            if not b1 and b2:
                ind01.append(i)
            elif b1 and not b2:
                ind10.append(i)


class CRZ(TensorflowControlledGate, base_gates.CRZ):

    def __init__(self, *args):
        base_gates.CRZ.__init__(self, *args)
        self.phase = tf.exp(1j * np.pi * self.theta)

    def _call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state0

    def _call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return self.phase * state1


class Toffoli(TensorflowGate, base_gates.Toffoli):

    def __init__(self, *args):
        base_gates.Toffoli.__init__(self, *args)


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