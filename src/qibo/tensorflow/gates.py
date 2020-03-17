# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import gates as base_gates
from typing import List


class TensorflowGate:
    """The base Tensorflow gate."""
    from qibo.config import DTYPECPX as dtype
    _slice = None

    def _base_slicer(self, targets: np.ndarray) -> np.ndarray:
        nbits = self.nqubits - len(targets)
        configs = np.arange(2**nbits, dtype=np.int32)

        for target in targets:
            mask = (1 << target) - 1
            not_mask = (1 << nbits) - 1 - mask
            configs = (configs & mask) | ((configs & not_mask) << 1)
            nbits += 1

        return configs

    def _calculate_slice(self) -> np.ndarray:
        qubits = self.nqubits - np.array(self.qubits) - 1
        slicer = self._base_slicer(qubits)

        target = 2 ** qubits[-1]
        control = (2 ** qubits[:-1]).sum()

        return np.concatenate([slicer + control, slicer + control + target])

    def call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    @property
    def slice(self) -> np.ndarray:
        if self._slice is None:
            self._slice = self._calculate_slice()
        return self._slice

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        """Implements the `Gate` on a given state."""
        # Set nqubits for the case the gate is called outside of the circuit
        if self._nqubits is None:
            self.nqubits = int(np.log2(state.shape[0]))
        if not isinstance(state, tf.Variable):
            raise TypeError("Gate {} called with a state that is not "
                            "tf.Variable.".format(self.name))

        n = len(self.slice) // 2
        # this implementation is slightly faster than
        # states = state.sparse_read(self.slice)
        # self.call_0(states[:n], states[n:])
        # self.call_1(states[:n], states[n:])
        # perhaps because slicing is better in numpy
        states = [state.sparse_read(self.slice[:n]),
                  state.sparse_read(self.slice[n:])]
        updates = tf.IndexedSlices(tf.concat([self.call_0(states[0], states[1]),
                                              self.call_1(states[0], states[1])],
                                             axis=0),
                                   self.slice,
                                   dense_shape=(2 ** self.nqubits,))
        state.scatter_update(updates)
        return state


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

    def update(self, theta):
        self.theta = tf.cast(theta, dtype=self.dtype)
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

    def update(self, theta):
        self.theta = tf.cast(theta, dtype=self.dtype)
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

    def update(self, theta):
        self.theta = tf.cast(theta, dtype=self.dtype)
        self.phase = tf.exp(1j * np.pi * self.theta)

    def call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return state0

    def call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        return self.phase * state1


class CNOT(X, base_gates.CNOT):

    def __init__(self, *args):
        base_gates.CNOT.__init__(self, *args)


class SWAP(X, base_gates.SWAP):

    def __init__(self, *args):
        base_gates.SWAP.__init__(self, *args)

    def _calculate_slice(self) -> np.ndarray:
        qubits = self.nqubits - np.array(self.qubits) - 1
        qubits.sort()
        slicer = self._base_slicer(qubits)

        targets = 2 ** qubits[-2:]
        control = (2 ** qubits[:-2]).sum()

        return np.array([slicer + control + targets[0],
                         slicer + control + targets[1]])


class CRZ(RZ, base_gates.CRZ):

    def __init__(self, *args):
        base_gates.CRZ.__init__(self, *args)

    def update(self, theta):
        self.theta = tf.cast(theta, dtype=self.dtype)
        self.phase = tf.cast(tf.exp(1j * np.pi * self.theta), dtype=self.dtype)


class Toffoli(X, base_gates.Toffoli):

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

        return tf.Variable(self.coefficients, dtype=state.dtype)
