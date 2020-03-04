# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import gates as base_gates
from typing import List


class TensorflowGate:
    """The base Tensorflow gate."""
    from qibo.config import DTYPECPX as dtype

    def _base_slicer(self, targets: np.ndarray) -> np.ndarray:
        nbits = self.nqubits - len(targets)
        configs = np.arange(2**nbits, dtype=np.int32)

        for target in targets:
            mask = (1 << target) - 1
            not_mask = (1 << nbits) - 1 - mask
            configs = (configs & mask) | ((configs & not_mask) << 1)
            nbits += 1

        return configs

    def call_0(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def call_1(self, state0: tf.Tensor, state1: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def _get_slices(self) -> List[np.ndarray]:
        qubits = self.nqubits - np.array(self.qubits) - 1
        slicer = self._base_slicer(qubits)

        target = 2 ** qubits[-1]
        control = (2 ** qubits[:-1]).sum()

        return [slicer + control, slicer + control + target]

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        """Implements the `Gate` on a given state."""
        # Set nqubits for the case the gate is called outside of the circuit
        if self._nqubits is None:
            self.nqubits = int(np.log2(state.shape[0]))

        slices = self._get_slices()
        states = [tf.gather(state, s) for s in slices]

        new0 = tf.IndexedSlices(self.call_0(states[0], states[1]),
                                slices[0], [self.nstates])
        new1 = tf.IndexedSlices(self.call_1(states[0], states[1]),
                                slices[1], [self.nstates])
        new_state = tf.add(new0, new1)

        if self.control_qubits or len(self.target_qubits) > 1:
            all_indices = np.concatenate(slices)
            mask = tf.constant(len(all_indices) * [1.0], dtype=self.dtype)
            mask = tf.convert_to_tensor(
                tf.IndexedSlices(mask, all_indices, [self.nstates]))
            return state * (1.0 - mask) + new_state

        return new_state


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


class CNOT(X, base_gates.CNOT):

    def __init__(self, *args):
        base_gates.CNOT.__init__(self, *args)


class SWAP(X, base_gates.SWAP):

    def __init__(self, *args):
        base_gates.SWAP.__init__(self, *args)

    def _get_slices(self) -> List[np.ndarray]:
        qubits = self.nqubits - np.array(self.qubits) - 1
        slicer = self._base_slicer(qubits)

        targets = 2 ** qubits[-2:]
        control = (2 ** qubits[:-2]).sum()

        return [slicer + control + targets[0], slicer + control + targets[1]]


class CRZ(RZ, base_gates.CRZ):

    def __init__(self, *args):
        base_gates.CRZ.__init__(self, *args)
        self.phase = tf.exp(1j * np.pi * self.theta)


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

        return tf.convert_to_tensor(self.coefficients, dtype=state.dtype)
