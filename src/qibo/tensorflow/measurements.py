# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import collections
import tensorflow as tf
from typing import Optional, Tuple


class GateResult:

    def __init__(self, nqubits: int,
                 decimal_samples: Optional[tf.Tensor] = None,
                 binary_samples: Optional[tf.Tensor] = None):
        self.qubits = nqubits
        if decimal_samples is not None and binary_samples is not None:
            raise ValueError("Measurement result object cannot be created "
                             "when samples are given both in decimal and "
                             "binary. Use one of the two.")
        if binary_samples is not None and binary_samples.shape[-1] != nqubits:
            raise ValueError("Binary samples are for {} qubits but the given "
                             "number of qubits is {}."
                             "".format(binary_samples.shape[-1], nqubits))

        self._decimal_samples = decimal_samples
        self._binary_samples = binary_samples

        self._frequencies = None

    @property
    def binary_samples(self) -> tf.Tensor:
        if self._binary_samples is None:
            _range = tf.range(self.nqubits, dtype=tf.int64)
            self._binary_samples = tf.math.mod(tf.bitwise.right_shift(
                self.decimal_samples[:, tf.newaxis], _range), 2)
        return self._binary_samples

    @property
    def decimal_samples(self) -> tf.Tensor:
        if self._decimal_samples is None:
            _range = tf.range(self.nqubits - 1, -1, -1, dtype=tf.int64)
            _range = tf.math(2, _range)[:, tf.newaxis]
            self._decimal_samples = tf.matmul(self._binary_samples,
                                              _range)[0]
        return self._decimal_samples

    def frequencies(self, binary: bool = False) -> collections.Counter:
        if self._frequencies is None:
            results, counts = np.unique(self.decimal_samples.numpy(),
                                        return_counts=True)
            self._frequencies = collections.Counter(
                {k: v for k, v in zip(results, counts)})

        if binary:
            return collections.Counter({{"0:b".format(k).zfill(self.nqubits): v
                                         for k, v in self._frequencies.items()}})
        return self._frequencies


#class CircuitResult:

    #def __init__(self, register_sets: List[Set[int]], result: GateResult):
