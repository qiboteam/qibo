# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import collections
import tensorflow as tf
from typing import Optional, Dict, Tuple, Set


class GateResult:

    def __init__(self, qubits: Tuple[int],
                 decimal_samples: Optional[tf.Tensor] = None,
                 binary_samples: Optional[tf.Tensor] = None):
        self.qubits = qubits
        if decimal_samples is not None and binary_samples is not None:
            raise ValueError("Measurement result object cannot be created "
                             "when samples are given both in decimal and "
                             "binary. Use one of the two.")
        if binary_samples is not None and binary_samples.shape[-1] != nqubits:
            raise ValueError("Binary samples are for {} qubits but the given "
                             "number of qubits is {}."
                             "".format(binary_samples.shape[-1], nqubits))

        self._decimal = decimal_samples
        self._binary = binary_samples
        self._frequencies = None

    @property
    def nqubits(self) -> int:
        return len(self.qubits)

    @property
    def qubit_map(self) -> Dict[int, int]:
        return {q: i for i, q in enumerate(self.qubits)}

    @property
    def binary(self) -> tf.Tensor:
        if self._binary is None:
            _range = tf.range(self.nqubits, dtype=tf.int64)
            self._binary = tf.math.mod(tf.bitwise.right_shift(
                self._decimal[:, tf.newaxis], _range), 2)
        return self._binary

    @property
    def decimal(self) -> tf.Tensor:
        if self._decimal is None:
            _range = tf.range(self.nqubits - 1, -1, -1, dtype=tf.int64)
            _range = tf.math(2, _range)[:, tf.newaxis]
            self._decimal = tf.matmul(self._binary, _range)[0]
        return self._decimal

    def frequencies(self, binary: bool = False) -> collections.Counter:
        if self._frequencies is None:
            res, cnts = np.unique(self.decimal.numpy(), return_counts=True)
            self._frequencies = collections.Counter(
                {k: v for k, v in zip(res, cnts)})

        if binary:
            return collections.Counter({{"0:b".format(k).zfill(self.nqubits): v
                                         for k, v in self._frequencies.items()}})
        return self._frequencies


class CircuitResult:

    def __init__(self,
                 register_qubits: Dict[str, Set[int]],
                 all_measured_qubits: Tuple[int],
                 all_decimal_samples: tf.Tensor):
        self.result = GateResult(all_measured_qubits,
                                 decimal_samples=all_decimal_samples)

        self.register_qubits = register_qubits
        self._register_results = None

    @property
    def register_results(self) -> Dict[str, GateResult]:
        if self._register_results is not None:
            return self._register_results

        self._register_results = {}
        for name, qubit_set in self.register_qubits.items():
            qubit_tuple = tuple(sorted(qubit_set))
            slicer = tuple(self.result.qubit_map[q] for q in qubit_tuple)
            samples = self.result.binary_samples[:, slicer]
            self._register_results[name] = GateResult(
                qubit_tuple, binary_samples=samples)

    @property
    def binary(self):
        return self.result.binary

    @property
    def register_binary(self, all: bool = True):
        return {k: v.binary for k, v in self.register_results.items()}

    @property
    def decimal(self):
        return self.result.decimal

    @property
    def register_decimal(self):
        return {k: v.decimal for k, v in self.register_results.items()}

    def frequencies(self, binary: bool = False):
        return self.result.frequencies(binary)

    def register_frequencies(self, binary: bool = False):
        return {k: v.frequencies(binary)
                for k, v in self.register_results.items()}