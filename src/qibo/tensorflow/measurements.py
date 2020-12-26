# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
from qibo import K
from qibo.base import measurements as base_measurements


class GateResult(base_measurements.GateResult):

    @staticmethod
    def _convert_to_binary(x, n):
        _range = K.range(n - 1, -1, -1, dtype=x.dtype)
        return K.mod(K.right_shift(x[:, K.newaxis], _range), 2)

    @staticmethod
    def _convert_to_decimal(x, n):
        _range = K.range(n - 1, -1, -1, dtype=x.dtype)
        _range = K.pow(2, _range)[:, K.newaxis]
        return K.matmul(x, _range)[:, 0]

    @staticmethod
    def _calculate_counts(decimal_samples):
        return K.unique(decimal_samples, return_counts=True)

    @staticmethod
    def _apply_bitflips(noiseless_samples, probs):
        fprobs = K.cast(probs, dtype='DTYPE')
        sprobs = K.random.uniform(noiseless_samples.shape,
                                  dtype=K.dtypes('DTYPE'))
        flip0 = K.cast(sprobs < fprobs[0], dtype=noiseless_samples.dtype)
        flip1 = K.cast(sprobs < fprobs[1], dtype=noiseless_samples.dtype)
        noisy_samples = noiseless_samples + (1 - noiseless_samples) * flip0
        noisy_samples = noisy_samples - noiseless_samples * flip1
        return noisy_samples


class CircuitResult(base_measurements.CircuitResult):

    @staticmethod
    def _calculate_register_results(register_qubits, gate_result):
        results = {}
        for name, qubit_tuple in register_qubits.items():
            slicer = tuple(gate_result.qubit_map[q] for q in qubit_tuple)
            samples = K.gather(gate_result.samples(True), slicer, axis=-1)
            results[name] = GateResult(qubit_tuple, binary_samples=samples)
        return results
