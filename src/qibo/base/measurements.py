# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import collections
from typing import Any, Optional, Dict, List, Tuple, Set
TensorType = Any


class GateResult:

    def __init__(self, qubits: Tuple[int],
                 decimal_samples: Optional[TensorType] = None,
                 binary_samples: Optional[TensorType] = None):
        self.qubits = qubits
        if decimal_samples is not None and binary_samples is not None:
            raise ValueError("Measurement result object cannot be created "
                             "when samples are given both in decimal and "
                             "binary. Use one of the two.")
        if binary_samples is not None and binary_samples.shape[-1] != len(qubits):
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
    def binary(self) -> TensorType:
        if self._binary is None:
            self._binary = self._convert_to_binary(self._decimal, self.nqubits)
        return self._binary

    @property
    def decimal(self) -> TensorType:
        if self._decimal is None:
            self._decimal = self._convert_to_decimal(self._binary, self.nqubits)
        return self._decimal

    def frequencies(self, binary: bool = False) -> collections.Counter:
        if self._frequencies is None:
            res, cnts = self._calculate_counts(self.decimal)
            self._frequencies = collections.Counter(
                {k: v for k, v in zip(res, cnts)})

        if binary:
            return collections.Counter(
                {"0:b".format(k).zfill(self.nqubits): v
                 for k, v in self._frequencies.items()})
        return self._frequencies

    @staticmethod
    def _convert_to_binary(x: TensorType, n: int) -> TensorType:
        raise NotImplementedError

    @staticmethod
    def _convert_to_decimal(x: TensorType, n: int) -> TensorType:
        raise NotImplementedError

    @staticmethod
    def _calculate_counts(decimal_samples: TensorType) -> Tuple[List[int]]:
        raise NotImplementedError


class CircuitResult:

    def __init__(self,
                 register_qubits: Dict[str, Set[int]],
                 measurement_gate_result: GateResult):
        self.register_qubits = register_qubits
        self.result = measurement_gate_result
        self._register_results = None

    @property
    def register_results(self) -> Dict[str, GateResult]:
        if self._register_results is None:
            self._register_results = self._calculate_register_results(
                self.register_qubits, self.result)
        return self._register_results

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

    @staticmethod
    def _calculate_register_results(register_qubits: Dict[str, Set[int]],
                                    gate_result: GateResult
                                    ) -> Dict[str, GateResult]:
        raise NotImplementedError