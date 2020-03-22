# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import collections
from typing import Any, Optional, Dict, List, Tuple, Set
TensorType = Any


class GateResult:
    """Object returned when user uses a `gates.M` on a state.

    Implements tools to convert samples from decimal to binary representation
    (and vice versa) and calculating the frequencies of shots.

    Args:
        qubits: Sorted tuple of qubit ids that the measurement gate acts on.
        decimal_samples: Tensor holding the measured samples in decimal
            representation. Has shape (nshots,).
        binary_samples: Tensor holding the measured samples in binary
            representation. Has shape (nshots, len(qubits)).
        Exactly one of `decimal_samples`, `binary_samples` should be given to
        create the object.
    """

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
        """Calculates frequencies of appearance of each measurement.

        Args:
            binary: If True the returned keys (measurement results) are in
                binary representation. Otherwise they are in decimal
                representation.

        Returns:
            A `collections.Counter` where the keys are the measurement results
            and the values are the number of appearances of each result in the
            measured shots.
        """
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
    """Object returned when user performs measurements using a circuit.

    Implements tools for dividing the global measurements from the circuit's
    `measurement_gate` to the corresponding registers.

    Args:
        register_qubits: Dictionary that maps register names to the
            corresponding sets of qubit ids. This is created in the
            `measurement_sets` variable of `models.Circuit`.
        measurement_gate_result: The `GateResult` resulting from the circuit's
            global measurement gate.
    """

    def __init__(self,
                 register_qubits: Dict[str, Set[int]],
                 measurement_gate_result: GateResult):
        self.register_qubits = register_qubits
        self.result = measurement_gate_result
        self._register_results = None

    @property
    def register_results(self) -> Dict[str, GateResult]:
        """Returns the individual `GateResult`s for each register."""
        if self._register_results is None:
            self._register_results = self._calculate_register_results(
                self.register_qubits, self.result)
        return self._register_results

    @property
    def binary(self) -> TensorType:
        return self.result.binary

    @property
    def decimal(self) -> TensorType:
        return self.result.decimal

    def frequencies(self, binary: bool = False) -> collections.Counter:
        return self.result.frequencies(binary)

    @property
    def register_binary(self) -> Dict[str, TensorType]:
        return {k: v.binary for k, v in self.register_results.items()}

    @property
    def register_decimal(self) -> Dict[str, TensorType]:
        return {k: v.decimal for k, v in self.register_results.items()}

    def register_frequencies(self, binary: bool = False) -> Dict[str, collections.Counter]:
        return {k: v.frequencies(binary)
                for k, v in self.register_results.items()}

    @staticmethod
    def _calculate_register_results(register_qubits: Dict[str, Set[int]],
                                    gate_result: GateResult
                                    ) -> Dict[str, GateResult]:
        """Calculates the individual register `GateResults`.

        This uses the `register_qubits` map to divide the bitstrings to their
        appropriate registers.
        """
        raise NotImplementedError