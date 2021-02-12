# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import math
import collections
from qibo import K
from qibo.config import raise_error
from qibo.abstractions.gates import M
from typing import Any, Optional, Dict, List, Set, Tuple, Union
TensorType = Any
ProbsType = Union[float, List[float], Dict[int, float]]


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

    def __init__(self, qubits, probabilities=None, nshots=None):
        self.qubits = qubits
        self.probabilities = probabilities
        self.nshots = nshots
        self._decimal = None
        self._binary = None
        self._frequencies = None

    @property
    def nqubits(self) -> int:
        return len(self.qubits)

    @property
    def qubit_map(self) -> Dict[int, int]:
        return {q: i for i, q in enumerate(self.qubits)}

    def _convert_to_binary(self):
        _range = K.range(self.nqubits - 1, -1, -1, dtype=self.decimal.dtype)
        return K.mod(K.right_shift(self.decimal[:, K.newaxis], _range), 2)

    def _convert_to_decimal(self):
        _range = K.range(self.nqubits - 1, -1, -1, dtype=self.binary.dtype)
        _range = K.pow(2, _range)[:, K.newaxis]
        return K.matmul(self.binary, _range)[:, 0]

    def _get_cpu(self): # pragma: no cover
        # case not covered by GitHub workflows because it requires OOM
        if not K.cpu_devices:
            raise_error(RuntimeError, "Cannot find CPU device to use for sampling.")
        return K.cpu_devices[0]

    def _sample_shots(self):
        if math.log2(self.nshots) + self.nqubits > 31: # pragma: no cover
            # case not covered by GitHub workflows because it requires large example
            # Use CPU to avoid "aborted" error
            with K.device(self._get_cpu()):
                result = K.sample_shots(self.probabilities, self.nshots)
        else:
            try:
                with K.device(K.default_device):
                    result = K.sample_shots(self.probabilities, self.nshots)
            except K.oom_error: # pragma: no cover
                # case not covered by GitHub workflows because it requires OOM
                # Force using CPU to perform sampling
                with K.device(self._get_cpu()):
                    result = K.sample_shots(self.probabilities, self.nshots)
        return result

    @property
    def decimal(self):
        if self._decimal is None:
            if self._binary is None:
                self._decimal = self._sample_shots()
            else:
                self._decimal = self._convert_to_decimal()
        return self._decimal

    @property
    def binary(self):
        if self._binary is None:
            self._binary = self._convert_to_binary()
        return self._binary

    @decimal.setter
    def decimal(self, x):
        self._decimal = x
        self._binary = None
        self._frequencies = None

    @binary.setter
    def binary(self, x):
        self._binary = x
        self._decimal = None
        self._frequencies = None

    def samples(self, binary: bool = True) -> TensorType:
        if binary:
            return self.binary
        return self.decimal

    def __getitem__(self, i: int) -> TensorType:
        return self.samples(binary=False)[i]

    def frequencies(self, binary: bool = True) -> collections.Counter:
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
            dsamples = self.samples(binary=False)
            res, cnts = K.unique(dsamples, return_counts=True)
            self._frequencies = collections.Counter(
                {k: v for k, v in zip(res, cnts)})
        if binary:
            return collections.Counter(
                {"{0:b}".format(k).zfill(self.nqubits): v
                 for k, v in self._frequencies.items()})
        return self._frequencies

    def apply_bitflips(self, p0: ProbsType, p1: Optional[ProbsType] = None
                       ) -> "GateResult":
        """Applies bitflip noise to the measured samples.

        Args:
            p0: Bitflip probability map. Can be:
                A dictionary that maps each measured qubit to the probability
                that it is flipped, a list or tuple that has the same length
                as the tuple of measured qubits or a single float number.
                If a single float is given the same probability will be used
                for all qubits.
            p1: Probability of asymmetric bitflip. If ``p1`` is given, ``p0``
                will be used as the probability for 0->1 and ``p1`` as the
                probability for 1->0. If ``p1`` is ``None`` the same probability
                ``p0`` will be used for both bitflips.

        Returns:
            A new :class:`qibo.core.measurements.GateResult` object that holds
            the noisy samples.
        """
        if p1 is None:
            probs = 2 * (M._get_bitflip_tuple(self.qubits, p0),)
        else:
            probs = (M._get_bitflip_tuple(self.qubits, p0),
                     M._get_bitflip_tuple(self.qubits, p1))

        # Calculate noisy samples
        noiseless_samples = self.samples()
        fprobs = K.cast(probs, dtype='DTYPE')
        sprobs = K.random_uniform(noiseless_samples.shape)
        flip0 = K.cast(sprobs < fprobs[0], dtype=noiseless_samples.dtype)
        flip1 = K.cast(sprobs < fprobs[1], dtype=noiseless_samples.dtype)
        noisy_samples = noiseless_samples + (1 - noiseless_samples) * flip0
        noisy_samples = noisy_samples - noiseless_samples * flip1
        noisy_result = self.__class__(self.qubits)
        noisy_result.binary = noisy_samples
        return noisy_result


class CircuitResult:
    """Object returned when user performs measurements using a circuit.

    Implements tools for dividing the global measurements from the circuit's
    `measurement_gate` to the corresponding registers.
    This object is created automatically every time a circuit that contains
    measurement gates is executed. The user does not have to worry about
    creating this object.

    Args:
        register_qubits: Dictionary that maps register names to the
            corresponding tuples of qubit ids. This is created in the
            `measurement_tuples` variable of :class:`qibo.abstractions.circuit.AbstractCircuit`.
        measurement_gate_result: The `GateResult` resulting from the circuit's
            global measurement gate.
    """

    def __init__(self,
                 register_qubits: Dict[str, Tuple[int]],
                 measurement_gate_result: GateResult):
        self.register_qubits = register_qubits
        self.result = measurement_gate_result
        self._register_results = None

    @property
    def register_results(self) -> Dict[str, GateResult]:
        """Returns the individual `GateResult`s for each register."""
        if self._register_results is None:
            samples = self.result.samples(True)
            self._register_results = {}
            for name, qubit_tuple in self.register_qubits.items():
                slicer = tuple(self.result.qubit_map[q] for q in qubit_tuple)
                register_samples = K.gather(samples, slicer, axis=-1)
                self._register_results[name] = GateResult(qubit_tuple)
                self._register_results[name].binary = register_samples
        return self._register_results

    def samples(self, binary: bool = True, registers: bool = False
                ) -> Union[TensorType, Dict[str, TensorType]]:
        """Returns raw measurement samples.

        Args:
            binary (bool): Return samples in binary or decimal form.
            registers (bool): Group samples according to registers.

        Returns:
            If `binary` is `True`
                samples are returned in binary form as a tensor
                of shape `(nshots, n_measured_qubits)`.
            If `binary` is `False`
                samples are returned in decimal form as a tensor
                of shape `(nshots,)`.
            If `registers` is `True`
                samples are returned in a `dict` where the keys are the register
                names and the values are the samples tensors for each register.
            If `registers` is `False`
                a single tensor is returned which contains samples from all the
                measured qubits, independently of their registers.
        """
        if not registers:
            return self.result.samples(binary)
        return {k: v.samples(binary) for k, v in self.register_results.items()}

    def frequencies(self, binary: bool = True, registers: bool = False
                    ) -> Union[collections.Counter, Dict[str, collections.Counter]]:
        """Returns the frequencies of measured samples.

        Args:
            binary (bool): Return frequency keys in binary or decimal form.
            registers (bool): Group frequencies according to registers.

        Returns:
            A `collections.Counter` where the keys are the observed values
            and the values the corresponding frequencies, that is the number
            of times each measured value/bitstring appears.

            If `binary` is `True`
                the keys of the `Counter` are in binary form, as strings of
                0s and 1s.
            If `binary` is `False`
                the keys of the `Counter` are integers.
            If `registers` is `True`
                a `dict` of `Counter` s is returned where keys are the name of
                each register.
            If `registers` is `False`
                a single `Counter` is returned which contains samples from all
                the measured qubits, independently of their registers.
        """
        if not registers:
            return self.result.frequencies(binary)
        return {k: v.frequencies(binary) for k, v in self.register_results.items()}

    def apply_bitflips(self, p0: ProbsType, p1: Optional[ProbsType] = None
                       ) -> "CircuitResult":
        """Applies bitflip noise to the measured samples.

        Args:
            p0: Bitflip probability map. Can be:
                A dictionary that maps each measured qubit to the probability
                that it is flipped, a list or tuple that has the same length
                as the tuple of measured qubits or a single float number.
                If a single float is given the same probability will be used
                for all qubits.
            p1: Probability of asymmetric bitflip. If ``p1`` is given, ``p0``
                will be used as the probability for 0->1 and ``p1`` as the
                probability for 1->0. If ``p1`` is ``None`` the same probability
                ``p0`` will be used for both bitflips.

        Returns:
            A new :class:`qibo.core.measurements.CircuitResult` object that
            holds the noisy samples.
        """
        noisy_result = self.result.apply_bitflips(p0, p1)
        return self.__class__(self.register_qubits, noisy_result)
