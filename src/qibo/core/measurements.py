# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import math
import collections
import sympy
from qibo import K
from qibo.config import raise_error, SHOT_BATCH_SIZE
from qibo.abstractions.gates import M
from typing import Any, Optional, Dict, List, Set, Tuple, Union
TensorType = Any
ProbsType = Union[float, List[float], Dict[int, float]]


class MeasurementResult:
    """Holds measurement results (shot samples and frequencies).

    Implements tools for calculating the frequencies and shot samples from
    a probability distribution and converting samples from decimal to binary
    representation and vice versa.

    If ``probabilities`` and ``nshots`` are not given during this object's
    initialization they should be specified later using
    :meth:`qibo.core.measurements.MeasurementResult.set_probabilities`.
    Alternatively binary or decimal samples can be added directly using the
    corresponding setters.

    Args:
        qubits (tuple): Sorted tuple of qubit ids that are measured.
        probabilities (Tensor): Tensor of probabilities to use for sampling
            measurements.
        nshots (int): Number of shots for the measurement.
    """

    def __init__(self, qubits, probabilities=None, nshots=0):
        self.qubits = tuple(qubits)
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

    def set_probabilities(self, probabilities, nshots=1):
        """Sets the probability distribution and number of shots for measurements.

        Calling this resets existing previous samples.

        Args:
            probabilities (Tensor): Tensor of size ``(2 ** len(qubits),)``
                that defines the probability distribution to sample measurements
                from.
            nshots (int): Number of measurement shots to sample.
        """
        self.reset()
        self.probabilities = probabilities
        self.nshots = nshots

    def add_shot(self, probabilities=None):
        """Adds a measurement shot to an existing measurement symbol.

        Useful for sampling more than one shots with collapse measurement gates.
        """
        if self.nshots:
            if probabilities is not None:
                self.probabilities = probabilities
            self.nshots += 1
            # sample new shot
            new_shot = K.cpu_fallback(K.sample_shots, self.probabilities, 1)
            self._decimal = K.concatenate([self.decimal, new_shot], axis=0)
            self._binary = None
        else:
            if probabilities is None:
                raise_error(ValueError, "Cannot add shots in measurement that "
                                        "for which the probability distribution "
                                        "is not specified.")
            self.set_probabilities(probabilities)

    def set_frequencies(self, frequencies):
        if self.has_samples():
            raise_error(RuntimeError, "Cannot set frequencies for measurement "
                                      "result that contains shots.")
        self._frequencies = frequencies

    def reset(self):
        """Resets the sampled shots contained in the ``MeasurementResult`` object."""
        self._decimal = None
        self._binary = None
        self._frequencies = None

    @property
    def decimal(self):
        """Returns sampled measurement shots in decimal form."""
        if self._decimal is None:
            if self._binary is None:
                self._decimal = self._sample_shots()
            else:
                self._decimal = self._convert_to_decimal()
        return self._decimal

    @property
    def binary(self):
        """Returns sampled measurement shots in binary form."""
        if self._binary is None:
            self._binary = self._convert_to_binary()
        return self._binary

    @decimal.setter
    def decimal(self, x):
        self.reset()
        self._decimal = x

    @binary.setter
    def binary(self, x):
        self.reset()
        self._binary = x

    def outcome(self, q=0):
        """Returns the latest outcome for the selected qubit."""
        if not self.nshots:
            nshots = int(self.binary.shape[0])
        else:
            nshots = self.nshots
        return self.binary[-1, q]

    def has_samples(self):
        """Checks if the measurement result has samples calculated."""
        return self._binary is not None or self._decimal is not None

    def samples(self, binary: bool = True) -> TensorType:
        if binary:
            return self.binary
        return self.decimal

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
            self._frequencies = self._calculate_frequencies()
        if binary:
            return collections.Counter(
                {"{0:b}".format(k).zfill(self.nqubits): v
                 for k, v in self._frequencies.items()})
        return self._frequencies

    def __getitem__(self, i: int) -> TensorType:
        return self.decimal[i]

    def _convert_to_binary(self):
        _range = K.range(self.nqubits - 1, -1, -1, dtype=K.dtypes('DTYPEINT'))
        return K.mod(K.right_shift(self.decimal[:, K.newaxis], _range), 2)

    def _convert_to_decimal(self):
        _range = K.range(self.nqubits - 1, -1, -1, dtype=K.dtypes('DTYPEINT'))
        _range = K.pow(2, _range)[:, K.newaxis]
        return K.matmul(self.binary, _range)[:, 0]

    def _sample_shots(self):
        self._frequencies = None
        if self.probabilities is None or not self.nshots:
            raise_error(RuntimeError, "Cannot sample measurement shots if "
                                      "a probability distribution is not "
                                      "provided.")
        if math.log2(self.nshots) + self.nqubits > 31: # pragma: no cover
            # case not covered by GitHub workflows because it requires large example
            # Use CPU to avoid "aborted" error
            with K.device(K.get_cpu()):
                result = K.sample_shots(self.probabilities, self.nshots)
        else:
            result = K.cpu_fallback(K.sample_shots, self.probabilities, self.nshots)
        return result

    def _calculate_frequencies(self):
        if self._binary is None and self._decimal is None:
            if self.probabilities is None or not self.nshots:
                raise_error(RuntimeError, "Cannot calculate measurement "
                                          "frequencies without a probability "
                                          "distribution or  samples.")
            freqs = K.sample_frequencies(self.probabilities, self.nshots)
            freqs = K.np.array(freqs)
            return collections.Counter(
                {k: v for k, v in enumerate(freqs) if v > 0})

        res, counts = K.unique(self.decimal, return_counts=True)
        res, counts = K.np.array(res), K.np.array(counts)
        return collections.Counter({k: v for k, v in zip(res, counts)})

    def apply_bitflips(self, p0: ProbsType, p1: Optional[ProbsType] = None):
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
            A new :class:`qibo.core.measurements.MeasurementResult` object that
            holds the noisy samples.
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


class MeasurementSymbol(sympy.Symbol):
    """``sympy.Symbol`` connected to a specific :class:`qibo.core.measurements.MeasurementResult`.

    Used by :class:`qibo.abstractions.gates.M` with ``collapse=True`` to allow
    controlling subsequent gates from the measurement results.
    """
    _counter = 0

    def __new__(cls, *args, **kwargs):
        name = "m{}".format(cls._counter)
        cls._counter += 1
        return super().__new__(cls=cls, name=name)

    def __init__(self, measurement_result, qubit=None):
        self.result = measurement_result
        self.qubit = qubit
        # create seperate ``MeasurementSymbol`` object that maps to the same
        # result for each measured qubit so that the user can use the symbol
        # to control subsequent parametrized gates
        if qubit is None:
            self.elements = [self.__class__(self.result, q)
                             for q in self.result.qubits]

    def samples(self, *args, **kwargs):
        return self.result.samples(*args, **kwargs)

    def frequencies(self, *args, **kwargs):
        return self.result.frequencies(*args, **kwargs)

    def outcome(self):
        if self.qubit is None:
            return self.result.outcome()
        return self.result.outcome(self.qubit)

    def __getitem__(self, i):
        return self.elements[i]

    def evaluate(self, expr):
        """Substitutes the symbol's value in the given expression.

        Args:
            expr (sympy.Expr): Sympy expression that involves the current
                measurement symbol.
        """
        if self.qubit is None and len(self.result.qubits) > 1:
            raise_error(NotImplementedError, "Symbolic measurements are not "
                                             "available for more than one "
                                             "measured qubits. Please use "
                                             "seperate measurement gates.")
        return expr.subs(self, self.outcome())


class MeasurementRegistersResult:
    """Holds measurement results grouped according to register.

    Divides a :class:`qibo.core.measurements.MeasurementResult` to multiple
    registers for easier access of the results by the user.

    Args:
        register_qubits (dict): Dictionary that maps register names to the
            corresponding tuples of qubit ids.
            For :class:`qibo.abstractions.circuit.AbstractCircuit` models, this
            dictionary is held at the `measurement_tuples` attribute.
        measurement_result (:class:`qibo.core.measurements.MeasurementResult`):
            The measurement object to split to registers.
    """

    def __init__(self, register_qubits: Dict[str, Tuple[int]],
                 measurement_result: MeasurementResult):
        self.register_qubits = register_qubits
        self.result = measurement_result
        self._samples = None
        self._frequencies = None

    def _calculate_register_frequencies(self):
        if self.result.has_samples():
            return self._calculate_register_samples()

        qubit_map = self.result.qubit_map
        frequencies = self.result.frequencies(True)
        results = {}
        for name, qubit_tuple in self.register_qubits.items():
            register_freqs = collections.Counter()
            for bitstring, freq in frequencies.items():
                idx = 0
                for i, q in enumerate(qubit_tuple):
                    if int(bitstring[qubit_map[q]]):
                        idx +=  2 ** (len(qubit_tuple) - i - 1)
                register_freqs[idx] += freq
            results[name] = MeasurementResult(qubit_tuple)
            results[name].set_frequencies(register_freqs)
        return results

    def _calculate_register_samples(self):
        qubit_map = self.result.qubit_map
        samples = self.result.samples(True)
        results = {}
        for name, qubit_tuple in self.register_qubits.items():
            slicer = tuple(qubit_map[q] for q in qubit_tuple)
            register_samples = K.gather(samples, slicer, axis=-1)
            results[name] = MeasurementResult(qubit_tuple)
            results[name].binary = register_samples
        return results

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
        if self._samples is None:
            self._frequencies = None
            self._samples = self._calculate_register_samples()
        return {k: v.samples(binary) for k, v in self._samples.items()}

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
        if self._frequencies is None:
            self._frequencies = self._calculate_register_frequencies()
        return {k: v.frequencies(binary) for k, v in self._frequencies.items()}

    def apply_bitflips(self, p0: ProbsType, p1: Optional[ProbsType] = None):
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
            A new :class:`qibo.core.measurements.MeasurementRegisterResult`
            object that holds the noisy samples.
        """
        noisy_result = self.result.apply_bitflips(p0, p1)
        return self.__class__(self.register_qubits, noisy_result)
