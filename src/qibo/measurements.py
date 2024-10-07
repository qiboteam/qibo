import collections

import numpy as np
import sympy

from qibo import gates
from qibo.config import raise_error


def _check_backend(backend):
    """This is only needed due to the circular import with qibo.backends."""
    from qibo.backends import _check_backend

    return _check_backend(backend)


def frequencies_to_binary(frequencies, nqubits):
    return collections.Counter(
        {"{:b}".format(k).zfill(nqubits): v for k, v in frequencies.items()}
    )


def apply_bitflips(result, p0, p1=None):
    gate = result.measurement_gate
    if p1 is None:
        probs = 2 * (gate._get_bitflip_tuple(gate.qubits, p0),)
    else:
        probs = (
            gate._get_bitflip_tuple(gate.qubits, p0),
            gate._get_bitflip_tuple(gate.qubits, p1),
        )
    noiseless_samples = result.samples()
    return result.backend.apply_bitflips(noiseless_samples, probs)


class MeasurementSymbol(sympy.Symbol):
    """``sympy.Symbol`` connected to measurement results.

    Used by :class:`qibo.gates.measurements.M` with ``collapse=True`` to allow
    controlling subsequent gates from the measurement results.
    """

    _counter = 0

    def __new__(cls, *args, **kwargs):
        name = f"m{cls._counter}"
        cls._counter += 1
        return super().__new__(cls=cls, name=name)

    def __init__(self, index, result):
        self.index = index
        self.result = result

    def __getstate__(self):
        return {"index": self.index, "result": self.result, "name": self.name}

    def __setstate__(self, data):
        self.index = data.get("index")
        self.result = data.get("result")
        self.name = data.get("name")

    def outcome(self):
        return self.result.samples(binary=True)[-1][self.index]

    def evaluate(self, expr):
        """Substitutes the symbol's value in the given expression.

        Args:
            expr (sympy.Expr): Sympy expression that involves the current
                measurement symbol.
        """
        return expr.subs(self, self.outcome())


class MeasurementResult:
    """Data structure for holding measurement outcomes.

    :class:`qibo.measurements.MeasurementResult` objects can be obtained
    when adding measurement gates to a circuit.

    Args:
        gate (:class:`qibo.gates.M`): Measurement gate associated with
            this result object.
        nshots (int): Number of measurement shots.
        backend (:class:`qibo.backends.abstract.AbstractBackend`): Backend
            to use for calculations.
    """

    def __init__(self, qubits):
        self.target_qubits = qubits
        self.circuit = None

        self._samples = None
        self._frequencies = None
        self._bitflip_p0 = None
        self._bitflip_p1 = None
        self._symbols = None

    def __repr__(self):
        qubits = self.target_qubits
        nshots = self.nshots
        return f"MeasurementResult(qubits={qubits}, nshots={nshots})"

    @property
    def raw(self) -> dict:
        samples = self._samples.tolist() if self.has_samples() else self._samples
        return {"samples": samples}

    @property
    def nshots(self) -> int:
        if self.has_samples():
            return len(self._samples)
        elif self._frequencies is not None:
            return sum(self._frequencies.values())

    def add_shot(self, probs, backend=None):
        backend = _check_backend(backend)
        qubits = sorted(self.target_qubits)
        shot = backend.sample_shots(probs, 1)
        bshot = backend.samples_to_binary(shot, len(qubits))
        if self._samples:
            self._samples.append(bshot[0])
        else:
            self._samples = [bshot[0]]
        return shot

    def add_shot_from_sample(self, sample):
        if self._samples:
            self._samples.append(sample)
        else:
            self._samples = [sample]

    def has_samples(self):
        return self._samples is not None

    def register_samples(self, samples):
        """Register samples array to the ``MeasurementResult`` object."""
        self._samples = samples

    def register_frequencies(self, frequencies):
        """Register frequencies to the ``MeasurementResult`` object."""
        self._frequencies = frequencies

    def reset(self):
        """Remove all registered samples and frequencies."""
        self._samples = None
        self._frequencies = None

    @property
    def symbols(self):
        """List of ``sympy.Symbols`` associated with the results of the measurement.

        These symbols are useful for conditioning parametrized gates on measurement outcomes.
        """
        if self._symbols is None:
            qubits = self.target_qubits
            self._symbols = [MeasurementSymbol(i, self) for i in range(len(qubits))]

        return self._symbols

    def samples(self, binary=True, registers=False, backend=None):
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
        """
        backend = _check_backend(backend)
        if self._samples is None:
            if self.circuit is None:
                raise_error(
                    RuntimeError, "Cannot calculate samples if circuit is not provided."
                )
            # calculate samples for the whole circuit so that
            # individual register samples are registered here
            self.circuit.final_state.samples()

        if binary:
            return self._samples

        qubits = self.target_qubits
        return backend.samples_to_decimal(self._samples, len(qubits))

    def frequencies(self, binary=True, registers=False, backend=None):
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
        """
        backend = _check_backend(backend)
        if self._frequencies is None:
            self._frequencies = backend.calculate_frequencies(
                self.samples(binary=False)
            )
        if binary:
            qubits = self.target_qubits
            return frequencies_to_binary(self._frequencies, len(qubits))

        return self._frequencies

    def apply_bitflips(self, p0, p1=None):  # pragma: no cover
        return apply_bitflips(self, p0, p1)
