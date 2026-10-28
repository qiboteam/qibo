from collections import Counter

import sympy

from qibo.config import raise_error


def _check_backend(backend):
    """This is only needed due to the circular import with qibo.backends."""
    from qibo.backends import _check_backend

    return _check_backend(backend)


def frequencies_to_binary(frequencies, nqubits):
    return Counter({f"{k:b}".zfill(nqubits): v for k, v in frequencies.items()})


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
    probs = result.backend.cast(probs, dtype=result.backend.float64)
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

        if self._frequencies is not None:
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

    @classmethod
    def join(cls, *results, backend=None):
        """Joins the outcomes of several measurements into a single result.

        The outcomes of all the measurements are put together shot by shot, in the order in
        which ``results`` are given: the bits of the first result are the leftmost ones, *i.e.*
        the most significant ones, and the bits of the last result are the rightmost ones. This
        is useful to retrieve all the outcomes of circuits with several ``collapse=True``
        measurements, *e.g.* when the same qubit is measured more than once, since these
        measurements are not merged into the result returned by the execution of the circuit.

        Example:

            .. testcode::

                from qibo import Circuit, gates
                from qibo.measurements import MeasurementResult

                # measure the same qubit three times
                circuit = Circuit(1, density_matrix=True)
                results = []
                for _ in range(3):
                    circuit.add(gates.H(0))
                    results.append(circuit.add(gates.M(0, collapse=True)))
                circuit(nshots=100)

                # the outcome of each shot is a bitstring with the three outcomes
                joint_result = MeasurementResult.join(*results)
                print(joint_result.samples(binary=True).shape)
                print(sum(joint_result.frequencies().values()))

            .. testoutput::

                (100, 3)
                100

        Args:
            *results (:class:`qibo.measurements.MeasurementResult`): results to be joined, as
                returned by :meth:`qibo.models.circuit.Circuit.add` when a measurement gate is
                added. The circuit must have been executed, and all the results must have the
                same number of shots.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend used to
                manipulate the samples. If ``None``, it uses the current backend.
                Defaults to ``None``.

        Returns:
            :class:`qibo.measurements.MeasurementResult`: result whose ``target_qubits`` are the
            ones of ``results``, in the same order and repeated if a qubit is measured more than
            once, and whose samples have one bit for each measured qubit.
        """
        if len(results) == 0:
            raise_error(ValueError, "At least one result is needed to join results.")

        if not all(result.has_samples() for result in results):
            raise_error(
                RuntimeError,
                "Cannot join results without samples, the circuit must be executed first.",
            )

        if len({result.nshots for result in results}) != 1:
            raise_error(
                ValueError, "All the results must have the same number of shots."
            )

        backend = _check_backend(backend)
        samples = backend.concatenate(
            [
                backend.cast(result.samples(binary=True), dtype=backend.int64)
                for result in results
            ],
            axis=1,
        )

        joint_result = cls(
            tuple(qubit for result in results for qubit in result.target_qubits)
        )
        joint_result.register_samples(samples)

        return joint_result

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
            self._symbols = [
                MeasurementSymbol(qubit, self) for qubit in range(len(qubits))
            ]

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
                self.samples(binary=False, backend=backend)
            )

        if binary:
            qubits = self.target_qubits
            return frequencies_to_binary(self._frequencies, len(qubits))

        return self._frequencies

    def apply_bitflips(self, p0, p1=None):  # pragma: no cover
        return apply_bitflips(self, p0, p1)
