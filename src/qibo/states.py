import collections

import numpy as np
import sympy

from qibo.config import raise_error


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
        name = "m{}".format(cls._counter)
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

    :class:`qibo.states.MeasurementResult` objects can be obtained
    when adding measurement gates to a circuit.

    Args:
        gate (:class:`qibo.gates.M`): Measurement gate associated with
            this result object.
        nshots (int): Number of measurement shots.
        backend (:class:`qibo.backends.abstract.AbstractBackend`): Backend
            to use for calculations.
    """

    def __init__(self, gate, nshots=0, backend=None):
        self.measurement_gate = gate
        self.backend = backend
        self.nshots = nshots
        self.circuit = None

        self._samples = None
        self._frequencies = None
        self._bitflip_p0 = None
        self._bitflip_p1 = None
        self._symbols = None

    def __repr__(self):
        qubits = self.measurement_gate.qubits
        nshots = self.nshots
        return f"MeasurementResult(qubits={qubits}, nshots={nshots})"

    def add_shot(self, probs):
        qubits = sorted(self.measurement_gate.target_qubits)
        shot = self.backend.sample_shots(probs, 1)
        bshot = self.backend.samples_to_binary(shot, len(qubits))
        if self._samples:
            self._samples.append(bshot[0])
        else:
            self._samples = [bshot[0]]
        self.nshots += 1
        return shot

    def has_samples(self):
        return self._samples is not None

    def register_samples(self, samples, backend=None):
        """Register samples array to the ``MeasurementResult`` object."""
        self._samples = samples
        self.nshots = len(samples)

    def register_frequencies(self, frequencies, backend=None):
        """Register frequencies to the ``MeasurementResult`` object."""
        self._frequencies = frequencies
        self.nshots = sum(frequencies.values())

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
            qubits = self.measurement_gate.target_qubits
            self._symbols = [MeasurementSymbol(i, self) for i in range(len(qubits))]

        return self._symbols

    def samples(self, binary=True, registers=False):
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
        if self._samples is None:
            if self.circuit is None:
                raise_error(
                    RuntimeError, "Cannot calculate samples if circuit is not provided."
                )
            # calculate samples for the whole circuit so that
            # individual register samples are registered here
            self.circuit.final_state.samples()
        if binary:
            return np.array(self._samples, dtype="int32")
        else:
            qubits = self.measurement_gate.target_qubits
            return self.backend.samples_to_decimal(self._samples, len(qubits))

    def frequencies(self, binary=True, registers=False):
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
        if self._frequencies is None:
            self._frequencies = self.backend.calculate_frequencies(
                self.samples(binary=False)
            )
        if binary:
            qubits = self.measurement_gate.target_qubits
            return frequencies_to_binary(self._frequencies, len(qubits))
        else:
            return self._frequencies

    def apply_bitflips(self, p0, p1=None):  # pragma: no cover
        return apply_bitflips(self, p0, p1)


class CircuitResult:
    """Data structure returned by circuit execution.

    Contains all the results produced by the circuit execution, such as
    the state vector or density matrix, measurement samples and frequencies.

    Args:
        backend (:class:`qibo.backends.abstract.AbstractBackend`): Backend
            to use for calculations.
        circuit (:class:`qibo.models.Circuit`): Circuit object that is producing
            this result.
        execution_result: Abstract raw data created by the circuit execution.
            The format of these data depends on the backend and they are processed
            by the backend. For simulation backends ``execution_result`` is a tensor
            holding the state vector or density matrix representation in the
            computational basis.
        nshots (int): Number of measurement shots, if measurements are performed.
    """

    def __init__(self, backend, circuit, execution_result, nshots=None):
        self.nqubits = circuit.nqubits
        self.measurements = circuit.measurements
        self.density_matrix = circuit.density_matrix
        self.circuit = circuit
        self.execution_result = execution_result
        self.backend = backend
        self.nshots = nshots

        self._measurement_gate = None
        self._samples = None
        self._frequencies = None
        self._repeated_execution_frequencies = None
        self._repeated_execution_probabilities = None
        self._bitflip_p0 = None
        self._bitflip_p1 = None
        self._symbols = None
        for gate in self.measurements:
            gate.result.reset()

    def state(self, numpy=False, decimals=-1, cutoff=1e-10, max_terms=20):
        """State's tensor representation as an backend tensor.

        Args:
            numpy (bool): If ``True`` the returned tensor will be a numpy array,
                otherwise it will follow the backend tensor type.
                Default is ``False``.
            decimals (int): If positive the Dirac representation of the state
                in the computational basis will be returned as a string.
                ``decimals`` will be the number of decimals of each amplitude.
                Default is -1.
            cutoff (float): Amplitudes with absolute value smaller than the
                cutoff are ignored from the Dirac representation.
                Ignored if ``decimals < 0``. Default is 1e-10.
            max_terms (int): Maximum number of terms in the Dirac representation.
                If the state contains more terms they will be ignored.
                Ignored if ``decimals < 0``. Default is 20.

        Returns:
            If ``decimals < 0`` a tensor representing the state in the computational
            basis, otherwise a string with the Dirac representation of the state
            in the computational basis.
        """
        tensor = self.backend.circuit_result_tensor(self)
        if decimals >= 0:
            return self.symbolic(decimals, cutoff, max_terms)
        if numpy:
            return self.backend.to_numpy(tensor)
        return tensor

    def symbolic(self, decimals=5, cutoff=1e-10, max_terms=20):
        """Dirac notation representation of the state in the computational basis.

        Args:
            decimals (int): Number of decimals for the amplitudes.
                Default is 5.
            cutoff (float): Amplitudes with absolute value smaller than the
                cutoff are ignored from the representation.
                Default is 1e-10.
            max_terms (int): Maximum number of terms to print. If the state
                contains more terms they will be ignored.
                Default is 20.

        Returns:
            A string representing the state in the computational basis.
        """
        state = self.backend.circuit_result_tensor(self)
        if self.density_matrix:
            terms = self.backend.calculate_symbolic_density_matrix(
                state, self.nqubits, decimals, cutoff, max_terms
            )
        else:
            terms = self.backend.calculate_symbolic(
                state, self.nqubits, decimals, cutoff, max_terms
            )
        return " + ".join(terms)

    def __repr__(self):
        return self.backend.circuit_result_representation(self)

    def __array__(self):
        """State's tensor representation as an array."""
        return self.state(numpy=True)

    def probabilities(self, qubits=None):
        """Calculates measurement probabilities by tracing out qubits.
        When noisy model is applied to a circuit and `circuit.density_matrix=False`,
        this method returns the average probability resulting from
        repeated execution. This probability distribution approximates the
        exact probability distribution obtained when `circuit.density_matrix=True`.

        Args:
            qubits (list, set): Set of qubits that are measured.
        """
        if self._repeated_execution_probabilities is not None:
            return self._repeated_execution_probabilities

        return self.backend.circuit_result_probabilities(self, qubits)

    def has_samples(self):
        if self.measurements:
            return self.measurements[0].result.has_samples()
        else:  # pragma: no cover
            return False

    @property
    def measurement_gate(self):
        """Single measurement gate containing all measured qubits.

        Useful for sampling all measured qubits at once when simulating.
        """
        if self._measurement_gate is None:
            from qibo import gates

            if not self.measurements:  # pragma: no cover
                raise_error(ValueError, "Circuit does not contain measurements.")

            for gate in self.measurements:
                if self._measurement_gate is None:
                    self._measurement_gate = gates.M(
                        *gate.init_args, **gate.init_kwargs
                    )
                else:
                    self._measurement_gate.add(gate)

        return self._measurement_gate

    def samples(self, binary=True, registers=False):
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
        qubits = self.measurement_gate.target_qubits
        if self._samples is None:
            if self.measurements[0].result.has_samples():
                self._samples = np.concatenate(
                    [gate.result.samples() for gate in self.measurements], axis=1
                )
            else:
                if self._frequencies is not None:
                    # generate samples that respect the existing frequencies
                    frequencies = self.frequencies(binary=False)
                    samples = np.concatenate(
                        [np.repeat(x, f) for x, f in frequencies.items()]
                    )
                    np.random.shuffle(samples)
                else:
                    # generate new samples
                    probs = self.probabilities(qubits)
                    samples = self.backend.sample_shots(probs, self.nshots)
                samples = self.backend.samples_to_binary(samples, len(qubits))
                if self.measurement_gate.has_bitflip_noise():
                    p0, p1 = self.measurement_gate.bitflip_map
                    bitflip_probabilities = [
                        [p0.get(q) for q in qubits],
                        [p1.get(q) for q in qubits],
                    ]
                    samples = self.backend.apply_bitflips(
                        samples, bitflip_probabilities
                    )
                # register samples to individual gate ``MeasurementResult``
                qubit_map = {
                    q: i for i, q in enumerate(self.measurement_gate.target_qubits)
                }
                self._samples = np.array(samples, dtype="int32")
                for gate in self.measurements:
                    rqubits = tuple(qubit_map.get(q) for q in gate.target_qubits)
                    gate.result.register_samples(
                        self._samples[:, rqubits], self.backend
                    )

        if registers:
            return {
                gate.register_name: gate.result.samples(binary)
                for gate in self.measurements
            }

        if binary:
            return self._samples
        else:
            return self.backend.samples_to_decimal(self._samples, len(qubits))

    def frequencies(self, binary=True, registers=False):
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
        if self._repeated_execution_frequencies is not None:
            return self._repeated_execution_frequencies

        qubits = self.measurement_gate.qubits
        if self._frequencies is None:
            if self.measurement_gate.has_bitflip_noise() and not self.has_samples():
                self._samples = self.samples()
            if not self.has_samples():
                # generate new frequencies
                probs = self.probabilities(qubits)
                self._frequencies = self.backend.sample_frequencies(probs, self.nshots)
                # register frequencies to individual gate ``MeasurementResult``
                qubit_map = {q: i for i, q in enumerate(qubits)}
                reg_frequencies = {}
                binary_frequencies = frequencies_to_binary(
                    self._frequencies, len(qubits)
                )
                for gate in self.measurements:
                    rfreqs = collections.Counter()
                    for bitstring, freq in binary_frequencies.items():
                        idx = 0
                        rqubits = gate.target_qubits
                        for i, q in enumerate(rqubits):
                            if int(bitstring[qubit_map.get(q)]):
                                idx += 2 ** (len(rqubits) - i - 1)
                        rfreqs[idx] += freq
                    gate.result.register_frequencies(rfreqs, self.backend)
            else:
                self._frequencies = self.backend.calculate_frequencies(
                    self.samples(binary=False)
                )

        if registers:
            return {
                gate.register_name: gate.result.frequencies(binary)
                for gate in self.measurements
            }

        if binary:
            return frequencies_to_binary(self._frequencies, len(qubits))

        return self._frequencies

    def apply_bitflips(self, p0, p1=None):
        return apply_bitflips(self, p0, p1)

    def expectation_from_samples(self, observable):
        """Computes the real expectation value of a diagonal observable from frequencies.

        Args:
            observable (Hamiltonian/SymbolicHamiltonian): diagonal observable in the computational basis.

        Returns:
            Real number corresponding to the expectation value.
        """
        freq = self.frequencies(binary=True)
        qubit_map = self.measurement_gate.qubits
        return observable.expectation_from_samples(freq, qubit_map)
