import collections
from qibo.config import raise_error


class CircuitResult:

    def __init__(self, backend, circuit, execution_result, nshots=None):
        self.backend = backend
        self.circuit = circuit
        self.nqubits = circuit.nqubits
        self.density_matrix = circuit.density_matrix
        self.execution_result = execution_result
        self.nshots = nshots

        self._samples = None
        self._frequencies = None
        self._bitflip_p0 = None
        self._bitflip_p1 = None

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
            terms = self.backend.calculate_symbolic_density_matrix(state, self.nqubits, decimals, cutoff, max_terms)
        else:
            terms = self.backend.calculate_symbolic(state, self.nqubits, decimals, cutoff, max_terms)
        return " + ".join(terms)

    def __repr__(self):
        return self.backend.circuit_result_representation(self)

    def __array__(self):
        """State's tensor representation as an array."""
        return self.state(numpy=True)

    def probabilities(self, qubits=None):
        """Calculates measurement probabilities by tracing out qubits.

        Args:
            qubits (list, set): Set of qubits that are measured.
        """
        return self.backend.circuit_result_probabilities(self, qubits)

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
        qubits = self.circuit.measurement_gate.qubits
        if self._samples is None:
            probs = self.probabilities(qubits)
            self._samples = self.backend.sample_shots(probs, self.nshots)
            if self.circuit.measurement_gate.has_bitflip_noise():
                p0, p1 = self.circuit.measurement_gate.bitflip_map
                bitflip_probabilities = [[p0.get(q) for q in qubits],
                                         [p1.get(q) for q in qubits]]
                noiseless_samples = self.backend.samples_to_binary(self._samples, len(qubits))
                noisy_samples = self.backend.apply_bitflips(noiseless_samples, bitflip_probabilities)
                self._samples = self.backend.samples_to_decimal(noisy_samples, len(qubits))

        if registers:
            qubit_map = {q: i for i, q in enumerate(qubits)}
            reg_samples = {}
            samples = self.backend.samples_to_binary(self._samples, len(qubits))
            for name, rqubits in self.circuit.measurement_tuples.items():
                rqubits = tuple(qubit_map.get(q) for q in rqubits)
                rsamples = samples[:, rqubits]
                if binary:
                    reg_samples[name] = rsamples
                else:
                    reg_samples[name] = self.backend.samples_to_decimal(rsamples, len(rqubits))
            return reg_samples

        if binary:
            return self.backend.samples_to_binary(self._samples, len(qubits))
        else:
            return self._samples

    @staticmethod
    def _frequencies_to_binary(frequencies, nqubits):
        return collections.Counter(
                {"{0:b}".format(k).zfill(nqubits): v
                 for k, v in frequencies.items()})

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
        qubits = self.circuit.measurement_gate.qubits
        if self._frequencies is None:
            if self.circuit.measurement_gate.has_bitflip_noise() and self._samples is None:
                self._samples = self.samples(binary=False)
            if self._samples is None:
                probs = self.probabilities(qubits)
                self._frequencies = self.backend.sample_frequencies(probs, self.nshots)
            else:
                self._frequencies = self.backend.calculate_frequencies(self._samples)

        if registers:
            qubit_map = {q: i for i, q in enumerate(qubits)}
            reg_frequencies = {}
            binary_frequencies = self._frequencies_to_binary(self._frequencies, len(qubits))
            for name, rqubits in self.circuit.measurement_tuples.items():
                rfreqs = collections.Counter()
                for bitstring, freq in binary_frequencies.items():
                    idx = 0
                    for i, q in enumerate(rqubits):
                        if int(bitstring[qubit_map.get(q)]):
                            idx += 2 ** (len(rqubits) - i - 1)
                    rfreqs[idx] += freq
                if binary:
                    reg_frequencies[name] = self._frequencies_to_binary(rfreqs, len(rqubits))
                else:
                    reg_frequencies[name] = rfreqs
            return reg_frequencies

        if binary:
            return self._frequencies_to_binary(self._frequencies, len(qubits))
        else:
            return self._frequencies

    def apply_bitflips(self, p0, p1=None):
        mgate = self.circuit.measurement_gate
        if p1 is None:
            probs = 2 * (mgate._get_bitflip_tuple(mgate.qubits, p0),)
        else:
            probs = (mgate._get_bitflip_tuple(mgate.qubits, p0),
                     mgate._get_bitflip_tuple(mgate.qubits, p1))
        noiseless_samples = self.samples()
        return self.backend.apply_bitflips(noiseless_samples, probs)
