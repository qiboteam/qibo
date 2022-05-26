import collections
from qibo.config import raise_error


class CircuitResult:

    def __init__(self, backend, circuit, execution_result, nshots):
        self.backend = backend
        self.circuit = circuit
        self.nqubits = circuit.nqubits
        self.density_matrix = circuit.density_matrix
        self.execution_result = execution_result
        self.nshots = nshots

        self._samples = None
        self._frequencies = None

    def __len__(self):
        """Number of components in the state's tensor representation."""
        return 2 ** self.nqubits

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
        tensor = self.backend.get_state_tensor(self)
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
        return self.backend.get_state_symbolic(self, result, decimals, cutoff, max_terms)

    def __repr__(self):
        return self.backend.get_state_repr(self)

    def __array__(self):
        """State's tensor representation as an array."""
        return self.state()

    def probabilities(self, qubits=None):
        """Calculates measurement probabilities by tracing out qubits.

        Exactly one of the following arguments should be given.

        Args:
            qubits (list, set): Set of qubits that are measured.
        """
        if qubits is None:
            qubits = self.measured_qubits
        return self.backend.get_state_probabilities(self, qubits)

    def samples(self, binary=True, registers=False):
        """Returns raw measurement samples.

        Args:
            binary (bool): Return samples in binary or decimal form.
            registers (bool): Group samples according to registers.
                Can be used only if ``registers`` were given when calling
                :meth:`qibo.abstractions.states.AbstractState.measure`.

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

        if registers:
            reg_samples = {}
            samples = self.backend.samples_to_binary(self._samples, len(qubits))
            for name, rqubits in self.circuit.mesurement_tuples.items():
                rqubits = tuple(qubit_map.get(q) for q in qubits)
                rsamples = self.backend.register_samples(samples, rqubits)
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
                Can be used only if ``registers`` were given when calling
                :meth:`qibo.abstractions.states.AbstractState.measure`.

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
        if self._frequencies is None:
            if self._samples is None:
                qubits = self.circuit.measurement_gate.qubits
                probs = self.probabilities(qubits)
                self._frequencies = self.backend.sample_frequencies(probs, self.nshots)
            else:
                self._frequencies = self.backend.calculate_frequencies(self._samples)

        if registers:
            reg_frequencies = {}
            for name, qubits in self.circuit.measurement_tuples.items():
                rfreqs = collections.Counter()
                for bitstring, freq in self._binary_frequencies.items():
                    idx = 0
                    for i, q in enumerate(qubits):
                        if int(bitstring[qubit_map.get(q)]):
                            idx += 2 ** (len(qubits) - i - 1)
                    rfreqs[idx] += freq
                if binary:
                    reg_frequencies[name] = self._frequencies_to_binary(rfreqs, len(qubits))
                else:
                    reg_frequencies[name] = rfreqs
            return reg_frequencies

        if binary:
            nqubits = len(self.circuit.measurement_gate.qubits)
            return self._frequencies_to_binary(self._frequencies, nqubits)
        else:
            return self._frequencies
