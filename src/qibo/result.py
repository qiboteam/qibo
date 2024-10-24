import collections
import warnings
from typing import Optional, Union

import numpy as np

from qibo import __version__, backends, gates
from qibo.config import raise_error
from qibo.measurements import apply_bitflips, frequencies_to_binary


def load_result(filename: str):
    """Loads the results of a circuit execution saved to disk.

    Args:
        filename (str): Path to the file containing the results.

    Returns:
        :class:`qibo.result.QuantumState` or :class:`qibo.result.MeasurementOutcomes` or :class:`qibo.result.CircuitResult`: result of circuit execution saved to disk, depending on saved filed.
    """
    payload = np.load(filename, allow_pickle=True).item()
    return globals()[payload.pop("dtype")].from_dict(payload)


class QuantumState:
    """Data structure to represent the final state after circuit execution.

    Args:
        state (np.ndarray): Input quantum state as np.ndarray.
        backend (qibo.backends.AbstractBackend): Backend used for the calculations. If not provided, the current backend is going to be used.
    """

    def __init__(self, state, backend=None):
        from qibo.backends import _check_backend

        self.backend = _check_backend(backend)
        self.density_matrix = len(state.shape) == 2
        self.nqubits = int(np.log2(state.shape[0]))
        self._state = state

    def symbolic(self, decimals: int = 5, cutoff: float = 1e-10, max_terms: int = 20):
        """Dirac notation representation of the state in the computational basis.

        Args:
            decimals (int, optional): Number of decimals for the amplitudes.
                Defaults to :math:`5`.
            cutoff (float, optional): Amplitudes with absolute value smaller than the
                cutoff are ignored from the representation. Defaults to  ``1e-10``.
            max_terms (int, optional): Maximum number of terms to print. If the state
                contains more terms they will be ignored. Defaults to :math:`20`.

        Returns:
            (str): A string representing the state in the computational basis.
        """
        if self.density_matrix:
            terms = self.backend.calculate_symbolic_density_matrix(
                self._state, self.nqubits, decimals, cutoff, max_terms
            )
        else:
            terms = self.backend.calculate_symbolic(
                self._state, self.nqubits, decimals, cutoff, max_terms
            )
        return " + ".join(terms)

    def state(self, numpy: bool = False):
        """State's tensor representation as a backend tensor.

        Args:
            numpy (bool, optional): If ``True`` the returned tensor will be a ``numpy`` array,
                otherwise it will follow the backend tensor type.
                Defaults to ``False``.

        Returns:
            The state in the computational basis.
        """
        if numpy:
            return np.array(self._state.tolist())

        return self._state

    def probabilities(self, qubits: Optional[Union[list, set]] = None):
        """Calculates measurement probabilities by tracing out qubits.

        When noisy model is applied to a circuit and `circuit.density_matrix=False`,
        this method returns the average probability resulting from
        repeated execution. This probability distribution approximates the
        exact probability distribution obtained when `circuit.density_matrix=True`.

        Args:
            qubits (list or set, optional): Set of qubits that are measured.
                If ``None``, ``qubits`` equates the total number of qubits.
                Defauts to ``None``.
        Returns:
            (np.ndarray): Probabilities over the input qubits.
        """

        if qubits is None:
            qubits = tuple(range(self.nqubits))

        if self.density_matrix:
            return self.backend.calculate_probabilities_density_matrix(
                self._state, qubits, self.nqubits
            )

        return self.backend.calculate_probabilities(self._state, qubits, self.nqubits)

    def __str__(self):
        return self.symbolic()

    def to_dict(self):
        """Returns a dictonary containinig all the information needed to rebuild the ``QuantumState``"""
        return {
            "state": self.state(numpy=True),
            "dtype": self.__class__.__name__,
            "qibo": __version__,
        }

    def dump(self, filename: str):
        """Writes to file the ``QuantumState`` for future reloading.

        Args:
            filename (str): Path to the file to write to.
        """
        with open(filename, "wb") as f:
            np.save(f, self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict):
        """Builds a ``QuantumState`` object starting from a dictionary.

        Args:
            payload (dict): Dictionary containing all the information
                to load the ``QuantumState`` object.

        Returns:
            :class:`qibo.result.QuantumState`: Quantum state object..
        """
        backend = backends.construct_backend("numpy")
        return cls(payload.get("state"), backend=backend)

    @classmethod
    def load(cls, filename: str):
        """Builds the ``QuantumState`` object stored in a file.

        Args:
            filename (str): Path to the file containing the ``QuantumState``.

        Returns:
            :class:`qibo.result.QuantumState`: Quantum state object.
        """
        payload = np.load(filename, allow_pickle=True).item()
        return cls.from_dict(payload)


class MeasurementOutcomes:
    """Object to store the outcomes of measurements after circuit execution.

    Args:
        measurements (:class:`qibo.gates.M`): Measurement gates.
        backend (:class:`qibo.backends.AbstractBackend`): Backend used for the calculations.
            If ``None``, then the current backend is used. Defaults to ``None``.
        probabilities (np.ndarray): Use these probabilities to generate samples and frequencies.
        samples (np.darray): Use these samples to generate probabilities and frequencies.
        nshots (int): Number of shots used for samples, probabilities and frequencies generation.
    """

    def __init__(
        self,
        measurements,
        backend=None,
        probabilities=None,
        samples: Optional[int] = None,
        nshots: int = 1000,
    ):
        self.backend = backend
        self.measurements = measurements
        self.nshots = nshots

        self._measurement_gate = None
        self._probs = probabilities
        self._samples = samples
        self._frequencies = None
        self._repeated_execution_frequencies = None

        if samples is not None:
            for m in measurements:
                indices = [self.measurement_gate.qubits.index(q) for q in m.qubits]
                m.result.register_samples(samples[:, indices])

    def frequencies(self, binary: bool = True, registers: bool = False):
        """Returns the frequencies of measured samples.

        Args:
            binary (bool, optional): If ``True``, returns frequency keys in binary form.
                If ``False``, returns them in decimal form. Defaults to ``True``.
            registers (bool, optional): Group frequencies according to registers.
                Defaults to ``False``.

        Returns:
            A :class:`collections.Counter` where the keys are the observed values
            and the values the corresponding frequencies, that is the number
            of times each measured value/bitstring appears.

            If ``binary`` is ``True``
                the keys of the :class:`collections.Counter` are in binary form,
                as strings of :math:`0` and :math`1`.
            If ``binary`` is ``False``
                the keys of the :class:`collections.Counter` are integers.
            If ``registers`` is ``True``
                a `dict` of :class:`collections.Counter` is returned where keys are
                the name of each register.
            If ``registers`` is ``False``
                a single :class:`collections.Counter` is returned which contains samples
                from all the measured qubits, independently of their registers.
        """
        qubits = self.measurement_gate.qubits

        if self._repeated_execution_frequencies is not None:
            if binary:
                return self._repeated_execution_frequencies

            return collections.Counter(
                {int(k, 2): v for k, v in self._repeated_execution_frequencies.items()}
            )

        if self._frequencies is None:
            if self.measurement_gate.has_bitflip_noise() and not self.has_samples():
                self._samples = self.samples()
            if not self.has_samples():
                # generate new frequencies
                self._frequencies = self.backend.sample_frequencies(
                    self._probs, self.nshots
                )
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
                    gate.result.register_frequencies(rfreqs)
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

    def probabilities(self, qubits: Optional[Union[list, set]] = None):
        """Calculate the probabilities as frequencies / nshots

        Returns:
            The array containing the probabilities of the measured qubits.
        """
        nqubits = len(self.measurement_gate.qubits)
        if qubits is None:
            qubits = range(nqubits)
        else:
            if not set(qubits).issubset(self.measurement_gate.qubits):
                raise_error(
                    RuntimeError,
                    f"Asking probabilities for qubits {qubits}, but only qubits {self.measurement_gate.qubits} were measured.",
                )
            qubits = [self.measurement_gate.qubits.index(q) for q in qubits]

        if self._probs is not None and not self.measurement_gate.has_bitflip_noise():
            return self.backend.calculate_probabilities(
                np.sqrt(self._probs), qubits, nqubits
            )

        probs = [0 for _ in range(2**nqubits)]
        for state, freq in self.frequencies(binary=False).items():
            probs[state] = freq / self.nshots
        probs = self.backend.cast(probs)
        self._probs = probs
        return self.backend.calculate_probabilities(
            self.backend.np.sqrt(probs), qubits, nqubits
        )

    def has_samples(self):
        """Check whether the samples are available already.

        Returns:
            (bool): ``True`` if the samples are available, ``False`` otherwise.
        """
        return self.measurements[0].result.has_samples() or self._samples is not None

    def samples(self, binary: bool = True, registers: bool = False):
        """Returns raw measurement samples.

        Args:
            binary (bool, optional): Return samples in binary or decimal form.
            registers (bool, optional): Group samples according to registers.

        Returns:
            If ``binary`` is ``True``
                samples are returned in binary form as a tensor
                of shape ``(nshots, n_measured_qubits)``.
            If ``binary`` is ``False``
                samples are returned in decimal form as a tensor
                of shape ``(nshots,)``.
            If ``registers`` is ``True``
                samples are returned in a ``dict`` where the keys are the register
                names and the values are the samples tensors for each register.
            If ``registers`` is ``False``
                a single tensor is returned which contains samples from all the
                measured qubits, independently of their registers.
        """
        qubits = self.measurement_gate.target_qubits
        if self._samples is None:
            if self.measurements[0].result.has_samples():
                self._samples = self.backend.np.concatenate(
                    [gate.result.samples() for gate in self.measurements],
                    axis=1,
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
                    samples = self.backend.sample_shots(self._probs, self.nshots)
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
                self._samples = samples
                for gate in self.measurements:
                    rqubits = tuple(qubit_map.get(q) for q in gate.target_qubits)
                    gate.result.register_samples(self._samples[:, rqubits])

        if registers:
            return {
                gate.register_name: gate.result.samples(binary)
                for gate in self.measurements
            }

        if binary:
            return self._samples

        return self.backend.samples_to_decimal(self._samples, len(qubits))

    @property
    def measurement_gate(self):
        """Single measurement gate containing all measured qubits.

        Useful for sampling all measured qubits at once when simulating.
        """
        if self._measurement_gate is None:
            for gate in self.measurements:
                if self._measurement_gate is None:
                    self._measurement_gate = gates.M(
                        *gate.init_args, **gate.init_kwargs
                    )
                else:
                    self._measurement_gate.add(gate)

        return self._measurement_gate

    def apply_bitflips(self, p0: float, p1: Optional[float] = None):
        """Apply bitflips to the measurements with probabilities `p0` and `p1`

        Args:
            p0 (float): Probability of the 0->1 flip.
            p1 (float): Probability of the 1->0 flip.
        """
        return apply_bitflips(self, p0, p1)

    def expectation_from_samples(self, observable):
        """Computes the real expectation value of a diagonal observable from frequencies.

        Args:
            observable (Hamiltonian/SymbolicHamiltonian): diagonal observable in the
                computational basis.

        Returns:
            (float): expectation value from samples.
        """
        freq = self.frequencies(binary=True)
        qubit_map = self.measurement_gate.qubits
        return observable.expectation_from_samples(freq, qubit_map)

    def to_dict(self):
        """Returns a dictonary containinig all the information needed to rebuild the :class:`qibo.result.MeasurementOutcomes`."""
        args = {
            "measurements": [m.to_json() for m in self.measurements],
            "probabilities": self._probs,
            "samples": self._samples,
            "nshots": self.nshots,
            "dtype": self.__class__.__name__,
            "qibo": __version__,
        }
        return args

    def dump(self, filename: str):
        """Writes to file the :class:`qibo.result.MeasurementOutcomes` for future reloading.

        Args:
            filename (str): Path to the file to write to.
        """
        with open(filename, "wb") as f:
            np.save(f, self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict):
        """Builds a :class:`qibo.result.MeasurementOutcomes` object starting from a dictionary.

        Args:
            payload (dict): Dictionary containing all the information to load the :class:`qibo.result.MeasurementOutcomes` object.

        Returns:
            A :class:`qibo.result.MeasurementOutcomes` object.
        """
        from qibo.backends import construct_backend

        if payload["probabilities"] is not None and payload["samples"] is not None:
            warnings.warn(
                "Both `probabilities` and `samples` found, discarding the `probabilities` and building out of the `samples`."
            )
            payload.pop("probabilities")
        backend = construct_backend("numpy")
        measurements = [gates.M.load(m) for m in payload.get("measurements")]
        return cls(
            measurements,
            backend=backend,
            probabilities=payload.get("probabilities"),
            samples=payload.get("samples"),
            nshots=payload.get("nshots"),
        )

    @classmethod
    def load(cls, filename: str):
        """Builds the :class:`qibo.result.MeasurementOutcomes` object stored in a file.

        Args:
            filename (str): Path to the file containing the :class:`qibo.result.MeasurementOutcomes`.

        Returns:
            A :class:`qibo.result.MeasurementOutcomes` object.
        """
        payload = np.load(filename, allow_pickle=True).item()
        return cls.from_dict(payload)


class CircuitResult(QuantumState, MeasurementOutcomes):
    """Object to store both the outcomes of measurements and the final state after circuit execution.

    Args:
        final_state (np.ndarray): Input quantum state as np.ndarray.
        measurements (qibo.gates.M): The measurement gates containing the measurements.
        backend (qibo.backends.AbstractBackend): Backend used for the calculations. If not provided, then the current backend is going to be used.
        probabilities (np.ndarray): Use these probabilities to generate samples and frequencies.
        samples (np.darray): Use these samples to generate probabilities and frequencies.
        nshots (int): Number of shots used for samples, probabilities and frequencies generation.
    """

    def __init__(
        self, final_state, measurements, backend=None, samples=None, nshots=1000
    ):
        QuantumState.__init__(self, final_state, backend)
        qubits = [q for m in measurements for q in m.target_qubits]
        if len(qubits) == 0:
            raise ValueError(
                "Circuit does not contain measurements. Use a `QuantumState` instead."
            )
        probs = QuantumState.probabilities(self, qubits) if samples is None else None
        MeasurementOutcomes.__init__(
            self,
            measurements,
            backend=backend,
            probabilities=probs,
            samples=samples,
            nshots=nshots,
        )

    def probabilities(self, qubits: Optional[Union[list, set]] = None):
        if self.measurement_gate.has_bitflip_noise():
            return MeasurementOutcomes.probabilities(self, qubits)
        return QuantumState.probabilities(self, qubits)

    def to_dict(self):
        """Returns a dictonary containinig all the information needed to rebuild the ``CircuitResult``."""
        args = MeasurementOutcomes.to_dict(self)
        args.update(QuantumState.to_dict(self))
        args.update({"dtype": self.__class__.__name__})
        return args

    @classmethod
    def from_dict(cls, payload: dict):
        """Builds a ``CircuitResult`` object starting from a dictionary.

        Args:
            payload (dict): Dictionary containing all the information to load the ``CircuitResult`` object.

        Returns:
            :class:`qibo.result.CircuitResult`: circuit result object.
        """
        state_load = {"state": payload.pop("state")}
        state = QuantumState.from_dict(state_load)
        measurements = MeasurementOutcomes.from_dict(payload)
        return cls(
            state.state(),
            measurements.measurements,
            backend=state.backend,
            samples=measurements.samples(),
            nshots=measurements.nshots,
        )
