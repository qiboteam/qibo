"""Module defining classes that store results of circuit execution."""

import collections
import warnings
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from sympy.core import basic

from qibo import __version__, gates
from qibo.config import raise_error
from qibo.measurements import apply_bitflips, frequencies_to_binary


def load_result(filename: str):
    """Loads the results of a circuit execution saved to disk.

    Args:
        filename (str): Path to the file containing the results.

    Returns:
        :class:`qibo.result.QuantumState` or
        :class:`qibo.result.MeasurementOutcomes` or
        :class:`qibo.result.CircuitResult`: result of circuit execution saved to disk,
        depending on saved filed.
    """
    payload = np.load(filename, allow_pickle=True).item()
    return globals()[payload.pop("dtype")].from_dict(payload)


class QuantumState:
    """Data structure to represent the final state after circuit execution.

    Args:
        state (ndarray): Input quantum state as ``ndarray``.
        backend (:class:`qibo.backends.abstract.Backend`): Backend used for the calculations.
            If not provided, the global backend is going to be used.
    """

    def __init__(self, state, backend=None):
        from qibo.backends import (  # pylint: disable=import-outside-toplevel
            _check_backend,
        )

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
            str: String representing the state in the computational basis.
        """
        terms = self.backend.calculate_symbolic(
            self._state, self.nqubits, decimals, cutoff, max_terms
        )
        return " + ".join(terms)

    def state(self, numpy: bool = False):
        """State's tensor representation as a backend tensor.

        .. note::
            If the state has Hamming weight :math:`k` and is computed using the
            ``HammingWeightBackend``, its dimension is :math:`d = \\binom{n}{k}`,
            where :math:`n` is the number of qubits.

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

        return self.backend.calculate_probabilities(
            self._state, qubits, self.nqubits, density_matrix=self.density_matrix
        )

    def __str__(self):
        return self.symbolic()

    def to_dict(self):
        """Returns a dictonary containinig all the information needed to
        rebuild the ``QuantumState``"""
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
        from qibo.backends import (  # pylint: disable=import-outside-toplevel
            construct_backend,
        )

        backend = construct_backend("numpy")
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
        backend (:class:`qibo.backends.abstract.Backend`): Backend used for the calculations.
            If ``None``, then the current backend is used. Defaults to ``None``.
        probabilities (np.ndarray): Use these probabilities to generate samples and frequencies.
        samples (np.darray): Use these samples to generate probabilities and frequencies.
        nshots (int): Number of shots used for samples, probabilities and frequencies generation.
        nqubits (int, optional): Total number of qubits in the circuit. When set,
            :meth:`probabilities` with ``qubits=None`` returns probabilities over
            all circuit qubits (not just the measured ones). If ``None``, defaults
            to the number of measured qubits. Defaults to ``None``.
    """

    def __init__(
        self,
        measurements,
        backend=None,
        probabilities=None,
        samples: Optional[int] = None,
        nshots: int = 1000,
        nqubits: Optional[int] = None,
    ):
        self.backend = backend
        self.measurements = measurements
        self.nshots = nshots
        self._nqubits = nqubits

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
                gate.register_name: gate.result.frequencies(
                    binary, backend=self.backend
                )
                for gate in self.measurements
            }

        if binary:
            return frequencies_to_binary(self._frequencies, len(qubits))

        return self._frequencies

    def probabilities(self, qubits: Optional[Union[list, set]] = None) -> ArrayLike:
        """Calculate the probabilities as frequencies / nshots

        Args:
            qubits (list or set, optional): Set of qubits for which to compute
                probabilities. If ``None`` and ``nqubits`` was provided at
                construction, probabilities are returned over all circuit
                qubits; otherwise only over the measured qubits.
                Defaults to ``None``.

        Returns:
            ArrayLike: The array containing the probabilities of the requested qubits.
        """
        measured_qubits = self.measurement_gate.qubits
        n_measured = len(measured_qubits)
        nqubits = self._nqubits if self._nqubits is not None else n_measured

        if qubits is None:
            qubits = range(nqubits)
        elif set(qubits).issubset(set(range(nqubits))):
            pass  # keep qubits as-is; they index into the full qubit space
        else:
            raise_error(
                RuntimeError,
                f"Asking probabilities for qubits {qubits}, "
                + f"but the system only has {nqubits} qubits "
                + f"(measured qubits: {measured_qubits}).",
            )

        # Build probability array in the measured-qubit space
        if self._probs is not None and not self.measurement_gate.has_bitflip_noise():
            measured_probs = self._probs
        else:
            measured_probs = [0] * 2**n_measured
            for state, freq in self.frequencies(binary=False).items():
                measured_probs[state] = freq / self.nshots
            measured_probs = self.backend.cast(
                measured_probs, dtype=self.backend.float64
            )
            self._probs = measured_probs

        if nqubits == n_measured:
            # No unmeasured qubits: use the standard path
            return self.backend.calculate_probabilities(
                self.backend.sqrt(measured_probs),
                list(qubits),
                n_measured,
            )

        # Expand measured probabilities into the full circuit qubit space.
        # Unmeasured qubits are placed in the |0⟩ state, consistent with the
        # standard qubit initialisation convention.
        full_probs = backend.zeros(2**nqubits, dtype=backend.float64)

        for measured_state in range(2**n_measured):
            p = float(measured_probs[measured_state])
            if p == 0:
                continue
            # Map the measured-state bits into the full-state index,
            # leaving unmeasured qubit bits as 0.
            full_state = 0
            m_bit_idx = 0
            for bit_pos in range(nqubits):
                if bit_pos in measured_qubits:
                    bit_val = (measured_state >> (n_measured - 1 - m_bit_idx)) & 1
                    m_bit_idx += 1
                    full_state |= bit_val << (nqubits - 1 - bit_pos)
            full_probs[full_state] = p

        return self.backend.calculate_probabilities(
            self.backend.sqrt(full_probs),
            list(qubits),
            nqubits,
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
                self._samples = self.backend.concatenate(
                    [
                        gate.result.samples(backend=self.backend)
                        for gate in self.measurements
                    ],
                    axis=1,
                )
            else:
                if self._frequencies is not None:
                    # generate samples that respect the existing frequencies
                    frequencies = self.frequencies(binary=False)
                    samples = [
                        self.backend.repeat(x, f) for x, f in frequencies.items()
                    ]
                    samples = self.backend.concatenate(samples)
                    self.backend.shuffle(samples)
                    samples = self.backend.cast(samples, dtype=self.backend.int64)
                else:
                    # generate new samples
                    samples = self.backend.sample_shots(self._probs, self.nshots)
                samples = self.backend.samples_to_binary(samples, len(qubits))
                if self.measurement_gate.has_bitflip_noise():
                    p0, p1 = self.measurement_gate.bitflip_map
                    bitflip_probabilities = self.backend.cast(
                        [
                            [p0.get(q) for q in qubits],
                            [p1.get(q) for q in qubits],
                        ],
                        dtype=self.backend.float64,
                    )
                    samples = self.backend.apply_bitflips(
                        samples, bitflip_probabilities
                    )
                # register samples to individual gate ``MeasurementResult``
                qubit_map = self.measurement_gate.target_qubits
                qubit_map = dict(zip(qubit_map, range(len(qubit_map))))
                self._samples = samples
                for gate in self.measurements:
                    rqubits = tuple(qubit_map.get(q) for q in gate.target_qubits)
                    gate.result.register_samples(self._samples[:, rqubits])

        if registers:
            return {
                gate.register_name: gate.result.samples(binary, backend=self.backend)
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

        return observable.expectation_from_samples(self.frequencies())

    def to_dict(self):
        """Returns a dictonary containinig all the information needed to rebuild the
        :class:`qibo.result.MeasurementOutcomes`."""
        args = {
            "measurements": [m.to_json() for m in self.measurements],
            "probabilities": self._probs,
            "samples": self._samples,
            "nshots": self.nshots,
            "nqubits": self._nqubits,
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
            payload (dict): Dictionary containing all the information to load the
                :class:`qibo.result.MeasurementOutcomes` object.

        Returns:
            :class:`qibo.result.MeasurementOutcomes`: Object storing the measurement outcomes.
        """
        from qibo.backends import construct_backend  # pylint: disable=C0415

        if payload["probabilities"] is not None and payload["samples"] is not None:
            warnings.warn(
                "Both `probabilities` and `samples` found, discarding the `probabilities`"
                + "and building out of the `samples`."
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
            nqubits=payload.get("nqubits"),
        )

    @classmethod
    def load(cls, filename: str):
        """Builds the :class:`qibo.result.MeasurementOutcomes` object stored in a file.

        Args:
            filename (str): Path to the file containing the
                :class:`qibo.result.MeasurementOutcomes`.

        Returns:
            :class:`qibo.result.MeasurementOutcomes`: instance of the
            ``MeasurementOutcomes`` class.
        """
        payload = np.load(filename, allow_pickle=True).item()
        return cls.from_dict(payload)

    @classmethod
    def from_samples(
        cls,
        samples,
        qubits=None,
        backend=None,
    ):
        """Constructs a :class:`qibo.result.MeasurementOutcomes` directly from
        a binary samples array.

        This is useful when building measurement outcomes from experimental data
        without needing to manually construct measurement gates.

        Args:
            samples (ArrayLike): Binary array of shape ``(nshots, nqubits)``.
                where each row is a measurement outcome with 0/1 values.
            qubits (tuple or list, optional): Qubit indices for the measured
                qubits. If ``None``, defaults to ``(0, 1, ..., nqubits - 1)``.
                Defaults to ``None``.
            backend (:class:`qibo.backends.abstract.Backend`, optional): Backend
                used for calculations. If ``None``, the current default backend
                is used. Defaults to ``None``.

        Returns:
            :class:`qibo.result.MeasurementOutcomes`: Object storing the
            measurement outcomes.

        Example:
            .. code-block:: python

                import numpy as np
                from qibo.result import MeasurementOutcomes

                samples = np.array([[0, 1], [1, 0], [1, 1]])
                result = MeasurementOutcomes.from_samples(samples)
                print(result.frequencies())  # Counter({'01': 1, '10': 1, '11': 1})
        """
        from qibo.backends import (  # pylint: disable=import-outside-toplevel
            _check_backend,
        )

        backend = _check_backend(backend)
        samples = np.array(samples)

        if samples.ndim != 2:
            raise_error(
                ValueError,
                f"samples must be a 2D array of shape (nshots, nqubits), "
                f"got shape {samples.shape}.",
            )

        if not np.all((samples == 0) | (samples == 1)):
            raise_error(
                ValueError, "samples array must contain only binary values (0 or 1)."
            )

        nshots, nqubits = samples.shape

        if qubits is None:
            qubits = tuple(range(nqubits))
        else:
            qubits = tuple(qubits)
            if len(qubits) != nqubits:
                raise_error(
                    ValueError,
                    f"Length of qubits ({len(qubits)}) does not match the number "
                    f"of columns in samples ({nqubits}).",
                )
            if len(set(qubits)) != len(qubits):
                raise_error(ValueError, "Qubit indices must be unique.")

        measurements = [gates.M(*qubits)]
        return cls(measurements, backend=backend, samples=samples, nshots=nshots)

    @classmethod
    def from_frequencies(
        cls,
        frequencies,
        nqubits=None,
        qubits=None,
        backend=None,
        seed=None,
    ):
        """Constructs a :class:`qibo.result.MeasurementOutcomes` from a
        frequencies dictionary.

        The frequencies are expanded into a binary samples array, shuffled to
        avoid ordering artifacts, and passed through the standard construction
        path so that all methods (``samples()``, ``frequencies()``,
        ``probabilities()``) work correctly.

        Args:
            frequencies (dict or Counter): Mapping from measurement outcomes to
                their counts. Keys can be binary strings (e.g. ``"010"``) or
                integers (e.g. ``2``). Values are non-negative integer counts.
            nqubits (int, optional): Total number of qubits in the circuit.
                When binary-string keys are used, the number of *measured*
                qubits is inferred from the key length; ``nqubits`` may be
                larger than or equal to that length, and the extra qubits are
                treated as unmeasured.
                When integer keys are used *without* ``qubits``, ``nqubits``
                is required and is interpreted as the number of *measured*
                qubits (backward-compatible behaviour).
                Defaults to ``None``.
            qubits (tuple or list, optional): Qubit indices for the measured
                qubits. If ``None``, defaults to ``(0, 1, ..., n_measured - 1)``
                where ``n_measured`` is the number of measured qubits.
                Defaults to ``None``.
            backend (:class:`qibo.backends.abstract.Backend`, optional): Backend
                used for calculations. If ``None``, the current default backend
                is used. Defaults to ``None``.
            seed (int, optional): Seed for sampling generation.

        Returns:
            :class:`qibo.result.MeasurementOutcomes`: Object storing the
            measurement outcomes.

        Raises:
            ValueError: If the number of measured qubits cannot be determined
                from the inputs.

        Example:
            .. code-block:: python

                from qibo.result import MeasurementOutcomes

                freq = {"00": 50, "11": 50}
                result = MeasurementOutcomes.from_frequencies(freq)
                print(result.frequencies())  # Counter({'00': 50, '11': 50})
        """
        from qibo.backends import (  # pylint: disable=import-outside-toplevel
            _check_backend,
        )

        backend = _check_backend(backend)
        frequencies = dict(frequencies)

        if len(frequencies) == 0:
            raise_error(ValueError, "frequencies dictionary must not be empty.")

        # Detect key type and normalise to integer-keyed dict
        first_key = next(iter(frequencies))
        if isinstance(first_key, str):
            if not all(isinstance(k, str) for k in frequencies):
                raise_error(
                    TypeError,
                    "All frequency keys must be of the same type (all strings or all integers).",
                )
            # Binary-string keys: infer n_measured from key length
            key_lengths = {len(k) for k in frequencies}
            if len(key_lengths) != 1:
                raise_error(
                    ValueError,
                    "All binary-string keys must have the same length, "
                    f"got lengths {key_lengths}.",
                )
            inferred_n_measured = key_lengths.pop()
            int_frequencies = {int(k, 2): v for k, v in frequencies.items()}
        else:
            # Integer keys
            inferred_n_measured = None
            int_frequencies = {int(k): v for k, v in frequencies.items()}

        # Resolve the number of measured qubits
        n_measured: int
        if qubits is not None:
            qubits = tuple(qubits)
            n_measured = len(qubits)
        elif inferred_n_measured is not None:
            n_measured = inferred_n_measured
        elif nqubits is not None:
            # Integer keys without qubits: nqubits gives the measured count
            n_measured = nqubits
        else:
            raise_error(
                ValueError,
                "Cannot determine the number of measured qubits. Provide "
                "`nqubits` or `qubits` when using integer keys in frequencies.",
            )

        # Validate consistency between inferred measured count and qubits
        if inferred_n_measured is not None and qubits is not None:
            if inferred_n_measured != len(qubits):
                raise_error(
                    ValueError,
                    f"Binary-string key length ({inferred_n_measured}) does not "
                    f"match the number of qubits provided ({len(qubits)}).",
                )

        if qubits is None:
            qubits = tuple(range(n_measured))

        # Resolve total nqubits for the circuit
        if nqubits is not None:
            if nqubits < n_measured:
                raise_error(
                    ValueError,
                    f"nqubits ({nqubits}) must be >= the number of measured "
                    f"qubits ({n_measured}).",
                )
            total_nqubits = nqubits
        else:
            total_nqubits = None  # will default to n_measured inside __init__

        # Expand frequencies into a binary samples array
        for state_int, count in int_frequencies.items():
            if not isinstance(count, (int, np.integer)):
                raise_error(
                    ValueError,
                    f"Frequency count for state {state_int} must be an integer, got {type(count)}.",
                )
            if count < 0:
                raise_error(
                    ValueError,
                    f"Frequency count for state {state_int} must be non-negative, got {count}.",
                )
        nshots = sum(int_frequencies.values())
        if nshots <= 0:
            raise_error(ValueError, "Total number of shots must be positive.")

        max_state = 2**n_measured - 1
        for state_int in int_frequencies:
            if state_int < 0 or state_int > max_state:
                raise_error(
                    ValueError,
                    f"State integer {state_int} is out of range for "
                    f"{n_measured} measured qubits (valid range: 0 to {max_state}).",
                )

        sample_rows = []
        for state_int, count in int_frequencies.items():
            if count == 0:
                continue
            # Convert integer state to binary row
            row = np.array(
                [(state_int >> (n_measured - 1 - i)) & 1 for i in range(n_measured)],
                dtype=int,
            )
            sample_rows.append(np.tile(row, (count, 1)))

        samples = np.concatenate(sample_rows, axis=0)
        # Shuffle to avoid ordering artifacts
        rng = np.random.default_rng(seed)
        rng.shuffle(samples)

        measurements = [gates.M(*qubits)]
        return cls(
            measurements,
            backend=backend,
            samples=samples,
            nshots=nshots,
            nqubits=total_nqubits,
        )


class CircuitResult(QuantumState, MeasurementOutcomes):
    """Object to store both the outcomes of measurements and the final state
    after circuit execution.

    Args:
        final_state (ndarray): Input quantum state as np.ndarray.
        measurements (:class:`qibo.gates.M`): The measurement gates containing the measurements.
        backend (:class:`qibo.backends.abstract.Backend`): Backend used for the calculations.
            If not provided, then the current backend is going to be used.
        probabilities (ndarray): Use these probabilities to generate samples and frequencies.
        samples (ndarray): Use these samples to generate probabilities and frequencies.
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
            nqubits=self.nqubits,
        )

    def probabilities(self, qubits: Optional[Union[list, set]] = None):
        if self.measurement_gate.has_bitflip_noise():
            return MeasurementOutcomes.probabilities(self, qubits)
        return QuantumState.probabilities(self, qubits)

    def to_dict(self):
        """Returns a dictonary containinig all the information needed to rebuild the
        ``CircuitResult``."""
        args = MeasurementOutcomes.to_dict(self)
        args.update(QuantumState.to_dict(self))
        args.update({"dtype": self.__class__.__name__})
        return args

    @classmethod
    def from_dict(cls, payload: dict):
        """Builds a ``CircuitResult`` object starting from a dictionary.

        Args:
            payload (dict): Dictionary containing all the information to load the
                ``CircuitResult`` object.

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
