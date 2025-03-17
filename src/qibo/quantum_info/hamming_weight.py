from typing import Optional, Union

import numpy as np
from scipy.special import binom

from qibo import gates
from qibo.backends import NumpyBackend
from qibo.config import raise_error
from qibo.result import MeasurementOutcomes, QuantumState


class HammingWeightResult(QuantumState, MeasurementOutcomes):

    def __init__(self, state, weight, nqubits, measurements, nshots, backend=None):
        from qibo.backends import _check_backend

        self.backend = _check_backend(backend)
        self.engine = self.backend.engine
        self.measurements = measurements
        self.nqubits = nqubits
        self.nshots = nshots
        self.weight = weight
        self.n_choose_k = int(binom(self.nqubits, self.weight))

        self._state = state
        self._samples = None
        self._probs = None
        self._frequencies = None
        self._measurement_gate = None
        self._repeated_execution_frequencies = None

        self._measurement_gate = self.measurement_gate

    def symbolic(self, decimals: int = 5, cutoff: float = 1e-10, max_terms: int = 20):

        terms = self.backend.calculate_symbolic(
            self._state, self.nqubits, self.weight, decimals, cutoff, max_terms
        )
        return " + ".join(terms)

    def full_state(self):
        if (
            self.backend._dict_indexes is None
            or list(self.backend._dict_indexes.keys())[0].count("1") != self.weight
        ):
            self.backend._dict_indexes = self.backend._get_lexicographical_order(
                self.nqubits, self.weight
            )

        state = self.engine.np.zeros(2**self.nqubits, dtype=self.engine.np.complex128)
        for i, j in self.backend._dict_indexes.values():
            state[j] = self._state[i]

        state = self.engine.cast(state)
        return state

    def probabilities(self, qubits: Optional[Union[list, set]] = None):
        if self.nshots is None or len(self.measurements) == 0:
            return self.exact_probabilities(qubits)
        else:
            return self.probabilities_from_samples(qubits)

    def exact_probabilities(self, qubits: Optional[Union[list, set]] = None):
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
            self._state, qubits, self.weight, self.nqubits
        )

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

        if len(self.measurements) == 0:
            raise_error(
                RuntimeError,
                "No measurements were performed. Cannot return samples.",
            )

        self._probs = self.exact_probabilities()
        return super().samples(binary=binary, registers=registers)

    def frequencies(self, binary: bool = True, registers: bool = False):

        if len(self.measurements) == 0:
            raise_error(
                RuntimeError,
                "No measurements were performed. Cannot return frequencies.",
            )

        if not self.has_samples():
            self._probs = self.exact_probabilities(self._measurement_gate.qubits)
            self._probs = [float(p) for p in self._probs]
        return super().frequencies(binary=binary, registers=registers)

    def probabilities_from_samples(self, qubits: Optional[Union[list, set]] = None):
        """Calculate the probabilities as frequencies / nshots

        Returns:
            The array containing the probabilities of the measured qubits.
        """
        if qubits is None:
            qubits = self.measurement_gate.qubits
        else:
            if not set(qubits).issubset(self.measurement_gate.qubits):
                raise_error(
                    RuntimeError,
                    f"Asking probabilities for qubits {qubits}, but only qubits {self.measurement_gate.qubits} were measured.",
                )
        qubits = [self.measurement_gate.qubits.index(q) for q in qubits]

        nqubits = len(self.measurement_gate.qubits)
        probs = [0 for _ in range(2**nqubits)]
        for state, freq in self.frequencies(binary=False).items():
            probs[state] = freq / self.nshots
        rtype = self.backend.np.real(probs).dtype
        probs = self.backend.cast(probs, dtype=rtype)
        self._probs = probs

        if nqubits != self.nqubits:
            self.backend._dict_indexes = None

        return super(self.backend.__class__, self.backend).calculate_probabilities(
            self.engine.np.sqrt(probs), qubits, nqubits
        )
