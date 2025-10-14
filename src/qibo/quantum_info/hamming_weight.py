"Module with the object that stores results from circuit execution using the HammingWeightBackend."

from typing import Optional, Union

import numpy as np
from scipy.special import binom

from qibo.backends import HammingWeightBackend
from qibo.config import raise_error
from qibo.result import MeasurementOutcomes, QuantumState


class HammingWeightResult(QuantumState, MeasurementOutcomes):
    """Object storing the results of a circuit execution with the
    :class:`qibo.backends.hamming_weight.HammingWeightBackend`.

    Args:
        state (ndarray): statevector with a fixed Hamming weight.
            The dimension of the state is :math:`d = \\binom{n}{k}`, where
            :math:`n` is the number of qubits and :math:`k` is the Hamming weight.
            The components of the state are ordered in lexicographical order.
        weight (int): Hamming weight of ``state``.
        nqubits (int): number of qubits of ``state``.
        measurements (list, optional): list of measurements gates :class:`qibo.gates.M`.
            Defaults to ``None``.
        nshots (int, optional): number of shots used for sampling the measurements.
            Defaults to :math:`1000`.
        engine (str, optional): engine to use in the execution of the
            :class:`qibo.backends.HammingWeightBackend`. It accepts ``"numpy"``, ``"numba"``,
            ``"cupy"``, and ``"cuquantum"``. If ``None``, defaults to the corresponding engine
            from the current backend. Defaults to ``None``.
    """

    def __init__(
        self,
        state,
        weight: int,
        nqubits: int,
        measurements=None,
        nshots: int = 1000,
        engine=None,
    ):  # pylint: disable=too-many-arguments

        backend = HammingWeightBackend(engine)
        QuantumState.__init__(self, state, backend)
        MeasurementOutcomes.__init__(self, measurements, backend, nshots=nshots)

        self.nqubits = nqubits
        self.weight = weight

        self._state = state

    def symbolic(self, decimals: int = 5, cutoff: float = 1e-10, max_terms: int = 20):
        """Dirac notation representation of the state in the computational basis.

        Args:
            decimals (int, optional): Number of decimals for the amplitudes.
                Defaults to :math:`5`.
            cutoff (float, optional): Amplitudes with absolute value smaller than the
                cutoff are ignored from the representation. Defaults to  :math:`1e-10`.
            max_terms (int, optional): Maximum number of terms to print. If the state
                contains more terms they will be ignored. Defaults to :math:`20`.

        Returns:
            str: String representing the state in the computational basis.
        """
        terms = self.backend.calculate_symbolic(
            self._state, self.nqubits, self.weight, decimals, cutoff, max_terms
        )
        return " + ".join(terms)

    def full_state(self):
        """Tensor representation of ``state`` in the entire computational basis.

        .. note::
            This method is inefficient in runtime and memory for a large number of qubits.
        """

        self.backend._dict_indexes = self.backend._get_lexicographical_order(
            self.nqubits, self.weight
        )

        state = self.backend.np.zeros(2**self.nqubits, dtype=self.backend.np.complex128)
        state = self.backend.cast(state, dtype=state.dtype)
        indices = list(self.backend._dict_indexes.values())
        indices.sort()
        indices = np.asarray(indices)
        state[list(indices[:, 1])] = self._state[list(indices[:, 0])]

        return state

    def probabilities(self, qubits: Optional[Union[list, set]] = None):
        """Calculate the probabilities of the measured qubits.

        If the number of shots is :math:`0` or no measurements were performed,
        the probabilities are calculated from the statevector.
        Otherwise, the probabilities are calculated from the samples.

        Args:
            qubits (list or set, optional): Set of qubits that are measured.
                If ``None``, ``qubits`` equates the total number of qubits.
                Defauts to ``None``.
        Returns:
            ndarray: Probabilities over the input qubits.
        """

        if self.nshots is None or len(self.measurements) == 0:
            return self._exact_probabilities(qubits)

        return self._probabilities_from_samples(qubits)

    def _exact_probabilities(self, qubits: Optional[Union[list, set]] = None):
        """Calculate measurement probabilities by tracing out qubits.

        Args:
            qubits (list or set, optional): Set of qubits that are measured.
                If ``None``, ``qubits`` equates the total number of qubits.
                Defauts to ``None``.

        Returns:
            ndarray: Probabilities over the input qubits.
        """

        if qubits is None:
            qubits = tuple(range(self.nqubits))

        return self.backend.calculate_probabilities(
            self._state, qubits, self.weight, self.nqubits
        )

    def _probabilities_from_samples(self, qubits: Optional[Union[list, set]] = None):
        """Calculate the probabilities as ``frequencies / nshots``.

        Args:
            qubits (list or set, optional): Set of qubits that are measured.
                If ``None``, ``qubits`` equates the total number of qubits.
                Defauts to ``None``.

        Returns:
            ndarray: Array containing the probabilities of the measured qubits.
        """
        if qubits is None:
            qubits = self.measurement_gate.qubits
        else:
            if not set(qubits).issubset(self.measurement_gate.qubits):
                raise_error(
                    RuntimeError,
                    f"Asking probabilities for qubits {qubits}, "
                    + f"but only qubits {self.measurement_gate.qubits} were measured.",
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

        return self.backend.calculate_full_probabilities(
            self.backend.np.sqrt(probs), qubits, nqubits
        )

    def samples(self, binary: bool = True, registers: bool = False):
        """Returns raw measurement samples.

        Args:
            binary (bool, optional): Return samples in binary or decimal form.
                Defaults to ``True``.
            registers (bool, optional): Group samples according to registers.
                Defaults to ``False``.

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

        self._probs = self._exact_probabilities()
        return super().samples(binary=binary, registers=registers)

    def frequencies(self, binary: bool = True, registers: bool = False):
        """Return the frequencies of measured samples.

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
        if len(self.measurements) == 0:
            raise_error(
                RuntimeError,
                "No measurements were performed. Cannot return frequencies.",
            )

        if not self.has_samples():
            self._probs = self._exact_probabilities(self._measurement_gate.qubits)

        return super().frequencies(binary=binary, registers=registers)
