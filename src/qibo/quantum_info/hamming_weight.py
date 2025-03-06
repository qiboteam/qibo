from itertools import product
from typing import Optional, Union

import numpy as np

from qibo import Circuit
from qibo.backends import CliffordBackend
from qibo.config import raise_error
from qibo.gates import M
from qibo.measurements import frequencies_to_binary

from ._clifford_utils import _decomposition_AG04, _decomposition_BM20, _string_product


class HammingWeightResult:

    def __init__(self, state, weight, nqubits, measurements, nshots, backend=None):
        from qibo.backends import _check_backend

        self.backend = _check_backend(backend)
        self.measurements = measurements
        self.nqubits = nqubits
        self.nshots = nshots
        self.weight = weight

        self._state = state

    def symbolic(self, decimals: int = 5, cutoff: float = 1e-10, max_terms: int = 20):

        terms = self.backend.calculate_symbolic(
            self._state, self.nqubits, self.weight, decimals, cutoff, max_terms
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
