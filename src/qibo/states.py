from typing import Union

import numpy as np
from tensorflow import Tensor


class QuantumState:
    """Data structure to represent the final state after circuit execution."""

    def __init__(self, state: Union[np.ndarray, Tensor], backend):
        self.backend = backend
        self.density_matrix = len(state.shape) == 2
        self.nqubits = int(np.log2(state.shape[0]))
        self.qubits = tuple(range(self.nqubits))
        self._repeated_execution_probabilities = None
        self._state = state

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
        if self.density_matrix:
            terms = self.backend.calculate_symbolic_density_matrix(
                self._state, self.nqubits, decimals, cutoff, max_terms
            )
        else:
            terms = self.backend.calculate_symbolic(
                self._state, self.nqubits, decimals, cutoff, max_terms
            )
        return " + ".join(terms)

    def state(self, dirac=False, decimals=5, cutoff=1e-10, max_terms=20):
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
            If ``dirac=True`` a string with the Dirac representation of the state
            in the computational basis, otherwise a tensor representing the state in
            the computational basis.
        """
        if dirac:
            return self.symbolic(decimals, cutoff, max_terms)
        else:
            return self._state

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
