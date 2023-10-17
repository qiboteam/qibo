import numpy as np


class QuantumState:
    """Data structure to represent the final state after circuit execution."""

    def __init__(self, state, backend):
        self.backend = backend
        self.density_matrix = len(state.shape) == 2
        self.nqubits = int(np.log2(state.shape[0]))
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

    def state(self, numpy=False):
        """State's tensor representation as a backend tensor.

        Args:
            numpy (bool): If ``True`` the returned tensor will be a numpy array,
                otherwise it will follow the backend tensor type.
                Default is ``False``.

        Returns:
            The state in the computational basis.
        """
        if numpy:
            return np.asarray(self._state)
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

        if qubits is None:
            qubits = tuple(range(self.nqubits))

        if self.density_matrix:
            return self.backend.calculate_probabilities_density_matrix(
                self._state, qubits, self.nqubits
            )
        else:
            return self.backend.calculate_probabilities(
                self._state, qubits, self.nqubits
            )

    def __str__(self):
        return self.symbolic()
