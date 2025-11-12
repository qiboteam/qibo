from dataclasses import dataclass
from functools import cached_property

from numpy.typing import ArrayLike

from qibo.error_mitigation.inverse_response_matrix import InverseResponseMatrix


@dataclass
class IterativeBayesianUnfolding(InverseResponseMatrix):

    iterations: int = 10

    @cached_property
    def transposed_inverse_response_matrix(self):
        return self.backend.np.transpose(self.inverse_response_matrix, (1, 0))

    def unfold(self, probabilities: ArrayLike) -> ArrayLike:
        N = len(probabilities)
        unfolded_probabilities = (
            self.backend.np.ones((N, 1), dtype=probabilities.dtype) / N
        )
        for _ in range(self.iterations):
            unfolded_probabilities *= self.transposed_inverse_response_matrix @ (
                probabilities / (self.inverse_response_matrix @ unfolded_probabilities)
            )
        return unfolded_probabilities

    def mitigate_frequencies(self, frequencies: ArrayLike) -> ArrayLike:
        probabilities = frequencies / sum(frequencies)
        mitigated_probabilities = self.unfold(probabilities)
        mitigated_frequencies = self.backend.np.round(
            mitigated_probabilities * sum(frequencies), decimals=0
        )
        mitigated_frequencies = (
            mitigated_frequencies / sum(mitigated_frequencies)
        ) * sum(frequencies)
        return mitigated_frequencies
