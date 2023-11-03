from enum import Enum, auto

import numpy as np

from ..config import raise_error
from ..hamiltonians import Hamiltonian


class DoubleBracketFlowMode(Enum):
    """Define DBF evolution."""

    canonical = auto()
    """Use canonical commutator."""
    group_commutator = auto()
    """Use group commutator."""


class DoubleBracketFlow:
    """
    Class implementing the Double Bracket flow algorithm.
    For more details, see https://arxiv.org/pdf/2206.11772.pdf

    Args:
        hamiltonian (Hamiltonian): Starting Hamiltonian

    Example:
        .. code-block:: python

        import numpy as np
        from qibo.models.double_bracket import DoubleBracketFlow, DoubleBracketFlowMode
        from qibo.quantum_info import random_hermitian

        nqubits = 4
        h0 = random_hermitian(2**nqubits)
        dbf = DoubleBracketFlow(Hamiltonian(nqubits=nqubits, matrix=h0))

        # diagonalized matrix
        dbf.h
    """

    def __init__(self, hamiltonian: Hamiltonian):
        # TODO: consider passing Mode here
        self.h = self.h0 = hamiltonian

    def __call__(self, step: float, mode: DoubleBracketFlowMode, d: np.array = None):
        if mode is DoubleBracketFlowMode.canonical:
            operator = self.backend.calculate_matrix_exp(
                1.0j * step,
                self.diagonal_h_matrix @ self.h.matrix
                - self.h.matrix @ self.diagonal_h_matrix,
            )
        elif mode is DoubleBracketFlowMode.group_commutator:
            if d is None:
                raise_error(ValueError, f"Cannot use group_commutator with matrix {d}")
            operator = (
                self.h.exp(-step)
                @ self.backend.calculate_matrix_exp(-step, d)
                @ self.h.exp(step)
                @ self.backend.calculate_matrix_exp(step, d)
            )
        self.h.matrix = (
            operator @ self.h.matrix @ self.backend.cast(np.matrix(operator).getH())
        )

    @property
    def diagonal_h_matrix(self):
        """Diagonal H matrix."""
        return self.backend.cast(np.diag(np.diag(self.backend.to_numpy(self.h.matrix))))

    @property
    def off_diagonal_norm(self):
        """Norm of off-diagonal part of H matrix."""
        off_diag_h = self.h.matrix - self.diagonal_h_matrix
        # TODO: make this trace backend agnostic (test on GPU)
        return np.real(
            np.trace(self.backend.cast(np.matrix(off_diag_h).getH()) @ off_diag_h)
        )

    @property
    def backend(self):
        """Get Hamiltonian's backend."""
        return self.h0.backend
