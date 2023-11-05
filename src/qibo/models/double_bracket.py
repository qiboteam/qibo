from enum import Enum, auto

import numpy as np

from ..config import raise_error
from ..hamiltonians import Hamiltonian


class FlowGeneratorType(Enum):
    """Define DBF evolution."""

    canonical = auto()
    """Use canonical commutator."""
    single_commutator = auto()
    """Use single commutator."""
    group_commutator = auto()
    """Use group commutator approximation"""
    # TODO: add double commutator (does it converge?)


class DoubleBracketFlow:
    """
    Class implementing the Double Bracket flow algorithm.
    For more details, see https://arxiv.org/pdf/2206.11772.pdf

    Args:
        hamiltonian (Hamiltonian): Starting Hamiltonian

    Example:
        .. code-block:: python

        import numpy as np
        from qibo.models.double_bracket import DoubleBracketFlow, FlowGeneratorType
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

    def __call__(self, step: float, mode: FlowGeneratorType, d: np.array = None):
        if mode is FlowGeneratorType.canonical:
            operator = self.backend.calculate_matrix_exp(
                1.0j * step,
                self.commutator(self.diagonal_h_matrix, self.h.matrix),
            )
        elif mode is FlowGeneratorType.single_commutator:
            if d is None:
                raise_error(ValueError, f"Cannot use group_commutator with matrix {d}")
            operator = self.backend.calculate_matrix_exp(
                1.0j * step,
                self.commutator(d, self.h.matrix),
            )
        elif mode is FlowGeneratorType.group_commutator:
            if d is None:
                raise_error(ValueError, f"Cannot use group_commutator with matrix {d}")
            operator = (
                self.h.exp(-step)
                @ self.backend.calculate_matrix_exp(-step, d)
                @ self.h.exp(step)
                @ self.backend.calculate_matrix_exp(step, d)
            )
        operator_dagger = self.backend.cast(
            np.matrix(self.backend.to_numpy(operator)).getH()
        )
        self.h.matrix = operator @ self.h.matrix @ operator_dagger

    @staticmethod
    def commutator(a, b):
        """Compute commutator between two arrays."""
        return a @ b - b @ a

    @property
    def diagonal_h_matrix(self):
        """Diagonal H matrix."""
        return self.backend.cast(np.diag(np.diag(self.backend.to_numpy(self.h.matrix))))

    @property
    def off_diagonal_norm(self):
        """Norm of off-diagonal part of H matrix."""
        off_diag_h = self.h.matrix - self.diagonal_h_matrix
        off_diag_h_dag = self.backend.cast(
            np.matrix(self.backend.to_numpy(off_diag_h)).getH()
        )
        return np.real(np.trace(self.backend.to_numpy(off_diag_h_dag @ off_diag_h)))

    @property
    def backend(self):
        """Get Hamiltonian's backend."""
        return self.h0.backend
