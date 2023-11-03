from enum import Enum, auto

import numpy as np
from scipy.linalg import expm

from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian


class DoubleBracketFlowMode(Enum):
    """Define DBF evolution."""

    canonical = auto()
    """Use canonical commutator."""
    group_commutator = auto()
    """Use group commutator."""


# TODO: provide backend agnostic implementation
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
        dbf.h"""

    def __init__(self, hamiltonian: Hamiltonian):
        # TODO: consider passing Mode here
        self.h = self.h0 = hamiltonian

    def __call__(self, step: float, mode: DoubleBracketFlowMode, d: np.array = None):
        if mode is DoubleBracketFlowMode.canonical:
            operator = expm(
                step
                * (
                    np.diag(np.diag(self.h.matrix)) @ self.h.matrix
                    - self.h.matrix @ np.diag(np.diag(self.h.matrix))
                )
            )
        elif mode is DoubleBracketFlowMode.group_commutator:
            if d is None:
                raise_error(ValueError, f"Cannot use group_commutator with matrix {d}")
            operator = (
                expm(1j * self.h.matrix * step)
                @ expm(1j * d * step)
                @ expm(-1j * self.h.matrix * step)
                @ expm(-1j * d * step)
            )
        self.h.matrix = operator @ self.h.matrix @ np.matrix(operator).getH()

    @property
    def off_diagonal_norm(self):
        off_diag_h = self.h.matrix - np.diag(np.diag(self.h.matrix))
        return np.real(np.trace(np.matrix(off_diag_h).getH() @ off_diag_h))
