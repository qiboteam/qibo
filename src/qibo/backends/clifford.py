"""Module defining the Clifford backend."""
import collections
from functools import reduce
from typing import Union

import numpy as np

from qibo import gates
from qibo.backends.numpy import NumpyBackend
from qibo.backends.tensorflow import TensorflowBackend
from qibo.config import raise_error


def _calculation_engine(backend):
    """Helper function to initialize the Clifford backend with the correct engine."""
    if backend.name == "qibojit":
        if (
            backend.platform == "cupy" or backend.platform == "cuquantum"
        ):  # pragma: no cover
            return backend.cp
        return backend.np

    return backend.np


class CliffordOperations:
    """Operations performed by Clifford gates on the phase-space representation of stabilizer states.

    See `Aaronson & Gottesman (2004) <https://arxiv.org/abs/quant-ph/0406196>`_.

    Args:
        engine (:class:`qibo.backends.abstract.Backend`): Backend used for the calculations.
    """

    def __init__(self, engine):
        self.engine = engine
        self.np = _calculation_engine(engine)

    def I(self, symplectic_matrix, q, nqubits):
        return symplectic_matrix

    def H(self, symplectic_matrix, q, nqubits):
        r, x, z = self._get_rxz(symplectic_matrix, nqubits)
        self._set_r(
            symplectic_matrix,
            r ^ (x[:, q] & z[:, q]),
        )
        symplectic_matrix[:, [q, nqubits + q]] = symplectic_matrix[:, [nqubits + q, q]]
        return symplectic_matrix

    def CNOT(self, symplectic_matrix, control_q, target_q, nqubits):
        r, x, z = self._get_rxz(symplectic_matrix, nqubits)
        self._set_r(
            symplectic_matrix,
            r
            ^ (x[:, control_q] & z[:, target_q])
            & (x[:, target_q] ^ z[:, control_q] ^ True),
        )

        symplectic_matrix[:-1, target_q] = x[:, target_q] ^ x[:, control_q]
        symplectic_matrix[:-1, nqubits + control_q] = z[:, control_q] ^ z[:, target_q]
        return symplectic_matrix

    def CZ(self, symplectic_matrix, control_q, target_q, nqubits):
        """Decomposition --> H-CNOT-H"""
        r, x, z = self._get_rxz(symplectic_matrix, nqubits)
        self._set_r(
            symplectic_matrix,
            r
            ^ (x[:, target_q] & z[:, target_q])
            ^ (
                x[:, control_q]
                & x[:, target_q]
                & (z[:, target_q] ^ z[:, control_q] ^ True)
            )
            ^ (x[:, target_q] & (z[:, target_q] ^ x[:, control_q])),
        )
        z_control_q = x[:, target_q] ^ z[:, control_q]
        z_target_q = z[:, target_q] ^ x[:, control_q]
        symplectic_matrix[:-1, nqubits + control_q] = z_control_q
        symplectic_matrix[:-1, nqubits + target_q] = z_target_q
        return symplectic_matrix

    def S(self, symplectic_matrix, q, nqubits):
        r, x, z = self._get_rxz(symplectic_matrix, nqubits)
        self._set_r(
            symplectic_matrix,
            r ^ (x[:, q] & z[:, q]),
        )
        symplectic_matrix[:-1, nqubits + q] = z[:, q] ^ x[:, q]
        return symplectic_matrix

    def Z(self, symplectic_matrix, q, nqubits):
        """Decomposition --> S-S"""

        r, x, z = self._get_rxz(symplectic_matrix, nqubits)
        self._set_r(
            symplectic_matrix,
            r ^ ((x[:, q] & z[:, q]) ^ x[:, q] & (z[:, q] ^ x[:, q])),
        )
        return symplectic_matrix

    def X(self, symplectic_matrix, q, nqubits):
        """Decomposition --> H-S-S-H"""

        r, x, z = self._get_rxz(symplectic_matrix, nqubits)
        self._set_r(
            symplectic_matrix,
            r ^ (z[:, q] & (z[:, q] ^ x[:, q])) ^ (z[:, q] & x[:, q]),
        )
        return symplectic_matrix

    def Y(self, symplectic_matrix, q, nqubits):
        """Decomposition --> S-S-H-S-S-H"""  # double check this, cause it should be
        # Y = i * HZHZ --> HSSHSS
        r, x, z = self._get_rxz(symplectic_matrix, nqubits)
        self._set_r(
            symplectic_matrix,
            r ^ (z[:, q] & (z[:, q] ^ x[:, q])) ^ (x[:, q] & (z[:, q] ^ x[:, q])),
        )
        return symplectic_matrix

    def SX(self, symplectic_matrix, q, nqubits):
        """Decomposition --> H-S-H"""

        r, x, z = self._get_rxz(symplectic_matrix, nqubits)
        self._set_r(
            symplectic_matrix,
            r ^ (z[:, q] & (z[:, q] ^ x[:, q])),
        )
        symplectic_matrix[:-1, q] = z[:, q] ^ x[:, q]
        return symplectic_matrix

    def SDG(self, symplectic_matrix, q, nqubits):
        """Decomposition --> S-S-S"""

        r, x, z = self._get_rxz(symplectic_matrix, nqubits)
        self._set_r(
            symplectic_matrix,
            r ^ (x[:, q] & (z[:, q] ^ x[:, q])),
        )
        symplectic_matrix[:-1, nqubits + q] = z[:, q] ^ x[:, q]
        return symplectic_matrix

    def SXDG(self, symplectic_matrix, q, nqubits):
        """Decomposition --> H-S-S-S-H"""

        r, x, z = self._get_rxz(symplectic_matrix, nqubits)
        self._set_r(
            symplectic_matrix,
            r ^ (z[:, q] & x[:, q]),
        )
        symplectic_matrix[:-1, q] = z[:, q] ^ x[:, q]
        return symplectic_matrix

    def RX(self, symplectic_matrix, q, nqubits, theta):
        if theta % (2 * np.pi) == 0:
            return self.I(symplectic_matrix, q, nqubits)
        elif (theta / np.pi - 1) % 2 == 0:
            return self.X(symplectic_matrix, q, nqubits)
        elif (theta / (np.pi / 2) - 1) % 4 == 0:
            return self.SX(symplectic_matrix, q, nqubits)
        else:  # theta == 3*pi/2 + 2*n*pi
            return self.SXDG(symplectic_matrix, q, nqubits)

    def RZ(self, symplectic_matrix, q, nqubits, theta):
        if theta % (2 * np.pi) == 0:
            return self.I(symplectic_matrix, q, nqubits)
        elif (theta / np.pi - 1) % 2 == 0:
            return self.Z(symplectic_matrix, q, nqubits)
        elif (theta / (np.pi / 2) - 1) % 4 == 0:
            return self.S(symplectic_matrix, q, nqubits)
        else:  # theta == 3*pi/2 + 2*n*pi
            return self.SDG(symplectic_matrix, q, nqubits)

    def RY(self, symplectic_matrix, q, nqubits, theta):
        if theta % (2 * np.pi) == 0:
            return self.I(symplectic_matrix, q, nqubits)
        elif (theta / np.pi - 1) % 2 == 0:
            return self.Y(symplectic_matrix, q, nqubits)
        elif (theta / (np.pi / 2) - 1) % 4 == 0:
            """Decomposition --> H-S-S"""

            r, x, z = self._get_rxz(symplectic_matrix, nqubits)
            self._set_r(
                symplectic_matrix,
                r ^ (x[:, q] & (z[:, q] ^ x[:, q])),
            )
            symplectic_matrix[:-1, [nqubits + q, q]] = symplectic_matrix[
                :-1, [q, nqubits + q]
            ]
            return symplectic_matrix
        else:  # theta == 3*pi/2 + 2*n*pi
            """Decomposition --> H-S-S-H-S-S-H-S-S"""

            r, x, z = self._get_rxz(symplectic_matrix, nqubits)
            self._set_r(
                symplectic_matrix,
                r ^ (z[:, q] & (z[:, q] ^ x[:, q])),
            )
            symplectic_matrix[:-1, [nqubits + q, q]] = symplectic_matrix[
                :-1, [q, nqubits + q]
            ]
            return symplectic_matrix

    def SWAP(self, symplectic_matrix, control_q, target_q, nqubits):
        """Decomposition --> CNOT-CNOT-CNOT"""

        r, x, z = self._get_rxz(symplectic_matrix, nqubits)
        self._set_r(
            symplectic_matrix,
            r
            ^ (
                x[:, control_q]
                & z[:, target_q]
                & (x[:, target_q] ^ z[:, control_q] ^ True)
            )
            ^ (
                (x[:, target_q] ^ x[:, control_q])
                & (z[:, target_q] ^ z[:, control_q])
                & (z[:, target_q] ^ x[:, control_q] ^ True)
            )
            ^ (
                x[:, target_q]
                & z[:, control_q]
                & (
                    x[:, control_q]
                    ^ x[:, target_q]
                    ^ z[:, control_q]
                    ^ z[:, target_q]
                    ^ True
                )
            ),
        )
        symplectic_matrix[
            :-1, [control_q, target_q, nqubits + control_q, nqubits + target_q]
        ] = symplectic_matrix[
            :-1, [target_q, control_q, nqubits + target_q, nqubits + control_q]
        ]
        return symplectic_matrix

    def iSWAP(self, symplectic_matrix, control_q, target_q, nqubits):
        """Decomposition --> H-CNOT-CNOT-H-S-S"""

        r, x, z = self._get_rxz(symplectic_matrix, nqubits)
        self._set_r(
            symplectic_matrix,
            r
            ^ (x[:, target_q] & z[:, target_q])
            ^ (x[:, control_q] & z[:, control_q])
            ^ (x[:, control_q] & (z[:, control_q] ^ x[:, control_q]))
            ^ (
                (z[:, control_q] ^ x[:, control_q])
                & (z[:, target_q] ^ x[:, target_q])
                & (x[:, target_q] ^ x[:, control_q] ^ True)
            )
            ^ (
                (x[:, target_q] ^ z[:, control_q] ^ x[:, control_q])
                & (x[:, target_q] ^ z[:, target_q] ^ x[:, control_q])
                & (
                    x[:, target_q]
                    ^ z[:, target_q]
                    ^ x[:, control_q]
                    ^ z[:, control_q]
                    ^ True
                )
            )
            ^ (x[:, control_q] & (x[:, target_q] ^ x[:, control_q] ^ z[:, control_q])),
        )
        z_control_q = x[:, target_q] ^ z[:, target_q] ^ x[:, control_q]
        z_target_q = x[:, target_q] ^ z[:, control_q] ^ x[:, control_q]
        symplectic_matrix[:-1, nqubits + control_q] = z_control_q
        symplectic_matrix[:-1, nqubits + target_q] = z_target_q
        symplectic_matrix[:-1, [control_q, target_q]] = symplectic_matrix[
            :-1, [target_q, control_q]
        ]
        return symplectic_matrix

    def FSWAP(self, symplectic_matrix, control_q, target_q, nqubits):
        """Decomposition --> X-CNOT-RY-CNOT-RY-CNOT-CNOT-X"""
        symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = self.CNOT(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = self.RY(symplectic_matrix, control_q, nqubits, np.pi / 2)
        symplectic_matrix = self.CNOT(symplectic_matrix, target_q, control_q, nqubits)
        symplectic_matrix = self.RY(symplectic_matrix, control_q, nqubits, -np.pi / 2)
        symplectic_matrix = self.CNOT(symplectic_matrix, target_q, control_q, nqubits)
        symplectic_matrix = self.CNOT(symplectic_matrix, control_q, target_q, nqubits)
        return self.X(symplectic_matrix, control_q, nqubits)

    def CY(self, symplectic_matrix, control_q, target_q, nqubits):
        """Decomposition --> S-CNOT-SDG"""

        r, x, z = self._get_rxz(symplectic_matrix, nqubits)
        self._set_r(
            symplectic_matrix,
            r
            ^ (x[:, target_q] & (z[:, target_q] ^ x[:, target_q]))
            ^ (
                x[:, control_q]
                & (x[:, target_q] ^ z[:, target_q])
                & (z[:, control_q] ^ x[:, target_q] ^ True)
            )
            ^ ((x[:, target_q] ^ x[:, control_q]) & (z[:, target_q] ^ x[:, target_q])),
        )

        x_target_q = x[:, control_q] ^ x[:, target_q]
        z_control_q = z[:, control_q] ^ z[:, target_q] ^ x[:, target_q]
        z_target_q = z[:, target_q] ^ x[:, control_q]
        symplectic_matrix[:-1, target_q] = x_target_q
        symplectic_matrix[:-1, nqubits + control_q] = z_control_q
        symplectic_matrix[:-1, nqubits + target_q] = z_target_q
        return symplectic_matrix

    def CRX(self, symplectic_matrix, control_q, target_q, nqubits, theta):
        # theta = 4 * n * pi
        if theta % (4 * np.pi) == 0:
            return self.I(symplectic_matrix, target_q, nqubits)
        # theta = pi + 4 * n * pi
        elif (theta / np.pi - 1) % 4 == 0:
            symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
            symplectic_matrix = self.CZ(symplectic_matrix, control_q, target_q, nqubits)
            symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
            return self.CY(symplectic_matrix, control_q, target_q, nqubits)
        # theta = 2 * pi + 4 * n * pi
        elif (theta / (2 * np.pi) - 1) % 2 == 0:
            symplectic_matrix = self.CZ(symplectic_matrix, control_q, target_q, nqubits)
            symplectic_matrix = self.Y(symplectic_matrix, target_q, nqubits)
            symplectic_matrix = self.CZ(symplectic_matrix, control_q, target_q, nqubits)
            return self.Y(symplectic_matrix, target_q, nqubits)
        # theta = 3 * pi + 4 * n * pi
        elif (theta / np.pi - 3) % 4 == 0:
            symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
            symplectic_matrix = self.CY(symplectic_matrix, control_q, target_q, nqubits)
            symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
            return self.CZ(symplectic_matrix, control_q, target_q, nqubits)

    def CRZ(self, symplectic_matrix, control_q, target_q, nqubits, theta):
        # theta = 4 * n * pi
        if theta % (4 * np.pi) == 0:
            return self.I(symplectic_matrix, target_q, nqubits)
        # theta = pi + 4 * n * pi
        elif (theta / np.pi - 1) % 4 == 0:
            symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
            symplectic_matrix = self.CY(symplectic_matrix, control_q, target_q, nqubits)
            symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
            return self.CNOT(symplectic_matrix, control_q, target_q, nqubits)
        # theta = 2 * pi + 4 * n * pi
        elif (theta / (2 * np.pi) - 1) % 2 == 0:
            symplectic_matrix = self.CZ(symplectic_matrix, control_q, target_q, nqubits)
            symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
            symplectic_matrix = self.CZ(symplectic_matrix, control_q, target_q, nqubits)
            return self.X(symplectic_matrix, target_q, nqubits)
        # theta = 3 * pi + 4 * n * pi
        elif (theta / np.pi - 3) % 4 == 0:
            symplectic_matrix = self.CNOT(
                symplectic_matrix, control_q, target_q, nqubits
            )
            symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
            symplectic_matrix = self.CY(symplectic_matrix, control_q, target_q, nqubits)
            return self.X(symplectic_matrix, target_q, nqubits)

    def CRY(self, symplectic_matrix, control_q, target_q, nqubits, theta):
        # theta = 4 * n * pi
        if theta % (4 * np.pi) == 0:
            return self.I(symplectic_matrix, target_q, nqubits)
        # theta = pi + 4 * n * pi
        elif (theta / np.pi - 1) % 4 == 0:
            symplectic_matrix = self.Z(symplectic_matrix, target_q, nqubits)
            symplectic_matrix = self.CNOT(
                symplectic_matrix, control_q, target_q, nqubits
            )
            symplectic_matrix = self.Z(symplectic_matrix, target_q, nqubits)
            return self.CZ(symplectic_matrix, control_q, target_q, nqubits)
        # theta = 2 * pi + 4 * n * pi
        elif (theta / (2 * np.pi) - 1) % 2 == 0:
            return self.CRZ(symplectic_matrix, control_q, target_q, nqubits, theta)
        # theta = 3 * pi + 4 * n * pi
        elif (theta / np.pi - 3) % 4 == 0:
            symplectic_matrix = self.CZ(symplectic_matrix, control_q, target_q, nqubits)
            symplectic_matrix = self.Z(symplectic_matrix, target_q, nqubits)
            symplectic_matrix = self.CNOT(
                symplectic_matrix, control_q, target_q, nqubits
            )
            return self.Z(symplectic_matrix, target_q, nqubits)

    def ECR(self, symplectic_matrix, control_q, target_q, nqubits):
        symplectic_matrix = self.S(symplectic_matrix, control_q, nqubits)
        symplectic_matrix = self.SX(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = self.CNOT(symplectic_matrix, control_q, target_q, nqubits)
        return self.X(symplectic_matrix, control_q, nqubits)

    # valid for a standard basis measurement only
    def M(self, state, qubits, nqubits, collapse=False):
        sample = []
        state_copy = state if collapse else state.copy()
        for q in qubits:
            x = CliffordOperations._get_x(state_copy, nqubits)
            p = x[nqubits:, q].nonzero()[0]
            # random outcome, affects the state
            if len(p) > 0:
                p = p[0].item() + nqubits
                h = self.np.array(
                    [i for i in x[:, q].nonzero()[0] if i != p], dtype=int
                )
                if h.shape[0] > 0:
                    state_copy = self._rowsum(
                        state_copy, h, p * self.np.ones(h.shape[0], dtype=int), nqubits
                    )
                state_copy[p - nqubits, :] = state_copy[p, :]
                outcome = self.np.random.randint(2, size=1).item()
                state_copy[p, :] = 0
                state_copy[p, -1] = outcome
                state_copy[p, nqubits + q] = 1
                sample.append(outcome)
            # determined outcome, state unchanged
            else:
                CliffordOperations._set_scratch(state_copy, 0)
                for i in x[:nqubits, q].nonzero()[0]:
                    state_copy = self._rowsum(
                        state_copy,
                        self.np.array([2 * nqubits]),
                        self.np.array([i + nqubits]),
                        nqubits,
                        include_scratch=True,
                    )
                sample.append(int(CliffordOperations._get_scratch(state_copy)[-1]))
        return sample

    @staticmethod
    def _get_r(symplectic_matrix, nqubits, include_scratch=False):
        return symplectic_matrix[
            : -1 + (2 * nqubits + 2) * int(include_scratch),
            -1,
        ]

    @staticmethod
    def _set_r(symplectic_matrix, val):
        symplectic_matrix[:-1, -1] = val

    @staticmethod
    def _get_x(symplectic_matrix, nqubits, include_scratch=False):
        return symplectic_matrix[
            : -1 + (2 * nqubits + 2) * int(include_scratch), :nqubits
        ]

    @staticmethod
    def _get_z(symplectic_matrix, nqubits, include_scratch=False):
        return symplectic_matrix[
            : -1 + (2 * nqubits + 2) * int(include_scratch), nqubits:-1
        ]

    @staticmethod
    def _get_rxz(symplectic_matrix, nqubits, include_scratch=False):
        return (
            CliffordOperations._get_r(symplectic_matrix, nqubits, include_scratch),
            CliffordOperations._get_x(symplectic_matrix, nqubits, include_scratch),
            CliffordOperations._get_z(symplectic_matrix, nqubits, include_scratch),
        )

    @staticmethod
    def _get_scratch(symplectic_matrix):
        return symplectic_matrix[-1, :]

    @staticmethod
    def _set_scratch(symplectic_matrix, val):
        symplectic_matrix[-1, :] = val

    def _exponent(self, x1, z1, x2, z2):
        """Computes the exponent to which i is raised for the product of the x and z Paulis encoded in the symplectic matrix.

        This is used in _rowsum.
        The computation is performed parallely over the separated Paulis
        x1[i], z1[i], x2[i] and z2[i].

        Args:
            x1 (ndarray): Bits of the first x Paulis.
            z1 (ndarray): Bits of the first z Paulis.
            x2 (ndarray): Bits of the second x Paulis.
            z2 (ndarray): Bits of the second z Paulis.

        Returns:
            (ndarray): Calculated exponents.
        """
        exp = self.np.zeros(x1.shape, dtype=int)
        x1_eq_z1 = (x1 ^ z1) == 0
        x1_neq_z1 = x1_eq_z1 ^ True
        x1_eq_0 = x1 == 0
        x1_eq_1 = x1 == 1
        ind2 = x1_eq_z1 & x1_eq_1
        ind3 = x1_eq_1 & x1_neq_z1
        ind4 = x1_eq_0 & x1_neq_z1
        exp[ind2] = z2[ind2].astype(int) - x2[ind2].astype(int)
        exp[ind3] = z2[ind3].astype(int) * (2 * x2[ind3].astype(int) - 1)
        exp[ind4] = x2[ind4].astype(int) * (1 - 2 * z2[ind4].astype(int))

        return exp

    def _rowsum(self, symplectic_matrix, h, i, nqubits, include_scratch: bool = False):
        """Updates the symplectic matrix by setting the h-th generator equal to the (i+h)-th one. This is done to keep track of the phase of the h-th row of the symplectic matrix (r[h]). The function is applied parallely over all the rows h and i passed.

        Args:
            symplectic_matrix (np.array): Input symplectic matrix.
            h (np.array): Indices of the rows encoding the generators to update.
            i (np.array): Indices of the rows encoding the generators to use.
            nqubits (int): Total number of qubits.
            include_scratch (bool): If ``True`` the scratch row (last row of the symplectic matrix) is inlcuded in the computation as well.

        Returns:
            (np.array): The updated symplectic matrix.
        """
        x, z = self._get_x(symplectic_matrix, nqubits, include_scratch), self._get_z(
            symplectic_matrix, nqubits, include_scratch
        )
        exponents = self._exponent(x[i, :], z[i, :], x[h, :], z[h, :])
        ind = (
            2 * symplectic_matrix[h, -1]
            + 2 * symplectic_matrix[i, -1]
            + self.np.sum(exponents, axis=-1)
        ) % 4 == 0
        r = self.np.ones(h.shape[0], dtype=bool)
        r[ind] = False

        symplectic_matrix[h, -1] = r
        symplectic_matrix[h, :nqubits] = x[i, :] ^ x[h, :]
        symplectic_matrix[h, nqubits:-1] = z[i, :] ^ z[h, :]

        return symplectic_matrix


class CliffordBackend(NumpyBackend):
    """Backend for the simulation of Clifford circuits following `Aaronson & Gottesman (2004) <https://arxiv.org/abs/quant-ph/0406196>`_.

    Args:
        engine (:class:`qibo.backends.abstract.Backend`): Backend used for the calculation.
    """

    def __init__(self, engine=None):
        super().__init__()

        if engine is None:
            from qibo.backends import GlobalBackend

            engine = GlobalBackend()

        if isinstance(engine, TensorflowBackend):
            raise_error(
                NotImplementedError,
                "TensorflowBackend for Clifford Simulation is not supported.",
            )

        self.engine = engine
        self.np = _calculation_engine(engine)

        self.name = "clifford"
        self.clifford_operations = CliffordOperations(engine)

    def cast(self, x, dtype=None, copy: bool = False):
        """Cast an object as the array type of the current backend.

        Args:
            x: Object to cast to array.
            copy (bool, optional): If ``True`` a copy of the object is created in memory.
                Defaults to ``False``.
        """
        return self.engine.cast(x, dtype=dtype, copy=copy)

    def calculate_frequencies(self, samples):
        res, counts = self.engine.np.unique(samples, return_counts=True)
        # The next two lines are necessary for the GPU backends
        res = [int(r) if not isinstance(r, str) else r for r in res]
        counts = [int(v) for v in counts]

        return collections.Counter({k: v for k, v in zip(res, counts)})

    def zero_state(self, nqubits: int):
        """Construct the zero state |00...00>.

        Args:
            nqubits (int): Number of qubits.

        Returns:
            (ndarray): Symplectic matrix for the zero state.
        """
        I = self.np.eye(nqubits)
        symplectic_matrix = self.np.zeros(
            (2 * nqubits + 1, 2 * nqubits + 1), dtype=bool
        )
        symplectic_matrix[:nqubits, :nqubits] = self.np.copy(I)
        symplectic_matrix[nqubits:-1, nqubits : 2 * nqubits] = self.np.copy(I)
        return symplectic_matrix

    def apply_gate_clifford(self, gate, symplectic_matrix, nqubits):
        operation = getattr(self.clifford_operations, gate.__class__.__name__)
        kwargs = (
            {"theta": gate.init_kwargs["theta"]} if "theta" in gate.init_kwargs else {}
        )

        return operation(symplectic_matrix, *gate.init_args, nqubits, **kwargs)

    def execute_circuit(self, circuit, initial_state=None, nshots: int = 1000):
        """Execute a Clifford circuits.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Input circuit.
            initial_state (ndarray, optional): The ``symplectic_matrix`` of the initial state.
                If ``None``, defaults to the zero state. Defaults to ``None``.
            nshots (int, optional): Number of shots to perform if ``circuit`` has measurements.
                Defaults to :math:`10^{3}`.

        Returns:
            (:class:`qibo.quantum_info.clifford.Clifford`): Object giving access to the final results.
        """
        for gate in circuit.queue:
            if not gate.clifford and not gate.__class__.__name__ == "M":
                raise_error(RuntimeError, "Circuit contains non-Clifford gates.")

        if circuit.repeated_execution and not nshots == 1:
            return self.execute_circuit_repeated(circuit, initial_state, nshots)

        try:
            from qibo.quantum_info.clifford import Clifford

            nqubits = circuit.nqubits

            state = self.zero_state(nqubits) if initial_state is None else initial_state

            for gate in circuit.queue:
                state = gate.apply_clifford(self, state, nqubits)

            clifford = Clifford(
                state,
                measurements=circuit.measurements,
                nshots=nshots,
                engine=self.engine,
            )

            return clifford

        except self.oom_error:  # pragma: no cover
            raise_error(
                RuntimeError,
                f"State does not fit in {self.device} memory."
                "Please switch the execution device to a "
                "different one using ``qibo.set_device``.",
            )

    def execute_circuit_repeated(self, circuit, initial_state=None, nshots: int = 1000):
        """Execute a Clifford circuits ``nshots`` times.

        This is used for all the simulations that involve repeated execution.
        For instance when collapsing measurement or noise channels are present.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): input circuit.
            initial_state (ndarray, optional): Symplectic_matrix of the initial state.
                If ``None``, defaults to :meth:`qibo.backends.clifford.CliffordBackend.zero_state`.
                Defaults to ``None``.
            nshots (int, optional): Number of times to repeat the execution.
                Defaults to :math:`1000`.

        Returns:
            (:class:`qibo.quantum_info.clifford.Clifford`): Object giving access to the final results.
        """
        circuit_copy = circuit.copy()
        samples = []
        for _ in range(nshots):
            res = self.execute_circuit(circuit_copy, initial_state, nshots=1)
            [m.result.reset() for m in circuit_copy.measurements]
            samples.append(res.samples())
        samples = self.np.vstack(samples)

        for m in circuit.measurements:
            m.result.register_samples(samples[:, m.target_qubits], self)

        from qibo.quantum_info.clifford import Clifford

        result = Clifford(
            self.zero_state(circuit.nqubits),
            measurements=circuit.measurements,
            nshots=nshots,
            engine=self.engine,
        )
        result.symplectic_matrix, result._samples = None, None

        return result

    def sample_shots(
        self,
        state,
        qubits: Union[tuple, list],
        nqubits: int,
        nshots: int,
        collapse: bool = False,
    ):
        """Sample shots by measuring the selected qubits from the provided symplectic matrix of a ``state``.

        Args:
            state (ndarray): symplectic matrix from which to sample shots from.
            qubits: (tuple or list): qubits to measure.
            nqubits (int): total number of qubits of the state.
            nshots (int): number of shots to sample.
            collapse (bool, optional): If ``True`` the input state is going to be
                collapsed with the last shot. Defaults to ``False``.

        Returns:
            (ndarray): Samples shots.
        """
        if isinstance(qubits, list):
            qubits = tuple(qubits)

        operation = CliffordOperations(self.engine)
        if collapse:
            samples = [
                operation.M(state, qubits, nqubits) for _ in range(nshots - 1)
            ]  # parallelize?
            samples.append(operation.M(state, qubits, nqubits, collapse))
        else:
            samples = [
                operation.M(state, qubits, nqubits) for _ in range(nshots)
            ]  # parallelize?

        return self.np.array(samples).reshape(nshots, len(qubits)).tolist()

    def symplectic_matrix_to_generators(
        self, symplectic_matrix, return_array: bool = False
    ):
        """Extract both the stabilizers and destabilizers generators from the input symplectic matrix.

        Args:
            symplectic_matrix (ndarray): The input symplectic_matrix.
            return_array (bool, optional): If ``True`` returns the generators as ``ndarrays``.
                If ``False``, generators are returned as strings. Defaults to ``False``.

        Returns:
            (list, list): Extracted generators and their corresponding phases, respectively.
        """
        bits_to_gate = {"00": "I", "01": "X", "10": "Z", "11": "Y"}

        nqubits = int((symplectic_matrix.shape[1] - 1) / 2)
        phases = (-1) ** symplectic_matrix[:-1, -1]
        tmp = 1 * symplectic_matrix[:-1, :-1]
        X, Z = tmp[:, :nqubits], tmp[:, nqubits:]
        generators = []
        for x, z in zip(X, Z):
            paulis = [bits_to_gate[f"{zz}{xx}"] for xx, zz in zip(x, z)]
            if return_array:
                paulis = [self.cast(getattr(gates, p)(0).matrix()) for p in paulis]
                matrix = reduce(self.np.kron, paulis)
                generators.append(matrix)
            else:
                generators.append("".join(paulis))

        if return_array:
            generators = self.cast(generators)

        return generators, phases
