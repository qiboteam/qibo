"""Module defining the matrix representation of gates used by the backends."""

import cmath
import math
from functools import cached_property

from qibo.config import raise_error


class NumpyMatrices:
    """Matrix representation of every gate as a numpy array."""

    def __init__(self, dtype):
        import numpy as np

        self.dtype = dtype
        self.engine = np

    def _cast(self, array, dtype):
        if isinstance(array, list):
            return self.engine.array(array, dtype=dtype)
        return array.astype(dtype)

    @cached_property
    def H(self):
        return self._cast([[1, 1], [1, -1]], dtype=self.dtype) / math.sqrt(2)

    @cached_property
    def X(self):
        return self._cast([[0, 1], [1, 0]], dtype=self.dtype)

    @cached_property
    def Y(self):
        return self._cast([[0j, -1j], [1j, 0j]], dtype=self.dtype)

    @cached_property
    def Z(self):
        return self._cast([[1, 0], [0, -1]], dtype=self.dtype)

    @cached_property
    def SX(self):
        return self._cast([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=self.dtype) / 2

    @cached_property
    def SXDG(self):
        return self._cast([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=self.dtype) / 2

    @cached_property
    def S(self):
        return self._cast([[1 + 0j, 0j], [0j, 1j]], dtype=self.dtype)

    @cached_property
    def SDG(self):
        return self._cast([[1 + 0j, 0j], [0j, -1j]], dtype=self.dtype)

    @cached_property
    def T(self):
        return self._cast(
            [[1 + 0j, 0], [0, cmath.exp(1j * math.pi / 4.0)]], dtype=self.dtype
        )

    @cached_property
    def TDG(self):
        return self._cast(
            [[1 + 0j, 0], [0, cmath.exp(-1j * math.pi / 4.0)]], dtype=self.dtype
        )

    def I(self, n=2):
        return self.engine.eye(n, dtype=self.dtype)

    def Align(self, delay, n=2):
        return self.I(n)

    def M(self):  # pragma: no cover
        raise_error(NotImplementedError)

    def RX(self, theta):
        cos = self.engine.cos(theta / 2.0) + 0j
        isin = -1j * self.engine.sin(theta / 2.0)
        return self._cast([[cos, isin], [isin, cos]], dtype=self.dtype)

    def RY(self, theta):
        cos = self.engine.cos(theta / 2.0) + 0j
        sin = self.engine.sin(theta / 2.0) + 0j
        return self._cast([[cos, -sin], [sin, cos]], dtype=self.dtype)

    def RZ(self, theta):
        phase = self.engine.exp(0.5j * theta)
        return self._cast([[self.engine.conj(phase), 0], [0, phase]], dtype=self.dtype)

    def PRX(self, theta, phi):
        cos = self.engine.cos(theta / 2)
        sin = self.engine.sin(theta / 2)
        exponent1 = -1.0j * self.engine.exp(-1.0j * phi)
        exponent2 = -1.0j * self.engine.exp(1.0j * phi)
        # The +0j is needed because of tensorflow casting issues
        return self._cast(
            [[cos + 0j, exponent1 * sin], [exponent2 * sin, cos + 0j]], dtype=self.dtype
        )

    def GPI(self, phi):
        phase = self.engine.exp(1.0j * phi)
        return self._cast([[0, self.engine.conj(phase)], [phase, 0]], dtype=self.dtype)

    def GPI2(self, phi):
        phase = self.engine.exp(1.0j * phi)
        return self._cast(
            [[1, -1.0j * self.engine.conj(phase)], [-1.0j * phase, 1]], dtype=self.dtype
        ) / math.sqrt(2)

    def U1(self, theta):
        phase = self.engine.exp(1j * theta)
        return self._cast([[1, 0], [0, phase]], dtype=self.dtype)

    def U2(self, phi, lam):
        eplus = self.engine.exp(1j * (phi + lam) / 2.0)
        eminus = self.engine.exp(1j * (phi - lam) / 2.0)
        return self._cast(
            [[self.engine.conj(eplus), -self.engine.conj(eminus)], [eminus, eplus]],
            dtype=self.dtype,
        ) / math.sqrt(2)

    def U3(self, theta, phi, lam):
        cost = self.engine.cos(theta / 2)
        sint = self.engine.sin(theta / 2)
        eplus = self.engine.exp(1j * (phi + lam) / 2.0)
        eminus = self.engine.exp(1j * (phi - lam) / 2.0)
        return self._cast(
            [
                [self.engine.conj(eplus) * cost, -self.engine.conj(eminus) * sint],
                [eminus * sint, eplus * cost],
            ],
            dtype=self.dtype,
        )

    def U1q(self, theta, phi):
        return self._cast(
            self.U3(theta, phi - math.pi / 2, math.pi / 2 - phi), dtype=self.dtype
        )

    @cached_property
    def CNOT(self):
        return self._cast(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=self.dtype
        )

    @cached_property
    def CY(self):
        return self._cast(
            [
                [1 + 0j, 0j, 0j, 0j],
                [0j, 1 + 0j, 0j, 0j],
                [0j, 0j, 0j, -1j],
                [0j, 0j, 1j, 0j],
            ],
            dtype=self.dtype,
        )

    @cached_property
    def CZ(self):
        return self._cast(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=self.dtype
        )

    @cached_property
    def CSX(self):
        a = (1 + 1j) / 2
        b = (1 - 1j) / 2
        return self._cast(
            [
                [1 + 0j, 0, 0, 0],
                [0, 1 + 0j, 0, 0],
                [0, 0, a, b],
                [0, 0, b, a],
            ],
            dtype=self.dtype,
        )

    @cached_property
    def CSXDG(self):
        a = (1 + 1j) / 2
        b = (1 - 1j) / 2
        return self._cast(
            [
                [1 + 0j, 0, 0, 0],
                [0, 1 + 0j, 0, 0],
                [0, 0, b, a],
                [0, 0, a, b],
            ],
            dtype=self.dtype,
        )

    def CRX(self, theta):
        cos = self.engine.cos(theta / 2.0) + 0j
        isin = -1j * self.engine.sin(theta / 2.0)
        matrix = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, cos, isin],
            [0, 0, isin, cos],
        ]
        return self._cast(matrix, dtype=self.dtype)

    def CRY(self, theta):
        cos = self.engine.cos(theta / 2.0) + 0j
        sin = self.engine.sin(theta / 2.0) + 0j
        matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, cos, -sin], [0, 0, sin, cos]]
        return self._cast(matrix, dtype=self.dtype)

    def CRZ(self, theta):
        phase = self.engine.exp(0.5j * theta)
        matrix = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, self.engine.conj(phase), 0],
            [0, 0, 0, phase],
        ]
        return self._cast(matrix, dtype=self.dtype)

    def CU1(self, theta):
        phase = self.engine.exp(1j * theta)
        matrix = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, phase],
        ]
        return self._cast(matrix, dtype=self.dtype)

    def CU2(self, phi, lam):
        eplus = self.engine.exp(1j * (phi + lam) / 2.0) / math.sqrt(2)
        eminus = self.engine.exp(1j * (phi - lam) / 2.0) / math.sqrt(2)
        matrix = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, self.engine.conj(eplus), -self.engine.conj(eminus)],
            [0, 0, eminus, eplus],
        ]
        return self._cast(matrix, dtype=self.dtype)

    def CU3(self, theta, phi, lam):
        cost = self.engine.cos(theta / 2)
        sint = self.engine.sin(theta / 2)
        eplus = self.engine.exp(1j * (phi + lam) / 2.0)
        eminus = self.engine.exp(1j * (phi - lam) / 2.0)
        matrix = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, self.engine.conj(eplus) * cost, -self.engine.conj(eminus) * sint],
            [0, 0, eminus * sint, eplus * cost],
        ]
        return self._cast(matrix, dtype=self.dtype)

    @cached_property
    def SWAP(self):
        return self._cast(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=self.dtype
        )

    @cached_property
    def iSWAP(self):
        return self._cast(
            [
                [1 + 0j, 0j, 0j, 0j],
                [0j, 0j, 1j, 0j],
                [0j, 1j, 0j, 0j],
                [0j, 0j, 0j, 1 + 0j],
            ],
            dtype=self.dtype,
        )

    @cached_property
    def SiSWAP(self):
        return self._cast(
            [
                [1 + 0j, 0j, 0j, 0j],
                [0j, 1 / math.sqrt(2) + 0j, 1j / math.sqrt(2), 0j],
                [0j, 1j / math.sqrt(2), 1 / math.sqrt(2) + 0j, 0j],
                [0j, 0j, 0j, 1 + 0j],
            ],
            dtype=self.dtype,
        )

    @cached_property
    def SiSWAPDG(self):
        return self._cast(
            [
                [1 + 0j, 0j, 0j, 0j],
                [0j, 1 / math.sqrt(2) + 0j, -1j / math.sqrt(2), 0j],
                [0j, -1j / math.sqrt(2), 1 / math.sqrt(2) + 0j, 0j],
                [0j, 0j, 0j, 1 + 0j],
            ],
            dtype=self.dtype,
        )

    @cached_property
    def FSWAP(self):
        return self._cast(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -1]], dtype=self.dtype
        )

    def fSim(self, theta, phi):
        cost = self.engine.cos(theta) + 0j
        isint = -1j * self.engine.sin(theta)
        phase = self.engine.exp(-1j * phi)
        return self._cast(
            [
                [1, 0, 0, 0],
                [0, cost, isint, 0],
                [0, isint, cost, 0],
                [0, 0, 0, phase],
            ],
            dtype=self.dtype,
        )

    @cached_property
    def SYC(self):
        cost = math.cos(math.pi / 2) + 0j
        isint = -1j * math.sin(math.pi / 2)
        phase = cmath.exp(-1j * math.pi / 6)
        return self._cast(
            [
                [1 + 0j, 0, 0, 0],
                [0, cost, isint, 0],
                [0, isint, cost, 0],
                [0, 0, 0, phase],
            ],
            dtype=self.dtype,
        )

    def GeneralizedfSim(self, u, phi):
        phase = self.engine.exp(-1j * phi)
        return self._cast(
            [
                [1, 0, 0, 0],
                [0, complex(u[0, 0]), complex(u[0, 1]), 0],
                [0, complex(u[1, 0]), complex(u[1, 1]), 0],
                [0, 0, 0, phase],
            ],
            dtype=self.dtype,
        )

    def RXX(self, theta):
        cos = self.engine.cos(theta / 2.0) + 0j
        isin = -1j * self.engine.sin(theta / 2.0)
        return self._cast(
            [
                [cos, 0, 0, isin],
                [0, cos, isin, 0],
                [0, isin, cos, 0],
                [isin, 0, 0, cos],
            ],
            dtype=self.dtype,
        )

    def RYY(self, theta):
        cos = self.engine.cos(theta / 2.0) + 0j
        isin = -1j * self.engine.sin(theta / 2.0)
        return self._cast(
            [
                [cos, 0, 0, -isin],
                [0, cos, isin, 0],
                [0, isin, cos, 0],
                [-isin, 0, 0, cos],
            ],
            dtype=self.dtype,
        )

    def RZZ(self, theta):
        phase = self.engine.exp(0.5j * theta)
        return self._cast(
            [
                [self.engine.conj(phase), 0, 0, 0],
                [0, phase, 0, 0],
                [0, 0, phase, 0],
                [0, 0, 0, self.engine.conj(phase)],
            ],
            dtype=self.dtype,
        )

    def RZX(self, theta):
        cos, sin = self.engine.cos(theta / 2) + 0j, self.engine.sin(theta / 2) + 0j
        return self._cast(
            [
                [cos, -1j * sin, 0, 0],
                [-1j * sin, cos, 0, 0],
                [0, 0, cos, 1j * sin],
                [0, 0, 1j * sin, cos],
            ],
            dtype=self.dtype,
        )

    def RXXYY(self, theta):
        cos, sin = self.engine.cos(theta / 2) + 0j, self.engine.sin(theta / 2) + 0j
        return self._cast(
            [
                [1, 0, 0, 0],
                [0, cos, -1j * sin, 0],
                [0, -1j * sin, cos, 0],
                [0, 0, 0, 1],
            ],
            dtype=self.dtype,
        )

    def MS(self, phi0, phi1, theta):
        plus = self.engine.exp(1.0j * (phi0 + phi1))
        minus = self.engine.exp(1.0j * (phi0 - phi1))
        cos = self.engine.cos(theta / 2) + 0j
        sin = self.engine.sin(theta / 2) + 0j
        return self._cast(
            [
                [cos, 0, 0, -1.0j * self.engine.conj(plus) * sin],
                [0, cos, -1.0j * self.engine.conj(minus) * sin, 0],
                [0, -1.0j * minus * sin, cos, 0],
                [-1.0j * plus * sin, 0, 0, cos],
            ],
            dtype=self.dtype,
        )

    def GIVENS(self, theta):
        return self._cast(
            [
                [1, 0, 0, 0],
                [0, self.engine.cos(theta), -self.engine.sin(theta), 0],
                [0, self.engine.sin(theta), self.engine.cos(theta), 0],
                [0, 0, 0, 1],
            ],
            dtype=self.dtype,
        )

    def RBS(self, theta):
        return self.GIVENS(-theta)

    @cached_property
    def ECR(self):
        return self._cast(
            [
                [0j, 0j, 1 + 0j, 1j],
                [0j, 0j, 1j, 1 + 0j],
                [1 + 0j, -1j, 0j, 0j],
                [-1j, 1 + 0j, 0j, 0j],
            ],
            dtype=self.dtype,
        ) / math.sqrt(2)

    @cached_property
    def TOFFOLI(self):
        return self._cast(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=self.dtype,
        )

    @cached_property
    def CCZ(self):
        return self._cast(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, -1],
            ],
            dtype=self.dtype,
        )

    def DEUTSCH(self, theta):
        sin = self.engine.sin(theta) + 0j  # 0j necessary for right tensorflow dtype
        cos = self.engine.cos(theta) + 0j
        return self._cast(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1j * cos, sin],
                [0, 0, 0, 0, 0, 0, sin, 1j * cos],
            ],
            dtype=self.dtype,
        )

    def FanOut(self, *q):
        # TODO: remove this loop after refactoring Gate.matrix method.

        # based on Backend.matrix_fused method
        rank = len(q)
        matrix = self.I(2**rank)

        gmatrix = self.CNOT
        eye = self.I(2 ** (rank - 2))
        gmatrix = self.engine.kron(gmatrix, eye)
        original_shape = gmatrix.shape
        gmatrix = self.engine.reshape(gmatrix, 2 * rank * (2,))

        for qubit in range(1, rank):
            qubits = [0, qubit]
            indices = qubits + [qub for qub in range(rank) if qub not in qubits]
            indices = self.engine.argsort(indices)
            transpose_indices = list(indices)
            transpose_indices.extend(indices + rank)
            _gmatrix = self.engine.transpose(gmatrix, transpose_indices)
            _gmatrix = self.engine.reshape(_gmatrix, original_shape)
            matrix = _gmatrix @ matrix

        return self._cast(matrix, dtype=self.dtype)

    def GeneralizedRBS(self, qubits_in, qubits_out, theta, phi):
        num_qubits_in, num_qubits_out = len(qubits_in), len(qubits_out)
        bitstring_length = num_qubits_in + num_qubits_out

        matrix = [
            [1 + 0j if l == k else 0j for l in range(2**bitstring_length)]
            for k in range(2**bitstring_length)
        ]
        exp, sin, cos = self.engine.exp(1j * phi), self.engine.sin(theta), self.engine.cos(theta)

        integer_in = int("1" * num_qubits_in + "0" * num_qubits_out, base=2)
        integer_out = int("0" * num_qubits_in + "1" * num_qubits_out, base=2)
        matrix[integer_in][integer_in] = exp * cos
        matrix[integer_in][integer_out] = -exp * sin
        matrix[integer_out][integer_in] = self.engine.conj(exp) * sin
        matrix[integer_out][integer_out] = self.engine.conj(exp) * cos

        return self._cast(matrix, dtype=self.dtype)

    def Unitary(self, u):
        return self.engine.asarray(u, dtype=self.dtype)

    def CallbackGate(self):  # pragma: no cover
        raise_error(NotImplementedError)

    def PartialTrace(self):  # pragma: no cover
        raise_error(NotImplementedError)

    def UnitaryChannel(self):  # pragma: no cover
        raise_error(NotImplementedError)

    def PauliNoiseChannel(self):  # pragma: no cover
        raise_error(NotImplementedError)

    def ResetChannel(self):  # pragma: no cover
        raise_error(NotImplementedError)

    def ThermalRelaxationChannel(self):  # pragma: no cover
        raise_error(NotImplementedError)

    def FusedGate(self):  # pragma: no cover
        raise_error(NotImplementedError)
