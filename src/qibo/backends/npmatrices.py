import sys

from qibo.config import raise_error

if sys.version_info.minor >= 8:
    from functools import cached_property  # pylint: disable=E0611
else:  # pragma: no cover
    # Custom ``cached_property`` because it is not available for Python < 3.8
    from functools import lru_cache

    def cached_property(func):
        @property
        def wrapper(self):
            return lru_cache()(func)(self)

        return wrapper


class NumpyMatrices:
    """Matrix representation of every gate as a numpy array."""

    def __init__(self, dtype):
        import numpy as np

        self.dtype = dtype
        self.np = np

    @cached_property
    def H(self):
        return self.np.array([[1, 1], [1, -1]], dtype=self.dtype) / self.np.sqrt(2)

    @cached_property
    def X(self):
        return self.np.array([[0, 1], [1, 0]], dtype=self.dtype)

    @cached_property
    def Y(self):
        return self.np.array([[0, -1j], [1j, 0]], dtype=self.dtype)

    @cached_property
    def Z(self):
        return self.np.array([[1, 0], [0, -1]], dtype=self.dtype)

    @cached_property
    def SX(self):
        return self.np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=self.dtype) / 2

    @cached_property
    def SXDG(self):
        return self.np.transpose(self.np.conj(self.SX))

    @cached_property
    def S(self):
        return self.np.array([[1, 0], [0, 1j]], dtype=self.dtype)

    @cached_property
    def SDG(self):
        return self.np.conj(self.S)

    @cached_property
    def T(self):
        return self.np.array(
            [[1, 0], [0, self.np.exp(1j * self.np.pi / 4.0)]], dtype=self.dtype
        )

    @cached_property
    def TDG(self):
        return self.np.conj(self.T)

    def I(self, n=2):
        return self.np.eye(n, dtype=self.dtype)

    def Align(self, n=2):
        return self.I(n)

    def M(self):  # pragma: no cover
        raise_error(NotImplementedError)

    def RX(self, theta):
        cos = self.np.cos(theta / 2.0) + 0j
        isin = -1j * self.np.sin(theta / 2.0)
        return self.np.array([[cos, isin], [isin, cos]], dtype=self.dtype)

    def RY(self, theta):
        cos = self.np.cos(theta / 2.0) + 0j
        sin = self.np.sin(theta / 2.0)
        return self.np.array([[cos, -sin], [sin, cos]], dtype=self.dtype)

    def RZ(self, theta):
        phase = self.np.exp(0.5j * theta)
        return self.np.array([[self.np.conj(phase), 0], [0, phase]], dtype=self.dtype)

    def GPI(self, phi):
        phase = self.np.exp(1.0j * phi)
        return self.np.array([[0, self.np.conj(phase)], [phase, 0]], dtype=self.dtype)

    def GPI2(self, phi):
        phase = self.np.exp(1.0j * phi)
        return self.np.array(
            [[1, -1.0j * self.np.conj(phase)], [-1.0j * phase, 1]], dtype=self.dtype
        ) / self.np.sqrt(2)

    def U1(self, theta):
        phase = self.np.exp(1j * theta)
        return self.np.array([[1, 0], [0, phase]], dtype=self.dtype)

    def U2(self, phi, lam):
        eplus = self.np.exp(1j * (phi + lam) / 2.0)
        eminus = self.np.exp(1j * (phi - lam) / 2.0)
        return self.np.array(
            [[self.np.conj(eplus), -self.np.conj(eminus)], [eminus, eplus]],
            dtype=self.dtype,
        ) / self.np.sqrt(2)

    def U3(self, theta, phi, lam):
        cost = self.np.cos(theta / 2)
        sint = self.np.sin(theta / 2)
        eplus = self.np.exp(1j * (phi + lam) / 2.0)
        eminus = self.np.exp(1j * (phi - lam) / 2.0)
        return self.np.array(
            [
                [self.np.conj(eplus) * cost, -self.np.conj(eminus) * sint],
                [eminus * sint, eplus * cost],
            ],
            dtype=self.dtype,
        )

    @cached_property
    def CNOT(self):
        return self.np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=self.dtype
        )

    @cached_property
    def CZ(self):
        return self.np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=self.dtype
        )

    @cached_property
    def CSX(self):
        a = (1 + 1j) / 2
        b = self.np.conj(a)
        return self.np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, a, b],
                [0, 0, b, a],
            ],
            dtype=self.dtype,
        )

    @cached_property
    def CSXDG(self):
        return self.np.transpose(self.np.conj(self.CSX))

    def CRX(self, theta):
        m = self.np.eye(4, dtype=self.dtype)
        m[2:, 2:] = self.RX(theta)
        return m

    def CRY(self, theta):
        m = self.np.eye(4, dtype=self.dtype)
        m[2:, 2:] = self.RY(theta)
        return m

    def CRZ(self, theta):
        m = self.np.eye(4, dtype=self.dtype)
        m[2:, 2:] = self.RZ(theta)
        return m

    def CU1(self, theta):
        m = self.np.eye(4, dtype=self.dtype)
        m[2:, 2:] = self.U1(theta)
        return m

    def CU2(self, phi, lam):
        m = self.np.eye(4, dtype=self.dtype)
        m[2:, 2:] = self.U2(phi, lam)
        return m

    def CU3(self, theta, phi, lam):
        m = self.np.eye(4, dtype=self.dtype)
        m[2:, 2:] = self.U3(theta, phi, lam)
        return m

    @cached_property
    def SWAP(self):
        return self.np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=self.dtype
        )

    @cached_property
    def iSWAP(self):
        return self.np.array(
            [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=self.dtype
        )

    @cached_property
    def FSWAP(self):
        return self.np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -1]], dtype=self.dtype
        )

    def fSim(self, theta, phi):
        cost = self.np.cos(theta) + 0j
        isint = -1j * self.np.sin(theta)
        phase = self.np.exp(-1j * phi)
        return self.np.array(
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
        cost = self.np.cos(self.np.pi / 2) + 0j
        isint = -1j * self.np.sin(self.np.pi / 2)
        phase = self.np.exp(-1j * self.np.pi / 6)
        return self.np.array(
            [
                [1, 0, 0, 0],
                [0, cost, isint, 0],
                [0, isint, cost, 0],
                [0, 0, 0, phase],
            ],
            dtype=self.dtype,
        )

    def GeneralizedfSim(self, u, phi):
        phase = self.np.exp(-1j * phi)
        return self.np.array(
            [
                [1, 0, 0, 0],
                [0, u[0, 0], u[0, 1], 0],
                [0, u[1, 0], u[1, 1], 0],
                [0, 0, 0, phase],
            ],
            dtype=self.dtype,
        )

    def RXX(self, theta):
        cos = self.np.cos(theta / 2.0) + 0j
        isin = -1j * self.np.sin(theta / 2.0)
        return self.np.array(
            [
                [cos, 0, 0, isin],
                [0, cos, isin, 0],
                [0, isin, cos, 0],
                [isin, 0, 0, cos],
            ],
            dtype=self.dtype,
        )

    def RYY(self, theta):
        cos = self.np.cos(theta / 2.0) + 0j
        isin = -1j * self.np.sin(theta / 2.0)
        return self.np.array(
            [
                [cos, 0, 0, -isin],
                [0, cos, isin, 0],
                [0, isin, cos, 0],
                [-isin, 0, 0, cos],
            ],
            dtype=self.dtype,
        )

    def RZZ(self, theta):
        phase = self.np.exp(0.5j * theta)
        return self.np.array(
            [
                [self.np.conj(phase), 0, 0, 0],
                [0, phase, 0, 0],
                [0, 0, phase, 0],
                [0, 0, 0, self.np.conj(phase)],
            ],
            dtype=self.dtype,
        )

    def RZX(self, theta):
        cos, sin = self.np.cos(theta / 2), self.np.sin(theta / 2)
        return self.np.array(
            [
                [cos, -1j * sin, 0, 0],
                [-1j * sin, cos, 0, 0],
                [0, 0, cos, 1j * sin],
                [0, 0, 1j * sin, cos],
            ],
            dtype=self.dtype,
        )

    def RXY(self, theta):
        cos, sin = self.np.cos(theta / 2), self.np.sin(theta / 2)
        return self.np.array(
            [
                [1, 0, 0, 0],
                [0, cos, -1j * sin, 0],
                [0, -1j * sin, cos, 0],
                [0, 0, 0, 1],
            ],
            dtype=self.dtype,
        )

    def MS(self, phi0, phi1, theta):
        plus = self.np.exp(1.0j * (phi0 + phi1))
        minus = self.np.exp(1.0j * (phi0 - phi1))

        return self.np.array(
            [
                [
                    self.np.cos(theta / 2),
                    0,
                    0,
                    -1.0j * self.np.conj(plus) * self.np.sin(theta / 2),
                ],
                [
                    0,
                    self.np.cos(theta / 2),
                    -1.0j * self.np.conj(minus) * self.np.sin(theta / 2),
                    0,
                ],
                [0, -1.0j * minus * self.np.sin(theta / 2), self.np.cos(theta / 2), 0],
                [-1.0j * plus * self.np.sin(theta / 2), 0, 0, self.np.cos(theta / 2)],
            ],
            dtype=self.dtype,
        )

    def GIVENS(self, theta):
        return self.np.array(
            [
                [1, 0, 0, 0],
                [0, self.np.cos(theta), -self.np.sin(theta), 0],
                [0, self.np.sin(theta), self.np.cos(theta), 0],
                [0, 0, 0, 1],
            ],
            dtype=self.dtype,
        )

    def RBS(self, theta):
        return self.GIVENS(-theta)

    @cached_property
    def ECR(self):
        return self.np.array(
            [[0, 0, 1, 1j], [0, 0, 1j, 1], [1, -1j, 0, 0], [-1j, 1, 0, 0]],
            dtype=self.dtype,
        ) / self.np.sqrt(2)

    @cached_property
    def TOFFOLI(self):
        return self.np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )

    def DEUTSCH(self, theta):
        sin = self.np.sin(theta) + 0j  # 0j necessary for right tensorflow dtype
        cos = self.np.cos(theta)
        return self.np.array(
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

    def Unitary(self, u):
        return self.np.array(u, dtype=self.dtype, copy=False)

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
