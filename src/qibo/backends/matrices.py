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


class Matrices:
    """Matrix representation of every gate as a numpy array."""

    def __init__(self, dtype):
        import numpy as np
        self.dtype = dtype
        self.np = np

    @cached_property
    def H(self):
        return self.np.array([
            [1, 1],
            [1, -1]
        ], dtype=self.dtype) / self.np.sqrt(2)

    @cached_property
    def X(self):
        return self.np.array([
            [0, 1],
            [1, 0]
        ], dtype=self.dtype)

    @cached_property
    def Y(self):
        return self.np.array([
            [0, -1j],
            [1j, 0]
        ], dtype=self.dtype)

    @cached_property
    def Z(self):
        return self.np.array([
            [1, 0],
            [0, -1]
        ], dtype=self.dtype)

    @cached_property
    def S(self):
        return self.np.array([
            [1, 0],
            [0, 1j]
        ], dtype=self.dtype)

    @cached_property
    def SDG(self):
        return self.np.conj(self.S)

    @cached_property
    def T(self):
        return self.np.array([
            [1, 0],
            [0, self.np.exp(1j * self.np.pi / 4.0)]
        ], dtype=self.dtype)

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
        return self.np.array([
            [cos, isin],
            [isin, cos]
        ], dtype=self.dtype)

    def RY(self, theta):
        cos = self.np.cos(theta / 2.0) + 0j
        sin = self.np.sin(theta / 2.0)
        return self.np.array([
            [cos, -sin],
            [sin, cos]
        ], dtype=self.dtype)

    def RZ(self, theta):
        phase = self.np.exp(0.5j * theta)
        return self.np.array([
            [self.np.conj(phase), 0],
            [0, phase]
        ], dtype=self.dtype)

    def U1(self, theta):
        phase = self.np.exp(1j * theta)
        return self.np.array([
            [1, 0],
            [0, phase]
        ], dtype=self.dtype)

    def U2(self, phi, lam):
        eplus = self.np.exp(1j * (phi + lam) / 2.0)
        eminus = self.np.exp(1j * (phi - lam) / 2.0)
        return self.np.array([
            [self.np.conj(eplus), - self.np.conj(eminus)],
            [eminus, eplus]
        ], dtype=self.dtype) / self.np.sqrt(2)

    def U3(self, theta, phi, lam):
        cost = self.np.cos(theta / 2)
        sint = self.np.sin(theta / 2)
        eplus = self.np.exp(1j * (phi + lam) / 2.0)
        eminus = self.np.exp(1j * (phi - lam) / 2.0)
        return self.np.array([
            [self.np.conj(eplus) * cost, - self.np.conj(eminus) * sint],
            [eminus * sint, eplus * cost]
        ], dtype=self.dtype)

    @cached_property
    def CNOT(self):
        return self.np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=self.dtype)

    @cached_property
    def CZ(self):
        return self.np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=self.dtype)

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
        return self.np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=self.dtype)

    @cached_property
    def FSWAP(self):
        return self.np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1]
        ], dtype=self.dtype)

    def fSim(self, theta, phi):
        cost = self.np.cos(theta) + 0j
        isint = -1j * self.np.sin(theta)
        phase = self.np.exp(-1j * phi)
        return self.np.array([
            [1, 0, 0, 0],
            [0, cost, isint, 0],
            [0, isint, cost, 0],
            [0, 0, 0, phase],
        ], dtype=self.dtype)

    def GeneralizedfSim(self, u, phi):
        phase = self.np.exp(-1j * phi)
        return self.np.array([
            [1, 0, 0, 0],
            [0, u[0, 0], u[0, 1], 0],
            [0, u[1, 0], u[1, 1], 0],
            [0, 0, 0, phase],
        ], dtype=self.dtype)

    @cached_property
    def TOFFOLI(self):
        return self.np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ])

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
