import numpy as np
from qibo.config import raise_error


class Matrices:
    """Matrix representation of every gate as a numpy array."""

    def __init__(self, dtype):
        self.dtype = dtype

    def H(self):
        return np.array([
            [1, 1], 
            [1, -1]
        ], dtype=self.dtype) / np.sqrt(2)

    def X(self):
        return np.array([
            [0, 1], 
            [1, 0]
        ], dtype=self.dtype)

    def Y(self):
        return np.array([
            [0, -1j], 
            [1j, 0]
        ], dtype=self.dtype)

    def Z(self):
        return np.array([
            [0, -1j], 
            [1j, 0]
        ], dtype=self.dtype)

    def S(self):
        return np.array([
            [1, 0], 
            [0, 1j]
        ], dtype=self.dtype)

    def SDG(self):
        return np.conj(self.S())

    def T(self):
        return np.array([
            [1, 0],
            [0, np.exp(1j * np.pi / 4.0)]
        ], dtype=self.dtype)

    def TDG(self):
        return np.conj(self.T())

    def I(self):
        return np.eye(2, dtype=self.dtype)

    def Align(self):
        raise_error(NotImplementedError)

    def M(self):
        raise_error(NotImplementedError)

    def RX(self, theta):
        cos = np.cos(theta / 2.0) + 0j
        isin = -1j * np.sin(theta / 2.0)
        return np.array([
            [cos, isin], 
            [isin, cos]
        ], dtype=self.dtype)

    def RY(self, theta):
        cos = np.cos(theta / 2.0) + 0j
        sin = np.sin(theta / 2.0)
        return np.array([
            [cos, -sin], 
            [sin, cos]
        ], dtype=self.dtype)

    def RZ(self, theta):
        phase = np.exp(0.5j * theta)
        return np.array([
            [np.conj(phase), 0], 
            [0, phase]
        ], dtype=self.dtype)

    def U1(self, theta):
        phase = np.exp(1j * theta)
        return np.array([
            [1, 0], 
            [0, phase]
        ], dtype=self.dtype)

    def U2(self, phi, lam):
        eplus = np.exp(1j * (phi + lam) / 2.0)
        eminus = np.exp(1j * (phi - lam) / 2.0)
        return np.array([
            [np.conj(eplus), - np.conj(eminus)],
            [eminus, eplus]
        ], dtype=self.dtype) / np.sqrt(2)

    def U3(self, theta, phi, lam):
        cost = np.cos(theta / 2)
        sint = np.sin(theta / 2)
        eplus = np.exp(1j * (phi + lam) / 2.0)
        eminus = np.exp(1j * (phi - lam) / 2.0)
        return np.array([
            [np.conj(eplus) * cost, - np.conj(eminus) * sint],
            [eminus * sint, eplus * cost]
        ], dtype=self.dtype)

    def CNOT(self):
        return np.array([
            [1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 0, 1], 
            [0, 0, 1, 0]
        ], dtype=self.dtype)

    def CZ(self):
        return np.array([
            [1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 1, 0], 
            [0, 0, 0, -1]
        ], dtype=self.dtype)

    def CRX(self, theta):
        m = np.eye(4, dtype=self.dtype)
        m[2:, 2:] = self.RX(theta)
        return m

    def CRY(self, theta):
        m = np.eye(4, dtype=self.dtype)
        m[2:, 2:] = self.RY(theta)
        return m

    def CRZ(self, theta):
        m = np.eye(4, dtype=self.dtype)
        m[2:, 2:] = self.RZ(theta)
        return m

    def CU1(self, theta):
        m = np.eye(4, dtype=self.dtype)
        m[2:, 2:] = self.U1(theta)
        return m

    def CU2(self, phi, lam):
        m = np.eye(4, dtype=self.dtype)
        m[2:, 2:] = self.U2(phi, lam)
        return m

    def CU3(self, theta, phi, lam):
        m = np.eye(4, dtype=self.dtype)
        m[2:, 2:] = self.U3(theta, phi, lam)
        return m

    def SWAP(self):
        return np.array([
            [1, 0, 0, 0], 
            [0, 0, 1, 0], 
            [0, 1, 0, 0], 
            [0, 0, 0, 1]
        ], dtype=self.dtype)

    def FSWAP(self):
        return np.array([
            [1, 0, 0, 0], 
            [0, 0, 1, 0], 
            [0, 1, 0, 0], 
            [0, 0, 0, -1]
        ], dtype=self.dtype)

    def fSim(self, theta, phi):
        cost = np.cos(theta) + 0j
        isint = -1j * np.sin(theta)
        phase = np.exp(-1j * phi)
        return np.array([
            [1, 0, 0, 0],
            [0, cost, isint, 0],
            [0, isint, cost, 0],
            [0, 0, 0, phase],
        ], dtype=self.dtype)

    def GeneralizedfSim(self, u, phi):
        phase = np.exp(-1j * phi)
        return np.array([
            [1, 0, 0, 0],
            [0, u[0, 0], u[0, 1], 0],
            [0, u[1, 0], u[1, 1], 0],
            [0, 0, 0, phase],
        ], dtype=self.dtype)

    def TOFFOLI(self):
        m = np.eye(8, dtype=self.dtype)
        m[-2, -2], m[-2, -1] = 0, 1
        m[-1, -2], m[-1, -1] = 1, 0
        return m

    def Unitary(self, u):
        return u.astype(self.dtype)

    def VariationalLayer(self, *args):
        raise_error(NotImplementedError)

    def Flatten(self):
        raise_error(NotImplementedError)

    def CallbackGate(self):
        raise_error(NotImplementedError)

    def PartialTrace(self):
        raise_error(NotImplementedError)

    def UnitaryChannel(self):
        raise_error(NotImplementedError)

    def PauliNoiseChannel(self):
        raise_error(NotImplementedError)

    def ResetChannel(self):
        raise_error(NotImplementedError)

    def ThermalRelaxationChannel(self):
        raise_error(NotImplementedError)

    def FusedGate(self):
        raise_error(NotImplementedError)
