import numpy as np


class Matrices:
    # TODO: Implement matrices for all gates

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

    def TOFFOLI(self):
        m = np.eye(8, dtype=self.dtype)
        m[-2, -2], m[-2, -1] = 0, 1
        m[-1, -2], m[-1, -1] = 1, 0
        return m