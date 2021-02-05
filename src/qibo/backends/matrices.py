import numpy as np


class NumpyMatrices:

    _NAMES = ["I", "H", "X", "Y", "Z", "CNOT", "CZ", "SWAP", "TOFFOLI"]

    def __init__(self, dtype):
        self._dtype = dtype
        self._I = None
        self._H = None
        self._X = None
        self._Y = None
        self._Z = None
        self._CNOT = None
        self._CZ = None
        self._SWAP = None
        self._TOFFOLI = None
        self.allocate_matrices()

    def allocate_matrices(self):
        for name in self._NAMES:
            getattr(self, f"_set{name}")()

    def cast(self, x):
        return x.astype(self.dtype)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype

    @property
    def I(self):
        return self._I

    @property
    def H(self):
        return self._H

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def Z(self):
        return self._Z

    @property
    def CNOT(self):
        return self._CNOT

    @property
    def CZ(self):
        return self._CZ

    @property
    def SWAP(self):
        return self._SWAP

    @property
    def TOFFOLI(self):
        return self._TOFFOLI

    def _setI(self):
        self._I = self.cast(np.eye(2, dtype=self.dtype))

    def _setH(self):
        m = np.ones((2, 2), dtype=self.dtype)
        m[1, 1] = -1
        self._H = self.cast(m / np.sqrt(2))

    def _setX(self):
        m = np.zeros((2, 2), dtype=self.dtype)
        m[0, 1], m[1, 0] = 1, 1
        self._X = self.cast(m)

    def _setY(self):
        m = np.zeros((2, 2), dtype=self.dtype)
        m[0, 1], m[1, 0] = -1j, 1j
        self._Y = self.cast(m)

    def _setZ(self):
        m = np.eye(2, dtype=self.dtype)
        m[1, 1] = -1
        self._Z = self.cast(m)

    def _setCNOT(self):
        m = np.eye(4, dtype=self.dtype)
        m[2, 2], m[2, 3] = 0, 1
        m[3, 2], m[3, 3] = 1, 0
        self._CNOT = self.cast(m)

    def _setCZ(self):
        m = np.diag([1, 1, 1, -1])
        self._CZ = self.cast(m)

    def _setSWAP(self):
        m = np.eye(4, dtype=self.dtype)
        m[1, 1], m[1, 2] = 0, 1
        m[2, 1], m[2, 2] = 1, 0
        self._SWAP = self.cast(m)

    def _setTOFFOLI(self):
        m = np.eye(8, dtype=self.dtype)
        m[-2, -2], m[-2, -1] = 0, 1
        m[-1, -2], m[-1, -1] = 1, 0
        self._TOFFOLI = self.cast(m)


class TensorflowMatrices(NumpyMatrices):

    def __init__(self, dtype):
        import tensorflow as tf
        self.tf = tf
        self.tftype = dtype
        if dtype == tf.complex128:
            super().__init__(np.complex128)
        elif dtype == tf.complex64:
            super().__init__(np.complex64)

    @NumpyMatrices.dtype.setter
    def dtype(self, dtype):
        self.tftype = dtype
        if dtype == self.tf.complex128:
            self._dtype = np.complex128
        elif dtype == self.tf.complex64:
            self._dtype = np.complex64

    def cast(self, x):
        return self.tf.cast(x, dtype=self.tftype)
