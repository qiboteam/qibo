import numpy as np
import tensorflow as tf
from qibo.config import DTYPES


class NumpyMatrices:

    _NAMES = ["I", "H", "X", "Y", "Z", "CNOT", "SWAP", "TOFFOLI"]

    def __init__(self):
        self.I = None
        self.H = None
        self.X = None
        self.Y = None
        self.Z = None
        self.CNOT = None
        self.SWAP = None
        self.TOFFOLI = None
        self.allocate_matrices()

    def allocate_matrices(self):
        for name in self._NAMES:
            setattr(self, name, getattr(self, f"_{name}")())

    @property
    def dtype(self):
        if DTYPES.get("DTYPECPX") == tf.complex128:
            return np.complex128
        elif DTYPES.get("DTYPECPX") == tf.complex64:
            return np.complex64
        else:
            raise TypeError("Unknown complex type {}."
                            "".format(DTYPES.get("DTYPECPX")))

    def _I(self):
        return np.eye(2, dtype=self.dtype)

    def _H(self):
        m = np.ones((2, 2), dtype=self.dtype)
        m[1, 1] = -1
        return m / np.sqrt(2)

    def _X(self):
        m = np.zeros((2, 2), dtype=self.dtype)
        m[0, 1], m[1, 0] = 1, 1
        return m

    def _Y(self):
        m = np.zeros((2, 2), dtype=self.dtype)
        m[0, 1], m[1, 0] = -1j, 1j
        return m

    def _Z(self):
        m = np.eye(2, dtype=self.dtype)
        m[1, 1] = -1
        return m

    def _CNOT(self):
        m = np.eye(4, dtype=self.dtype)
        m[2, 2], m[2, 3] = 0, 1
        m[3, 2], m[3, 3] = 1, 0
        return m.reshape(4 * (2,))

    def _SWAP(self):
        m = np.eye(4, dtype=self.dtype)
        m[1, 1], m[1, 2] = 0, 1
        m[2, 1], m[2, 2] = 1, 0
        return m.reshape(4 * (2,))

    def _TOFFOLI(self):
        m = np.eye(8, dtype=self.dtype)
        m[-2, -2], m[-2, -1] = 0, 1
        m[-1, -2], m[-1, -1] = 1, 0
        return m.reshape(6 * (2,))


class TensorflowMatrices(NumpyMatrices):

    def allocate_matrices(self):
        for name in self._NAMES:
            matrix = tf.convert_to_tensor(getattr(self, f"_{name}")(),
                                          dtype=DTYPES.get('DTYPECPX'))
            setattr(self, name, matrix)
