import numpy as np
import tensorflow as tf


class GateMatrices:
    # TODO: Add docstrings

    _AVAILABLE_GATES = ["I", "H", "X", "Y", "Z", "CNOT"]

    def __init__(self, dtype):
        self.dtype = dtype

        self.I = tf.convert_to_tensor(self._npI(), dtype=self.dtype)
        self.H = tf.convert_to_tensor(self._npH(), dtype=self.dtype)
        self.X = tf.convert_to_tensor(self._npX(), dtype=self.dtype)
        self.Y = tf.convert_to_tensor(self._npY(), dtype=self.dtype)
        self.Z = tf.convert_to_tensor(self._npZ(), dtype=self.dtype)
        self.CNOT = tf.convert_to_tensor(self._npCNOT(), dtype=self.dtype)
        self.SWAP = tf.convert_to_tensor(self._npSWAP(), dtype=self.dtype)
        self.Toffoli = tf.convert_to_tensor(self._npToffoli(), dtype=self.dtype)

    @property
    def nptype(self):
        if self.dtype == tf.complex128:
            return np.complex128
        elif self.dtype == tf.complex64:
            return np.complex64
        else:
            raise TypeError("Unknown complex type {}.".format(self.dtype))

    def _npI(self):
        return np.eye(2, dtype=self.nptype)

    def _npH(self):
        m = np.ones((2, 2), dtype=self.nptype)
        m[1, 1] = -1
        return m / np.sqrt(2)

    def _npX(self):
        m = np.zeros((2, 2), dtype=self.nptype)
        m[0, 1], m[1, 0] = 1, 1
        return m

    def _npY(self):
        m = np.zeros((2, 2), dtype=self.nptype)
        m[0, 1], m[1, 0] = -1j, 1j
        return m

    def _npZ(self):
        m = np.eye(2, dtype=self.nptype)
        m[1, 1] = -1
        return m

    def _npCNOT(self):
        m = np.eye(4, dtype=self.nptype)
        m[2, 2], m[2, 3] = 0, 1
        m[3, 2], m[3, 3] = 1, 0
        return m.reshape(4 * (2,))

    def _npSWAP(self):
        m = np.eye(4, dtype=self.nptype)
        m[1, 1], m[1, 2] = 0, 1
        m[2, 1], m[2, 2] = 1, 0
        return m.reshape(4 * (2,))

    def _npToffoli(self):
        m = np.eye(8, dtype=self.nptype)
        m[-2, -2], m[-2, -1] = 0, 1
        m[-1, -2], m[-1, -1] = 1, 0
        return m.reshape(6 * (2,))