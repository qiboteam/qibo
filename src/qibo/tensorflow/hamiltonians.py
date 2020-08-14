import numpy as np
import tensorflow as tf
from qibo.config import raise_error
from qibo.base import hamiltonians


NUMERIC_TYPES = (np.int, np.float, np.complex,
                 np.int32, np.int64, np.float32,
                 np.float64, np.complex64, np.complex128)
ARRAY_TYPES = (tf.Tensor, np.ndarray)


class TensorflowHamiltonian(hamiltonians.Hamiltonian):
    """Implementation of :class:`qibo.base.hamiltonians.Hamiltonian` using
    TensorFlow.
    """
    NUMERIC_TYPES = NUMERIC_TYPES
    ARRAY_TYPES = ARRAY_TYPES
    K = tf

    def _calculate_exp(self, a):
        if self._eigenvectors is None:
            return tf.linalg.expm(-1j * a * self.matrix)
        else:
            expd = tf.linalg.diag(tf.exp(-1j * a * self._eigenvalues))
            ud = tf.transpose(tf.math.conj(self._eigenvectors))
            return tf.matmul(self._eigenvectors, tf.matmul(expd, ud))

    def expectation(self, state, normalize=False):
        statec = tf.math.conj(state)
        hstate = self @ state
        ev = tf.math.real(tf.reduce_sum(statec * hstate))
        if normalize:
            norm = tf.reduce_sum(tf.square(tf.abs(state)))
            return ev / norm
        return ev

    def _real(self, o):
        if isinstance(o, tf.Tensor):
            if o.shape:
                return np.array(o)[0].real
            else:
                return np.array(o).real
        return super(TensorflowHamiltonian, self)._real(o)

    def __mul__(self, o):
        """Multiplication to scalar operator."""
        if isinstance(o, tf.Tensor):
            o = tf.cast(o, dtype=self.matrix.dtype)
        return super(TensorflowHamiltonian, self).__mul__(o)


class NumpyHamiltonian(TensorflowHamiltonian):
    """Implementation of :class:`qibo.base.hamiltonians.Hamiltonian` using
    numpy.
    """
    import scipy
    K = np

    def _calculate_exp(self, a):
        if self._eigenvectors is None:
            return self.scipy.linalg.expm(-1j * a * self.matrix)
        else:
            expd = np.diag(np.exp(-1j * a * self._eigenvalues))
            ud = np.transpose(np.conj(self._eigenvectors))
            return self._eigenvectors @ (expd @ ud)

    def expectation(self, state, normalize=False):
        statec = np.conj(state)
        hstate = self @ state
        ev = np.sum(statec * hstate).real
        if normalize:
            return ev / (np.abs(state) ** 2).sum()
        return ev
