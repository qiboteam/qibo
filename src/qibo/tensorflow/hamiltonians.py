import numpy as np
import tensorflow as tf
from qibo.base import hamiltonians


NUMERIC_TYPES = (np.int, np.float, np.complex,
                 np.int32, np.int64, np.float32,
                 np.float64, np.complex64, np.complex128)


class TensorflowHamiltonian(hamiltonians.Hamiltonian):
    """Implementation of :class:`qibo.base.hamiltonians.Hamiltonian` using
    TensorFlow.
    """
    NUMERIC_TYPES = NUMERIC_TYPES
    K = tf

    def _calculate_eigenvalues(self):
        return self.K.linalg.eigvalsh(self.matrix)

    def _calculate_eigenvectors(self):
        return self.K.linalg.eigh(self.matrix)

    def _calculate_exp(self, a):
        if self._eigenvectors is None:
            return tf.linalg.expm(-1j * a * self.matrix)
        else:
            expd = tf.linalg.diag(tf.exp(-1j * a * self._eigenvalues))
            ud = tf.transpose(tf.math.conj(self._eigenvectors))
            return tf.matmul(self._eigenvectors, tf.matmul(expd, ud))

    def _eye(self, n=None):
        if n is None:
            n = int(self.matrix.shape[0])
        return self.K.eye(n, dtype=self.matrix.dtype)

    def expectation(self, state, normalize=False):
        statec = tf.math.conj(state)
        hstate = self @ state
        ev = tf.math.real(tf.reduce_sum(statec * hstate))
        if normalize:
            norm = tf.reduce_sum(tf.square(tf.abs(state)))
            return ev / norm
        return ev

    def __mul__(self, o):
        """Multiplication to scalar operator."""
        if isinstance(o, (np.ndarray, tf.Tensor)):
            new_matrix = self.matrix * tf.cast(o, dtype=self.matrix.dtype)
            return self.__class__(self.nqubits, new_matrix)
        else:
            return super(TensorflowHamiltonian, self).__mul__(o)

    def __matmul__(self, o):
        """Matrix multiplication with other Hamiltonians or state vectors."""
        if isinstance(o, self.__class__):
            new_matrix = self.K.matmul(self.matrix, o.matrix)
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, (tf.Tensor, np.ndarray)):
            rank = len(tuple(o.shape))
            if rank == 1: # vector
                return self.K.matmul(self.matrix, o[:, self.K.newaxis])[:, 0]
            elif rank == 2: # matrix
                return self.K.matmul(self.matrix, o)
            else:
                raise ValueError(f'Cannot multiply Hamiltonian with '
                                  'rank-{rank} tensor.')
        else:
            raise NotImplementedError(f'Hamiltonian matrix multiplication to '
                                       '{type(o)} not implemented.')


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

    def __mul__(self, o):
        """Multiplication to scalar operator."""
        if isinstance(o, np.ndarray):
            new_matrix = self.matrix * o.astype(self.matrix.dtype)
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, tf.Tensor): # pragma: no cover
            new_matrix = self.matrix * o.numpy().astype(self.matrix.dtype)
            return self.__class__(self.nqubits, new_matrix)
        else:
            return hamiltonians.Hamiltonian.__mul__(self, o)

    def __matmul__(self, o):
        if isinstance(o, tf.Tensor): # pragma: no cover
            return TensorflowHamiltonian.__matmul__(self, o.numpy())
        else:
            return TensorflowHamiltonian.__matmul__(self, o)
