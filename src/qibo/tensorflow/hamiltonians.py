import numpy as np
import tensorflow as tf
from qibo import matrices
from qibo.config import DTYPES
from qibo.base import hamiltonians


NUMERIC_TYPES = (np.int, np.float, np.complex,
                 np.int32, np.int64, np.float32,
                 np.float64, np.complex64, np.complex128)


class TensorflowHamiltonian(hamiltonians.Hamiltonian):

    NUMERIC_TYPES = NUMERIC_TYPES

    def __init__(self, nqubits, matrix):
        super(TensorflowHamiltonian, self).__init__(nqubits, matrix)
        self.matrix = tf.cast(matrix, dtype=DTYPES.get('DTYPECPX'))

    def _calculate_eigenvalues(self):
        return tf.linalg.eigvalsh(self.matrix)

    def _calculate_eigenvectors(self):
        return tf.linalg.eigh(self.matrix)

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
        return tf.eye(n, dtype=self.matrix.dtype)

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
            new_matrix = tf.matmul(self.matrix, o.matrix)
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, (tf.Tensor, np.ndarray)):
            rank = len(tuple(o.shape))
            if rank == 1: # vector
                return tf.matmul(self.matrix, o[:, tf.newaxis])[:, 0]
            elif rank == 2: # matrix
                return tf.matmul(self.matrix, o)
            else:
                raise ValueError(f'Cannot multiply Hamiltonian with '
                                  'rank-{rank} tensor.')
        else:
            raise NotImplementedError(f'Hamiltonian matrix multiplication to '
                                       '{type(o)} not implemented.')
