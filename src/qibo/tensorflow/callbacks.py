import numpy as np
import tensorflow as tf
from qibo.base import callbacks
from qibo.config import DTYPES, EIGVAL_CUTOFF, raise_error
from typing import Union


class PartialTrace(callbacks.PartialTrace):

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        """Calculates reduced density matrix.

        Traces out all qubits contained in `self.partition`.
        """
        # Cast state in the proper shape
        if not (isinstance(state, np.ndarray) or isinstance(state, tf.Tensor)):
            raise_error(TypeError, "State of unknown type {} was given in callback "
                                   "calculation.".format(type(state)))
        if self._nqubits is None:
            self.nqubits = int(np.log2(tuple(state.shape)[0]))

        shape = (1 + int(is_density_matrix)) * self.nqubits * (2,)
        state = tf.reshape(state, shape)

        if is_density_matrix:
            rho = tf.einsum(self.traceout(), state)
        else:
            rho = tf.tensordot(state, tf.math.conj(state),
                               axes=[self.partition, self.partition])
        return tf.reshape(rho, (self.rho_dim, self.rho_dim))


class EntanglementEntropy(callbacks.EntanglementEntropy):
    _log2 = tf.cast(tf.math.log(2.0), dtype=DTYPES.get('DTYPE'))

    def _entropy(self, rho: tf.Tensor) -> tf.Tensor:
        """Calculates entropy by diagonalizing the density matrix."""
        # Diagonalize
        eigvals = tf.math.real(tf.linalg.eigvalsh(rho))
        # Treating zero and negative eigenvalues
        masked_eigvals = tf.gather(eigvals, tf.where(eigvals > EIGVAL_CUTOFF))[:, 0]
        spectrum = -1 * tf.math.log(masked_eigvals)
        if self.compute_spectrum:
            self.spectrum.append(spectrum)
        entropy = tf.reduce_sum(masked_eigvals * spectrum)
        return entropy / self._log2

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        # Construct reduced density matrix
        rho = PartialTrace.__call__(self, state, is_density_matrix)
        # Calculate entropy of reduced density matrix
        return self._entropy(rho)


class Norm(callbacks.Norm):

    @staticmethod
    def norm(state: tf.Tensor, is_density_matrix: bool = False) -> tf.Tensor:
        """"""
        if is_density_matrix:
            return tf.linalg.trace(state)
        return tf.sqrt(tf.reduce_sum(tf.square(tf.abs(state))))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        return self.norm(state, is_density_matrix)


class Overlap(callbacks.Overlap):

    def __init__(self, state: Union[np.ndarray, tf.Tensor]):
        super().__init__()
        self.statec = tf.math.conj(tf.cast(state, dtype=DTYPES.get('DTYPECPX')))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        if is_density_matrix:
            raise_error(NotImplementedError, "Overlap callback is not implemented "
                                             "for density matrices.")
        return tf.abs(tf.reduce_sum(self.statec * state))


class Energy(callbacks.Energy):

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        if is_density_matrix:
            return tf.linalg.trace(tf.matmul(self.hamiltonian.matrix,
                                             state))
        return self.hamiltonian.expectation(state)


class Gap(callbacks.Gap):
    pass
