import numpy as np
import tensorflow as tf
from qibo.base import callbacks
from qibo.config import DTYPES, EIGVAL_CUTOFF, raise_error
from typing import Union


class PartialTrace(callbacks.PartialTrace):

    def set_nqubits(self, state):
        if not (isinstance(state, np.ndarray) or isinstance(state, tf.Tensor)):
            raise_error(TypeError, "State of unknown type {} was given in callback "
                                   "calculation.".format(type(state)))
        if self._nqubits is None:
            self.nqubits = int(np.log2(tuple(state.shape)[0]))

    def state_vector_call(self, state):
        self.set_nqubits(state)
        state = tf.reshape(state, self.nqubits * (2,))
        rho = tf.tensordot(state, tf.math.conj(state),
                           axes=[self.partition, self.partition])
        return tf.reshape(rho, (self.rho_dim, self.rho_dim))

    def density_matrix_call(self, state):
        self.set_nqubits(state)
        state = tf.reshape(state, 2 * self.nqubits * (2,))
        rho = tf.einsum(self.traceout(), state)
        return tf.reshape(rho, (self.rho_dim, self.rho_dim))


class EntanglementEntropy(callbacks.EntanglementEntropy):
    _log2 = tf.cast(tf.math.log(2.0), dtype=DTYPES.get('DTYPE'))

    def entropy(self, rho: tf.Tensor) -> tf.Tensor:
        # Diagonalize
        eigvals = tf.math.real(tf.linalg.eigvalsh(rho))
        # Treating zero and negative eigenvalues
        masked_eigvals = tf.gather(eigvals, tf.where(eigvals > EIGVAL_CUTOFF))[:, 0]
        spectrum = -1 * tf.math.log(masked_eigvals)
        if self.compute_spectrum:
            self.spectrum.append(spectrum)
        entropy = tf.reduce_sum(masked_eigvals * spectrum)
        return entropy / self._log2


class Norm(callbacks.Norm):

    def state_vector_call(self, state):
        return tf.sqrt(tf.reduce_sum(tf.square(tf.abs(state))))

    def density_matrix_call(self, state):
        return tf.linalg.trace(state)


class Overlap(callbacks.Overlap):

    def __init__(self, state: Union[np.ndarray, tf.Tensor]):
        super().__init__()
        self.statec = tf.math.conj(tf.cast(state, dtype=DTYPES.get('DTYPECPX')))

    def state_vector_call(self, state):
        return tf.abs(tf.reduce_sum(self.statec * state))

    def density_matrix_call(self, state):
        raise_error(NotImplementedError, "Overlap callback is not implemented "
                                          "for density matrices.")


class Energy(callbacks.Energy):

    def density_matrix_call(self, state):
        return tf.linalg.trace(tf.matmul(self.hamiltonian.matrix, state))


class Gap(callbacks.Gap):
    pass
