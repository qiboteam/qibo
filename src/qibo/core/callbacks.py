import math
from qibo import K
from qibo.base import callbacks
from qibo.config import EIGVAL_CUTOFF, raise_error


class PartialTrace(callbacks.PartialTrace):

    def set_nqubits(self, state):
        if not isinstance(state, K.tensor_types):
            raise_error(TypeError, "State of unknown type {} was given in callback "
                                   "calculation.".format(type(state)))
        if self._nqubits is None:
            self.nqubits = int(math.log2(tuple(state.shape)[0]))

    def state_vector_call(self, state):
        self.set_nqubits(state)
        state = K.reshape(state, self.nqubits * (2,))
        rho = K.tensordot(state, K.conj(state),
                           axes=[self.partition, self.partition])
        return K.reshape(rho, (self.rho_dim, self.rho_dim))

    def density_matrix_call(self, state):
        self.set_nqubits(state)
        state = K.reshape(state, 2 * self.nqubits * (2,))
        rho = K.einsum(self.traceout(), state)
        return K.reshape(rho, (self.rho_dim, self.rho_dim))


class EntanglementEntropy(callbacks.EntanglementEntropy):
    _log2 = K.cast(math.log(2.0), dtype='DTYPE')

    def entropy(self, rho):
        # Diagonalize
        eigvals = K.real(K.eigvalsh(rho))
        # Treating zero and negative eigenvalues
        drop_condition = eigvals > EIGVAL_CUTOFF
        masked_eigvals = K.gather(eigvals, condition=drop_condition)[:, 0]
        spectrum = -1 * K.log(masked_eigvals)
        if self.compute_spectrum:
            self.spectrum.append(spectrum)
        entropy = K.sum(masked_eigvals * spectrum)
        return entropy / self._log2


class Norm(callbacks.Norm):

    def state_vector_call(self, state):
        return K.sqrt(K.sum(K.square(K.abs(state))))

    def density_matrix_call(self, state):
        return K.trace(state)


class Overlap(callbacks.Overlap):

    def __init__(self, state):
        super().__init__()
        self.statec = K.conj(K.cast(state, dtype='DTYPECPX'))

    def state_vector_call(self, state):
        return K.abs(K.sum(self.statec * state))

    def density_matrix_call(self, state):
        raise_error(NotImplementedError, "Overlap callback is not implemented "
                                          "for density matrices.")


class Energy(callbacks.Energy):

    def density_matrix_call(self, state):
        return K.trace(K.matmul(self.hamiltonian.matrix, state))


class Gap(callbacks.Gap):
    pass
