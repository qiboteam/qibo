import math
from abc import ABC, abstractmethod
from qibo import K
from qibo.abstractions import callbacks
from qibo.abstractions.states import AbstractState
from qibo.config import EIGVAL_CUTOFF, raise_error, log


class BackendCallback(callbacks.Callback, ABC):

    def __getitem__(self, k):
        if isinstance(k, int):
            if k >= len(self._results):
                raise_error(IndexError, "Attempting to access callbacks {} run but "
                                        "the callback has been used in {} executions."
                                        "".format(k, len(self._results)))
            return self._results[k]
        if isinstance(k, slice) or isinstance(k, list) or isinstance(k, tuple):
            from qibo import K
            return K.stack(self._results[k])
        raise_error(IndexError, "Unrecognized type for index {}.".format(k))

    @abstractmethod
    def state_vector_call(self, state): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def density_matrix_call(self, state): # pragma: no cover
        raise_error(NotImplementedError)

    def __call__(self, state):
        if isinstance(state, AbstractState):
            state = state.tensor
        return getattr(self, self._active_call)(state)


class PartialTrace(BackendCallback, callbacks.PartialTrace):

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


class EntanglementEntropy(BackendCallback, callbacks.EntanglementEntropy):
    _log2 = K.cast(math.log(2.0), dtype='DTYPE')

    def __init__(self, partition=None, compute_spectrum=False):
        callbacks.EntanglementEntropy.__init__(
            self, partition, compute_spectrum)
        self.partial_trace = PartialTrace(partition)

    @callbacks.Callback.nqubits.setter
    def nqubits(self, n: int):
        self._nqubits = n
        self.partial_trace.nqubits = n

    def entropy(self, rho):
        """Calculates entropy of a density matrix via exact diagonalization."""
        # Diagonalize
        eigvals = K.real(K.eigvalsh(rho))
        # Treating zero and negative eigenvalues
        drop_condition = eigvals > EIGVAL_CUTOFF
        masked_eigvals = K.gather(eigvals, condition=drop_condition)
        spectrum = -1 * K.log(masked_eigvals)
        if self.compute_spectrum:
            self.spectrum.append(spectrum)
        entropy = K.sum(masked_eigvals * spectrum)
        return entropy / self._log2

    def state_vector_call(self, state):
        rho = self.partial_trace.state_vector_call(state)
        return self.entropy(rho)

    def density_matrix_call(self, state):
        rho = self.partial_trace.density_matrix_call(state)
        return self.entropy(rho)


class Norm(BackendCallback, callbacks.Norm):

    def state_vector_call(self, state):
        return K.sqrt(K.sum(K.square(K.abs(state))))

    def density_matrix_call(self, state):
        return K.trace(state)


class Overlap(BackendCallback, callbacks.Overlap):

    def __init__(self, state):
        super().__init__()
        self.statec = K.conj(K.cast(state, dtype='DTYPECPX'))

    def state_vector_call(self, state):
        return K.abs(K.sum(self.statec * state))

    def density_matrix_call(self, state):
        raise_error(NotImplementedError, "Overlap callback is not implemented "
                                          "for density matrices.")


class Energy(BackendCallback, callbacks.Energy):

    def state_vector_call(self, state):
        return self.hamiltonian.expectation(state)

    def density_matrix_call(self, state):
        return K.trace(K.matmul(self.hamiltonian.matrix, state))


class Gap(BackendCallback, callbacks.Gap):

    def __init__(self, mode="gap", check_degenerate=True):
        callbacks.Gap.__init__(self, mode, check_degenerate)
        self._evolution = None

    @property
    def evolution(self):
        """:class:`qibo.evolution.AdiabaticEvolution` model used by the callback."""
        return self._evolution

    @evolution.setter
    def evolution(self, ev: "models.AdiabaticEvolution"):
        """Sets the :class:`qibo.evolution.AdiabaticEvolution` model."""
        from qibo.models import AdiabaticEvolution
        if not isinstance(ev, AdiabaticEvolution):
            t = type(ev)
            raise_error(TypeError, "Cannot add gap callback to {}.".format(t))
        self._evolution = ev

    def state_vector_call(self, state):
        if self.evolution is None:
            raise_error(ValueError, "Gap callback can only be used in "
                                    "adiabatic evolution models.")
        hamiltonian = self.evolution.hamiltonian()
        # Call the eigenvectors so that they are cached for the ``exp`` call
        hamiltonian.eigenvectors()
        eigvals = hamiltonian.eigenvalues()
        if isinstance(self.mode, int):
            return K.real(eigvals[self.mode])

        # case: self.mode == "gap"
        excited = 1
        gap = K.real(eigvals[excited] - eigvals[0])
        if not self.check_degenerate:
            return gap

        while K.equal(gap, 0):
            gap = K.real(eigvals[excited] - eigvals[0])
            excited += 1
        if excited > 1:
            log.warning("The Hamiltonian is degenerate. Using eigenvalue {} "
                        "to calculate gap.".format(excited))
        return gap

    def density_matrix_call(self, state):
        raise_error(NotImplementedError, "Gap callback is not implemented for "
                                         "density matrices.")
