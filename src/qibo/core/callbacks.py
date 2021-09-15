import math
from abc import ABC, abstractmethod
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from qibo import K
from qibo.abstractions import callbacks as abstract_callbacks
from qibo.abstractions.states import AbstractState
from qibo.config import EIGVAL_CUTOFF, raise_error, log
from qibo.core.minimum_gap_class import Method_1, Method_2, Method_3



class BackendCallback(abstract_callbacks.Callback, ABC):

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
    def _state_vector_call(self, state): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def _density_matrix_call(self, state): # pragma: no cover
        raise_error(NotImplementedError)

    def __call__(self, state):
        if isinstance(state, AbstractState):
            state = state.tensor
        return getattr(self, self._active_call)(state)


class EntanglementEntropy(BackendCallback, abstract_callbacks.EntanglementEntropy):

    def __init__(self, partition=None, compute_spectrum=False):
        abstract_callbacks.EntanglementEntropy.__init__(
            self, partition, compute_spectrum)
        self.partial_trace = None

    @abstract_callbacks.Callback.density_matrix.setter
    def density_matrix(self, x):
        abstract_callbacks.Callback.density_matrix.fset(self, x) # pylint: disable=no-member
        if self.partial_trace is not None:
            self.partial_trace.density_matrix = x

    @abstract_callbacks.Callback.nqubits.setter
    def nqubits(self, n: int):
        from qibo import gates
        if self._nqubits is not None and self._nqubits != n:
            raise_error(RuntimeError,
                        f"Changing EntanglementEntropy nqubits from {self._nqubits} to {n}.")
        self._nqubits = n
        if self.partition is None:
            self.partition = list(range(n // 2 + n % 2))
        if len(self.partition) <= self.nqubits // 2:
            self.partition = [i for i in range(self.nqubits)
                              if i not in set(self.partition)]
        self.partial_trace = gates.PartialTrace(*self.partition)
        self.partial_trace.nqubits = n
        self.partial_trace.density_matrix = self.density_matrix

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
        return entropy / math.log(2.0)

    def _set_nqubits(self, state):
        if not isinstance(state, K.tensor_types):
            raise_error(TypeError, "State of unknown type {} was given in callback "
                                   "calculation.".format(type(state)))
        self.nqubits = int(math.log2(tuple(state.shape)[0]))

    def _state_vector_call(self, state):
        self._set_nqubits(state)
        rho = self.partial_trace.state_vector_partial_trace(state)
        return self.entropy(rho)

    def _density_matrix_call(self, state):
        self._set_nqubits(state)
        rho = self.partial_trace.density_matrix_partial_trace(state)
        return self.entropy(rho)


class Norm(BackendCallback, abstract_callbacks.Norm):

    def _state_vector_call(self, state):
        return K.sqrt(K.sum(K.square(K.abs(state))))

    def _density_matrix_call(self, state):
        return K.trace(state)


class Overlap(BackendCallback, abstract_callbacks.Overlap):

    def __init__(self, state):
        super().__init__()
        self.statec = K.conj(K.cast(state, dtype='DTYPECPX'))

    def _state_vector_call(self, state):
        return K.abs(K.sum(self.statec * state))

    def _density_matrix_call(self, state):
        raise_error(NotImplementedError, "Overlap callback is not implemented "
                                          "for density matrices.")


class Energy(BackendCallback, abstract_callbacks.Energy):

    def _state_vector_call(self, state):
        return self.hamiltonian.expectation(state)

    def _density_matrix_call(self, state):
        return K.trace(K.matmul(self.hamiltonian.matrix, state))


class Gap(BackendCallback, abstract_callbacks.Gap):

    def __init__(self, mode="gap", check_degenerate=True):
        abstract_callbacks.Gap.__init__(self, mode, check_degenerate)
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

    def _state_vector_call(self, state):
        if self.evolution is None:
            raise_error(ValueError, "Gap callback can only be used in "
                                    "adiabatic evolution models.")
        hamiltonian = self.evolution.solver.current_hamiltonian
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

        while K.less(gap, EIGVAL_CUTOFF):
            gap = K.real(eigvals[excited] - eigvals[0])
            excited += 1
        if excited > 1:
            log.warning("The Hamiltonian is degenerate. Using eigenvalue {} "
                        "to calculate gap.".format(excited))
        return gap

    def _density_matrix_call(self, state):
        raise_error(NotImplementedError, "Gap callback is not implemented for "
                                         "density matrices.")

class MinimumGap(Gap):

    def __init__(self, n_steps: int, method=3, mode="gap", check_degenerate=True, precision = None, method_2_mode='minimum'):
        Gap.__init__(self, mode, check_degenerate)
        self._method = method
        self._n_steps = n_steps
        self._steps_left = n_steps
        self._energy_levels: List[List[float]] = []
        self._precision = precision
        self._method_2_mode = method_2_mode
        self._gap = None
        self._degenerated_levels = None
 
    @property
    def minimum_gap(self):
        if self._gap is None:
            raise_error(ValueError, 'Gap still not computed.')
        return min(self._gap[0])

    @property
    def gaps(self):
        if self._gap is None:
            raise_error(ValueError, 'Gap still not computed.')
        return self._gap[0]

    def _decrease_step(self):
        self._steps_left -= 1

    def _save_energy_levels(self, eigenvalues: List[float]):
        if self._steps_left == self._n_steps-1:
            self._energy_levels = [[] for _ in range(len(eigenvalues))]
        
        for i, eigenvalue in enumerate(list(eigenvalues)):
            self._energy_levels[i].append(eigenvalue) 

    def _state_vector_call(self, state):
        if self.evolution is None:
            raise_error(ValueError, "Gap callback can only be used in "
                                    "adiabatic evolution models.")
        self._decrease_step()
        
        hamiltonian = self.evolution.solver.current_hamiltonian
        # Call the eigenvectors so that they are cached for the ``exp`` call
        hamiltonian.eigenvectors()
        eigvals = hamiltonian.eigenvalues()
        self._save_energy_levels(eigvals)
        if isinstance(self.mode, int):
            return K.real(eigvals[self.mode])

        if self._steps_left == -1:
            if self._method == 1:
                self._gap, self._degenerated_levels = Method_1(self._energy_levels).compute_gap()
                return min(self._gap[0])

            if self._method == 2:
                self._gap, self._degenerated_levels = Method_2(self._energy_levels).compute_gap(self._method_2_mode)
                return min(self._gap[0])

            if self._method == 3:
                self._gap, self._degenerated_levels = Method_3(self._energy_levels).compute_gap(precision = self._precision)
                return min(self._gap[0])

            raise_error(NotImplementedError, 'Only available methods are: 1, 2, 3')
        return

    def plot_energies(self, T=1):
        if self._gap == None:
            raise_error(ValueError, 'Minimum Gap not computed.')

        dt = T/float(self._n_steps)
        fig, ax = plt.subplots()
        times = np.arange(0, T+dt, dt)
        for i,j in enumerate(self._degenerated_levels):
            if i==0:
                label='ground state'
            else:
                label=self._ordinal(i)+' excited state'
            ax.plot(times, j[:], label=label, color='C'+str(i))

        if len(times) > 100:
            reduction = int(len(times)/100)
            times_reduced = times[::reduction]
        else:
            times_reduced = times[:]
            reduction = 1

        for i,j in enumerate(times_reduced):

            plt.plot([j,j], 
                    [self._degenerated_levels[self._gap[1][i*reduction][0]][i*reduction], 
                    self._degenerated_levels[self._gap[1][i*reduction][1]][i*reduction]], 
                    c='purple', 
                    alpha=0.5)

        plt.ylabel('Energy')
        plt.xlabel('Schedule')
        plt.title('Energy during adiabatic evolution')
        ax.legend()
        fig.tight_layout()
        #fig.savefig('images/energy_levels.png', dpi=300, bbox_inches='tight')
        fig, ax = plt.subplots()
        ax.plot(times, self._gap[0], label='gap energy', color='C0')
        plt.ylabel('Energy')
        plt.xlabel('Schedule')
        plt.title('Energy during adiabatic evolution')
        ax.legend()
        fig.tight_layout()
        #fig.savefig('images/minimum_gap.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _ordinal(self, number):
        return "%d%s"%(number,{1:"st",2:"nd",3:"rd"}.get(number if number<20 else number%10,"th"))
        

