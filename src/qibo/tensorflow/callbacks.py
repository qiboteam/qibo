import numpy as np
import tensorflow as tf
from qibo.config import DTYPES, EIGVAL_CUTOFF, raise_error
from typing import List, Optional, Union


class Callback:
    """Base callback class.

    All Tensorflow callbacks should inherit this class and implement its
    `__call__` method.

    Results of a callback can be accessed by indexing the corresponding object.
    """

    def __init__(self):
        self._results = []
        self._nqubits = None

    @property
    def nqubits(self) -> int: # pragma: no cover
        """Total number of qubits in the circuit that the callback was added in."""
        # abstract method
        return self._nqubits

    @nqubits.setter
    def nqubits(self, n: int): # pragma: no cover
        # abstract method
        self._nqubits = n

    def __getitem__(self, k) -> tf.Tensor:
        if isinstance(k, int):
            if k >= len(self._results):
                raise_error(IndexError, "Attempting to access callbacks {} run but "
                                        "the callback has been used in {} executions."
                                        "".format(k, len(self._results)))
            return self._results[k]
        if isinstance(k, slice) or isinstance(k, list) or isinstance(k, tuple):
            return tf.stack(self._results[k])
        raise_error(IndexError, "Unrecognized type for index {}.".format(k))

    def __call__(self, state: tf.Tensor) -> tf.Tensor: # pragma: no cover
        # abstract method
        raise_error(NotImplementedError)

    def append(self, result: tf.Tensor):
        self._results.append(result)

    def extend(self, result: tf.Tensor):
        self._results.extend(result)


class PartialTrace(Callback):
    """Calculates reduced density matrix of a state.

    This is used by the :class:`qibo.tensorflow.callbacks.EntanglementEntropy`
    callback. It can also be used as a standalone callback in order to access
    a reduced density matrix in the middle of a circuit execution.

    Args:
        partition (list): List with qubit ids that defines the first subsystem.
            If `partition` is not given then the first subsystem is the first
            half of the qubits.
    """

    def __init__(self, partition: Optional[List[int]] = None):
        super(PartialTrace, self).__init__()
        self.partition = partition
        self.rho_dim = None
        self._traceout = None

    @property
    def nqubits(self) -> int:
        return self._nqubits

    @nqubits.setter
    def nqubits(self, n: int):
        self._nqubits = n
        if self.partition is None: # pragma: no cover
            self.partition = list(range(n // 2 + n % 2))

        if len(self.partition) < n // 2:
            # Revert parition so that we diagonalize a smaller matrix
            self.partition = [i for i in range(n)
                              if i not in set(self.partition)]
        self.rho_dim = 2 ** (n - len(self.partition))
        self._traceout = None

    @property
    def _traceout_str(self):
        """Einsum string used to trace out when state is density matrix."""
        if self._traceout is None:
            from qibo.tensorflow.einsum import DefaultEinsum
            partition = set(self.partition)
            self._traceout = DefaultEinsum.partialtrace_str(partition, self.nqubits)
        return self._traceout

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
            rho = tf.einsum(self._traceout_str, state)
        else:
            rho = tf.tensordot(state, tf.math.conj(state),
                               axes=[self.partition, self.partition])
        return tf.reshape(rho, (self.rho_dim, self.rho_dim))


class EntanglementEntropy(PartialTrace):
    """Von Neumann entanglement entropy callback.

    .. math::
        S = \\mathrm{Tr} \\left ( \\rho \\log _2 \\rho \\right )

    Args:
        partition (list): List with qubit ids that defines the first subsystem
            for the entropy calculation.
            If `partition` is not given then the first subsystem is the first
            half of the qubits.

    Example:
        ::

            from qibo import models, gates, callbacks
            # create entropy callback where qubit 0 is the first subsystem
            entropy = callbacks.EntanglementEntropy([0])
            # initialize circuit with 2 qubits and add gates
            c = models.Circuit(2)
            # add callback gates between normal gates
            c.add(gates.CallbackGate(entropy))
            c.add(gates.H(0))
            c.add(gates.CallbackGate(entropy))
            c.add(gates.CNOT(0, 1))
            c.add(gates.CallbackGate(entropy))
            # execute the circuit
            final_state = c()
            print(entropy[:])
            # Should print [0, 0, 1] which is the entanglement entropy
            # after every gate in the calculation.
    """
    _log2 = tf.cast(tf.math.log(2.0), dtype=DTYPES.get('DTYPE'))

    @classmethod
    def _entropy(cls, rho: tf.Tensor) -> tf.Tensor:
      """Calculates entropy by diagonalizing the density matrix."""
      # Diagonalize
      eigvals = tf.math.real(tf.linalg.eigvalsh(rho))
      # Treating zero and negative eigenvalues
      masked_eigvals = tf.gather(eigvals, tf.where(eigvals > EIGVAL_CUTOFF))[:, 0]
      entropy = - tf.reduce_sum(masked_eigvals * tf.math.log(masked_eigvals))
      return entropy / cls._log2

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        # Construct reduced density matrix
        rho = super(EntanglementEntropy, self).__call__(state, is_density_matrix)
        # Calculate entropy of reduced density matrix
        return self._entropy(rho)


class Norm(Callback):
    """State norm callback.

    .. math::
        \\mathrm{Norm} = \\left \\langle \\Psi | \\Psi \\right \\rangle
        = \\mathrm{Tr} (\\rho )
    """

    @staticmethod
    def norm(state: tf.Tensor, is_density_matrix: bool = False) -> tf.Tensor:
        """"""
        if is_density_matrix:
            return tf.linalg.trace(state)
        return tf.sqrt(tf.reduce_sum(tf.square(tf.abs(state))))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        return self.norm(state, is_density_matrix)


class Overlap(Callback):
    """State overlap callback.

    Calculates the overlap between the circuit state and a given target state:

    .. math::
        \\mathrm{Overlap} = |\\left \\langle \\Phi | \\Psi \\right \\rangle |

    Args:
        state (np.ndarray): Target state to calculate overlap with.
        normalize (bool): If ``True`` the states are normalized for the overlap
            calculation.
    """

    def __init__(self, state: Union[np.ndarray, tf.Tensor]):
        super(Overlap, self).__init__()
        self.statec = tf.math.conj(tf.cast(state, dtype=DTYPES.get('DTYPECPX')))

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        if is_density_matrix:
            raise_error(NotImplementedError, "Overlap callback is not implemented "
                                             "for density matrices.")

        return tf.abs(tf.reduce_sum(self.statec * state))


class Energy(Callback):
    """Energy expectation value callback.

    Calculates the expectation value of a given Hamiltonian as:

    .. math::
        \\left \\langle H \\right \\rangle =
        \\left \\langle \\Psi | H | \\Psi \\right \\rangle
        = \\mathrm{Tr} (\\rho H)

    assuming that the state is normalized.

    Args:
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): Hamiltonian
            object to calculate its expectation value.
    """

    def __init__(self, hamiltonian: "hamiltonians.Hamiltonian"):
        super(Energy, self).__init__()
        self.hamiltonian = hamiltonian

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        if is_density_matrix:
            return tf.linalg.trace(tf.matmul(self.hamiltonian.matrix,
                                             state))
        return self.hamiltonian.expectation(state)


class Gap(Callback):
    """Callback for calculating the gap of adiabatic evolution Hamiltonians.

    Can also be used to calculate the Hamiltonian eigenvalues at each time step
    during the evolution.
    Note that this callback can only be added in
    :class:`qibo.evolution.AdiabaticEvolution` models.

    Args:
        mode (str/int): Defines which quantity this callback calculates.
            If ``mode == 'gap'`` then the difference between ground state and
            first excited state energy (gap) is calculated.
            If ``mode`` is an integer, then the energy of the corresponding
            eigenstate is calculated.

    Example:
        ::

            from qibo import models, callbacks
            # define easy and hard Hamiltonians for adiabatic evolution
            h0 = hamiltonians.X(3)
            h1 = hamiltonians.TFIM(3, h=1.0)
            # define callbacks for logging the ground state, first excited
            # and gap energy
            ground = callbacks.Gap(0)
            excited = callbacks.Gap(1)
            gap = callbacks.Gap()
            # define and execute the ``AdiabaticEvolution`` model
            evolution = AdiabaticEvolution(h0, h1, lambda t: t, dt=1e-1,
                                           callbacks=[gap, ground, excited])
            final_state = evolution(final_time=1.0)
            # print results
            print(ground[:])
            print(excited[:])
            print(gap[:])
    """

    def __init__(self, mode: Union[str, int] = "gap"):
        super(Gap, self).__init__()
        if isinstance(mode, str):
            if mode != "gap":
                raise_error(ValueError, "Unsupported mode {} for gap callback."
                                        "".format(mode))
        elif not isinstance(mode, int):
            raise_error(TypeError, "Gap callback mode should be integer or "
                                   "string but is {}.".format(type(mode)))
        self._evolution = None
        self.mode = mode

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

    def __call__(self, state: tf.Tensor, is_density_matrix: bool = False
                 ) -> tf.Tensor:
        if self.evolution is None:
            raise_error(ValueError, "Gap callback can only be used in "
                                    "adiabatic evolution models.")
        if is_density_matrix:
            raise_error(NotImplementedError, "Adiabatic evolution gap callback "
                                             "is not implemented for density "
                                             "matrices.")
        hamiltonian = self.evolution.hamiltonian()
        # Call the eigenvectors so that they are cached for the ``exp`` call
        hamiltonian.eigenvectors()
        if isinstance(self.mode, int):
            return tf.math.real(hamiltonian.eigenvalues()[self.mode])
        # case: self.mode == "gap"
        return tf.math.real(hamiltonian.eigenvalues()[1] -
                            hamiltonian.eigenvalues()[0])
