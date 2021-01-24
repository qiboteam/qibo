from qibo.config import raise_error
from typing import List, Optional, Set, Union


class Callback:
    """Base callback class.

    Callbacks should inherit this class and implement its
    `state_vector_call` and `density_matrix_call` methods.

    Results of a callback can be accessed by indexing the corresponding object.
    """

    def __init__(self):
        self._results = []
        self._nqubits = None
        self._density_matrix = False
        self._active_call = "state_vector_call"

    @property
    def nqubits(self): # pragma: no cover
        """Total number of qubits in the circuit that the callback was added in."""
        # abstract method
        return self._nqubits

    @nqubits.setter
    def nqubits(self, n: int): # pragma: no cover
        # abstract method
        self._nqubits = n

    @property
    def density_matrix(self):
        return self._density_matrix

    @density_matrix.setter
    def density_matrix(self, x):
        self._density_matrix = x
        if x:
            self._active_call = "density_matrix_call"
        else:
            self._active_call = "state_vector_call"

    @property
    def results(self):
        return self._results

    def append(self, x):
        self._results.append(x)

    def extend(self, x):
        self._results.extend(x)


class PartialTrace(Callback):
    """Calculates reduced density matrix of a state.

    This is used by the :class:`qibo.core.callbacks.EntanglementEntropy`
    callback. It can also be used as a standalone callback in order to access
    a reduced density matrix in the middle of a circuit execution.

    Args:
        partition (list): List with qubit ids that defines the first subsystem.
            If `partition` is not given then the first subsystem is the first
            half of the qubits.
    """
    from qibo.config import EINSUM_CHARS

    def __init__(self, partition: Optional[List[int]] = None):
        super().__init__()
        self.partition = partition
        self.rho_dim = None
        self._traceout = None

    @Callback.nqubits.setter
    def nqubits(self, n: int):
        self._nqubits = n
        if self.partition is None: # pragma: no cover
            self.partition = list(range(n // 2 + n % 2))

        if len(self.partition) <= n // 2:
            # Revert parition so that we diagonalize a smaller matrix
            self.partition = [i for i in range(n)
                              if i not in set(self.partition)]
        self.rho_dim = 2 ** (n - len(self.partition))
        self._traceout = None

    @classmethod
    def einsum_string(cls, qubits: Set[int], nqubits: int,
                      measuring: bool = False) -> str:
        """Generates einsum string for partial trace of density matrices.

        This method is also used in :meth:`qibo.core.cgates.M.prepare`.

        Args:
            qubits (list): Set of qubit ids that are traced out.
            nqubits (int): Total number of qubits in the state.
            measuring (bool): If True non-traced-out indices are multiplied and
                the output has shape (nqubits - len(qubits),).
                If False the output has shape 2 * (nqubits - len(qubits),).

        Returns:
            String to use in einsum for performing partial density of a
            density matrix.
        """
        if (2 - int(measuring)) * nqubits > len(cls.EINSUM_CHARS): # pragma: no cover
            # case not tested because it requires large instance
            raise_error(NotImplementedError, "Not enough einsum characters.")

        left_in, right_in, left_out, right_out = [], [], [], []
        for i in range(nqubits):
            left_in.append(cls.EINSUM_CHARS[i])
            if i in qubits:
                right_in.append(cls.EINSUM_CHARS[i])
            else:
                left_out.append(cls.EINSUM_CHARS[i])
                if measuring:
                    right_in.append(cls.EINSUM_CHARS[i])
                else:
                    right_in.append(cls.EINSUM_CHARS[i + nqubits])
                    right_out.append(cls.EINSUM_CHARS[i + nqubits])

        left_in, left_out = "".join(left_in), "".join(left_out)
        right_in, right_out = "".join(right_in), "".join(right_out)
        return f"{left_in}{right_in}->{left_out}{right_out}"

    def traceout(self):
        if self._traceout is None:
            partition = set(self.partition)
            self._traceout = self.einsum_string(partition, self.nqubits)
        return self._traceout


class EntanglementEntropy(Callback):
    """Von Neumann entanglement entropy callback.

    .. math::
        S = \\mathrm{Tr} \\left ( \\rho \\log _2 \\rho \\right )

    Args:
        partition (list): List with qubit ids that defines the first subsystem
            for the entropy calculation.
            If `partition` is not given then the first subsystem is the first
            half of the qubits.
        compute_spectrum (bool): Compute the entanglement spectrum. Default is False.

    Example:
        ::

            from qibo import models, gates, callbacks
            # create entropy callback where qubit 0 is the first subsystem
            entropy = callbacks.EntanglementEntropy([0], compute_spectrum=True)
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
            print(entropy.spectrum)
            # Print the entanglement spectrum.
    """

    def __init__(self, partition: Optional[List[int]] = None,
                 compute_spectrum: bool = False):
        super().__init__()
        self.compute_spectrum = compute_spectrum
        self.spectrum = list()


class Norm(Callback):
    """State norm callback.

    .. math::
        \\mathrm{Norm} = \\left \\langle \\Psi | \\Psi \\right \\rangle
        = \\mathrm{Tr} (\\rho )
    """
    pass


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
    pass


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
        super().__init__()
        self.hamiltonian = hamiltonian


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
        self.mode = mode
