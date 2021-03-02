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
        self.partition = partition
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
        check_degenerate (bool): If ``True`` the excited state number is
            increased until a non-zero gap is found. This is used to find the
            proper gap in the case of degenerate Hamiltonians.
            This flag is relevant only if ``mode`` is ``'gap'``.
            Default is ``True``.

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

    def __init__(self, mode: Union[str, int] = "gap", check_degenerate: bool = True):
        super(Gap, self).__init__()
        if isinstance(mode, str):
            if mode != "gap":
                raise_error(ValueError, "Unsupported mode {} for gap callback."
                                        "".format(mode))
        elif not isinstance(mode, int):
            raise_error(TypeError, "Gap callback mode should be integer or "
                                   "string but is {}.".format(type(mode)))
        self.mode = mode
        self.check_degenerate = check_degenerate
