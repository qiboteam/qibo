from typing import List, Optional, Union

from qibo.config import raise_error


class Callback:
    """Base callback class.

    Results of a callback can be accessed by indexing the corresponding object.
    """

    def __init__(self):
        self._results = []
        self._nqubits = None

    @property
    def nqubits(self):  # pragma: no cover
        """Total number of qubits in the circuit that the callback was added in."""
        # abstract method
        return self._nqubits

    @nqubits.setter
    def nqubits(self, n: int):  # pragma: no cover
        # abstract method
        self._nqubits = n

    @property
    def results(self):
        return self._results

    def append(self, x):
        self._results.append(x)

    def extend(self, x):
        self._results.extend(x)

    def __getitem__(self, k):
        if not isinstance(k, (int, slice, list, tuple)):
            raise_error(IndexError, f"Unrecognized type for index {k}.")

        if isinstance(k, int) and k >= len(self._results):
            raise_error(
                IndexError,
                f"Attempting to access callbacks {k} run but "
                + f"the callback has been used in {len(self._results)} executions.",
            )

        return self._results[k]

    def apply(self, backend, state):  # pragma: no cover
        pass

    def apply_density_matrix(self, backend, state):  # pragma: no cover
        pass


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
        .. testcode::

            from qibo import Circuit, gates
            from qibo.callbacks import EntanglementEntropy

            # create entropy callback where qubit 0 is the first subsystem
            entropy = EntanglementEntropy([0], compute_spectrum=True)
            # initialize circuit with 2 qubits and add gates
            circuit = Circuit(2)
            # add callback gates between normal gates
            circuit.add(gates.CallbackGate(entropy))
            circuit.add(gates.H(0))
            circuit.add(gates.CallbackGate(entropy))
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.CallbackGate(entropy))
            # execute the circuit
            final_state = circuit()
            print(entropy[:])
            # Should print [0, 0, 1] which is the entanglement entropy
            # after every gate in the calculation.
            print(entropy.spectrum)
            # Print the entanglement spectrum.
        .. testoutput::
            :hide:

            ...
    """

    def __init__(
        self,
        partition: Optional[List[int]] = None,
        compute_spectrum: bool = False,
        base: float = 2,
        check_hermitian: bool = False,
    ):
        super().__init__()
        self.partition = partition
        self.compute_spectrum = compute_spectrum
        self.base = base
        self.check_hermitian = check_hermitian
        self.spectrum = list()

    @Callback.nqubits.setter
    def nqubits(self, n: int):
        if self._nqubits is not None and self._nqubits != n:
            raise_error(
                RuntimeError,
                f"Changing EntanglementEntropy nqubits from {self._nqubits} to {n}.",
            )
        self._nqubits = n
        if self.partition is None:
            self.partition = list(range(n // 2 + n % 2))
        if len(self.partition) <= self._nqubits // 2:
            self.partition = [
                i for i in range(self._nqubits) if i not in set(self.partition)
            ]

    def apply(self, backend, state):
        from qibo.quantum_info.entropies import entanglement_entropy

        entropy, spectrum = entanglement_entropy(
            state,
            bipartition=self.partition,
            base=self.base,
            check_hermitian=self.check_hermitian,
            return_spectrum=True,
            backend=backend,
        )
        self.append(entropy)
        if self.compute_spectrum:
            self.spectrum.append(spectrum)
        return entropy

    def apply_density_matrix(self, backend, state):
        return self.apply(backend, state)


class State(Callback):
    """Callback to keeps track of the full state during circuit execution.

    Warning: Keeping many copies of states in memory requires a lot of memory
    for circuits with many qubits.

    Args:
        copy (bool): If ``True`` the state vector or density matrix is
            copied in memory. Otherwise a reference to the existing array
            is stored in the callback.
            The callback will not work as expected if ``copy=False``
            is used with a backend that performs in-place updates,
            such as qibojit.
            Default is True
    """

    def __init__(self, copy=True):
        super().__init__()
        self.copy = copy

    def apply(self, backend, state):
        self.append(backend.cast(state, copy=self.copy))
        return state

    def apply_density_matrix(self, backend, state):
        self.append(backend.cast(state, copy=self.copy))
        return state


class Norm(Callback):
    """State norm callback.

    .. math::
        \\mathrm{Norm} = \\left \\langle \\Psi | \\Psi \\right \\rangle
        = \\mathrm{Tr} (\\rho )
    """

    def apply(self, backend, state):
        norm = backend.calculate_vector_norm(state)
        self.append(norm)
        return norm

    def apply_density_matrix(self, backend, state):
        norm = backend.calculate_matrix_norm(state)
        self.append(norm)
        return norm


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

    def __init__(self, state):
        super().__init__()
        self.state = state

    def apply(self, backend, state):
        overlap = backend.calculate_overlap(self.state, state)
        self.append(overlap)
        return overlap

    def apply_density_matrix(self, backend, state):
        overlap = backend.calculate_overlap_density_matrix(self.state, state)
        self.append(overlap)
        return overlap


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

    def __init__(self, hamiltonian: "hamiltonians.Hamiltonian"):  # type: ignore
        super().__init__()
        self.hamiltonian = hamiltonian

    def apply(self, backend, state):
        assert type(self.hamiltonian.backend) == type(backend)
        expectation = self.hamiltonian.expectation(state)
        self.append(expectation)
        return expectation

    def apply_density_matrix(self, backend, state):
        assert type(self.hamiltonian.backend) == type(backend)
        expectation = self.hamiltonian.expectation(state)
        self.append(expectation)
        return expectation


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

        .. testcode::

            from qibo import callbacks, hamiltonians
            from qibo.models import AdiabaticEvolution
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
        .. testoutput::
            :hide:

            ...
    """

    def __init__(self, mode: Union[str, int] = "gap", check_degenerate: bool = True):
        super().__init__()
        if not isinstance(mode, (int, str)):
            raise_error(
                TypeError,
                f"Gap callback mode should be integer or string but is {type(mode)}.",
            )
        elif isinstance(mode, str) and mode != "gap":
            raise_error(ValueError, f"Unsupported mode {mode} for gap callback.")
        self.mode = mode
        self.check_degenerate = check_degenerate
        self.evolution = None

    def apply(self, backend, state):
        from qibo.config import EIGVAL_CUTOFF, log

        if self.evolution is None:
            raise_error(
                RuntimeError,
                "Gap callback can only be used in " "adiabatic evolution models.",
            )
        hamiltonian = self.evolution.solver.current_hamiltonian  # pylint: disable=E1101
        assert type(hamiltonian.backend) == type(backend)
        # Call the eigenvectors so that they are cached for the ``exp`` call
        hamiltonian.eigenvectors()
        eigvals = hamiltonian.eigenvalues()
        if isinstance(self.mode, int):
            gap = backend.np.real(eigvals[self.mode])
            self.append(gap)
            return gap

        # case: self.mode == "gap"
        excited = 1
        gap = backend.np.real(eigvals[excited] - eigvals[0])

        if not self.check_degenerate:
            self.append(gap)
            return gap

        while backend.np.less(gap, EIGVAL_CUTOFF):
            gap = backend.np.real(eigvals[excited] - eigvals[0])
            excited += 1
        if excited > 1:
            log.warning(
                f"The Hamiltonian is degenerate. Using eigenvalue {excited} to calculate gap."
            )
        self.append(gap)
        return gap

    def apply_density_matrix(self, backend, state):
        raise_error(
            NotImplementedError,
            "Gap callback is not implemented for " "density matrices.",
        )
