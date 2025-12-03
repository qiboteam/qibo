import abc
from string import ascii_letters
from typing import List, Optional, Tuple, Union

from scipy.sparse import _matrix

from qibo.config import raise_error


class Backend(abc.ABC):
    def __init__(self):
        super().__init__()
        self.name = "backend"
        self.platform = None

        self.dtype = "complex128"
        self.matrices = None

        self.device = "/CPU:0"
        self.nthreads = 1
        self.supports_multigpu = False
        self.oom_error = MemoryError

        # computation engine
        self.np = None

    def __reduce__(self):
        """Allow pickling backend objects that have references to modules."""
        return self.__class__, tuple()

    def __repr__(self):
        if self.platform is None:
            return self.name

        return f"{self.name} ({self.platform})"

    @property
    @abc.abstractmethod
    def qubits(self) -> Optional[list[Union[int, str]]]:  # pragma: no cover
        """Return the qubit names of the backend. If :class:`SimulationBackend`, return None."""
        raise_error(NotImplementedError)

    @property
    @abc.abstractmethod
    def connectivity(
        self,
    ) -> Optional[list[tuple[Union[int, str], Union[int, str]]]]:  # pragma: no cover
        """Return the available qubit pairs of the backend. If :class:`SimulationBackend`, return None."""
        raise_error(NotImplementedError)

    @property
    @abc.abstractmethod
    def natives(self) -> Optional[list[str]]:  # pragma: no cover
        """Return the native gates of the backend. If :class:`SimulationBackend`, return None."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def set_dtype(self, dtype: str):  # pragma: no cover
        """Set data type of arrays created using the backend.

        .. note::
            The data types ``float32`` and ``float64`` are intended to be used when the circuits
            to be simulated only contain gates with real-valued matrix representations.
            Using one of the aforementioned data types with circuits that contain complex-valued
            matrices will raise a casting error.

        .. note::
            List of gates that always admit a real-valued matrix representation:
            :class:`qibo.gates.I`, :class:`qibo.gates.X`, :class:`qibo.gates.Z`,
            :class:`qibo.gates.H`, :class:`qibo.gates.Align`, :class:`qibo.gates.RY`,
            :class:`qibo.gates.CNOT`, :class:`qibo.gates.CZ`, :class:`qibo.gates.CRY`,
            :class:`qibo.gates.SWAP`, :class:`qibo.gates.FSWAP`, :class:`qibo.gates.GIVENS`,
            :class:`qibo.gates.RBS`, :class:`qibo.gates.TOFFOLI`, :class:`qibo.gates.CCZ`,
            and :class:`qibo.gates.FanOut`.

        .. note::
            The following parametrized gates can have real-valued matrix representations
            depending on the values of their parameters:
            :class:`qibo.gates.RX`, :class:`qibo.gates.RZ`, :class:`qibo.gates.U1`,
            :class:`qibo.gates.U2`, :class:`qibo.gates.U3`, :class:`qibo.gates.CRX`,
            :class:`qibo.gates.CRZ`, :class:`qibo.gates.CU1`, :class:`qibo.gates.CU2`,
            :class:`qibo.gates.CU3`, :class:`qibo.gates.fSim`, :class:`qibo.gates.GeneralizedfSim`,
            :class:`qibo.gates.RXX`, :class:`qibo.gates.RYY`, :class:`qibo.gates.RZZ`,
            :class:`qibo.gates.RZX`, and :class:`qibo.gates.GeneralizedRBS`.

        Args:
            dtype (str): the options are the following: ``complex128``, ``complex64``,
                ``float64``, and ``float32``.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def set_device(self, device: str):  # pragma: no cover
        """Set simulation device.

        Args:
            device (str): Device such as '/CPU:0', '/GPU:0', etc.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def set_threads(self, nthreads: int):  # pragma: no cover
        """Set number of threads for CPU simulation.

        Args:
            nthreads (int): Number of threads.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def cast(self, x, dtype=None, copy: bool = False):  # pragma: no cover
        """Cast an object as the array type of the current backend.

        Args:
            x: Object to cast to array.
            dtype (str or type, optional): data type of ``x`` after casting.
                Options are ``"complex128"``, ``"complex64"``, ``"float64"``,
                or ``"float32"``. If ``None``, defaults to ``Backend.dtype``.
                Defaults to ``None``.
            copy (bool, optional): If ``True`` a copy of the object is created in memory.
                Defaults to ``False``.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def is_sparse(self, x):  # pragma: no cover
        """Determine if a given array is a sparse tensor."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def to_numpy(self, x):  # pragma: no cover
        """Cast a given array to numpy."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def compile(self, func):  # pragma: no cover
        """Compile the given method.

        Available only for the tensorflow backend.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def zero_state(self, nqubits: int):  # pragma: no cover
        """Generate :math:`|000 \\cdots 0 \\rangle` state vector as an array."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def zero_density_matrix(self, nqubits: int):  # pragma: no cover
        """Generate :math:`|000\\cdots0\\rangle\\langle000\\cdots0|` density matrix as an array."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def identity_density_matrix(
        self, nqubits: int, normalize: bool = True
    ):  # pragma: no cover
        """Generate density matrix

        .. math::
            \\rho = \\frac{1}{2^\\text{nqubits}} \\, \\sum_{k=0}^{2^\\text{nqubits} - 1} \\,
                |k \\rangle \\langle k|

        if ``normalize=True``. If ``normalize=False``, returns the unnormalized
        Identity matrix, which is equivalent to :func:`numpy.eye`.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def plus_state(self, nqubits: int):  # pragma: no cover
        """Generate :math:`|+++\\cdots+\\rangle` state vector as an array."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def plus_density_matrix(self, nqubits: int):  # pragma: no cover
        """Generate :math:`|+++\\cdots+\\rangle\\langle+++\\cdots+|` density matrix as an array."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def matrix(self, gate: "qibo.gates.abstract.Gate"):  # pragma: no cover
        """Convert a :class:`qibo.gates.Gate` to the corresponding matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def matrix_parametrized(self, gate: "qibo.gates.abstract.Gate"):  # pragma: no cover
        """Equivalent to :meth:`qibo.backends.abstract.Backend.matrix` for parametrized gates."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def matrix_fused(self, gate):  # pragma: no cover
        """Fuse matrices of multiple gates."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_gate(self, gate, state, nqubits: int):  # pragma: no cover
        """Apply a gate to state vector."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_gate_density_matrix(self, gate, state, nqubits: int):  # pragma: no cover
        """Apply a gate to density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_gate_half_density_matrix(
        self, gate, state, nqubits: int
    ):  # pragma: no cover
        """Apply a gate to one side of the density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_channel(self, channel, state, nqubits: int):  # pragma: no cover
        """Apply a channel to state vector."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_channel_density_matrix(
        self, channel, state, nqubits: int
    ):  # pragma: no cover
        """Apply a channel to density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def collapse_state(
        self, state, qubits, shot, nqubits: int, normalize: bool = True
    ):  # pragma: no cover
        """Collapse state vector according to measurement shot."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def collapse_density_matrix(
        self, state, qubits, shot, nqubits: int, normalize: bool = True
    ):  # pragma: no cover
        """Collapse density matrix according to measurement shot."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def reset_error_density_matrix(self, gate, state, nqubits: int):  # pragma: no cover
        """Apply reset error to density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def thermal_error_density_matrix(
        self, gate, state, nqubits: int
    ):  # pragma: no cover
        """Apply thermal relaxation error to density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def execute_circuit(
        self, circuit, initial_state=None, nshots: int = None
    ):  # pragma: no cover
        """Execute a :class:`qibo.models.circuit.Circuit`."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def execute_circuits(
        self, circuits, initial_states=None, nshots: int = None
    ):  # pragma: no cover
        """Execute multiple :class:`qibo.models.circuit.Circuit` in parallel."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def execute_circuit_repeated(
        self, circuit: "qibo.models.circuit.Circuit", nshots: int, initial_state=None
    ):  # pragma: no cover
        """Execute a :class:`qibo.models.circuit.Circuit` multiple times.

        Useful for noise simulation using state vectors or for simulating gates
        controlled by measurement outcomes.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def execute_distributed_circuit(
        self, circuit, initial_state=None, nshots: int = None
    ):  # pragma: no cover
        """Execute a :class:`qibo.models.circuit.Circuit` using multiple GPUs."""
        raise_error(NotImplementedError)

    def expectation_observable_dense(self, circuit: "Circuit", observable: "ndarray"):
        """Compute the expectation value of a generic dense hamiltonian starting from the state.

        Args:
            circuit (Circuit): the circuit to calculate the expectation value from.
            observable (ndarray): the matrix corresponding to the observable.
        Returns:
            float: The calculated expectation value.
        """
        result = (
            circuit._final_state
            if circuit._final_state is not None
            else self.execute_circuit(circuit)
        )
        state = result.state()
        if circuit.density_matrix:
            return self.calculate_expectation_density_matrix(observable, state, False)
        return self.calculate_expectation_state(observable, state, False)

    def expectation_diagonal_observable_dense_from_samples(
        self,
        circuit: "Circuit",
        observable: "ndarray",
        nqubits: int,
        nshots: int,
        qubit_map: Optional[Tuple[int, ...]] = None,
    ) -> float:
        """Compute the expectation value of a dense Hamiltonian diagonal in a defined basis
        starting from the samples (measured in the same basis).

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): the circuit to calculate the expectation value from.
            observable (ndarray): the (diagonal) matrix corresponding to the observable.
            nqubits (int): the number of qubits of the observable.
            nshots (int): how many shots to execute the circuit with.
            qubit_map (Tuple[int, ...], optional): optional qubits reordering.

        Returns:
            float: The calculated expectation value.
        """
        result = (
            circuit._final_state
            if circuit._final_state is not None
            else self.execute_circuit(circuit, nshots=nshots)
        )

        freq = result.frequencies()
        diag = self.np.diagonal(observable)
        if self.np.count_nonzero(observable - self.np.diag(diag)) != 0:
            raise_error(
                NotImplementedError,
                "Observable is not diagonal. Expectation of non-diagonal observables starting "
                + "from samples is currently supported for `qibo.hamiltonians.SymbolicHamiltonian` only.",
            )
        diag = self.np.reshape(diag, nqubits * (2,))
        if qubit_map is None:
            qubit_map = tuple(range(nqubits))
        diag = self.np.transpose(diag, qubit_map).ravel()
        # select only the elements with non-zero counts
        diag = diag[[int(state, 2) for state in freq.keys()]]
        counts = self.cast(list(freq.values()), dtype=diag.dtype) / sum(freq.values())
        return self.np.real(self.np.sum(diag * counts))

    def expectation_diagonal_observable_symbolic_from_samples(
        self,
        circuit: "Circuit",
        nqubits: int,
        terms_qubits: List[Tuple[int, ...]],
        terms_coefficients: List[float],
        nshots: int,
        qubit_map: Optional[Union[Tuple[int, ...], List[int]]] = None,
    ) -> float:
        """Compute the expectation value of a symbolic observable diagonal in the computational basis,
        starting from the samples.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): the circuit to calculate the expectation value from.
            nqubits (int): number of qubits of the observable.
            terms_qubits (List[Tuple[int, ...]]): the qubits each term of the (diagonal) symbolic observable is acting on.
            terms_coefficients (List[float]): the coefficient of each term of the (diagonal) symbolic observable.
            constant (float): the constant term of the observable. Defaults to ``0.``.
            nshots (int): how many shots to execute the circuit with.
            qubit_map (Tuple[int, ...]): custom qubit ordering.

        Returns:
            float: The calculated expectation value.
        """
        result = (
            circuit._final_state
            if circuit._final_state is not None
            else self.execute_circuit(circuit, nshots=nshots)
        )
        if qubit_map is None:
            qubit_map = range(nqubits)
        qubit_map = list(qubit_map)

        freq = result.frequencies()
        keys = list(freq.keys())
        counts = list(freq.values())
        counts = self.cast(counts, dtype=self.np.float64) / sum(counts)
        expvals = []
        for qubits, coefficient in zip(terms_qubits, terms_coefficients):
            expvals.extend(
                [
                    coefficient
                    * (-1) ** [state[qubit_map.index(q)] for q in qubits].count("1")
                    for state in keys
                ]
            )
        expvals = self.cast(expvals, dtype=counts.dtype).reshape(
            len(terms_coefficients), len(freq)
        )
        return self.np.sum(expvals @ counts)

    def expectation_observable_symbolic_from_samples(
        self,
        circuit,
        diagonal_terms_coefficients: List[List[float]],
        diagonal_terms_observables: List[List[str]],
        diagonal_terms_qubits: List[List[Tuple[int, ...]]],
        nqubits: int,
        constant: float,
        nshots: int,
    ) -> float:
        """Compute the expectation value of a general symbolic observable defined by groups of terms
        that can be diagonalized simultaneously, starting from the samples.

        Args:
            circuit (Circuit): the circuit to calculate the expectation value from.
            diagonal_terms_coefficients (List[float]): the coefficients of each term of the (diagonal) symbolic observable.
            diagonal_terms_observables (List[List[str]]): the lists of strings defining the observables
                for each group of terms, e.g. ``[['IXZ', 'YII'], ['IYZ', 'XIZ']]``.
            diagonal_terms_qubits (List[Tuple[int, ...]]): the qubits each term of the groups is acting on,
                e.g. ``[[(0,1,2), (1,3)], [(2,1,3), (2,4)]]``.
            nqubits (int): number of qubits of the observable.
            constant (float): the constant term of the observable.
            nshots (int): how many shots to execute the circuit with.

        Returns:
            float: The calculated expectation value.
        """
        from qibo import gates  # pylint: disable=import-outside-toplevel

        rotated_circuits = []
        qubit_maps = []
        # loop over the terms that can be diagonalized simultaneously
        for terms_qubits, terms_observables in zip(
            diagonal_terms_qubits, diagonal_terms_observables
        ):
            # for each term that can be diagonalized simultaneously
            # preapare the basis rotation for the measurement
            # if nshots is None, additionally construct the matrix of
            # the global observable

            measurements = {}
            for qubits, observable in zip(terms_qubits, terms_observables):
                # Only care about non-I terms
                # prepare the measurement basis and append it to the circuit
                for qubit, factor in zip(qubits, observable):
                    if factor != "I" and qubit not in measurements:
                        measurements[qubit] = gates.M(
                            qubit, basis=getattr(gates, factor)
                        )

            # Get the qubits we want to measure for each term
            qubit_maps.append(measurements.keys())

            circ_copy = circuit.copy(True)
            circ_copy.add(list(measurements.values()))
            rotated_circuits.append(circ_copy)

        # execute the circuits
        # the results are saved in the circuit._final_state
        # that are used inside the calculation of the expectation
        # values
        if len(rotated_circuits) > 1:
            _ = self.execute_circuits(rotated_circuits, nshots=nshots)
        else:
            _ = self.execute_circuit(rotated_circuits[0], nshots=nshots)

        # construct the expectation value for each diagonal term
        # and sum all together
        expval = 0.0
        for circ, terms_qubits, terms_coefficients, qmap in zip(
            rotated_circuits,
            diagonal_terms_qubits,
            diagonal_terms_coefficients,
            qubit_maps,
        ):
            expval += self.expectation_diagonal_observable_symbolic_from_samples(
                circ,
                nqubits,
                terms_qubits,
                terms_coefficients,
                nshots,
                qmap,
            )
        return constant + expval

    def expectation_observable_symbolic(
        self,
        circuit: "Circuit",
        terms: List[str],
        term_qubits: List[Tuple[int, ...]],
        term_coefficients: List[float],
        nqubits: int,
    ):
        """Compute the expectation value of a general symbolic observable that is a sum of terms.

        In particular, each term of the observable is contracted with
        the corresponding subspace defined by the qubits it acts on.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): the circuit to calculate the expectation value from.
            terms (List[str]): the lists of strings defining the observables for each term, e.g.
                ``['ZXZ', 'YI', 'IYZ', 'X']``.
            term_coefficients (List[float]): the coefficients of each term.
            term_qubits (List[Tuple[int, ...]]): the qubits each term is acting on, e.g.
                ``[(0,1,2), (1,3), (2,1,3), (4,)]``.
            nqubits (int): number of qubits of the observable.

        Returns:
            float: The calculated expectation value.
        """
        # get the final state
        result = (
            circuit._final_state
            if circuit._final_state is not None
            else self.execute_circuit(circuit)
        )
        # get the state and separate it in the single qubits
        # subspaces
        state = result.state()
        N = len(state.shape) * nqubits
        shape = N * (2,)
        state = self.np.reshape(state, shape)
        # prepare the state indices for the contraction
        if circuit.density_matrix:
            state_indices = [ascii_letters[i] for i in range(N)]
        else:
            state_indices = [ascii_letters[i] for i in range(2 * N)]
            state_dag_indices = state_indices[:nqubits]
            state_indices = state_indices[nqubits:]
            state_dag_string = "".join(state_dag_indices)
        state_string = "".join(state_indices)

        # for each term get the matrices
        # acting on the separate qubits
        # and contract them with the corresponding
        # subspace of the state
        expval = 0.0
        for term, qubits, coefficient in zip(terms, term_qubits, term_coefficients):
            # per qubit matrices
            term_matrices = {
                qubit: getattr(self.matrices, factor)
                for factor, qubit in zip(term, qubits)
                if factor != "I"
            }
            qubits, matrices = zip(*term_matrices.items())
            # prepare the observable/state indices
            # for contraction
            if circuit.density_matrix:
                obs_indices = [
                    state_indices[i + nqubits] + state_indices[i] for i in qubits
                ]
                obs_string = ",".join(obs_indices)
                new_string = state_string[:]
                for q in set(range(nqubits)) - set(qubits):
                    new_string = (
                        new_string[:q] + new_string[q + nqubits] + new_string[q + 1 :]
                    )
                # contraction:
                # for a 3 qubits density matrix and an observable
                # acting on qubits (0,1), you have
                # "da,fc,abcdbf->"
                expval += self.np.real(
                    coefficient
                    * self.np.einsum(
                        f"{obs_string},{new_string}->",
                        *matrices,
                        state,
                    )
                )
            else:
                obs_indices = [state_dag_indices[i] + state_indices[i] for i in qubits]
                obs_string = ",".join(obs_indices)
                new_string = state_string[:]
                for q in set(range(nqubits)) - set(qubits):
                    new_string = (
                        new_string[:q] + state_dag_string[q] + new_string[q + 1 :]
                    )
                # contraction:
                # for a 3 qubits density matrix and an observable
                # acting on qubits (0,1), you have
                # "abc,ad,cf,dbf->"
                expval += self.np.real(
                    coefficient
                    * self.np.einsum(
                        f"{state_dag_string},{obs_string},{new_string}->",
                        self.np.conj(state),
                        *matrices,
                        state,
                    )
                )
        return expval

    @abc.abstractmethod
    def calculate_symbolic(
        self,
        state,
        nqubits: int,
        decimals: int = 5,
        cutoff: float = 1e-10,
        max_terms: int = 20,
    ):  # pragma: no cover
        """Dirac representation of a state vector."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_symbolic_density_matrix(
        self,
        state,
        nqubits: int,
        decimals: int = 5,
        cutoff: float = 1e-10,
        max_terms: int = 20,
    ):  # pragma: no cover
        """Dirac representation of a density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_probabilities(self, state, qubits, nqubits: int):  # pragma: no cover
        """Calculate probabilities given a state vector."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_probabilities_density_matrix(
        self, state, qubits, nqubits: int
    ):  # pragma: no cover
        """Calculate probabilities given a density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def set_seed(self, seed):  # pragma: no cover
        """Set the seed of the random number generator."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def sample_shots(self, probabilities, nshots: int):  # pragma: no cover
        """Sample measurement shots according to a probability distribution."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def aggregate_shots(self, shots):  # pragma: no cover
        """Collect shots to a single array."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def samples_to_binary(self, samples, nqubits: int):  # pragma: no cover
        """Convert samples from decimal representation to binary."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def samples_to_decimal(self, samples, nqubits: int):  # pragma: no cover
        """Convert samples from binary representation to decimal."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_frequencies(self, samples):  # pragma: no cover
        """Calculate measurement frequencies from shots."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def update_frequencies(
        self, frequencies, probabilities, nsamples: int
    ):  # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def sample_frequencies(self, probabilities, nshots: int):  # pragma: no cover
        """Sample measurement frequencies according to a probability distribution."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_vector_norm(
        self, state, order: Union[int, float, str] = 2
    ):  # pragma: no cover
        """Calculate norm of an :math:`1`-dimensional array.

        For specifications on possible values of the parameter ``order``
        for the ``tensorflow`` backend, please refer to
        `tensorflow.norm <https://www.tensorflow.org/api_docs/python/tf/norm>`_.
        For all other backends, please refer to
        `numpy.linalg.norm <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>`_.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_matrix_norm(
        self, state, order: Union[int, float, str] = "nuc"
    ):  # pragma: no cover
        """Calculate norm of a :math:`2`-dimensional array.

        Default is the ``nuclear`` norm.
        If ``order="nuc"``, it returns the nuclear norm of ``state``,
        assuming ``state`` is Hermitian (also known as trace norm).
        For specifications on the other  possible values of the
        parameter ``order`` for the ``tensorflow`` backend, please refer to
        `tensorflow.norm <https://www.tensorflow.org/api_docs/python/tf/norm>`_.
        For all other backends, please refer to
        `numpy.linalg.norm <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>`_.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_overlap(self, state1, state2):  # pragma: no cover
        """Calculate overlap of two state vectors."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_overlap_density_matrix(self, state1, state2):  # pragma: no cover
        """Calculate overlap of two density matrices."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_eigenvalues(
        self, matrix, k: int = 6, hermitian: bool = True
    ):  # pragma: no cover
        """Calculate eigenvalues of a matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_eigenvectors(
        self, matrix, k: int = 6, hermitian: bool = True
    ):  # pragma: no cover
        """Calculate eigenvectors of a matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_expectation_state(
        self, hamiltonian, state, normalize: bool
    ):  # pragma: no cover
        """Calculate expectation value of a state vector given the observable matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_expectation_density_matrix(
        self, hamiltonian, state, normalize: bool
    ):  # pragma: no cover
        """Calculate expectation value of a density matrix given the observable matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_matrix_exp(
        self,
        matrix,
        phase: Union[float, int, complex] = 1,
        eigenvectors=None,
        eigenvalues=None,
    ):  # pragma: no cover
        """Calculate the exponential :math:`e^{\\theta \\, A}` of a matrix :math:`A`
        and ``phase`` :math:`\\theta`.

        If the eigenvectors and eigenvalues are given the matrix diagonalization is
        used for exponentiation.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_matrix_log(
        self, matrix, base: Union[float, int] = 2, eigenvectors=None, eigenvalues=None
    ):  # pragma: no cover
        """Calculate the logarithm :math:`\\log_{b}(A)` with a ``base`` :math:`b`
        of a matrix :math:`A`.

        If the eigenvectors and eigenvalues are given the matrix diagonalization is
        used for exponentiation.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_matrix_power(
        self, matrix, power: Union[float, int], precision_singularity: float = 1e-14
    ):  # pragma: no cover
        """Calculate the (fractional) ``power`` :math:`\\alpha` of ``matrix`` :math:`A`,
        i.e. :math:`A^{\\alpha}`.

        .. note::
            For the ``pytorch`` backend, this method relies on a copy of the original tensor.
            This may break the gradient flow. For the GPU backends (i.e. ``cupy`` and
            ``cuquantum``), this method falls back to CPU whenever ``power`` is not
            an integer.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_matrix_sqrt(
        self, matrix, precision_singularity: float = 1e-14
    ):  # pragma: no cover
        """Calculate the square root of ``matrix`` :math:`A`, i.e. :math:`A^{1/2}`.

        .. note::
            For the ``pytorch`` backend, this method relies on a copy of the original tensor.
            This may break the gradient flow. For the GPU backends (i.e. ``cupy`` and
            ``cuquantum``), this method falls back to CPU.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_singular_value_decomposition(self, matrix):  # pragma: no cover
        """Calculate the Singular Value Decomposition of ``matrix``."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_jacobian_matrix(
        self, circuit, parameters, initial_state=None, return_complex: bool = True
    ):  # pragma: no cover
        """Calculate the Jacobian matrix of ``circuit`` with respect to varables ``params``."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def assert_allclose(
        self, value, target, rtol: float = 1e-7, atol: float = 0.0
    ):  # pragma: no cover
        raise_error(NotImplementedError)

    def assert_circuitclose(
        self, circuit, target_circuit, rtol: float = 1e-7, atol: float = 0.0
    ):
        value = self.execute_circuit(circuit)._state
        target = self.execute_circuit(target_circuit)._state
        self.assert_allclose(value, target, rtol=rtol, atol=atol)
