import abc

from qibo.config import raise_error


class Backend(abc.ABC):
    def __init__(self):
        super().__init__()
        self.name = "backend"
        self.platform = None

        self.precision = "double"
        self.dtype = "complex128"
        self.matrices = None

        self.device = "/CPU:0"
        self.nthreads = 1
        self.supports_multigpu = False
        self.oom_error = MemoryError

    def __reduce__(self):
        """Allow pickling backend objects that have references to modules."""
        return self.__class__, tuple()

    def __repr__(self):
        if self.platform is None:
            return self.name
        else:
            return f"{self.name} ({self.platform})"

    @abc.abstractmethod
    def set_precision(self, precision):  # pragma: no cover
        """Set complex number precision.

        Args:
            precision (str): 'single' or 'double'.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def set_device(self, device):  # pragma: no cover
        """Set simulation device.

        Args:
            device (str): Device such as '/CPU:0', '/GPU:0', etc.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def set_threads(self, nthreads):  # pragma: no cover
        """Set number of threads for CPU simulation.

        Args:
            nthreads (int): Number of threads.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def cast(self, x, copy=False):  # pragma: no cover
        """Cast an object as the array type of the current backend.

        Args:
            x: Object to cast to array.
            copy (bool): If ``True`` a copy of the object is created in memory.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def issparse(self, x):  # pragma: no cover
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
    def zero_state(self, nqubits):  # pragma: no cover
        """Generate :math:`|000 \\cdots 0 \\rangle` state vector as an array."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def zero_density_matrix(self, nqubits):  # pragma: no cover
        """Generate :math:`|000\\cdots0\\rangle\\langle000\\cdots0|` density matrix as an array."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def identity_density_matrix(
        self, nqubits, normalize: bool = True
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
    def plus_state(self, nqubits):  # pragma: no cover
        """Generate :math:`|+++\\cdots+\\rangle` state vector as an array."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def plus_density_matrix(self, nqubits):  # pragma: no cover
        """Generate :math:`|+++\\cdots+\\rangle\\langle+++\\cdots+|` density matrix as an array."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def asmatrix(self, gate):  # pragma: no cover
        """Convert a :class:`qibo.gates.Gate` to the corresponding matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def asmatrix_parametrized(self, gate):  # pragma: no cover
        """Equivalent to :meth:`qibo.backends.abstract.Backend.asmatrix` for parametrized gates."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def asmatrix_fused(self, gate):  # pragma: no cover
        """Fuse matrices of multiple gates."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def control_matrix(self, gate):  # pragma: no cover
        """ "Calculate full matrix representation of a controlled gate."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_gate(self, gate, state, nqubits):  # pragma: no cover
        """Apply a gate to state vector."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_gate_density_matrix(self, gate, state, nqubits):  # pragma: no cover
        """Apply a gate to density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_gate_half_density_matrix(self, gate, state, nqubits):  # pragma: no cover
        """Apply a gate to one side of the density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_channel(self, channel, state, nqubits):  # pragma: no cover
        """Apply a channel to state vector."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_channel_density_matrix(self, channel, state, nqubits):  # pragma: no cover
        """Apply a channel to density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def collapse_state(
        self, state, qubits, shot, nqubits, normalize=True
    ):  # pragma: no cover
        """Collapse state vector according to measurement shot."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def collapse_density_matrix(
        self, state, qubits, shot, nqubits, normalize=True
    ):  # pragma: no cover
        """Collapse density matrix according to measurement shot."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def reset_error_density_matrix(self, gate, state, nqubits):  # pragma: no cover
        """Apply reset error to density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def thermal_error_density_matrix(self, gate, state, nqubits):  # pragma: no cover
        """Apply thermal relaxation error to density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def execute_circuit(
        self, circuit, initial_state=None, nshots=None
    ):  # pragma: no cover
        """Execute a :class:`qibo.models.circuit.Circuit`."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def execute_circuit_repeated(
        self, circuit, initial_state=None, nshots=None
    ):  # pragma: no cover
        """Execute a :class:`qibo.models.circuit.Circuit` multiple times.

        Useful for noise simulation using state vectors or for simulating gates
        controlled by measurement outcomes.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def execute_distributed_circuit(
        self, circuit, initial_state=None, nshots=None
    ):  # pragma: no cover
        """Execute a :class:`qibo.models.circuit.Circuit` using multiple GPUs."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def circuit_result_representation(self, result):  # pragma: no cover
        """Represent a quantum state based on circuit execution results.

        Args:
            result (:class:`qibo.states.CircuitResult`): Result object that contains
                the data required to represent the state.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def circuit_result_tensor(self, result):  # pragma: no cover
        """State vector or density matrix representing a quantum state as an array.

        Args:
            result (:class:`qibo.states.CircuitResult`): Result object that contains
                the data required to represent the state.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def circuit_result_probabilities(self, result, qubits=None):  # pragma: no cover
        """Calculates measurement probabilities by tracing out qubits.

        Args:
            result (:class:`qibo.states.CircuitResult`): Result object that contains
                the data required to represent the state.
            qubits (list, set): Set of qubits that are measured.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_symbolic(
        self, state, nqubits, decimals=5, cutoff=1e-10, max_terms=20
    ):  # pragma: no cover
        """Dirac representation of a state vector."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_symbolic_density_matrix(
        self, state, nqubits, decimals=5, cutoff=1e-10, max_terms=20
    ):  # pragma: no cover
        """Dirac representation of a density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_probabilities(self, state, qubits, nqubits):  # pragma: no cover
        """Calculate probabilities given a state vector."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_probabilities_density_matrix(
        self, state, qubits, nqubits
    ):  # pragma: no cover
        """Calculate probabilities given a density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def set_seed(self, seed):  # pragma: no cover
        """Set the seed of the random number generator."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def sample_shots(self, probabilities, nshots):  # pragma: no cover
        """Sample measurement shots according to a probability distribution."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def aggregate_shots(self, shots):  # pragma: no cover
        """Collect shots to a single array."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def samples_to_binary(self, samples, nqubits):  # pragma: no cover
        """Convert samples from decimal representation to binary."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def samples_to_decimal(self, samples, nqubits):  # pragma: no cover
        """Convert samples from binary representation to decimal."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_frequencies(self, samples):  # pragma: no cover
        """Calculate measurement frequencies from shots."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def update_frequencies(
        self, frequencies, probabilities, nsamples
    ):  # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def sample_frequencies(self, probabilities, nshots):  # pragma: no cover
        """Sample measurement frequencies according to a probability distribution."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def partial_trace(self, state, qubits, nqubits):  # pragma: no cover
        """Trace out specific qubits of a state vector."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def partial_trace_density_matrix(self, state, qubits, nqubits):  # pragma: no cover
        """Trace out specific qubits of a density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def entanglement_entropy(self, rho):  # pragma: no cover
        """Calculate entangelement entropy of a reduced density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_norm(self, state):  # pragma: no cover
        """Calculate norm of a state vector."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_norm_density_matrix(self, state):  # pragma: no cover
        """Calculate norm (trace) of a density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_overlap(self, state1, state2):  # pragma: no cover
        """Calculate overlap of two state vectors."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_overlap_density_matrix(self, state1, state2):  # pragma: no cover
        """Calculate norm of two density matrices."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_eigenvalues(self, matrix, k=6):  # pragma: no cover
        """Calculate eigenvalues of a matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_eigenvectors(self, matrix, k=6):  # pragma: no cover
        """Calculate eigenvectors of a matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_matrix_exp(
        self, matrix, a, eigenvectors=None, eigenvalues=None
    ):  # pragma: no cover
        """Calculate matrix exponential of a matrix.

        If the eigenvectors and eigenvalues are given the matrix diagonalization is
        used for exponentiation.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_expectation_state(
        self, hamiltonian, state, normalize
    ):  # pragma: no cover
        """Calculate expectation value of a state vector given the observable matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_expectation_density_matrix(
        self, hamiltonian, state, normalize
    ):  # pragma: no cover
        """Calculate expectation value of a density matrix given the observable matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_hamiltonian_matrix_product(
        self, matrix1, matrix2
    ):  # pragma: no cover
        """Multiply two matrices."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_hamiltonian_state_product(self, matrix, state):  # pragma: no cover
        """Multiply a matrix to a state vector or density matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def assert_allclose(self, value, target, rtol=1e-7, atol=0.0):  # pragma: no cover
        raise_error(NotImplementedError)

    def assert_circuitclose(self, circuit, target_circuit, rtol=1e-7, atol=0.0):
        value = self.execute_circuit(circuit)
        target = self.execute_circuit(target_circuit)
        self.assert_allclose(value, target, rtol=rtol, atol=atol)

    @abc.abstractmethod
    def test_regressions(self, name):  # pragma: no cover
        """Correct outcomes for tests that involve random numbers.

        The outcomes of such tests depend on the backend.
        """
        raise_error(NotImplementedError)
