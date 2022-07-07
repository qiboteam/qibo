import abc
from qibo.config import raise_error


class Backend(abc.ABC):

    def __init__(self):
        super().__init__()
        self.name = "backend"
        self.platform= None

        self.precision = "double"
        self.dtype = "complex128"
        self.matrices = None

        self.device = "/CPU:0"
        self.nthreads = 1
        self.supports_multigpu = False
        self.oom_error = MemoryError

    def __repr__(self):
        if self.platform is None:
            return self.name
        else:
            return f"{self.name} ({self.platform})"

    @abc.abstractmethod
    def set_precision(self, precision): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def set_device(self, device): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def set_threads(self, nthreads): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def cast(self, x, copy=False): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def issparse(self, x): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def to_numpy(self, x): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def compile(self, func): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def zero_state(self, nqubits): # pragma: no cover
        """Generate |000...0> state as an array."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def zero_density_matrix(self, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def asmatrix(self, gate): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def asmatrix_parametrized(self, gate): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def asmatrix_fused(self, gate): # pragma: no cover
        """Fuse matrices of multiple gates."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def control_matrix(self, gate): # pragma: no cover
        """"Calculate full matrix representation of a controlled gate."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_gate(self, gate, state, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_gate_density_matrix(self, gate, state, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_gate_half_density_matrix(self, gate, state, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_channel(self, gate): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_channel_density_matrix(self, gate): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def collapse_state(self, state, qubits, shot, nqubits, normalize=True): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def collapse_density_matrix(self, state, qubits, shot, nqubits, normalize=True): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def execute_circuit(self, circuit, nshots=None): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def get_state_repr(self, result): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def get_state_tensor(self, result): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_symbolic(self, state, nqubits, decimals=5, cutoff=1e-10, max_terms=20): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_symbolic_density_matrix(self, state, nqubits, decimals=5, cutoff=1e-10, max_terms=20): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_probabilities(self, result): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_probabilities_density_matrix(self, state, qubits, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def set_seed(self, seed): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def sample_shots(self, probabilities, nshots): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def aggregate_shots(self, shots): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def samples_to_binary(self, samples, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def samples_to_decimal(self, samples, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def sample_frequencies(self, probabilities, nshots): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_frequencies(self, samples): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def update_frequencies(self, frequencies, probabilities, nsamples): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def partial_trace(self, state, qubits, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def partial_trace_density_matrix(self, state, qubits, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def entanglement_entropy(self, rho): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_norm(self, state): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_norm_density_matrix(self, state): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_overlap(self, state1, state2): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_overlap_density_matrix(self, state1, state2): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_eigenvalues(self, matrix, k=6): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_eigenvectors(self, matrix, k=6): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_matrix_exp(self, matrix, a, eigenvectors=None, eigenvalues=None): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_expectation_state(self, matrix, state, normalize): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_expectation_density_matrix(self, matrix, state, normalize): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_hamiltonian_state_product(self, matrix, state): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def assert_allclose(self, value, target, rtol=1e-7, atol=0.0): # pragma: no cover
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
