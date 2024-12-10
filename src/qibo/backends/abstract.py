import abc
from typing import Optional, Union

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
    def matrix(self, gate):  # pragma: no cover
        """Convert a :class:`qibo.gates.Gate` to the corresponding matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def matrix_parametrized(self, gate):  # pragma: no cover
        """Equivalent to :meth:`qibo.backends.abstract.Backend.matrix` for parametrized gates."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def matrix_fused(self, gate):  # pragma: no cover
        """Fuse matrices of multiple gates."""
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
    def execute_circuits(
        self, circuits, initial_states=None, nshots=None
    ):  # pragma: no cover
        """Execute multiple :class:`qibo.models.circuit.Circuit` in parallel."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def execute_circuit_repeated(
        self, circuit, nshots, initial_state=None
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
    def calculate_vector_norm(self, state, order=2):  # pragma: no cover
        """Calculate norm of an :math:`1`-dimensional array.

        For specifications on possible values of the parameter ``order``
        for the ``tensorflow`` backend, please refer to
        `tensorflow.norm <https://www.tensorflow.org/api_docs/python/tf/norm>`_.
        For all other backends, please refer to
        `numpy.linalg.norm <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>`_.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_matrix_norm(self, state, order="nuc"):  # pragma: no cover
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
    def calculate_matrix_exp(
        self, a, matrix, eigenvectors=None, eigenvalues=None
    ):  # pragma: no cover
        """Calculate matrix exponential of a matrix.
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
        value = self.execute_circuit(circuit)._state
        target = self.execute_circuit(target_circuit)._state
        self.assert_allclose(value, target, rtol=rtol, atol=atol)

    @abc.abstractmethod
    def _test_regressions(self, name):  # pragma: no cover
        """Correct outcomes for tests that involve random numbers.

        The outcomes of such tests depend on the backend.
        """
        raise_error(NotImplementedError)
