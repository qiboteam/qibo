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
    def calculate_norm(self, state, order=2):  # pragma: no cover
        """Calculate norm of a state vector.

        For specifications on possible values of the parameter ``order``
        for the ``tensorflow`` backend, please refer to
        `tensorflow.norm <https://www.tensorflow.org/api_docs/python/tf/norm>`_.
        For all other backends, please refer to
        `numpy.linalg.norm <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>`_.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_norm_density_matrix(self, state, order="nuc"):  # pragma: no cover
        """Calculate norm of a density matrix. Default is the ``nuclear`` norm.

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

    # --------------------------------------------------------------------------------------------
    # New methods introduced by the refactor:
    # in my view this might be considered as some sort of the core of the backend,
    # i.e. the computation engine that defines how the single small operations
    # are performed and it is going to be completely abstract. All the methods defined
    # above are possibly going to be combination of the core methods below and, therefore,
    # directly implemented in the abstract backend.

    # array creation and manipulation
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def array(self, x: Union[list, tuple], **kwargs):
        """Construct a native array of the backend starting from a `list` or `tuple`.

        Args:
            x (list | tuple): input list or tuple.
            kwargs: keyword argument passed to the `Backend.cast` method.
        """
        return self.cast(x, **kwargs)

    @abc.abstractmethod
    def eye(self, *args, **kwargs):
        """Numpy-like eye: https://numpy.org/devdocs/reference/generated/numpy.eye.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def zeros(self, *args, **kwargs):
        """Numpy-like zeros: https://numpy.org/devdocs/reference/generated/numpy.zeros.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def ones(self, *args, **kwargs):
        """Numpy-like ones: https://numpy.org/devdocs/reference/generated/numpy.ones.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def copy(self, *args, **kwargs):
        """Numpy-like copy: https://numpy.org/devdocs/reference/generated/numpy.copy.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def reshape(self, *args, **kwargs):
        """Numpy-like reshape: https://numpy.org/devdocs/reference/generated/numpy.reshape.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def ravel(self, *args, **kwargs):
        """Numpy-like ravel: https://numpy.org/devdocs/reference/generated/numpy.ravel.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def transpose(self, *args, **kwargs):
        """Numpy-like transpose: https://numpy.org/devdocs/reference/generated/numpy.transpose.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def concatenate(self, *args, **kwargs):
        """Numpy-like concatenate: https://numpy.org/devdocs/reference/generated/numpy.concatenate.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def expand_dims(self, *args, **kwargs):
        """Numpy-like expand_dims: https://numpy.org/devdocs/reference/generated/numpy.expand_dims.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def squeeze(self, *args, **kwargs):
        """Numpy-like squeeze: https://numpy.org/devdocs/reference/generated/numpy.squeeze.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def stack(self, *args, **kwargs):
        """Numpy-like stack: https://numpy.org/devdocs/reference/generated/numpy.stack.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def vstack(self, *args, **kwargs):
        """Numpy-like vstack: https://numpy.org/devdocs/reference/generated/numpy.vstack.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def unique(self, *args, **kwargs):
        """Numpy-like unique: https://numpy.org/devdocs/reference/generated/numpy.unique.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def where(self, *args, **kwargs):
        """Numpy-like where: https://numpy.org/doc/stable/reference/generated/numpy.where.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def flip(self, *args, **kwargs):
        """Numpy-like flip: https://numpy.org/doc/stable/reference/generated/numpy.flip.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def swapaxes(self, *args, **kwargs):
        """Numpy-like swapaxes: https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def diagonal(self, *args, **kwargs):
        """Numpy-like diagonal: https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def nonzero(self, *args, **kwargs):
        """Numpy-like nonzero: https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def sign(self, *args, **kwargs):
        """Numpy-like element-wise sign function: https://numpy.org/doc/stable/reference/generated/numpy.sign.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def delete(a, obj, axis=None):
        """Numpy-like delete function: https://numpy.org/doc/stable/reference/generated/numpy.delete.html"""
        raise NotImplementedError

    # linear algebra
    # ^^^^^^^^^^^^^^

    @abc.abstractmethod
    def einsum(self, *args, **kwargs):
        """Numpy-like einsum: https://numpy.org/devdocs/reference/generated/numpy.einsum.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def matmul(self, *args, **kwargs):
        """Numpy-like matmul: https://numpy.org/devdocs/reference/generated/numpy.matmul.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def multiply(self, *args, **kwargs):
        """Numpy-like multiply: https://numpy.org/doc/stable/reference/generated/numpy.multiply.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def prod(self, *args, **kwargs):
        """Numpy-like prod: https://numpy.org/doc/stable/reference/generated/numpy.prod.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def tensordot(self, *args, **kwargs):
        """Numpy-like tensordot: https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def kron(self, *args, **kwargs):
        """Numpy-like kron: https://numpy.org/doc/stable/reference/generated/numpy.kron.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def outer(self, *args, **kwargs):
        """Numpy-like outer: https://numpy.org/doc/stable/reference/generated/numpy.outer.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def diag(self, *args, **kwargs):
        """Numpy-like diag: https://numpy.org/devdocs/reference/generated/numpy.diag.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def trace(self, *args, **kwargs):
        """Numpy-like trace: https://numpy.org/devdocs/reference/generated/numpy.trace.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def linalg_svd(self, *args, **kwargs):
        """Numpy-like linalg.svd: https://numpy.org/devdocs/reference/generated/numpy.linalg.svd.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def linalg_norm(self, *args, **kwargs):
        """Numpy-like linalg.norm: https://numpy.org/devdocs/reference/generated/numpy.linalg.norm.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def det(self, *args, **kwargs):
        """Numpy-like matrix determinant: https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def qr(self, *args, **kwargs):
        """Numpy linear algebra QR decomposition: https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def inverse(self, *args, **kwargs):
        """Numpy linear algebra inverse: https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def eigvalsh(self, *args, **kwargs):
        """Numpy-like eigvalsh: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def eigvals(self, *args, **kwargs):
        """Eigenvalues of a matrix: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def eigh(self, *args, **kwargs):
        """Numpy-like eigvals: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def eig(self, *args, **kwargs):
        """Numpy-like eig: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def expm(self, *args, **kwargs):
        """Scipy-like expm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html"""
        raise NotImplementedError

    # randomization
    # ^^^^^^^^^^^^^

    @abc.abstractmethod
    def random_choice(self, *args, **kwargs):
        """Numpy-like random.choice: https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def seed(self, *args, **kwargs):
        """Numpy-like random seed: https://numpy.org/devdocs/reference/random/generated/numpy.random.seed.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def permutation(self, *args, **kwargs):
        """Numpy-like random permutation: https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def multinomial(self, *args, **kwargs):
        """Numpy-like multinomial: https://numpy.org/doc/2.0/reference/random/generated/numpy.random.multinomial.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def default_rng(self, *args, **kwargs):
        """Numpy-like random default_rng: https://numpy.org/doc/stable/reference/random/generator.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def rand(self, *args, **kwargs):
        """Numpy-like random rand: https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html"""
        raise NotImplementedError

    # logical operations
    # ^^^^^^^^^^^^^^^^^^

    @abc.abstractmethod
    def less(self, *args, **kwargs):
        """Numpy-like less: https://numpy.org/doc/stable/reference/generated/numpy.less.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def any(self, *args, **kwargs):
        """Numpy-like any: https://numpy.org/doc/stable/reference/generated/numpy.any.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def allclose(self, *args, **kwargs):
        """Numpy-like allclose: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def right_shift(self, *args, **kwargs):
        """Numpy-like element-wise right shift: https://numpy.org/doc/stable/reference/generated/numpy.right_shift.html"""
        raise NotImplementedError

    # mathematical operations
    # ^^^^^^^^^^^^^^^^^^^^^^^

    @abc.abstractmethod
    def sum(self, *args, **kwargs):
        """Numpy-like sum: https://numpy.org/devdocs/reference/generated/numpy.sum.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def conj(self, *args, **kwargs):
        """Numpy-like conj: https://numpy.org/devdocs/reference/generated/numpy.conj.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def exp(self, *args, **kwargs):
        """Numpy-like exp: https://numpy.org/devdocs/reference/generated/numpy.exp.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def log(self, *args, **kwargs):
        """Numpy-like log: https://numpy.org/doc/stable/reference/generated/numpy.log.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def log2(self, *args, **kwargs):
        """Numpy-like log2: https://numpy.org/doc/stable/reference/generated/numpy.log2.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def real(self, *args, **kwargs):
        """Numpy-like real: https://numpy.org/devdocs/reference/generated/numpy.real.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def imag(self, *args, **kwargs):
        """Numpy-like imag: https://numpy.org/doc/stable/reference/generated/numpy.imag.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def abs(self, *args, **kwargs):
        """Numpy-like abs: https://numpy.org/devdocs/reference/generated/numpy.abs.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def pow(self, *args, **kwargs):
        """Numpy-like element-wise power: https://numpy.org/doc/stable/reference/generated/numpy.power.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def square(self, *args, **kwargs):
        """Numpy-like element-wise square: https://numpy.org/doc/stable/reference/generated/numpy.square.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def sqrt(self, *args, **kwargs):
        """Numpy-like sqrt: https://numpy.org/devdocs/reference/generated/numpy.sqrt.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def mean(self, *args, **kwargs):
        """Numpy-like mean: https://numpy.org/doc/stable/reference/generated/numpy.mean.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def std(self, *args, **kwargs):
        """Numpy-like standard deviation: https://numpy.org/doc/stable/reference/generated/numpy.std.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def cos(self, *args, **kwargs):
        """Numpy-like cos: https://numpy.org/devdocs/reference/generated/numpy.cos.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def sin(self, *args, **kwargs):
        """Numpy-like sin: https://numpy.org/devdocs/reference/generated/numpy.sin.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def arccos(self, *args, **kwargs):
        """Numpy-like arccos: https://numpy.org/doc/stable/reference/generated/numpy.arccos.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def arctan2(self, *args, **kwargs):
        """Numpy-like arctan2: https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def angle(self, *args, **kwargs):
        """Numpy-like angle: https://numpy.org/doc/stable/reference/generated/numpy.angle.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def mod(self, *args, **kwargs):
        """Numpy-like element-wise modulus: https://numpy.org/doc/stable/reference/generated/numpy.mod.html"""
        raise NotImplementedError

    # misc
    # ^^^^

    @abc.abstractmethod
    def sort(self, *args, **kwargs):
        """Numpy-like sort: https://numpy.org/doc/stable/reference/generated/numpy.sort.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def count_nonzero(self, *args, **kwargs):
        """Numpy-like count_nonzero: https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def finfo(self, *args, **kwargs):
        """Numpy-like finfo: https://numpy.org/doc/stable/reference/generated/numpy.finfo.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def device(
        self,
    ):
        """Computation device, e.g. CPU, GPU, ..."""
        raise NotImplementedError

    @abc.abstractmethod
    def __version__(
        self,
    ):
        """Version of the backend engine."""
        raise_error(NotImplementedError)

    # Optimization
    # ^^^^^^^^^^^^^

    @abc.abstractmethod
    def jacobian(self, *args, **kwargs):
        """Compute the Jacobian matrix"""
        raise NotImplementedError
