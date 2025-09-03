import math
from typing import List, Optional, Tuple, Union

from qibo import __version__
from qibo.config import raise_error


class Backend:
    def __init__(self):
        super().__init__()

        self.device = "/CPU:0"
        self.dtype = "complex128"
        self.engine = None
        self.matrices = None
        self.name = "backend"
        self.nthreads = 1
        self.numeric_types = (
            int,
            float,
            complex,
            "complex128",
            "complex64",
            "float64",
            "float32",
        )
        self.oom_error = MemoryError
        self.platform = None
        self.supports_multigpu = False
        self.tensor_types = None
        self.versions = {"qibo": __version__}

    def __reduce__(self):
        """Allow pickling backend objects that have references to modules."""
        return self.__class__, tuple()

    def __repr__(self):
        if self.platform is None:
            return self.name

        return f"{self.name} ({self.platform})"

    @property
    def qubits(self) -> Optional[list[Union[int, str]]]:  # pragma: no cover
        """Return the qubit names of the backend.

        Returns:
            List[int] or List[str] or None: For hardware backends, return list of qubit names.
            For simulation backends, return ``None``.
        """
        return None

    @property
    def connectivity(
        self,
    ) -> Optional[List[Tuple[Union[int, str], Union[int, str]]]]:  # pragma: no cover
        """Return available qubit pairs of the backend.

        Returns:
            List[Tuple[int]] or List[Tuple[str]] or None: For hardware backends, return
            available qubit pairs. For simulation backends, return ``None``.
        """
        return None

    @property
    def natives(self) -> Optional[list[str]]:  # pragma: no cover
        """Return the native gates of the backend.

        Returns:
            List[str] or None: For hardware backends, return the native gates of the backend.
            For the simulation backends, return ``None``.
        """
        return None

    def set_dtype(self, dtype: str) -> None:  # pragma: no cover
        """Set data type of arrays created using the backend. Works in-place.

        .. note::
            The data types ``float32`` and ``float64`` are intended to be used when the circuits
            to be simulated only contain gates with real-valued matrix representations.
            Using one of the aforementioned data types with circuits that contain complex-valued
            matrices will raise a casting error.

        .. note::
            List of gates that have a real-valued matrix representation:
            :class:`qibo.gates.I`, :class:`qibo.gates.X`, :class:`qibo.gates.Z`,
            :class:`qibo.gates.H`, :class:`qibo.gates.Align`, :class:`qibo.gates.RY`,
            :class:`qibo.gates.CNOT`, :class:`qibo.gates.CZ`, :class:`qibo.gates.CRY`,
            :class:`qibo.gates.SWAP`, :class:`qibo.gates.FSWAP`, :class:`qibo.gates.GIVENS`,
            :class:`qibo.gates.RBS`, :class:`qibo.gates.TOFFOLI`, and :class:`qibo.gates.CCZ`.

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
        if dtype not in self.numeric_types:
            raise_error(
                ValueError,
                f"Unknown ``dtype`` ``{dtype}``."
                + f"``dtype`` must be one of the following options: {self.numeric_types}",
            )

        if dtype != self.dtype:
            self.dtype = dtype

            if self.matrices:
                self.matrices = self.matrices.__class__(self.dtype)

    def set_device(self, device: str) -> None:  # pragma: no cover
        """Set simulation device. Works in-place.

        Args:
            device (str): Device index, *e.g.* ``/CPU:0`` for CPU, or ``/GPU:1`` for
                the second GPU in a multi-GPU environment.
        """
        raise_error(NotImplementedError)

    def set_threads(self, nthreads: int) -> None:  # pragma: no cover
        """Set number of threads for CPU backend simulations that accept it. Works in-place.

        Args:
            nthreads (int): Number of threads.
        """
        raise_error(NotImplementedError)

    def cast(self, x, dtype=None, copy: bool = False) -> "ndarray":  # pragma: no cover
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

    def is_sparse(self, array) -> bool:  # pragma: no cover
        """Determine if a given array is a sparse tensor."""
        raise_error(NotImplementedError)

    def to_numpy(self, array) -> "ndarray":  # pragma: no cover
        """Cast a given array to numpy."""
        raise_error(NotImplementedError)

    def compile(self, func):  # pragma: no cover
        """Compile the given method.

        Available only for the ``tensorflow`` backend.
        """
        raise_error(NotImplementedError)

    def identity(self, dims: int, dtype=None):
        if dtype is None:
            dtype = self.dtype
        return self.engine.eye(dims, dtype=dtype)

    def ones(self, shape, dtype=None):
        if dtype is None:  # pragma: no cover
            dtype = self.dtype
        return self.engine.ones(shape, dtype=dtype)

    def outer(self, array_1, array_2):
        return self.engine.outer(array_1, array_2)

    def zeros(self, shape, dtype=None):
        if dtype is None:  # pragma: no cover
            dtype = self.dtype
        return self.engine.zeros(shape, dtype=dtype)

    def zero_state(
        self, nqubits: int, density_matrix: bool = False, dtype=None
    ) -> "ndarray":  # pragma: no cover
        """Generate the :math:`n`-fold tensor product of the single-qubit :math:`\\ket{0}` state.

        Args:
            nqubits (int): Number of qubits :math:`n`.
            density_matrix (bool, optional): If ``True``, returns the density matrix
                :math:`\\ketbra{0}^{\\otimes \\, n}`. If ``False``, returns the statevector
                :math:`\\ket{0}^{\\otimes \\, n}`. Defaults to ``False``.

        Returns:
            ndarray: Array representation of the :math:`n`-qubit zero state.
        """
        if dtype is None:
            dtype = self.dtype

        dims = 2**nqubits
        shape = 2 * (dims,) if density_matrix else dims
        indexes = [0, 0] if density_matrix else 0

        state = self.zeros(shape, dtype=dtype)
        state[indexes] = 1

        return state

    def plus_state(
        self, nqubits: int, density_matrix: bool = False, dtype=None
    ):  # pragma: no cover
        """Generate :math:`|+++\\cdots+\\rangle` state vector as an array."""
        if dtype is None:
            dtype = self.dtype

        dims = 2**nqubits
        normalization = dims if density_matrix else math.sqrt(dims)
        shape = 2 * (dims,) if density_matrix else dims

        state = self.ones(shape, dtype=dtype)
        state /= normalization

        return state

    def maximally_mixed_state(
        self, nqubits: int, dtype=None
    ) -> "ndarray":  # pragma: no cover
        """Generate the :math:`n`-qubit density matrix for the maximally mixed state.

        .. math::
            \\rho = \\frac{I}{2^{n}} \\, ,

        where :math:`I` is the :math:`2^{n} \\times 2^{n}` identity operator.

        Args:
            nqubits (int): Number of qubits :math:`n`.

        """
        if dtype is None:
            dtype = self.dtype

        dims = 2**nqubits
        state = self.identity(dims, dtype=dtype)
        state /= dims

        return state

    def matrix(self, gate: "qibo.gates.abstract.Gate"):  # pragma: no cover
        """Convert a :class:`qibo.gates.Gate` to the corresponding matrix."""
        raise_error(NotImplementedError)

    def matrix_parametrized(self, gate: "qibo.gates.abstract.Gate"):  # pragma: no cover
        """Equivalent to :meth:`qibo.backends.abstract.Backend.matrix` for parametrized gates."""
        raise_error(NotImplementedError)

    def matrix_fused(self, gate):  # pragma: no cover
        """Fuse matrices of multiple gates."""
        raise_error(NotImplementedError)

    def apply_gate(self, gate, state, nqubits: int):  # pragma: no cover
        """Apply a gate to state vector."""
        raise_error(NotImplementedError)

    def apply_gate_density_matrix(self, gate, state, nqubits: int):  # pragma: no cover
        """Apply a gate to density matrix."""
        raise_error(NotImplementedError)

    def apply_gate_half_density_matrix(
        self, gate, state, nqubits: int
    ):  # pragma: no cover
        """Apply a gate to one side of the density matrix."""
        raise_error(NotImplementedError)

    def apply_channel(self, channel, state, nqubits: int):  # pragma: no cover
        """Apply a channel to state vector."""
        raise_error(NotImplementedError)

    def apply_channel_density_matrix(
        self, channel, state, nqubits: int
    ):  # pragma: no cover
        """Apply a channel to density matrix."""
        raise_error(NotImplementedError)

    def collapse_state(
        self, state, qubits, shot, nqubits: int, normalize: bool = True
    ):  # pragma: no cover
        """Collapse state vector according to measurement shot."""
        raise_error(NotImplementedError)

    def collapse_density_matrix(
        self, state, qubits, shot, nqubits: int, normalize: bool = True
    ):  # pragma: no cover
        """Collapse density matrix according to measurement shot."""
        raise_error(NotImplementedError)

    def reset_error_density_matrix(self, gate, state, nqubits: int):  # pragma: no cover
        """Apply reset error to density matrix."""
        raise_error(NotImplementedError)

    def thermal_error_density_matrix(
        self, gate, state, nqubits: int
    ):  # pragma: no cover
        """Apply thermal relaxation error to density matrix."""
        raise_error(NotImplementedError)

    def execute_circuit(
        self, circuit, initial_state=None, nshots: int = None
    ):  # pragma: no cover
        """Execute a :class:`qibo.models.circuit.Circuit`."""
        raise_error(NotImplementedError)

    def execute_circuits(
        self, circuits, initial_states=None, nshots: int = None
    ):  # pragma: no cover
        """Execute multiple :class:`qibo.models.circuit.Circuit` in parallel."""
        raise_error(NotImplementedError)

    def execute_circuit_repeated(
        self, circuit: "qibo.models.circuit.Circuit", nshots: int, initial_state=None
    ):  # pragma: no cover
        """Execute a :class:`qibo.models.circuit.Circuit` multiple times.

        Useful for noise simulation using state vectors or for simulating gates
        controlled by measurement outcomes.
        """
        raise_error(NotImplementedError)

    def execute_distributed_circuit(
        self, circuit, initial_state=None, nshots: int = None
    ):  # pragma: no cover
        """Execute a :class:`qibo.models.circuit.Circuit` using multiple GPUs."""
        raise_error(NotImplementedError)

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

    def calculate_probabilities(self, state, qubits, nqubits: int):  # pragma: no cover
        """Calculate probabilities given a state vector."""
        raise_error(NotImplementedError)

    def calculate_probabilities_density_matrix(
        self, state, qubits, nqubits: int
    ):  # pragma: no cover
        """Calculate probabilities given a density matrix."""
        raise_error(NotImplementedError)

    def set_seed(self, seed):  # pragma: no cover
        """Set the seed of the random number generator."""
        raise_error(NotImplementedError)

    def sample_shots(self, probabilities, nshots: int):  # pragma: no cover
        """Sample measurement shots according to a probability distribution."""
        raise_error(NotImplementedError)

    def aggregate_shots(self, shots):  # pragma: no cover
        """Collect shots to a single array."""
        raise_error(NotImplementedError)

    def samples_to_binary(self, samples, nqubits: int):  # pragma: no cover
        """Convert samples from decimal representation to binary."""
        raise_error(NotImplementedError)

    def samples_to_decimal(self, samples, nqubits: int):  # pragma: no cover
        """Convert samples from binary representation to decimal."""
        raise_error(NotImplementedError)

    def calculate_frequencies(self, samples):  # pragma: no cover
        """Calculate measurement frequencies from shots."""
        raise_error(NotImplementedError)

    def update_frequencies(
        self, frequencies, probabilities, nsamples: int
    ):  # pragma: no cover
        raise_error(NotImplementedError)

    def sample_frequencies(self, probabilities, nshots: int):  # pragma: no cover
        """Sample measurement frequencies according to a probability distribution."""
        raise_error(NotImplementedError)

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

    def calculate_overlap(self, state1, state2):  # pragma: no cover
        """Calculate overlap of two state vectors."""
        raise_error(NotImplementedError)

    def calculate_overlap_density_matrix(self, state1, state2):  # pragma: no cover
        """Calculate overlap of two density matrices."""
        raise_error(NotImplementedError)

    def calculate_eigenvalues(
        self, matrix, k: int = 6, hermitian: bool = True
    ):  # pragma: no cover
        """Calculate eigenvalues of a matrix."""
        raise_error(NotImplementedError)

    def calculate_eigenvectors(
        self, matrix, k: int = 6, hermitian: bool = True
    ):  # pragma: no cover
        """Calculate eigenvectors of a matrix."""
        raise_error(NotImplementedError)

    def calculate_expectation_state(
        self, hamiltonian, state, normalize: bool
    ):  # pragma: no cover
        """Calculate expectation value of a state vector given the observable matrix."""
        raise_error(NotImplementedError)

    def calculate_expectation_density_matrix(
        self, hamiltonian, state, normalize: bool
    ):  # pragma: no cover
        """Calculate expectation value of a density matrix given the observable matrix."""
        raise_error(NotImplementedError)

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

    def calculate_matrix_log(
        self, matrix, base: Union[float, int] = 2, eigenvectors=None, eigenvalues=None
    ):  # pragma: no cover
        """Calculate the logarithm :math:`\\log_{b}(A)` with a ``base`` :math:`b`
        of a matrix :math:`A`.

        If the eigenvectors and eigenvalues are given the matrix diagonalization is
        used for exponentiation.
        """
        raise_error(NotImplementedError)

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

    def calculate_singular_value_decomposition(self, matrix):  # pragma: no cover
        """Calculate the Singular Value Decomposition of ``matrix``."""
        raise_error(NotImplementedError)

    def calculate_jacobian_matrix(
        self, circuit, parameters, initial_state=None, return_complex: bool = True
    ):  # pragma: no cover
        """Calculate the Jacobian matrix of ``circuit`` with respect to varables ``params``."""
        raise_error(NotImplementedError)

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
