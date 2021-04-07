from abc import ABC, abstractmethod
from qibo.config import raise_error, log


class AbstractBackend(ABC):

    def __init__(self):
        self.backend = None
        self.name = "base"

        self.precision = 'double'
        self._dtypes = {'DTYPEINT': 'int64', 'DTYPE': 'float64',
                        'DTYPECPX': 'complex128'}

        self.cpu_devices = []
        self.gpu_devices = []
        self.default_device = None

        self.matrices = None
        self.numeric_types = None
        self.tensor_types = None
        self.native_types = None
        self.Tensor = None
        self.random = None
        self.newaxis = None
        self.oom_error = None
        self.optimization = None

    def dtypes(self, name):
        if name in self._dtypes:
            dtype = self._dtypes.get(name)
        else:
            dtype = name
        return getattr(self.backend, dtype)

    def set_precision(self, dtype):
        if dtype == 'single':
            self._dtypes['DTYPE'] = 'float32'
            self._dtypes['DTYPECPX'] = 'complex64'
        elif dtype == 'double':
            self._dtypes['DTYPE'] = 'float64'
            self._dtypes['DTYPECPX'] = 'complex128'
        else:
            raise_error(ValueError, f'dtype {dtype} not supported.')
        self.precision = dtype
        if self.matrices is not None:
            self.matrices.dtype = self.dtypes('DTYPECPX')

    def set_device(self, name):
        parts = name[1:].split(":")
        if name[0] != "/" or len(parts) < 2 or len(parts) > 3:
            raise_error(ValueError, "Device name should follow the pattern: "
                                    "/{device type}:{device number}.")
        device_type, device_number = parts[-2], int(parts[-1])
        if device_type == "CPU":
            ndevices = len(self.cpu_devices)
        elif device_type == "GPU":
            ndevices = len(self.gpu_devices)
        else:
            raise_error(ValueError, f"Unknown device type {device_type}.")
        if device_number >= ndevices:
            raise_error(ValueError, f"Device {name} does not exist.")
        self.default_device = name

    def get_cpu(self): # pragma: no cover
        """Returns default CPU device to use for OOM fallback."""
        # case not covered by GitHub workflows because it requires OOM""
        if not self.cpu_devices:
            raise_error(RuntimeError, "Cannot find CPU device to fall back to.")
        return self.cpu_devices[0]

    def cpu_fallback(self, func, *args):
        """Executes a function on CPU if the default devices raises OOM."""
        try:
            return func(*args)
        except self.oom_error: # pragma: no cover
            # case not covered by GitHub workflows because it requires OOM
            # Force using CPU to perform sampling
            log.warn("Falling back to CPU because the GPU is out-of-memory.")
            with self.device(self.get_cpu()):
                return func(*args)

    @abstractmethod
    def cast(self, x, dtype='DTYPECPX'): # pragma: no cover
        """Casts tensor to the given dtype."""
        raise_error(NotImplementedError)

    @abstractmethod
    def diag(self, x, dtype='DTYPECPX'): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def reshape(self, x, shape): # pragma: no cover
        """Reshapes tensor in the given shape."""
        raise_error(NotImplementedError)

    @abstractmethod
    def stack(self, x, axis=None): # pragma: no cover
        """Stacks a list of tensors to a single tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def concatenate(self, x, axis=None): # pragma: no cover
        """Concatenates a list of tensor along a given axis."""
        raise_error(NotImplementedError)

    @abstractmethod
    def expand_dims(self, x, axis): # pragma: no cover
        """Creates a new axis of dimension one."""
        raise_error(NotImplementedError)

    @abstractmethod
    def copy(self, x): # pragma: no cover
        """Creates a copy of the tensor in memory."""
        raise_error(NotImplementedError)

    @abstractmethod
    def range(self, start, finish, step, dtype=None): # pragma: no cover
        """Creates a tensor of integers from start to finish."""
        raise_error(NotImplementedError)

    @abstractmethod
    def eye(self, dim, dtype='DTYPECPX'): # pragma: no cover
        """Creates the identity matrix as a tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def zeros(self, shape, dtype='DTYPECPX'): # pragma: no cover
        """Creates tensor of zeros with the given shape and dtype."""
        raise_error(NotImplementedError)

    @abstractmethod
    def ones(self, shape, dtype='DTYPECPX'): # pragma: no cover
        """Creates tensor of ones with the given shape and dtype."""
        raise_error(NotImplementedError)

    @abstractmethod
    def zeros_like(self, x): # pragma: no cover
        """Creates tensor of zeros with shape and dtype of the given tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def ones_like(self, x): # pragma: no cover
        """Creates tensor of ones with shape and dtype of the given tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def real(self, x): # pragma: no cover
        """Real part of a given complex tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def imag(self, x): # pragma: no cover
        """Imaginary part of a given complex tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def conj(self, x): # pragma: no cover
        """Elementwise complex conjugate of a tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def mod(self, x): # pragma: no cover
        """Elementwise mod operation."""
        raise_error(NotImplementedError)

    @abstractmethod
    def right_shift(self, x, y): # pragma: no cover
        """Elementwise bitwise right shift."""
        raise_error(NotImplementedError)

    @abstractmethod
    def exp(self, x): # pragma: no cover
        """Elementwise exponential."""
        raise_error(NotImplementedError)

    @abstractmethod
    def sin(self, x): # pragma: no cover
        """Elementwise sin."""
        raise_error(NotImplementedError)

    @abstractmethod
    def cos(self, x): # pragma: no cover
        """Elementwise cos."""
        raise_error(NotImplementedError)

    @abstractmethod
    def pow(self, base, exponent): # pragma: no cover
        """Elementwise power."""
        raise_error(NotImplementedError)

    @abstractmethod
    def square(self, x): # pragma: no cover
        """Elementwise square."""
        raise_error(NotImplementedError)

    @abstractmethod
    def sqrt(self, x): # pragma: no cover
        """Elementwise square root."""
        raise_error(NotImplementedError)

    @abstractmethod
    def log(self, x): # pragma: no cover
        """Elementwise natural logarithm."""
        raise_error(NotImplementedError)

    @abstractmethod
    def abs(self, x): # pragma: no cover
        """Elementwise absolute value."""
        raise_error(NotImplementedError)

    @abstractmethod
    def expm(self, x): # pragma: no cover
        """Matrix exponential."""
        raise_error(NotImplementedError)

    @abstractmethod
    def trace(self, x): # pragma: no cover
        """Matrix trace."""
        raise_error(NotImplementedError)

    @abstractmethod
    def sum(self, x, axis=None): # pragma: no cover
        """Sum of tensor elements."""
        raise_error(NotImplementedError)

    @abstractmethod
    def matmul(self, x, y): # pragma: no cover
        """Matrix multiplication of two tensors."""
        raise_error(NotImplementedError)

    @abstractmethod
    def outer(self, x, y): # pragma: no cover
        """Outer (Kronecker) product of two tensors."""
        raise_error(NotImplementedError)

    @abstractmethod
    def kron(self, x, y): # pragma: no cover
        """Outer (Kronecker) product of two tensors."""
        raise_error(NotImplementedError)

    @abstractmethod
    def einsum(self, *args): # pragma: no cover
        """Generic tensor operation based on Einstein's summation convention."""
        raise_error(NotImplementedError)

    @abstractmethod
    def tensordot(self, x, y, axes=None): # pragma: no cover
        """Generalized tensor product of two tensors."""
        raise_error(NotImplementedError)

    @abstractmethod
    def transpose(self, x, axes=None): # pragma: no cover
        """Tensor transpose."""
        raise_error(NotImplementedError)

    @abstractmethod
    def inv(self, x): # pragma: no cover
        """Matrix inversion."""
        raise_error(NotImplementedError)

    @abstractmethod
    def eigh(self, x): # pragma: no cover
        """Hermitian matrix eigenvalues and eigenvectors."""
        raise_error(NotImplementedError)

    @abstractmethod
    def eigvalsh(self, x): # pragma: no cover
        """Hermitian matrix eigenvalues."""
        raise_error(NotImplementedError)

    @abstractmethod
    def unique(self, x, return_counts=False): # pragma: no cover
        """Identifies unique elements in a tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def less(self, x, y): # pragma: no cover
        """Compares the values of two tensors element-wise. Returns a bool tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def array_equal(self, x, y): # pragma: no cover
        """Checks if two arrays are equal element-wise. Returns a single bool.

        Used in :meth:`qibo.tensorflow.hamiltonians.TrotterHamiltonian.construct_terms`.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def squeeze(self, x, axis=None): # pragma: no cover
        """Removes axis of unit length."""
        raise_error(NotImplementedError)

    @abstractmethod
    def gather(self, x, indices=None, condition=None, axis=0): # pragma: no cover
        """Indexing of one-dimensional tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def gather_nd(self, x, indices): # pragma: no cover
        """Indexing of multi-dimensional tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def initial_state(self, nqubits, is_matrix=False): # pragma: no cover
        """Creates the default initial state |00...0> as a tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def transpose_state(self, pieces, state, nqubits, order): # pragma: no cover
        """Transposes state pieces to the full state.

        Used by :class:`qibo.core.states.DistributedState`.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def random_uniform(self, shape, dtype='DTYPE'): # pragma: no cover
        """Samples array of given shape from a uniform distribution in [0, 1]."""
        raise_error(NotImplementedError)

    @abstractmethod
    def sample_shots(self, probs, nshots): # pragma: no cover
        """Samples measurement shots from a given probability distribution.

        Args:
            probs (Tensor): Tensor with the probability distribution on the
                measured bitsrings.
            nshots (int): Number of measurement shots to sample.

        Returns:
            Measurements in decimal as a tensor of shape ``(nshots,)``.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def sample_frequencies(self, probs, nshots): # pragma: no cover
        """Samples measurement frequencies from a given probability distribution.

        Args:
            probs (Tensor): Tensor with the probability distribution on the
                measured bitsrings.
            nshots (int): Number of measurement shots to sample.

        Returns:
            Frequencies of measurements as a ``collections.Counter``.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def compile(self, func): # pragma: no cover
        """Compiles the graph of a given function.

        Relevant for Tensorflow, not numpy.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def device(self, device_name): # pragma: no cover
        """Used to execute code in specific device if supported by backend."""
        raise_error(NotImplementedError)

    def executing_eagerly(self):
        """Checks if we are in eager or compiled mode.

        Relevant for the Tensorflow backends only.
        """
        return True

    @abstractmethod
    def set_seed(self, seed): # pragma: no cover
        """Sets the seed for random number generation.

        Args:
            seed (int): Integer to use as seed.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def create_gate_cache(self, gate): # pragma: no cover
        """Calculates data required for applying gates to states.

        These can be einsum index strings or tensors of qubit ids and it
        depends on the underlying backend.

        Args:
            gate (:class:`qibo.abstractions.abstract_gates.BackendGate`): Gate
                object to calculate its cache.

        Returns:
            Custom cache object that holds all the required data as
            attributes.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def state_vector_call(self, gate, state): # pragma: no cover
        """Applies gate to state vector.

        Args:
            gate (:class:`qibo.abstractions.abstract_gates.BackendGate`): Gate
                object to apply to state.
            state (Tensor): State vector as a ``Tensor`` supported by this
                backend.

        Returns:
            State vector after applying the gate as a ``Tensor``.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def state_vector_matrix_call(self, gate, state): # pragma: no cover
        """Applies gate to state vector using the gate's unitary matrix representation.

        This method is useful for the ``custom`` backend for which some gates
        do not require the unitary matrix.

        Args:
            gate (:class:`qibo.abstractions.abstract_gates.BackendGate`): Gate
                object to apply to state.
            state (Tensor): State vector as a ``Tensor`` supported by this
                backend.

        Returns:
            State vector after applying the gate as a ``Tensor``.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def density_matrix_call(self, gate, state): # pragma: no cover
        """Applies gate to density matrix.

        Args:
            gate (:class:`qibo.abstractions.abstract_gates.BackendGate`): Gate
                object to apply to state.
            state (Tensor): Density matrix as a ``Tensor`` supported by this
                backend.

        Returns:
            Density matrix after applying the gate as a ``Tensor``.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def density_matrix_matrix_call(self, gate, state): # pragma: no cover
        """Applies gate to density matrix using the gate's unitary matrix representation.

        This method is useful for the ``custom`` backend for which some gates
        do not require the unitary matrix.

        Args:
            gate (:class:`qibo.abstractions.abstract_gates.BackendGate`): Gate
                object to apply to state.
            state (Tensor): Density matrix as a ``Tensor`` supported by this
                backend.

        Returns:
            Density matrix after applying the gate as a ``Tensor``.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def state_vector_collapse(self, gate, state, result): # pragma: no cover
        """Collapses state vector to a given result."""
        raise_error(NotImplementedError)

    @abstractmethod
    def density_matrix_collapse(self, gate, state, result): # pragma: no cover
        """Collapses density matrix to a given result."""
        raise_error(NotImplementedError)
