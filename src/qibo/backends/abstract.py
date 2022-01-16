import os
from abc import ABC, abstractmethod
from qibo.config import raise_error, log


class AbstractBackend(ABC):

    TEST_REGRESSIONS = {}

    def __init__(self):
        self.backend = None
        self.name = "base"
        self.is_hardware = False

        self.precision = 'double'
        self._dtypes = {'DTYPEINT': 'int64', 'DTYPE': 'float64',
                        'DTYPECPX': 'complex128'}

        self.cpu_devices = []
        self.gpu_devices = []
        self.default_device = []
        self.nthreads = None
        import psutil
        # using physical cores by default
        self.nthreads = psutil.cpu_count(logical=False)

        self.is_custom = False
        self._matrices = None
        self.numeric_types = None
        self.tensor_types = None
        self.native_types = None
        self.Tensor = None
        self.random = None
        self.newaxis = None
        self.oom_error = None
        self.optimization = None
        self.supports_multigpu = False
        self.supports_gradients = False

    def test_regressions(self, name):
        """Correct outcomes for tests that involve random numbers.

        The outcomes of such tests depend on the backend.
        """
        return self.TEST_REGRESSIONS.get(name)

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
        self.matrices.allocate_matrices()

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
        with self.device(self.default_device):
            self.matrices.allocate_matrices()

    def set_threads(self, nthreads):
        """Set number of OpenMP threads.

        Args:
            num_threads (int): number of threads.
        """
        if not isinstance(nthreads, int): # pragma: no cover
            raise_error(RuntimeError, "Number of threads must be integer.")
        if nthreads < 1: # pragma: no cover
            raise_error(RuntimeError, "Number of threads must be positive.")
        self.nthreads = nthreads

    @abstractmethod
    def set_platform(self, platform):  # pragma: no cover
        """Sets the platform used by the backend.

        Not all backends support different platforms.
        'qibojit' GPU supports two platforms ('cupy', 'cuquantum').
        'qibolab' supports multiple platforms depending on the quantum hardware.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def get_platform(self):  # pragma: no cover
        """Returns the name of the activated platform.

        See :meth:`qibo.backends.abstract.AbstractBackend.set_platform` for
        more details on platforms.
        """
        raise_error(NotImplementedError)

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
            log.warn(f"Falling back to CPU for '{func.__name__}' because the GPU is out-of-memory.")
            with self.device(self.get_cpu()):
                return func(*args)

    @property
    def matrices(self):
        if self._matrices is None:
            from qibo.backends.matrices import Matrices
            self._matrices = Matrices(self)
        return self._matrices

    def circuit_class(self, accelerators=None, density_matrix=False):
        """Returns class used to create circuit model.

        Useful for hardware backends which use different circuit models.

        Args:
            accelerators (dict): Dictionary that maps device names to the number of
                times each device will be used.
                See :class:`qibo.core.distcircuit.DistributedCircuit` for more
                details.
            density_matrix (bool): If ``True`` it creates a circuit for density
                matrix simulation. Default is ``False`` which corresponds to
                state vector simulation.
        """
        # this method returns circuit objects defined in ``qibo.core`` which
        # are used for classical simulation.
        # Hardware backends should redefine this method to return the
        # corresponding hardware circuit objects.
        if density_matrix:
            if accelerators is not None:
                raise_error(NotImplementedError, "Distributed circuits are not "
                                                 "implemented for density "
                                                 "matrices.")
            from qibo.core.circuit import DensityMatrixCircuit
            return DensityMatrixCircuit
        elif accelerators is not None:
            from qibo.core.distcircuit import DistributedCircuit
            return DistributedCircuit
        else:
            from qibo.core.circuit import Circuit
            return Circuit

    def create_gate(self, cls, *args, **kwargs):
        """Create gate objects supported by the backend.

        Useful for hardware backends which use different gate objects.
        """
        # this method returns gate objects defined in ``qibo.core.gates`` which
        # are used for classical simulation.
        # Hardware backends should redefine this method to return the
        # corresponding hardware gate objects (with pulse represenntation etc).
        from qibo.abstractions.abstract_gates import BaseBackendGate
        return BaseBackendGate.__new__(cls)

    @abstractmethod
    def to_numpy(self, x): # pragma: no cover
        """Convert tensor to numpy."""
        raise_error(NotImplementedError)

    @abstractmethod
    def to_complex(self, re, img): # pragma: no cover
        """Creates complex number from real numbers."""
        raise_error(NotImplementedError)

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
        """Creates the default initial state ``|00...0>`` as a tensor."""
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
    def _state_vector_call(self, gate, state): # pragma: no cover
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
    def _density_matrix_call(self, gate, state): # pragma: no cover
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
    def _density_matrix_half_call(self, gate, state): # pragma: no cover
        """Half gate application to density matrix."""
        raise_error(NotImplementedError)

    @abstractmethod
    def density_matrix_half_matrix_call(self, gate, state): # pragma: no cover
        """Half gate application to density matrix using the gate's unitary matrix representation."""
        raise_error(NotImplementedError)

    @abstractmethod
    def state_vector_collapse(self, gate, state, result): # pragma: no cover
        """Collapses state vector to a given result."""
        raise_error(NotImplementedError)

    @abstractmethod
    def density_matrix_collapse(self, gate, state, result): # pragma: no cover
        """Collapses density matrix to a given result."""
        raise_error(NotImplementedError)

    @abstractmethod
    def on_cpu(self): # pragma: no cover
        """Used as `with K.on_cpu():` to perform following operations on CPU."""
        raise_error(NotImplementedError)

    @abstractmethod
    def cpu_tensor(self, x, dtype=None): # pragma: no cover
        """Creates backend tensors to be casted on CPU only.

        Used by :class:`qibo.core.states.DistributedState` to save state pieces
        on CPU instead of GPUs during a multi-GPU simulation.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def cpu_cast(self, x, dtype='DTYPECPX'): # pragma: no cover
        """Forces tensor casting on CPU.

        In contrast to simply `K.cast` which uses the current default device.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def cpu_assign(self, state, i, piece): # pragma: no cover
        """Assigns updated piece to a state object by transfering from GPU to CPU.

        Args:
            state (:class:`qibo.core.states.DistributedState`): State object to
                assign the updated piece to.
            i (int): Index to assign the updated piece to.
            piece (K.Tensor): GPU tensor to transfer to CPU and assign to the
                piece of given index.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def transpose_state(self, pieces, state, nqubits, order): # pragma: no cover
        """Transposes distributed state pieces to obtain the full state vector.

        Used by :meth:`qibo.backends.abstract.AbstractMultiGpu.calculate_tensor`.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def assert_allclose(self, value, target, rtol=1e-7, atol=0.0): # pragma: no cover
        """Check that two arrays are equal. Useful for testing."""
        raise_error(NotImplementedError)


class AbstractCustomOperators:  # pragma: no cover
    """Abstraction for backends that are based on custom operators.

    Such backends are `qibojit <https://github.com/qiboteam/qibojit>`_ and
    `qibotf <https://github.com/qiboteam/qibotf>`_.
    """

    def __init__(self):
        self._gate_ops = {
            "x": self.apply_x,
            "y": self.apply_y,
            "z": self.apply_z,
            "m": self.collapse_state,
            "u1": self.apply_z_pow,
            "cx": self.apply_x,
            "cz": self.apply_z,
            "cu1": self.apply_z_pow,
            "swap": self.apply_swap,
            "fsim": self.apply_fsim,
            "generalizedfsim": self.apply_fsim,
            "ccx": self.apply_x
            }

    def get_gate_op(self, gate):
        """Finds the custom operator function that corresponds to the given gate.

        Args:
            gate (qibo.abstractions.abstract_gates.Gate): Gate object to apply.

        Returns:
            A callable that applies the custom operator corresponding to the
            given gate.
        """
        if gate.name in self._gate_ops:
            return self._gate_ops.get(gate.name)
        elif gate.__class__.__name__ == "_ThermalRelaxationChannelB":
            return self.apply_two_qubit_gate
        n = len(gate.target_qubits)
        if n == 1:
            return self.apply_gate
        elif n == 2:
            return self.apply_two_qubit_gate
        else:
            return self.apply_multi_qubit_gate

    @abstractmethod
    def apply_gate(self, state, gate, nqubits, targets, qubits=None):
        """Applies one-qubit gate using matrix multiplication.

        The gate may be controlled on arbitrary number of qubits.

        Args:
            state: State vector as a backend supported tensor.
            gate: Gate matrix as a backend supported tensor.
            nqubits (int): Total number of qubits in the system.
            targets (tuple): Target qubit ids that the gate acts on.
            qubits: Sorted list of target and control qubit ids sorted as a
                backend supported tensor.

        Returns:
            The state vector after the gate is applied as a backend supported
            tensor.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def apply_x(self, state, nqubits, targets, qubits=None):
        """Applies Pauli-X gate.

        See :meth:`qibo.backends.abstract.AbstractCustomOperators.apply_gate`
        for information on arguments.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def apply_y(self, state, nqubits, targets, qubits=None):
        """Applies Pauli-Y gate.

        See :meth:`qibo.backends.abstract.AbstractCustomOperators.apply_gate`
        for information on arguments.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def apply_z(self, state, nqubits, targets, qubits=None):
        """Applies Pauli-Z gate.

        See :meth:`qibo.backends.abstract.AbstractCustomOperators.apply_gate`
        for information on arguments.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def apply_z_pow(self, state, gate, nqubits, targets, qubits=None):
        """Applies U1 gate.

        See :meth:`qibo.backends.abstract.AbstractCustomOperators.apply_gate`
        for information on arguments.
        The ``gate`` argument here corresponds to the phase to be applied, not
        the full matrix.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def apply_two_qubit_gate(self, state, gate, nqubits, targets, qubits=None):
        """Applies two-qubit gate using matrix multiplication.

        See :meth:`qibo.backends.abstract.AbstractCustomOperators.apply_gate`
        for information on arguments.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def apply_swap(self, state, nqubits, targets, qubits=None):
        """Applies SWAP gate.

        See :meth:`qibo.backends.abstract.AbstractCustomOperators.apply_gate`
        for information on arguments.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def apply_fsim(self, state, gate, nqubits, targets, qubits=None):
        """Applies fSim gate.

        See :meth:`qibo.backends.abstract.AbstractCustomOperators.apply_gate`
        for information on arguments.
        The ``gate`` argument here is a tensor of length 5 corresponding to the
        non-zero elements of :class:`qibo.abstractions.gates.GeneralizedfSim`.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def apply_multi_qubit_gate(self, state, gate, nqubits, targets, qubits=None):
        """Applies multi-qubit gate with three or more targets using matrix multiplication.

        See :meth:`qibo.backends.abstract.AbstractCustomOperators.apply_gate`
        for information on arguments.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def collapse_state(self, state, qubits, result, nqubits, normalize=True):
        """Collapses state according to the given measurement result.

        Args:
            state: State vector as a backend supported tensor.
            qubits: Sorted list of target qubit ids sorted as a backend
                supported tensor.
            result (int): Measurement result on the target qubits converted
                from binary to decimal.
            nqubits (int): Total number of qubits in the system.
            normalize (bool): If ``True`` the collapsed state is normalized.

        Returns:
            State after collapse as a backend supported tensor.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def swap_pieces(self, piece0, piece1, new_global, nlocal): # pragma: no cover
        """Swaps two distributed state pieces in order to change the global qubits.

        Useful to apply SWAP gates on distributed states.
        """
        raise_error(NotImplementedError)
