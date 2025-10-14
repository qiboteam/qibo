"""Module defining the Backend class."""

import math
from collections import Counter
from typing import List, Optional, Tuple, Union

from qibo import __version__
from qibo.backends import einsum_utils
from qibo.config import SHOT_BATCH_SIZE, log, raise_error
from qibo.result import CircuitResult, MeasurementOutcomes, QuantumState


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
        )
        self.oom_error = MemoryError
        self.platform = None
        self.supports_multigpu = False
        self.tensor_types = ()
        self.versions = {"qibo": __version__}

    def __reduce__(self) -> Tuple["Backend", tuple]:
        """Allow pickling backend objects that have references to modules."""
        return self.__class__, tuple()

    def __repr__(self) -> str:
        if self.platform is None:
            return self.name

        return f"{self.name} ({self.platform})"

    @property
    def qubits(self) -> Optional[List[Union[int, str]]]:  # pragma: no cover
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

    def cast(
        self, array, dtype=None, copy: bool = False  # pylint: disable=unused-argument
    ) -> "ndarray":  # pragma: no cover
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

    def compile(self, func):  # pragma: no cover
        """Compile the given method.

        Available only for the ``TensorflowBackend`` in ``qiboml``.
        """
        return func

    def is_sparse(self, array) -> bool:
        """Determine if a given array is a sparse tensor."""
        from scipy.sparse import issparse  # pylint: disable=import-outside-toplevel

        return issparse(array)

    def set_device(self, device: str) -> None:  # pragma: no cover
        """Set simulation device. Works in-place.

        Args:
            device (str): Device index, *e.g.* ``/CPU:0`` for CPU, or ``/GPU:1`` for
                the second GPU in a multi-GPU environment.
        """
        raise_error(NotImplementedError)

    def set_dtype(self, dtype: str) -> None:  # pragma: no cover
        """Set data type of arrays created using the backend. Works in-place.

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
        dtypes_str = ("float32", "float64", "complex64", "complex128")

        if dtype not in self.numeric_types and dtype not in dtypes_str:
            raise_error(
                ValueError,
                f"Unknown ``dtype`` ``{dtype}``. For this backend ({self}), "
                + f"``dtype`` must be either one of the following string: {dtypes_str}, "
                + f"or one of the following options: {self.numeric_types}",
            )

        if dtype != self.dtype:
            self.dtype = dtype

            if self.matrices:
                self.matrices = self.matrices.__class__(self.dtype)

    def set_seed(self, seed: Union[int, None]) -> None:
        """Set the seed of the random number generator. Works in-place."""
        self.engine.random.seed(seed)

    def set_threads(self, nthreads: int) -> None:  # pragma: no cover
        """Set number of threads for CPU backend simulations that accept it. Works in-place.

        Args:
            nthreads (int): Number of threads.
        """
        raise_error(NotImplementedError)

    def to_numpy(self, array) -> "ndarray":  # pragma: no cover
        """Cast a given array to numpy."""
        raise_error(NotImplementedError)

    ########################################################################################
    ######## Methods related to data types                                          ########
    ########################################################################################

    @property
    def complex64(self):
        return getattr(self.engine, "complex64")

    @property
    def complex128(self):
        return getattr(self.engine, "complex128")

    @property
    def float32(self):
        return getattr(self.engine, "float32")

    @property
    def float64(self):
        return getattr(self.engine, "float64")

    @property
    def int8(self):
        return getattr(self.engine, "int8")

    @property
    def int32(self):
        return getattr(self.engine, "int32")

    @property
    def int64(self):
        return getattr(self.engine, "int64")

    @property
    def uint8(self):
        return getattr(self.engine, "uint8")

    ########################################################################################
    ######## Methods related to array manipulation                                  ########
    ########################################################################################

    def abs(self, array, **kwargs) -> Union[int, float, complex, "ndarray"]:
        return self.engine.abs(array, **kwargs)

    def all(self, array, **kwargs) -> Union["ndarray", bool]:
        return self.engine.all(array, **kwargs)

    def angle(self, array, **kwargs) -> "ndarray":
        return self.engine.angle(array, **kwargs)

    def any(self, array, **kwargs) -> Union["ndarray", bool]:
        return self.engine.any(array, **kwargs)

    def arccos(self, array, **kwargs) -> "ndarray":
        return self.engine.arccos(array, **kwargs)

    def arctan2(self, array_1, array_2, **kwargs) -> "ndarray":
        return self.engine.arctan2(array_1, array_2, **kwargs)

    def argsort(self, array, axis=None, **kwargs) -> "ndarray":
        return self.engine.argsort(array, axis, **kwargs)

    def block(self, arrays) -> "ndarray":
        return self.engine.block(arrays)

    def block_diag(self, *arrays) -> "ndarray":
        from scipy.linalg import block_diag  # pylint: disable=import-outside-toplevel

        return block_diag(*arrays)

    def ceil(self, array, **kwargs) -> "ndarray":
        return self.engine.ceil(array, **kwargs)

    def concatenate(self, tup, **kwargs) -> "ndarray":
        return self.engine.concatenate(tup, **kwargs)

    def conj(self, array) -> "ndarray":
        return self.engine.conj(array)

    def copy(self, array, **kwargs) -> "ndarray":
        return self.engine.copy(array, **kwargs)

    def cos(self, array, **kwargs) -> "ndarray":
        return self.engine.cos(array, **kwargs)

    def csr_matrix(self, array):
        from scipy.sparse import csr_matrix  # pylint: disable=import-outside-toplevel

        return csr_matrix(array)

    def det(self, array) -> "ndarray":
        return self.engine.linalg.det(array)

    def diag(self, array, **kwargs) -> "ndarray":
        return self.engine.diag(array, **kwargs)

    def dot(self, array_1, array_2, **kwargs) -> "ndarray":
        return self.engine.dot(array_1, array_2, **kwargs)

    def eig(self, array, **kwargs):
        return self.engine.linalg.eig(array, **kwargs)

    def eigh(self, array, **kwargs):
        return self.engine.linalg.eigh(array, **kwargs)

    def eigsh(self, array, **kwargs):
        from scipy.sparse.linalg import (  # pylint: disable=import-outside-toplevel
            eigsh,
        )

        return eigsh(array, **kwargs)

    def eigvalsh(self, array, **kwargs):
        return self.engine.linalg.eigvalsh(array, **kwargs)

    def eigvals(self, array, **kwargs):
        return self.engine.linalg.eigvals(array, **kwargs)

    def einsum(self, subscripts: str, *operands, **kwargs) -> "ndarray":
        return self.engine.einsum(subscripts, *operands, **kwargs)

    def empty(self, shape, **kwargs) -> "ndarray":
        return self.engine.empty(shape, **kwargs)

    def exp(self, array, **kwargs):
        return self.engine.exp(array, **kwargs)

    def expm(self, array) -> "ndarray":
        if self.is_sparse(array):
            from scipy.sparse.linalg import (  # pylint: disable=import-outside-toplevel
                expm,
            )
        else:
            from scipy.linalg import expm  # pylint: disable=import-outside-toplevel

        return expm(array)

    def flatnonzero(self, array) -> "ndarray":
        return self.engine.flatnonzero(array)

    def floor(self, array, **kwargs) -> "ndarray":
        return self.engine.floor(array, **kwargs)

    def identity(self, dims: int, dtype=None, sparse: bool = False) -> "ndarray":
        if dtype is None:
            dtype = self.dtype

        if sparse:
            return self._identity_sparse(dims, dtype)

        return self.engine.eye(dims, dtype=dtype)

    def imag(self, array) -> Union[int, float, "ndarray"]:
        return self.engine.imag(array)

    def inv(self, array) -> "ndarray":
        return self.engine.linalg.inv(array)

    def kron(self, array_1, array_2) -> "ndarray":
        return self.engine.kron(array_1, array_2)

    def log(self, array, **kwargs) -> "ndarray":
        return self.engine.log(array, **kwargs)

    def logm(self, array, **kwargs) -> "ndarray":
        from scipy.linalg import logm  # pylint: disable=import-outside-toplevel

        return logm(array, **kwargs)

    def log2(self, array, **kwargs) -> "ndarray":
        return self.engine.log2(array, **kwargs)

    def log10(self, array, **kwargs) -> "ndarray":
        return self.engine.log10(array, **kwargs)

    def matmul(self, array_1, array_2, **kwargs) -> "ndarray":
        return self.engine.matmul(array_1, array_2, **kwargs)

    def matrix_norm(
        self, state, order: Union[int, float, str] = "nuc", **kwargs
    ):  # pragma: no cover
        """Calculate norm of a :math:`2`-dimensional array.

        Default is the ``nuclear`` norm.
        If ``order="nuc"``, it returns the nuclear norm of ``state``,
        assuming ``state`` is Hermitian (also known as trace norm).
        For specifications on the other  possible values of the
        parameter ``order`` for the ``tensorflow`` backend, please refer to
        `tensorflow.norm <https://www.tensorflow.org/api_docs/python/tf/norm>`_.
        For all other backends, please refer to
        `numpy.linalg.norm
        <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>`_.
        """
        dtype = kwargs.get("dtype", None)

        if dtype is None:
            dtype = self.dtype

        state = self.cast(state, dtype=dtype)  # pylint: disable=E1111

        return self.engine.linalg.norm(state, order, **kwargs)

    def mean(self, array, **kwargs) -> Union[float, complex, "ndarray"]:
        return self.engine.mean(array, **kwargs)

    def nonzero(self, array) -> "ndarray":
        return self.engine.nonzero(array)

    def ones(self, shape, dtype=None) -> "ndarray":  # pragma: no cover
        if dtype is None:
            dtype = self.dtype
        return self.engine.ones(shape, dtype=dtype)

    def outer(self, array_1, array_2) -> "ndarray":  # pragma: no cover
        return self.engine.outer(array_1, array_2)

    def prod(self, array, **kwargs) -> "ndarray":
        return self.engine.prod(array, **kwargs)

    def qr(self, array, **kwargs) -> Tuple["ndarray", ...]:
        return self.engine.linalg.qr(array, **kwargs)

    def real(self, array) -> Union[int, float, "ndarray"]:
        return self.engine.real(array)

    def random_choice(self, array, **kwargs) -> "ndarray":  # pragma: no cover
        return self.engine.random.choice(array, **kwargs)

    def random_integers(self, low, high=None, size=None, dtype=None):
        if dtype is None:
            dtype = self.dtype

        return self.engine.random.randint(low, high, size=size, dtype=dtype)

    def random_sample(self, size: int):
        return self.engine.random.random(size)

    def ravel(self, array, **kwargs) -> "ndarray":
        return self.engine.ravel(array, **kwargs)

    def reshape(
        self, array, shape: Union[Tuple[int, ...], List[int]], **kwargs
    ) -> "ndarray":
        return self.engine.reshape(array, shape, **kwargs)

    def round(self, array, decimals: int = 0, **kwargs):
        return self.engine.round(array, decimals, **kwargs)

    def shuffle(self, array, **kwargs) -> "ndarray":
        self.engine.random.shuffle(array, **kwargs)

    def sin(self, array, **kwargs) -> "ndarray":
        return self.engine.sin(array, **kwargs)

    def sort(self, array, **kwargs) -> "ndarray":
        return self.engine.sort(array, **kwargs)

    def sqrt(self, array):
        return self.engine.sqrt(array)

    def squeeze(
        self, array, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> "ndarray":
        return self.engine.squeeze(array, axis)

    def std(self, array, **kwargs) -> Union[float, "ndarray"]:
        return self.engine.std(array, **kwargs)

    def sum(self, array, axis=None, **kwargs) -> Union[int, float, complex, "ndarray"]:
        return self.engine.sum(array, axis=axis, **kwargs)

    def swapaxes(self, array, axis_1: int, axis_2: int) -> "ndarray":
        return self.engine.swapaxes(array, axis_1, axis_2)

    def tensordot(self, array_1, array_2, axes: Union[int, Tuple[int, ...]] = 2):
        return self.engine.tensordot(array_1, array_2, axes=axes)

    def trace(self, array) -> Union[int, float]:
        return self.engine.trace(array)

    def transpose(
        self, array, axes: Union[Tuple[int, ...], List[int]] = None
    ) -> "ndarray":
        return self.engine.transpose(array, axes)

    def tril(self, array, k: int = 0) -> "ndarray":
        return self.engine.tril(array, k=k)

    def tril_indices(self, n: int, k: int = 0, m: int = None):
        return self.engine.tril_indices(n, k, m)

    def unique(self, array, **kwargs) -> Union["ndarray", Tuple["ndarray", "ndarray"]]:
        return self.engine.unique(array, **kwargs)

    def vector_norm(
        self, state, order: Union[int, float, str] = 2, dtype=None
    ) -> float:  # pragma: no cover
        """Calculate norm of an :math:`1`-dimensional array.

        For specifications on possible values of the parameter ``order``
        for the ``tensorflow`` backend, please refer to
        `tensorflow.norm <https://www.tensorflow.org/api_docs/python/tf/norm>`_.
        For all other backends, please refer to
        `numpy.linalg.norm
        <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>`_.
        """
        if dtype is None:
            dtype = self.dtype

        state = self.cast(state, dtype=dtype)  # pylint: disable=E1111

        return self.engine.linalg.norm(state, order)

    def vstack(self, arrays, **kwargs) -> "ndarray":
        return self.engine.vstack(arrays, **kwargs)

    def zeros(self, shape, dtype=None) -> "ndarray":  # pragma: no cover
        if dtype is None:
            dtype = self.dtype
        return self.engine.zeros(shape, dtype=dtype)

    def zeros_like(self, array, dtype=None, **kwargs) -> "ndarray":
        return self.engine.zeros_like(array, dtype=dtype, **kwargs)

    ########################################################################################
    ######## Methods related to linear algebra operations                           ########
    ########################################################################################

    def eigenvalues(self, matrix, k: int = 6, hermitian: bool = True):
        """Calculate eigenvalues of a matrix."""
        if self.is_sparse(matrix):
            log.warning(
                "Calculating sparse matrix eigenvectors because "
                "sparse modules do not provide ``eigvals`` method."
            )
            return self.eigenvectors(matrix, k=k)[0]

        if hermitian:
            return self.eigvalsh(matrix)

        return self.eigvals(matrix)

    def eigenvectors(
        self, matrix, k: int = 6, hermitian: bool = True
    ):  # pragma: no cover
        """Calculate eigenvectors of a matrix."""
        if self.is_sparse(matrix):
            if k < matrix.shape[0]:
                return self.eigsh(matrix, k=k, which="SA")

            matrix = self.to_numpy(matrix)  # pylint: disable=E1111

        if hermitian:
            return self.eigh(matrix)

        return self.eig(matrix)

    def expectation_value(
        self, hamiltonian, state, normalize: bool = False, dtype=None
    ):  # pragma: no cover
        """Calculate expectation value of a state vector given the observable matrix."""
        if dtype is None:
            dtype = self.dtype

        density_matrix = bool(len(state.shape) == 2)

        state = self.cast(state, dtype=dtype)  # pylint: disable=E1111
        prod = hamiltonian @ state
        exp_val = (
            self.trace(prod) if density_matrix else self.sum(self.conj(state) * prod)
        )
        exp_val = self.real(exp_val)

        if normalize:
            norm = (
                self.trace(state) if density_matrix else self.sum(self.abs(state) ** 2)
            )
            norm = self.real(norm)
            exp_val /= norm

        return exp_val

    def jacobian(
        self, circuit, parameters, initial_state=None, return_complex: bool = True
    ):  # pragma: no cover
        """Calculate the Jacobian matrix of ``circuit`` with respect to varables ``params``."""
        raise_error(
            NotImplementedError,
            "This method is only implemented in backends that allow automatic differentiation, "
            + "e.g. ``PytorchBackend`` and ``TensorflowBackend``.",
        )

    def matrix_exp(
        self,
        matrix,
        phase: Union[float, int, complex] = 1,
        eigenvectors=None,
        eigenvalues=None,
    ):
        """Calculate the exponential :math:`e^{\\theta \\, A}` of a matrix :math:`A`
        and ``phase`` :math:`\\theta`.

        If the eigenvectors and eigenvalues are given the matrix diagonalization is
        used for exponentiation.
        """
        if eigenvectors is None or self.is_sparse(matrix):
            return self.expm(phase * matrix)

        expd = self.exp(phase * eigenvalues)
        ud = self.transpose(self.conj(eigenvectors))

        return (eigenvectors * expd) @ ud

    def matrix_log(
        self, matrix, base: Union[float, int] = 2, eigenvectors=None, eigenvalues=None
    ):
        """Calculate the logarithm :math:`\\log_{b}(A)` with a ``base`` :math:`b`
        of a matrix :math:`A`.

        If the eigenvectors and eigenvalues are given the matrix diagonalization is
        used for exponentiation.
        """
        if eigenvectors is None:
            # to_numpy and cast needed for GPUs
            log_matrix = self.logm(matrix) / float(self.log(base))

            return log_matrix

        log_matrix = self.log(eigenvalues) / float(self.log(base))
        ud = self.transpose(self.conj(eigenvectors))

        return (eigenvectors * log_matrix) @ ud

    def matrix_power(
        self,
        matrix,
        power: Union[float, int],
        precision_singularity: float = 1e-14,
        dtype=None,
    ):  # pragma: no cover
        """Calculate the (fractional) ``power`` :math:`\\alpha` of ``matrix`` :math:`A`,
        i.e. :math:`A^{\\alpha}`.

        .. note::
            For the ``pytorch`` backend, this method relies on a copy of the original tensor.
            This may break the gradient flow. For the GPU backends (i.e. ``cupy`` and
            ``cuquantum``), this method falls back to CPU whenever ``power`` is not
            an integer.
        """
        if not isinstance(power, (float, int)):
            raise_error(
                TypeError,
                f"``power`` must be either float or int, but it is type {type(power)}.",
            )

        if dtype is None:
            dtype = self.dtype

        if power < 0.0:
            # negative powers of singular matrices via SVD
            determinant = self.det(matrix)
            if abs(determinant) < precision_singularity:
                return self._negative_power_singular_matrix(
                    matrix, power, precision_singularity, dtype=dtype
                )

        from scipy.linalg import (  # pylint: disable=import-outside-toplevel
            fractional_matrix_power,
        )

        return fractional_matrix_power(matrix, power)

    def matrix_sqrt(self, array):  # pragma: no cover
        """Calculate the square root of ``matrix`` :math:`A`, i.e. :math:`A^{1/2}`.

        .. note::
            For the ``pytorch`` backend, this method relies on a copy of the original tensor.
            This may break the gradient flow. For the GPU backends (i.e. ``cupy`` and
            ``cuquantum``), this method falls back to CPU.
        """
        return self.matrix_power(array, power=0.5)

    def partial_trace(
        self, state, traced_qubits: Union[Tuple[int, ...], List[int]]
    ) -> "ndarray":
        state = self.cast(state, dtype=state.dtype)  # pylint: disable=E1111

        nqubits = math.log2(state.shape[0])

        if not nqubits.is_integer():
            raise_error(
                ValueError,
                "dimension(s) of ``state`` must be a power of 2, "
                + f"but it is {2**nqubits}.",
            )

        nqubits = int(nqubits)

        statevector = bool(len(state.shape) == 1)

        factor = 1 if statevector else 2
        state = self.reshape(state, factor * nqubits * (2,))

        if statevector:
            axes = 2 * [list(traced_qubits)]
            rho = self.tensordot(state, self.conj(state), axes)
            shape = 2 * (2 ** (nqubits - len(traced_qubits)),)

            return self.reshape(rho, shape)

        order = tuple(sorted(traced_qubits))
        order += tuple(set(list(range(nqubits))) ^ set(traced_qubits))
        order += tuple(elem + nqubits for elem in order)
        shape = 2 * (2 ** len(traced_qubits), 2 ** (nqubits - len(traced_qubits)))

        state = self.transpose(state, order)
        state = self.reshape(state, shape)

        return self.engine.einsum("abac->bc", state)

    def singular_value_decomposition(self, array) -> Tuple["ndarray", ...]:
        """Calculate the Singular Value Decomposition of ``matrix``."""
        return self.engine.linalg.svd(array)

    ########################################################################################
    ######## Methods related to the creation and manipulation of quantum objects    ########
    ########################################################################################

    def depolarizing_error_density_matrix(self, gate, state, nqubits):
        state = self.cast(state, dtype=state.dtype)  # pylint: disable=E1111
        shape = state.shape
        target_qubits = gate.target_qubits
        lam = gate.init_kwargs["lam"]

        trace = self.partial_trace(state, target_qubits)
        trace = self.reshape(trace, 2 * (nqubits - len(target_qubits)) * (2,))
        identity = self.maximally_mixed_state(len(target_qubits))
        identity = self.reshape(identity, 2 * len(target_qubits) * (2,))
        identity = self.tensordot(trace, identity, 0)

        qubits = list(range(nqubits))
        for j in target_qubits:
            qubits.pop(qubits.index(j))
        qubits.sort()
        qubits += list(target_qubits)

        qubit_1 = list(range(nqubits - len(target_qubits))) + list(
            range(2 * (nqubits - len(target_qubits)), 2 * nqubits - len(target_qubits))
        )
        qubit_2 = list(
            range(nqubits - len(target_qubits), 2 * (nqubits - len(target_qubits)))
        )
        qubit_2 += list(range(2 * nqubits - len(target_qubits), 2 * nqubits))
        qs = [qubit_1, qubit_2]

        order = []
        for qj in qs:
            qj = [qj[qubits.index(i)] for i in range(len(qubits))]
            order += qj

        identity = self.reshape(self.transpose(identity, order), shape)
        state = (1 - lam) * state + lam * identity

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

    def overlap_statevector(self, state_1, state_2, dtype=None):
        """Calculate overlap of two pure quantum states."""
        if dtype is None:
            dtype = self.dtype

        state_1 = self.cast(state_1, dtype=dtype)  # pylint: disable=E1111
        state_2 = self.cast(state_2, dtype=dtype)  # pylint: disable=E1111

        return self.sum(self.conj(state_1) * state_2)

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

    def reset_error_density_matrix(self, gate, state, nqubits: int):  # pragma: no cover
        """Apply reset error to density matrix."""
        from qibo.gates.gates import X  # pylint: disable=import-outside-toplevel

        state = self.cast(state, dtype=state.dtype)  # pylint: disable=E1111
        shape = state.shape
        qubit = gate.target_qubits[0]
        p_0, p_1 = gate.init_kwargs["p_0"], gate.init_kwargs["p_1"]
        trace = self.partial_trace(state, (qubit,))
        trace = self.reshape(trace, 2 * (nqubits - 1) * (2,))
        zero = self.zero_state(nqubits=1, density_matrix=True)
        zero = self.tensordot(trace, zero, 0)
        order = list(range(2 * nqubits - 2))
        order.insert(qubit, 2 * nqubits - 2)
        order.insert(qubit + nqubits, 2 * nqubits - 1)
        zero = self.reshape(self.transpose(zero, order), shape)
        state = (1 - p_0 - p_1) * state + p_0 * zero

        return state + p_1 * self.apply_gate(X(qubit), zero, nqubits)

    def thermal_error_density_matrix(
        self, gate, state, nqubits: int
    ):  # pragma: no cover
        """Apply thermal relaxation error to density matrix."""
        state = self.cast(state, dtype=state.dtype)  # pylint: disable=E1111
        shape = state.shape
        state = self.apply_gate(gate, state.ravel(), 2 * nqubits)
        return self.reshape(state, shape)

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

        state = self.zeros(shape, dtype=dtype)
        if density_matrix:
            state[0, 0] = 1
        else:
            state[0] = 1

        return state

    ########################################################################################
    ######## Methods related to circuit execution                                   ########
    ########################################################################################

    def apply_bitflips(self, noiseless_samples, bitflip_probabilities):
        sprobs = self.random_sample(noiseless_samples.shape)
        sprobs = self.cast(sprobs, dtype="float64")  # pylint: disable=E1111

        flip_0 = self.cast(  # pylint: disable=E1111
            sprobs < bitflip_probabilities[0], dtype=noiseless_samples.dtype
        )
        flip_1 = self.cast(  # pylint: disable=E1111
            sprobs < bitflip_probabilities[1], dtype=noiseless_samples.dtype
        )

        noisy_samples = noiseless_samples + (1 - noiseless_samples) * flip_0
        noisy_samples = noisy_samples - noiseless_samples * flip_1

        return noisy_samples

    def apply_gate(self, gate, state, nqubits: int) -> "ndarray":
        """Apply a gate to quantum state."""

        density_matrix = bool(len(state.shape) == 2)

        shape = nqubits * (2,)
        if density_matrix:
            shape *= 2

        state = self.reshape(state, shape=shape)

        if gate.is_controlled_by and density_matrix:
            return self._apply_gate_controlled_by_density_matrix(gate, state, nqubits)

        if gate.is_controlled_by:
            return self._apply_gate_controlled_by(gate, state, nqubits)

        matrix = gate.matrix(self)
        matrix = self.reshape(matrix, 2 * len(gate.qubits) * (2,))

        if density_matrix:
            matrix_conj = self.conj(matrix)
            left, right = einsum_utils.apply_gate_density_matrix_string(
                gate.qubits, nqubits
            )
            state = self.engine.einsum(right, state, matrix_conj)
            state = self.engine.einsum(left, state, matrix)
        else:
            opstring = einsum_utils.apply_gate_string(gate.qubits, nqubits)
            state = self.engine.einsum(opstring, state, matrix)

        shape = (2**nqubits,)
        if density_matrix:
            shape *= 2

        return self.reshape(state, shape)

    def apply_gate_half_density_matrix(
        self, gate, state, nqubits: int
    ):  # pragma: no cover
        """Apply a gate to one side of the density matrix."""
        if gate.is_controlled_by:
            raise_error(
                NotImplementedError,
                "Gate density matrix half call is "
                "not implemented for ``controlled_by``"
                "gates.",
            )

        state = self.cast(state, dtype=state.dtype)  # pylint: disable=E1111
        state = self.reshape(state, 2 * nqubits * (2,))
        matrix = gate.matrix(self)

        matrix = self.reshape(matrix, 2 * len(gate.qubits) * (2,))
        left, _ = einsum_utils.apply_gate_density_matrix_string(gate.qubits, nqubits)
        state = self.engine.einsum(left, state, matrix)

        return self.reshape(state, 2 * (2**nqubits,))

    def apply_channel(self, channel, state, nqubits: int):
        """Apply a ``channel`` to quantum ``state``."""

        density_matrix = bool(len(state.shape) == 2)

        if density_matrix:
            state = self.cast(state, dtype=state.dtype)  # pylint: disable=E1111

            new_state = (1 - channel.coefficient_sum) * state
            for coeff, gate in zip(channel.coefficients, channel.gates):
                new_state += coeff * self.apply_gate(gate, state, nqubits)

            return new_state

        probabilities = channel.coefficients + (1 - self.sum(channel.coefficients),)

        index = int(self.sample_shots(probabilities, 1)[0])
        if index != len(channel.gates):
            gate = channel.gates[index]
            state = self.apply_gate(gate, state, nqubits)

        return state

    def calculate_symbolic(
        self, state, nqubits, decimals=5, cutoff=1e-10, max_terms=20
    ):
        # state = self.to_numpy(state)
        density_matrix = bool(len(state.shape) == 2)
        ind_j = self.nonzero(state)
        if density_matrix:
            ind_k = ind_j[1]
        ind_j = ind_j[0]

        terms = []
        if density_matrix:
            for j, k in zip(ind_j, ind_k):
                b_j = bin(j)[2:].zfill(nqubits)
                b_k = bin(k)[2:].zfill(nqubits)
                if self.abs(state[j, k]) >= cutoff:
                    x = self.round(state[j, k], decimals=decimals)
                    terms.append(f"{x}|{b_j}><{b_k}|")
                if len(terms) >= max_terms:
                    terms.append("...")
                    return terms
        else:
            for j in ind_j:
                b = bin(j)[2:].zfill(nqubits)
                if self.abs(state[j]) >= cutoff:
                    x = self.round(state[j], decimals=decimals)
                    terms.append(f"{x}|{b}>")
                if len(terms) >= max_terms:
                    terms.append("...")
                    return terms

        return terms

    def collapse_state(
        self,
        state,
        qubits: Union[Tuple[int, ...], List[int]],
        shot: int,
        nqubits: int,
        normalize: bool = True,
        density_matrix: bool = False,
    ) -> "ndarray":
        """Collapse quantum state according to measurement shot."""

        if density_matrix:
            return self._collapse_density_matrix(
                state, qubits, shot, nqubits, normalize
            )

        return self._collapse_statevector(state, qubits, shot, nqubits, normalize)

    def execute_circuit(
        self, circuit, initial_state=None, nshots: int = 1000
    ):  # pragma: no cover
        """Execute a :class:`qibo.models.circuit.Circuit`."""
        nqubits = circuit.nqubits
        density_matrix = circuit.density_matrix

        if isinstance(initial_state, type(circuit)):
            if not bool(initial_state.density_matrix == density_matrix):
                raise_error(
                    ValueError,
                    f"Cannot set circuit with density_matrix {initial_state.density_matrix} as"
                    + f"initial state for circuit with density_matrix {density_matrix}.",
                )

            if not bool(
                initial_state.accelerators == circuit.accelerators
            ):  # pragma: no cover
                raise_error(
                    ValueError,
                    f"Cannot set circuit with accelerators {initial_state.density_matrix} as"
                    + f"initial state for circuit with accelerators {density_matrix}.",
                )

            return self.execute_circuit(initial_state + circuit, None, nshots)

        if initial_state is not None:
            initial_state = self.cast(  # pylint: disable=E1111
                initial_state, dtype=initial_state.dtype
            )  # pylint: disable=E1111
            valid_shape = 2 * (2**nqubits,) if density_matrix else (2**nqubits,)
            if tuple(initial_state.shape) != valid_shape:
                raise_error(
                    ValueError,
                    f"Given initial state has shape {initial_state.shape}"
                    + f"instead of the expected {valid_shape}.",
                )

        if circuit.repeated_execution:
            if not circuit.measurements and not circuit.has_collapse:
                raise_error(
                    RuntimeError,
                    "Attempting to perform noisy simulation with `density_matrix=False` "
                    + "and no Measurement gate in the Circuit. If you wish to retrieve the "
                    + "statistics of the outcomes please include measurements in the circuit, "
                    + "otherwise set `density_matrix=True` to recover the final state.",
                )

            return self.execute_circuit_repeated(circuit, nshots, initial_state)

        if circuit.accelerators:  # pragma: no cover
            return self.execute_distributed_circuit(circuit, initial_state, nshots)

        try:
            result = self._execute_circuit(
                circuit, initial_state=initial_state, nshots=nshots
            )
        except self.oom_error:
            raise_error(
                RuntimeError,
                f"State does not fit in {self.device} memory."
                "Please switch the execution device to a "
                "different one using ``qibo.set_device``.",
            )

        return result

    def execute_circuits(
        self, circuits, initial_states=None, nshots: int = None, processes=None
    ):  # pragma: no cover
        """Execute multiple :class:`qibo.models.circuit.Circuit` in parallel."""
        from qibo.parallel import (  # pylint: disable=import-outside-toplevel
            parallel_circuits_execution,
        )

        return parallel_circuits_execution(
            circuits, initial_states, nshots, processes, backend=self
        )

    def execute_circuit_repeated(
        self, circuit: "qibo.models.circuit.Circuit", nshots: int, initial_state=None
    ):  # pragma: no cover
        """Execute a :class:`qibo.models.circuit.Circuit` multiple times.

        Useful for noise simulation using state vectors or for simulating gates
        controlled by measurement outcomes.

        Execute the circuit `nshots` times to retrieve probabilities, frequencies
        and samples. Note that this method is called only if a unitary channel
        is present in the circuit (i.e. noisy simulation) and `density_matrix=False`, or
        if some collapsing measurement is performed.
        """
        density_matrix = circuit.density_matrix

        if circuit.has_collapse and not circuit.measurements and not density_matrix:
            raise_error(
                RuntimeError,
                "The circuit contains only collapsing measurements (`collapse=True`) but "
                + "`density_matrix=False`. Please set `density_matrix=True` to retrieve "
                + "the final state after execution.",
            )

        results, final_states = [], []
        nqubits = circuit.nqubits

        if not density_matrix:
            samples = []
            target_qubits = [
                measurement.target_qubits for measurement in circuit.measurements
            ]
            target_qubits = sum(target_qubits, tuple())

        state_copy = (
            self.zero_state(nqubits, density_matrix=density_matrix)
            if initial_state is None
            else self.cast(initial_state, copy=True)
        )

        for _ in range(nshots):
            state = self.cast(  # pylint: disable=E1111
                state_copy, dtype=state_copy.dtype, copy=True
            )

            if not density_matrix and circuit.accelerators:  # pragma: no cover
                state = self.execute_distributed_circuit(  # pylint: disable=E1111
                    circuit, state
                )
            else:
                for gate in circuit.queue:
                    if gate.symbolic_parameters:
                        gate.substitute_symbols()
                    state = gate.apply(self, state, nqubits)

            if density_matrix:
                final_states.append(state)

            if circuit.measurements:
                result = CircuitResult(
                    state, circuit.measurements, backend=self, nshots=1
                )
                sample = result.samples()[0]
                results.append(sample)
                if not density_matrix:
                    samples.append("".join([str(int(s)) for s in sample]))
                for gate in circuit.measurements:
                    gate.result.reset()

        if density_matrix:  # this implies also it has_collapse
            assert circuit.has_collapse
            final_state = self.mean(final_states, axis=0)
            if circuit.measurements:
                final_result = CircuitResult(
                    final_state,
                    circuit.measurements,
                    backend=self,
                    samples=self.aggregate_shots(results),
                    nshots=nshots,
                )
            else:
                final_result = QuantumState(final_state, backend=self)

            circuit._final_state = final_result  # pylint: disable=protected-access

            return final_result

        final_result = MeasurementOutcomes(
            circuit.measurements,
            backend=self,
            samples=self.aggregate_shots(results),
            nshots=nshots,
        )
        final_result._repeated_execution_frequencies = self.calculate_frequencies(
            samples
        )

        circuit._final_state = final_result  # pylint: disable=protected-access

        return final_result

    def execute_distributed_circuit(
        self, circuit, initial_state=None, nshots: int = None
    ):  # pragma: no cover
        """Execute a :class:`qibo.models.circuit.Circuit` using multiple GPUs."""
        raise_error(
            NotImplementedError, f"{self} does not support distributed execution."
        )

    def matrix(self, gate: "qibo.gates.abstract.Gate") -> "ndarray":
        """Convert a gate to its matrix representation in the computational basis."""
        name = gate.__class__.__name__
        _matrix = getattr(self.matrices, name)
        if name == "I":
            _matrix = _matrix(2 ** len(gate.target_qubits))
        elif name == "Align":
            _matrix = _matrix(0, 2)
        elif callable(_matrix):
            return self.matrix_parametrized(gate)

        return self.cast(_matrix, dtype=_matrix.dtype)  # pylint: disable=E1111

    def matrix_parametrized(self, gate: "qibo.gates.abstract.Gate"):
        """Convert a parametrized gate to its matrix representation in the computational basis."""
        name = gate.__class__.__name__
        _matrix = getattr(self.matrices, name)
        if name == "GeneralizedRBS":
            _matrix = _matrix(
                qubits_in=gate.init_args[0],
                qubits_out=gate.init_args[1],
                theta=gate.init_kwargs["theta"],
                phi=gate.init_kwargs["phi"],
            )
        elif name == "FanOut":
            _matrix = _matrix(*gate.init_args)
        else:
            _matrix = _matrix(*gate.parameters)

        return self.cast(_matrix, dtype=_matrix.dtype)  # pylint: disable=E1111

    def matrix_fused(self, fgate):
        """Fuse matrices of multiple gates."""
        rank = len(fgate.target_qubits)
        matrix = self.identity(2**rank, sparse=True)

        for gate in fgate.gates:
            gmatrix = gate.matrix(self)
            # add controls if controls were instantiated using
            # the ``Gate.controlled_by`` method
            num_controls = len(gate.control_qubits)
            if num_controls > 0:
                gmatrix = self.block_diag(
                    self.identity(2 ** len(gate.qubits) - len(gmatrix)), gmatrix
                )
            # Kronecker product with identity is needed to make the
            # original matrix have shape (2**rank x 2**rank)
            eye = self.identity(2 ** (rank - len(gate.qubits)))
            gmatrix = self.kron(gmatrix, eye)
            # Transpose the new matrix indices so that it targets the
            # target qubits of the original gate
            original_shape = gmatrix.shape
            gmatrix = self.reshape(gmatrix, 2 * rank * (2,))

            qubits = list(gate.qubits)
            indices = qubits + [q for q in fgate.target_qubits if q not in qubits]
            indices = self.argsort(indices)
            transpose_indices = list(indices)
            transpose_indices.extend(indices + rank)
            gmatrix = self.transpose(gmatrix, transpose_indices)
            gmatrix = self.reshape(gmatrix, original_shape)
            # fuse the individual gate matrix to the total ``FusedGate`` matrix
            # we are using sparse matrices to improve perfomances
            matrix = self.csr_matrix(gmatrix) @ matrix

        return self.cast(matrix.toarray(), dtype=matrix.dtype)

    ########################################################################################
    ######## Methods related to the execution and post-processing of measurements   ########
    ########################################################################################

    def aggregate_shots(self, shots) -> "ndarray":
        """Collect shots to a single array."""
        return self.cast(shots, dtype=shots[0].dtype)  # pylint: disable=E1111

    def calculate_frequencies(self, samples):
        """Calculate measurement frequencies from shots."""
        res, counts = self.unique(samples, return_counts=True)
        res = self.to_numpy(res).tolist()
        counts = self.to_numpy(counts).tolist()
        return Counter(dict(zip(res, counts)))

    def calculate_probabilities(
        self, state, qubits, nqubits, density_matrix: bool = False
    ):
        if density_matrix:
            order = tuple(sorted(qubits))
            order += tuple(i for i in range(nqubits) if i not in qubits)
            order = order + tuple(i + nqubits for i in order)
            shape = 2 * (2 ** len(qubits), 2 ** (nqubits - len(qubits)))
            state = self.reshape(state, 2 * nqubits * (2,))
            state = self.reshape(self.transpose(state, order), shape)
            probs = self.abs(self.engine.einsum("abab->a", state))
            probs = self.reshape(probs, len(qubits) * (2,))
        else:
            rtype = self.real(state).dtype
            unmeasured_qubits = tuple(set(list(range(nqubits))) ^ set(qubits))
            state = self.reshape(self.abs(state) ** 2, nqubits * (2,))
            probs = self.cast(state, dtype=rtype)  # pylint: disable=E1111
            probs = self.sum(probs, axis=unmeasured_qubits)

        return self._order_probabilities(probs, qubits, nqubits).ravel()

    def sample_shots(self, probabilities, nshots: int):
        """Sample measurement shots according to a probability distribution."""
        return self.random_choice(
            range(len(probabilities)), size=nshots, p=probabilities
        )

    def sample_frequencies(self, probabilities, nshots: int):
        """Sample measurement frequencies according to a probability distribution."""
        nprobs = probabilities / self.sum(probabilities)
        frequencies = self.zeros(len(nprobs), dtype=self.engine.int64)

        for _ in range(nshots // SHOT_BATCH_SIZE):
            frequencies = self.update_frequencies(frequencies, nprobs, SHOT_BATCH_SIZE)

        frequencies = self.update_frequencies(
            frequencies, nprobs, nshots % SHOT_BATCH_SIZE
        )

        return Counter({i: int(f) for i, f in enumerate(frequencies) if f > 0})

    def samples_to_binary(self, samples, nqubits: int):
        """Convert samples from decimal representation to binary."""
        qrange = (self.engine.arange(nqubits - 1, -1, -1, dtype=self.engine.int32),)
        return self.engine.mod(self.engine.right_shift(samples[:, None], qrange), 2)

    def samples_to_decimal(self, samples, nqubits: int):
        """Convert samples from binary representation to decimal."""
        qrange = self.engine.arange(nqubits - 1, -1, -1, dtype=self.int32)
        qrange = (2**qrange)[:, None]
        samples = self.cast(samples, dtype=self.int32)  # pylint: disable=E1111
        return (samples @ qrange)[:, 0]

    def update_frequencies(self, frequencies, probabilities, nsamples: int):
        samples = self.sample_shots(probabilities, nsamples)
        res, counts = self.unique(samples, return_counts=True)
        frequencies[res] += counts
        return frequencies

    ########################################################################################
    ######## Helper methods for testing                                             ########
    ########################################################################################

    def assert_allclose(
        self, value, target, rtol: float = 1e-7, atol: float = 0.0
    ):  # pragma: no cover
        if isinstance(value, (CircuitResult, QuantumState)):
            value = value.state()
        if isinstance(target, (CircuitResult, QuantumState)):
            target = target.state()

        self.engine.testing.assert_allclose(value, target, rtol=rtol, atol=atol)

    def assert_circuitclose(
        self, circuit, target_circuit, rtol: float = 1e-7, atol: float = 0.0
    ):
        value = self.execute_circuit(circuit).state()
        target = self.execute_circuit(target_circuit).state()

        self.assert_allclose(value, target, rtol=rtol, atol=atol)

    ########################################################################################
    ######## Helper methods                                                         ########
    ########################################################################################

    def _apply_gate_controlled_by(self, gate, state, nqubits: int) -> "ndarray":
        matrix = gate.matrix(self)
        matrix = self.reshape(matrix, 2 * len(gate.target_qubits) * (2,))
        ncontrol = len(gate.control_qubits)
        nactive = nqubits - ncontrol
        order, targets = einsum_utils.control_order(gate, nqubits)
        state = self.transpose(state, order)
        # Apply `einsum` only to the part of the state where all controls
        # are active. This should be `state[-1]`
        state = self.reshape(state, (2**ncontrol,) + nactive * (2,))
        opstring = einsum_utils.apply_gate_string(targets, nactive)
        updates = self.engine.einsum(opstring, state[-1], matrix)
        # Concatenate the updated part of the state `updates` with the
        # part of of the state that remained unaffected `state[:-1]`.
        state = self.engine.concatenate([state[:-1], updates[None]], axis=0)
        state = self.reshape(state, nqubits * (2,))
        # Put qubit indices back to their proper places
        state = self.transpose(state, einsum_utils.reverse_order(order))

        return self.reshape(state, shape=(2**nqubits,))

    def _apply_gate_controlled_by_density_matrix(
        self, gate, state, nqubits: int
    ) -> "ndarray":
        matrix = gate.matrix(self)
        matrix = self.reshape(matrix, 2 * len(gate.target_qubits) * (2,))
        matrixc = self.engine.conj(matrix)
        ncontrol = len(gate.control_qubits)
        nactive = nqubits - ncontrol
        dims_ctrl = 2**ncontrol

        order, targets = einsum_utils.control_order_density_matrix(gate, nqubits)
        state = self.transpose(state, order)
        state = self.reshape(state, 2 * (dims_ctrl,) + 2 * nactive * (2,))

        leftc, rightc = einsum_utils.apply_gate_density_matrix_controlled_string(
            targets, nactive
        )
        state01 = state[: dims_ctrl - 1, dims_ctrl - 1]
        state01 = self.engine.einsum(rightc, state01, matrixc)
        state10 = state[dims_ctrl - 1, : dims_ctrl - 1]
        state10 = self.engine.einsum(leftc, state10, matrix)

        left, right = einsum_utils.apply_gate_density_matrix_string(targets, nactive)
        state11 = state[dims_ctrl - 1, dims_ctrl - 1]
        state11 = self.engine.einsum(right, state11, matrixc)
        state11 = self.engine.einsum(left, state11, matrix)

        state00 = state[range(dims_ctrl - 1)]
        state00 = state00[:, range(dims_ctrl - 1)]
        state01 = self.engine.concatenate([state00, state01[:, None]], axis=1)
        state10 = self.engine.concatenate([state10, state11[None]], axis=0)
        state = self.engine.concatenate([state01, state10[None]], axis=0)
        state = self.reshape(state, 2 * nqubits * (2,))
        state = self.transpose(state, einsum_utils.reverse_order(order))

        return self.reshape(state, 2 * (2**nqubits,))

    def _append_zeros(self, state, qubits, results):
        """Helper function for the ``collapse_state`` method."""
        for q, r in zip(qubits, results):
            state = self.engine.expand_dims(state, q)
            state = (
                self.engine.concatenate([self.zeros_like(state), state], q)
                if r == 1
                else self.engine.concatenate([state, self.zeros_like(state)], q)
            )
        return state

    def _collapse_density_matrix(
        self,
        state,
        qubits: Union[Tuple[int, ...], List[int]],
        shot,
        nqubits: int,
        normalize: bool = True,
    ):  # pragma: no cover
        state = self.cast(state, dtype=state.dtype)  # pylint: disable=E1111
        shape = state.shape
        binshot = list(self.samples_to_binary(shot, len(qubits))[0])
        order = list(qubits) + [qubit + nqubits for qubit in qubits]
        order.extend(qubit for qubit in range(nqubits) if qubit not in qubits)
        order.extend(qubit + nqubits for qubit in range(nqubits) if qubit not in qubits)
        state = self.reshape(state, 2 * nqubits * (2,))
        state = self.transpose(state, order)
        subshape = 2 * (2 ** len(qubits),) + 2 * (nqubits - len(qubits)) * (2,)
        state = self.reshape(state, subshape)[int(shot), int(shot)]
        dims = 2 ** (len(state.shape) // 2)

        if normalize:
            norm = self.trace(self.reshape(state, 2 * (dims,)))
            state = state / norm

        qubits = qubits + [qubit + nqubits for qubit in qubits]
        state = self._append_zeros(state, qubits, 2 * binshot)

        return self.reshape(state, shape)

    def _collapse_statevector(
        self,
        state,
        qubits: Union[Tuple[int, ...], List[int]],
        shot,
        nqubits: int,
        normalize: bool = True,
    ):
        state = self.cast(state, dtype=state.dtype)  # pylint: disable=E1111
        shape = state.shape
        binshot = list(self.samples_to_binary(shot, len(qubits))[0])
        state = self.reshape(state, nqubits * (2,))
        order = list(qubits) + [
            qubit for qubit in range(nqubits) if qubit not in qubits
        ]
        state = self.transpose(state, order)
        subshape = (2 ** len(qubits),) + (nqubits - len(qubits)) * (2,)
        state = self.reshape(state, subshape)[int(shot)]

        if normalize:
            norm = self.sqrt(self.sum(self.engine.abs(state) ** 2))
            state = state / norm

        state = self._append_zeros(state, qubits, binshot)

        return self.engine.reshape(state, shape)

    def _execute_circuit(self, circuit, initial_state=None, nshots=1000):
        nqubits = circuit.nqubits
        density_matrix = circuit.density_matrix

        state = (
            self.zero_state(nqubits, density_matrix=density_matrix)
            if initial_state is None
            else self.cast(initial_state, dtype=initial_state.dtype)
        )

        for gate in circuit.queue:
            state = gate.apply(self, state, nqubits)

        if circuit.has_unitary_channel:
            # here we necessarily have `density_matrix=True`, otherwise
            # execute_circuit_repeated would have been called
            if circuit.measurements:
                circuit._final_state = CircuitResult(
                    state, circuit.measurements, backend=self, nshots=nshots
                )
                return circuit._final_state

            circuit._final_state = QuantumState(state, backend=self)
            return circuit._final_state

        if circuit.measurements:
            circuit._final_state = CircuitResult(
                state, circuit.measurements, backend=self, nshots=nshots
            )
            return circuit._final_state

        circuit._final_state = QuantumState(state, backend=self)

        return circuit._final_state

    def _identity_sparse(self, dims: int, dtype=None):
        from scipy.sparse import eye  # pylint: disable=import-outside-toplevel

        if dtype is None:
            dtype = self.dtype

        return eye(dims, dtype=dtype)

    def _negative_power_singular_matrix(
        self, matrix, power: Union[float, int], precision_singularity: float, dtype=None
    ):
        """Calculate negative power of singular matrix."""
        if dtype is None:
            dtype = self.dtype

        u_matrix, s_matrix, vh_matrix = self.singular_value_decomposition(matrix)
        # cast needed because of different dtypes in `torch`
        s_matrix = self.cast(s_matrix, dtype=dtype)  # pylint: disable=E1111
        s_matrix_inv = self.engine.where(
            self.abs(s_matrix) < precision_singularity, 0.0, s_matrix**power
        )

        return self.inv(vh_matrix) @ self.diag(s_matrix_inv) @ self.inv(u_matrix)

    def _order_probabilities(self, probs, qubits, nqubits):
        """Arrange probabilities according to the given ``qubits`` ordering."""
        unmeasured, reduced = [], {}
        for i in range(nqubits):
            if i in qubits:
                reduced[i] = i - len(unmeasured)
            else:
                unmeasured.append(i)
        return self.transpose(probs, [reduced.get(i) for i in qubits])
