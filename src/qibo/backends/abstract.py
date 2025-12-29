"""Module defining the Backend class."""

import math
from collections import Counter
from functools import reduce
from importlib.util import find_spec, module_from_spec
from string import ascii_letters
from typing import List, Optional, Tuple, Union

from numpy.typing import ArrayLike, DTypeLike

from qibo import __version__
from qibo.backends import einsum_utils
from qibo.config import SHOT_BATCH_SIZE, log, raise_error
from qibo.gates.abstract import Gate
from qibo.result import CircuitResult, MeasurementOutcomes, QuantumState


class Backend:  # pylint: disable=R0904
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

        # load the quantum info basic operations
        spec = find_spec("qibo.quantum_info._quantum_info")
        self.qinfo = module_from_spec(spec)
        spec.loader.exec_module(self.qinfo)
        self.qinfo.ENGINE = self

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
            For simulation backends, returns ``None``.
        """
        # needs to be None and not NotImplementedError because of the transpiler
        return None

    @property
    def connectivity(
        self,
    ) -> Optional[List[Tuple[Union[int, str], Union[int, str]]]]:  # pragma: no cover
        """Return available qubit pairs of the backend.

        Returns:
            List[Tuple[int]] or List[Tuple[str]] or None: For hardware backends, return
            available qubit pairs. For simulation backends, returns ``None``.
        """
        # needs to be None and not NotImplementedError because of the transpiler
        return None

    @property
    def natives(self) -> Optional[List[str]]:  # pragma: no cover
        """Return the native gates of the backend.

        Returns:
            List[str] or None: For hardware backends, return the native gates of the backend.
            For the simulation backends, return ``None``.
        """
        # needs to be None and not NotImplementedError because of the transpiler
        return None

    def cast(
        self,
        array: ArrayLike,
        dtype: DTypeLike = None,
        copy: bool = False,  # pylint: disable=unused-argument
    ) -> ArrayLike:  # pragma: no cover
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

    def is_sparse(self, array: ArrayLike) -> bool:  # pragma: no cover
        """Determine if a given array is a sparse tensor."""
        raise_error(NotImplementedError)

    def set_device(self, device: str) -> None:  # pragma: no cover
        """Set simulation device. Works in-place.

        Args:
            device (str): Device index, *e.g.* ``/CPU:0`` for CPU, or ``/GPU:1`` for
                the second GPU in a multi-GPU environment.
        """
        raise_error(NotImplementedError)

    def set_dtype(self, dtype: str) -> None:
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

            if self.matrices is not None:
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

    def to_numpy(self, array: ArrayLike) -> ArrayLike:  # pragma: no cover
        """Cast a given array to numpy."""
        raise_error(NotImplementedError)

    ########################################################################################
    ######## Methods related to data types                                          ########
    ########################################################################################

    @property
    def complex64(self) -> DTypeLike:
        return self.engine.complex64

    @property
    def complex128(self) -> DTypeLike:
        return self.engine.complex128

    @property
    def float32(self) -> DTypeLike:
        return self.engine.float32

    @property
    def float64(self) -> DTypeLike:
        return self.engine.float64

    @property
    def int8(self) -> DTypeLike:
        return self.engine.int8

    @property
    def int16(self) -> DTypeLike:
        return self.engine.int16

    @property
    def int32(self) -> DTypeLike:
        return self.engine.int32

    @property
    def int64(self) -> DTypeLike:
        return self.engine.int64

    @property
    def uint8(self) -> DTypeLike:
        return self.engine.uint8

    ########################################################################################
    ######## Methods related to array manipulation                                  ########
    ########################################################################################

    def abs(self, array: ArrayLike, **kwargs) -> Union[int, float, complex, ArrayLike]:
        return self.engine.abs(array, **kwargs)

    def all(self, array: ArrayLike, **kwargs) -> Union[bool, ArrayLike]:
        return self.engine.all(array, **kwargs)

    def allclose(self, array_1: ArrayLike, array_2: ArrayLike, **kwargs) -> bool:
        return self.engine.allclose(array_1, array_2, **kwargs)

    def angle(self, array: ArrayLike, **kwargs) -> ArrayLike:
        return self.engine.angle(array, **kwargs)

    def any(self, array: ArrayLike, **kwargs) -> Union[ArrayLike, bool]:
        return self.engine.any(array, **kwargs)

    def append(
        self, array: ArrayLike, values: ArrayLike, axis: Optional[int] = None
    ) -> ArrayLike:
        return self.engine.append(array, values, axis)

    def arange(self, *args, **kwargs) -> ArrayLike:
        return self.engine.arange(*args, **kwargs)

    def arccos(self, array: ArrayLike, **kwargs) -> ArrayLike:
        return self.engine.arccos(array, **kwargs)

    def arctan2(self, array_1: ArrayLike, array_2: ArrayLike, **kwargs) -> ArrayLike:
        return self.engine.arctan2(array_1, array_2, **kwargs)

    def argsort(
        self, array: ArrayLike, axis: Optional[int] = None, **kwargs
    ) -> ArrayLike:
        return self.engine.argsort(array, axis, **kwargs)

    def array_equal(self, array_1: ArrayLike, array_2: ArrayLike, **kwargs) -> bool:
        return self.engine.array_equal(array_1, array_2, **kwargs)

    def block(self, arrays: ArrayLike) -> ArrayLike:  # pragma: no cover
        return self.engine.block(arrays)

    def block_diag(self, *arrays: ArrayLike) -> ArrayLike:  # pragma: no cover
        raise_error(NotImplementedError)

    def ceil(self, array: ArrayLike, **kwargs) -> ArrayLike:
        return self.engine.ceil(array, **kwargs)

    def concatenate(self, tup: Tuple[ArrayLike, ...], **kwargs) -> ArrayLike:
        return self.engine.concatenate(tup, **kwargs)

    def conj(self, array: ArrayLike) -> ArrayLike:
        return self.engine.conj(array)

    def copy(self, array: ArrayLike, **kwargs) -> ArrayLike:
        return self.engine.copy(array, **kwargs)

    def cos(self, array: ArrayLike, **kwargs) -> ArrayLike:  # pragma: no cover
        return self.engine.cos(array, **kwargs)

    def count_nonzero(self, array: ArrayLike, **kwargs) -> ArrayLike:
        return self.engine.count_nonzero(array, **kwargs)

    def csr_matrix(self, array: ArrayLike, **kwargs) -> ArrayLike:  # pragma: no cover
        raise_error(NotImplementedError)

    def default_rng(self, seed: Optional[int] = None) -> ArrayLike:
        return self.engine.random.default_rng(seed)

    def delete(self, *args, **kwargs) -> ArrayLike:
        return self.engine.delete(*args, **kwargs)

    def det(self, array: ArrayLike) -> ArrayLike:
        return self.engine.linalg.det(array)

    def diag(self, array: ArrayLike, **kwargs) -> ArrayLike:
        return self.engine.diag(array, **kwargs)

    def dot(self, array_1: ArrayLike, array_2: ArrayLike, **kwargs) -> ArrayLike:
        return self.engine.dot(array_1, array_2, **kwargs)

    def eig(
        self, array: ArrayLike, **kwargs
    ) -> Tuple[ArrayLike, ArrayLike]:  # pragma: no cover
        return self.engine.linalg.eig(array, **kwargs)

    def eigh(self, array: ArrayLike, **kwargs) -> Tuple[ArrayLike, ArrayLike]:
        return self.engine.linalg.eigh(array, **kwargs)

    def eigsh(
        self, array: ArrayLike, **kwargs
    ) -> Tuple[ArrayLike, ArrayLike]:  # pragma: no cover
        raise_error(NotImplementedError)

    def eigvalsh(self, array: ArrayLike, **kwargs) -> ArrayLike:
        return self.engine.linalg.eigvalsh(array, **kwargs)

    def eigvals(self, array: ArrayLike, **kwargs) -> ArrayLike:  # pragma: no cover
        return self.engine.linalg.eigvals(array, **kwargs)

    def einsum(
        self, subscripts: str, *operands: List[ArrayLike], **kwargs
    ) -> ArrayLike:
        return self.engine.einsum(subscripts, *operands, **kwargs)

    def empty(self, shape: Union[int, Tuple[int, ...]], **kwargs) -> ArrayLike:
        return self.engine.empty(shape, **kwargs)

    def exp(self, array: ArrayLike, **kwargs) -> Union[float, complex, ArrayLike]:
        return self.engine.exp(array, **kwargs)

    def expand_dims(
        self, array: ArrayLike, axis: Union[int, Tuple[int, ...]]
    ) -> ArrayLike:
        return self.engine.expand_dims(array, axis)

    def expm(self, array: ArrayLike) -> ArrayLike:  # pragma: no cover
        raise_error(NotImplementedError)

    def flatnonzero(self, array: ArrayLike) -> ArrayLike:
        return self.engine.flatnonzero(array)

    def flip(
        self, array: ArrayLike, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> ArrayLike:
        return self.engine.flip(array, axis=axis)

    def floor(self, array: ArrayLike, **kwargs) -> ArrayLike:  # pragma: no cover
        return self.engine.floor(array, **kwargs)

    def hstack(self, arrays: Tuple[ArrayLike, ...], **kwargs) -> ArrayLike:
        return self.engine.hstack(arrays, **kwargs)

    def identity(
        self, dims: int, dtype: DTypeLike = None, sparse: bool = False, **kwargs
    ) -> ArrayLike:
        if dtype is None:
            dtype = self.dtype

        return (
            self._identity_sparse(dims, dtype, **kwargs)
            if sparse
            else self.engine.eye(dims, dtype=dtype, **kwargs)
        )

    def imag(self, array: ArrayLike) -> Union[int, float, ArrayLike]:
        return self.engine.imag(array)

    def inv(self, array: ArrayLike) -> ArrayLike:
        return self.engine.linalg.inv(array)

    def kron(self, array_1: ArrayLike, array_2: ArrayLike) -> ArrayLike:
        return self.engine.kron(array_1, array_2)

    def log(self, array: ArrayLike, **kwargs) -> ArrayLike:
        return self.engine.log(array, **kwargs)

    def logm(self, array: ArrayLike, **kwargs) -> ArrayLike:  # pragma: no cover
        raise_error(NotImplementedError)

    def log2(self, array: ArrayLike, **kwargs) -> ArrayLike:
        return self.engine.log2(array, **kwargs)

    def log10(self, array: ArrayLike, **kwargs) -> ArrayLike:  # pragma: no cover
        return self.engine.log10(array, **kwargs)

    def matmul(self, array_1: ArrayLike, array_2: ArrayLike, **kwargs) -> ArrayLike:
        return self.engine.matmul(array_1, array_2, **kwargs)

    def matrix_norm(
        self, state: ArrayLike, order: Union[int, float, str] = "nuc", **kwargs
    ) -> Union[float, ArrayLike]:
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

    def mean(self, array: ArrayLike, **kwargs) -> Union[float, complex, ArrayLike]:
        return self.engine.mean(array, **kwargs)

    def nonzero(self, array: ArrayLike) -> ArrayLike:
        return self.engine.nonzero(array)

    def ones(
        self, shape: Union[int, Tuple[int, ...]], dtype: Optional[DTypeLike] = None
    ) -> ArrayLike:
        if dtype is None:
            dtype = self.dtype
        return self.engine.ones(shape, dtype=dtype)

    def outer(self, array_1: ArrayLike, array_2: ArrayLike) -> ArrayLike:
        return self.engine.outer(array_1, array_2)

    def prod(self, array: ArrayLike, **kwargs) -> ArrayLike:
        return self.engine.prod(array, **kwargs)

    def qr(self, array: ArrayLike, **kwargs) -> Tuple[ArrayLike, ...]:
        return self.engine.linalg.qr(array, **kwargs)

    def real(self, array: ArrayLike) -> Union[int, float, ArrayLike]:
        return self.engine.real(array)

    def random_choice(
        self,
        array: ArrayLike,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        replace: bool = True,
        p: Optional[ArrayLike] = None,
        seed=None,
        **kwargs,
    ) -> ArrayLike:
        dtype = kwargs.get("dtype", self.float64)

        if size is None:  # pragma: no cover
            size = 1

        if seed is not None:  # pragma: no cover
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed
            result = local_state.choice(array, size=size, replace=replace, p=p)

            return self.cast(result, dtype=dtype)

        result = self.engine.random.choice(array, size=size, replace=replace, p=p)

        return self.cast(result, dtype=dtype)

    def random_integers(
        self,
        low: int,
        high: Optional[int] = None,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        seed=None,
        **kwargs,
    ) -> ArrayLike:
        dtype = kwargs.get("dtype", self.int64)

        if high is None:
            high = low
            low = 0

        if size is None:
            size = 1

        if seed is not None:  # pragma: no cover
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed

            return self.cast(local_state.integers(low, high, size), dtype=dtype)

        return self.cast(self.engine.random.randint(low, high, size), dtype=dtype)

    def random_normal(
        self,
        mean: Union[float, int],
        stddev: Union[float, int],
        size: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
        seed=None,
        dtype: Optional[DTypeLike] = None,
    ) -> ArrayLike:
        if dtype is None:
            dtype = self.float64

        if seed is not None:  # pragma: no cover
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed

            # local rng usually only has standard normal implemented
            distribution = local_state.standard_normal(size)
            distribution *= stddev
            distribution += mean

            return self.cast(distribution, dtype=dtype)

        return self.cast(self.engine.random.normal(mean, stddev, size), dtype=dtype)

    def random_sample(self, size: int, seed=None, **kwargs) -> ArrayLike:
        dtype = kwargs.get("dtype", self.float64)

        if seed is not None:  # pragma: no cover
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed

            return self.cast(local_state.random(size), dtype=dtype)

        return self.cast(self.engine.random.random(size), dtype=dtype)

    def random_uniform(
        self,
        low: Union[float, int] = 0.0,
        high: Union[float, int] = 1.0,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        seed=None,
        **kwargs,
    ) -> ArrayLike:
        dtype = kwargs.get("dtype", self.float64)

        if size is None:  # pragma: no cover
            size = 1

        if seed is not None:
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed

            return self.cast(local_state.uniform(low, high, size), dtype=dtype)

        return self.cast(self.engine.random.uniform(low, high, size), dtype=dtype)

    def ravel(self, array: ArrayLike, **kwargs) -> ArrayLike:
        return self.engine.ravel(array, **kwargs)

    def repeat(
        self,
        array: ArrayLike,
        repeats: Union[int, List[int], Tuple[int, ...]],
        axis: Optional[int] = None,
    ) -> ArrayLike:
        return self.engine.repeat(array, repeats, axis)

    def reshape(
        self, array: ArrayLike, shape: Union[Tuple[int, ...], List[int]], **kwargs
    ) -> ArrayLike:
        return self.engine.reshape(array, shape, **kwargs)

    def right_shift(self, *args, **kwargs) -> ArrayLike:
        return self.engine.right_shift(*args, **kwargs)

    def round(self, array: ArrayLike, decimals: int = 0, **kwargs) -> ArrayLike:
        return self.engine.round(array, decimals, **kwargs)

    def shuffle(self, array: ArrayLike, **kwargs) -> ArrayLike:
        self.engine.random.shuffle(array, **kwargs)

    def sign(self, array: ArrayLike, **kwargs) -> ArrayLike:  # pragma: no cover
        return self.engine.sign(array, **kwargs)

    def sin(self, array: ArrayLike, **kwargs) -> ArrayLike:  # pragma: no cover
        return self.engine.sin(array, **kwargs)

    def sort(self, array: ArrayLike, **kwargs) -> ArrayLike:
        return self.engine.sort(array, **kwargs)

    def sqrt(self, array: ArrayLike) -> ArrayLike:
        return self.engine.sqrt(array)

    def squeeze(
        self, array: ArrayLike, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> ArrayLike:
        return self.engine.squeeze(array, axis)

    def std(self, array: ArrayLike, **kwargs) -> Union[float, ArrayLike]:
        return self.engine.std(array, **kwargs)

    def sum(
        self, array: ArrayLike, axis: Optional[int] = None, **kwargs
    ) -> Union[int, float, complex, ArrayLike]:
        return self.engine.sum(array, axis=axis, **kwargs)

    def swapaxes(self, array: ArrayLike, axis_1: int, axis_2: int) -> ArrayLike:
        return self.engine.swapaxes(array, axis_1, axis_2)

    def tanh(self, array: ArrayLike, **kwargs) -> ArrayLike:  # pragma: no cover
        return self.engine.tanh(array, **kwargs)

    def tensordot(
        self,
        array_1: ArrayLike,
        array_2: ArrayLike,
        axes: Union[int, Tuple[int, ...]] = 2,
    ) -> ArrayLike:
        return self.engine.tensordot(array_1, array_2, axes=axes)

    def trace(self, array: ArrayLike) -> Union[int, float]:
        return self.engine.trace(array)

    def transpose(
        self, array: ArrayLike, axes: Union[Tuple[int, ...], List[int]] = None
    ) -> ArrayLike:
        return self.engine.transpose(array, axes)

    def tril(self, array: ArrayLike, offset: int = 0) -> ArrayLike:
        return self.engine.tril(array, offset)

    def tril_indices(
        self, row: int, offset: int = 0, col: Optional[int] = None, **kwargs
    ):
        if col is None:
            col = row
        return self.engine.tril_indices(row, offset, col, **kwargs)

    def triu(self, array: ArrayLike, offset: int = 0) -> ArrayLike:
        return self.engine.triu(array, offset)

    def unique(
        self, array: ArrayLike, **kwargs
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
        return self.engine.unique(array, **kwargs)

    def vector_norm(
        self,
        state: ArrayLike,
        order: Union[int, float, str] = 2,
        dtype: Optional[DTypeLike] = None,
        **kwargs,
    ) -> float:
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

        return self.engine.linalg.norm(state, order, **kwargs)

    def vstack(self, arrays: Tuple[ArrayLike, ...], **kwargs) -> ArrayLike:
        return self.engine.vstack(arrays, **kwargs)

    def where(self, *args, **kwargs) -> ArrayLike:
        return self.engine.where(*args, **kwargs)

    def zeros(
        self, shape: Union[int, Tuple[int, ...]], dtype: Optional[DTypeLike] = None
    ) -> ArrayLike:
        if dtype is None:
            dtype = self.dtype
        return self.engine.zeros(shape, dtype=dtype)

    def zeros_like(
        self, array: ArrayLike, dtype: Optional[DTypeLike] = None, **kwargs
    ) -> ArrayLike:
        return self.engine.zeros_like(array, dtype=dtype, **kwargs)

    ########################################################################################
    ######## Methods related to linear algebra operations                           ########
    ########################################################################################

    def eigenvalues(
        self, matrix: ArrayLike, k: int = 6, hermitian: bool = True
    ) -> ArrayLike:
        """Calculate eigenvalues of a matrix."""
        if self.is_sparse(matrix):
            log.warning(
                "Calculating sparse matrix eigenvectors because "
                "sparse modules do not provide ``eigvals`` method."
            )
            return self.eigenvectors(matrix, k=k)[0]

        if hermitian:
            return self.eigvalsh(matrix)

        return self.eigvals(matrix)  # pragma: no cover

    def eigenvectors(
        self, matrix: ArrayLike, k: int = 6, hermitian: bool = True
    ) -> ArrayLike:
        """Calculate eigenvectors of a matrix."""
        if self.is_sparse(matrix):
            if k < matrix.shape[0]:
                return self.eigsh(matrix, k=k, which="SA")

        if hermitian:
            return self.eigh(matrix)

        return self.eig(matrix)  # pragma: no cover

    def jacobian(
        self,
        circuit: "Circuit",  # type: ignore
        parameters: ArrayLike,
        initial_state: Optional[ArrayLike] = None,
        return_complex: bool = True,
    ) -> ArrayLike:  # pragma: no cover
        """Calculate the Jacobian matrix of ``circuit`` with respect to varables ``params``."""
        raise_error(
            NotImplementedError,
            "This method is only implemented in backends that allow automatic differentiation, "
            + "e.g. ``PytorchBackend`` and ``TensorflowBackend``.",
        )

    def matrix_exp(
        self,
        matrix: ArrayLike,
        phase: Union[float, int, complex] = 1,
        eigenvectors: Optional[ArrayLike] = None,
        eigenvalues: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """Calculate the exponential :math:`e^{\\theta \\, A}` of a matrix :math:`A`
        and ``phase`` :math:`\\theta`.

        If the eigenvectors and eigenvalues are given the matrix diagonalization is
        used for exponentiation.
        """
        if eigenvectors is None or self.is_sparse(matrix):
            _matrix = self.expm(phase * matrix)  # pylint: disable=E1111

            return self.cast(
                _matrix, dtype=_matrix.dtype
            )  # for GPU backends on qibojit

        expd = self.exp(phase * eigenvalues)
        ud = self.transpose(self.conj(eigenvectors))

        return (eigenvectors * expd) @ ud

    def matrix_log(
        self,
        matrix: ArrayLike,
        base: Union[float, int] = 2,
        eigenvectors: Optional[ArrayLike] = None,
        eigenvalues: Optional[ArrayLike] = None,
    ) -> ArrayLike:
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
        matrix: ArrayLike,
        power: Union[float, int],
        precision_singularity: float = 1e-14,
        dtype: Optional[DTypeLike] = None,
    ) -> ArrayLike:  # pragma: no cover
        raise_error(NotImplementedError)

    def matrix_sqrt(self, array: ArrayLike) -> ArrayLike:
        """Calculate the square root of ``matrix`` :math:`A`, i.e. :math:`A^{1/2}`.

        .. note::
            For the ``pytorch`` backend, this method relies on a copy of the original tensor.
            This may break the gradient flow. For the GPU backends (i.e. ``cupy`` and
            ``cuquantum``), this method falls back to CPU.
        """
        return self.matrix_power(array, power=0.5)

    def partial_trace(
        self, state: ArrayLike, traced_qubits: Union[Tuple[int, ...], List[int]]
    ) -> ArrayLike:
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

        return self.einsum("abac->bc", state)

    def singular_value_decomposition(self, array: ArrayLike) -> Tuple[ArrayLike, ...]:
        """Calculate the Singular Value Decomposition of ``matrix``."""
        return self.engine.linalg.svd(array)

    ########################################################################################
    ######## Methods related to the creation and manipulation of quantum objects    ########
    ########################################################################################

    def depolarizing_error_density_matrix(
        self, gate: Gate, state: ArrayLike, nqubits: int
    ) -> ArrayLike:
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
            qj = [qj[qubits.index(qub)] for qub in range(len(qubits))]
            order += qj

        identity = self.reshape(self.transpose(identity, order), shape)
        state = (1 - lam) * state + lam * identity

        return state

    def maximally_mixed_state(
        self, nqubits: int, dtype: Optional[DTypeLike] = None
    ) -> ArrayLike:
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

    def minus_state(
        self,
        nqubits: int,
        density_matrix: bool = False,
        dtype: Optional[DTypeLike] = None,
    ):
        if dtype is None:
            dtype = self.dtype

        state = self.cast([1, -1], dtype=dtype)  # pylint: disable=E1111
        state = reduce(self.kron, [state] * nqubits)

        state /= 2 ** (nqubits / 2)

        if density_matrix:
            state = self.outer(state, self.conj(state))

        return state

    def overlap_statevector(
        self, state_1: ArrayLike, state_2: ArrayLike, dtype: Optional[DTypeLike] = None
    ) -> Union[float, complex]:
        """Calculate overlap of two pure quantum states."""
        if dtype is None:
            dtype = self.dtype

        state_1 = self.cast(state_1, dtype=dtype)  # pylint: disable=E1111
        state_2 = self.cast(state_2, dtype=dtype)  # pylint: disable=E1111

        return self.sum(self.conj(state_1) * state_2)

    def plus_state(
        self,
        nqubits: int,
        density_matrix: bool = False,
        dtype: Optional[DTypeLike] = None,
    ) -> ArrayLike:
        """Generate :math:`|+++\\cdots+\\rangle` state vector as an array."""
        if dtype is None:
            dtype = self.dtype

        dims = 2**nqubits
        normalization = dims if density_matrix else math.sqrt(dims)
        shape = 2 * (dims,) if density_matrix else dims

        state = self.ones(shape, dtype=dtype)
        state /= normalization

        return state

    def reset_error_density_matrix(
        self, gate: Gate, state: ArrayLike, nqubits: int  # type: ignore
    ) -> ArrayLike:
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
        self, gate: Gate, state: ArrayLike, nqubits: int
    ) -> ArrayLike:
        """Apply thermal relaxation error to density matrix."""
        state = self.cast(state, dtype=state.dtype)  # pylint: disable=E1111
        shape = state.shape
        state = self.apply_gate(gate, state.ravel(), 2 * nqubits)
        return self.reshape(state, shape)

    def zero_state(
        self,
        nqubits: int,
        density_matrix: bool = False,
        dtype: Optional[DTypeLike] = None,
    ) -> ArrayLike:
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

    def apply_bitflips(
        self, noiseless_samples: ArrayLike, bitflip_probabilities: ArrayLike
    ) -> ArrayLike:
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

    def apply_channel(
        self, channel: "Channel", state: ArrayLike, nqubits: int  # type: ignore
    ) -> ArrayLike:
        """Apply a ``channel`` to quantum ``state``."""

        density_matrix = bool(len(state.shape) == 2)

        if density_matrix:
            state = self.cast(state, dtype=state.dtype)  # pylint: disable=E1111

            new_state = (1 - channel.coefficient_sum) * state
            for coeff, gate in zip(channel.coefficients, channel.gates):
                new_state += coeff * self.apply_gate(gate, state, nqubits)

            return new_state

        probabilities = channel.coefficients + (1 - sum(channel.coefficients),)

        index = int(self.sample_shots(probabilities, 1)[0])
        if index != len(channel.gates):
            gate = channel.gates[index]
            state = self.apply_gate(gate, state, nqubits)

        return state

    def apply_gate(self, gate: Gate, state: ArrayLike, nqubits: int) -> ArrayLike:
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
            state = self.einsum(right, state, matrix_conj)
            state = self.einsum(left, state, matrix)
        else:
            opstring = einsum_utils.apply_gate_string(gate.qubits, nqubits)
            state = self.einsum(opstring, state, matrix)

        shape = (2**nqubits,)
        if density_matrix:
            shape *= 2

        return self.reshape(state, shape)

    def apply_gate_half_density_matrix(
        self, gate: Gate, state: ArrayLike, nqubits: int
    ) -> ArrayLike:
        """Apply a gate to one side of the density matrix."""
        if gate.is_controlled_by:  # pragma: no cover
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
        state = self.einsum(left, state, matrix)

        return self.reshape(state, 2 * (2**nqubits,))

    def calculate_symbolic(
        self,
        state: ArrayLike,
        nqubits: int,
        decimals: int = 5,
        cutoff: float = 1e-10,
        max_terms: int = 20,
    ) -> List[str]:
        # state = self.to_numpy(state)
        density_matrix = bool(len(state.shape) == 2)
        ind_j = self.nonzero(state)
        if density_matrix:
            ind_k = ind_j[1]
        ind_j = ind_j[0]

        terms = []
        if density_matrix:
            for j, k in zip(ind_j, ind_k):
                j, k = int(j), int(k)
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
                j = int(j)
                b = bin(int(j))[2:].zfill(nqubits)
                if self.abs(state[j]) >= cutoff:
                    x = self.round(state[j], decimals=decimals)
                    terms.append(f"{x}|{b}>")
                if len(terms) >= max_terms:
                    terms.append("...")
                    return terms

        return terms

    def collapse_state(
        self,
        state: ArrayLike,
        qubits: Union[Tuple[int, ...], List[int]],
        shot: int,
        nqubits: int,
        normalize: bool = True,
        density_matrix: bool = False,
    ) -> ArrayLike:
        """Collapse quantum state according to measurement shot."""

        if density_matrix:
            return self._collapse_density_matrix(
                state, qubits, shot, nqubits, normalize
            )

        return self._collapse_statevector(state, qubits, shot, nqubits, normalize)

    def execute_circuit(
        self,
        circuit: "Circuit",  # type: ignore
        initial_state: Optional[ArrayLike] = None,
        nshots: int = 1000,
    ) -> Union[CircuitResult, MeasurementOutcomes, QuantumState]:
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
        self,
        circuits: List["Circuit"],  # type: ignore
        initial_states: Optional[ArrayLike] = None,
        nshots: Optional[int] = None,
        processes: Optional[int] = None,
    ) -> List[
        Union[CircuitResult, MeasurementOutcomes, QuantumState]
    ]:  # pragma: no cover
        """Execute multiple :class:`qibo.models.circuit.Circuit` in parallel."""
        from qibo.parallel import (  # pylint: disable=import-outside-toplevel
            parallel_circuits_execution,
        )

        return parallel_circuits_execution(
            circuits, initial_states, nshots, processes, backend=self
        )

    def execute_circuit_repeated(
        self,
        circuit: "Circuit",  # type: ignore
        nshots: int,
        initial_state: Optional[ArrayLike] = None,
    ) -> ArrayLike:  # pragma: no cover
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
            final_states = self.cast(  # pylint: disable=E1111
                final_states, dtype=final_states[0].dtype
            )
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
        self,
        circuit: "Circuit",  # type: ignore
        initial_state: Optional[ArrayLike] = None,
        nshots: Optional[int] = None,
    ) -> Union[CircuitResult, MeasurementOutcomes, QuantumState]:  # pragma: no cover
        """Execute a :class:`qibo.models.circuit.Circuit` using multiple GPUs."""
        raise_error(
            NotImplementedError, f"{self} does not support distributed execution."
        )

    def matrix(self, gate: Gate) -> ArrayLike:
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

    def matrix_parametrized(self, gate: Gate) -> ArrayLike:
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

    def matrix_fused(self, fgate: Gate) -> ArrayLike:
        """Fuse matrices of multiple gates."""
        rank = len(fgate.target_qubits)
        matrix = self.identity(2**rank, sparse=True)

        for gate in fgate.gates:
            gmatrix = gate.matrix(self)
            # add controls if controls were instantiated using
            # the ``Gate.controlled_by`` method
            num_controls = len(gate.control_qubits)
            if num_controls > 0:
                gmatrix = self.block_diag(  # pylint: disable=E1111
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
            indices = self.argsort(
                self.cast(indices, dtype=self.int64)  # pylint: disable=E1111
            )  # required by cupy
            indices = [int(elem) for elem in indices]
            transpose_indices = indices
            transpose_indices.extend([ind + rank for ind in indices])
            gmatrix = self.transpose(gmatrix, transpose_indices)
            gmatrix = self.reshape(gmatrix, original_shape)
            # fuse the individual gate matrix to the total ``FusedGate`` matrix
            # we are using sparse matrices to improve perfomances
            matrix = self.csr_matrix(gmatrix) @ matrix

        return self.cast(matrix.toarray(), dtype=matrix.dtype)

    ########################################################################################
    ######## Methods related to the execution and post-processing of measurements   ########
    ########################################################################################

    def aggregate_shots(self, shots: ArrayLike) -> ArrayLike:
        """Collect shots to a single array."""
        return self.cast(shots, dtype=shots[0].dtype)  # pylint: disable=E1111

    def calculate_frequencies(self, samples: ArrayLike) -> Counter:
        """Calculate measurement frequencies from shots."""
        res, counts = self.unique(samples, return_counts=True)
        res = self.to_numpy(res).tolist()
        counts = self.to_numpy(counts).tolist()
        return Counter(dict(zip(res, counts)))

    def calculate_probabilities(
        self,
        state: ArrayLike,
        qubits: Union[List[int], Tuple[int, ...]],
        nqubits: int,
        density_matrix: bool = False,
    ) -> ArrayLike:
        if density_matrix:
            order = tuple(sorted(qubits))
            order += tuple(qubit for qubit in range(nqubits) if qubit not in qubits)
            order = order + tuple(qubit + nqubits for qubit in order)
            shape = 2 * (2 ** len(qubits), 2 ** (nqubits - len(qubits)))
            state = self.reshape(state, 2 * nqubits * (2,))
            state = self.reshape(self.transpose(state, order), shape)
            probs = self.abs(self.einsum("abab->a", state))
            probs = self.reshape(probs, len(qubits) * (2,))
        else:
            rtype = self.real(state).dtype
            unmeasured_qubits = tuple(set(list(range(nqubits))) ^ set(qubits))
            state = self.reshape(self.abs(state) ** 2, nqubits * (2,))
            probs = self.cast(state, dtype=rtype)  # pylint: disable=E1111
            probs = self.sum(probs, axis=unmeasured_qubits)

        return self._order_probabilities(probs, qubits, nqubits).ravel()

    def sample_frequencies(self, probabilities: ArrayLike, nshots: int) -> Counter:
        """Sample measurement frequencies according to a probability distribution."""
        nprobs = probabilities / self.sum(probabilities)
        frequencies = self.zeros(len(nprobs), dtype=self.int64)

        for _ in range(nshots // SHOT_BATCH_SIZE):
            frequencies = self.update_frequencies(frequencies, nprobs, SHOT_BATCH_SIZE)

        frequencies = self.update_frequencies(
            frequencies, nprobs, nshots % SHOT_BATCH_SIZE
        )

        return Counter({i: int(f) for i, f in enumerate(frequencies) if f > 0})

    def sample_shots(self, probabilities: ArrayLike, nshots: int) -> ArrayLike:
        """Sample measurement shots according to a probability distribution."""
        return self.random_choice(
            self.arange(len(probabilities)),
            size=nshots,
            p=probabilities,
            dtype=self.int64,
        )

    def samples_to_binary(self, samples: ArrayLike, nqubits: int) -> ArrayLike:
        """Convert samples from decimal representation to binary."""
        qrange = self.arange(nqubits - 1, -1, -1, dtype=self.int32)
        return self.right_shift(samples[:, None], qrange) % 2

    def samples_to_decimal(self, samples: ArrayLike, nqubits: int) -> ArrayLike:
        """Convert samples from binary representation to decimal."""
        qrange = self.arange(nqubits - 1, -1, -1, dtype=self.int32)
        qrange = (2**qrange)[:, None]
        samples = self.cast(samples, dtype=self.int32)  # pylint: disable=E1111
        return (samples @ qrange)[:, 0]

    def update_frequencies(
        self, frequencies: ArrayLike, probabilities: ArrayLike, nsamples: int
    ) -> ArrayLike:
        samples = self.sample_shots(probabilities, nsamples)
        res, counts = self.unique(samples, return_counts=True)
        frequencies[res] += counts
        return frequencies

    ########################################################################################
    ######## Methods related to expectation values of Hamiltonians                  ########
    ########################################################################################

    def expectation_value(self, hamiltonian, state, normalize):
        density_matrix = bool(len(state.shape) == 2)

        if density_matrix:
            ev = self.real(
                self.trace(self.cast(hamiltonian @ state))
            )  # pylint: disable=E1111
            if normalize:  # pragma: no cover
                norm = self.real(self.trace(state))
                ev /= norm
            return ev

        statec = self.conj(state)
        hstate = hamiltonian @ state
        ev = self.real(self.sum(statec * hstate))
        if normalize:  # pragma: no cover
            ev /= self.sum(self.abs(state) ** 2)

        return ev

    def exp_value_diagonal_observable_dense_from_samples(
        self,
        circuit: "Circuit",  # type: ignore
        observable: ArrayLike,
        nqubits: int,
        nshots: int,
        qubit_map: Optional[Tuple[int, ...]] = None,
    ) -> float:
        """Compute the expectation value of a dense Hamiltonian diagonal in a defined basis
        starting from the samples (measured in the same basis).

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): the circuit to calculate
                the expectation value from.
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
        diag = self.diag(observable)
        if self.count_nonzero(observable - self.diag(diag)) != 0:
            raise_error(
                NotImplementedError,
                "Observable is not diagonal. Expectation of non-diagonal observables starting "
                + "from samples is currently supported for "
                + "`qibo.hamiltonians.SymbolicHamiltonian` only.",
            )
        diag = self.reshape(diag, nqubits * (2,))
        if qubit_map is None:
            qubit_map = tuple(range(nqubits))
        diag = self.transpose(diag, qubit_map).ravel()
        # select only the elements with non-zero counts
        diag = diag[[int(state, 2) for state in freq.keys()]]
        counts = self.cast(list(freq.values()), dtype=diag.dtype) / sum(freq.values())
        return self.real(self.sum(diag * counts))

    def exp_value_diagonal_observable_symbolic_from_samples(
        self,
        circuit: "Circuit",  # type: ignore
        nqubits: int,
        terms_qubits: List[Tuple[int, ...]],
        terms_coefficients: List[float],
        nshots: int,
        qubit_map: Optional[Union[Tuple[int, ...], List[int]]] = None,
        constant: Union[float, int] = 0.0,
    ) -> float:
        """Compute the expectation value of a symbolic observable diagonal in the
        computational basis, starting from the samples.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): the circuit to calculate
                the expectation value from.
            nqubits (int): number of qubits of the observable.
            terms_qubits (List[Tuple[int, ...]]): the qubits each term of the (diagonal)
                symbolic observable is acting on.
            terms_coefficients (List[float]): the coefficient of each term of the (diagonal)
                symbolic observable.
            nshots (int): how many shots to execute the circuit with.
            qubit_map (Tuple[int, ...]): custom qubit ordering.
            constant (float): the constant term of the observable. Defaults to :math:`0.0`.

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
        counts = self.cast(counts, dtype=self.float64) / sum(counts)
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
        return self.sum(expvals @ counts) + constant

    def exp_value_observable_dense(self, circuit: "Circuit", observable: ArrayLike):  # type: ignore
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
        return self.expectation_value(observable, state, normalize=False)

    def exp_value_observable_symbolic(
        self,
        circuit: "Circuit",  # type: ignore
        terms: List[str],
        term_qubits: List[Tuple[int, ...]],
        term_coefficients: List[float],
        nqubits: int,
    ):
        """Compute the expectation value of a general symbolic observable that is a sum of terms.

        In particular, each term of the observable is contracted with
        the corresponding subspace defined by the qubits it acts on.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): the circuit to calculate
                the expectation value from.
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
        dims = len(state.shape) * nqubits
        shape = dims * (2,)
        state = self.reshape(state, shape)
        # prepare the state indices for the contraction
        if circuit.density_matrix:
            state_indices = [ascii_letters[elem] for elem in range(dims)]
        else:
            state_indices = [ascii_letters[elem] for elem in range(2 * dims)]
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
                expval += self.real(
                    coefficient
                    * self.einsum(
                        f"{obs_string},{new_string}->",
                        *matrices,
                        state,
                    )
                )
            else:
                obs_indices = [
                    state_dag_indices[qubit] + state_indices[qubit] for qubit in qubits
                ]
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
                expval += self.real(
                    coefficient
                    * self.einsum(
                        f"{state_dag_string},{obs_string},{new_string}->",
                        self.conj(state),
                        *matrices,
                        state,
                    )
                )
        return expval

    def exp_value_observable_symbolic_from_samples(
        self,
        circuit,
        diagonal_terms_coefficients: List[List[float]],
        diagonal_terms_observables: List[List[str]],
        diagonal_terms_qubits: List[List[Tuple[int, ...]]],
        nqubits: int,
        constant: float,
        nshots: int,
    ) -> float:
        """Compute the expectation value of a general symbolic observable defined by groups
        of terms that can be diagonalized simultaneously, starting from the samples.

        Args:
            circuit (Circuit): the circuit to calculate the expectation value from.
            diagonal_terms_coefficients (List[float]): the coefficients of each term of the
                (diagonal) symbolic observable.
            diagonal_terms_observables (List[List[str]]): the lists of strings defining the
                observables for each group of terms, e.g. ``[['IXZ', 'YII'], ['IYZ', 'XIZ']]``.
            diagonal_terms_qubits (List[Tuple[int, ...]]): the qubits each term of the groups
                is acting on, e.g. ``[[(0,1,2), (1,3)], [(2,1,3), (2,4)]]``.
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
            expval += self.exp_value_diagonal_observable_symbolic_from_samples(
                circ,
                nqubits,
                terms_qubits,
                terms_coefficients,
                nshots,
                qmap,
            )
        return constant + expval

    ########################################################################################
    ######## Methods for testing                                                    ########
    ########################################################################################

    def assert_allclose(
        self,
        value: Union[ArrayLike, CircuitResult, QuantumState],
        target: Union[ArrayLike, CircuitResult, QuantumState],
        rtol: float = 1e-7,
        atol: float = 0.0,
    ) -> None:
        if isinstance(value, (CircuitResult, QuantumState)):
            value = value.state()
        if isinstance(target, (CircuitResult, QuantumState)):
            target = target.state()

        self.engine.testing.assert_allclose(value, target, rtol=rtol, atol=atol)

    def assert_circuitclose(
        self,
        circuit: "Circuit",  # type: ignore
        target_circuit: "Circuit",  # type: ignore
        rtol: float = 1e-7,
        atol: float = 0.0,
    ) -> None:
        value = self.execute_circuit(circuit).state()
        target = self.execute_circuit(target_circuit).state()

        self.assert_allclose(value, target, rtol=rtol, atol=atol)

    ########################################################################################
    ######## Helper methods                                                         ########
    ########################################################################################

    def _apply_gate_controlled_by(
        self, gate: Gate, state: ArrayLike, nqubits: int
    ) -> ArrayLike:
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
        updates = self.einsum(opstring, state[-1], matrix)
        # Concatenate the updated part of the state `updates` with the
        # part of of the state that remained unaffected `state[:-1]`.
        state = self.concatenate([state[:-1], updates[None]], axis=0)
        state = self.reshape(state, nqubits * (2,))
        # Put qubit indices back to their proper places
        state = self.transpose(state, einsum_utils.reverse_order(order))

        return self.reshape(state, shape=(2**nqubits,))

    def _apply_gate_controlled_by_density_matrix(
        self, gate: Gate, state: ArrayLike, nqubits: int
    ) -> ArrayLike:
        matrix = gate.matrix(self)
        matrix = self.reshape(matrix, 2 * len(gate.target_qubits) * (2,))
        matrixc = self.conj(matrix)
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
        state01 = self.einsum(rightc, state01, matrixc)
        state10 = state[dims_ctrl - 1, : dims_ctrl - 1]
        state10 = self.einsum(leftc, state10, matrix)

        left, right = einsum_utils.apply_gate_density_matrix_string(targets, nactive)
        state11 = state[dims_ctrl - 1, dims_ctrl - 1]
        state11 = self.einsum(right, state11, matrixc)
        state11 = self.einsum(left, state11, matrix)

        state00 = state[range(dims_ctrl - 1)]
        state00 = state00[:, range(dims_ctrl - 1)]
        state01 = self.concatenate([state00, state01[:, None]], axis=1)
        state10 = self.concatenate([state10, state11[None]], axis=0)
        state = self.concatenate([state01, state10[None]], axis=0)
        state = self.reshape(state, 2 * nqubits * (2,))
        state = self.transpose(state, einsum_utils.reverse_order(order))

        return self.reshape(state, 2 * (2**nqubits,))

    def _append_zeros(
        self, state: ArrayLike, qubits: Union[List[int], Tuple[int, ...]], results
    ) -> ArrayLike:
        """Helper function for the ``collapse_state`` method."""
        for q, r in zip(qubits, results):
            state = self.expand_dims(state, q)
            state = (
                self.concatenate([self.zeros_like(state), state], axis=q)
                if r == 1
                else self.concatenate([state, self.zeros_like(state)], axis=q)
            )
        return state

    def _collapse_density_matrix(
        self,
        state: ArrayLike,
        qubits: Union[Tuple[int, ...], List[int]],
        shot: ArrayLike,
        nqubits: int,
        normalize: bool = True,
    ) -> ArrayLike:
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
        state: ArrayLike,
        qubits: Union[Tuple[int, ...], List[int]],
        shot: ArrayLike,
        nqubits: int,
        normalize: bool = True,
    ) -> ArrayLike:
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
            norm = self.sqrt(self.sum(self.abs(state) ** 2))
            state = state / norm

        state = self._append_zeros(state, qubits, binshot)

        return self.reshape(state, shape)

    def _execute_circuit(
        self,
        circuit: "Circuit",  # type: ignore
        initial_state: Optional[ArrayLike] = None,
        nshots: int = 1000,
    ) -> Union[CircuitResult, QuantumState]:
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

    def _identity_sparse(
        self, dims: int, dtype: Optional[DTypeLike] = None, **kwargs
    ) -> ArrayLike:  # pragma: no cover
        raise_error(NotImplementedError)

    def _negative_power_singular_matrix(
        self,
        matrix: ArrayLike,
        power: Union[float, int],
        precision_singularity: float,
        dtype: Optional[DTypeLike] = None,
    ) -> ArrayLike:
        """Calculate negative power of singular matrix."""
        if dtype is None:  # pragma: no cover
            dtype = self.dtype

        u_matrix, s_matrix, vh_matrix = self.singular_value_decomposition(matrix)
        # cast needed because of different dtypes in `torch`
        s_matrix = self.cast(s_matrix, dtype=dtype)  # pylint: disable=E1111
        s_matrix_inv = self.where(
            self.abs(s_matrix) < precision_singularity, 0.0, s_matrix**power
        )

        return self.inv(vh_matrix) @ self.diag(s_matrix_inv) @ self.inv(u_matrix)

    def _order_probabilities(
        self, probs: ArrayLike, qubits: Union[List[int], Tuple[int, ...]], nqubits: int
    ) -> ArrayLike:
        """Arrange probabilities according to the given ``qubits`` ordering."""
        unmeasured, reduced = [], {}
        for qubit in range(nqubits):
            if qubit in qubits:
                reduced[qubit] = qubit - len(unmeasured)
            else:
                unmeasured.append(qubit)
        return self.transpose(probs, [reduced.get(qubit) for qubit in qubits])
