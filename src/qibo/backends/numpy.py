"""Module defining the Numpy backend."""

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from scipy.linalg import block_diag, expm, fractional_matrix_power, logm
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse import eye as eye_sparse
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import expm as expm_sparse

from qibo import __version__
from qibo.backends.abstract import Backend
from qibo.backends.npmatrices import NumpyMatrices
from qibo.config import raise_error


class NumpyBackend(Backend):
    def __init__(self):
        super().__init__()
        self.engine = np
        self.matrices = NumpyMatrices(self.dtype)
        self.name = "numpy"
        self.numeric_types += (
            self.int8,
            self.int32,
            self.int64,
            self.float32,
            self.float64,
            self.complex64,
            self.complex128,
        )
        self.tensor_types = (self.engine.ndarray,)
        self.versions[self.name] = self.engine.__version__

    def cast(
        self, array: ArrayLike, dtype: Optional[DTypeLike] = None, copy: bool = False
    ) -> ArrayLike:
        """Cast an object as the array type of the current backend.

        Args:
            array (ArrayLike): Object to cast to array.
            dtype (str or type, optional): data type of ``array`` after casting.
                Options are ``"complex128"``, ``"complex64"``, ``"float64"``,
                or ``"float32"``. If ``None``, defaults to ``Backend.dtype``.
                Defaults to ``None``.
            copy (bool, optional): If ``True`` a copy of the object is created in memory.
                Defaults to ``False``.

        Returns:
            ArrayLike: ``array`` casted to ``dtype``, possibly copied in memory.
        """
        if dtype is None:
            dtype = self.dtype

        if isinstance(array, self.tensor_types):
            return array.astype(dtype, copy=copy)

        if self.is_sparse(array):
            return array.astype(dtype, copy=copy)

        return self.engine.asarray(array, dtype=dtype, copy=copy if copy else None)

    def is_sparse(self, array: ArrayLike) -> bool:
        """Determine if a given array is a sparse tensor.

        Args:
            array (ArrayLike): array to determine the sparsity of.

        Returns:
            bool: ``True`` if ``array`` is sparse, ``False`` otherwise.
        """
        return issparse(array)

    def set_device(self, device: str) -> None:
        """Set simulation device. Works in-place.

        Args:
            device (str): Device index, *e.g.* ``/CPU:0`` for CPU, or ``/GPU:1`` for
                the second GPU in a multi-GPU environment.
        """
        if device != "/CPU:0":
            raise_error(
                ValueError, f"Device {device} is not available for {self} backend."
            )

    def set_threads(self, nthreads: int) -> None:
        """Set number of threads for CPU backend simulations that accept it. Works in-place.

        Args:
            nthreads (int): Number of threads.
        """
        if nthreads > 1:
            raise_error(ValueError, "``numpy`` does not support more than one thread.")

    def to_numpy(self, array: ArrayLike) -> ArrayLike:
        """Convert ``array`` to a ``numpy.ndarray``.

        Args:
            array (ArrayLike): array to be converted to ``numpy.ndarray``.

        Returns:
            ArrayLike: Original array converted to ``numpy.ndarray``.
        """
        return array.toarray() if self.is_sparse(array) else array

    ########################################################################################
    ######## Methods related to array manipulation                                  ########
    ########################################################################################

    def block_diag(self, *arrays: ArrayLike) -> ArrayLike:
        """Create a block diagonal array from provided ``arrays``.

        Args:
            arrays (ArrayLike): input arrays.

        Returns:
            ArrayLike: Array with ``arrays`` on the diagonal of the last two dimensions.
        """
        return block_diag(*arrays)

    def coo_matrix(self, array: ArrayLike, **kwargs) -> ArrayLike:  # pragma: no cover
        return coo_matrix(array, **kwargs)

    def csr_matrix(self, array: ArrayLike, **kwargs) -> ArrayLike:
        """Return the sparse version of ``array`` in compressed sparse row format.

        Args:
            array (ArrayLike): input array.

        Returns:
            ArrayLike: The compressed-sparse-row version of ``array``.
        """
        return csr_matrix(array, **kwargs)

    def eigsh(self, array: ArrayLike, **kwargs) -> Tuple[ArrayLike, ArrayLike]:
        """Compute the eigenvalues and right eigenvectors of a two-dimensional sparse ``array``
        that is assumed to be Hermitian.

        Args:
            array (ArrayLike): input sparse array.
            kwargs (optional): additional options for this fuction.
                For more details, see the corresponding engine's documentation.

        Returns:
            Tuple[ArrayLike, ArrayLike]: Tuple with, respectively, array of eigenvalues and
            array of column-stacked eigenvectors.
        """
        return eigsh(array, **kwargs)

    def expm(self, array: ArrayLike) -> ArrayLike:
        """Compute the matrix exponential of an ``array``.

        Args:
            array (ArrayLike): input array.

        Returns:
            ArrayLike: The resulting matrix exponential.
        """
        func = expm_sparse if self.is_sparse(array) else expm
        return func(array)

    def logm(self, array: ArrayLike, **kwargs) -> ArrayLike:
        """Compute the matrix logarithm of an ``array``.

        Args:
            array (ArrayLike): input array.

        Returns:
            ArrayLike: The resulting matrix logarithm.
        """
        return logm(array, **kwargs)

    ########################################################################################
    ######## Methods related to linear algebra operations                           ########
    ########################################################################################

    def matrix_power(
        self,
        matrix: ArrayLike,
        power: Union[float, int],
        precision_singularity: float = 1e-14,
        dtype: Optional[DTypeLike] = None,
    ) -> ArrayLike:
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

        return fractional_matrix_power(matrix, power)

    ########################################################################################
    ######## Helper methods                                                         ########
    ########################################################################################

    def _identity_sparse(
        self, dims: int, dtype: Optional[DTypeLike] = None, **kwargs
    ) -> ArrayLike:
        if dtype is None:  # pragma: no cover
            dtype = self.dtype

        sparsity_format = kwargs.get("format", "csr")

        return eye_sparse(dims, dtype=dtype, format=sparsity_format, **kwargs)
