"""Module defining the Numpy backend."""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

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

    def set_device(self, device: str) -> None:
        if device != "/CPU:0":
            raise_error(
                ValueError, f"Device {device} is not available for {self} backend."
            )

    def set_threads(self, nthreads: int) -> None:
        if nthreads > 1:
            raise_error(ValueError, "``numpy`` does not support more than one thread.")

    def cast(
        self, array: ArrayLike, dtype: Optional[DTypeLike] = None, copy: bool = False
    ) -> ArrayLike:
        if dtype is None:
            dtype = self.dtype

        if isinstance(array, self.tensor_types):
            return array.astype(dtype, copy=copy)

        if self.is_sparse(array):
            return array.astype(dtype, copy=copy)

        return self.engine.asarray(array, dtype=dtype, copy=copy if copy else None)

    def to_numpy(self, array: ArrayLike) -> ArrayLike:
        return array.toarray() if self.is_sparse(array) else array
