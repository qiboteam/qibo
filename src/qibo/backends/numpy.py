"""Module defining the Numpy backend."""

import numpy as np

from qibo import __version__
from qibo.backends.abstract import Backend
from qibo.backends.npmatrices import NumpyMatrices
from qibo.config import raise_error
from qibo.result import CircuitResult, MeasurementOutcomes, QuantumState


class NumpyBackend(Backend):
    def __init__(self):
        super().__init__()
        self.engine = np
        self.name = "numpy"
        self.matrices = NumpyMatrices(self.dtype)
        self.tensor_types = np.ndarray
        self.numeric_types += (
            self.int8,
            self.int32,
            self.int64,
            self.float32,
            self.float64,
            self.complex64,
            self.complex128,
        )
        self.versions[self.name] = self.engine.__version__

    def set_device(self, device):
        if device != "/CPU:0":
            raise_error(
                ValueError, f"Device {device} is not available for {self} backend."
            )

    def set_threads(self, nthreads):
        if nthreads > 1:
            raise_error(ValueError, "``numpy`` does not support more than one thread.")

    def cast(self, x, dtype=None, copy: bool = False):
        if dtype is None:
            dtype = self.dtype

        if isinstance(x, self.tensor_types):
            return x.astype(dtype, copy=copy)

        if self.is_sparse(x):
            return x.astype(dtype, copy=copy)

        return np.asarray(x, dtype=dtype, copy=copy if copy else None)

    def to_numpy(self, array):
        if self.is_sparse(array):
            return array.toarray()
        return array
