from abc import ABC, abstractmethod
from qibo.config import raise_error


class BaseBackend(ABC):

    def __init__(self):
        self.backend = None
        self.name = "base"

    @abstractmethod
    def sum(self, x, axis=None):
        raise_error(NotImplementedError)

    @abstractmethod
    def einsum(self, *args):
        raise_error(NotImplementedError)

    @abstractmethod
    def matmul(self, x, y):
        raise_error(NotImplementedError)


class NumpyBackend(BaseBackend):

    def __init__(self):
        import numpy as np
        self.backend = np
        self.name = "numpy"

    def sum(self, x, axis=None):
        return self.backend.sum(x, axis=axis)

    def einsum(self, *args):
        return self.backend.einsum(*args)

    def matmul(self, x, y):
        return self.backend.matmul(x, y)



class TensorflowBackend(NumpyBackend):

    def __init__(self):
        import tensorflow as tf
        self.backend = tf
        self.name = "tensorflow"

    def sum(self, x, axis=None):
        return self.backend.reduce_sum(x, axis=axis)


function_names = [m for m in dir(BaseBackend) if m[:2] != "__"]

factory = {
    'numpy': NumpyBackend,
    'tensorflow': TensorflowBackend
}
