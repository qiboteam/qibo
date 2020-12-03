import numpy as np
import tensorflow as tf
from qibo.config import DTYPES
from qibo.numpy.matrices import NumpyMatrices


class TensorflowMatrices(NumpyMatrices):

    def cast(self, x: np.ndarray) -> tf.Tensor:
        return tf.convert_to_tensor(x, dtype=DTYPES.get('DTYPECPX'))
