import collections
import os

import numpy as np

from qibo import __version__
from qibo.backends.npmatrices import NumpyMatrices
from qibo.backends.numpy import NumpyBackend
from qibo.config import TF_LOG_LEVEL, log, raise_error


class TensorflowMatrices(NumpyMatrices):
    # Redefine parametrized gate matrices for backpropagation to work

    def __init__(self, dtype):
        super().__init__(dtype)
        import tensorflow as tf  # pylint: disable=import-error
        import tensorflow.experimental.numpy as tnp  # pylint: disable=import-error

        self.tf = tf
        self.np = tnp
        self.np.linalg = tf.linalg

    def _cast(self, x, dtype):
        return self.tf.cast(x, dtype=dtype)

    def Unitary(self, u):
        return self._cast(u, dtype=self.dtype)


class TensorflowBackend(NumpyBackend):
    def __init__(self):
        super().__init__()
        self.name = "tensorflow"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(TF_LOG_LEVEL)

        import tensorflow as tf  # pylint: disable=import-error
        import tensorflow.experimental.numpy as tnp  # pylint: disable=import-error

        if TF_LOG_LEVEL >= 2:
            tf.get_logger().setLevel("ERROR")

        tnp.experimental_enable_numpy_behavior()
        self.tf = tf
        self.np = tnp
        self.np.flatnonzero = np.flatnonzero
        self.np.copy = np.copy

        self.versions = {
            "qibo": __version__,
            "numpy": np.__version__,
            "tensorflow": tf.__version__,
        }

        self.matrices = TensorflowMatrices(self.dtype)

        from tensorflow.python.framework import (  # pylint: disable=E0611,import-error
            errors_impl,
        )

        self.oom_error = errors_impl.ResourceExhaustedError

        cpu_devices = tf.config.list_logical_devices("CPU")
        gpu_devices = tf.config.list_logical_devices("GPU")
        if gpu_devices:  # pragma: no cover
            # CI does not use GPUs
            self.device = gpu_devices[0].name
        elif cpu_devices:
            self.device = cpu_devices[0].name

        self.nthreads = 0

        self.tensor_types = (np.ndarray, tf.Tensor, tf.Variable)

    def set_device(self, device):  # pragma: no cover
        self.device = device

    def set_threads(self, nthreads):
        log.warning(
            "`set_threads` is not supported by the tensorflow "
            "backend. Please use tensorflow's thread setters: "
            "`tf.config.threading.set_inter_op_parallelism_threads` "
            "or `tf.config.threading.set_intra_op_parallelism_threads` "
            "to switch the number of threads."
        )

    def cast(self, x, dtype=None, copy=False):
        if dtype is None:
            dtype = self.dtype
        x = self.tf.cast(x, dtype=dtype)
        if copy:
            return self.tf.identity(x)
        return x

    def is_sparse(self, x):
        return isinstance(x, self.tf.sparse.SparseTensor)

    def to_numpy(self, x):
        return np.array(x)

    def compile(self, func):
        return self.tf.function(func)

    def zero_state(self, nqubits):
        idx = self.tf.constant([[0]], dtype="int32")
        state = self.tf.zeros((2**nqubits,), dtype=self.dtype)
        update = self.tf.constant([1], dtype=self.dtype)
        state = self.tf.tensor_scatter_nd_update(state, idx, update)
        return state

    def zero_density_matrix(self, nqubits):
        idx = self.tf.constant([[0, 0]], dtype="int32")
        state = self.tf.zeros(2 * (2**nqubits,), dtype=self.dtype)
        update = self.tf.constant([1], dtype=self.dtype)
        state = self.tf.tensor_scatter_nd_update(state, idx, update)
        return state

    def matrix(self, gate):
        npmatrix = super().matrix(gate)
        return self.tf.cast(npmatrix, dtype=self.dtype)

    def matrix_parametrized(self, gate):
        npmatrix = super().matrix_parametrized(gate)
        return self.tf.cast(npmatrix, dtype=self.dtype)

    def matrix_fused(self, gate):
        npmatrix = super().matrix_fused(gate)
        return self.tf.cast(npmatrix, dtype=self.dtype)

    def execute_circuit(self, circuit, initial_state=None, nshots=1000):
        with self.tf.device(self.device):
            return super().execute_circuit(circuit, initial_state, nshots)

    def execute_circuit_repeated(self, circuit, nshots, initial_state=None):
        with self.tf.device(self.device):
            return super().execute_circuit_repeated(circuit, nshots, initial_state)

    def sample_shots(self, probabilities, nshots):
        # redefining this because ``tnp.random.choice`` is not available
        logits = self.tf.math.log(probabilities)[self.tf.newaxis]
        samples = self.tf.random.categorical(logits, nshots)[0]
        return samples

    def samples_to_binary(self, samples, nqubits):
        # redefining this because ``tnp.right_shift`` is not available
        qrange = self.np.arange(nqubits - 1, -1, -1, dtype="int32")
        samples = self.tf.cast(samples, dtype="int32")
        samples = self.tf.bitwise.right_shift(samples[:, self.np.newaxis], qrange)
        return self.tf.math.mod(samples, 2)

    def calculate_frequencies(self, samples):
        # redefining this because ``tnp.unique`` is not available
        res, _, counts = self.tf.unique_with_counts(samples, out_idx="int64")
        res, counts = self.np.array(res), self.np.array(counts)
        if res.dtype == "string":
            res = [r.numpy().decode("utf8") for r in res]
        else:
            res = [int(r) for r in res]
        return collections.Counter({k: int(v) for k, v in zip(res, counts)})

    def update_frequencies(self, frequencies, probabilities, nsamples):
        # redefining this because ``tnp.unique`` and tensor update is not available
        samples = self.sample_shots(probabilities, nsamples)
        res, _, counts = self.tf.unique_with_counts(samples, out_idx="int64")
        frequencies = self.tf.tensor_scatter_nd_add(
            frequencies, res[:, self.tf.newaxis], counts
        )
        return frequencies

    def calculate_norm(self, state, order=2):
        state = self.cast(state)
        return self.tf.norm(state, ord=order)

    def calculate_norm_density_matrix(self, state, order="nuc"):
        state = self.cast(state)
        if order == "nuc":
            return self.np.trace(state)
        return self.tf.norm(state, ord=order)

    def calculate_eigenvalues(self, matrix, k: int = 6, hermitian: bool = True):
        if hermitian:
            return self.tf.linalg.eigvalsh(matrix)
        return self.tf.linalg.eigvals(matrix)

    def calculate_eigenvectors(self, matrix, k: int = 6, hermitian: bool = True):
        if hermitian:
            return self.tf.linalg.eigh(matrix)
        return self.tf.linalg.eig(matrix)

    def calculate_matrix_exp(self, a, matrix, eigenvectors=None, eigenvalues=None):
        if eigenvectors is None or self.is_sparse(matrix):
            return self.tf.linalg.expm(-1j * a * matrix)
        return super().calculate_matrix_exp(a, matrix, eigenvectors, eigenvalues)

    def calculate_hamiltonian_matrix_product(self, matrix1, matrix2):
        if self.is_sparse(matrix1) or self.is_sparse(matrix2):
            raise_error(
                NotImplementedError,
                "Multiplication of sparse matrices is not supported with Tensorflow.",
            )
        return super().calculate_hamiltonian_matrix_product(matrix1, matrix2)

    def calculate_hamiltonian_state_product(self, matrix, state):
        rank = len(tuple(state.shape))
        if rank == 1:  # vector
            return self.np.matmul(matrix, state[:, self.np.newaxis])[:, 0]
        elif rank == 2:  # matrix
            return self.np.matmul(matrix, state)
        else:
            raise_error(
                ValueError,
                f"Cannot multiply Hamiltonian with rank-{rank} tensor.",
            )

    def _test_regressions(self, name):
        if name == "test_measurementresult_apply_bitflips":
            return [
                [4, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 2, 1, 1, 4, 0, 0, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 4, 0, 0, 0, 4],
            ]
        elif name == "test_probabilistic_measurement":
            if "GPU" in self.device:  # pragma: no cover
                return {0: 273, 1: 233, 2: 242, 3: 252}
            else:
                return {0: 271, 1: 239, 2: 242, 3: 248}
        elif name == "test_unbalanced_probabilistic_measurement":
            if "GPU" in self.device:  # pragma: no cover
                return {0: 196, 1: 153, 2: 156, 3: 495}
            else:
                return {0: 168, 1: 188, 2: 154, 3: 490}
        elif name == "test_post_measurement_bitflips_on_circuit":
            return [
                {5: 30},
                {5: 12, 7: 6, 4: 6, 1: 5, 6: 1},
                {3: 7, 6: 4, 2: 4, 7: 4, 0: 4, 5: 3, 4: 2, 1: 2},
            ]
