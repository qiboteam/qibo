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

    def RX(self, theta):
        cos = self.np.cos(theta / 2.0) + 0j
        isin = -1j * self.np.sin(theta / 2.0)
        return self.tf.cast([[cos, isin], [isin, cos]], dtype=self.dtype)

    def RY(self, theta):
        cos = self.np.cos(theta / 2.0) + 0j
        sin = self.np.sin(theta / 2.0) + 0j
        return self.tf.cast([[cos, -sin], [sin, cos]], dtype=self.dtype)

    def RZ(self, theta):
        phase = self.np.exp(0.5j * theta)
        return self.tf.cast([[self.np.conj(phase), 0], [0, phase]], dtype=self.dtype)

    def U1(self, theta):
        phase = self.np.exp(1j * theta)
        return self.tf.cast([[1, 0], [0, phase]], dtype=self.dtype)

    def U2(self, phi, lam):
        eplus = self.np.exp(1j * (phi + lam) / 2.0)
        eminus = self.np.exp(1j * (phi - lam) / 2.0)
        return self.tf.cast(
            [[self.np.conj(eplus), -self.np.conj(eminus)], [eminus, eplus]],
            dtype=self.dtype,
        ) / self.np.sqrt(2)

    def U3(self, theta, phi, lam):
        cost = self.np.cos(theta / 2)
        sint = self.np.sin(theta / 2)
        eplus = self.np.exp(1j * (phi + lam) / 2.0)
        eminus = self.np.exp(1j * (phi - lam) / 2.0)
        return self.tf.cast(
            [
                [self.np.conj(eplus) * cost, -self.np.conj(eminus) * sint],
                [eminus * sint, eplus * cost],
            ],
            dtype=self.dtype,
        )

    def CRX(self, theta):
        r = self.RX(theta)
        return self.tf.cast(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, r[0, 0], r[0, 1]],
                [0, 0, r[1, 0], r[1, 1]],
            ],
            dtype=self.dtype,
        )

    def CRY(self, theta):
        r = self.RY(theta)
        return self.tf.cast(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, r[0, 0], r[0, 1]],
                [0, 0, r[1, 0], r[1, 1]],
            ],
            dtype=self.dtype,
        )

    def CRZ(self, theta):
        r = self.RZ(theta)
        return self.tf.cast(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, r[0, 0], r[0, 1]],
                [0, 0, r[1, 0], r[1, 1]],
            ],
            dtype=self.dtype,
        )

    def CU1(self, theta):
        r = self.U1(theta)
        return self.tf.cast(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, r[0, 0], r[0, 1]],
                [0, 0, r[1, 0], r[1, 1]],
            ],
            dtype=self.dtype,
        )

    def CU2(self, phi, lam):
        r = self.U2(phi, lam)
        return self.tf.cast(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, r[0, 0], r[0, 1]],
                [0, 0, r[1, 0], r[1, 1]],
            ],
            dtype=self.dtype,
        )

    def CU3(self, theta, phi, lam):
        r = self.U3(theta, phi, lam)
        return self.tf.cast(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, r[0, 0], r[0, 1]],
                [0, 0, r[1, 0], r[1, 1]],
            ],
            dtype=self.dtype,
        )

    def fSim(self, theta, phi):
        cost = self.np.cos(theta) + 0j
        isint = -1j * self.np.sin(theta)
        phase = self.np.exp(-1j * phi)
        return self.tf.cast(
            [
                [1, 0, 0, 0],
                [0, cost, isint, 0],
                [0, isint, cost, 0],
                [0, 0, 0, phase],
            ],
            dtype=self.dtype,
        )

    def GeneralizedfSim(self, u, phi):
        phase = self.np.exp(-1j * phi)
        return self.tf.cast(
            [
                [1, 0, 0, 0],
                [0, u[0, 0], u[0, 1], 0],
                [0, u[1, 0], u[1, 1], 0],
                [0, 0, 0, phase],
            ],
            dtype=self.dtype,
        )

    def Unitary(self, u):
        return self.tf.cast(u, dtype=self.dtype)


class TensorflowBackend(NumpyBackend):
    def __init__(self):
        super().__init__()
        self.name = "tensorflow"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(TF_LOG_LEVEL)
        import tensorflow as tf  # pylint: disable=import-error
        import tensorflow.experimental.numpy as tnp  # pylint: disable=import-error

        tnp.experimental_enable_numpy_behavior()
        self.tf = tf
        self.np = tnp

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

        import psutil

        self.nthreads = psutil.cpu_count(logical=True)

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

    def issparse(self, x):
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

    def asmatrix(self, gate):
        npmatrix = super().asmatrix(gate)
        return self.tf.cast(npmatrix, dtype=self.dtype)

    def asmatrix_parametrized(self, gate):
        npmatrix = super().asmatrix_parametrized(gate)
        return self.tf.cast(npmatrix, dtype=self.dtype)

    def asmatrix_fused(self, gate):
        npmatrix = super().asmatrix_fused(gate)
        return self.tf.cast(npmatrix, dtype=self.dtype)

    def execute_circuit(
        self, circuit, initial_state=None, nshots=None, return_array=False
    ):
        with self.tf.device(self.device):
            return super().execute_circuit(circuit, initial_state, nshots, return_array)

    def execute_circuit_repeated(self, circuit, initial_state=None, nshots=None):
        with self.tf.device(self.device):
            return super().execute_circuit_repeated(circuit, initial_state, nshots)

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
        return collections.Counter({int(k): int(v) for k, v in zip(res, counts)})

    def update_frequencies(self, frequencies, probabilities, nsamples):
        # redefining this because ``tnp.unique`` and tensor update is not available
        samples = self.sample_shots(probabilities, nsamples)
        res, _, counts = self.tf.unique_with_counts(samples, out_idx="int64")
        frequencies = self.tf.tensor_scatter_nd_add(
            frequencies, res[:, self.tf.newaxis], counts
        )
        return frequencies

    def entanglement_entropy(self, rho):
        # redefining this because ``tnp.linalg`` is not available
        from qibo.config import EIGVAL_CUTOFF

        # Diagonalize
        eigvals = self.np.real(self.tf.linalg.eigvalsh(rho))
        # Treating zero and negative eigenvalues
        masked_eigvals = eigvals[eigvals > EIGVAL_CUTOFF]
        spectrum = -1 * self.np.log(masked_eigvals)
        entropy = self.np.sum(masked_eigvals * spectrum) / self.np.log(2.0)
        return entropy, spectrum

    def calculate_eigenvalues(self, matrix, k=6):
        return self.tf.linalg.eigvalsh(matrix)

    def calculate_eigenvectors(self, matrix, k=6):
        return self.tf.linalg.eigh(matrix)

    def calculate_matrix_exp(self, a, matrix, eigenvectors=None, eigenvalues=None):
        if eigenvectors is None or self.issparse(matrix):
            return self.tf.linalg.expm(-1j * a * matrix)
        else:
            return super().calculate_matrix_exp(a, matrix, eigenvectors, eigenvalues)

    def calculate_hamiltonian_matrix_product(self, matrix1, matrix2):
        if self.issparse(matrix1) or self.issparse(matrix2):
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
                "Cannot multiply Hamiltonian with " "rank-{} tensor.".format(rank),
            )

    def test_regressions(self, name):
        if name == "test_measurementresult_apply_bitflips":
            return [
                [4, 0, 0, 1, 0, 2, 2, 4, 4, 0],
                [4, 0, 0, 1, 0, 2, 2, 4, 4, 0],
                [4, 0, 0, 1, 0, 0, 0, 4, 4, 0],
                [4, 0, 0, 0, 0, 0, 0, 4, 4, 0],
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
                {5: 16, 7: 10, 6: 2, 3: 1, 4: 1},
                {3: 6, 5: 6, 7: 5, 2: 4, 4: 3, 0: 2, 1: 2, 6: 2},
            ]
