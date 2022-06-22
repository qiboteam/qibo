import os
import collections
import numpy as np
from qibo.backends.numpy import NumpyBackend
from qibo.config import log, raise_error, TF_LOG_LEVEL


class TensorflowBackend(NumpyBackend):

    def __init__(self):
        super().__init__()
        self.name = "tensorflow"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(TF_LOG_LEVEL)
        import tensorflow as tf
        import tensorflow.experimental.numpy as tnp  # pylint: disable=E0401
        tnp.experimental_enable_numpy_behavior()
        self.tf = tf
        self.np = tnp
        
        from tensorflow.python.framework import errors_impl  # pylint: disable=E0611
        self.oom_error = errors_impl.ResourceExhaustedError

        import psutil
        self.nthreads = psutil.cpu_count(logical=True)

        self.tensor_types = (np.ndarray, tf.Tensor, tf.Variable)

    def set_device(self, device):
        # TODO: Implement this
        raise_error(NotImplementedError)

    def set_threads(self, nthreads):
        log.warning("`set_threads` is not supported by the tensorflow "
                    "backend. Please use tensorflow's thread setters: "
                    "`tf.config.threading.set_inter_op_parallelism_threads` "
                    "or `tf.config.threading.set_intra_op_parallelism_threads` "
                    "to switch the number of threads.")

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

    def zero_state(self, nqubits):
        idx = self.tf.constant([[0]], dtype="int32")
        state = self.tf.zeros((2 ** nqubits,), dtype=self.dtype)
        update = self.tf.constant([1], dtype=self.dtype)
        state = self.tf.tensor_scatter_nd_update(state, idx, update)
        return state

    def zero_density_matrix(self, nqubits):
        idx = self.tf.constant([[0, 0]], dtype="int32")
        state = self.tf.zeros(2 * (2 ** nqubits,), dtype=self.dtype)
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

    def sample_shots(self, probabilities, nshots):
        # redefining this because ``tnp.random.choice`` is not available
        logits = self.tf.math.log(probabilities)[self.tf.newaxis]
        samples = self.tf.random.categorical(logits, nshots)[0]
        return samples

    def samples_to_binary(self, samples, nqubits):
        # redefining this because ``tnp.right_shift`` is not available
        qrange = self.np.arange(nqubits - 1, -1, -1, dtype="int64")
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
        frequencies = self.tf.tensor_scatter_nd_add(frequencies, res[:, self.tf.newaxis], counts)
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

    def calculate_matrix_product(self, hamiltonian, o):
        if isinstance(o, hamiltonian.__class__):
            new_matrix = self.np.dot(hamiltonian.matrix, o.matrix)
            return hamiltonian.__class__(hamiltonian.nqubits, new_matrix, backend=self)

        if isinstance(o, self.tensor_types):
            rank = len(tuple(o.shape))
            if rank == 1: # vector
                return self.np.matmul(hamiltonian.matrix, o[:, self.np.newaxis])[:, 0]
            elif rank == 2: # matrix
                return self.np.matmul(hamiltonian.matrix, o)
            else:
                raise_error(ValueError, "Cannot multiply Hamiltonian with "
                                        "rank-{} tensor.".format(rank))

        raise_error(NotImplementedError, "Hamiltonian matmul to {} not "
                                         "implemented.".format(type(o)))

    def test_regressions(self, name):
        if name == "test_measurementresult_apply_bitflips":
            return [
                [4, 0, 0, 1, 0, 2, 2, 4, 4, 0],
                [4, 0, 0, 1, 0, 2, 2, 4, 4, 0],
                [4, 0, 0, 1, 0, 0, 0, 4, 4, 0],
                [4, 0, 0, 0, 0, 0, 0, 4, 4, 0]
            ]
        elif name == "test_probabilistic_measurement": 
            return {0: 271, 1: 239, 2: 242, 3: 248}
        elif name == "test_unbalanced_probabilistic_measurement": 
            return {0: 168, 1: 188, 2: 154, 3: 490}
        elif name == "test_post_measurement_bitflips_on_circuit": 
            return [
                {5: 30}, {5: 16, 7: 10, 6: 2, 3: 1, 4: 1},
                {3: 6, 5: 6, 7: 5, 2: 4, 4: 3, 0: 2, 1: 2, 6: 2}
            ]
        else:
            return None
