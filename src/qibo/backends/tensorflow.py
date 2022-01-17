import os
from qibo.backends.abstract import AbstractBackend, AbstractCustomOperators
from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error, log, TF_LOG_LEVEL


class Optimization:

    def __init__(self):
        import tensorflow as tf  # pylint: disable=E0401
        self.Variable = tf.Variable
        self.GradientTape = tf.GradientTape
        self.optimizers = tf.optimizers


class TensorflowBackend(NumpyBackend):

    description = "Uses `tf.einsum` to apply gates to states via matrix " \
                  "multiplication."

    TEST_REGRESSIONS_CPU = {
        "test_measurementresult_apply_bitflips": [
            [4, 0, 0, 1, 0, 2, 2, 4, 4, 0],
            [4, 0, 0, 1, 0, 2, 2, 4, 4, 0],
            [4, 0, 0, 1, 0, 0, 0, 4, 4, 0],
            [4, 0, 0, 0, 0, 0, 0, 4, 4, 0]
        ],
        "test_probabilistic_measurement": {0: 271, 1: 239, 2: 242, 3: 248},
        "test_unbalanced_probabilistic_measurement": {0: 168, 1: 188, 2: 154, 3: 490},
        "test_post_measurement_bitflips_on_circuit": [
                {5: 30}, {5: 16, 7: 10, 6: 2, 3: 1, 4: 1},
                {3: 6, 5: 6, 7: 5, 2: 4, 4: 3, 0: 2, 1: 2, 6: 2}
            ],
    }
    TEST_REGRESSIONS_GPU = {
        "test_measurementresult_apply_bitflips": [
            [4, 0, 0, 1, 0, 2, 2, 4, 4, 0],
            [4, 0, 0, 1, 0, 2, 2, 4, 4, 0],
            [4, 0, 0, 1, 0, 0, 0, 4, 4, 0],
            [4, 0, 0, 0, 0, 0, 0, 4, 4, 0]
        ],
        "test_probabilistic_measurement": {0: 273, 1: 233, 2: 242, 3: 252},
        "test_unbalanced_probabilistic_measurement": {0: 196, 1: 153, 2: 156, 3: 495},
        "test_post_measurement_bitflips_on_circuit": [
                {5: 30}, {5: 16, 7: 10, 6: 2, 3: 1, 4: 1},
                {3: 6, 5: 6, 7: 5, 2: 4, 4: 3, 0: 2, 1: 2, 6: 2}
            ],
    }

    def __init__(self):
        super().__init__()
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(TF_LOG_LEVEL)
        import tensorflow as tf  # pylint: disable=E0401
        self.backend = tf
        self.name = "tensorflow"

        self.cpu_devices = tf.config.list_logical_devices("CPU")
        self.gpu_devices = tf.config.list_logical_devices("GPU")
        if self.gpu_devices: # pragma: no cover
            # CI does not use GPUs
            self.default_device = self.gpu_devices[0].name
        elif self.cpu_devices:
            self.default_device = self.cpu_devices[0].name
        else: # pragma: no cover
            # case not tested by GitHub workflows because it requires no device
            raise_error(RuntimeError, "Unable to find Tensorflow devices.")

        self.tensor_types = (self.np.ndarray, tf.Tensor, tf.Variable)
        self.native_types = (tf.Tensor, tf.Variable)
        self.Tensor = tf.Tensor
        self.random = tf.random
        self.newaxis = tf.newaxis
        from tensorflow.python.framework import errors_impl  # pylint: disable=E0611,E0401
        self.oom_error = errors_impl.ResourceExhaustedError
        self.optimization = Optimization()

        # seed to use in the measurement frequency custom op
        self._seed = None
        # seed can be modified using ``K.set_seed``

        self.supports_gradients = True

    def test_regressions(self, name):
        if "GPU" in self.default_device:  # pragma: no cover
            # Ci does not use GPUs
            return self.TEST_REGRESSIONS_GPU.get(name)
        else:
            return self.TEST_REGRESSIONS_CPU.get(name)

    def set_device(self, name):
        AbstractBackend.set_device(self, name)

    def set_threads(self, nthreads):
        log.warning("`set_threads` is not supported by the tensorflow "
                    "backend. Please use tensorflow's thread setters: "
                    "`tf.config.threading.set_inter_op_parallelism_threads` "
                    "or `tf.config.threading.set_intra_op_parallelism_threads` "
                    "to switch the number of threads.")
        AbstractBackend.set_threads(self, nthreads)

    def to_numpy(self, x):
        if isinstance(x, self.np.ndarray):
            return x
        return x.numpy()

    def to_complex(self, re, img):
        return self.backend.complex(re, img)

    def cast(self, x, dtype='DTYPECPX'):
        print(dtype)
        print(str(dtype))
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        if isinstance(x, self.np.ndarray):
            dtypestr = dtype.__repr__().split(".")[1]
            x = x.astype(getattr(self.np, dtypestr))
        return self.backend.cast(x, dtype=dtype)

    def diag(self, x, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.cast(self.backend.linalg.diag(x), dtype=dtype)

    def concatenate(self, x, axis=None):
        return self.backend.concat(x, axis=axis)

    def copy(self, x):
        return x + self.backend.zeros_like(x)

    def range(self, start, finish, step, dtype=None):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.range(start, finish, step, dtype=dtype)

    def real(self, x):
        return self.backend.math.real(x)

    def imag(self, x):
        return self.backend.math.imag(x)

    def conj(self, x):
        return self.backend.math.conj(x)

    def mod(self, x, y):
        return self.backend.math.mod(x, y)

    def right_shift(self, x, y):
        return self.backend.bitwise.right_shift(x, y)

    def pow(self, base, exponent):
        return self.backend.math.pow(base, exponent)

    def square(self, x):
        return self.backend.square(x)

    def sqrt(self, x):
        return self.backend.math.sqrt(x)

    def log(self, x):
        return self.backend.math.log(x)

    def trace(self, x):
        return self.backend.linalg.trace(x)

    def expm(self, x):
        return self.backend.linalg.expm(x)

    def sum(self, x, axis=None):
        return self.backend.reduce_sum(x, axis=axis)

    def outer(self, x, y):
        return self.tensordot(x, y, axes=0)

    def kron(self, x, y):
        dim = int(x.shape[0]) * int(y.shape[0])
        z = self.transpose(self.outer(x, y), axes=[0, 2, 1, 3])
        return self.reshape(z, (dim, dim))

    def inv(self, x):
        raise_error(NotImplementedError)

    def unique(self, x, return_counts=False):
        if return_counts:
            res, _, counts = self.backend.unique_with_counts(
                x, out_idx=self.dtypes('DTYPEINT'))
            return res, counts
        res, _  = self.backend.unique(x, out_idx=self.dtypes('DTYPEINT'))
        return res

    def gather(self, x, indices=None, condition=None, axis=0):
        if indices is not None:
            return self.backend.gather(x, indices, axis=axis)

        if condition is None:
            raise_error(ValueError, "Gather call is missing indices and "
                                    "condition.")
        indices = self.backend.where(condition)
        return self.backend.gather(x, indices, axis=axis)[:, 0]

    def gather_nd(self, x, indices):
        return self.backend.gather_nd(x, indices)

    def initial_state(self, nqubits, is_matrix=False): # pragma: no cover
        dim = 1 + is_matrix
        shape = dim * (2 ** nqubits,)
        idx = self.backend.constant([dim * [0]], dtype=self.dtypes('DTYPEINT'))
        state = self.backend.zeros(shape, dtype=self.dtypes('DTYPECPX'))
        update = self.backend.constant([1], dtype=self.dtypes('DTYPECPX'))
        state = self.backend.tensor_scatter_nd_update(state, idx, update)
        return state

    def random_uniform(self, shape, dtype='DTYPE'):
        return self.backend.random.uniform(shape, dtype=self.dtypes(dtype))

    def sample_shots(self, probs, nshots):
        from qibo.config import SHOT_BATCH_SIZE
        logits = self.log(probs)[self.newaxis]
        samples = [self.random.categorical(
            logits, SHOT_BATCH_SIZE, dtype=self.dtypes('DTYPEINT'))[0]
            for _ in range(nshots // SHOT_BATCH_SIZE)]
        samples.append(self.random.categorical(
                logits, nshots % SHOT_BATCH_SIZE,
                dtype=self.dtypes('DTYPEINT'))[0])
        return self.concatenate(samples, axis=0)

    def sample_frequencies(self, probs, nshots):
        logits = self.log(probs)[self.newaxis]
        samples = self.random.categorical(logits, nshots, dtype=self.dtypes('DTYPEINT'))[0]
        res, counts = self.unique(samples, return_counts=True)
        frequencies = self.zeros(int(probs.shape[0]), dtype=self.dtypes('DTYPEINT'))
        frequencies = self.backend.tensor_scatter_nd_add(frequencies, res[:, self.newaxis], counts)
        return frequencies

    def compile(self, func):
        return self.backend.function(func)

    def device(self, device_name):
        return self.backend.device(device_name)

    def on_cpu(self):
        return self.device(self.cpu_devices[0])

    def cpu_tensor(self, x, dtype=None):
        if dtype is None:
            dtype = x.dtype
        return self.backend.Variable(x, dtype=dtype)

    def cpu_cast(self, x, dtype='DTYPECPX'):
        dtype = self._dtypes.get(dtype)
        with self.on_cpu():
            return self.cast(x, dtype=dtype)

    def cpu_assign(self, state, i, piece):
        state.pieces[i].assign(piece)

    def executing_eagerly(self):
        return self.backend.executing_eagerly()

    def set_seed(self, seed):
        self._seed = seed
        self.backend.random.set_seed(seed)
