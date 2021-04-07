import os
from qibo.backends import abstract, numpy
from qibo.config import raise_error, log, LOG_LEVEL


class Optimization:

    def __init__(self):
        import tensorflow as tf
        self.Variable = tf.Variable
        self.GradientTape = tf.GradientTape
        self.optimizers = tf.optimizers


class TensorflowBackend(numpy.NumpyBackend):

    description = "Base class for Tensorflow backends."

    def __init__(self):
        super().__init__()
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(LOG_LEVEL)
        import tensorflow as tf
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

        from qibo.backends import matrices
        self.matrices = matrices.TensorflowMatrices(self.dtypes('DTYPECPX'))

        self.tensor_types = (self.np.ndarray, tf.Tensor, tf.Variable)
        self.native_types = (tf.Tensor, tf.Variable)
        self.Tensor = tf.Tensor
        self.random = tf.random
        self.newaxis = tf.newaxis
        from tensorflow.python.framework import errors_impl # pylint: disable=E0611
        self.oom_error = errors_impl.ResourceExhaustedError
        self.optimization = Optimization()

        # seed to use in the measurement frequency custom op
        self._seed = None
        # seed can be modified using ``K.set_seed``

    def set_device(self, name):
        abstract.AbstractBackend.set_device(self, name)

    def cast(self, x, dtype='DTYPECPX'):
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
        raise_error(NotImplementedError)

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

    def transpose_state(self, pieces, state, nqubits, order): # pragma: no cover
        pieces = self.reshape(self.backend.stack(pieces), nqubits * (2,))
        return self.reshape(self.transpose(pieces, order), state.shape)

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

    def executing_eagerly(self):
        return self.backend.executing_eagerly()

    def set_seed(self, seed):
        self._seed = seed
        self.backend.random.set_seed(seed)


class TensorflowCustomBackend(TensorflowBackend):

    description = "Uses precompiled primitives to apply gates to states. " \
                  "This is the fastest simulation engine."

    def __init__(self):
        from qibo.tensorflow import custom_operators as op
        if not op._custom_operators_loaded: # pragma: no cover
            # CI can compile custom operators so this case is not tested
            raise_error(RuntimeError, "Cannot initialize Tensorflow custom "
                                      "backend if custom operators are not "
                                      "compiled.")

        super().__init__()
        self.name = "custom"
        self.custom_gates = True
        self.custom_einsum = None
        self.op = op
        from qibo.config import get_threads
        self.get_threads = get_threads

    def initial_state(self, nqubits, is_matrix=False):
        return self.op.initial_state(nqubits, self.dtypes('DTYPECPX'),
                                    is_matrix=is_matrix,
                                    omp_num_threads=self.get_threads())

    def transpose_state(self, pieces, state, nqubits, order):
        return self.op.transpose_state(pieces, state, nqubits, order,
                                       self.get_threads())

    def sample_frequencies(self, probs, nshots):
        from qibo.config import SHOT_CUSTOM_OP_THREASHOLD
        if nshots < SHOT_CUSTOM_OP_THREASHOLD:
            return super().sample_frequencies(probs, nshots)
        # Generate random seed using tf
        dtype = self.dtypes('DTYPEINT')
        seed = self.backend.random.uniform(
            shape=tuple(), maxval=int(1e8), dtype=dtype)
        nqubits = int(self.np.log2(tuple(probs.shape)[0]))
        shape = self.cast(2 ** nqubits, dtype='DTYPEINT')
        frequencies = self.zeros(shape, dtype='DTYPEINT')
        frequencies = self.op.measure_frequencies(
            frequencies, probs, nshots, nqubits, seed, self.get_threads())
        return frequencies

    def create_einsum_cache(self, qubits, nqubits, ncontrol=None): # pragma: no cover
        raise_error(NotImplementedError)

    def einsum_call(self, cache, state, matrix): # pragma: no cover
        raise_error(NotImplementedError)

    def create_gate_cache(self, gate):
        cache = self.GateCache()
        qubits = [gate.nqubits - q - 1 for q in gate.control_qubits]
        qubits.extend(gate.nqubits - q - 1 for q in gate.target_qubits)
        cache.qubits_tensor = self.cast(sorted(qubits), "int32")
        if gate.density_matrix:
            cache.target_qubits_dm = [q + gate.nqubits for q in gate.target_qubits]
        return cache

    def state_vector_call(self, gate, state):
        return gate.gate_op(state, gate.cache.qubits_tensor, gate.nqubits,
                            *gate.target_qubits, self.get_threads())

    def state_vector_matrix_call(self, gate, state):
        return gate.gate_op(state, gate.matrix, gate.cache.qubits_tensor, # pylint: disable=E1121
                            gate.nqubits, *gate.target_qubits,
                            self.get_threads())

    def density_matrix_call(self, gate, state):
        state = gate.gate_op(state, gate.cache.qubits_tensor + gate.nqubits,
                             2 * gate.nqubits, *gate.target_qubits,
                             self.get_threads())
        state = gate.gate_op(state, gate.cache.qubits_tensor, 2 * gate.nqubits,
                             *gate.cache.target_qubits_dm, self.get_threads())
        return state

    def density_matrix_matrix_call(self, gate, state):
        state = gate.gate_op(state, gate.matrix, gate.cache.qubits_tensor + gate.nqubits, # pylint: disable=E1121
                             2 * gate.nqubits, *gate.target_qubits,
                             self.get_threads())
        adjmatrix = self.conj(gate.matrix)
        state = gate.gate_op(state, adjmatrix, gate.cache.qubits_tensor,
                             2 * gate.nqubits, *gate.cache.target_qubits_dm,
                             self.get_threads())
        return state

    def state_vector_collapse(self, gate, state, result):
        return gate.gate_op(state, gate.cache.qubits_tensor, result,
                            gate.nqubits, True, self.get_threads())

    def density_matrix_collapse(self, gate, state, result):
        state = gate.gate_op(state, gate.cache.qubits_tensor + gate.nqubits, result,
                             2 * gate.nqubits, False, self.get_threads())
        state = gate.gate_op(state, gate.cache.qubits_tensor, result,
                             2 * gate.nqubits, False, self.get_threads())
        return state / self.trace(state)


class TensorflowDefaultEinsumBackend(TensorflowBackend):
    """Gate application backend that based on default ``einsum``.

    This is the most efficient implementation for GPU, however its
    backpropagation is not working properly for complex numbers.
    The user should switch to :class:`qibo.core.einsum.MatmulEinsum`
    if automatic differentiation is required.
    """

    description = "Uses `tf.einsum` to apply gates to states via matrix " \
                  "multiplication."

    def __init__(self):
        super().__init__()
        from qibo.backends import einsum
        self.name = "tensorflow_defaulteinsum"
        self.custom_gates = False

    def create_einsum_cache(self, qubits, nqubits, ncontrol=None):
        return numpy.NumpyDefaultEinsumBackend.create_einsum_cache(
            self, qubits, nqubits, ncontrol)

    def einsum_call(self, cache, state, matrix):
        return numpy.NumpyDefaultEinsumBackend.einsum_call(
            self, cache, state, matrix)


class TensorflowMatmulEinsumBackend(TensorflowBackend):

    """Gate application backend based on ``matmul``.

    For Tensorflow this is more efficient than ``einsum`` on CPU but slower on GPU.
    The matmul version implemented here is not the most efficient possible.
    The implementation algorithm is the following.

    Assume that we are applying
    a two qubit gate of shape (4, 4) to qubits 0 and 3 of a five qubit state
    vector of shape 5 * (2,). We perform the following steps:

    * Reshape the state to (2, 4, 2, 2)
    * Transpose to (2, 2, 4, 2) to bring the target qubits in the beginning.
    * Reshape to (4, 8).
    * Apply the gate using the matmul (4, 4) x (4, 8).
    * Reshape to the original shape 5 * (2,) and traspose so that the final
      qubit order agrees with the initial.
    """

    description = "Uses `tf.matmul` as well as transpositions and reshapes " \
                  "to apply gates to states via matrix multiplication."

    def __init__(self):
        from qibo.backends import einsum
        super().__init__()
        self.name = "tensorflow_matmuleinsum"
        self.custom_gates = False

    def create_einsum_cache(self, qubits, nqubits, ncontrol=None):
        return numpy.NumpyMatmulEinsumBackend.create_einsum_cache(
            self, qubits, nqubits, ncontrol)

    def einsum_call(self, cache, state, matrix):
        return numpy.NumpyMatmulEinsumBackend.einsum_call(
            self, cache, state, matrix)
