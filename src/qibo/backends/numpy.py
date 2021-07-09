from qibo.abstractions.states import AbstractState
from qibo.backends import abstract, einsum_utils
from qibo.config import raise_error, log


class NumpyBackend(abstract.AbstractBackend):

    description = "Uses `np.einsum` to apply gates to states via matrix " \
                  "multiplication."

    def __init__(self):
        super().__init__()
        import numpy as np
        self.backend = np
        self.name = "numpy"
        self.np = np

        self.numeric_types = (np.int, np.float, np.complex, np.int32,
                              np.int64, np.float32, np.float64,
                              np.complex64, np.complex128)
        self.tensor_types = (np.ndarray,)
        self.native_types = (np.ndarray,)
        self.Tensor = np.ndarray
        self.random = np.random
        self.newaxis = np.newaxis
        self.oom_error = MemoryError
        self.optimization = None

    def set_device(self, name):
        log.warning("Numpy does not support device placement. "
                    "Aborting device change.")

    def set_threads(self, nthreads):
        log.warning("Numpy backend supports only single-thread execution. "
                    "Cannot change the number of threads.")
        abstract.AbstractBackend.set_threads(self, nthreads)

    def to_numpy(self, x):
        return x

    def to_complex(self, re, img): # pragma: no cover
        return re + 1j * img

    def cast(self, x, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        if isinstance(x, self.backend.ndarray):
            return x.astype(dtype)
        return self.backend.array(x, dtype=dtype)

    def diag(self, x, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.diag(x).astype(dtype)

    def reshape(self, x, shape):
        return self.backend.reshape(x, shape)

    def stack(self, x, axis=0):
        return self.backend.stack(x, axis=axis)

    def concatenate(self, x, axis=0):
        return self.backend.concatenate(x, axis=axis)

    def expand_dims(self, x, axis):
        return self.backend.expand_dims(x, axis)

    def copy(self, x):
        return self.backend.copy(x)

    def range(self, start, finish, step, dtype=None):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.arange(start, finish, step, dtype=dtype)

    def eye(self, shape, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.eye(shape, dtype=dtype)

    def zeros(self, shape, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.ones(shape, dtype=dtype)

    def zeros_like(self, x):
        return self.backend.zeros_like(x)

    def ones_like(self, x):
        return self.backend.ones_like(x)

    def real(self, x):
        return self.backend.real(x)

    def imag(self, x):
        return self.backend.imag(x)

    def conj(self, x):
        return self.backend.conj(x)

    def mod(self, x, y):
        return self.backend.mod(x, y)

    def right_shift(self, x, y):
        return self.backend.right_shift(x, y)

    def exp(self, x):
        return self.backend.exp(x)

    def sin(self, x):
        return self.backend.sin(x)

    def cos(self, x):
        return self.backend.cos(x)

    def pow(self, base, exponent):
        return base ** exponent

    def square(self, x):
        return x ** 2

    def sqrt(self, x):
        return self.backend.sqrt(x)

    def log(self, x):
        return self.backend.log(x)

    def abs(self, x):
        return self.backend.abs(x)

    def trace(self, x):
        return self.backend.trace(x)

    def expm(self, x):
        from scipy import linalg
        return linalg.expm(x)

    def sum(self, x, axis=None):
        return self.backend.sum(x, axis=axis)

    def matmul(self, x, y):
        return self.backend.matmul(x, y)

    def outer(self, x, y):
        return self.backend.outer(x, y)

    def kron(self, x, y):
        return self.backend.kron(x, y)

    def einsum(self, *args):
        return self.backend.einsum(*args)

    def tensordot(self, x, y, axes=None):
        return self.backend.tensordot(x, y, axes=axes)

    def transpose(self, x, axes=None):
        return self.backend.transpose(x, axes)

    def inv(self, x):
        return self.backend.linalg.inv(x)

    def eigh(self, x):
        return self.backend.linalg.eigh(x)

    def eigvalsh(self, x):
        return self.backend.linalg.eigvalsh(x)

    def less(self, x, y):
        return self.backend.less(x, y)

    def array_equal(self, x, y):
        return self.np.array_equal(x, y)

    def unique(self, x, return_counts=False):
        # Uses numpy backend always (even on Tensorflow)
        return self.np.unique(x, return_counts=return_counts)

    def squeeze(self, x, axis=None):
        return self.backend.squeeze(x, axis=axis)

    def gather(self, x, indices=None, condition=None, axis=0):
        if indices is None:
            if condition is None:
                raise_error(ValueError, "Gather call requires either indices "
                                        "or condition.")
            indices = condition
        if axis < 0:
            axis += len(x.shape)
        idx = axis * (slice(None),) + (indices,)
        return x[idx]

    def gather_nd(self, x, indices):
        return x[tuple(indices)]

    def initial_state(self, nqubits, is_matrix=False):
        if is_matrix:
            state = self.zeros(2 * (2 ** nqubits,))
            state[0, 0] = 1
        else:
            state = self.zeros((2 ** nqubits,))
            state[0] = 1
        return state

    def transpose_state(self, pieces, state, nqubits, order): # pragma: no cover
        raise_error(NotImplementedError)

    def random_uniform(self, shape, dtype='DTYPE'):
        return self.backend.random.random(shape).astype(self.dtypes(dtype))

    def sample_shots(self, probs, nshots):
        return self.random.choice(range(len(probs)), size=nshots, p=probs)

    def sample_frequencies(self, probs, nshots):
        from qibo.config import SHOT_BATCH_SIZE
        def update_frequencies(nsamples, frequencies):
            samples = self.random.choice(range(len(probs)), size=nsamples, p=probs)
            res, counts = self.backend.unique(samples, return_counts=True)
            frequencies[res] += counts
            return frequencies

        frequencies = self.zeros(int(probs.shape[0]), dtype=self.dtypes('DTYPEINT'))
        for _ in range(nshots // SHOT_BATCH_SIZE):
            frequencies = update_frequencies(SHOT_BATCH_SIZE, frequencies)
        frequencies = update_frequencies(nshots % SHOT_BATCH_SIZE, frequencies)
        return frequencies

    def compile(self, func):
        return func

    def device(self, device_name):
        class DummyModule:

            def __enter__(self, *args):
                pass

            def __exit__(self, *args):
                pass

        return DummyModule()

    def set_seed(self, seed):
        self.backend.random.seed(seed)

    def create_einsum_cache(self, qubits, nqubits, ncontrol=None):
        return einsum_utils.EinsumCache(qubits, nqubits, ncontrol)

    def einsum_call(self, cache, state, matrix):
        return self.einsum(cache, state, matrix)

    class GateCache:
        pass

    def create_gate_cache(self, gate):
        cache = self.GateCache()
        s = 1 + gate.density_matrix
        cache.tensor_shape = self.cast(s * gate.nqubits * (2,), dtype='DTYPEINT')
        cache.flat_shape = self.cast(s * (2 ** gate.nqubits,), dtype='DTYPEINT')
        if gate.is_controlled_by:
            cache.control_cache = einsum_utils.ControlCache(gate)
            nactive = gate.nqubits - len(gate.control_qubits)
            targets = cache.control_cache.targets
            ncontrol = len(gate.control_qubits)
            cache.calculation_cache = self.create_einsum_cache(
                targets, nactive, ncontrol)
        else:
            cache.calculation_cache = self.create_einsum_cache(gate.qubits, gate.nqubits)
        return cache

    def state_vector_call(self, gate, state):
        state = self.reshape(state, gate.cache.tensor_shape)
        if gate.is_controlled_by:
            ncontrol = len(gate.control_qubits)
            nactive = gate.nqubits - ncontrol
            state = self.transpose(state, gate.cache.control_cache.order(False))
            # Apply `einsum` only to the part of the state where all controls
            # are active. This should be `state[-1]`
            state = self.reshape(state, (2 ** ncontrol,) + nactive * (2,))
            updates = self.einsum_call(gate.cache.calculation_cache.vector, state[-1],
                                     gate.matrix)
            # Concatenate the updated part of the state `updates` with the
            # part of of the state that remained unaffected `state[:-1]`.
            state = self.concatenate([state[:-1], updates[self.newaxis]], axis=0)
            state = self.reshape(state, gate.nqubits * (2,))
            # Put qubit indices back to their proper places
            state = self.transpose(state, gate.cache.control_cache.reverse(False))
        else:
            einsum_str = gate.cache.calculation_cache.vector
            state = self.einsum_call(einsum_str, state, gate.matrix)
        return self.reshape(state, gate.cache.flat_shape)

    def state_vector_matrix_call(self, gate, state):
        return self.state_vector_call(gate, state)

    def density_matrix_call(self, gate, state):
        state = self.reshape(state, gate.cache.tensor_shape)
        if gate.is_controlled_by:
            ncontrol = len(gate.control_qubits)
            nactive = gate.nqubits - ncontrol
            n = 2 ** ncontrol
            state = self.transpose(state, gate.cache.control_cache.order(True))
            state = self.reshape(state, 2 * (n,) + 2 * nactive * (2,))
            state01 = self.gather(state, indices=range(n - 1), axis=0)
            state01 = self.squeeze(self.gather(state01, indices=[n - 1], axis=1), axis=1)
            state01 = self.einsum_call(gate.cache.calculation_cache.right0,
                                     state01, self.conj(gate.matrix))
            state10 = self.gather(state, indices=range(n - 1), axis=1)
            state10 = self.squeeze(self.gather(state10, indices=[n - 1], axis=0), axis=0)
            state10 = self.einsum_call(gate.cache.calculation_cache.left0,
                                       state10, gate.matrix)

            state11 = self.squeeze(self.gather(state, indices=[n - 1], axis=0), axis=0)
            state11 = self.squeeze(self.gather(state11, indices=[n - 1], axis=0), axis=0)
            state11 = self.einsum_call(gate.cache.calculation_cache.right, state11, self.conj(gate.matrix))
            state11 = self.einsum_call(gate.cache.calculation_cache.left, state11, gate.matrix)

            state00 = self.gather(state, indices=range(n - 1), axis=0)
            state00 = self.gather(state00, indices=range(n - 1), axis=1)
            state01 = self.concatenate([state00, state01[:, self.newaxis]], axis=1)
            state10 = self.concatenate([state10, state11[self.newaxis]], axis=0)
            state = self.concatenate([state01, state10[self.newaxis]], axis=0)
            state = self.reshape(state, 2 * gate.nqubits * (2,))
            state = self.transpose(state, gate.cache.control_cache.reverse(True))
        else:
            state = self.einsum_call(gate.cache.calculation_cache.right, state,
                                   self.conj(gate.matrix))
            state = self.einsum_call(gate.cache.calculation_cache.left, state, gate.matrix)
        return self.reshape(state, gate.cache.flat_shape)

    def density_matrix_matrix_call(self, gate, state):
        return self.density_matrix_call(gate, state)

    def density_matrix_half_call(self, gate, state):
        if gate.is_controlled_by: # pragma: no cover
            raise_error(NotImplementedError, "Gate density matrix half call is "
                                             "not implemented for ``controlled_by``"
                                             "gates.")
        state = self.reshape(state, gate.cache.tensor_shape)
        state = self.einsum_call(gate.cache.calculation_cache.left, state, gate.matrix)
        return self.reshape(state, gate.cache.flat_shape)

    def density_matrix_half_matrix_call(self, gate, state):
        return self.density_matrix_half_call(gate, state)

    def _append_zeros(self, state, qubits, results):
        """Helper method for `state_vector_collapse` and `density_matrix_collapse`."""
        for q, r in zip(qubits, results):
            state = self.expand_dims(state, axis=q)
            if r:
                state = self.concatenate([self.zeros_like(state), state], axis=q)
            else:
                state = self.concatenate([state, self.zeros_like(state)], axis=q)
        return state

    def state_vector_collapse(self, gate, state, result):
        state = self.reshape(state, gate.cache.tensor_shape)
        substate = self.gather_nd(self.transpose(state, gate.cache.order), result)
        norm = self.sum(self.square(self.abs(substate)))
        state = substate / self.cast(self.sqrt(norm), dtype=state.dtype)
        state = self._append_zeros(state, sorted(gate.target_qubits), result)
        return self.reshape(state, gate.cache.flat_shape)

    def density_matrix_collapse(self, gate, state, result):
        density_matrix_result = 2 * result
        sorted_qubits = sorted(gate.target_qubits)
        sorted_qubits = sorted_qubits + [q + gate.nqubits for q in sorted_qubits]
        state = self.reshape(state, gate.cache.tensor_shape)
        substate = self.gather_nd(self.transpose(state, gate.cache.order),
                                  density_matrix_result)
        n = 2 ** (len(tuple(substate.shape)) // 2)
        norm = self.trace(self.reshape(substate, (n, n)))
        state = substate / norm
        state = self._append_zeros(state, sorted_qubits, density_matrix_result)
        return self.reshape(state, gate.cache.flat_shape)

    def assert_allclose(self, value, target, rtol=1e-7, atol=0.0):
        self.np.testing.assert_allclose(value, target, rtol=rtol, atol=atol)


class JITCustomBackend(NumpyBackend): # pragma: no cover

    description = "Uses custom operators based on numba.jit for CPU and " \
                  "custom CUDA kernels loaded with cupy GPU."

    def __init__(self):
        from qibo.backends import Backend
        if not Backend.check_availability("qibojit"): # pragma: no cover
            # CI can compile custom operators so this case is not tested
            raise_error(RuntimeError, "Cannot initialize qibojit "
                                      "if the qibojit is not installed.")
        from qibojit import custom_operators as op # pylint: disable=E0401
        super().__init__()
        self.name = "qibojit"
        self.op = op

        try:
            from cupy import cuda # pylint: disable=E0401
            ngpu = cuda.runtime.getDeviceCount()
        except:
            ngpu = 0

        import os
        if "OMP_NUM_THREADS" in os.environ: # pragma: no cover
            self.set_threads(int(os.environ.get("OMP_NUM_THREADS")))
        if "NUMBA_NUM_THREADS" in os.environ: # pragma: no cover
            self.set_threads(int(os.environ.get("NUMBA_NUM_THREADS")))

        # TODO: reconsider device management
        self.cpu_devices = ["/CPU:0"]
        self.gpu_devices = [f"/GPU:{i}" for i in range(ngpu)]
        if self.gpu_devices: # pragma: no cover
            # CI does not use GPUs
            self.default_device = self.gpu_devices[0]
            self.set_engine("cupy")
        elif self.cpu_devices:
            self.default_device = self.cpu_devices[0]
            self.set_engine("numba")
            self.set_threads(self.nthreads)

    def set_engine(self, name): # pragma: no cover
        """Switcher between ``cupy`` for GPU and ``numba`` for CPU."""
        if name == "numba":
            import numpy as xp
            self.tensor_types = (xp.ndarray,)
        elif name == "cupy":
            import cupy as xp # pylint: disable=E0401
            self.tensor_types = (self.np.ndarray, xp.ndarray)
        else:
            raise_error(ValueError, "Unknown engine {}.".format(name))
        self.backend = xp
        self.numeric_types = (int, float, complex, xp.int32,
                              xp.int64, xp.float32, xp.float64,
                              xp.complex64, xp.complex128)
        self.native_types = (xp.ndarray,)
        self.Tensor = xp.ndarray
        self.random = xp.random
        self.newaxis = xp.newaxis
        self.op.set_backend(name)
        with self.device(self.default_device):
            self.matrices.allocate_matrices()

    def set_device(self, name):
        abstract.AbstractBackend.set_device(self, name)
        if "GPU" in name:
            self.set_engine("cupy")
        else:
            self.set_engine("numba")

    def set_threads(self, nthreads):
        abstract.AbstractBackend.set_threads(self, nthreads)
        import numba # pylint: disable=E0401
        numba.set_num_threads(nthreads)

    def to_numpy(self, x):
        if isinstance(x, self.np.ndarray):
            return x
        elif isinstance(x, AbstractState):
            x = x.numpy()
        return self.op.to_numpy(x)

    def cast(self, x, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.op.cast(x, dtype=dtype)

    def check_shape(self, shape):
        if self.op.get_backend() == "cupy" and isinstance(shape, self.Tensor):
            shape = shape.get()
        return shape

    def reshape(self, x, shape):
        return super().reshape(x, self.check_shape(shape))

    def eye(self, shape, dtype='DTYPECPX'):
        return super().eye(self.check_shape(shape), dtype=dtype)

    def zeros(self, shape, dtype='DTYPECPX'):
        return super().zeros(self.check_shape(shape), dtype=dtype)

    def ones(self, shape, dtype='DTYPECPX'):
        return super().ones(self.check_shape(shape), dtype=dtype)

    def expm(self, x):
        if self.op.get_backend() == "cupy":
            # Fallback to numpy because cupy does not have expm
            if isinstance(x, self.native_types):
                x = x.get()
            return self.backend.asarray(super().expm(x))
        return super().expm(x)

    def unique(self, x, return_counts=False):
        if self.op.get_backend() == "cupy":
            if isinstance(x, self.native_types):
                x = x.get()
            # Uses numpy backend always
        return super().unique(x, return_counts)

    def gather(self, x, indices=None, condition=None, axis=0):
        if self.op.get_backend() == "cupy":
            # Fallback to numpy because cupy does not support tuple indexing
            if isinstance(x, self.native_types):
                x = x.get()
            if isinstance(indices, self.native_types):
                indices = indices.get()
            if isinstance(condition, self.native_types):
                condition = condition.get()
            result = super().gather(x, indices, condition, axis)
            return self.backend.asarray(result)
        return super().gather(x, indices, condition, axis)

    def device(self, device_name):
        # assume tf naming convention '/GPU:0'
        if self.op.get_backend() == "numba":
            return super().device(device_name)
        else: # pragma: no cover
            device_id = int(device_name.split(":")[-1])
            return self.backend.cuda.Device(device_id)

    def initial_state(self, nqubits, is_matrix=False):
        return self.op.initial_state(nqubits, self.dtypes('DTYPECPX'),
                                    is_matrix=is_matrix)

    def sample_frequencies(self, probs, nshots):
        from qibo.config import SHOT_METROPOLIS_THRESHOLD
        if nshots < SHOT_METROPOLIS_THRESHOLD:
            return super().sample_frequencies(probs, nshots)
        if self.op.get_backend() == "cupy":
            probs = probs.get()
        dtype = self._dtypes.get('DTYPEINT')
        seed = self.np.random.randint(0, int(1e8), dtype=dtype)
        nqubits = int(self.np.log2(tuple(probs.shape)[0]))
        frequencies = self.np.zeros(2 ** nqubits, dtype=dtype)
        frequencies = self.op.measure_frequencies(
            frequencies, probs, nshots, nqubits, seed, self.nthreads)
        return frequencies

    def create_einsum_cache(self, qubits, nqubits, ncontrol=None): # pragma: no cover
        raise_error(NotImplementedError)

    def einsum_call(self, cache, state, matrix): # pragma: no cover
        raise_error(NotImplementedError)

    def create_gate_cache(self, gate):
        cache = self.GateCache()
        qubits = [gate.nqubits - q - 1 for q in gate.control_qubits]
        qubits.extend(gate.nqubits - q - 1 for q in gate.target_qubits)
        cache.qubits_tensor = tuple(sorted(qubits))
        if gate.density_matrix:
            cache.target_qubits_dm = [q + gate.nqubits for q in gate.target_qubits]
        return cache

    def state_vector_call(self, gate, state):
        return gate.gate_op(state, gate.nqubits, *gate.target_qubits,
                            gate.cache.qubits_tensor)

    def state_vector_matrix_call(self, gate, state):
        return gate.gate_op(state, gate.matrix, gate.nqubits, *gate.target_qubits,
                            gate.cache.qubits_tensor)

    def density_matrix_call(self, gate, state):
        qubits = tuple(x + gate.nqubits for x in gate.cache.qubits_tensor)
        shape = state.shape
        state = gate.gate_op(state.flatten(), 2 * gate.nqubits,
                             *gate.target_qubits, qubits)
        state = gate.gate_op(state, 2 * gate.nqubits, *gate.cache.target_qubits_dm,
                             gate.cache.qubits_tensor)
        return self.reshape(state, shape)

    def density_matrix_matrix_call(self, gate, state):
        qubits = tuple(x + gate.nqubits for x in gate.cache.qubits_tensor)
        shape = state.shape
        state = gate.gate_op(state.flatten(), gate.matrix, 2 * gate.nqubits,
                             *gate.target_qubits, qubits)
        adjmatrix = self.conj(gate.matrix)
        state = gate.gate_op(state, adjmatrix, 2 * gate.nqubits, *gate.cache.target_qubits_dm,
                             gate.cache.qubits_tensor)
        return self.reshape(state, shape)

    def density_matrix_half_call(self, gate, state):
        qubits = tuple(x + gate.nqubits for x in gate.cache.qubits_tensor)
        shape = state.shape
        state = gate.gate_op(state.flatten(), 2 * gate.nqubits,
                             *gate.target_qubits, qubits)
        return self.reshape(state, shape)

    def density_matrix_half_matrix_call(self, gate, state):
        qubits = tuple(x + gate.nqubits for x in gate.cache.qubits_tensor)
        shape = state.shape
        state = gate.gate_op(state.flatten(), gate.matrix, 2 * gate.nqubits,
                             *gate.target_qubits, qubits)
        return self.reshape(state, shape)

    def _result_tensor(self, result):
        n = len(result)
        return int(sum(2 ** (n - i - 1) * r for i, r in enumerate(result)))

    def state_vector_collapse(self, gate, state, result):
        result = self._result_tensor(result)
        return gate.gate_op(state, gate.cache.qubits_tensor, result, gate.nqubits, True)

    def density_matrix_collapse(self, gate, state, result):
        result = self._result_tensor(result)
        qubits = tuple(x + gate.nqubits for x in gate.cache.qubits_tensor)
        shape = state.shape
        state = gate.gate_op(state.flatten(), qubits, result, 2 * gate.nqubits, False)
        state = gate.gate_op(state, gate.cache.qubits_tensor, result,
                             2 * gate.nqubits, False)
        state = self.reshape(state, shape)
        return state / self.trace(state)

    def assert_allclose(self, value, target, rtol=1e-7, atol=0.0):
        if self.op.get_backend() == "cupy":
            if isinstance(value, self.backend.ndarray):
                value = value.get()
            if isinstance(target, self.backend.ndarray):
                target = target.get()
        self.np.testing.assert_allclose(value, target, rtol=rtol, atol=atol)
