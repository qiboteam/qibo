from qibo import __version__
from qibo.backends.npmatrices import NumpyMatrices
from qibo.backends.numpy import NumpyBackend
from qibo.config import log, raise_error


class JaxBackend(NumpyBackend):
    def __init__(self):
        super().__init__()
        self.name = "jax"

        import jax
        import jax.numpy as jnp  # pylint: disable=import-error
        import numpy

        jax.config.update("jax_enable_x64", True)

        self.jax = jax
        self.numpy = numpy

        self.np = jnp
        self.tensor_types = (jnp.ndarray, numpy.ndarray)

    def set_precision(self, precision):
        if precision != self.precision:
            if precision == "single":
                self.precision = precision
                self.dtype = self.np.complex64
            elif precision == "double":
                self.precision = precision
                self.dtype = self.np.complex128
            else:
                raise_error(ValueError, f"Unknown precision {precision}.")
            if self.matrices:
                self.matrices = self.matrices.__class__(self.dtype)

    def cast(self, x, dtype=None, copy=False):
        if dtype is None:
            dtype = self.dtype
        if isinstance(x, self.tensor_types):
            return x.astype(dtype)
        elif self.issparse(x):
            return x.astype(dtype)
        return self.np.array(x, dtype=dtype, copy=copy)

    def set_seed(self, seed):
        self.jax.random.key(seed)

    def sample_shots(self, probabilities, nshots):
        return self.jax.random.choice(
            range(len(probabilities)), size=nshots, p=probabilities
        )

    def zero_state(self, nqubits):
        state = self.np.zeros(2**nqubits, dtype=self.dtype)
        state = state.at[0].set(1)
        return state

    def zero_density_matrix(self, nqubits):
        state = self.np.zeros(2 * (2**nqubits,), dtype=self.dtype)
        state = state.at[0, 0].set(1)
        return state

    def plus_state(self, nqubits):
        state = self.np.ones(2**nqubits, dtype=self.dtype)
        state /= self.np.sqrt(2**nqubits)
        return state

    def plus_density_matrix(self, nqubits):
        state = self.np.ones(2 * (2**nqubits,), dtype=self.dtype)
        state /= 2**nqubits
        return state

    def sample_shots(self, probabilities, nshots):
        return self.numpy.random.choice(
            range(len(probabilities)), size=nshots, p=probabilities
        )

    def update_frequencies(self, frequencies, probabilities, nsamples):
        samples = self.sample_shots(probabilities, nsamples)
        res, counts = self.np.unique(samples, return_counts=True)
        frequencies.at[res].add(counts)
        return frequencies
