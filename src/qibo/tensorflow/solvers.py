import tensorflow as tf


class BaseSolver:

    def __init__(self, dt, hamiltonian):
        self.t = 0
        self.dt = dt
        self.hamiltonian = hamiltonian

    def __call__(self, state):
        raise NotImplementedError


class ExponentialPropagator(BaseSolver):

    def __call__(self, state):
        propagator = tf.linalg.expm(
            -1j * self.dt * self.hamiltonian(self.t).hamiltonian)
        self.t += self.dt
        return tf.matmul(propagator, state[:, tf.newaxis])[:, 0]


factory = {
    "exp": ExponentialPropagator
}
