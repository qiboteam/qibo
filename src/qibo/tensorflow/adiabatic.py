import tensorflow as tf


class AdiabaticEvolution:

    def __init__(self, h0, h1, s, total_time):
        if s(0) != 0:
            raise ValueError("s(0) should be 0 but is {}.".format(s(0)))
        if s(total_time) != 1:
            raise ValueError("s(T) should be 1 but is {}."
                             "".format(s(total_time)))

        self.total_time = total_time
        self.s = s

        self.h0 = h0
        self.h1 = h1

    def hamiltonian(self, t):
        return (1 - self.s(t)) * self.h0 + self.s(t) * self.h1

    def execute(self, dt, initial_state=None):
        state = self._cast_initial_state(initial_state)

        nsteps = int(self.total_time / dt)
        for n in range(nsteps):
            t = n * dt
            propagator = tf.linalg.expm(-1j * dt * self.hamiltonian(t).hamiltonian)
            state = tf.matmul(propagator, state[:, tf.newaxis])[:, 0]
        return state

    def __call__(self, dt, initial_state=None):
        return self.execute(dt, initial_state)

    def _cast_initial_state(self, initial_state=None):
        if initial_state is None:
            return self.h0.eigenvectors()[:, 0]
        return initial_state
