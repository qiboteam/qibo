import tensorflow as tf
from qibo.tensorflow import solvers


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

    def execute(self, dt, solver="exp", initial_state=None):
        state = self._cast_initial_state(initial_state)

        solver = solvers.factory[solver](dt, self.hamiltonian)
        nsteps = int(self.total_time / solver.dt)
        for _ in range(nsteps):
            state = solver(state)
        return state

    def __call__(self, dt, solver="exp", initial_state=None):
        return self.execute(dt, solver, initial_state)

    def _cast_initial_state(self, initial_state=None):
        if initial_state is None:
            return self.h0.eigenvectors()[:, 0]
        return initial_state
