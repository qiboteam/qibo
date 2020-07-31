from qibo import K, hamiltonians


class BaseSolver:

    def __init__(self, dt, hamiltonian):
        self.t = 0
        self.dt = dt
        if isinstance(hamiltonian, hamiltonians.Hamiltonian):
            self.hamiltonian = lambda t: hamiltonian
        else:
            self.hamiltonian = hamiltonian

    def __call__(self, state):
        raise NotImplementedError


class TimeIndependentExponential(BaseSolver):

    def __init__(self, dt, hamiltonian):
        super(TimeIndependentExponential, self).__init__(dt, hamiltonian)
        self.propagator = K.linalg.expm(-1j * dt * hamiltonian.hamiltonian)

    def __call__(self, state):
        self.t += self.dt
        return K.matmul(self.propagator, state[:, K.newaxis])[:, 0]


class Exponential(BaseSolver):

    def __new__(cls, dt, hamiltonian):
        if isinstance(hamiltonian, hamiltonians.Hamiltonian):
            return TimeIndependentExponential(dt, hamiltonian)
        else:
            return super(Exponential, cls).__new__(cls)

    def __call__(self, state):
        propagator = K.linalg.expm(
            -1j * self.dt * self.hamiltonian(self.t).hamiltonian)
        self.t += self.dt
        return K.matmul(propagator, state[:, K.newaxis])[:, 0]


class RungeKutta4(BaseSolver):

    def __call__(self, state):
        state = state[:, K.newaxis]
        ham1 = self.hamiltonian(self.t).hamiltonian
        ham2 = self.hamiltonian(self.t + self.dt / 2.0).hamiltonian
        ham3 = self.hamiltonian(self.t + self.dt).hamiltonian
        k1 = K.matmul(ham1, state)
        k2 = K.matmul(ham2, state + self.dt * k1 / 2.0)
        k3 = K.matmul(ham2, state + self.dt * k2 / 2.0)
        k4 = K.matmul(ham3, state + self.dt * k3)
        self.t += self.dt
        return (state - 1j * self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0)[:, 0]


factory = {
    "exp": Exponential,
    "rk4": RungeKutta4
}
