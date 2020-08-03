from qibo import K, hamiltonians


class BaseSolver:
    """Basic solver that should be inherited by all solvers.

    Args:
        dt (float): Time step size.
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): Hamiltonian object
            that the state evolves under.
    """

    def __init__(self, dt, hamiltonian):
        self.t = 0
        self.dt = dt
        if dt <= 0:
            raise ValueError(f"Time step dt should be positive but is {dt}.")
        if isinstance(hamiltonian, hamiltonians.Hamiltonian):
            self.hamiltonian = lambda t: hamiltonian
        else:
            self.hamiltonian = hamiltonian

    def __call__(self, state): # pragma: no cover
        raise NotImplementedError


class TimeIndependentExponential(BaseSolver):
    """Exact solver that uses the matrix exponential of the Hamiltonian:

    .. math::
        U(t) = e^{-i H t}

    Calculates the evolution operator during initialization and thus can be
    used only for Hamiltonians without explicit time dependence.
    """

    def __init__(self, dt, hamiltonian):
        super(TimeIndependentExponential, self).__init__(dt, hamiltonian)
        self.propagator = hamiltonian.exp(-1j * dt)

    def __call__(self, state):
        self.t += self.dt
        return K.matmul(self.propagator, state[:, K.newaxis])[:, 0]


class Exponential(BaseSolver):
    """Solver that uses the matrix exponential of the Hamiltonian:

    .. math::
        U(t) = e^{-i H(t) \\delta t}

    Calculates the evolution operator in every step and thus is compatible with
    time-dependent Hamiltonians.
    """

    def __new__(cls, dt, hamiltonian):
        if isinstance(hamiltonian, hamiltonians.Hamiltonian):
            return TimeIndependentExponential(dt, hamiltonian)
        else:
            return super(Exponential, cls).__new__(cls)

    def __call__(self, state):
        propagator = self.hamiltonian(self.t).exp(-1j * self.dt)
        self.t += self.dt
        return K.matmul(propagator, state[:, K.newaxis])[:, 0]


class RungeKutta4(BaseSolver):
    """Solver based on the 4th order Runge-Kutta method."""

    def __call__(self, state):
        state = state[:, K.newaxis]
        ham1 = self.hamiltonian(self.t)
        ham2 = self.hamiltonian(self.t + self.dt / 2.0)
        ham3 = self.hamiltonian(self.t + self.dt)
        k1 = ham1 @ state
        k2 = ham2 @ (state + self.dt * k1 / 2.0)
        k3 = ham2 @ (state + self.dt * k2 / 2.0)
        k4 = ham3 @ (state + self.dt * k3)
        self.t += self.dt
        return (state - 1j * self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0)[:, 0]


factory = {
    "exp": Exponential,
    "rk4": RungeKutta4
}
