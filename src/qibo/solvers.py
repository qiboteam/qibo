from qibo import K, hamiltonians
from qibo.config import raise_error


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
        if isinstance(hamiltonian, hamiltonians.Hamiltonian):
            self.hamiltonian = lambda t: hamiltonian
        else:
            self.hamiltonian = hamiltonian

    def __call__(self, state): # pragma: no cover
        # abstract method
        raise_error(NotImplementedError)


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


class RungeKutta45(BaseSolver):
    """Solver based on the 5th order Runge-Kutta method."""

    def __call__(self, state):
        state = state[:, K.newaxis]
        ham1 = self.hamiltonian(self.t)
        ham2 = self.hamiltonian(self.t + self.dt / 4.0)
        ham3 = self.hamiltonian(self.t + 3 * self.dt / 8.0)
        ham4 = self.hamiltonian(self.t + 12 * self.dt / 13.0)
        ham5 = self.hamiltonian(self.t + self.dt)
        ham6 = self.hamiltonian(self.t + self.dt / 2.0)
        k1 = ham1 @ state
        k2 = ham2 @ (state + self.dt * k1 / 4.0)
        k3 = ham3 @ (state + self.dt * (3 * k1 + 9 * k2) / 32.0)
        k4 = ham4 @ (state + self.dt * (1932 * k1 -
                                        7200 * k2 + 7296 * k3) / 2197.0)
        k5 = ham5 @ (state + self.dt * (439 * k1 / 216.0 - 8 *
                                        k2 + 3680 * k3 / 513.0 - 845 * k4 / 4104.0))
        k6 = ham6 @ (state + self.dt * (-8 * k1 / 27.0 + 2 * k2 -
                                        3544 * k3 / 2565 + 1859 * k4 / 4104 - 11 * k5 / 40.0))
        self.t += self.dt
        return (state - 1j * self.dt * (16 * k1 / 135.0 + 6656 * k3 / 12825.0 + 28561 * k4 / 56430.0 -
                                        9 * k5 / 50.0 + 2 * k6 / 55.0))[:, 0]


factory = {
    "exp": Exponential,
    "rk4": RungeKutta4,
    "rk45": RungeKutta45
}
