from qibo import K
from qibo.base import hamiltonians

HAMILTONIAN_TYPES = (hamiltonians.Hamiltonian, hamiltonians.LocalHamiltonian)


class BaseSolver:
    """Basic solver that should be inherited by all solvers.

    Args:
        dt (float): Time step size.
        hamiltonian (:class:`qibo.base.hamiltonians.Hamiltonian`): Hamiltonian object
            that the state evolves under.
    """

    def __init__(self, dt, hamiltonian):
        self.t = 0
        self.dt = dt
        if issubclass(type(hamiltonian), HAMILTONIAN_TYPES):
            self.hamiltonian = lambda t: hamiltonian
        else:
            self.hamiltonian = hamiltonian

    def __call__(self, state): # pragma: no cover
        raise NotImplementedError


class TrotterizedExponential(BaseSolver):
    """Solver that uses Trotterized exponentials.

    Created automatically from the :class:`qibo.solvers.Exponential` if the
    given Hamiltonian object is a
    :class:`qibo.base.hamiltonians.LocalHamiltonian`.
    """

    def __call__(self, state):
        circuit = self.hamiltonian(self.t).circuit(self.dt)
        self.t += self.dt
        return circuit(state)


class Exponential(BaseSolver):
    """Solver that uses the matrix exponential of the Hamiltonian:

    .. math::
        U(t) = e^{-i H(t) \\delta t}

    Calculates the evolution operator in every step and thus is compatible with
    time-dependent Hamiltonians.
    """

    def __new__(cls, dt, hamiltonian):
        if isinstance(hamiltonian, hamiltonians.LocalHamiltonian):
            # FIXME: This won't work for time-dependent local Hamiltonians
            return TrotterizedExponential(dt, hamiltonian)
        else:
            return super(Exponential, cls).__new__(cls)

    def __call__(self, state):
        propagator = self.hamiltonian(self.t).exp(self.dt)
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
