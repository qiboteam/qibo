from qibo.config import raise_error
from qibo.hamiltonians.abstract import AbstractHamiltonian
from qibo.hamiltonians.adiabatic import BaseAdiabaticHamiltonian
from qibo.hamiltonians.hamiltonians import SymbolicHamiltonian


class BaseSolver:
    """Basic solver that should be inherited by all solvers.

    Args:
        dt (float): Time step size.
        hamiltonian (:class:`qibo.hamiltonians.abstract.AbstractHamiltonian`): Hamiltonian object
            that the state evolves under.
    """

    def __init__(self, dt, hamiltonian):
        self.dt = dt
        if isinstance(hamiltonian, AbstractHamiltonian):
            self.backend = hamiltonian.backend
            self.hamiltonian = lambda t: hamiltonian
        else:
            self.backend = hamiltonian(0).backend
            self.hamiltonian = hamiltonian
        self.t = 0

    @property
    def t(self):
        """Solver's current time."""
        return self._t

    @t.setter
    def t(self, new_t):
        """Updates solver's current time."""
        self._t = new_t
        self.current_hamiltonian = self.hamiltonian(self.t)

    def __call__(self, state):  # pragma: no cover
        # abstract method
        raise_error(NotImplementedError)


class TrotterizedExponential(BaseSolver):
    """Solver that uses Trotterized exponentials.

    Created automatically from the :class:`qibo.solvers.Exponential` if the
    given Hamiltonian object is a
    :class:`qibo.hamiltonians.hamiltonians.SymbolicHamiltonian`.
    """

    def __init__(self, dt, hamiltonian):
        super().__init__(dt, hamiltonian)
        if isinstance(self.hamiltonian, BaseAdiabaticHamiltonian):
            self.circuit = lambda t, dt: self.hamiltonian.circuit(self.dt, t=self.t)
        else:
            self.circuit = lambda t, dt: self.hamiltonian(self.t).circuit(self.dt)

    def __call__(self, state):
        circuit = self.circuit(self.t, self.dt)
        self.t += self.dt
        result = self.backend.execute_circuit(circuit, initial_state=state)
        return result.state()


class Exponential(BaseSolver):
    """Solver that uses the matrix exponential of the Hamiltonian:

    .. math::
        U(t) = e^{-i H(t) \\delta t}

    Calculates the evolution operator in every step and thus is compatible with
    time-dependent Hamiltonians.
    """

    def __call__(self, state):
        propagator = self.current_hamiltonian.exp(self.dt)
        self.t += self.dt
        return (propagator @ state[:, None])[:, 0]


class RungeKutta4(BaseSolver):
    """Solver based on the 4th order Runge-Kutta method."""

    def __call__(self, state):
        ham1 = self.current_hamiltonian
        ham2 = self.hamiltonian(self.t + self.dt / 2.0)
        ham3 = self.hamiltonian(self.t + self.dt)
        k1 = ham1 @ state
        k2 = ham2 @ (state + self.dt * k1 / 2.0)
        k3 = ham2 @ (state + self.dt * k2 / 2.0)
        k4 = ham3 @ (state + self.dt * k3)
        self.t += self.dt
        return state - 1j * self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


class RungeKutta45(BaseSolver):
    """Solver based on the 5th order Runge-Kutta method."""

    def __call__(self, state):
        ham1 = self.current_hamiltonian
        ham2 = self.hamiltonian(self.t + self.dt / 4.0)
        ham3 = self.hamiltonian(self.t + 3 * self.dt / 8.0)
        ham4 = self.hamiltonian(self.t + 12 * self.dt / 13.0)
        ham5 = self.hamiltonian(self.t + self.dt)
        ham6 = self.hamiltonian(self.t + self.dt / 2.0)
        k1 = ham1 @ state
        k2 = ham2 @ (state + self.dt * k1 / 4.0)
        k3 = ham3 @ (state + self.dt * (3 * k1 + 9 * k2) / 32.0)
        k4 = ham4 @ (state + self.dt * (1932 * k1 - 7200 * k2 + 7296 * k3) / 2197.0)
        k5 = ham5 @ (
            state
            + self.dt
            * (439 * k1 / 216.0 - 8 * k2 + 3680 * k3 / 513.0 - 845 * k4 / 4104.0)
        )
        k6 = ham6 @ (
            state
            + self.dt
            * (
                -8 * k1 / 27.0
                + 2 * k2
                - 3544 * k3 / 2565
                + 1859 * k4 / 4104
                - 11 * k5 / 40.0
            )
        )
        self.t += self.dt
        return state - 1j * self.dt * (
            16 * k1 / 135.0
            + 6656 * k3 / 12825.0
            + 28561 * k4 / 56430.0
            - 9 * k5 / 50.0
            + 2 * k6 / 55.0
        )


def get_solver(solver_name, dt, hamiltonian):
    if solver_name == "exp":
        if isinstance(hamiltonian, AbstractHamiltonian):
            h0 = hamiltonian
        elif isinstance(hamiltonian, BaseAdiabaticHamiltonian):
            h0 = hamiltonian.h0
        else:
            h0 = hamiltonian(0)

        if isinstance(h0, SymbolicHamiltonian):
            return TrotterizedExponential(dt, hamiltonian)
        else:
            return Exponential(dt, hamiltonian)

    elif solver_name == "rk4":
        return RungeKutta4(dt, hamiltonian)

    elif solver_name == "rk45":
        return RungeKutta45(dt, hamiltonian)

    else:  # pragma: no cover
        raise_error(ValueError, f"Unknown solver {solver_name}.")
