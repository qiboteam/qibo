"""Models for time evolution of state vectors."""
import numpy as np
from qibo import solvers, optimizers, hamiltonians
from qibo.tensorflow import circuit
from qibo.config import log, raise_error


class StateEvolution:
    """Unitary time evolution of a state vector under a Hamiltonian.

    Args:
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): Hamiltonian to
            evolve under.
        dt (float): Time step to use for the numerical integration of
            Schrondiger's equation.
        solver (str): Solver to use for integrating Schrodinger's equation.
        callbacks (list): List of callbacks to calculate during evolution.

    Example:
        ::

            import numpy as np
            from qibo import models, hamiltonians
            # create critical (h=1.0) TFIM Hamiltonian for three qubits
            hamiltonian = hamiltonians.TFIM(3, h=1.0)
            # initialize evolution model with step dt=1e-2
            evolve = models.StateEvolution(hamiltonian, dt=1e-2)
            # initialize state to |+++>
            initial_state = np.ones(8) / np.sqrt(8)
            # execute evolution for total time T=2
            final_state2 = evolve(T=2, initial_state)
    """

    def __init__(self, hamiltonian, dt, solver="exp", callbacks=[]):
        self.nqubits = hamiltonian.nqubits
        if dt <= 0:
            raise_error(ValueError, f"Time step dt should be positive but is {dt}.")
        self.dt = dt
        self.solver = solvers.factory[solver](self.dt, hamiltonian)
        self.callbacks = callbacks

    def execute(self, final_time, start_time=0.0, initial_state=None):
        """Runs unitary evolution for a given total time.

        Args:
            final_time (float): Final time of evolution.
            start_time (float): Initial time of evolution. Defaults to t=0.
            initial_state (np.ndarray): Initial state of the evolution.

        Returns:
            Final state vector a ``tf.Tensor``.
        """
        state = self._cast_initial_state(initial_state)
        self.solver.t = start_time
        nsteps = int((final_time - start_time) / self.solver.dt)
        for callback in self.callbacks:
            callback.append(callback(state))
        for _ in range(nsteps):
            state = self.solver(state)
            for callback in self.callbacks:
                callback.append(callback(state))
        return state

    def __call__(self, final_time, start_time=0.0, initial_state=None):
        """Equivalent to :meth:`qibo.models.StateEvolution.execute`."""
        return self.execute(final_time, start_time, initial_state)

    def _cast_initial_state(self, initial_state=None):
        """Casts initial state as a Tensorflow tensor."""
        if initial_state is None:
            raise_error(ValueError, "StateEvolution cannot be used without initial "
                                    "state.")
        return circuit.TensorflowCircuit._cast_initial_state(
            self, initial_state)


class AdiabaticEvolution(StateEvolution):
    """Adiabatic evolution of a state vector under the following Hamiltonian:

    .. math::
        H(t) = (1 - s(t)) H_0 + s(t) H_1

    Args:
        h0 (:class:`qibo.hamiltonians.Hamiltonian`): Easy Hamiltonian.
        h1 (:class:`qibo.hamiltonians.Hamiltonian`): Problem Hamiltonian.
            These Hamiltonians should be time-independent.
        s (callable): Function of time that defines the scheduling of the
            adiabatic evolution. Can be either a function of time s(t) or a
            function with two arguments s(t, p) where p corresponds to a vector
            of parameters to be optimized.
        dt (float): Time step to use for the numerical integration of
            Schrondiger's equation.
        solver (str): Solver to use for integrating Schrodinger's equation.
        callbacks (list): List of callbacks to calculate during evolution.
    """
    ATOL = 1e-7 # Tolerance for checking s(0) = 0 and s(T) = 1.

    def __init__(self, h0, h1, s, dt, solver="exp", callbacks=[]):
        if not isinstance(h0, hamiltonians.Hamiltonian):
            raise_error(TypeError, f"h0 should be a hamiltonians.Hamiltonian object "
                                    "but is {type(h0)}.")
        if not isinstance(h1, hamiltonians.Hamiltonian):
            raise_error(TypeError, f"h1 should be a hamiltonians.Hamiltonian object "
                                    "but is {type(h1)}.")
        if h0.nqubits != h1.nqubits:
            raise_error(ValueError, "H0 has {} qubits while H1 has {}."
                                    "".format(h0.nqubits, h1.nqubits))
        ham = lambda t: h0
        ham.nqubits = h0.nqubits
        super(AdiabaticEvolution, self).__init__(ham, dt, solver, callbacks)
        self.h0 = h0
        self.h1 = h1

        # Flag to control if loss messages are shown during optimization
        self.opt_messages = False
        self.opt_history = {"params": [], "loss": []}

        self._schedule = None
        self._param_schedule = None
        nparams = s.__code__.co_argcount
        if nparams == 1: # given ``s`` is a function of time only
            self.schedule = s
        elif nparams == 2: # given ``s`` has undefined parameters
            self._param_schedule = s
        else:
            raise_error(ValueError, f"Scheduling function shoud take one or two "
                                     "arguments but it takes {nparams}.")

    @property
    def schedule(self):
        """Returns scheduling as a function of time."""
        if self._schedule is None:
            raise_error(ValueError, "Cannot access scheduling function before setting "
                                    "its free parameters.")
        return self._schedule

    @schedule.setter
    def schedule(self, f):
        """Sets scheduling s(t) function."""
        s0 = f(0)
        if np.abs(s0) > self.ATOL:
            raise_error(ValueError, f"s(0) should be 0 but is {s0}.")
        s1 = f(1)
        if np.abs(s1 - 1) > self.ATOL:
            raise_error(ValueError, f"s(1) should be 1 but is {s1}.")
        self._schedule = f

    def execute(self, final_time, start_time=0.0, initial_state=None):
        """"""
        if start_time != 0:
            raise_error(NotImplementedError, "Adiabatic evolution supports only t=0 "
                                             "as initial time.")
        self.set_hamiltonian(final_time - start_time)
        return super(AdiabaticEvolution, self).execute(
            final_time, start_time, initial_state)

    def set_parameters(self, params):
        """Sets the variational parameters of the scheduling function."""
        if self._param_schedule is not None:
            self.schedule = lambda t: self._param_schedule(t, params[:-1])
        self.set_hamiltonian(params[-1])

    def set_hamiltonian(self, final_time):
        def hamiltonian(t):
            # Disable warning that ``schedule`` is not Callable
            st = self.schedule(t / final_time) # pylint: disable=E1102
            return self.h0 * (1 - st) + self.h1 * st
        self.solver.hamiltonian = hamiltonian

    def _cast_initial_state(self, initial_state=None):
        """Casts initial state as a Tensorflow tensor.

        If initial state is not given the ground state of ``h0`` is used, which
        is the common practice in adiabatic evolution.
        """
        if initial_state is None:
            return self.h0.eigenvectors()[:, 0]
        return super(AdiabaticEvolution, self)._cast_initial_state(initial_state)

    def _loss(self, params):
        """Expectation value of H1 for a choice of scheduling parameters.

        Returns a ``tf.Tensor``.
        """
        self.set_parameters(params)
        final_state = super(AdiabaticEvolution, self).execute(params[-1])
        loss = self.h1.expectation(final_state, normalize=True)
        if self.opt_messages:
            self.opt_history["params"].append(params)
            self.opt_history["loss"].append(loss)
            log.info(f"Params: {params}  -  <H1> = {loss}")
        return loss

    def _nploss(self, params):
        """Expectation value of H1 for a choice of scheduling parameters.

        Returns a ``np.ndarray``.
        """
        loss = self._loss(params).numpy()
        return loss

    def minimize(self, initial_parameters, method="BFGS", options=None,
                 messages=False):
        """Optimize the free parameters of the scheduling function.

        Args:
            initial_parameters (np.ndarray): Initial guess for the variational
                parameters that are optimized.
                The last element of the given array should correspond to the
                guess for the total evolution time T.
            method (str): The desired minimization method.
                One of ``"cma"`` (genetic optimizer), ``"sgd"`` (gradient descent) or
                any of the methods supported by
                `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.
            options (dict): a dictionary with options for the different optimizers.
            messages (bool): If ``True`` the loss evolution is shown during
                optimization.
        """
        import numpy as np
        self.opt_messages = messages
        if method == "sgd":
            loss = self._loss
        else:
            loss = self._nploss

        result, parameters = optimizers.optimize(loss, initial_parameters,
                                                 method, options)
        if isinstance(parameters, np.ndarray) and not len(parameters.shape): # pragma: no cover
            # some optimizers like ``Powell`` return number instead of list
            parameters = [parameters]
        self.set_parameters(parameters)
        return result, parameters
