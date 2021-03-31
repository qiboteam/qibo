"""Models for time evolution of state vectors."""
from qibo import solvers, optimizers, K
from qibo.abstractions import hamiltonians
from qibo.core import circuit, states
from qibo.config import log, raise_error
from qibo.callbacks import Norm, Gap


class StateEvolution:
    """Unitary time evolution of a state vector under a Hamiltonian.

    Args:
        hamiltonian (:class:`qibo.abstractions.hamiltonians.Hamiltonian`): Hamiltonian
            to evolve under.
        dt (float): Time step to use for the numerical integration of
            Schrondiger's equation.
        solver (str): Solver to use for integrating Schrodinger's equation.
        callbacks (list): List of callbacks to calculate during evolution.
        accelerators (dict): Dictionary of devices to use for distributed
            execution. See :class:`qibo.core.distcircuit.DistributedCircuit`
            for more details. This option is available only when the Trotter
            decomposition is used for the time evolution.
        memory_device (str): Name of device where the full state will be saved.
            Relevant only for distributed execution (when ``accelerators`` is
            given).

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

    def __init__(self, hamiltonian, dt, solver="exp", callbacks=[],
                 accelerators=None, memory_device="/CPU:0"):
        if isinstance(hamiltonian, hamiltonians.HAMILTONIAN_TYPES):
            ham = hamiltonian
        else:
            ham = hamiltonian(0)
            if not isinstance(ham, hamiltonians.HAMILTONIAN_TYPES):
                raise TypeError("Hamiltonian type {} not understood."
                                "".format(type(ham)))
        self.nqubits = ham.nqubits
        if dt <= 0:
            raise_error(ValueError, f"Time step dt should be positive but is {dt}.")
        self.dt = dt

        if (accelerators is not None and
            (not isinstance(ham, hamiltonians.TrotterHamiltonian)
             or solver != "exp")):
            raise_error(NotImplementedError, "Distributed evolution is only "
                                             "implemented using the Trotter "
                                             "exponential solver.")
        if isinstance(ham, hamiltonians.TrotterHamiltonian):
            ham.circuit(dt, accelerators, memory_device)
        self.solver = solvers.factory[solver](self.dt, hamiltonian)

        self.callbacks = callbacks
        self.accelerators = accelerators
        self.state_cls = states.VectorState
        self.normalize_state = self._create_normalize_state(solver)
        self.calculate_callbacks = self._create_calculate_callbacks(
            accelerators, memory_device)

    def _create_normalize_state(self, solver):
        if "rk" in solver:
            norm = Norm()
            log.info('Normalizing state during RK solution.')
            return lambda s: s / K.cast(norm(s), dtype=s.dtype)
        else:
            return lambda s: s

    def _create_calculate_callbacks(self, accelerators, memory_device):
        def calculate_callbacks(state):
            for callback in self.callbacks:
                callback.append(callback(state))

        if accelerators is None:
            return calculate_callbacks

        def calculate_callbacks_distributed(state):
            with K.device(memory_device):
                if not isinstance(state, K.tensor_types):
                    state = state.tensor
                calculate_callbacks(state)

        return calculate_callbacks_distributed

    def execute(self, final_time, start_time=0.0, initial_state=None):
        """Runs unitary evolution for a given total time.

        Args:
            final_time (float): Final time of evolution.
            start_time (float): Initial time of evolution. Defaults to t=0.
            initial_state (np.ndarray): Initial state of the evolution.

        Returns:
            Final state vector a ``tf.Tensor`` or a
            :class:`qibo.tensorflow.distutils.DistributedState` when a
            distributed execution is used.
        """
        state = self.get_initial_state(initial_state)
        self.solver.t = start_time
        nsteps = int((final_time - start_time) / self.solver.dt)
        self.calculate_callbacks(state)
        for _ in range(nsteps):
            state = self.solver(state)
            if self.callbacks:
                state = self.normalize_state(state)
                self.calculate_callbacks(state)
        state = self.normalize_state(state)
        return state

    def __call__(self, final_time, start_time=0.0, initial_state=None):
        """Equivalent to :meth:`qibo.models.StateEvolution.execute`."""
        return self.execute(final_time, start_time, initial_state)

    def get_initial_state(self, state=None):
        """"""
        if state is None:
            raise_error(ValueError, "StateEvolution cannot be used without "
                                    "initial state.")
        if self.accelerators is None:
            return circuit.Circuit.get_initial_state(self, state)
        else:
            c = self.solver.hamiltonian(0).circuit(self.solver.dt)
            return c.get_initial_state(state)


class AdiabaticEvolution(StateEvolution):
    """Adiabatic evolution of a state vector under the following Hamiltonian:

    .. math::
        H(t) = (1 - s(t)) H_0 + s(t) H_1

    Args:
        h0 (:class:`qibo.abstractions.hamiltonians.Hamiltonian`): Easy Hamiltonian.
        h1 (:class:`qibo.abstractions.hamiltonians.Hamiltonian`): Problem Hamiltonian.
            These Hamiltonians should be time-independent.
        s (callable): Function of time that defines the scheduling of the
            adiabatic evolution. Can be either a function of time s(t) or a
            function with two arguments s(t, p) where p corresponds to a vector
            of parameters to be optimized.
        dt (float): Time step to use for the numerical integration of
            Schrondiger's equation.
        solver (str): Solver to use for integrating Schrodinger's equation.
        callbacks (list): List of callbacks to calculate during evolution.
        accelerators (dict): Dictionary of devices to use for distributed
            execution. See :class:`qibo.tensorflow.distcircuit.DistributedCircuit`
            for more details. This option is available only when the Trotter
            decomposition is used for the time evolution.
        memory_device (str): Name of device where the full state will be saved.
            Relevant only for distributed execution (when ``accelerators`` is
            given).
    """
    ATOL = 1e-7 # Tolerance for checking s(0) = 0 and s(T) = 1.

    def __init__(self, h0, h1, s, dt, solver="exp", callbacks=[],
                 accelerators=None, memory_device="/CPU:0"):
        if not issubclass(type(h0), hamiltonians.HAMILTONIAN_TYPES):
            raise_error(TypeError, "h0 should be a hamiltonians.Hamiltonian "
                                   "object but is {}.".format(type(h0)))
        if type(h1) != type(h0):
            raise_error(TypeError, "h1 should be of the same type {} of h0 but "
                                   "is {}.".format(type(h0), type(h1)))
        if h0.nqubits != h1.nqubits:
            raise_error(ValueError, "H0 has {} qubits while H1 has {}."
                                    "".format(h0.nqubits, h1.nqubits))
        if isinstance(h0, hamiltonians.TrotterHamiltonian):
            if not h1.is_compatible(h0):
                h0 = h1.make_compatible(h0)
        super(AdiabaticEvolution, self).__init__(h0, dt, solver, callbacks,
                                                 accelerators, memory_device)
        self.h0 = h0
        self.h1 = h1

        # Set evolution model to "Gap" callback if one exists
        for callback in self.callbacks:
            if isinstance(callback, Gap):
                callback.evolution = self

        # Flag that remembers if ``set_hamiltonian`` have not been called
        self.set_hamiltonian_flag = True

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
        if abs(s0) > self.ATOL:
            raise_error(ValueError, f"s(0) should be 0 but is {s0}.")
        s1 = f(1)
        if abs(s1 - 1) > self.ATOL:
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

    def set_hamiltonian(self, total_time):
        self.set_hamiltonian_flag = False
        def hamiltonian(t):
            # Disable warning that ``schedule`` is not Callable
            st = self.schedule(t / total_time) # pylint: disable=E1102
            return self.h0 * (1 - st) + self.h1 * st
        self.solver.hamiltonian = hamiltonian

    def hamiltonian(self, t=None, total_time=None):
        """Returns the adiabatic evolution Hamiltonian at a given time.

        Args:
            t (float): Time to calculate the Hamiltonian. If no time is given
                the current time set in the solver is used.
            total_time (float): Total time of adiabatic evolution. Required
                only if the user wants to access the Hamiltonian before
                executing the model.

        Returns:
            A :class:`qibo.abstractions.hamiltonians.Hamiltonian` object representing
            the adiabatic evolution Hamiltonian at time ``t``.
        """
        if total_time is not None:
            self.set_hamiltonian(total_time)
        else:
            if self.set_hamiltonian_flag:
                raise_error(RuntimeError, "Cannot access adiabatic evolution "
                                          "Hamiltonian before setting the "
                                          "the total evolution time.")
        if t is None or t == self.solver.t:
            return self.solver.current_hamiltonian
        return self.solver.hamiltonian(t)

    def get_initial_state(self, state=None):
        """Casts initial state as a tensor.

        If initial state is not given the ground state of ``h0`` is used, which
        is the common practice in adiabatic evolution.
        """
        if state is None:
            if self.accelerators is None:
                return self.h0.ground_state()
            else:
                from qibo.core.states import DistributedState
                c = self.hamiltonian(0).circuit(self.solver.dt)
                state = DistributedState.plus_state(c)
                return c.get_initial_state(state)
        return super(AdiabaticEvolution, self).get_initial_state(state)

    @staticmethod
    def _loss(params, adiabatic_evolution, h1, opt_messages, opt_history):
        """Expectation value of H1 for a choice of scheduling parameters.

        Returns a ``tf.Tensor``.
        """
        adiabatic_evolution.set_parameters(params)
        final_state = super(AdiabaticEvolution, adiabatic_evolution).execute(params[-1])
        loss = h1.expectation(final_state, normalize=True)
        if opt_messages:
            opt_history["params"].append(params)
            opt_history["loss"].append(loss)
            log.info(f"Params: {params}  -  <H1> = {loss}")
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
        self.opt_messages = messages
        if method == "sgd":
            loss = self._loss
        else:
            loss = lambda p, ae, h1, msg, hist: self._loss(p, ae, h1, msg, hist).numpy()

        result, parameters, extra = optimizers.optimize(loss, initial_parameters,
                                                 args=(self, self.h1, self.opt_messages, self.opt_history),
                                                 method=method, options=options)
        if isinstance(parameters, K.tensor_types) and not len(parameters.shape): # pragma: no cover
            # some optimizers like ``Powell`` return number instead of list
            parameters = [parameters]
        self.set_parameters(parameters)
        return result, parameters, extra
