import numpy as np

from qibo.config import raise_error
from qibo.models.evolution import StateEvolution
from qibo.models.utils import vqe_loss


class VQE:
    """This class implements the variational quantum eigensolver algorithm.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): Circuit that
            implements the variaional ansatz.
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): Hamiltonian object.

    Example:
        .. testcode::

            import numpy as np

            from qibo import Circuit, gates
            from qibo.hamiltonians import XXZ
            from qibo.models import VQE

            # create circuit ansatz for two qubits
            circuit = Circuit(2)
            circuit.add(gates.RY(0, theta=0))
            # create XXZ Hamiltonian for two qubits
            hamiltonian = XXZ(2)
            # create VQE model for the circuit and Hamiltonian
            vqe = VQE(circuit, hamiltonian)
            # optimize using random initial variational parameters
            initial_parameters = np.random.uniform(0, 2, 1)
            vqe.minimize(initial_parameters)
    """

    from qibo import optimizers

    def __init__(self, circuit, hamiltonian):
        """Initialize circuit ansatz and hamiltonian."""
        self.circuit = circuit
        self.hamiltonian = hamiltonian
        self.backend = hamiltonian.backend

    def minimize(
        self,
        initial_state,
        method="Powell",
        loss_func=None,
        jac=None,
        hess=None,
        hessp=None,
        bounds=None,
        constraints=(),
        tol=None,
        callback=None,
        options=None,
        compile=False,
        processes=None,
    ):
        """Search for parameters which minimizes the hamiltonian expectation.

        Args:
            initial_state (array): a initial guess for the parameters of the
                variational circuit.
            method (str): the desired minimization method.
                See :meth:`qibo.optimizers.optimize` for available optimization
                methods.
            loss (callable): loss function, the default one is :func:`qibo.models.utils.vqe_loss`.
            jac (dict): Method for computing the gradient vector for scipy optimizers.
            hess (dict): Method for computing the hessian matrix for scipy optimizers.
            hessp (callable): Hessian of objective function times an arbitrary
                vector for scipy optimizers.
            bounds (sequence or Bounds): Bounds on variables for scipy optimizers.
            constraints (dict): Constraints definition for scipy optimizers.
            tol (float): Tolerance of termination for scipy optimizers.
            callback (callable): Called after each iteration for scipy optimizers.
            options (dict): a dictionary with options for the different optimizers.
            compile (bool): whether the TensorFlow graph should be compiled.
            processes (int): number of processes when using the paralle BFGS method.

        Return:
            The final expectation value.
            The corresponding best parameters.
            The optimization result object. For scipy methods it returns
            the ``OptimizeResult``, for ``'cma'`` the ``CMAEvolutionStrategy.result``,
            and for ``'sgd'`` the options used during the optimization.
        """
        if loss_func is None:
            loss_func = vqe_loss
        if compile:
            loss = self.hamiltonian.backend.compile(loss_func)
        else:
            loss = loss_func

        if method == "cma":
            # TODO: check if we can use this shortcut
            # dtype = getattr(self.hamiltonian.backend.np, self.hamiltonian.backend._dtypes.get('DTYPE'))
            dtype = self.hamiltonian.backend.np.float64
            loss = (
                (lambda p, c, h: loss_func(p, c, h).item())
                if str(dtype) == "torch.float64"
                else (lambda p, c, h: dtype(loss_func(p, c, h)))
            )
        elif method != "sgd":
            loss = lambda p, c, h: self.hamiltonian.backend.to_numpy(loss_func(p, c, h))
        result, parameters, extra = self.optimizers.optimize(
            loss,
            initial_state,
            args=(self.circuit, self.hamiltonian),
            method=method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            callback=callback,
            options=options,
            compile=compile,
            processes=processes,
            backend=self.hamiltonian.backend,
        )
        self.circuit.set_parameters(parameters)
        return result, parameters, extra

    def energy_fluctuation(self, state):
        """
        Evaluate energy fluctuation

        .. math::
            \\Xi_{k}(\\mu) = \\sqrt{\\langle\\mu|\\hat{H}^2|\\mu\\rangle - \\langle\\mu|\\hat{H}|\\mu\\rangle^2} \\,

        for a given state :math:`|\\mu\\rangle`.

        Args:
            state (np.ndarray): quantum state to be used to compute the energy fluctuation with H.
        """
        return self.hamiltonian.energy_fluctuation(state)


class AAVQE:
    """This class implements the Adiabatically Assisted Variational Quantum Eigensolver
    algorithm. See https://arxiv.org/abs/1806.02287.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): variational ansatz.
        easy_hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): initial Hamiltonian object.
        problem_hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): problem Hamiltonian object.
        s (callable): scheduling function of time that defines the adiabatic
            evolution. It must verify boundary conditions: s(0) = 0 and s(1) = 1.
        nsteps (float): number of steps of the adiabatic evolution.
        t_max (float): total time evolution.
        bounds_tolerance (float): tolerance for checking s(0) = 0 and s(1) = 1.
        time_tolerance (float): tolerance for checking if time is greater than t_max.

    Example:
        .. testcode::

            import numpy as np

            from qibo import Circuit, gates
            from qibo.hamiltonians import X, XXZ
            from qibo.models import AAVQE

            # create circuit ansatz for two qubits
            circuit = Circuit(2)
            circuit.add(gates.RY(0, theta=0))
            circuit.add(gates.RY(1, theta=0))
            # define the easy and the problem Hamiltonians.
            easy_hamiltonian = X(2)
            problem_hamiltonian = XXZ(2)
            # define a scheduling function with only one parameter
            # and boundary conditions s(0) = 0, s(1) = 1
            s = lambda t: t
            # create AAVQE model
            aavqe = AAVQE(
                circuit,
                easy_hamiltonian,
                problem_hamiltonian,
                s,
                nsteps=10,
                t_max=1
            )
            # optimize using random initial variational parameters
            np.random.seed(0)
            initial_parameters = np.random.uniform(0, 2*np.pi, 2)
            ground_energy, params = aavqe.minimize(initial_parameters)
    """

    def __init__(
        self,
        circuit,
        easy_hamiltonian,
        problem_hamiltonian,
        s,
        nsteps=10,
        t_max=1,
        bounds_tolerance=1e-7,
        time_tolerance=1e-7,
    ):
        if nsteps <= 0:  # pragma: no cover
            raise_error(
                ValueError,
                f"Number of steps nsteps should be positive but is {nsteps}.",
            )
        if t_max <= 0:  # pragma: no cover
            raise_error(
                ValueError,
                f"Maximum time t_max should be positive but is {t_max}.",
            )
        if easy_hamiltonian.nqubits != problem_hamiltonian.nqubits:  # pragma: no cover
            raise_error(
                ValueError,
                f"The easy Hamiltonian has {easy_hamiltonian.nqubits} qubits "
                + f"while problem Hamiltonian has {problem_hamiltonian.nqubits}.",
            )

        self.ATOL = bounds_tolerance
        self.ATOL_TIME = time_tolerance

        self._circuit = circuit
        self._h0 = easy_hamiltonian
        self._h1 = problem_hamiltonian
        self._nsteps = nsteps
        self._t_max = t_max
        self._dt = 1.0 / (nsteps - 1)

        self._schedule = None
        nparams = s.__code__.co_argcount
        if not nparams == 1:  # pragma: no cover
            raise_error(
                ValueError,
                "Scheduling function must take only one argument,"
                + f"but the function proposed takes {nparams}.",
            )
        self.set_schedule(s)

    def set_schedule(self, func):
        """Set scheduling function s(t) as func."""
        # check boundary conditions
        s0 = func(0)
        if abs(s0) > self.ATOL:  # pragma: no cover
            raise_error(ValueError, f"s(0) should be 0 but it is {s0}.")
        s1 = func(1)
        if abs(s1 - 1) > self.ATOL:  # pragma: no cover
            raise_error(ValueError, f"s(1) should be 1 but it is {s1}.")
        self._schedule = func

    def schedule(self, t):
        """Returns scheduling function evaluated at time t: s(t/Tmax)."""
        if self._schedule is None:  # pragma: no cover
            raise_error(ValueError, "Cannot access scheduling before it is set.")
        if (t - self._t_max) > self.ATOL_TIME:  # pragma: no cover
            raise_error(
                ValueError,
                f"t cannot be greater than {self._t_max}, but it is {t}.",
            )

        s = self._schedule(t / self._t_max)
        if (abs(s) - 1) > self.ATOL:  # pragma: no cover
            raise_error(ValueError, f"s cannot be greater than 1 but it is {s}.")
        return s

    def hamiltonian(self, t):
        """Returns the adiabatic evolution Hamiltonian at a given time."""
        if (t - self._t_max) > self.ATOL:  # pragma: no cover
            raise_error(
                ValueError,
                f"t cannot be greater than {self._t_max}, but it is {t}.",
            )
        # boundary conditions  s(0)=0, s(total_time)=1
        st = self.schedule(t)
        return self._h0 * (1 - st) + self._h1 * st

    def minimize(
        self,
        params,
        method="BFGS",
        jac=None,
        hess=None,
        hessp=None,
        bounds=None,
        constraints=(),
        tol=None,
        options=None,
        compile=False,
        processes=None,
    ):
        """
        Performs minimization to find the ground state of the problem Hamiltonian.

        Args:
            params (np.ndarray or list): initial guess for the parameters of the variational circuit.
            method (str): optimizer to employ.
            jac (dict): Method for computing the gradient vector for scipy optimizers.
            hess (dict): Method for computing the hessian matrix for scipy optimizers.
            hessp (callable): Hessian of objective function times an arbitrary
                            vector for scipy optimizers.
            bounds (sequence or Bounds): Bounds on variables for scipy optimizers.
            constraints (dict): Constraints definition for scipy optimizers.
            tol (float): Tolerance of termination for scipy optimizers.
            options (dict): a dictionary with options for the different optimizers.
            compile (bool): whether the TensorFlow graph should be compiled.
            processes (int): number of processes when using the parallel BFGS method.
        """
        from qibo import models

        t = 0.0
        while (t - self._t_max) <= self.ATOL_TIME:
            H = self.hamiltonian(t)
            vqe = models.VQE(self._circuit, H)
            best, params, _ = vqe.minimize(
                params,
                method=method,
                jac=jac,
                hess=hess,
                hessp=hessp,
                bounds=bounds,
                constraints=constraints,
                tol=tol,
                options=options,
                compile=compile,
                processes=processes,
            )
            t += self._dt
        return best, params


class QAOA:
    """Quantum Approximate Optimization Algorithm (QAOA) model.

    The QAOA is introduced in `arXiv:1411.4028 <https://arxiv.org/abs/1411.4028>`_.

    Args:
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): problem Hamiltonian
            whose ground state is sought.
        mixer (:class:`qibo.hamiltonians.Hamiltonian`): mixer Hamiltonian.
            Must be of the same type and act on the same number of qubits as ``hamiltonian``.
            If ``None``, :class:`qibo.hamiltonians.X` is used.
        solver (str): solver used to apply the exponential operators.
            Default solver is 'exp' (:class:`qibo.solvers.Exponential`).
        callbacks (list): List of callbacks to calculate during evolution.
        accelerators (dict): Dictionary of devices to use for distributed
            execution. This option is available only when ``hamiltonian``
            is a :class:`qibo.hamiltonians.SymbolicHamiltonian`.

    Example:
        .. testcode::

            import numpy as np
            from qibo import models, hamiltonians
            # create XXZ Hamiltonian for four qubits
            hamiltonian = hamiltonians.XXZ(4)
            # create QAOA model for this Hamiltonian
            qaoa = models.QAOA(hamiltonian)
            # optimize using random initial variational parameters
            # and default options and initial state
            initial_parameters = 0.01 * np.random.random(4)
            best_energy, final_parameters, extra = qaoa.minimize(initial_parameters, method="BFGS")
    """

    from qibo import hamiltonians, optimizers

    def __init__(
        self, hamiltonian, mixer=None, solver="exp", callbacks=[], accelerators=None
    ):
        from qibo.hamiltonians.abstract import AbstractHamiltonian

        # list of QAOA variational parameters (angles)
        self.params = None
        # problem hamiltonian
        if not isinstance(hamiltonian, AbstractHamiltonian):
            raise_error(TypeError, f"Invalid Hamiltonian type {type(hamiltonian)}.")
        self.hamiltonian = hamiltonian
        self.nqubits = hamiltonian.nqubits
        # mixer hamiltonian (default = -sum(sigma_x))
        if mixer is None:
            trotter = isinstance(
                self.hamiltonian, self.hamiltonians.SymbolicHamiltonian
            )
            self.mixer = self.hamiltonians.X(
                self.nqubits, dense=not trotter, backend=self.hamiltonian.backend
            )
        else:
            if type(mixer) != type(hamiltonian):
                raise_error(
                    TypeError,
                    f"Given Hamiltonian is of type {type(hamiltonian)} "
                    + f"while mixer is of type {type(mixer)}.",
                )
            if mixer.nqubits != hamiltonian.nqubits:
                raise_error(
                    ValueError,
                    f"Given Hamiltonian acts on {hamiltonian.nqubits} qubits "
                    + f"while mixer acts on {mixer.nqubits}.",
                )
            self.mixer = mixer

        # create circuits for Trotter Hamiltonians
        if accelerators is not None and (
            not isinstance(self.hamiltonian, self.hamiltonians.SymbolicHamiltonian)
            or solver != "exp"
        ):
            raise_error(
                NotImplementedError,
                "Distributed QAOA is implemented "
                + "only with SymbolicHamiltonian and "
                + "exponential solver.",
            )
        if isinstance(self.hamiltonian, self.hamiltonians.SymbolicHamiltonian):
            self.hamiltonian.circuit(1e-2, accelerators)
            self.mixer.circuit(1e-2, accelerators)

        # evolution solvers
        from qibo.solvers import get_solver

        self.ham_solver = get_solver(solver, 1e-2, self.hamiltonian)
        self.mix_solver = get_solver(solver, 1e-2, self.mixer)

        self.callbacks = callbacks
        self.backend = (
            hamiltonian.backend
        )  # to avoid error with _create_calculate_callbacks
        self.accelerators = accelerators
        self.normalize_state = StateEvolution._create_normalize_state(self, solver)
        self.calculate_callbacks = StateEvolution._create_calculate_callbacks(
            self, accelerators
        )

    def set_parameters(self, p):
        """Sets the variational parameters.

        Args:
            p (np.ndarray): 1D-array holding the new values for the variational
                parameters. Length should be an even number.
        """
        self.params = p

    def _apply_exp(self, state, solver, p):
        """Helper method for ``execute``."""
        solver.dt = p
        state = solver(state)
        if self.callbacks:
            state = self.normalize_state(state)
            self.calculate_callbacks(state)
        return state

    def execute(self, initial_state=None):
        """Applies the QAOA exponential operators to a state.

        Args:
            initial_state (np.ndarray): Initial state vector.

        Returns:
            State vector after applying the QAOA exponential gates.
        """
        if initial_state is None:
            state = self.hamiltonian.backend.plus_state(self.nqubits)
        else:
            state = self.hamiltonian.backend.cast(initial_state)

        self.calculate_callbacks(state)
        n = int(self.params.shape[0])
        for i in range(n // 2):
            state = self._apply_exp(state, self.ham_solver, self.params[2 * i])
            state = self._apply_exp(state, self.mix_solver, self.params[2 * i + 1])
        return self.normalize_state(state)

    def __call__(self, initial_state=None):
        """Equivalent to :meth:`qibo.models.QAOA.execute`."""
        return self.execute(initial_state)

    def minimize(
        self,
        initial_p,
        initial_state=None,
        method="Powell",
        loss_func=None,
        loss_func_param=dict(),
        jac=None,
        hess=None,
        hessp=None,
        bounds=None,
        constraints=(),
        tol=None,
        callback=None,
        options=None,
        compile=False,
        processes=None,
    ):
        """Optimizes the variational parameters of the QAOA. A few loss functions are
        provided for QAOA optimizations such as expected value (default), CVar which is introduced in
        `Quantum 4, 256 <https://quantum-journal.org/papers/q-2020-04-20-256/>`_, and
        Gibbs loss function which is introduced in
        `PRR 2, 023074 (2020) <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023074>`_.

        Args:
            initial_p (np.ndarray): initial guess for the parameters.
            initial_state (np.ndarray): initial state vector of the QAOA.
            method (str): the desired minimization method.
                See :meth:`qibo.optimizers.optimize` for available optimization
                methods.
            loss_func (function): the desired loss function. If it is None, the expectation is used.
            loss_func_param (dict): a dictionary to pass in the loss function parameters.
            jac (dict): Method for computing the gradient vector for scipy optimizers.
            hess (dict): Method for computing the hessian matrix for scipy optimizers.
            hessp (callable): Hessian of objective function times an arbitrary
                vector for scipy optimizers.
            bounds (sequence or Bounds): Bounds on variables for scipy optimizers.
            constraints (dict): Constraints definition for scipy optimizers.
            tol (float): Tolerance of termination for scipy optimizers.
            callback (callable): Called after each iteration for scipy optimizers.
            options (dict): a dictionary with options for the different optimizers.
            compile (bool): whether the TensorFlow graph should be compiled.
            processes (int): number of processes when using the paralle BFGS method.

        Return:
            The final energy (expectation value of the ``hamiltonian``).
            The corresponding best parameters.
            The optimization result object. For scipy methods it
            returns the ``OptimizeResult``, for ``'cma'`` the
            ``CMAEvolutionStrategy.result``, and for ``'sgd'``
            the options used during the optimization.

        Example:
            .. testcode::

                from qibo import hamiltonians
                from qibo.models.utils import cvar, gibbs

                h = hamiltonians.XXZ(3)
                qaoa = models.QAOA(h)
                initial_p = [0.314, 0.22, 0.05, 0.59]
                best, params, _ = qaoa.minimize(initial_p)
                best, params, _ = qaoa.minimize(initial_p, loss_func=cvar, loss_func_param={'alpha':0.1})
                best, params, _ = qaoa.minimize(initial_p, loss_func=gibbs, loss_func_param={'eta':0.1})

        """
        if len(initial_p) % 2 != 0:
            raise_error(
                ValueError,
                "Initial guess for the parameters must "
                + "contain an even number of values but "
                + "contains {len(initial_p)}.",
            )

        def _loss(params, qaoa, hamiltonian, state):
            if state is not None:
                state = hamiltonian.backend.cast(state, copy=True)
            qaoa.set_parameters(params)
            state = qaoa(state)
            if loss_func is None:
                return hamiltonian.expectation(state)
            else:
                func_hyperparams = {
                    key: loss_func_param[key]
                    for key in loss_func_param
                    if key in loss_func.__code__.co_varnames
                }
                param = {**func_hyperparams, "hamiltonian": hamiltonian, "state": state}

                return loss_func(**param)

        if method == "sgd":
            loss = lambda p, c, h, s: _loss(self.hamiltonian.backend.cast(p), c, h, s)
        else:
            loss = lambda p, c, h, s: self.hamiltonian.backend.to_numpy(
                _loss(p, c, h, s)
            )

        result, parameters, extra = self.optimizers.optimize(
            loss,
            initial_p,
            args=(self, self.hamiltonian, initial_state),
            method=method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            callback=callback,
            options=options,
            compile=compile,
            processes=processes,
            backend=self.backend,
        )
        self.set_parameters(parameters)
        return result, parameters, extra


class FALQON(QAOA):
    """Feedback-based ALgorithm for Quantum OptimizatioN (FALQON) model.

    The FALQON is introduced in `arXiv:2103.08619 <https://arxiv.org/abs/2103.08619>`_.
    It inherits the QAOA class.

    Args:
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): problem Hamiltonian
            whose ground state is sought.
        mixer (:class:`qibo.hamiltonians.Hamiltonian`): mixer Hamiltonian.
            If ``None``, :class:`qibo.hamiltonians.X` is used.
        solver (str): solver used to apply the exponential operators.
            Default solver is 'exp' (:class:`qibo.solvers.Exponential`).
        callbacks (list): List of callbacks to calculate during evolution.
        accelerators (dict): Dictionary of devices to use for distributed
            execution. This option is available only when ``hamiltonian``
            is a :class:`qibo.hamiltonians.SymbolicHamiltonian`.

    Example:
        .. testcode::

            import numpy as np
            from qibo import models, hamiltonians
            # create XXZ Hamiltonian for four qubits
            hamiltonian = hamiltonians.XXZ(4)
            # create FALQON model for this Hamiltonian
            falqon = models.FALQON(hamiltonian)
            # optimize using random initial variational parameters
            # and default options and initial state
            delta_t = 0.01
            max_layers = 3
            best_energy, final_parameters, extra = falqon.minimize(delta_t, max_layers)
    """

    def __init__(
        self, hamiltonian, mixer=None, solver="exp", callbacks=[], accelerators=None
    ):
        super().__init__(hamiltonian, mixer, solver, callbacks, accelerators)
        self.evol_hamiltonian = 1j * (
            self.hamiltonian @ self.mixer - self.mixer @ self.hamiltonian
        )

    def minimize(
        self, delta_t, max_layers, initial_state=None, tol=None, callback=None
    ):
        """Optimizes the variational parameters of the FALQON.

        Args:
            delta_t (float): initial guess for the time step. A too large delta_t will make the algorithm fail.
            max_layers (int): maximum number of layers allowed for the FALQON.
            initial_state (np.ndarray): initial state vector of the FALQON.
            tol (float): Tolerance of energy change. If not specified, no check is done.
            callback (callable): Called after each iteration for scipy optimizers.
            options (dict): a dictionary with options for the different optimizers.

        Return:
            The final energy (expectation value of the ``hamiltonian``).
            The corresponding best parameters.
            extra: variable with historical data for the energy and callbacks.
        """
        parameters = np.array([delta_t, 0])

        def _loss(params, falqon, hamiltonian):
            falqon.set_parameters(params)
            state = falqon(initial_state)
            return hamiltonian.expectation(state)

        energy = [np.inf]
        callback_result = []
        for _ in range(1, max_layers + 1):
            beta = self.hamiltonian.backend.to_numpy(
                _loss(parameters, self, self.evol_hamiltonian)
            )

            if tol is not None:
                energy.append(
                    self.hamiltonian.backend.to_numpy(
                        _loss(parameters, self, self.hamiltonian)
                    )
                )
                if abs(energy[-1] - energy[-2]) < tol:
                    break

            if callback is not None:
                callback_result.append(callback(parameters))

            parameters = np.concatenate([parameters, [delta_t, delta_t * beta]])

        self.set_parameters(parameters)
        final_loss = _loss(parameters, self, self.hamiltonian)
        extra = {"energies": energy, "callbacks": callback_result}
        return final_loss, parameters, extra
