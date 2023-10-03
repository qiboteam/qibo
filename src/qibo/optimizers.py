import time
from datetime import datetime

import numpy as np

from qibo import backends


class Optimizer:
    """Parent optimizer class

    Args:
        initial_parameters (np.ndarray): array with initial values for gate parameters
        args (tuple): tuple containing loss function arguments
        loss (function): loss function to train on
        save (bool): Flag to set logging on
        loss_function (function): loss function to train on
        params (np.ndarray): array with current values for gate parameters
        backend (qibo.backends.GlobalBackend): backend on which to run circuit
        simulation_start (time.time): simulation start time
        ftime (time.time): start time variable for timing internal processes
        etime (time.time): end time variable for timing internal processes
        name (str): name of current optimisation process
        iteration (int): training iteration number
        initparams (np.ndarray or list): initial parameters
    """

    def __init__(self, initial_parameters, args=(), loss=None, save=False):
        # saving to class objects
        self.loss_function = loss
        if not isinstance(args, tuple):
            self.args = (args,)
        else:
            self.args = args

        self.params = initial_parameters
        self.backend = backends.GlobalBackend()

        # logging
        self.simulation_start = time.time()
        self.ftime = time.time()
        self.etime = None
        self.name = f'Run_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

        self.iteration = 0
        self.save = save

        if not isinstance(initial_parameters, np.ndarray) and not isinstance(
            initial_parameters, list
        ):
            raise TypeError(
                "Parameters must be a list of Parameter objects or a numpy array"
            )

    def set_options(self, updates):
        """Updates self.options dictionary"""
        self.options.update(updates)

    def fun(self, x):
        """Wrapper function to save and preprocess gate parameters

        Args:
            x (np.array): circuit parameters

        Returns:
            (float): loss value"""

        val = self.loss_function(x, *self.args)

        # timing
        self.etime = time.time()
        duration = self.etime - self.ftime
        self.ftime = self.etime

        # saving
        if self.save:
            self.file.write(
                f"Iteration {self.iteration} | loss: {val} | duration: {duration}\n"
            )

        self.iteration += 1
        return val

    def cleanup(self):
        """Cleans up log file and closes it"""

        self.file.write(f"Total simulation time: {time.time()-self.simulation_start}\n")
        self.file.close()


class CMAES(Optimizer):
    """Genetic optimizer based on `pycma <https://github.com/CMA-ES/pycma>`_.

    Example:
    .. code-block:: python

        from qibo.models.variational import VariationalCircuit
        from qibo import gates
        from qibo.derivative import create_hamiltonian
        from qibo.optimizers import CMAES

        def black_box_loss(params, circuit, hamiltonian):
            circuit.set_parameters(params)
            state = circuit().state()

            results = hamiltonian.expectation(state)

            return (results - 0.4) ** 2

        c = VariationalCircuit(1, density_matrix=True)

        c.add(gates.H(q=0))
        for _ in range(3):
            c.add(gates.RZ(q=0, theta=theta))
            c.add(gates.RY(q=0, theta=theta))
        c.add(gates.M(0))

        hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

        parameters = np.array([0.1] * 6)

        optimizer = CMAES(
            initial_parameters=parameters,
            args=(circuit, hamiltonian),
            loss=simple_loss
        )

        fbest, xbest, r, it = optimizer.fit()

    Args:
        intial_parameters (np.ndarray): array with initial values for gate parameters
        hamiltonian (qibo.hamiltonians.SymbolicHamiltonian): hamiltonian applied to final circuit state
        args (tuple): tuple containing loss function arguments. Circuit must be first argument.
        loss (function): loss function to train on
        save (bool): Flag to set logging on
    """

    def __init__(self, initial_parameters, args=(), loss=None, save=False, **kwargs):
        super().__init__(initial_parameters, args, loss, save)

        self.options = {}
        self.set_options(kwargs)

        # logging
        self.filename = f"results/cma_{self.name}.txt"
        if self.save:
            self.file = open(self.filename, "w")

    def fit(self):
        """Performs the optimizations.

        Returns:
            (float): best loss value
            (np.ndarray): best parameter values
            (cma.evolution_strategy.CMAEvolutionStrategy): full CMA Evolution Strategy object
        """

        import cma

        r = cma.fmin2(
            self.fun,
            self.params,
            sigma0=1.7,
            **self.options,
        )

        return r[1].result.fbest, r[1].result.xbest, r, self.iteration


class Powell(Optimizer):
    def __init__(self, initial_parameters, args=(), loss=None, save=False, **kwargs):
        """Powell optimization approaches based on ``scipy.optimize.minimize``.

        For more details check the `scipy` documentation <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html>`_.

        Example:
        .. code-block:: python

            from qibo.models.variational import VariationalCircuit
            from qibo import gates
            from qibo.derivative import create_hamiltonian
            from qibo.optimizers import Powell

            def black_box_loss(params, circuit, hamiltonian):
                circuit.set_parameters(params)
                state = circuit().state()

                results = hamiltonian.expectation(state)

                return (results - 0.4) ** 2

            c = VariationalCircuit(1, density_matrix=True)

            c.add(gates.H(q=0))
            for _ in range(3):
                c.add(gates.RZ(q=0, theta=theta))
                c.add(gates.RY(q=0, theta=theta))
            c.add(gates.M(0))

            hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

            parameters = np.array([0.1] * 6)

            optimizer = Powell(
                initial_parameters=parameters,
                args=(circuit, hamiltonian),
                loss=simple_loss
            )

            fbest, xbest, r, it = optimizer.fit()

        Args:
            loss (callable): Loss as a function of variational parameters to be
                optimized.
            initial_parameters (np.ndarray): Initial guess for the variational
                parameters.
            args (tuple): optional arguments for the loss function.
            method (str): Name of method supported by ``scipy.optimize.minimize`` and ``'parallel_L-BFGS-B'`` for
                a parallel version of L-BFGS-B algorithm.
            jac (dict): Method for computing the gradient vector for scipy optimizers.
            hess (dict): Method for computing the hessian matrix for scipy optimizers.
            hessp (callable): Hessian of objective function times an arbitrary
                vector for scipy optimizers.
            bounds (sequence or Bounds): Bounds on variables for scipy optimizers.
            constraints (dict): Constraints definition for scipy optimizers.
            tol (float): Tolerance of termination for scipy optimizers.
            callback (callable): Called after each iteration for scipy optimizers.
            options (dict): Dictionary with options accepted by
                ``scipy.optimize.minimize``.
            processes (int): number of processes when using the parallel BFGS method.
        """
        super().__init__(initial_parameters, args, loss, save)

        self.options = {
            "method": "Powell",
            "jac": None,
            "hess": None,
            "hessp": None,
            "bounds": None,
            "constraints": (),
            "tol": None,
            "callback": None,
            "options": {"disp": True, "maxfevals": 100},
            "processes": None,
            "backend": None,
        }
        self.set_options(kwargs)

        # logging
        self.filename = f"results/powell_{self.name}.txt"
        if self.save:
            self.file = open(self.filename, "w")

    def fit(self):
        """Performs the optimizations.

        Returns:
            (float): best loss value
            (np.ndarray): best parameter values
            (scipy.optimize.OptimizeResult): full scipy OptimizeResult object
            (int): iteration number
        """

        from scipy.optimize import minimize

        r = minimize(
            self.fun,
            self.params,
            method=self.options["method"],
            jac=self.options["jac"],
            hess=self.options["hess"],
            hessp=self.options["hessp"],
            bounds=self.options["bounds"],
            constraints=self.options["constraints"],
            tol=self.options["tol"],
            callback=self.options["callback"],
            options=self.options["options"],
        )

        return r.fun, r.x, r, self.iteration


class ParallelBFGS(Optimizer):  # pragma: no cover
    """Computes the L-BFGS-B using parallel evaluation using multiprocessing.
    This implementation here is based on https://doi.org/10.32614/RJ-2019-030.

    Example:
    .. code-block:: python

        from qibo.models.variational import VariationalCircuit
        from qibo import gates
        from qibo.derivative import create_hamiltonian
        from qibo.optimizers import ParallelBFGS

        def black_box_loss(params, circuit, hamiltonian):
            circuit.set_parameters(params)
            state = circuit().state()

            results = hamiltonian.expectation(state)

            return (results - 0.4) ** 2

        c = VariationalCircuit(1, density_matrix=True)

        c.add(gates.H(q=0))
        for _ in range(3):
            c.add(gates.RZ(q=0, theta=theta))
            c.add(gates.RY(q=0, theta=theta))
        c.add(gates.M(0))

        hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

        parameters = np.array([0.1] * 6)

        optimizer = ParallelBFGS(
            initial_parameters=parameters,
            args=(circuit, hamiltonian),
            loss=simple_loss
        )

        fbest, xbest, r, it = optimizer.fit()

    Args:
        function (function): loss function which returns a numpy object.
        args (tuple): optional arguments for the loss function.
        bounds (list): list of bound values for ``scipy.optimize.minimize`` L-BFGS-B.
        callback (function): function callback ``scipy.optimize.minimize`` L-BFGS-B.
        options (dict): follows ``scipy.optimize.minimize`` syntax for L-BFGS-B.
        processes (int): number of processes when using the paralle BFGS method.
    """

    import numpy as np

    def __init__(self, initial_parameters, loss, args=(), save=False, **kwargs):
        super().__init__(initial_parameters, args, loss, save)
        self.function_value = None
        self.jacobian_value = None

        self.options = {
            "xval": None,
            "precision": self.np.finfo("float64").eps,
            "bounds": None,
            "callback": None,
            "options": None,
            "processes": None,
            "save": False,
        }

        self.set_options(kwargs)

        # logging
        self.filename = f"results/bfgs_{self.name}.txt"
        if self.save:
            self.file = open(self.filename, "w")

    def fit(self):
        """Performs the optimizations.

        Returns:
            (float): best loss value
            (np.ndarray): best parameter values
            (scipy.optimize.OptimizeResult): full scipy OptimizeResult object
            (int): iteration number
        """

        from scipy.optimize import minimize

        out = minimize(
            fun=self.fun,
            x0=self.params,
            jac=self.jac,
            method="L-BFGS-B",
            bounds=self.options["bounds"],
            callback=self.options["callback"],
            options=self.options["options"],
        )
        out.hess_inv = out.hess_inv * self.np.identity(len(self.params))
        return out.fun, out.x, out, self.iteration

    @staticmethod
    def _eval_approx(eps_at, fun, x, eps):
        """Approximate evaluation

        Args:
            eps_at (int): parameter index where approximation occurs
            fun (function): loss function
            x (np.ndarray): circuit parameters
            eps (float): approximation delta
        Returns:
            (float): approximated loss value
        """
        if eps_at == 0:
            x_ = x
        else:
            x_ = x.copy()
            if eps_at <= len(x):
                x_[eps_at - 1] += eps
            else:
                x_[eps_at - 1 - len(x)] -= eps
        return fun(x_)

    def evaluate(self, x, eps=1e-8):
        """Handles function evaluation

        Args:
            x (np.ndarray): circuit parameters
            eps (float): approximation delta
        Returns
            (float): loss value
        """
        if not (
            self.options["xval"] is not None
            and all(abs(self.options["xval"] - x) <= self.options["precision"] * 2)
        ):
            eps_at = range(len(x) + 1)
            self.options["xval"] = x.copy()

            def operation(epsi):
                return self._eval_approx(
                    epsi, lambda y: self.loss_function(y, *self.args), x, eps
                )

            from joblib import Parallel, delayed

            ret = Parallel(self.options["processes"], prefer="threads")(
                delayed(operation)(epsi) for epsi in eps_at
            )
            self.function_value = ret[0]
            self.jacobian_value = (ret[1 : (len(x) + 1)] - self.function_value) / eps

    def fun(self, x):
        """Saves and returns loss function value

        Args:
            x (np.ndarray): circuit parameters
        Returns
            (float): loss value
        """

        self.evaluate(x)
        if self.save:
            self.file.write(
                f"Iteration {self.iteration} | loss: {self.function_value}\n"
            )
            self.iteration += 1
        return self.function_value

    def jac(self, x):
        """Evaluates the Jacobian

        Args:
            x (np.ndarray): circuit parameters
        Returns
            (float): jacobian value
        """

        self.evaluate(x)
        return self.jacobian_value


class BasinHopping(Optimizer):
    """Global optimizer based on `scipy.optimize.basinhopping <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html>`_.

    Example:
    .. code-block:: python

        from qibo.models.variational import VariationalCircuit
        from qibo import gates
        from qibo.derivative import create_hamiltonian
        from qibo.optimizers import BasinHopping

        def black_box_loss(params, circuit, hamiltonian):
            circuit.set_parameters(params)
            state = circuit().state()

            results = hamiltonian.expectation(state)

            return (results - 0.4) ** 2

        c = VariationalCircuit(1, density_matrix=True)

        c.add(gates.H(q=0))
        for _ in range(3):
            c.add(gates.RZ(q=0, theta=theta))
            c.add(gates.RY(q=0, theta=theta))
        c.add(gates.M(0))

        hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

        parameters = np.array([0.1] * 6)

        optimizer = BasinHopping(
            initial_parameters=parameters,
            args=(circuit, hamiltonian),
            loss=simple_loss
        )

        fbest, xbest, r, it = optimizer.fit()

    Args:
        intial_parameters (np.ndarray): array with initial values for gate parameters
        args (tuple): tuple containing loss function arguments. Circuit must be first argument.
        loss (function): loss function to train on
        save (bool): Flag to set logging on
    """

    def __init__(self, initial_parameters, args=(), loss=None, save=False, **kwargs):
        super().__init__(initial_parameters, args, loss, save)
        self.args = args
        self.options = kwargs

        # logging
        self.filename = f"results/basin_{self.name}.txt"
        if self.save:
            self.file = open(self.filename, "w")

    def fit(self):
        """Performs the optimizations.

        Returns:
            (float): best loss value
            (np.ndarray): best parameter values
            (scipy.optimize.OptimizeResult): full scipy OptimizeResult object
            (int): iteration number
        """
        from scipy.optimize import basinhopping

        if "options" in self.options:
            options = self.options.pop("options")

        r = basinhopping(
            self.fun,
            self.params,
            niter=1,
            minimizer_kwargs={"options": {"maxfev": 100}},
        )

        return r.fun, r.x, r, self.iteration


class BFGS(Optimizer):
    """BFGS optimization approach based on ``scipy.optimize.minimize``.

    For more details check the `scipy` documentation <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html>`_.

    Example:
    .. code-block:: python

        from qibo.models.variational import VariationalCircuit
        from qibo import gates
        from qibo.derivative import create_hamiltonian
        from qibo.optimizers import BFGS

        def black_box_loss(params, circuit, hamiltonian):
            circuit.set_parameters(params)
            state = circuit().state()

            results = hamiltonian.expectation(state)

            return (results - 0.4) ** 2

        c = VariationalCircuit(1, density_matrix=True)

        c.add(gates.H(q=0))
        for _ in range(3):
            c.add(gates.RZ(q=0, theta=theta))
            c.add(gates.RY(q=0, theta=theta))
        c.add(gates.M(0))

        hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

        parameters = np.array([0.1] * 6)

        optimizer = BFGS(
            initial_parameters=parameters,
            args=(circuit, hamiltonian),
            loss=simple_loss
        )

        fbest, xbest, r, it = optimizer.fit()

    Args:
        function (function): loss function which returns a numpy object.
        args (tuple): optional arguments for the loss function.
        bounds (list): list of bound values for ``scipy.optimize.minimize`` L-BFGS-B.
        callback (function): function callback ``scipy.optimize.minimize`` L-BFGS-B.
        options (dict): follows ``scipy.optimize.minimize`` syntax for L-BFGS-B.
        processes (int): number of processes when using the paralle BFGS method.
    """

    import numpy as np

    def __init__(
        self,
        initial_parameters,
        args=(),
        loss=None,
        jacobian=None,
        save=False,
        **kwargs,
    ):
        super().__init__(initial_parameters, loss=loss, save=True)

        self.save = save
        self.jacobian = jacobian
        self.args = args

        self.options = {
            "xval": None,
            "bounds": None,
            "callback": None,
            "options": {"gtol": 1e-7, "maxiter": 10000},
        }

        self.set_options(kwargs)

        self.function_value = None
        self.jacobian_value = None

        # logging
        self.filename = f"results/bfgs_{self.name}.txt"
        if self.save:
            self.file = open(self.filename, "w")

    def fit(self):
        """Executes parallel L-BFGS-B minimization.
        Args:
            x0 (numpy.array): guess for initial solution.

        Returns:
            scipy.minimize result object
        """
        from scipy.optimize import minimize

        r = minimize(
            fun=self.fun,
            x0=self.params,
            method="BFGS",
            bounds=self.options["bounds"],
            callback=self.options["callback"],
            options=self.options["options"],
        )

        return r.fun, r.x, r, self.iteration

    def fun(self, x):
        val = self.loss_function(x, *self.args)

        # timing
        self.etime = time.time()
        duration = self.etime - self.ftime
        self.ftime = self.etime

        # saving
        if self.save:
            self.file.write(
                f"Iteration {self.iteration} | loss: {val} | duration: {duration}\n"
            )

        self.iteration += 1
        return val

    def jac(self, x):
        if self.jacobian is None:
            return None
        res = self.jacobian(x, *self.args)
        return res.T
