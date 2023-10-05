import numpy as np

import cma
from scipy.optimize import minimize
from scipy.optimize import basinhopping

from qibo import backends
from qibo.config import log, raise_error


class Optimizer:
    """
    Agnostic optimizer class, on top of which specific optimizers are built.

    Args:
        initial_parameters (np.ndarray or list): array with initial values 
            for gate parameters.
        loss (callable): loss function to train on.
        args (tuple): tuple containing loss function arguments.
    """

    def __init__(self, initial_parameters, args=(), loss=None):
        # saving to class objects
        self.loss = loss
        if not isinstance(args, tuple):
            self.args = (args,)
        else:
            self.args = args

        self.params = initial_parameters
        self.backend = backends.GlobalBackend()

        self.options = {}

        if not isinstance(initial_parameters, np.ndarray) and not isinstance(
            initial_parameters, list
        ):
            raise_error(
                TypeError,
                "Parameters must be a list of Parameter objects or a numpy array."
            )

    def set_options(self, updates):
        """Update self.options dictionary"""
        self.options.update(updates)

    def fit(self):
        """Compute the optimization strategy."""
        raise_error(
            NotImplementedError,
            "fit method is not implemented in the parent Optimizer class."
        )


class CMAES(Optimizer):
    """Genetic optimizer based on `pycma <https://github.com/CMA-ES/pycma>`_."""

    def __init__(self, initial_parameters, args=(), options={}, loss=None):
        """

        Args:
        """
        super().__init__(initial_parameters, args, loss)

        self.options = {}
        self.set_options(options)

    def fit(self):
        """Performs the optimizations.

        Returns:
            (float): best loss value
            (np.ndarray): best parameter values
            (cma.evolution_strategy.CMAEvolutionStrategy): full CMA Evolution Strategy object
        """

        log.info(f"Optimization is performed using the optimizer: {type(self).__name__}")

        r = cma.fmin2(
            objective_function=self.loss,
            x0=self.params,
            args=self.args,
            sigma0=1.7,
            **self.options,
        )

        return r[1].result.fbest, r[1].result.xbest, r


class ScipyMinimizer(Optimizer):
    def __init__(self, initial_parameters, args=(), loss=None, options={}, method="Powell", **kwargs):
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
        super().__init__(initial_parameters, args, loss)
        self.method = method 

        self.options = {
            "method": self.method,
            "jac": None,
            "hess": None,
            "hessp": None,
            "bounds": None,
            "constraints": (),
            "tol": None,
            "callback": None,
            "options": options,
            "processes": None,
            "backend": None,
        }
        self.set_options(kwargs)

    def fit(self):
        """Performs the optimizations.

        Returns:
            (float): best loss value
            (np.ndarray): best parameter values
            (scipy.optimize.OptimizeResult): full scipy OptimizeResult object
            (int): iteration number
        """

        log.info(f"Optimization is performed using the optimizer: {type(self).__name__}.{self.method}")

        r = minimize(
            self.loss,
            self.params,
            args=self.args,
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

        return r.fun, r.x, r


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

    def __init__(self, initial_parameters, loss, args=(), **kwargs):
        super().__init__(initial_parameters, args, loss)
        
        self.function_value = None
        self.jacobian_value = None

        self.options = {
            "xval": None,
            "precision": np.finfo("float64").eps,
            "bounds": None,
            "callback": None,
            "options": None,
            "processes": None,
        }

        self.set_options(kwargs)


    def fit(self):
        """Performs the optimizations.

        Returns:
            (float): best loss value
            (np.ndarray): best parameter values
            (scipy.optimize.OptimizeResult): full scipy OptimizeResult object
            (int): iteration number
        """

        log.info(f"Optimization is performed using the optimizer: {type(self).__name__}")

        out = minimize(
            fun=self.fun,
            x0=self.params,
            jac=self.jac,
            method="L-BFGS-B",
            bounds=self.options["bounds"],
            callback=self.options["callback"],
            options=self.options["options"],
        )
        out.hess_inv = out.hess_inv * np.identity(len(self.params))
        return out.fun, out.x, out

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
                    epsi, lambda y: self.loss(y, *self.args), x, eps
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
    """
    Global optimizer based on `scipy.optimize.basinhopping <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html>`_.
    """

    def __init__(self, initial_parameters, args=(), loss=None, **kwargs):
        super().__init__(initial_parameters, args, loss)

        self.args = args
        self.options = kwargs

    def fit(self):
        """Performs the optimizations.

        Returns:
            (float): best loss value
            (np.ndarray): best parameter values
            (scipy.optimize.OptimizeResult): full scipy OptimizeResult object
            (int): iteration number
        """

        log.info(f"Optimization is performed using the optimizer: {type(self).__name__}")

        if "options" in self.options:
            options = self.options.pop("options")

        r = basinhopping(
            self.loss,
            self.params,
            niter=1,
            minimizer_kwargs={"args": self.args},
        )

        return r.fun, r.x, r