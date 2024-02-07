"""Optimization algorithms inherited from Scipy's minimization module."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from scipy.optimize import Bounds, minimize, show_options

from qibo.config import log
from qibo.optimizers.abstract import Optimizer


@dataclass
class ScipyMinimizer(Optimizer):
    """
    Optimization approaches based on `scipy.optimize.minimize`.

    Attributes:
        method (Optional[str]): optimization method among the minimizers provided by scipy, defaults to "Powell".
        jac (Optional[dict]): method for computing the gradient vector.
        hess (Optional[dict]): method for computing the hessian matrix.
        hessp (Optional[callable]): hessian of objective function times an arbitrary vector.
        bounds (Union[None, List[Tuple], Bounds]): bounds on variables.
        constraints (Optional[dict]): constraints definition.
        tol (Optional[float]): tolerance for termination.
        callback (Optional[callable]): a callable called after each optimization iteration.
        verbosity (bool): verbosity level of the optimization. If `True`, logging messages are displayed.
    """

    method: Optional[str] = "Powell"
    jac: Optional[dict] = None
    hess: Optional[dict] = None
    hessp: Optional[callable] = None
    bounds: Union[None, List[Tuple], Bounds] = None
    constraints: Optional[dict] = None
    tol: Optional[float] = None
    callback: Optional[callable] = None

    def __str__(self):
        return f"scipy_minimizer_{self.method}"

    def show_fit_options(self):
        """Return available extra options for chosen minimizer."""
        return show_options(solver="minimize", method=self.method)

    def fit(
        self,
        initial_parameters: Union[List, ndarray],
        loss: callable,
        args: Tuple,
        fit_options: Optional[dict] = None,
    ):
        """Perform the optimizations via ScipyMinimizer.

        Args:
            initial_parameters (np.ndarray or list): array with initial values
                for gate parameters.
            loss (callable): loss function to train on.
            args (tuple): tuple containing loss function arguments.
            fit_options (dict): dictionary containing extra options which depend
                on the chosen `"method"`. This argument is called "options" in the
                Scipy's documentation and we recommend to fill it according to the
                official documentation.

        Returns:
            tuple: best loss value (float), best parameter values (np.ndarray), full scipy OptimizeResult object.
        """

        if fit_options is None:
            options = {}
        else:
            options = fit_options

        if self.verbosity:
            log.info(f"Optimization is performed using the optimizer: {self.__str__()}")

        r = minimize(loss, initial_parameters, args=args, **options)

        return r.fun, r.x, r


@dataclass
class ParallelBFGS(ScipyMinimizer):
    """
    Computes the L-BFGS-B with parallel evaluation using multiprocessing.
    This implementation here is based on https://doi.org/10.32614/RJ-2019-030.

    Attributes:
        jac (Optional[dict]): Method for computing the gradient vector.
        hess (Optional[dict]): Method for computing the hessian matrix.
        hessp (Optional[callable]): Hessian of objective function times an arbitrary vector.
        bounds (Union[None, List[Tuple], Bounds]): Bounds on variables.
        constraints (Optional[dict]): Constraints definition.
        tol (Optional[float]): Tolerance for termination.
        callback (Optional[callable]): A callable called after each optimization iteration.
        processes (int): number of processes to be computed in parallel.
        verbosity (bool): verbosity level of the optimization. If `True`, logging messages are displayed.
    """

    processes: int = field(default=1)
    xval: float = field(init=False, default=None)
    function_value: float = field(init=False, default=None)
    jacobian_value: float = field(init=False, default=None)
    precision: float = field(init=False, default=np.finfo("float64").eps)

    def __post_init__(self):
        self.xval = None
        self.function_value = None
        self.jacobian_value = None
        self.precision = np.finfo("float64").eps

    def __str__(self):
        return f"scipy_minimizer_ParallelBFGS"

    def show_fit_options(self):
        """Return available extra options for chosen minimizer."""
        return show_options(solver="minimize", method="L-BFGS-B")

    def fit(
        self,
        initial_parameters: Union[List, ndarray],
        loss: callable,
        args: Tuple,
        fit_options: Optional[dict] = None,
    ):
        """Performs the optimizations via ParallelBFGS.

        Args:
            initial_parameters (np.ndarray or list): array with initial values
                for gate parameters.
            loss (callable): loss function to train on.
            args (tuple): tuple containing loss function arguments.
            fit_options (dict): specific options accepted by the L-BFGS-B minimizer.
                Use the method `ParallelBFGS.show_fit_options()` to visualize all
                the available options.

        Returns:
            tuple: best loss value (float), best parameter values (np.ndarray), full scipy OptimizeResult object.
        """

        if self.verbosity:
            log.info(f"Optimization is performed using the optimizer: {self.__str__()}")

        self.loss = loss
        self.args = args
        self.params = initial_parameters

        if fit_options is None:
            self.fit_options = {}
        else:
            self.fit_options = fit_options

        out = minimize(
            fun=self.fun,
            x0=initial_parameters,
            jac=self.jac,
            method="L-BFGS-B",
            **self.fit_options,
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
            self.xval is not None and all(abs(self.xval - x) <= self.precision * 2)
        ):
            eps_at = range(len(x) + 1)
            self.xval = x.copy()

            def operation(epsi):
                return self._eval_approx(
                    epsi, lambda y: self.loss(y, *self.args), x, eps
                )

            from joblib import Parallel, delayed

            ret = Parallel(self.processes, prefer="threads")(
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
