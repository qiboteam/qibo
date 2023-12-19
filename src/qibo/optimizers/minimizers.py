"""Optimization algorithms inherited from Scipy's minimization module."""
import inspect

import numpy as np
from scipy.optimize import minimize

from qibo.config import log
from qibo.optimizers.abstract import Optimizer, check_fit_arguments, check_options


class ScipyMinimizer(Optimizer):
    """
    Optimization approaches based on ``scipy.optimize.minimize``.

    Args:
        options (dict): options which can be provided to the general
            Scipy's minimizer. See `scipy.optimize.minimize` documentation.
            By default, the `"method"` option is set to `"Powell"`.

    """

    def __init__(self, options={}):
        self.options = {"method": "Powell"}
        self.name = "scipy_minimizer"
        self._fit_function = minimize
        check_options(function=self._fit_function, options=options)
        self.set_options(options)

    def get_options_list(self):
        """Return all available optimizer's options."""
        default_arguments = ["fun", "x0", "args", "options"]
        customizable_arguments = []
        for arg in list(inspect.signature(self._fit_function).parameters):
            if arg not in default_arguments:
                customizable_arguments.append(arg)
        return customizable_arguments

    def get_fit_options_list(self):
        print(
            f"Please have a look to the `options` argument of `scipy.optimize.minimize(method='{self.options['method']}')`"
        )

    def fit(self, initial_parameters, loss=None, args=(), fit_options={}):
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

        check_fit_arguments(args=args, initial_parameters=initial_parameters)

        log.info(
            f"Optimization is performed using the optimizer: {type(self).__name__}.{self.options['method']}"
        )

        # update options with minimizer extra options
        self.set_options({"options": fit_options})

        r = self._fit_function(
            loss,
            initial_parameters,
            args=args,
            **self.options,
        )

        return r.fun, r.x, r


class ParallelBFGS(Optimizer):  # pragma: no cover
    """
    Computes the L-BFGS-B with parallel evaluation using multiprocessing.
    This implementation here is based on https://doi.org/10.32614/RJ-2019-030.

    Args:
        processes (int): number of parallel processes used to evaluate functions.
        options (dict): possible arguments accepted by
            `scipy.optimize.minimize` class.
    """

    def __init__(
        self,
        processes=1,
        options={},
    ):
        self.options = {}
        self.name = "parallel_bfgs"
        self.xval = None
        self.function_value = None
        self.jacobian_value = None
        self.processes = processes
        self.precision = np.finfo("float64").eps
        self._fit_function = minimize

        # check if options are compatible with the function and update class options
        check_options(function=self._fit_function, options=options)

    def get_options_list(self):
        """Return all available optimizer's options."""
        default_arguments = ["fun", "x0", "args", "options", "method"]
        customizable_arguments = []
        for arg in list(inspect.signature(self._fit_function).parameters):
            if arg not in default_arguments:
                customizable_arguments.append(arg)
        return customizable_arguments

    def get_fit_options_list(self):
        print(
            f"Please have a look to the `options` argument of `scipy.optimize.minimize(method='L-BFGS-B')`"
        )

    def fit(self, initial_parameters, loss=None, args=(), fit_options={}):
        """Performs the optimizations via ParallelBFGS.

        Args:
            initial_parameters (np.ndarray or list): array with initial values
                for gate parameters.
            loss (callable): loss function to train on.
            args (tuple): tuple containing loss function arguments.
            fit_options (dict): specific options accepted by the L-BFGS-B minimizer.
                This argument corresponds to Scipy's `"options"`.

        Returns:
            tuple: best loss value (float), best parameter values (np.ndarray), full scipy OptimizeResult object.
        """

        check_fit_arguments(args=args, initial_parameters=initial_parameters)

        log.info(
            f"Optimization is performed using the optimizer: {type(self).__name__}"
        )

        # update options with L-BFGS-B extra options
        self.set_options({"options": fit_options})
        self.loss = loss
        self.args = args
        self.params = initial_parameters

        out = self._fit_function(
            fun=self.fun,
            x0=initial_parameters,
            jac=self.jac,
            method="L-BFGS-B",
            **self.options,
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
