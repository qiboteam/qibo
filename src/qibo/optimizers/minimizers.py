"""Optimization algorithms inherited from Scipy's minimization module."""

import numpy as np
from scipy.optimize import minimize

from qibo.config import log
from qibo.optimizers.abstract import Optimizer, check_options


class ScipyMinimizer(Optimizer):
    def __init__(
        self,
        initial_parameters,
        loss=None,
        args=(),
        options={"method": "Powell"},
        minimizer_kwargs={},
    ):
        """
        Optimization approaches based on ``scipy.optimize.minimize``.

        Args:
            initial_parameters (np.ndarray or list): array with initial values
                for gate parameters.
            loss (callable): loss function to train on.
            args (tuple): tuple containing loss function arguments.
            options (dict): options which can be provided to the general
                Scipy's minimizer. See `scipy.optimize.minimize` documentation.
                By default, the `"method"` option is set to `"Powell"`.
            minimizer_kwargs (dict): extra options to customize the selected
                minimizer. This arguments correspond to Scipy's "options".

        """
        super().__init__(initial_parameters, args, loss)

        # check if options are compatible with the function and update class options
        check_options(function=minimize, options=options)
        self.set_options(options)
        self.set_options({"options": minimizer_kwargs})

        self.method = self.options["method"]

    def fit(self):
        """Perform the optimizations via ScipyMinimizer.

        Returns:
            (float): best loss value.
            (np.ndarray): best parameter values.
            (scipy.optimize.OptimizeResult): full scipy OptimizeResult object.
        """

        log.info(
            f"Optimization is performed using the optimizer: {type(self).__name__}.{self.method}"
        )

        print("scipy", self.loss, self.params, self.args, self.options)
        import numpy as np

        np.random.seed(42)
        r = minimize(
            self.loss,
            self.params,
            args=self.args,
            **self.options,
        )

        return r.fun, r.x, r


class ParallelBFGS(Optimizer):  # pragma: no cover
    def __init__(
        self,
        initial_parameters,
        loss,
        processes=1,
        args=(),
        options={},
        minimizer_kwargs={},
    ):
        """
        Computes the L-BFGS-B using parallel evaluation using multiprocessing.
        This implementation here is based on https://doi.org/10.32614/RJ-2019-030.

        Args:
            initial_parameters (np.ndarray or list): array with initial values
                for gate parameters.
            loss (callable): loss function to train on.
            processes (int): number of parallel processes used to evaluate functions.
            args (tuple): tuple containing loss function arguments.
            options (dict): possible arguments accepted by
                `scipy.optimize.minimize` class.
            minimizer_kwargs (dict): specific options accepted by the L-BFGS-B minimizer.
                This argument corresponds to Scipy's `"options"`.
        """
        super().__init__(initial_parameters, args, loss)

        self.xval = None
        self.function_value = None
        self.jacobian_value = None
        self.processes = processes
        self.precision = np.finfo("float64").eps

        # check if options are compatible with the function and update class options
        check_options(function=minimize, options=options)
        self.set_options(options)
        self.set_options({"options": minimizer_kwargs})

    def fit(self):
        """Performs the optimizations via ParallelBFGS.

        Returns:
            (float): best loss value
            (np.ndarray): best parameter values
            (scipy.optimize.OptimizeResult): full scipy OptimizeResult object
        """

        log.info(
            f"Optimization is performed using the optimizer: {type(self).__name__}"
        )

        out = minimize(
            fun=self.fun,
            x0=self.params,
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
