"""Meta-heuristic optimization algorithms."""
import inspect

import cma
from scipy.optimize import basinhopping

from qibo.config import log
from qibo.optimizers.abstract import Optimizer, check_fit_arguments, check_options


class CMAES(Optimizer):
    """
    Covariance Matrix Adaptation Evolution Strategy based on
    `pycma <https://github.com/CMA-ES/pycma>`_.

    Args:
        options (dict): optimizer's options. These correspond the the arguments
            which can be set into the `cma.fmin2` method. Please look at the
            official documentation https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.html#fmin2
            to see all the available options. The only default set parameter
            is `sigma0`, which we have set to 0.5, as it is a reasonable value
            when considering variational parameters as rotation angles in a circuit.
    """

    def __init__(self, options={"sigma0": 0.5}):
        self.options = {}
        self.name = "cmaes"
        self._fit_function = cma.fmin2

        # check if options are compatible with the function and update class options
        check_options(function=self._fit_function, options=options)
        self.set_options(options)

    def get_options_list(self):
        """Return all available optimizer's options."""
        default_arguments = ["objective_function", "x0", "args", "options"]
        customizable_arguments = ()
        for arg in list(inspect.signature(self._fit_function).parameters):
            if arg not in default_arguments:
                customizable_arguments += (arg,)
        return customizable_arguments

    def get_fit_options_list(self, keyword=None):
        """
        Return all the available fit options for the optimizer.

        Args:
            keyword (str): keyword to help the research of fit options into the
                options dictionary.
        """
        return cma.CMAOptions(keyword)

    def fit(self, initial_parameters, loss, args=(), fit_options={}):
        """Perform the optimizations via CMA-ES.

        Args:
            initial_parameters (np.ndarray or list): array with initial values
                for gate parameters.
            loss (callable): loss function to train on.
            args (tuple): tuple containing loss function arguments.
            fit_options (dict): fit extra options. To have a look to all
                possible options please import the `cma` package and type `cma.CMAOptions()`.

        Returns:
            tuple: best loss value (float), best parameter values (np.ndarray), full cma result object.
        """

        check_fit_arguments(args=args, initial_parameters=initial_parameters)

        log.info(
            f"Optimization is performed using the optimizer: {type(self).__name__}"
        )

        # update options dictionary with extra `cma.fmin` options.
        self.set_options({"options": fit_options})

        r = self._fit_function(
            objective_function=loss,
            x0=initial_parameters,
            args=args,
            **self.options,
        )

        return r[1].result.fbest, r[1].result.xbest, r


class BasinHopping(Optimizer):
    """
    Global optimizer based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html.

    Args:
        options (dict): additional information compatible with the
            `scipy.optimize.basinhopping` optimizer. The only default set parameter is `niter=10`.
        minimizer_kwargs (dict): the Basin-Hopping optimizer makes use of an
            extra Scipy's minimizer to compute the optimizaton. This argument
            can be used to setup the extra minimization routine. For example,
            one can set:

            .. code-block:: python

                minimizer_kwargs = {
                    method = "BFGS",
                    jac = None
                }

    """

    def __init__(
        self,
        options={},
        minimizer_kwargs={},
    ):
        self.options = {"niter": 10}
        self.name = "basinhopping"
        self._fit_function = basinhopping
        # check if options are compatible with the function and update class options
        check_options(function=self._fit_function, options=options)
        self.options = options
        self.minimizer_kwargs = minimizer_kwargs

    def get_options_list(self):
        """Return all available optimizer's options."""
        default_arguments = ["func", "x0", "options", "minimizer_kwargs"]
        customizable_arguments = []
        for arg in list(inspect.signature(self._fit_function).parameters):
            if arg not in default_arguments:
                customizable_arguments.append(arg)
        return customizable_arguments

    def get_fit_options_list(self):
        log.info(f"No `fit_options` are required for the Basin-Hopping optimizer.")

    def fit(self, initial_parameters, loss, args=()):
        """Perform the optimizations via Basin-Hopping strategy.

        Args:
            initial_parameters (np.ndarray or list): array with initial values
                for gate parameters.
            loss (callable): loss function to train on.
            args (tuple): tuple containing loss function arguments.

        Returns:
            tuple: best loss value (float), best parameter values (np.ndarray), full scipy OptimizeResult object.
        """

        check_fit_arguments(args=args, initial_parameters=initial_parameters)

        log.info(
            f"Optimization is performed using the optimizer: {type(self).__name__}"
        )

        self.minimizer_kwargs.update({"args": args})
        self.set_options({"minimizer_kwargs": self.minimizer_kwargs})

        r = self._fit_function(
            loss,
            initial_parameters,
            **self.options,
        )

        return r.fun, r.x, r
