"""Meta-heuristic optimization algorithms."""

import cma
import numpy as np
from scipy.optimize import basinhopping

from qibo.config import log, raise_error
from qibo.optimizers.abstract import Optimizer, check_options


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
        super().__init__(options)
        self.name = "cmaes"
        # check if options are compatible with the function and update class options
        check_options(function=cma.fmin2, options=options)
        self.set_options(options)

    def fit(self, initial_parameters, loss, args=(), fit_options={}):
        """Perform the optimizations via CMA-ES.

        Args:
            loss (callable): loss function to train on.
            initial_parameters (np.ndarray or list): array with initial values
                for gate parameters.
            args (tuple): tuple containing loss function arguments.
            fit_options (dict): fit extra options. To have a look to all
                possible options please import the `cma` package and type `cma.CMAOptions()`.

        Returns:
            tuple: best loss value (float), best parameter values (np.ndarray), full cma result object.
        """
        if not isinstance(args, tuple):
            raise_error(TypeError, "Loss function args must be provided as a tuple.")
        else:
            self.args = args

        if not isinstance(initial_parameters, np.ndarray) and not isinstance(
            initial_parameters, list
        ):
            raise_error(
                TypeError,
                "Parameters must be a list of Parameter objects or a numpy array.",
            )

        log.info(
            f"Optimization is performed using the optimizer: {type(self).__name__}"
        )

        # update options dictionary with extra `cma.fmin` options.
        self.set_options({"options": fit_options})

        r = cma.fmin2(
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
        options={"niter": 10},
        minimizer_kwargs={},
    ):
        super().__init__(options)
        self.name = "basinhopping"
        # check if options are compatible with the function and update class options
        check_options(function=basinhopping, options=options)
        self.options = options
        self.minimizer_kwargs = minimizer_kwargs

    def fit(self, loss, initial_parameters, args=()):
        """Perform the optimizations via Basin-Hopping strategy.

        Args:
            loss (callable): loss function to train on.
            initial_parameters (np.ndarray or list): array with initial values
                for gate parameters.
            args (tuple): tuple containing loss function arguments.

        Returns:
            tuple: best loss value (float), best parameter values (np.ndarray), full scipy OptimizeResult object.
        """
        if not isinstance(args, tuple):
            raise_error(TypeError, "Loss function args must be provided as a tuple.")
        else:
            self.args = args

        if not isinstance(initial_parameters, np.ndarray) and not isinstance(
            initial_parameters, list
        ):
            raise_error(
                TypeError,
                "Parameters must be a list of Parameter objects or a numpy array.",
            )

        log.info(
            f"Optimization is performed using the optimizer: {type(self).__name__}"
        )

        self.minimizer_kwargs.update({"args": args})
        self.set_options({"minimizer_kwargs": self.minimizer_kwargs})

        r = basinhopping(
            loss,
            initial_parameters,
            **self.options,
        )

        return r.fun, r.x, r
