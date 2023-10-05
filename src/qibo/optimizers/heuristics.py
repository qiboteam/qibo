"""Meta-heuristic optimization algorithms."""

import cma
from scipy.optimize import basinhopping

from qibo.optimizers.abstract import Optimizer, check_options
from qibo.config import log

class CMAES(Optimizer):

    def __init__(self, initial_parameters, args=(), loss=None, options={"sigma0": 0.5}):
        """
        Covariance Matrix Adaptation Evolution Strategy based on 
        `pycma <https://github.com/CMA-ES/pycma>`_.
        
        Args:
            initial_parameters (np.ndarray or list): array with initial values 
                for gate parameters.
            loss (callable): loss function to train on.
            args (tuple): tuple containing loss function arguments.
            options (dict): options which can be set into the `cma.fmin2` method.
                To have a look to all possible options the command 
                `cma.CMAOptions()` can be used. 
        """
        super().__init__(initial_parameters, args, loss)

        # check if options are compatible with the function and update class options
        check_options(function=cma.fmin2, options=options)
        self.set_options(options)

    def fit(self):
        """Perform the optimizations via CMA-ES.

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
            **self.options,
        )

        return r[1].result.fbest, r[1].result.xbest, r

class BasinHopping(Optimizer):

    def __init__(self, initial_parameters, args=(), loss=None, options={"niter":10}, minimizer_kwargs={}):
        """
        Global optimizer based on 
        `scipy.optimize.basinhopping <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html>`_.

        Args:
            initial_parameters (np.ndarray or list): array with initial values 
                for gate parameters.
            loss (callable): loss function to train on.
            args (tuple): tuple containing loss function arguments.
            options (dict): additional information compatible with the 
                `scipy.optimize.basinhopping` optimizer.
            minimizer_kwargs (dict): extra keyword arguments to be passed to the 
                local minimizer. See `scipy.optimize.basinhopping` documentation.
        """
        super().__init__(initial_parameters, args, loss)

        self.args = args
        # check if options are compatible with the function and update class options
        check_options(function=basinhopping, options=options)
        self.options = options
        
        self.minimizer_kwargs = minimizer_kwargs
        self.minimizer_kwargs.update({"args": args})
        self.set_options({"minimizer_kwargs":self.minimizer_kwargs})

    def fit(self):
        """Perform the optimizations via Basin-Hopping strategy.

        Returns:
            (float): best loss value
            (np.ndarray): best parameter values
            (scipy.optimize.OptimizeResult): full scipy OptimizeResult object
        """

        log.info(f"Optimization is performed using the optimizer: {type(self).__name__}")

        r = basinhopping(
            self.loss,
            self.params,
            **self.options,
        )

        return r.fun, r.x, r