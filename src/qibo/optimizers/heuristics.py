"""Meta-heuristic optimization algorithms."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import cma
from numpy import ndarray
from scipy.optimize import basinhopping

from qibo.config import log
from qibo.optimizers.abstract import Optimizer


@dataclass
class CMAES(Optimizer):
    """
    Covariance Matrix Adaptation Evolution Strategy based on
    `pycma <https://github.com/CMA-ES/pycma>`_.

    Args:
        verbosity (Optional[bool]): verbosity level of the optimization. If `True`, logging messages are displayed.
        sigma0 (Optional[float]): scalar, initial standard deviation in each coordinate.
            sigma0 should be about 1/4th of the search domain width (where the
            optimum is to be expected).
        restarts (Optional[int]): number of restarts with increasing population size.
        restart_from_best (Optional[bool]): which point to restart from.
        iconpopsize (Optional[int]): multiplier for increasing the population size popsize before each restart.
        callback (Optional[callable]): a callable called after each optimization iteration.
    """

    sigma0: Optional[float] = field(default=0.5)
    restarts: Optional[int] = field(default=0)
    restarts_from_best: Optional[bool] = field(default=False)
    iconpopsize: Optional[int] = field(default=2)
    callback: Optional[callable] = field(default=None)

    def __str__(self):
        return "cmaes"

    def show_fit_options(self, keyword: str = None):
        """
        Return all the available fit options for the optimizer.

        Args:
            keyword (str): keyword to help the research of fit options into the
                options dictionary.
        """
        return cma.CMAOptions(keyword)

    def fit(
        self,
        initial_parameters: Union[List, ndarray],
        loss: callable,
        args: Tuple,
        fit_options: Optional[dict] = None,
    ):
        """Perform the optimizations via CMA-ES.

        Args:
            initial_parameters (np.ndarray or list): array with initial values
                for gate parameters.
            loss (callable): loss function to train on.
            args (tuple): tuple containing loss function arguments.
            fit_options (dict): fit extra options. To have a look to all
                possible options please use `CMAES.show_fit_options()`.

        Returns:
            tuple: best loss value (float), best parameter values (np.ndarray), full cma result object.
        """

        log.info(f"Optimization is performed using the optimizer: {self.__str__()}")

        r = cma.fmin2(
            objective_function=loss,
            x0=initial_parameters,
            args=args,
            sigma0=self.sigma0,
            restarts=self.restarts,
            restart_from_best=self.restarts_from_best,
            incpopsize=self.iconpopsize,
            callback=self.callback,
            options=fit_options,
        )

        return r[1].result.fbest, r[1].result.xbest, r


@dataclass
class BasinHopping(Optimizer):
    """
    Global optimizer based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html.
    Note that the Basin-Hopping optimizer combines a global stepping algorithm
    together with a local minimization (which is implemented using an extra scipy minimizer).
    It is designed to mimic the natural process of energy minimization of clusters
    of atoms and it works well for similar problems with “funnel-like, but rugged” energy landscapes.

    Args:
        verbosity (Optional[bool]): verbosity level of the optimization. If `True`, logging messages are displayed.
        niter (Optional[int]): The number of basin-hopping iterations. There will
            be a total of `niter+1` runs of the local minimizer.
        T (Optional[float]): the “temperature” parameter for the acceptance or
            rejection criterion. Higher “temperatures” mean that larger jumps in
            function value will be accepted. For best results T should be comparable
            to the separation (in function value) between local minima.
        stepsize (Optional[float]): maximum step size for use in the random displacement.
        take_step (Optional[callable]): replace the default step-taking routine with this routine.
        accept_test (Optional[callable]): accept test function. It must be of shape
            `accept_test(f_new=f_new, x_new=x_new, f_old=f_old, x_old=x_old)` and
            return a boolean variable. If `True`, the new point is accepted, if
            `False`, the step is rejected. It can also return `force accept`, which
            will override any other tests in order to accept the step.
        callback (Optional[callable]): a callable called after each optimization iteration.
        target_accept_rate (Optional[float]): the target acceptance rate that is
            used to adjust the stepsize. If the current acceptance rate is greater
            than the target, then the stepsize is increased. Otherwise, it is decreased.
        niter_success (Optional[int]): stop the run if the global minimum
            candidate remains the same for this number of iterations.
        minimizer_kwargs (Optional[dict]): extra keyword arguments to be passed
            to the local minimizer. To visualize all the possible options available
            for a fixed scipy `minimizer` please use the method
            `BasinHopping.show_fit_options(method="method")`, where `"method"` is
            selected among the ones provided by `scipy.optimize.minimize`.
    """

    niter: Optional[int] = field(default=10)
    T: Optional[float] = field(default=1.0)
    stepsize: Optional[float] = field(default=0.5)
    take_step: Optional[callable] = field(default=None)
    accept_test: Optional[callable] = field(default=None)
    callback: Optional[callable] = field(default=None)
    target_accept_rate: Optional[float] = field(default=0.5)
    niter_success: Optional[int] = field(default=None)
    minimizer_kwargs: Optional[dict] = field(default=None)
    disp: bool = field(init=False, default=False)

    def __post_init__(self):
        if self.verbosity:
            self.disp = True

    def __str__(self):
        return "basinhopping"

    def show_fit_options_list(self):
        log.info(f"No `fit_options` are required for the Basin-Hopping optimizer.")

    def fit(
        self, initial_parameters: Union[List, ndarray], loss: callable, args: Tuple
    ):
        """Perform the optimizations via Basin-Hopping strategy.

        Args:
            initial_parameters (np.ndarray or list): array with initial values
                for gate parameters.
            loss (callable): loss function to train on.
            args (tuple): tuple containing loss function arguments.

        Returns:
            tuple: best loss value (float), best parameter values (np.ndarray), full scipy OptimizeResult object.
        """

        log.info(
            f"Optimization is performed using the optimizer: {type(self).__name__}"
        )

        if self.minimizer_kwargs is None:
            options = {}
        else:
            options = self.minimizer_kwargs

        options.update({"args": args})

        r = basinhopping(
            func=loss,
            x0=initial_parameters,
            niter=self.niter,
            T=self.T,
            stepsize=self.stepsize,
            take_step=self.take_step,
            accept_test=self.accept_test,
            callback=self.callback,
            niter_success=self.niter_success,
            target_accept_rate=self.target_accept_rate,
            minimizer_kwargs=options,
            disp=self.disp,
        )

        return r.fun, r.x, r
