import numpy as np

from qibo import backends
from qibo.config import raise_error


def check_options(function, options):
    """
    Check if given options dictionary is compatible with optimization
    method arguments.
    """

    for arg in options:
        if arg not in function.__code__.co_varnames:
            raise_error(
                TypeError,
                f"Given argument {arg} is not accepted by {function.__code__.co_name} function.",
            )


class Optimizer:
    """Qibo optimizer class."""

    def __init__(self, options={}):
        """
        Args:
            options (dict): optimizer's parameters. This dictionary has to be
                filled according to each specific optimizer interfate.
        """
        self.options = options

    def set_options(self, updates):
        """Update self.options dictionary"""
        self.options.update(updates)

    def fit(self, loss, initial_parameters, args=(), fit_options={}):
        """
        Compute the optimization strategy given loss function generalities.

        Args:
            loss (callable): loss function to train on.
            initial_parameters (np.ndarray or list): array with initial values
                for gate parameters.
            args (tuple): tuple containing loss function arguments.
            fit_options (dict): extra options to customize the fit.
        """
        if not isinstance(args, tuple):
            self.args = (args,)
        else:
            self.args = args

        self.params = initial_parameters

        if not isinstance(initial_parameters, np.ndarray) and not isinstance(
            initial_parameters, list
        ):
            raise_error(
                TypeError,
                "Parameters must be a list of Parameter objects or a numpy array.",
            )