import numpy as np

from qibo import backends
from qibo.config import raise_error


def check_options(function, options):
    """
    Check if given options dictionary is compatible with optimization
    method arguments.
    """

    print(function.__code__.co_varnames)
    for arg in options:
        if arg not in function.__code__.co_varnames:
            raise_error(
                TypeError,
                f"Given argument {arg} is not accepted by {function.__code__.co_name} function.",
            )


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

        self.options = {}

        if not isinstance(initial_parameters, np.ndarray) and not isinstance(
            initial_parameters, list
        ):
            raise_error(
                TypeError,
                "Parameters must be a list of Parameter objects or a numpy array.",
            )

    def set_options(self, updates):
        """Update self.options dictionary"""
        self.options.update(updates)

    def fit(self):
        """Compute the optimization strategy."""
        raise_error(
            NotImplementedError,
            "fit method is not implemented in the parent Optimizer class.",
        )
