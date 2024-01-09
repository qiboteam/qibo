import inspect
from abc import abstractmethod

import numpy as np

from qibo.config import raise_error


def check_options(function, options):
    """
    Check if given options dictionary is compatible with optimization
    method arguments.
    """

    for arg in options:
        if arg not in list(inspect.signature(function).parameters):
            raise_error(
                TypeError,
                f"Given argument {arg} is not accepted by {function.__code__.co_name} function, which is the fitting function of the chosen optimizer.",
            )


def check_fit_arguments(args, initial_parameters):
    """
    Check loss function args and initial parameters Types.

    Args:
        args (tuple): list of loss function extra arguments.
        initial_parameters (np.ndarray or list): initial parameters.
    """
    # check whether if args are a tuple
    if not isinstance(args, tuple):
        raise_error(TypeError, "Loss function args must be provided as a tuple.")
    else:
        args = args
    # check whether params are a list or a numpy array
    if not isinstance(initial_parameters, np.ndarray) and not isinstance(
        initial_parameters, list
    ):
        raise_error(
            TypeError,
            "Parameters must be a list of Parameter objects or a numpy array.",
        )


class Optimizer:
    """Qibo optimizer class."""

    def __init__(self, options={}):
        """
        Args:
            options (dict): optimizer's parameters. This dictionary has to be
                filled according to each specific optimizer interface.
        """
        pass

    def set_options(self, updates):
        """Update self.options dictionary"""
        # check the new options are arguments of the fitting function
        check_options(self._fit_function, updates)
        self.options.update(updates)

    @abstractmethod
    def get_options_list(self):  # pragma: no cover
        """Return all available optimizer options."""
        raise_error(NotImplementedError)

    @abstractmethod
    def get_fit_options_list(self):  # pragma: no cover
        """Return all available fitting options."""
        raise_error(NotImplementedError)

    @abstractmethod
    def fit(
        self,
        initial_parameters,
        loss,
        args=(),
        fit_options={},
    ):  # pragma: no cover
        """
        Compute the optimization strategy according to the chosen optimizer.

        Args:
            initial_parameters (np.ndarray or list): array with initial values
                for gate parameters.
            loss (callable): loss function to train on.
            args (tuple): tuple containing loss function arguments.
            fit_options (dict): extra options to customize the fit.
        """
        raise_error(NotImplementedError)
