"""Gradient descent strategies to optimize quantum models."""

from typing import List, Optional, Tuple, Union

from numpy import ndarray

from qibo.backends import construct_backend
from qibo.config import log
from qibo.optimizers.abstract import Optimizer


class TensorflowSGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer using Tensorflow backpropagation.
    See `tf.keras.Optimizers https://www.tensorflow.org/api_docs/python/tf/keras/optimizers.
    for a list of the available optimizers.

    Args:
        optimizer_name (str): `tensorflow.keras.optimizer`, see
            https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
            for the list of available optimizers.
        compile (bool):  if ``True`` the Tensorflow optimization graph is compiled.
        **optimizer_options (dict): a dictionary containing the keywords arguments
            to customize the selected keras optimizer. In order to properly
            customize your optimizer please refer to https://www.tensorflow.org/api_docs/python/tf/keras/optimizers.
    """

    def __init__(
        self, optimizer_name: str = "Adagrad", compile: bool = True, **optimizer_options
    ):

        self.optimizer_name = optimizer_name
        self.compile = compile

        if optimizer_options is None:
            options = {}
        else:
            options = optimizer_options

        self.backend = construct_backend("tensorflow")
        self.optimizer = getattr(
            self.backend.tf.optimizers.legacy, self.optimizer_name
        )(**options)

    def __str__(self):
        return f"tensorflow_{self.optimizer_name}"

    def fit(
        self,
        initial_parameters: Union[List, ndarray],
        loss: callable,
        args: Union[Tuple] = None,
        epochs: int = 10000,
        nmessage: int = 100,
        loss_threshold: Optional[float] = None,
    ):
        """
        Compute the SGD optimization according to the chosen optimizer.

        Args:
            initial_parameters (np.ndarray or list): array with initial values
                for gate parameters.
            loss (callable): loss function to train on.
            args (tuple): tuple containing loss function arguments.
            epochs (int): number of optimization iterations [default 10000].
            nmessage (int): Every how many epochs to print
                a message of the loss function [default 100].
            loss_threshold (float): if this loss function value is reached, training
                stops [default None].

        Returns:
            (float): best loss value
            (np.ndarray): best parameter values
            (list): loss function history
        """

        vparams = self.backend.tf.Variable(
            initial_parameters, dtype=self.backend.tf.float64
        )
        print(vparams)
        loss_history = []

        def sgd_step():
            """Compute one SGD optimization step according to the chosen optimizer."""
            with self.backend.tf.GradientTape() as tape:
                tape.watch(vparams)
                loss_value = loss(vparams, *args)

            grads = tape.gradient(loss_value, [vparams])
            self.optimizer.apply_gradients(zip(grads, [vparams]))
            return loss_value

        if self.compile:
            self.backend.compile(loss)
            self.backend.compile(sgd_step)

        # SGD procedure: loop over epochs
        for epoch in range(epochs):  # pragma: no cover
            # early stopping if loss_threshold has been set
            if (
                loss_threshold is not None
                and (epoch != 0)
                and (loss_history[-1] <= loss_threshold)
            ):
                break

            loss_value = sgd_step().numpy()
            loss_history.append(loss_value)

            if epoch % nmessage == 0:
                log.info("ite %d : loss %f", epoch, loss_value)

        return loss(vparams, *args).numpy(), vparams.numpy(), loss_history
