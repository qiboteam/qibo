"""Gradient descent strategies to optimize quantum models."""

import tensorflow as tf

from qibo.backends import TensorflowBackend
from qibo.config import log, raise_error
from qibo.optimizers.abstract import Optimizer, check_options


class TensorFlowSGD(Optimizer):
    def __init__(
        self,
        initial_parameters,
        args=(),
        loss=None,
        optimizer="Adagrad",
        options={"learning_rate": 0.001},
    ):
        """
        Stochastic Gradient Descent (SGD) optimizer using Tensorflow backpropagation.

        See `tf.keras.Optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_
        for a list of the available optimizers.

        Args:
            initial_parameters (np.ndarray or list): array with initial values
                for gate parameters.
            loss (callable): loss function to train on.
            args (tuple): tuple containing loss function arguments.
            optimizer (str): `tensorflow.keras.optimizer`, see
                <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>_
                for the list of available optimizers.
            options (dict): options which can be provided to the chosen optimizer.
                See the same reference of above for the complete list of options once
                the optimizer is selected.
        """
        super().__init__(initial_parameters, args, loss)
        # This optimizer works only with tensorflow backend
        self.backend = TensorflowBackend()

        self.options = options

        # options are automatically checked inside the tf.keras.optimizer
        self.optimizer = getattr(self.backend.tf.optimizers, optimizer)(**self.options)

    def fit(self, epochs, nmessage):
        """
        Compute the SGD optimization according to the chosen optimizer.

        Args:
            epochs (int): number of optimization iterations [default 10000].
            nmessage (int): Every how many epochs to print
                a message of the loss function.

        Returns:
            (float): best loss value
            (np.ndarray): best parameter values
            (list): loss function history
        """

        vparams = self.backend.tf.Variable(self.params)
        loss_history = []

        def sgd_step():
            """Compute one SGD optimization step according to the chosen optimizer."""
            with self.backend.tf.GradientTape() as tape:
                loss = self.loss(vparams, *self.args)

            grads = tape.gradient(loss, [vparams])
            self.optimizer.apply_gradients(zip(grads, [vparams]))
            return loss

        self.backend.compile(self.loss)
        self.backend.compile(sgd_step)

        # SGD procedure: loop over epochs
        for epoch in range(epochs):
            loss = sgd_step()
            loss_history.append(loss.numpy())
            if epoch % nmessage == 0:
                log.info("ite %d : loss %f", epoch, loss_history[-1])

        return self.loss(vparams, *self.args).numpy(), vparams.numpy(), loss_history
