"""Gradient descent strategies to optimize quantum models."""

from qibo import set_backend
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

        Example:
        .. testcode::
            import numpy as np
            from qibo import models, hamiltonians, gates, set_backend
            from qibo.optimizers.gradient_based import TensorFlowSGD

            # tensorflow backend is needed to use the TensorFlowSGD optimizer.
            set_backend("tensorflow")

            # define a dummy model
            nqubits = 2
            nlayers = 3

            c = models.Circuit(nqubits)
            for l in range(nlayers):
                for q in range(nqubits):
                    c.add(gates.RY(q=q, theta=0))
                    c.add(gates.RY(q=q, theta=0))
                for q in range(nqubits-1):
                    c.add(gates.CNOT(q0=q, q1=q+1))
            c.add(gates.M(*range(nqubits)))

            # define a loss function
            h = hamiltonians.Z(nqubits)
            def loss(parameters, circuit, hamiltonian):
                circuit.set_parameters(parameters)
                return hamiltonian.expectation(circuit().state())

            # initialize parameters
            params = np.random.randn(2 * nqubits * nlayers)

            # initialize optimizer
            options = {"learning_rate": 0.05}
            opt = TensorFlowSGD(initial_parameters=params, loss=loss, args=(c,h), options=options)
            # perform the training
            res = opt.fit(epochs=50, nmessage=1)
        """

        super().__init__(initial_parameters, args, loss)
        # This optimizer works only with tensorflow backend
        self.backend = TensorflowBackend()

        self.options = options

        # options are automatically checked inside the tf.keras.optimizer
        self.optimizer = getattr(self.backend.tf.optimizers, optimizer)(**self.options)

    def fit(self, epochs=100000, nmessage=100, loss_treshold=None):
        """
        Compute the SGD optimization according to the chosen optimizer.

        Args:
            epochs (int): number of optimization iterations [default 10000].
            nmessage (int): Every how many epochs to print
                a message of the loss function [default 100].
            loss_treshold (float): if this loss function value is reached, training
                stops [default None].

        Returns:
            (float): best loss value
            (np.ndarray): best parameter values
            (list): loss function history
        """

        vparams = self.backend.tf.Variable(
            self.params, dtype=self.backend.tf.complex128
        )
        loss_history = []

        def sgd_step():
            """Compute one SGD optimization step according to the chosen optimizer."""
            with self.backend.tf.GradientTape() as tape:
                tape.watch(vparams)
                loss = self.loss(vparams, *self.args)

            grads = tape.gradient(loss, [vparams])
            self.optimizer.apply_gradients(zip(grads, [vparams]))

            return loss

        self.backend.compile(self.loss)
        self.backend.compile(sgd_step)

        # SGD procedure: loop over epochs
        for epoch in range(epochs):
            # early stopping if loss_treshold has been set
            if (
                loss_treshold is not None
                and (epoch != 0)
                and (loss_history[-1] <= loss_treshold)
            ):
                continue

            loss = sgd_step()
            loss_history.append(loss.numpy())

            if epoch % nmessage == 0:
                log.info("ite %d : loss %f", epoch, loss_history[-1])

        return self.loss(vparams, *self.args).numpy(), vparams.numpy(), loss_history
