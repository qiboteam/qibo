"""Gradient descent strategies to optimize quantum models."""
import inspect

from qibo.backends import TensorflowBackend
from qibo.config import log, raise_error
from qibo.optimizers.abstract import Optimizer, check_fit_arguments


class TensorflowSGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer using Tensorflow backpropagation.
    See `tf.keras.Optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_
    for a list of the available optimizers.

    Args:
        optimizer (str): `tensorflow.keras.optimizer`, see
            https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
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
        opt = TensorFlowSGD(options=options)
        # perform the training
        res = opt.fit(loss=loss, initial_parameters=params, args=(circuit, hamiltonian), fit_options={"epochs": 100}, nmessage=1)
    """

    def __init__(
        self,
        optimizer="Adagrad",
        options={"learning_rate": 0.001},
    ):
        self.options = {}
        self.name = "tensorflow"
        # This optimizer works only with tensorflow backend
        self.backend = TensorflowBackend()
        self.options = options

        # options are automatically checked inside the tf.keras.optimizer
        self.optimizer = getattr(self.backend.tf.optimizers, optimizer)(**self.options)
        self.name += f"_{self.optimizer.name}"

    def set_options(self, updates):
        """Update self.options dictionary"""
        for arg in updates:
            if arg not in self.get_options_list():
                raise_error(
                    TypeError,
                    f"Given argument {arg} is not accepted by {self.name}.",
                )
        self.options.update(updates)
        self.optimizer = getattr(self.backend.tf.optimizers, self.optimizer.name)(
            **self.options
        )

    def get_options_list(self):
        """Return list of available options of the chosen optimizer."""
        opt_class = type(self.optimizer)
        return list(inspect.signature(opt_class).parameters)

    def get_fit_options_list(self):
        """Return fit options list."""
        fit_options_list = ["epochs", "nmessage", "loss_threshold"]
        return fit_options_list

    def fit(
        self,
        initial_parameters,
        loss,
        args=(),
        fit_options={"epochs": 10000, "nmessage": 100, "loss_threshold": None},
    ):
        """
        Compute the SGD optimization according to the chosen optimizer.

        Args:
            initial_parameters (np.ndarray or list): array with initial values
                for gate parameters.
            loss (callable): loss function to train on.
            args (tuple): tuple containing loss function arguments.
            fit_options (dics): extra options to customize the fit. The default
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
        check_fit_arguments(args=args, initial_parameters=initial_parameters)

        default_fit_options = {"epochs": 10000, "nmessage": 100, "loss_threshold": None}

        # update the options with new ones
        default_fit_options.update(fit_options)

        vparams = self.backend.tf.Variable(
            initial_parameters, dtype=self.backend.tf.complex128
        )
        loss_history = []

        def sgd_step():
            """Compute one SGD optimization step according to the chosen optimizer."""
            with self.backend.tf.GradientTape() as tape:
                tape.watch(vparams)
                loss_value = loss(vparams, *args)

            grads = tape.gradient(loss_value, [vparams])
            self.optimizer.apply_gradients(zip(grads, [vparams]))

            return loss_value

        self.backend.compile(loss)
        self.backend.compile(sgd_step)

        # SGD procedure: loop over epochs
        for epoch in range(default_fit_options["epochs"]):  # pragma: no cover
            # early stopping if loss_threshold has been set
            if (
                default_fit_options["loss_threshold"] is not None
                and (epoch != 0)
                and (loss_history[-1] <= default_fit_options["loss_threshold"])
            ):
                break

            loss_value = sgd_step().numpy()
            loss_history.append(loss_value)

            if epoch % default_fit_options["nmessage"] == 0:
                log.info("ite %d : loss %f", epoch, loss_value)

        return loss(vparams, *args).numpy(), vparams.numpy(), loss_history