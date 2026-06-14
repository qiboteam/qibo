import numpy as np

from qibo.config import log, raise_error

__all__ = [
    "ParallelBFGS",
    "QuantumNaturalGradient",
    "cmaes",
    "newtonian",
    "optimize",
    "sgd",
]


def optimize(
    loss,
    initial_parameters,
    args=(),
    method="Powell",
    jac=None,
    hess=None,
    hessp=None,
    bounds=None,
    constraints=(),
    tol=None,
    callback=None,
    options=None,
    compile=False,
    processes=None,
    backend=None,
):
    """Main optimization method. Selects one of the following optimizers:
        - :meth:`qibo.optimizers.cmaes`
        - :meth:`qibo.optimizers.newtonian`
        - :meth:`qibo.optimizers.sgd`

    Args:
        loss (callable): Loss as a function of ``parameters`` and optional extra
            arguments. Make sure the loss function returns a tensor for ``method=sgd``
            and numpy object for all the other methods.
        initial_parameters (np.ndarray): Initial guess for the variational
            parameters that are optimized.
        args (tuple): optional arguments for the loss function.
        method (str): Name of optimizer to use. Can be ``'cma'``, ``'sgd'`` or
            one of the Newtonian methods supported by
            :meth:`qibo.optimizers.newtonian` and ``'parallel_L-BFGS-B'``. ``sgd`` is
            only available for backends based on tensorflow.
        jac (dict): Method for computing the gradient vector for scipy optimizers.
        hess (dict): Method for computing the hessian matrix for scipy optimizers.
        hessp (callable): Hessian of objective function times an arbitrary
            vector for scipy optimizers.
        bounds (sequence or Bounds): Bounds on variables for scipy optimizers.
        constraints (dict): Constraints definition for scipy optimizers.
        tol (float): Tolerance of termination for scipy optimizers.
        callback (callable): Called after each iteration for scipy optimizers.
        options (dict): Dictionary with options. See the specific optimizer
            bellow for a list of the supported options.
        compile (bool): If ``True`` the Tensorflow optimization graph is compiled.
            This is relevant only for the ``'sgd'`` optimizer.
        processes (int): number of processes when using the parallel BFGS method.

    Returns:
        (float, float, custom): Final best loss value; best parameters obtained by the optimizer;         extra: optimizer-specific return object. For scipy methods it
        returns the ``OptimizeResult``, for ``'cma'`` the ``CMAEvolutionStrategy.result``,
        and for ``'sgd'`` the options used during the optimization.


    Example:
        .. testcode::

            import numpy as np
            from qibo import Circuit, gates
            from qibo.optimizers import optimize

            # create custom loss function
            # make sure the return type matches the optimizer requirements.
            def myloss(parameters, circuit):
                circuit.set_parameters(parameters)
                return np.square(np.sum(circuit().state())) # returns numpy array

            # create circuit ansatz for two qubits
            circuit = Circuit(2)
            circuit.add(gates.RY(0, theta=0))

            # optimize using random initial variational parameters
            initial_parameters = np.random.uniform(0, 2, 1)
            best, params, extra = optimize(myloss, initial_parameters, args=(circuit))

            # set parameters to circuit
            circuit.set_parameters(params)
    """
    from qibo.backends import _check_backend  # pylint: disable=import-outside-toplevel

    backend = _check_backend(backend)
    if method == "cma":
        if bounds is not None:  # pragma: no cover
            raise_error(
                RuntimeError,
                "The keyword 'bounds' cannot be used with the cma optimizer. Please use 'options' instead as defined by the cma documentation: ex. options['bounds'] = [0.0, 1.0].",
            )

        return cmaes(
            loss, backend.to_numpy(initial_parameters), args, callback, options
        )

    if method == "sgd":
        return sgd(loss, initial_parameters, args, callback, options, compile, backend)

    return newtonian(
        loss,
        initial_parameters,
        args,
        method,
        jac,
        hess,
        hessp,
        bounds,
        constraints,
        tol,
        callback,
        options,
        processes,
        backend,
    )


def cmaes(loss, initial_parameters, args=(), callback=None, options=None):
    """Genetic optimizer based on `pycma <https://github.com/CMA-ES/pycma>`_.

    Args:
        loss (callable): Loss as a function of variational parameters to be
            optimized.
        initial_parameters (np.ndarray): Initial guess for the variational
            parameters.
        args (tuple): optional arguments for the loss function.
        callback (list[callable]): List of callable called after each optimization
            iteration. According to cma-es implementation take ``CMAEvolutionStrategy``
            instance as argument.
            See: https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html.
        options (dict): Dictionary with options accepted by the ``cma``
            optimizer. The user can use ``import cma; cma.CMAOptions()`` to view the
            available options.
    """
    import cma

    es = cma.CMAEvolutionStrategy(initial_parameters, sigma0=1.7, inopts=options)

    if callback is not None:
        while not es.stop():
            solutions = es.ask()
            objective_values = [loss(x, *args) for x in solutions]
            for solution in solutions:
                callback(solution)
            es.tell(solutions, objective_values)
            es.logger.add()
    else:
        es.optimize(loss, args=args)

    return es.result.fbest, es.result.xbest, es.result


def newtonian(
    loss,
    initial_parameters,
    args=(),
    method="Powell",
    jac=None,
    hess=None,
    hessp=None,
    bounds=None,
    constraints=(),
    tol=None,
    callback=None,
    options=None,
    processes=None,
    backend=None,
):
    """Newtonian optimization approaches based on ``scipy.optimize.minimize``.

    For more details check the `scipy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

    .. note::
        When using the method ``parallel_L-BFGS-B`` the ``processes`` option controls the
        number of processes used by the parallel L-BFGS-B algorithm through the ``multiprocessing`` library.
        By default ``processes=None``, in this case the total number of logical cores are used.
        Make sure to select the appropriate number of processes for your computer specification,
        taking in consideration memory and physical cores. In order to obtain optimal results
        you can control the number of threads used by each process with the ``qibo.set_threads`` method.
        For example, for small-medium size circuits you may benefit from single thread per process, thus set
        ``qibo.set_threads(1)`` before running the optimization.

    Args:
        loss (callable): Loss as a function of variational parameters to be
            optimized.
        initial_parameters (np.ndarray): Initial guess for the variational
            parameters.
        args (tuple): optional arguments for the loss function.
        method (str): Name of method supported by ``scipy.optimize.minimize`` and ``'parallel_L-BFGS-B'`` for
            a parallel version of L-BFGS-B algorithm.
        jac (dict): Method for computing the gradient vector for scipy optimizers.
        hess (dict): Method for computing the hessian matrix for scipy optimizers.
        hessp (callable): Hessian of objective function times an arbitrary
            vector for scipy optimizers.
        bounds (sequence or Bounds): Bounds on variables for scipy optimizers.
        constraints (dict): Constraints definition for scipy optimizers.
        tol (float): Tolerance of termination for scipy optimizers.
        callback (callable): Called after each iteration for scipy optimizers.
        options (dict): Dictionary with options accepted by
            ``scipy.optimize.minimize``.
        processes (int): number of processes when using the parallel BFGS method.
    """
    if method == "parallel_L-BFGS-B":  # pragma: no cover
        o = ParallelBFGS(
            loss,
            args=args,
            processes=processes,
            bounds=bounds,
            callback=callback,
            options=options,
        )
        m = o.run(backend.to_numpy(initial_parameters))
    else:
        from scipy.optimize import minimize

        m = minimize(
            loss,
            backend.to_numpy(initial_parameters),
            args=args,
            method=method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            callback=callback,
            options=options,
        )
    return m.fun, m.x, m


def sgd(
    loss,
    initial_parameters,
    args=(),
    callback=None,
    options=None,
    compile=False,
    backend=None,
):
    """Stochastic Gradient Descent (SGD) optimizer using Tensorflow backpropagation.

    See `tf.keras.Optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_
    for a list of the available optimizers for Tensorflow.
    See `torch.optim <https://pytorch.org/docs/stable/optim.html>`_ for a list of the available
    optimizers for PyTorch.

    Args:
        loss (callable): Loss as a function of variational parameters to be
            optimized.
        initial_parameters (np.ndarray): Initial guess for the variational
            parameters.
        args (tuple): optional arguments for the loss function.
        callback (callable): Called after each iteration.
        options (dict): Dictionary with options for the SGD optimizer. Supports
            the following keys:

            - ``'optimizer'`` (str, default: ``'Adagrad'``): Name of optimizer.
            - ``'learning_rate'`` (float, default: ``'1e-3'``): Learning rate.
            - ``'nepochs'`` (int, default: ``1e6``): Number of epochs for optimization.
            - ``'nmessage'`` (int, default: ``1e3``): Every how many epochs to print
              a message of the loss function.
    """

    sgd_options = {
        "nepochs": 1000000,
        "nmessage": 1000,
        "optimizer": "Adagrad",
        "learning_rate": 0.001,
    }
    if options is not None:
        sgd_options.update(options)

    if backend.platform == "tensorflow":  # pragma: no cover
        return _sgd_tf(
            loss,
            initial_parameters,
            args,
            sgd_options,
            compile,
            backend,
            callback=callback,
        )

    if backend.platform == "pytorch":  # pragma: no cover
        if compile:
            log.warning(
                "PyTorch does not support compilation of the optimization graph."
            )
        return _sgd_torch(
            loss, initial_parameters, args, sgd_options, backend, callback=callback
        )

    raise_error(RuntimeError, "SGD optimizer requires Tensorflow or PyTorch backend.")


def _sgd_torch(
    loss, initial_parameters, args, sgd_options, backend, callback=None
):  # pragma: no cover

    vparams = initial_parameters
    optimizer = getattr(backend.engine.optim, sgd_options["optimizer"])(
        params=[vparams], lr=sgd_options["learning_rate"]
    )

    for e in range(sgd_options["nepochs"]):
        optimizer.zero_grad()
        l = loss(vparams, *args)
        l.backward()
        optimizer.step()
        if callback is not None:
            callback(backend.to_numpy(vparams))
        if e % sgd_options["nmessage"] == 1:
            log.info("ite %d : loss %f", e, l.item())

    return loss(vparams, *args).item(), vparams.detach().numpy(), sgd_options


def _sgd_tf(
    loss, initial_parameters, args, sgd_options, compile, backend, callback=None
):  # pragma: no cover

    vparams = backend.tf.Variable(initial_parameters)
    optimizer = getattr(backend.tf.optimizers, sgd_options["optimizer"])(
        learning_rate=sgd_options["learning_rate"]
    )

    def opt_step():
        with backend.tf.GradientTape() as tape:
            l = loss(vparams, *args)
        grads = tape.gradient(l, [vparams])
        optimizer.apply_gradients(zip(grads, [vparams]))
        return l

    if compile:
        loss = backend.compile(loss)
        opt_step = backend.compile(opt_step)

    for e in range(sgd_options["nepochs"]):
        l = opt_step()
        if callback is not None:
            callback(vparams)
        if e % sgd_options["nmessage"] == 1:
            log.info("ite %d : loss %f", e, l.numpy())

    return loss(vparams, *args).numpy(), vparams.numpy(), sgd_options


class ParallelBFGS:  # pragma: no cover
    """Computes the L-BFGS-B using parallel evaluation using multiprocessing.
    This implementation here is based on https://doi.org/10.32614/RJ-2019-030.

    Args:
        function (function): loss function which returns a numpy object.
        args (tuple): optional arguments for the loss function.
        bounds (list): list of bound values for ``scipy.optimize.minimize`` L-BFGS-B.
        callback (function): function callback ``scipy.optimize.minimize`` L-BFGS-B.
        options (dict): follows ``scipy.optimize.minimize`` syntax for L-BFGS-B.
        processes (int): number of processes when using the paralle BFGS method.
    """

    import numpy as np

    def __init__(
        self,
        function,
        args=(),
        bounds=None,
        callback=None,
        options=None,
        processes=None,
    ):
        self.function = function
        self.args = args
        self.xval = None
        self.function_value = None
        self.jacobian_value = None
        self.precision = np.finfo("float64").eps
        self.bounds = bounds
        self.callback = callback
        self.options = options
        self.processes = processes

    def run(self, x0):
        """Executes parallel L-BFGS-B minimization.
        Args:
            x0 (numpy.array): guess for initial solution.

        Returns:
            scipy.minimize result object
        """
        from scipy.optimize import minimize

        out = minimize(
            fun=self.fun,
            x0=x0,
            jac=self.jac,
            method="L-BFGS-B",
            bounds=self.bounds,
            callback=self.callback,
            options=self.options,
        )
        out.hess_inv = out.hess_inv * np.identity(len(x0))
        return out

    @staticmethod
    def _eval_approx(eps_at, fun, x, eps):
        if eps_at == 0:
            x_ = x
        else:
            x_ = x.copy()
            if eps_at <= len(x):
                x_[eps_at - 1] += eps
            else:
                x_[eps_at - 1 - len(x)] -= eps
        return fun(x_)

    def evaluate(self, x, eps=1e-8):
        if not (
            self.xval is not None and all(abs(self.xval - x) <= self.precision * 2)
        ):
            eps_at = range(len(x) + 1)
            self.xval = x.copy()

            def operation(epsi):
                return self._eval_approx(
                    epsi, lambda y: self.function(y, *self.args), x, eps
                )

            from joblib import Parallel, delayed

            ret = Parallel(self.processes, prefer="threads")(
                delayed(operation)(epsi) for epsi in eps_at
            )
            self.function_value = ret[0]
            self.jacobian_value = (ret[1 : (len(x) + 1)] - self.function_value) / eps

    def fun(self, x):
        self.evaluate(x)
        return self.function_value

    def jac(self, x):
        self.evaluate(x)
        return self.jacobian_value


class QuantumNaturalGradient:
    r"""Quantum Natural Gradient optimizer for parametrized circuits.

    Implements the Quantum Natural Gradient update rule

    .. math::

        \theta_{t + 1} = \theta_t - \eta (F(\theta_t) + \lambda I)^+ \nabla L,

    where :math:`F(\theta)` is the quantum Fisher information matrix (QFIM).

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): Parametrized circuit to optimize.
        loss_fn (Callable): Loss function. The first two arguments must be the circuit
            and backend used for execution.
        loss_kwargs (dict, optional): Additional keyword arguments for ``loss_fn``.
        initial_parameters (ArrayLike, optional): Initial circuit parameters. If ``None``,
            the parameters already stored in ``circuit`` are used.
        gradient_fn (Callable, optional): Function returning the Euclidean gradient with
            respect to the circuit parameters. The first two arguments must be the circuit
            and backend. If ``None``, a central finite-difference gradient is used.
        gradient_kwargs (dict, optional): Additional keyword arguments for
            ``gradient_fn``.
        learning_rate (float, optional): QNG step size. Defaults to ``0.01``.
        regularization (float, optional): Non-negative diagonal regularization added to
            the QFIM before solving the natural-gradient system. Defaults to ``1e-8``.
        finite_difference_epsilon (float, optional): Shift used when ``gradient_fn`` is
            not provided. Defaults to ``1e-6``.
        qfim_kwargs (dict, optional): Additional keyword arguments passed to
            :func:`qibo.quantum_info.quantum_fisher_information_matrix`, such as
            ``initial_state`` or ``return_complex``.
        callback (Callable, optional): Function called after every optimization step with
            keyword arguments ``iter_num``, ``loss``, ``parameters``, and
            ``natural_gradient``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Execution backend.
            If ``None``, the current qibo backend is used.

    References:
        J. Stokes, J. Izaac, N. Killoran, and G. Carleo, *Quantum Natural Gradient*,
        `Quantum 4, 269 (2020) <https://doi.org/10.22331/q-2020-05-25-269>`_.
    """

    def __init__(
        self,
        circuit,
        loss_fn,
        loss_kwargs=None,
        initial_parameters=None,
        gradient_fn=None,
        gradient_kwargs=None,
        learning_rate=0.01,
        regularization=1e-8,
        finite_difference_epsilon=1e-6,
        qfim_kwargs=None,
        callback=None,
        backend=None,
    ):
        from qibo.backends import _check_backend
        from qibo.models.circuit import Circuit

        if not isinstance(circuit, Circuit):
            raise_error(
                TypeError,
                "``circuit`` must be an instance of ``qibo.models.circuit.Circuit``.",
            )
        if not callable(loss_fn):
            raise_error(TypeError, "``loss_fn`` must be callable.")
        if gradient_fn is not None and not callable(gradient_fn):
            raise_error(TypeError, "``gradient_fn`` must be callable or ``None``.")
        if learning_rate <= 0:
            raise_error(ValueError, "``learning_rate`` must be positive.")
        if regularization < 0:
            raise_error(ValueError, "``regularization`` must be non-negative.")
        if finite_difference_epsilon <= 0:
            raise_error(ValueError, "``finite_difference_epsilon`` must be positive.")

        self.circuit = circuit
        self.loss_fn = loss_fn
        self.loss_kwargs = {} if loss_kwargs is None else loss_kwargs
        self.gradient_fn = gradient_fn
        self.gradient_kwargs = {} if gradient_kwargs is None else gradient_kwargs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.finite_difference_epsilon = finite_difference_epsilon
        self.qfim_kwargs = {} if qfim_kwargs is None else qfim_kwargs
        self.callback = callback
        self.backend = _check_backend(backend)

        self.n_calls_loss = 0
        self.n_calls_gradient = 0
        self.n_calls_qfim = 0

        if initial_parameters is not None:
            self.circuit.set_parameters(self._as_flat_array(initial_parameters))

        self.parameters = self.current_parameters()
        if len(self.parameters) == 0:
            raise_error(
                ValueError,
                "``QuantumNaturalGradient`` requires a parametrized circuit.",
            )

    def _as_flat_array(self, values):
        if hasattr(self.backend, "to_numpy"):
            values = self.backend.to_numpy(values)
        return np.asarray(values, dtype=float).reshape(-1)

    def _as_scalar(self, value):
        if hasattr(self.backend, "to_numpy"):
            value = self.backend.to_numpy(value)
        return float(np.asarray(value).reshape(()))

    def current_parameters(self):
        """Return current circuit parameters as a flat NumPy array."""
        return self._as_flat_array(
            self.circuit.get_parameters(output_format="flatlist")
        )

    def loss(self, circuit=None, backend=None):
        """Evaluate the loss and increment the loss-call counter."""
        self.n_calls_loss += 1
        if circuit is None:
            circuit = self.circuit
        if backend is None:
            backend = self.backend
        return self._as_scalar(self.loss_fn(circuit, backend, **self.loss_kwargs))

    def metric_tensor(self):
        """Calculate the QFIM for the current circuit parameters."""
        from qibo.quantum_info import quantum_fisher_information_matrix

        parameters = self.current_parameters()
        qfim_parameters = self.backend.cast(parameters, dtype=self.backend.float64)
        kwargs = dict(self.qfim_kwargs)
        self.n_calls_qfim += 1
        try:
            metric = quantum_fisher_information_matrix(
                self.circuit,
                parameters=qfim_parameters,
                backend=self.backend,
                **kwargs,
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                "``QuantumNaturalGradient`` requires a qibo backend whose "
                + "``quantum_fisher_information_matrix`` supports automatic "
                + "differentiation, or a custom compatible backend. "
                + f"Current backend platform is {self.backend.platform!r}."
            ) from exc

        metric = self._as_flat_matrix(metric)
        if metric.shape != (len(parameters), len(parameters)):
            raise_error(
                ValueError,
                "The QFIM shape must match the number of trainable parameters. "
                + f"Expected {(len(parameters), len(parameters))}, got {metric.shape}.",
            )
        return metric

    def _as_flat_matrix(self, values):
        if hasattr(self.backend, "to_numpy"):
            values = self.backend.to_numpy(values)
        values = np.asarray(values, dtype=float)
        if values.ndim != 2:
            raise_error(ValueError, "The QFIM must be a two-dimensional matrix.")
        return values

    def gradient(self):
        """Return the Euclidean gradient of the loss."""
        self.n_calls_gradient += 1
        if self.gradient_fn is not None:
            gradient = self.gradient_fn(
                self.circuit, self.backend, **self.gradient_kwargs
            )
            gradient = self._as_flat_array(gradient)
        else:
            gradient = self._finite_difference_gradient()

        if len(gradient) != len(self.current_parameters()):
            raise_error(
                ValueError,
                "The gradient length must match the number of trainable parameters. "
                + f"Expected {len(self.current_parameters())}, got {len(gradient)}.",
            )
        return gradient

    def _finite_difference_gradient(self):
        params = self.current_parameters()
        gradient = np.zeros_like(params, dtype=float)
        epsilon = self.finite_difference_epsilon

        for index in range(len(params)):
            shifted = params.copy()
            shifted[index] += epsilon
            self.circuit.set_parameters(shifted)
            loss_plus = self.loss()

            shifted[index] -= 2 * epsilon
            self.circuit.set_parameters(shifted)
            loss_minus = self.loss()

            gradient[index] = (loss_plus - loss_minus) / (2 * epsilon)

        self.circuit.set_parameters(params)
        return gradient

    def natural_gradient(self):
        """Solve the regularized QNG linear system."""
        metric = self.metric_tensor()
        gradient = self.gradient()
        if self.regularization:
            metric = metric + self.regularization * np.eye(len(gradient))

        try:
            return np.linalg.solve(metric, gradient)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(metric) @ gradient

    def run(self, steps=100):
        """Run QNG optimization for ``steps`` iterations.

        Args:
            steps (int, optional): Number of optimization iterations. Defaults to ``100``.

        Returns:
            Tuple with final loss, loss history, and final parameters.
        """
        if steps < 0:
            raise_error(ValueError, "``steps`` must be non-negative.")

        losses = [self.loss()]
        for iter_num in range(steps):
            params = self.current_parameters()
            natural_gradient = self.natural_gradient()
            updated_parameters = params - self.learning_rate * natural_gradient
            self.circuit.set_parameters(updated_parameters)

            loss_value = self.loss()
            losses.append(loss_value)

            if self.callback is not None:
                self.callback(
                    iter_num=iter_num + 1,
                    loss=loss_value,
                    parameters=updated_parameters,
                    natural_gradient=natural_gradient,
                )

        self.parameters = self.current_parameters()
        return (
            losses[-1],
            self.backend.cast(losses, dtype=self.backend.float64),
            self.backend.cast(self.parameters, dtype=self.backend.float64),
        )

    def __call__(self, steps=100):
        """Run QNG optimization for ``steps`` iterations."""
        return self.run(steps=steps)
