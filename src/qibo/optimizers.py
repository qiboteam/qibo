from qibo.parallel import ParallelResources, _executor


def optimize(loss, initial_parameters, args=(), method='Powell',
             jac=None, hess=None, hessp=None, bounds=None, constraints=(),
             tol=None, callback=None, options=None, compile=False, processes=None):
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
            :meth:`qibo.optimizers.newtonian` and ``'parallel_L-BFGS-B'``.
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
        loss (float): final best loss value.
        xbest (float): best parameters obtained by the optimizer.
        extra: optimizer specific return object containing. For scipy methods it
            returns the ``OptimizeResult``, for ``'cma'`` the ``CMAEvolutionStrategy.result``,
            and for ``'sgd'`` the options used during the optimization.

    Example:
        ::

            import numpy as np
            from qibo import gates, models
            from qibo.optimizers import optimize

            # create custom loss function
            # make sure the return type matches the optimizer requirements.
            def myloss(parameters, circuit):
                circuit.set_parameters(parameters)
                return np.square(np.sum(circuit())) # returns numpy array

            # create circuit ansatz for two qubits
            circuit = models.Circuit(2)
            circuit.add(gates.RY(0, theta=0))

            # optimize using random initial variational parameters
            initial_parameters = np.random.uniform(0, 2, 1)
            best, params, extra = optimize(myloss, initial_parameters, args=(circuit))

            # set parameters to circuit
            circuit.set_parameters(params)
    """
    if method == "cma":
        return cmaes(loss, initial_parameters, args, options)
    elif method == "sgd":
        return sgd(loss, initial_parameters, args, options, compile)
    else:
        return newtonian(loss, initial_parameters, args, method,
                         jac, hess, hessp, bounds, constraints, tol,
                         callback, options, processes)


def cmaes(loss, initial_parameters, args=(), options=None):
    """Genetic optimizer based on `pycma <https://github.com/CMA-ES/pycma>`_.

    Args:
        loss (callable): Loss as a function of variational parameters to be
            optimized.
        initial_parameters (np.ndarray): Initial guess for the variational
            parameters.
        args (tuple): optional arguments for the loss function.
        options (dict): Dictionary with options accepted by the ``cma``
            optimizer. The user can use ``import cma; cma.CMAOptions()`` to view the
            available options.
    """
    import cma
    r = cma.fmin2(loss, initial_parameters, 1.7, options=options, args=args)
    return r[1].result.fbest, r[1].result.xbest, r


def newtonian(loss, initial_parameters, args=(), method='Powell',
              jac=None, hess=None, hessp=None, bounds=None, constraints=(),
              tol=None, callback=None, options=None, processes=None):
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
    if method == 'parallel_L-BFGS-B':
        from qibo.parallel import _check_parallel_configuration
        _check_parallel_configuration(processes)
        o = ParallelBFGS(loss, args=args, processes=processes,
                         bounds=bounds, callback=callback, options=options)
        m = o.run(initial_parameters)
    else:
        from scipy.optimize import minimize
        m = minimize(loss, initial_parameters, args=args, method=method,
                     jac=jac, hess=hess, hessp=hessp, bounds=bounds, constraints=constraints,
                     tol=tol, callback=callback, options=options)
    return m.fun, m.x, m


def sgd(loss, initial_parameters, args=(), options=None, compile=False):
    """Stochastic Gradient Descent (SGD) optimizer using Tensorflow backpropagation.

    See `tf.keras.Optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_
    for a list of the available optimizers.

    Args:
        loss (callable): Loss as a function of variational parameters to be
            optimized.
        initial_parameters (np.ndarray): Initial guess for the variational
            parameters.
        args (tuple): optional arguments for the loss function.
        options (dict): Dictionary with options for the SGD optimizer. Supports
            the following keys:
              - ``'optimizer'`` (str, default: ``'Adagrad'``): Name of optimizer.
              - ``'learning_rate'`` (float, default: ``'1e-3'``): Learning rate.
              - ``'nepochs'`` (int, default: ``1e6``): Number of epochs for optimization.
              - ``'nmessage'`` (int, default: ``1e3``): Every how many epochs to print
                a message of the loss function.
    """
    # check if gates are using the MatmulEinsum backend
    compatible_backends = {
        "tensorflow_defaulteinsum", "tensorflow_matmuleinsum"}
    from qibo.core.circuit import Circuit
    for argument in args:
        if isinstance(argument, Circuit):
            from qibo import K
            if K.name not in compatible_backends:  # pragma: no cover
                from qibo.config import raise_error
                raise_error(
                    RuntimeError, "SGD requires native Tensorflow backend.")

    from qibo import K
    from qibo.config import log
    sgd_options = {"nepochs": 1000000,
                   "nmessage": 1000,
                   "optimizer": "Adagrad",
                   "learning_rate": 0.001}
    if options is not None:
        sgd_options.update(options)

    # proceed with the training
    vparams = K.optimization.Variable(initial_parameters)
    optimizer = getattr(K.optimization.optimizers, sgd_options["optimizer"])(
        learning_rate=sgd_options["learning_rate"])

    def opt_step():
        with K.optimization.GradientTape() as tape:
            l = loss(vparams, *args)
        grads = tape.gradient(l, [vparams])
        optimizer.apply_gradients(zip(grads, [vparams]))
        return l

    if compile:
        loss = K.compile(loss)
        opt_step = K.compile(opt_step)

    for e in range(sgd_options["nepochs"]):
        l = opt_step()
        if e % sgd_options["nmessage"] == 1:
            log.info('ite %d : loss %f', e, l.numpy())

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
    import multiprocessing as mp
    import functools
    import itertools
    from qibo import K

    def __init__(self, function, args=(), bounds=None,
                 callback=None, options=None, processes=None):
        ParallelResources().arguments = args
        ParallelResources().custom_function = function
        self.xval = None
        self.function_value = None
        self.jacobian_value = None
        self.precision = self.K.np.finfo(self.K.qnp.dtypes("DTYPE")).eps
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
        ParallelResources().lock = self.mp.Lock()
        with self.mp.Pool(processes=self.processes) as self.pool:
            from scipy.optimize import minimize
            out = minimize(fun=self.fun, x0=x0, jac=self.jac, method='L-BFGS-B',
                           bounds=self.bounds, callback=self.callback, options=self.options)
        ParallelResources().reset()
        out.hess_inv = out.hess_inv * self.K.np.identity(len(x0))
        return out

    @staticmethod
    def _eval_approx(eps_at, fun, x, eps):
        if eps_at == 0:
            x_ = x
        else:
            x_ = x.copy()
            if eps_at <= len(x):
                x_[eps_at-1] += eps
            else:
                x_[eps_at-1-len(x)] -= eps
        return fun(x_)

    def evaluate(self, x, eps=1e-8):
        if not (self.xval is not None and all(abs(self.xval - x) <= self.precision*2)):
            eps_at = range(len(x)+1)
            self.xval = x.copy()
            ret = self.pool.starmap(self._eval_approx, zip(eps_at,
                                    self.itertools.repeat(_executor),
                                    self.itertools.repeat(x),
                                    self.itertools.repeat(eps)))
            self.function_value = ret[0]
            self.jacobian_value = (
                ret[1:(len(x)+1)] - self.function_value) / eps

    def fun(self, x):
        self.evaluate(x)
        return self.function_value

    def jac(self, x):
        self.evaluate(x)
        return self.jacobian_value
