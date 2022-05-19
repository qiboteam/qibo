# Optimizers

def optimize(loss, initial_parameters, args=(), method='Powell',
             jac=None, hess=None, hessp=None, bounds=None, constraints=(),
             tol=None, callback=None, options=None, compile=False):
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
            :meth:`qibo.optimizers.newtonian`. ``sgd`` is
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

    Returns:
        (float, float, custom): Final best loss value; best parameters obtained by the optimizer;         extra: optimizer-specific return object. For scipy methods it
        returns the ``OptimizeResult``, for ``'cma'`` the ``CMAEvolutionStrategy.result``,
        and for ``'sgd'`` the options used during the optimization.


    Example:
        .. testcode::

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
                         callback, options)


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
              tol=None, callback=None, options=None):
    """Newtonian optimization approaches based on ``scipy.optimize.minimize``.

    For more details check the `scipy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

    Args:
        loss (callable): Loss as a function of variational parameters to be
            optimized.
        initial_parameters (np.ndarray): Initial guess for the variational
            parameters.
        args (tuple): optional arguments for the loss function.
        method (str): Name of method supported by ``scipy.optimize.minimize`.
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
    """
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
    from qibo import K
    from qibo.config import log, raise_error
    if not K.supports_gradients:
        raise_error(RuntimeError, "SGD optimizer requires Tensorflow backend.")

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
