def optimize(loss, initial_parameters, method='Powell',
             options=None, compile=False):
    """Main optimization method. Selects one of the following optimizers:
        - :meth:`qibo.optimizers.cma`
        - :meth:`qibo.optimizers.newtonian`
        - :meth:`qibo.optimizers.sgd`

    Args:
        loss (callable): Loss as a function of ``parameters``.
        initial_parameters (np.ndarray): Initial guess for the variational
            parameters that are optimized.
        method (str): Name of optimizer to use. Can be ``'cma'``, ``'sgd'`` or
            one of the Newtonian methods supported by
            :meth:`qibo.optimizers.newtonian`.
        options (dict): Dictionary with options. See the specific optimizer
            bellow for a list of the supported options.
        compile (bool): If ``True`` the Tensorflow optimization graph is compiled.
            This is relevant only for the ``'sgd'`` optimizer.
    """
    if method == "cma":
        return cma(loss, initial_parameters, options)
    elif method == "sgd":
        return sgd(loss, initial_parameters, options, compile)
    else:
        return newtonian(loss, initial_parameters, method, options)


def cma(loss, initial_parameters, options=None):
    """Genetic optimizer based on `pycma <https://github.com/CMA-ES/pycma>`_.

    Args:
        loss (callable): Loss as a function of variational parameters to be
            optimized.
        initial_parameters (np.ndarray): Initial guess for the variational
            parameters.
        options (dict): Dictionary with options accepted by the ``cma``.
            optimizer. The user can use ``cma.CMAOptions()`` to view the
            available options.
    """
    import cma
    r = cma.fmin2(loss, initial_parameters, 1.7, options=options)
    return r[1].result.fbest, r[1].result.xbest


def newtonian(loss, initial_parameters, method='Powell', options=None):
    """Newtonian optimization approaches based on ``scipy.optimize.minimize``.

    For more details check the `scipy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

    Args:
        loss (callable): Loss as a function of variational parameters to be
            optimized.
        initial_parameters (np.ndarray): Initial guess for the variational
            parameters.
        method (str): Name of method supported by ``scipy.optimize.minimize``.
        options (dict): Dictionary with options accepted by
            ``scipy.optimize.minimize``.
    """
    from scipy.optimize import minimize
    m = minimize(loss, initial_parameters, method=method, options=options)
    return m.fun, m.x


def sgd(loss, initial_parameters, options=None, compile=False):
    """Stochastic Gradient Descent (SGD) optimizer using Tensorflow backpropagation.

    See `tf.keras.Optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_
    for a list of the available optimizers.

    Args:
        loss (callable): Loss as a function of variational parameters to be
            optimized.
        initial_parameters (np.ndarray): Initial guess for the variational
            parameters.
        options (dict): Dictionary with options for the SGD optimizer. Supports
            the following keys:
              - ``'optimizer'`` (str, default: ``'Adagrad'``): Name of optimizer.
              - ``'learning_rate'`` (float, default: ``'1e-3'``): Learning rate.
              - ``'nepochs'`` (int, default: ``1e6``): Number of epochs for optimization.
              - ``'nmessage'`` (int, default: ``1e3``): Every how many epochs to print
                a message of the loss function.
    """
    from qibo import K
    from qibo.config import log
    sgd_options = {"nepochs": 1000000,
                    "nmessage": 1000,
                    "optimizer": "Adagrad",
                    "learning_rate": 0.001}
    if options is not None:
        sgd_options.update(options)

    # proceed with the training
    vparams = K.Variable(initial_parameters)
    optimizer = getattr(K.optimizers, sgd_options["optimizer"])(
        learning_rate=sgd_options["learning_rate"])

    def opt_step():
        with K.GradientTape() as tape:
            l = loss(vparams)
        grads = tape.gradient(l, [vparams])
        optimizer.apply_gradients(zip(grads, [vparams]))
        return l

    if compile:
        opt_step = K.function(opt_step)

    for e in range(sgd_options["nepochs"]):
        l = opt_step()
        if e % sgd_options["nmessage"] == 1:
            log.info('ite %d : loss %f', e, l.numpy())

    return loss(vparams).numpy(), vparams.numpy()
