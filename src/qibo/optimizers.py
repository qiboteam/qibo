def cma(loss, initial_parameters): # pragma: no cover
    """Genetic optimizer."""
    # cma is not tested because it takes a lot of time
    import cma
    r = cma.fmin2(loss, initial_parameters, 1.7)
    return r[1].result.fbest, r[1].result.xbest


def newtonian(loss, initial_parameters, method='Powell', options=None):
    """Newtonian approaches based on ``scipy.optimize.minimize``."""
    from scipy.optimize import minimize
    m = minimize(loss, initial_parameters, method=method, options=options)
    return m.fun, m.x


def sgd(loss, initial_parameters, options=None, compile=False):
    """Stochastic Gradient Descent optimizer using Tensorflow backpropagation."""
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


def optimize(loss, initial_parameters, method='Powell',
             options=None, compile=False):
    """Main optimization method.

    Used by :class:`qibo.models.VQE` and :class:`qibo.models.AdiabaticEvolution`.

    Args:
        loss (callable): Loss as a function of ``parameters``.
        initial_parameters (np.ndarray): Initial guess for the variational
            parameters that are optimized.
        method (str): The desired minimization method.
            One of ``"cma"`` (genetic optimizer), ``"sgd"`` (gradient descent) or
            any of the methods supported by
            `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.
        options (dict): a dictionary with options for the different optimizers.
        compile (bool): If ``True`` the Tensorflow optimization graph is compiled.
            Relevant only for the ``"sgd"`` optimizer.
    """
    if method == "cma": # pragma: no cover
        # cma is not tested because it takes a lot of time
        return cma(loss, initial_parameters)
    elif method == "sgd":
        return sgd(loss, initial_parameters, options, compile)
    else:
        return newtonian(loss, initial_parameters, method, options)
