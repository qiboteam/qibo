def cma(loss, initial_parameters): # pragma: no cover
    # Genetic optimizer
    import cma
    r = cma.fmin2(loss, initial_parameters, 1.7)
    return r[1].result.fbest, r[1].result.xbest


def newtonian(loss, initial_parameters, method='Powell', options=None):
    # Newtonian approaches
    from scipy.optimize import minimize
    m = minimize(loss, initial_parameters, method=method, options=options)
    return m.fun, m.x


def sgd(loss, initial_parameters, options=None, compile=False):
    from qibo import K
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
            print('ite %d : loss %f' % (e, l.numpy()))

    return loss(vparams).numpy(), vparams.numpy()


def optimize(loss, initial_parameters, method='Powell',
             options=None, compile=False):
    if method == "cma": # pragma: no cover
        return cma(loss, initial_parameters)
    elif method == "sgd":
        return sgd(loss, initial_parameters, options, compile)
    else:
        return newtonian(loss, initial_parameters, method, options)
