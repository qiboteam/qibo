def optimize(loss, initial_parameters, method='Powell',
             options=None, compile=False, processes=None, args=()):
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
            :meth:`qibo.optimizers.newtonian` and ``'parallel_L-BFGS-B'``.
        options (dict): Dictionary with options. See the specific optimizer
            bellow for a list of the supported options.
        compile (bool): If ``True`` the Tensorflow optimization graph is compiled.
            This is relevant only for the ``'sgd'`` optimizer.
        args (tuple): optional arguments for the loss function.
    """
    if method == "cma":
        return cma(loss, initial_parameters, options, args)
    elif method == "sgd":
        return sgd(loss, initial_parameters, options, compile, args)
    else:
        return newtonian(loss, initial_parameters, method, options, processes, args)


def cma(loss, initial_parameters, options=None, args=()):
    """Genetic optimizer based on `pycma <https://github.com/CMA-ES/pycma>`_.

    Args:
        loss (callable): Loss as a function of variational parameters to be
            optimized.
        initial_parameters (np.ndarray): Initial guess for the variational
            parameters.
        options (dict): Dictionary with options accepted by the ``cma``.
            optimizer. The user can use ``cma.CMAOptions()`` to view the
            available options.
        args (tuple): optional arguments for the loss function.
    """
    import cma
    r = cma.fmin2(loss, initial_parameters, 1.7, options=options, args=args)
    return r[1].result.fbest, r[1].result.xbest


def newtonian(loss, initial_parameters, method='Powell', options=None, processes=None, args=()):
    """Newtonian optimization approaches based on ``scipy.optimize.minimize``.

    For more details check the `scipy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

    Args:
        loss (callable): Loss as a function of variational parameters to be
            optimized.
        initial_parameters (np.ndarray): Initial guess for the variational
            parameters.
        method (str): Name of method supported by ``scipy.optimize.minimize`` and ``'parallel_L-BFGS-B'`` for
            a parallel version of L-BFGS-B algorithm.
        options (dict): Dictionary with options accepted by
            ``scipy.optimize.minimize``.
        processes (int): number of processes when using the paralle BFGS method.
        args (tuple): optional arguments for the loss function.
    """
    if method == 'parallel_L-BFGS-B':
        from qibo.config import raise_error, get_device
        if "GPU" in get_device(): # pragma: no cover
            raise_error(RuntimeError, "Parallel L-BFGS-B cannot be used with GPU.")
        o = ParallelBFGS(loss, args=args, options=options, processes=processes)
        m = o.run(initial_parameters)
    else:
        from scipy.optimize import minimize
        m = minimize(loss, initial_parameters, args=args, method=method, options=options)
    return m.fun, m.x


def sgd(loss, initial_parameters, options=None, compile=False, args=()):
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
            l = loss(vparams, *args)
        grads = tape.gradient(l, [vparams])
        optimizer.apply_gradients(zip(grads, [vparams]))
        return l

    if compile:
        opt_step = K.function(opt_step)

    for e in range(sgd_options["nepochs"]):
        l = opt_step()
        if e % sgd_options["nmessage"] == 1:
            log.info('ite %d : loss %f', e, l.numpy())

    return loss(vparams, *args).numpy(), vparams.numpy()


class ParallelBFGSResources: # pragma: no cover
    """Auxiliary singleton class for sharing memory objects in a
    multiprocessing environment when performing a parallel_L-BFGS-B
    minimization procedure.

    This class takes care of duplicating resources for each process
    and calling the respective loss function.
    """
    import multiprocessing as mp
    mp.set_start_method('fork') # enforce on Darwin

    # private objects holding the state
    _instance = None
    # dict with of shared objects
    _objects_per_process = {}
    custom_loss = None
    lock = None
    args = ()

    def __new__(cls, *args, **kwargs):
        """Creates singleton instance."""
        if cls._instance is None:
            cls._instance = super(ParallelBFGSResources, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def loss(self, params):
        """Computes loss (custom_loss) for a specific set of parameters.
        This class performs the lock mechanism to duplicate objects
        for each process.
        """
        # lock to avoid race conditions
        self.lock.acquire()
        # get process name
        pname = self.mp.current_process().name
        # check if there are objects already stored
        args = self._objects_per_process.get(pname, None)
        if args is None:
            args = []
            for obj in self.args:
                try:
                    # copy object if copy method is available
                    copy = obj.copy(deep=True)
                except AttributeError:
                    # if copy is not implemented just use the original object
                    copy = obj
                except Exception as e:
                    # use print otherwise the message will not appear
                    print('Exception in ParallelBFGSResources', str(e))
                args.append(copy)
            args = tuple(args)
            self._objects_per_process[pname] = args
        # unlock
        self.lock.release()
        # finally compute the loss function
        return self.custom_loss(params, *args)


class ParallelBFGS: # pragma: no cover
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
    import numpy as np
    import functools
    import itertools
    from qibo.config import DTYPES

    def __init__(self, function, args=(), bounds=None,
                 callback=None, options=None, processes=None):
        ParallelBFGSResources().args = args
        ParallelBFGSResources().custom_loss = function
        self.xval = None
        self.function_value = None
        self.jacobian_value = None
        self.precision = self.np.finfo(self.DTYPES.get("DTYPE").as_numpy_dtype).eps
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
        ParallelBFGSResources().lock = self.mp.Lock()
        with self.mp.Pool(processes=self.processes) as self.pool:
            from scipy.optimize import minimize
            out = minimize(fun=self.fun, x0=x0, jac=self.jac, method='L-BFGS-B',
                           bounds=self.bounds, callback=self.callback, options=self.options)
        out.hess_inv = out.hess_inv * self.np.identity(len(x0))
        return out

    @staticmethod
    def loss(params):
        """Returns singletong loss."""
        return ParallelBFGSResources().loss(params)

    @staticmethod
    def _eval_approx(eps_at, fun, x, eps):
        if eps_at == 0:
            x_ = x
        else:
            x_= x.copy()
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
                                    self.itertools.repeat(self.loss),
                                    self.itertools.repeat(x),
                                    self.itertools.repeat(eps)))
            self.function_value = ret[0]
            self.jacobian_value = (ret[1:(len(x)+1)] - self.function_value ) / eps

    def fun(self, x):
        self.evaluate(x)
        return self.function_value

    def jac(self, x):
        self.evaluate(x)
        return self.jacobian_value
