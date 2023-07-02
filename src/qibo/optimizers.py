from scipy.optimize import basinhopping
#from qibo.noise import NoiseModel
from qibo.config import log, raise_error
from qibo.models import Circuit
from qibo import gates
import numpy as np
import random
import tensorflow as tf


class Optimizer:
    """Parent optimizer"""
    
    def __init__(self, loss, args=(), method="sgd", options=None):
        self.loss = loss
        self.method = method
        self.args = args
        self.options = options #shift rule, epochs, J_iteration, fubini, ...
        self.backend = None
        self.shift_rule = "psr"

        # natural gradient
        self.natgrad = True
        if self.natgrad:
            print("Using Natural Gradient")

        if not isinstance(args[0], np.ndarray):
            raise("First argument to loss function must be a numpy array with circuit parameters")
        
        self.params = args[0]
        self.nparams = len(self.params)
        self.nparams_main = len(self.params)

        if not isinstance(args[1], Circuit):
            raise("Second argument to loss function must be a Circuit")
        
        self._circuit_main = args[1]
        self._circuit = self._circuit_main
        self.nqubits = self._circuit_main.nqubits


    def fit(self, X, y):
        """Performs the optimizations and returns f_best, x_best."""

        if not isinstance(X, np.ndarray):
            raise("X must be a numpy array")
        
        self.labels = X
        self.nsample = len(self.labels)
        
        if not isinstance(y, np.ndarray):
            raise("y must be a numpy array")
        
        self.features = y

        if self.method == "cma":

            cmaes(self.loss, self.params, self.args, self.options)

        elif self.method == "sgd":

            if self.backend is None:
                from qibo.backends import GlobalBackend

                self.backend = GlobalBackend()
            return self.sgd(self.options)
        
        elif self.method == "basinhopping":
            return
        
        else:
            if self.backend is None:
                from qibo.backends import GlobalBackend

                self.backend = GlobalBackend()

            return newtonian(
                self.loss,
                self.params,
                self.args,
                self.method,
                self.options
            )
        
    def one_prediction(self, feature):
        results = tf.Variable(initial_value=np.zeros(2**self.nqubits), dtype=tf.float64)
        _ = self.loss(self.params, self._circuit, feature, 0, results)
        return results[0]
        
    def set_parameters(self, new_params):
        """
        Function which set the new parameters into the circuit
        Args:
            new_params: np.array of the new parameters; it has to be (3 * nlayers) long
        """
        self.params = new_params
    
    def forward_psr(self, original, factor, **kwargs):
        """
        This function calculates the forward shifted parameter for the parameter shift rule (PSR).
        Args:
            original: original parameter value
            factor: multiplicative factor which rescales based on size of data point
        Returns: shifted parameter value
        """

        shifted = original + np.pi / 2 / factor

        return shifted
    
    def backward_psr(self, original, factor, **kwargs):
        """
        This function calculates the backward shifted parameter for the parameter shift rule (PSR).
        Args:
            original: original parameter value
            factor: multiplicative factor which rescales based on size of data point
        Returns: shifted parameter value
        """

        shifted = original - np.pi / 2 / factor

        return shifted
    

    def shift_parameter(self, i, this_feature, forward_func, backward_func, param):
        """
        Stochastic parameter shift's execution on a single parameter
        Args:
            i: integer index which identify the parameter into self.params
            this_feature: np.array 2**nqubits-long containing the state vector assciated to a data
        Returns: derivative of the observable (here the prediction) with respect to self.params[i]
        """

        original = self.params.copy()
        shifted = self.params.copy()


        # parameters multiplied by data input must be rescaled
        if i % 2 == 0:
            factor = this_feature
        else:
            factor = 1

        # forward
        shifted[i] = forward_func(original=original[i], factor=factor, param=param)

        self.set_parameters(shifted)
   
        forward = self.one_prediction(this_feature)

        # backward
        self.set_parameters(original)

        shifted[i] = backward_func(original=original[i], factor=factor, param=param)

        self.set_parameters(shifted)

        backward = self.one_prediction(this_feature)

        # result
        self.set_parameters(original)

        result = (forward - backward) * factor / 2 ## factor 2 to make gradient equal to actual

        return result
    

    def parameter_shift(self, this_feature):
        """
        Full parameter-shift rule's implementation
        Args:
            this_feature: np.array 2**nqubits-long containing the state vector assciated to a data
        Returns: np.array of the observable's gradients with respect to the variational parameters
        """

        # parameter shift
        if self.shift_rule == "psr":

            param = 1
            forward_func = self.forward_psr
            backward_func = self.backward_psr
            factor = 1

        # stochastic parameter shift
        elif self.shift_rule == "spsr":

            param = random.random() # s
            forward_func = self.forward_stoch
            backward_func = self.backward_stoch
            factor = 1

        # numerical differentiation (central difference)
        else:
            
            param = 0.05 # dtheta
            forward_func = self.forward_diff
            backward_func = self.backward_diff
            factor = 2*param


        obs_gradients = np.zeros(self.nparams, dtype=np.float64)
        for ipar in range(self.nparams):
            obs_gradients[ipar] = self.shift_parameter(ipar, this_feature, forward_func, backward_func, param=param) / factor
        return obs_gradients
    
    def _update_layered_circuit(self, matrix, idx, gate, gate_symbol=None):
        """Helper method for :meth:`qibo.models.circuit.Circuit.draw`."""

        if isinstance(gate, gates.CallbackGate):
            targets = list(range(self.nqubits))
        else:
            targets = list(gate.target_qubits)
        controls = list(gate.control_qubits)

        # identify boundaries
        qubits = targets + controls
        qubits.sort()
        min_qubits_id = qubits[0]
        max_qubits_id = qubits[-1]

        # identify column
        col = idx[targets[0]] if not controls and len(targets) == 1 else max(idx)

        # extend matrix
        for iq in range(self.nqubits):
            matrix[iq].extend((1 + col - len(matrix[iq])) * [None])

        # fill
        for iq in range(min_qubits_id, max_qubits_id + 1):
            if iq in targets:
                matrix[iq][col] = gate
            elif iq in controls:
                matrix[iq][col] = None
            else:
                matrix[iq][col] = None

        # update indexes
        if not controls and len(targets) == 1:
            idx[targets[0]] += 1
        else:
            idx = [col + 1] * self.nqubits

        return matrix, idx
    
    def layered_circuit(self):
        """Draw text circuit using unicode symbols.

        Args:
            line_wrap (int): maximum number of characters per line. This option
                split the circuit text diagram in chunks of line_wrap characters.
            legend (bool): If ``True`` prints a legend below the circuit for
                callbacks and channels. Default is ``False``.

        Return:
            String containing text circuit diagram.
        """
        # build string representation of gates
        matrix = [[] for _ in range(self.nqubits)]
        idx = [0] * self.nqubits

        for gate in self._circuit_main.queue:
            if isinstance(gate, gates.FusedGate):
                # start fused gate
                matrix, idx = self._update_layered_circuit(matrix, idx, gate, "[")
                # draw gates contained in the fused gate
                for subgate in gate.gates:
                    matrix, idx = self._update_layered_circuit(matrix, idx, subgate)
                # end fused gate
                matrix, idx = self._update_layered_circuit(matrix, idx, gate, "]")
            else:
                matrix, idx = self._update_layered_circuit(matrix, idx, gate)

        return matrix
    
    def generate_fubini(self, feature, method="variance"):
        """Generate the Fubini-Study metric tensor"""

        fubini = np.zeros((self.nparams, self.nparams))
        original = self.params.copy()

        if method == "hessian":

            shifted = self.params.copy()

            phi = self.retrieve_state(feature)

            for i in range(self.nparams):
                if i % 2 == 0:
                    factor = feature
                else:
                    factor = 1
                shifted[i] = self.forward_diff(original=original[i], factor=factor, param=np.pi)
                self.set_parameters(shifted)
                phi_prime = self.retrieve_state(feature)

                self.set_parameters(original)
                fubini[i, i] = 1/4 * (1 - (np.abs(np.dot(phi, phi_prime)))**2)

        elif method == "variance":

            # trainable and non trainable should be separated into different columns! Not sure
            matrix = self.layered_circuit()

            # empty circuit
            self.subcircuit = Circuit(self.nqubits, density_matrix=True)

            for col in range(len(matrix[0])):

                variational_flag = False

                for gate in matrix[:, col]:
                   
                    if isinstance(gate, gates.ParametrizedGate):
                        variational_flag = True

                if variational_flag:

                    for i in range(self.nqubits):
                        self.subcircuit.add(gates.M(i), basis=gates.Z) # change basis

                    #   3) remove measurement gate

                    # add gate to circuit


            # run through all layers one by one
            for cparam in range(0, self.nparams, 2):

                if cparam % 4 == 0:
                    basis = gates.Z
                else:
                    basis = gates.Y


                # build subcircuit
                self._circuit = self.ansatz(self.layers, cparam, basis=basis)
                self.nparams = cparam

                if cparam > 0:
                    self.set_parameters(self.params[:cparam])

                state = self.retrieve_state(feature)

                self.set_parameters(original)

                fubini[cparam, cparam] = state[0] - state[0]**2
                fubini[cparam + 1, cparam + 1] = state[0] - state[0]**2

            # restore full circuit
            self._circuit = self._circuit_main
            self.nparams = self.nparams_main


        return fubini
        
    def dloss(self, features, labels):
        """
        This function calculates the loss function's gradients with respect to self.params
        Args:
            features: np.matrix containig the n_sample-long vector of states
            labels: np.array of the labels related to features
        Returns: np.array of length self.nparams containing the loss function's gradients
        """
        loss_gradients = np.zeros(self.nparams)
        loss = 0

        if self.natgrad:
            fubini = np.zeros((self.nparams, self.nparams))

        for feat, label in zip(features, labels):
        
            results = tf.Variable(initial_value=np.zeros(2**self._circuit.nqubits), dtype=tf.float64)
            with tf.GradientTape() as tape:
                local_loss = self.loss(self.params, self._circuit, feat, label, results)

            loss += local_loss
            grads = tape.gradient(local_loss, results)

            obs_gradients = self.parameter_shift(feat)  # d<B> N params, N label gradients

            if self.natgrad:
                fubini += self.generate_fubini(feat)

            for i in range(self.nparams):
                loss_gradients[i] += obs_gradients[i] * grads[0]

        # gradient average
        loss_gradients /= len(features)

        # Fubini-Study Metric renormalisation
        if self.natgrad:

            fubini /= len(features)
            loss_gradients = np.dot(np.linalg.inv(fubini), loss_gradients)

        return loss_gradients, (loss.numpy() / len(features))
        
    def AdamDescent(
        self,
        learning_rate,
        m,
        v,
        features,
        labels,
        iteration,
        beta_1=0.85,
        beta_2=0.99,
        epsilon=1e-8,
    ):
        """
        Implementation of the Adam optimizer: during a run of this function parameters are updated.
        Furthermore, new values of m and v are calculated.
        Args:
            learning_rate: np.float value of the learning rate
            m: momentum's value before the execution of the Adam descent
            v: velocity's value before the execution of the Adam descent
            features: np.matrix containig the n_sample-long vector of states
            labels: np.array of the labels related to features
            iteration: np.integer value corresponding to the current training iteration
            beta_1: np.float value of the Adam's beta_1 parameter; default 0.85
            beta_2: np.float value of the Adam's beta_2 parameter; default 0.99
            epsilon: np.float value of the Adam's epsilon parameter; default 1e-8
        Returns: np.float new values of momentum and velocity
        """

        grads, loss = self.dloss(features, labels)

        for i in range(self.nparams):
            m[i] = beta_1 * m[i] + (1 - beta_1) * grads[i]
            v[i] = beta_2 * v[i] + (1 - beta_2) * grads[i] * grads[i]
            mhat = m[i] / (1.0 - beta_1 ** (iteration + 1))
            vhat = v[i] / (1.0 - beta_2 ** (iteration + 1))
            self.params[i] -= learning_rate * mhat / (np.sqrt(vhat) + epsilon)

        return m, v, loss
        
    def sgd(self, options):
        """
        This function performs the full Adam descent's procedure
        Args:
            epochs: np.integer value corresponding to the epochs of training
            learning_rate: np.float value of the learning rate
            batches: np.integer value of the number of batches which divide the dataset
            J_treshold: np.float value of the desired loss function's treshold
        Returns: list of loss values, one for each epoch
        """
        options = {
            "epochs": 1000,
            "learning_rate": 0.045,
            "batches": 1,
            "J_threshold": 1e-5,
            "shift_rule": "psr"
        }

        losses = []
        indices = []

        # create index list
        idx = np.arange(0, self.nsample)

        m = np.zeros(self.nparams)
        v = np.zeros(self.nparams)

        # create index blocks on which we run
        for ib in range(options["batches"]):
            indices.append(np.arange(ib, self.nsample, options["batches"]))

        iteration = 0


        for epoch in range(options["epochs"]):
            if epoch != 0 and losses[-1] < options["J_threshold"]:
                print(
                    "Desired sensibility is reached, here we stop: ",
                    iteration,
                    " iteration",
                )
                break
            # shuffle index list
            np.random.shuffle(idx)
            # run over the batches
            for ib in range(options["batches"]):
                iteration += 1

                features = self.features[idx[indices[ib]]]

                labels = self.labels[idx[indices[ib]]]
                # update parameters
                m, v, this_loss = self.AdamDescent(
                    options["learning_rate"], m, v, features, labels, iteration
                )
                # track the training
                print(
                    "Iteration ",
                    iteration,
                    " epoch ",
                    epoch + 1,
                    " | loss: ",
                    this_loss,
                )
                # in case one wants to plot J as a function of the iterations
                losses.append(this_loss)





##############################################################
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
        if bounds is not None:  # pragma: no cover
            raise_error(
                RuntimeError,
                "The keyword 'bounds' cannot be used with the cma optimizer. Please use 'options' instead as defined by the cma documentation: ex. options['bounds'] = [0.0, 1.0].",
            )
        return cmaes(loss, initial_parameters, args, options)
    elif method == "sgd":
        if backend is None:
            from qibo.backends import GlobalBackend

            backend = GlobalBackend()
        return sgd(loss, initial_parameters, args, options, compile, backend)
    else:
        if backend is None:
            from qibo.backends import GlobalBackend

            backend = GlobalBackend()
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
        m = o.run(initial_parameters)
    else:
        from scipy.optimize import minimize

        m = minimize(
            loss,
            initial_parameters,
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


def sgd(loss, initial_parameters, args=(), options=None, compile=False, backend=None):
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
    #if not backend.name == "tensorflow":
    #    raise_error(RuntimeError, "SGD optimizer requires Tensorflow backend.")


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
        self.precision = self.np.finfo("float64").eps
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
        out.hess_inv = out.hess_inv * self.np.identity(len(x0))
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
