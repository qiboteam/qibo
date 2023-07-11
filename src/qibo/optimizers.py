from scipy.optimize import basinhopping
#from qibo.noise import NoiseModel
from qibo.config import log, raise_error
from qibo.models import Circuit
from qibo import backends
from qibo.symbols import Symbol
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.derivative import calculate_gradients

import numpy as np
import random
import tensorflow as tf


class Optimizer:
    """Parent optimizer"""
    
    def __init__(self, initial_parameters, args=(), loss=None, **kwargs):
        if not loss:
            self.loss_function = self.base_loss
        else:
            self.loss_function = loss
        self.args = args
        self.backend = None
        self.inital_parameters = initial_parameters

        # natural gradient
        self.natgrad = False
        if self.natgrad:
            print("Using Natural Gradient")

        if not isinstance(initial_parameters, list) and not isinstance(initial_parameters, np.ndarray):
            raise("Parameters must be a list of Parameter objects or a numpy array")


    def base_loss(self, result, label):
        """Standard squared error loss"""
        loss = (result - label) ** 2

        return loss

    def set_options(self, updates):
        self.options.update(updates)


class SGD(Optimizer):

    def __init__(self, circuit, parameters, args=(), loss=None, **kwargs):
        super().__init__(parameters, args, loss=loss, **kwargs)

        # circuit
        if not isinstance(circuit, Circuit):
            raise("Circuit is not of correct object type")
        
        self._circuit = circuit
        self.nqubits = self._circuit.nqubits

        # parameters
        self.paramInputs = parameters
        self.params = self._get_params(trainable=True)
        self.nparams = len(self.params)

        # options
        self.options = {
            "epochs": 1000,
            "learning_rate": 0.045,
            "batches": 1,
            "J_threshold": 1e-3,
            "shift_rule": "psr"
        }
        self.set_options(kwargs)

    def _get_params(self, trainable=False, feature=None):
        """Creates an array with the trainable parameters"""
        if isinstance(self.paramInputs, np.ndarray):
            return self.paramInputs
        else:
            params = []
            count = 0
            for Param in self.paramInputs:
                if trainable:
                    params += Param._trainablep
                else:
                    trainablep = self.params[count:count+Param.nparams]
                    count += Param.nparams
                    params.append(Param.get_params(trainablep, feature=feature)) 

            return params

    def loss(self, feature, label):
        """Calculates loss and its derivative"""
        params = self._get_params(trainable=False, feature=feature)
        result = tf.Variable(self.run_circuit(params, nshots=1024))
        with tf.GradientTape() as tape:
            loss = self.loss_function(result, label)
        loss_grad = tape.gradient(loss, result)
        return loss.numpy(), loss_grad
    
    def create_hamiltonian(self, qubit, nqubit):
        """
        Creates appropriate Hamiltonian for Fubini-Matrix generation
        Args:
            qubit: qubit number whose state we are interested in
            nqubit: total number of qubits, which determines size of Hamiltonian
        Return:
            hamiltonian: SymbolicHamiltonian
        """
        standard = np.array([[1, 0], [0, 1]])
        target = np.array([[1, 0], [0, 0]])
        hams = []

        for i in range(nqubit):
            if i == qubit:
                hams.append(Symbol(i, target))
            else:
                hams.append(Symbol(i, standard))

        # create Hamiltonian
        obs = np.prod(hams)
        hamiltonian = SymbolicHamiltonian(obs)

        return hamiltonian
        
    def run_circuit(self, parameters, nshots=1024):
        """
        User-facing function which runs the circuit with given parameters and returns the result
        Args:
            parameters: trainable parameters of type Parameter
            nshots: number of shots to average circuit expectation values
        Return:
            results
        """
        self._circuit.set_parameters(parameters)

        state = self._circuit().state()

        # run through parametrized gate
        hamiltonian = self.create_hamiltonian(0, self.nqubits)

        results = hamiltonian.expectation(state)

        return results
        
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

            local_loss, loss_grad = self.loss(feat, label)

            loss += local_loss

            obs_gradients = calculate_gradients(self, feature=feat)  # d<B> N params, N label gradients

            if self.natgrad:
                fubini = None
            #    fubini += generate_fubini(self, feat)
            

            for i in range(self.nparams):
                loss_gradients[i] += obs_gradients[i] * loss_grad

        # gradient average
        loss_gradients /= len(features)
        loss /= len(features)

        # Fubini-Study Metric renormalisation
        if self.natgrad:

            fubini /= len(features)
            loss_gradients = np.dot(np.linalg.inv(fubini), loss_gradients)

        return loss_gradients, loss
           
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
            #np.random.shuffle(idx)
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

    def fit(self, X, y):
        """Performs the optimizations and returns f_best, x_best."""

        if not isinstance(X, np.ndarray):
            raise("X must be a numpy array")
        
        self.labels = y
        self.nsample = len(self.labels)
        
        self.features = X
        
        if not isinstance(y, np.ndarray):
            raise("y must be a numpy array")
        
        if self.backend is None:
            from qibo.backends import GlobalBackend

            self.backend = GlobalBackend()

        return self.sgd(self.options)
    

class CMAES(Optimizer):
    def __init__(self, initial_parameters, args=(), loss=None, **kwargs):
        super().__init__(initial_parameters, args, loss, **kwargs)

    def fit(loss, initial_parameters, args=(), options=None):
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
    

class Newtonian(Optimizer):
    def __init__(self, initial_parameters, args=(), loss=None, **kwargs):
        super().__init__(initial_parameters, args, loss, **kwargs)

    def fit(self,
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
            backend=None):
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
                self.loss,
                args=self.args,
                processes=processes,
                bounds=bounds,
                callback=callback,
                options=options,
            )
            m = o.run(self.initial_parameters)
        else:
            from scipy.optimize import minimize

            m = minimize(
                self.loss,
                self.initial_parameters,
                args=self.args,
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
