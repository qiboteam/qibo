import random
import time
from datetime import datetime

import numpy as np

from qibo import backends
from qibo.config import log, raise_error
from qibo.derivative import (
    build_graph,
    calculate_circuit_gradients,
    create_hamiltonian,
    error_mitigation,
    execute_circuit,
    generate_fubini,
)
from qibo.models import Circuit


class Optimizer:
    """Parent optimizer class

    Args:
        initial_parameters (np.ndarray): array with initial values for gate parameters
        args (tuple): tuple containing loss function arguments
        loss (function): loss function to train on
        save (bool): Flag to set logging on
        loss_function (function): loss function to train on
        params (np.ndarray): array with current values for gate parameters
        backend (qibo.backends.GlobalBackend): backend on which to run circuit
        simulation_start (time.time): simulation start time
        ftime (time.time): start time variable for timing internal processes
        etime (time.time): end time variable for timing internal processes
        name (str): name of current optimisation process
        iteration (int): training iteration number
        initparams (np.ndarray or list): initial parameters
    """

    def __init__(self, initial_parameters, args=(), loss=None, save=False):
        # saving to class objects
        self.loss_function = loss
        if not isinstance(args, tuple):
            self.args = (args,)
        else:
            self.args = args

        self.params = initial_parameters
        self.backend = backends.GlobalBackend()

        # logging
        self.simulation_start = time.time()
        self.ftime = time.time()
        self.etime = None
        self.name = f'Run_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

        self.iteration = 0
        self.save = save

        if not isinstance(initial_parameters, np.ndarray) and not isinstance(
            initial_parameters, list
        ):
            raise TypeError(
                "Parameters must be a list of Parameter objects or a numpy array"
            )

    def set_options(self, updates):
        """Updates self.options dictionary"""
        self.options.update(updates)

    def fun(self, x):
        """Wrapper function to save and preprocess gate parameters

        Args:
            x (np.array): circuit parameters

        Returns:
            (float): loss value"""

        val = self.loss_function(x, *self.args)

        # timing
        self.etime = time.time()
        duration = self.etime - self.ftime
        self.ftime = self.etime

        # saving
        if self.save:
            self.file.write(
                f"Iteration {self.iteration} | loss: {val} | duration: {duration}\n"
            )

        self.iteration += 1
        return val

    def cleanup(self):
        """Cleans up log file and closes it"""

        self.file.write(
            f"##### Total simulation time: {time.time()-self.simulation_start}\n"
        )
        self.file.close()


class SGD(Optimizer):
    """Stochastic Gradient Descent object to optimise function parameters.

    Example:
    .. code-block:: python

        from qibo import Circuit
        from qibo import gates

        circuit = Circuit(nqubits=1)
        c.add(gates.RY(q=0, theta=Parameter(lambda x, th1, th2: th1 * x + th2, [0.1, 0.1], feature=[0.1])))
        c.add(gates.RZ(q=0, theta=Parameter(lambda x, th1, th2: th1 * x + th2, [0.1, 0.1], feature=[0.1])))
        c.add(gates.M(q=0))

        parameters = [0.2, 0.4, 0.2, 0.9]

        optimizer = SGD(circuit=circuit, parameters=parameters)
        X = np.array([0.1, 0.2, 0.3])
        y = np.array([0.2, 0.5, 0.7])
        losses = optimizer.fit(X, y)

    Args:
        circuit (Circuit): the circuit whose parameters will be optimised
        parameters (np.ndarray): array with initial values for gate parameters
        hamiltonian (qibo.hamiltonians.SymbolicHamiltonian): hamiltonian applied to final circuit state
        args (tuple): tuple containing loss function arguments
        loss (function): loss function to train on
        save (bool): Flag to set logging on
    """

    def __init__(
        self,
        circuit,
        parameters,
        hamiltonian=None,
        args=(),
        loss=None,
        loss_derivative=None,
        save=False,
        **kwargs,
    ):
        super().__init__(parameters, args, loss=loss, save=save)

        # circuit
        if not isinstance(circuit, Circuit):
            raise ("Circuit is not of correct object type")

        if isinstance(parameters, list):
            parameters = np.array(parameters)

        if not isinstance(parameters, np.ndarray):
            raise ("Parameters must be a numpy array")

        if not loss:
            raise ("You must provide a loss function")

        self._circuit = circuit
        self.nqubits = self._circuit.nqubits

        # parameters
        self.params = parameters.astype(np.float64)
        self.nparams = len(self.params)

        # loss function derivative
        self.loss_func_deriv = loss_derivative

        # hamiltonian
        if not hamiltonian:
            self.hamiltonian = [create_hamiltonian(0, self.nqubits, self.backend)]
        elif not isinstance(hamiltonian, list):
            self.hamiltonian = [hamiltonian]
        else:
            self.hamiltonian = hamiltonian

        # error mitigation
        self.cdr_params = None
        self.calibration = None

        # options
        self.options = {
            "epochs": 100,
            "learning_rate": 0.045,
            "batches": 1,
            "J_threshold": 1e-3,
            "shift_rule": "psr",
            "var_gates": None,
            "nshots": 1024,
            "natgrad": False,
            "fubini_freq": 0.2,
            "mitigation": False,
            "noise_model": None,
            "adam": False,
            "deterministic": False,
            "beta_1": 0.85,
            "beta_2": 0.99,
            "vanilla": False,
            "qadam": False,
        }
        self.set_options(kwargs)

        # name
        self.name_appendix = ""
        if self.options["adam"]:
            self.name_appendix += "adam"
        if self.options["natgrad"]:
            self.name_appendix += "natgrad"
        if self.options["qadam"]:
            self.name_appendix += "qadam"

        # logging
        self.param_history = np.zeros((self.options["epochs"], self.nparams))
        self.filename = f"results/{self.name}_{self.name_appendix}.txt"
        if save:
            self.file = open(self.filename, "w")

        # natural gradient graph initialisation
        if self.options["natgrad"]:
            self.NGgraph = build_graph(self._circuit, self.nparams, self.nqubits)

    def calculate_loss_func_grad(self, results, labels, idx, delta=1e-6):
        """Calculates loss function derivative with respect to parameter idx

        Args:
            results (np.ndarray): predicted values
            labels (np.ndarray): true values
            idx (int): parameter number with respect to which we calculate the gradient
            delta (float): size of the finite difference perturbation

        Returns:
            (np.ndarray): gradients of loss function w.r.t to feature idx
        """

        grads = np.empty(self.nlabels)

        for lab in range(self.nlabels):
            shifted = np.copy(results)

            # forward
            shifted[idx][lab] += delta
            forward = self.loss_function(
                np.copy(shifted).squeeze(), labels.squeeze(), *self.args
            )

            # backward
            shifted[idx][lab] -= 2 * delta
            backward = self.loss_function(
                np.copy(shifted).squeeze(), labels.squeeze(), *self.args
            )

            # grad
            grads[lab] = (forward - backward) / (2 * delta)

        return grads

    def run_circuit(self, feature, N=1):
        """Backend function which runs the circuit for one feature N times

        Args:
            feature (int or np.ndarray): single input to the circuit
            N (int): number of runs
        Return:
            (np.ndarray): array of expectation values"""

        # set parameters
        self._circuit.set_variational_parameters(self.params, [feature])

        # run circuit
        exp_v = np.zeros((len(self.hamiltonian), N))
        for i, hamiltonian in enumerate(self.hamiltonian):
            for n in range(N):
                exp_v[i, n] = execute_circuit(
                    self.backend,
                    self._circuit,
                    hamiltonian,
                    self.options["nshots"],
                    calibration=self.calibration,
                    initial_state=None,
                    cdr_params=None,
                    deterministic=self.options["deterministic"],
                )

        return exp_v

    def predict(self, feature, N=1):
        """User-facing function which runs the circuit for given input
        features and returns the result.

        Args:
            feature (int or np.ndarray): single input to the circuit
            N (int): number of runs

        Return:
            (np.ndarray): array of expectation values
        """

        if isinstance(feature, np.ndarray):
            results = np.zeros((len(feature), self.nlabels, N))
            for i, feat in enumerate(feature):
                results[i] = self.run_circuit(feat, N)
            return np.squeeze(results)

        else:
            return np.squeeze(self.run_circuit(feature))

    def dloss(self, features, labels):
        """This function calculates the loss function's gradients with respect to self.params,
        as well as the loss per feature.

        Args:
            features (np.ndarray): array containing all possible input states
            labels (np.ndarray): array of the true labels
        Returns:
            (np.array): loss function's gradients
            (float): mean loss
        """

        # setup
        circ_grads = np.zeros(self.nparams)
        results = np.zeros((self.nsample, self.nlabels))
        loss = 0

        # natural gradient setup
        if self.options["natgrad"]:
            fubini = np.zeros((self.nparams, self.nparams))
            scount = int(self.options["fubini_freq"] * self.nsample)
            scount = min(max(scount, 5), self.nsample)
            sample = random.sample([i for i in range(self.nsample)], scount)

        # calculate CDR parameters anew at each epoch
        if self.options["mitigation"]:
            self._circuit.set_variational_parameters(self.params)

            self.cdr_params, self.calibration = error_mitigation(
                self._circuit.to_clifford(),
                self.nqubits,
                self.hamiltonian,
                self.backend,
                self.options["noise_model"],
                self.options["nshots"],
            )
            self.cdr_params = (1, 0)

        # iterate through all data points
        for i, feat in enumerate(features):
            results[i] = self.predict(feat)

            obs_gradients = np.zeros((self.nlabels, self.nparams))
            for h, ham in enumerate(self.hamiltonian):
                obs_gradients[h] = calculate_circuit_gradients(
                    circuit=self._circuit,
                    ham=ham,
                    nparams=self.nparams,
                    shift_rule=self.options["shift_rule"],
                    cdr_params=self.cdr_params,
                    calibration=self.calibration,
                    nshots=self.options["nshots"],
                    deterministic=self.options["deterministic"],
                    var_gates=self.options["var_gates"],
                )  # d<B> N params, N label gradients

            if self.loss_func_deriv is None:
                loss_func_grad = self.calculate_loss_func_grad(
                    np.copy(results), labels, i
                )
            else:
                grad = self.loss_func_deriv(
                    results.squeeze(), labels.squeeze(), *self.args
                )
                loss_func_grad = np.array([grad[i]])

            circ_grads += np.dot(loss_func_grad.T, obs_gradients)

            if self.options["natgrad"] and i in sample:
                self.NGgraph.update_parameters(self._circuit.get_parameters())
                fubini += generate_fubini(
                    self.NGgraph,
                    self.nparams,
                    self.nqubits,
                    initparams=self._circuit.initparams,
                    nshots=self.options["nshots"],
                    mitigation=self.options["mitigation"],
                    noise_model=self.options["noise_model"],
                    deterministic=self.options["deterministic"],
                )  # separate pull request

        # gradient average

        loss = (
            self.loss_function(results.squeeze(), labels.squeeze(), *self.args)
            / self.nsample
        )
        loss_gradients = circ_grads / self.nsample

        # Fubini-Study Metric renormalisation
        if self.options["natgrad"]:
            fubini /= scount  # len(features)
            loss_gradients = np.dot(np.linalg.inv(fubini), loss_gradients)

        # save data
        if self.save:
            self.file.write(
                f"Grads (absolute value: {np.linalg.norm(loss_gradients)}): {loss_gradients.tolist()}\nParams {self.params.tolist()}\nypred: {results.tolist()}\n"
            )

        plot(
            results, self.features, self.labels, self.epoch, loss, name_prependix="sgd"
        )

        return loss_gradients, loss

    def GradientDescent(
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
        """Implementation of one iteration of Gradient Descent.

        During a run of this function parameters are updated.
        Furthermore, new values of m and v are calculated.

        Args:
            learning_rate (float): np.float value of the learning rate
            m (np.ndarray): momentum's value before the execution of the Adam descent
            v (np.ndarray): velocity's value before the execution of the Adam descent
            features (np.ndarray): array containing the n_sample-long vector of states
            labels (np.ndarray): np.array of the labels related to features
            iteration (int): current training iteration
            beta_1 (float): beta_1 parameter; default 0.85
            beta_2 (flaot): beta_2 parameter; default 0.99
            epsilon (float): epsilon parameter; default 1e-8
        Returns:
            (float) new momentum
            (float) new velocity
            (float) new loss
        """
        grads, loss = self.dloss(features, labels)
        # in case one wants to plot J as a function of the iterations
        self.losses.append(loss)
        self.param_history[self.epoch] = self.params

        if self.options["adam"]:
            m = beta_1 * m + (1 - beta_1) * grads
            v = beta_2 * v + (1 - beta_2) * grads * grads
            mhat = m / (1.0 - beta_1 ** (iteration + 1))
            vhat = v / (1.0 - beta_2 ** (iteration + 1))
            self.params -= learning_rate * mhat / (np.sqrt(vhat) + epsilon)

            return m, v, loss

        # QADAM
        elif self.options["qadam"]:
            it = self.iteration + 1
            beta_1 /= it
            beta_2 /= it
            m = beta_1 * m + (1 - beta_1) * grads
            v = beta_2 * v + (1 - beta_2) * grads * grads

            self.params -= learning_rate * m / (np.sqrt(v) + epsilon)

            return m, v, loss

        # vanilla gradient descent
        else:
            self.params -= learning_rate * grads
            return 0, 0, loss

    def sgd(self, options):
        """This function performs the full Gradient Descent

        Args:
            options (dict): gradient descent options
        Returns:
            (list) history of loss values, one for each epoch
        """

        self.losses = []
        indices = []

        # create index list
        idx = np.arange(0, self.nsample)

        m = np.zeros(self.nparams)
        v = np.zeros(self.nparams)

        # create index blocks on which we run
        for ib in range(options["batches"]):
            indices.append(np.arange(ib, self.nsample, options["batches"]))

        if self.save:
            self.file.write(
                f"Number of features: {self.nsample}\n"
                f"Number of parameters: {len(self.params)}\n"
                f"Epochs: {self.options['epochs']}\n"
                f"learning rate: {self.options['learning_rate']}\n"
                f"nshots: {self.options['nshots']}\n"
                f"natgrad: {self.options['natgrad']}\n"
                f"mitigation: {self.options['mitigation']}\n"
                f"noise_model: {self.options['noise_model']}\n"
                f"adam: {self.options['adam']}\n"
                f"qadam: {self.options['qadam']}\n"
                f"beta_1: {self.options['beta_1']}\n"
                f"beta_2: {self.options['beta_2']}\n\n\n"
            )
            self.ftime = time.time()
            self.simulation_start = time.time()
            self.etime = time.time()

        for self.epoch in range(options["epochs"]):
            iteration = 0

            if self.epoch != 0 and self.losses[-1] < options["J_threshold"]:
                print(
                    "Desired sensibility is reached, here we stop: ",
                    self.epoch,
                    " epochs",
                )
                break

            # shuffle index list
            # np.random.shuffle(idx)
            # run over the batches
            for ib in range(options["batches"]):
                iteration += 1

                features = self.features[idx[indices[ib]]]

                labels = self.labels[idx[indices[ib]]]

                # update parameters
                m, v, this_loss = self.GradientDescent(
                    options["learning_rate"],
                    m,
                    v,
                    features,
                    labels,
                    iteration,
                    beta_1=options["beta_1"],
                    beta_2=options["beta_2"],
                )

                # track the training
                print(
                    "Iteration ",
                    iteration,
                    " epoch ",
                    self.epoch + 1,
                    " | loss: ",
                    this_loss,
                )

                if self.save:
                    etime = time.time()
                    self.file.write(
                        f"Iteration {iteration}, epoch {self.epoch + 1} | loss: {this_loss} | duration: {etime-self.etime}\n\n"
                    )
                    self.etime = etime

        if self.save:
            value = min(self.losses)
            idx = self.losses.index(value)
            self.params = self.param_history[idx]
            ypred, ysigma = get_error(self, self.features, "sgd", self.name_appendix)

            plot(
                ypred,
                self.features,
                self.labels,
                idx,
                value,
                ysigma,
                self.name,
                name_appendix=self.name_appendix,
                name_prependix="sgd",
            )

            self.file.write(f"Params {self.params.tolist()}\n")
            self.file.write(f"Best J: {min(self.losses)}\n")
            self.cleanup()

        return self.losses

    def setup(self, X, y):
        """Input and output setup

        Args:
            X (np.ndarray): array containing input values
            y (np.ndarray): array containing true output values
        """

        if not isinstance(y, np.ndarray):
            raise ("y must be a numpy array")

        self.features = X
        if X is None:
            self.nsample = 1
            self.features = np.array([1.0])
        else:
            self.nsample = len(self.features)

        self.labels = y
        if y.ndim == 1:
            self.labels = self.labels.reshape(-1, 1)

        self.nlabels = self.labels.shape[1]

        if self.backend is None:
            from qibo.backends import GlobalBackend

            self.backend = GlobalBackend()

    def fit(self, y=None, X=None):
        """Performs the optimizations and returns f_best, x_best.

        Args:
            X (np.ndarray): array containing input values
            y (np.ndarray): array containing true output values
        """

        random.seed(42)  # CHANGE
        self.setup(X, y)

        return self.sgd(self.options)


class CMAES(Optimizer):
    """Genetic optimizer based on `pycma <https://github.com/CMA-ES/pycma>`_.

    Example:
    .. code-block:: python

        from qibo import Circuit
        from qibo import gates

        circuit = Circuit(nqubits=1)
        c.add(gates.RY(q=0, theta=Parameter(lambda x, th1, th2: th1 * x + th2, [0.1, 0.1], feature=[0.1])))
        c.add(gates.RZ(q=0, theta=Parameter(lambda x, th1, th2: th1 * x + th2, [0.1, 0.1], feature=[0.1])))
        c.add(gates.M(q=0))

        parameters = [0.2, 0.4, 0.2, 0.9]

        optimizer = SGD(circuit=circuit, parameters=parameters)
        X = np.array([0.1, 0.2, 0.3])
        y = np.array([0.2, 0.5, 0.7])
        losses = optimizer.fit(X, y)

    Args:
        intial_parameters (np.ndarray): array with initial values for gate parameters
        hamiltonian (qibo.hamiltonians.SymbolicHamiltonian): hamiltonian applied to final circuit state
        args (tuple): tuple containing loss function arguments. Circuit must be first argument.
        loss (function): loss function to train on
        save (bool): Flag to set logging on
    """

    def __init__(self, initial_parameters, args=(), loss=None, save=False, **kwargs):
        self._circuit = args[0]
        super().__init__(initial_parameters, args, loss, save)
        self.options = {}
        self.set_options(kwargs)
        self.filename = f"results/cma_{self.name}.txt"
        if self.save:
            self.file = open(self.filename, "w")

    def fit(self):
        """Performs the optimizations.

        Returns:
            (float): best loss value
            (np.ndarray): best parameter values
            (cma.evolution_strategy.CMAEvolutionStrategy): full CMA Evolution Strategy object
        """

        import cma

        r = cma.fmin2(
            self.fun,
            self.params,
            sigma0=1.7,
            **self.options,
        )

        return r[1].result.fbest, r[1].result.xbest, r, self.iteration


class Newtonian(Optimizer):
    def __init__(self, initial_parameters, args=(), loss=None, save=False, **kwargs):
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
        super().__init__(initial_parameters, args, loss, save)
        self._circuit = args[0]

        self.options = {
            "method": "Powell",
            "jac": None,
            "hess": None,
            "hessp": None,
            "bounds": None,
            "constraints": (),
            "tol": None,
            "callback": None,
            "options": {"disp": True, "maxfevals": 100},
            "processes": None,
            "backend": None,
        }
        self.set_options(kwargs)

        self.filename = f"results/newtonian_{self.name}.txt"
        if self.save:
            self.file = open(self.filename, "w")

    def fit(self):
        """Performs the optimizations.

        Returns:
            (float): best loss value
            (np.ndarray): best parameter values
            (scipy.optimize.OptimizeResult): full scipy OptimizeResult object
            (int): iteration number
        """

        if self.options["method"] == "parallel_L-BFGS-B":  # pragma: no cover
            o = ParallelBFGS(
                self.fun,
                processes=self.options["processes"],
                bounds=self.options["bounds"],
                callback=self.options["callback"],
                options=self.options["options"],
            )
            m = o.run(self.params)
        else:
            from scipy.optimize import minimize

            m = minimize(
                self.fun,
                self.params,
                method=self.options["method"],
                jac=self.options["jac"],
                hess=self.options["hess"],
                hessp=self.options["hessp"],
                bounds=self.options["bounds"],
                constraints=self.options["constraints"],
                tol=self.options["tol"],
                callback=self.options["callback"],
                options=self.options["options"],
            )
        return m.fun, m.x, m, self.iteration


class ParallelBFGS(Optimizer):  # pragma: no cover
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

    def __init__(self, initial_parameters, loss, args=(), save=False, **kwargs):
        super().__init__(initial_parameters, args, loss, save)
        self.function_value = None
        self.jacobian_value = None
        self._circuit = args[0]

        self.options = {
            "xval": None,
            "precision": self.np.finfo("float64").eps,
            "bounds": None,
            "callback": None,
            "options": None,
            "processes": None,
            "save": False,
        }

        self.set_options(kwargs)

        self.filename = f"results/bfgs_{self.name}.txt"
        if self.save:
            self.file = open(self.filename, "w")

    def fit(self):
        """Performs the optimizations.

        Returns:
            (float): best loss value
            (np.ndarray): best parameter values
            (scipy.optimize.OptimizeResult): full scipy OptimizeResult object
            (int): iteration number
        """

        from scipy.optimize import minimize

        out = minimize(
            fun=self.fun,
            x0=self.params,
            jac=self.jac,
            method="L-BFGS-B",
            bounds=self.options["bounds"],
            callback=self.options["callback"],
            options=self.options["options"],
        )
        out.hess_inv = out.hess_inv * self.np.identity(len(self.params))
        return out.fun, out.x, out, self.iteration

    @staticmethod
    def _eval_approx(eps_at, fun, x, eps):
        """Approximate evaluation

        Args:
            eps_at (int): parameter index where approximation occurs
            fun (function): loss function
            x (np.ndarray): circuit parameters
            eps (float): approximation delta
        Returns:
            (float): approximated loss value
        """
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
        """Handles function evaluation

        Args:
            x (np.ndarray): circuit parameters
            eps (float): approximation delta
        Returns
            (float): loss value
        """
        if not (
            self.options["xval"] is not None
            and all(abs(self.options["xval"] - x) <= self.options["precision"] * 2)
        ):
            eps_at = range(len(x) + 1)
            self.options["xval"] = x.copy()

            def operation(epsi):
                return self._eval_approx(
                    epsi, lambda y: self.loss_function(y, *self.args), x, eps
                )

            from joblib import Parallel, delayed

            ret = Parallel(self.options["processes"], prefer="threads")(
                delayed(operation)(epsi) for epsi in eps_at
            )
            self.function_value = ret[0]
            self.jacobian_value = (ret[1 : (len(x) + 1)] - self.function_value) / eps

    def fun(self, x):
        """Saves and returns loss function value

        Args:
            x (np.ndarray): circuit parameters
        Returns
            (float): loss value
        """

        self.evaluate(x)
        if self.save:
            self.file.write(
                f"Iteration {self.iteration} | loss: {self.function_value}\n"
            )
            self.iteration += 1
        return self.function_value

    def jac(self, x):
        """Evaluates the Jacobian

        Args:
            x (np.ndarray): circuit parameters
        Returns
            (float): jacobian value
        """

        self.evaluate(x)
        return self.jacobian_value


class BasinHopping(Optimizer):
    """Global optimizer based on `scipy.optimize.basinhopping <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html>`_.

    Example:
    .. code-block:: python

        from qibo import Circuit
        from qibo import gates

        circuit = Circuit(nqubits=1)
        c.add(gates.RY(q=0, theta=Parameter(lambda x, th1, th2: th1 * x + th2, [0.1, 0.1], feature=[0.1])))
        c.add(gates.RZ(q=0, theta=Parameter(lambda x, th1, th2: th1 * x + th2, [0.1, 0.1], feature=[0.1])))
        c.add(gates.M(q=0))

        parameters = [0.2, 0.4, 0.2, 0.9]

        optimizer = BasinHopping(
            initial_parameters=parameters,
            args=(circuit, hamiltonian),
            loss=cma_loss,
        )

        fbest = optimizer.fit()

    Args:
        intial_parameters (np.ndarray): array with initial values for gate parameters
        args (tuple): tuple containing loss function arguments. Circuit must be first argument.
        loss (function): loss function to train on
        save (bool): Flag to set logging on
    """

    def __init__(self, initial_parameters, args=(), loss=None, save=False, **kwargs):
        super().__init__(initial_parameters, args, loss, save)
        self.args = args
        self.options = kwargs
        self._circuit = args[0]
        self.filename = f"results/basin_{self.name}.txt"
        if self.save:
            self.file = open(self.filename, "w")

    def fit(self):
        """Performs the optimizations.

        Returns:
            (float): best loss value
            (np.ndarray): best parameter values
            (scipy.optimize.OptimizeResult): full scipy OptimizeResult object
            (int): iteration number
        """
        from scipy.optimize import basinhopping

        if "options" in self.options:
            options = self.options.pop("options")

        r = basinhopping(
            self.fun,
            self.params,
            niter=1,
            minimizer_kwargs={"options": {"maxfev": 100}},
        )

        return r.fun, r.x, r, self.iteration


def get_error(optimizer, xtrain, name_prependix, name_appendix):
    ypredictions = optimizer.predict(xtrain, N=10)

    np.save(
        f"predictions/{name_prependix}_{optimizer.name}_{name_appendix}.dat",
        ypredictions,
    )
    ypred = np.mean(ypredictions, axis=ypredictions.ndim - 1)
    ysigma = np.std(ypredictions, axis=ypredictions.ndim - 1)

    return ypred, ysigma


scaler = lambda x: x  # (1 - x + 0.1) / (1 + x + 0.1)
import matplotlib.pyplot as plt


def plot(
    yprediction,
    xtrain,
    ytrain,
    epoch,
    loss,
    ysigma=None,
    name=None,
    name_prependix=None,
    name_appendix=None,
    params=None,
    xscale="log",
):
    # new predictions
    if yprediction.ndim == 2:
        cols = yprediction.shape[1]
    else:
        cols = 1
    # new plot
    fig, ax = plt.subplots(nrows=1, ncols=cols, figsize=(8, 6))
    fig.suptitle(f"Epoch {epoch+1}, J={loss:.4}")

    for col in range(cols):
        if cols > 1:
            if col == 0:
                ax[col].set_ylabel("y")
            else:
                ax[col].set_ylabel("")
            ax[col].set_xlabel("x")

            ax[col].set_xscale(xscale)
            train = ytrain[:, col]
            pred = yprediction[:, col]
            pred = scaler(pred)
            if ysigma is not None:
                sigma = ysigma[:, col]
                ax[col].fill_between(
                    xtrain, pred + sigma, pred - sigma, alpha=0.3, color="royalblue"
                )

            ax[col].plot(xtrain, train, label="Classical PDF", color="black")
            ax[col].plot(
                xtrain,
                pred,
                label="Quantum PDF model",
                # zorder=10,
                # marker=".",
                # markersize=12,
                alpha=0.7,
                color="royalblue",
                lw=2,
            )

            ax[col].legend()

        else:
            ax.set_xscale(xscale)

            yprediction = scaler(yprediction)

            if ysigma is not None:
                ax.fill_between(
                    xtrain,
                    yprediction + ysigma,
                    yprediction - ysigma,
                    alpha=0.3,
                    color="royalblue",
                )
            ax.plot(xtrain, ytrain, label="Classical PDF", color="black")
            ax.plot(
                xtrain,
                yprediction,
                label="Quantum PDF model",
                # zorder=10,
                # marker=".",
                # markersize=12,
                color="royalblue",
                lw=2,
                alpha=0.7,
            )
            ax.legend()

            plt.xscale(xscale)
            plt.xlabel("x")
            plt.ylabel("y")
    if name is not None:
        plt.savefig(
            f"results/{name_prependix}_{name}_{name_appendix}.png", bbox_inches="tight"
        )
        plt.show()
    else:
        plt.savefig("Plot.png", bbox_inches="tight")
    plt.close()


class BFGS(Optimizer):  # pragma: no cover
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
        jacobian,
        args=(),
        bounds=None,
        callback=None,
        options=None,
        processes=None,
    ):
        super().__init__(np.random.randn(12), loss=function, save=True)
        self.filename = f"results/Scipy_bfgs_{self.name}.txt"
        if self.save:
            self.file = open(self.filename, "w")
        self.function = function
        self.jacobian = jacobian
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
            method="BFGS",
            bounds=self.bounds,
            callback=self.callback,
            options={"gtol": 1e-7, "ftol": 1e-10, "xtol": 1e-10, "maxiter": 10000},
        )

        return out

    def fun(self, x):
        res = self.loss_function(x, *self.args)
        return res

    def jac(self, x):
        res = self.jacobian(x, *self.args)
        return res.T
