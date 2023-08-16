import random
import time
from datetime import datetime

import numpy as np

from qibo import backends
from qibo.config import log, raise_error
from qibo.derivative import (
    calculate_circuit_gradients,
    create_hamiltonian,
    error_mitigation,
    execute_circuit,
)
from qibo.gates import gates
from qibo.models import Circuit


class VariationalCircuit(Circuit):
    """
    Circuit architecture used in Quantum Machine Learning and Quantum Optimisation
    procedures
    """

    def optimize(
        self,
        X,
        y,
        initial_parameters,
        loss,
        args=(),
        method="sgd",
        hamiltonian=None,
        **kwargs,
    ):
        if method == "sgd":
            optimizer = SGD(self, initial_parameters, hamiltonian, args, loss, **kwargs)

        return optimizer.fit(X, y)


class Optimizer:
    """Parent optimizer"""

    def __init__(self, initial_parameters, args=(), loss=None, save=False):
        # saving to class objects
        self.loss_function = loss
        self.args = args
        self.params = initial_parameters
        self.backend = backends.GlobalBackend()

        # logging
        self.simulation_start = time.time()
        self.ftime = time.time()
        self.etime = None
        self.name = f'Run_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        self.iteration = 0
        self.filename = f"results/{self.name}.txt"
        self.save = save
        if save:
            self.file = open(self.filename, "w")

        self.initparams = None

        if not isinstance(initial_parameters, np.ndarray) and not isinstance(
            initial_parameters, list
        ):
            raise TypeError(
                "Parameters must be a list of Parameter objects or a numpy array"
            )

    def set_options(self, updates):
        """Updates options dictionary"""
        self.options.update(updates)

    def set_params(self):
        self.initparams = self._get_initparams()

    def _get_initparams(self):
        """Retrieve parameter values or objects directly from gates"""

        params = []
        for gate in self._circuit.queue:
            if isinstance(gate, gates._Rn_):
                params.append(gate.initparams)

        if isinstance(params[0], (float, int)):
            params = self.params

        return params

    def _get_gate_params(self, feature=None):
        """Retrieve gate parameters based on initial parameter values given to gates
        Args:
            feature (int or np.ndarray): input feature if embedded in Parameter lambda function

        Returns:
            (list) gate parameters
        """

        # for array
        if isinstance(self.initparams, np.ndarray):
            return self.initparams

        # for Parameter objects
        else:
            params = []
            count = 0
            for Param in self.initparams:
                trainablep = self.params[count : count + Param.nparams]
                count += Param.nparams
                # update trainable params and retrieve gate param
                params.append(Param.get_params(trainablep, feature=feature))

            return params

    def fun(self, x):
        """Wrapper function to save and preprocess gate parameters"""

        if isinstance(self.initparams[0], (float, int)):
            val = self.loss_function(x, *self.args)
        else:
            val = self.loss_function(x, self.initparams, *self.args)

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
        c.add(gates.RY(q=0, theta=0))
        c.add(gates.RZ(q=0, theta=0))
        c.add(gates.M(q=0))

        parameters = []
        for _ in range(2):
            parameters.append(Parameter(lambda x, th1, th2: th1 * x + th2, [0.1, 0.1], featurep=True))

        optimizer = SGD(circuit=circuit, parameters=parameters)
        X = np.array([0.1, 0.2, 0.3])
        y = np.array([0.2, 0.5, 0.7])
        losses = optimizer.fit(X, y)

    Args:
        circuit (Circuit): the circuit whose parameters will be optimised
        parameters (np.ndarray or list of Parameter objects): initial gate parameters
        hamiltonian (SymbolicHamiltonian): hamiltonian applied to final circuit state
        args (list): additional loss function arguments
        loss (lambda): loss function applied between
    """

    def __init__(
        self,
        circuit,
        parameters,
        hamiltonian=None,
        args=(),
        loss=None,
        save=False,
        **kwargs,
    ):
        super().__init__(parameters, args, loss=loss, save=save)

        # circuit
        if not isinstance(circuit, Circuit):
            raise ("Circuit is not of correct object type")

        self._circuit = circuit
        self.nqubits = self._circuit.nqubits

        # parameters
        self.initparams = self._get_initparams()
        self.params = parameters
        self.nparams = len(self.params)

        # hamiltonian
        if not hamiltonian:
            self.hamiltonian = [create_hamiltonian(0, self.nqubits, self.backend)]
        elif not isinstance(hamiltonian, list):
            self.hamiltonian = [hamiltonian]
        else:
            self.hamiltonian = hamiltonian

        # error mitigation
        self.cdr_params = None

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
            "mitigation": False,
            "noise_model": None,
            "adam": True,
            "deterministic": False,
            "beta_1": 0.85,
            "beta_2": 0.99,
        }
        self.set_options(kwargs)

        # name
        self.name_appendix = ""
        if self.options["adam"]:
            self.name_appendix += "adam"
        if self.options["natgrad"]:
            self.name_appendix += "natgrad"

        # logging
        self.param_history = np.zeros((self.options["epochs"], self.nparams))
        self.filename = f"results/{self.name}_{self.name_appendix}.txt"
        if save:
            self.file = open(self.filename, "w")

        # natural gradient graph initialisation
        if self.options["natgrad"]:
            print("ok")
            """
            self.NGgraph = build_graph(
                self._circuit, self.nparams, self.nqubits, self.initparams
            )
            """

    def calculate_loss_func_grad(self, results, labels, idx, delta=1e-6):
        """
        Calculates loss function derivative with respect to parameter idx
        Args:
            result: predicted values
            labels: true values
            idx: parameter number with respect to which we calculate the gradient
            delta: size of the finite difference perturbation
        """

        grads = np.empty(self.nlabels)
        for lab in range(self.nlabels):
            shifted = results[idx : idx + 1, :]
            shifted[0][lab] += delta
            forward = self.loss_function(
                np.copy(shifted), labels[idx : idx + 1, :], self.args
            )
            shifted[0][lab] -= 2 * delta
            backward = self.loss_function(
                np.copy(shifted), labels[idx : idx + 1, :], self.args
            )
            grads[lab] = (forward - backward) / (2 * delta)
        return grads

    def run_circuit(self, feature, N=1):
        """Backend function which runs the circuit for one feature N times
        Args:
            feature: single input value to the system
        Return:
            results: expectation value"""

        # set parameters
        parameters = self._get_gate_params(feature=feature)
        self._circuit.set_parameters(parameters)

        # run circuit
        exp_v = np.zeros((len(self.hamiltonian), N))
        for i, hamiltonian in enumerate(self.hamiltonian):
            for n in range(N):
                exp_v[i, n] = execute_circuit(
                    self.backend,
                    self._circuit,
                    hamiltonian,
                    self.options["nshots"],
                    initial_state=None,
                    cdr_params=self.cdr_params,
                    deterministic=self.options["deterministic"],
                )

        return exp_v

    def predict(self, feature, N=1):
        """
         User-facing function which runs the circuit for given input features and returns the result.
         Args:
             feature: input values which are run through the circuit
        Return:
             results
        """

        if isinstance(feature, np.ndarray):
            results = np.zeros((len(feature), self.nlabels, N))
            for i, feat in enumerate(feature):
                results[i] = self.run_circuit(feat, N)
            return np.squeeze(results)
        else:
            return np.squeeze(self.run_circuit(feature))

    def dloss(self, features, labels):
        """
        This function calculates the loss function's gradients with respect to self.params,
        as well as the loss per feature.
        Args:
            features: np.matrix containig the n_sample-long vector of states
            labels: np.array of the labels related to features
        Returns:
            np.array of length self.nparams containing the loss function's gradients
            float of mean loss per feature
        """
        circ_grads = np.zeros(self.nparams)
        results = np.zeros((self.nsample, self.nlabels))
        loss = 0

        # natural gradient setup
        if self.options["natgrad"]:
            fubini = np.zeros((self.nparams, self.nparams))
            scount = int(0.2 * self.nsample)
            sample = random.sample([i for i in range(self.nsample)], scount)

        # calculate CDR parameters anew at each epoch
        if self.options["mitigation"]:
            parameters = self._get_gate_params(feature=1.0)
            self._circuit.set_parameters(parameters)
            self.cdr_params = error_mitigation(
                self._circuit.to_clifford(),
                self.nqubits,
                self.hamiltonian,
                self.backend,
                self.options["noise_model"],
                self.options["nshots"],
            )

        # iterate through all data points
        for i, feat in enumerate(features):
            # predict current output
            results[i] = self.predict(feat)

            obs_gradients = np.zeros((self.nlabels, self.nparams))
            for h, ham in enumerate(self.hamiltonian):
                obs_gradients[h] = calculate_circuit_gradients(
                    self._circuit,
                    ham,
                    self.initparams,
                    self.nparams,
                    self.options["shift_rule"],
                    self.cdr_params,
                    self.options["nshots"],
                    self.options["deterministic"],
                    var_gates=self.options["var_gates"],
                )  # d<B> N params, N label gradients

            loss_func_grad = self.calculate_loss_func_grad(results, labels, i)

            circ_grads += np.dot(loss_func_grad.T, obs_gradients)

            if self.options["natgrad"] and i in sample:
                print("ok")
                """
                self.NGgraph.update_parameters(self._circuit.get_parameters())
                fubini += generate_fubini(
                    self.NGgraph,
                    self.nparams,
                    self.nqubits,
                    self.initparams,
                    noise_model=self.options["noise_model"],
                    deterministic=self.options["deterministic"],
                )  # separate pull request
                """

        # gradient average
        loss = self.loss_function(results, labels, self.args) / self.nsample
        loss_gradients = circ_grads / self.nsample

        # Fubini-Study Metric renormalisation
        if self.options["natgrad"]:
            fubini /= scount
            loss_gradients = np.dot(np.linalg.inv(fubini), loss_gradients)

        # save data
        if self.save:
            self.file.write(
                f"Grads (absolute value: {np.linalg.norm(loss_gradients)}): {loss_gradients.tolist()}\nParams {self.params.tolist()}\nypred: {results.tolist()}\n"
            )

        plot(results, self.features, self.labels, self.epoch, loss)

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
        """
        Implementation of the gradient descent: during a run of this function parameters are updated.
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

        if self.options["adam"]:
            m = beta_1 * m + (1 - beta_1) * grads
            v = beta_2 * v + (1 - beta_2) * grads * grads
            mhat = m / (1.0 - beta_1 ** (iteration + 1))
            vhat = v / (1.0 - beta_2 ** (iteration + 1))
            self.params -= learning_rate * mhat / (np.sqrt(vhat) + epsilon)

            return m, v, loss

        # QADAM
        elif self.options["natgrad"]:
            beta_1 /= self.iteration + 1
            beta_2 /= self.iteration + 1
            m = beta_1 * m + (1 - beta_1) * grads
            v = beta_2 * v + (1 - beta_2) * grads * grads

            self.params -= learning_rate * m / (np.sqrt(v) + epsilon)

            return m, v, loss

        # vanilla gradient descent
        else:
            self.params -= learning_rate * grads
            return 0, 0, loss

    def sgd(self, options):
        """
        This function performs the full gradient descent's procedure
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
                f"beta_1: {self.options['beta_1']}\n"
                f"beta_2: {self.options['beta_2']}\n\n\n"
            )
            self.ftime = time.time()
            self.simulation_start = time.time()
            self.etime = time.time()

        for self.epoch in range(options["epochs"]):
            iteration = 0

            if self.epoch != 0 and losses[-1] < options["J_threshold"]:
                print(
                    "Desired sensibility is reached, here we stop: ",
                    iteration,
                    " iteration",
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

                # in case one wants to plot J as a function of the iterations
                losses.append(this_loss)
                self.param_history[self.epoch] = self.params

        if self.save:
            value = min(losses)
            idx = losses.index(value)
            self.parameters = self.param_history[idx]
            ypred, ysigma = get_error(self, self.features, self.name_appendix)

            plot(
                ypred,
                self.features,
                self.labels,
                idx,
                value,
                ysigma,
                self.name,
                self.name_appendix,
            )

            self.file.write(f"Params {self.params.tolist()}\n")
            self.file.write(f"Best J: {min(losses)}\n")
            self.cleanup()

        return losses

    def setup(self, X, y):
        if not isinstance(X, np.ndarray):
            raise ("X must be a numpy array")

        if not isinstance(y, np.ndarray):
            raise ("y must be a numpy array")

        self.features = X
        self.nsample = len(self.features)

        self.labels = y
        if y.ndim == 1:
            self.labels = self.labels.reshape(-1, 1)

        self.nlabels = self.labels.shape[1]

        if self.backend is None:
            from qibo.backends import GlobalBackend

            self.backend = GlobalBackend()

    def fit(self, X, y):
        """Performs the optimizations and returns f_best, x_best."""

        self.setup(X, y)

        return self.sgd(self.options)


class CMAES(Optimizer):
    def __init__(self, initial_parameters, args=(), loss=None, save=False, **kwargs):
        self._circuit = args[0]
        super().__init__(initial_parameters, args, loss, save)
        self.options = {}
        self.set_options(kwargs)
        self.set_params()

    def fit(self):
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

        r = cma.fmin2(
            self.fun,
            self.params,
            sigma0=1.7,
            **self.options,
        )

        return r[1].result.fbest, r[1].result.xbest, r


class Newtonian(Optimizer):
    def __init__(self, initial_parameters, args=(), loss=None, save=False, **kwargs):
        """
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
        self.options = {
            "method": "Powell",
            "jac": None,
            "hess": None,
            "hessp": None,
            "bounds": None,
            "constraints": (),
            "tol": None,
            "callback": None,
            "options": {"disp": True, "maxiter": 100},
            "processes": None,
            "backend": None,
        }
        self.set_options(kwargs)

    def fit(self):
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
                self.loss_function,
                self.params,
                args=self.args,
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
        return m.fun, m.x, m


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

    def fit(self):
        """Executes parallel L-BFGS-B minimization.

        Returns:
            scipy.minimize result object
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
        return out.fun, out.x, out

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
        self.evaluate(x)
        if self.save:
            self.file.write(
                f"Iteration {self.iteration} | loss: {self.function_value}\n"
            )
            self.iteration += 1
        return self.function_value

    def jac(self, x):
        self.evaluate(x)
        return self.jacobian_value


def get_error(optimizer, xtrain, name_appendix):
    ypredictions = optimizer.predict(xtrain, N=10)

    np.save(f"predictions/{optimizer.name}_{name_appendix}.dat", ypredictions)
    ypred = np.mean(ypredictions, axis=ypredictions.ndim - 1)
    ysigma = np.std(ypredictions, axis=ypredictions.ndim - 1)

    return ypred, ysigma


scaler = lambda x: x
import matplotlib.pyplot as plt


def plot(
    yprediction,
    xtrain,
    ytrain,
    epoch,
    loss,
    ysigma=None,
    name=None,
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
        plt.savefig(f"results/{name}_{name_appendix}.png", bbox_inches="tight")
        plt.show()
    else:
        plt.savefig("Plot.png", bbox_inches="tight")
    plt.close()


class BasinHopping(Optimizer):
    def __init__(self, initial_parameters, args=(), loss=None, save=False, **kwargs):
        super().__init__(initial_parameters, args, loss, save)
        self.args = args
        self.options = kwargs
        self._circuit = args[0]
        self.set_params()

    def fit(self):
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
        from scipy.optimize import basinhopping

        r = basinhopping(self.fun, self.params, **self.options)

        return r.fun
