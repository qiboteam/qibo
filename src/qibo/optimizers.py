import random
import time
from datetime import datetime

import numpy as np
from scipy.optimize import basinhopping

from qibo import backends
from qibo.config import log, raise_error
from qibo.derivative import calculate_gradients, create_hamiltonian, execute_circuit
from qibo.models import Circuit
from qibo.models.error_mitigation import CDR, calibration_matrix


class Optimizer:
    """Parent optimizer"""

    def __init__(self, initial_parameters, args=(), loss=None):
        self.loss_function = loss
        self.args = args
        self.initial_parameters = initial_parameters
        self.backend = backends.GlobalBackend()
        self.ftime = None
        self.etime = None
        self.name = f'Run_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

        if not isinstance(initial_parameters, list) and not isinstance(
            initial_parameters, np.ndarray
        ):
            raise ("Parameters must be a list of Parameter objects or a numpy array")

    def set_options(self, updates):
        """Updates options dictionary"""
        self.options.update(updates)


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
        self, circuit, parameters, hamiltonian=None, args=(), loss=None, **kwargs
    ):
        super().__init__(parameters, args, loss=loss)

        # circuit
        if not isinstance(circuit, Circuit):
            raise ("Circuit is not of correct object type")

        self._circuit = circuit
        self.nqubits = self._circuit.nqubits

        # parameters
        self.paramInputs = parameters
        self.params = self._get_params(trainable=True)
        self.nparams = len(self.params)

        # hamiltonian
        if not hamiltonian:
            self.hamiltonian = [create_hamiltonian(0, self.nqubits, self.backend)]
        else:
            self.hamiltonian = hamiltonian

        # error mitigation
        self.cdr_params = None

        # options
        self.options = {
            "epochs": 1000,
            "learning_rate": 0.045,
            "batches": 1,
            "J_threshold": 1e-3,
            "shift_rule": "psr",
            "nshots": 1024,
            "natgrad": False,
            "mitigation": False,
            "noise_model": None,
            "adam": True,
            "save": False,
            "filename": f"results/{self.name}.txt",
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
                    params += Param._variational_parameters
                else:
                    trainablep = self.params[count : count + Param.nparams]
                    count += Param.nparams
                    # update trainable params and retrieve gate param
                    params.append(Param.get_params(trainablep, feature=feature))

            return params

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

    def run_circuit(self, feature):
        """Backend function which runs the circuit for one feature
        Args:
            feature: single input value to the system
        Return:
            results: expectation value"""

        # set parameters
        parameters = self._get_params(trainable=False, feature=feature)
        self._circuit.set_parameters(parameters)

        # run circuit
        if isinstance(self.hamiltonian, list):
            exp_v = np.empty(len(self.hamiltonian))
            for i, hamiltonian in enumerate(self.hamiltonian):
                exp_v[i] = execute_circuit(
                    self.backend,
                    self._circuit,
                    hamiltonian,
                    self.options["nshots"],
                    initial_state=None,
                    cdr_params=self.cdr_params,
                )

        else:
            exp_v = execute_circuit(
                self.backend,
                self._circuit,
                self.hamiltonian,
                self.options["nshots"],
                initial_state=None,
                cdr_params=self.cdr_params,
            )

        return exp_v

    def predict(self, feature):
        """
         User-facing function which runs the circuit for given input features returns the result
         Args:
             feature: input values which are run through the circuit
        Return:
             results
        """

        if isinstance(feature, np.ndarray):
            results = np.zeros((len(feature), self.nlabels))
            for i, feat in enumerate(feature):
                results[i, :] = self.run_circuit(feat)
            return results
        else:
            return self.run_circuit(feature)

    def error_mitigation(self, circuit):
        """Fit CDR regression model to noisy states"""

        calibration = calibration_matrix(
            nqubits=self.nqubits, backend=self.backend, nshots=self.options["nshots"]
        )

        montecarlo = random.randrange(len(self.hamiltonian))

        _, _, optimal_params, _ = CDR(
            circuit=circuit,
            observable=self.hamiltonian[montecarlo],
            backend=self.backend,
            nshots=self.options["nshots"],
            noise_model=self.options["noise_model"],
            full_output=True,
            readout={"calibration_matrix": calibration},
        )

        return optimal_params

    def dloss(self, features, labels):
        """
        This function calculates the loss function's gradients with respect to self.params
        Args:
            features: np.matrix containig the n_sample-long vector of states
            labels: np.array of the labels related to features
        Returns: np.array of length self.nparams containing the loss function's gradients
        """
        circ_grads = np.zeros(self.nparams)
        results = np.zeros((self.nsample, self.nlabels))
        loss = 0

        # calculate CDR parameters anew at each epoch
        if self.options["mitigation"]:
            parameters = self._get_params(trainable=False, feature=1.0)
            self._circuit.set_parameters(parameters)
            self.cdr_params = self.error_mitigation(
                self._circuit.to_clifford(),
                self.nqubits,
                self.hamiltonian,
                self.backend,
                self.options["noise_model"],
                self.options["nshots"],
            )

        # iterate through all data points
        for i, feat in enumerate(features):
            # duration measurement
            if self.options["save"]:
                ftime = time.time()
                self.file.write(f"Feature {feat}, duration {ftime-self.ftime}\n")
                self.ftime = ftime

            # predict current output
            results[i] = self.predict(feat)

            obs_gradients = np.zeros((self.nlabels, self.nparams))
            for h, ham in enumerate(self.hamiltonian):
                obs_gradients[h] = calculate_gradients(
                    self._circuit,
                    self.options["shift_rule"],
                    self.nparams,
                    self.paramInputs,
                    self.cdr_params,
                    ham,
                    self.options["nshots"],
                )  # d<B> N params, N label gradients

            loss_func_grad = self.calculate_loss_func_grad(results, labels, i)

            circ_grads += np.dot(loss_func_grad.T, obs_gradients)

        # gradient average
        loss = self.loss_function(results, labels, self.args)
        loss_gradients = circ_grads / (self.nsample)

        return loss_gradients, loss / len(features)

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

        if self.options["save"]:
            self.file.write(
                f"Grads (absolute value: {np.linalg.norm(grads)}): {grads.tolist()}\nParams {self.params}\n"
            )

        if self.options["adam"]:
            for i in range(self.nparams):
                m[i] = beta_1 * m[i] + (1 - beta_1) * grads[i]
                v[i] = beta_2 * v[i] + (1 - beta_2) * grads[i] * grads[i]
                mhat = m[i] / (1.0 - beta_1 ** (iteration + 1))
                vhat = v[i] / (1.0 - beta_2 ** (iteration + 1))
                self.params[i] -= learning_rate * mhat / (np.sqrt(vhat) + epsilon)
            return m, v, loss

        # standard gradient descent
        else:
            for i in range(self.nparams):
                self.params[i] -= learning_rate * grads[i]
            return 0, 0, loss

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

        if self.options["save"]:
            self.file = open(self.options["filename"], "w")
            self.file.write(
                f"Epochs: {self.options['epochs']}, \
                            learning rate: {self.options['learning_rate']}, \
                            nshots: {self.options['nshots']}, \
                            natgrad: {self.options['natgrad']}, \
                            mitigation: {self.options['mitigation']}, \
                            noise_model: {self.options['noise_model']}, \
                            adam: {self.options['adam']}\n"
            )
            self.ftime = time.time()
            self.etime = time.time()

        for epoch in range(options["epochs"]):
            if epoch != 0 and losses[-1] < options["J_threshold"]:
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

                if self.options["save"]:
                    etime = time.time()
                    self.file.write(
                        f"Iteration {iteration}, epoch {epoch + 1} | loss: {this_loss} | duration: {etime-self.etime}\n"
                    )
                    self.etime = etime

                # in case one wants to plot J as a function of the iterations
                losses.append(this_loss)

        if self.options["save"]:
            self.file.write(f"Params {self.params}\n")
            self.file.close()

        return losses

    def fit(self, X, y):
        """Performs the optimizations and returns f_best, x_best."""

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

        return self.sgd(self.options)


class CMAES(Optimizer):
    def __init__(self, initial_parameters, args=(), loss=None, **kwargs):
        super().__init__(initial_parameters, args, loss, **kwargs)

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

        r = cma.fmin2(self.loss_function, self.initial_parameters, 1.7, args=self.args)
        return r[1].result.fbest, r[1].result.xbest, r


class Newtonian(Optimizer):
    def __init__(self, initial_parameters, args=(), loss=None, **kwargs):
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
        super().__init__(initial_parameters, args, loss, **kwargs)
        self.options = {
            "method": "Powell",
            "jac": None,
            "hess": None,
            "hessp": None,
            "bounds": None,
            "constraints": (),
            "tol": None,
            "callback": None,
            "options": None,
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
                self.loss_function,
                args=self.args,
                processes=self.options["processes"],
                bounds=self.options["bounds"],
                callback=self.options["callback"],
                options=self.options["options"],
            )
            m = o.run(self.initial_parameters)
        else:
            from scipy.optimize import minimize

            m = minimize(
                self.loss_function,
                self.initial_parameters,
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

    def __init__(self, initial_parameters, loss, args=(), **kwargs):
        super().__init__(initial_parameters, args, loss=loss)
        self.function_value = None
        self.jacobian_value = None

        self.options = {
            "xval": None,
            "precision": self.np.finfo("float64").eps,
            "bounds": None,
            "callback": None,
            "options": None,
            "processes": None,
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
            x0=self.initial_parameters,
            jac=self.jac,
            method="L-BFGS-B",
            bounds=self.options["bounds"],
            callback=self.options["callback"],
            options=self.options["options"],
        )
        out.hess_inv = out.hess_inv * self.np.identity(len(self.initial_parameters))
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
        return self.function_value

    def jac(self, x):
        self.evaluate(x)
        return self.jacobian_value
