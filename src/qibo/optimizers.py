from scipy.optimize import basinhopping
#from qibo.noise import NoiseModel
from qibo.config import log, raise_error
from qibo.models import Circuit
from qibo import backends
from qibo.symbols import Symbol
from qibo.hamiltonians import SymbolicHamiltonian
from qibo import gates
import numpy as np
import random
import tensorflow as tf

class Node:
    """Parent class to create gate nodes"""

    def __init__(self, gate, trainable_params, gate_params):
        self.gate = gate
        self.trainable_params = trainable_params # index of optimisable parameters
        self.gate_params = gate_params # gate parameters
        self.prev = None
        self.next = None

class ConvergeNode(Node):
    """Node for two-qubit gates"""
    def __init__(self, gate, trainable_params, gate_params):
        super().__init__(gate, trainable_params, gate_params)
        self.prev_target = None
        self.next_target = None
        self.waiting = None

class Graph:
    """Creates a graph representation of a circuit"""

    def __init__(self, nqubits, gates, trainable_params, gate_params):
        self.gates = gates
        self.trainable_params = trainable_params
        self.gate_params = gate_params
        self.nqubits = nqubits

    def build_graph(self):
        """
        Builds graph based on the circuit gates and associates parameters to each gate.
        """

        # setup
        start = [None]*self.nqubits
        ends = [None]*self.nqubits
        depth = [0]*self.nqubits
        nodes = list()

        count = 0
        # run through each gate in circuit queue
        for i, gate in enumerate(self.gates):

            n = len(gate.init_args) - 1
            
            # store parameters for ParametrizedGate
            if isinstance(gate, gates.ParametrizedGate):
                trainp = self.trainable_params[count]
                gatep = self.gate_params[count]
                count += 1
            else:
                trainp = None
                gatep = None

            # two-qubit gates
            if n == 1:

                node = ConvergeNode(gate, trainp, gatep)
                control = gate.init_args[0]
                target = gate.init_args[1]

                # control qubit
                # start of graph
                if start[control] is None:
                    start[control] = i
                    ends[control] = i
                # link to existing graph node
                else:
                    nodes[ends[control]].next = i
                    node.prev = ends[control]
                    ends[control] = i

                # target qubit
                # start of graph
                if start[target] is None:
                    start[target] = i
                    ends[target] = i
                # link to existing graph node
                else:
                    nodes[ends[target]].next = i
                    node.prev = ends[target]
                    ends[target] = i  

                depth[control] += 1
                depth[target] += 1                

            # one-qubit gate
            else:

                node = Node(gate, trainp, gatep)
                qubit = gate.init_args[0]

                # start of graph
                if start[qubit] is None:
                    start[qubit] = i
                    ends[qubit] = i
                # link to existing graph node
                else:
                    nodes[ends[qubit]].next = i
                    node.prev = ends[qubit]
                    ends[qubit] = i

                depth[qubit] += 1

            # add node to list
            nodes.append(node)

        self.start = start
        self.end = ends
        self.nodes = nodes
        self.depth = max(depth)

    def _determine_basis(self, gate):
        gname = gate.name

        if gname == "rx":
            return gates.X
        elif gname == "ry":
            return gates.Y
        else:
            return gates.Z

    def run_layer(self, layer):
        """Runs through one layer of the circuit parameters
        Args:
            layer: int, layer number N
        Returns: 
            c: circuit up to nth layer
            trainable_qubits: qubits on which we find trainable gates
            affected_params: which trainable parameters are linked to the trainable gates
        """

        # empty circuit
        c = Circuit(self.nqubits, density_matrix=True)

        current = self.start[:]

        trainable_qubits = []
        affected_params = []

        # run through layer up to N
        for iter in range(layer+1):

            # run through all qubits
            for q in range(self.nqubits):

                node = self.nodes[current[q]]

                # wait for both qubits to reach two-qubit node
                if isinstance(node, ConvergeNode):

                    # first arrived
                    if node.waiting is None:
                        node.waiting = q
                    # second arrived
                    elif node.waiting != q:
                        c.add(node.gate)
                        control = node.gate.init_args[0]
                        target = node.gate.init_args[1]
                        current[control] = node.next
                        current[target] = node.next_target
                        node.waiting = None
                
                # replace last layer by M gate
                elif iter == layer and isinstance(node.gate, gates.ParametrizedGate):     

                    c.add(gates.M(q, basis=self._determine_basis(node.gate)))
                    trainable_qubits.append(q)
                    affected_params.append(node.trainable_params)
    
                # simple one-qubit node
                else:

                    c.add(node.gate)
                    if node.next:
                        current[q] = node.next

        return c, trainable_qubits, affected_params


class Optimizer:
    """Parent optimizer"""
    
    def __init__(self, loss, circuit, args=(), method="sgd", options=None):
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

        if not isinstance(args, np.ndarray):
            raise("First argument to loss function must be a numpy array with circuit parameters")
        
        self.params = args
        self.nparams = len(self.params)
        self.nparams_main = len(self.params)

        if not isinstance(circuit, Circuit):
            raise("Circuit is not of correct object type")
        
        self._circuit_main = circuit
        self._circuit = self._circuit_main
        self.nqubits = self._circuit_main.nqubits
        self.Parameters = None
        self.results = tf.Variable(initial_value=np.zeros(2**self._circuit.nqubits), dtype=tf.float64)



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
        
    def run_circuit(self, parameters, nshots=1024):
        """
        User-facing function which runs the circuit with given parameters and returns the result
        Args:
            parameters: trainable parameters of type Parameter
            nshots: number of shots to average circuit expectation values
        Return:
            results
        """
        param_values = []

        for Param in parameters:
            param_values.append(Param._gatep)

        if self.natgrad:
            self.Parameters = parameters

        self._circuit.set_parameters(param_values)

        self.results.assign_add(self._circuit(nshots=nshots).probabilities(qubits=[0])) # CHANGE

        return self.results
        
    def one_prediction(self, feature):
        """
        Runs the user loss function and returns the result of the measurements for gradient calculation
        Args:
            feature: feature value given to loss function
        Return:
            float, circuit expectation result
        """
        # label value is insignificant
        _ = self.loss(self, self.params, feature, 0)
        return self.results[0]
        
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
    
    def _create_hamiltonian(self, qubit, nqubit):
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

            # trainable and gate parameters
            gate_params = self._circuit.associate_gates_with_parameters()
            trainable_params = []
            for Param in self.Parameters:
                trainable_params.append(Param._trainablep)

            # build graph from circuit gates
            graph = Graph(self.nqubits, self._circuit.queue, trainable_params, gate_params)
            graph.build_graph()

            # run through layers
            for i in range(graph.depth):
                c, qubits, affected_param = graph.run_layer(i)
                if len(qubits) == 0:
                    continue

                state = c().state()
                print(c.draw())
                print("state", i, state)

                # run through parametrized gate
                for qubit, params in zip(qubits, affected_param):
                    hamiltonian = self._create_hamiltonian(qubit, self.nqubits)
                    print(hamiltonian.matrix)

                    result = hamiltonian.expectation(state)

                    for p in params:
                        # update Fubini-Study matrix
                        fubini[p, p] = result - result**2
                print("result: ", result - result**2)
                #exit(0)

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
            self.results = tf.Variable(initial_value=np.zeros(2**self._circuit.nqubits), dtype=tf.float64)
            with tf.GradientTape() as tape:
                local_loss = self.loss(self, self.params, feat, label)

            loss += local_loss
            grads = tape.gradient(local_loss, self.results)

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


class Parameter:

    def __init__(self, value, trainablep):
        self._gatep = value
        self._trainablep = trainablep


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
