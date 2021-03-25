import math
from qibo import get_backend
from qibo.config import raise_error, log
from qibo.core.circuit import Circuit as StateCircuit
from qibo.core.circuit import DensityMatrixCircuit
from qibo.evolution import StateEvolution, AdiabaticEvolution
from typing import Dict, Optional
import numpy as np
from qibo import gates


class Circuit(StateCircuit):
    """"""

    @classmethod
    def _constructor(cls, *args, **kwargs):
        if kwargs["density_matrix"]:
            if kwargs["accelerators"] is not None:
                raise_error(NotImplementedError,
                            "Distributed circuits are not implemented for "
                            "density matrices.")
            circuit_cls = DensityMatrixCircuit
            kwargs = {}
        elif kwargs["accelerators"] is None:
            circuit_cls = StateCircuit
            kwargs = {}
        else:
            try:
                from qibo.tensorflow.distcircuit import DistributedCircuit
            except ModuleNotFoundError: # pragma: no cover
                # CI installs all required libraries by default
                raise_error(ModuleNotFoundError,
                            "Cannot create distributed circuit because some "
                            "required libraries are missing.")
            circuit_cls = DistributedCircuit
            kwargs.pop("density_matrix")
        return circuit_cls, args, kwargs

    def __new__(cls, nqubits: int,
                accelerators: Optional[Dict[str, int]] = None,
                memory_device: str = "/CPU:0",
                density_matrix: bool = False):
        circuit_cls, args, kwargs = cls._constructor(
                  nqubits, accelerators=accelerators,
                  memory_device=memory_device,
                  density_matrix=density_matrix
                )
        return circuit_cls(*args, **kwargs)

    @classmethod
    def from_qasm(cls, qasm_code: str,
                  accelerators: Optional[Dict[str, int]] = None,
                  memory_device: str = "/CPU:0",
                  density_matrix: bool = False):
      circuit_cls, args, kwargs = cls._constructor(
                qasm_code, accelerators=accelerators,
                memory_device=memory_device,
                density_matrix=density_matrix
              )
      return circuit_cls.from_qasm(*args, **kwargs)


def QFT(nqubits: int, with_swaps: bool = True,
        accelerators: Optional[Dict[str, int]] = None,
        memory_device: str = "/CPU:0") -> Circuit:
    """Creates a circuit that implements the Quantum Fourier Transform.

    Args:
        nqubits (int): Number of qubits in the circuit.
        with_swaps (bool): Use SWAP gates at the end of the circuit so that the
            qubit order in the final state is the same as the initial state.
        accelerators (dict): Accelerator device dictionary in order to use a
            distributed circuit
            If ``None`` a simple (non-distributed) circuit will be used.
        memory_device (str): Device to use for memory in case a distributed circuit
            is used. Ignored for non-distributed circuits.

    Returns:
        A qibo.models.Circuit that implements the Quantum Fourier Transform.

    Example:
        ::

            import numpy as np
            from qibo.models import QFT
            nqubits = 6
            c = QFT(nqubits)
            # Random normalized initial state vector
            init_state = np.random.random(2 ** nqubits) + 1j * np.random.random(2 ** nqubits)
            init_state = init_state / np.sqrt((np.abs(init_state)**2).sum())
            # Execute the circuit
            final_state = c(init_state)
    """
    if accelerators is not None:
        if not with_swaps:
            raise_error(NotImplementedError, "Distributed QFT is only implemented "
                                             "with SWAPs.")
        return _DistributedQFT(nqubits, accelerators, memory_device)

    from qibo import gates

    circuit = Circuit(nqubits)
    for i1 in range(nqubits):
        circuit.add(gates.H(i1))
        for i2 in range(i1 + 1, nqubits):
            theta = math.pi / 2 ** (i2 - i1)
            circuit.add(gates.CU1(i2, i1, theta))

    if with_swaps:
        for i in range(nqubits // 2):
            circuit.add(gates.SWAP(i, nqubits - i - 1))

    return circuit


def _DistributedQFT(nqubits: int,
                    accelerators: Optional[Dict[str, int]] = None,
                    memory_device: str = "/CPU:0"):
    """QFT with the order of gates optimized for reduced multi-device communication."""
    from qibo import gates

    circuit = Circuit(nqubits, accelerators, memory_device)
    icrit = nqubits // 2 + nqubits % 2
    if accelerators is not None:
        circuit.global_qubits = range(circuit.nlocal, nqubits) # pylint: disable=E1101
        if icrit < circuit.nglobal: # pylint: disable=E1101
            raise_error(NotImplementedError, "Cannot implement QFT for {} qubits "
                                             "using {} global qubits."
                                             "".format(nqubits, circuit.nglobal)) # pylint: disable=E1101

    for i1 in range(nqubits):
        if i1 < icrit:
            i1eff = i1
        else:
            i1eff = nqubits - i1 - 1
            circuit.add(gates.SWAP(i1, i1eff))

        circuit.add(gates.H(i1eff))
        for i2 in range(i1 + 1, nqubits):
            theta = math.pi / 2 ** (i2 - i1)
            circuit.add(gates.CU1(i2, i1eff, theta))

    return circuit


class VQE(object):
    """This class implements the variational quantum eigensolver algorithm.

    Args:
        circuit (:class:`qibo.abstractions.circuit.AbstractCircuit`): Circuit that
            implements the variaional ansatz.
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): Hamiltonian object.

    Example:
        ::

            import numpy as np
            from qibo import gates, models, hamiltonians
            # create circuit ansatz for two qubits
            circuit = models.Circuit(2)
            circuit.add(gates.RY(0, theta=0))
            # create XXZ Hamiltonian for two qubits
            hamiltonian = hamiltonians.XXZ(2)
            # create VQE model for the circuit and Hamiltonian
            vqe = models.VQE(circuit, hamiltonian)
            # optimize using random initial variational parameters
            initial_parameters = np.random.uniform(0, 2, 1)
            vqe.minimize(initial_parameters)
    """
    from qibo import optimizers

    def __init__(self, circuit, hamiltonian):
        """Initialize circuit ansatz and hamiltonian."""
        self.circuit = circuit
        self.hamiltonian = hamiltonian

    def minimize(self, initial_state, method='Powell', jac=None, hess=None,
                 hessp=None, bounds=None, constraints=(), tol=None, callback=None,
                 options=None, compile=False, processes=None):
        """Search for parameters which minimizes the hamiltonian expectation.

        Args:
            initial_state (array): a initial guess for the parameters of the
                variational circuit.
            method (str): the desired minimization method.
                See :meth:`qibo.optimizers.optimize` for available optimization
                methods.
            jac (dict): Method for computing the gradient vector for scipy optimizers.
            hess (dict): Method for computing the hessian matrix for scipy optimizers.
            hessp (callable): Hessian of objective function times an arbitrary
                vector for scipy optimizers.
            bounds (sequence or Bounds): Bounds on variables for scipy optimizers.
            constraints (dict): Constraints definition for scipy optimizers.
            tol (float): Tolerance of termination for scipy optimizers.
            callback (callable): Called after each iteration for scipy optimizers.
            options (dict): a dictionary with options for the different optimizers.
            compile (bool): whether the TensorFlow graph should be compiled.
            processes (int): number of processes when using the paralle BFGS method.

        Return:
            The final expectation value.
            The corresponding best parameters.
            The optimization result object. For scipy methods it
                returns the ``OptimizeResult``, for ``'cma'`` the
                ``CMAEvolutionStrategy.result``, and for ``'sgd'``
                the options used during the optimization.
        """
        def _loss(params, circuit, hamiltonian):
            circuit.set_parameters(params)
            final_state = circuit()
            return hamiltonian.expectation(final_state)

        if compile:
            if get_backend() == "custom":
                raise_error(RuntimeError, "Cannot compile VQE that uses custom operators. "
                                          "Set the compile flag to False.")
            from qibo import K
            loss = K.compile(_loss)

        if method == 'sgd':
            # check if gates are using the MatmulEinsum backend
            from qibo.core.gates import BackendGate
            for gate in self.circuit.queue:
                if not isinstance(gate, BackendGate):
                    raise_error(RuntimeError, 'SGD VQE requires native Tensorflow '
                                              'gates because gradients are not '
                                              'supported in the custom kernels.')
            loss = _loss
        else:
            loss = lambda p, c, h: _loss(p, c, h).numpy()
        result, parameters, extra = self.optimizers.optimize(loss, initial_state,
                                                             args=(self.circuit, self.hamiltonian),
                                                             method=method, jac=jac, hess=hess, hessp=hessp,
                                                             bounds=bounds, constraints=constraints,
                                                             tol=tol, callback=callback, options=options,
                                                             compile=compile, processes=processes)
        self.circuit.set_parameters(parameters)
        return result, parameters, extra


class QAOA(object):
    """ Quantum Approximate Optimization Algorithm (QAOA) model.

    The QAOA is introduced in `arXiv:1411.4028 <https://arxiv.org/abs/1411.4028>`_.

    Args:
        hamiltonian (:class:`qibo.abstractions.hamiltonians.Hamiltonian`): problem Hamiltonian
            whose ground state is sought.
        mixer (:class:`qibo.abstractions.hamiltonians.Hamiltonian`): mixer Hamiltonian.
            If ``None``, :class:`qibo.hamiltonians.X` is used.
        solver (str): solver used to apply the exponential operators.
            Default solver is 'exp' (:class:`qibo.solvers.Exponential`).
        callbacks (list): List of callbacks to calculate during evolution.
        accelerators (dict): Dictionary of devices to use for distributed
            execution. See :class:`qibo.tensorflow.distcircuit.DistributedCircuit`
            for more details. This option is available only when ``hamiltonian``
            is a :class:`qibo.abstractions.hamiltonians.TrotterHamiltonian`.
        memory_device (str): Name of device where the full state will be saved.
            Relevant only for distributed execution (when ``accelerators`` is
            given).

    Example:
        ::

            import numpy as np
            from qibo import models, hamiltonians
            # create XXZ Hamiltonian for four qubits
            hamiltonian = hamiltonians.XXZ(4)
            # create QAOA model for this Hamiltonian
            qaoa = models.QAOA(hamiltonian)
            # optimize using random initial variational parameters
            # and default options and initial state
            initial_parameters = 0.01 * np.random.random(4)
            best_energy, final_parameters = qaoa.minimize(initial_parameters, method="BFGS")
    """
    from qibo import hamiltonians, optimizers, K
    from qibo.core import states
    from qibo.abstractions.hamiltonians import HAMILTONIAN_TYPES

    def __init__(self, hamiltonian, mixer=None, solver="exp", callbacks=[],
                 accelerators=None, memory_device="/CPU:0"):
        # list of QAOA variational parameters (angles)
        self.params = None
        # problem hamiltonian
        if not isinstance(hamiltonian, self.HAMILTONIAN_TYPES):
            raise_error(TypeError, "Invalid Hamiltonian type {}."
                                   "".format(type(hamiltonian)))
        self.hamiltonian = hamiltonian
        self.nqubits = hamiltonian.nqubits
        # mixer hamiltonian (default = -sum(sigma_x))
        if mixer is None:
            trotter = isinstance(
                self.hamiltonian, self.hamiltonians.TrotterHamiltonian)
            self.mixer = self.hamiltonians.X(self.nqubits, trotter=trotter)
        else:
            if type(mixer) != type(hamiltonian):
                  raise_error(TypeError, "Given Hamiltonian is of type {} "
                                         "while mixer is of type {}."
                                         "".format(type(hamiltonian),
                                                   type(mixer)))
            self.mixer = mixer

        # create circuits for Trotter Hamiltonians
        if (accelerators is not None and (
            not isinstance(self.hamiltonian, self.hamiltonians.TrotterHamiltonian)
            or solver != "exp")):
            raise_error(NotImplementedError, "Distributed QAOA is implemented "
                                             "only with TrotterHamiltonian and "
                                             "exponential solver.")
        if isinstance(self.hamiltonian, self.hamiltonians.TrotterHamiltonian):
            self.hamiltonian.circuit(1e-2, accelerators, memory_device)
            self.mixer.circuit(1e-2, accelerators, memory_device)

        # evolution solvers
        from qibo import solvers
        self.ham_solver = solvers.factory[solver](1e-2, self.hamiltonian)
        self.mix_solver = solvers.factory[solver](1e-2, self.mixer)

        self.state_cls = self.states.VectorState
        self.callbacks = callbacks
        self.accelerators = accelerators
        self.normalize_state = StateEvolution._create_normalize_state(
            self, solver)
        self.calculate_callbacks = StateEvolution._create_calculate_callbacks(
            self, accelerators, memory_device)

    def set_parameters(self, p):
        """Sets the variational parameters.

        Args:
            p (np.ndarray): 1D-array holding the new values for the variational
                parameters. Length should be an even number.
        """
        self.params = p

    def _apply_exp(self, state, solver, p):
        """Helper method for ``execute``."""
        solver.dt = p
        state = solver(state)
        if self.callbacks:
            state = self.normalize_state(state)
            self.calculate_callbacks(state)
        return state

    def execute(self, initial_state=None):
        """Applies the QAOA exponential operators to a state.

        Args:
            initial_state (np.ndarray): Initial state vector.

        Returns:
            State vector after applying the QAOA exponential gates.
        """
        state = self.get_initial_state(initial_state)
        self.calculate_callbacks(state)
        n = int(self.params.shape[0])
        for i in range(n // 2):
            state = self._apply_exp(state, self.ham_solver,
                                    self.params[2 * i])
            state = self._apply_exp(state, self.mix_solver,
                                    self.params[2 * i + 1])
        return self.normalize_state(state)

    def __call__(self, initial_state=None):
        """Equivalent to :meth:`qibo.models.QAOA.execute`."""
        return self.execute(initial_state)

    def get_initial_state(self, state=None):
        """"""
        if self.accelerators is not None:
            c = self.hamiltonian.circuit(self.params[0])
            if state is None:
                state = self.states.DistributedState.plus_state(c)
            return c.get_initial_state(state)

        if state is None:
            return self.state_cls.plus_state(self.nqubits).tensor
        return StateCircuit.get_initial_state(self, state)

    def minimize(self, initial_p, initial_state=None, method='Powell',
                 jac=None, hess=None, hessp=None, bounds=None, constraints=(),
                 tol=None, callback=None, options=None, compile=False, processes=None):
        """Optimizes the variational parameters of the QAOA.

        Args:
            initial_p (np.ndarray): initial guess for the parameters.
            initial_state (np.ndarray): initial state vector of the QAOA.
            method (str): the desired minimization method.
                See :meth:`qibo.optimizers.optimize` for available optimization
                methods.
            jac (dict): Method for computing the gradient vector for scipy optimizers.
            hess (dict): Method for computing the hessian matrix for scipy optimizers.
            hessp (callable): Hessian of objective function times an arbitrary
                vector for scipy optimizers.
            bounds (sequence or Bounds): Bounds on variables for scipy optimizers.
            constraints (dict): Constraints definition for scipy optimizers.
            tol (float): Tolerance of termination for scipy optimizers.
            callback (callable): Called after each iteration for scipy optimizers.
            options (dict): a dictionary with options for the different optimizers.
            compile (bool): whether the TensorFlow graph should be compiled.
            processes (int): number of processes when using the paralle BFGS method.

        Return:
            The final energy (expectation value of the ``hamiltonian``).
            The corresponding best parameters.
            The optimization result object. For scipy methods it
                returns the ``OptimizeResult``, for ``'cma'`` the
                ``CMAEvolutionStrategy.result``, and for ``'sgd'``
                the options used during the optimization.
        """
        if len(initial_p) % 2 != 0:
            raise_error(ValueError, "Initial guess for the parameters must "
                                    "contain an even number of values but "
                                    "contains {}.".format(len(initial_p)))

        def _loss(params, qaoa, hamiltonian):
            qaoa.set_parameters(params)
            state = qaoa(initial_state)
            return hamiltonian.expectation(state)

        if method == "sgd":
            from qibo import K
            loss = lambda p, c, h: _loss(K.cast(p), c, h)
        else:
            loss = lambda p, c, h: _loss(p, c, h).numpy()

        result, parameters, extra = self.optimizers.optimize(loss, initial_p, args=(self, self.hamiltonian),
                                                             method=method, jac=jac, hess=hess, hessp=hessp,
                                                             bounds=bounds, constraints=constraints,
                                                             tol=tol, callback=callback, options=options,
                                                             compile=compile, processes=processes)
        self.set_parameters(parameters)
        return result, parameters, extra


class Grover:
    """Model that performs Grover's algorithm.

    # TODO: Add a few more details on Grover's algorithm and/or reference.

    Args:
        oracle (:class:`qibo.core.circuit.Circuit`): quantum circuit that flips
            the sign using a Grover ancilla initialized with -X-H-
            and expected to have the total size of the circuit.
        initial_state (:class:`qibo.core.circuit.Circuit`): quantum circuit
            that initializes the state. Leave empty if |000..00>
        superposition (:class:`qibo.core.circuit.Circuit`): quantum circuit that
            takes an initial state to a superposition. Expected to use the first
            set of qubits to store the relevant superposition.
        sup_qubits (int): number of qubits that store the relevant superposition.
            Leave empty if superposition does not use ancillas.
        sup_size (int): how many states are in a superposition.
            Leave empty if its an equal superposition of quantum states.
        num_sol (int): number of expected solutions. Needed for normal Grover.
            Leave empty for iterative version.
        check (function): function that returns True if the solution has been
            found. Required of iterative approach.
            First argument should be the bitstring to check.
        check_args (tuple): arguments needed for the check function.
            The found bitstring not included.
    """
    def __init__(self, oracle, superposition=None, initial_state=None, sup_qubits=None, sup_size=None, num_sol=None, check=None, check_args=()):
        self.oracle = oracle
        self.initial_state = initial_state

        if superposition:
            self.superposition = superposition
        else:
            if not sup_qubits:
                raise_error(ValueError)
                # TODO: Consider a better way for the user to pass superposition qubits
                # and the number of ancillas
            self.superposition = Circuit(sup_qubits)
            self.superposition.add((gates.H(i) for i in range(sup_qubits)))

        if sup_qubits:
            self.sup_qubits = sup_qubits
        else:
            self.sup_qubits = self.superposition.nqubits

        if sup_size:
            self.sup_size = sup_size
        else:
            self.sup_size = int(2**self.superposition.nqubits)

        self.check = check
        self.check_args = check_args
        self.num_sol = num_sol

    def initialize(self):
        """Initialize the Grover algorithm with the superposition and Grover ancilla."""
        c = Circuit(self.oracle.nqubits)
        c.add(gates.X(self.oracle.nqubits - 1))
        c.add(gates.H(self.oracle.nqubits - 1))
        c.add(self.superposition.on_qubits(*range(self.superposition.nqubits)))
        return c

    def diffusion(self):
        """Construct the diffusion operator out of the superposition circuit."""
        nqubits = self.superposition.nqubits
        c = Circuit(nqubits + 1)
        c.add(self.superposition.invert().on_qubits(*range(nqubits)))
        if self.initial_state:
            c.add(self.initial_state.on_qubits(*range(self.initial_state.nqubits)))
        c.add([gates.X(i) for i in range(nqubits)])
        c.add(gates.X(nqubits).controlled_by(*range(nqubits)))
        c.add([gates.X(i) for i in range(nqubits)])
        if self.initial_state:
            c.add(self.initial_state.invert().on_qubits(*range(self.initial_state.nqubits)))
        c.add(self.superposition.on_qubits(*range(nqubits)))
        return c

    def step(self):
        """Combine oracle and diffusion for a Grover step."""
        c = Circuit(self.oracle.nqubits)
        c += self.oracle
        diffusion = self.diffusion()
        qubits = list(range(diffusion.nqubits - 1))
        qubits.append(self.oracle.nqubits - 1)
        c.add(diffusion.on_qubits(*qubits))
        return c

    def circuit(self, iterations):
        """Creates circuit that performs Grover's algorithm with a set amount of iterations.

        Args:
            iterations (int): number of times to repeat the Grover step.

        Returns:
            :class:`qibo.core.circuit.Circuit` that performs Grover's algorithm.
        """
        c = Circuit(self.oracle.nqubits)
        c += self.initialize()
        for _ in range(iterations):
            c += self.step()
        c.add(gates.M(*range(self.sup_qubits)))
        return c

    def iterative(self):
        """Iterative approach of Grover for when the number of solutions is not known."""
        k = 1
        lamda = 6/5 # TODO: Perhaps allow user to control these values? (Has to be between 1 and 4/3)
        total_iterations = 0
        while True:
            it = np.random.randint(k + 1)
            if it != 0:
                total_iterations += it
                circuit = self.circuit(it)
                result = circuit(nshots=1)
                measured = result.frequencies(binary=True).most_common(1)[0][0]
                if self.check(measured, *self.check_args):
                    return measured, total_iterations
            k = min(lamda * k, np.sqrt(self.sup_size))
            if total_iterations > 2*self.sup_size:
                raise_error(TimeoutError, "Cancelling iterative method as too "
                                          "many iterations have taken place.")

    def execute(self, nshots=100, freq=False):
        """Execute Grover's algorithm.

        If the number of solutions is given, calculates iterations,
        otherwise it uses an iterative approach.

        Args:
            nshots (int): number of shots in order to get the frequencies.
            freq (bool): print the full frequencies after the exact Grover algorithm.
        """
        #TODO: How do we want to return this information.
        if self.num_sol:
            it = np.int(np.pi * np.sqrt(self.sup_size / self.num_sol) / 4)
            circuit = self.circuit(it)
            result = circuit(nshots=nshots).frequencies(binary=True)
            if freq:
                log.info("Result of sampling Grover's algorihm")
                log.info(result)
            log.info(f"Most common states found using Grover's algorithm with {it} iterations:")
            most_common = result.most_common(self.num_sol)
            for i in most_common:
                log.info(i[0])
                if self.check:
                    if self.check(i[0], *self.check_args):
                        log.info('Solution checked and successful.')
                    else:
                        log.info('Not a solution of the problem. Something went wrong.')
        else:
            if not self.check:
                raise_error(ValueError, "Check function needed for iterative approach.")
            measured, total_iterations = self.iterative()
            log.info('Solution found in an iterative process.')
            log.info(f'Solution: {measured}\n')
            log.info(f'Total Grover iterations taken: {total_iterations}')
