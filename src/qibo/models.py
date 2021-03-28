import math
from qibo import get_backend
from qibo.config import raise_error
from qibo.core.circuit import Circuit as StateCircuit
from qibo.core.circuit import DensityMatrixCircuit
from qibo.evolution import StateEvolution, AdiabaticEvolution
from typing import Dict, Optional


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
            for gate in self.circuit.queue:
                _ = gate.cache
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
