from qibo.config import BACKEND_NAME, raise_error
if BACKEND_NAME != "tensorflow": # pragma: no cover
    # case not tested because backend is preset to TensorFlow
    raise_error(NotImplementedError, "Only Tensorflow backend is implemented.")
from qibo.tensorflow.circuit import TensorflowCircuit as SimpleCircuit
from qibo.tensorflow.distcircuit import TensorflowDistributedCircuit as DistributedCircuit
from qibo.evolution import StateEvolution, AdiabaticEvolution
from typing import Dict, Optional


class Circuit(DistributedCircuit):
    """Factory class for circuits.

    Creates both normal and distributed circuits.
    """

    def __new__(cls, nqubits: int,
                accelerators: Optional[Dict[str, int]] = None,
                memory_device: str = "/CPU:0"):
        if accelerators is None:
            return SimpleCircuit(nqubits)
        else:
            return DistributedCircuit(nqubits, accelerators, memory_device)

    @classmethod
    def from_qasm(cls, qasm_code: str,
                  accelerators: Optional[Dict[str, int]] = None,
                  memory_device: str = "/CPU:0"):
      if accelerators is None:
          return SimpleCircuit.from_qasm(qasm_code)
      else:
          return DistributedCircuit.from_qasm(qasm_code,
                                              accelerators=accelerators,
                                              memory_device=memory_device)


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

    import numpy as np
    from qibo import gates

    circuit = Circuit(nqubits)
    for i1 in range(nqubits):
        circuit.add(gates.H(i1))
        for i2 in range(i1 + 1, nqubits):
            theta = np.pi / 2 ** (i2 - i1)
            circuit.add(gates.CZPow(i2, i1, theta))

    if with_swaps:
        for i in range(nqubits // 2):
            circuit.add(gates.SWAP(i, nqubits - i - 1))

    return circuit


def _DistributedQFT(nqubits: int,
                    accelerators: Optional[Dict[str, int]] = None,
                    memory_device: str = "/CPU:0") -> DistributedCircuit:
    """QFT with the order of gates optimized for reduced multi-device communication."""
    import numpy as np
    from qibo import gates

    circuit = Circuit(nqubits, accelerators, memory_device)
    icrit = nqubits // 2 + nqubits % 2
    if accelerators is not None:
        circuit.global_qubits = range(circuit.nlocal, nqubits)
        if icrit < circuit.nglobal:
            raise_error(NotImplementedError, "Cannot implement QFT for {} qubits "
                                             "using {} global qubits."
                                             "".format(nqubits, circuit.nglobal))

    for i1 in range(nqubits):
        if i1 < icrit:
            i1eff = i1
        else:
            i1eff = nqubits - i1 - 1
            circuit.add(gates.SWAP(i1, i1eff))

        circuit.add(gates.H(i1eff))
        for i2 in range(i1 + 1, nqubits):
            theta = np.pi / 2 ** (i2 - i1)
            circuit.add(gates.CZPow(i2, i1eff, theta))

    return circuit


class VQE(object):
    """This class implements the variational quantum eigensolver algorithm.

    Args:
        circuit (:class:`qibo.base.circuit.BaseCircuit`): Circuit that
            implements the variaional ansatz.
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): Hamiltonian object.

    Example:
        ::

            import numpy as np
            from qibo import gates, models, hamiltonians
            # create circuit ansatz for two qubits
            circuit = models.Circuit(2)
            circuit.add(gates.RY(q, theta=0))
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

    def minimize(self, initial_state, method='Powell', options=None, compile=True):
        """Search for parameters which minimizes the hamiltonian expectation.

        Args:
            initial_state (array): a initial guess for the parameters of the
                variational circuit.
            method (str): the desired minimization method.
                See :meth:`qibo.optimizers.optimize` for available optimization
                methods.
            options (dict): a dictionary with options for the different optimizers.
            compile (bool): whether the TensorFlow graph should be compiled.

        Return:
            The final expectation value.
            The corresponding best parameters.
        """
        def loss(params):
            self.circuit.set_parameters(params)
            final_state = self.circuit()
            return self.hamiltonian.expectation(final_state)

        if compile:
            if not self.circuit.using_tfgates:
                raise_error(RuntimeError, "Cannot compile VQE that uses custom operators. "
                                          "Set the compile flag to False.")
            from qibo import K
            loss = K.function(loss)

        if method == 'sgd':
            # check if gates are using the MatmulEinsum backend
            from qibo.tensorflow.gates import TensorflowGate
            for gate in self.circuit.queue:
                if not isinstance(gate, TensorflowGate):
                    raise_error(RuntimeError, 'SGD VQE requires native Tensorflow '
                                              'gates because gradients are not '
                                              'supported in the custom kernels.')

            result, parameters = self.optimizers.optimize(loss, initial_state,
                                                          "sgd", options,
                                                          compile)
        else:
            result, parameters = self.optimizers.optimize(
                lambda p: loss(p).numpy(), initial_state, method, options)

        self.circuit.set_parameters(parameters)
        return result, parameters


class QAOA(object):
    """ Quantum Approximate Optimization Algorithm (QAOA) model.

    The QAOA is introduced in `arXiv:1411.4028 <https://arxiv.org/abs/1411.4028>`_.

    Args:
        hamiltonian (:class:`qibo.base.hamiltonians.Hamiltonian`): problem Hamiltonian
            whose ground state is sought.
        mixer (:class:`qibo.base.hamiltonians.Hamiltonian`): mixer Hamiltonian.
            If ``None``, :class:`qibo.hamiltonians.X` is used.
        solver (str): solver used to apply the exponential operators.
            Default solver is 'exp' (:class:`qibo.solvers.Exponential`).
        callbacks (list): List of callbacks to calculate during evolution.
        accelerators (dict): Dictionary of devices to use for distributed
            execution. See :class:`qibo.tensorflow.distcircuit.TensorflowDistributedCircuit`
            for more details. This option is available only when ``hamiltonian``
            is a :class:`qibo.base.hamiltonians.TrotterHamiltonian`.
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
    from qibo import hamiltonians, optimizers
    from qibo.config import K, DTYPES
    from qibo.base.hamiltonians import HAMILTONIAN_TYPES

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
            if state is None:
                state = "ones"
            c = self.hamiltonian.circuit(self.params[0])
            return c.get_initial_state(state)

        if state is None:
            # Generate |++...+> state
            dtype = self.DTYPES.get('DTYPECPX')
            n = self.K.cast(2 ** self.nqubits, dtype=self.DTYPES.get('DTYPEINT'))
            state = self.K.ones(n, dtype=dtype)
            norm = self.K.cast(2 ** float(self.nqubits / 2.0), dtype=dtype)
            return state / norm
        return SimpleCircuit._cast_initial_state(self, state)

    def minimize(self, initial_p, initial_state=None, method='Powell', options=None):
        """Optimizes the variational parameters of the QAOA.

        Args:
            initial_p (np.ndarray): initial guess for the parameters.
            initial_state (np.ndarray): initial state vector of the QAOA.
            method (str): the desired minimization method.
                See :meth:`qibo.optimizers.optimize` for available optimization
                methods.
            options (dict): a dictionary with options for the different optimizers.

        Return:
            The final energy (expectation value of the ``hamiltonian``).
            The corresponding best parameters.
        """
        if len(initial_p) % 2 != 0:
            raise_error(ValueError, "Initial guess for the parameters must "
                                    "contain an even number of values but "
                                    "contains {}.".format(len(initial_p)))

        def _loss(p):
            self.set_parameters(p)
            state = self(initial_state)
            return self.hamiltonian.expectation(state)

        if method == "sgd":
            import tensorflow as tf
            loss = lambda p: _loss(tf.cast(
                p, dtype=self.DTYPES.get('DTYPECPX')))
        else:
            import numpy as np
            loss = lambda p: _loss(p).numpy()

        result, parameters = self.optimizers.optimize(loss, initial_p, method,
                                                      options)
        self.set_parameters(parameters)
        return result, parameters
