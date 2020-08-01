from qibo.config import BACKEND_NAME
if BACKEND_NAME != "tensorflow": # pragma: no cover
    raise NotImplementedError("Only Tensorflow backend is implemented.")
from qibo.tensorflow.circuit import TensorflowCircuit as SimpleCircuit
from qibo.tensorflow.distcircuit import TensorflowDistributedCircuit as DistributedCircuit
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
            raise NotImplementedError("Distributed QFT is only implemented "
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
            raise NotImplementedError("Cannot implement QFT for {} qubits "
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
                One of ``"cma"`` (genetic optimizer), ``"sgd"`` (gradient descent) or
                any of the methods supported by `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.
            options (dict): a dictionary with options for the different optimizers.

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
                raise RuntimeError("Cannot compile VQE that uses custom operators. "
                                   "Set the compile flag to False.")
            from qibo import K
            loss = K.function(loss)

        if method == 'cma': # pragma: no cover
            # Genetic optimizer
            import cma
            r = cma.fmin2(lambda p: loss(p).numpy(), initial_state, 1.7)
            result = r[1].result.fbest
            parameters = r[1].result.xbest

        elif method == 'sgd':
            # check if gates are using the MatmulEinsum backend
            from qibo.tensorflow.gates import TensorflowGate
            for gate in self.circuit.queue:
                if not isinstance(gate, TensorflowGate): # pragma: no cover
                    raise RuntimeError('SGD VQE requires native Tensorflow '
                                       'gates because gradients are not '
                                       'supported in the custom kernels.')

            sgd_options = {"nepochs": 1000000,
                           "nmessage": 1000,
                           "optimizer": "Adagrad",
                           "learning_rate": 0.001}
            if options is not None:
                sgd_options.update(options)

            # proceed with the training
            from qibo import K
            vparams = K.Variable(initial_state)
            optimizer = getattr(K.optimizers, sgd_options["optimizer"])(
              learning_rate=sgd_options["learning_rate"])

            def opt_step():
                with K.GradientTape() as tape:
                    l = loss(vparams)
                grads = tape.gradient(l, [vparams])
                optimizer.apply_gradients(zip(grads, [vparams]))
                return l

            if compile:
                opt_step = K.function(opt_step)

            for e in range(sgd_options["nepochs"]):
                l = opt_step()
                if e % sgd_options["nmessage"] == 1:
                    print('ite %d : loss %f' % (e, l.numpy()))

            result = loss(vparams).numpy()
            parameters = vparams.numpy()

        else:
            # Newtonian approaches
            import numpy as np
            from scipy.optimize import minimize
            n = self.hamiltonian.nqubits
            m = minimize(lambda p: loss(p).numpy(), initial_state,
                         method=method, options=options)
            result = m.fun
            parameters = m.x

        self.circuit.set_parameters(parameters)
        return result, parameters


class StateEvolution:
    """Unitary time evolution of a state vector under a Hamiltonian.

    Args:
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): Hamiltonian to
            evolve under.

    Example:
        ::

            import numpy as np
            from qibo import models, hamiltonians
            # create critical (h=1.0) TFIM Hamiltonian for three qubits
            hamiltonian = hamiltonians.TFIM(3, h=1.0)
            # initialize evolution model
            evolution = models.StateEvolution(hamiltonian)
            # initialize state to |+++>
            initial_state = np.ones(8) / np.sqrt(8)
            # evolve the state for T=1
            final_state = evolution(1, initial_state)
    """

    from qibo import solvers

    def __init__(self, hamiltonian):
        self.nqubits = hamiltonian.nqubits
        self.hamiltonian = hamiltonian

    def execute(self, total_time, dt=None, initial_state=None, solver="exp",
                callbacks=[]):
        """Runs unitary evolution for a given total time.

        Args:
            total_time (float): Total time to evolve for. Initial time is t=0.
            dt (float): Time step to use for the numerical integration of
                Schrondiger's equation.
                Not required if the Hamiltonian is time-independent and the
                exponential solver is used.
            initial_state (np.ndarray): Initial state of the evolution.
            solver (str): Solver to use for integrating Schrodinger's equation.
            callbacks (list): List of callbacks to calculate during evolution.

        Returns:
            Final state vector a ``tf.Tensor``.
        """
        state = self._cast_initial_state(initial_state)
        if dt is None:
            dt = total_time

        solver = self.solvers.factory[solver](dt, self.hamiltonian)
        nsteps = int(total_time / solver.dt)
        for callback in callbacks:
            callback.append(callback(state))
        for _ in range(nsteps):
            state = solver(state)
            for callback in callbacks:
                callback.append(callback(state))
        return state

    def __call__(self, total_time, dt=None, initial_state=None, solver="exp",
                 callbacks=[]):
        """Equivalent to :meth:`qibo.models.StateEvolution.execute`."""
        return self.execute(total_time, dt, initial_state, solver, callbacks)

    def _cast_initial_state(self, initial_state=None):
        """Casts initial state as a Tensorflow tensor."""
        if initial_state is None:
            raise ValueError("StateEvolution cannot be used without initial "
                             "state.")
        return SimpleCircuit._cast_initial_state(self, initial_state)


class AdiabaticEvolution(StateEvolution):
    """Adiabatic evolution of a state vector under the following Hamiltonian:

    .. math::
        H(t) = (1 - s(t)) H_0 + s(t) H_1

    Args:
        h0 (:class:`qibo.hamiltonians.Hamiltonian`): Easy Hamiltonian.
        h1 (:class:`qibo.hamiltonians.Hamiltonian`): Problem Hamiltonian.
        s (callable): Function of time that defines the scheduling of the
            adiabatic evolution.
    """

    def __init__(self, h0, h1, s):
        if h0.nqubits != h1.nqubits:
            raise ValueError("H0 has {} qubits while H1 has {}."
                             "".format(h0.nqubits, h1.nqubits))
        if s(0) != 0:
            raise ValueError("s(0) should be 0 but is {}.".format(s(0)))
        self.nqubits = h0.nqubits
        self.s = s
        self.h0 = h0
        self.h1 = h1

    # disable pylint warning because ``hamiltonian`` is defined as an
    # attribute given by user in ``StateEvolution``
    def hamiltonian(self, t): # pylint: disable=E0202
        """Calculates the Hamiltonian at a given time.

        Args:
            t (float): Time value.

        Returns:
            :class:`qibo.hamiltonians.Hamiltonian` object corresponding to the
            evolution Hamiltonian at the given time.
        """
        return (1 - self.s(t)) * self.h0 + self.s(t) * self.h1

    def execute(self, total_time, *args, **kwargs):
        """"""
        st = self.s(total_time)
        if st != 1:
            raise ValueError("s(T) should be 1 but is {}.".format(st))
        return super(AdiabaticEvolution, self).execute(total_time, *args,
                                                       **kwargs)

    def _cast_initial_state(self, initial_state=None):
        """Casts initial state as a Tensorflow tensor.

        If initial state is not given the ground state of ``h0`` is used, which
        is the common practice in adiabatic evolution.
        """
        if initial_state is None:
            return self.h0.eigenvectors()[:, 0]
        return super(AdiabaticEvolution, self)._cast_initial_state(initial_state)
