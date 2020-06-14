from qibo.config import BACKEND_NAME
if BACKEND_NAME != "tensorflow":
    raise NotImplementedError("Only Tensorflow backend is implemented.")
from qibo.tensorflow.circuit import TensorflowCircuit as SimpleCircuit
from qibo.tensorflow.distcircuit import TensorflowDistributedCircuit as DistributedCircuit
from typing import Dict, Optional


class Circuit:
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
        memory_device: str = "/CPU:0",
        backend: str = "Custom") -> Circuit:
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
        backend: Which backend to use among ``'Custom'``, ``'DefaultEinsum'``
            or ``'MatmulEinsum'``.

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
        if backend != "Custom":
            raise TypeError("Distributed QFT supports only custom operator gates.")
        return _DistributedQFT(nqubits, accelerators, memory_device, with_swaps)

    import numpy as np
    if backend == "Custom":
        from qibo import gates
    elif backend == "DefaultEinsum" or backend == "MatmulEinsum":
        from qibo.tensorflow import gates
    else:
        raise ValueError("Unknown backend {}.".format(backend))

    circuit = Circuit(nqubits)
    for i1 in range(nqubits):
        circuit.add(gates.H(i1).with_backend(backend))
        m = 2
        for i2 in range(i1 + 1, nqubits):
            theta = np.pi / 2 ** (m - 1)
            circuit.add(gates.CZPow(i2, i1, theta).with_backend(backend))
            m += 1

    if with_swaps:
        for i in range(nqubits // 2):
            circuit.add(gates.SWAP(i, nqubits - i - 1).with_backend(backend))

    return circuit


def _DistributedQFT(nqubits: int,
                    accelerators: Dict[str, int],
                    memory_device: str = "/CPU:0",
                    with_swaps: bool = True) -> DistributedCircuit:
    """QFT with the order of gates optimized for reduced multi-device communication."""
    import numpy as np
    from qibo.tensorflow import cgates as gates

    circuit = Circuit(nqubits, accelerators, memory_device)
    nqubits = circuit.nqubits
    nglobal = circuit.nglobal

    for i1 in range(nqubits - nglobal):
        for i2 in range(i1):
            theta = np.pi / 2 ** (i1 - i2)
            circuit.add(gates.CZPow(i1, i2, theta))
        circuit.add(gates.H(i1))

    for i2 in range(nglobal):
        for i1 in range(nqubits - nglobal, nqubits):
            theta = np.pi / 2 ** (i1 - i2)
            circuit.add(gates.CZPow(i1, i2, theta))

    for i1 in range(nqubits - nglobal, nqubits):
        for i2 in range(nglobal, i1):
            theta = np.pi / 2 ** (i1 - i2)
            circuit.add(gates.CZPow(i1, i2, theta))
        circuit.add(gates.H(i1))

    if with_swaps:
        for i in range(nglobal, nqubits // 2):
            circuit.add(gates.SWAP(i, nqubits - i - 1))
        for i in range(nglobal):
            circuit.add(gates.SWAP(i, nqubits - i - 1))

    return circuit


class VQE(object):
    """This class implements the variational quantum eigensolver algorithm.

    Args:
        ansatz (function): a python function which takes as input an array of parameters.
        hamiltonian (qibo.hamiltonians): a hamiltonian object.

    Example:
        ::

            import numpy as np
            from qibo import gates
            from qibo.hamiltonians import XXZ
            from qibo.models import VQE, Circuit
            def ansatz(theta):
                c = Circuit(2)
                c.add(gates.RY(q, theta[0]))
                return c
            v = VQE(ansats, XXZ(2))
            initial_state = np.random.uniform(0, 2, 1)
            v.minimize(initial_state)
    """
    def __init__(self, ansatz, hamiltonian):
        """Initialize ansatz and hamiltonian."""
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian

    def minimize(self, initial_state, method='BFGS', options=None, compile=True):
        """Search for parameters which minimizes the hamiltonian expectation.

        Args:
            initial_state (array): a initial guess for the circuit.
            method (str): the desired minimization method.
                One of ``"cma"`` (genetic optimizer), ``"sgd"`` (gradient descent) or
                any of the methods supported by `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.
            options (dict): a dictionary with options for the different optimizers.

        Return:
            The final expectation value.
            The corresponding best parameters.
        """
        def loss(params):
            s = self.ansatz(params)()
            return self.hamiltonian.expectation(s)

        if compile:
            circuit = self.ansatz(initial_state)
            if not circuit.using_tfgates:
                raise RuntimeError("Cannot compile VQE that uses custom operators.")
            from qibo.config import K
            loss = K.function(loss)

        if method == 'cma':
            # Genetic optimizer
            import cma
            r = cma.fmin2(lambda p: loss(p).numpy(), initial_state, 1.7)
            result = r[1].result.fbest
            parameters = r[1].result.xbest

        elif method == 'sgd':
            # check if gates are using the MatmulEinsum backend
            from qibo.tensorflow.gates import TensorflowGate
            circuit = self.ansatz(initial_state)
            for gate in circuit.queue:
                if not isinstance(gate, TensorflowGate):
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
            from qibo.config import K
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

        return result, parameters
