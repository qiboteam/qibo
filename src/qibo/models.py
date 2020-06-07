from qibo.config import BACKEND_NAME
if BACKEND_NAME == "tensorflow":
    from qibo.tensorflow.circuit import TensorflowCircuit as Circuit
else:
    raise NotImplementedError("Only Tensorflow backend is implemented.")


def QFT(nqubits: int, with_swaps: bool = True, gates=None) -> Circuit:
    """Creates a circuit that implements the Quantum Fourier Transform.

    Args:
        nqubits (int): Number of qubits in the circuit.
        with_swaps (bool): Use SWAP gates at the end of the circuit so that the
            qubit order in the final state is the same as the initial state.
        gates: Which gates module will be used.
            The user can choose between the native tensorflow gates (:class:`qibo.tensorflow.gates`)
            or the gates that use custom operators (:class:`qibo.tensorflow.cgates`).
            If ``gates`` is ``None`` then custom gates will be used.

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
    import numpy as np
    if gates is None:
        from qibo import gates

    circuit = Circuit(nqubits)
    for i1 in range(nqubits):
        circuit.add(gates.H(i1))
        m = 2
        for i2 in range(i1 + 1, nqubits):
            theta = np.pi / 2 ** (m - 1)
            circuit.add(gates.CZPow(i2, i1, theta))
            m += 1

    if with_swaps:
        for i in range(nqubits // 2):
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
