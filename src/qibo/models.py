from qibo.config import BACKEND_NAME
if BACKEND_NAME == "tensorflow":
    from qibo.tensorflow.circuit import TensorflowCircuit as Circuit
else:
    raise NotImplementedError("Only Tensorflow backend is implemented.")


def QFTCircuit(nqubits: int, with_swaps: bool = True) -> Circuit:
    """Creates a circuit that implements the Quantum Fourier Transform.

    Args:
        nqubits: Number of qubits in the circuit.
        with_swaps: Use SWAP gates at the end of the circuit so that the final
            qubit ordering agrees with the initial state.

    Returns:
        A qibo.models.Circuit that implements the Quantum Fourier Transform.
    """
    from qibo import gates

    circuit = Circuit(nqubits)
    for i1 in range(nqubits):
        circuit.add(gates.H(i1))
        m = 2
        for i2 in range(i1 + 1, nqubits):
            theta = 1.0 / 2 ** (m - 1)
            circuit.add(gates.CRZ(i2, i1, theta))
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
            v = VQE(ansats, XXZ(2))
            initial_state = np.random.uniform(0, 2*np.pi, 1)
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
            options (dict): a dictionary with options for the different optimizers.

        Return:
            The final expectation value.
            The corresponding best parameters.
        """
        def loss(params):
            s = self.ansatz(params)
            return self.hamiltonian.expectation(s)

        if compile:
            from qibo.config import K
            loss = K.function(loss)

        if method == 'cma':
            # Genetic optimizer
            import cma
            r = cma.fmin2(lambda p: loss(p).numpy(), initial_state, 1.7)
            result = r[1].result.fbest
            parameters = r[1].result.xbest
        elif method == 'sgd':
            from qibo.config import K
            vparams = K.Variable(initial_state)
            opt = K.optimizers.Adagrad(learning_rate=0.001)
            with K.GradientTape() as t:
                l = loss(vparams)
            trainable_variables = [vparams]
            grad = t.gradient(l, trainable_variables)
            for e in range(1000000):
                opt.apply_gradients(zip(grad, trainable_variables))
                if e % 1000 == True:
                    print('ite %d : loss %f' % (e, loss(vparams).numpy()))
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
