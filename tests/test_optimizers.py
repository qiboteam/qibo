import numpy as np
import pennylane as qml

import qibo
from qibo.backends import GlobalBackend
from qibo.derivative import create_hamiltonian
from qibo.optimizers import CMAES, SGD, Newtonian, ParallelBFGS


class Parameter:
    """Trainable parameters and possibly features are linked through a lambda function
    which returns the final gate parameter"""

    def __init__(self, func, trainablep, features=None):
        self._variational_parameters = trainablep
        self._featurep = features
        self.nparams = len(trainablep)
        self.lambdaf = func

    def _apply_func(self, fixed_params=None):
        """Apply lambda function and return final gate parameter"""
        params = []
        if self._featurep is not None:
            if isinstance(self._featurep, list):
                params.extend(self._featurep)
            else:
                params.append(self._featurep)
        if fixed_params:
            params.extend(fixed_params)
        else:
            params.extend(self._variational_parameters)
        return self.lambdaf(*params)

    def _update_params(self, trainablep=None, feature=None):
        """Update gate trainable parameter and feature values"""
        if trainablep:
            self._variational_parameters = trainablep
        if feature and self._featurep:
            self._featurep = feature

    def get_gate_params(self, trainablep=None, feature=None):
        """Update values with trainable parameter and calculate current gate parameter"""
        self._update_params(trainablep=trainablep, feature=feature)
        return self._apply_func()

    def get_indices(self, start_index):
        """Return list of respective indices of trainable parameters within
        the optimizer's trainable parameter list"""
        return [start_index + i for i in range(self.nparams)]

    def get_fixed_part(self, trainablep_idx):
        """Retrieve parameter constant unaffected by a specific trainable parameter"""
        params = self._variational_parameters.copy()
        params[trainablep_idx] = 0.0
        return self._apply_func(fixed_params=params)

    def get_scaling_factor(self, trainablep_idx):
        """Get scaling factor multiplying a specific trainable parameter"""
        fixed = self.get_fixed_part(trainablep_idx)
        trainablep = self._variational_parameters
        trainablep[trainablep_idx] = 1.0
        gate_value = self.get_params(trainablep=trainablep)
        return gate_value - fixed


def ansatz(layers, nqubits):
    """
    The circuit's ansatz: a sequence of RZ and RY with a beginning H gate
    Args:
        layers: integer, number of layers which compose the circuit
    Returns: abstract qibo circuit
    """
    c = qibo.models.Circuit(nqubits, density_matrix=True)

    for i in range(nqubits):
        c.add(qibo.gates.H(q=i))

        for _ in range(layers):
            c.add(qibo.gates.RZ(q=i, theta=0))
            c.add(qibo.gates.RY(q=i, theta=0))

        c.add(qibo.gates.M(i))

    return c


def loss_func_1qubit(ypred, ytrue, other_args=None):
    loss = 0
    for i in range(len(ypred)):
        loss += (ytrue[i] - ypred[i]) ** 2

    return loss


def test_sgd_optimizer():
    """Test single qubit SGD optimizer"""

    circuit = ansatz(3, 1)
    parameters = []
    for _ in range(6):
        parameters.append(
            Parameter(lambda x, th1, th2: th1 * x + th2, [0.1, 0.1], features=True)
        )

    optimizer = SGD(circuit=circuit, parameters=parameters, loss=loss_func_1qubit)
    X = np.array([0.1, 0.2, 0.3])
    y = np.array([0.2, 0.5, 0.7])
    losses = optimizer.fit(X, y)

    assert losses[-1] < 0.001

    return losses


def loss_func_2qubit(ypred, ytrue, other_args=None):
    loss = 0
    for i in range(len(ypred)):
        for j in range(len(ypred[0])):
            loss += (ytrue[i][j] - ypred[i][j]) ** 2

    return loss


def test_multiqubit_sgd_optimizer():
    """Test 2-qubit, 2-hamiltonian ansatz with SGD Optimiser"""

    circuit = ansatz(3, 2)
    print(circuit.draw())

    parameters = []
    for i in range(12):
        parameters.append(
            Parameter(
                lambda x, th1, th2: th1 * x + th2, [i / 100, i / 53], features=True
            )
        )

    hamiltonians = [create_hamiltonian(i, 2, GlobalBackend()) for i in range(2)]
    optimizer = SGD(
        circuit=circuit,
        parameters=parameters,
        loss=loss_func_2qubit,
        hamiltonian=hamiltonians,
        epochs=100,
    )
    X = np.array([0.1, 0.2, 0.3])
    y = np.array([[0.1, 0.2], [0.3, 0.5], [0.4, 0.5]])
    losses = optimizer.fit(X, y)

    assert losses[-1] < 0.01


def test_sgd_methods():
    circuit = ansatz(3, 2)

    parameters = []
    for i in range(0, 24, 2):
        parameters.append(
            Parameter(lambda x, th1, th2: th1 * x + th2, [i, i + 1], features=True)
        )

    hamiltonians = [create_hamiltonian(i, 2, GlobalBackend()) for i in range(2)]
    optimizer = SGD(
        circuit=circuit,
        parameters=parameters,
        loss=loss_func_2qubit,
        hamiltonian=hamiltonians,
    )

    # _get_params
    trainablep = optimizer._get_params(trainable=True)
    assert trainablep == [i for i in range(24)]
    gatep = optimizer._get_params(trainable=False, feature=0.5)
    assert gatep == [
        1.0,
        4.0,
        7.0,
        10.0,
        13.0,
        16.0,
        19.0,
        22.0,
        25.0,
        28.0,
        31.0,
        34.0,
    ]

    # calculate_loss_function
    ypred = np.array([[0.2, 0.2], [0.3, 0.4], [0.4, 0.5]])
    ytrue = np.array([[0.2, 0.2], [0.3, 0.5], [0.4, 0.5]])
    optimizer.nlabels = 2
    grads = optimizer.calculate_loss_func_grad(ypred, ytrue, 1)
    assert np.allclose(grads, np.array([0.0, -0.2]))

    # run_circuit
    expectation_values = optimizer.run_circuit(0.1)
    assert np.allclose(
        np.array(expectation_values), np.array([-0.1116947, -0.31656565]), atol=0.08
    )


def cma_loss(params, circuit, hamiltonian):
    circuit.set_parameters(params)
    state = circuit().state()

    results = hamiltonian.expectation(state)

    return (results - 0.4) ** 2


def test_cma_optimizer():
    circuit = ansatz(3, 1)

    hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    parameters = np.array([0.1] * 6)

    optimizer = CMAES(
        initial_parameters=parameters, args=(circuit, hamiltonian), loss=cma_loss
    )

    fbest, xbest, r = optimizer.fit()

    assert fbest < 1e-5


def test_newtonian_optimizer():
    circuit = ansatz(3, 1)

    hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    parameters = np.array([0.1] * 6)

    optimizer = Newtonian(
        initial_parameters=parameters, args=(circuit, hamiltonian), loss=cma_loss
    )

    fbest, xbest, r = optimizer.fit()

    assert fbest < 1e-5


def test_parallel_bfgs_optimizer():
    circuit = ansatz(3, 1)

    hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    parameters = np.array([0.1] * 6)

    optimizer = ParallelBFGS(
        initial_parameters=parameters, args=(circuit, hamiltonian), loss=cma_loss
    )

    fbest, xbest, r = optimizer.fit()

    assert fbest < 1e-5


if __name__ == "__main__":
    # test_parallel_bfgs_optimizer()
    # test_newtonian_optimizer()
    # test_multiqubit_sgd_optimizer()
    test_sgd_optimizer()
    # test_sgd_methods()
