import numpy as np
import pennylane as qml

import qibo
from qibo.backends import GlobalBackend
from qibo.derivative import Parameter, create_hamiltonian
from qibo.optimizers import CMAES, SGD, BasinHopping, Newtonian, ParallelBFGS


def ansatz(layers, nqubits, variational=False):
    """
    The circuit's ansatz: a sequence of RZ and RY with a beginning H gate
    Args:
        layers: integer, number of layers which compose the circuit
    Returns: abstract qibo circuit
    """
    if variational:
        c = qibo.models.variational.VariationalCircuit(nqubits, density_matrix=True)
    else:
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
            Parameter(lambda x, th1, th2: th1 * x + th2, [0.1, 0.1], featurep=[0.1])
        )

    optimizer = SGD(circuit=circuit, parameters=parameters, loss=loss_func_1qubit)
    X = np.array([0.1, 0.2, 0.3])
    y = np.array([0.2, 0.5, 0.7])
    losses = optimizer.fit(X, y)

    assert losses[-1] < 0.001

    return losses


def test_optimizer_parameter():
    c = qibo.models.Circuit(1, density_matrix=True)

    for i in range(1):
        c.add(qibo.gates.H(q=i))

        for _ in range(3):
            c.add(
                qibo.gates.RZ(
                    q=i,
                    theta=Parameter(
                        lambda x, th1, th2: th1 * x + th2, [0.1, 0.1], featurep=[0.1]
                    ),
                )
            )
            c.add(
                qibo.gates.RY(
                    q=i,
                    theta=Parameter(
                        lambda x, th1, th2: th1 * x + th2, [0.1, 0.1], featurep=[0.1]
                    ),
                )
            )
        c.add(qibo.gates.M(i))

    np.random.seed(42)
    # parameters = np.random.rand(12)
    parameters = [0.1] * 12

    optimizer = SGD(circuit=c, parameters=parameters, loss=loss_func_1qubit)
    X = np.array([0.1, 0.2, 0.3])
    y = np.array([0.2, 0.5, 0.7])
    losses = optimizer.fit(X, y)

    assert losses[-1] < 0.001

    return losses


def test_variational_circuit():
    VC = ansatz(3, 1, variational=True)
    parameters = []
    for _ in range(6):
        parameters.append(
            Parameter(lambda x, th1, th2: th1 * x + th2, [0.1, 0.1], featurep=[0.1])
        )

    X = np.array([0.1, 0.2, 0.3])
    y = np.array([0.2, 0.5, 0.7])
    losses = VC.optimize(X, y, parameters, loss_func_1qubit, method="sgd")

    assert losses[-1] < 0.001


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
                lambda x, th1, th2: th1 * x + th2, [i / 100, i / 53], featurep=[0.1]
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
            Parameter(lambda x, th1, th2: th1 * x + th2, [i, i + 1], featurep=[0.1])
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


def test_loss(x):
    return np.sum(x**2 + 6 * x - 1000)


def test_basin_hopping_optimizer():
    circuit = ansatz(3, 1)

    hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    parameters = np.array([0.1] * 6)

    optimizer = BasinHopping(
        initial_parameters=parameters,
        minimizer_kwargs=(circuit, hamiltonian),
        loss=test_loss,
    )

    fbest, xbest, r = optimizer.fit()

    assert fbest < 1e-5


if __name__ == "__main__":
    # test_parallel_bfgs_optimizer()
    # test_newtonian_optimizer()
    # test_multiqubit_sgd_optimizer()
    # test_sgd_optimizer()
    # test_sgd_methods()
    # test_variational_circuit()
    # test_basin_hopping_optimizer()
    test_optimizer_parameter()
