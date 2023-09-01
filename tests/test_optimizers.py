import numpy as np

import qibo
from qibo.backends import GlobalBackend
from qibo.derivative import create_hamiltonian
from qibo.models.variational import VariationalCircuit
from qibo.optimizers import CMAES, SGD, BasinHopping, ParallelBFGS, Powell
from qibo.parameter import Parameter


def ansatz(layers, nqubits, theta=0):
    """
    The circuit's ansatz: a sequence of RZ and RY with a beginning H gate

    Developed by Michael Tsesmelis (ACSE-mct22)

    Args:
        layers (int): number of layers which compose the circuit
        nqubits (int): number of qubits in circuit
        theta (float or `class:parameter:Parameter`): fixed theta value, if required
    Returns: abstract qibo circuit
    """

    c = VariationalCircuit(nqubits, density_matrix=True)

    for i in range(nqubits):
        c.add(qibo.gates.H(q=i))

        for _ in range(layers):
            c.add(qibo.gates.RZ(q=i, theta=theta))
            c.add(qibo.gates.RY(q=i, theta=theta))

        c.add(qibo.gates.M(i))

    return c


def loss_func_1qubit(ypred, ytrue):
    """Simple Least-Squared Errors loss function

    Developed by Michael Tsesmelis (ACSE-mct22)
    """

    loss = np.sum(np.square(ytrue - ypred))

    return loss


def test_sgd_optimizer():
    """Test single qubit SGD optimizer

    Developed by Michael Tsesmelis (ACSE-mct22)
    """

    circuit = ansatz(
        3,
        1,
        theta=Parameter(lambda x, th1, th2: th1 * x + th2, [0.1, 0.1], feature=[0.1]),
    )
    parameters = [0.1] * 12

    optimizer = SGD(
        circuit=circuit,
        parameters=parameters,
        adam=True,
        loss=loss_func_1qubit,
        deterministic=True,
        epochs=10,
    )
    X = np.array([0.1, 0.2, 0.3])
    y = np.array([0.2, 0.5, 0.7])
    losses = optimizer.fit(y, X)

    assert np.isclose(losses[-1], 0.016386939869709672)

    return losses


def loss_func_2qubit(ypred, ytrue):
    """Simple Least-Squares Error for 2 qubits

    Developed by Michael Tsesmelis (ACSE-mct22)
    """

    loss = np.sum(np.square(ytrue - ypred))

    return loss


def test_multiqubit_sgd_optimizer():
    """Test 2-qubit, 2-hamiltonian ansatz with SGD Optimiser

    Developed by Michael Tsesmelis (ACSE-mct22)
    """

    nqubits = 2
    layers = 3

    c = VariationalCircuit(nqubits, density_matrix=True)

    for i in range(nqubits):
        c.add(qibo.gates.H(q=i))

        for _ in range(layers):
            c.add(
                qibo.gates.RZ(
                    q=i,
                    theta=Parameter(
                        lambda x, th1, th2: th1 * x + th2,
                        [i / 100, i / 53],
                        feature=[0.1],
                    ),
                )
            )
            c.add(
                qibo.gates.RY(
                    q=i,
                    theta=Parameter(
                        lambda x, th1, th2: th1 * x + th2,
                        [i / 100, i / 53],
                        feature=[0.1],
                    ),
                )
            )

        c.add(qibo.gates.M(i))

    parameters = [0.1] * 24

    hamiltonians = [create_hamiltonian(i, 2, GlobalBackend()) for i in range(2)]
    optimizer = SGD(
        circuit=c,
        parameters=parameters,
        loss=loss_func_2qubit,
        hamiltonian=hamiltonians,
        epochs=10,
    )

    X = np.array([0.1, 0.2, 0.3])
    y = np.array([[0.1, 0.2], [0.3, 0.5], [0.4, 0.5]])
    losses = optimizer.fit(y, X)

    assert losses[-1] < 0.1


def test_sgd_methods():
    """Test built-in SGD functions

    Developed by Michael Tsesmelis (ACSE-mct22)
    """

    nqubits = 2
    layers = 3

    c = VariationalCircuit(nqubits, density_matrix=True)

    for i in range(nqubits):
        c.add(qibo.gates.H(q=i))

        for _ in range(layers):
            c.add(
                qibo.gates.RZ(
                    q=i,
                    theta=Parameter(
                        lambda x, th1, th2: th1 * x + th2,
                        [i / 100, i / 53],
                        feature=[0.1],
                    ),
                )
            )
            c.add(
                qibo.gates.RY(
                    q=i,
                    theta=Parameter(
                        lambda x, th1, th2: th1 * x + th2,
                        [i / 100, i / 53],
                        feature=[0.1],
                    ),
                )
            )

        c.add(qibo.gates.M(i))

    parameters = [i for i in range(24)]

    hamiltonians = [create_hamiltonian(i, 2, GlobalBackend()) for i in range(2)]
    optimizer = SGD(
        circuit=c,
        parameters=parameters,
        loss=loss_func_2qubit,
        hamiltonian=hamiltonians,
        deterministic=True,
    )

    # test _get_params
    optimizer._circuit.set_variational_parameters(optimizer.params, feature=[0.5])
    gatep = optimizer._circuit.get_parameters()
    gatep = [v[0] for v in gatep]

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

    # test calculate_loss_function
    ypred = np.array([[0.2, 0.2], [0.3, 0.4], [0.4, 0.5]])
    ytrue = np.array([[0.2, 0.2], [0.3, 0.5], [0.4, 0.5]])
    optimizer.nlabels = 2
    grads = optimizer.calculate_loss_func_grad(ypred, ytrue, 1)
    assert np.allclose(grads, np.array([0.0, -0.2]))

    # test run_circuit
    expectation_values = optimizer.predict(0.1)

    assert np.allclose(
        np.array(expectation_values), np.array([0.1116947, 0.31656565]), atol=0.08
    )


def black_box_loss(params, circuit, hamiltonian):
    """Simple black-box optimizer loss function

    Developed by Michael Tsesmelis (ACSE-mct22)
    """

    circuit.set_parameters(params)
    state = circuit().state()

    results = hamiltonian.expectation(state)

    return (results - 0.4) ** 2


def test_cma_optimizer():
    """Test CMA global optimizer

    Developed by Michael Tsesmelis (ACSE-mct22)
    """

    circuit = ansatz(3, 1)

    hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    parameters = np.array([0.1] * 6)

    optimizer = CMAES(
        initial_parameters=parameters,
        args=(circuit, hamiltonian),
        loss=black_box_loss,
        options={"maxiter": 50},
    )

    fbest, xbest, r, it = optimizer.fit()

    assert fbest < 1e-3


def test_powell_optimizer():
    """Test Powell optimizer

    Developed by Michael Tsesmelis (ACSE-mct22)
    """

    circuit = ansatz(3, 1)

    hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    parameters = np.array([0.1] * 6)

    optimizer = Powell(
        initial_parameters=parameters, args=(circuit, hamiltonian), loss=black_box_loss
    )

    fbest, xbest, r, it = optimizer.fit()

    assert fbest < 1e-5


def test_parallel_bfgs_optimizer():
    """Test parallel BFGS optimizer

    Developed by Michael Tsesmelis (ACSE-mct22)
    """

    circuit = ansatz(3, 1)

    hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    parameters = np.array([0.1] * 6)

    optimizer = ParallelBFGS(
        initial_parameters=parameters, args=(circuit, hamiltonian), loss=black_box_loss
    )

    fbest, xbest, r, it = optimizer.fit()

    assert fbest < 1e-5


def test_basin_hopping_optimizer():
    """Test BasinHopping optimizer

    Developed by Michael Tsesmelis (ACSE-mct22)
    """

    circuit = ansatz(3, 1)

    hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    parameters = np.array([0.1] * 6)

    optimizer = BasinHopping(
        initial_parameters=parameters,
        args=(circuit, hamiltonian),
        loss=black_box_loss,
    )

    fbest, xbest, r, it = optimizer.fit()

    assert fbest < 1e-5


def test_single_output_optimizer():
    """Test SGD with single label, no feature

    Developed by Michael Tsesmelis (ACSE-mct22)
    """

    c = VariationalCircuit(1, density_matrix=True)
    c.add(qibo.gates.RX(q=0, theta=30.0))
    c.add(qibo.gates.RY(q=0, theta=30.0))
    c.add(qibo.gates.M(0))

    np.random.seed(42)
    pv = np.array([30, 30])

    sgd = SGD(c, pv, loss=loss_func_1qubit, natgrad=True)
    losses = sgd.fit(y=np.array([0.3]))

    assert np.allclose(losses, [0.1058, 0.0630, 0.0062, 0.00060], atol=0.01)
