import numpy as np
import pennylane as qml
import pytest

import qibo
from qibo.backends import GlobalBackend
from qibo.derivative import Parameter
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.optimizers import CMAES, SGD
from qibo.symbols import I, Z


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
            Parameter(lambda x, th1, th2: th1 * x + th2, [0.1, 0.1], featurep=True)
        )

    optimizer = SGD(circuit=circuit, parameters=parameters, loss=loss_func_1qubit)
    X = np.array([0.1, 0.2, 0.3])
    y = np.array([0.2, 0.5, 0.7])
    losses = optimizer.fit(X, y)

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

    parameters = []
    for _ in range(12):
        parameters.append(
            Parameter(lambda x, th1, th2: th1 * x + th2, [0.1, 0.1], featurep=True)
        )

    hamiltonians = [create_hamiltonian(i, 2, GlobalBackend()) for i in range(2)]
    optimizer = SGD(
        circuit=circuit,
        parameters=parameters,
        loss=loss_func_2qubit,
        hamiltonian=hamiltonians,
    )
    X = np.array([0.1, 0.2, 0.3])
    y = np.array([[0.1, 0.2], [0.3, 0.5], [0.4, 0.5]])
    losses = optimizer.fit(X, y)

    assert losses[-1] < 0.001


def create_hamiltonian(qubit, nqubit, backend):
    """
    Creates appropriate Hamiltonian for a given list of qubits
    Args:
        qubit: qubit numbers whose states we are interested in
        nqubit: total number of qubits, which determines size of Hamiltonian
    Return:
        hamiltonian: SymbolicHamiltonian
    """
    if not isinstance(qubit, list):
        qubit = [qubit]

    hams = []

    for i in range(nqubit):
        if i in qubit:
            hams.append(Z(i))
        else:
            hams.append(I(i))

    # create Hamiltonian
    obs = np.prod([Z(i) for i in range(1)])
    hamiltonian = SymbolicHamiltonian(obs, backend=backend)

    return hamiltonian


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


if __name__ == "__main__":
    # test_multiqubit_sgd_optimizer()
    test_sgd_optimizer()
