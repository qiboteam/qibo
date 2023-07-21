import numpy as np
import pennylane as qml
import pytest

import qibo
from qibo.derivative import Parameter
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.optimizers import CMAES, SGD
from qibo.symbols import Symbol


def ansatz(layers, nqubits):
    """
    The circuit's ansatz: a sequence of RZ and RY with a beginning H gate
    Args:
        layers: integer, number of layers which compose the circuit
    Returns: abstract qibo circuit
    """

    c = qibo.models.Circuit(nqubits, density_matrix=True)

    c.add(qibo.gates.H(q=0))

    for _ in range(layers):
        c.add(qibo.gates.RZ(q=0, theta=0))
        c.add(qibo.gates.RY(q=0, theta=0))

    c.add(qibo.gates.M(0))

    return c


def loss_func(ypred, ytrue, other_args=None):
    loss = 0
    for i in range(len(ypred)):
        loss += (ytrue[i] - ypred[i]) ** 2

    return loss


def test_sgd_optimizer():
    circuit = ansatz(3, 1)
    parameters = []
    for _ in range(6):
        parameters.append(
            Parameter(lambda x, th1, th2: th1 * x + th2, [0.1, 0.1], featurep=True)
        )

    optimizer = SGD(circuit=circuit, parameters=parameters, loss=loss_func)
    X = np.array([0.1, 0.2, 0.3])
    y = np.array([0.2, 0.5, 0.7])
    losses = optimizer.fit(X, y)

    assert losses[-1] < 0.001


def create_hamiltonian(qubit, nqubit):
    """
    Creates appropriate Hamiltonian for Fubini-Matrix generation
    Args:
        qubit: qubit number whose state we are interested in
        nqubit: total number of qubits, which determines size of Hamiltonian
    Return:
        hamiltonian: SymbolicHamiltonian
    """
    standard = np.array([[1, 0], [0, 1]])
    target = np.array([[1, 0], [0, 0]])
    hams = []

    for i in range(nqubit):
        if i == qubit:
            hams.append(Symbol(i, target))
        else:
            hams.append(Symbol(i, standard))

    # create Hamiltonian
    obs = np.prod(hams)
    hamiltonian = SymbolicHamiltonian(obs)

    return hamiltonian


def cma_loss(params, circuit, hamiltonian):
    circuit.set_parameters(params)
    state = circuit().state()

    results = hamiltonian.expectation(state)

    return (results - 0.4) ** 2


def test_cma_optimizer():
    circuit = ansatz(3, 1)

    hamiltonian = create_hamiltonian(0, 1)

    parameters = np.array([0.1] * 6)

    optimizer = CMAES(
        initial_parameters=parameters, args=(circuit, hamiltonian), loss=cma_loss
    )

    fbest, xbest, r = optimizer.fit()

    assert fbest < 1e-5
