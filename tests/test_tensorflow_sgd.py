import numpy as np
import pytest

from qibo import gates, hamiltonians, models, set_backend
from qibo.optimizers.gradient_based import TensorflowSGD


def test_tensorflow_sgd():
    # tensorflow backend is needed to use the TensorFlowSGD optimizer.
    set_backend("tensorflow")
    pytest.skip("Skipping SGD test for unsupported backend.")

    # define a dummy model
    nqubits = 1
    nlayers = 2

    c = models.Circuit(nqubits)
    for l in range(nlayers):
        for q in range(nqubits):
            c.add(gates.RY(q=q, theta=0))
            c.add(gates.RY(q=q, theta=0))
        for q in range(nqubits - 1):
            c.add(gates.CNOT(q0=q, q1=q + 1))
    c.add(gates.M(*range(nqubits)))

    # define a loss function
    h = hamiltonians.Z(nqubits)

    def loss(parameters, circuit, hamiltonian):
        circuit.set_parameters(parameters)
        return hamiltonian.expectation(circuit().state())

    # set optimizers options and fit
    options = {"learning_rate": 0.05}

    # test full sgd
    np.random.seed(42)
    params = np.random.randn(2 * nqubits * nlayers)
    opt = TensorflowSGD(
        initial_parameters=params, loss=loss, args=(c, h), options=options
    )
    result_full = opt.fit(epochs=100, nmessage=1)

    assert np.isclose(result_full[0], -1, atol=1e-3)

    # test with early stopping
    np.random.seed(42)
    params = np.random.randn(2 * nqubits * nlayers)
    opt = TensorflowSGD(
        initial_parameters=params, loss=loss, args=(c, h), options=options
    )
    result_early_stopping = opt.fit(epochs=100, nmessage=1, loss_treshold=-0.5)

    assert result_full[0] < result_early_stopping[0]
