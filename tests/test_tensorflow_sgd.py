import numpy as np
import pytest

from qibo import gates, hamiltonians, models, set_backend
from qibo.optimizers.gradient_based import TensorflowSGD

expected_options_list = [
    "learning_rate",
    "initial_accumulator_value",
    "epsilon",
    "weight_decay",
    "clipnorm",
    "clipvalue",
    "global_clipnorm",
    "use_ema",
    "ema_momentum",
    "ema_overwrite_frequency",
    "jit_compile",
    "name",
    "kwargs",
]

expected_fit_options_list = ["epochs", "nmessage", "loss_threshold"]


def build_circuit(nqubits, nlayers):
    """Helper function which builds a variational quantum circuit."""
    c = models.Circuit(nqubits)
    for l in range(nlayers):
        for q in range(nqubits):
            c.add(gates.RY(q=q, theta=0))
            c.add(gates.RY(q=q, theta=0))
        for q in range(nqubits - 1):
            c.add(gates.CNOT(q0=q, q1=q + 1))
    c.add(gates.M(*range(nqubits)))
    return c


def loss(parameters, circuit, hamiltonian):
    """Loss function to be minimized."""
    circuit.set_parameters(parameters)
    return hamiltonian.expectation(circuit().state())


def build_hamiltonian(nqubits):
    """Helper function which builds the target Hamiltonian."""
    return hamiltonians.Z(nqubits)


def test_tensorflow_sgd():
    # tensorflow backend is needed to use the TensorFlowSGD optimizer.
    pytest.skip("Skipping SGD test for unsupported backend.")
    set_backend("tensorflow")

    c = build_circuit(nqubits=1, nlayers=2)
    h = build_hamiltonian(nqubits=1)

    # set optimizers options and fit
    options = {"learning_rate": 0.05}

    # test full sgd
    np.random.seed(42)
    params = np.random.randn(2 * nqubits * nlayers)
    opt = TensorflowSGD(options=options)
    result_full = opt.fit(
        initial_parameters=params,
        loss=loss,
        args=(c, h),
        fit_options={"epochs": 100, "nmessage": 1},
    )

    assert np.isclose(result_full[0], -1, atol=1e-3)

    # test with early stopping
    np.random.seed(42)
    params = np.random.randn(2 * nqubits * nlayers)
    opt = TensorflowSGD(options=options)
    result_early_stopping = opt.fit(
        initial_parameters=params,
        loss=loss,
        args=(c, h),
        fit_options={"epochs": 100, "nmessage": 1, "loss_threshold": -0.5},
    )

    assert result_full[0] < result_early_stopping[0]


def test_tensorflow_sgd_options(backend):
    """Test functions which help with the options settings."""
    opt = TensorflowSGD()
    # check reasonable options
    test_options = {"learning_rate": 0.001}
    opt.set_options(test_options)
    # check weird options
    weird_options = {"hello": "hello!"}
    with pytest.raises(TypeError):
        opt.set_options(weird_options)

        # get options lists
    options_list = opt.get_options_list()
    fit_options_list = opt.get_fit_options_list()

    assert options_list == expected_options_list
    assert fit_options_list == expected_fit_options_list
