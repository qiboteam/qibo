import numpy as np
import pytest
import tensorflow as tf

import qibo
from qibo import gates, hamiltonians
from qibo.backends import GlobalBackend
from qibo.derivative import (
    finite_differences,
    parameter_shift,
    stochastic_parameter_shift,
)
from qibo.models import Circuit
from qibo.models.parameter import Parameter
from qibo.symbols import Z

qibo.set_backend("tensorflow")


# defining an observable
def hamiltonian(nqubits, backend=GlobalBackend()):
    return hamiltonians.hamiltonians.SymbolicHamiltonian(
        np.prod([Z(i) for i in range(nqubits)]), backend=backend
    )


# defining a dummy circuit
def circuit(nqubits=1):
    c = Circuit(nqubits)
    # all gates for which generator eigenvalue is implemented
    c.add(gates.RX(q=0, theta=0))
    c.add(gates.RY(q=0, theta=0))
    c.add(gates.RZ(q=0, theta=0))
    c.add(gates.M(0))

    return c


@pytest.mark.parametrize("nshots, atol", [(None, 1e-8), (100000, 1e-2)])
@pytest.mark.parametrize(
    "scale_factor, grads",
    [(1, [-8.51104358e-02, -5.20075970e-01, 0]), (0.5, [-0.02405061, -0.13560379, 0])],
)
def test_standard_parameter_shift(backend, nshots, atol, scale_factor, grads):
    # initializing the circuit
    c = circuit(nqubits=1)

    # some parameters
    # we know the derivative's values with these params
    test_params = np.linspace(0.1, 1, 3)
    test_params *= scale_factor
    c.set_parameters(test_params)

    test_hamiltonian = hamiltonian(nqubits=1, backend=backend)

    # testing parameter out of bounds
    with pytest.raises(ValueError):
        grad_0 = parameter_shift(
            circuit=c, hamiltonian=test_hamiltonian, parameter_index=5
        )

    # testing hamiltonian type
    with pytest.raises(TypeError):
        grad_0 = parameter_shift(
            circuit=c, hamiltonian=c, parameter_index=0, nshots=nshots
        )

    # executing all the procedure
    grad_0 = parameter_shift(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=0,
        scale_factor=scale_factor,
        nshots=nshots,
    )
    grad_1 = parameter_shift(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=1,
        scale_factor=scale_factor,
        nshots=nshots,
    )
    grad_2 = parameter_shift(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=2,
        scale_factor=scale_factor,
        nshots=nshots,
    )

    # check of known values
    # calculated using tf.GradientTape
    backend.assert_allclose(grad_0, grads[0], atol=atol)
    backend.assert_allclose(grad_1, grads[1], atol=atol)
    backend.assert_allclose(grad_2, grads[2], atol=atol)


def gradient_exact():
    backend = GlobalBackend()

    test_params = tf.Variable(np.linspace(0.1, 1, 3))

    with tf.GradientTape() as tape:
        c = circuit(nqubits=1)
        c.set_parameters(test_params)

        ham = hamiltonian(1)
        results = ham.expectation(
            backend.execute_circuit(circuit=c, initial_state=None).state()
        )

    gradients = tape.gradient(results, test_params)

    return gradients


@pytest.mark.parametrize("nshots, atol", [(None, 1e-1), (100000, 1e-1)])
def test_finite_differences(backend, nshots, atol):
    # exact gradients
    grads = gradient_exact()

    # initializing the circuit
    c = circuit(nqubits=1)

    # some parameters
    # we know the derivative's values with these params
    test_params = np.linspace(0.1, 1, 3)
    c.set_parameters(test_params)

    test_hamiltonian = hamiltonian(nqubits=1)

    # testing parameter out of bounds
    with pytest.raises(ValueError):
        grad_0 = finite_differences(
            circuit=c, hamiltonian=test_hamiltonian, parameter_index=5
        )

    # testing hamiltonian type
    with pytest.raises(TypeError):
        grad_0 = finite_differences(
            circuit=c, hamiltonian=c, parameter_index=0, nshots=nshots
        )

    # executing all the procedure
    grad_0 = finite_differences(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=0,
        nshots=nshots,
    )
    grad_1 = finite_differences(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=1,
        nshots=nshots,
    )
    grad_2 = finite_differences(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=2,
        nshots=nshots,
    )

    # check of known values
    # calculated using tf.GradientTape
    backend.assert_allclose(grad_0, grads[0], atol=atol)
    backend.assert_allclose(grad_1, grads[1], atol=atol)
    backend.assert_allclose(grad_2, grads[2], atol=atol)


@pytest.mark.parametrize("nshots, atol", [(None, 1e-1), (1024, 1e-1)])
def test_spsr(backend, nshots, atol):
    # exact gradients
    grads = gradient_exact()

    # initializing the circuit
    c = circuit(nqubits=1)

    # some parameters
    # we know the derivative's values with these params
    test_params = []
    param_values = np.linspace(0.1, 1, 3) * 0.5

    for i in range(3):
        test_params.append(
            Parameter(lambda th1, th2: th1 + th2, [param_values[i], param_values[i]])
        )

    parameter_values = [param.get_params() for param in test_params]
    print(parameter_values)
    c.set_parameters(parameter_values)

    test_hamiltonian = hamiltonian(nqubits=1, backend=GlobalBackend())

    # testing parameter out of bounds
    with pytest.raises(ValueError):
        grad_0 = stochastic_parameter_shift(
            circuit=c,
            hamiltonian=test_hamiltonian,
            parameter_index=5,
            parameter=Parameter(lambda th: 2 * th, [0.1]),
        )

    # testing hamiltonian type
    with pytest.raises(TypeError):
        grad_0 = stochastic_parameter_shift(
            circuit=c,
            hamiltonian=c,
            parameter_index=0,
            parameter=Parameter(lambda th: 2 * th, [0.1]),
            nshots=nshots,
        )

    # executing all the procedure
    grads_0 = stochastic_parameter_shift(c, test_hamiltonian, 0, test_params[0])[0]
    grads_1 = stochastic_parameter_shift(c, test_hamiltonian, 1, test_params[1])[0]
    grads_2 = stochastic_parameter_shift(c, test_hamiltonian, 2, test_params[2])[0]

    print(grads_0, grads_1, grads_2, grads)

    # check of known values
    backend.assert_allclose(grads_0, grads[0], atol=atol)
    backend.assert_allclose(grads_1, grads[1], atol=atol)
    backend.assert_allclose(grads_2, grads[2], atol=atol)
