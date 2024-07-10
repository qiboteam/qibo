import numpy as np
import pytest

from qibo import Circuit, gates, hamiltonians
from qibo.derivative import finite_differences, parameter_shift
from qibo.symbols import Z


# defining an observable
def hamiltonian(nqubits, backend):
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
    backend.set_seed(42)

    # some parameters
    # we know the derivative's values with these params
    test_params = np.linspace(0.1, 1, 3)
    test_params *= scale_factor
    c.set_parameters(test_params)

    test_hamiltonian = hamiltonian(nqubits=1, backend=backend)

    # testing parameter out of bounds
    with pytest.raises(ValueError):
        grad = parameter_shift(
            circuit=c, hamiltonian=test_hamiltonian, parameter_index=5
        )

    # testing hamiltonian type
    with pytest.raises(TypeError):
        grad = parameter_shift(
            circuit=c, hamiltonian=c, parameter_index=0, nshots=nshots
        )

    # executing all the procedure
    grad = [
        parameter_shift(
            circuit=c,
            hamiltonian=test_hamiltonian,
            parameter_index=i,
            scale_factor=scale_factor,
            nshots=nshots,
        )
        for i in range(3)
    ]

    # check of known values
    for i in range(3):
        backend.assert_allclose(grad[i], grads[i], atol=atol)


@pytest.mark.parametrize("step_size", [10**-i for i in range(5, 10, 1)])
def test_finite_differences(backend, step_size):
    # initializing the circuit
    c = circuit(nqubits=1)
    backend.set_seed(42)

    # some parameters
    test_params = np.linspace(0.1, 1, 3)
    grads = [-8.51104358e-02, -5.20075970e-01, 0]
    atol = 1e-6
    c.set_parameters(test_params)

    test_hamiltonian = hamiltonian(nqubits=1, backend=backend)

    # testing parameter out of bounds
    with pytest.raises(ValueError):
        grad_0 = finite_differences(
            circuit=c, hamiltonian=test_hamiltonian, parameter_index=5
        )

    # testing hamiltonian type
    with pytest.raises(TypeError):
        grad_0 = finite_differences(circuit=c, hamiltonian=c, parameter_index=0)

    # executing all the procedure
    grad_0 = finite_differences(
        circuit=c, hamiltonian=test_hamiltonian, parameter_index=0, step_size=step_size
    )
    grad_1 = finite_differences(
        circuit=c, hamiltonian=test_hamiltonian, parameter_index=1, step_size=step_size
    )
    grad_2 = finite_differences(
        circuit=c, hamiltonian=test_hamiltonian, parameter_index=2, step_size=step_size
    )

    # check of known values
    # calculated using tf.GradientTape
    backend.assert_allclose(grad_0, grads[0], atol=atol)
    backend.assert_allclose(grad_1, grads[1], atol=atol)
    backend.assert_allclose(grad_2, grads[2], atol=atol)
