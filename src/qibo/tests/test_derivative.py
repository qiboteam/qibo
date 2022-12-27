import numpy as np
import pytest

from qibo import gates, hamiltonians
from qibo.derivative import parameter_shift
from qibo.models import Circuit


# defining an observable
def hamiltonian(nqubits, backend):
    return (1 / nqubits) * hamiltonians.Z(nqubits, backend=backend)


# defining a dummy circuit
def circuit(nqubits=1):
    c = Circuit(nqubits)
    # all gates for which generator eigenvalue is implemented
    c.add(gates.RX(q=0, theta=0))
    c.add(gates.RY(q=0, theta=0))
    c.add(gates.RZ(q=0, theta=0))
    c.add(gates.M(0))
    return c


@pytest.mark.parametrize(
    "scale_factor, grads",
    [(1, [8.51104358e-02, 5.20075970e-01, 0]), (0.5, [0.02405061, 0.13560379, 0])],
)
def test_derivative(backend, scale_factor, grads):

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
        grad_0 = parameter_shift(circuit=c, hamiltonian=c, parameter_index=0)

    # executing all the procedure
    grad_0 = parameter_shift(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=0,
        scale_factor=scale_factor,
    )
    grad_1 = parameter_shift(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=1,
        scale_factor=scale_factor,
    )
    grad_2 = parameter_shift(
        circuit=c,
        hamiltonian=test_hamiltonian,
        parameter_index=2,
        scale_factor=scale_factor,
    )

    # check of known values
    # calculated using tf.GradientTape
    backend.assert_allclose(grad_0, grads[0], atol=1e-8)
    backend.assert_allclose(grad_1, grads[1], atol=1e-8)
    backend.assert_allclose(grad_2, grads[2], atol=1e-8)
