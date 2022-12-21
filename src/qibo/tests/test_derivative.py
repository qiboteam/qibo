import numpy as np
import pytest

from qibo import gates, hamiltonians
from qibo.derivative import parameter_shift, rescaled_parameter_shift
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


def test_derivative(backend):

    # initializing the circuit
    c = circuit(nqubits=1)

    # some parameters
    # we know the derivative's values with these params
    test_params = np.linspace(0.1, 1, 3)
    c.set_parameters(test_params)

    test_hamiltonian = hamiltonian(nqubits=1, backend=backend)

    # ------------------- normal parameter shift rule --------------------------

    # testing parameter out of bounds
    with pytest.raises(ValueError):
        grad_0 = parameter_shift(
            circuit=c, hamiltonian=test_hamiltonian, parameter_index=5
        )

    # testing hamiltonian type
    with pytest.raises(TypeError):
        grad_0 = parameter_shift(circuit=c, hamiltonian=c, parameter_index=0)

    # executing all the procedure
    grad_0 = parameter_shift(circuit=c, hamiltonian=test_hamiltonian, parameter_index=0)
    grad_1 = parameter_shift(circuit=c, hamiltonian=test_hamiltonian, parameter_index=1)
    grad_2 = parameter_shift(circuit=c, hamiltonian=test_hamiltonian, parameter_index=2)

    # check of known values
    # calculated using tf.GradientTape
    backend.assert_allclose(grad_0, 8.51104358e-02, atol=1e-10)
    backend.assert_allclose(grad_1, 5.20075970e-01, atol=1e-10)
    backend.assert_allclose(grad_2, 0, atol=1e-10)

    # ------------------- rescaled parameter shift rule ------------------------

    # params * scale_factor
    x = 0.5
    test_params *= 0.5
    c.set_parameters(test_params)

    # testing parameter out of bounds
    with pytest.raises(ValueError):
        grad_0_res = rescaled_parameter_shift(
            circuit=c, hamiltonian=test_hamiltonian, parameter_index=5, scale_factor=x
        )

    # testing hamiltonian type
    with pytest.raises(TypeError):
        grad_0_res = rescaled_parameter_shift(
            circuit=c, hamiltonian=c, parameter_index=0, scale_factor=x
        )

    # executing all the procedure
    grad_0_res = rescaled_parameter_shift(
        circuit=c, hamiltonian=test_hamiltonian, parameter_index=0, scale_factor=x
    )
    grad_1_res = rescaled_parameter_shift(
        circuit=c, hamiltonian=test_hamiltonian, parameter_index=1, scale_factor=x
    )
    grad_2_res = rescaled_parameter_shift(
        circuit=c, hamiltonian=test_hamiltonian, parameter_index=2, scale_factor=x
    )

    # check of known values
    # calculated using tf.GradientTape
    backend.assert_allclose(grad_0_res, 0.02405061, atol=1e-8)
    backend.assert_allclose(grad_1_res, 0.13560379, atol=1e-8)
    backend.assert_allclose(grad_2_res, 0, atol=1e-8)
