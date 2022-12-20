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
    c = Circuit(nqubits=1)
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

    # testing parameter out of bounds
    with pytest.raises(ValueError):
        grad_0 = parameter_shift(
            circuit=c, hamiltonian=test_hamiltonian, parameter_index=5
        )

    # testing hamiltonian type
    with pytest.raises(TypeError):
        grad_0 = parameter_shift(circuit=c, hamiltonian=c, parameter_shift=0)

    # executing all the procedure
    grad_0 = parameter_shift(circuit=c, hamiltonian=test_hamiltonian, parameter_index=0)
    grad_1 = parameter_shift(circuit=c, hamiltonian=test_hamiltonian, parameter_index=1)
    grad_2 = parameter_shift(circuit=c, hamiltonian=test_hamiltonian, parameter_index=2)

    assert isinstance(test_hamiltonian, hamiltonians.AbstractHamiltonian)
    # check of known values
    # calculated using tf.GradientTape
    assert round(grad_0, 10) == 8.51104358e-02
    assert round(grad_1, 8) == 5.20075970e-01
    assert round(grad_2, 10) == 0.0000000000
