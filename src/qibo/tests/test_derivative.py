import numpy as np
import pytest

import qibo
from qibo import gates, hamiltonians
from qibo.derivative import parameter_shift
from qibo.models import Circuit


# defining an observable
def hamiltonian(nqubits=1):
    m0 = (1 / nqubits) * hamiltonians.Z(nqubits).matrix
    ham = hamiltonians.Hamiltonian(nqubits, m0)
    return ham


# defining a dummy circuit
def circuit(nqubits=1):
    c = Circuit(nqubits=1)
    c.add(gates.RY(q=0, theta=0))
    c.add(gates.RX(q=0, theta=0))
    c.add(gates.M(0))
    return c


def test_parameter_index():

    # initializing the circuit
    c = circuit(nqubits=1)

    # some parameters
    test_params = np.random.randn(2)
    c.set_parameters(test_params)

    test_hamiltonian = hamiltonian()

    # testing parameter out of bounds
    with pytest.raises(ValueError):
        grad_0 = parameter_shift(
            circuit=c, hamiltonian=test_hamiltonian, parameter_index=3
        )


def test_hamiltonian_type():

    c = circuit(nqubits=1)

    test_params = np.random.randn(2)
    c.set_parameters(test_params)

    # testing hamiltonian type
    with pytest.raises(TypeError):
        grad_0 = parameter_shift(circuit=c, hamiltonian=c, parameter_shift=0)
