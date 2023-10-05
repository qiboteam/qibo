import numpy as np

import qibo
from qibo import hamiltonians
from qibo.models import Circuit
from qibo.optimizers.heuristics import CMAES, BasinHopping
from qibo.optimizers.minimizers import ParallelBFGS, ScipyMinimizer


def create_hamiltonian(nqubits=3):
    """Precomputes Hamiltonian."""
    return hamiltonians.Z(nqubits)


def ansatz(layers=1, nqubits=3, theta=0):
    """
    The circuit's ansatz: a sequence of RZ and RY with a beginning H gate

    Args:
        layers (int): number of layers which compose the circuit
        nqubits (int): number of qubits in circuit
        theta (float or `class:parameter:Parameter`): fixed theta value, if required
    Returns: abstract qibo circuit
    """

    c = Circuit(nqubits=nqubits)

    for i in range(nqubits):
        c.add(qibo.gates.H(q=i))

        for _ in range(layers):
            c.add(qibo.gates.RZ(q=i, theta=theta))
            c.add(qibo.gates.RY(q=i, theta=theta))

    c.add(qibo.gates.M(*range(nqubits)))

    return c


def black_box_loss(params, circuit, hamiltonian):
    """Simple black-box optimizer loss function"""
    circuit.set_parameters(params)
    return hamiltonian.expectation(circuit().state())


def test_scipyminimizer(backend):
    """Test ScipyMinimizer."""

    circuit = ansatz()
    hamiltonian = create_hamiltonian()
    parameters = np.random.randn(len(circuit.get_parameters()))

    optimizer = ScipyMinimizer(
        initial_parameters=parameters, args=(circuit, hamiltonian), loss=black_box_loss
    )

    result = optimizer.fit()
    assert np.isclose(result[0], -3, atol=1e-8)


def test_parallel_bfgs(backend):
    """Test parallel BFGS optimizer"""

    circuit = ansatz()
    hamiltonian = create_hamiltonian()
    parameters = np.random.randn(len(circuit.get_parameters()))

    optimizer = ParallelBFGS(
        initial_parameters=parameters, args=(circuit, hamiltonian), loss=black_box_loss
    )

    result = optimizer.fit()
    assert np.isclose(result[0], -3, atol=1e-8)


def test_basinhopping(backend):
    """Test BasinHopping optimizer"""

    circuit = ansatz()
    hamiltonian = create_hamiltonian()
    parameters = np.random.randn(len(circuit.get_parameters()))

    optimizer = BasinHopping(
        initial_parameters=parameters,
        args=(circuit, hamiltonian),
        loss=black_box_loss,
    )

    result = optimizer.fit()
    assert np.isclose(result[0], -3, atol=1e-8)


def test_cmaes(backend):
    """Test CMA global optimizer"""

    circuit = ansatz()
    hamiltonian = create_hamiltonian()
    parameters = np.random.randn(len(circuit.get_parameters()))

    optimizer = CMAES(
        initial_parameters=parameters, args=(circuit, hamiltonian), loss=black_box_loss
    )

    result = optimizer.fit()
    assert np.isclose(result[0], -3, atol=1e-6)
