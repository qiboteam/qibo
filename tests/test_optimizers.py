import numpy as np
import pytest

import qibo
from qibo import hamiltonians
from qibo.models import Circuit
from qibo.optimizers.heuristics import CMAES, BasinHopping
from qibo.optimizers.minimizers import ParallelBFGS, ScipyMinimizer


def create_hamiltonian(nqubits=1):
    """Precomputes Hamiltonian."""
    return hamiltonians.Z(nqubits)


def ansatz(layers=1, nqubits=1, theta=0):
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

    optimizer = ScipyMinimizer()

    result = optimizer.fit(
        initial_parameters=parameters, loss=black_box_loss, args=(circuit, hamiltonian)
    )

    # get options lists
    options_list = optimizer.get_options_list()
    fit_options_list = optimizer.get_fit_options_list()

    assert np.isclose(result[0], -1, atol=1e-4)


def test_parallel_bfgs(backend):
    """Test parallel BFGS optimizer"""

    circuit = ansatz()
    hamiltonian = create_hamiltonian()
    parameters = np.random.randn(len(circuit.get_parameters()))

    optimizer = ParallelBFGS()

    result = optimizer.fit(
        initial_parameters=parameters, args=(circuit, hamiltonian), loss=black_box_loss
    )

    # get options lists
    options_list = optimizer.get_options_list()
    fit_options_list = optimizer.get_fit_options_list()

    assert np.isclose(result[0], -1, atol=1e-4)


def test_basinhopping(backend):
    """Test BasinHopping optimizer"""

    circuit = ansatz()
    hamiltonian = create_hamiltonian()
    parameters = np.random.randn(len(circuit.get_parameters()))

    optimizer = BasinHopping()

    result = optimizer.fit(
        initial_parameters=parameters,
        args=(circuit, hamiltonian),
        loss=black_box_loss,
    )

    # get options lists
    options_list = optimizer.get_options_list()
    fit_options_list = optimizer.get_fit_options_list()

    assert np.isclose(result[0], -1, atol=1e-4)


def test_cmaes(backend):
    """Test CMA global optimizer"""

    circuit = ansatz()
    hamiltonian = create_hamiltonian()
    parameters = np.random.randn(len(circuit.get_parameters()))

    optimizer = CMAES()

    result = optimizer.fit(
        initial_parameters=parameters, args=(circuit, hamiltonian), loss=black_box_loss
    )

    # get options lists
    options_list = optimizer.get_options_list()
    fit_options_list = optimizer.get_fit_options_list()
    reduced_fit_options_list = optimizer.get_fit_options_list(keyword="tol")

    assert np.isclose(result[0], -1, atol=1e-4)
    assert len(reduced_fit_options_list) < len(fit_options_list)


def test_optimizer_arguments(backend):
    """Test option dictionary mechanism."""

    # setting reasonable options
    options = {"sigma0": 0.5}
    opt = CMAES(options=options)

    # setting wrong options
    options.update({"hello": "hello"})
    with pytest.raises(TypeError):
        opt = CMAES(options=options)

    # passing wrong args data structure (not a tuple)
    circuit = ansatz()
    hamiltonian = create_hamiltonian()
    parameters = np.random.randn(len(circuit.get_parameters()))
    optimizer = ScipyMinimizer()

    with pytest.raises(TypeError):
        _ = optimizer.fit(
            initial_parameters=parameters,
            loss=black_box_loss,
            args=[circuit, hamiltonian],
        )

    # passing parameters not in array or list shape
    parameters = {"parameters": parameters}

    with pytest.raises(TypeError):
        _ = optimizer.fit(
            initial_parameters=parameters,
            loss=black_box_loss,
            args=(circuit, hamiltonian),
        )
