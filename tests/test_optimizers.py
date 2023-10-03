import numpy as np

import qibo
from qibo import hamiltonians
from qibo.backends import GlobalBackend, matrices
from qibo.hamiltonians import Hamiltonian
from qibo.models import Circuit
from qibo.optimizers import BFGS, CMAES, BasinHopping, ParallelBFGS, Powell


def create_hamiltonian(qubit=0, nqubits=1, backend=None):
    """Precomputes Hamiltonian.

    Args:
        nqubits (int): number of qubits.
        z_qubit (int): qubit where the Z measurement is applied, must be z_qubit < nqubits
        backend (:class:`qibo.backends.abstract.Backend`): Backend object to use for execution.
            If ``None`` the currently active global backend is used.
            Default is ``None``.

    Returns:
        (:class:`qibo.hamiltonians.Hamiltonian`): hamiltonian object.
    """
    eye = matrices.I
    if qubit == 0:
        h = hamiltonians.Z(1).matrix
        for _ in range(nqubits - 1):
            h = np.kron(h, eye)

    elif qubit == nqubits - 1:
        h = eye
        for _ in range(nqubits - 2):
            h = np.kron(eye, h)
        h = np.kron(h, hamiltonians.Z(1).matrix)
    else:
        h = eye
        for _ in range(nqubits - 1):
            if _ + 1 == qubit:
                h = np.kron(matrices.Z, h)
            else:
                h = np.kron(eye, h)
    return Hamiltonian(nqubits, h, backend=backend)


def ansatz(layers, nqubits, theta=0):
    """
    The circuit's ansatz: a sequence of RZ and RY with a beginning H gate

    Args:
        layers (int): number of layers which compose the circuit
        nqubits (int): number of qubits in circuit
        theta (float or `class:parameter:Parameter`): fixed theta value, if required
    Returns: abstract qibo circuit
    """

    c = Circuit(nqubits, density_matrix=True)

    for i in range(nqubits):
        c.add(qibo.gates.H(q=i))

        for _ in range(layers):
            c.add(qibo.gates.RZ(q=i, theta=theta))
            c.add(qibo.gates.RY(q=i, theta=theta))

        c.add(qibo.gates.M(i))

    return c


def black_box_loss(params, circuit, hamiltonian):
    """Simple black-box optimizer loss function"""

    circuit.set_parameters(params)
    state = circuit().state()

    results = hamiltonian.expectation(state)

    return (results - 0.4) ** 2


def test_cma_optimizer():
    """Test CMA global optimizer"""

    circuit = ansatz(3, 1)

    hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    parameters = np.array([0.1] * 6)

    optimizer = CMAES(
        initial_parameters=parameters,
        args=(circuit, hamiltonian),
        loss=black_box_loss,
        options={"maxiter": 50},
    )

    fbest, xbest, r, it = optimizer.fit()

    assert fbest < 1e-3


def test_powell_optimizer():
    """Test Powell optimizer"""

    circuit = ansatz(3, 1)

    hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    parameters = np.array([0.1] * 6)

    optimizer = Powell(
        initial_parameters=parameters, args=(circuit, hamiltonian), loss=black_box_loss
    )

    fbest, xbest, r, it = optimizer.fit()

    assert fbest < 1e-5


def test_parallel_bfgs_optimizer():
    """Test parallel BFGS optimizer"""

    circuit = ansatz(3, 1)

    hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    parameters = np.array([0.1] * 6)

    optimizer = ParallelBFGS(
        initial_parameters=parameters, args=(circuit, hamiltonian), loss=black_box_loss
    )

    fbest, xbest, r, it = optimizer.fit()

    assert fbest < 1e-5


def test_bfgs_optimizer():
    """Test BFGS optimizer"""

    circuit = ansatz(3, 1)

    hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    parameters = np.array([0.1] * 6)

    optimizer = BFGS(
        initial_parameters=parameters, args=(circuit, hamiltonian), loss=black_box_loss
    )

    fbest, xbest, r, it = optimizer.fit()

    assert fbest < 1e-5


def test_basin_hopping_optimizer():
    """Test BasinHopping optimizer"""

    circuit = ansatz(3, 1)

    hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    parameters = np.array([0.1] * 6)

    optimizer = BasinHopping(
        initial_parameters=parameters,
        args=(circuit, hamiltonian),
        loss=black_box_loss,
    )

    fbest, xbest, r, it = optimizer.fit()

    assert fbest < 1e-5
