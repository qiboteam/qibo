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


@pytest.mark.parametrize("step_size", [10**-i for i in range(5, 10, 1)])
def test_finite_differences(backend, step_size):
    # initializing the circuit
    c = circuit(nqubits=1)

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


import tensorflow as tf

from qibo import gates
from qibo.backends import GlobalBackend, set_backend
from qibo.derivative import VAR, create_hamiltonian, finite_differences, parameter_shift
from qibo.models.circuit import Circuit

set_backend("tensorflow")


def gradient_exact():
    """Calculates exact gradient of a circuit"""

    backend = GlobalBackend()

    test_params = tf.Variable(np.linspace(0.1, 1, 3))

    with tf.GradientTape() as tape:
        c = circuit(nqubits=1)
        c.set_parameters(test_params)

        ham = create_hamiltonian(0, 1, GlobalBackend())
        results = ham.expectation(
            backend.execute_circuit(circuit=c, initial_state=None).state()
        )

    gradients = tape.gradient(results, test_params)

    return gradients


one_qubit = tf.constant(
    [[-1.0 - 0.0j, 0.0 - 0.0j], [0.0 - 0.0j, 1.0 + 0.0j]], dtype=tf.complex128
)

two_qubit = tf.constant(
    [
        [-1.0 - 0.0j, 0.0 - 0.0j, 0.0 - 0.0j, 0.0 - 0.0j],
        [0.0 - 0.0j, 1.0 + 0.0j, 0.0 - 0.0j, 0.0 + 0.0j],
        [0.0 - 0.0j, 0.0 - 0.0j, -1.0 - 0.0j, 0.0 - 0.0j],
        [0.0 - 0.0j, 0.0 + 0.0j, 0.0 - 0.0j, 1.0 + 0.0j],
    ],
    dtype=tf.complex128,
)


@pytest.mark.parametrize(
    "qubit, nqubits, matrix", [(0, 1, one_qubit), (1, 2, two_qubit)]
)
def test_create_hamiltonian(qubit, nqubits, matrix):
    """Test the `create_hamiltonian` function"""

    ham = create_hamiltonian(qubit, nqubits, GlobalBackend())
    assert np.allclose(ham.matrix, matrix)


def test_psr_commuting_gate():
    """Test PSR with commuting gates, i.e. [X, Y]=0"""

    # hyperparameters
    scale_factor = 0.5
    nshots = None
    test_hamiltonian = create_hamiltonian(0, 1, GlobalBackend())

    # separating gates
    c = Circuit(1)
    c.add(gates.H(q=0))
    c.add(gates.RY(q=0, theta=0))
    c.add(gates.RY(q=0, theta=0))
    c.add(gates.M(0))

    params = np.array([0.1, 0.2])
    c.set_parameters(params)

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

    # single gate
    c2 = Circuit(1)
    c2.add(gates.H(q=0))
    c2.add(gates.RY(q=0, theta=0))
    c2.add(gates.M(0))

    params = np.array([0.3])
    c2.set_parameters(params)

    grad_2 = parameter_shift(
        circuit=c2,
        hamiltonian=test_hamiltonian,
        parameter_index=0,
        scale_factor=scale_factor,
        nshots=nshots,
    )

    assert np.isclose(grad_0, grad_1, atol=1e-5)
    assert np.isclose(grad_1, grad_2, atol=1e-5)


def RXRYN(phi):
    """Simple generator with non-commuting elements"""

    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0.0]])

    return 0.3 * X - phi * Y


def spsr_circuit_RXRY_decomposed(phi, s, shift):
    """RXRY decomposed into its 3 constituents for SPSR"""

    ham = create_hamiltonian(0, 1, GlobalBackend())

    c1 = Circuit(nqubits=1)
    start = gates.OneQubitGate(
        0, "H", exponentiated=True, generator=RXRYN, scaling=s, phi=phi
    )
    c1.add(start)
    c1.add(VAR(start).return_gate(item="phi", angle=shift))
    c1.add(
        gates.OneQubitGate(
            0, "H", exponentiated=True, generator=RXRYN, scaling=(1 - s), phi=phi
        )
    )
    c1.add(gates.M(0))

    backend = GlobalBackend()

    val = ham.expectation(
        backend.execute_circuit(circuit=c1, initial_state=None).state()
    )

    return val


def spsr_circuit_RXRY(phi):
    """RXRY gate and circuit used for SPSR"""

    ham = create_hamiltonian(0, 1, GlobalBackend())
    c1 = Circuit(nqubits=1)
    c1.add(gates.RXRY(0, phi, 1.0))
    c1.add(gates.M(0))

    backend = GlobalBackend()

    val = ham.expectation(
        backend.execute_circuit(circuit=c1, initial_state=None).state()
    )

    c1.set_parameters([phi + 0.001, 1.0])
    forward = ham.expectation(
        backend.execute_circuit(circuit=c1, initial_state=None).state()
    )

    c1.set_parameters([phi - 0.001, 1.0])
    backward = ham.expectation(
        backend.execute_circuit(circuit=c1, initial_state=None).state()
    )

    diff = (forward - backward) / 0.002

    return val, diff


def test_spsr_RXRY():
    """Test `stochastic_parameter_shift` decomposed into its constituents on RXRY"""

    np.random.seed(1430)
    backend = GlobalBackend()

    angles = np.linspace(0.1, 2 * np.pi, 50)

    evals = [spsr_circuit_RXRY(theta1) for theta1 in angles]
    fdiff = [res[1] for res in evals]
    evals = [res[0] for res in evals]

    # spsr
    pos_vals = np.array(
        [
            [
                spsr_circuit_RXRY_decomposed(theta1, s=s, shift=np.pi / 4)
                for s in np.random.uniform(size=10)
            ]
            for theta1 in angles
        ]
    )

    neg_vals = np.array(
        [
            [
                spsr_circuit_RXRY_decomposed(theta1, s=s, shift=-np.pi / 4)
                for s in np.random.uniform(size=10)
            ]
            for theta1 in angles
        ]
    )

    spsr_vals = (pos_vals - neg_vals).mean(axis=1)

    assert np.allclose(fdiff, spsr_vals, atol=0.05)


def circuit_rxry(nqubits=1):
    """Small circuit ansatz"""

    c = Circuit(nqubits)
    # all gates for which generator eigenvalue is implemented
    c.add(gates.H(q=0))
    c.add(gates.RX(q=0, theta=0))
    c.add(gates.OneQubitGate(0, "H", exponentiated=True, generator=RXRYN, phi=0.1))
    c.add(gates.RZ(q=0, theta=0))
    c.add(gates.M(0))

    return c


def gradient_exact_rxry():
    """Calculates exact gradient of a circuit"""

    backend = GlobalBackend()

    test_params = tf.Variable(np.linspace(0.1, 1, 3))

    with tf.GradientTape() as tape:
        c = circuit(nqubits=1)
        c.set_parameters(test_params)

        ham = create_hamiltonian(0, 1, GlobalBackend())
        results = ham.expectation(
            backend.execute_circuit(circuit=c, initial_state=None).state()
        )

    gradients = tape.gradient(results, test_params)

    return gradients


# def test_stochastic_parameter_shift():
