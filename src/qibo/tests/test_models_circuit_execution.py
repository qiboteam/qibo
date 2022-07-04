import pytest
import numpy as np
from qibo import gates
from qibo.models import Circuit


def test_eager_execute(backend, accelerators):
    c = Circuit(4, accelerators)
    c.add((gates.H(i) for i in range(4)))
    final_state = backend.execute_circuit(c)
    target_state = np.ones(16) / 4.0
    backend.assert_allclose(final_state, target_state)


def test_compiled_execute(backend):
    def create_circuit(theta = 0.1234):
        c = Circuit(2)
        c.add(gates.X(0))
        c.add(gates.X(1))
        c.add(gates.CU1(0, 1, theta))
        return c

    # Try to compile circuit without gates
    empty_c = Circuit(2)
    with pytest.raises(RuntimeError):
        empty_c.compile()

    # Run eager circuit
    c1 = create_circuit()
    r1 = backend.execute_circuit(c1)

    # Run compiled circuit
    c2 = create_circuit()
    c2.compile(backend)
    r2 = c2()
    np.testing.assert_allclose(r1, r2)


def test_compiling_twice_exception(backend):
    """Check that compiling a circuit a second time raises error."""
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1)])
    c.compile()
    with pytest.raises(RuntimeError):
        c.compile()

# TODO: Test circuit execution with measurements
# TODO: Test compiled circuit execution with measurements

@pytest.mark.linux
def test_memory_error(backend, accelerators):
    """Check that ``RuntimeError`` is raised if device runs out of memory."""
    c = Circuit(40, accelerators)
    c.add((gates.H(i) for i in range(0, 40, 5)))
    with pytest.raises(RuntimeError):
        final_state = backend.execute_circuit(c)


def test_repeated_execute(backend, accelerators):
    c = Circuit(4, accelerators)
    thetas = np.random.random(4)
    c.add((gates.RY(i, t) for i, t in enumerate(thetas)))
    target_state = backend.execute_circuit(c).state(numpy=True)
    target_state = np.array(20 * [target_state])
    c.repeated_execution = True
    final_state = backend.execute_circuit(c, nshots=20)
    final_state = [backend.to_numpy(x) for x in final_state]
    backend.assert_allclose(final_state, target_state)


def test_final_state_property(backend):
    """Check accessing final state using the circuit's property."""
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1)])

    with pytest.raises(RuntimeError):
        final_state = c.final_state

    backend.execute_circuit(c)
    target_state = np.ones(4) / 2
    backend.assert_allclose(c.final_state, target_state)


def test_density_matrix_circuit(backend):
    from qibo.tests.utils import random_density_matrix
    theta = 0.1234
    initial_rho = random_density_matrix(3)

    c = Circuit(3, density_matrix=True)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.CNOT(0, 1))
    c.add(gates.H(2))
    final_rho = backend.execute_circuit(c, np.copy(initial_rho))

    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                     [0, 0, 0, 1], [0, 0, 1, 0]])
    m1 = np.kron(np.kron(h, h), np.eye(2))
    m2 = np.kron(cnot, np.eye(2))
    m3 = np.kron(np.eye(4), h)
    target_rho = m1.dot(initial_rho).dot(m1.T.conj())
    target_rho = m2.dot(target_rho).dot(m2.T.conj())
    target_rho = m3.dot(target_rho).dot(m3.T.conj())
    backend.assert_allclose(final_rho, target_rho)
