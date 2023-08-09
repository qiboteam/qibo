"""Test all methods defined in `qibo/core/circuit.py`."""
import numpy as np
import pytest
from qibo import gates, K
from qibo.models import Circuit


def test_circuit_init(backend, accelerators):
    c = Circuit(2, accelerators)
    assert c.param_tensor_types == K.tensor_types


def test_set_nqubits(backend):
    c = Circuit(4)
    c.add(gates.H(0))
    assert c.queue[0].nqubits == 4
    gate = gates.H(1)
    gate.nqubits = 3
    with pytest.raises(RuntimeError):
        c.add(gate)


@pytest.mark.parametrize("nqubits", [5, 6])
def test_circuit_add_layer(backend, nqubits, accelerators):
    c = Circuit(nqubits, accelerators)
    qubits = list(range(nqubits))
    pairs = [(2 * i, 2 * i + 1) for i in range(nqubits // 2)]
    params = nqubits * [0.1]
    c.add(gates.VariationalLayer(qubits, pairs, gates.RY, gates.CZ, params))
    assert len(c.queue) == nqubits // 2 + nqubits % 2
    for gate in c.queue:
        assert isinstance(gate, gates.Unitary)


def test_circuit_unitary(backend):
    from qibo import matrices
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.CNOT(0, 1))
    c.add(gates.X(0))
    c.add(gates.Y(1))
    h = np.kron(matrices.H, matrices.H)
    target_matrix = np.kron(matrices.X, matrices.Y) @ matrices.CNOT @ h
    K.assert_allclose(c.unitary(), target_matrix)


def test_circuit_unitary_bigger(backend):
    from qibo import matrices
    c = Circuit(4)
    c.add(gates.H(i) for i in range(4))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CZ(1, 2))
    c.add(gates.CNOT(0, 3))
    h = np.kron(matrices.H, matrices.H)
    h = np.kron(h, h)
    m1 = np.kron(matrices.CNOT, np.eye(4))
    m2 = np.kron(np.kron(np.eye(2), matrices.CZ), np.eye(2))
    m3 = np.kron(matrices.CNOT, np.eye(4)).reshape(8 * (2,))
    m3 = np.transpose(m3, [0, 2, 3, 1, 4, 6, 7, 5]).reshape((16, 16))
    target_matrix = m3 @ m2 @ m1 @ h
    K.assert_allclose(c.unitary(), target_matrix)

# :meth:`qibo.core.circuit.Circuit.fuse` is tested in `test_core_fusion.py`

def test_eager_execute(backend, accelerators):
    c = Circuit(4, accelerators)
    c.add((gates.H(i) for i in range(4)))
    target_state = np.ones(16) / 4.0
    K.assert_allclose(c(), target_state)


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
    r1 = c1.execute()

    # Run compiled circuit
    c2 = create_circuit()
    c2.compile()
    r2 = c2()
    init_state = c2.get_initial_state()
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
        final_state = c()


def test_repeated_execute(backend, accelerators):
    c = Circuit(4, accelerators)
    thetas = np.random.random(4)
    c.add((gates.RY(i, t) for i, t in enumerate(thetas)))
    c.repeated_execution = True
    target_state = np.array(20 * [c()])
    final_state = c(nshots=20)
    K.assert_allclose(final_state, target_state)


def test_final_state_property(backend):
    """Check accessing final state using the circuit's property."""
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1)])

    with pytest.raises(RuntimeError):
        final_state = c.final_state

    _ = c()
    target_state = np.ones(4) / 2
    K.assert_allclose(c.final_state, target_state)


def test_get_initial_state(backend):
    c = Circuit(2)
    final_state = c.get_initial_state()
    target_state = np.zeros(4)
    target_state[0] = 1
    K.assert_allclose(final_state, target_state)
    with pytest.raises(ValueError):
        state = c.get_initial_state(np.zeros(2**3))
    with pytest.raises(ValueError):
        final_state = c.get_initial_state(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        final_state = c.get_initial_state(np.zeros((2, 2, 2)))
    with pytest.raises(TypeError):
        final_state = c.get_initial_state(0)
    c = Circuit(2)
    c.check_initial_state_shape = False
    with pytest.raises(TypeError):
        final_state = c.get_initial_state(0)


def test_density_matrix_circuit(backend):
    from qibo.tests.utils import random_density_matrix
    theta = 0.1234
    initial_rho = random_density_matrix(3)

    c = Circuit(3, density_matrix=True)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.CNOT(0, 1))
    c.add(gates.H(2))
    final_rho = c(np.copy(initial_rho))

    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                     [0, 0, 0, 1], [0, 0, 1, 0]])
    m1 = np.kron(np.kron(h, h), np.eye(2))
    m2 = np.kron(cnot, np.eye(2))
    m3 = np.kron(np.eye(4), h)
    target_rho = m1.dot(initial_rho).dot(m1.T.conj())
    target_rho = m2.dot(target_rho).dot(m2.T.conj())
    target_rho = m3.dot(target_rho).dot(m3.T.conj())
    K.assert_allclose(final_rho, target_rho)


def test_density_matrix_circuit_initial_state(backend):
    from qibo.tests.utils import random_state
    initial_psi = random_state(3)
    c = Circuit(3, density_matrix=True)
    final_rho = c(np.copy(initial_psi))
    target_rho = np.outer(initial_psi, initial_psi.conj())
    K.assert_allclose(final_rho, target_rho)
    final_rho = c(initial_psi)
    K.assert_allclose(final_rho, target_rho)
