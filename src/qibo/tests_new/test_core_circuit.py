"""Test all methods defined in `qibo/core/circuit.py`."""
import numpy as np
import pytest
import qibo
from qibo import gates
from qibo.models import Circuit


def test_circuit_init(backend, accelerators):
    from qibo import K
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(2, accelerators)
    assert c.param_tensor_types == K.tensor_types
    qibo.set_backend(original_backend)


def test_set_nqubits(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(4)
    c.add(gates.H(0))
    assert c.queue[0].nqubits == 4
    gate = gates.H(1)
    gate.nqubits = 3
    with pytest.raises(RuntimeError):
        c.add(gate)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("nqubits", [5, 6])
def test_circuit_add_layer(backend, nqubits, accelerators):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(nqubits, accelerators)
    qubits = list(range(nqubits))
    pairs = [(2 * i, 2 * i + 1) for i in range(nqubits // 2)]
    params = nqubits * [0.1]
    c.add(gates.VariationalLayer(qubits, pairs, gates.RY, gates.CZ, params))
    assert len(c.queue) == nqubits // 2 + nqubits % 2
    for gate in c.queue:
        assert isinstance(gate, gates.Unitary)
    qibo.set_backend(original_backend)

# TODO: Test `_fuse_copy`
# TODO: Test `fuse`

def test_eager_execute(backend, accelerators):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(4, accelerators)
    c.add((gates.H(i) for i in range(4)))
    target_state = np.ones(16) / 4.0
    np.testing.assert_allclose(c(), target_state)
    qibo.set_backend(original_backend)


def test_compiled_execute(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
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
    from qibo import K
    c2 = create_circuit()
    if K.op is not None:
        with pytest.raises(RuntimeError):
            c2.compile()
    else:
        c2.compile()
        r2 = c2()
        init_state = c2.get_initial_state()
        np.testing.assert_allclose(r1, r2)
    qibo.set_backend(original_backend)


def test_compiling_twice_exception():
    """Check that compiling a circuit a second time raises error."""
    from qibo import K
    original_backend = qibo.get_backend()
    if "tensorflow" not in K.available_backends: # pragma: no cover
        pytest.skip("Skipping compilation test because Tensorflow is not available.")
    qibo.set_backend("tensorflow")
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1)])
    c.compile()
    with pytest.raises(RuntimeError):
        c.compile()
    qibo.set_backend(original_backend)

# TODO: Test circuit execution with measurements
# TODO: Test compiled circuit execution with measurements

@pytest.mark.linux
def test_memory_error(backend, accelerators):
    """Check that ``RuntimeError`` is raised if device runs out of memory."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(40, accelerators)
    c.add((gates.H(i) for i in range(0, 40, 5)))
    with pytest.raises(RuntimeError):
        final_state = c()
    qibo.set_backend(original_backend)


def test_repeated_execute(backend, accelerators):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(4, accelerators)
    thetas = np.random.random(4)
    c.add((gates.RY(i, t) for i, t in enumerate(thetas)))
    c.repeated_execution = True
    target_state = np.array(20 * [c()])
    final_state = c(nshots=20)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_final_state_property(backend):
    """Check accessing final state using the circuit's property."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1)])

    with pytest.raises(RuntimeError):
        final_state = c.final_state

    _ = c()
    target_state = np.ones(4) / 2
    np.testing.assert_allclose(c.final_state, target_state)
    qibo.set_backend(original_backend)


def test_get_initial_state(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(2)
    final_state = c.get_initial_state()
    target_state = np.zeros(4)
    target_state[0] = 1
    np.testing.assert_allclose(final_state, target_state)
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
    qibo.set_backend(original_backend)


def test_density_matrix_circuit(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    from qibo.tests_new.test_core_gates import random_density_matrix
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
    np.testing.assert_allclose(final_rho, target_rho)
    qibo.set_backend(original_backend)


def test_density_matrix_circuit_initial_state(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    from qibo.tests_new.test_core_gates import random_state
    initial_psi = random_state(3)
    c = Circuit(3, density_matrix=True)
    final_rho = c(np.copy(initial_psi))
    target_rho = np.outer(initial_psi, initial_psi.conj())
    np.testing.assert_allclose(final_rho, target_rho)
    final_rho = c(initial_psi)
    np.testing.assert_allclose(final_rho, target_rho)
    qibo.set_backend(original_backend)
