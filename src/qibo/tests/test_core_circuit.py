import numpy as np
import pytest
import qibo
from qibo import gates
from qibo.models import Circuit


try:
    import tensorflow as tf
    BACKENDS = ["custom", "defaulteinsum", "matmuleinsum",
                "numpy_defaulteinsum", "numpy_matmuleinsum"]
except ModuleNotFoundError: # pragma: no cover
    BACKENDS = ["numpy_defaulteinsum", "numpy_matmuleinsum"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_circuit_init(backend, accelerators=None):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(2, accelerators)
    if "numpy" in backend:
        assert c.param_tensor_types == (np.ndarray,)
    else:
        assert c.param_tensor_types == (np.ndarray, tf.Tensor, tf.Variable)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("nqubits", [5, 6])
def test_circuit_add_layer(backend, nqubits, accelerators=None):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(nqubits, accelerators)
    qubits = list(range(nqubits))
    pairs = [(2 * i, 2 * i + 1) for i in range(nqubits // 2)]
    params = nqubits * [0.1]
    c.add(gates.VariationalLayer(qubits, pairs, gates.RY, gates.CZ, params))
    assert len(c.queue) == nqubits // 2 + nqubits % 2
    if backend == "custom":
        target_gate_cls = gates.Unitary
    else:
        from qibo.core.gates import Unitary as target_gate_cls
    for gate in c.queue:
        assert isinstance(gate, target_gate_cls)
    qibo.set_backend(original_backend)

# TODO: Test `_fuse_copy`
# TODO: Test `fuse`

@pytest.mark.parametrize("backend", BACKENDS)
def test_eager_execute(backend, accelerators=None):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(4, accelerators)
    c.add((gates.H(i) for i in range(4)))
    target_state = np.ones(16) / 4.0
    np.testing.assert_allclose(c(), target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", BACKENDS)
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
    c2 = create_circuit()
    if backend == "custom":
        with pytest.raises(RuntimeError):
            c2.compile()
    else:
        c2.compile()
        r2 = c2()
        init_state = c2.get_initial_state()
        r3, _ = c2._execute_for_compile(np.reshape(init_state, (2, 2)))
        r3 = np.reshape(r3, (4,))
        np.testing.assert_allclose(r1, r2)
        np.testing.assert_allclose(r1, r3)
    qibo.set_backend(original_backend)


def test_compiling_twice_exception():
    """Check that compiling a circuit a second time raises error."""
    original_backend = qibo.get_backend()
    qibo.set_backend("matmuleinsum")
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1)])
    c.compile()
    with pytest.raises(RuntimeError):
        c.compile()
    qibo.set_backend(original_backend)

# TODO: Test circuit execution with measurements
# TODO: Test compiled circuit execution with measurements

@pytest.mark.linux
@pytest.mark.parametrize("backend", BACKENDS)
def test_memory_error(backend, accelerators=None):
    """Check that ``RuntimeError`` is raised if device runs out of memory."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(40, accelerators)
    c.add((gates.H(i) for i in range(0, 40, 5)))
    with pytest.raises(RuntimeError):
        final_state = c()
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_repeated_execute(backend, accelerators=None):
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_repeated_execute_with_noise_channel(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    thetas = np.random.random(4)
    c = Circuit(4)
    c.add((gates.RY(i, t) for i, t in enumerate(thetas)))

    c.add((gates.PauliNoiseChannel(i, px=0.1, py=0.2, pz=0.3, seed=1234)
          for i in range(4)))
    final_state = c(nshots=20)

    np.random.seed(1234)
    target_state = []
    for _ in range(20):
        noiseless_c = Circuit(4)
        noiseless_c.add((gates.RY(i, t) for i, t in enumerate(thetas)))
        for i in range(4):
            if np.random.random() < 0.1:
                noiseless_c.add(gates.X(i))
            if np.random.random() < 0.2:
                noiseless_c.add(gates.Y(i))
            if np.random.random() < 0.3:
                noiseless_c.add(gates.Z(i))
        target_state.append(noiseless_c())
    target_state = np.stack(target_state)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend"), BACKENDS)
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


@pytest.mark.parametrize("backend", BACKENDS)
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

# TODO: Test `DensityMatrixCircuit`
