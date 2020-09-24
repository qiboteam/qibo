"""
Testing Tensorflow circuits.
"""
import numpy as np
import pytest
import qibo
from qibo import gates
from qibo.models import Circuit

_BACKENDS = ["custom", "defaulteinsum", "matmuleinsum"]
_DEVICE_BACKENDS = [("custom", None), ("matmuleinsum", None),
                    ("custom", {"/GPU:0": 1, "/GPU:1": 1})]


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_circuit_addition_result(backend, accelerators):
    """Check if circuit addition works properly on Tensorflow circuit."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c1 = Circuit(2, accelerators)
    c1.add(gates.H(0))
    c1.add(gates.H(1))

    c2 = Circuit(2, accelerators)
    c2.add(gates.CNOT(0, 1))

    c3 = c1 + c2

    c = Circuit(2, accelerators)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.CNOT(0, 1))

    np.testing.assert_allclose(c3.execute().numpy(), c.execute().numpy())
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_custom_circuit(backend):
    """Check consistency between Circuit and custom circuits"""
    import tensorflow as tf
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.CU1(0, 1, theta))
    r1 = c.execute().numpy()

    # custom circuit
    def custom_circuit(initial_state, theta):
        l1 = gates.X(0)(initial_state)
        l2 = gates.X(1)(l1)
        o = gates.CU1(0, 1, theta)(l2)
        return o

    init2 = c._default_initial_state()
    init3 = c._default_initial_state()
    if backend != "custom":
        init2 = tf.reshape(init2, (2, 2))
        init3 = tf.reshape(init3, (2, 2))

    r2 = custom_circuit(init2, theta).numpy().ravel()
    np.testing.assert_allclose(r1, r2)

    tf_custom_circuit = tf.function(custom_circuit)
    if backend == "custom":
        with pytest.raises(NotImplementedError):
            r3 = tf_custom_circuit(init3, theta).numpy().ravel()
    else:
        r3 = tf_custom_circuit(init3, theta).numpy().ravel()
        np.testing.assert_allclose(r2, r3)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_compiled_circuit(backend):
    """Check that compiling with `Circuit.compile` does not break results."""
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
    r1 = c1.execute().numpy()

    # Run compiled circuit
    c2 = create_circuit()
    if backend == "custom":
        with pytest.raises(RuntimeError):
            c2.compile()
    else:
        c2.compile()
        r2 = c2.execute().numpy()
        init_state = c2._default_initial_state()
        r3, _ = c2._execute_for_compile(init_state.numpy().reshape((2, 2)))
        r3 = r3.numpy().ravel()
        np.testing.assert_allclose(r1, r2)
        np.testing.assert_allclose(r1, r3)
    qibo.set_backend(original_backend)


def test_compiling_twice_exception():
    """Check that compiling a circuit a second time raises error."""
    from qibo.tensorflow import gates
    original_backend = qibo.get_backend()
    qibo.set_backend("matmuleinsum")
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1)])
    c.compile()
    with pytest.raises(RuntimeError):
        c.compile()
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_circuit_custom_compilation(backend):
    import tensorflow as tf
    from qibo.config import DTYPES
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    init_state = tf.cast(np.ones(4) / 2.0, dtype=DTYPES.get('DTYPECPX'))

    def run_circuit(initial_state):
        c = Circuit(2)
        c.add(gates.X(0))
        c.add(gates.X(1))
        c.add(gates.CU1(0, 1, theta))
        return c.execute(initial_state)

    r1 = run_circuit(init_state).numpy()
    compiled_circuit = tf.function(run_circuit)
    if backend == "custom":
        with pytest.raises(NotImplementedError):
            r2 = compiled_circuit(init_state)
    else:
        r2 = compiled_circuit(init_state)
        np.testing.assert_allclose(r1, r2)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_bad_initial_state(backend, accelerators):
    """Check that errors are raised when bad initial state is passed."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    import tensorflow as tf
    c = Circuit(2, accelerators)
    c.add([gates.H(0), gates.H(1)])
    with pytest.raises(ValueError):
        final_state = c(initial_state=np.zeros(2**3))
    with pytest.raises(ValueError):
        final_state = c(initial_state=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        final_state = c(initial_state=np.zeros((2, 2, 2)))
    with pytest.raises(TypeError):
        final_state = c(initial_state=0)
    c = Circuit(2, accelerators)
    c.check_initial_state_shape = False
    with pytest.raises(TypeError):
        final_state = c(initial_state=0)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_final_state_property(backend, accelerators):
    """Check accessing final state using the circuit's property."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    import tensorflow as tf
    c = Circuit(2, accelerators)
    c.add([gates.H(0), gates.H(1)])

    with pytest.raises(RuntimeError):
        final_state = c.final_state

    _ = c()
    final_state = c.final_state.numpy()
    target_state = np.ones(4) / 2
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
@pytest.mark.parametrize("deep", [False, True])
def test_circuit_copy(backend, accelerators, deep):
    """Check that circuit copy execution is equivalent to original circuit."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234

    c1 = Circuit(2, accelerators)
    c1.add([gates.X(0), gates.X(1), gates.CU1(0, 1, theta)])
    if not deep and accelerators is not None:
        with pytest.raises(ValueError):
            c2 = c1.copy(deep)
    else:
        c2 = c1.copy(deep)
        target_state = c1.execute().numpy()
        final_state = c2.execute().numpy()
        np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.linux
@pytest.mark.parametrize("accelerators", [None, {"/GPU:0": 2}])
def test_memory_error(accelerators):
    """Check that ``RuntimeError`` is raised if device runs out of memory."""
    c = Circuit(40, accelerators=accelerators)
    c.add((gates.H(i) for i in range(0, 40, 5)))
    with pytest.raises(RuntimeError):
        final_state = c()
