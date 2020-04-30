"""
Testing tensorflow backend.
"""
import numpy as np
import pytest
from qibo.models import Circuit
from qibo import gates

_EINSUM_BACKENDS = ["DefaultEinsum", "MatmulEinsum"]


def test_circuit_addition_result():
    """Check if circuit addition works properly on Tensorflow circuit."""
    c1 = Circuit(2)
    c1.add(gates.H(0))
    c1.add(gates.H(1))

    c2 = Circuit(2)
    c2.add(gates.CNOT(0, 1))

    c3 = c1 + c2

    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.CNOT(0, 1))

    np.testing.assert_allclose(c3.execute().numpy(), c.execute().numpy())


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_hadamard(einsum_choice):
    """Check Hadamard gate is working properly."""
    c = Circuit(2)
    c.add(gates.H(0).with_backend(einsum_choice))
    c.add(gates.H(1).with_backend(einsum_choice))
    final_state = c.execute().numpy()
    target_state = np.ones_like(final_state) / 2
    np.testing.assert_allclose(final_state, target_state)


def test_flatten():
    """Check flatten gate is working properly."""
    target_state = np.ones(4) / 2.0
    c = Circuit(2)
    c.add(gates.Flatten(target_state))
    final_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)


def test_xgate():
    """Check X gate is working properly."""
    c = Circuit(2)
    c.add(gates.X(0))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[2] = 1.0
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_multicontrol_xgate(einsum_choice):
    """Check that fallback method for X works for more than two controls."""
    c = Circuit(4)
    c.add(gates.X(0).with_backend(einsum_choice))
    c.add(gates.X(1).with_backend(einsum_choice))
    c.add(gates.X(2).with_backend(einsum_choice))
    c.add(gates.X(3).with_backend(einsum_choice).controlled_by(0, 1, 2))
    c.add(gates.X(0).with_backend(einsum_choice))
    c.add(gates.X(2).with_backend(einsum_choice))
    final_state = c.execute().numpy()

    c = Circuit(4)
    c.add(gates.X(1))
    c.add(gates.X(3))
    target_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)


def test_rz_phase0():
    """Check RZ gate is working properly when qubit is on |0>."""
    theta = 0.1234

    c = Circuit(2)
    c.add(gates.RZ(0, theta))
    final_state = c.execute().numpy()

    target_state = np.zeros_like(final_state)
    target_state[0] = np.exp(-1j * theta / 2.0)
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_rz_phase1(einsum_choice):
    """Check RZ gate is working properly when qubit is on |1>."""
    theta = 0.1234

    c = Circuit(2)
    c.add(gates.X(0).with_backend(einsum_choice))
    c.add(gates.RZ(0, theta).with_backend(einsum_choice))
    final_state = c.execute().numpy()

    target_state = np.zeros_like(final_state)
    target_state[2] = np.exp(1j * theta / 2.0)
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_rx(einsum_choice):
    """Check RX gate is working properly."""
    theta = 0.1234

    c = Circuit(1)
    c.add(gates.H(0).with_backend(einsum_choice))
    c.add(gates.RX(0, theta=theta).with_backend(einsum_choice))
    final_state = c.execute().numpy()

    phase = np.exp(1j * theta / 2.0)
    gate = np.array([[phase.real, -1j * phase.imag],
                    [-1j * phase.imag, phase.real]])
    target_state = gate.dot(np.ones(2)) / np.sqrt(2)
    np.testing.assert_allclose(final_state, target_state)


def test_ry():
    """Check RY gate is working properly."""
    theta = 0.1234

    c = Circuit(1)
    c.add(gates.H(0))
    c.add(gates.RY(0, theta))
    final_state = c.execute().numpy()

    phase = np.exp(1j * theta / 2.0)
    gate = np.array([[phase.real, -phase.imag],
                     [phase.imag, phase.real]])
    target_state = gate.dot(np.ones(2)) / np.sqrt(2)
    np.testing.assert_allclose(final_state, target_state)


def test_cnot_no_effect():
    """Check CNOT gate is working properly on |00>."""
    c = Circuit(2)
    c.add(gates.CNOT(0, 1))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[0] = 1.0
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_cnot(einsum_choice):
    """Check CNOT gate is working properly on |10>."""
    c = Circuit(2)
    c.add(gates.X(0).with_backend(einsum_choice))
    c.add(gates.CNOT(0, 1).with_backend(einsum_choice))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[3] = 1.0
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_czpow(einsum_choice):
    """Check CZPow gate is working properly on |11>."""
    theta = 0.1234

    c = Circuit(2)
    c.add(gates.X(0).with_backend(einsum_choice))
    c.add(gates.X(1).with_backend(einsum_choice))
    c.add(gates.CZPow(0, 1, theta).with_backend(einsum_choice))
    final_state = c.execute().numpy()

    phase = np.exp(1j * theta)
    target_state = np.zeros_like(final_state)
    target_state[3] = phase
    np.testing.assert_allclose(final_state, target_state)


def test_doubly_controlled_by_rx_no_effect():
    theta = 0.1234

    c = Circuit(3)
    c.add(gates.X(0))
    c.add(gates.RX(2, theta).controlled_by(0, 1))
    c.add(gates.X(0))
    final_state = c.execute().numpy()

    target_state = np.zeros_like(final_state)
    target_state[0] = 1.0

    np.testing.assert_allclose(final_state, target_state)


def test_doubly_controlled_by_rx():
    theta = 0.1234

    c = Circuit(3)
    c.add(gates.RX(2, theta))
    target_state = c.execute().numpy()

    c = Circuit(3)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.RX(2, theta).controlled_by(0, 1))
    c.add(gates.X(0))
    c.add(gates.X(1))
    final_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)


def test_swap():
    """Check SWAP gate is working properly on |01>."""
    c = Circuit(2)
    c.add(gates.X(1))
    c.add(gates.SWAP(0, 1))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[2] = 1.0
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_multiple_swap(einsum_choice):
    """Check SWAP gate is working properly when called multiple times."""
    c = Circuit(4)
    c.add(gates.X(0).with_backend(einsum_choice))
    c.add(gates.X(2).with_backend(einsum_choice))
    c.add(gates.SWAP(0, 1).with_backend(einsum_choice))
    c.add(gates.SWAP(2, 3).with_backend(einsum_choice))
    final_state = c.execute().numpy()

    c = Circuit(4)
    c.add(gates.X(1))
    c.add(gates.X(3))
    target_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_controlled_by_swap(einsum_choice):
    """Check controlled SWAP using controlled by."""
    c = Circuit(3)
    c.add(gates.SWAP(1, 2).controlled_by(0).with_backend(einsum_choice))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[0] = 1.0
    np.testing.assert_allclose(final_state, target_state)

    c = Circuit(3)
    c.add(gates.X(0).with_backend(einsum_choice))
    c.add(gates.SWAP(1, 2).controlled_by(0).with_backend(einsum_choice))
    c.add(gates.X(0).with_backend(einsum_choice))
    final_state = c.execute().numpy()
    c = Circuit(3)
    c.add(gates.SWAP(1, 2))
    target_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)


def test_doubly_controlled_by_swap():
    """Check controlled SWAP using controlled by two qubits."""
    c = Circuit(4)
    c.add(gates.X(0))
    c.add(gates.SWAP(1, 2).controlled_by(0, 3))
    c.add(gates.X(0))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[0] = 1.0
    np.testing.assert_allclose(final_state, target_state)

    c = Circuit(4)
    c.add(gates.X(0))
    c.add(gates.X(3))
    c.add(gates.SWAP(1, 2).controlled_by(0, 3))
    c.add(gates.X(0))
    c.add(gates.X(3))
    final_state = c.execute().numpy()
    c = Circuit(4)
    c.add(gates.SWAP(1, 2))
    target_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)


def test_toffoli_no_effect():
    """Check Toffoli gate is working properly on |010>."""
    c = Circuit(3)
    c.add(gates.X(1))
    c.add(gates.TOFFOLI(0, 1, 2))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[2] = 1.0
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_toffoli(einsum_choice):
    """Check Toffoli gate is working properly on |110>."""
    c = Circuit(3)
    c.add(gates.X(0).with_backend(einsum_choice))
    c.add(gates.X(1).with_backend(einsum_choice))
    c.add(gates.TOFFOLI(0, 1, 2).with_backend(einsum_choice))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[-1] = 1.0
    np.testing.assert_allclose(final_state, target_state)


def test_unitary_common_gates():
    """Check that `Unitary` gate can create common gates."""
    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.H(1))
    target_state = c.execute().numpy()

    c = Circuit(2)
    c.add(gates.Unitary(np.array([[0, 1], [1, 0]]), 0))
    c.add(gates.Unitary(np.array([[1, 1], [1, -1]]) / np.sqrt(2), 1))
    final_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_unitary_random_gate(einsum_choice):
    """Check that `Unitary` gate can apply random matrices."""
    init_state = np.ones(4) / 2.0
    matrix = np.random.random([4, 4])
    target_state = matrix.dot(init_state)

    c = Circuit(2)
    c.add(gates.H(0).with_backend(einsum_choice))
    c.add(gates.H(1).with_backend(einsum_choice))
    c.add(gates.Unitary(matrix, 0, 1, name="random").with_backend(einsum_choice))
    final_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)


def test_unitary_controlled_by():
    """Check that `controlled_by` works as expected with `Unitary`."""
    matrix = np.random.random([2, 2])

    # No effect
    c = Circuit(2)
    c.add(gates.Unitary(matrix, 1).controlled_by(0))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[0] = 1.0
    np.testing.assert_allclose(final_state, target_state)

    # With effect
    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.Unitary(matrix, 1).controlled_by(0))
    c.add(gates.X(0))
    final_state = c.execute().numpy()

    c = Circuit(2)
    c.add(gates.Unitary(matrix, 1))
    target_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)


def test_custom_circuit():
    """Check consistency between Circuit and custom circuits"""
    theta = 0.1234

    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.CZPow(0, 1, theta))
    r1 = c.execute().numpy()

    # custom circuit
    def custom_circuit(initial_state, theta):
        l1 = gates.X(0)(initial_state)
        l2 = gates.X(1)(l1)
        o = gates.CZPow(0, 1, theta)(l2)
        return o

    init = c._default_initial_state()
    r2 = custom_circuit(init, theta).numpy().ravel()
    np.testing.assert_allclose(r1, r2)

    import tensorflow as tf
    tf_custom_circuit = tf.function(custom_circuit)
    r3 = tf_custom_circuit(init, theta).numpy().ravel()
    np.testing.assert_allclose(r2, r3)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_compiled_circuit(einsum_choice):
    """Check that compiling with `Circuit.compile` does not break results."""
    def create_circuit(theta = 0.1234):
        c = Circuit(2)
        c.add(gates.X(0).with_backend(einsum_choice))
        c.add(gates.X(1).with_backend(einsum_choice))
        c.add(gates.CZPow(0, 1, theta).with_backend(einsum_choice))
        return c

    # Run eager circuit
    c1 = create_circuit()
    r1 = c1.execute().numpy()

    # Run compiled circuit
    c2 = create_circuit()
    c2.compile()
    r2 = c2.execute().numpy()

    np.testing.assert_allclose(r1, r2)


def test_circuit_custom_compilation():
    theta = 0.1234
    init_state = np.ones(4) / 2.0

    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.CZPow(0, 1, theta))
    r1 = c.execute(init_state).numpy()

    def run_circuit(initial_state):
        c = Circuit(2)
        c.add(gates.X(0))
        c.add(gates.X(1))
        c.add(gates.CZPow(0, 1, theta))
        return c.execute(initial_state)

    import tensorflow as tf
    compiled_circuit = tf.function(run_circuit)
    init_state = tf.cast(init_state.reshape((2, 2)), dtype=c.dtype)
    r2 = compiled_circuit(init_state)

    np.testing.assert_allclose(r1, r2)


def test_variable_theta():
    """Check that parametrized gates accept `tf.Variable` parameters."""
    import tensorflow as tf
    from qibo.config import DTYPE
    theta1 = tf.Variable(0.1234, dtype=DTYPE)
    theta2 = tf.Variable(0.4321, dtype=DTYPE)

    cvar = Circuit(2)
    cvar.add(gates.RX(0, theta1))
    cvar.add(gates.RY(1, theta2))
    final_state = cvar().numpy()

    c = Circuit(2)
    c.add(gates.RX(0, 0.1234))
    c.add(gates.RY(1, 0.4321))
    target_state = c().numpy()

    np.testing.assert_allclose(final_state, target_state)


def test_variable_backpropagation():
    """Check that backpropagation works when using `tf.Variable` parameters."""
    import tensorflow as tf
    from qibo.config import DTYPE
    theta = tf.Variable(0.1234, dtype=DTYPE)

    # TODO: Fix parametrized gates so that `Circuit` can be defined outside
    # of the gradient tape
    with tf.GradientTape() as tape:
        c = Circuit(1)
        c.add(gates.X(0))
        c.add(gates.RZ(0, theta).with_backend("MatmulEinsum"))
        loss = tf.math.real(c()[-1])
    grad = tape.gradient(loss, theta)

    target_loss = np.cos(theta.numpy() / 2.0)
    np.testing.assert_allclose(loss.numpy(), target_loss)

    target_grad = - np.sin(theta.numpy() / 2.0) / 2.0
    np.testing.assert_allclose(grad.numpy(), target_grad)


def test_two_variables_backpropagation():
    """Check that backpropagation works when using `tf.Variable` parameters."""
    import tensorflow as tf
    from qibo.config import DTYPE
    theta = tf.Variable([0.1234, 0.4321], dtype=DTYPE)

    # TODO: Fix parametrized gates so that `Circuit` can be defined outside
    # of the gradient tape
    with tf.GradientTape() as tape:
        c = Circuit(2)
        c.add(gates.RX(0, theta[0]).with_backend("MatmulEinsum"))
        c.add(gates.RY(1, theta[1]).with_backend("MatmulEinsum"))
        loss = tf.math.real(c()[0])
    grad = tape.gradient(loss, theta)

    t = np.array([0.1234, 0.4321]) / 2.0
    target_loss = np.cos(t[0]) * np.cos(t[1])
    np.testing.assert_allclose(loss.numpy(), target_loss)

    target_grad1 = - np.sin(t[0]) * np.cos(t[1])
    target_grad2 = - np.cos(t[0]) * np.sin(t[1])
    target_grad = np.array([target_grad1, target_grad2]) / 2.0
    np.testing.assert_allclose(grad.numpy(), target_grad)


def test_circuit_copy():
    """Check that circuit copy execution is equivalent to original circuit."""
    theta = 0.1234

    c1 = Circuit(2)
    c1.add([gates.X(0), gates.X(1), gates.CZPow(0, 1, theta)])
    c2 = c1.copy()

    target_state = c1.execute().numpy()
    final_state = c2.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)


def test_circuit_with_noise_gates():
    """Check that ``circuit.with_noise()`` adds the proper noise channels."""
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])
    noisy_c = c.with_noise((0.1, 0.2, 0.3))

    assert noisy_c.depth == 9
    for i in [1, 2, 4, 5, 7, 8]:
        assert isinstance(noisy_c.queue[i], gates.NoiseChannel)


def test_circuit_with_noise_execution():
    """Check ``circuit.with_noise()`` execution."""
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1)])
    noisy_c = c.with_noise((0.1, 0.2, 0.3))

    target_c = Circuit(2)
    target_c.add(gates.H(0))
    target_c.add(gates.NoiseChannel(0, 0.1, 0.2, 0.3))
    target_c.add(gates.NoiseChannel(1, 0.1, 0.2, 0.3))
    target_c.add(gates.H(1))
    target_c.add(gates.NoiseChannel(0, 0.1, 0.2, 0.3))
    target_c.add(gates.NoiseChannel(1, 0.1, 0.2, 0.3))

    final_state = noisy_c().numpy()
    target_state = target_c().numpy()
    np.testing.assert_allclose(target_state, final_state)


def test_circuit_with_noise_with_measurements():
    """Check ``circuit.with_noise() when using measurement noise."""
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1)])
    c.add(gates.M(0))
    noisy_c = c.with_noise(3 * (0.1,), measurement_noise = (0.3, 0.0, 0.0))

    target_c = Circuit(2)
    target_c.add(gates.H(0))
    target_c.add(gates.NoiseChannel(0, 0.1, 0.1, 0.1))
    target_c.add(gates.NoiseChannel(1, 0.1, 0.1, 0.1))
    target_c.add(gates.H(1))
    target_c.add(gates.NoiseChannel(0, 0.3, 0.0, 0.0))
    target_c.add(gates.NoiseChannel(1, 0.1, 0.1, 0.1))

    final_state = noisy_c().numpy()
    target_state = target_c().numpy()
    np.testing.assert_allclose(target_state, final_state)


def test_circuit_with_noise_noise_map():
    """Check ``circuit.with_noise() when giving noise map."""
    noise_map = {0: (0.1, 0.2, 0.1), 1: (0.2, 0.3, 0.0),
                 2: (0.0, 0.0, 0.0)}

    c = Circuit(3)
    c.add([gates.H(0), gates.H(1), gates.X(2)])
    c.add(gates.M(2))
    noisy_c = c.with_noise(noise_map, measurement_noise = (0.3, 0.0, 0.0))

    target_c = Circuit(3)
    target_c.add(gates.H(0))
    target_c.add(gates.NoiseChannel(0, 0.1, 0.2, 0.1))
    target_c.add(gates.NoiseChannel(1, 0.2, 0.3, 0.0))
    target_c.add(gates.H(1))
    target_c.add(gates.NoiseChannel(0, 0.1, 0.2, 0.1))
    target_c.add(gates.NoiseChannel(1, 0.2, 0.3, 0.0))
    target_c.add(gates.X(2))
    target_c.add(gates.NoiseChannel(0, 0.1, 0.2, 0.1))
    target_c.add(gates.NoiseChannel(1, 0.2, 0.3, 0.0))
    target_c.add(gates.NoiseChannel(2, 0.3, 0.0, 0.0))

    final_state = noisy_c().numpy()
    target_state = target_c().numpy()
    np.testing.assert_allclose(target_state, final_state)


def test_circuit_with_noise_noise_map_exceptions():
    """Check that proper exceptions are raised when noise map is invalid."""
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1)])
    with pytest.raises(ValueError):
        noisy_c = c.with_noise((0.2, 0.3))
    with pytest.raises(ValueError):
        noisy_c = c.with_noise({0: (0.2, 0.3, 0.1), 1: (0.3, 0.1)})
    with pytest.raises(ValueError):
        noisy_c = c.with_noise({0: (0.2, 0.3, 0.1)})
    with pytest.raises(TypeError):
        noisy_c = c.with_noise({0, 1})
    with pytest.raises(ValueError):
        noisy_c = c.with_noise((0.2, 0.3, 0.1),
                               measurement_noise=(0.5, 0.0, 0.0))


def test_circuit_with_noise_exception():
    """Check that calling ``with_noise`` in a noisy circuit raises error."""
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1), gates.NoiseChannel(0, px=0.2)])
    with pytest.raises(ValueError):
        noisy_c = c.with_noise((0.2, 0.3, 0.0))
