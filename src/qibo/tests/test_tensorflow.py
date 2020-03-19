"""
Testing tensorflow backend.
"""
import numpy as np
from qibo.models import Circuit
from qibo import gates


def test_circuit_sanity():
    """Check if the number of qbits is preserved."""
    c = Circuit(2)
    assert c.nqubits == 2
    assert c.size == 2


def test_circuit_add():
    """Check if circuit depth increases with the add method."""
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.CNOT(0, 1))
    assert c.depth == 3


def test_circuit_addition():
    """Check if circuit addition increases depth."""
    c1 = Circuit(2)
    c1.add(gates.H(0))
    c1.add(gates.H(1))
    assert c1.depth == 2

    c2 = Circuit(2)
    c2.add(gates.CNOT(0, 1))
    assert c2.depth == 1

    c3 = c1 + c2
    assert c3.depth == 3


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


def test_hadamard():
    """Check Hadamard gate is working properly."""
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))
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


def test_multicontrol_xgate():
    """Check that fallback method for X works for more than two controls."""
    c = Circuit(4)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.X(3).controlled_by(0, 1, 2))
    c.add(gates.X(0))
    c.add(gates.X(2))
    final_state = c.execute().numpy()

    c = Circuit(4)
    c.add(gates.X(1))
    c.add(gates.X(3))
    target_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)


def test_rz_no_effect():
    """Check RZ gate is working properly when qubit is on |0>."""
    c = Circuit(2)
    c.add(gates.RZ(0, 0.1234))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[0] = 1.0
    np.testing.assert_allclose(final_state, target_state)


def test_rz_phase():
    """Check RZ gate is working properly when qubit is on |1>."""
    theta = 0.1234

    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.RZ(0, theta))
    final_state = c.execute().numpy()

    target_state = np.zeros_like(final_state)
    target_state[2] = np.exp(1j * np.pi * theta)
    np.testing.assert_allclose(final_state, target_state)


def test_rx():
    """Check RX gate is working properly."""
    theta = 0.1234

    c = Circuit(1)
    c.add(gates.H(0))
    c.add(gates.RX(0, theta))
    final_state = c.execute().numpy()

    phase = np.exp(1j * np.pi * theta / 2.0)
    gate = phase * np.array([[phase.real, -1j * phase.imag],
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

    phase = np.exp(1j * np.pi * theta / 2.0)
    gate = phase * np.array([[phase.real, -phase.imag],
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


def test_cnot():
    """Check CNOT gate is working properly on |10>."""
    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.CNOT(0, 1))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[3] = 1.0
    np.testing.assert_allclose(final_state, target_state)


def test_crz():
    """Check CRZ gate is working properly on |11>."""
    theta = 0.1234

    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.CRZ(0, 1, theta))
    final_state = c.execute().numpy()

    phase = np.exp(1j * np.pi * theta)
    target_state = np.zeros_like(final_state)
    target_state[-1] = phase
    np.testing.assert_allclose(final_state, target_state)


def test_controlled_by_rz():
    """Check RZ.controlled_by falls back to CRZ."""
    theta = 0.1234

    c = Circuit(2)
    c.add(gates.CRZ(0, 1, theta))
    print(c.queue)
    target_state = c.execute().numpy()

    c = Circuit(2)
    c.add(gates.RZ(1, theta).controlled_by(0))
    print(c.queue)
    final_state = c.execute().numpy()

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


def test_multiple_swap():
    """Check SWAP gate is working properly when called multiple times."""
    c = Circuit(4)
    c.add(gates.X(0))
    c.add(gates.X(2))
    c.add(gates.SWAP(0, 1))
    c.add(gates.SWAP(2, 3))
    final_state = c.execute().numpy()

    c = Circuit(4)
    c.add(gates.X(1))
    c.add(gates.X(3))
    target_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)


def test_controlled_by_swap():
    """Check controlled SWAP using controlled by."""
    c = Circuit(3)
    c.add(gates.SWAP(1, 2).controlled_by(0))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[0] = 1.0
    np.testing.assert_allclose(final_state, target_state)

    c = Circuit(3)
    c.add(gates.X(0))
    c.add(gates.SWAP(1, 2).controlled_by(0))
    c.add(gates.X(0))
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


def test_toffoli():
    """Check Toffoli gate is working properly on |110>."""
    c = Circuit(3)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.TOFFOLI(0, 1, 2))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[-1] = 1.0
    np.testing.assert_allclose(final_state, target_state)


def test_custom_circuit():
    """Check consistency between Circuit and custom circuits"""
    theta = 0.1234

    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.CRZ(0, 1, theta))
    r1 = c.execute().numpy()

    # custom circuit
    def custom_circuit(initial_state, theta):
        l1 = gates.X(0)(initial_state)
        l2 = gates.X(1)(l1)
        o = gates.CRZ(0, 1, theta)(l2)
        return o

    init = c._default_initial_state()
    r2 = custom_circuit(init, theta).numpy().ravel()
    np.testing.assert_allclose(r1, r2)

    import tensorflow as tf
    tf_custom_circuit = tf.function(custom_circuit)
    r3 = tf_custom_circuit(init, theta).numpy().ravel()
    np.testing.assert_allclose(r2, r3)


def test_compiled_circuit():
    """Check that compiling with `Circuit.compile` does not break results."""
    def create_circuit(theta = 0.1234):
        c = Circuit(2)
        c.add(gates.X(0))
        c.add(gates.X(1))
        c.add(gates.CRZ(0, 1, theta))
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
    c.add(gates.CRZ(0, 1, theta))
    r1 = c.execute(init_state).numpy()

    def run_circuit(initial_state):
        c = Circuit(2)
        c.add(gates.X(0))
        c.add(gates.X(1))
        c.add(gates.CRZ(0, 1, theta))
        return c.execute(initial_state)

    import tensorflow as tf
    compiled_circuit = tf.function(run_circuit)
    r2 = compiled_circuit(init_state)

    np.testing.assert_allclose(r1, r2)