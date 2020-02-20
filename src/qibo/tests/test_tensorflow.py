"""
Testing tensorflow backend.
"""
import numpy as np
from qibo.models import Circuit
from qibo import gates
import pytest


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


def test_hadamard():
    """Check Hadamard gate is working properly."""
    c = Circuit(1)
    c.add(gates.H(0))
    final_state = c.execute()
    target_state = np.ones_like(final_state) / np.sqrt(2)
    np.testing.assert_allclose(final_state, target_state)


def test_flatten():
    """Check flatten gate is working properly."""
    target_state = np.ones(4) / 2.0
    c = Circuit(2)
    c.add(gates.Flatten(target_state))
    final_state = c.execute()
    np.testing.assert_allclose(final_state, target_state)


def test_xgate():
    """Check X gate is working properly."""
    c = Circuit(2)
    c.add(gates.X(0))
    final_state = c.execute()
    target_state = np.zeros_like(final_state)
    target_state[2] = 1.0
    np.testing.assert_allclose(final_state, target_state)


def test_cnot_no_effect():
    """Check CNOT gate is working properly on |00>."""
    c = Circuit(2)
    c.add(gates.CNOT(0, 1))
    final_state = c.execute()
    target_state = np.zeros_like(final_state)
    target_state[0] = 1.0
    np.testing.assert_allclose(final_state, target_state)


def test_cnot():
    """Check CNOT gate is working properly on |10>."""
    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.CNOT(0, 1))
    final_state = c.execute()
    target_state = np.zeros_like(final_state)
    target_state[3] = 1.0
    np.testing.assert_allclose(final_state, target_state)


def test_rz_no_effect():
    """Check RZ gate is working properly when qubit is on |0>."""
    c = Circuit(2)
    c.add(gates.RZ(0, 0.1234))
    final_state = c.execute()
    target_state = np.zeros_like(final_state)
    target_state[0] = 1.0
    np.testing.assert_allclose(final_state, target_state)


def test_rz_phase():
    """Check RZ gate is working properly when qubit is on |1>."""
    theta = 0.1234

    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.RZ(0, theta))
    final_state = c.execute()

    target_state = np.zeros_like(final_state)
    target_state[2] = np.exp(1j * np.pi * theta)
    np.testing.assert_allclose(final_state, target_state)


def test_rx():
    """Check RX gate is working properly."""
    theta = 0.1234

    c = Circuit(1)
    c.add(gates.H(0))
    c.add(gates.RX(0, theta))
    final_state = c.execute()

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
    final_state = c.execute()

    phase = np.exp(1j * np.pi * theta / 2.0)
    gate = phase * np.array([[phase.real, -phase.imag],
                             [phase.imag, phase.real]])
    target_state = gate.dot(np.ones(2)) / np.sqrt(2)
    np.testing.assert_allclose(final_state, target_state)


def test_crz():
    """Check CRZ gate is working properly on |11>."""
    theta = 0.1234

    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.CRZ(0, 1, theta))
    final_state = c.execute()

    phase = np.exp(1j * np.pi * theta)
    target_state = np.zeros_like(final_state)
    target_state[-1] = phase
    np.testing.assert_allclose(final_state, target_state)


def test_custom_circuit():
    """Check consistency between Circuit and custom circuits"""
    theta = 0.1234

    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.CRZ(0, 1, theta))
    r1 = c.execute()

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
