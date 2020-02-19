"""
Testing tensorflow backend.
"""
import numpy as np
from qibo import models
from qibo import gates


def test_hadamard():
    """Check Hadamard gate is working properly."""
    c = models.Circuit(1)
    c.add(gates.H(0))
    final_state = c.run()
    target_state = np.ones_like(final_state) / np.sqrt(2)
    np.testing.assert_allclose(final_state, target_state)


def test_flatten():
    """Check flatten gate is working properly."""
    target_state = np.ones(4) / 2.0
    c = models.Circuit(2)
    c.add(gates.Flatten(target_state))
    final_state = c.run()
    np.testing.assert_allclose(final_state, target_state)


def test_xgate():
    """Check X gate is working properly."""
    c = models.Circuit(2)
    c.add(gates.X(0))
    final_state = c.run()
    target_state = np.zeros_like(final_state)
    target_state[2] = 1.0
    np.testing.assert_allclose(final_state, target_state)


def test_rz_no_effect():
    """Check RZ gate is working properly."""
    c = models.Circuit(2)
    c.add(gates.RZ(0, 0.1234))
    final_state = c.run()
    target_state = np.zeros_like(final_state)
    target_state[0] = 1.0
    np.testing.assert_allclose(final_state, target_state)


def test_rz_phase():
    """Check RZ gate is working properly."""
    theta = 0.1234

    c = models.Circuit(2)
    c.add(gates.X(0))
    c.add(gates.RZ(0, theta))
    final_state = c.run()

    target_state = np.zeros_like(final_state)
    target_state[2] = np.exp(1j * theta)
    np.testing.assert_allclose(final_state, target_state)
