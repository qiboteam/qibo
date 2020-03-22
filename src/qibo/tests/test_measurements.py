import numpy as np
import pytest
from qibo import gates, models


def test_measurement_gate():
    """Check that measurement gate works when called on a state."""
    state = np.zeros(4)
    state[0] = 1
    state = state.reshape((2, 2))

    measurements = gates.M(0)(state, nshots=100).decimal_samples.numpy()
    target_measurements = np.zeros_like(measurements)
    assert measurements.shape == (100,)
    np.testing.assert_allclose(measurements, target_measurements)

    state[0], state[-1] = 0, 1
    measurements = gates.M(0)(state, nshots=100).decimal_samples.numpy()
    target_measurements = np.ones_like(measurements)
    assert measurements.shape == (100,)
    np.testing.assert_allclose(measurements, target_measurements)


def test_multiple_qubit_measurement_gate():
    """Check that multiple qubit measurement gate works when called on state."""
    state = np.zeros(4)
    state[0] = 0
    state[2] = 1
    state = state.reshape((2, 2))

    measurements = gates.M(0, 1)(state, nshots=100).decimal_samples.numpy()
    target_measurements = 2 * np.ones_like(measurements)
    assert measurements.shape == (100,)
    np.testing.assert_allclose(measurements, target_measurements)


def test_controlled_measurement_error():
    """Check that using `controlled_by` in measurements raises error."""
    with pytest.raises(NotImplementedError):
        m = gates.M(0).controlled_by(1)


def test_measurement_circuit():
    """Check that measurement gate works as part of circuit."""
    c = models.Circuit(2)
    c.add(gates.M(0))

    measurements = c(nshots=100).decimal_samples.numpy()
    target_measurements = np.zeros_like(measurements)
    assert measurements.shape == (100,)
    np.testing.assert_allclose(measurements, target_measurements)


def test_multiple_qubit_measurement_circuit():
    """Check that multiple measurement gates fuse correctly."""
    c = models.Circuit(2)
    c.add(gates.X(0))
    c.add(gates.M(0))
    c.add(gates.M(1))

    measurements = c(nshots=100).decimal_samples.numpy()
    target_measurements = 2 * np.ones_like(measurements)
    assert measurements.shape == (100,)
    np.testing.assert_allclose(measurements, target_measurements)

    final_state = c.final_state.numpy()
    target_state = np.zeros_like(final_state)
    target_state[2] = 1
    np.testing.assert_allclose(final_state, target_state)


def test_multiple_measurement_gates_circuit():
    """Check that measurement gates with different number of targets."""
    c = models.Circuit(4)
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.M(0, 1))
    c.add(gates.M(2))
    c.add(gates.X(3))

    measurements = c(nshots=100).decimal_samples.numpy()
    target_measurements = 3 * np.ones_like(measurements)
    assert measurements.shape == (100,)
    np.testing.assert_allclose(measurements, target_measurements)

    final_state = c.final_state.numpy()
    c = models.Circuit(4)
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.X(3))
    target_state = c().numpy()
    np.testing.assert_allclose(final_state, target_state)


def test_measurement_compiled_circuit():
    """Check that measurement gates work when compiling the circuit."""
    c = models.Circuit(2)
    c.add(gates.X(0))
    c.add(gates.M(0))
    c.add(gates.M(1))
    c.compile()

    measurements = c(nshots=100).decimal_samples.numpy()
    target_measurements = 2 * np.ones_like(measurements)
    assert measurements.shape == (100,)
    np.testing.assert_allclose(measurements, target_measurements)

    final_state = c.final_state.numpy()
    target_state = np.zeros_like(final_state)
    target_state[2] = 1
    np.testing.assert_allclose(final_state, target_state)