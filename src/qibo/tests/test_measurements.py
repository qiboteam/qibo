import numpy as np
import pytest
from qibo import gates, models


def test_measurement_gate():
    """Check that measurement gate works when called on a state."""
    state = np.zeros(4)
    state[0] = 1
    state = state.reshape((2, 2))

    measurements = gates.M(0)(state, nshots=100).numpy()
    target_measurements = np.zeros_like(measurements)
    assert measurements.shape == (100,)
    np.testing.assert_allclose(measurements, target_measurements)

    state[0], state[-1] = 0, 1
    measurements = gates.M(0)(state, nshots=100).numpy()
    target_measurements = np.ones_like(measurements)
    assert measurements.shape == (100,)
    np.testing.assert_allclose(measurements, target_measurements)


def test_multiple_qubit_measurement_gate():
    state = np.zeros(4)
    state[0] = 0
    state[2] = 1
    state = state.reshape((2, 2))

    measurements = gates.M(0, 1)(state, nshots=100).numpy()
    target_measurements = 2 * np.ones_like(measurements)
    assert measurements.shape == (100,)
    np.testing.assert_allclose(measurements, target_measurements)


def test_controlled_measurement_error():
    """Check that using `controlled_by` in measurements raises error."""
    with pytest.raises(NotImplementedError):
        m = gates.M(0).controlled_by(1)


def test_measurement_circuit():
    c = models.Circuit(2)
    c.add(gates.M(0))

    measurements = c(nshots=100).numpy()
    target_measurements = np.zeros_like(measurements)
    assert measurements.shape == (100,)
    np.testing.assert_allclose(measurements, target_measurements)
