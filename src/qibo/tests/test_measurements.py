import numpy as np
import pytest
from qibo import gates


def test_measurement_gate():
    """Check that measurement gate works when called on a state."""
    state = np.zeros(4)
    state[0] = 1
    state = state.reshape((2, 2))

    measurements = gates.M(0)(state, nshots=100)
    target_meas = np.zeros_like(measurements)
    np.testing.assert_allclose(measurements, target_meas)

    state[0], state[-1] = 0, 1
    measurements = gates.M(0)(state, nshots=100)
    target_meas = np.ones_like(measurements)
    np.testing.assert_allclose(measurements, target_meas)


def test_controlled_measurement_error():
    """Check that using `controlled_by` in measurements raises error."""
    with pytest.raises(NotImplementedError):
        m = gates.M(0).controlled_by(1)
