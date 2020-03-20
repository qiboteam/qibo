import numpy as np
from qibo import gates


def test_measurement_gate():
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
