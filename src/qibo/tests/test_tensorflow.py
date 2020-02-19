import numpy as np
from qibo import models
from qibo import gates


def test_basic_hadamard():
    c = models.Circuit(1)
    c.add(gates.H(0))
    final_state = c.run()
    target_state = np.ones_like(final_state) / np.sqrt(2)
    np.testing.assert_allclose(final_state, target_state)