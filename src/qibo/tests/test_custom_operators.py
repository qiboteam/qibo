"""
Testing Tensorflow custom operators circuit.
"""
import pytest
import numpy as np
import tensorflow as tf
from qibo.tensorflow.custom_operators import initial_state


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_initial_state(dtype):
  """Check that initial_state updates first element properly."""
  final_state = tf.zeros(10, dtype=dtype)
  initial_state(final_state)
  exact_state = np.array([1] + [0]*9, dtype=dtype)
  np.testing.assert_allclose(final_state, exact_state)
