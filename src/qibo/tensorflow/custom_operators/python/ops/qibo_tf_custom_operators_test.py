"""
Testing Tensorflow custom operators circuit.
"""
import numpy as np
import tensorflow as tf
from qibo.tensorflow.custom_operators import initial_state


def test_initial_state():
  """Check that initial_state updates first element properly."""
  final_state = tf.zeros(10, dtype=tf.complex128)
  initial_state(final_state)
  exact_state = np.zeros(10, dtype=np.complex128)
  exact_state[0] = 1
  np.testing.assert_allclose(final_state, exact_state)
