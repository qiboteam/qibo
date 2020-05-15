"""Tests for initial_state ops."""
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test
from qibo.tensorflow.custom_operators.python.ops.qibo_tf_custom_operators import initial_state


class InitialStateTest(test.TestCase):

  def testInitialState(self):
    with self.test_session():
      a = tf.zeros(10, dtype=tf.complex128)
      initial_state(a)
      b = np.zeros(10, dtype=np.complex128)
      b[0] = 1
      self.assertAllClose(a, b)


if __name__ == '__main__':
  test.main()
