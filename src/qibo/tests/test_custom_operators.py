"""
Testing Tensorflow custom operators circuit.
"""
import pytest
import numpy as np
import tensorflow as tf
from qibo.tensorflow import custom_operators as op

_atol = 1e-6


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("compile", [False, True])
def test_initial_state(dtype, compile):
  """Check that initial_state updates first element properly."""
  def apply_operator(dtype):
    """Apply the initial_state operator"""
    a = tf.zeros(10, dtype=dtype)
    return op.initial_state(a)

  func = apply_operator
  if compile:
      func = tf.function(apply_operator)
  final_state = func(dtype)
  exact_state = np.array([1] + [0]*9, dtype=dtype)
  np.testing.assert_allclose(final_state, exact_state)


@pytest.mark.parametrize(("nqubits", "target", "dtype", "compile"),
                         [(5, 4, np.float32, False),
                          (4, 2, np.float32, True),
                          (4, 2, np.float64, False),
                          (3, 0, np.float64, True),
                          (8, 5, np.float64, False)])
def test_apply_gate(nqubits, target, dtype, compile):
    """Check that `op.apply_gate` agrees with `tf.einsum`."""
    def apply_operator(state, gate):
      return op.apply_gate(state, gate, nqubits, target)

    state = tf.complex(tf.random.uniform((2 ** nqubits,), dtype=dtype),
                       tf.random.uniform((2 ** nqubits,), dtype=dtype))
    gate = tf.complex(tf.random.uniform((2, 2), dtype=dtype),
                      tf.random.uniform((2, 2), dtype=dtype))

    einsum_str = {3: "abc,Aa->Abc",
                  4: "abcd,Cc->abCd",
                  5: "abcde,Ee->abcdE",
                  8: "abcdefgh,Ff->abcdeFgh"}
    target_state = tf.reshape(state, nqubits * (2,))
    target_state = tf.einsum(einsum_str[nqubits], target_state, gate)
    target_state = target_state.numpy().ravel()

    if compile:
        apply_operator = tf.function(apply_operator)
    state = apply_operator(state, gate)
    np.testing.assert_allclose(target_state, state.numpy(), atol=_atol)
