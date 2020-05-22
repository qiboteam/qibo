"""
Testing Tensorflow custom operators circuit.
"""
import pytest
import numpy as np
import tensorflow as tf
from qibo import models, gates
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


def tensorflow_random_complex(shape, dtype):
  _re = tf.random.uniform(shape, dtype=dtype)
  _im = tf.random.uniform(shape, dtype=dtype)
  return tf.complex(_re, _im)


@pytest.mark.parametrize(("nqubits", "target", "dtype", "compile", "einsum_str"),
                         [(5, 4, np.float32, False, "abcde,Ee->abcdE"),
                          (4, 2, np.float32, True, "abcd,Cc->abCd"),
                          (4, 2, np.float64, False, "abcd,Cc->abCd"),
                          (3, 0, np.float64, True, "abc,Aa->Abc"),
                          (8, 5, np.float64, False, "abcdefgh,Ff->abcdeFgh")])
def test_apply_gate(nqubits, target, dtype, compile, einsum_str):
    """Check that `op.apply_gate` agrees with `tf.einsum`."""
    def apply_operator(state, gate):
      return op.apply_gate(state, gate, nqubits, target)

    state = tensorflow_random_complex((2 ** nqubits,), dtype)
    gate = tensorflow_random_complex((2, 2), dtype)

    target_state = tf.reshape(state, nqubits * (2,))
    target_state = tf.einsum(einsum_str, target_state, gate)
    target_state = target_state.numpy().ravel()

    if compile:
        apply_operator = tf.function(apply_operator)
    state = apply_operator(state, gate)
    np.testing.assert_allclose(target_state, state.numpy(), atol=_atol)


@pytest.mark.skip
@pytest.mark.parametrize("nqubits", [2, 3, 4, 5])
def test_apply_gate_cx(nqubits):
    """Check that multiply-controlled X gate works."""
    state = (np.random.random((2 ** nqubits,)) +
             1j * np.random.random((2 ** nqubits,))).astype(np.complex128)
    gate = np.eye(2 ** nqubits, dtype=state.dtype)
    gate[-2, -2], gate[-2, -1] = 0, 1
    gate[-1, -2], gate[-1, -1] = 1, 0
    target_state = gate.dot(state)

    xgate = np.array([[0, 1], [1, 0]]).astype(state.dtype)
    controls = list(range(nqubits - 1))
    state = apply_gate(state, xgate, nqubits, nqubits - 1, controls)

    np.testing.assert_allclose(target_state, state)
