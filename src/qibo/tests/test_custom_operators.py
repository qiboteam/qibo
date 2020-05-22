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


def test_apply_with_circuit():
    """Check that `op.apply_gate` agrees with qibo circuits."""
    # Temporary test since `op.apply_gate` will be integrated to circuits

    c = models.Circuit(3)
    c.add(gates.H(0))
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.Y(2))
    target_state = c().numpy()

    state = tf.zeros_like(target_state)
    op.initial_state(state)
    op.apply_gate(state, c.queue[0].matrix, c.nqubits, 0)
    op.apply_gate(state, c.queue[1].matrix, c.nqubits, 1)
    op.apply_gate(state, c.queue[2].matrix, c.nqubits, 2)
    np.testing.assert_allclose(target_state, state.numpy())


def test_apply_with_circuit_controlled():
    """Check that `op.apply_gate` agrees with qibo circuits."""
    # Temporary test since `op.apply_gate` will be integrated to circuits
    theta = 0.4321

    c = models.Circuit(2)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.CZPow(0, 1, theta=theta))
    target_state = c().numpy()

    zpow = np.array([[1, 0], [0, np.exp(1j * theta)]])
    state = tf.zeros_like(target_state)
    op.initial_state(state)
    op.apply_gate(state, c.queue[0].matrix, c.nqubits, 0)
    op.apply_gate(state, c.queue[1].matrix, c.nqubits, 1)
    op.apply_gate(state, zpow, c.nqubits, 1, [0])
    np.testing.assert_allclose(target_state, state.numpy())


def test_apply_gate_controlled():
    nqubits = 5
    dtype = tf.float64
    state = tf.complex(tf.random.uniform((2 ** nqubits,), dtype=dtype),
                       tf.random.uniform((2 ** nqubits,), dtype=dtype))
    gate = tf.complex(tf.random.uniform((2, 2), dtype=dtype),
                      tf.random.uniform((2, 2), dtype=dtype))

    target = 3
    controls = [1]
    einsum_str = "abcd,Cc->abCd"

    # Apply controlled gate in numpy
    target_state = state.numpy().reshape(nqubits * (2,))
    slicer = nqubits * [slice(None)]
    for c in controls:
        slicer[c] = 1
    target_state[slicer] = np.einsum(einsum_str, target_state[slicer], gate.numpy())
    target_state = target_state.ravel()

    op.apply_gate(state, gate, nqubits, target, controls)
    np.testing.assert_allclose(target_state, state.numpy())
