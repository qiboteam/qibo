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
    """Check that ``op.apply_gate`` agrees with ``tf.einsum``."""
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


@pytest.mark.parametrize(("nqubits", "compile"),
                         [(2, True), (3, False), (4, True), (5, False)])
def test_apply_gate_cx(nqubits, compile):
    """Check ``op.apply_gate`` for multiply-controlled X gates."""
    state = tensorflow_random_complex((2 ** nqubits,), dtype=tf.float64)

    target_state = state.numpy()
    gate = np.eye(2 ** nqubits, dtype=target_state.dtype)
    gate[-2, -2], gate[-2, -1] = 0, 1
    gate[-1, -2], gate[-1, -1] = 1, 0
    target_state = gate.dot(target_state)

    xgate = tf.cast([[0, 1], [1, 0]], dtype=state.dtype)
    controls = list(range(nqubits - 1))
    def apply_operator(state):
      return op.apply_gate(state, xgate, nqubits, nqubits - 1, controls)
    if compile:
        apply_operator = tf.function(apply_operator)
    state = apply_operator(state)

    np.testing.assert_allclose(target_state, state.numpy())


@pytest.mark.parametrize(("nqubits", "target", "controls", "einsum_str"),
                         [(3, 0, [1, 2], "a,Aa->A"),
                          (4, 3, [0, 1, 2], "a,Aa->A"),
                          (5, 3, [1], "abcd,Cc->abCd"),
                          (5, 2, [1, 4], "abc,Bb->aBc"),
                          (6, 3, [0, 2, 5], "abc,Bb->aBc"),
                          (6, 3, [0, 2, 4, 5], "ab,Bb->aB")])
def test_apply_gate_controlled(nqubits, target, controls, einsum_str):
    """Check ``op.apply_gate`` for random controlled gates."""
    state = tensorflow_random_complex((2 ** nqubits,), dtype=tf.float64)
    gate = tensorflow_random_complex((2, 2), dtype=tf.float64)

    target_state = state.numpy().reshape(nqubits * (2,))
    slicer = nqubits * [slice(None)]
    for c in controls:
        slicer[c] = 1
    slicer = tuple(slicer)
    target_state[slicer] = np.einsum(einsum_str, target_state[slicer], gate)
    target_state = target_state.ravel()

    state = op.apply_gate(state, gate, nqubits, target, controls)
    np.testing.assert_allclose(target_state, state.numpy())


def test_apply_gate_error():
    """Check that ``TypeError`` is raised for invalid ``controls``."""
    state = tensorflow_random_complex((2 ** 2,), dtype=tf.float64)
    gate = tensorflow_random_complex((2, 2), dtype=tf.float64)
    with pytest.raises(TypeError):
        state = op.apply_gate(state, gate, 2, 0, "a")


@pytest.mark.parametrize(("nqubits", "target", "gate"),
                         [(3, 0, "x"), (4, 3, "x"),
                          (5, 2, "y"), (3, 1, "z")])
def test_apply_pauli_gate(nqubits, target, gate):
    """Check ``apply_x``, ``apply_y`` and ``apply_z`` kernels."""
    matrices = {"x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
                "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
                "z": np.array([[1, 0], [0, -1]], dtype=np.complex128)}
    state = tensorflow_random_complex((2 ** nqubits,), dtype=tf.float64)
    target_state = tf.cast(state.numpy(), dtype=state.dtype)

    state = getattr(op, "apply_{}".format(gate))(state, nqubits, target)
    target_state = op.apply_gate(state, matrices[gate], nqubits, target)

    np.testing.assert_allclose(target_state.numpy(), state.numpy())


@pytest.mark.parametrize(("nqubits", "target", "controls"),
                         [(3, 0, []), (3, 2, [1]),
                          (3, 2, [0, 1]), (6, 1, [0, 2, 4])])
def test_apply_zpow_gate(nqubits, target, controls):
    """Check ``apply_zpow`` (including CZPow case)."""
    import itertools
    phase = np.exp(1j * 0.1234)
    qubits = controls[:]
    qubits.append(target)
    qubits.sort()
    matrix = np.ones(2 ** nqubits, dtype=np.complex128)
    for i, conf in enumerate(itertools.product([0, 1], repeat=nqubits)):
        if np.array(conf)[qubits].prod():
            matrix[i] = phase

    state = tensorflow_random_complex((2 ** nqubits,), dtype=tf.float64)

    target_state = np.diag(matrix).dot(state.numpy())
    state = op.apply_zpow(state, phase, nqubits, target, controls)

    np.testing.assert_allclose(target_state, state.numpy())
