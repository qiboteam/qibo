"""
Testing Tensorflow custom operators circuit.
"""
import pytest
import numpy as np
import tensorflow as tf
from qibo.tensorflow import custom_operators as op
from qibo.tests import utils

_atol = 1e-6


def qubits_tensor(nqubits, targets, controls=[]):
    qubits = list(nqubits - np.array(controls) - 1)
    qubits.extend(nqubits - np.array(targets) - 1)
    qubits = sorted(qubits)
    return qubits


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


@pytest.mark.parametrize(("nqubits", "target", "dtype", "compile", "einsum_str"),
                         [(5, 4, np.float32, False, "abcde,Ee->abcdE"),
                          (4, 2, np.float32, True, "abcd,Cc->abCd"),
                          (4, 2, np.float64, False, "abcd,Cc->abCd"),
                          (3, 0, np.float64, True, "abc,Aa->Abc"),
                          (8, 5, np.float64, False, "abcdefgh,Ff->abcdeFgh")])
def test_apply_gate(nqubits, target, dtype, compile, einsum_str):
    """Check that ``op.apply_gate`` agrees with ``tf.einsum``."""
    def apply_operator(state, gate):
      qubits = qubits_tensor(nqubits, [target])
      return op.apply_gate(state, gate, qubits, nqubits, target)

    state = utils.random_tensorflow_complex((2 ** nqubits,), dtype)
    gate = utils.random_tensorflow_complex((2, 2), dtype)

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
    state = utils.random_tensorflow_complex((2 ** nqubits,), dtype=tf.float64)

    target_state = state.numpy()
    gate = np.eye(2 ** nqubits, dtype=target_state.dtype)
    gate[-2, -2], gate[-2, -1] = 0, 1
    gate[-1, -2], gate[-1, -1] = 1, 0
    target_state = gate.dot(target_state)

    xgate = tf.cast([[0, 1], [1, 0]], dtype=state.dtype)
    controls = list(range(nqubits - 1))
    def apply_operator(state):
      qubits = qubits_tensor(nqubits, [nqubits - 1], controls)
      return op.apply_gate(state, xgate, qubits, nqubits, nqubits - 1)
    if compile:
        apply_operator = tf.function(apply_operator)
    state = apply_operator(state)

    np.testing.assert_allclose(target_state, state.numpy())


@pytest.mark.parametrize(("nqubits", "target", "controls", "compile", "einsum_str"),
                         [(3, 0, [1, 2], False, "a,Aa->A"),
                          (4, 3, [0, 1, 2], True, "a,Aa->A"),
                          (5, 3, [1], True, "abcd,Cc->abCd"),
                          (5, 2, [1, 4], True, "abc,Bb->aBc"),
                          (6, 3, [0, 2, 5], False, "abc,Bb->aBc"),
                          (6, 3, [0, 2, 4, 5], False, "ab,Bb->aB")])
def test_apply_gate_controlled(nqubits, target, controls, compile, einsum_str):
    """Check ``op.apply_gate`` for random controlled gates."""
    state = utils.random_tensorflow_complex((2 ** nqubits,), dtype=tf.float64)
    gate = utils.random_tensorflow_complex((2, 2), dtype=tf.float64)

    target_state = state.numpy().reshape(nqubits * (2,))
    slicer = nqubits * [slice(None)]
    for c in controls:
        slicer[c] = 1
    slicer = tuple(slicer)
    target_state[slicer] = np.einsum(einsum_str, target_state[slicer], gate)
    target_state = target_state.ravel()

    def apply_operator(state):
      qubits = qubits_tensor(nqubits, [target], controls)
      return op.apply_gate(state, gate, qubits, nqubits, target)
    if compile:
        apply_operator = tf.function(apply_operator)

    state = apply_operator(state)
    np.testing.assert_allclose(target_state, state.numpy())


@pytest.mark.parametrize(("nqubits", "target", "gate"),
                         [(3, 0, "x"), (4, 3, "x"),
                          (5, 2, "y"), (3, 1, "z")])
@pytest.mark.parametrize("compile", [False, True])
def test_apply_pauli_gate(nqubits, target, gate, compile):
    """Check ``apply_x``, ``apply_y`` and ``apply_z`` kernels."""
    matrices = {"x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
                "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
                "z": np.array([[1, 0], [0, -1]], dtype=np.complex128)}
    state = utils.random_tensorflow_complex((2 ** nqubits,), dtype=tf.float64)
    target_state = tf.cast(state.numpy(), dtype=state.dtype)
    qubits = qubits_tensor(nqubits, [target])
    target_state = op.apply_gate(state, matrices[gate], qubits, nqubits, target)

    def apply_operator(state):
      qubits = qubits_tensor(nqubits, [target])
      return getattr(op, "apply_{}".format(gate))(state, qubits, nqubits, target)
    if compile:
        apply_operator = tf.function(apply_operator)
    state = apply_operator(state)

    np.testing.assert_allclose(target_state.numpy(), state.numpy())


@pytest.mark.parametrize(("nqubits", "target", "controls"),
                         [(3, 0, []), (3, 2, [1]),
                          (3, 2, [0, 1]), (6, 1, [0, 2, 4])])
@pytest.mark.parametrize("compile", [False, True])
def test_apply_zpow_gate(nqubits, target, controls, compile):
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

    state = utils.random_tensorflow_complex((2 ** nqubits,), dtype=tf.float64)

    target_state = np.diag(matrix).dot(state.numpy())

    def apply_operator(state):
      qubits = qubits_tensor(nqubits, [target], controls)
      return op.apply_z_pow(state, phase, qubits, nqubits, target)
    if compile:
        apply_operator = tf.function(apply_operator)
    state = apply_operator(state)

    np.testing.assert_allclose(target_state, state.numpy())


@pytest.mark.parametrize(("nqubits", "targets", "controls",
                          "compile", "einsum_str"),
                         [(3, [0, 1], [], False, "abc,ABab->ABc"),
                          (4, [0, 2], [], True, "abcd,ACac->AbCd"),
                          (3, [0, 1], [2], False, "ab,ABab->AB"),
                          (4, [0, 3], [1], True, "abc,ACac->AbC"),
                          (4, [2, 3], [0], False, "abc,BCbc->aBC"),
                          (5, [1, 4], [2], False, "abcd,BDbd->aBcD"),
                          (6, [1, 3], [0, 4], True, "abcd,ACac->AbCd"),
                          (6, [0, 5], [1, 2, 3], False, "abc,ACac->AbC")])
def test_apply_twoqubit_gate_controlled(nqubits, targets, controls,
                                        compile, einsum_str):
    """Check ``op.apply_twoqubit_gate`` for random gates."""
    state = utils.random_tensorflow_complex((2 ** nqubits,), dtype=tf.float64)
    gate = utils.random_tensorflow_complex((4, 4), dtype=tf.float64)
    gatenp = gate.numpy().reshape(4 * (2,))

    target_state = state.numpy().reshape(nqubits * (2,))
    slicer = nqubits * [slice(None)]
    for c in controls:
        slicer[c] = 1
    slicer = tuple(slicer)
    target_state[slicer] = np.einsum(einsum_str, target_state[slicer], gatenp)
    target_state = target_state.ravel()

    def apply_operator(state):
      qubits = qubits_tensor(nqubits, targets, controls)
      return op.apply_two_qubit_gate(state, gate, qubits, nqubits, *targets)
    if compile:
        apply_operator = tf.function(apply_operator)

    state = apply_operator(state)
    np.testing.assert_allclose(target_state, state.numpy())


@pytest.mark.parametrize(("nqubits", "targets", "controls",
                          "compile", "einsum_str"),
                         [(3, [0, 1], [], False, "abc,ABab->ABc"),
                          (4, [0, 2], [], True, "abcd,ACac->AbCd"),
                          (3, [1, 2], [0], False, "ab,ABab->AB"),
                          (4, [0, 1], [2], False, "abc,ABab->ABc"),
                          (5, [0, 1], [2], False, "abcd,ABab->ABcd"),
                          (5, [3, 4], [2], False, "abcd,CDcd->abCD"),
                          (4, [0, 3], [1], False, "abc,ACac->AbC"),
                          (4, [2, 3], [0], True, "abc,BCbc->aBC"),
                          (5, [1, 4], [2], False, "abcd,BDbd->aBcD"),
                          (6, [1, 3], [0, 4], True, "abcd,ACac->AbCd"),
                          (6, [0, 5], [1, 2, 3], False, "abc,ACac->AbC")])
def test_apply_fsim(nqubits, targets, controls, compile, einsum_str):
    """Check ``op.apply_twoqubit_gate`` for random gates."""
    state = utils.random_tensorflow_complex((2 ** nqubits,), dtype=tf.float64)
    rotation = utils.random_tensorflow_complex((2, 2), dtype=tf.float64)
    phase = utils.random_tensorflow_complex((1,), dtype=tf.float64)

    target_state = state.numpy().reshape(nqubits * (2,))
    gatenp = np.eye(4, dtype=target_state.dtype)
    gatenp[1:3, 1:3] = rotation.numpy()
    gatenp[3, 3] = phase.numpy()[0]
    gatenp = gatenp.reshape(4 * (2,))

    slicer = nqubits * [slice(None)]
    for c in controls:
        slicer[c] = 1
    slicer = tuple(slicer)
    target_state[slicer] = np.einsum(einsum_str, target_state[slicer], gatenp)
    target_state = target_state.ravel()

    gate = tf.concat([tf.reshape(rotation, (4,)), phase], axis=0)
    def apply_operator(state):
      qubits = qubits_tensor(nqubits, targets, controls)
      return op.apply_fsim(state, gate, qubits, nqubits, *targets)
    if compile:
        apply_operator = tf.function(apply_operator)

    state = apply_operator(state)
    np.testing.assert_allclose(target_state, state.numpy())


@pytest.mark.parametrize("compile", [False, True])
def test_apply_swap_with_matrix(compile):
    """Check ``apply_swap`` for two qubits."""
    state = utils.random_tensorflow_complex((2 ** 2,), dtype=tf.float64)
    matrix = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]])
    target_state = matrix.dot(state.numpy())

    def apply_operator(state):
      qubits = qubits_tensor(2, [0, 1])
      return op.apply_swap(state, qubits, 2, 0, 1)
    if compile:
        apply_operator = tf.function(apply_operator)
    state = apply_operator(state)
    np.testing.assert_allclose(target_state, state.numpy())


@pytest.mark.parametrize(("nqubits", "targets", "controls"),
                         [(2, [0, 1], []), (3, [0, 2], []), (4, [1, 3], []),
                          (3, [1, 2], [0]), (4, [0, 2], [1]), (4, [2, 3], [0]),
                          (5, [3, 4], [1, 2]), (6, [1, 4], [0, 2, 5])])
@pytest.mark.parametrize("compile", [False, True])
def test_apply_swap_general(nqubits, targets, controls, compile):
    """Check ``apply_swap`` for more general cases."""
    state = utils.random_tensorflow_complex((2 ** nqubits,), dtype=tf.float64)

    target0, target1 = targets
    for q in controls:
        if q < targets[0]:
            target0 -= 1
        if q < targets[1]:
            target1 -= 1

    target_state = state.numpy().reshape(nqubits * (2,))
    order = list(range(nqubits - len(controls)))
    order[target0], order[target1] = target1, target0
    slicer = tuple(1 if q in controls else slice(None) for q in range(nqubits))
    reduced_state = target_state[slicer]
    reduced_state = np.transpose(reduced_state, order)
    target_state[slicer] = reduced_state

    def apply_operator(state):
      qubits = qubits_tensor(nqubits, targets, controls)
      return op.apply_swap(state, qubits, nqubits, *targets)
    if compile:
        apply_operator = tf.function(apply_operator)
    state = apply_operator(state)
    np.testing.assert_allclose(target_state.ravel(), state.numpy())


# this test fails when compiling due to in-place updates of the state
@pytest.mark.parametrize("gate", ["h", "x", "z", "swap"])
@pytest.mark.parametrize("compile", [False])
def test_custom_op_toy_callback(gate, compile):
    """Check calculating ``callbacks`` using intermediate state values."""
    import functools
    state = utils.random_tensorflow_complex((2 ** 2,), dtype=tf.float64)
    mask = utils.random_tensorflow_complex((2 ** 2,), dtype=tf.float64)

    matrices = {"h": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
                "x": np.array([[0, 1], [1, 0]]),
                "z": np.array([[1, 0], [0, -1]])}
    for k, v in matrices.items():
        matrices[k] = np.kron(v, np.eye(2))
    matrices["swap"] = np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                                 [0, 1, 0, 0], [0, 0, 0, 1]])

    target_state = state.numpy()
    target_c1 = mask.numpy().dot(target_state)
    target_state = matrices[gate].dot(target_state)
    target_c2 = mask.numpy().dot(target_state)
    assert target_c1 != target_c2
    target_callback = [target_c1, target_c2]

    htf = tf.cast(np.array([[1, 1], [1, -1]]) / np.sqrt(2), dtype=state.dtype)
    qubits_t1 = qubits_tensor(2, [0])
    qubits_t2 = qubits_tensor(2, [0, 1])
    apply_gate = {"h": functools.partial(op.apply_gate, gate=htf, qubits=qubits_t1,
                                         nqubits=2, target=0),
                  "x": functools.partial(op.apply_x, qubits=qubits_t1,
                                         nqubits=2, target=0),
                  "z": functools.partial(op.apply_z, qubits=qubits_t1,
                                         nqubits=2, target=0),
                  "swap": functools.partial(op.apply_swap, qubits=qubits_t2,
                                            nqubits=2, target1=0, target2=1)}

    def apply_operator(state):
        c1 = tf.reduce_sum(mask * state)
        state0 = apply_gate[gate](state)
        c2 = tf.reduce_sum(mask * state0)
        return state0, tf.stack([c1, c2])
    if compile: # pragma: no cover
        # case not tested because it fails
        apply_operator = tf.function(apply_operator)
    state, callback = apply_operator(state)

    np.testing.assert_allclose(target_state, state.numpy())
    np.testing.assert_allclose(target_callback, callback.numpy())


def check_unimplemented_error(func, *args): # pragma: no cover
    # method not tested by GitHub workflows because it requires GPU
    error = tf.python.framework.errors_impl.UnimplementedError
    with pytest.raises(error):
        func(*args)


@pytest.mark.parametrize("nqubits", [3, 4, 7, 8, 9, 10])
@pytest.mark.parametrize("ndevices", [2, 4, 8])
def test_transpose_state(nqubits, ndevices):
    for _ in range(10):
        # Generate global qubits randomly
        all_qubits = np.arange(nqubits)
        np.random.shuffle(all_qubits)
        qubit_order = list(all_qubits)
        state = utils.random_tensorflow_complex((2 ** nqubits,), dtype=tf.float64)

        state_tensor = state.numpy().reshape(nqubits * (2,))
        target_state = np.transpose(state_tensor, qubit_order).ravel()

        new_state = tf.zeros_like(state)
        shape = (ndevices, int(state.shape[0]) // ndevices)
        state = tf.reshape(state, shape)
        pieces = [state[i] for i in range(ndevices)]
        if tf.config.list_physical_devices("GPU"): # pragma: no cover
            # case not tested by GitHub workflows because it requires GPU
            check_unimplemented_error(op.transpose_state,
                                      pieces, new_state, nqubits, qubit_order)
        else:
            new_state = op.transpose_state(pieces, new_state, nqubits, qubit_order)
            np.testing.assert_allclose(target_state, new_state.numpy())


@pytest.mark.parametrize("nqubits", [4, 5, 7, 8, 9, 10])
def test_swap_pieces_zero_global(nqubits):
    state = utils.random_tensorflow_complex((2 ** nqubits,), dtype=tf.float64)
    target_state = tf.cast(np.copy(state.numpy()), dtype=state.dtype)
    shape = (2, int(state.shape[0]) // 2)
    state = tf.reshape(state, shape)

    for _ in range(10):
        local = np.random.randint(1, nqubits)

        qubits_t = qubits_tensor(nqubits, [0, local])
        target_state = op.apply_swap(target_state, qubits_t, nqubits, 0, local)
        target_state = tf.reshape(target_state, shape)

        piece0, piece1 = state[0], state[1]
        if tf.config.list_physical_devices("GPU"): # pragma: no cover
            # case not tested by GitHub workflows because it requires GPU
            check_unimplemented_error(op.swap_pieces,
                                      piece0, piece1, local - 1, nqubits - 1)
        else:
            op.swap_pieces(piece0, piece1, local - 1, nqubits - 1)
            np.testing.assert_allclose(target_state[0], piece0.numpy())
            np.testing.assert_allclose(target_state[1], piece1.numpy())


@pytest.mark.parametrize("nqubits", [5, 7, 8, 9, 10])
def test_swap_pieces(nqubits):
    state = utils.random_tensorflow_complex((2 ** nqubits,), dtype=tf.float64)
    target_state = tf.cast(np.copy(state.numpy()), dtype=state.dtype)
    shape = (2, int(state.shape[0]) // 2)

    for _ in range(10):
        global_qubit = np.random.randint(0, nqubits)
        local_qubit = np.random.randint(0, nqubits)
        while local_qubit == global_qubit:
            local_qubit = np.random.randint(0, nqubits)

        transpose_order = ([global_qubit] + list(range(global_qubit)) +
                           list(range(global_qubit + 1, nqubits)))

        qubits_t = qubits_tensor(nqubits, [global_qubit, local_qubit])
        target_state = op.apply_swap(target_state, qubits_t, nqubits, global_qubit, local_qubit)
        target_state = tf.reshape(target_state, nqubits * (2,))
        target_state = tf.transpose(target_state, transpose_order)
        target_state = tf.reshape(target_state, shape)

        state = tf.reshape(state, nqubits * (2,))
        state = tf.transpose(state, transpose_order)
        state = tf.reshape(state, shape)
        piece0, piece1 = state[0], state[1]
        if tf.config.list_physical_devices("GPU"): # pragma: no cover
            # case not tested by GitHub workflows because it requires GPU
            check_unimplemented_error(op.swap_pieces,
                                      piece0, piece1, local_qubit - 1, nqubits - 1)
        else:
            op.swap_pieces(piece0, piece1,
                           local_qubit - int(global_qubit < local_qubit),
                           nqubits - 1)
            np.testing.assert_allclose(target_state[0], piece0.numpy())
            np.testing.assert_allclose(target_state[1], piece1.numpy())
