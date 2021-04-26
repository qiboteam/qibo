"""
Testing Tensorflow custom operators circuit.
"""
import itertools
import pytest
import numpy as np
from qibo import K, get_threads


_atol = 1e-6


def qubits_tensor(nqubits, targets, controls=[]):
    qubits = list(nqubits - np.array(controls) - 1)
    qubits.extend(nqubits - np.array(targets) - 1)
    qubits = sorted(qubits)
    return qubits


def random_complex(shape, dtype=None):
    x = np.random.random(shape) + 1j * np.random.random(shape)
    if dtype is not None:
        x = K.cast(x.astype(dtype))
    return K.cast(x)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("compile", [False, True])
def test_initial_state(dtype, compile):
  """Check that initial_state updates first element properly."""
  def apply_operator(dtype):
    """Apply the initial_state operator"""
    return K.op.initial_state(nqubits=4, dtype=dtype,
                            is_matrix=False, omp_num_threads=get_threads())

  func = apply_operator
  if compile:
      func = K.compile(apply_operator)
  final_state = func(dtype)
  exact_state = np.array([1] + [0]*15, dtype=dtype)
  np.testing.assert_allclose(final_state, exact_state)


@pytest.mark.parametrize(("nqubits", "target", "dtype", "compile", "einsum_str"),
                         [(5, 4, np.complex64, False, "abcde,Ee->abcdE"),
                          (4, 2, np.complex64, True, "abcd,Cc->abCd"),
                          (4, 2, np.complex128, False, "abcd,Cc->abCd"),
                          (3, 0, np.complex128, True, "abc,Aa->Abc"),
                          (8, 5, np.complex128, False, "abcdefgh,Ff->abcdeFgh")])
def test_apply_gate(nqubits, target, dtype, compile, einsum_str):
    """Check that ``K.op.apply_gate`` agrees with einsum gate implementation."""
    def apply_operator(state, gate):
      qubits = qubits_tensor(nqubits, [target])
      return K.op.apply_gate(state, gate, qubits, nqubits, target, get_threads())

    state = random_complex((2 ** nqubits,), dtype=dtype)
    gate = random_complex((2, 2), dtype=dtype)

    target_state = K.reshape(state, nqubits * (2,))
    target_state = K.einsum(einsum_str, target_state, gate)
    target_state = target_state.numpy().ravel()

    if compile:
        apply_operator = K.compile(apply_operator)
    state = apply_operator(state, gate)
    np.testing.assert_allclose(target_state, state, atol=_atol)


@pytest.mark.parametrize(("nqubits", "compile"),
                         [(2, True), (3, False), (4, True), (5, False)])
def test_apply_gate_cx(nqubits, compile):
    """Check ``K.op.apply_gate`` for multiply-controlled X gates."""
    state = random_complex((2 ** nqubits,))

    target_state = np.array(state)
    gate = np.eye(2 ** nqubits, dtype=target_state.dtype)
    gate[-2, -2], gate[-2, -1] = 0, 1
    gate[-1, -2], gate[-1, -1] = 1, 0
    target_state = np.dot(gate, target_state)

    xgate = K.cast([[0, 1], [1, 0]], dtype=state.dtype)
    controls = list(range(nqubits - 1))
    def apply_operator(state):
      qubits = qubits_tensor(nqubits, [nqubits - 1], controls)
      return K.op.apply_gate(state, xgate, qubits, nqubits, nqubits - 1, get_threads())
    if compile:
        apply_operator = K.compile(apply_operator)
    state = apply_operator(state)

    np.testing.assert_allclose(target_state, state)


@pytest.mark.parametrize(("nqubits", "target", "controls", "compile", "einsum_str"),
                         [(3, 0, [1, 2], False, "a,Aa->A"),
                          (4, 3, [0, 1, 2], True, "a,Aa->A"),
                          (5, 3, [1], True, "abcd,Cc->abCd"),
                          (5, 2, [1, 4], True, "abc,Bb->aBc"),
                          (6, 3, [0, 2, 5], False, "abc,Bb->aBc"),
                          (6, 3, [0, 2, 4, 5], False, "ab,Bb->aB")])
def test_apply_gate_controlled(nqubits, target, controls, compile, einsum_str):
    """Check ``K.op.apply_gate`` for random controlled gates."""
    state = random_complex((2 ** nqubits,))
    gate = random_complex((2, 2))

    target_state = state.numpy().reshape(nqubits * (2,))
    slicer = nqubits * [slice(None)]
    for c in controls:
        slicer[c] = 1
    slicer = tuple(slicer)
    target_state[slicer] = np.einsum(einsum_str, target_state[slicer], gate)
    target_state = target_state.ravel()

    def apply_operator(state):
      qubits = qubits_tensor(nqubits, [target], controls)
      return K.op.apply_gate(state, gate, qubits, nqubits, target, get_threads())
    if compile:
        apply_operator = K.compile(apply_operator)

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
    state = random_complex((2 ** nqubits,))
    target_state = K.cast(state, dtype=state.dtype)
    qubits = qubits_tensor(nqubits, [target])
    target_state = K.op.apply_gate(state, matrices[gate], qubits, nqubits, target, get_threads())

    def apply_operator(state):
      qubits = qubits_tensor(nqubits, [target])
      return getattr(K.op, "apply_{}".format(gate))(state, qubits, nqubits, target, get_threads())
    if compile:
        apply_operator = K.compile(apply_operator)
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

    state = random_complex((2 ** nqubits,))

    target_state = np.diag(matrix).dot(state.numpy())

    def apply_operator(state):
      qubits = qubits_tensor(nqubits, [target], controls)
      return K.op.apply_z_pow(state, phase, qubits, nqubits, target, get_threads())
    if compile:
        apply_operator = K.compile(apply_operator)
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
    """Check ``K.op.apply_twoqubit_gate`` for random gates."""
    state = random_complex((2 ** nqubits,))
    gate = random_complex((4, 4))
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
      return K.op.apply_two_qubit_gate(state, gate, qubits, nqubits, *targets, get_threads())
    if compile:
        apply_operator = K.compile(apply_operator)

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
    """Check ``K.op.apply_twoqubit_gate`` for random gates."""
    state = random_complex((2 ** nqubits,))
    rotation = random_complex((2, 2))
    phase = random_complex((1,))

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

    gate = K.concatenate([K.reshape(rotation, (4,)), phase], axis=0)
    def apply_operator(state):
      qubits = qubits_tensor(nqubits, targets, controls)
      return K.op.apply_fsim(state, gate, qubits, nqubits, *targets, get_threads())
    if compile:
        apply_operator = K.compile(apply_operator)

    state = apply_operator(state)
    np.testing.assert_allclose(target_state, state.numpy())


@pytest.mark.parametrize("compile", [False, True])
def test_apply_swap_with_matrix(compile):
    """Check ``apply_swap`` for two qubits."""
    state = random_complex((2 ** 2,))
    matrix = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]])
    target_state = matrix.dot(state.numpy())

    def apply_operator(state):
      qubits = qubits_tensor(2, [0, 1])
      return K.op.apply_swap(state, qubits, 2, 0, 1, get_threads())
    if compile:
        apply_operator = K.compile(apply_operator)
    state = apply_operator(state)
    np.testing.assert_allclose(target_state, state.numpy())


@pytest.mark.parametrize(("nqubits", "targets", "controls"),
                         [(2, [0, 1], []), (3, [0, 2], []), (4, [1, 3], []),
                          (3, [1, 2], [0]), (4, [0, 2], [1]), (4, [2, 3], [0]),
                          (5, [3, 4], [1, 2]), (6, [1, 4], [0, 2, 5])])
@pytest.mark.parametrize("compile", [False, True])
def test_apply_swap_general(nqubits, targets, controls, compile):
    """Check ``apply_swap`` for more general cases."""
    state = random_complex((2 ** nqubits,))

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
      return K.op.apply_swap(state, qubits, nqubits, *targets, get_threads())
    if compile:
        apply_operator = K.compile(apply_operator)
    state = apply_operator(state)
    np.testing.assert_allclose(target_state.ravel(), state.numpy())


@pytest.mark.parametrize("nqubits,targets,results",
                         [(2, [0], [1]), (2, [1], [0]), (3, [1], [1]),
                          (4, [1, 3], [1, 0]), (5, [1, 2, 4], [0, 1, 1]),
                          (15, [4, 7], [0, 0]), (16, [8, 12, 15], [1, 0, 1])])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_collapse_state(nqubits, targets, results, dtype):
    """Check ``collapse_state`` kernel."""
    atol = 1e-7 if dtype == np.complex64 else 1e-14
    state = random_complex((2 ** nqubits,), dtype=dtype)
    slicer = nqubits * [slice(None)]
    for t, r in zip(targets, results):
        slicer[t] = r
    slicer = tuple(slicer)
    initial_state = np.reshape(state, nqubits * (2,))
    target_state = np.zeros_like(initial_state)
    target_state[slicer] = initial_state[slicer]
    norm = (np.abs(target_state) ** 2).sum()
    target_state = target_state.ravel() / np.sqrt(norm)

    qubits = sorted(nqubits - np.array(targets) - 1)
    b2d = 2 ** np.arange(len(results) - 1, -1, -1)
    result = np.array(results).dot(b2d)
    state = K.op.collapse_state(state, qubits, result, nqubits)
    np.testing.assert_allclose(state, target_state, atol=atol)


# this test fails when compiling due to in-place updates of the state
@pytest.mark.parametrize("gate", ["h", "x", "z", "swap"])
@pytest.mark.parametrize("compile", [False])
def test_custom_op_toy_callback(gate, compile):
    """Check calculating ``callbacks`` using intermediate state values."""
    import functools
    state = random_complex((2 ** 2,))
    mask = random_complex((2 ** 2,))

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

    htf = K.cast(np.array([[1, 1], [1, -1]]) / np.sqrt(2), dtype=state.dtype)
    qubits_t1 = qubits_tensor(2, [0])
    qubits_t2 = qubits_tensor(2, [0, 1])
    apply_gate = {"h": functools.partial(K.op.apply_gate, gate=htf, qubits=qubits_t1,
                                         nqubits=2, target=0, omp_num_threads=get_threads()),
                  "x": functools.partial(K.op.apply_x, qubits=qubits_t1,
                                         nqubits=2, target=0, omp_num_threads=get_threads()),
                  "z": functools.partial(K.op.apply_z, qubits=qubits_t1,
                                         nqubits=2, target=0, omp_num_threads=get_threads()),
                  "swap": functools.partial(K.op.apply_swap, qubits=qubits_t2,
                                            nqubits=2, target1=0, target2=1,
                                            omp_num_threads=get_threads())}

    def apply_operator(state):
        c1 = K.sum(mask * state)
        state0 = apply_gate[gate](state)
        c2 = K.sum(mask * state0)
        return state0, K.stack([c1, c2])
    if compile: # pragma: no cover
        # case not tested because it fails
        apply_operator = K.compile(apply_operator)
    state, callback = apply_operator(state)

    np.testing.assert_allclose(target_state, state.numpy())
    np.testing.assert_allclose(target_callback, callback.numpy())


def check_unimplemented_error(func, *args): # pragma: no cover
    # method not tested by GitHub workflows because it requires GPU
    from tensorflow.python.framework import errors_impl # pylint: disable=no-name-in-module
    error = errors_impl.UnimplementedError
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
        state = random_complex((2 ** nqubits,))

        state_tensor = state.numpy().reshape(nqubits * (2,))
        target_state = np.transpose(state_tensor, qubit_order).ravel()

        new_state = K.zeros_like(state)
        shape = (ndevices, int(state.shape[0]) // ndevices)
        state = K.reshape(state, shape)
        pieces = [state[i] for i in range(ndevices)]
        if K.gpu_devices: # pragma: no cover
            # case not tested by GitHub workflows because it requires GPU
            check_unimplemented_error(K.op.transpose_state,
                                      pieces, new_state, nqubits, qubit_order, get_threads())
        else:
            new_state = K.op.transpose_state(pieces, new_state, nqubits, qubit_order, get_threads())
            np.testing.assert_allclose(target_state, new_state.numpy())


@pytest.mark.parametrize("nqubits", [4, 5, 7, 8, 9, 10])
def test_swap_pieces_zero_global(nqubits):
    state = random_complex((2 ** nqubits,))
    target_state = K.cast(np.copy(state.numpy()))
    shape = (2, int(state.shape[0]) // 2)
    state = K.reshape(state, shape)

    for _ in range(10):
        local = np.random.randint(1, nqubits)

        qubits_t = qubits_tensor(nqubits, [0, local])
        target_state = K.op.apply_swap(target_state, qubits_t, nqubits, 0, local, get_threads())
        target_state = K.reshape(target_state, shape)

        piece0, piece1 = state[0], state[1]
        if K.gpu_devices: # pragma: no cover
            # case not tested by GitHub workflows because it requires GPU
            check_unimplemented_error(K.op.swap_pieces,
                                      piece0, piece1, local - 1, nqubits - 1, get_threads())
        else:
            K.op.swap_pieces(piece0, piece1, local - 1, nqubits - 1, get_threads())
            np.testing.assert_allclose(target_state[0], piece0.numpy())
            np.testing.assert_allclose(target_state[1], piece1.numpy())


@pytest.mark.parametrize("nqubits", [5, 7, 8, 9, 10])
def test_swap_pieces(nqubits):
    state = random_complex((2 ** nqubits,))
    target_state = K.cast(np.copy(state.numpy()), dtype=state.dtype)
    shape = (2, int(state.shape[0]) // 2)

    for _ in range(10):
        global_qubit = np.random.randint(0, nqubits)
        local_qubit = np.random.randint(0, nqubits)
        while local_qubit == global_qubit:
            local_qubit = np.random.randint(0, nqubits)

        transpose_order = ([global_qubit] + list(range(global_qubit)) +
                           list(range(global_qubit + 1, nqubits)))

        qubits_t = qubits_tensor(nqubits, [global_qubit, local_qubit])
        target_state = K.op.apply_swap(target_state, qubits_t, nqubits, global_qubit, local_qubit, get_threads())
        target_state = K.reshape(target_state, nqubits * (2,))
        target_state = K.transpose(target_state, transpose_order)
        target_state = K.reshape(target_state, shape)

        state = K.reshape(state, nqubits * (2,))
        state = K.transpose(state, transpose_order)
        state = K.reshape(state, shape)
        piece0, piece1 = state[0], state[1]
        if K.gpu_devices: # pragma: no cover
            # case not tested by GitHub workflows because it requires GPU
            check_unimplemented_error(K.op.swap_pieces,
                                      piece0, piece1, local_qubit - 1, nqubits - 1, get_threads())
        else:
            K.op.swap_pieces(piece0, piece1,
                           local_qubit - int(global_qubit < local_qubit),
                           nqubits - 1, get_threads())
            np.testing.assert_allclose(target_state[0], piece0.numpy())
            np.testing.assert_allclose(target_state[1], piece1.numpy())


@pytest.mark.skip("tf.tensor_scatter_nd_update bug on GPU (tensorflow#42581)")
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("compile", [False, True])
def test_initial_state_gradient(dtype, compile): # pragma: no cover
    # Test skipped due to `tf.tensor_scatter_nd_update` bug on GPU
    def grad_default(var):
        update = np.array([1]).astype(dtype)
        with K.optimization.GradientTape() as tape:
            loss = K.backend.tensor_scatter_nd_update(var, [[0]], update)
        return tape.gradient(loss, var)

    def grad_custom(var):
        with K.optimization.GradientTape() as tape:
            loss = K.op.initial_state(var)
        return tape.gradient(loss, var)

    if compile:
        grad_default = K.compile(grad_default)
        grad_custom = K.compile(grad_custom)

    zeros = K.optimization.Variable(K.zeros(10, dtype=dtype))
    grad_reference = grad_default(zeros)
    grad_custom_op = grad_custom(zeros)
    np.testing.assert_allclose(grad_reference, grad_custom_op)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("inttype", [np.int32, np.int64])
def test_measure_frequencies(dtype, inttype):
    import sys
    probs = np.ones(16, dtype=dtype) / 16
    frequencies = np.zeros(16, dtype=inttype)
    frequencies = K.op.measure_frequencies(frequencies, probs, nshots=1000,
                                           nqubits=4, omp_num_threads=1,
                                           seed=1234)
    if sys.platform == "linux":
        target_frequencies = [60, 50, 68, 64, 53, 53, 67, 54, 64, 53, 67,
                              69, 76, 57, 64, 81]
    elif sys.platform == "darwin": # pragma: no cover
        target_frequencies = [57, 51, 62, 63, 55, 70, 52, 47, 75, 58, 63,
                              73, 68, 72, 60, 74]
    assert np.sum(frequencies) == 1000
    np.testing.assert_allclose(frequencies, target_frequencies)


NONZERO = list(itertools.combinations(range(8), r=1))
NONZERO.extend(itertools.combinations(range(8), r=2))
NONZERO.extend(itertools.combinations(range(8), r=3))
NONZERO.extend(itertools.combinations(range(8), r=4))
@pytest.mark.parametrize("nonzero", NONZERO)
def test_measure_frequencies_sparse_probabilities(nonzero):
    import sys
    probs = np.zeros(8, dtype=np.float64)
    for i in nonzero:
        probs[i] = 1
    probs = probs / np.sum(probs)
    frequencies = np.zeros(8, dtype=np.int64)
    frequencies = K.op.measure_frequencies(frequencies, probs, nshots=1000,
                                           nqubits=3, omp_num_threads=1,
                                           seed=1234)
    assert np.sum(frequencies) == 1000
    for i, freq in enumerate(frequencies):
        if i in nonzero:
            assert freq != 0
        else:
            assert freq == 0


def test_backend_sample_frequencies_seed():
    """Check that frequencies generated using custom operator are different in each call."""
    from qibo import K
    from qibo.config import SHOT_CUSTOM_OP_THREASHOLD
    nshots = SHOT_CUSTOM_OP_THREASHOLD + 1
    probs = np.random.random(8)
    probs = probs / np.sum(probs)
    frequencies1 = K.sample_frequencies(probs, nshots)
    frequencies2 = K.sample_frequencies(probs, nshots)
    np.testing.assert_raises(AssertionError, np.testing.assert_allclose,
                             frequencies1, frequencies2)
