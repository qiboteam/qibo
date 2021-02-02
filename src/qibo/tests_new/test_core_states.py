"""Tests methods defined in `qibo/core/states.py`."""
import pytest
import numpy as np
import qibo
from qibo import K
from qibo.core import states


def test_state_shape_and_dtype(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    state = states.VectorState.zstate(3)
    assert state.shape == (8,)
    assert state.dtype == K.dtypes('DTYPECPX')
    state = states.MatrixState.zstate(3)
    assert state.shape == (8, 8)
    assert state.dtype == K.dtypes('DTYPECPX')
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("nqubits", [None, 2])
def test_vector_state_tensor_setter(backend, nqubits):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    state = states.VectorState(nqubits)
    with pytest.raises(AttributeError):
        tensor = state.tensor
    state.tensor = np.ones(4)
    assert state.nqubits == 2
    np.testing.assert_allclose(state.tensor, np.ones(4))
    np.testing.assert_allclose(np.array(state), np.ones(4))
    np.testing.assert_allclose(state.numpy(), np.ones(4))
    with pytest.raises(ValueError):
        state.tensor = np.zeros(5)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("nqubits", [None, 2])
def test_matrix_state_tensor_setter(backend, nqubits):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    # TODO: Fix this
    qibo.set_backend(original_backend)


def test_zstate_initialization(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    state = states.VectorState.zstate(4)
    target_state = np.zeros(16)
    target_state[0] = 1
    np.testing.assert_allclose(state.tensor, target_state)
    state = states.MatrixState.zstate(3)
    target_state = np.zeros((8, 8))
    target_state[0, 0] = 1
    np.testing.assert_allclose(state.tensor, target_state)
    qibo.set_backend(original_backend)


def test_xstate_initialization(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    state = states.VectorState.xstate(4)
    target_state = np.ones(16) / 4
    np.testing.assert_allclose(state.tensor, target_state)
    state = states.MatrixState.xstate(3)
    target_state = np.ones((8, 8)) / 8
    np.testing.assert_allclose(state.tensor, target_state)
    qibo.set_backend(original_backend)


def test_vector_state_to_density_matrix(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    vector = np.random.random(32) + 1j * np.random.random(32)
    vector = vector / np.sqrt((np.abs(vector) ** 2).sum())
    state = states.VectorState.from_tensor(vector)
    mstate = state.to_density_matrix()
    target_matrix = np.outer(vector, vector.conj())
    np.testing.assert_allclose(mstate.tensor, target_matrix)
    state = states.MatrixState.from_tensor(target_matrix)
    with pytest.raises(RuntimeError):
        state.to_density_matrix()
    qibo.set_backend(original_backend)


def test_vector_state_tracout():
    from qibo import gates
    state = states.VectorState.zstate(3)
    mgate = gates.M(0)
    qubits = [0]
    assert state._traceout(qubits=qubits) == [1, 2]
    assert state._traceout(measurement_gate=mgate) == (1, 2)
    with pytest.raises(ValueError):
        unmeasured = state._traceout()
    with pytest.raises(ValueError):
        unmeasured = state._traceout(qubits, mgate)


def test_matrix_state_tracout():
    from qibo import gates
    state = states.MatrixState.zstate(2)
    mgate = gates.M(0)
    mgate.density_matrix = True
    qubits = [0]
    assert state._traceout(qubits=qubits) == "abab->a"
    assert state._traceout(measurement_gate=mgate) == "abab->a"


@pytest.mark.parametrize("state_type", ["VectorState", "MatrixState"])
def test_state_probabilities(backend, state_type):
    # TODO: Test this both for `VectorState` and `MatrixState`
    state = getattr(states, state_type).xstate(4)
    probs = state.probabilities(qubits=[0, 1])
    target_probs = np.ones((2, 2)) / 4
    np.testing.assert_allclose(probs, target_probs)


@pytest.mark.parametrize("registers", [None, {"a": (0,), "b": (2,)}])
def test_state_measure(registers):
    from qibo import gates
    state = states.VectorState.zstate(4)
    mgate = gates.M(0, 2)
    assert state.measurements is None
    with pytest.raises(RuntimeError):
        samples = state.samples()
    state.measure(mgate, nshots=100, registers=registers)
    target_samples = np.zeros((100, 2))
    np.testing.assert_allclose(state.samples(), target_samples)
    assert state.frequencies() == {"00": 100}
    if registers is not None:
        target_freqs = {"a": {"0": 100}, "b": {"0": 100}}
    else:
        target_freqs = {"00": 100}
    assert state.frequencies(registers=True) == target_freqs


@pytest.mark.parametrize("registers", [None, {"a": (0,), "b": (2,)}])
def test_state_set_measurements(registers):
    from qibo import gates
    state = states.VectorState.zstate(3)
    samples = np.array(50 * [0] + 50 * [1])
    state.set_measurements([0, 2], samples, registers)
    target_samples = np.array(50 * [[0, 0]] + 50 * [[0, 1]])
    np.testing.assert_allclose(state.samples(), target_samples)
    assert state.frequencies() == {"00": 50, "01": 50}
    if registers is not None:
        target_freqs = {"a": {"0": 100}, "b": {"0": 50, "1": 50}}
    else:
        target_freqs = {"00": 50, "01": 50}
    assert state.frequencies(registers=True) == target_freqs


def test_state_apply_bitflips():
    state = states.VectorState.zstate(3)
    with pytest.raises(RuntimeError):
        state.apply_bitflips(0.1)
    # Bitflips are tested in measurement tests


def test_distributed_state_init():
    from qibo.models import Circuit
    circuit = Circuit(4, accelerators={"/GPU:0": 2})
    state = states.DistributedState(circuit)
    assert state.circuit == circuit
    circuit = Circuit(4)
    with pytest.raises(TypeError):
        state = states.DistributedState(circuit)
    with pytest.raises(NotImplementedError):
        state.tensor = [0, 0]


@pytest.mark.parametrize("init_type", ["z", "x"])
def test_distributed_state_constructors(init_type):
    """Tests `zstate` and `xstate` for `DistributedState`."""
    from qibo.models import Circuit
    from qibo.tensorflow.distutils import DistributedQubits
    c = Circuit(6, {"/GPU:0": 2, "/GPU:1": 2})
    c.queues.qubits = DistributedQubits(range(c.nglobal), c.nqubits) # pylint: disable=E1101
    state = getattr(states.DistributedState, f"{init_type}state")(c)

    final_state = state.numpy()
    if init_type == "z":
        target_state = np.zeros_like(final_state)
        target_state[0] = 1
    else:
        target_state = np.ones_like(final_state) / 8
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("nqubits", [5, 6])
def test_user_initialization(nqubits):
    import itertools
    from qibo.models import Circuit
    from qibo.tensorflow.distutils import DistributedQubits
    target_state = (np.random.random(2 ** nqubits) +
                    1j * np.random.random(2 ** nqubits))
    c = Circuit(nqubits, {"/GPU:0": 2, "/GPU:1": 2})
    c.queues.qubits = DistributedQubits(range(c.nglobal), c.nqubits) # pylint: disable=E1101
    state = states.DistributedState.from_tensor(target_state, c)
    np.testing.assert_allclose(state.tensor, target_state)

    target_state = target_state.reshape(nqubits * (2,))
    for i, s in enumerate(itertools.product([0, 1], repeat=c.nglobal)): # pylint: disable=E1101
        target_piece = target_state[s]
        np.testing.assert_allclose(state.pieces[i], target_piece.ravel())


def test_distributed_state_copy():
    from qibo.models import Circuit
    from qibo.tensorflow.distutils import DistributedQubits
    c = Circuit(4, {"/GPU:0": 2, "/GPU:1": 2})
    c.queues.qubits = DistributedQubits(range(c.nglobal), c.nqubits) # pylint: disable=E1101
    state = states.DistributedState.zstate(c)
    cstate = state.copy()
    np.testing.assert_allclose(state.tensor, cstate.tensor)
