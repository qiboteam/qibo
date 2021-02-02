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
    qibo.set_backend(original_backend)


def test_vector_state_tracout():
    pass


def test_state_probabilities():
    # TODO: Test this both for `VectorState` and `MatrixState`
    pass


def test_state_measure():
    # TODO: Also test `state.samples` and `state.frequencies` here
    pass


def test_state_set_measurements():
    # TODO: Also test `state.samples` and `state.frequencies` here
    pass


def test_state_apply_bitflips():
    pass


def test_distributed_state_init():
    pass


def test_distributed_state_circuit_setter():
    pass


def test_distributed_state_tensor():
    pass


def test_distributed_state_constructors():
    """Tests `from_tensor`, `zstate` and `xstate` for `DistributedState`."""
    pass


def test_distributed_state_copy():
    pass
