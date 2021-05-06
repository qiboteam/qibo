"""Tests methods defined in `qibo/abstractions/states.py`."""
import pytest
from qibo.abstractions.states import AbstractState


def test_abstract_state_init():
    AbstractState.__abstractmethods__ = set()
    state = AbstractState(5) # pylint: disable=E0110
    assert state.nqubits == 5
    assert len(state) == 32
    state = AbstractState() # pylint: disable=E0110
    with pytest.raises(AttributeError):
        nqubits = state.nqubits
    with pytest.raises(AttributeError):
        nstates = len(state)


@pytest.mark.parametrize("nqubits", [None, 2])
def test_abstract_state_tensor_getter_setter(nqubits):
    AbstractState.__abstractmethods__ = set()
    state = AbstractState(nqubits) # pylint: disable=E0110
    with pytest.raises(AttributeError):
        tensor = state.tensor
    state.tensor = [0, 0, 0, 0]
    assert state.tensor == [0, 0, 0, 0]
    assert state.nqubits == 2
    with pytest.raises(ValueError):
        state.tensor = [0, 0]


def test_abstract_state_from_tensor():
    AbstractState.__abstractmethods__ = set()
    state = AbstractState.from_tensor([0, 1]) # pylint: disable=E0110
    assert state.nqubits == 1
    assert state.tensor == [0, 1]


def test_abstract_state_getitem():
    AbstractState.__abstractmethods__ = set()
    state = AbstractState.from_tensor([0, 1, 0, 1]) # pylint: disable=E0110
    assert state[1] == 1
    assert state[2] == 0
    with pytest.raises(IndexError):
        state[5]


def test_abstract_state_copy():
    AbstractState.__abstractmethods__ = set()
    state = AbstractState.from_tensor([0, 1])
    cstate = state.copy()
    assert cstate.nqubits == state.nqubits
    assert len(cstate) == len(state)
    assert cstate.tensor == state.tensor
    assert cstate.measurements == state.measurements
