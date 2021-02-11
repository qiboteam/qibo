"""Tests methods defined in `qibo/core/states.py`."""
import pytest
import numpy as np
import qibo
from qibo import K
from qibo.core import states


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


@pytest.mark.parametrize("init_type", ["zero", "plus"])
def test_distributed_state_constructors(init_type):
    """Tests `zero_state` and `plus_state` for `DistributedState`."""
    from qibo.models import Circuit
    from qibo.tensorflow.distutils import DistributedQubits
    c = Circuit(6, {"/GPU:0": 2, "/GPU:1": 2})
    c.queues.qubits = DistributedQubits(range(c.nglobal), c.nqubits) # pylint: disable=E1101
    state = getattr(states.DistributedState, f"{init_type}_state")(c)

    final_state = state.numpy()
    if init_type == "zero":
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
    state = states.DistributedState.zero_state(c)
    cstate = state.copy()
    np.testing.assert_allclose(state.tensor, cstate.tensor)
