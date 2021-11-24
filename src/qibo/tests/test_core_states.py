"""Tests methods defined in `qibo/core/states.py`."""
import pytest
import numpy as np
from qibo import K
from qibo.core import states


def test_state_shape_and_dtype(backend):
    state = states.VectorState.zero_state(3)
    assert state.shape == (8,)
    assert state.dtype == K.dtypes('DTYPECPX')
    state = states.MatrixState.zero_state(3)
    assert state.shape == (8, 8)
    assert state.dtype == K.dtypes('DTYPECPX')


@pytest.mark.parametrize("nqubits", [None, 2])
def test_vector_state_tensor_setter(backend, nqubits):
    state = states.VectorState(nqubits)
    with pytest.raises(AttributeError):
        tensor = state.tensor
    state.tensor = np.ones(4)
    assert state.nqubits == 2
    K.assert_allclose(state.tensor, np.ones(4))
    K.assert_allclose(np.array(state), np.ones(4))
    K.assert_allclose(state.numpy(), np.ones(4))
    K.assert_allclose(state.state(numpy=True), np.ones(4))
    K.assert_allclose(state.state(numpy=False), np.ones(4))
    with pytest.raises(ValueError):
        state.tensor = np.zeros(5)


@pytest.mark.parametrize("nqubits", [None, 2])
def test_matrix_state_tensor_setter(backend, nqubits):
    # TODO: Fix this
    pass


def test_zero_state_initialization(backend):
    state = states.VectorState.zero_state(4)
    target_state = np.zeros(16)
    target_state[0] = 1
    K.assert_allclose(state.tensor, target_state)
    state = states.MatrixState.zero_state(3)
    target_state = np.zeros((8, 8))
    target_state[0, 0] = 1
    K.assert_allclose(state.tensor, target_state)


def test_plus_state_initialization(backend):
    state = states.VectorState.plus_state(4)
    target_state = np.ones(16) / 4
    K.assert_allclose(state.tensor, target_state)
    state = states.MatrixState.plus_state(3)
    target_state = np.ones((8, 8)) / 8
    K.assert_allclose(state.tensor, target_state)


def test_vector_state_to_density_matrix(backend):
    vector = np.random.random(32) + 1j * np.random.random(32)
    vector = vector / np.sqrt((np.abs(vector) ** 2).sum())
    state = states.VectorState.from_tensor(vector)
    mstate = state.to_density_matrix()
    target_matrix = np.outer(vector, vector.conj())
    K.assert_allclose(mstate.tensor, target_matrix)
    state = states.MatrixState.from_tensor(target_matrix)
    with pytest.raises(RuntimeError):
        state.to_density_matrix()


@pytest.mark.parametrize("target", range(5))
def test_state_vector_representation(target):
    from qibo import models, gates
    c = models.Circuit(5)
    c.add(gates.H(target))
    result = c()
    bstring = target * "0" + "1" + (4 - target) * "0"
    target_str = f"(0.70711+0j)|00000> + (0.70711+0j)|{bstring}>"
    assert str(result) == target_str
    assert result.state(symbolic=True) == target_str
    assert result.symbolic(decimals=1) == f"(0.7+0j)|00000> + (0.7+0j)|{bstring}>"
    assert result.symbolic(decimals=2) == f"(0.71+0j)|00000> + (0.71+0j)|{bstring}>"


def test_state_vector_representation_max_terms():
    from qibo import models, gates
    c = models.Circuit(5)
    c.add(gates.H(i) for i in range(5))
    result = c()
    assert result.symbolic(max_terms=3) == "(0.17678+0j)|00000> + (0.17678+0j)|00001> + (0.17678+0j)|00010> + ..."
    assert result.symbolic(max_terms=5) == "(0.17678+0j)|00000> + (0.17678+0j)|00001> + (0.17678+0j)|00010> + (0.17678+0j)|00011> + (0.17678+0j)|00100> + ..."


@pytest.mark.parametrize("state_type", ["VectorState", "MatrixState"])
@pytest.mark.parametrize("use_gate", [False, True])
def test_state_probabilities(backend, state_type, use_gate):
    state = getattr(states, state_type).plus_state(4)
    if use_gate:
        from qibo import gates
        mgate = gates.M(0, 1)
        probs = state.probabilities(measurement_gate=mgate)
    else:
        probs = state.probabilities(qubits=[0, 1])
    target_probs = np.ones((2, 2)) / 4
    K.assert_allclose(probs, target_probs)


def test_state_probabilities_errors():
    from qibo import gates
    state = states.VectorState.zero_state(3)
    mgate = gates.M(0)
    qubits = [0]
    with pytest.raises(ValueError):
        probs = state.probabilities()
    with pytest.raises(ValueError):
        probs = state.probabilities(qubits, mgate)


@pytest.mark.parametrize("registers", [None, {"a": (0,), "b": (2,)}])
def test_state_measure(registers):
    from qibo import gates
    state = states.VectorState.zero_state(4)
    mgate = gates.M(0, 2)
    assert state.measurements is None
    with pytest.raises(RuntimeError):
        samples = state.samples()
    state.measure(mgate, nshots=100, registers=registers)
    target_samples = np.zeros((100, 2))
    K.assert_allclose(state.samples(), target_samples)
    assert state.frequencies() == {"00": 100}
    if registers is not None:
        target_freqs = {"a": {"0": 100}, "b": {"0": 100}}
    else:
        target_freqs = {"00": 100}
    assert state.frequencies(registers=True) == target_freqs


@pytest.mark.parametrize("registers", [None, {"a": (0,), "b": (2,)}])
def test_state_set_measurements(registers):
    from qibo import gates
    state = states.VectorState.zero_state(3)
    samples = K.cast(50 * [0] + 50 * [1], dtype=K.dtypes("DTYPEINT"))
    state.set_measurements([0, 2], samples, registers)
    target_samples = np.array(50 * [[0, 0]] + 50 * [[0, 1]])
    K.assert_allclose(state.samples(), target_samples)
    assert state.frequencies() == {"00": 50, "01": 50}
    if registers is not None:
        target_freqs = {"a": {"0": 100}, "b": {"0": 50, "1": 50}}
    else:
        target_freqs = {"00": 50, "01": 50}
    assert state.frequencies(registers=True) == target_freqs


def test_state_apply_bitflips():
    state = states.VectorState.zero_state(3)
    with pytest.raises(RuntimeError):
        state.apply_bitflips(0.1)
    # Bitflips are tested in measurement tests


@pytest.mark.parametrize("dense", [True, False])
def test_vector_state_expectation(backend, dense):
    from qibo.hamiltonians import XXZ
    ham = XXZ(nqubits=5, delta=0.5, dense=dense)
    matrix = K.to_numpy(ham.matrix)

    state = np.random.random(32) + 1j * np.random.random(32)
    norm = np.sum(np.abs(state) ** 2)
    target_ev = np.sum(state.conj() * matrix.dot(state)).real
    state = states.VectorState.from_tensor(state)

    K.assert_allclose(state.expectation(ham), target_ev)
    K.assert_allclose(state.expectation(ham, True), target_ev / norm)


@pytest.mark.parametrize("dense", [True, False])
def test_matrix_state_expectation(backend, dense):
    from qibo.hamiltonians import TFIM
    ham = TFIM(nqubits=2, h=1.0, dense=dense)
    matrix = K.to_numpy(ham.matrix)

    state = np.random.random((4, 4)) + 1j * np.random.random((4, 4))
    state = state + state.T.conj()
    norm = np.trace(state)
    target_ev = np.trace(matrix.dot(state)).real
    state = states.MatrixState.from_tensor(state)

    K.assert_allclose(state.expectation(ham), target_ev)
    K.assert_allclose(state.expectation(ham, True), target_ev / norm)
