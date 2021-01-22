"""
Testing Tensorflow gates.
"""
import numpy as np
import pytest
import qibo
from qibo import gates
from qibo.models import Circuit
from qibo.tests import utils

_BACKENDS = ["custom", "defaulteinsum", "matmuleinsum",
             "numpy_defaulteinsum", "numpy_matmuleinsum"]
_DEVICE_BACKENDS = [("custom", None), ("matmuleinsum", None),
                    ("custom", {"/GPU:0": 1, "/GPU:1": 1})]


@pytest.mark.parametrize("backend", _BACKENDS)
def test_generalized_fsim_error(backend):
    """Check GenerelizedfSim gate raises error for wrong unitary shape."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    phi = np.random.random()
    rotation = utils.random_numpy_complex((4, 4))
    c = Circuit(2)
    with pytest.raises(ValueError):
        c.add(gates.GeneralizedfSim(0, 1, rotation, phi))
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_unitary_bad_shape(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    matrix = np.random.random((8, 8))
    with pytest.raises(ValueError):
        gate = gates.Unitary(matrix, 0, 1)

    if backend == "custom":
        with pytest.raises(NotImplementedError):
            gate = gates.Unitary(matrix, 0, 1, 2)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_unitary_various_type_initialization(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    if "numpy" in backend:
        matrix = utils.random_numpy_complex((4, 4))
    else:
        import tensorflow as tf
        matrix = utils.random_tensorflow_complex((4, 4), dtype=tf.float64)
    gate = gates.Unitary(matrix, 0, 1)
    with pytest.raises(TypeError):
        gate = gates.Unitary("abc", 0, 1)
    qibo.set_backend(original_backend)


def test_control_unitary_error():
    matrix = np.random.random((4, 4))
    gate = gates.Unitary(matrix, 0, 1)
    with pytest.raises(ValueError):
        unitary = gate.control_unitary(np.random.random((16, 16)))


@pytest.mark.parametrize("backend", _BACKENDS)
def test_construct_unitary_controlled_by(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    rotation = np.array([[np.cos(theta / 2.0), -np.sin(theta / 2.0)],
                         [np.sin(theta / 2.0), np.cos(theta / 2.0)]])
    target_matrix = np.eye(4, dtype=rotation.dtype)
    target_matrix[2:, 2:] = rotation
    gate = gates.RY(0, theta).controlled_by(1)
    np.testing.assert_allclose(gate.unitary, target_matrix)

    gate = gates.RY(0, theta).controlled_by(1, 2)
    with pytest.raises(NotImplementedError):
        unitary = gate.unitary
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_construct_unitary_errors(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    gate = gates.M(0)
    with pytest.raises(ValueError):
        matrix = gate.unitary

    pairs = list((i, i + 1) for i in range(0, 5, 2))
    theta = 2 * np.pi * np.random.random(6)
    gate = gates.VariationalLayer(range(6), pairs, gates.RY, gates.CZ, theta)
    with pytest.raises(ValueError):
        gate.construct_unitary()
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_collapse_after_measurement(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    qubits = [0, 2, 3]

    c1 = Circuit(5)
    c1.add((gates.H(i) for i in range(5)))
    c1.add(gates.M(*qubits))
    result = c1(nshots=1)
    c2 = Circuit(5)
    bitstring = result.samples(binary=True)[0]
    c2.add(gates.Collapse(*qubits, result=bitstring))
    c2.add((gates.H(i) for i in range(5)))
    final_state = c2(initial_state=c1.final_state)

    ct = Circuit(5)
    for i, r in zip(qubits, bitstring):
        if r:
            ct.add(gates.X(i))
    ct.add((gates.H(i) for i in qubits))
    target_state = ct()
    np.testing.assert_allclose(final_state, target_state, atol=1e-15)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_noise_channel_repeated(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    thetas = np.random.random(4)
    probs = 0.1 * np.random.random([4, 3]) + 0.2
    gatelist = [gates.X, gates.Y, gates.Z]

    c = Circuit(4)
    c.add((gates.RY(i, t) for i, t in enumerate(thetas)))
    c.add((gates.PauliNoiseChannel(i, px, py, pz, seed=123)
           for i, (px, py, pz) in enumerate(probs)))
    final_state = c(nshots=40)

    np.random.seed(123)
    target_state = []
    for _ in range(40):
        noiseless_c = Circuit(4)
        noiseless_c.add((gates.RY(i, t) for i, t in enumerate(thetas)))
        for i, ps in enumerate(probs):
            for p, gate in zip(ps, gatelist):
                if np.random.random() < p:
                    noiseless_c.add(gate(i))
        target_state.append(noiseless_c())
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_reset_channel_repeated(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    initial_state = utils.random_numpy_state(5)
    c = Circuit(5)
    c.add(gates.ResetChannel(2, p0=0.3, p1=0.3, seed=123))
    final_state = c(np.copy(initial_state), nshots=30)

    np.random.seed(123)
    target_state = []
    for _ in range(30):
        noiseless_c = Circuit(5)
        if np.random.random() < 0.3:
            noiseless_c.add(gates.Collapse(2))
        if np.random.random() < 0.3:
            noiseless_c.add(gates.Collapse(2))
            noiseless_c.add(gates.X(2))
        target_state.append(noiseless_c(np.copy(initial_state)))
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_thermal_relaxation_channel_repeated(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    initial_state = utils.random_numpy_state(5)
    c = Circuit(5)
    c.add(gates.ThermalRelaxationChannel(4, t1=1.0, t2=0.6, time=0.8,
                                         excited_population=0.8, seed=123))
    final_state = c(np.copy(initial_state), nshots=30)

    pz, p0, p1 = c.queue[0].calculate_probabilities(1.0, 0.6, 0.8, 0.8)
    np.random.seed(123)
    target_state = []
    for _ in range(30):
        noiseless_c = Circuit(5)
        if np.random.random() < pz:
            noiseless_c.add(gates.Z(4))
        if np.random.random() < p0:
            noiseless_c.add(gates.Collapse(4))
        if np.random.random() < p1:
            noiseless_c.add(gates.Collapse(4))
            noiseless_c.add(gates.X(4))
        target_state.append(noiseless_c(np.copy(initial_state)))
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)
