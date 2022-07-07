import pytest
import numpy as np
from qibo import gates


####################### Test `asmatrix` #######################
GATES = [
    ("H", (0,), np.array([[1, 1], [1, -1]]) / np.sqrt(2)),
    ("X", (0,), np.array([[0, 1], [1, 0]])),
    ("Y", (0,), np.array([[0, -1j], [1j, 0]])),
    ("Z", (1,), np.array([[1, 0], [0, -1]])),
    ("S", (2,), np.array([[1, 0], [0, 1j]])),
    ("T", (2,), np.array([[1, 0], [0, np.exp(1j * np.pi / 4.0)]])),
    ("CNOT", (0, 1), np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                               [0, 0, 0, 1], [0, 0, 1, 0]])),
    ("CZ", (1, 3), np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                             [0, 0, 1, 0], [0, 0, 0, -1]])),
    ("SWAP", (2, 4), np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                               [0, 1, 0, 0], [0, 0, 0, 1]])),
    ("FSWAP", (2, 4), np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                               [0, 1, 0, 0], [0, 0, 0, -1]])),
    ("TOFFOLI", (1, 2, 3), np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 0, 0, 1, 0]]))
]
@pytest.mark.parametrize("gate,qubits,target_matrix", GATES)
def test_asmatrix(backend, gate, qubits, target_matrix):
    gate = getattr(gates, gate)(*qubits)
    backend.assert_allclose(gate.asmatrix(backend), target_matrix)


GATES = [
    ("RX", lambda x: np.array([[np.cos(x / 2.0), -1j * np.sin(x / 2.0)],
                               [-1j * np.sin(x / 2.0), np.cos(x / 2.0)]])),
    ("RY", lambda x: np.array([[np.cos(x / 2.0), -np.sin(x / 2.0)],
                               [np.sin(x / 2.0), np.cos(x / 2.0)]])),
    ("RZ", lambda x: np.diag([np.exp(-1j * x / 2.0), np.exp(1j * x / 2.0)])),
    ("U1", lambda x: np.diag([1, np.exp(1j * x)])),
    ("CU1", lambda x: np.diag([1, 1, 1, np.exp(1j * x)]))
]
@pytest.mark.parametrize("gate,target_matrix", GATES)
def test_asmatrix_rotations(backend, gate, target_matrix):
    """Check that `_construct_unitary` method constructs the proper matrix."""
    theta = 0.1234
    if gate == "CU1":
        gate = getattr(gates, gate)(0, 1, theta)
    else:
        gate = getattr(gates, gate)(0, theta)
    backend.assert_allclose(gate.asmatrix(backend), target_matrix(theta))
    backend.assert_allclose(gate.asmatrix(backend), target_matrix(theta))


def test_control_matrix(backend):
    theta = 0.1234
    rotation = np.array([[np.cos(theta / 2.0), -np.sin(theta / 2.0)],
                         [np.sin(theta / 2.0), np.cos(theta / 2.0)]])
    target_matrix = np.eye(4, dtype=rotation.dtype)
    target_matrix[2:, 2:] = rotation
    gate = gates.RY(0, theta).controlled_by(1)
    backend.assert_allclose(gate.asmatrix(backend), target_matrix)

    gate = gates.RY(0, theta).controlled_by(1, 2)
    with pytest.raises(NotImplementedError):
        matrix = backend.control_matrix(gate)


def test_control_matrix_unitary(backend):
    u = np.random.random((2, 2))
    gate = gates.Unitary(u, 0).controlled_by(1)
    matrix = backend.control_matrix(gate)
    target_matrix = np.eye(4, dtype=backend.dtype)
    target_matrix[2:, 2:] = u
    backend.assert_allclose(matrix, target_matrix)

    u = np.random.random((16, 16))
    gate = gates.Unitary(u, 0, 1, 2, 3).controlled_by(4)
    with pytest.raises(ValueError):
        matrix = backend.control_matrix(gate)


def test_plus_density_matrix(backend):
    matrix = backend.plus_density_matrix(4)
    target_matrix = np.ones((16, 16)) / 16
    backend.assert_allclose(matrix, target_matrix)
