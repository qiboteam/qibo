import platform
import sys

import numpy as np
import pytest

from qibo import construct_backend, gates, list_available_backends, set_backend
from qibo.backends import MetaBackend

from .conftest import AVAILABLE_BACKENDS

####################### Test `matrix` #######################
GATES = [
    ("H", (0,), np.array([[1, 1], [1, -1]]) / np.sqrt(2)),
    ("X", (0,), np.array([[0, 1], [1, 0]])),
    ("Y", (0,), np.array([[0, -1j], [1j, 0]])),
    ("Z", (1,), np.array([[1, 0], [0, -1]])),
    ("S", (2,), np.array([[1, 0], [0, 1j]])),
    ("T", (2,), np.array([[1, 0], [0, np.exp(1j * np.pi / 4.0)]])),
    (
        "CNOT",
        (0, 1),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
    ),
    ("CZ", (1, 3), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])),
    (
        "SWAP",
        (2, 4),
        np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
    ),
    (
        "FSWAP",
        (2, 4),
        np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -1]]),
    ),
    (
        "TOFFOLI",
        (1, 2, 3),
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        ),
    ),
]


@pytest.mark.parametrize("gate,qubits,target_matrix", GATES)
def test_matrix(backend, gate, qubits, target_matrix):
    gate = getattr(gates, gate)(*qubits)
    backend.assert_allclose(gate.matrix(backend), target_matrix)


GATES = [
    (
        "RX",
        lambda x: np.array(
            [
                [np.cos(x / 2.0), -1j * np.sin(x / 2.0)],
                [-1j * np.sin(x / 2.0), np.cos(x / 2.0)],
            ]
        ),
    ),
    (
        "RY",
        lambda x: np.array(
            [[np.cos(x / 2.0), -np.sin(x / 2.0)], [np.sin(x / 2.0), np.cos(x / 2.0)]]
        ),
    ),
    ("RZ", lambda x: np.diag([np.exp(-1j * x / 2.0), np.exp(1j * x / 2.0)])),
    ("U1", lambda x: np.diag([1, np.exp(1j * x)])),
    ("CU1", lambda x: np.diag([1, 1, 1, np.exp(1j * x)])),
]


@pytest.mark.parametrize("gate,target_matrix", GATES)
def test_matrix_rotations(backend, gate, target_matrix):
    """Check that `_construct_unitary` method constructs the proper matrix."""
    theta = 0.1234
    if gate == "CU1":
        gate = getattr(gates, gate)(0, 1, theta)
    else:
        gate = getattr(gates, gate)(0, theta)
    backend.assert_allclose(gate.matrix(backend), target_matrix(theta))
    backend.assert_allclose(gate.matrix(backend), target_matrix(theta))


def test_plus_density_matrix(backend):
    matrix = backend.plus_density_matrix(4)
    target_matrix = np.ones((16, 16)) / 16
    backend.assert_allclose(matrix, target_matrix)


def test_set_backend_error():
    with pytest.raises(ValueError):
        set_backend("non-existing-backend")


def test_metabackend_load_error():
    with pytest.raises(ValueError):
        MetaBackend.load("non-existing-backend")


def test_construct_backend(backend):
    assert isinstance(
        construct_backend(backend.name, platform=backend.platform), backend.__class__
    )


def test_list_available_backends():
    qulacs = (
        False if platform.system() == "Darwin" and sys.version_info[1] == 9 else True
    )
    available_backends = {
        "numpy": True,
        "qulacs": qulacs,
        "qibojit": {
            platform: any(platform in backend for backend in AVAILABLE_BACKENDS)
            for platform in ["numba", "cupy", "cuquantum"]
        },
        "qibolab": False,
        "qibo-cloud-backends": False,
        "qibotn": {"cutensornet": False, "qmatchatea": False, "qutensornet": True},
        # "qiboml": {"tensorflow": False, "pytorch": True},
    }
    assert available_backends == list_available_backends(
        "qibojit",
        "qibolab",
        "qibo-cloud-backends",
        "qibotn",  # "qiboml"
    )
