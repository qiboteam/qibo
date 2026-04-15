import sys

import numpy as np
import pytest

from qibo import Circuit, construct_backend, gates, list_available_backends, set_backend
from qibo.backends import MetaBackend

from .conftest import AVAILABLE_BACKENDS


def test_validate_nqubits_exceeds_max(backend):
    """Test that state allocation raises ValueError when nqubits exceeds limits."""
    import qibo

    original_sv = qibo.get_max_qubits()
    original_dm = qibo.get_max_qubits_dm()
    try:
        qibo.set_max_qubits(5)
        qibo.set_max_qubits_dm(4)
        # state vector checks use MAX_QUBITS
        with pytest.raises(ValueError, match="state vector"):
            backend.zero_state(10)
        with pytest.raises(ValueError, match="set_max_qubits\\("):
            backend.plus_state(10)
        with pytest.raises(ValueError, match="state vector"):
            backend.minus_state(10)
        # density matrix checks use MAX_QUBITS_DM
        with pytest.raises(ValueError, match="density matrix"):
            backend.zero_state(10, density_matrix=True)
        with pytest.raises(ValueError, match="set_max_qubits_dm\\("):
            backend.plus_state(10, density_matrix=True)
        with pytest.raises(ValueError, match="density matrix"):
            backend.minus_state(10, density_matrix=True)
        with pytest.raises(ValueError, match="density matrix"):
            backend.maximally_mixed_state(10)
        # state vector within SV limit but exceeding DM limit should work
        state = backend.zero_state(5)
        assert state.shape == (2**5,)
        # density matrix within DM limit should work
        state_dm = backend.zero_state(4, density_matrix=True)
        assert state_dm.shape == (2**4, 2**4)
        # density matrix exceeding DM limit should fail even if within SV limit
        with pytest.raises(ValueError, match="density matrix"):
            backend.zero_state(5, density_matrix=True)
    finally:
        qibo.set_max_qubits(original_sv)
        qibo.set_max_qubits_dm(original_dm)


def test_validate_nqubits_unlimited(backend):
    """Test that setting -1 disables the qubit limit."""
    import qibo

    original_sv = qibo.get_max_qubits()
    original_dm = qibo.get_max_qubits_dm()
    try:
        qibo.set_max_qubits(-1)
        qibo.set_max_qubits_dm(-1)
        # large qubit counts should not raise (only allocate small states to avoid OOM)
        state = backend.zero_state(10)
        assert state.shape == (2**10,)
        state_dm = backend.zero_state(10, density_matrix=True)
        assert state_dm.shape == (2**10, 2**10)
    finally:
        qibo.set_max_qubits(original_sv)
        qibo.set_max_qubits_dm(original_dm)


def test_validate_nqubits_invalid_type(backend):
    """Test that _validate_nqubits rejects non-integer or non-positive nqubits."""
    with pytest.raises(ValueError, match="positive integer"):
        backend.zero_state(-1)
    with pytest.raises(ValueError, match="positive integer"):
        backend.zero_state(0)


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
    matrix = backend.plus_state(4, density_matrix=True)
    target_matrix = backend.ones((16, 16)) / 16
    backend.assert_allclose(matrix, target_matrix)


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("nqubits", [2, 5])
def test_minus_state(backend, nqubits, density_matrix):
    state = backend.minus_state(nqubits, density_matrix=density_matrix)

    target = Circuit(nqubits, density_matrix=density_matrix)
    target.add(gates.X(qubit) for qubit in range(nqubits))
    target.add(gates.H(qubit) for qubit in range(nqubits))
    target = backend.execute_circuit(target).state()

    backend.assert_allclose(state, target)


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
    qulacs = False if sys.version_info[1] == 13 else True
    available_backends = {
        "numpy": True,
        "qulacs": qulacs,
        "qibojit": {
            platform: any(platform in backend for backend in AVAILABLE_BACKENDS)
            for platform in ["numba", "cupy", "cuquantum"]
        },
        "qibolab": False,
        "qibo-cloud-backends": False,
        # "qibotn": {"cutensornet": False, "qmatchatea": False, "qutensornet": True},
        # "qiboml": {"tensorflow": False, "pytorch": True},
    }
    assert available_backends == list_available_backends(
        "qibojit",
        "qibolab",
        "qibo-cloud-backends",  # , "qibotn", "qiboml"
    )
