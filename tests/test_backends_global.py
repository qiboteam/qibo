import networkx as nx
import pytest

import qibo
from qibo import Circuit, matrices
from qibo.backends import _Global, get_backend
from qibo.backends.numpy import NumpyBackend
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.pipeline import Passes
from qibo.transpiler.placer import Random
from qibo.transpiler.router import Sabre
from qibo.transpiler.unroller import NativeGates, Unroller


def test_set_get_backend():
    qibo.set_backend("numpy")
    assert str(qibo.get_backend()) == "numpy"
    assert qibo.get_backend().name == "numpy"


def test_set_precision():
    import numpy as np

    assert qibo.get_precision() == "double"
    qibo.set_precision("single")
    assert matrices.I.dtype == np.complex64
    assert qibo.get_precision() == "single"
    qibo.set_precision("double")
    assert matrices.I.dtype == np.complex128
    assert qibo.get_precision() == "double"
    with pytest.raises(ValueError):
        qibo.set_precision("test")


def test_set_device():
    qibo.set_backend("numpy")
    qibo.set_device("/CPU:0")
    assert qibo.get_device() == "/CPU:0"
    with pytest.raises(ValueError):
        qibo.set_device("test")
    with pytest.raises(ValueError):
        qibo.set_device("/GPU:0")


def test_set_threads():
    with pytest.raises(ValueError):
        qibo.set_threads(-2)
    with pytest.raises(TypeError):
        qibo.set_threads("test")

    qibo.set_backend("numpy")
    assert qibo.get_threads() == 1
    with pytest.raises(ValueError):
        qibo.set_threads(10)


def test_set_shot_batch_size():
    original_batch_size = qibo.get_batch_size()
    qibo.set_batch_size(1024)
    assert qibo.get_batch_size() == 1024
    from qibo.config import SHOT_BATCH_SIZE

    assert SHOT_BATCH_SIZE == 1024
    with pytest.raises(TypeError):
        qibo.set_batch_size("test")
    with pytest.raises(ValueError):
        qibo.set_batch_size(-10)
    with pytest.raises(ValueError):
        qibo.set_batch_size(2**35)
    qibo.set_batch_size(original_batch_size)


def test_set_metropolis_threshold():
    original_threshold = qibo.get_metropolis_threshold()
    qibo.set_metropolis_threshold(100)
    assert qibo.get_metropolis_threshold() == 100
    from qibo.config import SHOT_METROPOLIS_THRESHOLD

    assert SHOT_METROPOLIS_THRESHOLD == 100
    with pytest.raises(TypeError):
        qibo.set_metropolis_threshold("test")
    with pytest.raises(ValueError):
        qibo.set_metropolis_threshold(-10)
    qibo.set_metropolis_threshold(original_threshold)


def test_circuit_execution():
    qibo.set_backend("numpy")
    circuit = Circuit(2)
    circuit.add(qibo.gates.H(0))
    circuit()
    circuit.unitary()


def test_gate_matrix():
    qibo.set_backend("numpy")
    gate = qibo.gates.H(0)
    gate.matrix


def test_check_backend(backend):
    # testing when backend is not None
    test = qibo.backends._check_backend(backend)

    assert test.name == backend.name
    assert test.__class__ == backend.__class__

    # testing when backend is None
    test = None
    test = qibo.backends._check_backend(test)
    target = get_backend()

    assert test.name == target.name
    assert test.__class__ == target.__class__


def _star_connectivity():
    Q = [i for i in range(5)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [(Q[i], Q[2]) for i in range(5) if i != 2]
    chip.add_edges_from(graph_list)
    return chip


def test_set_get_transpiler():
    connectivity = _star_connectivity()
    transpiler = Passes(
        connectivity=connectivity,
        passes=[
            Preprocessing(),
            Random(seed=0),
            Sabre(),
            Unroller(NativeGates.default()),
        ],
    )

    qibo.set_transpiler(transpiler)
    assert qibo.get_transpiler() == transpiler
    assert qibo.get_transpiler_name() == str(transpiler)


def test_default_transpiler_sim():
    backend = NumpyBackend()
    assert (
        backend.natives is None
        and backend.connectivity is None
        and backend.qubits is None
    )


CONNECTIVITY = [
    [("A1", "A2"), ("A2", "A3"), ("A3", "A4"), ("A4", "A5")],
    [("A1", "A2")],
    [],
]


@pytest.mark.parametrize("connectivity", CONNECTIVITY)
def test_default_transpiler_hw(connectivity):
    class TempBackend(NumpyBackend):
        def __init__(self):
            super().__init__()
            self.name = "tempbackend"

        @property
        def qubits(self):
            return ["A1", "A2", "A3", "A4", "A5"]

        @property
        def connectivity(self):
            return connectivity

        @property
        def natives(self):
            return ["CZ", "GPI2"]

    backend = TempBackend()
    _Global._backend = backend
    transpiler = _Global.transpiler()

    assert list(transpiler.connectivity.nodes) == ["A1", "A2", "A3", "A4", "A5"]
    assert list(transpiler.connectivity.edges) == connectivity
    assert (
        NativeGates.CZ in transpiler.native_gates
        and NativeGates.GPI2 in transpiler.native_gates
    )
