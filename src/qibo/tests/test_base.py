"""
Test imports and basic functionality that is indepedent of calculation backend.
"""
import numpy as np
import pytest
import qibo
from qibo.models import *
from qibo.gates import *



def test_importing_full_qibo():
    """Checks accessing `models` and `gates` from `qibo`."""
    import qibo
    c = qibo.models.Circuit(2)
    c.add(qibo.gates.H(0))
    assert c.nqubits == 2
    assert c.depth == 1
    assert c.ngates == 1


def test_importing_qibo_modules():
    """Checks importing `models` and `gates` from qibo."""
    from qibo import models, gates
    c = models.Circuit(2)
    c.add(gates.H(0))
    assert c.nqubits == 2
    assert c.depth == 1
    assert c.ngates == 1


def test_gate_types():
    """Check ``BaseCircuit.gate_types`` property."""
    import collections
    c = Circuit(3)
    c.add(H(0))
    c.add(H(1))
    c.add(X(2))
    c.add(CNOT(0, 2))
    c.add(CNOT(1, 2))
    c.add(TOFFOLI(0, 1, 2))
    target_counter = collections.Counter({"h": 2, "x": 1, "cx": 2, "ccx": 1})
    assert target_counter == c.gate_types


def test_gates_of_type():
    """Check ``BaseCircuit.gates_of_type`` method."""
    c = Circuit(3)
    c.add(H(0))
    c.add(H(1))
    c.add(CNOT(0, 2))
    c.add(X(1))
    c.add(CNOT(1, 2))
    c.add(TOFFOLI(0, 1, 2))
    c.add(H(2))
    h_gates = c.gates_of_type(H)
    cx_gates = c.gates_of_type("cx")
    assert h_gates == [(0, c.queue[0]), (1, c.queue[1]), (6, c.queue[6])]
    assert cx_gates == [(2, c.queue[2]), (4, c.queue[4])]
    with pytest.raises(TypeError):
        c.gates_of_type(5)


def test_summary():
    """Check ``BaseCircuit.summary()`` method."""
    c = Circuit(3)
    c.add(H(0))
    c.add(H(1))
    c.add(CNOT(0, 2))
    c.add(CNOT(1, 2))
    c.add(TOFFOLI(0, 1, 2))
    c.add(H(2))
    target_summary = "\n".join(["Circuit depth = 5",
                                "Total number of gates = 6",
                                "Number of qubits = 3",
                                "Most common gates:",
                                "h: 3", "cx: 2", "ccx: 1"])
    assert c.summary() == target_summary


@pytest.mark.parametrize("deep", [False, True])
def test_circuit_copy(deep):
    """Check that ``circuit.copy()`` copies gates properly."""
    c1 = Circuit(2)
    c1.add([H(0), H(1), CNOT(0, 1)])
    c2 = c1.copy(deep)
    assert c2.depth == c1.depth
    assert c2.ngates == c1.ngates
    assert c2.nqubits == c1.nqubits
    for g1, g2 in zip(c1.queue, c2.queue):
        if deep:
            assert g1.__class__ == g2.__class__
            assert g1.target_qubits == g2.target_qubits
            assert g1.control_qubits == g2.control_qubits
        else:
            assert g1 is g2


def test_circuit_copy_with_measurements():
    """Check that ``circuit.copy()`` copies measurements properly."""
    c1 = Circuit(4)
    c1.add([H(0), H(3), CNOT(0, 2)])
    c1.add(M(0, 1, register_name="a"))
    c1.add(M(3, register_name="b"))
    c2 = c1.copy()

    assert c2.measurement_gate is c1.measurement_gate
    assert c2.measurement_tuples == {"a": (0, 1), "b": (3,)}


def test_base_gate_errors():
    """Check errors in ``base.gates.Gate`` for coverage."""
    gate = H(0)
    with pytest.raises(ValueError):
        nqubits = gate.nqubits
    with pytest.raises(ValueError):
        nstates = gate.nstates
    # Access nstates
    gate2 = H(0)
    gate2.nqubits = 3
    _ = gate2.nstates

    with pytest.raises(ValueError):
        gate.nqubits = 2
        gate.nqubits = 3
    with pytest.raises(RuntimeError):
        cgate = gate.controlled_by(1)
    with pytest.raises(RuntimeError):
        gate = H(0).controlled_by(1).controlled_by(2)


@pytest.mark.parametrize("precision", ["single", "double"])
def test_state_precision(precision):
    """Check ``set_precision`` in state dtype."""
    import qibo
    import tensorflow as tf
    original_precision = qibo.get_precision()
    qibo.set_precision(precision)
    c1 = Circuit(2)
    c1.add([H(0), H(1)])
    final_state = c1()
    if precision == "single":
        expected_dtype = tf.complex64
    else:
        expected_dtype = tf.complex128
    assert final_state.dtype == expected_dtype
    qibo.set_precision(original_precision)


@pytest.mark.parametrize("precision", ["single", "double"])
def test_precision_dictionary(precision):
    """Check if ``set_precision`` changes the ``DTYPES`` dictionary."""
    import qibo
    import tensorflow as tf
    from qibo.config import DTYPES
    original_precision = qibo.get_precision()
    qibo.set_precision(precision)
    if precision == "single":
        assert DTYPES.get("DTYPECPX") == tf.complex64
    else:
        assert DTYPES.get("DTYPECPX") == tf.complex128
    qibo.set_precision(original_precision)


def test_matrices_dtype():
    """Check if ``set_precision`` changes matrices types."""
    import qibo
    original_precision = qibo.get_precision()
    # Check that matrices can be imported
    from qibo import matrices
    assert matrices.I.dtype == np.complex128
    np.testing.assert_allclose(matrices.I, np.eye(2))
    # Check that matrices precision is succesfully switched
    qibo.set_precision("single")
    assert matrices.H.dtype == np.complex64
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    np.testing.assert_allclose(matrices.H, H)
    qibo.set_precision("double")
    # Check that ``qibo.matrices`` also works.
    np.testing.assert_allclose(qibo.matrices.H, H)
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                     [0, 0, 0, 1], [0, 0, 1, 0]])
    np.testing.assert_allclose(qibo.matrices.CNOT, CNOT)
    qibo.set_precision(original_precision)


def test_modifying_matrices_error():
    """Check that modifying matrices raises ``AttributeError``."""
    from qibo import matrices
    with pytest.raises(AttributeError):
        matrices.I = np.zeros((2, 2))


@pytest.mark.parametrize("backend", ["custom", "defaulteinsum", "matmuleinsum"])
def test_set_backend(backend):
    """Check ``set_backend`` for switching gate backends."""
    import qibo
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    from qibo import gates
    assert qibo.get_backend() == backend
    if backend == "custom":
        from qibo.tensorflow import cgates as custom_gates
        assert isinstance(gates.H(0), custom_gates.TensorflowGate)
    else:
        from qibo.tensorflow import gates as native_gates
        from qibo.tensorflow import einsum
        einsums = {"defaulteinsum": einsum.DefaultEinsum,
                   "matmuleinsum": einsum.MatmulEinsum}
        h = gates.H(0)
        assert isinstance(h, native_gates.TensorflowGate)
        assert isinstance(h.einsum, einsums[backend]) # pylint: disable=no-member
    qibo.set_backend(original_backend)


def test_switcher_errors():
    """Check set precision and backend errors."""
    import qibo
    with pytest.raises(RuntimeError):
        qibo.set_precision('test')
    with pytest.raises(RuntimeError):
        qibo.set_backend('test')


def test_switcher_warnings():
    """Check set precision and backend warnings."""
    import qibo
    from qibo import gates
    g = gates.H(0)
    qibo.set_precision("double")
    with pytest.warns(RuntimeWarning):
        qibo.set_precision("single")
        qibo.set_precision("double")
    with pytest.warns(RuntimeWarning):
        qibo.set_backend("matmuleinsum")
        qibo.set_backend("custom")


def test_set_device():
    """Check device switcher and errors in device name."""
    import qibo
    original_device = qibo.get_device()
    qibo.set_device("/CPU:0")
    with pytest.raises(ValueError):
        qibo.set_device("test")
    with pytest.raises(ValueError):
        qibo.set_device("/TPU:0")
    with pytest.raises(ValueError):
        qibo.set_device("/gpu:10")
    with pytest.raises(ValueError):
        qibo.set_device("/GPU:10")
    qibo.set_device(original_device)
