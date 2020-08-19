import re
import numpy as np
import pytest
import cirq
from qibo import gates
from qibo.models import Circuit
from qibo.tests import utils

_QIBO_TO_CIRQ = {"CNOT": "CNOT", "RY": "Ry", "TOFFOLI": "TOFFOLI"}
_ATOL = 1e-6


def assert_gates_equivalent(qibo_gate, cirq_gate):
    """Asserts that qibo gate is equivalent to cirq gate.

    Checks that:
        * Gate type agrees.
        * Target and control qubits agree.
        * Parameter (if applicable) agrees.

    Cirq gate parameters are extracted by parsing the gate string.
    """
    pieces = [x for x in re.split("[()]", str(cirq_gate)) if x]
    if len(pieces) == 2:
        gatename, targets = pieces
        theta = None
    elif len(pieces) == 3:
        gatename, theta, targets = pieces
    else: # pragma: no cover
        # case not tested because it fails
        raise RuntimeError("Cirq gate parsing failed with {}.".format(pieces))

    qubits = list(int(x) for x in targets.replace(" ", "").split(","))
    targets = (qubits.pop(),)
    controls = set(qubits)

    assert _QIBO_TO_CIRQ[qibo_gate.__class__.__name__] == gatename
    assert qibo_gate.target_qubits == targets
    assert set(qibo_gate.control_qubits) == controls
    if theta is not None:
        if "π" in theta:
            theta = np.pi * float(theta.replace("π", ""))
        else: # pragma: no cover
            # case doesn't happen in tests (could remove)
            theta = float(theta)
        np.testing.assert_allclose(theta, qibo_gate.parameter)


def assert_circuit_same_gates(circuit1, circuit2):
    """Asserts that two circuits contain the same gates."""
    for g1, g2 in zip(circuit1.queue, circuit2.queue):
        assert g1.__class__ == g2.__class__
        assert g1.target_qubits == g2.target_qubits
        assert g1.control_qubits == g2.control_qubits
    mg1 = circuit1.measurement_gate
    mg2 = circuit2.measurement_gate
    assert mg1.__class__ == mg2.__class__
    if mg1 is not None:
        assert mg1.target_qubits == mg2.target_qubits
    assert circuit1.measurement_tuples == circuit2.measurement_tuples


@pytest.mark.parametrize(("target", "controls", "free"),
                         [(0, (1,), ()), (2, (0, 1), ()),
                          (3, (0, 1, 4), (2, 5)),
                          (7, (0, 1, 2, 3, 4), (5, 6))])
def test_x_decomposition_gates(target, controls, free):
    """Check that decomposition of multi-control ``X`` agrees with Cirq."""
    gate = gates.X(target).controlled_by(*controls)
    qibo_decomp = gate.decompose(*free, use_toffolis=False)

    # Calculate the decomposition using Cirq.
    nqubits = max((target,) + controls + free) + 1
    qubits = [cirq.LineQubit(i) for i in range(nqubits)]
    controls = [qubits[i] for i in controls]
    free = [qubits[i] for i in free]
    cirq_decomp = cirq.decompose_multi_controlled_x(controls, qubits[target], free)

    assert len(qibo_decomp) == len(cirq_decomp)
    for qibo_gate, cirq_gate in zip(qibo_decomp, cirq_decomp):
        assert_gates_equivalent(qibo_gate, cirq_gate)


@pytest.mark.parametrize(("target", "controls", "free"),
                         [(0, (1,), ()), (2, (0, 1), ()),
                          (3, (0, 1, 4), (2, 5)),
                          (7, (0, 1, 2, 3, 4), (5, 6)),
                          (5, (0, 2, 4, 6, 7), (1, 3)),
                          (8, (0, 2, 4, 6, 9), (3, 5, 7))])
@pytest.mark.parametrize("use_toffolis", [True, False])
def test_x_decomposition_execution(target, controls, free, use_toffolis):
    """Check that applying the decomposition is equivalent to applying the multi-control gate."""
    gate = gates.X(target).controlled_by(*controls)
    nqubits = max((target,) + controls + free) + 1
    init_state = utils.random_numpy_state(nqubits)

    targetc = Circuit(nqubits)
    targetc.add(gate)
    target_state = targetc(np.copy(init_state)).numpy()

    c = Circuit(nqubits)
    c.add(gate.decompose(*free, use_toffolis=use_toffolis))
    final_state = c(np.copy(init_state)).numpy()

    np.testing.assert_allclose(final_state, target_state, atol=_ATOL)


@pytest.mark.parametrize("use_toffolis", [True, False])
def test_x_decomposition_errors(use_toffolis):
    """Check ``X`` decomposition errors."""
    gate = gates.X(0).controlled_by(1, 2, 3, 4)
    with pytest.raises(ValueError):
        decomp = gate.decompose(2, 3, use_toffolis=use_toffolis)
    c = Circuit(6)
    c.add(gate)
    with pytest.raises(ValueError):
        decomp = gate.decompose(5, 6, use_toffolis=use_toffolis)


def test_x_decompose_few_controls():
    """Check ``X`` decomposition with len(controls) < 3."""
    gate = gates.X(0)
    decomp = gate.decompose(1, 2)
    assert len(decomp) == 1
    assert isinstance(decomp[0], gates.X)


def test_circuit_decompose():
    """Check ``circuit.decompose`` agrees with multi-control ``X`` decomposition."""
    c = Circuit(6)
    c.add(gates.RX(0, 0.1234))
    c.add(gates.RY(1, 0.4321))
    c.add((gates.H(i) for i in range(2, 6)))
    c.add(gates.CNOT(0, 1))
    c.add(gates.X(3).controlled_by(0, 1, 2, 4))

    decomp_c = c.decompose(5)

    init_state = utils.random_numpy_state(c.nqubits)
    target_state = c(np.copy(init_state)).numpy()
    final_state = decomp_c(np.copy(init_state)).numpy()
    np.testing.assert_allclose(final_state, target_state, atol=_ATOL)

    target_c = Circuit(c.nqubits)
    target_c.add(gates.RX(0, 0.1234))
    target_c.add(gates.RY(1, 0.4321))
    target_c.add((gates.H(i) for i in range(2, 6)))
    target_c.add(gates.CNOT(0, 1))
    target_c.add(gates.X(3).controlled_by(0, 1, 2, 4).decompose(5))
    assert_circuit_same_gates(decomp_c, target_c)


def test_circuit_decompose_with_measurements():
    """Check ``circuit.decompose`` for circuit with measurements."""
    c = Circuit(8)
    c.add(gates.X(4).controlled_by(0, 2, 5, 6, 7))
    c.add(gates.M(0, 2, 4, 6, register_name="A"))
    c.add(gates.M(1, 3, 5, 7, register_name="B"))

    target_c = Circuit(8)
    target_c.add(gates.X(4).controlled_by(0, 2, 5, 6, 7).decompose(1, 3))
    target_c.add(gates.M(0, 2, 4, 6, register_name="A"))
    target_c.add(gates.M(1, 3, 5, 7, register_name="B"))

    decomp_c = c.decompose(1, 3)
    assert_circuit_same_gates(decomp_c, target_c)
