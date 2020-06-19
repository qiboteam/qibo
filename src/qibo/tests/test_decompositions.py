import re
import numpy as np
import pytest
import cirq
from qibo import gates
from qibo.models import Circuit

_QIBO_TO_CIRQ = {"CNOT": "CNOT", "RY": "Ry", "TOFFOLI": "TOFFOLI"}
_ATOL = 1e-6


def random_initial_state(nqubits, dtype=np.complex128):
    """Generates a random normalized state vector."""
    x = np.random.random(2 ** nqubits) + 1j * np.random.random(2 ** nqubits)
    return (x / np.sqrt((np.abs(x) ** 2).sum())).astype(dtype)


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
    else:
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
        else:
            theta = float(theta)
        np.testing.assert_allclose(theta, qibo_gate.theta)


@pytest.mark.parametrize(("target", "controls", "free"),
                         [(0, (1,), ()), (2, (0, 1), ()),
                          (3, (0, 1, 4), (2, 5)),
                          (7, (0, 1, 2, 3, 4), (5, 6))])
def test_x_decomposition_gates(target, controls, free):
    """Check that decomposition of multi-control ``X`` agrees with Cirq."""
    gate = gates.X(target).controlled_by(*controls)
    qibo_decomp = gate.decompose(*free, use_toffolis=False)

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
    gate = gates.X(target).controlled_by(*controls)
    nqubits = max((target,) + controls + free) + 1
    init_state = random_initial_state(nqubits)

    targetc = Circuit(nqubits)
    targetc.add(gate)
    target_state = targetc(np.copy(init_state)).numpy()

    c = Circuit(nqubits)
    c.add(gate.decompose(*free, use_toffolis=use_toffolis))
    final_state = c(np.copy(init_state)).numpy()

    np.testing.assert_allclose(final_state, target_state, atol=_ATOL)


@pytest.mark.parametrize("use_toffolis", [True, False])
def test_x_decomposition_errors(use_toffolis):
    gate = gates.X(0).controlled_by(1, 2, 3, 4)
    with pytest.raises(ValueError):
        decomp = gate.decompose(2, 3, use_toffolis=use_toffolis)
    c = Circuit(6)
    c.add(gate)
    with pytest.raises(ValueError):
        decomp = gate.decompose(5, 6, use_toffolis=use_toffolis)


def test_x_decomposition_execution_cirq():
    # TODO: Remove this test
    init_state = random_initial_state(8)
    sim = cirq.Simulator()

    qubits = [cirq.LineQubit(i) for i in range(8)]
    controls = qubits[:5]
    free = qubits[5:-1]
    cirq_decomp = cirq.decompose_multi_controlled_x(controls, qubits[-1], free)

    targetc = cirq.Circuit()
    targetc.append(cirq.X.controlled(5)(*(controls + [qubits[-1]])))
    target_state = sim.simulate(targetc, initial_state=np.copy(init_state),
                                qubit_order=qubits).final_state

    c = cirq.Circuit()
    c.append(cirq_decomp)
    final_state = sim.simulate(c, initial_state=np.copy(init_state),
                               qubit_order=qubits).final_state

    np.testing.assert_allclose(final_state, target_state, atol=_ATOL)
