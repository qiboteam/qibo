import pytest
from qibo.base import gates

# TODO: Add tests for `abstract_gates.py`

@pytest.mark.parametrize("gatename", ["H", "X", "Y", "Z", "I"])
def test_one_qubit_gates_init(gatename):
    gate = getattr(gates, gatename)(0)
    assert gate.target_qubits == (0,)


@pytest.mark.parametrize("controls,instance",
                         [((1,), "CNOT"), ((1, 2), "TOFFOLI"),
                          ((1, 2, 4), "X")])
def test_x_controlled_by(controls, instance):
    gate = gates.X(0).controlled_by(*controls)
    assert gate.target_qubits == (0,)
    assert gate.control_qubits == controls
    assert isinstance(gate, getattr(gates, instance))


@pytest.mark.parametrize(("target", "controls", "free"),
                         [(0, (1,), ()), (2, (0, 1), ()),
                          (3, (0, 1, 4), (2, 5)),
                          (7, (0, 1, 2, 3, 4), (5, 6))])
def test_x_decompose_with_cirq(target, controls, free):
    """Check that decomposition of multi-control ``X`` agrees with Cirq."""
    import cirq
    from qibo.tests import cirq_utils
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
        cirq_utils.assert_gates_equivalent(qibo_gate, cirq_gate)


def test_x_decompose_with_few_controls():
    """Check ``X`` decomposition with less than three controls."""
    gate = gates.X(0)
    decomp = gate.decompose(1, 2)
    assert len(decomp) == 1
    assert isinstance(decomp[0], gates.X)


@pytest.mark.parametrize("use_toffolis", [True, False])
def test_x_decomposition_errors(use_toffolis):
    """Check ``X`` decomposition errors."""
    gate = gates.X(0).controlled_by(1, 2, 3, 4)
    with pytest.raises(ValueError):
        decomp = gate.decompose(2, 3, use_toffolis=use_toffolis)
    gate.nqubits = 6
    with pytest.raises(ValueError):
        decomp = gate.decompose(5, 6, use_toffolis=use_toffolis)


@pytest.mark.parametrize("controls,instance", [((1,), "CZ"), ((1, 2), "Z")])
def test_z_controlled_by(controls, instance):
    gate = gates.Z(0).controlled_by(*controls)
    assert gate.target_qubits == (0,)
    assert gate.control_qubits == controls
    assert isinstance(gate, getattr(gates, instance))


@pytest.mark.parametrize("result,target_result",
                         [(1, [1, 1, 1]),
                          ([0, 1, 0], [0, 0, 1])])
def test_collapse_init(result, target_result):
    # also tests ``Collapse.result`` getter and setter
    gate = gates.Collapse(0, 2, 1, result=result)
    assert gate.target_qubits == (0, 2, 1)
    assert gate.sorted_qubits == [0, 1, 2]
    assert gate.result == target_result


def test_collapse_controlled_by():
    gate = gates.Collapse(0)
    with pytest.raises(NotImplementedError):
        gate.controlled_by(1)
