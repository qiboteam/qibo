"""Tests methods defined in `qibo/abstractions/gates.py` and `qibo/abstractions/abstract_gates.py`."""
import pytest
from qibo.abstractions import gates


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


@pytest.mark.parametrize("targets,p0,p1",
                         [((0,), None, None),
                          ((0, 1, 2), None, None),
                          ((0, 3, 2), 0.2, 0.1)])
def test_measurement_init(targets, p0, p1):
    # also tests `_get_bitflip_map`
    gate = gates.M(*targets, p0=p0, p1=p1)
    assert gate.target_qubits == targets
    p0map = {q: 0 if p0 is None else p0 for q in targets}
    p1map = {q: 0 if p1 is None else p1 for q in targets}
    assert gate.bitflip_map == (p0map, p1map)


def test_measurement_einsum_string():
    func = gates.M.einsum_string
    estr = func({0, 2, 4}, 5)
    assert estr == "abcdeagcie->bdgi"
    estr = func({0, 2, 4}, 5, measuring=True)
    assert estr == "abcdeabcde->bd"
    estr = func({0, 1, 3, 5, 6}, 10, measuring=False)
    assert estr == "abcdefghijabmdofgrst->cehijmorst"
    estr = func({0, 1, 3, 5, 6}, 10, measuring=True)
    assert estr == "abcdefghijabcdefghij->cehij"


def test_measurement_add():
    gate = gates.M(0, 2)
    assert gate.target_qubits == (0, 2)
    assert gate.bitflip_map == 2 * ({0: 0, 2: 0},)
    gate.add(gates.M(1, 3, p0=0.3, p1=0.0))
    assert gate.target_qubits == (0, 2, 1, 3)
    assert gate.bitflip_map == ({0: 0, 1: 0.3, 2: 0, 3: 0.3},
                                {0: 0, 1: 0, 2: 0, 3: 0})


def test_measurement_errors():
    gate = gates.M(0)
    with pytest.raises(NotImplementedError):
        gate.controlled_by(1)


@pytest.mark.parametrize("gatename,params",
                         [("RX", (0.1234,)), ("RY", (0.1234,)),
                          ("RZ", (0.1234,)), ("U1", (0.1234,)),
                          ("U2", (0.1234, 0.4321)),
                          ("U3", (0.1234, 0.4321, 0.5678))])
def test_one_qubit_rotations_init(gatename, params):
    gate = getattr(gates, gatename)(0, *params)
    assert gate.target_qubits == (0,)
    if len(params) == 1:
        params = params[0]
    assert gate.parameters == params


@pytest.mark.parametrize("gatename,params",
                         [("RX", (0.1234,)), ("RY", (0.1234,)),
                          ("RZ", (0.1234,)), ("U1", (0.1234,)),
                          ("U2", (0.1234, 0.4321)),
                          ("U3", (0.1234, 0.4321, 0.5678))])
def test_one_qubit_rotations_controlled_by(gatename, params):
    gate = getattr(gates, gatename)(0, *params).controlled_by(1)
    assert gate.target_qubits == (0,)
    assert gate.control_qubits == (1,)
    assert isinstance(gate, getattr(gates, f"C{gatename}"))
    gate = getattr(gates, gatename)(1, *params).controlled_by(0, 3)
    assert gate.target_qubits == (1,)
    assert gate.control_qubits == (0, 3)
    if len(params) == 1:
        params = params[0]
    assert gate.parameters == params


def test_cnot_and_cz_init():
    gate = gates.CNOT(0, 1)
    assert gate.target_qubits == (1,)
    assert gate.control_qubits == (0,)
    gate = gates.CZ(3, 2)
    assert gate.target_qubits == (2,)
    assert gate.control_qubits == (3,)

# :meth:`qibo.abstractions.gates.CNOT.decompose` is tested in
# ``test_x_decompose_with_cirq`` above

@pytest.mark.parametrize("gatename,params",
                         [("CRX", (0.1234,)), ("CRY", (0.1234,)),
                          ("CRZ", (0.1234,)), ("CU1", (0.1234,)),
                          ("CU2", (0.1234, 0.4321)),
                          ("CU3", (0.1234, 0.4321, 0.5678))])
def test_two_qubit_controlled_rotations_init(gatename, params):
    gate = getattr(gates, gatename)(0, 2, *params)
    assert gate.target_qubits == (2,)
    assert gate.control_qubits == (0,)


def test_swap_init():
    gate = gates.SWAP(4, 3)
    assert gate.target_qubits == (4, 3)


def test_fsim_init():
    import numpy as np
    gate = gates.fSim(0, 1, 0.1234, 0.4321)
    assert gate.target_qubits == (0, 1)
    matrix = np.random.random((2, 2))
    gate = gates.GeneralizedfSim(0, 1, matrix, 0.4321)
    assert gate.target_qubits == (0, 1)
    assert gate.parameters == (matrix, 0.4321)
    matrix = np.random.random((3, 3))
    with pytest.raises(ValueError):
        gate = gates.GeneralizedfSim(0, 1, matrix, 0.4321)


def test_toffoli_init():
    gate = gates.TOFFOLI(0, 2, 1)
    assert gate.target_qubits == (1,)
    assert gate.control_qubits == (0, 2)

# :meth:`qibo.abstractions.gates.TOFFOLI.decompose` and
# :meth:`qibo.abstractions.gates.TOFFOLI.congruent`
# are tested in `test_x_decompose_with_cirq`

@pytest.mark.parametrize("targets", [(0,), (2, 0), (1, 3, 2)])
def test_unitary_init(targets):
    import numpy as np
    matrix = np.random.random(2 * (2 ** len(targets),))
    gate = gates.Unitary(matrix, *targets)
    assert gate.target_qubits == targets
    assert gate.nparams == 4 ** len(targets)


@pytest.mark.parametrize("targets", [range(5), range(6)])
def test_variational_layer_init(targets):
    import numpy as np
    targets = tuple(targets)
    pairs = [(i, i + 1) for i in range(0, len(targets) - 1, 2)]
    params = np.random.random(len(targets))
    gate = gates.VariationalLayer(targets, pairs, gates.RY, gates.CZ, params)
    assert gate.target_qubits == targets
    assert gate.params == {q: p for q, p in zip(targets, params)}
    assert gate.params2 == {}

    pairs = [(0, 1)]
    with pytest.raises(ValueError):
        gate = gates.VariationalLayer(targets, pairs, gates.RY, gates.CZ, params)


def test_variational_layer_errors():
    import numpy as np
    pairs = [(i, i + 1) for i in range(0, 5, 2)]
    with pytest.raises(ValueError):
        gate = gates.VariationalLayer(range(6), pairs, gates.RY, gates.CZ,
                                      np.zeros(6), np.zeros(7))
    with pytest.raises(ValueError):
        gate = gates.VariationalLayer(range(10), pairs,
                                      gates.RY, gates.CZ,
                                      np.zeros(10), np.zeros(10))
    gate = gates.VariationalLayer(range(6), pairs, gates.RY, gates.CZ,
                                  np.zeros(6), np.zeros(6))
    np.testing.assert_allclose(gate.parameters, np.zeros(12))


def test_flatten():
    gate = gates.Flatten([1, 2, 3, 4])
    assert gate.coefficients == [1, 2, 3, 4]
    with pytest.raises(NotImplementedError):
        gate.on_qubits(0, 2)

# TODO: Test :class:`qibo.abstractions.gates.CallbackGate` in
# `test_core_gates.py`

def test_kraus_channel_init():
    import numpy as np
    ops = [((0,), np.random.random((2, 2))),
           ((0, 1), np.random.random((4, 4))),
           ((0, 2), np.random.random((4, 4))),
           ((3,), np.random.random((2, 2)))]
    gate = gates.KrausChannel(ops)
    gate.target_qubits == (0, 1, 2, 3)
    for g in gate.gates:
        assert isinstance(g, gates.Unitary)

    ops.append(((4,), np.random.random((4, 4))))
    with pytest.raises(ValueError):
        gate = gates.KrausChannel(ops)


def test_unitary_channel_init():
    import numpy as np
    ops = [((0,), np.random.random((2, 2))),
           ((0, 1), np.random.random((4, 4))),
           ((0, 2), np.random.random((4, 4))),
           ((3,), np.random.random((2, 2)))]
    gate = gates.UnitaryChannel(4 * [0.1], ops)
    gate.target_qubits == (0, 1, 2, 3)
    for g in gate.gates:
        assert isinstance(g, gates.Unitary)

    with pytest.raises(ValueError):
        gate = gates.UnitaryChannel(2 * [0.1], ops)
    with pytest.raises(ValueError):
        gate = gates.UnitaryChannel(4 * [-0.1], ops)
    with pytest.raises(ValueError):
        gate = gates.UnitaryChannel(4 * [0.5], ops)


def test_pauli_noise_channel_init():
    gate = gates.PauliNoiseChannel(0, 0.1, 0.2, 0.3)
    assert gate.target_qubits == (0,)
    assert isinstance(gate.gates[0], gates.X)
    assert isinstance(gate.gates[1], gates.Y)
    assert isinstance(gate.gates[2], gates.Z)


def test_reset_channel_init():
    gate = gates.ResetChannel(0, 0.1, 0.2)
    assert gate.target_qubits == (0,)
    assert isinstance(gate.gates[0], gates.M)
    assert isinstance(gate.gates[1], gates.X)

# TODO: Add thermal relaxation channel init test

def test_qubit_getter_and_setter():
    from qibo.abstractions import abstract_gates
    gate = abstract_gates.Gate()
    gate.target_qubits = (0, 3)
    gate.control_qubits = (1, 4, 2)
    assert gate.qubits == (1, 2, 4, 0, 3)

    gate = abstract_gates.Gate()
    with pytest.raises(ValueError):
        gate.target_qubits = (1, 1)
    gate = abstract_gates.Gate()
    with pytest.raises(ValueError):
        gate.control_qubits = (1, 1)
    gate = abstract_gates.Gate()
    gate.target_qubits = (0, 1)
    with pytest.raises(ValueError):
        gate.control_qubits = (1,)


def test_nqubits_getter_and_setter():
    from qibo.abstractions import abstract_gates
    gate = abstract_gates.Gate()
    gate.target_qubits = (0, 1)
    gate.control_qubits = (2,)
    gate.nqubits = 10
    assert gate.nqubits == 10
    assert gate.nstates == 1024


def test_nqubits_getter_and_setter_errors():
    from qibo.abstractions import abstract_gates
    gate = abstract_gates.Gate()
    with pytest.raises(ValueError):
        nqubits = gate.nqubits
    with pytest.raises(ValueError):
        nstates = gate.nstates
    gate.nqubits = 4
    with pytest.raises(ValueError):
        gate.nqubits = 5


def test_density_matrix_getter_and_setter():
    from qibo.abstractions import abstract_gates
    gate = abstract_gates.Gate()
    gate.target_qubits = (0, 1)
    gate.control_qubits = (2,)
    assert gate._active_call == "state_vector_call"
    gate.density_matrix = True
    assert gate._active_call == "density_matrix_call"

    gate.nqubits = 4
    gate.is_prepared = True
    with pytest.raises(RuntimeError):
        gate.density_matrix = False

def test_gates_commute():
    assert gates.H(0).commutes(gates.X(1))
    assert gates.H(0).commutes(gates.H(1))
    assert gates.H(0).commutes(gates.H(0))
    assert not gates.H(0).commutes(gates.Y(0))
    assert not gates.CNOT(0, 1).commutes(gates.SWAP(1, 2))
    assert not gates.CNOT(0, 1).commutes(gates.H(1))
    assert not gates.CNOT(0, 1).commutes(gates.Y(0).controlled_by(2))
    assert not gates.CNOT(2, 3).commutes(gates.CNOT(3, 0))
    assert gates.CNOT(0, 1).commutes(gates.Y(2).controlled_by(0))


def test_on_qubits():
    gate = gates.CNOT(0, 1).on_qubits(2, 3)
    assert gate.target_qubits == (3,)
    assert gate.control_qubits == (2,)
    assert isinstance(gate, gates.CNOT)


def test_controlled_by():
    gate = gates.RX(0, 0.1234).controlled_by(1, 2, 3)
    assert gate.target_qubits == (0,)
    assert gate.control_qubits == (1, 2, 3)
    assert gate.is_controlled_by
    assert isinstance(gate, gates.RX)

    with pytest.raises(RuntimeError):
        gate = gates.CNOT(0, 1).controlled_by(2)

    gate = gates.RY(0, 0.1234)
    gate.nqubits = 5
    with pytest.raises(RuntimeError):
        gate = gate.controlled_by(2)


def test_on_qubits_controlled_by():
    gate = gates.H(0).controlled_by(1, 2)
    gate = gate.on_qubits(5, 4, 6)
    assert gate.target_qubits == (5,)
    assert gate.control_qubits == (4, 6)
    assert isinstance(gate, gates.H)
    assert gate.is_controlled_by


def test_decompose():
    decomp_gates = gates.H(0).decompose(1)
    assert len(decomp_gates) == 1
    assert isinstance(decomp_gates[0], gates.H)


def test_special_gate():
    from qibo.abstractions import abstract_gates
    gate = abstract_gates.SpecialGate()
    assert not gate.commutes(gates.H(0))
    with pytest.raises(NotImplementedError):
        gate.on_qubits(1)
