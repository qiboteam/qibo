"""Test circuit result measurements and measurement gate and as part of circuit."""

import json
import pickle

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.measurements import MeasurementResult
from qibo.models import QFT


def assert_result(
    backend,
    result,
    decimal_samples=None,
    binary_samples=None,
    decimal_frequencies=None,
    binary_frequencies=None,
):
    if decimal_frequencies is not None:
        assert result.frequencies(False) == decimal_frequencies
    if binary_frequencies is not None:
        assert result.frequencies(True) == binary_frequencies
    if decimal_samples is not None:
        backend.assert_allclose(result.samples(False), decimal_samples)
    if binary_samples is not None:
        backend.assert_allclose(result.samples(True), binary_samples)


def assert_dicts_equal(backend, d1, d2):
    assert d1.keys() == d2.keys()
    for k, v in d1.items():
        if isinstance(v, dict):
            assert v == d2[k]
        else:
            backend.assert_allclose(v, d2[k])


def assert_register_result(
    backend,
    result,
    decimal_samples=None,
    binary_samples=None,
    decimal_frequencies=None,
    binary_frequencies=None,
):
    if decimal_samples:
        register_result = result.samples(binary=False, registers=True)
        assert_dicts_equal(backend, register_result, decimal_samples)
    if binary_samples:
        register_result = result.samples(binary=True, registers=True)
        assert_dicts_equal(backend, register_result, binary_samples)
    if decimal_frequencies:
        register_result = result.frequencies(binary=False, registers=True)
        assert_dicts_equal(backend, register_result, decimal_frequencies)
    if binary_frequencies:
        register_result = result.frequencies(binary=True, registers=True)
        assert_dicts_equal(backend, register_result, binary_frequencies)


@pytest.mark.parametrize("n", [0, 1])
@pytest.mark.parametrize("nshots", [100, 1000000])
def test_measurement_gate(backend, n, nshots):
    circuit = Circuit(2)
    if n:
        circuit.add(gates.X(1))
    circuit.add(gates.M(1))
    result = backend.execute_circuit(circuit, nshots=nshots)
    assert_result(
        backend,
        result,
        n * np.ones(nshots),
        n * np.ones((nshots, 1)),
        {n: nshots},
        {str(n): nshots},
    )


def test_multiple_qubit_measurement_gate(backend):
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.M(0, 1))
    result = backend.execute_circuit(circuit, nshots=100)
    target_binary_samples = np.zeros((100, 2))
    target_binary_samples[:, 0] = 1
    assert_result(
        backend,
        result,
        2 * np.ones((100,)),
        target_binary_samples,
        {2: 100},
        {"10": 100},
    )


def test_measurement_gate_errors(backend):
    gate = gates.M(0)
    # attempting to use `controlled_by`
    with pytest.raises(NotImplementedError):
        gate.controlled_by(1)
    # attempting to construct unitary
    with pytest.raises(NotImplementedError):
        matrix = gate.matrix(backend)


def test_measurement_circuit(backend, accelerators):
    circuit = Circuit(4, accelerators)
    circuit.add(gates.X(0))
    circuit.add(gates.M(0))
    result = backend.execute_circuit(circuit, nshots=100)
    assert_result(
        backend, result, np.ones((100,)), np.ones((100, 1)), {1: 100}, {"1": 100}
    )


@pytest.mark.parametrize("registers", [False, True])
def test_measurement_qubit_order_simple(backend, registers):
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    if registers:
        circuit.add(gates.M(1, 0))
    else:
        circuit.add(gates.M(1))
        circuit.add(gates.M(0))
    result = backend.execute_circuit(circuit, nshots=100)

    target_binary_samples = np.zeros((100, 2))
    target_binary_samples[:, 1] = 1
    assert_result(
        backend, result, np.ones(100), target_binary_samples, {1: 100}, {"01": 100}
    )


@pytest.mark.parametrize("nshots", [100, 1000000])
def test_measurement_qubit_order(backend, accelerators, nshots):
    circuit = Circuit(6, accelerators)
    circuit.add(gates.X(0))
    circuit.add(gates.X(1))
    circuit.add(gates.M(1, 5, 2, 0))
    result = backend.execute_circuit(circuit, nshots=nshots)

    target_binary_samples = np.zeros((nshots, 4))
    target_binary_samples[:, 0] = 1
    target_binary_samples[:, 3] = 1
    assert_result(
        backend,
        result,
        9 * np.ones(nshots),
        target_binary_samples,
        {9: nshots},
        {"1001": nshots},
    )


def test_multiple_measurement_gates_circuit(backend):
    circuit = Circuit(4)
    circuit.add(gates.X(1))
    circuit.add(gates.X(2))
    circuit.add(gates.M(0, 1))
    circuit.add(gates.M(2))
    circuit.add(gates.X(3))
    result = backend.execute_circuit(circuit, nshots=100)

    target_binary_samples = np.ones((100, 3))
    target_binary_samples[:, 0] = 0
    assert_result(
        backend, result, 3 * np.ones(100), target_binary_samples, {3: 100}, {"011": 100}
    )


def test_circuit_with_unmeasured_qubits(backend, accelerators):
    circuit = Circuit(5, accelerators)
    circuit.add(gates.X(4))
    circuit.add(gates.X(2))
    circuit.add(gates.M(0, 2))
    circuit.add(gates.X(3))
    circuit.add(gates.M(1, 4))
    result = backend.execute_circuit(circuit, nshots=100)

    target_binary_samples = np.zeros((100, 4))
    target_binary_samples[:, 1] = 1
    target_binary_samples[:, 3] = 1
    assert_result(
        backend,
        result,
        5 * np.ones(100),
        target_binary_samples,
        {5: 100},
        {"0101": 100},
    )


def test_circuit_addition_with_measurements(backend):
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.X(1))

    meas_circuit = Circuit(2)
    circuit.add(gates.M(0, 1))

    circuit += meas_circuit
    result = backend.execute_circuit(circuit, nshots=100)

    assert_result(
        backend,
        result,
        3 * np.ones(100),
        np.ones((100, 2)),
        {3: 100},
        {"11": 100},
    )


def test_circuit_addition_with_measurements_in_both_circuits(backend, accelerators):
    circuit_1 = Circuit(4, accelerators)
    circuit_1.add(gates.X(0))
    circuit_1.add(gates.X(1))
    circuit_1.add(gates.M(1, register_name="a"))

    circuit_2 = Circuit(4, accelerators)
    circuit_2.add(gates.X(0))
    circuit_2.add(gates.M(0, register_name="b"))

    circuit = circuit_1 + circuit_2
    result = backend.execute_circuit(circuit, nshots=100)
    assert_result(
        backend,
        result,
        binary_frequencies={"10": 100},
    )


def test_circuit_copy_with_measurements(backend, accelerators):
    circuit_1 = Circuit(6, accelerators)
    circuit_1.add([gates.X(0), gates.X(1), gates.X(3)])
    circuit_1.add(gates.M(5, 1, 3, register_name="a"))
    circuit_1.add(gates.M(2, 0, register_name="b"))
    circuit_2 = circuit_1.copy(deep=True)

    r1 = backend.execute_circuit(circuit_1, nshots=100)
    r2 = backend.execute_circuit(circuit_2, nshots=100)

    backend.assert_allclose(r1.samples(), r2.samples())
    rg1 = r1.frequencies(registers=True)
    rg2 = r2.frequencies(registers=True)
    assert rg1.keys() == rg2.keys()
    for k in rg1.keys():
        assert rg1[k] == rg2[k]


def test_measurement_compiled_circuit(backend):
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.M(0))
    circuit.add(gates.M(1))
    circuit.compile(backend)
    result = circuit(nshots=100)
    target_binary_samples = np.zeros((100, 2))
    target_binary_samples[:, 0] = 1
    assert_result(
        backend,
        result,
        2 * np.ones((100,)),
        target_binary_samples,
        {2: 100},
        {"10": 100},
    )

    target_state = np.zeros_like(circuit.final_state.state())
    target_state[2] = 1
    backend.assert_allclose(circuit.final_state._state, target_state)


def test_final_state(backend, accelerators):
    """Check that final state is logged correctly when using measurements."""
    circuit = Circuit(4, accelerators)
    circuit.add(gates.X(1))
    circuit.add(gates.X(2))
    circuit.add(gates.M(0, 1))
    circuit.add(gates.M(2))
    circuit.add(gates.X(3))
    result = backend.execute_circuit(circuit, nshots=100)
    circuit = Circuit(4, accelerators)
    circuit.add(gates.X(1))
    circuit.add(gates.X(2))
    circuit.add(gates.X(3))
    target_state = backend.execute_circuit(circuit)
    backend.assert_allclose(circuit.final_state, target_state)


def test_measurement_gate_bitflip_errors():
    gate = gates.M(0, 1, p0=2 * [0.1])
    with pytest.raises(ValueError):
        gate = gates.M(0, 1, p0=4 * [0.1])
    with pytest.raises(KeyError):
        gate = gates.M(0, 1, p0={0: 0.1, 2: 0.2})
    with pytest.raises(TypeError):
        gate = gates.M(0, 1, p0="test")


def test_register_measurements(backend):
    circuit = Circuit(3)
    circuit.add(gates.X(0))
    circuit.add(gates.X(1))
    circuit.add(gates.M(0, 2))
    circuit.add(gates.M(1))
    result = backend.execute_circuit(circuit, nshots=100)

    decimal_samples = {"register0": 2 * np.ones((100,)), "register1": np.ones((100,))}
    binary_samples = {"register0": np.zeros((100, 2)), "register1": np.ones((100, 1))}
    binary_samples["register0"][:, 0] = 1
    decimal_frequencies = {"register0": {2: 100}, "register1": {1: 100}}
    binary_frequencies = {"register0": {"10": 100}, "register1": {"1": 100}}
    print(type(result))
    assert_register_result(
        backend,
        result,
        decimal_samples,
        binary_samples,
        decimal_frequencies,
        binary_frequencies,
    )


def test_measurement_qubit_order_multiple_registers(backend, accelerators):
    circuit = Circuit(6, accelerators)
    circuit.add(gates.X(0))
    circuit.add(gates.X(1))
    circuit.add(gates.X(3))
    circuit.add(gates.M(5, 1, 3, register_name="a"))
    circuit.add(gates.M(2, 0, register_name="b"))
    result = backend.execute_circuit(circuit, nshots=100)

    # Check full result
    target_binary_samples = np.zeros((100, 5))
    target_binary_samples[:, 1] = 1
    target_binary_samples[:, 2] = 1
    target_binary_samples[:, 4] = 1
    assert_result(
        backend,
        result,
        13 * np.ones((100,)),
        target_binary_samples,
        {13: 100},
        {"01101": 100},
    )

    decimal_samples = {"a": 3 * np.ones((100,)), "b": np.ones((100,))}
    binary_samples = {"a": np.zeros((100, 3)), "b": np.zeros((100, 2))}
    binary_samples["a"][:, 1] = 1
    binary_samples["a"][:, 2] = 1
    binary_samples["b"][:, 1] = 1
    decimal_frequencies = {"a": {3: 100}, "b": {1: 100}}
    binary_frequencies = {"a": {"011": 100}, "b": {"01": 100}}
    assert_register_result(
        backend,
        result,
        decimal_samples,
        binary_samples,
        decimal_frequencies,
        binary_frequencies,
    )


def test_registers_in_circuit_with_unmeasured_qubits(backend, accelerators):
    """Check that register measurements are unaffected by unmeasured qubits."""
    circuit = Circuit(5, accelerators)
    circuit.add(gates.X(1))
    circuit.add(gates.X(2))
    circuit.add(gates.M(0, 2, register_name="A"))
    circuit.add(gates.X(3))
    circuit.add(gates.M(1, 4, register_name="B"))
    result = backend.execute_circuit(circuit, nshots=100)

    target = {}
    decimal_samples = {"A": np.ones((100,)), "B": 2 * np.ones((100,))}
    binary_samples = {"A": np.zeros((100, 2)), "B": np.zeros((100, 2))}
    binary_samples["A"][:, 1] = 1
    binary_samples["B"][:, 0] = 1
    decimal_frequencies = {"A": {1: 100}, "B": {2: 100}}
    binary_frequencies = {"A": {"01": 100}, "B": {"10": 100}}
    assert_register_result(
        backend,
        result,
        decimal_samples,
        binary_samples,
        decimal_frequencies,
        binary_frequencies,
    )


def test_measurement_density_matrix(backend):
    circuit = Circuit(2, density_matrix=True)
    circuit.add(gates.X(0))
    circuit.add(gates.M(0, 1))
    result = backend.execute_circuit(circuit, nshots=100)
    target_binary_samples = np.zeros((100, 2))
    target_binary_samples[:, 0] = 1
    assert_result(
        backend,
        result,
        decimal_samples=2 * np.ones((100,)),
        binary_samples=target_binary_samples,
        decimal_frequencies={2: 100},
        binary_frequencies={"10": 100},
    )


def test_measurement_result_vs_circuit_result(backend, accelerators):
    circuit = Circuit(6, accelerators)
    circuit.add([gates.X(0), gates.X(1), gates.X(3)])
    ma = circuit.add(gates.M(5, 1, 3, register_name="a"))
    mb = circuit.add(gates.M(2, 0, register_name="b"))
    result = backend.execute_circuit(circuit, nshots=100)

    ma_freq = ma.frequencies()
    mb_freq = mb.frequencies()
    frequencies = result.frequencies(registers=True)
    assert ma_freq == frequencies.get("a")
    assert mb_freq == frequencies.get("b")


@pytest.mark.parametrize("nqubits", [1, 4])
@pytest.mark.parametrize("outcome", [0, 1])
def test_measurement_basis(backend, nqubits, outcome):
    circuit = Circuit(nqubits)
    if outcome:
        circuit.add(gates.X(q) for q in range(nqubits))
    circuit.add(gates.H(q) for q in range(nqubits))
    circuit.add(gates.M(*range(nqubits), basis=gates.X))
    result = backend.execute_circuit(circuit, nshots=100)
    assert result.frequencies() == {nqubits * str(outcome): 100}


def test_measurement_basis_list(backend):
    circuit = Circuit(4, wire_names=["q0", "q1", "q2", "q3"])
    circuit.add(gates.H(0))
    circuit.add(gates.X(2))
    circuit.add(gates.H(2))
    circuit.add(gates.X(3))
    circuit.add(gates.M(0, 1, 2, 3, basis=[gates.X, gates.Z, gates.X, gates.Z]))
    result = backend.execute_circuit(circuit, nshots=100)
    assert result.frequencies() == {"0011": 100}
    assert (
        str(circuit)
        == """q0: ─H─H───M─
q1: ───────M─
q2: ─X─H─H─M─
q3: ─X─────M─"""
    )


def test_measurement_basis_list_error():
    circuit = Circuit(4)
    with pytest.raises(ValueError):
        circuit.add(gates.M(0, 1, 2, 3, basis=[gates.X, gates.Z, gates.X]))


def test_measurement_same_qubit_different_registers_error():
    circuit = Circuit(4)
    circuit.add(gates.M(0, 1, 3, register_name="a"))
    with pytest.raises(KeyError):
        circuit.add(gates.M(1, 2, 3, register_name="a"))


def test_measurementsymbol_pickling(backend):
    circuit = QFT(3)
    circuit.add(gates.M(0, 2, basis=[gates.X, gates.Z]))
    backend.execute_circuit(circuit).samples()
    for symbol in circuit.measurements[0].result.symbols:
        dumped_symbol = pickle.dumps(symbol)
        new_symbol = pickle.loads(dumped_symbol)
        assert symbol.index == new_symbol.index
        assert symbol.name == new_symbol.name
        backend.assert_allclose(symbol.result.samples(), new_symbol.result.samples())


def test_measurementresult_nshots(backend):
    gate = gates.M(*range(3))
    result = MeasurementResult(gate.qubits)
    # nshots starting from samples
    nshots = 10
    samples = backend.cast(
        [[i % 2, i % 2, i % 2] for i in range(nshots)], backend.engine.int64
    )
    result.register_samples(samples)
    assert result.nshots == nshots
    # nshots starting from frequencies
    result = MeasurementResult(gate.qubits)
    states, counts = np.unique(samples, axis=0, return_counts=True)
    to_str = lambda x: [str(item) for item in x]
    states = ["".join(to_str(s)) for s in states.tolist()]
    freq = dict(zip(states, counts.tolist()))
    result.register_frequencies(freq)
    assert result.nshots == nshots


def test_measurement_serialization(backend):
    kwargs = {
        "register_name": "test",
        "collapse": False,
        "basis": ["Z", "X", "Y"],
        "p0": 0.1,
        "p1": 0.2,
    }
    gate = gates.M(*range(3), **kwargs)
    samples = backend.cast(np.random.randint(2, size=(100, 3)), backend.engine.int64)
    gate.result.register_samples(samples)
    dump = gate.to_json()
    load = gates.M.from_dict(json.loads(dump))
    for k, v in kwargs.items():
        assert load.init_kwargs[k] == v
    backend.assert_allclose(samples, load.result.samples())
