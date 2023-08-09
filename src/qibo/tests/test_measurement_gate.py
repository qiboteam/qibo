"""Test :class:`qibo.abstractions.gates.M` as standalone and as part of circuit."""
import pytest
import numpy as np
from qibo import models, gates, K


def assert_result(result, decimal_samples=None, binary_samples=None,
                  decimal_frequencies=None, binary_frequencies=None):
    if decimal_frequencies is not None:
        assert result.frequencies(False) == decimal_frequencies
    if binary_frequencies is not None:
        assert result.frequencies(True) == binary_frequencies
    if decimal_samples is not None:
        K.assert_allclose(result.samples(False), decimal_samples)
    if binary_samples is not None:
        K.assert_allclose(result.samples(True), binary_samples)


@pytest.mark.parametrize("n", [0, 1])
@pytest.mark.parametrize("nshots", [100, 1000000])
def test_measurement_gate(backend, n, nshots):
    state = np.zeros(4)
    state[-n] = 1
    result = gates.M(0)(K.cast(state), nshots=nshots)
    assert_result(result, n * np.ones(nshots), n * np.ones((nshots, 1)),
                  {n: nshots}, {str(n): nshots})


def test_multiple_qubit_measurement_gate(backend):
    state = np.zeros(4)
    state[2] = 1
    result = gates.M(0, 1)(K.cast(state), nshots=100)
    target_binary_samples = np.zeros((100, 2))
    target_binary_samples[:, 0] = 1
    assert_result(result, 2 * np.ones((100,)), target_binary_samples,
                  {2: 100}, {"10": 100})


def test_measurement_gate_errors(backend):
    gate = gates.M(0)
    # attempting to use `controlled_by`
    with pytest.raises(NotImplementedError):
        gate.controlled_by(1)
    # attempting to construct unitary
    with pytest.raises(ValueError):
        matrix = gate.matrix
    # calling on bad state
    with pytest.raises(TypeError):
        gate("test", 100)


def test_measurement_circuit(backend, accelerators):
    c = models.Circuit(4, accelerators)
    c.add(gates.X(0))
    c.add(gates.M(0))
    result = c(nshots=100)
    assert_result(result,
                  np.ones((100,)), np.ones((100, 1)),
                  {1: 100}, {"1": 100})


def test_gate_after_measurement_error(backend, accelerators):
    c = models.Circuit(4, accelerators)
    c.add(gates.X(0))
    c.add(gates.M(0))
    c.add(gates.X(1))
    # TODO: Change this to NotImplementedError
    with pytest.raises(ValueError):
        c.add(gates.H(0))


@pytest.mark.parametrize("registers", [False, True])
def test_measurement_qubit_order_simple(backend, registers):
    c = models.Circuit(2)
    c.add(gates.X(0))
    if registers:
        c.add(gates.M(1, 0))
    else:
        c.add(gates.M(1))
        c.add(gates.M(0))
    result = c(nshots=100)

    target_binary_samples = np.zeros((100, 2))
    target_binary_samples[:, 1] = 1
    assert_result(result, np.ones(100), target_binary_samples,
                  {1: 100}, {"01": 100})


@pytest.mark.parametrize("nshots", [100, 1000000])
def test_measurement_qubit_order(backend, accelerators, nshots):
    c = models.Circuit(6, accelerators)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.M(1, 5, 2, 0))
    result = c(nshots=nshots)

    target_binary_samples = np.zeros((nshots, 4))
    target_binary_samples[:, 0] = 1
    target_binary_samples[:, 3] = 1
    assert_result(result, 9 * np.ones(nshots), target_binary_samples,
                  {9: nshots}, {"1001": nshots})


def test_multiple_measurement_gates_circuit(backend):
    c = models.Circuit(4)
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.M(0, 1))
    c.add(gates.M(2))
    c.add(gates.X(3))
    result = c(nshots=100)

    target_binary_samples = np.ones((100, 3))
    target_binary_samples[:, 0] = 0
    assert_result(result, 3 * np.ones(100), target_binary_samples,
                  {3: 100}, {"011": 100})


def test_circuit_with_unmeasured_qubits(backend, accelerators):
    c = models.Circuit(5, accelerators)
    c.add(gates.X(4))
    c.add(gates.X(2))
    c.add(gates.M(0, 2))
    c.add(gates.X(3))
    c.add(gates.M(1, 4))
    result = c(nshots=100)

    target_binary_samples = np.zeros((100, 4))
    target_binary_samples[:, 1] = 1
    target_binary_samples[:, 3] = 1
    assert_result(result, 5 * np.ones(100), target_binary_samples,
                  {5: 100}, {"0101": 100})


def test_circuit_addition_with_measurements(backend):
    c = models.Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))

    meas_c = models.Circuit(2)
    c.add(gates.M(0, 1))

    c += meas_c
    assert len(c.measurement_gate.target_qubits) == 2
    assert c.measurement_tuples == {"register0": (0, 1)}


def test_circuit_addition_with_measurements_in_both_circuits(backend, accelerators):
    c1 = models.Circuit(4, accelerators)
    c1.add(gates.H(0))
    c1.add(gates.H(1))
    c1.add(gates.M(1, register_name="a"))

    c2 = models.Circuit(4, accelerators)
    c2.add(gates.X(0))
    c2.add(gates.M(0, register_name="b"))

    c = c1 + c2
    assert len(c.measurement_gate.target_qubits) == 2
    assert c.measurement_tuples == {"a": (1,), "b": (0,)}


def test_gate_after_measurement_with_addition_error(backend, accelerators):
    c = models.Circuit(4, accelerators)
    c.add(gates.H(0))
    c.add(gates.M(1))

    # Try to add gate to qubit that is already measured
    c2 = models.Circuit(4, accelerators)
    c2.add(gates.H(1))
    with pytest.raises(ValueError):
        c += c2
    # Try to add measurement to qubit that is already measured
    c2 = models.Circuit(4, accelerators)
    c2.add(gates.M(1, register_name="a"))
    with pytest.raises(ValueError):
        c += c2


def test_circuit_copy_with_measurements(backend, accelerators):
    c1 = models.Circuit(6, accelerators)
    c1.add([gates.X(0), gates.X(1), gates.X(3)])
    c1.add(gates.M(5, 1, 3, register_name="a"))
    c1.add(gates.M(2, 0, register_name="b"))
    c2 = c1.copy()

    r1 = c1(nshots=100)
    r2 = c2(nshots=100)

    K.assert_allclose(r1.samples(), r2.samples())
    rg1 = r1.frequencies(registers=True)
    rg2 = r2.frequencies(registers=True)
    assert rg1.keys() == rg2.keys()
    for k in rg1.keys():
        assert rg1[k] == rg2[k]


def test_measurement_compiled_circuit(backend):
    from qibo import K
    if K.is_custom:
        # use native gates because custom gates do not support compilation
        pytest.skip("Custom backend does not support compilation.")
    c = models.Circuit(2)
    c.add(gates.X(0))
    c.add(gates.M(0))
    c.add(gates.M(1))
    c.compile()
    result = c(nshots=100)
    target_binary_samples = np.zeros((100, 2))
    target_binary_samples[:, 0] = 1
    assert_result(result, 2 * np.ones((100,)), target_binary_samples,
                   {2: 100}, {"10": 100})

    target_state = np.zeros_like(c.final_state)
    target_state[2] = 1
    K.assert_allclose(c.final_state, target_state)


def test_final_state(backend, accelerators):
    """Check that final state is logged correctly when using measurements."""
    c = models.Circuit(4, accelerators)
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.M(0, 1))
    c.add(gates.M(2))
    c.add(gates.X(3))
    result = c(nshots=100)
    c = models.Circuit(4, accelerators)
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.X(3))
    target_state = c()
    K.assert_allclose(c.final_state, target_state)


def test_measurement_gate_bitflip_errors():
    gate = gates.M(0, 1, p0=2 * [0.1])
    with pytest.raises(ValueError):
        gate = gates.M(0, 1, p0=4 * [0.1])
    with pytest.raises(KeyError):
        gate = gates.M(0, 1, p0={0: 0.1, 2: 0.2})
    with pytest.raises(TypeError):
        gate = gates.M(0, 1, p0="test")
