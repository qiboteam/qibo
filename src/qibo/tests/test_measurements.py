import collections
import numpy as np
import pytest
from qibo import gates, models
from typing import Optional

_ACCELERATORS = [None, {"/GPU:0": 2}, {"/GPU:0": 1, "/GPU:1": 1}]


def assert_results(result,
                   decimal_samples: Optional[np.ndarray] = None,
                   binary_samples: Optional[np.ndarray] = None,
                   decimal_frequencies: Optional[collections.Counter] = None,
                   binary_frequencies: Optional[collections.Counter] = None):
    if decimal_samples is not None:
        np.testing.assert_allclose(
            result.samples(False).numpy(), decimal_samples)
    if binary_samples is not None:
        np.testing.assert_allclose(
            result.samples(True).numpy(), binary_samples)
    if decimal_frequencies is not None:
        assert result.frequencies(
            False) == collections.Counter(decimal_frequencies)
    if binary_frequencies is not None:
        assert result.frequencies(
            True) == collections.Counter(binary_frequencies)


def assert_register_results(
        result,
        decimal_samples: Optional[np.ndarray] = None,
        binary_samples: Optional[np.ndarray] = None,
        decimal_frequencies: Optional[collections.Counter] = None,
        binary_frequencies: Optional[collections.Counter] = None):
    if decimal_samples is not None:
        register_result = result.samples(binary=False, registers=True)
        assert register_result.keys() == decimal_samples.keys()
        for k, v in register_result.items():
            np.testing.assert_allclose(v.numpy(), decimal_samples[k])
    if binary_samples is not None:
        register_result = result.samples(binary=True, registers=True)
        assert register_result.keys() == binary_samples.keys()
        for k, v in register_result.items():
            np.testing.assert_allclose(v.numpy(), binary_samples[k])

    if decimal_frequencies is not None:
        register_result = result.frequencies(binary=False, registers=True)
        assert register_result.keys() == decimal_frequencies.keys()
        for k, v in register_result.items():
            assert v == collections.Counter(decimal_frequencies[k])
    if binary_frequencies is not None:
        register_result = result.frequencies(binary=True, registers=True)
        assert register_result.keys() == binary_frequencies.keys()
        for k, v in register_result.items():
            assert v == collections.Counter(binary_frequencies[k])


def test_gate_result_initialization_errors():
    """Check ``ValueError``s during the initialization of ``GateResult`` object."""
    from qibo.base import measurements
    decimal_samples = np.random.randint(0, 4, (100,))
    binary_samples = np.random.randint(0, 2, (100, 2))
    with pytest.raises(ValueError):
        res = measurements.GateResult((0, 1), decimal_samples=decimal_samples,
                                      binary_samples=binary_samples)
    binary_samples = np.random.randint(0, 2, (100, 4))
    with pytest.raises(ValueError):
        res = measurements.GateResult((0, 1), binary_samples=binary_samples)


def test_convert_to_binary():
    """Check that `_convert_to_binary` method works properly."""
    # Create a result object to access `_convert_to_binary`
    state = np.zeros(4)
    state[0] = 1
    result = gates.M(0)(state, nshots=100)

    import itertools
    nbits = 5
    binary_samples = result._convert_to_binary(np.arange(2 ** nbits),
                                               nbits).numpy()
    target_samples = np.array(list(itertools.product([0, 1], repeat=nbits)))
    np.testing.assert_allclose(binary_samples, target_samples)


def test_convert_to_decimal():
    """Check that `_convert_to_decimal` method works properly."""
    # Create a result object to access `_convert_to_decimal`
    state = np.zeros(4)
    state[0] = 1
    result = gates.M(0)(state, nshots=100)

    import itertools
    nbits = 5
    binary_samples = np.array(list(itertools.product([0, 1], repeat=nbits)))
    decimal_samples = result._convert_to_decimal(binary_samples, nbits).numpy()
    target_samples = np.arange(2 ** nbits)
    np.testing.assert_allclose(decimal_samples, target_samples)


def test_measurement_gate():
    """Check that measurement gate works when called on the state |00>."""
    state = np.zeros(4)
    state[0] = 1
    result = gates.M(0)(state, nshots=100)
    assert_results(result,
                   decimal_samples=np.zeros((100,)),
                   binary_samples=np.zeros((100, 1)),
                   decimal_frequencies={0: 100},
                   binary_frequencies={"0": 100})


def test_measurement_gate2():
    """Check that measurement gate works when called on the state |11>."""
    state = np.zeros(4)
    state[-1] = 1
    result = gates.M(1)(state, nshots=100)
    assert_results(result,
                   decimal_samples=np.ones((100,)),
                   binary_samples=np.ones((100, 1)),
                   decimal_frequencies={1: 100},
                   binary_frequencies={"1": 100})


def test_measurement_gate_errors():
    """Check various errors that are raised by the measurement gate."""
    state = np.zeros(4)
    state[-1] = 1
    # add targets after calling
    gate = gates.M(1)
    result = gate(state, nshots=100)
    with pytest.raises(RuntimeError):
        gate._add((0,))
    # try to set unmeasured qubits before setting ``nqubits``
    gate = gates.M(1)
    with pytest.raises(RuntimeError):
        gate._set_unmeasured_qubits()
    # try to set unmeasured qubit a second time
    gate = gates.M(1)
    gate.nqubits = 3
    gate._unmeasured_qubits = (0, 2)
    with pytest.raises(RuntimeError):
        gate._set_unmeasured_qubits()
    # get reduced target qubits
    gate = gates.M(1)
    gate.nqubits = 3
    assert gate.reduced_target_qubits == [0]


def test_multiple_qubit_measurement_gate():
    """Check that multiple qubit measurement gate works when called on |10>."""
    state = np.zeros(4)
    state[2] = 1
    result = gates.M(0, 1)(state, nshots=100)

    target_binary_samples = np.zeros((100, 2))
    target_binary_samples[:, 0] = 1
    assert_results(result,
                   decimal_samples=2 * np.ones((100,)),
                   binary_samples=target_binary_samples,
                   decimal_frequencies={2: 100},
                   binary_frequencies={"10": 100})


def test_controlled_measurement_error():
    """Check that using `controlled_by` in measurements raises error."""
    with pytest.raises(NotImplementedError):
        m = gates.M(0).controlled_by(1)


@pytest.mark.parametrize("accelerators", _ACCELERATORS)
def test_measurement_circuit(accelerators):
    """Check that measurement gate works as part of circuit."""
    c = models.Circuit(2, accelerators)
    c.add(gates.X(0))
    c.add(gates.M(0))
    result = c(nshots=100)
    assert_results(result,
                   decimal_samples=np.ones((100,)),
                   binary_samples=np.ones((100, 1)),
                   decimal_frequencies={1: 100},
                   binary_frequencies={"1": 100})


@pytest.mark.parametrize("accelerators", _ACCELERATORS)
def test_gate_after_measurement_error(accelerators):
    """Check that reusing measured qubits is not allowed."""
    c = models.Circuit(2, accelerators)
    c.add(gates.X(0))
    c.add(gates.M(0))
    c.add(gates.X(1))
    # TODO: Change this to NotImplementedError
    with pytest.raises(ValueError):
        c.add(gates.H(0))


def test_register_name_error():
    """Check that using the same register name twice results to error."""
    c = models.Circuit(2)
    c.add(gates.X(0))
    c.add(gates.M(0, register_name="a"))
    with pytest.raises(KeyError):
        c.add(gates.M(1, register_name="a"))


@pytest.mark.parametrize("accelerators", _ACCELERATORS)
def test_multiple_qubit_measurement_circuit(accelerators):
    """Check multiple measurement gates in circuit."""
    c = models.Circuit(2, accelerators)
    c.add(gates.X(1))
    c.add(gates.M(0))
    c.add(gates.M(1))
    result = c(nshots=100)

    target_binary_samples = np.zeros((100, 2))
    target_binary_samples[:, 1] = 1
    assert_results(result,
                   decimal_samples=np.ones((100,)),
                   binary_samples=target_binary_samples,
                   decimal_frequencies={1: 100},
                   binary_frequencies={"01": 100})


def test_measurement_qubit_order_simple():
    """Check that measurement results follow order defined by user."""
    c = models.Circuit(2)
    c.add(gates.X(0))
    c.add(gates.M(1, 0))
    result1 = c(nshots=100)

    c = models.Circuit(2)
    c.add(gates.X(0))
    c.add(gates.M(1))
    c.add(gates.M(0))
    result2 = c(nshots=100)

    target_binary_samples = np.zeros((100, 2))
    target_binary_samples[:, 1] = 1
    target = {"decimal_samples": np.ones((100,)),
              "binary_samples": target_binary_samples,
              "decimal_frequencies": {1: 100},
              "binary_frequencies": {"01": 100}}
    assert_results(result1, **target)
    assert_results(result2, **target)


@pytest.mark.parametrize("accelerators", _ACCELERATORS)
def test_measurement_qubit_order(accelerators):
    """Check that measurement results follow order defined by user."""
    c = models.Circuit(6, accelerators)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.M(1, 5, 2, 0))
    result = c(nshots=100)

    target_binary_samples = np.zeros((100, 4))
    target_binary_samples[:, 0] = 1
    target_binary_samples[:, 3] = 1
    assert_results(result,
                   decimal_samples=9 * np.ones((100,)),
                   binary_samples=target_binary_samples,
                   decimal_frequencies={9: 100},
                   binary_frequencies={"1001": 100})


@pytest.mark.parametrize("accelerators", _ACCELERATORS)
def test_measurement_qubit_order_multiple_registers(accelerators):
    """Check that measurement results follow order defined by user."""
    c = models.Circuit(6, accelerators)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.X(3))
    c.add(gates.M(5, 1, 3, register_name="a"))
    c.add(gates.M(2, 0, register_name="b"))
    result = c(nshots=100)

    # Check full result
    target_binary_samples = np.zeros((100, 5))
    target_binary_samples[:, 1] = 1
    target_binary_samples[:, 2] = 1
    target_binary_samples[:, 4] = 1
    assert_results(result,
                   decimal_samples=13 * np.ones((100,)),
                   binary_samples=target_binary_samples,
                   decimal_frequencies={13: 100},
                   binary_frequencies={"01101": 100})

    target = {}
    target["decimal_samples"] = {"a": 3 * np.ones((100,)),
                                 "b": np.ones((100,))}
    target["binary_samples"] = {"a": np.zeros((100, 3)),
                                "b": np.zeros((100, 2))}
    target["binary_samples"]["a"][:, 1] = 1
    target["binary_samples"]["a"][:, 2] = 1
    target["binary_samples"]["b"][:, 1] = 1

    target["decimal_frequencies"] = {"a": {3: 100}, "b": {1: 100}}
    target["binary_frequencies"] = {"a": {"011": 100}, "b": {"01": 100}}
    assert_register_results(result, **target)


def test_multiple_measurement_gates_circuit():
    """Check multiple gates with multiple qubits each in the same circuit."""
    c = models.Circuit(4)
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.M(0, 1))
    c.add(gates.M(2))
    c.add(gates.X(3))
    result = c(nshots=100)

    target_binary_samples = np.ones((100, 3))
    target_binary_samples[:, 0] = 0
    assert_results(result,
                   decimal_samples=3 * np.ones((100,)),
                   binary_samples=target_binary_samples,
                   decimal_frequencies={3: 100},
                   binary_frequencies={"011": 100})


@pytest.mark.parametrize("accelerators", _ACCELERATORS)
def test_final_state(accelerators):
    """Check that final state is logged correctly when using measurements."""
    c = models.Circuit(4, accelerators)
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.M(0, 1))
    c.add(gates.M(2))
    c.add(gates.X(3))
    result = c(nshots=100)
    logged_final_state = c.final_state.numpy()

    c = models.Circuit(4, accelerators)
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.X(3))
    target_state = c().numpy()

    np.testing.assert_allclose(logged_final_state, target_state)


@pytest.mark.parametrize("accelerators", _ACCELERATORS)
def test_circuit_with_unmeasured_qubits(accelerators):
    """Check that unmeasured qubits are not taken into account."""
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
    assert_results(result,
                   decimal_samples=5 * np.ones((100,)),
                   binary_samples=target_binary_samples,
                   decimal_frequencies={5: 100},
                   binary_frequencies={"0101": 100})


def test_measurement_compiled_circuit():
    """Check that measurements and final state work for compiled circuits."""
    # use native gates because custom gates do not support compilation
    import qibo
    original_backend = qibo.get_backend()
    qibo.set_backend("matmuleinsum")
    c = models.Circuit(2)
    c.add(gates.X(0))
    c.add(gates.M(0))
    c.add(gates.M(1))
    c.compile()
    result = c(nshots=100)

    target_binary_samples = np.zeros((100, 2))
    target_binary_samples[:, 0] = 1
    assert_results(result,
                   decimal_samples=2 * np.ones((100,)),
                   binary_samples=target_binary_samples,
                   decimal_frequencies={2: 100},
                   binary_frequencies={"10": 100})

    final_state = c.final_state.numpy()
    target_state = np.zeros_like(final_state)
    target_state[2] = 1

    # reset backend for next tests
    qibo.set_backend(original_backend)

    np.testing.assert_allclose(final_state, target_state)


def test_register_measurements():
    """Check register measurements are split properly."""
    c = models.Circuit(3)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.M(0, 2))
    c.add(gates.M(1))
    result = c(nshots=100)

    target = {}
    target["decimal_samples"] = {"register0": 2 * np.ones((100,)),
                                 "register1": np.ones((100,))}
    target["binary_samples"] = {"register0": np.zeros((100, 2)),
                                "register1": np.ones((100, 1))}
    target["binary_samples"]["register0"][:, 0] = 1

    target["decimal_frequencies"] = {"register0": {2: 100},
                                     "register1": {1: 100}}
    target["binary_frequencies"] = {"register0": {"10": 100},
                                    "register1": {"1": 100}}
    assert_register_results(result, **target)


@pytest.mark.parametrize("accelerators", _ACCELERATORS)
def test_registers_in_circuit_with_unmeasured_qubits(accelerators):
    """Check that register measurements are unaffected by unmeasured qubits."""
    c = models.Circuit(5, accelerators)
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.M(0, 2, register_name="A"))
    c.add(gates.X(3))
    c.add(gates.M(1, 4, register_name="B"))
    result = c(nshots=100)

    target = {}
    target["decimal_samples"] = {"A": np.ones((100,)),
                                 "B": 2 * np.ones((100,))}
    target["binary_samples"] = {"A": np.zeros((100, 2)),
                                "B": np.zeros((100, 2))}
    target["binary_samples"]["A"][:, 1] = 1
    target["binary_samples"]["B"][:, 0] = 1
    target["decimal_frequencies"] = {"A": {1: 100}, "B": {2: 100}}
    target["binary_frequencies"] = {"A": {"01": 100}, "B": {"10": 100}}
    assert_register_results(result, **target)


@pytest.mark.parametrize("accelerators", _ACCELERATORS)
def test_probabilistic_measurement(accelerators):
    import tensorflow as tf
    tf.random.set_seed(1234)

    c = models.Circuit(2, accelerators)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.M(0, 1))
    result = c(nshots=1000)

    # update reference values based on device
    if tf.config.list_physical_devices("GPU"): # pragma: no cover
        # case not tested in GitHub workflows because it requires GPU
        decimal_freqs = {0: 273, 1: 233, 2: 242, 3: 252}
        binary_freqs = {"00": 273, "01": 233, "10": 242, "11": 252}
    else:
        decimal_freqs = {0: 271, 1: 239, 2: 242, 3: 248}
        binary_freqs = {"00": 271, "01": 239, "10": 242, "11": 248}
    assert sum(binary_freqs.values()) == 1000
    assert_results(result,
                   decimal_frequencies=decimal_freqs,
                   binary_frequencies=binary_freqs)


def test_unbalanced_probabilistic_measurement():
    import tensorflow as tf
    tf.random.set_seed(1234)

    state = np.array([1, 1, 1, np.sqrt(3)]) / 2.0
    c = models.Circuit(2)
    c.add(gates.Flatten(state))
    c.add(gates.M(0, 1))
    result = c(nshots=1000)

    # update reference values based on device
    if tf.config.list_physical_devices("GPU"): # pragma: no cover
        # case not tested in GitHub workflows because it requires GPU
        decimal_freqs = {0: 196, 1: 153, 2: 156, 3: 495}
        binary_freqs = {"00": 196, "01": 153, "10": 156, "11": 495}
    else:
        decimal_freqs = {0: 168, 1: 188, 2: 154, 3: 490}
        binary_freqs = {"00": 168, "01": 188, "10": 154, "11": 490}
    assert sum(binary_freqs.values()) == 1000
    assert_results(result,
                   decimal_frequencies=decimal_freqs,
                   binary_frequencies=binary_freqs)


def test_circuit_addition_with_measurements():
    """Check if measurements are transferred during circuit addition."""
    c = models.Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))

    meas_c = models.Circuit(2)
    c.add(gates.M(0, 1))

    c += meas_c
    assert len(c.measurement_gate.target_qubits) == 2
    assert c.measurement_tuples == {"register0": (0, 1)}


@pytest.mark.parametrize("accelerators", _ACCELERATORS)
def test_circuit_addition_with_measurements_in_both_circuits(accelerators):
    """Check if measurements of two circuits are added during circuit addition."""
    c1 = models.Circuit(2, accelerators)
    c1.add(gates.H(0))
    c1.add(gates.H(1))
    c1.add(gates.M(1, register_name="a"))

    c2 = models.Circuit(2, accelerators)
    c2.add(gates.X(0))
    c2.add(gates.M(0, register_name="b"))

    c = c1 + c2
    assert len(c.measurement_gate.target_qubits) == 2
    assert c.measurement_tuples == {"a": (1,), "b": (0,)}


@pytest.mark.parametrize("accelerators", _ACCELERATORS)
def test_gate_after_measurement_with_addition_error(accelerators):
    """Check that measured qubits cannot be reused by adding gates."""
    c = models.Circuit(2, accelerators)
    c.add(gates.H(0))
    c.add(gates.M(1))

    # Try to add gate to qubit that is already measured
    c2 = models.Circuit(2, accelerators)
    c2.add(gates.H(1))
    with pytest.raises(ValueError):
        c += c2
    # Try to add measurement to qubit that is already measured
    c2 = models.Circuit(2, accelerators)
    c2.add(gates.M(1, register_name="a"))
    with pytest.raises(ValueError):
        c += c2


def test_registers_with_same_name_error():
    """Check that registers with the same name cannot be added."""
    c1 = models.Circuit(2)
    c1.add(gates.H(0))
    c1.add(gates.M(0))

    c2 = models.Circuit(2)
    c2.add(gates.H(1))
    c2.add(gates.M(1))

    with pytest.raises(KeyError):
        c = c1 + c2


@pytest.mark.parametrize("accelerators", _ACCELERATORS)
def test_copy_measurements(accelerators):
    """Check that ``circuit.copy()`` properly copies measurements."""
    c1 = models.Circuit(6, accelerators)
    c1.add([gates.X(0), gates.X(1), gates.X(3)])
    c1.add(gates.M(5, 1, 3, register_name="a"))
    c1.add(gates.M(2, 0, register_name="b"))
    c2 = c1.copy()

    r1 = c1(nshots=100)
    r2 = c2(nshots=100)

    np.testing.assert_allclose(r1.samples().numpy(), r2.samples().numpy())
    rg1 = r1.frequencies(registers=True)
    rg2 = r2.frequencies(registers=True)
    assert rg1.keys() == rg2.keys()
    for k in rg1.keys():
        assert rg1[k] == rg2[k]
