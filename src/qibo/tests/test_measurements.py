import collections
import numpy as np
import pytest
from qibo import gates, models
from typing import Optional


def assert_results(result,
                   decimal_samples: Optional[np.ndarray] = None,
                   binary_samples: Optional[np.ndarray] = None,
                   decimal_frequencies: Optional[collections.Counter] = None,
                   binary_frequencies: Optional[collections.Counter] = None):
  if decimal_samples is not None:
      np.testing.assert_allclose(result.samples(False).numpy(), decimal_samples)
  if binary_samples is not None:
      np.testing.assert_allclose(result.samples(True).numpy(), binary_samples)
  if decimal_frequencies is not None:
      assert result.frequencies(False) == collections.Counter(decimal_frequencies)
  if binary_frequencies is not None:
      assert result.frequencies(True) == collections.Counter(binary_frequencies)


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


def test_convert_to_binary():
    """Check that `_convert_to_binary` method works properly."""
    # Create a result object to access `_convert_to_binary`
    state = np.zeros(4)
    state[0] = 1
    state = state.reshape((2, 2))
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
    state = state.reshape((2, 2))
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
    state = state.reshape((2, 2))
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
    state = state.reshape((2, 2))
    result = gates.M(1)(state, nshots=100)
    assert_results(result,
                   decimal_samples=np.ones((100,)),
                   binary_samples=np.ones((100, 1)),
                   decimal_frequencies={1: 100},
                   binary_frequencies={"1": 100})


def test_multiple_qubit_measurement_gate():
    """Check that multiple qubit measurement gate works when called on |10>."""
    state = np.zeros(4)
    state[2] = 1
    state = state.reshape((2, 2))
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


def test_measurement_circuit():
    """Check that measurement gate works as part of circuit."""
    c = models.Circuit(2)
    c.add(gates.X(0))
    c.add(gates.M(0))
    result = c(nshots=100)
    assert_results(result,
                   decimal_samples=np.ones((100,)),
                   binary_samples=np.ones((100, 1)),
                   decimal_frequencies={1: 100},
                   binary_frequencies={"1": 100})


def test_gate_after_measurement_error():
    """Check that reusing measured qubits is not allowed."""
    c = models.Circuit(2)
    c.add(gates.X(0))
    c.add(gates.M(0))
    c.add(gates.X(1))
    with pytest.raises(ValueError):
        c.add(gates.H(0))


def test_multiple_qubit_measurement_circuit():
    """Check multiple measurement gates in circuit."""
    c = models.Circuit(2)
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


def test_final_state():
    """Check that final state is logged correctly when using measurements."""
    c = models.Circuit(4)
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.M(0, 1))
    c.add(gates.M(2))
    c.add(gates.X(3))
    result = c(nshots=100)
    logged_final_state = c.final_state.numpy()

    c = models.Circuit(4)
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.X(3))
    target_state = c().numpy()

    np.testing.assert_allclose(logged_final_state, target_state)


def test_circuit_with_unmeasured_qubits():
    """Check that unmeasured qubits are not taken into account."""
    c = models.Circuit(5)
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.M(0, 2))
    c.add(gates.X(3))
    c.add(gates.M(1, 4))
    result = c(nshots=100)

    target_binary_samples = np.zeros((100, 4))
    target_binary_samples[:, 1] = 1
    target_binary_samples[:, 2] = 1
    assert_results(result,
                   decimal_samples=6 * np.ones((100,)),
                   binary_samples=target_binary_samples,
                   decimal_frequencies={6: 100},
                   binary_frequencies={"0110": 100})


def test_measurement_compiled_circuit():
    """Check that measurements and final state work for compiled circuits."""
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


def test_registers_in_circuit_with_unmeasured_qubits():
    """Check that register measurements are unaffected by unmeasured qubits."""
    c = models.Circuit(5)
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


def test_probabilistic_measurement():
    import tensorflow as tf
    tf.random.set_seed(1234)

    c = models.Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.M(0, 1))
    result = c(nshots=1000)

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

    decimal_freqs = {0: 168, 1: 188, 2: 154, 3: 490}
    binary_freqs = {"00": 168, "01": 188, "10": 154, "11": 490}
    assert sum(binary_freqs.values()) == 1000
    assert_results(result,
                   decimal_frequencies=decimal_freqs,
                   binary_frequencies=binary_freqs)
