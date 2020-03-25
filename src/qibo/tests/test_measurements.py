import collections
import numpy as np
import pytest
from qibo import gates, models
from typing import Dict, Optional, Union


def _assert_results(result: Dict[str, Union[np.ndarray, collections.Counter]],
                    target: Dict[str, Union[np.ndarray, collections.Counter]]):
    if "decimal_samples" in target:
        np.testing.assert_allclose(result["decimal_samples"],
                                   target["decimal_samples"])
    if "binary_samples" in target:
        np.testing.assert_allclose(result["binary_samples"],
                                   target["binary_samples"])
    if "decimal_frequencies" in target:
        assert (result["decimal_frequencies"] ==
                collections.Counter(target["decimal_frequencies"]))
    if "binary_frequencies" in target:
        assert (result["binary_frequencies"] ==
                collections.Counter(target["binary_frequencies"]))


def assert_results(result, **target):
    results = {}
    if "decimal_samples" in target:
        results["decimal_samples"] = result.samples(False)
    if "binary_samples" in target:
        results["binary_samples"] = result.samples(True)
    if "decimal_frequencies" in target:
        results["decimal_frequencies"] = result.frequencies(False)
    if "binary_samples" in target:
        results["binary_frequencies"] = result.frequencies(True)
    _assert_results(results, target)


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
    """Check that multiple measurement gates in circuit."""
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
    """Check using many gates with multiple qubits each in the same circuit."""
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
    c = models.Circuit(2)
    c.add(gates.X(0))
    c.add(gates.M(0))
    c.add(gates.M(1))
    result = c(nshots=100)
