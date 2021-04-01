"""Test :class:`qibo.abstractions.gates.M` when used with registers."""
import pytest
import numpy as np
import qibo
from qibo import models, gates


def assert_dicts_equal(d1, d2):
    assert d1.keys() == d2.keys()
    for k, v in d1.items():
        if isinstance(v, dict):
            assert v == d2[k]
        else:
            np.testing.assert_allclose(v, d2[k])


def assert_register_result(result, decimal_samples=None, binary_samples=None,
                           decimal_frequencies=None, binary_frequencies=None):
    if decimal_samples:
        register_result = result.samples(binary=False, registers=True)
        assert_dicts_equal(register_result, decimal_samples)
    if binary_samples:
        register_result = result.samples(binary=True, registers=True)
        assert_dicts_equal(register_result, binary_samples)
    if decimal_frequencies:
        register_result = result.frequencies(binary=False, registers=True)
        assert_dicts_equal(register_result, decimal_frequencies)
    if binary_frequencies:
        register_result = result.frequencies(binary=True, registers=True)
        assert_dicts_equal(register_result, binary_frequencies)


def test_register_measurements(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = models.Circuit(3)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.M(0, 2))
    c.add(gates.M(1))
    result = c(nshots=100)

    decimal_samples = {"register0": 2 * np.ones((100,)),
                        "register1": np.ones((100,))}
    binary_samples = {"register0": np.zeros((100, 2)),
                      "register1": np.ones((100, 1))}
    binary_samples["register0"][:, 0] = 1
    decimal_frequencies = {"register0": {2: 100}, "register1": {1: 100}}
    binary_frequencies = {"register0": {"10": 100}, "register1": {"1": 100}}
    assert_register_result(result, decimal_samples, binary_samples,
                           decimal_frequencies, binary_frequencies)
    qibo.set_backend(original_backend)


def test_register_name_error(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = models.Circuit(2)
    c.add(gates.X(0))
    c.add(gates.M(0, register_name="a"))
    with pytest.raises(KeyError):
        c.add(gates.M(1, register_name="a"))
    qibo.set_backend(original_backend)


def test_registers_with_same_name_error(backend):
    """Check that circuits that contain registers with the same name cannot be added."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c1 = models.Circuit(2)
    c1.add(gates.H(0))
    c1.add(gates.M(0))

    c2 = models.Circuit(2)
    c2.add(gates.H(1))
    c2.add(gates.M(1))

    with pytest.raises(KeyError):
        c = c1 + c2
    qibo.set_backend(original_backend)


def test_measurement_qubit_order_multiple_registers(backend, accelerators):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = models.Circuit(6, accelerators)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.X(3))
    c.add(gates.M(5, 1, 3, register_name="a"))
    c.add(gates.M(2, 0, register_name="b"))
    result = c(nshots=100)

    # Check full result
    from qibo.tests_new.test_measurement_gate import assert_result
    target_binary_samples = np.zeros((100, 5))
    target_binary_samples[:, 1] = 1
    target_binary_samples[:, 2] = 1
    target_binary_samples[:, 4] = 1
    assert_result(result, 13 * np.ones((100,)), target_binary_samples,
                  {13: 100}, {"01101": 100})

    decimal_samples = {"a": 3 * np.ones((100,)), "b": np.ones((100,))}
    binary_samples = {"a": np.zeros((100, 3)), "b": np.zeros((100, 2))}
    binary_samples["a"][:, 1] = 1
    binary_samples["a"][:, 2] = 1
    binary_samples["b"][:, 1] = 1
    decimal_frequencies = {"a": {3: 100}, "b": {1: 100}}
    binary_frequencies = {"a": {"011": 100}, "b": {"01": 100}}
    assert_register_result(result, decimal_samples, binary_samples,
                           decimal_frequencies, binary_frequencies)
    qibo.set_backend(original_backend)


def test_registers_in_circuit_with_unmeasured_qubits(backend, accelerators):
    """Check that register measurements are unaffected by unmeasured qubits."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = models.Circuit(5, accelerators)
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.M(0, 2, register_name="A"))
    c.add(gates.X(3))
    c.add(gates.M(1, 4, register_name="B"))
    result = c(nshots=100)

    target = {}
    decimal_samples = {"A": np.ones((100,)), "B": 2 * np.ones((100,))}
    binary_samples = {"A": np.zeros((100, 2)), "B": np.zeros((100, 2))}
    binary_samples["A"][:, 1] = 1
    binary_samples["B"][:, 0] = 1
    decimal_frequencies = {"A": {1: 100}, "B": {2: 100}}
    binary_frequencies = {"A": {"01": 100}, "B": {"10": 100}}
    assert_register_result(result, decimal_samples, binary_samples,
                           decimal_frequencies, binary_frequencies)
    qibo.set_backend(original_backend)
