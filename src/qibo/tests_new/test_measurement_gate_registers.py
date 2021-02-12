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


def assert_register_results(result, decimal_samples=None, binary_samples=None,
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
    assert_register_results(result, decimal_samples, binary_samples,
                            decimal_frequencies, binary_frequencies)
    qibo.set_backend(original_backend)
