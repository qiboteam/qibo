from os import remove

import numpy as np
import pytest

from qibo import gates, models
from qibo.result import CircuitResult, MeasurementOutcomes, load_result


def test_circuit_result_error(backend):
    c = models.Circuit(1)
    state = np.array([1, 0])
    with pytest.raises(Exception) as exc_info:
        CircuitResult(state, c.measurements, backend)
    assert (
        str(exc_info.value)
        == "Circuit does not contain measurements. Use a `QuantumState` instead."
    )


def test_measurement_gate_dump_load(backend):
    c = models.Circuit(2)
    c.add(gates.M(1, 0, basis=[gates.Z, gates.X]))
    m = c.measurements
    load = m[0].to_json()
    new_m = gates.M.load(load)
    assert new_m.to_json() == load


@pytest.mark.parametrize("agnostic_load", [False, True])
def test_measurementoutcomes_dump_load(backend, agnostic_load):
    c = models.Circuit(2)
    c.add(gates.M(1, 0, basis=[gates.Z, gates.X]))
    # just to trigger repeated execution and test MeasurementOutcomes
    c.has_collapse = True
    measurement = backend.execute_circuit(c, nshots=100)
    freq = measurement.frequencies()
    measurement.dump("tmp.npy")
    if agnostic_load:
        loaded_meas = load_result("tmp.npy")
    else:
        loaded_meas = MeasurementOutcomes.load("tmp.npy")
    loaded_freq = loaded_meas.frequencies()
    for state, f in freq.items():
        assert loaded_freq[state] == f
    remove("tmp.npy")


@pytest.mark.parametrize("agnostic_load", [False, True])
def test_circuitresult_dump_load(backend, agnostic_load):
    c = models.Circuit(2, density_matrix=True)
    c.add(gates.M(1, 0, basis=[gates.Z, gates.X]))
    # trigger repeated execution to build the CircuitResult object
    # from samples and recover the same exact frequencies
    c.has_collapse = True
    result = backend.execute_circuit(c)
    freq = result.frequencies()
    # set probabilities to trigger the warning
    result._probs = result.probabilities()
    result.dump("tmp.npy")
    if agnostic_load:
        loaded_res = load_result("tmp.npy")
    else:
        loaded_res = CircuitResult.load("tmp.npy")
    loaded_freq = loaded_res.frequencies()
    for state, f in freq.items():
        assert loaded_freq[state] == f
    assert np.sum(result.state() - loaded_res.state()) == 0
    remove("tmp.npy")
