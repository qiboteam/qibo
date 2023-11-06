import numpy as np
import pytest

from qibo import gates, models
from qibo import Circuit, gates
from qibo.result import CircuitResult, MeasurementOutcomes, load_result


@pytest.mark.parametrize("qubits", [[0, 2], [0], [1, 2]])
def test_measurementoutcomes_probabilties(backend, qubits):
    c = models.Circuit(3)
    c.add(gates.X(0))
    c.add(gates.M(0, 2))
    global_probs = c().probabilities(qubits=[0, 2])
    probabilities = c().probabilities(qubits=qubits)
    c.has_collapse = True
    if 1 in qubits:
        with pytest.raises(Exception) as exc_info:
            repeated_probabilities = c().probabilities(qubits=qubits)
            assert (
                str(exc_info.value)
                == "Asking probabilities for qubits that were not measured."
            )
    else:
        repeated_probabilities = c().probabilities(qubits=qubits)
        result = MeasurementOutcomes(
            c.measurements, backend, probabilities=global_probs
        )
        backend.assert_allclose(probabilities, repeated_probabilities, atol=1e-3)
        assert np.sum(repeated_probabilities - probabilities) == 0
        assert np.sum(result.probabilities(qubits) - probabilities) == 0


def test_circuit_result_error(backend):
    c = models.Circuit(1)
    state = np.array([1, 0])
    with pytest.raises(Exception) as exc_info:
        CircuitResult(state, c.measurements, backend)
    assert (
        str(exc_info.value)
        == "Circuit does not contain measurements. Use a `QuantumState` instead."
    )


def test_measurementoutcomes_errors(backend):
    c = models.Circuit(1)
    c.add(gates.M(0))
    samples = [np.array([1, 0]) for _ in range(5)]
    with pytest.raises(Exception) as exc_info:
        MeasurementOutcomes(c.measurements, backend)
    assert (
        str(exc_info.value)
        == "You have to provide either the `probabilities` or the `samples` to build a `MeasurementOutcomes` object."
    )
    with pytest.raises(Exception) as exc_info:
        MeasurementOutcomes(
            c.measurements, backend, probabilities=np.array([1, 0]), samples=samples
        )
    assert (
        str(exc_info.value)
        == "Both the `probabilities` and the `samples` were provided to build the `MeasurementOutcomes` object. Don't know which one to use."
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
    from os import remove

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
    from os import remove

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
