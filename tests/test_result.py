from os import remove

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.result import CircuitResult, MeasurementOutcomes, load_result


@pytest.mark.parametrize("qubits", [None, [0], [1, 2]])
def test_measurementoutcomes_probabilities(backend, qubits):
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0, 2))
    global_probs = backend.execute_circuit(circuit).probabilities(qubits=[0, 2])
    probabilities = (
        backend.execute_circuit(circuit).probabilities(qubits=qubits)
        if qubits is not None
        else backend.execute_circuit(circuit).probabilities(qubits=[0, 2])
    )
    circuit.has_collapse = True
    if qubits is not None and 1 in qubits:
        with pytest.raises(RuntimeError) as excinfo:
            repeated_probabilities = backend.execute_circuit(circuit).probabilities(
                qubits=qubits
            )
            assert (
                str(excinfo.value)
                == f"Asking probabilities for qubits {qubits}, but only qubits [0,2] were measured."
            )
    else:
        repeated_probabilities = backend.execute_circuit(
            circuit, nshots=1000
        ).probabilities(qubits=qubits)
        result = MeasurementOutcomes(
            circuit.measurements, backend=backend, probabilities=global_probs
        )
        backend.assert_allclose(probabilities, repeated_probabilities, atol=1e-1)
        backend.assert_allclose(result.probabilities(qubits), probabilities, atol=1e-1)


def test_measurementoutcomes_samples_from_measurements(backend):
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0, 2))
    res = backend.execute_circuit(circuit)
    samples = res.samples()
    outcome = MeasurementOutcomes(circuit.measurements, backend=backend)
    backend.assert_allclose(samples, outcome.samples())


def test_circuit_result_error(backend):
    c = Circuit(1)
    state = np.array([1, 0])
    with pytest.raises(Exception) as exc_info:
        CircuitResult(state, c.measurements, backend)
    assert (
        str(exc_info.value)
        == "Circuit does not contain measurements. Use a `QuantumState` instead."
    )


def test_measurement_gate_dump_load(backend):
    c = Circuit(2)
    c.add(gates.M(1, 0, basis=[gates.Z, gates.X]))
    m = c.measurements
    load = m[0].to_json()
    new_m = gates.M.load(load)
    assert new_m.to_json() == load


@pytest.mark.parametrize("agnostic_load", [False, True])
def test_measurementoutcomes_dump_load(backend, agnostic_load):
    c = Circuit(2)
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
    c = Circuit(2, density_matrix=True)
    c.add(gates.M(1, 0, basis=[gates.Z, gates.X]))
    # trigger repeated execution to build the CircuitResult object
    # from samples and recover the same exact frequencies
    c.has_collapse = True
    result = backend.execute_circuit(c)
    freq = result.frequencies()
    # set probabilities to trigger the warning
    result._probs = result.probabilities()
    result.dump("tmp.npy")
    loaded_res = (
        load_result("tmp.npy") if agnostic_load else CircuitResult.load("tmp.npy")
    )
    loaded_freq = loaded_res.frequencies()
    for state, f in freq.items():
        assert loaded_freq[state] == f
    assert backend.np.sum(result.state() - backend.cast(loaded_res.state())) == 0
    remove("tmp.npy")
