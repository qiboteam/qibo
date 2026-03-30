import collections
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
        backend.assert_allclose(repeated_probabilities, probabilities, atol=1e-1)
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
    circuit = Circuit(1)
    state = np.array([1, 0])
    with pytest.raises(Exception) as exc_info:
        CircuitResult(state, circuit.measurements, backend)
    assert (
        str(exc_info.value)
        == "Circuit does not contain measurements. Use a `QuantumState` instead."
    )


def test_measurement_gate_dump_load(backend):
    circuit = Circuit(2)
    circuit.add(gates.M(1, 0, basis=[gates.Z, gates.X]))
    m = circuit.measurements
    load = m[0].to_json()
    new_m = gates.M.load(load)
    assert new_m.to_json() == load


@pytest.mark.parametrize("agnostic_load", [False, True])
def test_measurementoutcomes_dump_load(backend, agnostic_load):
    circuit = Circuit(2)
    circuit.add(gates.M(1, 0, basis=[gates.Z, gates.X]))
    # just to trigger repeated execution and test MeasurementOutcomes
    circuit.has_collapse = True
    measurement = backend.execute_circuit(circuit, nshots=100)
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
    circuit = Circuit(2, density_matrix=True)
    circuit.add(gates.M(1, 0, basis=[gates.Z, gates.X]))
    # trigger repeated execution to build the CircuitResult object
    # from samples and recover the same exact frequencies
    circuit.has_collapse = True
    result = backend.execute_circuit(circuit)
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
    assert backend.sum(result.state() - backend.cast(loaded_res.state())) == 0
    remove("tmp.npy")


def test_measurementoutcomes_from_samples(backend):
    """Test constructing MeasurementOutcomes from a raw samples array."""
    samples = np.array(
        [[0, 1], [1, 0], [1, 1], [0, 1], [1, 0]],
        dtype=int,
    )
    result = MeasurementOutcomes.from_samples(samples, backend=backend)

    # Check samples round-trip
    backend.assert_allclose(result.samples(), samples)

    # Check frequencies
    freq = result.frequencies(binary=True)
    assert freq["01"] == 2
    assert freq["10"] == 2
    assert freq["11"] == 1

    # Check nshots
    assert result.nshots == 5

    # Check decimal samples
    dec = result.samples(binary=False)
    assert dec.shape == (5,)

    # Check probabilities sum to 1
    probs = result.probabilities()
    backend.assert_allclose(backend.sum(probs), 1.0)


def test_measurementoutcomes_from_samples_custom_qubits(backend):
    """Test from_samples with explicit qubit indices."""
    samples = np.array([[0, 1], [1, 0]], dtype=int)
    result = MeasurementOutcomes.from_samples(samples, qubits=[2, 5], backend=backend)
    assert result.measurement_gate.qubits == (2, 5)
    assert result.nshots == 2


def test_measurementoutcomes_from_samples_errors(backend):
    """Test from_samples raises errors on invalid input."""
    # 1D array
    with pytest.raises(ValueError):
        MeasurementOutcomes.from_samples(np.array([0, 1, 1]), backend=backend)

    # qubits length mismatch
    with pytest.raises(ValueError):
        MeasurementOutcomes.from_samples(
            np.array([[0, 1], [1, 0]]), qubits=[0, 1, 2], backend=backend
        )


def test_measurementoutcomes_from_frequencies_binary_keys(backend):
    """Test constructing MeasurementOutcomes from binary-string frequencies."""
    freq_input = {"00": 30, "01": 20, "10": 10, "11": 40}
    result = MeasurementOutcomes.from_frequencies(freq_input, backend=backend)

    # Check nshots
    assert result.nshots == 100

    # Check frequencies round-trip
    freq_out = result.frequencies(binary=True)
    for key, count in freq_input.items():
        assert freq_out[key] == count

    # Check nqubits inferred correctly
    assert len(result.measurement_gate.qubits) == 2

    # Check probabilities sum to 1
    probs = result.probabilities()
    backend.assert_allclose(backend.sum(probs), 1.0)


def test_measurementoutcomes_from_frequencies_integer_keys(backend):
    """Test constructing MeasurementOutcomes from integer-keyed frequencies."""
    freq_input = {0: 50, 3: 50}
    result = MeasurementOutcomes.from_frequencies(
        freq_input, nqubits=2, backend=backend
    )

    assert result.nshots == 100

    freq_out = result.frequencies(binary=True)
    assert freq_out["00"] == 50
    assert freq_out["11"] == 50


def test_measurementoutcomes_from_frequencies_integer_keys_with_qubits(backend):
    """Test from_frequencies with integer keys and explicit qubits."""
    freq_input = {0: 25, 1: 75}
    result = MeasurementOutcomes.from_frequencies(
        freq_input, qubits=[3, 7], backend=backend
    )

    assert result.nshots == 100
    assert result.measurement_gate.qubits == (3, 7)

    freq_out = result.frequencies(binary=True)
    assert freq_out["00"] == 25
    assert freq_out["01"] == 75


def test_measurementoutcomes_from_frequencies_counter(backend):
    """Test from_frequencies accepts a collections.Counter."""
    freq_input = collections.Counter({"010": 10, "111": 20, "000": 30})
    result = MeasurementOutcomes.from_frequencies(freq_input, backend=backend)

    assert result.nshots == 60
    assert len(result.measurement_gate.qubits) == 3

    freq_out = result.frequencies(binary=True)
    assert freq_out["010"] == 10
    assert freq_out["111"] == 20
    assert freq_out["000"] == 30


def test_measurementoutcomes_from_frequencies_errors(backend):
    """Test from_frequencies raises errors on invalid input."""
    # Integer keys without nqubits or qubits
    with pytest.raises(ValueError):
        MeasurementOutcomes.from_frequencies({0: 10, 1: 20}, backend=backend)

    # Empty dict
    with pytest.raises(ValueError):
        MeasurementOutcomes.from_frequencies({}, backend=backend)

    # Inconsistent binary key lengths
    with pytest.raises(ValueError):
        MeasurementOutcomes.from_frequencies({"00": 10, "111": 20}, backend=backend)


def test_measurementoutcomes_from_frequencies_dump_load(backend, tmp_path):
    """Test that from_frequencies results can be dumped and reloaded."""
    freq_input = {"00": 40, "01": 10, "10": 20, "11": 30}
    result = MeasurementOutcomes.from_frequencies(freq_input, backend=backend)
    freq = result.frequencies()

    filepath = str(tmp_path / "tmp_from_freq.npy")
    result.dump(filepath)
    loaded = MeasurementOutcomes.load(filepath)
    loaded_freq = loaded.frequencies()

    for state, f in freq.items():
        assert loaded_freq[state] == f


def test_measurementoutcomes_from_frequencies_negative_count(backend):
    with pytest.raises(ValueError):
        MeasurementOutcomes.from_frequencies({"00": 10, "01": -1}, backend=backend)


def test_measurementoutcomes_from_frequencies_overflow_state(backend):
    with pytest.raises(ValueError):
        MeasurementOutcomes.from_frequencies({4: 1}, nqubits=2, backend=backend)


def test_measurementoutcomes_from_samples_nonbinary(backend):
    with pytest.raises(ValueError):
        MeasurementOutcomes.from_samples(np.array([[0, 2], [1, 0]]), backend=backend)
