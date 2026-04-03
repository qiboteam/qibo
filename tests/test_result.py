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


def test_measurementoutcomes_from_samples_duplicate_qubits(backend):
    """Test from_samples raises error on duplicate qubit indices."""
    with pytest.raises(ValueError):
        MeasurementOutcomes.from_samples(
            np.array([[0, 1], [1, 0]]), qubits=[0, 0], backend=backend
        )


def test_measurementoutcomes_from_frequencies_mixed_key_types(backend):
    """Test from_frequencies raises error on mixed string/int keys."""
    with pytest.raises(TypeError):
        MeasurementOutcomes.from_frequencies({"00": 10, 1: 20}, backend=backend)


def test_measurementoutcomes_from_frequencies_key_length_qubits_mismatch(backend):
    """Test from_frequencies raises error when binary-string key length does not
    match the number of qubits provided."""
    # Keys have length 2, but 3 qubits are provided
    with pytest.raises(ValueError, match="Binary-string key length"):
        MeasurementOutcomes.from_frequencies(
            {"00": 10, "11": 20}, qubits=[0, 1, 2], backend=backend
        )
    # Keys have length 3, but only 2 qubits are provided
    with pytest.raises(ValueError, match="Binary-string key length"):
        MeasurementOutcomes.from_frequencies(
            {"000": 5, "111": 15}, qubits=[0, 1], backend=backend
        )


def test_measurementoutcomes_from_frequencies_nqubits_larger(backend):
    """Test from_frequencies with nqubits larger than key length (unmeasured qubits)."""
    result = MeasurementOutcomes.from_frequencies(
        {"00": 10, "11": 20}, nqubits=3, backend=backend
    )
    # 2 measured qubits, 3 total circuit qubits
    assert len(result.measurement_gate.qubits) == 2
    probs = result.probabilities()
    # probabilities should span 2^3 = 8 states
    assert len(probs) == 8
    backend.assert_allclose(backend.sum(probs), 1.0)


def test_measurementoutcomes_from_frequencies_nqubits_too_small(backend):
    """Test from_frequencies raises error when nqubits < number of measured qubits."""
    with pytest.raises(ValueError):
        MeasurementOutcomes.from_frequencies(
            {"000": 10, "111": 20}, nqubits=2, backend=backend
        )


def test_measurementoutcomes_from_frequencies_non_integer_count(backend):
    """Test from_frequencies raises error on non-integer counts."""
    with pytest.raises(ValueError):
        MeasurementOutcomes.from_frequencies({"00": 1.5}, backend=backend)


def test_measurementoutcomes_from_frequencies_all_zero_counts(backend):
    """Test from_frequencies raises error when all counts are zero."""
    with pytest.raises(ValueError):
        MeasurementOutcomes.from_frequencies({"00": 0, "11": 0}, backend=backend)


def test_measurementoutcomes_from_frequencies_zero_count_entry(backend):
    """Test from_frequencies correctly skips entries with zero count."""
    freq_input = {"00": 10, "01": 0, "11": 20}
    result = MeasurementOutcomes.from_frequencies(freq_input, backend=backend)

    assert result.nshots == 30
    freq_out = result.frequencies(binary=True)
    assert freq_out["00"] == 10
    assert freq_out.get("01", 0) == 0
    assert freq_out["11"] == 20


def test_measurementoutcomes_from_frequencies_matches_circuit_probabilities(backend):
    """Test that from_frequencies with nqubits produces probabilities matching
    CircuitResult.probabilities() when the circuit has unmeasured qubits."""
    from qibo import Circuit
    from qibo.gates import M, X

    circ = Circuit(3)
    circ.add(X(0))
    circ.add(M(0, 1))

    res = backend.execute_circuit(circ)

    m = MeasurementOutcomes.from_frequencies(
        res.frequencies(), nqubits=3, backend=backend
    )

    # Both should return arrays of length 2^3 = 8 and match exactly
    res_probs = res.probabilities()
    m_probs = m.probabilities()
    assert len(res_probs) == len(m_probs)
    backend.assert_allclose(m_probs, res_probs, atol=1e-1)

    # Marginalising over the unmeasured qubit (qubit 2) should also match
    res_measured = res.probabilities(qubits=[0, 1])
    m_measured = m.probabilities(qubits=[0, 1])
    backend.assert_allclose(m_measured, res_measured, atol=1e-1)


def test_measurementoutcomes_from_frequencies_marginal_over_unmeasured(backend):
    """Test that marginalising over unmeasured qubits recovers the measured
    qubit probabilities."""
    freq_input = {"10": 1000}
    result = MeasurementOutcomes.from_frequencies(
        freq_input, nqubits=3, backend=backend
    )

    # Full probabilities should have length 8
    full_probs = result.probabilities()
    assert len(full_probs) == 8
    backend.assert_allclose(backend.sum(full_probs), 1.0)

    # Probabilities for just the measured qubits should recover original
    measured_probs = result.probabilities(qubits=[0, 1])
    assert len(measured_probs) == 4
    backend.assert_allclose(measured_probs[2], 1.0, atol=1e-10)  # state "10" = index 2
