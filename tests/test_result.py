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


def test_measurementoutcomes_from_frequencies_nqubits_mismatch(backend):
    """Test from_frequencies raises error when key length mismatches nqubits."""
    with pytest.raises(ValueError):
        MeasurementOutcomes.from_frequencies(
            {"00": 10, "11": 20}, nqubits=3, backend=backend
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


def test_quantumstate_symbolic_and_str(backend):
    """Cover QuantumState.symbolic()."""
    from qibo.result import QuantumState

    # |0> state for 1 qubit
    state = backend.cast(np.array([1.0, 0.0], dtype=complex))
    qs = QuantumState(state, backend=backend)

    symbolic_repr = qs.symbolic()
    assert isinstance(symbolic_repr, str)
    assert len(symbolic_repr) > 0

    # __str__ delegates to symbolic()
    assert str(qs) == symbolic_repr


def test_frequencies_from_probabilities_only(backend):
    """Cover probabilities-based frequency generation"""
    m0 = gates.M(0, register_name="a")
    m1 = gates.M(1, register_name="b")
    # probabilities for 2-qubit system: |00>=0.5, |01>=0.0, |10>=0.0, |11>=0.5
    probs = np.array([0.5, 0.0, 0.0, 0.5], dtype=np.float64)
    result = MeasurementOutcomes(
        [m0, m1], backend=backend, probabilities=probs, nshots=1000
    )

    freq = result.frequencies(binary=True)
    assert isinstance(freq, dict)
    total = sum(freq.values())
    assert total == 1000
    # With these probabilities, only "00" and "11" should appear
    for key in freq:
        assert key in ("00", "11")


def test_frequencies_registers_true(backend):
    """Cover frequencies(registers=True)"""
    m0 = gates.M(0, register_name="a")
    m1 = gates.M(1, register_name="b")
    probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    result = MeasurementOutcomes(
        [m0, m1], backend=backend, probabilities=probs, nshots=100
    )

    reg_freq = result.frequencies(registers=True)
    assert isinstance(reg_freq, dict)
    # Should have register names as keys
    assert set(reg_freq.keys()) == {"a", "b"}
    for name, counter in reg_freq.items():
        total = sum(counter.values())
        assert total == 100


def test_samples_from_existing_frequencies(backend):
    """Cover samples()."""
    m = gates.M(0, 1)
    result = MeasurementOutcomes([m], backend=backend, nshots=100)
    # Manually inject frequencies (integer-keyed: state -> count)
    result._frequencies = collections.Counter({0: 60, 3: 40})

    samples = result.samples(binary=True)
    assert samples.shape == (100, 2)
    # All values should be binary
    assert np.all((np.array(samples) == 0) | (np.array(samples) == 1))

    # Verify frequencies match what we injected
    freq = result.frequencies(binary=False)
    assert freq[0] == 60
    assert freq[3] == 40


def test_samples_with_bitflip_noise(backend):
    """Cover bitflip noise in samples()"""
    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.M(0, 1, p0=0.5, p1=0.5))
    result = backend.execute_circuit(c, nshots=1000)

    # The result has probabilities but not samples yet;
    # calling samples() should trigger the bitflip path
    s = result.samples(binary=True)
    assert s.shape == (1000, 2)


def test_samples_registers_true(backend):
    """Cover samples(registers=True)"""
    samples = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=int)
    m0 = gates.M(0, register_name="a")
    m1 = gates.M(1, register_name="b")
    result = MeasurementOutcomes([m0, m1], backend=backend, samples=samples, nshots=4)

    reg_samples = result.samples(registers=True)
    assert isinstance(reg_samples, dict)
    assert set(reg_samples.keys()) == {"a", "b"}
    for name, s in reg_samples.items():
        assert s.shape[0] == 4


def test_measurement_gate_add_merge(backend):
    """Cover measurement_gate.add(gate).

    Create MeasurementOutcomes with multiple M gates so that the
    measurement_gate property merges them via .add().
    """
    m0 = gates.M(0, register_name="a")
    m1 = gates.M(1, register_name="b")
    samples = np.array([[0, 1], [1, 0]], dtype=int)
    result = MeasurementOutcomes([m0, m1], backend=backend, samples=samples, nshots=2)

    mg = result.measurement_gate
    # The merged gate should contain both qubits
    assert set(mg.qubits) == {0, 1}


def test_apply_bitflips_method(backend):
    """Cover apply_bitflips() method."""
    samples = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=int)
    result = MeasurementOutcomes.from_samples(samples, backend=backend)

    flipped = result.apply_bitflips(0.0)
    # With p0=0.0, no bits should flip
    backend.assert_allclose(flipped, samples)


def test_expectation_from_samples_method(backend):
    """Cover expectation_from_samples()."""
    from qibo.hamiltonians import Hamiltonian

    # Z operator on 1 qubit: diag(1, -1)
    matrix = np.array([[1.0, 0.0], [0.0, -1.0]])
    obs = Hamiltonian(1, matrix, backend=backend)

    # All samples are |0>, so <Z> should be 1.0
    samples = np.array([[0]] * 100, dtype=int)
    result = MeasurementOutcomes.from_samples(samples, backend=backend)

    exp_val = result.expectation_from_samples(obs)
    backend.assert_allclose(exp_val, 1.0, atol=1e-10)


def test_circuitresult_probabilities_bitflip(backend):
    """Cover CircuitResult.probabilities bitflip path."""
    c = Circuit(1)
    c.add(gates.X(0))
    c.add(gates.M(0, p0=0.2, p1=0.2))
    result = backend.execute_circuit(c, nshots=10000)

    probs = result.probabilities()
    # Probabilities should sum to 1
    backend.assert_allclose(backend.sum(probs), 1.0)
    # With X gate, qubit is |1>, but bitflip noise should cause some |0>
    # So prob[0] should be > 0 and prob[1] should be > 0
    assert float(probs[0]) > 0
    assert float(probs[1]) > 0
