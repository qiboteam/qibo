"""Test measurement objects defined in `qibo/core/measurements.py`."""
import pytest
import numpy as np
import qibo
from qibo import K
from qibo.core import measurements


def test_measurementresult_init():
    result = measurements.MeasurementResult((0, 1))
    assert result.qubits == (0, 1)
    assert result.nqubits == 2
    assert result.qubit_map == {0: 0, 1: 1}


def test_measurementresult_errors():
    """Try to sample shots and frequencies without probability distribution."""
    result = measurements.MeasurementResult((0, 1))
    with pytest.raises(RuntimeError):
        samples = result.samples()
    with pytest.raises(RuntimeError):
        samples = result.frequencies()
    result.binary = np.random.randint(0, 2, (100, 2))
    with pytest.raises(RuntimeError):
        result.set_frequencies({0: 100})


def test_measurementresult_add_shots(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    result = measurements.MeasurementResult((0, 1))
    with pytest.raises(ValueError):
        result.add_shot()
    probs = np.array([1, 0, 0, 0], dtype=np.float64)
    result.add_shot(probabilities=probs)
    assert result.nshots == 1
    np.testing.assert_allclose(result.decimal, [0])
    np.testing.assert_allclose(result.binary, [[0, 0]])
    probs = np.array([0, 0, 0, 1], dtype=np.float64)
    result.add_shot(probabilities=probs)
    assert result.nshots == 2
    np.testing.assert_allclose(result.decimal, [0, 3])
    np.testing.assert_allclose(result.binary, [[0, 0], [1, 1]])
    qibo.set_backend(original_backend)


def test_measurementresult_outcome(backend):
    import collections
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    result = measurements.MeasurementResult((0,))
    result.decimal = np.zeros(1, dtype=np.int64)
    assert result.outcome() == 0
    result.decimal = np.ones(1, dtype=np.int64)
    assert result.outcome() == 1
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize("dsamples,bsamples",
                         [([0, 3, 2, 3, 1],
                           [[0, 0], [1, 1], [1, 0], [1, 1], [0, 1]]),
                          ([0, 6, 5, 3, 1],
                           [[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1]])])
def test_measurementresult_conversions(backend, binary, dsamples, bsamples):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    qubits = tuple(range(len(bsamples[0])))
    result1 = measurements.MeasurementResult(qubits)
    result1.decimal = K.cast(dsamples, dtype='DTYPEINT')
    result2 = measurements.MeasurementResult(qubits)
    result2.binary = K.cast(bsamples, dtype='DTYPEINT')
    np.testing.assert_allclose(result1.samples(binary=True), bsamples)
    np.testing.assert_allclose(result2.samples(binary=True), bsamples)
    np.testing.assert_allclose(result1.samples(binary=False), dsamples)
    np.testing.assert_allclose(result2.samples(binary=False), dsamples)
    # test ``__getitem__``
    for i, target in enumerate(dsamples):
        np.testing.assert_allclose(result1[i], target)
    qibo.set_backend(original_backend)


def test_measurementresult_frequencies(backend):
    import collections
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    result = measurements.MeasurementResult((0, 1, 2))
    result.decimal = [0, 6, 5, 3, 5, 5, 6, 1, 1, 2, 4]
    dfreqs = {0: 1, 1: 2, 2: 1, 3: 1, 4: 1, 5: 3, 6: 2}
    bfreqs = {"000": 1, "001": 2, "010": 1, "011": 1, "100": 1,
              "101": 3, "110": 2}
    assert result.frequencies(binary=True) == bfreqs
    assert result.frequencies(binary=False) == dfreqs
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("i,p0,p1",
                         [(0, 0.2, None), (1, 0.2, 0.1),
                          (2, (0.1, 0.0, 0.2), None),
                          (3, {0: 0.2, 1: 0.1, 2: 0.0}, None)])
def test_measurementresult_apply_bitflips(backend, i, p0, p1):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    result = measurements.MeasurementResult((0, 1, 2))
    result.decimal = K.zeros(10, dtype='DTYPEINT')
    K.set_seed(123)
    noisy_result = result.apply_bitflips(p0, p1)
    if "numpy" in backend:
        targets = [
            [0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 0, 0]
        ]
    else:
        targets = [
            [4, 0, 0, 1, 0, 2, 2, 4, 4, 0],
            [4, 0, 0, 1, 0, 2, 2, 4, 4, 0],
            [4, 0, 0, 1, 0, 0, 0, 4, 4, 0],
            [4, 0, 0, 0, 0, 0, 0, 4, 4, 0]
        ]
    np.testing.assert_allclose(noisy_result.samples(binary=False), targets[i])
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("probs", [0.2, {0: 0.1, 1: 0.2, 2: 0.8, 3: 0.3}])
def test_measurementresult_apply_bitflips_random_samples(backend, probs):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    qubits = tuple(range(4))
    samples = np.random.randint(0, 2, (20, 4))
    result = measurements.MeasurementResult(qubits)
    result.binary = np.copy(samples)
    K.set_seed(123)
    noisy_result = result.apply_bitflips(probs)

    K.set_seed(123)
    if isinstance(probs, dict):
        probs = np.array([probs[q] for q in qubits])
    sprobs = np.array(K.random_uniform(samples.shape))
    target_samples = (samples + (sprobs < probs)) % 2
    np.testing.assert_allclose(noisy_result.samples(), target_samples)
    qibo.set_backend(original_backend)


def test_measurementresult_apply_bitflips_random_samples_asymmetric(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    qubits = tuple(range(4))
    samples = np.random.randint(0, 2, (20, 4))
    result = measurements.MeasurementResult(qubits)
    result.binary = np.copy(samples)
    p1_map = {0: 0.2, 1: 0.0, 2: 0.0, 3: 0.1}
    K.set_seed(123)
    noisy_result = result.apply_bitflips(p0=0.2, p1=p1_map)

    p0 = 0.2 * np.ones(4)
    p1 = np.array([0.2, 0.0, 0.0, 0.1])
    K.set_seed(123)
    sprobs = np.array(K.random_uniform(samples.shape))
    target_samples = np.copy(samples).ravel()
    ids = (np.where(target_samples == 0)[0], np.where(target_samples == 1)[0])
    target_samples[ids[0]] = samples.ravel()[ids[0]] + (sprobs < p0).ravel()[ids[0]]
    target_samples[ids[1]] = samples.ravel()[ids[1]] - (sprobs < p1).ravel()[ids[1]]
    target_samples = target_samples.reshape(samples.shape)
    np.testing.assert_allclose(noisy_result.samples(), target_samples)
    qibo.set_backend(original_backend)


def test_measurementresult_apply_bitflips_errors():
    """Check errors raised by `MeasurementResult.apply_bitflips`."""
    result = measurements.MeasurementResult((0, 1, 3))
    result.binary = np.random.randint(0, 2, (20, 3))
    # Passing wrong qubit ids in bitflip error map
    with pytest.raises(KeyError):
        noisy_result = result.apply_bitflips({0: 0.1, 2: 0.2})
    # Passing wrong bitflip error map type
    with pytest.raises(TypeError):
        noisy_result = result.apply_bitflips("test")
    # Passing negative bitflip probability
    with pytest.raises(ValueError):
        noisy_result = result.apply_bitflips(-0.4)


def test_measurementsymbol_counter():
    symbol1 = measurements.MeasurementSymbol(measurements.MeasurementResult((0, 1)))
    symbol2 = measurements.MeasurementSymbol(measurements.MeasurementResult((1, 3)))
    assert symbol1.result.qubits == (0, 1)
    assert symbol2.result.qubits == (1, 3)
    assert symbol1.name[0] == "m" # pylint: disable=E1101
    assert symbol2.name[0] == "m" # pylint: disable=E1101
    assert int(symbol1.name[1:]) + 3 == int(symbol2.name[1:]) # pylint: disable=E1101


def test_measurementsymbol_evaluate(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    result = measurements.MeasurementSymbol(measurements.MeasurementResult((0, 1)))
    expr = 2 * result
    with pytest.raises(NotImplementedError):
        value = result.evaluate(expr)
    result = measurements.MeasurementSymbol(measurements.MeasurementResult((0,)))
    result.result.set_probabilities(np.array([0., 1.]), nshots=1)
    expr = 2 * result
    value = result.evaluate(expr)
    assert value == 2
    qibo.set_backend(original_backend)


def test_measurementregistersresult_samples(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    samples = np.random.randint(0, 2, (20, 4))
    result = measurements.MeasurementResult((0, 1, 2, 3))
    result.binary = samples
    qubits = {"a": (0, 2), "b": (1, 3)}
    result = measurements.MeasurementRegistersResult(qubits, result)
    register_samples = result.samples(registers=True)
    assert register_samples.keys() == qubits.keys()
    np.testing.assert_allclose(register_samples["a"], samples[:, [0, 2]])
    np.testing.assert_allclose(register_samples["b"], samples[:, [1, 3]])
    qibo.set_backend(original_backend)


def test_measurementregistersresult_frequencies(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    probs = np.random.random(16)
    probs = probs / np.sum(probs)
    result = measurements.MeasurementResult((0, 1, 2, 3), probs, nshots=1000000)
    frequencies = result.frequencies()
    qubits = {"a": (0, 1), "b": (2, 3)}
    result = measurements.MeasurementRegistersResult(qubits, result)
    register_frequencies = result.frequencies(registers=True)
    assert register_frequencies.keys() == qubits.keys()
    rkeys = ["00", "01", "10", "11"]
    target_frequencies_a = {k: sum(frequencies[f"{k}{l}"] for l in rkeys)
                            for k in rkeys}
    target_frequencies_b = {k: sum(frequencies[f"{l}{k}"] for l in rkeys)
                            for k in rkeys}
    assert register_frequencies["a"] == target_frequencies_a
    assert register_frequencies["b"] == target_frequencies_b
    qibo.set_backend(original_backend)
