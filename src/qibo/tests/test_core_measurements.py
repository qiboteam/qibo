import pytest
import numpy as np
import qibo
from qibo import K
from qibo.core import measurements


def test_gateresult_init():
    samples = K.zeros(10, dtype='DTYPEINT')
    result = measurements.GateResult((0, 1), decimal_samples=samples)
    assert result.qubits == (0, 1)
    assert result.nqubits == 2
    assert result.qubit_map == {0: 0, 1: 1}


def test_gateresult_init_errors():
    dsamples = K.zeros(10, dtype='DTYPEINT')
    bsamples = K.zeros((10, 2), dtype='DTYPEINT')
    with pytest.raises(ValueError):
        result = measurements.GateResult((0, 1), decimal_samples=dsamples,
                                         binary_samples=bsamples)
    with pytest.raises(ValueError):
        result = measurements.GateResult((0, 1))
    bsamples = K.zeros((10, 5), dtype='DTYPEINT')
    with pytest.raises(ValueError):
        result = measurements.GateResult((0, 1), binary_samples=bsamples)



@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize("dsamples,bsamples",
                         [([0, 3, 2, 3, 1],
                           [[0, 0], [1, 1], [1, 0], [1, 1], [0, 1]]),
                          ([0, 6, 5, 3, 1],
                           [[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1]])])
def test_gateresult_binary_decimal_conversions(binary, dsamples, bsamples):
    decsamples = K.cast(dsamples, dtype='DTYPEINT')
    binsamples = K.cast(bsamples, dtype='DTYPEINT')
    qubits = tuple(range(int(binsamples.shape[-1])))
    result1 = measurements.GateResult(qubits, decimal_samples=decsamples)
    result2 = measurements.GateResult(qubits, binary_samples=binsamples)
    np.testing.assert_allclose(result1.samples(binary=True), binsamples)
    np.testing.assert_allclose(result2.samples(binary=True), binsamples)
    np.testing.assert_allclose(result1.samples(binary=False), decsamples)
    np.testing.assert_allclose(result2.samples(binary=False), decsamples)
    # test ``__getitem__``
    for i, target in enumerate(dsamples):
        np.testing.assert_allclose(result1[i], target)


def test_gateresult_frequencies():
    import collections
    samples = [0, 6, 5, 3, 5, 5, 6, 1, 1, 2, 4]
    result = measurements.GateResult((0, 1, 2), decimal_samples=samples)
    dfreqs = {0: 1, 1: 2, 2: 1, 3: 1, 4: 1, 5: 3, 6: 2}
    bfreqs = {"000": 1, "001": 2, "010": 1, "011": 1, "100": 1,
              "101": 3, "110": 2}
    assert result.frequencies(binary=True) == bfreqs
    assert result.frequencies(binary=False) == dfreqs


@pytest.mark.parametrize("i,p0,p1",
                         [(0, 0.2, None), (1, 0.2, 0.1),
                          (2, (0.1, 0.0, 0.2), None),
                          (3, {0: 0.2, 1: 0.1, 2: 0.0}, None)])
def test_gateresult_apply_bitflips(i, p0, p1):
    samples = K.zeros(10, dtype='DTYPEINT')
    result = measurements.GateResult((0, 1, 2), decimal_samples=samples)
    K.set_seed(123)
    noisy_result = result.apply_bitflips(p0, p1)
    targets = [
        [4, 0, 0, 1, 0, 2, 2, 4, 4, 0],
        [4, 0, 0, 1, 0, 2, 2, 4, 4, 0],
        [4, 0, 0, 1, 0, 0, 0, 4, 4, 0],
        [4, 0, 0, 0, 0, 0, 0, 4, 4, 0]
    ]
    np.testing.assert_allclose(noisy_result.samples(binary=False), targets[i])


# TODO: Test ``CircuitResult``
