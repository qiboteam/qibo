import numpy as np
import pytest

from qibo.parameter import Parameter


def test_parameter():
    # single feature
    param = Parameter(
        lambda x, th1, th2, th3: x**2 * th1 + th2 * th3**2,
        features=[7.0],
        trainable=[1.5, 2.0, 3.0],
    )

    indices = param.trainable_parameter_indices(10)
    assert indices == [10, 11, 12]

    fixed = param.unaffected_by(1)
    assert fixed == 73.5

    factor = param.partial_derivative(3)
    assert factor == 12.0

    param.trainable = [15.0, 10.0, 7.0]
    param.features = [5.0]
    gate_value = param()
    assert gate_value == 865

    # single feature, no list
    param2 = Parameter(
        lambda x, th1, th2, th3: x**2 * th1 + th2 * th3**2,
        features=[7.0],
        trainable=[1.5, 2.0, 3.0],
    )

    gate_value2 = param2()
    assert gate_value2 == 91.5

    # multiple features
    param = Parameter(
        lambda x1, x2, th1, th2, th3: x1**2 * th1 + x2 * th2 * th3,
        features=[7.0, 4.0],
        trainable=[1.5, 2.0, 3.0],
    )

    fixed = param.unaffected_by(1)
    assert fixed == 73.5
    assert param.nparams == 3
    assert param.nfeat == 2

    factor = param.partial_derivative(4)
    assert factor == 8.0

    param.trainable = np.array([15.0, 10.0, 7.0])
    param.features = [5.0, 3.0]
    gate_value = param()
    assert gate_value == 585


def test_parameter_errors():
    param = Parameter(
        lambda x, th1, th2, th3: x**2 * th1 + th2 * th3**2,
        features=[7.0],
        trainable=[1.5, 2.0, 3.0],
    )

    param.trainable = [1, 1, 1]
    param.features = 1

    try:
        param()
        assert False
    except Exception as e:
        assert True

    param.trainable = [1, 1]
    param.features = [1]
    try:
        param()
        assert False
    except Exception as e:
        assert True

    param.trainable = [1, 1, 1]
    param.features = [1, 1]
    try:
        param()
        assert False
    except Exception as e:
        assert True

    # test type error due to wrong initialization
    with pytest.raises(TypeError):
        param = Parameter(func=lambda x, y: x + y**2)
