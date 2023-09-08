import numpy as np
import pytest

from qibo.parameter import Parameter


def test_parameter():
    # single feature
    param = Parameter(
        lambda x, th1, th2, th3: x**2 * th1 + th2 * th3**2,
        [1.5, 2.0, 3.0],
        feature=[7.0],
    )

    indices = param.get_indices(10)
    assert indices == [10, 11, 12]

    fixed = param.get_fixed_part(1)
    assert fixed == 73.5

    factor = param.get_partial_derivative(3)
    assert factor == 12.0

    param.update_parameters(trainable=[15.0, 10.0, 7.0], feature=[5.0])
    gate_value = param()
    assert gate_value == 865

    # single feature, no list
    param2 = Parameter(
        lambda x, th1, th2, th3: x**2 * th1 + th2 * th3**2,
        [1.5, 2.0, 3.0],
        feature=[7.0],
    )

    gate_value2 = param2()
    assert gate_value2 == 91.5

    # multiple features
    param = Parameter(
        lambda x1, x2, th1, th2, th3: x1**2 * th1 + x2 * th2 * th3,
        [1.5, 2.0, 3.0],
        feature=[7.0, 4.0],
    )

    fixed = param.get_fixed_part(1)
    assert fixed == 73.5

    factor = param.get_partial_derivative(4)
    assert factor == 8.0

    param.update_parameters(trainable=np.array([15.0, 10.0, 7.0]), feature=[5.0, 3.0])
    gate_value = param()
    assert gate_value == 585


def test_parameter_errors():
    with pytest.raises(ValueError) as e_info:
        param = Parameter(
            lambda x, t1, th2, th3: x**2 * t1 + th2 * th3**2,
            [1.5, 2.0, 3.0],
            feature=[7.0],
        )

    with pytest.raises(ValueError) as e_info:
        param = Parameter(
            lambda x, th1, th2, th3: x**2 * th1 + th2 * th3**2,
            [1.5, 2.0],
            feature=[3.0, 7.0],
        )

    with pytest.raises(ValueError) as e_info:
        param = Parameter(
            lambda j, th1, th2, th3: j**2 * th1 + th2 * th3**2,
            [1.5, 2.0, 3.0],
            feature=[7.0],
        )

    with pytest.raises(ValueError) as e_info:
        param = Parameter(
            lambda x, th1, th2, th3: x**2 * th1 + th2 * th3**2,
            [1.5, 2.0, 3.0],
            feature=[7.0],
        )
        param.update_parameters((1, 1, 1), [1])

    with pytest.raises(ValueError) as e_info:
        param = Parameter(
            lambda x, th1, th2, th3: x**2 * th1 + th2 * th3**2,
            [1.5, 2.0, 3.0],
            feature=[7.0],
        )
        param.update_parameters([1, 1, 1], (1))

    with pytest.raises(ValueError) as e_info:
        param = Parameter(
            lambda x, th1, th2, th3: x**2 * th1 + th2 * th3**2,
            [1.5, 2.0, 3.0],
            feature=[7.0],
        )
        param.update_parameters([1, 1], [1])

    with pytest.raises(ValueError) as e_info:
        param = Parameter(
            lambda x, th1, th2, th3: x**2 * th1 + th2 * th3**2,
            [1.5, 2.0, 3.0],
            feature=[7.0],
        )
        param.update_parameters([1, 1, 1], [1, 1])

    with pytest.raises(ValueError) as e_info:
        param = Parameter(
            lambda x, th1, th2, th3: x**2 * th1 + th2 * th3**2,
            [1.5, 2.0],
            feature=[7.0],
        )


if __name__ == "__main__":
    test_parameter()
