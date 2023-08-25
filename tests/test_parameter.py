import numpy as np

from qibo import gates
from qibo.models.circuit import VariationalCircuit
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


def test_variational_circuit():
    c = VariationalCircuit(1)
    c.add(gates.RX(q=0, theta=Parameter(lambda th1, th2: th1**2 + th2, [0.1, 0.1])))
    c.add(gates.RY(q=0, theta=Parameter(lambda th1, th2: th1**2 + th2, [0.4, 0.1])))
    c.add(gates.RZ(q=0, theta=Parameter(lambda th1, th2: th1**2 + th2, [0.3, 0.1])))
    c.add(gates.M(0))

    # _get_initparams
    true = np.array([0.1, 0.1, 0.4, 0.1, 0.3, 0.1])
    Params = c._get_initparams()
    check = []
    for Param in Params:
        check.extend(Param._trainable)

    assert np.allclose(check, true)

    # _get_train_params
    train_params = c._get_train_params()

    assert np.allclose(check, train_params)

    # set_variational_parameters
    true = [(120.0,), (10010.0,), (420.0,)]
    c.set_variational_parameters([10.0, 20, 100, 10, 20, 20])
    circuit_params = c.get_parameters()

    assert circuit_params == true
