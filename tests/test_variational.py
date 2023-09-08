import numpy as np
import pytest

from qibo import gates
from qibo.models.variational import VariationalCircuit
from qibo.parameter import Parameter


def test_variational_circuit():
    c = VariationalCircuit(1)
    c.add(gates.RX(q=0, theta=Parameter(lambda th1, th2: th1**2 + th2, [0.1, 0.1])))
    c.add(gates.RY(q=0, theta=Parameter(lambda th1, th2: th1**2 + th2, [0.4, 0.1])))
    c.add(gates.RZ(q=0, theta=Parameter(lambda th1, th2: th1**2 + th2, [0.3, 0.1])))
    c.add(gates.M(0))

    c()

    # _get_initparams
    true = np.array([0.1, 0.1, 0.4, 0.1, 0.3, 0.1])
    Params = c.gate_parameters
    check = []
    for Param in Params:
        check.extend(Param._trainable)

    assert np.allclose(check, true)

    # _get_train_params
    train_params = c.trainable_parameters

    assert np.allclose(check, train_params)

    # set_variational_parameters
    true = [(120.0,), (10010.0,), (420.0,)]
    c.set_variational_parameters([10.0, 20, 100, 10, 20, 20])
    circuit_params = c.get_parameters()

    assert circuit_params == true

    c.set_variational_parameters([(10.0,), (20,), (100,), (10,), (20,), (20,)])
    circuit_params = c.get_parameters()

    assert circuit_params == true

    with pytest.raises(ValueError) as e_info:
        c.set_variational_parameters((10.0, 20, 100, 10, 20, 20))
