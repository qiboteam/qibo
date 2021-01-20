import numpy as np
import pytest
import qibo
from qibo import gates
from qibo.models import Circuit


try:
    import tensorflow as tf
    BACKENDS = ["custom", "defaulteinsum", "matmuleinsum",
                "numpy_defaulteinsum", "numpy_matmuleinsum"]
except ModuleNotFoundError: # pragma: no cover
    BACKENDS = ["defaulteinsum", "matmuleinsum"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("trainable", [True, False])
def test_set_parameters_with_list(backend, trainable):
    """Check updating parameters of circuit with list."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    params = [0.123, 0.456, (0.789, 0.321)]
    c = Circuit(3)
    if trainable:
        c.add(gates.RX(0, theta=0, trainable=trainable))
    else:
        c.add(gates.RX(0, theta=params[0], trainable=trainable))
    c.add(gates.RY(1, theta=0))
    c.add(gates.CZ(1, 2))
    c.add(gates.fSim(0, 2, theta=0, phi=0))
    c.add(gates.H(2))
    # execute once
    final_state = c()

    target_c = Circuit(3)
    target_c.add(gates.RX(0, theta=params[0]))
    target_c.add(gates.RY(1, theta=params[1]))
    target_c.add(gates.CZ(1, 2))
    target_c.add(gates.fSim(0, 2, theta=params[2][0], phi=params[2][1]))
    target_c.add(gates.H(2))

    # Attempt using a flat np.ndarray/list
    for new_params in (np.random.random(4), list(np.random.random(4))):
        if trainable:
            c.set_parameters(new_params)
        else:
            new_params[0] = params[0]
            c.set_parameters(new_params[1:])
    target_params = [new_params[0], new_params[1], (new_params[2], new_params[3])]
    target_c.set_parameters(target_params)
    np.testing.assert_allclose(c(), target_c())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("trainable", [True, False])
def test_circuit_set_parameters_ungates(backend, trainable):
    """Check updating parameters of circuit with list."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    params = [0.1, 0.2, 0.3, (0.4, 0.5), (0.6, 0.7, 0.8)]
    if trainable:
        trainable_params = list(params)
    else:
        trainable_params = [0.1, 0.3, (0.4, 0.5)]

    c = Circuit(3)
    c.add(gates.RX(0, theta=0))
    if trainable:
        c.add(gates.CRY(0, 1, theta=0, trainable=trainable))
    else:
        c.add(gates.CRY(0, 1, theta=params[1], trainable=trainable))
    c.add(gates.CZ(1, 2))
    c.add(gates.U1(2, theta=0))
    c.add(gates.CU2(0, 2, phi=0, lam=0))
    if trainable:
        c.add(gates.U3(1, theta=0, phi=0, lam=0, trainable=trainable))
    else:
        c.add(gates.U3(1, *params[4], trainable=trainable))
    # execute once
    final_state = c()

    target_c = Circuit(3)
    target_c.add(gates.RX(0, theta=params[0]))
    target_c.add(gates.CRY(0, 1, theta=params[1]))
    target_c.add(gates.CZ(1, 2))
    target_c.add(gates.U1(2, theta=params[2]))
    target_c.add(gates.CU2(0, 2, *params[3]))
    target_c.add(gates.U3(1, *params[4]))
    c.set_parameters(trainable_params)
    np.testing.assert_allclose(c(), target_c())

    # Attempt using a flat list
    npparams = np.random.random(8)
    if trainable:
        trainable_params = np.copy(npparams)
    else:
        npparams[1] = params[1]
        npparams[5:] = params[4]
        trainable_params = np.delete(npparams, [1, 5, 6, 7])
    target_c = Circuit(3)
    target_c.add(gates.RX(0, theta=npparams[0]))
    target_c.add(gates.CRY(0, 1, theta=npparams[1]))
    target_c.add(gates.CZ(1, 2))
    target_c.add(gates.U1(2, theta=npparams[2]))
    target_c.add(gates.CU2(0, 2, *npparams[3:5]))
    target_c.add(gates.U3(1, *npparams[5:]))
    c.set_parameters(trainable_params)
    np.testing.assert_allclose(c(), target_c())
    qibo.set_backend(original_backend)
