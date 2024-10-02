import sys

import numpy as np
import pytest

from qibo import gates, models
from qibo.quantum_info import infidelity


def test_torch_gradients(backend):
    if backend.name != "pytorch":
        pytest.skip("Test only valid for PyTorch backend.")
    backend.np.manual_seed(42)
    nepochs = 1001
    optimizer = backend.np.optim.Adam
    target_state = backend.np.rand(2, dtype=backend.np.complex128)
    target_state = target_state / backend.np.norm(target_state)
    params = backend.np.rand(2, dtype=backend.np.float64, requires_grad=True)
    c = models.Circuit(1)
    c.add(gates.RX(0, params[0]))
    c.add(gates.RY(0, params[1]))

    initial_params = params.clone()
    initial_loss = infidelity(
        target_state, backend.execute_circuit(c).state(), backend=backend
    )

    optimizer = optimizer([params])
    for _ in range(nepochs):
        optimizer.zero_grad()
        c.set_parameters(params)
        final_state = backend.execute_circuit(c).state()
        loss = infidelity(target_state, final_state, backend=backend)
        loss.backward()
        optimizer.step()

    assert initial_loss > loss
    assert initial_params[0] != params[0]


@pytest.mark.skipif(
    sys.platform != "linux", reason="Tensorflow available only when testing on linux."
)
def test_torch_tensorflow_gradients(backend):
    if backend.name != "pytorch":
        pytest.skip("Test only valid for PyTorch backend.")

    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    from qibo.backends.tensorflow import (  # pylint: disable=import-outside-toplevel
        TensorflowBackend,
    )

    target_state = backend.np.tensor([0.0, 1.0], dtype=backend.np.complex128)
    param = backend.np.tensor([0.1], dtype=backend.np.float64, requires_grad=True)
    c = models.Circuit(1)
    c.add(gates.RX(0, param[0]))

    optimizer = backend.np.optim.SGD
    optimizer = optimizer([param], lr=1)
    c.set_parameters(param)
    final_state = backend.execute_circuit(c).state()
    loss = infidelity(target_state, final_state, backend=backend)
    loss.backward()
    torch_param_grad = param.grad.clone().item()
    optimizer.step()
    torch_param = param.clone().item()

    tf_backend = TensorflowBackend()

    target_state = tf.constant([0.0, 1.0], dtype=tf.complex128)
    param = tf.Variable([0.1], dtype=tf.float64)
    c = models.Circuit(1)
    c.add(gates.RX(0, param[0]))

    optimizer = tf.optimizers.SGD(learning_rate=1.0)

    with tf.GradientTape() as tape:
        c.set_parameters(param)
        final_state = tf_backend.execute_circuit(c).state()
        loss = infidelity(target_state, final_state, backend=tf_backend)

    grads = tape.gradient(loss, [param])
    tf_param_grad = grads[0].numpy()[0]
    optimizer.apply_gradients(zip(grads, [param]))
    tf_param = param.numpy()[0]

    assert np.allclose(torch_param_grad, tf_param_grad, atol=1e-7, rtol=1e-7)
    assert np.allclose(torch_param, tf_param, atol=1e-7, rtol=1e-7)
