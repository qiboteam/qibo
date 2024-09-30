import numpy as np
import pytest
import tensorflow as tf
import torch

import qibo
from qibo import gates, models


def test_torch_gradients():
    qibo.set_backend("pytorch")
    torch.manual_seed(42)
    nepochs = 1001
    optimizer = torch.optim.Adam
    target_state = torch.rand(2, dtype=torch.complex128)
    target_state = target_state / torch.norm(target_state)
    params = torch.rand(2, dtype=torch.float64, requires_grad=True)
    c = models.Circuit(1)
    c.add(gates.RX(0, params[0]))
    c.add(gates.RY(0, params[1]))

    initial_params = params.clone()
    initial_loss = 1 - torch.abs(torch.sum(torch.conj(target_state) * c().state()))

    optimizer = optimizer([params])
    for _ in range(nepochs):
        optimizer.zero_grad()
        c.set_parameters(params)
        final_state = c().state()
        fidelity = torch.abs(torch.sum(torch.conj(target_state) * final_state))
        loss = 1 - fidelity
        loss.backward()
        optimizer.step()

    assert initial_loss > loss
    assert initial_params[0] != params[0]


def test_torch_tensorflow_gradients():
    qibo.set_backend("pytorch")
    target_state = torch.tensor([0.0, 1.0], dtype=torch.complex128)
    param = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
    c = models.Circuit(1)
    c.add(gates.RX(0, param[0]))

    optimizer = torch.optim.SGD
    optimizer = optimizer([param], lr=1)
    c.set_parameters(param)
    final_state = c().state()
    fidelity = torch.abs(torch.sum(torch.conj(target_state) * final_state))
    loss = 1 - fidelity
    loss.backward()
    torch_param_grad = param.grad.clone().item()
    optimizer.step()
    torch_param = param.clone().item()

    qibo.set_backend("tensorflow")

    target_state = tf.constant([0.0, 1.0], dtype=tf.complex128)
    param = tf.Variable([0.1], dtype=tf.float64)
    c = models.Circuit(1)
    c.add(gates.RX(0, param[0]))

    optimizer = tf.optimizers.SGD(learning_rate=1.0)

    with tf.GradientTape() as tape:
        c.set_parameters(param)
        final_state = c().state()
        fidelity = tf.abs(tf.reduce_sum(tf.math.conj(target_state) * final_state))
        loss = 1 - fidelity

    grads = tape.gradient(loss, [param])
    tf_param_grad = grads[0].numpy()[0]
    optimizer.apply_gradients(zip(grads, [param]))
    tf_param = param.numpy()[0]

    assert np.allclose(torch_param_grad, tf_param_grad, atol=1e-7, rtol=1e-7)
    assert np.allclose(torch_param, tf_param, atol=1e-7, rtol=1e-7)
