"""Tests Tensorflow's backpropagation when using `tf.Variable` parameters."""
import numpy as np
import pytest
import qibo
from qibo import K, gates
from qibo.models import Circuit


def test_variable_backpropagation(backend):
    if backend == "custom":
        pytest.skip("Custom gates do not support automatic differentiation.")
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    if "numpy" in K.name:
        qibo.set_backend(original_backend)
        pytest.skip("Backpropagation is not supported by {}.".format(K.name))

    theta = K.optimization.Variable(0.1234, dtype=K.dtypes('DTYPE'))
    # TODO: Fix parametrized gates so that `Circuit` can be defined outside
    # of the gradient tape
    with K.optimization.GradientTape() as tape:
        c = Circuit(1)
        c.add(gates.X(0))
        c.add(gates.RZ(0, theta))
        loss = K.real(c()[-1])
    grad = tape.gradient(loss, theta)

    target_loss = np.cos(theta / 2.0)
    np.testing.assert_allclose(loss, target_loss)

    target_grad = - np.sin(theta / 2.0) / 2.0
    np.testing.assert_allclose(grad, target_grad)
    qibo.set_backend(original_backend)


def test_two_variables_backpropagation(backend):
    if backend == "custom":
        pytest.skip("Custom gates do not support automatic differentiation.")
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    if "numpy" in K.name:
        qibo.set_backend(original_backend)
        pytest.skip("Backpropagation is not supported by {}.".format(K.name))

    theta = K.optimization.Variable([0.1234, 0.4321], dtype=K.dtypes('DTYPE'))
    # TODO: Fix parametrized gates so that `Circuit` can be defined outside
    # of the gradient tape
    with K.optimization.GradientTape() as tape:
        c = Circuit(2)
        c.add(gates.RX(0, theta[0]))
        c.add(gates.RY(1, theta[1]))
        loss = K.real(c()[0])
    grad = tape.gradient(loss, theta)

    t = np.array([0.1234, 0.4321]) / 2.0
    target_loss = np.cos(t[0]) * np.cos(t[1])
    np.testing.assert_allclose(loss, target_loss)

    target_grad1 = - np.sin(t[0]) * np.cos(t[1])
    target_grad2 = - np.cos(t[0]) * np.sin(t[1])
    target_grad = np.array([target_grad1, target_grad2]) / 2.0
    np.testing.assert_allclose(grad, target_grad)
    qibo.set_backend(original_backend)
