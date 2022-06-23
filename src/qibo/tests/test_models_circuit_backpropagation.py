"""Tests Tensorflow's backpropagation when using `tf.Variable` parameters."""
import pytest
import numpy as np
from qibo import gates
from qibo.models import Circuit


def construct_tensorflow_backend():
    try:
        from qibo.backends import construct_backend
        backend = construct_backend("tensorflow")
    except ModuleNotFoundError: # pragma: no cover
        pytest.skip("Skipping backpropagation test because tensorflow is not available.")
    return backend


def test_variable_backpropagation():
    backend = construct_tensorflow_backend()
    import tensorflow as tf
    theta = tf.Variable(0.1234, dtype="float64")
    # TODO: Fix parametrized gates so that `Circuit` can be defined outside
    # of the gradient tape
    with tf.GradientTape() as tape:
        c = Circuit(1)
        c.add(gates.X(0))
        c.add(gates.RZ(0, theta))
        result = backend.execute_circuit(c)
        loss = tf.math.real(result.state()[-1])
    grad = tape.gradient(loss, theta)

    target_loss = np.cos(theta / 2.0)
    backend.assert_allclose(loss, target_loss)

    target_grad = - np.sin(theta / 2.0) / 2.0
    backend.assert_allclose(grad, target_grad)


def test_two_variables_backpropagation():
    backend = construct_tensorflow_backend()
    import tensorflow as tf
    theta = tf.Variable([0.1234, 0.4321], dtype="float64")
    # TODO: Fix parametrized gates so that `Circuit` can be defined outside
    # of the gradient tape
    with tf.GradientTape() as tape:
        c = Circuit(2)
        c.add(gates.RX(0, theta[0]))
        c.add(gates.RY(1, theta[1]))
        result = backend.execute_circuit(c)
        loss = tf.math.real(result.state()[0])
    grad = tape.gradient(loss, theta)

    t = np.array([0.1234, 0.4321]) / 2.0
    target_loss = np.cos(t[0]) * np.cos(t[1])
    backend.assert_allclose(loss, target_loss)

    target_grad1 = - np.sin(t[0]) * np.cos(t[1])
    target_grad2 = - np.cos(t[0]) * np.sin(t[1])
    target_grad = np.array([target_grad1, target_grad2]) / 2.0
    backend.assert_allclose(grad, target_grad)
