"""Tests Tensorflow's backpropagation when using `tf.Variable` parameters."""

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.backends import MissingBackend

pytest.skip("To be moved to `qiboml`.", allow_module_level=True)


def construct_tensorflow_backend():
    try:
        from qibo.backends import construct_backend

        backend = construct_backend(backend="qiboml", platform="tensorflow")
    except MissingBackend:
        pytest.skip(
            "Skipping backpropagation test because tensorflow is not available."
        )
    return backend


def test_variable_backpropagation():
    backend = construct_tensorflow_backend()

    theta = backend.engine.Variable(0.1234, dtype=backend.float64)
    # TODO: Fix parametrized gates so that `Circuit` can be defined outside
    # of the gradient tape
    with backend.engine.GradientTape() as tape:
        circuit = Circuit(1)
        circuit.add(gates.X(0))
        circuit.add(gates.RZ(0, theta))
        result = backend.execute_circuit(circuit)
        loss = backend.real(result.state()[-1])
    grad = tape.gradient(loss, theta)
    grad = backend.real(grad)

    target_loss = backend.cos(theta / 2.0)
    backend.assert_allclose(loss, target_loss)

    target_grad = -backend.sin(theta / 2.0) / 2.0
    backend.assert_allclose(grad, target_grad)


def test_two_variables_backpropagation():
    backend = construct_tensorflow_backend()

    theta = backend.engine.Variable([0.1234, 0.4321], dtype=backend.float64)
    # TODO: Fix parametrized gates so that `Circuit` can be defined outside
    # of the gradient tape
    with backend.engine.GradientTape() as tape:
        circuit = Circuit(2)
        circuit.add(gates.RX(0, theta[0]))
        circuit.add(gates.RY(1, theta[1]))
        result = backend.execute_circuit(circuit)
        loss = backend.real(result.state()[0])
    grad = tape.gradient(loss, theta)

    t = np.array([0.1234, 0.4321]) / 2.0
    target_loss = backend.cos(t[0]) * backend.cos(t[1])
    backend.assert_allclose(loss, target_loss)

    target_grad1 = -backend.sin(t[0]) * backend.cos(t[1])
    target_grad2 = -backend.cos(t[0]) * backend.sin(t[1])
    target_grad = np.array([target_grad1, target_grad2]) / 2.0
    backend.assert_allclose(grad, target_grad)
