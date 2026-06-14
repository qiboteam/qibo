"""Tests for qibo.optimizers."""

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from qibo.optimizers import QuantumNaturalGradient


def test_quantum_natural_gradient_uses_qfim(monkeypatch):
    backend = NumpyBackend()
    circuit = Circuit(2)
    circuit.add(gates.RY(0, theta=1.0))
    circuit.add(gates.RZ(1, theta=1.0))
    qfim_calls = []

    def loss_fn(circuit, backend):
        params = np.asarray(circuit.get_parameters(output_format="flatlist"))
        target = np.asarray([0.0, -1.0])
        return float(np.sum((params - target) ** 2))

    def gradient_fn(circuit, backend):
        params = np.asarray(circuit.get_parameters(output_format="flatlist"))
        target = np.asarray([0.0, -1.0])
        return 2 * (params - target)

    def qfim(circuit, parameters=None, **kwargs):
        qfim_calls.append(np.asarray(parameters))
        return np.diag([2.0, 4.0])

    import qibo.quantum_info as quantum_info

    monkeypatch.setattr(quantum_info, "quantum_fisher_information_matrix", qfim)

    optimizer = QuantumNaturalGradient(
        circuit,
        loss_fn=loss_fn,
        gradient_fn=gradient_fn,
        learning_rate=0.5,
        regularization=0.0,
        backend=backend,
    )
    final_loss, losses, final_parameters = optimizer(steps=1)

    np.testing.assert_allclose(final_parameters, [0.5, 0.5])
    assert final_loss < losses[0]
    assert optimizer.n_calls_qfim == 1
    np.testing.assert_allclose(qfim_calls[0], [1.0, 1.0])


def test_quantum_natural_gradient_regularizes_singular_qfim(monkeypatch):
    backend = NumpyBackend()
    circuit = Circuit(2)
    circuit.add(gates.RY(0, theta=1.0))
    circuit.add(gates.RZ(1, theta=1.0))

    def loss_fn(circuit, backend):
        params = np.asarray(circuit.get_parameters(output_format="flatlist"))
        return float(np.sum(params**2))

    def gradient_fn(circuit, backend):
        return np.asarray([1.0, 2.0])

    import qibo.quantum_info as quantum_info

    monkeypatch.setattr(
        quantum_info,
        "quantum_fisher_information_matrix",
        lambda *args, **kwargs: np.zeros((2, 2)),
    )

    optimizer = QuantumNaturalGradient(
        circuit,
        loss_fn=loss_fn,
        gradient_fn=gradient_fn,
        learning_rate=0.25,
        regularization=1.0,
        backend=backend,
    )
    _, _, final_parameters = optimizer(steps=1)

    np.testing.assert_allclose(final_parameters, [0.75, 0.5])


def test_quantum_natural_gradient_finite_difference_gradient(monkeypatch):
    backend = NumpyBackend()
    circuit = Circuit(1)
    circuit.add(gates.RY(0, theta=0.4))

    def loss_fn(circuit, backend):
        param = circuit.get_parameters(output_format="flatlist")[0]
        return float(param**2)

    import qibo.quantum_info as quantum_info

    monkeypatch.setattr(
        quantum_info,
        "quantum_fisher_information_matrix",
        lambda *args, **kwargs: np.eye(1),
    )

    optimizer = QuantumNaturalGradient(
        circuit,
        loss_fn=loss_fn,
        learning_rate=0.1,
        regularization=0.0,
        finite_difference_epsilon=1e-7,
        backend=backend,
    )
    final_loss, losses, final_parameters = optimizer(steps=1)

    np.testing.assert_allclose(final_parameters, [0.32], atol=1e-6)
    assert final_loss < losses[0]
    assert optimizer.n_calls_gradient == 1


def test_quantum_natural_gradient_uses_separate_gradient_kwargs(monkeypatch):
    backend = NumpyBackend()
    circuit = Circuit(1)
    circuit.add(gates.RY(0, theta=1.0))

    def loss_fn(circuit, backend, offset):
        param = circuit.get_parameters(output_format="flatlist")[0]
        return float((param - offset) ** 2)

    def gradient_fn(circuit, backend, scale):
        param = circuit.get_parameters(output_format="flatlist")[0]
        return np.asarray([scale * param])

    import qibo.quantum_info as quantum_info

    monkeypatch.setattr(
        quantum_info,
        "quantum_fisher_information_matrix",
        lambda *args, **kwargs: np.eye(1),
    )

    optimizer = QuantumNaturalGradient(
        circuit,
        loss_fn=loss_fn,
        loss_kwargs={"offset": 0.0},
        gradient_fn=gradient_fn,
        gradient_kwargs={"scale": 3.0},
        learning_rate=0.1,
        regularization=0.0,
        backend=backend,
    )
    final_loss, losses, final_parameters = optimizer(steps=1)

    np.testing.assert_allclose(final_parameters, [0.7])
    assert final_loss < losses[0]


def test_quantum_natural_gradient_errors():
    backend = NumpyBackend()
    circuit = Circuit(1)

    with pytest.raises(ValueError):
        _ = QuantumNaturalGradient(
            circuit, loss_fn=lambda circuit, backend: 0, backend=backend
        )

    circuit.add(gates.RY(0, theta=0.1))

    with pytest.raises(ValueError):
        _ = QuantumNaturalGradient(
            circuit,
            loss_fn=lambda circuit, backend: 0,
            learning_rate=0,
            backend=backend,
        )
