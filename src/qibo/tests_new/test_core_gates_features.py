"""Test special features of core gates."""
import pytest
import numpy as np
import qibo
from qibo import K, gates
from qibo.models import Circuit
from qibo.tests_new.test_core_gates import random_state


@pytest.mark.parametrize("nqubits,targets", [(5, [2, 4]), (6, [3, 5])])
def test_collapse_gate_distributed(backend, accelerators, nqubits, targets):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    initial_state = random_state(nqubits)
    c = Circuit(nqubits, accelerators)
    c.add(gates.Collapse(*targets))
    final_state = c(np.copy(initial_state))
    slicer = nqubits * [slice(None)]
    for t in targets:
        slicer[t] = 0
    slicer = tuple(slicer)
    initial_state = initial_state.reshape(nqubits * (2,))
    target_state = np.zeros_like(initial_state)
    target_state[slicer] = initial_state[slicer]
    norm = (np.abs(target_state) ** 2).sum()
    target_state = target_state.ravel() / np.sqrt(norm)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_rx_parameter_setter(backend):
    """Check that the parameter setter of RX gate is working properly."""
    def exact_state(theta):
        phase = np.exp(1j * theta / 2.0)
        gate = np.array([[phase.real, -1j * phase.imag],
                        [-1j * phase.imag, phase.real]])
        return gate.dot(np.ones(2)) / np.sqrt(2)

    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    gate = gates.RX(0, theta=theta)
    initial_state = K.cast(np.ones(2) / np.sqrt(2))
    final_state = gate(initial_state)
    target_state = exact_state(theta)
    np.testing.assert_allclose(final_state, target_state)

    theta = 0.4321
    gate.parameters = theta
    initial_state = K.cast(np.ones(2) / np.sqrt(2))
    final_state = gate(initial_state)
    target_state = exact_state(theta)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)
