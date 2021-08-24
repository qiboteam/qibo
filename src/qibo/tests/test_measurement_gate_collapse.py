"""Test :class:`qibo.abstractions.gates.M` as standalone and as part of circuit."""
import pytest
import numpy as np
from qibo import models, gates, K
from qibo.tests.utils import random_state, random_density_matrix


@pytest.mark.parametrize("nqubits,targets",
                         [(2, [1]), (3, [1]), (4, [1, 3]), (5, [0, 3, 4]),
                          (6, [1, 3]), (4, [0, 2])])
def test_measurement_collapse(backend, nqubits, targets):
    initial_state = random_state(nqubits)
    gate = gates.M(*targets, collapse=True)
    final_state = gate(K.cast(np.copy(initial_state)), nshots=1)
    results = gate.result.binary[0]
    slicer = nqubits * [slice(None)]
    for t, r in zip(targets, results):
        slicer[t] = int(r)
    slicer = tuple(slicer)
    initial_state = initial_state.reshape(nqubits * (2,))
    target_state = np.zeros_like(initial_state)
    target_state[slicer] = initial_state[slicer]
    norm = (np.abs(target_state) ** 2).sum()
    target_state = target_state.ravel() / np.sqrt(norm)
    K.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("nqubits,targets",
                         [(2, [1]), (3, [1]), (4, [1, 3]), (5, [0, 3, 4])])
def test_measurement_collapse_density_matrix(backend, nqubits, targets):
    initial_rho = random_density_matrix(nqubits)
    gate = gates.M(*targets, collapse=True)
    gate.density_matrix = True
    final_rho = gate(K.cast(np.copy(initial_rho)), nshots=1)
    results = gate.result.binary[0]
    target_rho = np.reshape(initial_rho, 2 * nqubits * (2,))
    for q, r in zip(targets, results):
        r = int(r)
        slicer = 2 * nqubits * [slice(None)]
        slicer[q], slicer[q + nqubits] = 1 - r, 1 - r
        target_rho[tuple(slicer)] = 0
        slicer[q], slicer[q + nqubits] = r, 1 - r
        target_rho[tuple(slicer)] = 0
        slicer[q], slicer[q + nqubits] = 1 - r, r
        target_rho[tuple(slicer)] = 0
    target_rho = np.reshape(target_rho, initial_rho.shape)
    target_rho = target_rho / np.trace(target_rho)
    K.assert_allclose(final_rho, target_rho)


def test_measurement_collapse_errors(backend):
    gate = gates.M(0, 1, collapse=True)
    state = np.ones(4) / 4
    with pytest.raises(ValueError):
        state = gate(K.cast(state), nshots=100)


def test_measurement_collapse_bitflip_noise(backend, accelerators):
    K.set_seed(123)
    c = models.Circuit(4, accelerators)
    with pytest.raises(NotImplementedError):
        output = c.add(gates.M(0, 1, p0=0.2, collapse=True))


@pytest.mark.parametrize("effect", [False, True])
def test_measurement_result_parameters(backend, accelerators, effect):
    c = models.Circuit(4, accelerators)
    if effect:
        c.add(gates.X(0))
    output = c.add(gates.M(0, collapse=True))
    c.add(gates.RX(1, theta=np.pi * output / 4))

    target_c = models.Circuit(4)
    if effect:
        target_c.add(gates.X(0))
        target_c.add(gates.RX(1, theta=np.pi / 4))
    K.assert_allclose(c(), target_c())


def set_device_seed(seed, accelerators):
    if accelerators:
        with K.on_cpu():
            K.set_seed(seed)
    else:
        K.set_seed(seed)


def test_measurement_result_parameters_random(backend, accelerators):
    test_device = K.cpu_devices[0] if accelerators else K.default_device
    initial_state = random_state(4)
    set_device_seed(123, accelerators)
    c = models.Circuit(4, accelerators)
    output = c.add(gates.M(1, collapse=True))
    c.add(gates.RY(0, theta=np.pi * output / 5))
    c.add(gates.RX(2, theta=np.pi * output / 4))
    result = c(initial_state=np.copy(initial_state))
    assert len(output.frequencies()) == 1

    set_device_seed(123, accelerators)
    with K.device(test_device):
        collapse = gates.M(1, collapse=True)
        target_state = collapse(K.cast(np.copy(initial_state)))
        if int(collapse.result.outcome()):
            target_state = gates.RY(0, theta=np.pi / 5)(target_state)
            target_state = gates.RX(2, theta=np.pi / 4)(target_state)
    K.assert_allclose(result, target_state)


@pytest.mark.parametrize("use_loop", [True, False])
def test_measurement_result_parameters_repeated_execution(backend, accelerators, use_loop):
    test_device = K.cpu_devices[0] if accelerators else K.default_device
    initial_state = random_state(4)
    set_device_seed(123, accelerators)
    c = models.Circuit(4, accelerators)
    output = c.add(gates.M(1, collapse=True))
    c.add(gates.RX(2, theta=np.pi * output / 4))
    if use_loop:
        final_states = []
        for _ in range(20):
            final_states.append(c(np.copy(initial_state)).state())
    else:
        final_states = c(initial_state=np.copy(initial_state), nshots=20)

    set_device_seed(123, accelerators)
    target_states = []
    with K.device(test_device):
        for _ in range(20):
            collapse = gates.M(1, collapse=True)
            target_state = collapse(K.cast(np.copy(initial_state)))
            if int(collapse.result.outcome()):
                target_state = gates.RX(2, theta=np.pi / 4)(target_state)
            target_states.append(np.copy(target_state))
        final_states = K.stack(final_states)
        target_states = K.stack(target_states)
    K.assert_allclose(final_states, target_states)


def test_measurement_result_parameters_repeated_execution_final_measurements(backend):
    initial_state = random_state(4)
    K.set_seed(123)
    c = models.Circuit(4)
    output = c.add(gates.M(1, collapse=True))
    c.add(gates.RY(0, theta=np.pi * output / 3))
    c.add(gates.RY(2, theta=np.pi * output / 4))
    c.add(gates.M(0, 1, 2, 3))
    result = c(initial_state=np.copy(initial_state), nshots=30)
    final_samples = result.samples(binary=False)

    K.set_seed(123)
    target_samples = []
    for _ in range(30):
        collapse = gates.M(1, collapse=True)
        target_state = collapse(K.cast(np.copy(initial_state)))
        if int(collapse.result.outcome()):
            target_state = gates.RY(0, theta=np.pi / 3)(target_state)
            target_state = gates.RY(2, theta=np.pi / 4)(target_state)
        with K.device(K.default_device):
            target_result = gates.M(0, 1, 2, 3)(target_state)
            target_samples.append(target_result.decimal[0])
    target_samples = K.stack(target_samples)
    K.assert_allclose(final_samples, target_samples)


def test_measurement_result_parameters_multiple_qubits(backend):
    initial_state = random_state(4)
    K.set_seed(123)
    c = models.Circuit(4)
    output = c.add(gates.M(0, 1, 2, collapse=True))
    c.add(gates.RY(1, theta=np.pi * output[0] / 5))
    c.add(gates.RX(3, theta=np.pi * output[2] / 3))
    result = c(initial_state=np.copy(initial_state))

    K.set_seed(123)
    collapse = gates.M(0, 1, 2, collapse=True)
    target_state = collapse(K.cast(np.copy(initial_state)))
    if int(collapse.result.outcome(0)):
        target_state = gates.RY(1, theta=np.pi / 5)(target_state)
    if int(collapse.result.outcome(2)):
        target_state = gates.RX(3, theta=np.pi / 3)(target_state)
    K.assert_allclose(result, target_state)
