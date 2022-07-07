"""Test :class:`qibo.gates.M` as standalone and as part of circuit."""
import pytest
import numpy as np
from qibo import models, gates
from qibo.tests.utils import random_state, random_density_matrix


@pytest.mark.parametrize("nqubits,targets",
                         [(2, [1]), (3, [1]), (4, [1, 3]), (5, [0, 3, 4]),
                          (6, [1, 3]), (4, [0, 2])])
def test_measurement_collapse(backend, nqubits, targets):
    initial_state = random_state(nqubits)
    c = models.Circuit(nqubits)
    m = c.add(gates.M(*targets, collapse=True))
    final_state = backend.execute_circuit(c, np.copy(initial_state), nshots=1)[0]
    if len(targets) > 1:
        results = m[0].result.samples[0]
    else:
        results = m.result.samples[0]
    slicer = nqubits * [slice(None)]
    for t, r in zip(targets, results):
        slicer[t] = int(r)
    slicer = tuple(slicer)
    initial_state = initial_state.reshape(nqubits * (2,))
    target_state = np.zeros_like(initial_state)
    target_state[slicer] = initial_state[slicer]
    norm = (np.abs(target_state) ** 2).sum()
    target_state = target_state.ravel() / np.sqrt(norm)
    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("nqubits,targets",
                         [(2, [1]), (3, [1]), (4, [1, 3]), (5, [0, 3, 4])])
def test_measurement_collapse_density_matrix(backend, nqubits, targets):
    initial_rho = random_density_matrix(nqubits)
    c = models.Circuit(nqubits, density_matrix=True)
    m = c.add(gates.M(*targets, collapse=True))
    final_rho = backend.execute_circuit(c, np.copy(initial_rho), nshots=1)[0]

    if len(targets) > 1:
        results = m[0].result.samples[0]
    else:
        results = m.result.samples[0]
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
    backend.assert_allclose(final_rho, target_rho)


def test_measurement_collapse_bitflip_noise(backend):
    c = models.Circuit(4)
    with pytest.raises(NotImplementedError):
        output = c.add(gates.M(0, 1, p0=0.2, collapse=True))


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("effect", [False, True])
def test_measurement_result_parameters(backend, effect, density_matrix):
    c = models.Circuit(4, density_matrix=density_matrix)
    if effect:
        c.add(gates.X(0))
    output = c.add(gates.M(0, collapse=True))
    c.add(gates.RX(1, theta=np.pi * output / 4))

    target_c = models.Circuit(4, density_matrix=density_matrix)
    if effect:
        target_c.add(gates.X(0))
        target_c.add(gates.RX(1, theta=np.pi / 4))

    final_state = backend.execute_circuit(c, nshots=1)[0]
    target_state = backend.execute_circuit(target_c)
    backend.assert_allclose(final_state, target_state)


def test_measurement_result_parameters_random(backend):
    initial_state = random_state(4)
    backend.set_seed(123)
    c = models.Circuit(4)
    output = c.add(gates.M(1, collapse=True))
    c.add(gates.RY(0, theta=np.pi * output / 5))
    c.add(gates.RX(2, theta=np.pi * output / 4))
    final_state = backend.execute_circuit(c, initial_state=np.copy(initial_state), nshots=1)[0]

    backend.set_seed(123)
    c = models.Circuit(4)
    m = c.add(gates.M(1, collapse=True))
    target_state = backend.execute_circuit(c, initial_state=np.copy(initial_state), nshots=1)[0]
    if int(m.outcome()):
        c = models.Circuit(4)
        c.add(gates.RY(0, theta=np.pi / 5))
        c.add(gates.RX(2, theta=np.pi / 4))
        target_state = backend.execute_circuit(c, initial_state=target_state)
    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("use_loop", [True, False])
def test_measurement_result_parameters_repeated_execution(backend, use_loop):
    initial_state = random_state(4)
    backend.set_seed(123)
    c = models.Circuit(4)
    output = c.add(gates.M(1, collapse=True))
    c.add(gates.RX(2, theta=np.pi * output / 4))
    if use_loop:
        final_states = []
        for _ in range(20):
            final_state = backend.execute_circuit(c, initial_state=np.copy(initial_state), nshots=1)
            final_states.append(final_state[0])
    else:
        final_states = backend.execute_circuit(c, initial_state=np.copy(initial_state), nshots=20)

    backend.set_seed(123)
    target_states = []
    for _ in range(20):
        c = models.Circuit(4)
        m = c.add(gates.M(1, collapse=True))
        target_state = backend.execute_circuit(c, np.copy(initial_state), nshots=1)[0]
        if int(m.outcome()):
            target_state = backend.apply_gate(gates.RX(2, theta=np.pi / 4), target_state, 4)
        target_states.append(backend.to_numpy(target_state))

    final_states = [backend.to_numpy(x) for x in final_states]
    backend.assert_allclose(final_states, target_states)


def test_measurement_result_parameters_repeated_execution_final_measurements(backend):
    initial_state = random_state(4)
    backend.set_seed(123)
    c = models.Circuit(4)
    output = c.add(gates.M(1, collapse=True))
    c.add(gates.RY(0, theta=np.pi * output / 3))
    c.add(gates.RY(2, theta=np.pi * output / 4))
    c.add(gates.M(0, 1, 2, 3))
    result = backend.execute_circuit(c, initial_state=np.copy(initial_state), nshots=30)
    final_samples = result.samples(binary=False)

    backend.set_seed(123)
    target_samples = []
    for _ in range(30):
        c = models.Circuit(4)
        m = c.add(gates.M(1, collapse=True))
        target_state = backend.execute_circuit(c, np.copy(initial_state), nshots=1)[0]
        c = models.Circuit(4)
        if int(m.outcome()):
            c.add(gates.RY(0, theta=np.pi / 3))
            c.add(gates.RY(2, theta=np.pi / 4))
        c.add(gates.M(0, 1, 2, 3))
        result = backend.execute_circuit(c, target_state, nshots=1)
        target_samples.append(result.samples(binary=False)[0])
    backend.assert_allclose(final_samples, target_samples)


def test_measurement_result_parameters_multiple_qubits(backend):
    initial_state = random_state(4)
    backend.set_seed(123)
    c = models.Circuit(4)
    output = c.add(gates.M(0, 1, 2, collapse=True))
    c.add(gates.RY(1, theta=np.pi * output[0] / 5))
    c.add(gates.RX(3, theta=np.pi * output[2] / 3))
    final_state = backend.execute_circuit(c, np.copy(initial_state), nshots=1)[0]

    backend.set_seed(123)
    c = models.Circuit(4)
    m = c.add(gates.M(0, 1, 2, collapse=True))
    target_state = backend.execute_circuit(c, np.copy(initial_state), nshots=1)[0]
    # not including in coverage because outcomes are probabilistic and may
    # not occur for the CI run
    if int(m[0].outcome()): # pragma: no cover
        target_state = backend.apply_gate(gates.RY(1, theta=np.pi / 5), target_state, 4)
    if int(m[2].outcome()): # pragma: no cover
        target_state = backend.apply_gate(gates.RX(3, theta=np.pi / 3), target_state, 4)
    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("nqubits,targets", [(5, [2, 4]), (6, [3, 5])])
def test_measurement_collapse_distributed(backend, accelerators, nqubits, targets):
    initial_state = random_state(nqubits)
    c = models.Circuit(nqubits, accelerators)
    output = c.add(gates.M(*targets, collapse=True))
    result = backend.execute_circuit(c, np.copy(initial_state), nshots=1)
    slicer = nqubits * [slice(None)]
    outcomes = [r.outcome() for r in output]
    for t, r in zip(targets, outcomes):
        slicer[t] = int(r)
    slicer = tuple(slicer)
    initial_state = initial_state.reshape(nqubits * (2,))
    target_state = np.zeros_like(initial_state)
    target_state[slicer] = initial_state[slicer]
    norm = (np.abs(target_state) ** 2).sum()
    target_state = target_state.ravel() / np.sqrt(norm)
    backend.assert_allclose(result[0], target_state)


def test_collapse_after_measurement(backend):
    qubits = [0, 2, 3]
    c = models.Circuit(5)
    c.add((gates.H(i) for i in range(5)))
    output = c.add(gates.M(*qubits, collapse=True))
    c.add((gates.H(i) for i in range(5)))
    final_state = backend.execute_circuit(c, nshots=1)[0]

    ct = models.Circuit(5)
    bitstring = [r.outcome() for r in output]
    for i, r in zip(qubits, bitstring):
        if r:
            ct.add(gates.X(i))
    ct.add((gates.H(i) for i in qubits))
    target_state = backend.execute_circuit(ct)
    backend.assert_allclose(final_state, target_state, atol=1e-15)
