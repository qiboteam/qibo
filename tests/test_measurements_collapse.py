"""Test :class:`qibo.gates.M` as standalone and as part of circuit."""

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.quantum_info import random_density_matrix, random_statevector


@pytest.mark.parametrize(
    "nqubits,targets",
    [(2, [1]), (3, [1]), (4, [1, 3]), (5, [0, 3, 4]), (6, [1, 3]), (4, [0, 2])],
)
def test_measurement_collapse(backend, nqubits, targets):
    initial_state = random_statevector(2**nqubits, backend=backend)
    circuit = Circuit(nqubits)
    for q in np.random.randint(nqubits, size=np.random.randint(nqubits, size=1)):
        circuit.add(gates.H(q))
    r = circuit.add(gates.M(*targets, collapse=True))
    circuit.add(gates.M(*targets))
    outcome = backend.execute_circuit(
        circuit, backend.cast(initial_state, copy=True), nshots=1
    )
    samples = r.samples()[0]
    backend.assert_allclose(samples, outcome.samples()[0])


@pytest.mark.parametrize(
    "nqubits,targets", [(2, [1]), (3, [1]), (4, [1, 3]), (5, [0, 3, 4])]
)
def test_measurement_collapse_density_matrix(backend, nqubits, targets):
    def assign_value(rho, index, value):
        if backend.platform == "tensorflow":
            rho_numpy = rho.numpy()
            rho_numpy[index] = value
            return rho.__class__(rho_numpy, rho.device)

        rho[index] = value
        return rho

    initial_rho = random_density_matrix(2**nqubits, backend=backend)
    circuit = Circuit(nqubits, density_matrix=True)
    r = circuit.add(gates.M(*targets, collapse=True))
    final_rho = backend.execute_circuit(circuit, backend.copy(initial_rho), nshots=1)

    samples = r.samples()[0]
    target_rho = backend.reshape(initial_rho, 2 * nqubits * (2,))
    for q, r in zip(targets, samples):
        r = int(r)
        slicer = 2 * nqubits * [slice(None)]
        slicer[q], slicer[q + nqubits] = 1 - r, 1 - r
        target_rho = assign_value(target_rho, tuple(slicer), 0)
        slicer[q], slicer[q + nqubits] = r, 1 - r
        target_rho = assign_value(target_rho, tuple(slicer), 0)
        slicer[q], slicer[q + nqubits] = 1 - r, r
        target_rho = assign_value(target_rho, tuple(slicer), 0)
    target_rho = backend.reshape(target_rho, initial_rho.shape)
    target_rho = target_rho / backend.trace(target_rho)
    backend.assert_allclose(final_rho, target_rho)


def test_measurement_collapse_bitflip_noise(backend):
    circuit = Circuit(4)
    with pytest.raises(NotImplementedError):
        output = circuit.add(gates.M(0, 1, p0=0.2, collapse=True))


@pytest.mark.parametrize("density_matrix", [True, False])
@pytest.mark.parametrize("effect", [False, True])
def test_measurement_result_parameters(backend, effect, density_matrix):
    circuit = Circuit(4, density_matrix=density_matrix)
    if effect:
        circuit.add(gates.X(0))
    r = circuit.add(gates.M(0, collapse=True))
    circuit.add(gates.RX(1, theta=np.pi * r.symbols[0] / 4))
    if not density_matrix:
        circuit.add(gates.M(0))

    target_circuit = Circuit(4, density_matrix=density_matrix)
    if effect:
        target_circuit.add(gates.X(0))
        target_circuit.add(gates.RX(1, theta=np.pi / 4))
    if not density_matrix:
        target_circuit.add(gates.M(0))

    final_state = backend.execute_circuit(circuit, nshots=1)
    target_state = backend.execute_circuit(target_circuit)
    if not density_matrix:
        final_state = final_state.samples()[0]
        target_state = target_state.samples()[0]
    backend.assert_allclose(final_state, target_state)


def test_measurement_result_parameters_random(backend):
    initial_state = random_density_matrix(2**4, backend=backend)
    backend.set_seed(123)
    circuit = Circuit(4, density_matrix=True)
    r = circuit.add(gates.M(1, collapse=True))
    circuit.add(gates.RY(0, theta=np.pi * r.symbols[0] / 5))
    circuit.add(gates.RX(2, theta=np.pi * r.symbols[0] / 4))
    final_state = backend.execute_circuit(
        circuit, initial_state=backend.copy(initial_state), nshots=1
    )

    backend.set_seed(123)
    circuit = Circuit(4, density_matrix=True)
    m = circuit.add(gates.M(1, collapse=True))
    target_state = backend.execute_circuit(
        circuit, initial_state=backend.copy(initial_state), nshots=1
    ).state()
    if int(m.symbols[0].outcome()):
        circuit = Circuit(4, density_matrix=True)
        circuit.add(gates.RY(0, theta=np.pi / 5))
        circuit.add(gates.RX(2, theta=np.pi / 4))
        target_state = backend.execute_circuit(circuit, initial_state=target_state)
    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("use_loop", [True, False])
def test_measurement_result_parameters_repeated_execution(backend, use_loop):
    initial_state = random_density_matrix(2**4, backend=backend)
    backend.set_seed(123)
    circuit = Circuit(4, density_matrix=True)
    r = circuit.add(gates.M(1, collapse=True))
    circuit.add(gates.RX(2, theta=np.pi * r.symbols[0] / 4))
    if use_loop:
        final_states = []
        for _ in range(20):
            final_state = backend.execute_circuit(
                circuit, initial_state=backend.copy(initial_state), nshots=1
            )
            final_states.append(final_state.state())
        final_states = backend.engine.mean(backend.cast(final_states), 0)
    else:
        final_states = backend.execute_circuit(
            circuit, initial_state=backend.copy(initial_state), nshots=20
        ).state()

    backend.set_seed(123)
    target_states = []
    for _ in range(20):
        circuit = Circuit(4, density_matrix=True)
        m = circuit.add(gates.M(1, collapse=True))
        target_state = backend.execute_circuit(
            circuit, backend.copy(initial_state), nshots=1
        ).state()
        if int(m.symbols[0].outcome()):
            target_state = backend.apply_gate(
                gates.RX(2, theta=np.pi / 4), target_state, 4
            )
        target_states.append(backend.to_numpy(target_state))

    target_states = np.asarray(target_states).mean(0)
    backend.assert_allclose(final_states, target_states)


def test_measurement_result_parameters_repeated_execution_final_measurements(backend):
    initial_state = random_density_matrix(2**4, backend=backend)
    backend.set_seed(123)
    circuit = Circuit(4, density_matrix=True)
    r = circuit.add(gates.M(1, collapse=True))
    circuit.add(gates.RY(0, theta=np.pi * r.symbols[0] / 3))
    circuit.add(gates.RY(2, theta=np.pi * r.symbols[0] / 4))
    circuit.add(gates.M(0, 1, 2, 3))
    result = backend.execute_circuit(
        circuit, initial_state=backend.cast(initial_state, copy=True), nshots=30
    )
    final_samples = result.samples(binary=False)

    backend.set_seed(123)
    target_samples = []
    for _ in range(30):
        circuit = Circuit(4, density_matrix=True)
        m = circuit.add(gates.M(1, collapse=True))
        target_state = backend.execute_circuit(
            circuit, backend.copy(initial_state), nshots=1
        ).state()
        circuit = Circuit(4, density_matrix=True)
        if int(m.symbols[0].outcome()):
            circuit.add(gates.RY(0, theta=np.pi / 3))
            circuit.add(gates.RY(2, theta=np.pi / 4))
        circuit.add(gates.M(0, 1, 2, 3))
        result = backend.execute_circuit(circuit, target_state, nshots=1)
        target_samples.append(result.samples(binary=False)[0])
    backend.assert_allclose(final_samples, target_samples)


def test_measurement_result_parameters_multiple_qubits(backend):
    initial_state = random_density_matrix(2**4, backend=backend)
    backend.set_seed(123)
    circuit = Circuit(4, density_matrix=True)
    r = circuit.add(gates.M(0, 1, 2, collapse=True))
    circuit.add(gates.RY(1, theta=np.pi * r.symbols[0] / 5))
    circuit.add(gates.RX(3, theta=np.pi * r.symbols[2] / 3))
    final_state = backend.execute_circuit(
        circuit, backend.copy(initial_state), nshots=1
    )

    backend.set_seed(123)
    circuit = Circuit(4, density_matrix=True)
    m = circuit.add(gates.M(0, 1, 2, collapse=True))
    target_state = backend.execute_circuit(
        circuit, backend.copy(initial_state), nshots=1
    ).state()
    # not including in coverage because outcomes are probabilistic and may
    # not occur for the CI run
    if int(m.symbols[0].outcome()):  # pragma: no cover
        target_state = backend.apply_gate(gates.RY(1, theta=np.pi / 5), target_state, 4)
    if int(m.symbols[2].outcome()):  # pragma: no cover
        target_state = backend.apply_gate(gates.RX(3, theta=np.pi / 3), target_state, 4)
    backend.assert_allclose(final_state, target_state)


@pytest.mark.skip(reason="this has to be updated for density matrices")
@pytest.mark.parametrize("nqubits,targets", [(5, [2, 4]), (6, [3, 5])])
def test_measurement_collapse_distributed(backend, accelerators, nqubits, targets):
    initial_state = random_density_matrix(2**nqubits, backend=backend)
    circuit = Circuit(nqubits, accelerators, density_matrix=True)
    m = circuit.add(gates.M(*targets, collapse=True))
    result = backend.execute_circuit(circuit, np.copy(initial_state), nshots=1).state()
    slicer = 2 * nqubits * [slice(None)]
    outcomes = [r.outcome() for r in m.symbols]
    for t, r in zip(targets, outcomes):
        slicer[t] = int(r)
    slicer = tuple(slicer)
    initial_state = initial_state.reshape(2 * nqubits * (2,))
    target_state = np.zeros_like(initial_state)
    target_state[slicer] = initial_state[slicer]
    norm = (np.abs(target_state) ** 2).sum()
    target_state = target_state.ravel() / np.sqrt(norm)
    backend.assert_allclose(result, target_state.reshape(2**nqubits, 2**nqubits))


def test_collapse_after_measurement(backend):
    qubits = [0, 2, 3]
    circuit = Circuit(5, density_matrix=True)
    circuit.add(gates.H(i) for i in range(5))
    m = circuit.add(gates.M(*qubits, collapse=True))
    circuit.add(gates.H(i) for i in range(5))
    final_state = backend.execute_circuit(circuit, nshots=1)

    ct = Circuit(5, density_matrix=True)
    bitstring = [r.outcome() for r in m.symbols]
    for i, r in zip(qubits, bitstring):
        if r:
            ct.add(gates.X(i))
    ct.add(gates.H(i) for i in qubits)
    target_state = backend.execute_circuit(ct)
    backend.assert_allclose(final_state, target_state, atol=1e-15)


def test_collapse_error(backend):
    circuit = Circuit(1)
    m = circuit.add(gates.M(0, collapse=True))
    with pytest.raises(Exception) as exc_info:
        backend.execute_circuit(circuit)
    assert (
        str(exc_info.value)
        == "The circuit contains only collapsing measurements (`collapse=True`) but `density_matrix=False`. Please set `density_matrix=True` to retrieve the final state after execution."
    )
