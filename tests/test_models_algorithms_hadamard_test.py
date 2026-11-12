"""Tests for the Hadamard and swap tests in `qibo/models/algorithms/hadamard_test.py`."""

import pytest

from qibo import Circuit, gates
from qibo.models import hadamard_test, swap_test
from qibo.quantum_info import random_unitary


def _random_circuit(nqubits, seed, backend):
    circuit = Circuit(nqubits)
    circuit.add(
        gates.Unitary(
            random_unitary(2**nqubits, seed=seed, backend=backend), *range(nqubits)
        )
    )
    return circuit


def _state(circuit, backend):
    return backend.execute_circuit(circuit).state()


@pytest.mark.parametrize("measure", [True, False])
@pytest.mark.parametrize("use_observable", [False, True])
@pytest.mark.parametrize("part", ["real", "imag"])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_hadamard_test(backend, nqubits, part, use_observable, measure):
    """The probability of the auxiliary qubit gives the part of the amplitude."""
    phi = _random_circuit(nqubits, 1, backend)
    psi = _random_circuit(nqubits, 2, backend)
    observable = _random_circuit(nqubits, 3, backend) if use_observable else None

    circuit = hadamard_test(phi, psi, observable, part, measure, backend=backend)

    ket = _state(psi, backend)
    if use_observable:
        ket = backend.matmul(observable.unitary(backend), ket)
    amplitude = backend.sum(backend.conj(_state(phi, backend)) * ket)
    target = float(amplitude.real if part == "real" else amplitude.imag)

    probability = backend.execute_circuit(circuit).probabilities(qubits=[0])[0]

    assert circuit.nqubits == nqubits + 1
    assert len(circuit.measurements) == int(measure)
    backend.assert_allclose(2 * float(probability) - 1, target, atol=1e-10)


def test_hadamard_test_controlled_gates(backend):
    """Gates that are already controlled are supported, and so are empty circuits."""
    phi = Circuit(2)
    phi.add(gates.H(0))
    phi.add(gates.CRY(0, 1, 0.7))
    psi = Circuit(2)
    observable = Circuit(2)
    observable.add(gates.CNOT(0, 1))
    observable.add(gates.Z(0))

    probability = backend.execute_circuit(
        hadamard_test(phi, psi, observable, backend=backend)
    ).probabilities(qubits=[0])[0]

    ket = backend.matmul(observable.unitary(backend), _state(psi, backend))
    target = float(backend.sum(backend.conj(_state(phi, backend)) * ket).real)

    backend.assert_allclose(2 * float(probability) - 1, target, atol=1e-10)


def test_hadamard_test_density_matrix(backend):
    """Keyword arguments are passed to the circuit."""
    phi = Circuit(1)
    psi = Circuit(1)
    psi.add(gates.X(0))

    circuit = hadamard_test(phi, psi, density_matrix=True, backend=backend)
    probability = backend.execute_circuit(circuit).probabilities(qubits=[0])[0]

    assert circuit.density_matrix
    backend.assert_allclose(float(probability), 0.5, atol=1e-10)


def test_hadamard_test_errors(backend):
    """The inputs are validated."""
    circuit = Circuit(1)
    measured = Circuit(1)
    measured.add(gates.M(0))

    with pytest.raises(TypeError):
        hadamard_test(circuit, "psi", backend=backend)
    with pytest.raises(ValueError):
        hadamard_test(circuit, measured, backend=backend)
    with pytest.raises(ValueError):
        hadamard_test(circuit, Circuit(2), backend=backend)
    with pytest.raises(ValueError):
        hadamard_test(circuit, circuit, Circuit(2), backend=backend)
    with pytest.raises(ValueError):
        hadamard_test(circuit, circuit, part="abs", backend=backend)


@pytest.mark.parametrize("measure", [True, False])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_swap_test(backend, nqubits, measure):
    """The probability of the auxiliary qubit gives the squared overlap."""
    phi = _random_circuit(nqubits, 4, backend)
    psi = _random_circuit(nqubits, 5, backend)

    circuit = swap_test(phi, psi, measure)

    overlap = backend.sum(backend.conj(_state(phi, backend)) * _state(psi, backend))
    probability = backend.execute_circuit(circuit).probabilities(qubits=[0])[0]

    assert circuit.nqubits == 2 * nqubits + 1
    assert len(circuit.measurements) == int(measure)
    backend.assert_allclose(
        2 * float(probability) - 1, float(backend.abs(overlap) ** 2), atol=1e-10
    )


def test_swap_test_identical_and_orthogonal(backend):
    """The auxiliary qubit is always in zero for equal states and uniform if orthogonal."""
    zero = Circuit(1)
    one = Circuit(1)
    one.add(gates.X(0))

    equal = backend.execute_circuit(swap_test(one, one)).probabilities(qubits=[0])
    orthogonal = backend.execute_circuit(swap_test(zero, one)).probabilities(qubits=[0])

    backend.assert_allclose(equal, [1.0, 0.0], atol=1e-10)
    backend.assert_allclose(orthogonal, [0.5, 0.5], atol=1e-10)


def test_swap_test_density_matrix(backend):
    """Keyword arguments are passed to the circuit."""
    circuit = swap_test(Circuit(1), Circuit(1), density_matrix=True)

    assert circuit.density_matrix


def test_swap_test_errors():
    """The inputs are validated."""
    measured = Circuit(1)
    measured.add(gates.M(0))

    with pytest.raises(TypeError):
        swap_test(Circuit(1), "psi")
    with pytest.raises(ValueError):
        swap_test(Circuit(1), measured)
    with pytest.raises(ValueError):
        swap_test(Circuit(1), Circuit(2))
