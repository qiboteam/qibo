"""Tests for linear combination of unitaries in `qibo/models/lcu.py`."""

import pytest

from qibo import Circuit, gates, models
from qibo.models.lcu import lcu_circuit
from qibo.quantum_info import random_unitary


@pytest.mark.parametrize("nsystem", [1, 2])
@pytest.mark.parametrize("nunitaries", [1, 2, 3, 4, 5])
def test_lcu_circuit(backend, nunitaries, nsystem):
    """The circuit is unitary and has the normalized linear combination as its block."""
    unitaries = []
    for seed in range(nunitaries):
        unitary = Circuit(nsystem)
        unitary.add(
            gates.Unitary(
                random_unitary(2**nsystem, seed=seed, backend=backend),
                *range(nsystem),
            )
        )
        unitaries.append(unitary)
    coefficients = backend.random_normal(
        0.0, 1.0, size=(nunitaries,), seed=nunitaries, dtype="float64"
    ) + 1j * backend.random_normal(
        0.0, 1.0, size=(nunitaries,), seed=nunitaries + 100, dtype="float64"
    )

    circuit, alpha, nauxiliary = lcu_circuit(unitaries, coefficients, backend=backend)

    target = sum(
        coefficient * unitary.unitary(backend)
        for coefficient, unitary in zip(coefficients, unitaries)
    )
    unitary = circuit.unitary(backend)

    assert alpha == pytest.approx(float(backend.sum(backend.abs(coefficients))))
    assert nauxiliary == max((nunitaries - 1).bit_length(), 1)
    assert circuit.nqubits == nauxiliary + nsystem
    backend.assert_allclose(
        backend.matmul(unitary, backend.dagger(unitary)),
        backend.identity(2 ** (nauxiliary + nsystem), dtype="complex128"),
        atol=1e-10,
    )
    backend.assert_allclose(
        unitary[: 2**nsystem, : 2**nsystem], target / alpha, atol=1e-10
    )


def test_lcu_circuit_controlled_gates(backend):
    """Gates that are already controlled are supported, and zero terms are skipped."""
    cnot = Circuit(3)
    cnot.add(gates.CNOT(0, 1))
    cnot.add(gates.RX(2, 0.3))
    controlled = Circuit(3)
    controlled.add(gates.H(0))
    controlled.add(gates.CRY(1, 2, 0.7))
    controlled.add(gates.TOFFOLI(0, 1, 2))
    controlled.add(gates.Y(0).controlled_by(1, 2))
    swap = Circuit(3)
    swap.add(gates.SWAP(0, 2))
    swap.add(gates.CZ(1, 0))
    swap.add(gates.SWAP(0, 1).controlled_by(2))
    unitaries = [cnot, controlled, swap, cnot]
    coefficients = [0.0, 0.3 - 0.4j, -0.5, 0.2j]

    circuit, alpha, nauxiliary = lcu_circuit(unitaries, coefficients, backend=backend)

    target = sum(
        coefficient * unitary.unitary(backend)
        for coefficient, unitary in zip(coefficients, unitaries)
    )

    assert alpha == pytest.approx(1.2)
    assert nauxiliary == 2
    backend.assert_allclose(
        circuit.unitary(backend)[:8, :8], target / alpha, atol=1e-10
    )


def test_lcu_circuit_exports():
    """The function is available from ``qibo.models``."""
    assert models.lcu_circuit is lcu_circuit


def test_lcu_circuit_errors(backend):
    unitary = Circuit(1)
    unitary.add(gates.X(0))
    other = Circuit(2)
    other.add(gates.X(0))

    with pytest.raises(ValueError):
        lcu_circuit([], [], backend=backend)
    with pytest.raises(ValueError):
        lcu_circuit([unitary, unitary], [1.0], backend=backend)
    with pytest.raises(ValueError):
        lcu_circuit([unitary, unitary], [[1.0, 1.0]], backend=backend)
    with pytest.raises(ValueError):
        lcu_circuit([unitary, other], [1.0, 1.0], backend=backend)
    with pytest.raises(ValueError):
        lcu_circuit([unitary, unitary], [0.0, 0.0], backend=backend)
