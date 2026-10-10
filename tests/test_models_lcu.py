"""Tests for linear combination of unitaries in `qibo/models/lcu.py`."""

import pytest

from qibo import Circuit, gates, models
from qibo.models.lcu import lcu_circuit, lcu_prepare, lcu_select
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


@pytest.mark.parametrize("zero_term", [False, True])
def test_lcu_circuit_oracles(backend, zero_term):
    """The circuit is the composition of the oracles of ``lcu_prepare`` and ``lcu_select``."""
    unitaries = []
    for seed in range(3):
        unitary = Circuit(2)
        unitary.add(gates.Unitary(random_unitary(4, seed=seed, backend=backend), 0, 1))
        unitaries.append(unitary)
    coefficients = [0.5 - 0.2j, 0.0 if zero_term else -0.7, 0.3j]

    circuit, alpha, nauxiliary = lcu_circuit(unitaries, coefficients, backend=backend)

    prepare_left, prepare_right, prepare_alpha, prepare_nauxiliary = lcu_prepare(
        coefficients, backend=backend
    )
    select = lcu_select(unitaries, nauxiliary, backend=backend)
    composition = Circuit(nauxiliary + 2)
    composition.add(prepare_right.on_qubits(*range(nauxiliary)))
    composition.add(select.on_qubits(*range(nauxiliary + 2)))
    composition.add(prepare_left.invert().on_qubits(*range(nauxiliary)))

    assert alpha == prepare_alpha
    assert nauxiliary == prepare_nauxiliary
    # a zero term only changes SELECT in columns that PREPARE does not populate
    backend.assert_allclose(
        circuit.unitary(backend)[:, :4],
        composition.unitary(backend)[:, :4],
        atol=1e-10,
    )


def test_lcu_exports():
    """The functions are available from ``qibo.models``."""
    assert models.lcu_circuit is lcu_circuit
    assert models.lcu_prepare is lcu_prepare
    assert models.lcu_select is lcu_select


@pytest.mark.parametrize("nterms", [1, 2, 3, 4, 5, 8])
def test_lcu_prepare(backend, nterms):
    """The oracles load the square roots of the normalized magnitudes, with and without
    the phases of the coefficients."""
    coefficients = backend.random_normal(
        0.0, 1.0, size=(nterms,), seed=nterms, dtype="float64"
    ) + 1j * backend.random_normal(
        0.0, 1.0, size=(nterms,), seed=nterms + 100, dtype="float64"
    )

    prepare_left, prepare_right, alpha, nauxiliary = lcu_prepare(
        coefficients, backend=backend
    )

    magnitudes = backend.abs(coefficients)
    padding = backend.zeros(2**nauxiliary - nterms, dtype="float64")
    amplitudes = backend.concatenate(
        (backend.sqrt(magnitudes / magnitudes.sum()), padding)
    )
    phases = backend.concatenate((coefficients / magnitudes, padding))

    assert alpha == pytest.approx(float(backend.sum(magnitudes)))
    assert nauxiliary == max((nterms - 1).bit_length(), 1)
    assert prepare_left.nqubits == nauxiliary
    assert prepare_right.nqubits == nauxiliary
    backend.assert_allclose(prepare_left().state(), amplitudes, atol=1e-10)
    backend.assert_allclose(prepare_right().state(), amplitudes * phases, atol=1e-10)


def test_lcu_prepare_errors(backend):
    with pytest.raises(ValueError):
        lcu_prepare([], backend=backend)
    with pytest.raises(ValueError):
        lcu_prepare([[1.0, 1.0]], backend=backend)
    with pytest.raises(ValueError):
        lcu_prepare([0.0, 0.0], backend=backend)


def test_lcu_prepare_zero_coefficients(backend):
    """Zero coefficients have zero amplitude."""
    prepare_left, prepare_right, alpha, nauxiliary = lcu_prepare(
        [0.0, 2.0, 0.0, -2j], backend=backend
    )

    half = 0.5**0.5
    backend.assert_allclose(
        prepare_left().state(), backend.cast([0.0, half, 0.0, half]), atol=1e-10
    )
    backend.assert_allclose(
        prepare_right().state(), backend.cast([0.0, half, 0.0, -1j * half]), atol=1e-10
    )
    assert alpha == pytest.approx(4.0)
    assert nauxiliary == 2


@pytest.mark.parametrize("extra", [0, 1])
@pytest.mark.parametrize("nsystem", [1, 2])
@pytest.mark.parametrize("nunitaries", [1, 2, 3, 4, 5])
def test_lcu_select(backend, nunitaries, nsystem, extra):
    """The circuit applies each unitary on the system, selected by the auxiliary qubits."""
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
    nauxiliary = max((nunitaries - 1).bit_length(), 1) + extra

    select = lcu_select(unitaries, nauxiliary if extra else None, backend=backend)

    identity = backend.identity(2**nsystem, dtype="complex128")
    basis = backend.identity(2**nauxiliary, dtype="complex128")
    target = sum(
        backend.kron(
            backend.outer(basis[position], backend.conj(basis[position])),
            unitaries[position].unitary(backend) if position < nunitaries else identity,
        )
        for position in range(2**nauxiliary)
    )

    assert select.nqubits == nauxiliary + nsystem
    backend.assert_allclose(select.unitary(backend), target, atol=1e-10)


def test_lcu_select_controlled_gates(backend):
    """Gates that are already controlled are supported, and identities are skipped."""
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
    identity = Circuit(3)
    unitaries = [cnot, controlled, identity, swap]

    select = lcu_select(unitaries, backend=backend)

    basis = backend.identity(4, dtype="complex128")
    target = sum(
        backend.kron(
            backend.outer(basis[position], backend.conj(basis[position])),
            unitary.unitary(backend),
        )
        for position, unitary in enumerate(unitaries)
    )

    backend.assert_allclose(select.unitary(backend), target, atol=1e-10)
    assert len(lcu_select([identity, identity], backend=backend).queue) == 0


def test_lcu_select_errors(backend):
    unitary = Circuit(1)
    unitary.add(gates.X(0))
    other = Circuit(2)
    other.add(gates.X(0))

    with pytest.raises(ValueError):
        lcu_select([], backend=backend)
    with pytest.raises(ValueError):
        lcu_select([unitary, other], backend=backend)
    with pytest.raises(ValueError):
        lcu_select([unitary] * 3, nauxiliary=1, backend=backend)
