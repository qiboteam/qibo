"""Tests for the Quantum Phase Estimation circuit defined in `qibo/models/qpe.py`."""

import math

import pytest

from qibo import Circuit, gates
from qibo.models import QFT, QPE
from qibo.quantum_info import bitstring_to_phase, random_unitary


def _estimated_phase(backend, circuit, nshots=100, **kwargs):
    frequencies = backend.execute_circuit(
        circuit, nshots=nshots, **kwargs
    ).frequencies()
    assert len(frequencies) == 1
    return bitstring_to_phase(next(iter(frequencies)))


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("numerator", [0, 1, 5])
@pytest.mark.parametrize("nqubits_counting", [3, 4])
def test_qpe_single_qubit_unitary(backend, nqubits_counting, numerator, density_matrix):
    phase = numerator / 2**nqubits_counting
    unitary = Circuit(1)
    unitary.add(gates.U1(0, theta=2 * math.pi * phase))
    qpe = QPE(unitary, nqubits_counting, density_matrix=density_matrix)
    assert qpe.nqubits == nqubits_counting + 1
    assert qpe.density_matrix == density_matrix

    # eigenstate |1> of the unitary
    circuit = Circuit(qpe.nqubits, density_matrix=density_matrix)
    circuit.add(gates.X(nqubits_counting))
    circuit.add(qpe)

    assert _estimated_phase(backend, circuit) == phase


@pytest.mark.parametrize(
    "build_unitary, build_eigenstate, nqubits_counting, phase",
    [
        # Pauli X, which is controlled by CNOT gates, with eigenstate |->
        (lambda: [gates.X(0)], lambda t: [gates.X(t), gates.H(t)], 3, 1 / 2),
        # CNOT, which is controlled by TOFFOLI gates, with eigenstate |1>|->
        (
            lambda: [gates.CNOT(0, 1)],
            lambda t: [gates.X(t), gates.X(t + 1), gates.H(t + 1)],
            3,
            1 / 2,
        ),
        # product of phase gates with eigenstate |11>
        (
            lambda: [
                gates.U1(0, theta=2 * math.pi / 8),
                gates.U1(1, theta=2 * math.pi * 3 / 16),
            ],
            lambda t: [gates.X(t), gates.X(t + 1)],
            4,
            5 / 16,
        ),
        # rotation with eigenstate |0>, whose eigenvalue is exp(-i theta / 2)
        (
            lambda: [gates.RZ(0, theta=-3 * math.pi / 2)],
            lambda t: [],
            3,
            3 / 8,
        ),
        # global phase, which becomes a relative phase when the unitary is controlled
        (
            lambda: [gates.H(0), gates.H(0), gates.GlobalPhase(2 * math.pi * 3 / 8)],
            lambda t: [],
            3,
            3 / 8,
        ),
    ],
)
def test_qpe_unitary_circuits(
    backend, build_unitary, build_eigenstate, nqubits_counting, phase
):
    unitary = Circuit(max(q for gate in build_unitary() for q in gate.qubits) + 1)
    unitary.add(build_unitary())
    qpe = QPE(unitary, nqubits_counting)

    circuit = Circuit(qpe.nqubits)
    circuit.add(build_eigenstate(nqubits_counting))
    circuit.add(qpe)

    assert _estimated_phase(backend, circuit) == phase


@pytest.mark.parametrize("nqubits_counting", [2, 3, 4])
def test_qpe_structure(nqubits_counting):
    unitary = Circuit(2)
    unitary.add(gates.U1(0, theta=0.3))
    qpe = QPE(unitary, nqubits_counting)

    assert qpe.nqubits == nqubits_counting + 2
    assert len(qpe.measurements) == 1
    assert qpe.measurements[0].qubits == tuple(range(nqubits_counting))

    # Hadamards, the controlled unitary applied 2^t - 1 times, inverse QFT and measurement
    nqft = QFT(nqubits_counting).ngates
    assert qpe.ngates == nqubits_counting + (2**nqubits_counting - 1) + nqft + 1
    assert [gate.target_qubits for gate in qpe.queue[:nqubits_counting]] == [
        (qubit,) for qubit in range(nqubits_counting)
    ]
    # the controlled gates act on the target qubits and are controlled by the counting qubits
    for gate in qpe.queue[
        nqubits_counting : nqubits_counting + 2**nqubits_counting - 1
    ]:
        assert gate.target_qubits == (nqubits_counting,)
        assert gate.control_qubits[0] < nqubits_counting


def test_qpe_without_measurement():
    unitary = Circuit(1)
    unitary.add(gates.U1(0, theta=0.3))
    qpe = QPE(unitary, 3, measure=False)
    assert qpe.measurements == []
    assert qpe.ngates == 3 + 7 + QFT(3).ngates


@pytest.mark.parametrize("nqubits_counting", [3, 4])
def test_qpe_matches_manual_construction(backend, nqubits_counting):
    phase = 0.3
    theta = 2 * math.pi * phase
    unitary = Circuit(1)
    unitary.add(gates.U1(0, theta=theta))

    circuit = Circuit(nqubits_counting + 1)
    circuit.add(gates.X(nqubits_counting))
    circuit.add(QPE(unitary, nqubits_counting, measure=False))

    target = Circuit(nqubits_counting + 1)
    target.add(gates.X(nqubits_counting))
    target.add(gates.H(qubit) for qubit in range(nqubits_counting))
    for qubit in range(nqubits_counting):
        power = 2 ** (nqubits_counting - 1 - qubit)
        target.add(gates.CU1(qubit, nqubits_counting, theta=power * theta))
    target.add(QFT(nqubits_counting).invert().on_qubits(*range(nqubits_counting)))

    backend.assert_circuitclose(circuit, target, atol=1e-8)


@pytest.mark.parametrize("phase", [0.3, 0.7, 0.9])
def test_qpe_phase_not_representable(backend, phase):
    nqubits_counting = 5
    unitary = Circuit(1)
    unitary.add(gates.U1(0, theta=2 * math.pi * phase))
    qpe = QPE(unitary, nqubits_counting, measure=False)

    circuit = Circuit(qpe.nqubits)
    circuit.add(gates.X(nqubits_counting))
    circuit.add(qpe)

    probabilities = backend.execute_circuit(circuit).probabilities(
        qubits=list(range(nqubits_counting))
    )
    estimate = int(backend.argmax(probabilities)) / 2**nqubits_counting
    # the most probable outcome is the closest value to the phase that is representable
    assert abs(estimate - phase) <= 2 ** (-nqubits_counting - 1)


@pytest.mark.parametrize("nqubits", [1, 2])
@pytest.mark.parametrize("kind", ["circuit", "gate", "matrix"])
def test_qpe_unitary_types(backend, kind, nqubits):
    """Estimates the eigenphases of a random unitary given as a circuit, gate or matrix."""
    nqubits_counting = 3
    numerators = [1, 3, 5, 6][: 2**nqubits]
    eigenvectors = random_unitary(2**nqubits, backend=backend)
    eigenvalues = backend.exp(
        2j
        * math.pi
        * backend.cast(numerators, dtype=backend.dtype)
        / 2**nqubits_counting
    )
    matrix = backend.matmul(
        backend.matmul(eigenvectors, backend.diag(eigenvalues)),
        backend.conj(eigenvectors).T,
    )
    target_qubits = range(nqubits_counting, nqubits_counting + nqubits)

    for index, numerator in enumerate(numerators):
        if kind == "matrix":
            unitary = matrix
        else:
            unitary = gates.Unitary(matrix, *range(nqubits))
        if kind == "circuit":
            unitary_circuit = Circuit(nqubits)
            unitary_circuit.add(unitary)
            unitary = unitary_circuit
        qpe = QPE(unitary, nqubits_counting)
        assert qpe.nqubits == nqubits_counting + nqubits

        # eigenstate |index> of the unitary, built from the eigenvectors
        circuit = Circuit(qpe.nqubits)
        circuit.add(
            gates.X(qubit)
            for qubit, bit in zip(target_qubits, f"{index:0{nqubits}b}")
            if bit == "1"
        )
        circuit.add(gates.Unitary(eigenvectors, *target_qubits))
        circuit.add(qpe)

        assert _estimated_phase(backend, circuit) == numerator / 2**nqubits_counting


def test_qpe_gate_qubits(backend):
    """The qubits of a gate are relabeled in increasing order, as in a smaller circuit."""
    nqubits_counting = 3
    matrix = random_unitary(4, backend=backend)
    single_qubit_matrix = random_unitary(2, backend=backend)
    cases = [
        (gates.Unitary(matrix, 5, 2), gates.Unitary(matrix, 1, 0)),
        (
            gates.Unitary(single_qubit_matrix, 4).controlled_by(1),
            gates.Unitary(single_qubit_matrix, 1).controlled_by(0),
        ),
        (gates.RX(3, theta=0.4), gates.RX(0, theta=0.4)),
        (gates.X(2).controlled_by(0, 5), gates.X(1).controlled_by(0, 2)),
    ]
    for gate, target_gate in cases:
        qpe = QPE(gate, nqubits_counting, measure=False)
        assert qpe.nqubits == nqubits_counting + len(gate.qubits)

        unitary = Circuit(len(gate.qubits))
        unitary.add(target_gate)
        backend.assert_circuitclose(
            qpe, QPE(unitary, nqubits_counting, measure=False), atol=1e-8
        )
        # the original gate is not modified
        assert gate.qubits != target_gate.qubits


def test_qpe_errors(backend):
    unitary = Circuit(1)
    unitary.add(gates.U1(0, theta=0.3))

    for nqubits_counting in (0, -1, 2.5, "3"):
        with pytest.raises(ValueError):
            QPE(unitary, nqubits_counting)

    for invalid_unitary in ("U", 0.5, [[1, 0], [0, 1]]):
        with pytest.raises(TypeError):
            QPE(invalid_unitary, 3)

    # matrices that are not square, that do not have dimension 2^n or have n = 0
    for matrix in (
        [1, 0],
        [[1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0, 0], [0, 1, 0, 0]],
        [[[1, 0], [0, 1]]],
    ):
        with pytest.raises(ValueError):
            QPE(backend.cast(matrix), 3)
    # matrix that is not unitary
    with pytest.raises(ValueError):
        QPE(backend.cast([[1, 1], [0, 1]]), 3)
    # gate that does not act on any qubit
    with pytest.raises(ValueError):
        QPE(gates.GlobalPhase(0.3), 3)

    # unitaries with gates that cannot be controlled
    with pytest.raises(NotImplementedError):
        QPE(gates.M(0), 3)
    unitary.add(gates.M(0))
    with pytest.raises(NotImplementedError):
        QPE(unitary, 3)
