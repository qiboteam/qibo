"""Tests for the Quantum Phase Estimation circuits defined in `qibo/models/qpe.py`."""

import math

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.measurements import MeasurementResult
from qibo.models import IQPE, QFT, QPE
from qibo.quantum_info import bitstring_to_phase, estimate_phase, random_unitary


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


@pytest.mark.parametrize("nqubits_counting", [np.int64(3), np.uint8(3), np.int32(3)])
def test_qpe_numpy_integer_counting_qubits(backend, nqubits_counting):
    unitary = Circuit(1)
    unitary.add(gates.U1(0, theta=0.3))

    qpe = QPE(unitary, nqubits_counting, measure=False)
    target = QPE(unitary, 3, measure=False)
    assert isinstance(qpe.nqubits, int)
    assert qpe.nqubits == target.nqubits == 4
    assert qpe.ngates == target.ngates
    backend.assert_circuitclose(qpe, target)


@pytest.mark.parametrize("function", [QPE, IQPE])
def test_phase_estimation_errors(backend, function):
    unitary = Circuit(1)
    unitary.add(gates.U1(0, theta=0.3))

    for nqubits_counting in (0, -1, 2.5, "3", True, False, np.int64(0), np.int8(-1)):
        with pytest.raises(ValueError):
            function(unitary, nqubits_counting)

    for invalid_unitary in ("U", 0.5, [[1, 0], [0, 1]]):
        with pytest.raises(TypeError):
            function(invalid_unitary, 3)

    # matrices that are not square, that do not have dimension 2^n or have n = 0
    for matrix in (
        [1, 0],
        [[1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0, 0], [0, 1, 0, 0]],
        [[[1, 0], [0, 1]]],
    ):
        with pytest.raises(ValueError):
            function(backend.cast(matrix), 3)
    # matrix that is not unitary
    with pytest.raises(ValueError):
        function(backend.cast([[1, 1], [0, 1]]), 3)
    # gate that does not act on any qubit
    with pytest.raises(ValueError):
        function(gates.GlobalPhase(0.3), 3)

    # unitaries with gates that cannot be controlled
    with pytest.raises(NotImplementedError):
        function(gates.M(0), 3)
    unitary.add(gates.M(0))
    with pytest.raises(NotImplementedError):
        function(unitary, 3)


@pytest.mark.parametrize("density_matrix", [True, False])
@pytest.mark.parametrize("nqubits", [1, 2])
@pytest.mark.parametrize("kind", ["circuit", "gate", "matrix"])
def test_iqpe_unitary_types(backend, kind, nqubits, density_matrix):
    """Estimates the eigenphases of a random unitary given as a circuit, gate or matrix."""
    nbits = 3
    numerators = [1, 3, 5, 6][: 2**nqubits]
    numerators = backend.cast(numerators, dtype=backend.dtype)
    eigenvectors = random_unitary(2**nqubits, backend=backend)
    eigenvalues = backend.exp(2j * math.pi * numerators / 2**nbits)
    matrix = backend.matmul(
        backend.matmul(eigenvectors, backend.diag(eigenvalues)),
        backend.dagger(eigenvectors),
    )
    target_qubits = range(1, 1 + nqubits)

    for index, numerator in enumerate(numerators):
        if kind == "matrix":
            unitary = matrix
        else:
            unitary = gates.Unitary(matrix, *range(nqubits))
        if kind == "circuit":
            unitary_circuit = Circuit(nqubits)
            unitary_circuit.add(unitary)
            unitary = unitary_circuit
        iqpe, results = IQPE(unitary, nbits, density_matrix=density_matrix)
        assert iqpe.nqubits == 1 + nqubits
        assert iqpe.density_matrix == density_matrix

        # eigenstate |index> of the unitary, built from the eigenvectors
        circuit = Circuit(iqpe.nqubits, density_matrix=density_matrix)
        circuit.add(
            gates.X(qubit)
            for qubit, bit in zip(target_qubits, f"{index:0{nqubits}b}")
            if bit == "1"
        )
        circuit.add(gates.Unitary(eigenvectors, *target_qubits))
        circuit.add(iqpe)
        backend.execute_circuit(circuit, nshots=20)

        joint_result = MeasurementResult.join(*results, backend=backend)
        # the phase is exactly represented, so all the shots give the same outcome
        assert len(joint_result.frequencies(backend=backend)) == 1
        assert estimate_phase(joint_result) == numerator / 2**nbits


@pytest.mark.parametrize("nbits", [1, 2, 4])
def test_iqpe_structure(nbits):
    unitary = Circuit(2)
    unitary.add(gates.U1(0, theta=0.3))
    iqpe, results = IQPE(unitary, nbits)

    assert iqpe.nqubits == 3
    assert not iqpe.density_matrix
    assert len(results) == nbits
    assert all(isinstance(result, MeasurementResult) for result in results)

    def count(gate_type):
        return sum(type(gate) is gate_type for gate in iqpe.queue)

    # the unitary is applied 2^t - 1 times, controlled by the counting qubit
    assert count(gates.CU1) == 2**nbits - 1
    for gate in iqpe.queue:
        if isinstance(gate, gates.CU1):
            assert gate.control_qubits == (0,)
            assert gate.target_qubits == (1,)
    assert count(gates.H) == 2 * nbits
    # corrections and resets conditioned on the outcomes of the previous steps
    assert count(gates.U1) == count(gates.RX) == nbits - 1
    for gate in iqpe.queue:
        if type(gate) in (gates.U1, gates.RX):
            assert gate.target_qubits == (0,)
            assert not gate.parameters[0].is_number

    # t measurements with collapse and a final one without it, all of the counting qubit
    measurements = [gate for gate in iqpe.queue if isinstance(gate, gates.M)]
    assert len(measurements) == nbits + 1
    assert all(gate.target_qubits == (0,) for gate in measurements)
    assert [gate.collapse for gate in measurements] == [True] * nbits + [False]
    assert len(iqpe.measurements) == 1
    # the results go from the most to the least significant bit
    assert results == [gate.result for gate in reversed(measurements[:-1])]


@pytest.mark.parametrize("numerator, nbits", [(5, 4), (1, 3), (6, 3)])
def test_iqpe_order_of_the_results(backend, numerator, nbits):
    phase = numerator / 2**nbits
    unitary = Circuit(1)
    unitary.add(gates.U1(0, theta=2 * math.pi * phase))
    iqpe, results = IQPE(unitary, nbits)

    circuit = Circuit(iqpe.nqubits)
    circuit.add(gates.X(1))
    circuit.add(iqpe)
    backend.execute_circuit(circuit, nshots=20)

    joint_result = MeasurementResult.join(*results, backend=backend)
    assert len(joint_result.frequencies(backend=backend)) == 1
    assert estimate_phase(joint_result) == phase
    # the first measurement is the one of the least significant bit
    bitstring = f"{numerator:0{nbits}b}"
    reversed_phase = int(bitstring[::-1], 2) / 2**nbits
    reversed_result = MeasurementResult.join(*reversed(results), backend=backend)
    assert estimate_phase(reversed_result) == reversed_phase


def test_iqpe_matches_qpe(backend):
    """The distribution of the outcomes is the same as in the canonical phase estimation."""
    nbits, nshots = 5, 400
    unitary = Circuit(1)
    unitary.add(gates.U1(0, theta=2 * math.pi * 0.3))

    qpe = QPE(unitary, nbits, measure=False)
    qpe_circuit = Circuit(qpe.nqubits)
    qpe_circuit.add(gates.X(nbits))
    qpe_circuit.add(qpe)
    probabilities = backend.execute_circuit(qpe_circuit).probabilities(
        qubits=list(range(nbits))
    )

    iqpe, results = IQPE(unitary, nbits)
    iqpe_circuit = Circuit(iqpe.nqubits)
    iqpe_circuit.add(gates.X(1))
    iqpe_circuit.add(iqpe)
    backend.execute_circuit(iqpe_circuit, nshots=nshots)
    joint_result = MeasurementResult.join(*results, backend=backend)
    frequencies = joint_result.frequencies(binary=False, backend=backend)

    # the tolerance is more than five standard deviations of the frequencies
    for outcome in range(2**nbits):
        frequency = frequencies.get(outcome, 0) / nshots
        assert abs(frequency - float(probabilities[outcome])) < 0.13
    assert estimate_phase(joint_result) == round(0.3 * 2**nbits) / 2**nbits


@pytest.mark.parametrize("nbits", [np.int64(3), np.uint8(3), np.int32(3)])
def test_iqpe_numpy_integer_bits(backend, nbits):
    unitary = Circuit(1)
    unitary.add(gates.U1(0, theta=0.3))

    iqpe, results = IQPE(unitary, nbits, density_matrix=True)
    target, target_results = IQPE(unitary, 3, density_matrix=True)
    assert isinstance(iqpe.nqubits, int)
    assert iqpe.density_matrix
    assert len(results) == len(target_results) == 3
    assert iqpe.ngates == target.ngates
