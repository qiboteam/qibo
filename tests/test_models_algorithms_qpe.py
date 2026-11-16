"""Tests for the Quantum Phase Estimation circuits defined in `qibo/models/algorithms/qpe.py`."""

import math

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.measurements import MeasurementResult
from qibo.models import (
    IQPE,
    QFT,
    QPE,
    hybrid_input_output,
    vectorial_phase_retrieval,
)
from qibo.quantum_info import bitstring_to_phase, estimate_phase, random_unitary


@pytest.mark.parametrize(
    "kwargs",
    [
        {"magnitudes": [1.0, 2.0]},
        {"phases": [0.0, 1.0]},
        {"beta": -0.1},
        {"beta": 1.5},
        {"beta": "0.9"},
        {"niterations": 0},
        {"niterations": 2.5},
        {"initial": [[1.0, 2.0], [3.0, 4.0]]},
    ],
)
def test_hybrid_input_output_errors(backend, kwargs):
    kwargs = {"magnitudes": backend.ones((5, 7)), "backend": backend, **kwargs}

    with pytest.raises(ValueError):
        hybrid_input_output(**kwargs)


@pytest.mark.parametrize("shape", [(15, 17), (16, 16)])
def test_hybrid_input_output_fixed_point(backend, shape):
    """The spectrum of the exact signal is a fixed point of the algorithm.

    The signal has lines at frequencies whose weights are positive, so its spectrum is real,
    and it is positive after multiplying by the triangular window.
    """
    signal, window = _signal(backend, shape)
    spectrum = backend.real(backend.fft(backend.fft(window * signal, axis=0), axis=1))

    result, estimate = hybrid_input_output(
        backend.abs(signal),
        phases=backend.angle(signal[0]),
        niterations=5,
        initial=spectrum,
        backend=backend,
    )

    assert float(backend.min(spectrum)) > 0
    backend.assert_allclose(result, spectrum, atol=1e-10)
    backend.assert_allclose(estimate, window * signal, atol=1e-10)


def test_hybrid_input_output_reproducible(backend):
    signal, _ = _signal(backend, (11, 13))
    kwargs = {"magnitudes": backend.abs(signal), "niterations": 20, "backend": backend}

    spectrum, estimate = hybrid_input_output(seed=3, **kwargs)
    spectrum_same, estimate_same = hybrid_input_output(seed=3, **kwargs)
    spectrum_other, _ = hybrid_input_output(seed=4, **kwargs)

    backend.assert_allclose(spectrum, spectrum_same)
    backend.assert_allclose(estimate, estimate_same)
    assert float(backend.max(backend.abs(spectrum - spectrum_other))) > 1e-6


def test_hybrid_input_output_signal_constraints(backend):
    """The estimate has the absolute values of the data, which include the window, and the
    known phases of the time zero."""
    signal, window = _signal(backend, (15, 17))

    spectrum, estimate = hybrid_input_output(
        backend.abs(signal),
        phases=backend.angle(signal[0]),
        niterations=50,
        seed=0,
        backend=backend,
    )

    assert spectrum.shape == signal.shape
    backend.assert_allclose(backend.abs(estimate), window * backend.abs(signal))
    backend.assert_allclose(estimate[0], window[0] * signal[0], atol=1e-12)


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


def test_qpe_without_measurement():
    unitary = Circuit(1)
    unitary.add(gates.U1(0, theta=0.3))
    qpe = QPE(unitary, 3, measure=False)
    assert qpe.measurements == []
    assert qpe.ngates == 3 + 7 + QFT(3).ngates


@pytest.mark.parametrize("orthogonal", [False, True])
@pytest.mark.parametrize(
    "nsamples, support, nreferences", [(16, 5, 1), (24, 8, 2), (40, 14, 3)]
)
def test_vectorial_phase_retrieval(backend, nsamples, support, nreferences, orthogonal):
    """If the support is exact, the phases of the signal are recovered up to a global phase.

    If the references are orthogonal to the target state, their signals vanish at the time zero,
    which makes the relative phase of this time undefined.
    """
    signal, target, references = _probabilities(
        backend, nsamples, support, nreferences, orthogonal=orthogonal
    )

    estimate, eigenvalues = vectorial_phase_retrieval(
        target, references, support, backend=backend
    )

    backend.assert_allclose(estimate, signal, atol=1e-8)
    assert abs(float(eigenvalues[0])) < 1e-10
    assert float(eigenvalues[1]) > 1e-4


def test_vectorial_phase_retrieval_eigenvalues(backend):
    """The smallest eigenvalue drops to zero at the size of the support."""
    _, target, references = _probabilities(backend, 24, 8, 2)

    smallest = [
        float(vectorial_phase_retrieval(target, references, s, backend=backend)[1][0])
        for s in (5, 6, 7, 8, 9)
    ]

    assert min(smallest[:3]) > 1e-3
    assert abs(smallest[3]) < 1e-10
    assert abs(smallest[4]) < 1e-10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target": [[1.0, 2.0]]},
        {"references": [[1.0, 2.0]]},
        {"references": [[[1.0] * 6] * 3]},
        {"references": [[[1.0] * 8] * 2]},
        {"references": []},
        {"support": -1},
        {"support": 8},
        {"support": 2.5},
        {"support": True},
    ],
)
def test_vectorial_phase_retrieval_errors(backend, kwargs):
    kwargs = {
        "target": backend.ones(8),
        "references": backend.ones((1, 3, 8)),
        "support": 3,
        "backend": backend,
        **kwargs,
    }

    with pytest.raises(ValueError):
        vectorial_phase_retrieval(**kwargs)


@pytest.mark.parametrize("nreferences", [2, 4])
def test_vectorial_phase_retrieval_noise(backend, nreferences):
    """The error is linear in the leakage of the spectra out of the support, and in the noise
    of the probabilities, which is the one of a finite number of shots."""
    nsamples, support = 32, 10

    for tail, noise in ((1e-3, 0.0), (0.0, 1e-6)):
        signal, target, references = _probabilities(
            backend, nsamples, support, nreferences, tail=tail, noise=noise
        )

        estimate, _ = vectorial_phase_retrieval(
            target, references, support, backend=backend
        )

        error = float(backend.max(backend.abs(estimate - signal)))
        assert 1e-8 < error < 1e-1 * float(backend.max(backend.abs(signal)))


def test_vectorial_phase_retrieval_undefined_phase(backend):
    """The references vanish at the time zero if they are orthogonal to the target state, which
    leaves the relative phase undefined, and the estimate does not depend on the probabilities
    that would define it."""
    signal, target, references = _probabilities(backend, 24, 8, 2, orthogonal=True)
    references[:, 0, 0] = 0

    for shift in (0.0, 0.03):
        noisy = references + 0.0
        noisy[:, 1, 0] += shift
        noisy[:, 2, 0] -= shift

        estimate, _ = vectorial_phase_retrieval(target, noisy, 8, backend=backend)

        backend.assert_allclose(estimate, signal, atol=1e-8)


def _estimated_phase(backend, circuit, nshots=100, **kwargs):
    frequencies = backend.execute_circuit(
        circuit, nshots=nshots, **kwargs
    ).frequencies()
    assert len(frequencies) == 1
    return bitstring_to_phase(next(iter(frequencies)))


def _probabilities(
    backend, nsamples, support, nreferences, noise=0.0, orthogonal=False, tail=0.0
):
    # probabilities that the circuits measure, which are the squared absolute values of
    # f1, f2, f1 + f2 and f1 + i f2, where the last two are divided by 2 as their states are
    # normalized, and the signals are the exact ones if there is no noise
    # the signals have spectra that are supported on the first frequencies, with a leakage out of
    # them if tail is not zero, and the global phase that makes the first value of the target
    # real. The value of a signal at the time zero is the mean of its spectrum, so the references
    # vanish there if they are orthogonal to the target state
    generator = backend.default_rng(nsamples)
    shape = (nreferences + 1, nsamples)
    spectra = generator.normal(size=shape) + 1j * generator.normal(size=shape)
    spectra[:, support:] *= tail
    if orthogonal:
        spectra[1:, :support] -= spectra[1:, :support].mean(axis=1, keepdims=True)
    signals = backend.ifft(backend.cast(spectra, dtype=backend.complex128), axis=1)
    signals = signals * backend.conj(signals[0, 0]) / backend.abs(signals[0, 0])
    target, references = signals[0], signals[1:]
    probabilities = backend.cast(
        [
            [
                backend.abs(reference) ** 2,
                backend.abs(target + reference) ** 2 / 2,
                backend.abs(target + 1j * reference) ** 2 / 2,
            ]
            for reference in references
        ],
        dtype=backend.float64,
    )
    generator = backend.default_rng(1)
    target_probabilities = backend.abs(target) ** 2
    if noise > 0:
        target_probabilities += noise * backend.cast(
            generator.normal(size=nsamples), dtype=backend.float64
        )
        probabilities += noise * backend.cast(
            generator.normal(size=probabilities.shape), dtype=backend.float64
        )

    return (
        target,
        backend.abs(target_probabilities),
        backend.abs(probabilities),
    )


def _signal(backend, shape):
    # signal with three lines, whose frequencies are the energies of H and of H_D, and the
    # triangular window of the algorithm, with the signed indices of the times
    nrows, ncols = shape
    rows = [j if j <= nrows - j else j - nrows for j in range(nrows)]
    cols = [m if m <= ncols - m else m - ncols for m in range(ncols)]
    times_rows = backend.cast(rows, dtype=backend.float64)
    times_cols = backend.cast(cols, dtype=backend.float64)
    signal = sum(
        weight
        * backend.exp(1j * energy * times_rows)[:, None]
        * backend.exp(1j * charge * times_cols)[None, :]
        for weight, energy, charge in zip(
            (0.5, 0.3, 0.2), (0.5, 1.0, 1.6), (0.2, 0.8, 1.4)
        )
    )
    window = backend.cast(
        [
            [(1 - 2 * abs(j) / nrows) * (1 - 2 * abs(m) / ncols) for m in cols]
            for j in rows
        ],
        dtype=backend.float64,
    )

    return signal, window
