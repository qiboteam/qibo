"""Tests for :class:`qibo.tomography.classical_shadows.ClassicalShadow`."""

from functools import reduce

import numpy as np
import pytest
from sympy import S

from qibo import Circuit, gates, hamiltonians, matrices
from qibo.models.encodings import entangling_layer
from qibo.noise import DepolarizingError, NoiseModel
from qibo.quantum_info import random_density_matrix, vectorization
from qibo.symbols import X, Y, Z
from qibo.tomography import ClassicalShadow, Tomography
from qibo.tomography.classical_shadows import BASES, SINGLE_QUBIT_CLIFFORDS

PAULIS = {
    "I": matrices.I,
    "X": matrices.X,
    "Y": matrices.Y,
    "Z": matrices.Z,
}

SYMBOLS = {"X": X, "Y": Y, "Z": Z}


@pytest.mark.parametrize(
    "method,nqubits",
    [
        ("local-clifford", 2),
        ("global-clifford", 1),
        ("global-clifford", 2),
        ("ultra-shallow", 2),
    ],
)
def test_call_basis_agreement(backend, method, nqubits):
    """Both bases and all observable types share snapshots and frame."""
    circuit = _prepare(nqubits)
    terms = _terms(nqubits)
    matrix = _matrix(terms, backend)

    merged = {}
    for string, coefficient in terms:
        merged[string] = merged.get(string, 0.0) + coefficient

    shadow = ClassicalShadow()
    kwargs = {
        "nsamples": 40,
        "method": method,
        "nbatches": 2,
        "chunksize": 25,
        "depth": 1,
        "ncalibration": 150,
        "seed": 7,
        "backend": backend,
    }
    estimate = shadow(circuit, terms, **kwargs)

    backend.assert_allclose(shadow(circuit, merged, **kwargs), estimate, atol=1e-10)

    observables = [
        matrix,
        hamiltonians.Hamiltonian(nqubits, matrix, backend=backend),
        hamiltonians.SymbolicHamiltonian(
            sum(
                coefficient
                * reduce(
                    lambda left, right: left * right,
                    [
                        SYMBOLS[char](qubit)
                        for qubit, char in enumerate(string)
                        if char != "I"
                    ],
                    S(1),
                )
                for string, coefficient in terms
            ),
            nqubits=nqubits,
            backend=backend,
        ),
    ]
    for observable in observables:
        computational = shadow(circuit, observable, basis="computational", **kwargs)
        backend.assert_allclose(computational, estimate, atol=1e-8)


def test_call_chunks(backend, monkeypatch):
    """Chunks never cross batches, and batches are almost equally sized."""
    sizes = []
    circuits = ClassicalShadow.circuits

    def spy(self, circuit, nsamples, *args, **kwargs):
        sizes.append(nsamples)
        return circuits(self, circuit, nsamples, *args, **kwargs)

    monkeypatch.setattr(ClassicalShadow, "circuits", spy)

    circuit = Circuit(1)
    circuit.add(gates.H(0))

    shadow = ClassicalShadow()
    kwargs = {"method": "local-clifford", "backend": backend}

    shadow(circuit, [("Z", 1.0)], 10, nbatches=2, chunksize=3, **kwargs)
    assert sizes == [3, 2, 3, 2]

    sizes.clear()
    shadow(circuit, [("Z", 1.0)], 7, nbatches=3, chunksize=100, **kwargs)
    assert sizes == [3, 2, 2]


def test_call_errors(backend):
    circuit = _prepare(2)
    shadow = ClassicalShadow()
    kwargs = {"method": "local-clifford", "backend": backend}

    with pytest.raises(ValueError):
        shadow(circuit, [("ZZ", 1.0)], 10, basis="xyz", **kwargs)

    for nbatches in (0, 11, 2.5):
        with pytest.raises(ValueError):
            shadow(circuit, [("ZZ", 1.0)], 10, nbatches=nbatches, **kwargs)

    for chunksize in (0, -1, 2.5):
        with pytest.raises(ValueError):
            shadow(circuit, [("ZZ", 1.0)], 10, chunksize=chunksize, **kwargs)

    for observable in ([("ZZZ", 1.0)], {"ZA": 1.0}, [("Z", 1.0)]):
        with pytest.raises(ValueError):
            shadow(circuit, observable, 10, **kwargs)

    with pytest.raises(ValueError):
        shadow(circuit, [("ZZ", 1.0)], 10, method="xyz", backend=backend)

    with pytest.raises(ValueError):
        shadow(circuit, [("ZZ", 1.0)], 10, depth=-1, **kwargs)

    measured = _prepare(2)
    measured.add(gates.M(0, 1))
    with pytest.raises(ValueError):
        shadow(measured, [("ZZ", 1.0)], 10, **kwargs)


@pytest.mark.parametrize(
    "method,nqubits,nsamples,ncalibration,nbatches,atol",
    [
        ("local-clifford", 1, 1500, None, 1, 0.2),
        ("local-clifford", 2, 800, None, 1, 0.4),
        ("local-clifford", 2, 800, None, 4, 0.7),
        ("global-clifford", 1, 1500, None, 1, 0.2),
        ("global-clifford", 2, 600, None, 1, 0.3),
        ("ultra-shallow", 2, 600, 2000, 1, 0.5),
    ],
)
def test_call_estimates(
    backend, method, nqubits, nsamples, ncalibration, nbatches, atol
):
    """Estimates are consistent with the exact expectation values."""
    circuit = _prepare(nqubits)
    terms = _terms(nqubits)

    state = backend.execute_circuit(circuit).state()
    exact = backend.real(backend.dagger(state) @ _matrix(terms, backend) @ state)

    estimate = ClassicalShadow()(
        circuit,
        terms,
        nsamples,
        method,
        nbatches=nbatches,
        depth=1,
        ncalibration=ncalibration,
        seed=11,
        backend=backend,
    )

    backend.assert_allclose(estimate, exact, atol=atol)


def test_call_ignored_arguments(backend):
    """``depth`` and ``ncalibration`` only matter for ``"ultra-shallow"``."""
    circuit = _prepare(2)
    shadow = ClassicalShadow()
    kwargs = {"seed": 3, "backend": backend}

    default = shadow(circuit, [("ZZ", 1.0)], 30, "local-clifford", **kwargs)
    modified = shadow(
        circuit,
        [("ZZ", 1.0)],
        30,
        "local-clifford",
        depth=5,
        ncalibration=3,
        **kwargs,
    )

    backend.assert_allclose(default, modified)


@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize(
    "estimates,median",
    [([3.0], 3.0), ([3.0, 1.0, 2.0], 2.0), ([4.0, 1.0, 2.0, 3.0], 2.5)],
)
def test_call_median_of_means(backend, monkeypatch, basis, estimates, median):
    """The result is the median of the estimates of each batch."""
    name = "_expectation_pauli" if basis == "pauli" else "_expectation_computational"
    monkeypatch.setattr(
        ClassicalShadow,
        name,
        lambda self, *args: backend.cast(estimates, dtype=backend.float64),
    )

    result = ClassicalShadow()(
        Circuit(1),
        None,
        10,
        basis=basis,
        nbatches=len(estimates),
        backend=backend,
    )

    backend.assert_allclose(result, median)


def test_call_reproducibility(backend):
    circuit = _prepare(2)
    shadow = ClassicalShadow()

    def estimate(seed):
        return shadow(
            circuit,
            [("ZZ", 1.0), ("XX", 1.0)],
            30,
            "local-clifford",
            nshots=2,
            seed=seed,
            backend=backend,
        )

    backend.assert_allclose(estimate(5), estimate(5))
    assert float(estimate(5)) != float(estimate(6))


@pytest.mark.parametrize(
    "method,nqubits,depth",
    [
        ("local-clifford", 3, 1),
        ("global-clifford", 1, 1),
        ("global-clifford", 2, 1),
        ("ultra-shallow", 4, 2),
        ("ultra-shallow", 3, 0),
        ("ultra-shallow", 1, 3),
    ],
)
def test_circuits(backend, method, nqubits, depth):
    circuit = Circuit(nqubits)
    circuit.add(gates.H(0))

    shadow = ClassicalShadow()
    shadow_circuits = shadow.circuits(
        circuit, 4, method, depth, seed=3, backend=backend
    )

    assert len(shadow_circuits) == 4
    assert len(circuit.queue) == 1

    ncnots = len(entangling_layer(nqubits, "shifted").queue) * depth
    for shadow_circuit in shadow_circuits:
        assert shadow_circuit.nqubits == nqubits
        # the state preparation is untouched and the measurement is last
        assert shadow_circuit.queue[0] is circuit.queue[0]
        assert isinstance(shadow_circuit.queue[-1], gates.M)
        assert shadow_circuit.queue[-1].qubits == tuple(range(nqubits))

        added = shadow_circuit.queue[1:-1]
        assert all(gate.clifford for gate in added)
        if method != "global-clifford":
            assert all(
                isinstance(gate, (gates.H, gates.S, gates.CNOT)) for gate in added
            )
            assert sum(isinstance(gate, gates.CNOT) for gate in added) == (
                ncnots if method == "ultra-shallow" else 0
            )

    def description(seed):
        return [
            [(type(gate).__name__, gate.qubits) for gate in shadow_circuit.queue]
            for shadow_circuit in shadow.circuits(
                circuit, 4, method, depth, seed=seed, backend=backend
            )
        ]

    assert description(3) == description(3)
    assert description(3) != description(4)


def test_circuits_depth(backend):
    """Layers of ``"ultra-shallow"``: 4 qubits have 3 CNOTs per entangling layer."""
    shadow = ClassicalShadow()

    for depth in range(4):
        shadow_circuit = shadow.circuits(
            Circuit(4), 1, "ultra-shallow", depth, seed=1, backend=backend
        )[0]
        cnots = [gate.qubits for gate in shadow_circuit.queue if gate.name == "cx"]
        assert cnots == [(0, 1), (2, 3), (1, 2)] * depth

    # no entangling layers is equivalent to local Cliffords
    def description(method):
        return [
            (type(gate).__name__, gate.qubits)
            for gate in shadow.circuits(
                Circuit(3), 2, method, 0, seed=9, backend=backend
            )[0].queue
        ]

    assert description("ultra-shallow") == description("local-clifford")


def test_circuits_errors(backend):
    shadow = ClassicalShadow()

    with pytest.raises(ValueError):
        shadow.circuits(Circuit(2), 1, "xyz", backend=backend)

    for depth in (-1, 1.5):
        with pytest.raises(ValueError):
            shadow.circuits(Circuit(2), 1, "ultra-shallow", depth, backend=backend)

    measured = Circuit(2)
    measured.add(gates.M(0))
    with pytest.raises(ValueError):
        shadow.circuits(measured, 1, "local-clifford", backend=backend)


@pytest.mark.parametrize("method", ["global-clifford", "local-clifford"])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_frame_operator_computational(backend, method, nqubits):
    """The frame is diagonal in the Pauli basis, with the same eigenvalues."""
    shadow = ClassicalShadow()
    dim = 2**nqubits

    pauli = backend.to_numpy(shadow.frame_operator(nqubits, method, backend=backend))
    frame = backend.to_numpy(
        shadow.frame_operator(nqubits, method, basis="computational", backend=backend)
    )
    inverse = backend.to_numpy(
        shadow.frame_operator(
            nqubits, method, True, basis="computational", backend=backend
        )
    )

    assert frame.shape == (4**nqubits, 4**nqubits)
    backend.assert_allclose(frame @ inverse, backend.identity(4**nqubits), atol=1e-10)
    backend.assert_allclose(backend.eigvalsh(frame), backend.sort(pauli), atol=1e-10)

    # Pauli strings are eigenvectors
    for index in range(4**nqubits):
        string = "".join(
            "IXYZ"[int(digit)] for digit in backend.base_repr(index, 4).zfill(nqubits)
        )
        vector = vectorization(
            _pauli_matrix(string, backend), order="row", backend=backend
        )
        backend.assert_allclose(frame @ vector, pauli[index] * vector, atol=1e-10)

    if method == "global-clifford":
        # the channel is (rho + tr(rho) I) / (d + 1)
        rho = random_density_matrix(dim, backend=backend)
        rho = rho @ rho.T
        vector = vectorization(rho, order="row", backend=backend)
        identity = backend.reshape(backend.identity(dim), -1)
        backend.assert_allclose(
            frame @ vector,
            (vector + backend.trace(rho) * identity) / (dim + 1),
            atol=1e-10,
        )


def test_frame_operator_computational_ultra_shallow(backend):
    shadow = ClassicalShadow()
    kwargs = {"depth": 1, "nsamples": 100, "seed": 2, "backend": backend}

    pauli = shadow.frame_operator(2, "ultra-shallow", **kwargs)
    frame = shadow.frame_operator(2, "ultra-shallow", basis="computational", **kwargs)

    backend.assert_allclose(backend.eigvalsh(frame), backend.sort(pauli), atol=1e-10)


def test_frame_operator_errors(backend):
    shadow = ClassicalShadow()

    with pytest.raises(ValueError):
        shadow.frame_operator(2, "xyz", backend=backend)

    with pytest.raises(ValueError):
        shadow.frame_operator(2, "local-clifford", basis="xyz", backend=backend)

    # too few samples give non-positive eigenvalues, which cannot be inverted
    with pytest.raises(ValueError):
        shadow.frame_operator(
            3, "ultra-shallow", True, depth=1, nsamples=20, seed=0, backend=backend
        )


def test_frame_operator_noisy_calibration(backend, monkeypatch):
    """Noise in the calibration circuits is included in the sampled frame."""
    shadow = ClassicalShadow()
    kwargs = {"depth": 1, "nsamples": 800, "seed": 4, "backend": backend}

    ideal = backend.to_numpy(shadow.frame_operator(2, "ultra-shallow", **kwargs))

    noise = NoiseModel()
    noise.add(DepolarizingError(0.3), gates.CNOT)
    execute = Tomography.execute
    monkeypatch.setattr(
        Tomography,
        "execute",
        lambda self, circuits, nshots=1000, backend=None: execute(
            self, [noise.apply(circuit) for circuit in circuits], nshots, backend
        ),
    )
    noisy = shadow.frame_operator(2, "ultra-shallow", **kwargs)

    backend.assert_allclose(noisy[0], 1.0)
    assert backend.sum(noisy[1:]) < 0.9 * backend.sum(ideal[1:])


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("nqubits", [1, 2, 3])
@pytest.mark.parametrize("method", ["global-clifford", "local-clifford"])
def test_frame_operator_pauli(backend, method, nqubits, inverse):
    weights = _weights(nqubits, backend)

    if method == "global-clifford":
        expected = backend.where(weights == 0, 1.0, 1 / (2**nqubits + 1))
    else:
        expected = 3.0 ** (-weights)
    if inverse:
        expected = 1 / expected

    frame = ClassicalShadow().frame_operator(nqubits, method, inverse, backend=backend)

    assert frame.dtype == backend.float64
    assert frame.shape == (4**nqubits,)
    backend.assert_allclose(frame, expected)


def test_frame_operator_ultra_shallow(backend):
    """The sampled frame has the structure and values of the ensemble."""
    shadow = ClassicalShadow()
    kwargs = {"nsamples": 1500, "seed": 1, "backend": backend}

    frame = backend.to_numpy(
        shadow.frame_operator(2, "ultra-shallow", depth=1, **kwargs)
    )
    inverse = backend.to_numpy(
        shadow.frame_operator(2, "ultra-shallow", True, depth=1, **kwargs)
    )

    assert frame.dtype == backend.float64
    assert frame.shape == (16,)
    backend.assert_allclose(frame[0], 1.0)
    backend.assert_allclose(frame * inverse, 1.0)
    assert backend.all(frame > 0)

    # the eigenvalue only depends on the support of the Pauli string
    supports = {}
    for index in range(16):
        digits = backend.base_repr(index, 4).zfill(2)
        supports.setdefault(tuple(int(digit != "0") for digit in digits), []).append(
            frame[index]
        )
    for values in supports.values():
        backend.assert_allclose(values, values[0])

    # exact values for two qubits and one entangling layer, by enumeration of the ensemble
    for support, exact in {(0, 1): 5 / 27, (1, 0): 5 / 27, (1, 1): 17 / 81}.items():
        backend.assert_allclose(supports[support][0], exact, atol=0.07)

    # without entangling layers it is the local Clifford frame
    local = shadow.frame_operator(2, "ultra-shallow", depth=0, **kwargs)

    backend.assert_allclose(local, 3.0 ** (-_weights(2, backend)), atol=0.07)


def test_single_qubit_cliffords(backend):
    assert len(SINGLE_QUBIT_CLIFFORDS) == 24

    unitaries = []
    for word in SINGLE_QUBIT_CLIFFORDS:
        circuit = Circuit(1)
        for name in word:
            circuit.add(getattr(gates, name)(0))
        unitaries.append(circuit.unitary(backend) if word else backend.identity(2))

    def key(unitary):
        flat = unitary.reshape(-1)
        phase = flat[backend.argmax(backend.abs(flat) > 1e-9)]
        return tuple(backend.round(unitary / (phase / abs(phase)), 6).reshape(-1))

    # 24 different elements up to a global phase
    keys = {key(unitary) for unitary in unitaries}
    assert len(keys) == 24

    # closed under multiplication, i.e. it is the whole group
    for left in unitaries:
        for right in unitaries:
            assert key(left @ right) in keys

    # and each element maps Pauli matrices to Pauli matrices
    for unitary in unitaries:
        for char in "XYZ":
            rotated = unitary @ PAULIS[char] @ backend.dagger(unitary)
            assert any(
                backend.allclose(rotated, sign * PAULIS[other], atol=1e-10)
                for other in "XYZ"
                for sign in (1, -1)
            )


@pytest.mark.parametrize(
    "method,nqubits",
    [("local-clifford", 2), ("ultra-shallow", 2), ("global-clifford", 1)],
)
def test_snapshots(backend, method, nqubits):
    circuit = _prepare(nqubits)
    shadow = ClassicalShadow()

    snapshots = list(shadow._snapshots(circuit, 6, method, 2, 3, 4, 1, 5, backend))

    assert len(snapshots) == 3
    for snapshot in snapshots:
        assert snapshot.shape == (2**nqubits, 2**nqubits)
        backend.assert_allclose(snapshot, backend.dagger(snapshot), atol=1e-10)
        backend.assert_allclose(backend.trace(snapshot), 1.0, atol=1e-10)
        assert backend.all(backend.eigvalsh(snapshot) > -1e-10)

    # a single sample is the projector U^dagger |b><b| U
    (snapshot,) = shadow._snapshots(circuit, 1, method, 1, 1, 1, 1, 5, backend)
    backend.assert_allclose(snapshot @ snapshot, snapshot, atol=1e-10)


def _matrix(terms, backend):
    return sum(
        coefficient * _pauli_matrix(string, backend) for string, coefficient in terms
    )


def _pauli_matrix(string, backend):
    return reduce(backend.kron, [PAULIS[char] for char in string])


def _prepare(nqubits):
    """Hadamard on the first qubit followed by a chain of CNOTs."""
    circuit = Circuit(nqubits)
    circuit.add(gates.H(0))
    for qubit in range(nqubits - 1):
        circuit.add(gates.CNOT(qubit, qubit + 1))
    return circuit


def _terms(nqubits):
    """Pauli strings with a repeated one, to check that coefficients are added."""
    first = "Z" * nqubits
    return [
        (first, 0.7),
        ("X" * nqubits, -0.3),
        ("Y" + "Z" * (nqubits - 1), 0.5),
        ("Z" + "I" * (nqubits - 1), 0.2),
        (first, 0.1),
        ("I" * nqubits, 0.5),
    ]


def _weights(nqubits, backend):
    """Number of non-identity factors of each Pauli string, in Pauli-basis order."""
    return np.array(
        [
            sum(digit != "0" for digit in backend.base_repr(index, 4).zfill(nqubits))
            for index in range(4**nqubits)
        ]
    )
