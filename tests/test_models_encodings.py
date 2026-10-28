"""Tests for qibo.models.encodings"""

import logging
import math
from functools import reduce
from itertools import combinations, product

import numpy as np
import pytest
from scipy.optimize import curve_fit
from scipy.special import binom

from qibo import Circuit, gates
from qibo.hamiltonians import FoldedXXZ
from qibo.models._encodings import (
    _add_wbd_gate,
    _ehrlich_algorithm,
    _ehrlich_codewords_up_to_k,
    _folded_merged,
    _folded_xxz_cswap,
    _get_int_type,
    _walsh_gray_code,
)
from qibo.models.encodings import (
    binary_encoder,
    comp_basis_encoder,
    dicke_state,
    entangling_layer,
    fanout_synthesis,
    folded_xxz_eigenstate,
    ghz_state,
    graph_state,
    hamming_weight_encoder,
    ladder_synthesis,
    leaf_separable_encoder,
    permutation_synthesis,
    phase_encoder,
    sparse_encoder,
    spin_s_encoder,
    unary_encoder,
    unary_encoder_random_gaussian,
    up_to_k_hamming_weight_encoder,
)
from qibo.quantum_info.metrics import fidelity
from qibo.quantum_info.random_ensembles import random_statevector
from qibo.transpiler import InverseCancellation


def _gaussian(x, a, b, c):
    """Gaussian used in the `unary_encoder_random_gaussian test"""
    return np.exp(a * x**2 + b * x + c)


@pytest.mark.parametrize(
    "basis_element", [5, "101", ["1", "0", "1"], [1, 0, 1], ("1", "0", "1"), (1, 0, 1)]
)
def test_comp_basis_encoder(backend, basis_element):
    with pytest.raises(TypeError):
        comp_basis_encoder(2.3)
    with pytest.raises(ValueError):
        comp_basis_encoder("0b001")
    with pytest.raises(ValueError):
        comp_basis_encoder("001", nqubits=2)
    with pytest.raises(TypeError):
        comp_basis_encoder("001", nqubits=3.1)
    with pytest.raises(ValueError):
        comp_basis_encoder(3)

    zero = np.array([1, 0], dtype=complex)
    one = np.array([0, 1], dtype=complex)
    target = np.kron(one, np.kron(zero, one))
    target = backend.cast(target, dtype=target.dtype)

    state = (
        comp_basis_encoder(basis_element, nqubits=3)
        if isinstance(basis_element, int)
        else comp_basis_encoder(basis_element)
    )

    state = backend.execute_circuit(state).state()

    backend.assert_allclose(state, target)


@pytest.mark.parametrize("kind", [None, list])
@pytest.mark.parametrize("rotation", ["RX", "RY", "RZ"])
def test_phase_encoder(backend, rotation, kind):
    sampler = np.random.default_rng(1)

    nqubits = 3

    with pytest.raises(TypeError):
        data = sampler.random(nqubits)
        data = backend.cast(data, dtype=data.dtype)
        phase_encoder(nqubits, data=data, rotation=True)
    with pytest.raises(ValueError):
        data = sampler.random(nqubits)
        data = backend.cast(data, dtype=data.dtype)
        phase_encoder(nqubits, data=data, rotation="rzz")

    phases = backend.random_sample(nqubits)

    gate = getattr(gates, rotation)
    target = reduce(
        backend.kron,
        [gate(qubit, phase).matrix(backend) for qubit, phase in enumerate(phases)],
    )[:, 0]

    if kind is not None:
        phases = kind(phases)

    state = phase_encoder(nqubits, data=phases, rotation=rotation, backend=backend)
    state.draw()
    state = backend.execute_circuit(state).state()

    backend.assert_allclose(state, target)


@pytest.mark.parametrize("keep_antictrls", [False, True])
@pytest.mark.parametrize("custom_codewords", [False, True])
@pytest.mark.parametrize("not_power_of_two", [False, True])
@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("data", [False, True])
@pytest.mark.parametrize("parametrization", ["hopf", "hyperspherical", "mottonen"])
@pytest.mark.parametrize("nqubits", [3, 4, 5])
def test_binary_encoder(
    backend,
    nqubits,
    parametrization,
    data,
    complex_data,
    not_power_of_two,
    custom_codewords,
    keep_antictrls,
):
    if complex_data:
        parametrization += "-complex"

    if parametrization in ("hopf", "hopf-complex"):
        if complex_data:
            with pytest.raises(NotImplementedError):
                test = binary_encoder(nqubits, "hopf-complex")

            pytest.skip(
                "``binary_encoder`` in Hopf coordinates not implemented for complex data."
            )
        else:
            with pytest.raises(ValueError):
                test = backend.random_sample(5)
                test = binary_encoder(
                    data=test,
                    parametrization=parametrization,
                    nqubits=nqubits,
                    backend=backend,
                )

    dims = (
        2**nqubits - 2
        if parametrization == "hyperspherical" and not_power_of_two
        else 2**nqubits
    )

    if parametrization in ("mottonen", "mottonen-complex") and not_power_of_two:
        with pytest.raises(ValueError):
            test = binary_encoder(
                nqubits=nqubits,
                data=backend.random_sample(5),
                parametrization=parametrization,
                backend=backend,
            )
        pytest.skip(
            "``mottonen`` parametrization requires data length to be a power of 2."
        )

    if parametrization in ("mottonen", "mottonen-complex") and (
        custom_codewords or keep_antictrls
    ):
        pytest.skip(
            "``codewords`` and ``keep_antictrls`` are not used by the ``mottonen`` parametrization."
        )

    if data:
        target = random_statevector(
            dims,
            dtype=backend.complex128 if complex_data else backend.float64,
            seed=10,
            backend=backend,
        )
    else:
        target = None

    codewords = backend.arange(dims) if custom_codewords else None

    circuit = binary_encoder(
        nqubits,
        parametrization=parametrization,
        data=target,
        backend=backend,
        codewords=codewords,
        keep_antictrls=keep_antictrls,
    )
    state = backend.execute_circuit(circuit).state()

    if not data:
        target = backend.zero_state(nqubits)

    if parametrization == "hyperspherical" and not_power_of_two:
        # need to insert zeros at the end of target to get
        # matching shapes
        trail_zeros = backend.zeros(
            2 ** math.ceil(math.log2(dims)) - dims, dtype=target.dtype
        )
        target = backend.concatenate((target, trail_zeros))

    backend.assert_allclose(state, target[: len(state)], atol=1e-10, rtol=1e-4)


@pytest.mark.parametrize("seed", [1])
@pytest.mark.parametrize("architecture", ["tree", "diagonal"])
@pytest.mark.parametrize("nqubits", [8])
def test_unary_encoder(backend, nqubits, architecture, seed):
    with pytest.raises(TypeError):
        data = backend.random_sample(nqubits, seed=seed)
        unary_encoder(nqubits, data=data, architecture=True)
    with pytest.raises(ValueError):
        data = backend.random_sample(nqubits, seed=seed)
        unary_encoder(nqubits, data=data, architecture="semi-diagonal")
    if architecture == "tree":
        with pytest.raises(ValueError):
            data = backend.random_sample(nqubits + 1, seed=seed)
            unary_encoder(nqubits, data=data, architecture=architecture)

    # sampling random data in interval [-1, 1]
    data = 2 * backend.random_sample(nqubits) - 1

    circuit = unary_encoder(
        nqubits, data=data, architecture=architecture, backend=backend
    )
    state = backend.execute_circuit(circuit).state()
    indexes = backend.flatnonzero(state)
    state = backend.real(state[indexes])

    backend.assert_allclose(
        state,
        backend.cast(data, dtype=backend.float64) / backend.vector_norm(data, 2),
        rtol=1e-5,
    )


@pytest.mark.parametrize("seed", [None, 10, np.random.default_rng(10)])
@pytest.mark.parametrize("nqubits", [8])
def test_unary_encoder_random_gaussian(backend, nqubits, seed):
    """Tests if encoded vector are random variables sampled from
    Gaussian distribution with 0.0 mean and variance close to the norm
    of the random Gaussian vector that was encoded."""
    with pytest.raises(TypeError):
        unary_encoder_random_gaussian("1", seed=seed)
    with pytest.raises(ValueError):
        unary_encoder_random_gaussian(-1, seed=seed)
    with pytest.raises(ValueError):
        unary_encoder_random_gaussian(3, seed=seed)
    with pytest.raises(TypeError):
        unary_encoder_random_gaussian(nqubits, architecture=True, seed=seed)
    with pytest.raises(NotImplementedError):
        unary_encoder_random_gaussian(nqubits, architecture="diagonal", seed=seed)
    with pytest.raises(TypeError):
        unary_encoder_random_gaussian(nqubits, seed="seed")

    samples = int(1e2)

    local_state = np.random.default_rng(seed) if seed in [None, 10] else seed

    amplitudes = []
    for _ in range(samples):
        circuit = unary_encoder_random_gaussian(
            nqubits, seed=local_state, backend=backend
        )
        state = backend.execute_circuit(circuit).state()
        indexes = np.flatnonzero(backend.to_numpy(state))
        state = np.real(state[indexes])
        amplitudes += [float(elem) for elem in list(state)]

    y, x = np.histogram(amplitudes, bins=50, density=True)
    x = (x[:-1] + x[1:]) / 2

    params, _ = curve_fit(_gaussian, x, y)

    stddev = np.sqrt(-1 / (2 * params[0]))
    mean = stddev**2 * params[1]

    theoretical_norm = (
        math.sqrt(2) * math.gamma((nqubits + 1) / 2) / math.gamma(nqubits / 2)
    )
    theoretical_norm = 1.0 / theoretical_norm

    backend.assert_allclose(0.0, mean, atol=1e-1)
    backend.assert_allclose(stddev, theoretical_norm, atol=1e-1)


@pytest.mark.parametrize("seed", [10])
@pytest.mark.parametrize("optimize_controls", [False, True])
@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("full_hwp", [False, True])
@pytest.mark.parametrize("data", [False, True])
@pytest.mark.parametrize("weight", [1, 2, 3])
@pytest.mark.parametrize("nqubits", [4, 5, 6])
def test_hamming_weight_encoder(
    backend,
    nqubits,
    weight,
    data,
    full_hwp,
    complex_data,
    optimize_controls,
    seed,
):
    n_choose_k = int(binom(nqubits, weight))
    dims = 2**nqubits
    dtype = backend.complex128 if complex_data else backend.float64

    initial_string = np.array([1] * weight + [0] * (nqubits - weight))
    indices = _ehrlich_algorithm(initial_string, False)
    indices = [int(string, 2) for string in indices]
    indices_lex = np.sort(np.copy(indices))

    if data:
        _data = random_statevector(n_choose_k, dtype=dtype, seed=seed, backend=backend)
        target = backend.zeros(dims, dtype=dtype)
        target[indices_lex] = _data
    else:
        _data = None
        target = comp_basis_encoder([0] * (nqubits - weight) + [1] * weight)
        target = backend.execute_circuit(target).state()

    circuit = hamming_weight_encoder(
        nqubits=nqubits,
        weight=weight,
        data=_data,
        full_hwp=full_hwp,
        optimize_controls=optimize_controls,
        backend=backend,
    )
    if full_hwp:
        circuit.queue = [
            gates.X(nqubits - 1 - qubit) for qubit in range(weight)
        ] + circuit.queue
    state = backend.execute_circuit(circuit).state()

    backend.assert_allclose(state, target, atol=1e-7)


@pytest.mark.parametrize("keep_antictrls", [False, True])
@pytest.mark.parametrize("custom_codewords", [False, True])
@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("data", [False, True])
@pytest.mark.parametrize("up_to_k", [1, 2, 7])
@pytest.mark.parametrize("nqubits", [4, 5, 6])
def test_up_to_k_hamming_weight_encoder(
    backend,
    nqubits,
    up_to_k,
    data,
    complex_data,
    custom_codewords,
    keep_antictrls,
):

    seed = 10
    dim = int(sum(binom(nqubits, weight) for weight in range(up_to_k + 1)))
    dtype = backend.complex128 if complex_data else backend.float64

    codewords = backend.arange(dim) if custom_codewords else None

    with pytest.raises(ValueError):
        _data = random_statevector(dim + 10, dtype=dtype, seed=seed, backend=backend)
        _ = up_to_k_hamming_weight_encoder(
            data=_data,
            nqubits=nqubits,
            up_to_k=up_to_k,
            codewords=codewords,
            keep_antictrls=keep_antictrls,
            backend=backend,
        )

    if data:
        _data = random_statevector(dim, dtype=dtype, seed=seed, backend=backend)
    else:
        _data = None

    if up_to_k > nqubits:
        with pytest.raises(ValueError):
            _ = list(_ehrlich_codewords_up_to_k(up_to_k, False, nqubits, backend))
    else:
        indices = _ehrlich_codewords_up_to_k(up_to_k, False, nqubits, backend)
        indices = [int(string, 2) for string in indices]
        indices_lex = np.sort(np.copy(indices))

        if data:
            target = backend.zeros(2**nqubits, dtype=dtype)
            if codewords is None:
                target[indices_lex] = _data
            else:
                target[codewords] = _data
        else:
            target = backend.zero_state(nqubits, dtype=dtype)

        circuit = up_to_k_hamming_weight_encoder(
            data=_data,
            nqubits=nqubits,
            up_to_k=up_to_k,
            codewords=codewords,
            keep_antictrls=keep_antictrls,
            backend=backend,
        )
        state = backend.execute_circuit(circuit).state()

        backend.assert_allclose(state, target, atol=1e-7)


@pytest.mark.parametrize("seed", [10])
@pytest.mark.parametrize("zip_input", [False, True])
@pytest.mark.parametrize("integers", [False, True])
@pytest.mark.parametrize("nqubits", [4, 7])
@pytest.mark.parametrize("method", ["li", "farias"])
def test_sparse_encoder(backend, method, nqubits, integers, zip_input, seed):
    dims = 2**nqubits
    sparsity = nqubits

    data = random_statevector(sparsity, seed=seed, backend=backend)
    np.random.seed(seed)
    indices = np.random.choice(range(dims), size=sparsity, replace=False)
    indices = backend.cast(indices, dtype=int)

    target = backend.cast(backend.zeros(dims))
    target[indices] = data

    if not integers:
        indices = [f"{elem:0{nqubits}b}" for elem in indices]

    data = zip(indices, data)
    if not zip_input:
        data = list(data)

    if integers and not zip_input:
        with pytest.raises(ValueError):
            circuit = sparse_encoder(data, method, nqubits=None, backend=backend)

        with pytest.raises(ValueError):
            circuit = sparse_encoder(
                data, method=True, nqubits=nqubits, backend=backend
            )

    _nqubits = nqubits if integers else None
    circuit = sparse_encoder(data, method, _nqubits, backend=backend)
    state = backend.execute_circuit(circuit).state()

    backend.assert_allclose(state, target)


def test_sparse_encoder_helpers_errors(backend):
    with pytest.raises(ValueError):
        _get_int_type(backend.engine.iinfo(backend.int64).max + 1, backend=backend)


@pytest.mark.parametrize("levels", [1, 2, 3, 4])
@pytest.mark.parametrize("nqudits", [1, 2, 3, 4])
def test_walsh_gray_code(backend, nqudits, levels):
    for weight in range(levels * nqudits + 1):
        compositions = _walsh_gray_code(nqudits, weight, levels, backend=backend)
        compositions = backend.to_numpy(compositions)
        target = {
            string
            for string in product(range(levels + 1), repeat=nqudits)
            if sum(string) == weight
        }
        assert len(compositions) == len(target)
        assert {
            tuple(int(part) for part in string) for string in compositions
        } == target
        # Gray property: consecutive compositions differ by e_i - e_j
        distances = np.sum(np.abs(np.diff(compositions, axis=0)), axis=1)
        assert np.all(distances == 2)
    # Table 3 of Zare Harofteh and Nepomechie, Ann. Phys. 538, e70239 (2026)
    if (nqudits, levels) == (3, 2):
        table = [
            [2, 1, 0],
            [1, 2, 0],
            [0, 2, 1],
            [1, 1, 1],
            [2, 0, 1],
            [1, 0, 2],
            [0, 1, 2],
        ]
        backend.assert_allclose(
            _walsh_gray_code(3, 3, 2, backend=backend), backend.cast(table)
        )


def test_spin_s_encoder_errors(backend):
    with pytest.raises(ValueError):
        spin_s_encoder(3, 0.3, 2, backend=backend)
    with pytest.raises(ValueError):
        spin_s_encoder(3, 1, 7, backend=backend)
    with pytest.raises(ValueError):
        spin_s_encoder(3, 1, 2, encoding="ternary", backend=backend)


@pytest.mark.parametrize("seed", [10])
@pytest.mark.parametrize("optimize_controls", [False, True])
@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("data", [False, True])
@pytest.mark.parametrize("encoding", ["binary", "unary"])
@pytest.mark.parametrize(
    "nqudits,spin,weight",
    [
        (1, 1, 1),
        (2, 0.5, 1),
        (3, 1, 0),
        (3, 1, 3),
        (3, 1.5, 4),
        (4, 0.5, 2),
        (2, 2, 3),
        (4, 1, 3),
        (6, 0.5, 3),
    ],
)
def test_spin_s_encoder(
    backend,
    nqudits,
    spin,
    weight,
    encoding,
    data,
    complex_data,
    optimize_controls,
    seed,
):
    levels = int(2 * spin)
    width = levels if encoding == "unary" else levels.bit_length()
    dtype = backend.complex128 if complex_data else backend.float64

    ditstrings = [
        string
        for string in product(range(levels + 1), repeat=nqudits)
        if sum(string) == weight
    ]
    indices = [
        int(
            "".join(
                (
                    "0" * (width - value) + "1" * value
                    if encoding == "unary"
                    else format(value, f"0{width}b")
                )
                for value in string
            ),
            2,
        )
        for string in ditstrings
    ]

    # single-state subspaces are only prepared up to a global phase
    data = data and len(ditstrings) > 1
    if data:
        _data = random_statevector(
            len(ditstrings), dtype=dtype, seed=seed, backend=backend
        )
    else:
        _data = None
    target = backend.zeros(2 ** (nqudits * width), dtype=dtype)
    target[indices] = _data if data else backend.cast([1] + [0] * (len(indices) - 1))

    if data:
        with pytest.raises(ValueError):
            circuit = spin_s_encoder(
                nqudits,
                spin,
                weight,
                data=_data[:-1],
                complex_data=complex_data,
                encoding=encoding,
                optimize_controls=optimize_controls,
                backend=backend,
            )

    circuit = spin_s_encoder(
        nqudits,
        spin,
        weight,
        data=_data,
        complex_data=complex_data,
        encoding=encoding,
        optimize_controls=optimize_controls,
        backend=backend,
    )
    assert circuit.nqubits == nqudits * width
    state = backend.execute_circuit(circuit).state()

    backend.assert_allclose(state, target, atol=1e-7)


@pytest.mark.parametrize("seed", [10])
@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("nqubits,weight", [(3, 1), (4, 2), (5, 2), (6, 3)])
def test_spin_s_encoder_spin_half(backend, nqubits, weight, complex_data, seed):
    """For spin 1/2, ``spin_s_encoder`` reduces to ``hamming_weight_encoder``."""
    dtype = backend.complex128 if complex_data else backend.float64
    n_choose_k = int(binom(nqubits, weight))
    data = random_statevector(n_choose_k, dtype=dtype, seed=seed, backend=backend)

    circuit_farias = hamming_weight_encoder(nqubits, weight, data=data, backend=backend)
    state_farias = backend.execute_circuit(circuit_farias).state()

    circuits = [
        spin_s_encoder(
            nqubits, 0.5, weight, data=data, encoding=encoding, backend=backend
        )
        for encoding in ("binary", "unary")
    ]
    # binary and unary encodings coincide for spin 1/2
    assert circuits[0].draw() == circuits[1].draw()

    circuit = circuits[1]
    assert circuit.nqubits == nqubits
    names = [gate.name for gate in circuit.queue]
    # same gate set as the fixed-Hamming-weight encoder: k initial X gates and
    # one RBS per new amplitude, with no anti-controls (i.e. no further X gates)
    assert names[:weight] == ["x"] * weight
    assert "x" not in names[weight:]
    assert names.count("rbs") == n_choose_k - 1
    assert set(names[weight:]) <= {"crz", "rbs", "rz"}

    state = backend.execute_circuit(circuit).state()
    backend.assert_allclose(state, state_farias, atol=1e-7)


@pytest.mark.parametrize("sigma", [(0, 2, 1, 3), [0, 2, 1, 3]])
def test_permutation_synthesis_errors(sigma, backend):

    with pytest.raises(TypeError):
        permutation_synthesis(
            backend.cast(sigma, dtype=backend.int64), m=2, backend=backend
        )
    with pytest.raises(ValueError):
        permutation_synthesis([0, 2, 1, 3, 10], m=2, backend=backend)
    with pytest.raises(ValueError):
        permutation_synthesis(sigma, m=3, backend=backend)


def test_entangling_layer_errors():
    with pytest.raises(TypeError):
        entangling_layer(10.5)
    with pytest.raises(ValueError):
        entangling_layer(-4)
    with pytest.raises(TypeError):
        entangling_layer(10, architecture=True)
    with pytest.raises(NotImplementedError):
        entangling_layer(10, architecture="qibo")
    with pytest.raises(TypeError):
        entangling_layer(10, closed_boundary="True")
    with pytest.raises(NotImplementedError):
        entangling_layer(10, entangling_gate=gates.GeneralizedfSim)
    with pytest.raises(NotImplementedError):
        entangling_layer(10, entangling_gate=gates.TOFFOLI)
    with pytest.raises(ValueError):
        entangling_layer(7, architecture="x")


@pytest.mark.parametrize("closed_boundary", [False, True])
@pytest.mark.parametrize("entangling_gate", ["CNOT", gates.CZ, gates.RBS])
@pytest.mark.parametrize(
    "architecture",
    [
        "diagonal",
        "even_layer",
        "next_nearest",
        "odd_layer",
        "pyramid",
        "shifted",
        "v",
        "x",
    ],
)
@pytest.mark.parametrize("nqubits", [4, 6])
def test_entangling_layer(nqubits, architecture, entangling_gate, closed_boundary):
    target_circuit = Circuit(nqubits)
    if architecture in ["next_nearest", "pyramid", "v", "x"]:
        if architecture == "next_nearest":
            target_circuit.add(
                _helper_entangling_test(entangling_gate, qubit, qubit + 2)
                for qubit in range(nqubits - 2)
            )
            if closed_boundary:
                target_circuit.add(
                    _helper_entangling_test(entangling_gate, nqubits - 1, 0)
                )
        elif architecture == "pyramid":
            for end in range(nqubits - 1, 1, -1):
                target_circuit.add(
                    _helper_entangling_test(entangling_gate, qubit)
                    for qubit in range(end)
                )
        elif architecture == "v":
            target_circuit.add(
                _helper_entangling_test(entangling_gate, qubit, qubit + 1)
                for qubit in range(nqubits - 1)
            )
            target_circuit.add(
                _helper_entangling_test(entangling_gate, qubit - 1, qubit)
                for qubit in range(nqubits - 2, 1, -1)
            )
    else:
        if architecture == "diagonal":
            target_circuit.add(
                _helper_entangling_test(entangling_gate, qubit)
                for qubit in range(nqubits - 1)
            )
        elif architecture == "even_layer":
            target_circuit.add(
                _helper_entangling_test(entangling_gate, qubit)
                for qubit in range(0, nqubits - 1, 2)
            )
        elif architecture == "odd_layer":
            target_circuit.add(
                _helper_entangling_test(entangling_gate, qubit)
                for qubit in range(1, nqubits - 1, 2)
            )
        else:
            target_circuit.add(
                _helper_entangling_test(entangling_gate, qubit)
                for qubit in range(0, nqubits - 1, 2)
            )
            target_circuit.add(
                _helper_entangling_test(entangling_gate, qubit)
                for qubit in range(1, nqubits - 1, 2)
            )

        if closed_boundary:
            target_circuit.add(_helper_entangling_test(entangling_gate, nqubits - 1, 0))

    circuit = entangling_layer(nqubits, architecture, entangling_gate, closed_boundary)

    for gate, target in zip(circuit.queue, target_circuit.queue):
        assert gate.__class__.__name__ == target.__class__.__name__
        assert gate.qubits == target.qubits
        assert gate.target_qubits == target.target_qubits
        assert gate.control_qubits == target.control_qubits
        assert gate.parameters == target.parameters


@pytest.mark.parametrize("closed_boundary", [False, True])
@pytest.mark.parametrize("entangling_gate", ["CNOT", gates.CZ, gates.RBS])
@pytest.mark.parametrize("nqubits", [1, 2, 3, 4, 5, 6, 7, 8])
def test_entangling_layer_all_to_all(nqubits, entangling_gate, closed_boundary):
    circuit = entangling_layer(nqubits, "all-to-all", entangling_gate, closed_boundary)

    pairs = [gate.qubits for gate in circuit.queue]
    target_pairs = list(combinations(range(nqubits), 2))

    # every pair of qubits is acted upon exactly once, with the lower index first
    assert len(pairs) == len(target_pairs)
    assert sorted(pairs) == target_pairs

    # depth is minimal, i.e. the chromatic index of the complete graph
    if nqubits > 1:
        assert circuit.depth == (nqubits - 1 if nqubits % 2 == 0 else nqubits)


def _helper_entangling_test(gate, qubit_0, qubit_1=None):
    """Creates two-qubit gate with of without parameters."""
    if qubit_1 is None:
        qubit_1 = qubit_0 + 1

    if callable(gate) and gate.__name__ == "RBS":
        return gate(qubit_0, qubit_1, 0.0)

    if gate == "CNOT":
        gate = gates.CNOT

    return gate(qubit_0, qubit_1)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_circuit_kwargs(backend, density_matrix):
    test = comp_basis_encoder(5, 7, density_matrix=density_matrix)
    assert test.density_matrix is density_matrix

    test = entangling_layer(5, density_matrix=density_matrix)
    assert test.density_matrix is density_matrix

    data = backend.cast(np.random.rand(5), dtype=backend.float64)
    test = phase_encoder(
        nqubits=5, data=data, density_matrix=density_matrix, backend=backend
    )
    assert test.density_matrix is density_matrix

    test = unary_encoder(
        5, "diagonal", data, density_matrix=density_matrix, backend=backend
    )
    assert test.density_matrix is density_matrix

    test = unary_encoder_random_gaussian(
        4, density_matrix=density_matrix, backend=backend
    )
    assert test.density_matrix is density_matrix


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("nqubits", [1, 2, 3, 4])
def test_ghz_circuit(backend, nqubits, density_matrix):
    if nqubits < 2:
        with pytest.raises(ValueError):
            GHZ_circ = ghz_state(nqubits, density_matrix=density_matrix)
    else:
        target = np.zeros(2**nqubits, dtype=complex)
        target[0] = 1 / np.sqrt(2)
        target[2**nqubits - 1] = 1 / np.sqrt(2)
        target = backend.cast(target, dtype=target.dtype)

        GHZ_circ = ghz_state(nqubits, density_matrix=density_matrix)
        state = backend.execute_circuit(GHZ_circ).state()

        if density_matrix:
            target = backend.outer(target, backend.conj(target.T))

        backend.assert_allclose(state, target)


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("all_to_all", [False, True])
@pytest.mark.parametrize(
    "nqubits, weight",
    [
        (2, -1),
        (2, 0),
        (2, 1),
        (3, 1),
        (3, 2),
        (4, 2),
        (4, 1),
    ],
)
def test_dicke_state(backend, nqubits, weight, all_to_all, density_matrix):
    if weight < 0 or weight > nqubits:
        with pytest.raises(ValueError):
            dicke_circ = dicke_state(
                nqubits, weight, all_to_all=all_to_all, density_matrix=density_matrix
            )
    else:
        # Build expected Dicke state vector
        target = np.zeros(2**nqubits, dtype=complex)
        bitstrings = combinations(range(nqubits), weight)
        for positions in bitstrings:
            index = sum(1 << (nqubits - 1 - p) for p in positions)
            target[index] = 1.0
        target /= np.sqrt(np.count_nonzero(target))

        target = backend.cast(target, dtype=target.dtype)

        dicke_circ = dicke_state(
            nqubits, weight, all_to_all=all_to_all, density_matrix=density_matrix
        )
        result = backend.execute_circuit(dicke_circ)
        state = result.state()

        if density_matrix:
            target = backend.outer(target, backend.conj(target.T))

        backend.assert_allclose(state, target)


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize(
    "nqubits, mqubits, weight",
    [
        (2, 1, 1),
        (3, 1, 0),
        (4, 1, 2),
        (4, 2, 2),
        (4, 3, 1),
        (5, 2, 3),
    ],
)
def test_wbd_gate(backend, nqubits, mqubits, weight, density_matrix):
    wbd_circ = Circuit(nqubits, density_matrix=density_matrix)
    first_register = list(range(mqubits, nqubits))
    second_register = list(range(mqubits))
    if mqubits > nqubits - mqubits:
        with pytest.raises(ValueError):
            _add_wbd_gate(
                wbd_circ, first_register, second_register, nqubits, mqubits, weight
            )
    else:
        _add_wbd_gate(
            wbd_circ, first_register, second_register, nqubits, mqubits, weight
        )

        # Build expected WBD state vector (Definition 2 from [2])
        # Test for all considered (0 <= l <= weight) input states: |0>^n-l |1>^l
        for l in range(weight + 1):
            initial = np.zeros(2 ** (nqubits), dtype=complex)
            target = np.zeros(2 ** (nqubits), dtype=complex)

            # initial state |0>^n-l |1>^l
            initial[2**l - 1] = 1

            comb_n_l = math.comb(nqubits, l)
            for i in range(min(l, mqubits) + 1):
                index = 2 ** (nqubits - mqubits) * (2 ** (i) - 1) + 2 ** (l - i) - 1
                target[index] = np.sqrt(
                    math.comb(mqubits, i)
                    * math.comb(nqubits - mqubits, l - i)
                    / comb_n_l
                )

            initial = backend.cast(initial, dtype=initial.dtype)
            target = backend.cast(target, dtype=target.dtype)

            if density_matrix:
                initial = backend.outer(initial, backend.conj(initial.T))
                target = backend.outer(target, backend.conj(target.T))

            result = backend.execute_circuit(wbd_circ, initial_state=initial)
            state = result.state()

            backend.assert_allclose(state, target)


@pytest.mark.parametrize(
    "matrix, expects_error, circuit1, circuit2",
    [
        # Test Case 1: 3-qubit graph
        ([[0, 1, 0], [1, 0, 1], [0, 1, 0]], False, True, False),
        # Test Case 2: 5-qubit  graph
        (
            [
                [0, 1, 0, 0, 1],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1],
                [1, 0, 0, 1, 0],
            ],
            False,
            False,
            True,
        ),
        # Test Case 3: Non-symmetric matrix (expected error)
        (
            [[0, 1, 0], [0, 0, 1], [0, 1, 0]],
            True,
            False,
            False,
        ),  # matrix[1,0] != matrix[0,1]
        # Test Case 4: Non-symmetric matrix (expected error)
        ([[0, 1], [0, 0]], True, False, False),  # matrix[1,0] != matrix[0,1]
    ],
)
def test_graph_state(backend, matrix, expects_error, circuit1, circuit2):

    if expects_error:
        # We expect a ValueError for non-symmetric matrices
        with pytest.raises(ValueError):
            graph_state(matrix, backend=backend)

    else:
        if circuit1:
            nqubits = 3
            target = Circuit(nqubits)
            target.add(gates.H(qubit) for qubit in range(nqubits))
            target.add(gates.CZ(0, 1))
            target.add(gates.CZ(1, 2))

        if circuit2:
            nqubits = 5
            target = Circuit(nqubits)
            target.add(gates.H(qubit) for qubit in range(nqubits))
            target.add(gates.CZ(0, 1))
            target.add(gates.CZ(0, 4))
            target.add(gates.CZ(1, 2))
            target.add(gates.CZ(2, 3))
            target.add(gates.CZ(3, 4))

        circuit = graph_state(matrix)
        backend.assert_circuitclose(circuit, target)


@pytest.mark.parametrize("method", [None, "parity"])
@pytest.mark.parametrize("nqubits", [6, 8, 10])
def test_fanout_and_ladder_synthesis(backend, nqubits, method):
    with pytest.raises(ValueError):
        qubits = [0, 1, 3]
        ladder_synthesis(qubits, return_circuit=True)

    with pytest.raises(ValueError):
        qubits = [0, 1, 3]
        fanout_synthesis(qubits, method=method)

    with pytest.raises(ValueError):
        fanout_synthesis(qubits=[0, 1, 2], method="foo")

    qubits = (0, 1)
    fanout = fanout_synthesis(qubits, method=method)
    target = gates.FanOut(*qubits).matrix(backend)
    backend.assert_allclose(fanout.unitary(backend), target)

    qubits = list(range(nqubits))
    fanout = fanout_synthesis(qubits, method=method)
    target = gates.FanOut(*qubits).matrix(backend)
    backend.assert_allclose(fanout.unitary(backend), target)

    ladder = ladder_synthesis(qubits, return_circuit=True)
    target = Circuit(nqubits)
    target.add(gates.CNOT(qubit, qubit + 1) for qubit in range(nqubits - 1))
    target = target.invert()
    backend.assert_allclose(ladder.unitary(backend), target.unitary(backend))


def _leaf_separable_data(
    nqubits,
    weight,
    leaf_size,
    seed,
    correlated=False,
    complex_leaves=False,
    distribution_phases=None,
):
    """Random leaf-separable amplitudes.

    If ``correlated`` is ``False``, the weights of the leaves are independent, conditioned on
    their sum, and the weight distribution is exactly given by a tree of weight distribution
    blocks. Otherwise, it is an arbitrary distribution.
    If ``complex_leaves`` is ``True``, the leaf states are complex.
    ``distribution_phases`` can be ``"sign"`` or ``"phase"``, to multiply the amplitudes of
    each weight distribution by a random sign or phase, respectively.
    """
    rng = np.random.default_rng(seed)
    sizes = [leaf_size] * (nqubits // leaf_size)
    if nqubits % leaf_size:
        sizes.append(nqubits % leaf_size)
    starts = np.cumsum([0] + sizes[:-1])
    states = np.sort(
        [
            sum(1 << (nqubits - 1 - qubit) for qubit in combination)
            for combination in combinations(range(nqubits), weight)
        ]
    )

    leaf_amplitudes = []
    for size in sizes:
        values = rng.normal(size=2**size)
        if complex_leaves:
            values = values + 1j * rng.normal(size=2**size)
        leaf_amplitudes.append(values)

    shape = (weight + 1,) * len(sizes)
    independent = rng.random((len(sizes), weight + 1)) + 0.1
    arbitrary = rng.random(shape) + 0.1
    phases = {
        None: np.ones(shape),
        "sign": np.sign(rng.normal(size=shape)),
        "phase": np.exp(1j * rng.uniform(0, 2 * np.pi, size=shape)),
    }[distribution_phases]

    data = []
    for state in states:
        amplitude = 1.0
        distribution = []
        for leaf, (start, size) in enumerate(zip(starts, sizes)):
            string = (state >> (nqubits - start - size)) & (2**size - 1)
            distribution.append(int(string).bit_count())
            amplitude *= leaf_amplitudes[leaf][string]
            if not correlated:
                amplitude *= np.sqrt(independent[leaf][distribution[-1]])
        if correlated:
            amplitude *= np.sqrt(arbitrary[tuple(distribution)])
        data.append(amplitude * phases[tuple(distribution)])
    data = np.array(data)

    return data / np.linalg.norm(data), states


@pytest.mark.parametrize("nqubits, mqubits", [(4, 2), (5, 3)])
@pytest.mark.parametrize(
    "phases",
    [None, [[0.0, 0.0, 0.0], [0.3, -1.2, 0.0], [0.7, 2.0, -0.4]]],
)
@pytest.mark.parametrize(
    "probabilities",
    [
        [[1.0], [0.3, 0.7], [0.2, 0.5, 0.3]],
        [[1.0], [0.0, 1.0], [1.0, 0.0, 0.0]],
    ],
)
def test_wbd_gate_probabilities(backend, probabilities, phases, nqubits, mqubits):
    weight = 2
    circuit = Circuit(nqubits)
    _add_wbd_gate(
        circuit,
        list(range(mqubits, nqubits)),
        list(range(mqubits)),
        nqubits,
        mqubits,
        weight,
        probabilities=probabilities,
        phases=phases,
    )

    for l in range(weight + 1):
        initial = np.zeros(2**nqubits, dtype=complex)
        target = np.zeros(2**nqubits, dtype=complex)
        initial[2**l - 1] = 1
        for i in range(min(l, mqubits) + 1):
            index = 2 ** (nqubits - mqubits) * (2**i - 1) + 2 ** (l - i) - 1
            target[index] = np.sqrt(probabilities[l][i])
            if phases is not None:
                target[index] *= np.exp(1j * phases[l][i])

        initial = backend.cast(initial, dtype=initial.dtype)
        target = backend.cast(target, dtype=target.dtype)
        state = backend.execute_circuit(circuit, initial_state=initial).state()

        backend.assert_allclose(state, target, atol=1e-8)


@pytest.mark.parametrize(
    "nqubits, weight, leaf_size",
    [
        (4, 2, 2),
        (5, 2, 3),
        (6, 2, 2),
        (6, 3, 3),
        (7, 2, 2),
        (7, 3, 3),
        (8, 2, 3),
        (8, 3, 3),
        (9, 3, 3),
        (10, 2, 2),
        (11, 2, 2),
    ],
)
def test_leaf_separable_encoder(backend, nqubits, weight, leaf_size):
    data, states = _leaf_separable_data(nqubits, weight, leaf_size, seed=7)
    target = np.zeros(2**nqubits, dtype=complex)
    target[states] = data
    target = backend.cast(target, dtype=target.dtype)
    data = backend.cast(data, dtype=backend.float64)

    circuit = leaf_separable_encoder(
        nqubits,
        weight,
        leaf_size,
        data,
        backend=backend,
    )
    state = backend.execute_circuit(circuit).state()

    backend.assert_allclose(state, target, atol=1e-8)


@pytest.mark.parametrize(
    "nqubits, weight, leaf_size", [(6, 2, 2), (7, 3, 3), (8, 2, 3)]
)
def test_leaf_separable_encoder_dicke(backend, nqubits, weight, leaf_size):
    circuit = leaf_separable_encoder(
        nqubits, weight, leaf_size, np.ones(math.comb(nqubits, weight)), backend=backend
    )
    target = dicke_state(nqubits, weight, all_to_all=True)

    backend.assert_allclose(
        backend.execute_circuit(circuit).state(),
        backend.execute_circuit(target).state(),
        atol=1e-8,
    )


@pytest.mark.parametrize("distribution", [(2, 0), (1, 1), (0, 2)])
def test_leaf_separable_encoder_single_distribution(backend, distribution):
    nqubits, weight, leaf_size = 6, 2, 3
    values = backend.random_sample((2, 2**leaf_size), seed=5) + 0.1
    states = backend.sort(
        [
            sum(1 << (nqubits - 1 - qubit) for qubit in combination)
            for combination in combinations(range(nqubits), weight)
        ]
    )

    data = np.zeros(len(states))
    for index, state in enumerate(states):
        strings = (state >> leaf_size, state & (2**leaf_size - 1))
        if tuple(int(string).bit_count() for string in strings) == distribution:
            data[index] = values[0][strings[0]] * values[1][strings[1]]
    data /= np.linalg.norm(data)
    target = np.zeros(2**nqubits, dtype=complex)
    target[states] = data
    target = backend.cast(target, dtype=backend.float64)
    data = backend.cast(data, dtype=data.dtype)

    circuit = leaf_separable_encoder(nqubits, weight, leaf_size, data, backend=backend)
    state = backend.execute_circuit(circuit).state()

    backend.assert_allclose(state, target, atol=1e-8)


def test_leaf_separable_encoder_approximation(backend):
    nqubits, weight, leaf_size = 8, 2, 2
    data, states = _leaf_separable_data(
        nqubits, weight, leaf_size, seed=3, correlated=True
    )
    target = np.zeros(2**nqubits)
    target[states] = data

    circuit = leaf_separable_encoder(nqubits, weight, leaf_size, data, backend=backend)
    state = backend.execute_circuit(circuit).state()
    fid = fidelity(state, target, backend=backend)

    # weight distributions that are not a product of conditional distributions are lost
    assert 0.9 < fid < 1 - 1e-6


@pytest.mark.parametrize("complex_leaves", [False, True])
@pytest.mark.parametrize("distribution_phases", ["sign", "phase"])
@pytest.mark.parametrize(
    "nqubits, weight, leaf_size",
    [(4, 2, 2), (5, 2, 3), (6, 2, 2), (6, 3, 3), (8, 3, 3), (9, 3, 3)],
)
def test_leaf_separable_encoder_phases(
    backend, nqubits, weight, leaf_size, distribution_phases, complex_leaves
):
    # two or three leaves: any phase of the weight distributions can be encoded
    data, states = _leaf_separable_data(
        nqubits,
        weight,
        leaf_size,
        seed=11,
        complex_leaves=complex_leaves,
        distribution_phases=distribution_phases,
    )
    target = np.zeros(2**nqubits, dtype=complex)
    target[states] = data
    target = backend.cast(target, dtype=target.dtype)

    circuit = leaf_separable_encoder(
        nqubits, weight, leaf_size, backend.cast(data), backend=backend
    )
    state = backend.execute_circuit(circuit).state()

    backend.assert_allclose(state, target, atol=1e-8)


@pytest.mark.parametrize(
    "nqubits, weight, leaf_size", [(8, 2, 2), (10, 2, 2), (11, 2, 2), (12, 3, 3)]
)
def test_leaf_separable_encoder_complex_leaves(backend, nqubits, weight, leaf_size):
    # phases that depend on the weight of each leaf are encoded for any number of leaves
    data, states = _leaf_separable_data(
        nqubits, weight, leaf_size, seed=5, complex_leaves=True
    )
    target = np.zeros(2**nqubits, dtype=complex)
    target[states] = data
    target = backend.cast(target, dtype=target.dtype)

    circuit = leaf_separable_encoder(nqubits, weight, leaf_size, data, backend=backend)
    state = backend.execute_circuit(circuit).state()

    backend.assert_allclose(state, target, atol=1e-8)


@pytest.mark.parametrize("distribution_phases", ["sign", "phase"])
def test_leaf_separable_encoder_phases_approximation(backend, distribution_phases):
    nqubits, weight, leaf_size = 10, 2, 2
    data, states = _leaf_separable_data(
        nqubits,
        weight,
        leaf_size,
        seed=2,
        distribution_phases=distribution_phases,
    )
    target = np.zeros(2**nqubits, dtype=complex)
    target[states] = data
    target = backend.cast(target, dtype=target.dtype)

    circuit = leaf_separable_encoder(nqubits, weight, leaf_size, data, backend=backend)
    state = backend.execute_circuit(circuit).state()
    fid = fidelity(state, target, backend=backend)

    # phases that are not a sum of phases of each node are only approximated
    assert 0.5 < fid < 1 - 1e-6


def test_leaf_separable_encoder_phases_convergence_warning(backend, caplog):
    nqubits, weight, leaf_size = 10, 2, 2
    # the alternating maximization of the phases reaches its maximum number of sweeps
    data, _ = _leaf_separable_data(
        nqubits, weight, leaf_size, seed=2, distribution_phases="phase"
    )
    with caplog.at_level(logging.WARNING):
        leaf_separable_encoder(nqubits, weight, leaf_size, data, backend=backend)
    assert "did not converge" in caplog.text

    # and it converges for other data
    caplog.clear()
    data, _ = _leaf_separable_data(
        8, weight, leaf_size, seed=0, distribution_phases="phase"
    )
    with caplog.at_level(logging.WARNING):
        leaf_separable_encoder(8, weight, leaf_size, data, backend=backend)
    assert "did not converge" not in caplog.text


def test_leaf_separable_encoder_errors(backend):
    data = backend.random_sample(math.comb(6, 2), seed=10)

    for weight, leaf_size in [(0, 2), (3, 2), (2, 6)]:
        with pytest.raises(ValueError):
            leaf_separable_encoder(6, weight, leaf_size, data, backend=backend)

    with pytest.raises(ValueError):
        leaf_separable_encoder(6, 2, 2, data[:-1], backend=backend)

    # generic amplitudes are not leaf-separable
    with pytest.raises(ValueError):
        leaf_separable_encoder(6, 2, 2, data, backend=backend)


@pytest.mark.parametrize(
    "nqubits,nmagnons,domains,modes,method",
    [
        (6, 0, None, None, "sopena"),
        (4, 3, None, None, "sopena"),
        (7, 3, None, None, "sopena"),
        (6, 2, None, [1, 2, 3], "sopena"),
        (6, 2, None, [0, 1], "sopena"),
        (6, 2, None, [2, 2], "sopena"),
        (6, 2, None, [1, 6], "sopena"),
        (7, 1, [[]], None, "sopena"),
        (7, 1, [[3]], None, "sopena"),
        (7, 1, [[3, 5]], None, "sopena"),
        (7, 1, [[2, 3]], None, "sopena"),
        (7, 1, [[6, 8]], None, "sopena"),
        (12, 1, [[3, 4], [6, 7]], None, "sopena"),
        (6, 0, None, None, "merged"),
        (7, 1, [[3, 5]], None, "merged"),
        (5, 1, [[3, 4]], None, "other"),
    ],
)
def test_folded_xxz_eigenstate_errors(
    backend, nqubits, nmagnons, domains, modes, method
):
    with pytest.raises(ValueError):
        folded_xxz_eigenstate(
            nqubits, nmagnons, domains, modes, method=method, backend=backend
        )


# larger cases are only for the method that needs fewer qubits
EIGENSTATE_CASES = [
    (*case, method)
    for case in [
        (4, 1, None, None, [2]),
        (5, 2, [], [1, 3], [1, 3]),
        (5, 2, None, None, [2, 4]),
        (8, 3, None, None, [2, 4, 6]),
        (5, 1, [[3, 4]], None, [2]),
        (6, 1, [[3, 4, 5]], [1], [1]),
        (7, 2, [[5, 6]], None, [2, 4]),
        # the seed state has no amplitude on the first basis element, so completing it
        # into a basis must not rely on that element; this size used to return ``NaN``
        (8, 1, None, None, [2]),
        (8, 1, [[3, 4], [7, 8]], None, [2]),
    ]
    for method in ["sopena", "merged"]
]
EIGENSTATE_CASES.extend(
    [
        (10, 2, [[5, 6], [9, 10]], None, [2, 5], "merged"),
        (10, 3, [[7, 8]], None, [2, 4, 6], "merged"),
        (9, 2, [[6, 7]], None, [2, 6], "merged"),
        (9, 2, [[5, 6, 7]], None, [2, 6], "merged"),
    ]
)


@pytest.mark.parametrize(
    "nsites,nmagnons,domains,modes,target_modes,method", EIGENSTATE_CASES
)
def test_folded_xxz_eigenstate(
    backend, nsites, nmagnons, domains, modes, target_modes, method
):
    circuit = folded_xxz_eigenstate(
        nsites, nmagnons, domains, modes, method=method, backend=backend
    )

    # the chain is in the first qubits, and the auxiliary qubits are in the zero state
    state = backend.execute_circuit(circuit).state()
    chain = state[:: 2 ** (circuit.nqubits - nsites)]
    backend.assert_allclose(backend.vector_norm(chain), 1.0, atol=1e-8)

    # the boundary qubits of the Hamiltonian are in the zero state
    vacuum = backend.zero_state(1)
    state = backend.kron(backend.kron(vacuum, chain), vacuum)
    hamiltonian = FoldedXXZ(nsites + 2, backend=backend)

    nsqueezed = nsites - nmagnons + 1 - 2 * len(domains or [])
    momenta = (
        math.pi * backend.cast(target_modes, dtype=backend.float64) / (nsqueezed + 1)
    )
    energy = -backend.sum(backend.cos(momenta))
    backend.assert_allclose(hamiltonian.matrix @ state, energy * state, atol=1e-8)


@pytest.mark.parametrize(
    "nsites,nmagnons,domains,target_modes",
    [
        (12, 1, [[3, 4], [7, 8], [11, 12]], [2]),
        (13, 1, [[4, 5], [8, 9], [12, 13]], [2]),
        (16, 1, [[3, 4], [7, 8], [11, 12]], [2]),
        (14, 2, [[5, 6], [9, 10], [13, 14]], [2, 7]),
    ],
)
@pytest.mark.parametrize("method", ["sopena", "merged"])
def test_folded_xxz_eigenstate_many_domains(
    backend, nsites, nmagnons, domains, target_modes, method
):
    """The circuits are too large to be simulated, so the first stage is simulated alone, and the
    other stages, which only permute basis states, are applied to the bit strings of its state.
    """
    if method == "merged":
        # relative phases are not simulated, so the gates are the ones that permute states
        circuit = _folded_merged(
            nsites, nmagnons, domains, target_modes, backend, relative_phase=False
        )
    else:
        circuit = folded_xxz_eigenstate(
            nsites, nmagnons, domains, method=method, backend=backend
        )
    nfree = nsites - 2 * len(domains)

    last = max(
        i
        for i, gate in enumerate(circuit.queue)
        if isinstance(gate, (gates.Unitary, gates.RBS))
    )
    first_stage = circuit.queue[: last + 1]
    free_fermions = Circuit(nfree)
    free_fermions.add(
        gate.on_qubits({q: q - nsites for q in gate.qubits}) for gate in first_stage
    )
    amplitudes = backend.to_numpy(backend.execute_circuit(free_fermions).state())

    chain = np.zeros(2**nsites, dtype=complex)
    for index in np.flatnonzero(abs(amplitudes) > 1e-12):
        bits = [0] * circuit.nqubits
        bits[nsites : nsites + nfree] = [int(bit) for bit in f"{index:0{nfree}b}"]
        for gate in circuit.queue[len(first_stage) :]:
            if all(bits[qubit] for qubit in gate.control_qubits):
                first, *others = gate.target_qubits
                if others:
                    bits[first], bits[others[0]] = bits[others[0]], bits[first]
                else:
                    bits[first] ^= 1
        assert not any(bits[nsites:])
        chain[int("".join(map(str, bits[:nsites])), 2)] += amplitudes[index]

    chain = backend.cast(chain, dtype=backend.complex128)
    backend.assert_allclose(backend.vector_norm(chain), 1.0, atol=1e-8)

    vacuum = backend.zero_state(1)
    state = backend.kron(backend.kron(vacuum, chain), vacuum)
    hamiltonian = FoldedXXZ(nsites + 2, dense=False, backend=backend)

    nsqueezed = nfree - nmagnons + 1
    momenta = (
        math.pi * backend.cast(target_modes, dtype=backend.float64) / (nsqueezed + 1)
    )
    energy = -backend.sum(backend.cos(momenta))
    backend.assert_allclose(hamiltonian @ state, energy * state, atol=1e-8)


@pytest.mark.parametrize("nsites,nmagnons,domains", [(8, 3, None), (7, 2, [[5, 6]])])
def test_folded_xxz_eigenstate_methods(backend, nsites, nmagnons, domains):
    chains = []
    for method in ("sopena", "merged"):
        circuit = folded_xxz_eigenstate(
            nsites, nmagnons, domains, method=method, backend=backend
        )
        state = backend.execute_circuit(circuit).state()
        chains.append(state[:: 2 ** (circuit.nqubits - nsites)])

    overlap = backend.sum(backend.conj(chains[0]) * chains[1])
    backend.assert_allclose(backend.abs(overlap), 1.0, atol=1e-8)


@pytest.mark.parametrize("absent", [(0, 1), (1, 1)])
def test_folded_xxz_cswap(backend, absent):
    circuit = Circuit(3)
    circuit.add(_folded_xxz_cswap(0, 1, 2, absent))
    swap = Circuit(3)
    swap.add(gates.SWAP(1, 2).controlled_by(0))

    # the gate is the controlled-SWAP up to the sign of the states with ``absent``
    matrix = circuit.unitary(backend)
    target = swap.unitary(backend)
    for index, bits in enumerate(product([0, 1], repeat=3)):
        if bits[1:] != absent:
            backend.assert_allclose(matrix[:, index], target[:, index], atol=1e-8)


@pytest.mark.parametrize(
    "nsites,nmagnons,domains", [(8, 3, None), (10, 2, [[5, 6], [9, 10]])]
)
def test_folded_xxz_eigenstate_cancelled(backend, nsites, nmagnons, domains):
    circuit = folded_xxz_eigenstate(
        nsites, nmagnons, domains, method="merged", backend=backend
    )

    # there are no gates left that cancel
    assert len(InverseCancellation()(circuit).queue) == len(circuit.queue)
