"""Tests for the helper functions of the Clifford module."""

from functools import reduce

import numpy as np
import pytest

from qibo.backends import CliffordBackend, NumpyBackend, _get_engine_name
from qibo.quantum_info._clifford_utils import (
    _one_qubit_paulis_string_product,
    _string_product,
)
from qibo.quantum_info.clifford import Clifford
from qibo.quantum_info.random_ensembles import random_clifford


def construct_clifford_backend(backend):
    if backend.__class__.__name__ in (
        "TensorflowBackend",
        "PyTorchBackend",
        "CuQuantumBackend",
    ):
        with pytest.raises(NotImplementedError):
            CliffordBackend(backend.name)
        pytest.skip("Clifford backend not defined for the this engine.")

    return CliffordBackend(_get_engine_name(backend))


def _pauli_string_to_matrix(string):
    """Matrix of a Pauli string with an optional global phase, e.g. ``"-iXZ"``."""
    paulis = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }
    phase = 1
    while string[0] in "-i":
        phase *= -1 if string[0] == "-" else 1j
        string = string[1:]

    return phase * reduce(np.kron, [paulis[pauli] for pauli in string])


@pytest.mark.parametrize("phase_2", ["", "-", "i", "-i"])
@pytest.mark.parametrize("phase_1", ["", "-", "i", "-i"])
@pytest.mark.parametrize("pauli_2", ["I", "X", "Y", "Z"])
@pytest.mark.parametrize("pauli_1", ["I", "X", "Y", "Z"])
def test_one_qubit_paulis_string_product_matrices(pauli_1, pauli_2, phase_1, phase_2):
    operator_1, operator_2 = phase_1 + pauli_1, phase_2 + pauli_2
    product = _one_qubit_paulis_string_product(operator_1, operator_2)

    target = _pauli_string_to_matrix(operator_1) @ _pauli_string_to_matrix(operator_2)
    np.testing.assert_allclose(_pauli_string_to_matrix(product), target, atol=1e-12)


def test_string_product_matrices():
    rng = np.random.default_rng(1234)
    for _ in range(200):
        nqubits = int(rng.integers(1, 4))
        operators = [
            str(rng.choice(["", "-"])) + "".join(rng.choice(list("IXYZ"), nqubits))
            for _ in range(int(rng.integers(2, 5)))
        ]

        target = reduce(np.matmul, [_pauli_string_to_matrix(op) for op in operators])
        product = _pauli_string_to_matrix(_string_product(operators))

        np.testing.assert_allclose(product, target, atol=1e-12)


@pytest.mark.parametrize("nqubits", [2, 3, 4])
def test_clifford_stabilizers_stabilize_state(backend, nqubits):
    construct_clifford_backend(backend)

    for seed in range(10):
        circuit = random_clifford(nqubits, seed=seed, backend=NumpyBackend())
        clifford = Clifford.from_circuit(circuit, platform=_get_engine_name(backend))
        state = backend.to_numpy(clifford.state())

        stabilizers = clifford.stabilizers()
        assert len(stabilizers) == 2**nqubits
        for stabilizer in stabilizers:
            matrix = _pauli_string_to_matrix(stabilizer)
            np.testing.assert_allclose(matrix @ state, state, atol=1e-10)
