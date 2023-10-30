import itertools

import numpy as np
import pytest

from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit
from qibo.transpiler.pipeline import transpose_qubits
from qibo.transpiler.star_connectivity import StarConnectivity


def generate_random_circuit(nqubits, depth, seed=None, middle_qubit=2):
    """Generate random circuits one-qubit rotations and CZ gates."""
    # find the number of qubits for hardware circuit
    if nqubits == 1:
        hardware_qubits = 1
    else:
        hardware_qubits = max(nqubits, middle_qubit + 1)

    pairs = list(itertools.combinations(range(hardware_qubits), 2))
    if seed is not None:  # pragma: no cover
        np.random.seed(seed)

    rotations = [gates.RX, gates.RY, gates.RZ]
    circuit = Circuit(hardware_qubits)
    for _ in range(depth):
        for i in range(hardware_qubits):
            # generate a random rotation
            rotation = rotations[int(np.random.randint(0, 3))]
            theta = 2 * np.pi * np.random.random()
            circuit.add(rotation(i, theta=theta))
        # add CZ gates on random qubit pairs
        for i in np.random.randint(0, len(pairs), len(pairs)):
            q1, q2 = pairs[i]
            circuit.add(gates.CZ(q1, q2))

    return circuit


@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("nqubits", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("middle_qubit", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("depth", [2, 10])
def test_fix_connectivity(nqubits, depth, middle_qubit, verbose):
    """Checks that the transpiled circuit can be executed and is equivalent to original."""
    original = generate_random_circuit(nqubits, depth, middle_qubit=middle_qubit)
    transpiler = StarConnectivity(middle_qubit=middle_qubit, verbose=verbose)
    transpiled, hardware_qubits = transpiler(original)
    # check that transpiled circuit can be executed
    assert transpiler.is_satisfied(transpiled)
    # check that execution results agree with original (using simulation)
    backend = NumpyBackend()
    final_state = backend.execute_circuit(transpiled).state()
    target_state = backend.execute_circuit(original).state()
    hardware_qubits = list(hardware_qubits.values())
    target_state = transpose_qubits(target_state, hardware_qubits)
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("nqubits", [2, 3, 4, 5])
@pytest.mark.parametrize("middle_qubit", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("unitary_dim", [1, 2])
@pytest.mark.parametrize("depth", [2, 10])
def test_fix_connectivity_unitaries(nqubits, unitary_dim, depth, middle_qubit):
    """Checks that the transpiled circuit can be executed and is equivalent to original
    when using unitaries."""

    from .test_transpilers_unitary_decompositions import random_unitary

    # find the number of qubits for hardware circuit
    n_hardware_qubits = max(nqubits, middle_qubit + 1)

    original = Circuit(n_hardware_qubits)
    pairs = list(itertools.combinations(range(n_hardware_qubits), unitary_dim))
    for _ in range(depth):
        qubits = pairs[int(np.random.randint(len(pairs)))]
        original.add(gates.Unitary(random_unitary(unitary_dim), *qubits))

    transpiler = StarConnectivity(middle_qubit=middle_qubit)
    transpiled, hardware_qubits = transpiler(original)
    # check that transpiled circuit can be executed
    assert transpiler.is_satisfied(transpiled)
    # check that execution results agree with original (using simulation)
    backend = NumpyBackend()
    final_state = backend.execute_circuit(transpiled).state()
    target_state = backend.execute_circuit(original).state()
    hardware_qubits = list(hardware_qubits.values())
    target_state = transpose_qubits(target_state, hardware_qubits)
    np.testing.assert_allclose(final_state, target_state)
