import pytest

from qibo.quantum_info.random_ensembles import random_clifford
from qibo.hamiltonians import XXZ
from qibo.hamiltonians.expectation_values import get_expval_hamiltonian

def get_XXZ_hamilt_explicit(nqubits, delta=0.5):
    """
    returns a list with the terms of the XXZ hamiltonians for a given
    number of qubits and delta

    format: [(coef, pauli_word)]
    """

    hamiltonian_xxz = []

    for pauli in "X Y Z".split():
        for i in range(nqubits):
            pauli_str = ["I"] * nqubits
            if i != nqubits - 1:
                pauli_str[i], pauli_str[i + 1] = pauli, pauli
            else:
                pauli_str[i], pauli_str[0] = pauli, pauli

            pauli_str = "".join(pauli_str)

            if pauli == "Z":
                hamiltonian_xxz.append((delta, pauli_str))
            else:
                hamiltonian_xxz.append((1.0, pauli_str))

    return hamiltonian_xxz

@pytest.mark.parametrize("nqubits", [4, 6, 8])
@pytest.mark.parametrize("nshots", [None, int(1e6)])
@pytest.mark.parametrize("hamiltonian_type", ["matrix", "like_list", "like_dict"])
def test_get_expval_hamiltonian(backend, nqubits, nshots, hamiltonian_type,):

    circuit = random_clifford(nqubits, backend=backend)

    hamiltonian_matrix = XXZ(nqubits=nqubits, backend=backend).matrix
    state = backend.execute_circuit(circuit).state()
    expval_matrix = (backend.np.conj(state) @ hamiltonian_matrix @ state).real

    if hamiltonian_type == "like_list":
        hamiltonian = get_XXZ_hamilt_explicit(nqubits, delta=0.5)
    if hamiltonian_type == "matrix":
        hamiltonian = XXZ(nqubits=nqubits, backend=backend).matrix

    expval = get_expval_hamiltonian(
        circuit,
        hamiltonian,
        nshots,
        backend,
    )

    backend.assert_allclose(expval, expval_matrix)