from itertools import product

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.backends import CliffordBackend, NumpyBackend
from qibo.quantum_info import Clifford, random_clifford

clifford_bkd = CliffordBackend()
numpy_bkd = NumpyBackend()

thetas = [
    th + 2 * i * np.pi for i in range(2) for th in [0, np.pi / 2, np.pi, 3 * np.pi / 2]
]
axes = ["RX", "RY", "RZ"]


@pytest.mark.parametrize("axis,theta", list(product(axes, thetas)))
def test_rotations(theta, axis):
    c = Circuit(3, density_matrix=True)
    qubits = np.random.randint(3, size=2)
    H_qubits = np.random.choice(range(3), size=2, replace=False)
    for q in H_qubits:
        c.add(gates.H(q))
    c.add(getattr(gates, axis)(qubits[0], theta=theta))
    c.add(getattr(gates, axis)(qubits[1], theta=theta))
    c.add(gates.M(0))
    clifford_state = clifford_bkd.execute_circuit(c).state()
    numpy_state = numpy_bkd.execute_circuit(c).state()
    numpy_bkd.assert_allclose(clifford_state, numpy_state, atol=1e-8)


@pytest.mark.parametrize("gate", ["I", "H", "S", "Z", "X", "Y", "SX", "SDG", "SXDG"])
def test_single_qubit_gates(gate):
    c = Circuit(3, density_matrix=True)
    qubits = np.random.randint(3, size=2)
    H_qubits = np.random.choice(range(3), size=2, replace=False)
    for q in H_qubits:
        c.add(gates.H(q))
    c.add(getattr(gates, gate)(qubits[0]))
    c.add(getattr(gates, gate)(qubits[1]))
    c.add(gates.M(0))
    clifford_state = clifford_bkd.execute_circuit(c).state()
    numpy_state = numpy_bkd.execute_circuit(c).state()
    numpy_bkd.assert_allclose(clifford_state, numpy_state, atol=1e-8)


@pytest.mark.parametrize("gate", ["CNOT", "CZ", "SWAP", "iSWAP", "FSWAP", "ECR"])
def test_two_qubits_gates(gate):
    c = Circuit(5, density_matrix=True)
    qubits = np.random.choice(range(5), size=4, replace=False).reshape(2, 2)
    H_qubits = np.random.choice(range(5), size=3, replace=False)
    for q in H_qubits:
        c.add(gates.H(q))
    c.add(getattr(gates, gate)(*qubits[0]))
    c.add(getattr(gates, gate)(*qubits[1]))
    c.add(gates.M(0))
    clifford_state = clifford_bkd.execute_circuit(c).state()
    numpy_state = numpy_bkd.execute_circuit(c).state()
    print(np.abs(clifford_state - numpy_state).sum())
    numpy_bkd.assert_allclose(clifford_state, numpy_state, atol=1e-8)


MEASURED_QUBITS = np.random.choice(range(5), size=3, replace=False)


@pytest.mark.parametrize(
    "prob_qubits",
    [
        range(5),
        np.random.choice(MEASURED_QUBITS, size=2, replace=False),
        [0],
        [1],
        [2],
        [3],
        [4],
    ],
)
def test_random_clifford_circuit(prob_qubits):
    c = random_clifford(5)
    # c.add(gates.M(*MEASURED_QUBITS))
    c.density_matrix = True
    c_copy = c.copy()
    c.add(gates.M(*MEASURED_QUBITS))
    c_copy.add(gates.M(*MEASURED_QUBITS))
    numpy_result = numpy_bkd.execute_circuit(c, nshots=1000)
    clifford_result = clifford_bkd.execute_circuit(c_copy)
    numpy_bkd.assert_allclose(numpy_result.state(), clifford_result.state())
    if not set(prob_qubits).issubset(set(MEASURED_QUBITS)):
        with pytest.raises(RuntimeError) as excinfo:
            numpy_bkd.assert_allclose(
                numpy_result.probabilities(prob_qubits),
                clifford_result.probabilities(prob_qubits),
            )
            assert (
                str(excinfo.value)
                == f"Asking probabilities for qubits {prob_qubits}, but only qubits {MEASURED_QUBITS} were measured."
            )
    else:
        print(f"Measured qubits: {MEASURED_QUBITS}")
        print(f"Probs qubits: {prob_qubits}")
        print(f"Numpy Frequencies: {numpy_result.frequencies()}")
        print(f"Clifford Frequencies: {clifford_result.frequencies()}")
        print(f"Numpy probs: {numpy_result.probabilities(MEASURED_QUBITS)}")
        print(f"Clifford probs: {clifford_result.probabilities()}")
        numpy_bkd.assert_allclose(
            numpy_result.probabilities(prob_qubits),
            clifford_result.probabilities(prob_qubits),
            atol=1e-1,
        )
