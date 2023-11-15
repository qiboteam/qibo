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


@pytest.mark.parametrize("gate", ["CNOT", "CZ", "SWAP", "iSWAP", "ECR"])
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


def test_random_clifford_circuit():
    for _ in range(4):
        measured_qubits = np.random.choice(range(5), size=3, replace=False)
        print("----------------- Numpy -------------------")
        c = random_clifford(5)
        c.add(gates.M(*measured_qubits))
        # for q in np.random.choice(range(5), size=3, replace=False):
        #    c.add(gates.M(q))
        result = numpy_bkd.execute_circuit(c)
        print(result.frequencies())
        print(result.probabilities(measured_qubits[:2]))
        print("----------------- Clifford -------------------")
        c = random_clifford(5)
        c.add(gates.M(*measured_qubits))
        result = clifford_bkd.execute_circuit(c)
        # print(result.samples())
        print(result.frequencies())
        print(result.probabilities(measured_qubits[:2]))
        assert False
