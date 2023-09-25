import numpy as np
import pytest

from qibo import gates


def test_callback_gate_errors(backend):
    from qibo import callbacks

    entropy = callbacks.EntanglementEntropy([0])
    gate = gates.CallbackGate(entropy)
    with pytest.raises(NotImplementedError):
        gate.on_qubits(2)
    with pytest.raises(NotImplementedError):
        gate.matrix(backend)


@pytest.mark.parametrize("nqubits", [2, 3])
def test_fused_gate_construct_unitary(backend, nqubits):
    gate = gates.FusedGate(0, 1)
    gate.append(gates.H(0))
    gate.append(gates.H(1))
    gate.append(gates.CZ(0, 1))
    hmatrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    czmatrix = np.diag([1, 1, 1, -1])
    target_matrix = czmatrix @ np.kron(hmatrix, hmatrix)
    if nqubits > 2:
        gate.append(gates.TOFFOLI(0, 1, 2))
        toffoli = np.eye(8)
        toffoli[-2:, -2:] = np.array([[0, 1], [1, 0]])
        target_matrix = toffoli @ np.kron(target_matrix, np.eye(2))
    backend.assert_allclose(gate.matrix(backend), target_matrix)
