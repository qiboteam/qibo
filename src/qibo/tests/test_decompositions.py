import numpy as np
import pytest
import cirq
from qibo import gates
from qibo.models import Circuit


def random_initial_state(nqubits, dtype=np.complex128):
    """Generates a random normalized state vector."""
    x = np.random.random(2 ** nqubits) + 1j * np.random.random(2 ** nqubits)
    return (x / np.sqrt((np.abs(x) ** 2).sum())).astype(dtype)


def test_x_decomposition_gates():
    gate = gates.X(7).controlled_by(0, 1, 2, 3, 4)
    qibo_decomp = gate.decompose(5, 6)

    qubits = [cirq.LineQubit(i) for i in range(8)]
    controls = qubits[:5]
    free = qubits[5:-1]
    cirq_decomp = cirq.decompose_multi_controlled_x(controls, qubits[-1], free)

    for x, y in zip(qibo_decomp, cirq_decomp):
        print(x, y)
        # TODO: Parse cirq gate attributes from str(y)
    assert False


@pytest.mark.skip
def test_x_decomposition_execution():
    gate = gates.X(7).controlled_by(0, 1, 2, 3, 4)
    init_state = random_initial_state(8)

    targetc = Circuit(8)
    targetc.add(gate)
    target_state = targetc(np.copy(init_state)).numpy()

    c = Circuit(8)
    c.add(gate.decompose(5, 6))
    final_state = c(np.copy(init_state)).numpy()

    np.testing.assert_allclose(final_state, target_state)
