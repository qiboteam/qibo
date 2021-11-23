"""Test style-qGAN model defined in `qibo/models/qgan.py`."""
import pytest
from qibo import gates
from qibo.models import Circuit, StyleQGAN


def test_qgan_default_init(backend):
    oracle = Circuit(5 + 1)
    oracle.add(gates.X(5).controlled_by(*range(5)))
    superposition = Circuit(5)
    superposition.add([gates.H(i) for i in range(5)])
    grover = Grover(oracle, superposition_circuit=superposition)
    assert grover.oracle == oracle
    assert grover.superposition == superposition
    assert grover.sup_qubits == 5
    assert grover.sup_size == 32
    assert not grover.iterative
    grover = Grover(oracle, superposition_circuit=superposition, superposition_size=int(2**5))
    assert grover.oracle == oracle
    assert grover.superposition == superposition
    assert grover.sup_qubits == 5
    assert grover.sup_size == 32
    assert not grover.iterative
    
def test_qgan_custom_init(backend):
    oracle = Circuit(5 + 1)
    oracle.add(gates.X(5).controlled_by(*range(5)))
    # try to initialize without passing `superposition_qubits`
    with pytest.raises(ValueError):
        grover = Grover(oracle)

    grover = Grover(oracle, superposition_qubits=4)
    assert grover.oracle == oracle
    assert grover.sup_qubits == 4
    assert grover.sup_size == 16
    assert grover.superposition.depth == 1
    assert grover.superposition.ngates == 4
