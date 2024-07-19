import pytest

from qibo import Circuit, gates
from qibo.transpiler._exceptions import BlockingError
from qibo.transpiler.blocks import (
    Block,
    CircuitBlocks,
    _check_multi_qubit_measurements,
    _count_multi_qubit_gates,
    _find_previous_gates,
    _find_successive_gates,
    _gates_on_qubit,
    _initial_block_decomposition,
    _remove_gates,
    _split_multi_qubit_measurements,
    block_decomposition,
)


def assert_gates_equality(gates_1: list, gates_2: list):
    """Check that the gates are the same."""
    for g_1, g_2 in zip(gates_1, gates_2):
        assert g_1.qubits == g_2.qubits
        assert g_1.__class__ == g_2.__class__


def test_count_2q_gates():
    block = Block(qubits=(0, 1), gates=[gates.CZ(0, 1), gates.CZ(0, 1), gates.H(0)])
    assert block._count_2q_gates() == 2


def test_add_gate_and_entanglement():
    block = Block(qubits=(0, 1), gates=[gates.H(0)])
    assert not block.entangled
    block.add_gate(gates.CZ(0, 1))
    assert block.entangled
    assert block._count_2q_gates() == 1


def test_add_gate_error():
    block = Block(qubits=(0, 1), gates=[gates.CZ(0, 1)])
    with pytest.raises(BlockingError):
        block.add_gate(gates.CZ(0, 2))


def test_fuse_blocks():
    block_1 = Block(qubits=(0, 1), gates=[gates.CZ(0, 1)])
    block_2 = Block(qubits=(0, 1), gates=[gates.H(0)])
    fused = block_1.fuse(block_2)
    assert_gates_equality(fused.gates, block_1.gates + block_2.gates)


def test_fuse_blocks_error():
    block_1 = Block(qubits=(0, 1), gates=[gates.CZ(0, 1)])
    block_2 = Block(qubits=(1, 2), gates=[gates.CZ(1, 2)])
    with pytest.raises(BlockingError):
        fused = block_1.fuse(block_2)


@pytest.mark.parametrize("qubits", [(0, 1), (2, 1)])
def test_commute_false(qubits):
    block_1 = Block(qubits=(0, 1), gates=[gates.CZ(0, 1)])
    block_2 = Block(qubits=qubits, gates=[gates.CZ(*qubits)])
    assert not block_1.commute(block_2)


def test_commute_true():
    block_1 = Block(qubits=(0, 1), gates=[gates.CZ(0, 1)])
    block_2 = Block(qubits=(2, 3), gates=[gates.CZ(2, 3)])
    assert block_1.commute(block_2)


def test_count_multi_qubit_gates():
    gatelist = [gates.CZ(0, 1), gates.H(0), gates.TOFFOLI(0, 1, 2)]
    assert _count_multi_qubit_gates(gatelist) == 2


def test_gates_on_qubit():
    gatelist = [gates.H(0), gates.H(1), gates.H(2), gates.H(0)]
    assert_gates_equality(_gates_on_qubit(gatelist, 0), [gatelist[0], gatelist[-1]])
    assert_gates_equality(_gates_on_qubit(gatelist, 1), [gatelist[1]])
    assert_gates_equality(_gates_on_qubit(gatelist, 2), [gatelist[2]])


def test_remove_gates():
    gatelist = [gates.H(0), gates.CZ(0, 1), gates.H(2), gates.CZ(0, 2)]
    remaining = [gates.CZ(0, 1), gates.H(2)]
    delete_list = [gatelist[0], gatelist[3]]
    _remove_gates(gatelist, delete_list)
    assert_gates_equality(gatelist, remaining)


def test_find_previous_gates():
    gatelist = [gates.H(0), gates.H(1), gates.H(2)]
    previous_gates = _find_previous_gates(gatelist, (0, 1))
    assert_gates_equality(previous_gates, gatelist[:2])


def test_find_successive_gates():
    gatelist = [gates.H(0), gates.CZ(2, 3), gates.H(1), gates.H(2), gates.CZ(2, 1)]
    successive_gates = _find_successive_gates(gatelist, (0, 1))
    assert_gates_equality(successive_gates, [gatelist[0], gatelist[2]])


def test_initial_block_decomposition():
    circ = Circuit(5)
    circ.add(gates.H(1))
    circ.add(gates.H(0))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(1, 2))
    circ.add(gates.H(3))
    circ.add(gates.H(4))
    blocks = _initial_block_decomposition(circ)
    assert_gates_equality(blocks[0].gates, [gates.H(1), gates.H(0), gates.CZ(0, 1)])
    assert len(blocks) == 4
    assert len(blocks[0].gates) == 3
    assert len(blocks[1].gates) == 1
    assert blocks[2].entangled
    assert not blocks[3].entangled
    assert len(blocks[3].gates) == 2


def test_check_measurements():
    circ = Circuit(2)
    circ.add(gates.H(1))
    circ.add(gates.M(0, 1))
    assert _check_multi_qubit_measurements(circ)
    circ = Circuit(2)
    circ.add(gates.H(1))
    circ.add(gates.M(0))
    circ.add(gates.M(1))
    assert not _check_multi_qubit_measurements(circ)


def test_split_measurements():
    circ = Circuit(2)
    circ.add(gates.H(1))
    circ.add(gates.M(0, 1))
    new_circ = _split_multi_qubit_measurements(circ)
    assert_gates_equality(new_circ.queue, [gates.H(1), gates.M(0), gates.M(1)])


def test_initial_block_decomposition_measurements():
    circ = Circuit(5)
    circ.add(gates.M(0))
    circ.add(gates.M(1))
    circ.add(gates.H(1))
    circ.add(gates.H(0))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.M(1))
    circ.add(gates.M(3))
    circ.add(gates.H(3))
    circ.add(gates.H(4))
    blocks = _initial_block_decomposition(circ)
    assert_gates_equality(
        blocks[0].gates,
        [gates.M(0), gates.M(1), gates.H(1), gates.H(0), gates.CZ(0, 1)],
    )
    assert_gates_equality(blocks[1].gates, [gates.CZ(0, 1), gates.M(1)])
    assert_gates_equality(blocks[2].gates, [gates.M(3), gates.H(3), gates.H(4)])


def test_initial_block_decomposition_error():
    circ = Circuit(3)
    circ.add(gates.TOFFOLI(0, 1, 2))
    with pytest.raises(BlockingError):
        blocks = _initial_block_decomposition(circ)


def test_block_decomposition_error():
    circ = Circuit(1)
    with pytest.raises(BlockingError):
        block_decomposition(circ)


def test_block_decomposition_no_fuse():
    circ = Circuit(4)
    circ.add(gates.H(1))
    circ.add(gates.H(0))
    circ.add(gates.H(0))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.H(0))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(1, 2))
    circ.add(gates.H(1))
    circ.add(gates.H(3))
    blocks = block_decomposition(circ, fuse=False)
    assert_gates_equality(
        blocks[0].gates,
        [gates.H(1), gates.H(0), gates.H(0), gates.CZ(0, 1), gates.H(0)],
    )
    assert len(blocks) == 4
    assert len(blocks[0].gates) == 5
    assert len(blocks[1].gates) == 1
    assert blocks[2].entangled
    assert not blocks[3].entangled


def test_block_decomposition():
    circ = Circuit(4)
    circ.add(gates.H(1))  # first block
    circ.add(gates.H(0))  # first block
    circ.add(gates.CZ(0, 1))  # first block
    circ.add(gates.H(0))  # first block
    circ.add(gates.CZ(0, 1))  # first block
    circ.add(gates.CZ(1, 2))  # second block
    circ.add(gates.CZ(1, 2))  # second block
    circ.add(gates.H(1))  # second block
    circ.add(gates.H(3))  # 4 block
    circ.add(gates.CZ(0, 1))  # 3 block
    circ.add(gates.CZ(0, 1))  # 3 block
    circ.add(gates.CZ(2, 3))  # 4 block
    circ.add(gates.CZ(0, 1))  # 3 block
    blocks = block_decomposition(circ)
    assert_gates_equality(
        blocks[0].gates,
        [gates.H(1), gates.H(0), gates.CZ(0, 1), gates.H(0), gates.CZ(0, 1)],
    )
    assert len(blocks) == 4
    assert blocks[0]._count_2q_gates() == 2
    assert len(blocks[0].gates) == 5
    assert blocks[0].qubits == (0, 1)
    assert blocks[1]._count_2q_gates() == 2
    assert len(blocks[1].gates) == 3
    assert blocks[3]._count_2q_gates() == 1
    assert len(blocks[3].gates) == 2
    assert blocks[3].qubits == (2, 3)
    assert blocks[2]._count_2q_gates() == 3
    assert len(blocks[2].gates) == 3


def test_block_decomposition_measurements():
    circ = Circuit(4)
    circ.add(gates.H(1))  # first block
    circ.add(gates.H(0))  # first block
    circ.add(gates.CZ(0, 1))  # first block
    circ.add(gates.H(0))  # first block
    circ.add(gates.M(0, 1))  # first block
    circ.add(gates.CZ(1, 2))  # second block
    circ.add(gates.CZ(1, 2))  # second block
    circ.add(gates.H(1))  # second block
    circ.add(gates.H(3))  # 4 block
    circ.add(gates.CZ(0, 1))  # 3 block
    circ.add(gates.CZ(0, 1))  # 3 block
    circ.add(gates.CZ(2, 3))  # 4 block
    circ.add(gates.M(0, 1))  # 3 block
    blocks = block_decomposition(circ)
    assert_gates_equality(
        blocks[0].gates,
        [gates.H(1), gates.H(0), gates.CZ(0, 1), gates.H(0), gates.M(0), gates.M(1)],
    )
    assert len(blocks) == 4
    assert blocks[0]._count_2q_gates() == 1
    assert len(blocks[0].gates) == 6
    assert blocks[0].qubits == (0, 1)
    assert blocks[1]._count_2q_gates() == 2
    assert len(blocks[1].gates) == 3
    assert blocks[3]._count_2q_gates() == 1
    assert len(blocks[3].gates) == 2
    assert blocks[3].qubits == (2, 3)
    assert blocks[2]._count_2q_gates() == 2
    assert len(blocks[2].gates) == 4


def test_circuit_blocks(backend):
    circ = Circuit(4)
    circ.add(gates.H(1))
    circ.add(gates.H(0))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.H(0))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(1, 2))
    circ.add(gates.CZ(1, 2))
    circ.add(gates.H(1))
    circ.add(gates.H(3))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(2, 3))
    circ.add(gates.CZ(0, 1))
    circuit_blocks = CircuitBlocks(circ, index_names=True)
    for index, block in enumerate(circuit_blocks()):
        assert block.name == index
    reconstructed_circ = circuit_blocks.circuit()
    backend.assert_allclose(
        backend.execute_circuit(circ).state(),
        backend.execute_circuit(reconstructed_circ).state(),
    )
    first_block = circuit_blocks.search_by_index(0)
    assert_gates_equality(
        first_block.gates,
        [gates.H(1), gates.H(0), gates.CZ(0, 1), gates.H(0), gates.CZ(0, 1)],
    )


def test_add_block():
    circ = Circuit(4)
    circ.add(gates.H(1))
    circ.add(gates.H(0))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.H(0))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(1, 2))
    circuit_blocks = CircuitBlocks(circ)
    new_block = Block(qubits=(2, 3), gates=[gates.CZ(2, 3)])
    circ.add(gates.CZ(2, 3))
    circuit_blocks.add_block(new_block)
    reconstructed_circ = circuit_blocks.circuit()
    assert_gates_equality(reconstructed_circ.queue, circ.queue)


def test_add_block_error():
    circ = Circuit(2)
    circ.add(gates.CZ(0, 1))
    circuit_blocks = CircuitBlocks(circ)
    new_block = Block(qubits=(2, 3), gates=[gates.CZ(2, 3)])
    with pytest.raises(BlockingError):
        circuit_blocks.add_block(new_block)


def test_remove_block():
    circ = Circuit(3)
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(1, 2))
    circuit_blocks = CircuitBlocks(circ)
    blocks = circuit_blocks()
    circuit_blocks.remove_block(blocks[0])
    remaining_block = circuit_blocks()
    assert_gates_equality(remaining_block[0].gates, [gates.CZ(1, 2)])


def test_remove_block_error():
    circ = Circuit(3)
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(1, 2))
    circuit_blocks = CircuitBlocks(circ)
    new_block = Block(qubits=(2, 3), gates=[gates.CZ(2, 3)])
    with pytest.raises(BlockingError):
        circuit_blocks.remove_block(new_block)


def test_search_by_index_error_no_indexes():
    circ = Circuit(2)
    circ.add(gates.CZ(0, 1))
    circuit_blocks = CircuitBlocks(circ)
    with pytest.raises(BlockingError):
        circuit_blocks.search_by_index(0)


def test_search_by_index_error_no_index_found():
    circ = Circuit(2)
    circ.add(gates.CZ(0, 1))
    circuit_blocks = CircuitBlocks(circ, index_names=True)
    with pytest.raises(BlockingError):
        circuit_blocks.search_by_index(1)


def test_block_on_qubits():
    block = Block(
        qubits=(0, 1),
        gates=[gates.H(0), gates.CZ(0, 1), gates.H(1), gates.CZ(1, 0), gates.M(1)],
    )
    new_block = block.on_qubits(new_qubits=(2, 3))
    assert new_block.gates[0].qubits == (2,)
    assert new_block.gates[1].qubits == (2, 3)
    assert new_block.gates[2].qubits == (3,)
    assert new_block.gates[3].qubits == (3, 2)
    assert new_block.gates[4].qubits == (3,)


####
import numpy as np
from qibo import gates
from qibo.transpiler.blocks import Block
import pytest


@pytest.mark.parametrize("gates", [[], [gates.CNOT(0, 1)]])
def test_block_kak_error(gates):
    """
    Test the KAK decomposition when KAK is not needed.
    """
    block = Block(qubits=(0, 1), gates=gates)
    with pytest.raises(BlockingError):
        block.kak_decompose()


# No ERROR
@pytest.mark.parametrize("gates", [[gates.H(0), gates.S(1), gates.TDG(0), gates.CNOT(0, 1), gates.CNOT(1, 0)]])

# ==== TODO: ERROR =====
# @pytest.mark.parametrize("gates", [[gates.H(0), gates.S(1), gates.CNOT(1, 0), gates.TDG(0), gates.CNOT(0, 1), gates.CNOT(1, 0)]])
def test_block_kak(gates):
    """
    Test the KAK decomposition of a block.
    """
    # Create a block with a sequence of gates
    block = Block(qubits=(0, 1), gates=gates)

    # Obtain the unitary matrix of the block
    U = block._unitary()

    # Perform the KAK decomposition
    A0, A1, K, B0, B1 = block.kak_decompose()

    # Reassemble the decomposed matrices
    orig = (np.kron(A0, A1) @ heisenberg_unitary(K) @ np.kron(B0.conj().T, B1.conj().T))

    # Check that the original and decomposed matrices are equal
    assert ( np.linalg.norm(orig - U) < 1e-8 )



# TODO: Move this helper function to a more appropriate location.
def heisenberg_unitary(k):
    """
    Generate the heisenberg unitary from parameters k.
    """
    from scipy.linalg import expm

    PAULI_X = np.array([[0, 1], [1, 0]])
    PAULI_Y = np.array([[0, -1j], [1j, 0]])
    PAULI_Z = np.array([[1, 0], [0, -1]])

    PXX = np.kron(PAULI_X, PAULI_X)
    PYY = np.kron(PAULI_Y, PAULI_Y)
    PZZ = np.kron(PAULI_Z, PAULI_Z)

    H = k[0] * np.eye(4) + k[1] * PXX + k[2] * PYY + k[3] * PZZ
    return expm(1j * H)
