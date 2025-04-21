from typing import Optional, Union

from qibo import Circuit, gates
from qibo.config import raise_error
from qibo.gates import Gate
from qibo.transpiler._exceptions import BlockingError


class Block:
    """A block contains a subset of gates acting on two qubits.

    Args:
        qubits (tuple): qubits where the block is acting.
        gates (list): list of gates that compose the block.
        name (str or int, optional): name of the block. Defaults to ``None``.
    """

    def __init__(
        self, qubits: tuple, gates: list, name: Optional[Union[str, int]] = None
    ):
        self.qubits = qubits
        self.gates = gates
        self.name = name

    @property
    def entangled(self):
        """Returns ``True`` if the block contains two-qubit gates."""
        return self._count_2q_gates() > 0

    def add_gate(self, gate: Gate):
        """Add a new gate to the block.

        Args:
            gate (:class:`qibo.gates.abstract.Gate`): gate to be added.
        """
        if not set(gate.qubits).issubset(self.qubits):
            raise_error(
                BlockingError,
                f"Gate acting on qubits {gate.qubits} can't be added "
                + f"to block acting on qubits {self._qubits}.",
            )
        self.gates.append(gate)

    def _count_2q_gates(self):
        """Return the number of two qubit gates in the block."""
        return _count_2q_gates(self.gates)

    @property
    def qubits(self):
        """Returns a sorted tuple with qubits of the block."""
        return tuple(sorted(self._qubits))

    @qubits.setter
    def qubits(self, qubits):
        self._qubits = qubits

    def fuse(self, block, name: Optional[str] = None):
        """Fuses the current block with a new one, the qubits they are acting on must coincide.

        Args:
            block (:class:`qibo.transpiler.blocks.Block`): block to fuse.
            name (str, optional): name of the fused block. Defaults to ``None``.

        Return:
            (:class:`qibo.transpiler.blocks.Block`): fusion of the two input blocks.
        """
        if not self.qubits == block.qubits:
            raise_error(
                BlockingError, "In order to fuse two blocks their qubits must coincide."
            )
        return Block(qubits=self.qubits, gates=self.gates + block.gates, name=name)

    def on_qubits(self, new_qubits: tuple):
        """Return a new block acting on the new qubits.

        Args:
            new_qubits (tuple): new qubits where the block is acting.
        """
        qubits_dict = dict(zip(self.qubits, new_qubits))
        new_gates = [gate.on_qubits(qubits_dict) for gate in self.gates]
        return Block(qubits=new_qubits, gates=new_gates, name=self.name)

    # TODO: use real QM properties to check commutation
    def commute(self, block):
        """Check if a block commutes with the current one.

        Args:
            block (:class:`qibo.transpiler.blocks.Block`): block to check commutation.

        Return:
            True if the two blocks don't share any qubit.
            False otherwise.
        """
        if len(set(self.qubits).intersection(block.qubits)) > 0:
            return False
        return True

    # TODO
    def kak_decompose(self):  # pragma: no cover
        """Return KAK decomposition of the block.
        This should be done only if the block is entangled and the number of
        two qubit gates is higher than the number after the decomposition.
        """
        raise_error(NotImplementedError, "KAK decomposition is not available yet.")


class CircuitBlocks:
    """A CircuitBlocks contains a quantum circuit decomposed in two qubits blocks.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit to be decomposed.
        index_names (bool, optional): assign names to the blocks. Defaults to ``False``.
    """

    def __init__(self, circuit: Circuit, index_names: bool = False):
        self.block_list = block_decomposition(circuit)
        self._index_names = index_names
        if index_names:
            for index, block in enumerate(self.block_list):
                block.name = index
        self.qubits = circuit.nqubits

    def __call__(self):
        return self.block_list

    def search_by_index(self, index: int):
        """Find a block from its index, requires index_names to be ``True``."""
        if not self._index_names:
            raise_error(
                BlockingError,
                "You need to assign index names in order to use search_by_index.",
            )
        for block in self.block_list:
            if block.name == index:
                return block
        raise_error(BlockingError, f"No block found with index {index}.")

    def add_block(self, block: "Block"):
        """Add a two qubits block."""
        if not set(block.qubits).issubset(range(self.qubits)):
            raise_error(
                BlockingError,
                "The block can't be added to the circuit because it acts on different qubits",
            )
        self.block_list.append(block)

    def circuit(self, circuit_kwargs=None):
        """Return the quantum circuit.

        Args:
            circuit_kwargs (dict): original circuit init_kwargs.
        """
        if circuit_kwargs is None:
            circuit = Circuit(self.qubits)
        else:
            circuit = Circuit(**circuit_kwargs)
        for block in self.block_list:
            for gate in block.gates:
                circuit.add(gate)
        return circuit

    def remove_block(self, block: "Block"):
        """Remove a block from the circuit blocks."""
        try:
            self.block_list.remove(block)
        except ValueError:
            raise_error(
                BlockingError,
                "The block you are trying to remove is not present in the circuit blocks.",
            )

    def return_last_block(self):
        """Return the last block in the circuit blocks."""
        if len(self.block_list) == 0:
            raise_error(
                BlockingError,
                "No blocks found in the circuit blocks.",
            )
        return self.block_list[-1]


def block_decomposition(circuit: Circuit, fuse: bool = True):
    """Decompose a circuit into blocks of gates acting on two qubits.
    Break measurements on multiple qubits into measurements of single qubit.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit to be decomposed.
        fuse (bool, optional): fuse adjacent blocks acting on the same qubits.

    Return:
        (list): list of blocks that act on two qubits.
    """
    if circuit.nqubits < 2:
        raise_error(
            BlockingError,
            "Only circuits with at least two qubits can be decomposed with block_decomposition.",
        )

    if _check_multi_qubit_measurements(circuit):
        circuit = _split_multi_qubit_measurements(circuit)
    initial_blocks = _initial_block_decomposition(circuit)

    if not fuse:
        return initial_blocks

    blocks = []
    while len(initial_blocks) > 0:
        first_block = initial_blocks[0]
        remove_list = [first_block]
        if len(initial_blocks[1:]) > 0:
            for second_block in initial_blocks[1:]:
                if second_block.qubits == first_block.qubits:
                    first_block = first_block.fuse(second_block)
                    remove_list.append(second_block)
                elif not first_block.commute(second_block):
                    break
        blocks.append(first_block)
        _remove_gates(initial_blocks, remove_list)

    return blocks


def _initial_block_decomposition(circuit: Circuit):
    """Helper method for :meth:'qibo.transpiler.blocks.block_decomposition'.

    Decompose a circuit into blocks of gates acting on two qubits.
    This decomposition is not minimal.

    Args:
        circuit (:class:`qibo.models.Circuit`): circuit to be decomposed.

    Return:
        blocks (list): list of blocks that act on two qubits.
    """
    blocks = []
    all_gates = list(circuit.queue)
    while _count_multi_qubit_gates(all_gates) > 0:
        for idx, gate in enumerate(all_gates):
            if len(gate.qubits) == 2:
                qubits = gate.qubits
                block_gates = _find_previous_gates(all_gates[0:idx], qubits)
                block_gates.append(gate)
                block_gates.extend(_find_successive_gates(all_gates[idx + 1 :], qubits))
                block = Block(qubits=qubits, gates=block_gates)
                _remove_gates(all_gates, block_gates)
                blocks.append(block)
                break
            elif len(gate.qubits) > 2:
                raise_error(
                    BlockingError,
                    "Gates targeting more than 2 qubits are not supported.",
                )

    # Now we need to deal with the remaining spare single qubit gates
    while len(all_gates) > 0:
        first_qubit = all_gates[0].qubits[0]
        block_gates = _gates_on_qubit(gatelist=all_gates, qubit=first_qubit)
        _remove_gates(all_gates, block_gates)
        # Add other single qubits if there are still single qubit gates
        if len(all_gates) > 0:
            second_qubit = all_gates[0].qubits[0]
            second_qubit_block_gates = _gates_on_qubit(
                gatelist=all_gates, qubit=second_qubit
            )
            block_gates += second_qubit_block_gates
            _remove_gates(all_gates, second_qubit_block_gates)
            block = Block(qubits=(first_qubit, second_qubit), gates=block_gates)
        # In case there are no other spare single qubit gates create a block using a following qubit as placeholder
        else:
            block = Block(
                qubits=(first_qubit, (first_qubit + 1) % circuit.nqubits),
                gates=block_gates,
            )
        blocks.append(block)

    return blocks


def _check_multi_qubit_measurements(circuit: Circuit):
    """Helper method for :meth:'qibo.transpiler.blocks.block_decomposition'.

    Return True if the circuit contains measurements acting on multiple qubits."""
    for gate in circuit.queue:
        if isinstance(gate, gates.M) and len(gate.qubits) > 1:
            return True
    return False


def _split_multi_qubit_measurements(circuit: Circuit):
    """Helper method for :meth:'qibo.transpiler.blocks.block_decomposition'.

    Return an equivalent circuit containinig measurements acting only on single qubits.
    """
    new_circuit = Circuit(circuit.nqubits)
    for gate in circuit.queue:
        if isinstance(gate, gates.M) and len(gate.qubits) > 1:
            for qubit in gate.qubits:
                new_circuit.add(gates.M(qubit))
        else:
            new_circuit.add(gate)
    return new_circuit


def _gates_on_qubit(gatelist, qubit):
    """Helper method for :meth:'qibo.transpiler.blocks.block_decomposition'.

    Return a list of all single qubit gates in gatelist acting on a specific qubit."""
    selected_gates = []
    for gate in gatelist:
        if gate.qubits[0] == qubit:
            selected_gates.append(gate)
    return selected_gates


def _remove_gates(gatelist, remove_list):
    """Helper method for :meth:'qibo.transpiler.blocks.block_decomposition'.

    Remove all gates present in remove_list from gatelist."""
    for gate in remove_list:
        gatelist.remove(gate)


def _count_2q_gates(gatelist: list):
    """Helper method for :meth:'qibo.transpiler.blocks.block_decomposition'.

    Return the number of two qubit gates in a list of gates."""
    return len([gate for gate in gatelist if len(gate.qubits) == 2])


def _count_multi_qubit_gates(gatelist: list):
    """Helper method for :meth:'qibo.transpiler.blocks.block_decomposition'.

    Return the number of multi qubit gates in a list of gates."""
    return len([gate for gate in gatelist if len(gate.qubits) >= 2])


def _find_successive_gates(gates_list: list, qubits: tuple):
    """Helper method for :meth:'qibo.transpiler.blocks.block_decomposition'.

    Return a list containing all gates acting on qubits until a new two qubit gate acting on qubits is found.
    """
    successive_gates = []
    for qubit in qubits:
        for gate in gates_list:
            if (len(gate.qubits) == 1) and (gate.qubits[0] == qubit):
                successive_gates.append(gate)
            elif (len(gate.qubits) == 2) and (qubit in gate.qubits):
                break
    return successive_gates


def _find_previous_gates(gates_list: list, qubits: tuple):
    """Helper method for :meth:'qibo.transpiler.blocks.block_decomposition'.

    Return a list containing all gates acting on qubits."""
    previous_gates = []
    for gate in gates_list:
        if gate.qubits[0] in qubits:
            previous_gates.append(gate)
    return previous_gates
