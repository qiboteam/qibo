from qibo import Circuit, gates
from qibo.transpiler.abstract import Router
from qibo.transpiler.router import ConnectivityError


# TODO: split into routing plus placer steps
class StarConnectivity(Router):
    """Transforms an arbitrary circuit to one that can be executed on hardware.

    This transpiler produces a circuit that respects the following connectivity:

             q
             |
        q -- q -- q
             |
             q

    by adding SWAP gates when needed.

    Args:
        connectivity (:class:`networkx.Graph`): chip connectivity, not used for this transpiler.
        middle_qubit (int, optional): qubit id of the qubit that is in the middle of the star.
    """

    def __init__(self, connectivity=None, middle_qubit: int = 2):
        self.middle_qubit = middle_qubit
        self.connectivity = connectivity

    def __call__(self, circuit: Circuit, initial_layout=None):
        """Apply the transpiler transformation on a given circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): The original Qibo circuit to transform.
                This circuit must contain up to two-qubit gates.

        Returns:
            (:class:`qibo.models.circuit.Circuit`, list): circuit that performs the same operation
                as the original but respects the hardware connectivity,
                and list that maps logical to hardware qubits.
        """

        middle_qubit = self.middle_qubit
        # find the number of qubits for hardware circuit
        if circuit.nqubits == 1:
            nqubits = 1
        else:
            nqubits = max(circuit.nqubits, middle_qubit + 1)
        # new circuit object that will be compatible with hardware connectivity
        new = circuit.__class__(nqubits)
        # list to maps logical to hardware qubits
        hardware_qubits = list(range(nqubits))

        # find initial qubit mapping
        for i, gate in enumerate(circuit.queue):
            if len(gate.qubits) == 2:
                if middle_qubit not in gate.qubits:
                    new_middle = _find_connected_qubit(
                        gate.qubits, circuit.queue[i + 1 :], hardware_qubits
                    )
                    hardware_qubits[middle_qubit], hardware_qubits[new_middle] = (
                        hardware_qubits[new_middle],
                        hardware_qubits[middle_qubit],
                    )
                break

        # the first SWAP is not needed as it can be applied via virtual mapping
        add_swap = False
        for i, gate in enumerate(circuit.queue):
            # map gate qubits to hardware
            qubits = tuple(hardware_qubits.index(q) for q in gate.qubits)
            if isinstance(gate, gates.M):
                new_gate = gates.M(*qubits, **gate.init_kwargs)
                new_gate.result = gate.result
                new.add(new_gate)
                continue

            if len(qubits) > 2:
                raise ConnectivityError(
                    "Gates targeting more than two qubits are not supported."
                )

            elif len(qubits) == 2 and middle_qubit not in qubits:
                # find which qubit should be moved to 0
                new_middle = _find_connected_qubit(
                    qubits, circuit.queue[i + 1 :], hardware_qubits
                )
                # update hardware qubits according to the swap
                hardware_qubits[middle_qubit], hardware_qubits[new_middle] = (
                    hardware_qubits[new_middle],
                    hardware_qubits[middle_qubit],
                )
                if add_swap:
                    new.add(gates.SWAP(middle_qubit, new_middle))
                # update gate qubits according to the new swap
                qubits = tuple(hardware_qubits.index(q) for q in gate.qubits)

            # add gate to the hardware circuit
            if isinstance(gate, gates.Unitary):
                # gates.Unitary requires matrix as first argument
                matrix = gate.init_args[0]
                new.add(gate.__class__(matrix, *qubits, **gate.init_kwargs))
            else:
                new.add(gate.__class__(*qubits, **gate.init_kwargs))
            if len(qubits) == 2:
                add_swap = True
        hardware_qubits_keys = ["q" + str(i) for i in range(5)]
        return new, dict(zip(hardware_qubits_keys, hardware_qubits))


def _find_connected_qubit(qubits, queue, hardware_qubits):
    """Helper method for :meth:`qibo.transpiler.StarConnectivity`.

    Finds which qubit should be mapped to hardware middle qubit
    by looking at the two-qubit gates that follow.
    """
    possible_qubits = set(qubits)
    for next_gate in queue:
        if len(next_gate.qubits) == 2:
            possible_qubits &= {hardware_qubits.index(q) for q in next_gate.qubits}
            if not possible_qubits:
                # freedom of choice
                return qubits[0]
            elif len(possible_qubits) == 1:
                return possible_qubits.pop()
    # freedom of choice
    return qubits[0]
