from qibo import gates
from qibo.config import log, raise_error
from qibo.transpiler.abstract import Router


def find_connected_qubit(qubits, queue, hardware_qubits):
    """Helper method for :meth:`qibolab.transpilers.fix_connecivity`.

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
        connectivity (networkx.Graph): chip connectivity, not used for this transpiler.
        middle_qubit (int): qubit id of the qubit that is in the middle of the star.
        verbose (bool): print info messages.
    """

    def __init__(self, connectivity=None, middle_qubit=2, verbose=False):
        self.middle_qubit = middle_qubit
        self.verbose = verbose

    def tlog(self, message):
        """Print messages only if ``verbose`` was set to ``True``."""
        if self.verbose:
            log.info(message)

    def is_satisfied(self, circuit):
        """Checks if a circuit respects connectivity constraints.

        Args:
            circuit (qibo.models.Circuit): Circuit model to check.
            middle_qubit (int): Hardware middle qubit.
            verbose (bool): If ``True`` it prints debugging log messages.

        Returns ``True`` if the following conditions are satisfied:
            - Circuit does not contain more than two-qubit gates.
            - All two-qubit gates have qubit 0 as target or control.

            otherwise returns ``False``.
        """
        for gate in circuit.queue:
            if len(gate.qubits) > 2 and not isinstance(gate, gates.M):
                self.tlog(f"{gate.name} acts on more than two qubits.")
                return False
            elif len(gate.qubits) == 2:
                if self.middle_qubit not in gate.qubits:
                    self.tlog(
                        "Circuit does not respect connectivity. "
                        f"{gate.name} acts on {gate.qubits}."
                    )
                    return False

        self.tlog("Circuit respects connectivity.")
        return True

    def __call__(self, circuit, initial_layout=None):
        """Apply the transpiler transformation on a given circuit.

        Args:
            circuit (qibo.models.Circuit): The original Qibo circuit to transform.
                This circuit must contain up to two-qubit gates.

        Returns:
            new (qibo.models.Circuit): Qibo circuit that performs the same operation
                as the original but respects the hardware connectivity.
            hardware_qubits (list): List that maps logical to hardware qubits.
                This is required for transforming final measurements.
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
                    new_middle = find_connected_qubit(
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
                raise_error(
                    NotImplementedError,
                    "Transpiler does not support gates targeting more than two-qubits.",
                )

            elif len(qubits) == 2 and middle_qubit not in qubits:
                # find which qubit should be moved to 0
                new_middle = find_connected_qubit(
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
                from qibo.backends import NumpyBackend

                backend = NumpyBackend()
                matrix = gate.matrix(backend)
                new.add(gate.__class__(matrix, *qubits, **gate.init_kwargs))
            else:
                new.add(gate.__class__(*qubits, **gate.init_kwargs))
            if len(qubits) == 2:
                add_swap = True
        hardware_qubits_keys = ["q" + str(i) for i in range(5)]
        return new, dict(zip(hardware_qubits_keys, hardware_qubits))
