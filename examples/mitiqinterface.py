from cirq import Circuit as cirqCircuit
from qibo import models
from qibo.models import Circuit as qiboCircuit
from mitiq.interface.mitiq_qiskit.conversions import from_qasm, to_qasm


def from_qibo_to_mitiq(circuit: qiboCircuit) -> cirqCircuit:
    """Returns a Cirq circuit equivalent to the input qibo circuit.

    Args:
        circuit: qibo Circuit to convert to a Cirq circuit.
    """
    c_qasm = models.Circuit.to_qasm(circuit)
    c_mitiq = from_qasm(c_qasm)
    return c_mitiq


def from_mitiq_to_qibo(circuit: cirqCircuit) -> qiboCircuit:
    """Returns a qibo circuit equivalent to the input Cirq circuit.

    Args:
        circuit: Cirq circuit to convert to a qibo circuit.
    ..testcode::
        from qibo import gates
        from mitiq.zne.scaling import fold_gates_from_left

        c = qiboCircuit(2)
        c.add(gates.H(0))
        c.add(gates.X(1))
        c_cirq = from_qibo_to_mitiq(c)
        print(c_cirq)
        folded = fold_gates_from_left(c_cirq, scale_factor=2.)
        print(folded)
        c_folded = from_mitiq_to_qibo(folded)
        print(c_folded.summary())
        #q_0: ───H───

        #q_1: ───X───
        #q_0: ───H───H───H───

        # q_1: ───X───────────
        # Circuit depth = 3
        # Total number of gates = 4
        # Number of qubits = 2
        # Most common gates:
        # h: 3
        # x: 1
    """
    c_qasm = to_qasm(circuit)
    c_qibo = models.Circuit.from_qasm(c_qasm)
    return c_qibo
