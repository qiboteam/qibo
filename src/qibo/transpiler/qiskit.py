from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps
from qiskit.transpiler import PassManager

from qibo.models.circuit import Circuit


@dataclass
class QiskitPasses:
    """A wrapper to qiskit's transpiler, passes through QASM to convert
    circuits: qibo -> qiskit -> transpile -> qibo.

    Args:
        pass_manager (qiskit.transpiler.PassManager): a qiskit ``PassManager``.
    """

    pass_manager: PassManager

    def __call__(self, circuit: Circuit) -> Circuit:
        init_kwargs = circuit.init_kwargs
        # convert to qasm
        qasm = circuit.to_qasm(extended_compatibility=True)
        # convert to qiskit circuit
        qiskit_circuit = QuantumCircuit.from_qasm_str(qasm)
        # transpile
        qiskit_circuit = self.pass_manager.run(qiskit_circuit)
        # convert back to qasm
        qasm = dumps(qiskit_circuit)
        # rebuild the qibo circuit
        return Circuit.from_qasm(qasm, **init_kwargs)
