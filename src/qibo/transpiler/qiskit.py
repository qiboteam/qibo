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
        circuit = circuit.to_qasm(extended_compatibility=True)
        # convert to qiskit circuit
        circuit = QuantumCircuit.from_qasm_str(circuit)
        # transpile
        circuit = self.pass_manager.run(circuit)
        # convert back to qasm
        circuit = dumps(circuit)
        # rebuild the qibo circuit
        return Circuit.from_qasm(circuit, **init_kwargs)
