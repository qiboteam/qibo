from qibo import K
from qibo.core.circuit import Circuit as StateCircuit
from typing import Dict, Optional


class Circuit(StateCircuit):
    """"""

    def __new__(cls, nqubits, accelerators=None, density_matrix=False):
        circuit_cls = K.circuit_class(accelerators, density_matrix)
        if accelerators is not None:
            return circuit_cls(nqubits, accelerators=accelerators)
        return circuit_cls(nqubits)

    @classmethod
    def from_qasm(cls, qasm_code, accelerators=None, density_matrix=False):
        circuit_cls = K.circuit_class(accelerators, density_matrix)
        if accelerators is not None:
            return circuit_cls.from_qasm(qasm_code, accelerators=accelerators)
        return circuit_cls.from_qasm(qasm_code)