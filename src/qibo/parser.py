import openqasm3

from qibo import gates
from qibo.config import raise_error


class QASMParser:

    def __init__(
        self,
    ):
        self.parser = openqasm3.parser
        self.defined_gates = {}

    def to_circuit(self, qasm_string: str):
        parsed = self.parser.parse(qasm_string)
        gates, measurements = [], []
        self.defined_gates, qubits = {}, {}
        for statement in parsed.statements:
            if isinstance(statement, openqasm3.ast.QuantumGate):
                gates.append(self._get_gate(statement))
            elif isinstance(statement, openqasm3.ast.QuantumMeasurementStatement):
                measurements.append(self._get_measurement(statement))
            elif isinstance(statement, openqasm3.ast.QubitDeclaration):
                qubits.update(self._get_qubits(statement))
            elif isinstance(statement, openqasm3.ast.QuantumGateDefinition):
                self._def_gate(statement)
            else:
                raise_error(RuntimeError, f"Unsupported {type(statement)} statement.")

    def _get_measurement(measurement):
        return getattr(gates, "M")(measurement.measure.qubit)

    def _get_qubit(qubit):
        return {qubit.name: qubit.size.value}

    def _get_gate(self, gate):
        init_args = {"q": gate.qubits.name}
        for arg in gate.arguments:
            init_args.update({arg.lhs.name: arg.rhs.value})
        try:
            return getattr(gates, gate.name.name.upper())(**init_args)
        except:
            return getattr(self.defined_gates, gate.name.name)(**init_args)

    def _def_gate(definition):
        name = definition.name.name
        gates = [self._get_gate(g) for g in definition.body]
        self.defined_gates.update({name: gates})
