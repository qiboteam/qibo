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
        qubits = [q.name for q in gate.qubits]
        init_args = []
        for arg in gate.arguments:
            init_args.append(eval(self._unroll_expression(arg).replace("pi", "np.pi")))
        try:
            gate = getattr(gates, gate.name.name.upper())
        except:
            gates = getattr(self.defined_gates, gate.name.name)
            for gate in gates:
                self._get_gate(gate)
        return gate(*qubits, *init_args)

    def _unroll_expression(self, expr):
        try:
            return expr.name
        except:
            try:
                return expr.value
            except:
                expr_dict = {}
                for attr in ("lhs", "op", "expression", "rhs"):
                    expr_dict[attr] = ""
                    try:
                        expr_dict[attr] += self._unroll_expression(getattr(expr, attr))
                    except:
                        continue
                return "".join(list(expr_dict.values()))

    def _def_gate(definition):
        name = definition.name.name
        gates = [self._get_gate(g) for g in definition.body]
        self.defined_gates.update({name: gates})
