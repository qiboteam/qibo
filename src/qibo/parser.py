import openqasm3

import qibo
from qibo.config import raise_error


class DefinedGate:

    def __init__(self, gates, qubits, args):
        self.gates = gates
        self.qubits = qubits
        self.args = args

    def get_gates(self, qubits, args):
        qubit_map = dict(zip(self.qubits, qubits))
        args_map = dict(zip(self.args, args))
        # print(qubit_map)
        # print(args_map)
        gates = []
        for gate in self.gates:
            new_qubits = [qubit_map[q] for q in gate.qubits]
            kwargs = gate.init_kwargs
            kwargs.pop("trainable", None)
            new_args = [args_map[arg] for arg in kwargs.values()]
            gates.append(gate.__class__(*new_qubits, *new_args))
        return gates


def _qibo_gate_name(gate):
    if gate == "cx":
        return "CNOT"
    else:
        return gate.upper()


class QASMParser:

    def __init__(
        self,
    ):
        self.parser = openqasm3.parser
        self.defined_gates = {}
        self.registers = {}

    def to_circuit(self, qasm_string: str):
        parsed = self.parser.parse(qasm_string)
        gates, measurements = [], []
        self.defined_gates, self.registers, nqubits = {}, {}, 0
        for statement in parsed.statements:
            # print(f"--------------- Statement -------------\n{statement}\n----------------------------------------------------")
            if isinstance(statement, openqasm3.ast.QuantumGate):
                [gates.append(gate) for gate in self._get_gate(statement)]
            elif isinstance(statement, openqasm3.ast.QuantumMeasurementStatement):
                measurements.append(self._get_measurement(statement))
            elif isinstance(statement, openqasm3.ast.QubitDeclaration):
                q_name, q_size = self._get_qubit(statement)
                self.registers.update({q_name: list(range(nqubits, nqubits + q_size))})
                nqubits += q_size
            elif isinstance(statement, openqasm3.ast.QuantumGateDefinition):
                self._def_gate(statement)
            elif isinstance(statement, openqasm3.ast.Include):
                continue
            else:
                raise_error(RuntimeError, f"Unsupported {type(statement)} statement.")
        c = qibo.Circuit(nqubits)
        for gate in gates:
            c.add(gate)
        print(c.draw())

    def _get_measurement(measurement):
        return getattr(qibo.gates, "M")(measurement.measure.qubit)

    def _get_qubit(self, qubit):
        if isinstance(qubit, openqasm3.ast.QubitDeclaration):
            return qubit.qubit.name, qubit.size.value
        if not isinstance(qubit, openqasm3.ast.Identifier):
            return self.registers[qubit.name.name][qubit.indices[0][0].value]
        else:
            return qubit.name

    def _get_gate(self, gate):
        print(gate.name.name)
        qubits = [self._get_qubit(q) for q in gate.qubits]
        # print(qubits)
        init_args = []
        for arg in gate.arguments:
            arg = self._unroll_expression(arg)
            try:
                arg = eval(arg.replace("pi", "np.pi"))
            except:
                pass
            init_args.append(arg)
        try:
            gates = [
                getattr(qibo.gates, _qibo_gate_name(gate.name.name))(
                    *qubits, *init_args
                )
            ]
        except:
            gates = self.defined_gates.get(gate.name.name).get_gates(qubits, init_args)
        return gates

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
                        val = self._unroll_expression(getattr(expr, attr))
                    except:
                        continue
                    expr_dict[attr] += str(val)
                return "".join(list(expr_dict.values()))

    def _def_gate(self, definition):
        name = definition.name.name
        print(f"> Def gate {name}")
        qubits = [self._get_qubit(q) for q in definition.qubits]
        args = [self._unroll_expression(expr) for expr in definition.arguments]
        gates = [self._get_gate(g)[0] for g in definition.body]
        self.defined_gates.update({name: DefinedGate(gates, qubits, args)})
