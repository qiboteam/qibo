import numpy as np
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
        gates = []
        for gate in self.gates:
            new_qubits = [qubit_map[q] for q in gate.qubits]
            kwargs = gate.init_kwargs
            new_args = [args_map.get(arg, arg) for arg in kwargs.values()]
            gates.append(gate.__class__(*new_qubits, *new_args))
        return gates


def _qibo_gate_name(gate):
    if gate == "cx":
        return "CNOT"
    elif gate == "id":
        return "I"
    elif gate == "ccx":
        return "TOFFOLI"
    else:
        return gate.upper()


class QASMParser:

    def __init__(
        self,
    ):
        self.parser = openqasm3.parser
        self.defined_gates = {}
        self.q_registers, self.c_registers = {}, set()

    def to_circuit(self, qasm_string: str, accelerators=None, density_matrix=False):
        parsed = self.parser.parse(qasm_string)
        gates = []
        self.defined_gates, self.q_registers, self.c_registers = {}, {}, {}

        nqubits = 0
        for statement in parsed.statements:
            if isinstance(statement, openqasm3.ast.QuantumGate):
                [gates.append(gate) for gate in self._get_gate(statement)]
            elif isinstance(statement, openqasm3.ast.QuantumMeasurementStatement):
                gates.append(self._get_measurement(statement))
            elif isinstance(statement, openqasm3.ast.QubitDeclaration):
                q_name, q_size = self._get_qubit(statement)
                self.q_registers.update(
                    {q_name: list(range(nqubits, nqubits + q_size))}
                )
                nqubits += q_size
            elif isinstance(statement, openqasm3.ast.QuantumGateDefinition):
                self._def_gate(statement)
            elif isinstance(statement, openqasm3.ast.Include):
                continue
            elif isinstance(statement, openqasm3.ast.ClassicalDeclaration):
                name = statement.identifier.name
                size = statement.type.size.value
                self.c_registers.update({name: size})
            else:
                raise_error(RuntimeError, f"Unsupported {type(statement)} statement.")
        c = qibo.Circuit(nqubits, accelerators, density_matrix)
        for gate in gates:
            c.add(gate)
        return c

    def _get_measurement(self, measurement):
        qubit = self._get_qubit(measurement.measure.qubit)
        register = measurement.target.name.name
        if register not in self.c_registers:
            raise_error(ValueError, f"Undefined measurement register `{register}`.")
        ind = measurement.target.indices[0][0].value
        if ind >= self.c_registers[register]:
            raise_error(
                IndexError, f"Index `{ind}` is out of bounds of register `{register}`."
            )
        return getattr(qibo.gates, "M")(qubit, register_name=register)

    def _get_qubit(self, qubit):
        if isinstance(qubit, openqasm3.ast.QubitDeclaration):
            return qubit.qubit.name, qubit.size.value
        if not isinstance(qubit, openqasm3.ast.Identifier):
            return self.q_registers[qubit.name.name][qubit.indices[0][0].value]
        else:
            return qubit.name

    def _get_gate(self, gate):
        qubits = [self._get_qubit(q) for q in gate.qubits]
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
            try:
                gates = self.defined_gates.get(gate.name.name).get_gates(
                    qubits, init_args
                )
            except:
                raise_error(
                    ValueError, f"Invalid gate declaration at span: {gate.span}"
                )
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
        qubits = [self._get_qubit(q) for q in definition.qubits]
        args = [self._unroll_expression(expr) for expr in definition.arguments]
        gates = [g for gate in definition.body for g in self._get_gate(gate)]
        self.defined_gates.update({name: DefinedGate(gates, qubits, args)})
