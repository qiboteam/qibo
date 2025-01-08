"""Qibo wrapper for QASM 3.0 parser."""

from itertools import repeat
from typing import Union

import numpy as np
import openqasm3

import qibo
from qibo.config import raise_error
from qibo.gates import FusedGate


class CustomQASMGate:
    """Object that handles the definition of custom gates in QASM via the `gate` command.

    Args:
        gates (list): List of gates composing the defined gate.
        qubits (list or tuple): Qubits identifiers (e.g. (q0, q1, q2, ...)).
        args (list or tuple): Arguments identifiers (e.g. (theta, alpha, gamma, ...)).
    """

    def __init__(
        self,
        name: str,
        gates: list,
        qubits: Union[list, tuple],
        args: Union[list, tuple],
    ):
        self.name = name
        self.gates = gates
        self.qubits = qubits
        self.args = args

    def get_gate(self, qubits: Union[list, tuple], args: Union[list, tuple]):
        """Returns the gates composing the defined gate applied on the
        specified qubits with the specified ``args`` as a unique :class:`qibo.gates.special.FusedGate`.

        Args:
            qubits (list or tuple): Qubits where to apply the gates.
            args (list or tuple): Arguments to evaluate the gates on.

        Returns:
            :class:`qibo.gates.special.FusedGate`: the composed gate evaluated on the input qubits with the input arguments.
        """
        if len(self.args) != len(args):
            raise_error(
                ValueError,
                f"Invalid `args` argument passed to the user-defined gate `{self.name}` upon construction. {args} was passed but something of the form {self.args} is expected.",
            )
        elif len(self.qubits) != len(qubits):
            raise_error(
                ValueError,
                f"Invalid `qubits` argument passed to the user-defined gate `{self.name}` upon construction. {qubits} was passed but something of the form {self.qubits} is expected.",
            )
        qubit_map = dict(zip(self.qubits, qubits))
        args_map = dict(zip(self.args, args))
        return self._construct_fused_gate(self.gates, qubits, qubit_map, args_map)

    def _construct_fused_gate(self, gates, qubits, qubit_map, args_map):
        """Constructs a :class:`qibo.gates.special.FusedGate` out of the provided list of gates on the specified qubits.

        Args:
            gates (list(:class:`qibo.gates.Gate`)): List of gates to build the fused gate from.
            qubits (list(int)): List of qubits to construct the gate on.
            qubit_map (dict): Mapping between the placeholders for the qubits contained in `gates` and the actual qubits indices to apply them on.
            args_map (dict): Mapping between the placeholders for the kwargs contained in `gates` and the actual kwargs values.

        Returns:
            (qibo.gates.special.FusedGate): The resulting fused gate.
        """
        fused_gate = FusedGate(*qubits)
        for gate in gates:
            if not isinstance(gate, FusedGate):
                new_qubits, new_args = self._compile_gate_qubits_and_args(
                    gate, qubit_map, args_map
                )
                fused_gate.append(gate.__class__(*new_qubits, *new_args))
            else:
                qubits = [qubit_map[q] for q in gate.qubits]
                fused_gate.append(
                    self._construct_fused_gate(gate.gates, qubits, qubit_map, args_map)
                )
        return fused_gate

    def _compile_gate_qubits_and_args(self, gate, qubit_map, args_map):
        """Compile the qubits and arguments placeholders contained in the input gate with their actual values.

        Args:
            gate (:class:`qibo.gates.Gate`): The input gate containing the qubits and arguments placeholders.
            qubit_map (dict): Mapping between the placeholders for the qubits contained in `gate` and the actual qubits indices to apply them on.
            args_map (dict): Mapping between the placeholders for the kwargs contained in `gate` and the actual kwargs values.

        Returns:
            tuple(list, list): The compiled qubits and arguments.
        """
        new_qubits = [qubit_map[q] for q in gate.qubits]
        new_args = [args_map.get(arg, arg) for arg in gate.init_kwargs.values()]
        return new_qubits, new_args


def _qibo_gate_name(gate):
    if gate == "cx":
        return "CNOT"

    if gate == "id":
        return "I"

    if gate == "ccx":
        return "TOFFOLI"

    if gate in ["u", "U"]:
        # `u` for QASM 2.0
        # `U` for QASM 3.0
        return "U3"

    return gate.upper()


class QASMParser:
    """Wrapper around the :class:`openqasm3.parser` for QASM 3.0."""

    def __init__(
        self,
    ):
        self.parser = openqasm3.parser
        self.defined_gates = {}
        self.q_registers = {}
        self.c_registers = set()

    def to_circuit(
        self, qasm_string: str, accelerators: dict = None, density_matrix: bool = False
    ):
        """Converts a QASM program into a :class:`qibo.models.Circuit`.

        Args:
            qasm_string (str): QASM program.
            accelerators (dict, optional): Maps device names to the number of times each
                device will be used. Defaults to ``None``.
            density_matrix (bool, optional): If ``True``, the constructed circuit would
                evolve density matrices.

        Returns:
            :class:`qibo.models.Circuit`: circuit constructed from QASM string.
        """
        parsed = self.parser.parse(qasm_string)
        gates = []
        self.defined_gates, self.q_registers, self.c_registers = {}, {}, {}

        nqubits = 0
        for statement in parsed.statements:
            if isinstance(statement, openqasm3.ast.QuantumGate):
                gates.append(self._get_gate(statement))
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
                self.c_registers.update({name: list(range(size))})
            else:
                raise_error(RuntimeError, f"Unsupported {type(statement)} statement.")
        circ = qibo.Circuit(
            nqubits,
            accelerators,
            density_matrix,
            wire_names=self._construct_wire_names(),
        )
        circ.add(self._merge_measurements(gates))
        return circ

    def _get_measurement(self, measurement):
        """Converts a :class:`openqasm3.ast.QuantumMeasurementStatement` statement
        into :class:`qibo.gates.measurements.M`."""
        qubit = self._get_qubit(measurement.measure.qubit)
        register = measurement.target.name.name
        if register not in self.c_registers:
            raise_error(ValueError, f"Undefined measurement register `{register}`.")
        ind = measurement.target.indices[0][0].value
        if ind >= len(self.c_registers[register]):
            raise_error(
                IndexError, f"Index `{ind}` is out of bounds of register `{register}`."
            )
        self.c_registers[register][ind] = qubit
        return getattr(qibo.gates, "M")(qubit, register_name=register)

    def _get_qubit(self, qubit):
        """Extracts the qubit from a :class:`openqasm3.ast.QubitDeclaration` statement."""
        if isinstance(qubit, openqasm3.ast.QubitDeclaration):
            return qubit.qubit.name, qubit.size.value

        if not isinstance(qubit, openqasm3.ast.Identifier):
            return self.q_registers[qubit.name.name][qubit.indices[0][0].value]

        return qubit.name

    def _get_gate(self, gate):
        """Converts a :class:`openqasm3.ast.QuantumGate` statement
        into :class:`qibo.gates.Gate`."""
        qubits = [self._get_qubit(q) for q in gate.qubits]
        init_args = []
        for arg in gate.arguments:
            arg = self._unroll_expression(arg)
            try:
                arg = eval(arg.replace("pi", "np.pi"))
            except:
                pass
            init_args.append(arg)
        # check whether the gate exists in qibo.gates already
        if _qibo_gate_name(gate.name.name) in dir(qibo.gates):
            try:
                gate = getattr(qibo.gates, _qibo_gate_name(gate.name.name))(
                    *qubits, *init_args
                )
            # the gate exists in qibo.gates but invalid construction
            except TypeError:
                raise_error(
                    ValueError, f"Invalid gate declaration at span: {gate.span}"
                )
        # check whether the gate was defined by the user
        elif gate.name.name in self.defined_gates:
            try:
                gate = self.defined_gates.get(gate.name.name).get_gate(
                    qubits, init_args
                )
            # the gate exists in self.defined_gates but invalid construction
            except ValueError:
                raise_error(
                    ValueError, f"Invalid gate declaration at span: {gate.span}"
                )
        # undefined gate
        else:
            raise_error(ValueError, f"Undefined gate at span: {gate.span}")
        return gate

    def _unroll_expression(self, expr):
        """Unrolls an argument definition expression to retrieve the
        complete argument as a string."""
        # check whether the expression is a simple string, e.g. `pi` or `theta`
        if "name" in dir(expr):
            return expr.name
        # check whether the expression is a single value, e.g. `0.1234`
        if "value" in dir(expr):
            return expr.value
        # the expression is composite, e.g. `2*pi` or `3*theta/2`
        expr_dict = {}
        for attr in ("lhs", "op", "expression", "rhs"):
            expr_dict[attr] = ""
            if attr in dir(expr):
                val = self._unroll_expression(getattr(expr, attr))
                expr_dict[attr] += str(val)

        return "".join(list(expr_dict.values()))

    def _def_gate(self, definition):
        """Converts a :class:`openqasm3.ast.QuantumGateDefinition` statement
        into :class:`qibo.parser.CustomQASMGate` object."""
        name = definition.name.name
        qubits = [self._get_qubit(q) for q in definition.qubits]
        args = [self._unroll_expression(expr) for expr in definition.arguments]
        gates = [self._get_gate(gate) for gate in definition.body]
        self.defined_gates.update({name: CustomQASMGate(name, gates, qubits, args)})

    def _merge_measurements(self, gates):
        """Merges separated measurements of a same register into a single one.
        This is needed because qibo doesn't allow to separetely define two measurements in a same register:

        # not allowed
        c.add(gates.M(0, register="m0"))
        c.add(gates.M(1, register="m0"))
        """
        updated_queue = []
        for gate in gates:
            if isinstance(gate, qibo.gates.M):
                if gate.register_name in self.c_registers:
                    updated_queue.append(
                        qibo.gates.M(
                            *self.c_registers.pop(gate.register_name),
                            register_name=gate.register_name,
                        )
                    )
            else:
                updated_queue.append(gate)
        return updated_queue

    def _construct_wire_names(self):
        """Builds the wires names from the declared quantum registers."""
        wire_names = []
        for reg_name, reg_qubits in self.q_registers.items():
            wires = sorted(
                zip(repeat(reg_name, len(reg_qubits)), reg_qubits), key=lambda x: x[1]
            )
            for wire in wires:
                wire_names.append(f"{wire[0]}{wire[1]}")
        return wire_names
