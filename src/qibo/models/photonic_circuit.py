from .circuit import Circuit
import perceval as pcvl
from qibo.gates.photonic_gates import PhotonicGate
from qibo.backends import _Global


class PhotonicCircuit(Circuit):

    def __init__(self, nmodes):
        super().__init__(nmodes)
        self._exp = pcvl.Experiment(nmodes)
        self._wire_type = "mode"

    def add(self, gate):
        if not isinstance(gate, PhotonicGate):
            raise TypeError("only photonic gates are supported in a photonic circuit")

        super().add(gate)
        self._exp.add(gate.wires, gate.photonic_component)

    def set_input_state(self, input_state: tuple[int, ...]):
        if len(input_state) != self.nqubits:
            raise ValueError(f"input state must have length {self.nqubits}")
        for v in input_state:
            if v < 0:
                raise ValueError("input state must have positive value")

        self._exp.with_input(pcvl.BasicState(input_state))

    def __call__(self, initial_state=None, nshots=1000):
        self.set_input_state(initial_state)
        backend = _Global.backend()
        return backend.execute_circuit(self._exp, initial_state, nshots)

    def to_pcvl(self):
        """Convert the circuit to a perceval experiment."""
        code = f"Circuit({self.nqubits})"

        for gate in self.queue:
            wires = gate.wires
            code += f"\n    .add({wires[0]}, {gate.__class__.__name__}({', '.join(map(str, gate.init_args[len(wires):]))}))"

        return code