import abc
from qibo.config import raise_error


class Engine(abc.ABC):

    def __init__(self):
        self.name = "engine"
        self.platform = None

    def __repr__(self):
        if self.platform is None:
            return self.name
        else:
            return f"{self.name} ({self.platform})"

    @abc.abstractmethod
    def apply_gate(self, gate):
        raise_error(NotImplementedError)


class Simulator(Engine):

    def __init__(self):
        super().__init__()
        self.name = "simulator"
        self.device = "/CPU:0"
        self.precision = "double"
        self.dtype = "complex128"
        # object that contains gate matrices
        self.matrices = None

    def get_precision(self):
        return self.precision

    def set_precision(self, precision):
        if precision != self.precision:
            if precision == "single":
                self.precision = precision
                self.dtype = "complex64"
            elif precision == "double":
                self.precision = precision
                self.dtype = "complex128"
            else:
                raise_error(ValueError, f"Unknown precision {precision}.")
            if self.matrices:
                self.matrices = self.matrices.__class__(self.dtype)

    def get_device(self):
        return self.device

    def set_device(self, device):
        if device != "/CPU:0":
            raise_error(ValueError, f"Device {device} is not available for {self} backend.")

    def asmatrix(self, gate):
        name = gate.__class__.__name__
        if gate.parameters:
            return getattr(self.matrices, name)(*gate.parameters)
        else:
            return getattr(self.matrices, name)

    @abc.abstractmethod
    def zero_state(self, nqubits):
        """Generate |000...0> state as an array."""
        pass

    def execute_circuit(self, circuit, initial_state=None, nshots=None):
        # TODO: Implement shots
        # TODO: Implement repeated execution
        # TODO: Implement callbacks
        # TODO: Implement density matrices
        nqubits = circuit.nqubits
        if initial_state is None:
            state = self.zero_state(nqubits)
        for gate in circuit.queue:
            state = self.apply_gate(gate, state, nqubits)
        # TODO: Consider implementing a final state setter in circuits?
        circuit._final_state = state
        return state