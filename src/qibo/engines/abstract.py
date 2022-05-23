import abc
from qibo.config import raise_error
from qibo.gates.abstract import ParametrizedGate, SpecialGate


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

    @abc.abstractmethod
    def execute_circuit(self, circuit, nshots=None):
        raise_error(NotImplementedError)


class Simulator(Engine):

    def __init__(self):
        super().__init__()
        self.name = "simulator"
        
        self.precision = "double"
        self.dtype = "complex128"
        self.matrices = None

        self.device = "/CPU:0"
        self.nthreads = 1

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

    def set_device(self, device):
        if device != "/CPU:0":
            raise_error(ValueError, f"Device {device} is not available for {self} backend.")

    @abc.abstractmethod
    def set_threads(self, nthreads):
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def to_numpy(self, x):
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def zero_state(self, nqubits):
        """Generate |000...0> state as an array."""
        raise_error(NotImplementedError)

    def asmatrix(self, gate):
        """Convert a gate to its matrix representation in the computational basis."""
        name = gate.__class__.__name__
        if isinstance(gate, ParametrizedGate):
            return getattr(self.matrices, name)(*gate.parameters)
        elif isinstance(gate, SpecialGate):
            return self.asmatrix_special(gate)
        else:
            return getattr(self.matrices, name)

    @abc.abstractmethod
    def asmatrix_special(self, gate):
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def control_matrix(self, gate):
        """"Calculate full matrix representation of a controlled gate."""
        raise_error(NotImplementedError)

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

    @abc.abstractmethod
    def assert_allclose(self, value, target, rtol=1e-7, atol=0.0): # pragma: no cover
        raise_error(NotImplementedError)