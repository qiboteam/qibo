import abc
from qibo.config import raise_error
from qibo.states import CircuitResult
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
    def execute_circuit(self, circuit, nshots=None): # pragma: no cover
        """Executes a circuit."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def get_state_repr(self, result): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def get_state_tensor(self, result): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def get_state_symbolic(self, result, decimals=5, cutoff=1e-10, max_terms=20): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_probabilities(self, result): # pragma: no cover
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
        self.supports_multigpu = False
        self.oom_error = MemoryError

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

    @abc.abstractmethod
    def set_device(self, device):
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def set_threads(self, nthreads):
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def cast(self, x):
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def to_numpy(self, x):
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def zero_state(self, nqubits):
        """Generate |000...0> state as an array."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def zero_density_matrix(self, nqubits):
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

    @abc.abstractmethod
    def apply_gate(self, gate):
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_gate_density_matrix(self, gate):
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_channel(self, gate):
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def apply_channel_density_matrix(self, gate):
        raise_error(NotImplementedError)

    def execute_circuit(self, circuit, initial_state=None, nshots=None):
        # TODO: Implement shots
        # TODO: Implement repeated execution
        # TODO: Implement callbacks
        if circuit.accelerators and not self.supports_multigpu:
            raise_error(NotImplementedError, f"{self} does not support distributed execution.")

        if isinstance(initial_state, CircuitResult):
            initial_state = initial_state.state()

        try:
            nqubits = circuit.nqubits
            if circuit.density_matrix:
                if initial_state is None:
                    state = self.zero_density_matrix(nqubits)
                else:
                    # cast to proper complex type
                    state = self.cast(initial_state)
                
                for gate in circuit.queue:
                    state = self.apply_gate_density_matrix(gate, state, nqubits)
            
            else:
                if initial_state is None:
                    state = self.zero_state(nqubits)
                else:
                    # cast to proper complex type
                    state = self.cast(initial_state)

                for gate in circuit.queue:
                    state = self.apply_gate(gate, state, nqubits)

            # TODO: Consider implementing a final state setter in circuits?
            circuit._final_state = CircuitResult(self, circuit, state, nshots)
            return circuit._final_state
        
        except self.oom_error:
            raise_error(RuntimeError, f"State does not fit in {self.device} memory."
                                       "Please switch the execution device to a "
                                       "different one using ``qibo.set_device``.")

    def get_state_repr(self, result):
        return self.get_state_symbolic(result)

    def get_state_tensor(self, result):
        return result.execution_result

    def get_state_symbolic(self, result, decimals=5, cutoff=1e-10, max_terms=20):
        import numpy as np
        state = self.to_numpy(self.get_state_tensor(result))
        terms = []
        if result.density_matrix:
            indi, indj = K.np.nonzero(state)
            for i, j in zip(indi, indj):
                bi = bin(i)[2:].zfill(self.nqubits)
                bj = bin(j)[2:].zfill(self.nqubits)
                if K.np.abs(state[i, j]) >= cutoff:
                    x = round(state[i, j], decimals)
                    terms.append(f"{x}|{bi}><{bj}|")
                if len(terms) >= max_terms:
                    terms.append("...")
                    break
    
        else:
            terms = []
            for i in np.nonzero(state)[0]:
                b = bin(i)[2:].zfill(result.nqubits)
                if np.abs(state[i]) >= cutoff:
                    x = round(state[i], decimals)
                    terms.append(f"{x}|{b}>")
                if len(terms) >= max_terms:
                    terms.append("...")
                    break
        
        return " + ".join(terms)

    @abc.abstractmethod
    def calculate_probabilities(self, result): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def sample_shots(self, probabilities, nshots):
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def samples_to_binary(self, samples, nqubits):
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def samples_to_decimal(self, samples, nqubits):
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def sample_frequencies(self, probabilities, nshots):
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_frequencies(self, samples):
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def assert_allclose(self, value, target, rtol=1e-7, atol=0.0): # pragma: no cover
        raise_error(NotImplementedError)

    def assert_circuitclose(self, circuit, target_circuit, rtol=1e-7, atol=0.0):
        value = self.execute_circuit(circuit)
        target = self.execute_circuit(target_circuit)
        self.assert_allclose(value, target)
