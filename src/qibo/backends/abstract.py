import abc
from qibo.config import raise_error
from qibo.states import CircuitResult
from qibo.gates.abstract import ParametrizedGate, SpecialGate
from qibo.gates.channels import Channel


class Backend(abc.ABC):

    def __init__(self):
        self.name = "backend"
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
    def calculate_probabilities(self, result): # pragma: no cover
        raise_error(NotImplementedError)


class Simulator(Backend):

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
    def cast(self, x, copy=False):
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

    @abc.abstractmethod
    def collapse_state(self, gate, state, nqubits):
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def collapse_density_matrix(self, gate, state, nqubits):
        raise_error(NotImplementedError)

    def execute_circuit(self, circuit, initial_state=None, nshots=None):
        # TODO: Implement callbacks
        if circuit.accelerators and not self.supports_multigpu:
            raise_error(NotImplementedError, f"{self} does not support distributed execution.")

        if isinstance(initial_state, CircuitResult):
            initial_state = initial_state.state()

        if circuit.repeated_execution:
            return self.execute_circuit_repeated(circuit, initial_state, nshots)

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

    def execute_circuit_repeated(self, circuit, initial_state=None, nshots=None):
        from qibo.gates.measurements import M
        from qibo.gates.channels import Channel
        results = []
        nqubits = circuit.nqubits
        for _ in range(nshots):
            if circuit.density_matrix:
                if initial_state is None:
                    state = self.zero_density_matrix(nqubits)
                else:
                    state = self.cast(initial_state, copy=True)
                
                for gate in circuit.queue:
                    if isinstance(gate, Channel):
                        state = self.apply_channel_density_matrix(gate, state, nqubits)
                    elif isinstance(gate, M):
                        state = self.collapse_density_matrix(gate, state, nqubits)
                    else:
                        if gate.symbolic_parameters:
                            gate.substitute_symbols()
                        state = self.apply_gate_density_matrix(gate, state, nqubits)

                if circuit.measurement_gate:
                    result = CircuitResult(self, circuit, state, 1)
                    results.append(result.samples(binary=False)[0])
                else:
                    results.append(state)

            else:
                if initial_state is None:
                    state = self.zero_state(nqubits)
                else:
                    state = self.cast(initial_state, copy=True)
                
                for gate in circuit.queue:
                    if isinstance(gate, Channel):
                        state = self.apply_channel(gate, state, nqubits)
                    elif isinstance(gate, M):
                        state = self.collapse_state(gate, state, nqubits)
                    else:
                        if gate.symbolic_parameters:
                            gate.substitute_symbols()
                        state = self.apply_gate(gate, state, nqubits)
                
                if circuit.measurement_gate:
                    result = CircuitResult(self, circuit, state, 1)
                    results.append(result.samples(binary=False)[0])
                else:
                    results.append(state)

        if circuit.measurement_gate:
            final_result = CircuitResult(self, circuit, state, nshots)
            final_result._samples = self.aggregate_shots(results)
            circuit._final_state = final_result
            return final_result
        else:
            circuit._final_state = CircuitResult(self, circuit, results[-1], nshots)
            return results

    def get_state_repr(self, result):
        return result.symbolic()

    def get_state_tensor(self, result):
        return result.execution_result

    @abc.abstractmethod
    def calculate_symbolic(self, state, nqubits, decimals=5, cutoff=1e-10, max_terms=20): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_symbolic_density_matrix(self, state, nqubits, decimals=5, cutoff=1e-10, max_terms=20): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def partial_trace(self, state, qubits, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def partial_trace_density_matrix(self, state, qubits, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_probabilities(self, state, qubits, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_probabilities_density_matrix(self, state, qubits, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def set_seed(self, seed): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def sample_shots(self, probabilities, nshots): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def aggregate_shots(self, shots): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def samples_to_binary(self, samples, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def samples_to_decimal(self, samples, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def sample_frequencies(self, probabilities, nshots): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_frequencies(self, samples): # pragma: no cover
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def assert_allclose(self, value, target, rtol=1e-7, atol=0.0): # pragma: no cover
        raise_error(NotImplementedError)

    def assert_circuitclose(self, circuit, target_circuit, rtol=1e-7, atol=0.0):
        value = self.execute_circuit(circuit)
        target = self.execute_circuit(target_circuit)
        self.assert_allclose(value, target)

    @abc.abstractmethod
    def test_regressions(self, name):  # pragma: no cover
        """Correct outcomes for tests that involve random numbers.

        The outcomes of such tests depend on the backend.
        """
        raise_error(NotImplementedError)
