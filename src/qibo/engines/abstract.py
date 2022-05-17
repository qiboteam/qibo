import abc


class Engine(abc.ABC):

    def __init__(self):
        self.name = "engine"

    def __repr__(self):
        return self.name

    @abc.abstractmethod
    def apply_gate(self, gate):
        pass


class Simulator(Engine):

    def __init__(self):
        self.name = "simulator"

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