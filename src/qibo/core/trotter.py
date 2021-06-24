import itertools
from qibo.models import Circuit


class TrotterCircuit:

    def __init__(self, nqubits, terms, dt, accelerators, memory_device):
        self.gates = {}
        self.dt = dt
        self.circuit = Circuit(nqubits, accelerators=accelerators,
                               memory_device=memory_device)
        for term in itertools.chain(terms, terms[::-1]):
            gate = term.expgate(dt / 2.0)
            self.gates[gate] = term
            self.circuit.add(gate)

    def set_dt(self, dt):
        params = {gate: term.exp(dt / 2.0) for gate, term in self.gates.items()}
        self.dt = dt
        self.circuit.set_parameters(params)
