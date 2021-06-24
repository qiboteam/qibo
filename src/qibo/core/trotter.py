import itertools
from qibo.models import Circuit


def reduce_terms(terms):
    orders = {}
    for term in terms:
        if len(term) in orders:
            orders[len(term)].append(term)
        else:
            orders[len(term)] = [term]

    reduced_terms = []
    for order in sorted(orders.keys())[::-1]:
        for child in orders[order]:
            flag = True
            for i, parent in enumerate(reduced_terms):
                if set(child.target_qubits).issubset(parent.target_qubits):
                    reduced_terms[i] = parent.merge(child)
                    flag = False
                    break
            if flag:
                reduced_terms.append(child)
    return reduced_terms


class TrotterCircuit:

    def __init__(self, nqubits, terms, dt, accelerators, memory_device):
        self.gates = {}
        self.dt = dt
        self.circuit = Circuit(nqubits, accelerators=accelerators, memory_device=memory_device)
        reduced_terms = reduce_terms(terms)
        for term in itertools.chain(reduced_terms, reduced_terms[::-1]):
            gate = term.expgate(dt / 2.0)
            self.gates[gate] = term
            self.circuit.add(gate)

    def set_dt(self, dt):
        params = {gate: term.exp(dt / 2.0) for gate, term in self.gates.items()}
        self.dt = dt
        self.circuit.set_parameters(params)
