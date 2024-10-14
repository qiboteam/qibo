from math import prod

from qibo import Circuit, gates, symbols
from qibo.config import raise_error
from qibo.hamiltonians import SymbolicHamiltonian


def Expectation(circuit: Circuit, observable: SymbolicHamiltonian) -> float:
    if len(circuit.measurements) > 0:
        raise_error(RuntimeError)
    exp_val = 0.0
    for term in observable.terms:
        Z_observable = SymbolicHamiltonian(
            prod([symbols.Z(q) for q in term.target_qubits]),
            nqubits=circuit.nqubits,
            backend=observable.backend,
        )
        measurements = [
            gates.M(factor.target_qubit, basis=factor.gate.__class__)
            for factor in term.factors
        ]
        circ_copy = circuit.copy(True)
        [circ_copy.add(m) for m in measurements]
        freq = observable.backend.execute_circuit(circ_copy).frequencies()
        exp_val += Z_observable.expectation_from_samples(freq)
    return exp_val
