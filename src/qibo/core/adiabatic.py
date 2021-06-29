from abc import ABC, abstractmethod
from qibo.config import raise_error
from qibo.core import hamiltonians, terms


class AdiabaticHamiltonian(ABC):

    def __new__(cls, h0, h1):
        if type(h1) != type(h0):
            raise_error(TypeError, "h1 should be of the same type {} of h0 but "
                                   "is {}.".format(type(h0), type(h1)))
        if isinstance(h0, hamiltonians.Hamiltonian):
            return BaseAdiabaticHamiltonian(h0, h1)
        elif isinstance(h0, hamiltonians.SymbolicHamiltonian):
            return SymbolicAdiabaticHamiltonian(h0, h1)
        else:
            raise_error(TypeError, "h0 should be a hamiltonians.Hamiltonian "
                                   "object but is {}.".format(type(h0)))

    def __init__(self, h0, h1): # pragma: no cover
        self.h0, self.h1 = h0, h1
        self.schedule = None
        self.total_time = None

    @abstractmethod
    def ground_state(self): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def __call__(self, t): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def circuit(self, dt, accelerators=None, memory_device="/CPU:0", t=0): # pragma: no cover
        raise_error(NotImplementedError)


class BaseAdiabaticHamiltonian:

    def __init__(self, h0, h1):
        if h0.nqubits != h1.nqubits:
            raise_error(ValueError, "H0 has {} qubits while H1 has {}."
                                    "".format(h0.nqubits, h1.nqubits))
        self.nqubits = h0.nqubits
        self.h0, self.h1 = h0, h1
        self.schedule = None
        self.total_time = None

    def ground_state(self):
        return self.h0.ground_state()

    def __call__(self, t):
        if t == 0:
            return self.h0
        if self.total_time is None or self.schedule is None:
            raise_error(RuntimeError, "Cannot access adiabatic evolution "
                                      "Hamiltonian before setting the "
                                      "the total evolution time and "
                                      "scheduling.")
        st = self.schedule(t / self.total_time) # pylint: disable=E1102
        return self.h0 * (1 - st) + self.h1 * st

    def circuit(self, dt, accelerators=None, memory_device="/CPU:0", t=0): # pragma: no cover
        raise_error(NotImplementedError, "Trotter circuit is not available "
                                         "for full matrix Hamiltonians.")


class TrotterCircuit(hamiltonians.TrotterCircuit):

    def set(self, dt, coefficients):
        params = {gate: group.to_term(coefficients).exp(dt / 2.0) for gate, group in self.gates.items()}
        self.dt = dt
        self.circuit.set_parameters(params)


class SymbolicAdiabaticHamiltonian(BaseAdiabaticHamiltonian):

    def __init__(self, h0, h1):
        super().__init__(h0, h1)
        self.trotter_circuit = None
        self.groups0 = terms.TermGroup.from_terms(self.h0.terms)
        self.groups1 = terms.TermGroup.from_terms(self.h1.terms)
        all_terms = []
        for group in self.groups0:
            for term in group:
                term.hamiltonian = self.h0
                all_terms.append(term)
        for group in self.groups1:
            for term in group:
                term.hamiltonian = self.h1
                all_terms.append(term)
        self.groups = terms.TermGroup.from_terms(all_terms)

    def circuit(self, dt, accelerators=None, memory_device="/CPU:0", t=0):
        if self.trotter_circuit is None:
            self.trotter_circuit = TrotterCircuit(self.groups, dt, self.nqubits,
                                                  accelerators, memory_device)
        st = self.schedule(t / self.total_time) if t != 0 else 0 # pylint: disable=E1102
        coefficients = {self.h0: 1 - st, self.h1: st}
        self.trotter_circuit.set(dt, coefficients)
        return self.trotter_circuit.circuit
