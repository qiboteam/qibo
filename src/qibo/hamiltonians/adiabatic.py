from abc import ABC, abstractmethod
from itertools import chain

from qibo.config import raise_error
from qibo.hamiltonians import hamiltonians, terms


class AdiabaticHamiltonian(ABC):
    """Constructor for Hamiltonians used in adiabatic evolution.

    This object is never constructed, it falls back to one of
    :class:`qibo.core.adiabatic.BaseAdiabaticHamiltonian` or
    :class:`qibo.core.adiabatic.SymbolicAdiabaticHamiltonian`.

    These objects allow constructing the adiabatic Hamiltonian of the
    form ``(1 - s) * H0 + s * H1`` efficiently.
    """

    def __new__(cls, h0, h1):
        if type(h1) != type(h0):
            raise_error(
                TypeError,
                f"h1 should be of the same type {type(h0)} of h0 but is {type(h1)}.",
            )
        if isinstance(h0, hamiltonians.Hamiltonian):
            return BaseAdiabaticHamiltonian(h0, h1)
        elif isinstance(h0, hamiltonians.SymbolicHamiltonian):
            return SymbolicAdiabaticHamiltonian(h0, h1)
        else:
            raise_error(
                TypeError,
                f"h0 should be a hamiltonians.Hamiltonian object but is {type(h0)}.",
            )

    def __init__(self, h0, h1):  # pragma: no cover
        self.h0, self.h1 = h0, h1
        self.schedule = None
        self.total_time = None

    @abstractmethod
    def ground_state(self):  # pragma: no cover
        """Returns the ground state of the ``H0`` Hamiltonian.

        Usually used as the initial state for adiabatic evolution.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def __call__(self, t):  # pragma: no cover
        """Hamiltonian object corresponding to the given time."""
        raise_error(NotImplementedError)

    @abstractmethod
    def circuit(self, dt, accelerators=None, t=0):  # pragma: no cover
        raise_error(NotImplementedError)


class BaseAdiabaticHamiltonian:
    """Adiabatic Hamiltonian that is a sum of :class:`qibo.hamiltonians.hamiltonians.Hamiltonian`."""

    def __init__(self, h0, h1):
        if h0.nqubits != h1.nqubits:
            raise_error(
                ValueError,
                f"H0 has {h0.nqubits} qubits while H1 has {h1.nqubits}.",
            )
        self.nqubits = h0.nqubits
        if h0.backend != h1.backend:  # pragma: no cover
            raise_error(ValueError, "H0 and H1 have different backends.")
        self.backend = h0.backend
        self.h0, self.h1 = h0, h1
        self.schedule = None
        self.total_time = None

    def ground_state(self):
        return self.h0.ground_state()

    def __call__(self, t):
        """Hamiltonian object corresponding to the given time.

        Returns:
            A :class:`qibo.hamiltonians.hamiltonians.Hamiltonian` object corresponding
            to the adiabatic Hamiltonian at a given time.
        """
        if t == 0:
            return self.h0
        if self.total_time is None or self.schedule is None:
            raise_error(
                RuntimeError,
                "Cannot access adiabatic evolution "
                "Hamiltonian before setting the "
                "the total evolution time and "
                "scheduling.",
            )
        st = self.schedule(t / self.total_time)  # pylint: disable=E1102
        return self.h0 * (1 - st) + self.h1 * st

    def circuit(self, dt, accelerators=None, t=0):  # pragma: no cover
        raise_error(
            NotImplementedError,
            "Trotter circuit is not available " "for full matrix Hamiltonians.",
        )


class SymbolicAdiabaticHamiltonian(BaseAdiabaticHamiltonian):
    """Adiabatic Hamiltonian that is sum of :class:`qibo.hamiltonians.hamiltonians.SymbolicHamiltonian`."""

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

    def circuit(self, dt, accelerators=None, t=0):
        """Circuit that implements the Trotterized evolution under the adiabatic Hamiltonian.

        Args:
            dt (float): Time step to use for exponentiation of the Hamiltonian.
            accelerators (dict): Dictionary with accelerators for distributed
                circuits.
            t (float): Time that the Hamiltonian should be calculated.

        Returns:
            :class:`qibo.models.Circuit`: Circuit implementing the Trotterized evolution.
        """
        from qibo import Circuit  # pylint: disable=import-outside-toplevel
        from qibo.hamiltonians.terms import (  # pylint: disable=import-outside-toplevel
            TermGroup,
        )

        # pylint: disable=E1102
        st = self.schedule(t / self.total_time) if t != 0 else 0
        # pylint: enable=E1102
        coefficients = {self.h0: 1 - st, self.h1: st}

        groups = self.groups
        circuit = Circuit(self.nqubits, accelerators=accelerators)
        circuit.add(
            group.to_term(coefficients).expgate(dt / 2.0)
            for group in chain(groups, groups[::-1])
        )

        return circuit
