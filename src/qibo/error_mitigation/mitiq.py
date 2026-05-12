from dataclasses import dataclass
from types import NoneType
from typing import Callable, Dict, Optional, Union

import mitiq

from qibo.config import raise_error
from qibo.error_mitigation.abstract import (
    ErrorMitigationRoutine,
    ReadoutMitigationRoutine,
)
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models.circuit import Circuit
from qibo.noise import NoiseModel


@dataclass
class Mitiq(ErrorMitigationRoutine):

    method: str = None
    method_kwargs: Optional[Dict] = None
    _run_method: str = None

    def __post_init__(self):
        if self.method is None:
            raise_error(RuntimeError, "Missing mandatory `method` argument.")
        self._execute_method = f"execute_with_{self.method}"
        self.method = getattr(getattr(mitiq, self.method), self.method)
        if self.method_kwargs is None:
            self.method_kwargs = {}

    @property
    def _execute(self) -> Callable:
        return getattr(self.method, self._execute_method)

    def _qibo_observable_to_mitiq(
        self, observable: SymbolicHamiltonian
    ) -> mitiq.Observable:
        pauli_strings = []
        for coefficient, pauli, qubits in zip(*observable.simple_terms):
            pauli_string = ""
            for q in range(max(qubits)):
                if q in qubits:
                    index = qubits.index(q)
                    pauli_string += pauli[index]
                else:
                    pauli_string += "I"
            pauli_strings.append(
                mitiq.PauliString(spec=pauli_string, coeff=coefficient)
            )
        return mitiq.Observable(*pauli_strings)

    def _observable(
        self, observable: Union[SymbolicHamiltonian, NoneType]
    ) -> Union[SymbolicHamiltonian, NoneType]:
        observable = super()._observable(observable)
        return self._qibo_observable_to_mitiq(observable)

    def __call__(
        self,
        circuit: Optional[Circuit] = None,
        observable: Optional[SymbolicHamiltonian] = None,
        nshots: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
        **mitiq_kwargs,
    ):
        breakpoint()
        circuit = self._circuit_preprocessing([self._circuit(circuit)], noise_model)[0]
        #observable = self._observable(observable)
        kwargs = self.method_kwargs.copy()
        kwargs.update(mitiq_kwargs)
        #if observable is not None:
        #    kwargs["observable"] = observable
        return self._execute(circuit, **kwargs)
