"""Abstract Error Mitigation Routine object"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import NoneType
from typing import List, Optional, Union

from qibo import Circuit
from qibo.backends import _check_backend, get_transpiler
from qibo.backends.abstract import Backend
from qibo.config import raise_error
from qibo.hamiltonians.abstract import AbstractHamiltonian
from qibo.noise import NoiseModel
from qibo.transpiler import Passes


@dataclass
class ErrorMitigationRoutine(ABC):

    circuit: Optional[Circuit] = None
    observable: Optional[AbstractHamiltonian] = None
    noise_model: Optional[NoiseModel] = None
    transpiler: Optional[Passes] = None
    backend: Optional[Backend] = None

    def __post_init__(
        self,
    ):
        self.backend = _check_backend(self.backend)
        if self.transpiler is None:
            self.transpiler = get_transpiler()

    def _circuit(self, circuit: Circuit) -> Circuit:
        if circuit is None:
            if self.circuit is None:
                raise_error(
                    RuntimeError,
                    "No circuit provided, please either initialize the the mitigation routine with a circuit, or provide it upon `__call__.`",
                )
            return self.circuit
        return circuit

    def _noise_model(
        self, noise_model: Union[NoiseModel, NoneType]
    ) -> Union[NoiseModel, NoneType]:
        if noise_model is None:
            return self.noise_model
        return noise_model

    def _observable(
        self, observable: Union[AbstractHamiltonian, NoneType]
    ) -> Union[AbstractHamiltonian, NoneType]:
        if observable is None:
            if self.observable is None:
                raise_error(
                    RuntimeError,
                    "No observable provided, please either initialize the the mitigation routine with an observable, or provide it upon `__call__.`",
                )
            return self.observable
        return observable

    @abstractmethod
    def __call__(
        self,
        circuit: Circuit,
        observable: AbstractHamiltonian,
        nshots: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
    ):
        pass

    def _circuit_preprocessing(
        self, circuits: List[Circuit], noise_model: Optional[NoiseModel] = None
    ) -> List[Circuit]:
        noise = self._noise_model(noise_model)
        new_circuits = []
        for circ in circuits:
            if noise is not None:
                circ = noise.apply(circ)
            circ, _ = self.transpiler(circ)
            new_circuits.append(circ)
        return new_circuits


@dataclass
class DataRegressionErrorMitigation(ErrorMitigationRoutine):

    @abstractmethod
    def sample_circuit(self, circuit: Optional[Circuit] = None):
        pass

    def regression():
        pass
