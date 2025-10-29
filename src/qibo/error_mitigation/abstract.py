"""Abstract Error Mitigation Routine object"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from types import NoneType
from typing import Callable, Iterable, List, Optional, Union

from qibo import Circuit
from qibo.backends import _check_backend, construct_backend, get_transpiler
from qibo.backends.abstract import Backend
from qibo.config import raise_error
from qibo.hamiltonians.abstract import AbstractHamiltonian
from qibo.models.error_mitigation import SIMULATION_BACKEND
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

    regression_model: Optional[Callable] = None
    model_parameters: Optional[Iterable[float]] = None
    simulation_backend: Optional[Backend] = None

    def __post_init__(self):
        super().__post_init__()
        if self.simulation_backend is None:
            self.simulation_backend = construct_backend("numpy")

    @abstractmethod
    def sample_circuit(self, circuit: Optional[Circuit] = None) -> Circuit:
        pass

    def _training_circuits(self, circuit: Optional[Circuit] = None) -> List[Circuit]:
        if circuit is None:
            return self.training_circuits
        return [self.sample_circuit(circuit) for _ in range(self.n_training_samples)]

    @cached_property
    def training_circuits(self) -> List[Circuit]:
        return [
            self.sample_circuit(self.circuit) for _ in range(self.n_training_samples)
        ]

    def regression(
        self,
        training_circuits: List[Circuit],
        observable: Optional[AbstractHamiltonian] = None,
        nshots: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
    ):
        observable = self._observable(observable)
        # first get the noisy expectations running with the current backend
        noisy_circuits = self._circuit_preprocessing(training_circuits, noise_model)
        noisy_expectations = [
            observable.expectation(circuit, nshots=nshots) for circuit in noisy_circuits
        ]
        # then switch to the simulation backend to get the exact values
        original_backend = observable.backend
        observable.backend = self.simulation_backend
        exact_expectations = [
            observable.expectation(circuit, nshots=nshots)
            for circuit in training_circuits
        ]
        # restore the original backend
        observable.backend = original_backend
        # cast to the simulation backend native array
        noisy_expectations = self.simulation_backend.cast(
            noisy_expectations, dtype=exact_expectations[0].dtype
        )
        # do the regression
        optimal_params = self.simulation_backend.curve_fit(
            self.regression_model,
            noisy_expectations,
            exact_expectations,
            self.model_parameters,
        )
        self.model_parameters = optimal_params
        return optimal_params

    def __call__(
        self,
        circuit: Optional[Circuit] = None,
        observable: Optional[AbstractHamiltonian] = None,
        nshots: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
    ):
        training_circuits = self._training_circuits(circuit)
        # if something is different retrain the regression model
        if circuit is not None or observable is not None or noise_model is not None:
            self.regression(training_circuits, observable, nshots, noise_model)
        observable = self._observable(observable)
        circuit = self._circuit(circuit)
        noisy_exp_val = observable.expectation(circuit, nshots=nshots)
        return self.regression_model(noisy_exp_val, *self.model_parameters)
