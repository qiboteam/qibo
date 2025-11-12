"""Abstract Error Mitigation Routine object"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from types import NoneType
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
from scipy.optimize import curve_fit

from qibo import Circuit
from qibo.backends import construct_backend, get_transpiler
from qibo.backends.abstract import Backend
from qibo.config import raise_error
from qibo.hamiltonians.abstract import AbstractHamiltonian
from qibo.measurements import MeasurementResult
from qibo.noise import NoiseModel
from qibo.transpiler import Passes


@dataclass
class ReadoutMitigationRoutine(ABC):

    nshots: Optional[int] = None
    _backend: Union[Backend, NoneType] = None
    _noise_model: NoiseModel = None
    _nqubits: int = None

    @property
    def backend(self):
        if self._backend is None:
            raise_error(RuntimeError, "Backend not initialized yet.")
        return self._backend

    @backend.setter
    def backend(self, new_backend: Backend):
        self._backend = new_backend

    @property
    def noise_model(self):
        if self._noise_model is None:
            raise_error(RuntimeError, "NoiseModel not initialized yet.")
        return self._noise_model

    @property
    def nqubits(self):
        if self._nqubits is None:
            raise_error(RuntimeError, "nqubits not initialized yet.")
        return self._nqubits

    @abstractmethod
    def __call__(self, measurement_result: MeasurementResult) -> MeasurementResult:
        pass

    @staticmethod
    def binary_to_integer_keys(
        frequencies: Dict[str, int | float],
    ) -> Dict[int, int | float]:
        return {int(key, 2): value for key, value in frequencies.items()}

    @staticmethod
    def integer_to_binary_keys(
        frequencies: Dict[int, int | float], nqubits: int
    ) -> Dict[str, int | float]:
        return {f"{i:0{nqubits}b}": value for i, value in enumerate(frequencies)}


class MitigatedMeasurementResult(MeasurementResult):
    pass


@dataclass
class ErrorMitigationRoutine(ABC):

    circuit: Optional[Circuit] = None
    observable: Optional[AbstractHamiltonian] = None
    noise_model: Optional[NoiseModel] = None
    transpiler: Optional[Passes] = None
    readout_mitigation: Optional[ReadoutMitigationRoutine] = None

    def __post_init__(
        self,
    ):
        if self.transpiler is None:
            self.transpiler = get_transpiler()
        if self.readout_mitigation is not None:
            self.readout_mitigation._noise_model = self.noise_model
            self.readout_mitigation._backend = self.backend

    @property
    def backend(self):
        if self.observable is not None:
            return self.observable.backend
        raise_error(RuntimeError, "No observable defined yet, no backend available.")

    def _circuit(self, circuit: Circuit) -> Circuit:
        if circuit is None:
            if self.circuit is None:
                raise_error(
                    RuntimeError,
                    "No circuit provided, please either initialize the the mitigation routine with a circuit, or provide it upon call.",
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
                    "No observable provided, please either initialize the the mitigation routine with an observable, or provide it upon call.",
                )
            return self.observable
        # this is mostly useful to have a consistent backend available
        self.observable = observable
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

    n_training_samples: Optional[int] = 50
    regression_model: Optional[Callable] = None
    model_parameters: Optional[Iterable[float]] = None
    simulation_backend: Optional[Backend] = None
    _is_trained: bool = False

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

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @is_trained.setter
    def is_trained(self, trained: bool):
        self._is_trained = trained

    @property
    def xdata_shape(self):
        return (self.n_training_samples,)

    @staticmethod
    def _cast_parameters(
        circuits: List[Circuit], src_backend: Backend, target_backend: Backend
    ):
        for circ in circuits:
            for gate in circ.parametrized_gates:
                params = src_backend.to_numpy(
                    src_backend.cast(gate.parameters, dtype=src_backend.np.float64)
                )
                gate.parameters = target_backend.cast(
                    params, dtype=target_backend.np.float64
                )

    def regression(
        self,
        training_circuits: Optional[List[Circuit]] = None,
        observable: Optional[AbstractHamiltonian] = None,
        nshots: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
    ):
        if training_circuits is None:
            training_circuits = self.training_circuits

        observable = self._observable(observable)

        # first get the noisy expectations running with the current backend
        noisy_circuits = self._circuit_preprocessing(training_circuits, noise_model)
        noisy_expectations = observable.backend.cast(
            [
                observable.expectation(circuit, nshots=nshots)
                for circuit in noisy_circuits
            ],
            dtype=observable.backend.np.float64,
        )
        # cast to numpy
        noisy_expectations = self.backend.to_numpy(noisy_expectations)
        noisy_expectations = np.reshape(noisy_expectations, self.xdata_shape)

        # then switch to the simulation backend to get the exact values
        original_backend = observable.backend
        observable.backend = self.simulation_backend
        # select the subset of circuits relevant for exact expectations
        N = self.xdata_shape[-1] if len(self.xdata_shape) > 1 else 1
        training_circuits = [
            training_circuits[i] for i in N * np.arange(noisy_expectations.shape[0])
        ]
        # cast circuits parameters to the simulation backend arrays
        self._cast_parameters(
            training_circuits, original_backend, self.simulation_backend
        )
        exact_expectations = self.simulation_backend.cast(
            [
                observable.expectation(circuit, nshots=nshots)
                for circuit in training_circuits
            ],
            dtype=self.simulation_backend.np.float64,
        )
        # cast to numpy
        exact_expectations = self.simulation_backend.to_numpy(exact_expectations)
        # restore the original backend
        observable.backend = original_backend

        # do the regression
        optimal_params, params_cov = curve_fit(
            self.regression_model,
            noisy_expectations,
            exact_expectations,
            self.model_parameters,
        )
        return optimal_params

    def _apply_model(
        self,
        circuit: Optional[Circuit] = None,
        observable: Optional[AbstractHamiltonian] = None,
        nshots: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
    ):
        circuit = self._circuit(circuit)
        circuit = self._circuit_preprocessing([circuit], noise_model)[0]
        observable = self._observable(observable)
        noisy_exp_val = observable.expectation(circuit, nshots=nshots)
        params = self.backend.cast(self.model_parameters, dtype=noisy_exp_val.dtype)
        return self.regression_model(noisy_exp_val, *params)

    def __call__(
        self,
        circuit: Optional[Circuit] = None,
        observable: Optional[AbstractHamiltonian] = None,
        nshots: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
    ):
        training_circuits = self._training_circuits(circuit)
        # if something is different retrain the regression model
        if (
            circuit is not None
            or observable is not None
            or noise_model is not None
            or not self.is_trained
        ):
            self.model_parameters = self.regression(
                training_circuits, observable, nshots, noise_model
            )
            self.is_trained = True
        return self._apply_model(circuit, observable, nshots, noise_model)
