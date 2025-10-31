from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional

import numpy as np

from qibo.config import raise_error
from qibo.error_mitigation.cdr import CDR
from qibo.error_mitigation.zne import ZNE
from qibo.hamiltonians.abstract import AbstractHamiltonian
from qibo.models.circuit import Circuit
from qibo.noise import NoiseModel


@dataclass
class VNCDR(ZNE, CDR):

    def __post_init__(self):
        super().__post_init__()

    def _regression_model_setup(self):
        if self.regression_model is None:
            self.regression_model = lambda x, *params: sum(
                [xx * p for xx, p in zip(x.T, params)]
            )  # lambda x, *params: x @ np.asarray(params)
            if self.model_parameters is None:
                self.model_parameters = np.ones(
                    (len(self.noise_levels),), dtype=np.float64
                )
        elif self.model_parameters is None:
            raise_error(
                ValueError,
                "Please provide also the parameters through the `model_parameters` argument when using a custom regression model.",
            )

    @property
    def xdata_shape(self):
        return (-1, len(self.noise_levels))

    def _training_circuits(self, circuit: Optional[Circuit] = None) -> List[Circuit]:
        if circuit is None:
            return self.training_circuits
        circuits = []
        for _ in range(self.n_training_samples):
            circ = self.sample_circuit(circuit)
            noisier_circuits = [
                self.build_noisy_circuit(circ, level) for level in self.noise_levels
            ]
            circuits.extend(noisier_circuits)
        return circuits

    @cached_property
    def training_circuits(self) -> List[Circuit]:
        circuits = []
        for _ in range(self.n_training_samples):
            circ = self.sample_circuit(self.circuit)
            noisier_circuits = [
                self.build_noisy_circuit(circ, level) for level in self.noise_levels
            ]
            circuits.extend(noisier_circuits)
        return circuits

    def _apply_model(
        self,
        circuit: Optional[Circuit] = None,
        observable: Optional[AbstractHamiltonian] = None,
        nshots: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
    ):
        noisy_circuits = self._noisy_circuits(circuit)
        noisy_circuits = self._circuit_preprocessing(noisy_circuits, noise_model)
        observable = self._observable(observable)
        noisy_exp_vals = [
            observable.expectation(circ, nshots=nshots) for circ in noisy_circuits
        ]
        noisy_exp_vals = self.backend.cast(
            noisy_exp_vals, dtype=noisy_exp_vals[0].dtype
        )
        params = self.backend.cast(self.model_parameters, dtype=noisy_exp_vals.dtype)
        return self.regression_model(noisy_exp_vals, *params)

    def __call__(
        self,
        circuit: Optional[Circuit] = None,
        observable: Optional[AbstractHamiltonian] = None,
        nshots: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
    ):
        return CDR.__call__(self, circuit, observable, nshots, noise_model)
