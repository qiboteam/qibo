from dataclasses import dataclass
from typing import Dict, Optional

import mitiq

from qibo.config import raise_error
from qibo.error_mitigation.abstract import (
    ErrorMitigationRoutine,
    ReadoutMitigationRoutine,
)
from qibo.hamiltonians.abstract import AbstractHamiltonian
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
        self._run_method = f"run_with_{method}"
        self.method = getattr(mitiq, self.method)
        if self.method_kwargs is None:
            self.method_kwargs = {}

    @property
    def _run(self):
        return getattr(self.method, self._run_method)

    def __call__(
        self,
        circuit: Circuit,
        observable: AbstractHamiltonian,
        nshots: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
    ):
        circuit = self._circuit_preprocessing([circuit], noise_model)
        return self._run(circuit[0], **self.method_kwargs)
