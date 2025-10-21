"""Error mitigation object"""

from dataclasses import dataclass
from typing import List, Optional

from qibo import Circuit
from qibo.backends.abstract import Backend
from qibo.hamiltonians.abstract import AbstractHamiltonian
from qibo.noise import NoiseModel
from qibo.transpiler import Passes


@dataclass
class Mitigator:

    passes: List
    transpiler: Optional[Passes] = None
    backend: Optional[Backend] = None

    def __call__(
        self,
        circuit: Circuit,
        observable: AbstractHamiltonian,
        nshots: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
    ):
        for routine in self.passes:
            circuit, exp_val = routine(circuit, observable, nshots, noise_model)
        return exp_val
