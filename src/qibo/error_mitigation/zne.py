import math
from dataclasses import dataclass
from functools import cache, cached_property
from typing import Iterable, List, Optional, Tuple

import numpy as np

from qibo import Circuit, gates
from qibo.config import raise_error
from qibo.error_mitigation.abstract import ErrorMitigationRoutine
from qibo.hamiltonians.abstract import AbstractHamiltonian
from qibo.noise import NoiseModel


@dataclass
class ZNE(ErrorMitigationRoutine):

    noise_levels: Iterable[int] = tuple(range(4))
    insertion_gate: str = "CNOT"
    global_unitary_folding: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.insertion_gate = getattr(gates, self.insertion_gate)
        if not self.insertion_gate in (gates.RX, gates.CNOT):
            raise_error(
                ValueError,
                f"Gate {self.insertion_gate} is not supported, please use either `RX` or `CNOT`.",
            )

    @staticmethod
    @cache
    def gammas(noise_levels: Tuple[int, ...], analytical: bool = True) -> np.ndarray:
        """Standalone function to compute the ZNE coefficients given the noise levels.

        Args:
            noise_levels (numpy.ndarray): array containing the different noise levels.
                Note that in the CNOT insertion paradigm this corresponds to
                the number of CNOT pairs to be inserted. The canonical ZNE
                noise levels are obtained as ``2 * c + 1``.
            analytical (bool, optional): if ``True``, computes the coeffients by solving the
                linear system. If ``False``, use the analytical solution valid
                for the CNOT insertion method. Default is ``True``.

        Returns:
            numpy.ndarray: the computed coefficients.
        """
        noise_levels = np.asarray(noise_levels)
        if analytical:
            noise_levels = 2 * noise_levels + 1
            a_matrix = np.array([noise_levels**i for i in range(len(noise_levels))])
            b_vector = np.zeros(len(noise_levels))
            b_vector[0] = 1
            zne_coefficients = np.linalg.solve(a_matrix, b_vector)
        else:
            max_noise_level = noise_levels[-1]
            zne_coefficients = np.array(
                [
                    1
                    / (2 ** (2 * max_noise_level) * math.factorial(i))
                    * (-1) ** i
                    / (1 + 2 * i)
                    * math.factorial(1 + 2 * max_noise_level)
                    / (
                        math.factorial(max_noise_level)
                        * math.factorial(max_noise_level - i)
                    )
                    for i in noise_levels
                ]
            )
        return zne_coefficients

    def build_noisy_circuit(
        self, circuit: Optional[Circuit] = None, num_insertions: int = 1
    ) -> Circuit:
        circuit = self._circuit(circuit)

        if self.global_unitary_folding:
            copy_c = circuit.copy(deep=True)
            noisy_circuit = copy_c
            for _ in range(num_insertions):
                noisy_circuit += copy_c.invert() + copy_c
            return noisy_circuit

        if (
            self.insertion_gate is gates.CNOT and circuit.nqubits < 2
        ):  # pragma: no cover
            raise_error(
                ValueError,
                "Provide a circuit with at least 2 qubits when using the 'CNOT' insertion gate. "
                + "Alternatively, try with the 'RX' insertion gate instead.",
            )

        # theta = np.pi / 2 # this or the actual angle??
        noisy_circuit = Circuit(**circuit.init_kwargs)

        for gate in circuit.queue:
            noisy_circuit.add(gate)
            if isinstance(gate, self.insertion_gate):
                gate_kwargs_1 = gate_kwargs_2 = {}
                if gate is gates.RX:
                    gate_kwargs_1 = {"theta": gate.init_kwargs["theta"]}
                    gate_kwargs_2 = {"theta": -gate.init_kwargs["theta"]}
                for _ in range(num_insertions):
                    noisy_circuit.add(
                        self.insertion_gate(*gate.qubits, **gate_kwargs_1)
                    )
                    noisy_circuit.add(
                        self.insertion_gate(*gate.qubits, **gate_kwargs_2)
                    )
        return noisy_circuit

    def _noisy_circuits(self, circuit: Optional[Circuit] = None):
        if circuit is None:
            return self.noisy_circuits
        return [
            self.build_noisy_circuit(circuit, num_insertions)
            for num_insertions in self.noise_levels
        ]

    @cached_property
    def noisy_circuits(self) -> List[Circuit]:
        return [
            self.build_noisy_circuit(self.circuit, num_insertions)
            for num_insertions in self.noise_levels
        ]

    def __call__(
        self,
        circuit: Optional[Circuit] = None,
        observable: Optional[AbstractHamiltonian] = None,
        nshots: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
    ):
        noisy_circuits = self._noisy_circuits(circuit)
        noisy_circuits = self._circuit_preprocessing(noisy_circuits, noise_model)
        observable = self._observable(observable)
        exp_vals = [
            observable.expectation(circ, nshots=nshots) for circ in noisy_circuits
        ]
        exp_vals = self.backend.np.stack(exp_vals)
        gammas = self.backend.cast(
            self.gammas(self.noise_levels), dtype=exp_vals[0].dtype
        )
        return sum(gammas * exp_vals)
