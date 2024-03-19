import re

import numpy as np
import qulacs
from qulacs import (  # pylint: disable=no-name-in-module
    QuantumCircuitSimulator,
    converter,
)
from qulacs.circuit import QuantumCircuitOptimizer  # pylint: disable=no-name-in-module

from qibo import __version__
from qibo.backends import NumpyBackend
from qibo.result import CircuitResult, QuantumState


class QulacsBackend(NumpyBackend):

    def __init__(self):
        super().__init__()
        self.name = "qulacs"
        self.versions = {"qibo": __version__, "qulacs": qulacs.__version__}
        self.device = "CPU"

    def circuit_to_qulacs(
        self, circuit: "qibo.Circuit"
    ) -> qulacs.QuantumCircuit:  # pylint: disable=no-member
        qasm_str = re.sub("^//.+\n", "", circuit.to_qasm())
        qasm_str = re.sub(r"creg\s.+;", "", qasm_str)
        qasm_str = re.sub(r"measure\s.+;", "", qasm_str)
        circ = converter.convert_QASM_to_qulacs_circuit(qasm_str.splitlines())
        return circ

    def execute_circuit(
        self,
        circuit: "qibo.Circuit",
        initial_state: np.ndarray = None,
        nshots: int = 1000,
    ):
        circ = self.circuit_to_qulacs(circuit)
        state = (
            qulacs.DensityMatrix(circuit.nqubits)  # pylint: disable=no-member
            if circuit.density_matrix
            else qulacs.QuantumState(circuit.nqubits)  # pylint: disable=no-member
        )
        sim = QuantumCircuitSimulator(circ, state)
        if initial_state is not None:
            sim.initialize_state(initial_state)
        sim.simulate()
        if circuit.density_matrix:
            dim = 2**circuit.nqubits
            state = (
                state.get_matrix()
                .reshape(2 * circuit.nqubits * (2,))
                .T.reshape(dim, dim)
            )
        else:
            state = state.get_vector().reshape(circuit.nqubits * (2,)).T.ravel()
        if len(circuit.measurements) > 0:
            return CircuitResult(
                state, circuit.measurements, backend=self, nshots=nshots
            )
        return QuantumState(state, backend=self)
