import re

import numpy as np
import qulacs
from qulacs import Observable, QuantumCircuitSimulator, converter
from qulacs.circuit import QuantumCircuitOptimizer

from qibo import __version__
from qibo.backends import NumpyBackend
from qibo.result import CircuitResult, QuantumState


class QulacsBackend(NumpyBackend):

    def __init__(self):
        super().__init__()
        self.name = "qulacs"
        self.versions = {"qibo": __version__, "qulacs": qulacs.__version__}

    def circuit_to_qulacs(
        self, circuit: "qibo.Circuit", optimize: bool = False
    ) -> qulacs.QuantumCircuit:
        qasm_str = re.sub("^//.+\n", "", circuit.to_qasm())
        qasm_str = re.sub(r"creg\s.+;", "", qasm_str)
        qasm_str = re.sub(r"measure\s.+;", "", qasm_str)
        circ = converter.convert_QASM_to_qulacs_circuit(qasm_str.splitlines())
        if optimize:
            opt = QuantumCircuitOptimizer()
            opt.optimize(circ)
        return circ

    def execute_circuit(
        self,
        circuit: "qibo.Circuit",
        initial_state: np.ndarray = None,
        nshots: int = 1000,
        optimize: bool = False,
    ):
        circ = self.circuit_to_qulacs(circuit, optimize=optimize)
        state = (
            qulacs.DensityMatrix(circuit.nqubits)
            if circuit.density_matrix
            else qulacs.QuantumState(circuit.nqubits)
        )
        sim = QuantumCircuitSimulator(circ, state)
        if initial_state is not None:
            sim.initialize_state(initial_state)
        sim.simulate()
        state = state.get_matrix() if circuit.density_matrix else state.get_vector()
        if len(circuit.measurements) > 0:
            return CircuitResult(
                state, circuit.measurements, backend=self, nshots=nshots
            )
        return QuantumState(state, backend=self)
