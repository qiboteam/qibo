import re

import numpy as np

from qibo import __version__
from qibo.backends import NumpyBackend
from qibo.result import CircuitResult, QuantumState


class QulacsBackend(NumpyBackend):

    def __init__(self):
        super().__init__()
        import qulacs
        from qulacs import QuantumCircuitSimulator, converter

        self.qulacs = qulacs
        self.simulator = QuantumCircuitSimulator
        self.converter = converter
        self.name = "qulacs"
        self.versions = {"qibo": __version__, "qulacs": qulacs.__version__}
        self.device = "CPU"

    def circuit_to_qulacs(
        self, circuit: "qibo.Circuit"
    ) -> "qulacs.QuantumCircuit":  # pylint: disable=no-member
        """
        Converts a qibo circuit in a qulacs circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Input circuit to convert.

        Returns:
            qulacs.QuantumCircuit: The converted qulacs circuit.
        """
        qasm_str = re.sub("^//.+\n", "", circuit.to_qasm())
        qasm_str = re.sub(r"creg\s.+;", "", qasm_str)
        qasm_str = re.sub(r"measure\s.+;", "", qasm_str)
        circ = self.converter.convert_QASM_to_qulacs_circuit(qasm_str.splitlines())
        return circ

    def execute_circuit(
        self,
        circuit: "qibo.Circuit",
        nshots: int = 1000,
    ):
        """Execute a circuit with qulacs.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Input circuit.
            nshots (int, optional): Number of shots to perform if ``circuit`` has measurements.
                Defaults to :math:`10^{3}`.

        Returns:
            :class:`qibo.result.CircuitResult`: Object storing to the final results.
        """
        circ = self.circuit_to_qulacs(circuit)
        state = (
            self.qulacs.DensityMatrix(circuit.nqubits)  # pylint: disable=no-member
            if circuit.density_matrix
            else self.qulacs.QuantumState(circuit.nqubits)  # pylint: disable=no-member
        )
        sim = self.simulator(circ, state)
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
