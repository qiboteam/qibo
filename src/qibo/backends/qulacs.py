import re

import numpy as np
import qulacs  # pylint: disable=import-error
from qulacs import (  # pylint: disable=no-name-in-module, import-error
    QuantumCircuitSimulator,
    converter,
)

from qibo import __version__
from qibo.backends import NumpyBackend
from qibo.config import raise_error
from qibo.result import CircuitResult, QuantumState


def circuit_to_qulacs(
    circuit: "qibo.Circuit",
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
    circ = converter.convert_QASM_to_qulacs_circuit(qasm_str.splitlines())
    return circ


class QulacsBackend(NumpyBackend):

    def __init__(self):
        super().__init__()

        self.name = "qulacs"
        self.versions = {"qibo": __version__, "qulacs": qulacs.__version__}
        self.device = "CPU"

    def execute_circuit(
        self,
        circuit: "qibo.Circuit",
        initial_state=None,
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
        if initial_state is not None:
            raise_error(
                NotImplementedError,
                "The use of an initial state is not supported yet by the `QulacsBackend`.",
            )
        circ = circuit_to_qulacs(circuit)
        state = (
            qulacs.DensityMatrix(circuit.nqubits)  # pylint: disable=no-member
            if circuit.density_matrix
            else qulacs.QuantumState(circuit.nqubits)  # pylint: disable=no-member
        )
        sim = QuantumCircuitSimulator(circ, state)
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
