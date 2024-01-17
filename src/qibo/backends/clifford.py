"""Module defining the Clifford backend."""
import collections
from functools import reduce
from typing import Union

import numpy as np

from qibo import gates
from qibo.backends.numpy import NumpyBackend
from qibo.backends.tensorflow import TensorflowBackend
from qibo.config import raise_error


def _calculation_engine(backend):
    """Helper function to initialize the Clifford backend with the correct engine."""
    if backend.name == "qibojit":
        if (
            backend.platform == "cupy" or backend.platform == "cuquantum"
        ):  # pragma: no cover
            return backend.cp
        return backend.np

    return backend.np


class CliffordBackend(NumpyBackend):
    """Backend for the simulation of Clifford circuits following `Aaronson & Gottesman (2004) <https://arxiv.org/abs/quant-ph/0406196>`_.

    Args:
        engine (:class:`qibo.backends.abstract.Backend`): Backend used for the calculation.
    """

    def __init__(self, engine=None):
        super().__init__()

        if engine is None:
            from qibo.backends import GlobalBackend

            engine = GlobalBackend()

        if isinstance(engine, TensorflowBackend):
            raise_error(
                NotImplementedError,
                "TensorflowBackend for Clifford Simulation is not supported.",
            )

        self.engine = engine
        self.np = _calculation_engine(engine)

        self.name = "clifford"
        self.clifford_operations = engine.clifford_operations

    def cast(self, x, dtype=None, copy: bool = False):
        """Cast an object as the array type of the current backend.

        Args:
            x: Object to cast to array.
            copy (bool, optional): If ``True`` a copy of the object is created in memory.
                Defaults to ``False``.
        """
        return self.engine.cast(x, dtype=dtype, copy=copy)

    def calculate_frequencies(self, samples):
        res, counts = self.engine.np.unique(samples, return_counts=True)
        # The next two lines are necessary for the GPU backends
        res = [int(r) if not isinstance(r, str) else r for r in res]
        counts = [int(v) for v in counts]

        return collections.Counter({k: v for k, v in zip(res, counts)})

    def zero_state(self, nqubits: int):
        """Construct the zero state |00...00>.

        Args:
            nqubits (int): Number of qubits.

        Returns:
            (ndarray): Symplectic matrix for the zero state.
        """
        I = self.np.eye(nqubits)
        symplectic_matrix = self.np.zeros(
            (2 * nqubits + 1, 2 * nqubits + 1), dtype=bool
        )
        symplectic_matrix[:nqubits, :nqubits] = self.np.copy(I)
        symplectic_matrix[nqubits:-1, nqubits : 2 * nqubits] = self.np.copy(I)
        return symplectic_matrix

    def reshape_clifford_state(self, state, nqubits):
        """Reshape the symplectic matrix to the shape needed by the engine.

        Args:
            state (ndarray): The input state.
            nqubits (int): Number of qubits.

        Returns:
            (ndarray): The reshaped state.
        """
        return self.engine.reshape_clifford_state(state, nqubits)

    def apply_gate_clifford(self, gate, symplectic_matrix, nqubits):
        operation = getattr(self.clifford_operations, gate.__class__.__name__)
        kwargs = (
            {"theta": gate.init_kwargs["theta"]} if "theta" in gate.init_kwargs else {}
        )

        return operation(symplectic_matrix, *gate.init_args, nqubits, **kwargs)

    def execute_circuit(self, circuit, initial_state=None, nshots: int = 1000):
        """Execute a Clifford circuits.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Input circuit.
            initial_state (ndarray, optional): The ``symplectic_matrix`` of the initial state.
                If ``None``, defaults to the zero state. Defaults to ``None``.
            nshots (int, optional): Number of shots to perform if ``circuit`` has measurements.
                Defaults to :math:`10^{3}`.

        Returns:
            (:class:`qibo.quantum_info.clifford.Clifford`): Object giving access to the final results.
        """
        for gate in circuit.queue:
            if not gate.clifford and not gate.__class__.__name__ == "M":
                raise_error(RuntimeError, "Circuit contains non-Clifford gates.")

        if circuit.repeated_execution and not nshots == 1:
            return self.execute_circuit_repeated(circuit, initial_state, nshots)

        try:
            from qibo.quantum_info.clifford import Clifford

            nqubits = circuit.nqubits

            state = self.zero_state(nqubits) if initial_state is None else initial_state

            state = self.reshape_clifford_state(state, nqubits)

            for gate in circuit.queue:
                state = gate.apply_clifford(self, state, nqubits)

            state = self.reshape_clifford_state(state, nqubits)

            clifford = Clifford(
                state,
                measurements=circuit.measurements,
                nshots=nshots,
                engine=self.engine,
            )

            return clifford

        except self.oom_error:  # pragma: no cover
            raise_error(
                RuntimeError,
                f"State does not fit in {self.device} memory."
                "Please switch the execution device to a "
                "different one using ``qibo.set_device``.",
            )

    def execute_circuit_repeated(self, circuit, initial_state=None, nshots: int = 1000):
        """Execute a Clifford circuits ``nshots`` times.

        This is used for all the simulations that involve repeated execution.
        For instance when collapsing measurement or noise channels are present.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): input circuit.
            initial_state (ndarray, optional): Symplectic_matrix of the initial state.
                If ``None``, defaults to :meth:`qibo.backends.clifford.CliffordBackend.zero_state`.
                Defaults to ``None``.
            nshots (int, optional): Number of times to repeat the execution.
                Defaults to :math:`1000`.

        Returns:
            (:class:`qibo.quantum_info.clifford.Clifford`): Object giving access to the final results.
        """
        circuit_copy = circuit.copy()
        samples = []
        for _ in range(nshots):
            res = self.execute_circuit(circuit_copy, initial_state, nshots=1)
            [m.result.reset() for m in circuit_copy.measurements]
            samples.append(res.samples())
        samples = self.np.vstack(samples)

        for m in circuit.measurements:
            m.result.register_samples(samples[:, m.target_qubits], self)

        from qibo.quantum_info.clifford import Clifford

        result = Clifford(
            self.zero_state(circuit.nqubits),
            measurements=circuit.measurements,
            nshots=nshots,
            engine=self.engine,
        )
        result.symplectic_matrix, result._samples = None, None

        return result

    def sample_shots(
        self,
        state,
        qubits: Union[tuple, list],
        nqubits: int,
        nshots: int,
        collapse: bool = False,
    ):
        """Sample shots by measuring the selected qubits from the provided symplectic matrix of a ``state``.

        Args:
            state (ndarray): symplectic matrix from which to sample shots from.
            qubits: (tuple or list): qubits to measure.
            nqubits (int): total number of qubits of the state.
            nshots (int): number of shots to sample.
            collapse (bool, optional): If ``True`` the input state is going to be
                collapsed with the last shot. Defaults to ``False``.

        Returns:
            (ndarray): Samples shots.
        """
        if isinstance(qubits, list):
            qubits = tuple(qubits)

        if collapse:
            samples = [
                self.clifford_operations.M(state, qubits, nqubits)
                for _ in range(nshots - 1)
            ]  # parallelize?
            samples.append(self.clifford_operations.M(state, qubits, nqubits, collapse))
        else:
            samples = [
                self.clifford_operations.M(state, qubits, nqubits)
                for _ in range(nshots)
            ]  # parallelize?

        return self.engine.cast(samples)

    def symplectic_matrix_to_generators(
        self, symplectic_matrix, return_array: bool = False
    ):
        """Extract both the stabilizers and destabilizers generators from the input symplectic matrix.

        Args:
            symplectic_matrix (ndarray): The input symplectic_matrix.
            return_array (bool, optional): If ``True`` returns the generators as ``ndarrays``.
                If ``False``, generators are returned as strings. Defaults to ``False``.

        Returns:
            (list, list): Extracted generators and their corresponding phases, respectively.
        """
        bits_to_gate = {"00": "I", "01": "X", "10": "Z", "11": "Y"}

        nqubits = int((symplectic_matrix.shape[1] - 1) / 2)
        phases = (-1) ** symplectic_matrix[:-1, -1]
        tmp = 1 * symplectic_matrix[:-1, :-1]
        X, Z = tmp[:, :nqubits], tmp[:, nqubits:]
        generators = []
        for x, z in zip(X, Z):
            paulis = [bits_to_gate[f"{zz}{xx}"] for xx, zz in zip(x, z)]
            if return_array:
                paulis = [self.cast(getattr(gates, p)(0).matrix()) for p in paulis]
                matrix = reduce(self.np.kron, paulis)
                generators.append(matrix)
            else:
                generators.append("".join(paulis))

        if return_array:
            generators = self.cast(generators)

        return generators, phases
