"""Module defining the Clifford backend."""

import collections
import itertools
from functools import reduce
from importlib.util import find_spec, module_from_spec
from typing import Union

import numpy as np

from qibo import gates
from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error


class CliffordBackend(NumpyBackend):
    """Backend for the simulation of Clifford circuits following
    `Aaronson & Gottesman (2004) <https://arxiv.org/abs/quant-ph/0406196>`_.

    Args:
        :class:`qibo.backends.abstract.Backend`: Backend used for the calculation.
    """

    def __init__(self, engine=None):
        super().__init__()

        if engine == "stim":
            import stim  # pylint: disable=C0415

            engine = "numpy"
            self.platform = "stim"
            self._stim = stim
        else:
            if engine is None:
                from qibo.backends import (  # pylint: disable=C0415
                    _check_backend,
                    _get_engine_name,
                )

                engine = _get_engine_name(_check_backend(engine))

            self.platform = engine

        spec = find_spec("qibo.backends._clifford_operations")
        self.engine = module_from_spec(spec)
        spec.loader.exec_module(self.engine)

        if engine == "numpy":
            pass
        elif engine == "numba":
            from numba import set_num_threads

            set_num_threads(1)

            from qibojit.backends import (  # pylint: disable=C0415
                clifford_operations_cpu,
            )

            for method in dir(clifford_operations_cpu):
                setattr(self.engine, method, getattr(clifford_operations_cpu, method))
        elif engine == "cupy":  # pragma: no cover
            from qibojit.backends import (  # pylint: disable=C0415
                clifford_operations_gpu,
            )

            for method in dir(clifford_operations_gpu):
                setattr(self.engine, method, getattr(clifford_operations_gpu, method))
        else:
            raise_error(
                NotImplementedError,
                f"Backend `{engine}` is not supported for Clifford Simulation.",
            )

        self.np = self.engine.np

        self.name = "clifford"

    def cast(self, x, dtype=None, copy: bool = False):
        """Cast an object as the array type of the current backend.

        Args:
            x: Object to cast to array.
            dtype (optional): data type of the array or tensor. If ``None``, defaults
                to the default data type of the current backend. Defaults to ``None``.
            copy (bool, optional): If ``True`` a copy of the object is created in memory.
                Defaults to ``False``.
        """
        return self.engine.cast(x, dtype=dtype, copy=copy)

    def calculate_frequencies(self, samples):
        res, counts = self.engine.np.unique(samples, return_counts=True)
        # The next two lines are necessary for the GPU backends
        res = [int(r) if not isinstance(r, str) else r for r in res]
        counts = [int(v) for v in counts]

        return collections.Counter(dict(zip(res, counts)))

    def zero_state(self, nqubits: int):
        """Construct the zero state :math`\\ket{00...00}`.

        Args:
            nqubits (int): number of qubits.

        Returns:
            ndarray: Symplectic matrix for the zero state.
        """
        identity = self.np.eye(nqubits)
        symplectic_matrix = self.np.zeros(
            (2 * nqubits + 1, 2 * nqubits + 1), dtype=bool
        )
        symplectic_matrix[:nqubits, :nqubits] = self.np.copy(identity)
        symplectic_matrix[nqubits:-1, nqubits : 2 * nqubits] = self.np.copy(identity)
        return symplectic_matrix

    def _clifford_pre_execution_reshape(self, state):
        """Reshape the symplectic matrix to the shape needed by the engine before circuit execution.

        Args:
            state (ndarray): Input state.

        Returns:
            ndarray: Reshaped state.
        """
        return self.engine._clifford_pre_execution_reshape(  # pylint: disable=protected-access
            state
        )

    def _clifford_post_execution_reshape(self, state, nqubits: int):
        """Reshape the symplectic matrix to the shape needed by the engine after circuit execution.

        Args:
            state (ndarray): Input state.
            nqubits (int): Number of qubits.

        Returns:
            ndarray: Reshaped state.
        """
        return self.engine._clifford_post_execution_reshape(  # pylint: disable=protected-access
            state, nqubits
        )

    def apply_gate_clifford(self, gate, symplectic_matrix, nqubits):
        """Apply a gate to a symplectic matrix."""
        if isinstance(gate, gates.Unitary):
                symplectic_matrix = self.apply_unitary(gate, symplectic_matrix, nqubits)

                return symplectic_matrix
        
        operation = getattr(self.engine, gate.__class__.__name__)

        kwargs = {}
        for param_name in ["theta", "phi"]:
            if param_name in gate.init_kwargs:
                kwargs[param_name] = gate.init_kwargs[param_name]


        return operation(symplectic_matrix, *gate.init_args, nqubits, **kwargs)
    
    def apply_unitary(self, gate, symplectic_matrix, nqubits):
        """Apply a unitary gate to a symplectic matrix."""
        temp_matrix = self._clifford_post_execution_reshape(symplectic_matrix, nqubits)

        qubit_indices = list(gate.qubits)
        m = len(qubit_indices)
        
        matrix = gate.matrix()
        symplectic_m = self._compute_symplectic_matrix(matrix, m)
        phase_m = self._get_phase_vector(matrix, m)

        symplectic_n = self._embed_clifford(symplectic_m, nqubits, qubit_indices)
        phase_n = self._embed_phase_vector(phase_m, nqubits, qubit_indices)

        original_symplectic = temp_matrix[:, :2*nqubits].copy()
        original_phases = temp_matrix[:, 2*nqubits].copy()

        temp_matrix[:, :2*nqubits] = (original_symplectic @ symplectic_n.T) % 2
        for i in range(2*nqubits):
            v_old = original_symplectic[i, :]  
            phase_old = original_phases[i]      

            linear_correction = np.dot(v_old, phase_n) % 2
            quadratic_correction = self._compute_quadratic_form(v_old, symplectic_n, nqubits)

            total_correction = (linear_correction + quadratic_correction) % 2
            temp_matrix[i, 2*nqubits] = (phase_old + total_correction) % 2

        result = self._clifford_pre_execution_reshape(temp_matrix)
        symplectic_matrix[:] = result  

        return symplectic_matrix

    def _pauli_string_to_matrix(self, pauli_str):
        """Convert Pauli string to matrix (tensor product)."""
        from qibo import matrices  # pylint: disable=C0415

        pauli_matrices = (getattr(matrices, p) for p in pauli_str)
        matrix = reduce(np.kron, pauli_matrices)
        return self.cast(matrix, dtype=matrix.dtype)

    def _pauli_to_binary(self, pauli_str, nqubits):
        """Convert Pauli string to binary vector of length :math`2*nqubits`."""
        pauli_symplectic = self.np.zeros((2*nqubits))
        for q, term in enumerate(pauli_str):
            if term in ['X', 'Y']:
                pauli_symplectic[q] = 1
            if term in ['Z', 'Y']:
                pauli_symplectic[q + nqubits] = 1

        return pauli_symplectic

    def _compute_symplectic_matrix(self, unitary, m):
        """Compute symplectic matrix for Clifford unitary on :math`m` qubits."""
        pauli_gens = []
        for i in range(m):
            p = ['I']*m; p[i] = 'X'
            pauli_gens.append(''.join(p))
        for i in range(m):
            p = ['I']*m; p[i] = 'Z'
            pauli_gens.append(''.join(p))

        symplectic = np.zeros((2*m, 2*m), dtype=int)
        symplectic = self.cast(symplectic, dtype=symplectic.dtype)

        for i, p_str in enumerate(pauli_gens):
            pauli = self._pauli_string_to_matrix(p_str)
            pauli_uconj = unitary @ pauli @ unitary.conj().T

            found = False
            for candidate_str in itertools.product('IXYZ', repeat=m):
                candidate_str = ''.join(candidate_str)
                candidate_P = self._pauli_string_to_matrix(candidate_str)
                
                for phase in [1, -1, 1j, -1j]:
                    if np.allclose(pauli_uconj, phase * candidate_P, atol=1e-10):
                        v = self._pauli_to_binary(candidate_str, m)
                        symplectic[:, i] = v
                        found = True
                        break
                if found:
                    break
                    
            if not found:
                raise ValueError(f"Could not find conjugated Pauli for {p_str}")

        return symplectic % 2

    def _get_phase_vector(self, unitary, m):
        """Compute phase vector :math`r_U` of length :math`2m` for Clifford unitary.
        :math`r_U[j] = 0` if :math`U g_j U^\\dagger = +P_j` else :math`1`."""
        pauli_gens = []
        for i in range(m):
            p = ['I']*m; p[i] = 'X'
            pauli_gens.append(''.join(p))
        for i in range(m):
            p = ['I']*m; p[i] = 'Z'
            pauli_gens.append(''.join(p))

        phase = np.zeros(2*m, dtype=int)
        for j, p_str in enumerate(pauli_gens):
            pauli = self._pauli_string_to_matrix(p_str)
            pauli_uconj = unitary @ pauli @ unitary.conj().T

            found = False
            for candidate_str in itertools.product('IXYZ', repeat=m):
                candidate_str = ''.join(candidate_str)
                candidate_P = self._pauli_string_to_matrix(candidate_str)
                if np.allclose(pauli_uconj, candidate_P):
                    phase[j] = 0
                    found = True
                    break
                elif np.allclose(pauli_uconj, -candidate_P):
                    phase[j] = 1
                    found = True
                    break
            if not found:
                raise ValueError(f"Phase vector: failed for {p_str}")
        return phase

    def _embed_clifford(self, symplectic_m, n, qubit_indices):
        """Embed m-qubit symplectic :math`S_U_m` into n-qubit system at qubit_indices."""
        symplectic_n = np.eye(2*n, dtype=int)
        symplectic_n = self.cast(symplectic_n, dtype=symplectic_m.dtype)

        x_indices = qubit_indices
        z_indices = [q + n for q in qubit_indices]
        full_indices = x_indices + z_indices

        symplectic_n[np.ix_(full_indices, full_indices)] = symplectic_m

        return symplectic_n % 2

    def _embed_phase_vector(self, phase_m, n, qubit_indices):
        """Embed m-qubit phase vector into n-qubit system."""
        phase_n = np.zeros(2*n, dtype=int)
        phase_n = self.cast(phase_n, dtype=phase_m.dtype)
        m = len(qubit_indices)
        
        qubit_indices = np.array(qubit_indices)
        phase_n[qubit_indices] = phase_m[:m]
        phase_n[qubit_indices + n] = phase_m[m:]

        return phase_n

    def _compute_quadratic_form(self, v, symplectic, n):
        """Compute quadratic form for phase correction."""
        v_x = v[:n]
        v_z = v[n:]

        #A = symplectic[:n, :n]      # X-X block
        B = symplectic[:n, n:]      # X-Z block
        C = symplectic[n:, :n]      # Z-X block
        #D = symplectic[n:, n:]      # Z-Z block

        term1 = (v_x.T @ B @ v_z) % 2
        term2 = (v_z.T @ C @ v_x) % 2
        
        return (term1 + term2) % 2

    def apply_channel(self, channel, state, nqubits):
        probabilities = channel.coefficients + (1 - np.sum(channel.coefficients),)
        index = self.np.random.choice(
            range(len(probabilities)), size=1, p=probabilities
        )[0]
        index = int(index)
        if index != len(channel.gates):
            gate = channel.gates[index]
            state = gate.apply_clifford(self, state, nqubits)
        return state

    def execute_circuit(  # pylint: disable=R1710
        self, circuit, initial_state=None, nshots: int = 1000
    ):
        """Execute a Clifford circuits.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Input circuit.
            initial_state (ndarray, optional): The ``symplectic_matrix`` of the initial state.
                If ``None``, defaults to the zero state. Defaults to ``None``.
            nshots (int, optional): Number of shots to perform if ``circuit`` has measurements.
                Defaults to :math:`10^{3}`.

        Returns:
            :class:`qibo.quantum_info.clifford.Clifford`: Object storing to the final results.
        """
        from qibo.quantum_info.clifford import Clifford  # pylint: disable=C0415

        if self.platform == "stim":
            circuit_stim = self._stim.Circuit()  # pylint: disable=E1101
            for gate in circuit.queue:
                circuit_stim.append(gate.__class__.__name__, list(gate.qubits))

            x_destab, z_destab, x_stab, z_stab, x_phases, z_phases = (
                self._stim.Tableau.from_circuit(  # pylint: disable=no-member
                    circuit_stim
                ).to_numpy()
            )
            symplectic_matrix = np.block([[x_destab, z_destab], [x_stab, z_stab]])
            symplectic_matrix = np.c_[symplectic_matrix, np.r_[x_phases, z_phases]]

            return Clifford(
                symplectic_matrix,
                measurements=circuit.measurements,
                nshots=nshots,
                _backend=self,
            )

        for gate in circuit.queue:
            if (
                not gate.clifford
                and not gate.__class__.__name__ == "M"
                and not isinstance(gate, gates.PauliNoiseChannel)
            ):
                raise_error(RuntimeError, "Circuit contains non-Clifford gates.")

        if circuit.repeated_execution and nshots != 1:
            return self.execute_circuit_repeated(circuit, nshots, initial_state)

        try:
            nqubits = circuit.nqubits

            state = self.zero_state(nqubits) if initial_state is None else initial_state
            state = self._clifford_pre_execution_reshape(state)
            for gate in circuit.queue:
                gate.apply_clifford(self, state, nqubits)

            state = self._clifford_post_execution_reshape(state, nqubits)

            clifford = Clifford(
                state,
                measurements=circuit.measurements,
                nshots=nshots,
                _backend=self,
            )

            return clifford

        except self.oom_error:  # pragma: no cover
            raise_error(
                RuntimeError,
                f"State does not fit in {self.device} memory."
                "Please switch the execution device to a "
                "different one using ``qibo.set_device``.",
            )

    def execute_circuit_repeated(self, circuit, nshots: int = 1000, initial_state=None):
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
            :class:`qibo.quantum_info.clifford.Clifford`: Object storing to the final results.
        """
        from qibo.quantum_info.clifford import Clifford  # pylint: disable=C0415

        circuit_copy = circuit.copy()
        samples = []
        for _ in range(nshots):
            res = self.execute_circuit(circuit_copy, initial_state, nshots=1)
            for measurement in circuit_copy.measurements:
                measurement.result.reset()
            samples.append(res.samples())
        samples = self.np.vstack(samples)

        for meas in circuit.measurements:
            meas.result.register_samples(samples[:, meas.target_qubits])

        result = Clifford(
            self.zero_state(circuit.nqubits),
            measurements=circuit.measurements,
            nshots=nshots,
            _backend=self,
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
    ):  # pylint: disable=W0221
        """Sample shots by measuring selected ``qubits`` in symplectic matrix of a ``state``.

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
            samples = [self.engine.M(state, qubits, nqubits) for _ in range(nshots - 1)]
            samples.append(self.engine.M(state, qubits, nqubits, collapse))
        else:
            samples = [self.engine.M(state, qubits, nqubits) for _ in range(nshots)]

        return self.engine.cast(samples, dtype=int)

    def symplectic_matrix_to_generators(
        self, symplectic_matrix, return_array: bool = False
    ):
        """Extract the stabilizers and destabilizers generators from symplectic matrix.

        Args:
            symplectic_matrix (ndarray): The input symplectic_matrix.
            return_array (bool, optional): If ``True`` returns the generators as ``ndarrays``.
                If ``False``, generators are returned as strings. Defaults to ``False``.

        Returns:
            (list, list): Extracted generators and their corresponding phases, respectively.
        """
        bits_to_gate = {"00": "I", "01": "X", "10": "Z", "11": "Y"}

        nqubits = int((symplectic_matrix.shape[1] - 1) / 2)
        phases = (-1) ** symplectic_matrix[:-1, -1].astype(np.int16)
        tmp = 1 * symplectic_matrix[:-1, :-1]
        X, Z = tmp[:, :nqubits], tmp[:, nqubits:]
        generators = []
        for x, z in zip(X, Z):
            paulis = [bits_to_gate[f"{zz}{xx}"] for xx, zz in zip(x, z)]
            if return_array:
                from qibo import matrices  # pylint: disable=C0415

                paulis = [self.cast(getattr(matrices, p)) for p in paulis]
                matrix = reduce(self.np.kron, paulis)
                generators.append(matrix)
            else:
                generators.append("".join(paulis))

        if return_array:
            generators = self.cast(generators)

        return generators, phases
