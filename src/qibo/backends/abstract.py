import math
from typing import List, Optional, Tuple, Union

from qibo import __version__
from qibo.backends import einsum_utils
from qibo.config import raise_error


class Backend:
    def __init__(self):
        super().__init__()

        self.device = "/CPU:0"
        self.dtype = "complex128"
        self.engine = None
        self.matrices = None
        self.name = "backend"
        self.nthreads = 1
        self.numeric_types = (
            int,
            float,
            complex,
            "complex128",
            "complex64",
            "float64",
            "float32",
        )
        self.oom_error = MemoryError
        self.platform = None
        self.supports_multigpu = False
        self.tensor_types = None
        self.versions = {"qibo": __version__}

    def __reduce__(self) -> Tuple["Backend", tuple]:
        """Allow pickling backend objects that have references to modules."""
        return self.__class__, tuple()

    def __repr__(self) -> str:
        if self.platform is None:
            return self.name

        return f"{self.name} ({self.platform})"

    @property
    def qubits(self) -> Optional[list[Union[int, str]]]:  # pragma: no cover
        """Return the qubit names of the backend.

        Returns:
            List[int] or List[str] or None: For hardware backends, return list of qubit names.
            For simulation backends, return ``None``.
        """
        return None

    @property
    def connectivity(
        self,
    ) -> Optional[List[Tuple[Union[int, str], Union[int, str]]]]:  # pragma: no cover
        """Return available qubit pairs of the backend.

        Returns:
            List[Tuple[int]] or List[Tuple[str]] or None: For hardware backends, return
            available qubit pairs. For simulation backends, return ``None``.
        """
        return None

    @property
    def natives(self) -> Optional[list[str]]:  # pragma: no cover
        """Return the native gates of the backend.

        Returns:
            List[str] or None: For hardware backends, return the native gates of the backend.
            For the simulation backends, return ``None``.
        """
        return None

    def set_dtype(self, dtype: str) -> None:  # pragma: no cover
        """Set data type of arrays created using the backend. Works in-place.

        .. note::
            The data types ``float32`` and ``float64`` are intended to be used when the circuits
            to be simulated only contain gates with real-valued matrix representations.
            Using one of the aforementioned data types with circuits that contain complex-valued
            matrices will raise a casting error.

        .. note::
            List of gates that have a real-valued matrix representation:
            :class:`qibo.gates.I`, :class:`qibo.gates.X`, :class:`qibo.gates.Z`,
            :class:`qibo.gates.H`, :class:`qibo.gates.Align`, :class:`qibo.gates.RY`,
            :class:`qibo.gates.CNOT`, :class:`qibo.gates.CZ`, :class:`qibo.gates.CRY`,
            :class:`qibo.gates.SWAP`, :class:`qibo.gates.FSWAP`, :class:`qibo.gates.GIVENS`,
            :class:`qibo.gates.RBS`, :class:`qibo.gates.TOFFOLI`, and :class:`qibo.gates.CCZ`.

        .. note::
            The following parametrized gates can have real-valued matrix representations
            depending on the values of their parameters:
            :class:`qibo.gates.RX`, :class:`qibo.gates.RZ`, :class:`qibo.gates.U1`,
            :class:`qibo.gates.U2`, :class:`qibo.gates.U3`, :class:`qibo.gates.CRX`,
            :class:`qibo.gates.CRZ`, :class:`qibo.gates.CU1`, :class:`qibo.gates.CU2`,
            :class:`qibo.gates.CU3`, :class:`qibo.gates.fSim`, :class:`qibo.gates.GeneralizedfSim`,
            :class:`qibo.gates.RXX`, :class:`qibo.gates.RYY`, :class:`qibo.gates.RZZ`,
            :class:`qibo.gates.RZX`, and :class:`qibo.gates.GeneralizedRBS`.

        Args:
            dtype (str): the options are the following: ``complex128``, ``complex64``,
                ``float64``, and ``float32``.
        """
        if dtype not in self.numeric_types:
            raise_error(
                ValueError,
                f"Unknown ``dtype`` ``{dtype}``."
                + f"``dtype`` must be one of the following options: {self.numeric_types}",
            )

        if dtype != self.dtype:
            self.dtype = dtype

            if self.matrices:
                self.matrices = self.matrices.__class__(self.dtype)

    def set_device(self, device: str) -> None:  # pragma: no cover
        """Set simulation device. Works in-place.

        Args:
            device (str): Device index, *e.g.* ``/CPU:0`` for CPU, or ``/GPU:1`` for
                the second GPU in a multi-GPU environment.
        """
        raise_error(NotImplementedError)

    def set_threads(self, nthreads: int) -> None:  # pragma: no cover
        """Set number of threads for CPU backend simulations that accept it. Works in-place.

        Args:
            nthreads (int): Number of threads.
        """
        raise_error(NotImplementedError)

    def cast(self, x, dtype=None, copy: bool = False) -> "ndarray":  # pragma: no cover
        """Cast an object as the array type of the current backend.

        Args:
            x: Object to cast to array.
            dtype (str or type, optional): data type of ``x`` after casting.
                Options are ``"complex128"``, ``"complex64"``, ``"float64"``,
                or ``"float32"``. If ``None``, defaults to ``Backend.dtype``.
                Defaults to ``None``.
            copy (bool, optional): If ``True`` a copy of the object is created in memory.
                Defaults to ``False``.
        """
        raise_error(NotImplementedError)

    def is_sparse(self, array) -> bool:  # pragma: no cover
        """Determine if a given array is a sparse tensor."""
        raise_error(NotImplementedError)

    def to_numpy(self, array) -> "ndarray":  # pragma: no cover
        """Cast a given array to numpy."""
        raise_error(NotImplementedError)

    def compile(self, func):  # pragma: no cover
        """Compile the given method.

        Available only for the ``tensorflow`` backend.
        """
        raise_error(NotImplementedError)

    ########################################################################################
    ######## Methods related to array manipulation                                  ########
    ########################################################################################

    def conj(self, array) -> "ndarray":
        return self.engine.conj(array)

    def identity(self, dims: int, dtype=None) -> "ndarray":
        if dtype is None:
            dtype = self.dtype
        return self.engine.eye(dims, dtype=dtype)

    def ones(self, shape, dtype=None) -> "ndarray":  # pragma: no cover
        if dtype is None:
            dtype = self.dtype
        return self.engine.ones(shape, dtype=dtype)

    def outer(self, array_1, array_2) -> "ndarray":  # pragma: no cover
        return self.engine.outer(array_1, array_2)

    def random_choice(self, array, **kwargs) -> "ndarray":  # pragma: no cover
        return self.engine.random.choice(array, **kwargs)

    def reshape(
        self, array, shape: Union[Tuple[int, ...], List[int]], **kwargs
    ) -> "ndarray":
        return self.engine.reshape(array, shape=shape, **kwargs)

    def sum(self, array, axis=None, **kwargs) -> Union[int, float, complex, "ndarray"]:
        return self.engine.sum(array, axis=axis, **kwargs)

    def trace(self, array) -> Union[int, float]:
        return self.engine.trace(array)

    def transpose(self, array, axes: Union[Tuple[int, ...], List[int]]) -> "ndarray":
        return self.engine.transpose(array, axes)

    def zeros(self, shape, dtype=None) -> "ndarray":  # pragma: no cover
        if dtype is None:
            dtype = self.dtype
        return self.engine.zeros(shape, dtype=dtype)

    ########################################################################################
    ######## Methods related to the creation and manipulation of quantum objects    ########
    ########################################################################################

    def zero_state(
        self, nqubits: int, density_matrix: bool = False, dtype=None
    ) -> "ndarray":  # pragma: no cover
        """Generate the :math:`n`-fold tensor product of the single-qubit :math:`\\ket{0}` state.

        Args:
            nqubits (int): Number of qubits :math:`n`.
            density_matrix (bool, optional): If ``True``, returns the density matrix
                :math:`\\ketbra{0}^{\\otimes \\, n}`. If ``False``, returns the statevector
                :math:`\\ket{0}^{\\otimes \\, n}`. Defaults to ``False``.

        Returns:
            ndarray: Array representation of the :math:`n`-qubit zero state.
        """
        if dtype is None:
            dtype = self.dtype

        dims = 2**nqubits
        shape = 2 * (dims,) if density_matrix else dims
        indexes = [0, 0] if density_matrix else 0

        state = self.zeros(shape, dtype=dtype)
        state[indexes] = 1

        return state

    def plus_state(
        self, nqubits: int, density_matrix: bool = False, dtype=None
    ):  # pragma: no cover
        """Generate :math:`|+++\\cdots+\\rangle` state vector as an array."""
        if dtype is None:
            dtype = self.dtype

        dims = 2**nqubits
        normalization = dims if density_matrix else math.sqrt(dims)
        shape = 2 * (dims,) if density_matrix else dims

        state = self.ones(shape, dtype=dtype)
        state /= normalization

        return state

    def maximally_mixed_state(
        self, nqubits: int, dtype=None
    ) -> "ndarray":  # pragma: no cover
        """Generate the :math:`n`-qubit density matrix for the maximally mixed state.

        .. math::
            \\rho = \\frac{I}{2^{n}} \\, ,

        where :math:`I` is the :math:`2^{n} \\times 2^{n}` identity operator.

        Args:
            nqubits (int): Number of qubits :math:`n`.

        """
        if dtype is None:
            dtype = self.dtype

        dims = 2**nqubits
        state = self.identity(dims, dtype=dtype)
        state /= dims

        return state

    def reset_error_density_matrix(self, gate, state, nqubits: int):  # pragma: no cover
        """Apply reset error to density matrix."""
        from qibo.gates import X  # pylint: disable=import-outside-toplevel
        from qibo.quantum_info.linalg_operations import (  # pylint: disable=import-outside-toplevel
            partial_trace,
        )

        state = self.cast(state)
        shape = state.shape
        q = gate.target_qubits[0]
        p_0, p_1 = gate.init_kwargs["p_0"], gate.init_kwargs["p_1"]
        trace = partial_trace(state, (q,), backend=self)
        trace = self.engine.reshape(trace, 2 * (nqubits - 1) * (2,))
        zero = self.zero_state(1, density_matrix=True)
        zero = self.engine.tensordot(trace, zero, 0)
        order = list(range(2 * nqubits - 2))
        order.insert(q, 2 * nqubits - 2)
        order.insert(q + nqubits, 2 * nqubits - 1)
        zero = self.engine.reshape(self.engine.transpose(zero, order), shape)
        state = (1 - p_0 - p_1) * state + p_0 * zero
        return state + p_1 * self.apply_gate_density_matrix(X(q), zero, nqubits)

    def thermal_error_density_matrix(
        self, gate, state, nqubits: int
    ):  # pragma: no cover
        """Apply thermal relaxation error to density matrix."""
        raise_error(NotImplementedError)

    ########################################################################################
    ######## Methods related to circuit execution                                   ########
    ########################################################################################

    def matrix(self, gate: "qibo.gates.abstract.Gate") -> "ndarray":  # pragma: no cover
        """Convert a gate to its matrix representation in the computational basis."""
        name = gate.__class__.__name__
        _matrix = getattr(self.matrices, name)
        if callable(_matrix):
            _matrix = _matrix(2 ** len(gate.target_qubits))
        return self.cast(_matrix, dtype=_matrix.dtype)

    def matrix_parametrized(self, gate: "qibo.gates.abstract.Gate"):  # pragma: no cover
        """Convert a parametrized gate to its matrix representation in the computational basis."""
        name = gate.__class__.__name__

        _matrix = getattr(self.matrices, name)
        if name == "GeneralizedRBS":
            _matrix = _matrix(
                qubits_in=gate.init_args[0],
                qubits_out=gate.init_args[1],
                theta=gate.init_kwargs["theta"],
                phi=gate.init_kwargs["phi"],
            )
        else:
            _matrix = _matrix(*gate.parameters)

        return self.cast(_matrix, dtype=_matrix.dtype)

    def matrix_fused(self, gate):  # pragma: no cover
        """Fuse matrices of multiple gates."""
        raise_error(NotImplementedError)

    def apply_gate(
        self, gate, state, nqubits: int, density_matrix: bool = False
    ) -> "ndarray":  # pragma: no cover
        """Apply a gate to state vector."""
        state = self.reshape(state, nqubits * (2,))

        if gate.is_controlled_by and density_matrix:
            return self._apply_gate_controlled_by_density_matrix(gate, state, nqubits)

        if gate.is_controlled_by:
            return self._apply_gate_controlled_by(gate, state, nqubits)

        matrix = gate.matrix(self)
        matrix = self.reshape(matrix, 2 * len(gate.qubits) * (2,))

        if density_matrix:
            matrix_conj = self.engine.conj(matrix)
            left, right = einsum_utils.apply_gate_density_matrix_string(
                gate.qubits, nqubits
            )
            state = self.engine.einsum(right, state, matrix_conj)
            state = self.engine.einsum(left, state, matrix)
        else:
            opstring = einsum_utils.apply_gate_string(gate.qubits, nqubits)
            state = self.engine.einsum(opstring, state, matrix)

        shape = (2**nqubits,)
        if density_matrix:
            shape *= 2

        return self.reshape(state, shape)

    def apply_gate_half_density_matrix(
        self, gate, state, nqubits: int
    ):  # pragma: no cover
        """Apply a gate to one side of the density matrix."""
        if gate.is_controlled_by:
            raise_error(
                NotImplementedError,
                "Gate density matrix half call is "
                "not implemented for ``controlled_by``"
                "gates.",
            )

        state = self.cast(state, dtype=state.dtype)
        state = self.reshape(state, 2 * nqubits * (2,))
        matrix = self.matrix(gate)

        matrix = self.reshape(matrix, 2 * len(gate.qubits) * (2,))
        left, _ = einsum_utils.apply_gate_density_matrix_string(gate.qubits, nqubits)
        state = self.engine.einsum(left, state, matrix)

        return self.reshape(state, 2 * (2**nqubits,))

    def apply_channel(
        self, channel, state, nqubits: int, density_matrix: bool = False
    ):  # pragma: no cover
        """Apply a ``channel`` to quantum ``state``."""

        if density_matrix:
            state = self.cast(state, dtype=state.dtype)

            new_state = (1 - channel.coefficient_sum) * state
            for coeff, gate in zip(channel.coefficients, channel.gates):
                new_state += coeff * self.apply_gate_density_matrix(
                    gate, state, nqubits
                )

            return new_state

        probabilities = channel.coefficients + (1 - self.sum(channel.coefficients),)

        index = int(self.sample_shots(probabilities, 1)[0])
        if index != len(channel.gates):
            gate = channel.gates[index]
            state = self.apply_gate(gate, state, nqubits)

        return state

    def collapse_state(
        self,
        state,
        qubits: Union[Tuple[int, ...], List[int]],
        shot: int,
        nqubits: int,
        normalize: bool = True,
        density_matrix: bool = False,
    ) -> "ndarray":
        """Collapse state vector according to measurement shot."""

        if density_matrix:
            return self._collapse_density_matrix(
                state, qubits, shot, nqubits, normalize
            )

        return self._collapse_statevector(state, qubits, shot, nqubits, normalize)

    def execute_circuit(
        self, circuit, initial_state=None, nshots: int = None
    ):  # pragma: no cover
        """Execute a :class:`qibo.models.circuit.Circuit`."""
        raise_error(NotImplementedError)

    def execute_circuits(
        self, circuits, initial_states=None, nshots: int = None
    ):  # pragma: no cover
        """Execute multiple :class:`qibo.models.circuit.Circuit` in parallel."""
        raise_error(NotImplementedError)

    def execute_circuit_repeated(
        self, circuit: "qibo.models.circuit.Circuit", nshots: int, initial_state=None
    ):  # pragma: no cover
        """Execute a :class:`qibo.models.circuit.Circuit` multiple times.

        Useful for noise simulation using state vectors or for simulating gates
        controlled by measurement outcomes.
        """
        raise_error(NotImplementedError)

    def execute_distributed_circuit(
        self, circuit, initial_state=None, nshots: int = None
    ):  # pragma: no cover
        """Execute a :class:`qibo.models.circuit.Circuit` using multiple GPUs."""
        raise_error(NotImplementedError)

    def calculate_symbolic(
        self,
        state,
        nqubits: int,
        decimals: int = 5,
        cutoff: float = 1e-10,
        max_terms: int = 20,
    ):  # pragma: no cover
        """Dirac representation of a state vector."""
        raise_error(NotImplementedError)

    def calculate_symbolic_density_matrix(
        self,
        state,
        nqubits: int,
        decimals: int = 5,
        cutoff: float = 1e-10,
        max_terms: int = 20,
    ):  # pragma: no cover
        """Dirac representation of a density matrix."""
        raise_error(NotImplementedError)

    def calculate_probabilities(self, state, qubits, nqubits: int):  # pragma: no cover
        """Calculate probabilities given a state vector."""
        raise_error(NotImplementedError)

    def calculate_probabilities_density_matrix(
        self, state, qubits, nqubits: int
    ):  # pragma: no cover
        """Calculate probabilities given a density matrix."""
        raise_error(NotImplementedError)

    def set_seed(self, seed):  # pragma: no cover
        """Set the seed of the random number generator."""
        raise_error(NotImplementedError)

    def sample_shots(self, probabilities, nshots: int):  # pragma: no cover
        """Sample measurement shots according to a probability distribution."""
        raise_error(NotImplementedError)

    def aggregate_shots(self, shots):  # pragma: no cover
        """Collect shots to a single array."""
        raise_error(NotImplementedError)

    def samples_to_binary(self, samples, nqubits: int):  # pragma: no cover
        """Convert samples from decimal representation to binary."""
        raise_error(NotImplementedError)

    def samples_to_decimal(self, samples, nqubits: int):  # pragma: no cover
        """Convert samples from binary representation to decimal."""
        raise_error(NotImplementedError)

    def calculate_frequencies(self, samples):  # pragma: no cover
        """Calculate measurement frequencies from shots."""
        raise_error(NotImplementedError)

    def update_frequencies(
        self, frequencies, probabilities, nsamples: int
    ):  # pragma: no cover
        raise_error(NotImplementedError)

    def sample_frequencies(self, probabilities, nshots: int):  # pragma: no cover
        """Sample measurement frequencies according to a probability distribution."""
        raise_error(NotImplementedError)

    def calculate_vector_norm(
        self, state, order: Union[int, float, str] = 2
    ):  # pragma: no cover
        """Calculate norm of an :math:`1`-dimensional array.

        For specifications on possible values of the parameter ``order``
        for the ``tensorflow`` backend, please refer to
        `tensorflow.norm <https://www.tensorflow.org/api_docs/python/tf/norm>`_.
        For all other backends, please refer to
        `numpy.linalg.norm <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>`_.
        """
        raise_error(NotImplementedError)

    def calculate_matrix_norm(
        self, state, order: Union[int, float, str] = "nuc"
    ):  # pragma: no cover
        """Calculate norm of a :math:`2`-dimensional array.

        Default is the ``nuclear`` norm.
        If ``order="nuc"``, it returns the nuclear norm of ``state``,
        assuming ``state`` is Hermitian (also known as trace norm).
        For specifications on the other  possible values of the
        parameter ``order`` for the ``tensorflow`` backend, please refer to
        `tensorflow.norm <https://www.tensorflow.org/api_docs/python/tf/norm>`_.
        For all other backends, please refer to
        `numpy.linalg.norm <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>`_.
        """
        raise_error(NotImplementedError)

    def calculate_overlap(self, state1, state2):  # pragma: no cover
        """Calculate overlap of two state vectors."""
        raise_error(NotImplementedError)

    def calculate_overlap_density_matrix(self, state1, state2):  # pragma: no cover
        """Calculate overlap of two density matrices."""
        raise_error(NotImplementedError)

    def calculate_eigenvalues(
        self, matrix, k: int = 6, hermitian: bool = True
    ):  # pragma: no cover
        """Calculate eigenvalues of a matrix."""
        raise_error(NotImplementedError)

    def calculate_eigenvectors(
        self, matrix, k: int = 6, hermitian: bool = True
    ):  # pragma: no cover
        """Calculate eigenvectors of a matrix."""
        raise_error(NotImplementedError)

    def calculate_expectation_state(
        self, hamiltonian, state, normalize: bool
    ):  # pragma: no cover
        """Calculate expectation value of a state vector given the observable matrix."""
        raise_error(NotImplementedError)

    def calculate_expectation_density_matrix(
        self, hamiltonian, state, normalize: bool
    ):  # pragma: no cover
        """Calculate expectation value of a density matrix given the observable matrix."""
        raise_error(NotImplementedError)

    def calculate_matrix_exp(
        self,
        matrix,
        phase: Union[float, int, complex] = 1,
        eigenvectors=None,
        eigenvalues=None,
    ):  # pragma: no cover
        """Calculate the exponential :math:`e^{\\theta \\, A}` of a matrix :math:`A`
        and ``phase`` :math:`\\theta`.

        If the eigenvectors and eigenvalues are given the matrix diagonalization is
        used for exponentiation.
        """
        raise_error(NotImplementedError)

    def calculate_matrix_log(
        self, matrix, base: Union[float, int] = 2, eigenvectors=None, eigenvalues=None
    ):  # pragma: no cover
        """Calculate the logarithm :math:`\\log_{b}(A)` with a ``base`` :math:`b`
        of a matrix :math:`A`.

        If the eigenvectors and eigenvalues are given the matrix diagonalization is
        used for exponentiation.
        """
        raise_error(NotImplementedError)

    def calculate_matrix_power(
        self, matrix, power: Union[float, int], precision_singularity: float = 1e-14
    ):  # pragma: no cover
        """Calculate the (fractional) ``power`` :math:`\\alpha` of ``matrix`` :math:`A`,
        i.e. :math:`A^{\\alpha}`.

        .. note::
            For the ``pytorch`` backend, this method relies on a copy of the original tensor.
            This may break the gradient flow. For the GPU backends (i.e. ``cupy`` and
            ``cuquantum``), this method falls back to CPU whenever ``power`` is not
            an integer.
        """
        raise_error(NotImplementedError)

    def calculate_matrix_sqrt(
        self, matrix, precision_singularity: float = 1e-14
    ):  # pragma: no cover
        """Calculate the square root of ``matrix`` :math:`A`, i.e. :math:`A^{1/2}`.

        .. note::
            For the ``pytorch`` backend, this method relies on a copy of the original tensor.
            This may break the gradient flow. For the GPU backends (i.e. ``cupy`` and
            ``cuquantum``), this method falls back to CPU.
        """
        raise_error(NotImplementedError)

    def calculate_singular_value_decomposition(self, matrix):  # pragma: no cover
        """Calculate the Singular Value Decomposition of ``matrix``."""
        raise_error(NotImplementedError)

    def calculate_jacobian_matrix(
        self, circuit, parameters, initial_state=None, return_complex: bool = True
    ):  # pragma: no cover
        """Calculate the Jacobian matrix of ``circuit`` with respect to varables ``params``."""
        raise_error(NotImplementedError)

    def assert_allclose(
        self, value, target, rtol: float = 1e-7, atol: float = 0.0
    ):  # pragma: no cover
        raise_error(NotImplementedError)

    def assert_circuitclose(
        self, circuit, target_circuit, rtol: float = 1e-7, atol: float = 0.0
    ):
        value = self.execute_circuit(circuit)._state
        target = self.execute_circuit(target_circuit)._state
        self.assert_allclose(value, target, rtol=rtol, atol=atol)

    def _apply_gate_controlled_by(self, gate, state, nqubits: int) -> "ndarray":
        matrix = gate.matrix(self)
        matrix = self.reshape(matrix, 2 * len(gate.target_qubits) * (2,))
        ncontrol = len(gate.control_qubits)
        nactive = nqubits - ncontrol
        order, targets = einsum_utils.control_order(gate, nqubits)
        state = self.transpose(state, order)
        # Apply `einsum` only to the part of the state where all controls
        # are active. This should be `state[-1]`
        state = self.reshape(state, (2**ncontrol,) + nactive * (2,))
        opstring = einsum_utils.apply_gate_string(targets, nactive)
        updates = self.engine.einsum(opstring, state[-1], matrix)
        # Concatenate the updated part of the state `updates` with the
        # part of of the state that remained unaffected `state[:-1]`.
        state = self.engine.concatenate([state[:-1], updates[None]], axis=0)
        state = self.reshape(state, nqubits * (2,))
        # Put qubit indices back to their proper places
        state = self.transpose(state, einsum_utils.reverse_order(order))

        return self.reshape(state, shape=(2**nqubits,))

    def _apply_gate_controlled_by_density_matrix(
        self, gate, state, nqubits: int
    ) -> "ndarray":
        matrix = gate.matrix(self)
        matrix = self.reshape(matrix, 2 * len(gate.target_qubits) * (2,))
        matrixc = self.engine.conj(matrix)
        ncontrol = len(gate.control_qubits)
        nactive = nqubits - ncontrol
        dims_ctrl = 2**ncontrol

        order, targets = einsum_utils.control_order_density_matrix(gate, nqubits)
        state = self.transpose(state, order)
        state = self.reshape(state, 2 * (dims_ctrl,) + 2 * nactive * (2,))

        leftc, rightc = einsum_utils.apply_gate_density_matrix_controlled_string(
            targets, nactive
        )
        state01 = state[: dims_ctrl - 1, dims_ctrl - 1]
        state01 = self.engine.einsum(rightc, state01, matrixc)
        state10 = state[dims_ctrl - 1, : dims_ctrl - 1]
        state10 = self.engine.einsum(leftc, state10, matrix)

        left, right = einsum_utils.apply_gate_density_matrix_string(targets, nactive)
        state11 = state[dims_ctrl - 1, dims_ctrl - 1]
        state11 = self.engine.einsum(right, state11, matrixc)
        state11 = self.engine.einsum(left, state11, matrix)

        state00 = state[range(dims_ctrl - 1)]
        state00 = state00[:, range(dims_ctrl - 1)]
        state01 = self.engine.concatenate([state00, state01[:, None]], axis=1)
        state10 = self.engine.concatenate([state10, state11[None]], axis=0)
        state = self.engine.concatenate([state01, state10[None]], axis=0)
        state = self.reshape(state, 2 * nqubits * (2,))
        state = self.transpose(state, einsum_utils.reverse_order(order))

        return self.reshape(state, 2 * (2**nqubits,))

    def _append_zeros(self, state, qubits, results):
        """Helper function for the ``collapse_state`` method."""
        for q, r in zip(qubits, results):
            state = self.engine.expand_dims(state, q)
            state = (
                self.engine.concatenate([self.engine.zeros_like(state), state], q)
                if r == 1
                else self.engine.concatenate([state, self.engine.zeros_like(state)], q)
            )
        return state

    def _collapse_density_matrix(
        self, state, qubits, shot, nqubits: int, normalize: bool = True
    ):  # pragma: no cover
        state = self.cast(state, dtype=state.dtype)
        shape = state.shape
        binshot = list(self.samples_to_binary(shot, len(qubits))[0])
        order = list(qubits) + [qubit + nqubits for qubit in qubits]
        order.extend(qubit for qubit in range(nqubits) if qubit not in qubits)
        order.extend(qubit + nqubits for qubit in range(nqubits) if qubit not in qubits)
        state = self.reshape(state, 2 * nqubits * (2,))
        state = self.transpose(state, order)
        subshape = 2 * (2 ** len(qubits),) + 2 * (nqubits - len(qubits)) * (2,)
        state = self.reshape(state, subshape)[int(shot), int(shot)]
        dims = 2 ** (len(state.shape) // 2)

        if normalize:
            norm = self.trace(self.reshape(state, 2 * (dims,)))
            state = state / norm

        qubits = qubits + [qubit + nqubits for qubit in qubits]
        state = self._append_zeros(state, qubits, 2 * binshot)

        return self.reshape(state, shape)

    def _collapse_state(self, state, qubits, shot, nqubits, normalize=True):
        state = self.cast(state)
        shape = state.shape
        binshot = list(self.samples_to_binary(shot, len(qubits))[0])
        state = self.reshape(state, nqubits * (2,))
        order = list(qubits) + [
            qubit for qubit in range(nqubits) if qubit not in qubits
        ]
        state = self.transpose(state, order)
        subshape = (2 ** len(qubits),) + (nqubits - len(qubits)) * (2,)
        state = self.reshape(state, subshape)[int(shot)]

        if normalize:
            norm = self.engine.sqrt(self.sum(self.engine.abs(state) ** 2))
            state = state / norm

        state = self._append_zeros(state, qubits, binshot)

        return self.engine.reshape(state, shape)
