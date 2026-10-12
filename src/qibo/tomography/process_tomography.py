"""Vanilla quantum process tomography with state tomography and linear inversion."""

from functools import reduce
from itertools import product

from numpy.typing import ArrayLike

from qibo import Circuit, gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.quantum_info._superoperator_transformations import (
    _operator_to_pauli_vectors_fht,
)
from qibo.quantum_info.superoperator_transformations import (
    pauli_to_choi,
    pauli_to_liouville,
)
from qibo.tomography.abstract import Tomography
from qibo.tomography.state_tomography import StateTomography


class ProcessTomography(Tomography):
    """Vanilla quantum process tomography by linear inversion.

    The process :math:`\\mathcal{E}` of :math:`n` qubits, which is the action of a circuit
    on the state of its qubits, is characterized by its Pauli transfer matrix,

    .. math::
        R_{ij} = \\frac{1}{2^{n}} \\text{tr}\\left(P_{i} \\, \\mathcal{E}(P_{j})\\right) \\, ,

    where :math:`P_{i}` are the :math:`4^{n}` Pauli strings. It maps the vector of Pauli
    coefficients :math:`s_{j} = \\text{tr}(P_{j} \\, \\rho)` of a state to that of its output,
    :math:`o_{i} = \\text{tr}(P_{i} \\, \\mathcal{E}(\\rho)) = \\sum_{j} R_{ij} \\, s_{j}`.

    The process is probed with the :math:`4^{n}` product states in which each qubit is in one
    of :math:`|0\\rangle`, :math:`|1\\rangle`, :math:`|+\\rangle`, or :math:`|{+i}\\rangle`,
    whose Pauli coefficients :math:`s^{(k)}` are linearly independent. The output of each of
    them is reconstructed with :class:`qibo.tomography.state_tomography.StateTomography`,
    which needs :math:`3^{n}` measurement settings, so there are :math:`12^{n}` circuits.
    Collecting the coefficients in the matrices :math:`S_{kj} = s^{(k)}_{j}` and
    :math:`O_{ki} = o^{(k)}_{i}`, which satisfy :math:`O = S \\, R^{T}`, the estimate is
    :math:`\\hat{R} = (S^{-1} \\, O)^{T}` [1, 2].

    The estimate is unbiased, and it is trace preserving, i.e. its first row is exactly
    :math:`(1, 0, \\dots, 0)`, but in general it is **not** completely positive, since the
    statistical errors can make the Choi matrix have negative eigenvalues. The number of circuits
    grows as :math:`12^{n}`, so this protocol is meant for few qubits. The errors in the
    preparation of the input states and in the measurements are not corrected, and they are
    part of the reconstructed process.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.tomography import ProcessTomography

            circuit = Circuit(1)
            circuit.add(gates.H(0))

            # Pauli transfer matrix of the Hadamard gate, from 12 circuits of 1000 shots
            estimate = ProcessTomography()(circuit, nshots=1000, seed=1)

    References:
        1. I. L. Chuang and M. A. Nielsen, *Prescription for experimental determination
           of the dynamics of a quantum black box*,
           `J. Mod. Opt. 44, 2455 (1997) <https://doi.org/10.1080/09500349708231894>`_.
        2. M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*,
           Cambridge University Press (2010), Sec. 8.4.2.
    """

    def __call__(
        self,
        circuit: Circuit,
        nshots: int = 1000,
        representation: str = "pauli",
        seed: int | None = None,
        backend: Backend = None,
    ) -> ArrayLike:
        """Estimates the process implemented by ``circuit``.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit that implements the
                process, without measurements. It can have noise channels, and it can be a
                circuit of density matrices.
            nshots (int, optional): number of shots per circuit. Defaults to :math:`1000`.
            representation (str, optional): representation of the estimated process. If
                ``"pauli"``, it is the Pauli transfer matrix, which is real, in the Pauli
                basis ordered as :math:`I, X, Y, Z` for each qubit, the first qubit being
                the most significant. This is the Pauli-Liouville representation, in the
                normalized Pauli basis, of :mod:`qibo.quantum_info`. If ``"liouville"``, or
                ``"choi"``, it is converted to that representation, with the row
                vectorization, using :func:`qibo.quantum_info.pauli_to_liouville`, or
                :func:`qibo.quantum_info.pauli_to_choi`. Defaults to ``"pauli"``.
            seed (int, optional): seed of the sampling of the outcomes. If ``None``, the
                generator of the backend is not reseeded. Defaults to ``None``.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend
                to be used in the execution. If ``None``, it uses the current
                backend. Defaults to ``None``.

        Returns:
            ArrayLike: :math:`4^{n} \\times 4^{n}` estimated process, in ``representation``.
        """
        representations = ("pauli", "liouville", "choi")
        if representation not in representations:
            raise_error(
                ValueError,
                f"``representation`` must be one of {representations}, "
                + f"but it is {representation}.",
            )

        backend = _check_backend(backend)

        prepared_circuits = self._prepared_circuits(circuit)

        # the sampling of the outcomes of all the circuits follows from one seed
        if seed is not None:
            backend.set_seed(seed)

        nqubits = circuit.nqubits
        dim = 2**nqubits

        # Pauli coefficients of the output state of each of the input states
        state_tomography = StateTomography()
        outputs = backend.zeros((4**nqubits, 4**nqubits), dtype=backend.float64)
        for index, prepared_circuit in enumerate(prepared_circuits):
            state = state_tomography(prepared_circuit, nshots, backend=backend)
            outputs[index] = backend.real(
                _operator_to_pauli_vectors_fht(state, nqubits, dim, backend=backend)
            )

        # Pauli coefficients, for I, X, Y, Z, of the input states |0>, |1>, |+>, |+i>,
        # which is a matrix that is invertible for each qubit, and so for all of them
        single_qubit_inputs = backend.cast(
            [[1, 0, 0, 1], [1, 0, 0, -1], [1, 1, 0, 0], [1, 0, 1, 0]],
            dtype=backend.float64,
        )
        inverse_inputs = reduce(
            backend.kron, [backend.inv(single_qubit_inputs)] * nqubits
        )

        transfer_matrix = backend.transpose(backend.matmul(inverse_inputs, outputs))

        if representation == "liouville":
            return pauli_to_liouville(transfer_matrix, normalize=True, backend=backend)

        if representation == "choi":
            return pauli_to_choi(transfer_matrix, normalize=True, backend=backend)

        return transfer_matrix

    def circuits(self, circuit: Circuit) -> list[Circuit]:
        """Builds the circuits that probe the process implemented by ``circuit``.

        Each input state is prepared before ``circuit``, and the output state is measured
        in all the settings of :meth:`qibo.tomography.state_tomography.StateTomography.circuits`.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit that implements the
                process, without measurements.

        Returns:
            list: :math:`12^{n}` circuits. The input states change the slowest, in the order
            of ``itertools.product(range(4), repeat=n)``, with
            :math:`|0\\rangle, |1\\rangle, |+\\rangle, |{+i}\\rangle` as :math:`0, 1, 2, 3`,
            and the first qubit changes the slowest. They are prepared from the zero state
            with :class:`qibo.gates.X`, for :math:`|1\\rangle`, :class:`qibo.gates.H`,
            for :math:`|+\\rangle`, and :class:`qibo.gates.H` followed by
            :class:`qibo.gates.S`, for :math:`|{+i}\\rangle`. For each of them, there are
            the :math:`3^{n}` circuits of the settings, in their order.
        """
        state_tomography = StateTomography()

        return [
            tomography_circuit
            for prepared_circuit in self._prepared_circuits(circuit)
            for tomography_circuit in state_tomography.circuits(prepared_circuit)
        ]

    def _prepared_circuits(self, circuit: Circuit) -> list[Circuit]:
        """Prepends each of the :math:`4^{n}` input states to ``circuit``."""
        self._check_circuit(circuit)

        prepared_circuits = []
        for states in product(range(4), repeat=circuit.nqubits):
            prepared_circuit = circuit.__class__(**circuit.init_kwargs)
            for qubit, state in enumerate(states):
                if state == 1:
                    prepared_circuit.add(gates.X(qubit))
                if state >= 2:
                    prepared_circuit.add(gates.H(qubit))
                if state == 3:
                    prepared_circuit.add(gates.S(qubit))
            prepared_circuit.add(circuit.queue)
            prepared_circuits.append(prepared_circuit)

        return prepared_circuits
