import math
from functools import cache
from inspect import signature
from itertools import product

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from sympy import S

from qibo import Circuit, gates, symbols
from qibo.backends import Backend, _check_backend, construct_backend
from qibo.config import raise_error
from qibo.gates.abstract import Gate
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.noise import NoiseModel
from qibo.symbols import Symbol
from qibo.tomography.abstract import Tomography
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.pipeline import Passes
from qibo.transpiler.placer import Random
from qibo.transpiler.router import Sabre
from qibo.transpiler.unroller import NativeGates, Unroller

SUPPORTED_NQUBITS = [1, 2]
"""Supported nqubits for GST."""

ANGLES = ["theta", "phi", "lam", "unitary"]
"""Angle names for parametrized gates."""

FIDUCIAL_BASES = (gates.Z, gates.X, gates.Y, gates.Z)
"""Bases in which the qubits are measured to estimate the Pauli operators
:math:`I`, :math:`X`, :math:`Y` and :math:`Z`, respectively."""

FIDUCIAL_STATES = ((gates.I,), (gates.X,), (gates.H,), (gates.H, gates.S))
"""Gates, in order of application, that prepare the fiducial states
:math:`|0\\rangle\\langle 0|`, :math:`|1\\rangle\\langle 1|`, :math:`|+\\rangle\\langle +|`
and :math:`|y+\\rangle\\langle y+|` from the zero state."""

GAUGE_MATRIX = ((1, 1, 1, 1), (0, 0, 1, 0), (0, 0, 0, 1), (1, -1, 0, 0))
"""Default single-qubit gauge matrix. Its columns are the fiducial states in the
Pauli basis."""

MEASURED_PAULIS = {
    nqubits: tuple(product(range(4), repeat=nqubits))[1:]
    for nqubits in SUPPORTED_NQUBITS
}
"""Pauli strings measured for each supported number of qubits, as tuples of the digits
:math:`\\{0, 1, 2, 3\\} \\equiv \\{I, X, Y, Z\\}`. The identity string is excluded, since
its expectation value is always :math:`1`."""


class GateSetTomography(Tomography):
    """Gate set tomography (GST) of one- and two-qubit circuits by linear inversion.

    Given a circuit implementing an operation :math:`O` on :math:`n \\in \\{1, 2\\}`
    qubits, each fiducial state
    :math:`\\rho_{k} \\in \\{ |0\\rangle\\langle 0|, |1\\rangle\\langle 1|,
    |+\\rangle\\langle +|, |y+\\rangle\\langle y+| \\}^{\\otimes n}` is prepared and
    :math:`O` is applied. The output is then measured in each Pauli basis
    :math:`M_{j} \\in \\{ I, X, Y, Z \\}^{\\otimes n}`, which estimates the
    :math:`4^{n} \\times 4^{n}` matrix

    .. math::
        \\tilde{O}_{jk} = \\text{tr}(M_{j} \\, O \\, \\rho_{k}) \\, .

    The same experiments for an empty circuit estimate the Gram matrix of the
    fiducial states and measurements,

    .. math::
        \\tilde{g}_{jk} = \\text{tr}(M_{j} \\, \\rho_{k}) \\, ,

    which accounts for state preparation and measurement errors. Then,
    the linear inversion of Ref. [1] (Sec. 3.2) gives the estimate of :math:`O` in the
    Pauli-Liouville representation, also known as Pauli transfer matrix,

    .. math::
        O^{\\text{PL}} = T \\, \\tilde{g}^{-1} \\, \\tilde{O} \\, T^{-1} \\, ,

    where :math:`T` is the gauge matrix. The estimate is defined up to a gauge
    transformation, and the default :math:`T` is the one that corresponds to the ideal
    fiducial states, which is the best *a priori* choice in Ref. [1] (Sec. 3.2). It is
    the tensor product of the single-qubit matrix

    .. math::
        T = \\begin{pmatrix}
            1 & 1 & 1 & 1 \\\\
            0 & 0 & 1 & 0 \\\\
            0 & 0 & 0 & 1 \\\\
            1 & -1 & 0 & 0
        \\end{pmatrix} \\, .

    .. note::
        The Gram matrix must be well-conditioned for its inverse not to amplify the
        statistical error. Its singular values can be inspected before the
        inversion, by calling the protocol on an empty circuit [1].

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.tomography import GateSetTomography

            circuit = Circuit(1)
            circuit.add(gates.X(0))

            # Pauli-Liouville representation of a Pauli X gate
            gst = GateSetTomography()
            estimate = gst(circuit, nshots=1000, pauli_liouville=True)

    References:
        1. E. Nielsen *et al.*, *Gate set tomography*,
           `Quantum 5, 557 (2021) <https://doi.org/10.22331/q-2021-10-05-557>`_,
           `arXiv:2009.07301 <https://arxiv.org/abs/2009.07301>`_.
        2. R. Blume-Kohout *et al.*, *Robust, self-consistent, closed-form tomography
           of quantum logic gates on a trapped ion qubit*,
           `arXiv:1310.4492 <https://arxiv.org/abs/1310.4492>`_.
        3. D. Greenbaum, *Introduction to quantum gate set tomography*,
           `arXiv:1509.02921 <https://arxiv.org/abs/1509.02921>`_.
        4. S. Endo, S. C. Benjamin and Y. Li, *Practical quantum error mitigation for
           near-future applications*, `Phys. Rev. X 8, 031027 (2018)
           <https://doi.org/10.1103/PhysRevX.8.031027>`_.
    """

    def __call__(
        self,
        circuit: Circuit,
        nshots: int = int(1e4),
        noise_model: NoiseModel | None = None,
        auxiliary: list[int] | None = None,
        pauli_liouville: bool = False,
        gauge_matrix: ArrayLike | None = None,
        gram_matrix: ArrayLike | None = None,
        transpiler: Passes | None = None,
        backend: Backend | None = None,
    ) -> ArrayLike:
        """Estimates the matrix of the operation implemented by ``circuit``.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit, on :math:`1` or
                :math:`2` qubits and without measurements, implementing the operation
                to be characterized. An empty circuit gives the Gram matrix. Two
                single-qubit gates acting on different qubits of a :math:`2`-qubit
                circuit are characterized simultaneously.
            nshots (int, optional): number of shots per circuit. Defaults to
                :math:`10^{4}`.
            noise_model (:class:`qibo.noise.NoiseModel`, optional): noise model applied
                to simulate noisy computations. Defaults to ``None``.
            auxiliary (list[int], optional): qubits of ``circuit`` that are swapped
                with fresh auxiliary qubits, in the zero state, right after the
                fiducial state is prepared, i.e. they are reset before ``circuit``
                acts on them. Then, the columns of the matrix do not depend on the
                fiducial state of those qubits. The Gram matrix is always estimated
                without auxiliary qubits. Defaults to ``None``.
            pauli_liouville (bool, optional): if ``True``, returns the estimate in
                the Pauli-Liouville representation. Defaults to ``False``.
            gauge_matrix (ArrayLike, optional): invertible :math:`4 \\times 4` gauge
                matrix of a single qubit, used if ``pauli_liouville=True``. For
                :math:`2` qubits, its tensor product with itself is used. If
                ``None``, it is the matrix of the ideal fiducial states, given in the
                class description. Defaults to ``None``.
            gram_matrix (ArrayLike, optional): Gram matrix used if
                ``pauli_liouville=True``. If ``None``, it is estimated with the same
                ``nshots``, ``noise_model`` and ``transpiler``. It can be provided to
                avoid estimating it again when characterizing several circuits.
                Defaults to ``None``.
            transpiler (:class:`qibo.transpiler.pipeline.Passes`, optional):
                transpiler applied to the circuits before their execution. If
                ``None`` and the backend is ``qibolab``, it uses a transpiler built from
                the connectivity and the native gates of the backend. The placement is
                the same for all the circuits. Defaults to ``None``.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend
                to be used in the execution. If ``None``, it uses the current
                backend. Defaults to ``None``.

        Returns:
            ArrayLike: :math:`4^{n} \\times 4^{n}` matrix :math:`\\tilde{O}`, or
            :math:`O^{\\text{PL}}` if ``pauli_liouville=True``.
        """
        backend = _check_backend(backend)

        if backend.name == "qibolab" and transpiler is None:
            connectivity = nx.Graph(backend.connectivity)
            connectivity.add_nodes_from(backend.qubits)
            transpiler = Passes(
                connectivity=connectivity,
                passes=[
                    Preprocessing(),
                    Sabre(),
                    Unroller(NativeGates[backend.natives]),
                ],
            )

        circuits = self.circuits(circuit, auxiliary)

        nqubits = circuit.nqubits
        dim = 4**nqubits

        if pauli_liouville:
            single_qubit_gauge = backend.cast(
                GAUGE_MATRIX if gauge_matrix is None else gauge_matrix,
                dtype=backend.complex128,
            )
            if (
                single_qubit_gauge.shape != (4, 4)
                or backend.det(single_qubit_gauge) == 0
            ):
                raise_error(
                    ValueError, "``gauge_matrix`` must be an invertible 4 x 4 matrix."
                )

            gauge = single_qubit_gauge
            for _ in range(nqubits - 1):
                gauge = backend.kron(gauge, single_qubit_gauge)

            if gram_matrix is None:
                gram_matrix = self(
                    Circuit(nqubits),
                    nshots,
                    noise_model,
                    transpiler=transpiler,
                    backend=backend,
                )
            gram_matrix = backend.cast(gram_matrix, dtype=backend.complex128)

        if noise_model is not None and backend.name != "qibolab":
            circuits = [noise_model.apply(gst_circuit) for gst_circuit in circuits]

        if transpiler is not None:
            circuits = [
                transpiler(gst_circuit, backend=backend)[0] for gst_circuit in circuits
            ]

        observables = [
            SymbolicHamiltonian(
                math.prod(
                    symbols.Z(qubit, backend=backend)
                    for qubit, pauli in enumerate(paulis)
                    if pauli != 0
                ),
                nqubits=nqubits,
                backend=backend,
            )
            for paulis in MEASURED_PAULIS[nqubits]
        ]

        # the results are ordered by fiducial state, and then by measured Pauli string
        results = iter(self.execute(circuits, nshots, backend))
        columns = [
            [1.0] + [next(results).expectation_from_samples(o) for o in observables]
            for _ in range(dim)
        ]
        matrix = backend.transpose(backend.cast(columns, dtype=backend.complex128))

        if pauli_liouville:
            return gauge @ backend.inv(gram_matrix) @ matrix @ backend.inv(gauge)

        return matrix

    def circuits(
        self, circuit: Circuit, auxiliary: list[int] | None = None
    ) -> list[Circuit]:
        """Builds the circuits that estimate the matrix of ``circuit``.

        Each circuit prepares a fiducial state, applies ``circuit``, and measures all
        the qubits in the basis of a Pauli string. The qubits are measured in the
        :math:`Z` basis for the identity, and the outcome is not used.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit, on :math:`1` or
                :math:`2` qubits and without measurements, implementing the operation
                to be characterized.
            auxiliary (list[int], optional): qubits of ``circuit`` that are swapped
                with fresh auxiliary qubits, appended to the circuit in the same order,
                right after the fiducial state is prepared. Defaults to ``None``.

        Returns:
            list: :math:`4^{n} (4^{n} - 1)` circuits, ordered by fiducial state, and
            then by Pauli string. The identity string is excluded, since its
            expectation value is always :math:`1`.
        """
        self._check_circuit(circuit)

        nqubits = circuit.nqubits
        if nqubits not in SUPPORTED_NQUBITS:
            raise_error(
                ValueError,
                f"``circuit`` has {nqubits} qubits, but GST supports circuits "
                + f"of {SUPPORTED_NQUBITS[0]} or {SUPPORTED_NQUBITS[1]} qubits.",
            )

        auxiliary = [] if auxiliary is None else list(auxiliary)
        if len(set(auxiliary)) != len(auxiliary) or any(
            not isinstance(qubit, int) or not 0 <= qubit < nqubits
            for qubit in auxiliary
        ):
            raise_error(
                ValueError,
                "``auxiliary`` must be a list of distinct qubits of ``circuit``, "
                + f"but it is {auxiliary}.",
            )

        circuits = []
        for state in product(FIDUCIAL_STATES, repeat=nqubits):
            for paulis in MEASURED_PAULIS[nqubits]:
                gst_circuit = Circuit(nqubits + len(auxiliary), density_matrix=True)

                for qubit, preparation in enumerate(state):
                    gst_circuit.add(gate(qubit) for gate in preparation)

                for index, qubit in enumerate(auxiliary):
                    gst_circuit.add(gates.SWAP(qubit, nqubits + index))

                gst_circuit.add(circuit.queue)
                gst_circuit.add(
                    gates.M(qubit, basis=FIDUCIAL_BASES[pauli])
                    for qubit, pauli in enumerate(paulis)
                )
                circuits.append(gst_circuit)

        return circuits


@cache
def _check_nqubits(nqubits):
    if nqubits not in SUPPORTED_NQUBITS:
        raise_error(
            ValueError,
            f"``nqubits`` given as {nqubits}. ``nqubits`` needs to be either 1 or 2.",
        )


@cache
def _gates(nqubits: int) -> list[tuple[Gate, ...]]:
    """Gates implementing all the GST state preparations.

    Args:
        nqubits (int): Number of qubits for the circuit.
    Returns:
        List(:class:`qibo.gates.Gate`): gates used to prepare the possible states.
    """

    return list(
        product(
            [(gates.I,), (gates.X,), (gates.H,), (gates.H, gates.S)], repeat=nqubits
        )
    )


@cache
def _measurements(nqubits: int) -> list[tuple[Gate, ...]]:
    """Measurement gates implementing all the GST measurement bases.

    Args:
        nqubits (int): Number of qubits for the circuit.
    Returns:
        List(:class:`qibo.gates.Gate`): gates implementing the possible measurement bases.
    """

    return list(product([gates.Z, gates.X, gates.Y, gates.Z], repeat=nqubits))


@cache
def _observables(nqubits: int) -> list[tuple[Symbol, ...]]:
    """All the observables measured in the GST protocol.

    Args:
        nqubits (int): number of qubits for the circuit.

    Returns:
        List[:class:`qibo.symbols.Symbol`]: all possible observables to be measured.
    """

    return list(product([symbols.I, symbols.Z, symbols.Z, symbols.Z], repeat=nqubits))


@cache
def _get_observable(j: int, nqubits: int, backend: str) -> SymbolicHamiltonian:
    """Return the :math:`j`-th observable.

    The :math:`j`-th observable is expressed as a base-:math:`4` indexing and is given by

    .. math::
        j \\in \\{0, 1, 2, 3\\}^{\\otimes n} \\equiv \\{ I, X, Y, Z\\}^{\\otimes n}.

    Args:
        j (int): index of the measurement basis (in base-4)
        nqubits (int): number of qubits.
        backend (str): name of the backend to be used in the computation.

    Returns:
        List[:class:`qibo.hamiltonians.SymbolicHamiltonian`]: Observables represented by
        symbolic Hamiltonians.
    """
    backend_args = backend.replace("(", "").replace(")", "").split(" ")
    if len(backend_args) == 2:
        backend = construct_backend(backend_args[0], platform=backend_args[1])
    else:
        backend = construct_backend(backend_args[0])

    if j == 0:
        _check_nqubits(nqubits)
    observables = _observables(nqubits)[j]
    observable = S(1)
    for q, obs in enumerate(observables):
        if obs is not symbols.I:
            observable *= obs(q, backend=backend)
    return SymbolicHamiltonian(observable, nqubits=nqubits, backend=backend)


@cache
def _prepare_state(k: int, nqubits: int):
    """Prepares the :math:`k`-th state for an :math:`n`-qubits (`nqubits`) circuit.
    Using base-4 indexing for :math:`k`,

    .. math::
        k \\in \\{0, 1, 2, 3\\}^{\\otimes n} \\equiv \\{ 0\\rangle\\langle0|, |1\\rangle\\langle1|,
        |+\\rangle\\langle +|, |y+\\rangle\\langle y+|\\}^{\\otimes n}.

    Args:
        k (int): index of the state to be prepared.
        nqubits (int): Number of qubits.

    Returns:
        list(:class:`qibo.gates.Gate`): gates that prepare the :math:`k`-th state.
    """

    _check_nqubits(nqubits)
    gates = _gates(nqubits)[k]
    return [gate(q) for q in range(len(gates)) for gate in gates[q]]


@cache
def _measurement_basis(j: int, nqubits: int):
    """Constructs the :math:`j`-th measurement basis element for an :math:`n`-qubits (`nqubits`) circuit.
    Base-4 indexing is used for the :math:`j`-th measurement basis and is given by

    .. math::
        j \\in \\{0, 1, 2, 3\\}^{\\otimes n} \\equiv \\{ I, X, Y, Z\\}^{\\otimes n}.

    Args:
        j (int): index of the measurement basis element.
        nqubits (int): number of qubits.

    Returns:
        List[:class:`qibo.gates.Gate`]: gates forming the :math:`j`-th element
            of the Pauli measurement basis.
    """

    _check_nqubits(nqubits)
    measurements = _measurements(nqubits)[j]
    return [gates.M(q, basis=measurements[q]) for q in range(len(measurements))]


def _extract_nqubits(gate, params=None):
    """A function to extract the number of qubits the gate acts on.
    Args:
        gate (:class:`qibo.gates.abstract.Gate`): gate
        params (list, optional): A list containing the angles for the gate.
    Returns:
        nqubits (int): Number of qubits that the gate acts on.
    """

    init_args = signature(gate).parameters
    if "unitary" in init_args and params is not None:
        nqubits = int(np.log2(np.shape(params[0])[0]))
    else:
        if "q" in init_args:
            nqubits = 1
        elif "q0" in init_args and "q1" in init_args and "q2" not in init_args:
            nqubits = 2
        else:
            nqubits = None
            raise_error(
                RuntimeError,
                f"Gate {gate} is not supported for `GST`, only 1- and 2-qubit gates are supported.",
            )
    return nqubits


def _get_nqubits_and_angles(
    gate: gates.abstract.Gate | tuple[gates.abstract.Gate, list[float]],
):
    """A function to extract information about a `qibo.gates.Gate`.

    Args:
        gate (:class:`qibo.gates.abstract.Gate` or tuple): Either a gate or a tuple consisting of a gate and a list of its parameters.
            Examples of a valid input:
            - ``gate = gates.Z`` for a non-parametrized gate.
            - ``gate = (gates.RX, [np.pi/3])`` or ``gate = (gates.PRX, [np.pi/2, np.pi/3])`` for a parametrized gate.
            - ``gate = (gates.Unitary, [np.array([[1, 0], [0, 1]])])`` for an arbitrary unitary operator.
    Returns:
        gate (:class:`qibo.gates.Gate`): Gate class.
        nqubits (int): Number of qubits that the gate acts on.
        angle_names (list[str]): If gate is a parametrized gate, ``angle_names`` contains a list containing the angle names of the
            parametrized gate. Else, ``None``.
        angle_values (dict[str, float]): If gate is a parametrized gate, ``angle_values`` is a dictionary containing the angle names
            of the parametrized gate and the respective angles. Else, an empty dictionary is returned.
        params (list[float]): Stores all the parameters of the gate in a list.
    """

    if isinstance(gate, tuple):
        angles = ANGLES
        gate, params = gate
        if not (isinstance(params, (list, tuple))):
            params = [params]
    else:
        angles = None
        params = None
    init_args = signature(gate).parameters
    nqubits = _extract_nqubits(gate, params)

    if angles is not None:
        angle_names = [arg for arg in init_args if arg in angles]
        angle_values = dict(zip(angle_names, params))
    else:
        angle_names = None
        angle_values = {}

    return gate, nqubits, angle_names, angle_values, params


def _extract_gate(
    gate: gates.abstract.Gate | tuple[gates.abstract.Gate, list[float]],
    qubits: int | tuple[int, ...] | None = None,
):
    """Receives a gate class / tuple of gate class and parameters and extracts an instance of a
        `qibo.gates.Gate` that can be applied directly to the circuit while also returning the number of
        qubits that the gate acts on.

    Args:
        gate (type or tuple): A gate class or a tuple consisting of the class and a list of its parameters.
            Examples of a valid input:
            - `gate = gates.Z` for a non-parametrized gate.
            - `gate = (gates.RX, [np.pi/3])` or `gate = (gates.PRX, [np.pi/2, np.pi/3])` for a parametrized gate.
            - `gate = (gates.Unitary, [np.array([[1, 0], [0, 1]])])` for an arbitrary unitary operator.
        qubits (int or tuple, optional): Specifies the qubit index (or indices) the gate should be applied to.
            Defaults to None, in which case qubit 0 (or qubits 0 and 1 for two-qubit gates) will be used by default.

    Returns:
        gate (:class:`qibo.gates.Gate`): An instance of the gate that can be applied directly to the circuit.
        nqubits (int): The number of qubits that the gate acts on.
    """
    gate, nqubits, _angle_names, angle_values, _params = _get_nqubits_and_angles(gate)
    # Construct gate instance
    qubits = (
        range(nqubits)
        if qubits is None
        else ((qubits,) if isinstance(qubits, int) else tuple(qubits))
    )
    if "unitary" in angle_values:
        gate = gate(angle_values["unitary"], *qubits, check_unitary=True)
        if not gate.unitary:
            raise_error(ValueError, "Unitary gate received non-unitary matrix.")
    else:
        gate = gate(*qubits, **angle_values)

    return gate, nqubits


@cache
def _get_swap_pairs(nqubits, ancilla):
    """Function that returns a tuple representing which qubits to swap. There are three
        scenarios:
        - If ``ancilla = 0``, ``swap_pairs = [(0, 2)]``.
        - If ``ancilla = 1``, ``swap_pairs = [(1, 2)]``.
        - If ``ancilla = 2``, ``swap_pairs = [(0, 2), (1, 3)]``.

    Args:
        nqubits (int): The number of qubits in the GST circuit.
        ancilla (int): Controls which qubits the SWAP gates are applied to.

    Returns:
        swap_pairs (list(tuple)): A list containing the tuple of the qubits to swap.
    """

    swap_pairs = (
        [(ancilla, nqubits - 1)]
        if ancilla < 2
        else [(0, nqubits - 2), (1, nqubits - 1)]
    )
    return swap_pairs


def _gate_tomography(
    nqubits: int,
    gate: Gate = None,
    nshots: int = int(1e4),
    noise_model: NoiseModel | None = None,
    backend: Backend | None = None,
    transpiler=None,
    ancilla=None,
):
    """Runs gate tomography for a 1 or 2 qubit gate.

    It obtains a :math:`4^{n} \\times 4^{n}` matrix, where :math:`n` is the number of qubits.
    This matrix needs to be post-processed to get the Pauli-Liouville representation of the gate.
    The matrix has elements :math:`\\text{tr}(M_{j} \\, \\rho_{k})` or
    :math:`\\text{tr}(M_{j} \\, O_{l} \\rho_{k})`, depending on whether the gate
    :math:`O_{l}` is present or not.

    Args:
        nqubits (int): number of qubits of the gate.
        gate (Union[qibo.gates.Gate, list[qibo.gates.Gate]], optional):
            Gate to perform gate tomography on. Supported configurations are:
                - A single single-qubit gate.
                - A single two-qubit gate.
                - Two single-qubit gates, one applied to each qubit register.
            If ``None``, gate set tomography will be performed on an empty circuit.
            Defaults to ``None``.
        nshots (int, optional): number of shots used.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): noise model applied to simulate
            noisy computations.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.
        ancilla (int, optional): Controls whether SWAP gates are applied to replace qubits 0
            and/or 1 with fresh ancilla qubits.

            - If `ancilla = 0`, a single SWAP gate is applied on qubit0 and an ancilla qubit
            - If `ancilla = 1`, a single SWAP gate is applied on qubit1 and an ancilla qubit
            - If `ancilla = 2`, SWAP gates are applied between qubit 0 and one ancilla qubit,
              and between qubit 1 and another ancilla qubit
            - If `ancilla = None`, no SWAP gates are used. Defaults to ``None``.

    Returns:
        ndarray: Matrix approximating the input gate.
    """

    # Check if gate is 1 or 2 qubit gate.
    _check_nqubits(nqubits)

    backend = _check_backend(backend)

    if ancilla is not None and ancilla >= 3:
        raise_error(
            ValueError,
            f"Unexpected ancilla value (ancilla={ancilla}).\n"
            f"    Permitted inputs ancilla=None;\n"
            f"                     ancilla=0 to apply SWAP to qubit0 (simulating reset of qubit0);\n"
            f"                     ancilla=1 to apply SWAP to qubit1 (simulating reset of qubit1);\n"
            f"                     ancilla=2 to apply SWAP to qubit0 and qubit1 (simulating reset of qubit0 and qubit1).",
        )
    if gate is not None:
        if isinstance(gate, gates.Gate):
            gate = [gate]
        if len(gate) == 1:
            _gate = gate[0]
            if nqubits != len(_gate.qubits):
                raise_error(
                    ValueError,
                    f"Mismatched inputs: nqubits given as {nqubits}. {_gate} is a {len(_gate.qubits)}-qubit gate.",
                )
        elif len(gate) > 2:
            raise_error(
                ValueError,
                f"Mismatched inputs: number of gates in gate = {len(gate)}. Supported configurations for _gates in gate are (1) single 1-qubit gate, (2) single 2-qubit gate, (3) two 1-qubit gates applied to each qubit register.",
            )

    # GST for empty circuit or with gates
    matrix_jk = 1j * np.zeros((4**nqubits, 4**nqubits))
    for k in range(4**nqubits):
        additional_qubits = 0 if ancilla is None else (1 if ancilla in (0, 1) else 2)
        circ = Circuit(nqubits + additional_qubits, density_matrix=True)

        circ.add(_prepare_state(k, nqubits))

        if ancilla is not None:
            swap_pairs = _get_swap_pairs(circ.nqubits, ancilla)
            for q1, q2 in swap_pairs:
                circ.add(gates.SWAP(q1, q2))

        if gate is not None:
            for _gate in gate:
                circ.add(_gate)

        for j in range(4**nqubits):
            if j == 0:
                exp_val = 1.0
            else:
                new_circ = circ.copy()
                measurements = _measurement_basis(j, nqubits)
                new_circ.add(measurements)
                observable = _get_observable(j, nqubits, backend=str(backend))
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                if transpiler is not None:
                    new_circ, _ = transpiler(new_circ, backend=backend)
                result = backend.execute_circuit(new_circ, nshots=nshots)
                exp_val = result.expectation_from_samples(observable)
            matrix_jk[j, k] = exp_val
    return backend.cast(matrix_jk, dtype=matrix_jk.dtype)


def GST(
    gate_set: tuple | set | list,
    nshots: int = int(1e4),
    noise_model: NoiseModel | None = None,
    include_empty: bool = False,
    pauli_liouville: bool = False,
    gauge_matrix: ArrayLike | None = None,
    backend: Backend | None = None,
    transpiler=None,
    two_qubit_basis_op_diff_registers=False,
    ancilla=None,
):
    """Run Gate Set Tomography on the input ``gate_set``.

    Example 1:
        Given the following ``gate_set``:

        .. code-block:: python

            gate_set = [(gates.RX, [np.pi/3]), gates.Z,
                        (gates.PRX, [np.pi/2, np.pi/3]), (gates.GPI, [np.pi/7]),
                        (gates.Unitary, [np.array([[1, 0], [0, 1]])]), gates.CNOT]

        one can simply run GST to extract calibration matrices for 1- and 2-qubits
        (``g_1q`` and ``g_2q`` respectively):

            .. code-block:: python

                g_1q, g_2q, *gates_GST = GST(gate_set=gate_set,
                                              nshots=int(1e4),
                                              include_empty=True,
                                              backend=NumpyBackend(),
                                              )

    Args:
        gate_set (tuple or set or list): set of :class:`qibo.gates.Gate` and parameters to run
            GST on. For instance, ``gate_set = [(gates.RX, [np.pi/3]), gates.Z, (gates.PRX,
            [np.pi/2, np.pi/3]), (gates.GPI, [np.pi/7]), (gates.Unitary,
            [np.array([[1, 0], [0, 1]])]), gates.CNOT]``.
        nshots (int, optional): number of shots used in Gate Set Tomography per gate.
            Defaults to :math:`10^{4}`.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): noise model applied to simulate
            noisy computations.
        include_empty (bool, optional): if ``True``, additionally performs gate set tomography
            for :math:`1`- and :math:`2`-qubit empty circuits, returning the corresponding empty
            matrices in the first and second position of the ouput list.
        pauli_liouville (bool, optional): if ``True``, returns the matrices in the
            Pauli-Liouville representation. Defaults to ``False``.
        gauge_matrix (ndarray, optional): gauge matrix transformation to the Pauli-Liouville
            representation. Defaults to

            .. math::
                \\begin{pmatrix}
                    1 & 1 & 1 & 1 \\\\
                    0 & 0 & 1 & 0 \\\\
                    0 & 0 & 0 & 1 \\\\
                    1 & -1 & 0 & 0 \\\\
                \\end{pmatrix}

        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.
        two_qubit_basis_op_diff_registers (bool): If ``True``, the input `gate_set` must
            contain exactly two :math:`1`-qubit gates, one for each qubit, and gate set tomography
            will be performed simultaneously on a :math:`2`-qubit circuit. If ``False``, gate set
            tomography will be performed separately for each gate in `gate_set`. 'Defaults to
            ``False``. (Not to be confused with a single two-qubit basis operation i.e. a single
            :math:`2`-qubit gate.)
        ancilla (int, optional): Controls whether SWAP gates are applied to replace qubits 0
            and/or 1 with fresh ancilla qubits.

            - If `ancilla = 0`, a single SWAP gate is applied on qubit0 and an ancilla qubit
            - If `ancilla = 1`, a single SWAP gate is applied on qubit1 and an ancilla qubit
            - If `ancilla = 2`, SWAP gates are applied between qubit 0 and one ancilla qubit,
              and between qubit 1 and another ancilla qubit
            - If `ancilla = None`, no SWAP gates are used. Defaults to ``None``.

    Returns:
        List[ArrayLike]: Input ``gate_set`` represented by matrices estimaded via GST.
    """

    backend = _check_backend(backend)
    if backend.name == "qibolab" and transpiler is None:  # pragma: no cover
        transpiler = Passes(
            connectivity=backend.platform.topology,
            passes=[
                Preprocessing(backend.platform.topology),
                Random(backend.platform.topology),
                Sabre(backend.platform.topology),
                Unroller(NativeGates.default()),
            ],
        )

    matrices = []
    empty_matrices = []
    if include_empty or pauli_liouville:
        for nqubits in SUPPORTED_NQUBITS:
            empty_matrix = _gate_tomography(
                nqubits=nqubits,
                gate=None,
                nshots=nshots,
                noise_model=noise_model,
                backend=backend,
                transpiler=transpiler,
                ancilla=ancilla,
            )
            empty_matrices.append(empty_matrix)

    # Check that gate_set has two single-qubit gates if two_qubit_basis_op_diff_registers=True.
    # Then, if gate_set has two single-qubit gates, extract its :class:`qibo.gates.Gate` and
    # append to gate for _gate_tomography.
    if two_qubit_basis_op_diff_registers:
        if len(gate_set) != 2:
            raise_error(RuntimeError, "Requires two single-qubit gates")
        gate = []
        for idx, _gate in enumerate(gate_set):
            params = None
            if isinstance(_gate, tuple):
                _g, params = _gate
            else:
                _g = _gate
            nqubits = _extract_nqubits(_g, params)
            if nqubits != 1:
                raise_error(RuntimeError, "Requires two single-qubit gates")
            _gate, nqubits = _extract_gate(_gate, idx)
            gate.append(_gate)

        matrices.append(
            _gate_tomography(
                nqubits=2,
                gate=gate,
                nshots=nshots,
                noise_model=noise_model,
                backend=backend,
                transpiler=transpiler,
                ancilla=ancilla,
            )
        )

    else:
        for _gate in gate_set:
            if _gate is not None:
                _gate, nqubits = _extract_gate(_gate)
                gate = [_gate]
            matrices.append(
                _gate_tomography(
                    nqubits=nqubits,
                    gate=gate,
                    nshots=nshots,
                    noise_model=noise_model,
                    backend=backend,
                    transpiler=transpiler,
                    ancilla=ancilla,
                )
            )

    if pauli_liouville:
        if gauge_matrix is not None and np.linalg.det(gauge_matrix) == 0:
            raise_error(ValueError, "Matrix is not invertible")
        gauge_matrix = backend.cast(
            [[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]]
        )
        PL_matrices = []
        gauge_matrix_1q = gauge_matrix
        gauge_matrix_2q = backend.kron(gauge_matrix, gauge_matrix)
        for matrix in matrices:
            gauge_matrix = gauge_matrix_1q if matrix.shape[0] == 4 else gauge_matrix_2q
            empty = empty_matrices[0] if matrix.shape[0] == 4 else empty_matrices[1]
            PL_matrices.append(
                gauge_matrix @ backend.inv(empty) @ matrix @ backend.inv(gauge_matrix)
            )
        matrices = PL_matrices

    if include_empty:
        matrices = empty_matrices + matrices

    return matrices
