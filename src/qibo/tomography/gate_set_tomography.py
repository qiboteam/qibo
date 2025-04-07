from inspect import signature
from itertools import product
from typing import List, Union

import numpy as np
from sympy import S

from qibo import Circuit, gates, symbols
from qibo.backends import _check_backend
from qibo.config import raise_error
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.pipeline import Passes
from qibo.transpiler.placer import Random
from qibo.transpiler.router import Sabre
from qibo.transpiler.unroller import NativeGates, Unroller

SUPPORTED_NQUBITS = [1, 2]
"""Supported nqubits for GST."""


def _check_nqubits(nqubits):
    if nqubits not in SUPPORTED_NQUBITS:
        raise_error(
            ValueError,
            f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2.",
        )


# @cache
def _gates(nqubits) -> List:
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


# @cache
def _measurements(nqubits: int) -> List:
    """Measurement gates implementing all the GST measurement bases.

    Args:
        nqubits (int): Number of qubits for the circuit.
    Returns:
        List(:class:`qibo.gates.Gate`): gates implementing the possible measurement bases.
    """

    return list(product([gates.Z, gates.X, gates.Y, gates.Z], repeat=nqubits))


# @cache
def _observables(nqubits: int) -> List:
    """All the observables measured in the GST protocol.

    Args:
        nqubits (int): number of qubits for the circuit.

    Returns:
        List[:class:`qibo.symbols.Symbol`]: all possible observables to be measured.
    """

    return list(product([symbols.I, symbols.Z, symbols.Z, symbols.Z], repeat=nqubits))


# @cache
def _get_observable(j: int, nqubits: int):
    """Returns the :math:`j`-th observable. The :math:`j`-th observable is expressed as a base-4 indexing and is given by

    .. math::
        j \\in \\{0, 1, 2, 3\\}^{\\otimes n} \\equiv \\{ I, X, Y, Z\\}^{\\otimes n}.

    Args:
        j (int): index of the measurement basis (in base-4)
        nqubits (int): number of qubits.

    Returns:
        List[:class:`qibo.hamiltonians.SymbolicHamiltonian`]: observables represented by symbolic Hamiltonians.
    """

    if j == 0:
        _check_nqubits(nqubits)
    observables = _observables(nqubits)[j]
    observable = S(1)
    for q, obs in enumerate(observables):
        if obs is not symbols.I:
            observable *= obs(q)
    return SymbolicHamiltonian(observable, nqubits=nqubits)


# @cache
def _prepare_state(k, nqubits):
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


# @cache
def _measurement_basis(j, nqubits):
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


def _gate_tomography(
    nqubits: int,
    gate_list=None,
    nshots: int = int(1e4),
    noise_model=None,
    backend=None,
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
        gate_list (list of :class:`qibo.gates.Gate`, optional):
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
        ndarray: matrix approximating the input gate.
    """

    # Check if gate is 1 or 2 qubit gate.
    _check_nqubits(nqubits)

    backend = _check_backend(backend)

    if ancilla is not None:
        if ancilla >= 3:
            raise_error(
                ValueError,
                f"Unexpected ancilla value (ancilla={ancilla}).\n"
                f"    Permitted inputs ancilla=None;\n"
                f"                     ancilla=0 to apply SWAP to qubit0 (simulating reset of qubit0);\n"
                f"                     ancilla=1 to apply SWAP to qubit1 (simulating reset of qubit1);\n"
                f"                     ancilla=2 to apply SWAP to qubit0 and qubit1 (simulating reset of qubit0 and qubit1).",
            )
    if gate_list is not None:
        if len(gate_list) == 1:
            gate = gate_list[0]
            if nqubits != len(gate.qubits):
                raise_error(
                    ValueError,
                    f"Mismatched inputs: nqubits given as {nqubits}. {gate} is a {len(gate.qubits)}-qubit gate.",
                )
        elif len(gate_list) > 2:
            raise_error(
                ValueError,
                f"Mismatched inputs: number of gates in gate_list = {len(gate_list)}. Supported configurations for gates in gate_list are (1) single 1-qubit gate, (2) single 2-qubit gate, (3) two 1-qubit gates applied to each qubit register.",
            )

    # GST for empty circuit or with gates
    matrix_jk = 1j * np.zeros((4**nqubits, 4**nqubits))
    for k in range(4**nqubits):

        additional_qubits = 0 if ancilla is None else (1 if ancilla in (0, 1) else 2)
        circ = Circuit(nqubits + additional_qubits, density_matrix=True)

        circ.add(_prepare_state(k, nqubits))

        if ancilla is not None:  # pragma: no cover
            swap_pairs = [(ancilla, 2)] if ancilla < 2 else [(0, 2), (1, 3)]
            for q1, q2 in swap_pairs:
                circ.add(gates.SWAP(q1, q2))

        if gate_list is not None:
            for gate in gate_list:
                circ.add(gate)

        for j in range(4**nqubits):
            if j == 0:
                exp_val = 1.0
            else:
                new_circ = circ.copy()
                measurements = _measurement_basis(j, nqubits)
                new_circ.add(measurements)
                observable = _get_observable(j, nqubits)
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                if transpiler is not None:
                    new_circ, _ = transpiler(new_circ)
                exp_val = observable.expectation_from_samples(
                    backend.execute_circuit(new_circ, nshots=nshots).frequencies()
                )
            matrix_jk[j, k] = exp_val
    return backend.cast(matrix_jk, dtype=matrix_jk.dtype)


def _extract_gate(gate, idx=None):
    """
    Args:
        gate (:class:`qibo.gates.Gate` and parameters): For instance, gate = gates.Z
        or gate = (gates.RX, [np.pi/3]).
        idx (int): The specific qubit index to apply the gate to. Defaults to ``None``.
    Returns:
        gate (:class:`qibo.gates.Gate`): Gate that can be applied directly to the circuit.
        nqubits (int): The number of qubits that the gate acts on.
    """

    nqubits = None
    if isinstance(gate, tuple):
        angles = ["theta", "phi", "lam", "unitary"]
        gate, params = gate
        params = [params] if isinstance(params[0], np.ndarray) else params
        init_args = signature(gate).parameters
        valid_angles = [arg for arg in init_args if arg in angles]
        angle_values = dict(zip(valid_angles, params))
    else:
        angle_values = {}
        init_args = signature(gate).parameters

    if "q" in init_args:
        nqubits = 1
    elif "q0" in init_args and "q1" in init_args and "q2" not in init_args:
        nqubits = 2
    else:
        raise_error(
            RuntimeError,
            f"Gate {gate} is not supported for `GST`, only 1- and 2-qubit gates are supported.",
        )

    if idx:
        if "unitary" in angle_values:
            gate = gate(angle_values["unitary"][0], idx)
        else:
            gate = gate(idx, **angle_values)

    else:
        if "unitary" in angle_values:
            gate = gate(angle_values["unitary"][0], *range(nqubits))
        else:
            gate = gate(*range(nqubits), **angle_values)

    return gate, nqubits


def GST(
    gate_set: Union[tuple, set, list],
    nshots=int(1e4),
    noise_model=None,
    include_empty=False,
    pauli_liouville=False,
    gauge_matrix=None,
    backend=None,
    transpiler=None,
    two_qubit_basis_op_diff_registers=False,
    ancilla=None,
):
    """Run Gate Set Tomography on the input ``gate_set``.

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
        two_qubit_basis_op_diff_registers (bool, optional): If ``True``, the input `gate_set` must
            contain exactly two :math:`1`-qubit gates, one for each qubit, and gate set tomography
            will be performed simultaneously on a :math:`2`-qubit circuit. If ``False``, gate set
            tomography will be performed separately for each gate in `gate_set`. 'Defaults to
            ``False``. (Not to be confused with single two-qubit basis operation i.e. single
            :math:`2`-qubit gate.)
        ancilla (int, optional): Controls whether SWAP gates are applied to replace qubits 0
            and/or 1 with fresh ancilla qubits.
            - If `ancilla = 0`, a single SWAP gate is applied on qubit0 and an ancilla qubit
            - If `ancilla = 1`, a single SWAP gate is applied on qubit1 and an ancilla qubit
            - If `ancilla = 2`, SWAP gates are applied between qubit 0 and one ancilla qubit,
              and between qubit 1 and another ancilla qubit
            - If `ancilla = None`, no SWAP gates are used. Defaults to ``None``.

    Returns:
        List(ndarray): input ``gate_set`` represented by matrices estimaded via GST.
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
                gate_list=None,
                nshots=nshots,
                noise_model=noise_model,
                backend=backend,
                transpiler=transpiler,
                ancilla=ancilla,
            )
            empty_matrices.append(empty_matrix)

    if two_qubit_basis_op_diff_registers:
        if len(gate_set) == 2:
            gate_set_nqubits = []
            for gate in gate_set:
                if isinstance(gate, tuple):
                    angles = ["theta", "phi", "lam", "unitary"]
                    gate, params = gate
                init_args = signature(gate).parameters
                if "q" in init_args:
                    nqubits = 1
                elif "q0" in init_args and "q1" in init_args and "q2" not in init_args:
                    nqubits = 2
                gate_set_nqubits.append(nqubits)
            if 2 in gate_set_nqubits:
                raise_error(RuntimeError, f"Requires two single-qubit gates")

        gate_list = []
        for idx in range(len(gate_set)):
            gate = gate_set[idx]
            gate, _ = _extract_gate(gate, idx)

            gate_list.append(gate)

        matrices.append(
            _gate_tomography(
                nqubits=2,
                gate_list=gate_list,
                nshots=nshots,
                noise_model=noise_model,
                backend=backend,
                transpiler=transpiler,
                ancilla=ancilla,
            )
        )

    else:
        for gate in gate_set:
            if gate is not None:
                gate, nqubits = _extract_gate(gate)
                gate = [gate]
            matrices.append(
                _gate_tomography(
                    nqubits=nqubits,
                    gate_list=gate,
                    nshots=nshots,
                    noise_model=noise_model,
                    backend=backend,
                    transpiler=transpiler,
                    ancilla=None,
                )
            )

    if pauli_liouville:
        if gauge_matrix is not None:
            if np.linalg.det(gauge_matrix) == 0:
                raise_error(ValueError, "Matrix is not invertible")
        else:
            gauge_matrix = backend.cast(
                [[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]]
            )
        PL_matrices = []
        gauge_matrix_1q = gauge_matrix
        gauge_matrix_2q = backend.np.kron(gauge_matrix, gauge_matrix)
        for matrix in matrices:
            gauge_matrix = gauge_matrix_1q if matrix.shape[0] == 4 else gauge_matrix_2q
            empty = empty_matrices[0] if matrix.shape[0] == 4 else empty_matrices[1]
            PL_matrices.append(
                gauge_matrix
                @ backend.np.linalg.inv(empty)
                @ matrix
                @ backend.np.linalg.inv(gauge_matrix)
            )
        matrices = PL_matrices

    if include_empty:
        matrices = empty_matrices + matrices

    return matrices
