from functools import cache
from inspect import signature
from itertools import product
from random import Random
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


@cache
def _gates(nqubits) -> List:
    """Gates implementing all the GST state preparations.

    Args:
        nqubits (int): Number of qubits for the circuit.
    Returns:
        list(:class:`qibo.gates.abstract.Gate`): list of the gates used to prepare the possible states.
    """

    return list(
        product(
            [(gates.I,), (gates.X,), (gates.H,), (gates.H, gates.S)], repeat=nqubits
        )
    )


@cache
def _measurements(nqubits: int) -> List:
    """Measurement gates implementing all the GST measurement bases.

    Args:
        nqubits (int): Number of qubits for the circuit.
    Returns:
        list(:class:`qibo.gates.abstract.Gate`): list of the gates implementing the possible measurement bases.
    """

    return list(product([gates.Z, gates.X, gates.Y, gates.Z], repeat=nqubits))


@cache
def _observables(nqubits: int) -> List:
    """All the observables measured in the GST protocol.

    Args:
        nqubits (int): number of qubits for the circuit.

    Returns:
        List[:class:`qibo.symbols.Symbol`]: all possible observables to be measured.
    """

    return list(product([symbols.I, symbols.Z, symbols.Z, symbols.Z], repeat=nqubits))


@cache
def _get_observable(j: int, nqubits: int):
    """Returns the j-th observable. Here,

    .. math::
        j \\in \\{0, 1, 2, 3\\}^{\\otimes n} \\equiv \\{ I, X, Y, Z\\}^{\\otimes n}.

    Args:
        j (int): index of the measurement basis.
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


@cache
def _prepare_state(k, nqubits):
    """Prepares the :math:`k`-th state for an :math:`n`-qubits (`nqubits`) circuit, where

    .. math::
        k \\in \\{0, 1, 2, 3\\}^{\\otimes n} \\equiv \\{ 0\\rangle\\langle0|, |1\\rangle\\langle1|,
        |+\\rangle\\langle +|, |y+\\rangle\\langle y+|\\}^{\\otimes n}.

    Args:
        k (int): The index of the state to be prepared.
        nqubits (int): Number of qubits.

    Returns:
        list(:class:`qibo.gates.abstract.Gate`): list of the gates that prepare the k-th state.
    """

    _check_nqubits(nqubits)
    gates = _gates(nqubits)[k]
    return [gate(q) for q in range(len(gates)) for gate in gates[q]]


@cache
def _measurement_basis(j, nqubits):
    """Constructs the j-th measurement basis for an :math:`n`-qubits (`nqubits`) circuit, where

    .. math::
        j \\in \\{0, 1, 2, 3\\}^{\\otimes n} \\equiv \\{ I, X, Y, Z\\}^{\\otimes n}.

    Args:
        j (int): The index of the measurement basis.
        nqubits (int): Number of qubits.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: list of gates forming the :math:`j`-th element
        of the Pauli measurement basis.
    """

    _check_nqubits(nqubits)
    measurements = _measurements(nqubits)[j]
    return [gates.M(q, basis=measurements[q]) for q in range(len(measurements))]


def _gate_tomography(
    nqubits,
    gate=None,
    nshots=int(1e4),
    noise_model=None,
    backend=None,
    transpiler=None,
):
    """Runs gate tomography for a 1 or 2 qubit gate to obtain a :math:`4^n` by :math:`4^n` matrix (where :math:`n`
    is the number of qubits in the circuit). This matrix needs to be processed further to get the Pauli-Liouville
    representation of the `gate`. The matrix has elements :math:`\\text{tr}(M_{j} \\, \\rho_{k})` or
    :math:`\\text{tr}(M_{j} \\, O_{l} \\rho_{k})` depending on whether the gate :math:`O_l` is present.

    Args:
        nqubits (int): Number of qubits of the gate.
        gate (:class:`qibo.gates.abstract.Gate`, optional): The gate to perform gate tomography on. If ``None``, then gate tomography will be performed for an empty circuit.
        nshots (int, optional): Number of shots used.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): Noise model applied to simulate noisy computation.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Calculation engine.
    Returns:
        ndarray: matrix approximating the input gate.
    """

    # Check if gate is 1 or 2 qubit gate.
    _check_nqubits(nqubits)

    backend = _check_backend(backend)

    if gate is not None:
        if nqubits != len(gate.qubits):
            raise_error(
                ValueError,
                f"Mismatched inputs: nqubits given as {nqubits}. {gate} is a {len(gate.qubits)}-qubit gate.",
            )
        gate = gate.__class__(*gate.qubits, **gate.init_kwargs)

    # GST for empty circuit or with gates
    matrix_jk = np.zeros((4**nqubits, 4**nqubits))
    for k in range(4**nqubits):
        circ = Circuit(nqubits, density_matrix=True)
        circ.add(_prepare_state(k, nqubits))

        if gate is not None:
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
    return matrix_jk


def GST(
    gate_set: Union[tuple, set, list],
    nshots=int(1e4),
    noise_model=None,
    include_empty=False,
    Pauli_Liouville=False,
    T=None,
    backend=None,
    transpiler=None,
):
    """Runs Gate Set Tomography on the input ``gate_set``.

    Args:
        gate_set (tuple, set, list): List of :class:`qibo.gates.abstract.Gate` to run GST on.
        nshots (int, optional): Number of shots used in Gate Set Tomography.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): Noise model applied to simulate noisy computation.
        include_empty (bool, optional): If ``True``, additionally performs gate set tomography for 1- and 2-qubits empty circuits and returns the corresponding empty circuit matrices in the first and second position of the ouput list.
        Pauli_Liouville (bool, optional): If ``True``, returns the gates' matrices in the Pauli-Liouville representation.
        T (ndarray): Gauge matrix transformation to the Pauli-Liouville representation. Defaults to ``array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Calculation engine.
    Returns:
        list(ndarray): Approximated matrices of the input gate_set.
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
    if include_empty or Pauli_Liouville:
        for nqubits in SUPPORTED_NQUBITS:
            empty_matrix = _gate_tomography(
                nqubits=nqubits,
                gate=None,
                nshots=nshots,
                noise_model=noise_model,
                backend=backend,
                transpiler=transpiler,
            )
            empty_matrices.append(empty_matrix)

    for gate in gate_set:
        if gate is not None:
            init_args = signature(gate).parameters
            if "q" in init_args:
                nqubits = 1
            elif "q0" in init_args and "q1" in init_args and "q2" not in init_args:
                nqubits = 2
            else:
                raise_error(
                    RuntimeError,
                    f"Gate {gate} is not supported for `GST`, only 1- and 2-qubits gates are supported.",
                )
            gate = gate(*range(nqubits))

        matrices.append(
            _gate_tomography(
                nqubits=nqubits,
                gate=gate,
                nshots=nshots,
                noise_model=noise_model,
                backend=backend,
                transpiler=transpiler,
            )
        )

    if Pauli_Liouville:
        if T is not None:
            if np.linalg.det(T) == 0:
                raise_error(ValueError, "Matrix is not invertible")
        else:
            T = np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])
        PL_matrices = []
        T_1q = T
        T_2q = np.kron(T, T)
        for matrix in matrices:
            T_matrix = T_1q if matrix.shape[0] == 4 else T_2q
            empty = empty_matrices[0] if matrix.shape[0] == 4 else empty_matrices[1]
            PL_matrices.append(
                T_matrix @ np.linalg.inv(empty) @ matrix @ np.linalg.inv(T_matrix)
            )
        matrices = PL_matrices

    if include_empty:
        matrices = empty_matrices + matrices

    return matrices
