"""Module defining the Hamming-weight-preserving backend."""

# pylint: disable=W0212

from functools import lru_cache
from itertools import combinations
from math import comb
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import binom

from qibo import gates
from qibo.backends import Backend
from qibo.config import raise_error
from qibo.gates.abstract import Gate


@lru_cache(maxsize=None)
def _global_dims(nqubits: int, weight: int) -> int:
    """Return the dimension of the fixed-weight subspace."""
    if weight < 0 or weight > nqubits:
        return 0
    return comb(nqubits, weight)


@lru_cache(maxsize=None)
def _global_binom_table(
    nqubits: int, weight: int, backend: Optional[Backend] = None
) -> ArrayLike:
    """Return binomial coefficients C(n, k) for 0 <= n <= nqubits and 0 <= k <= weight."""
    table = backend.zeros((nqubits + 1, weight + 1), dtype=backend.int64)
    table[:, 0] = 1
    for n in range(1, nqubits + 1):
        upto = min(n, weight)
        for k in range(1, upto + 1):
            if k == n:
                table[n, k] = 1
            else:
                table[n, k] = table[n - 1, k] + table[n - 1, k - 1]
    return table


def _rank_fixed_weight_bitstrings(
    bitstrings: ArrayLike,
    nqubits: int,
    weight: int,
    backend: Optional[Backend] = None,
) -> ArrayLike:
    """Return lexicographic ranks of fixed-weight bitstrings."""
    bits = backend.cast(bitstrings, dtype=bool)
    table = _global_binom_table(nqubits, weight, backend)
    ranks = backend.zeros(bits.shape[0], dtype=backend.int64)
    remaining = backend.full(bits.shape[0], weight, dtype=backend.int64)

    for pos in range(nqubits):
        ones_mask = bits[:, pos]
        if backend.any(ones_mask):
            rem_positions = nqubits - pos - 1
            ranks[ones_mask] += table[rem_positions, remaining[ones_mask]]
            remaining[ones_mask] -= 1

    return ranks


def _lex_rank_from_one_positions(
    nqubits: int, one_positions, weight: int, table: ArrayLike
) -> int:
    """Return lexicographic rank from sorted positions of ones."""
    rank = 0
    remaining = weight
    for pos in one_positions:
        rank += table[nqubits - pos - 1, remaining]
        remaining -= 1
    return rank


def _lex_rank_from_zero_positions(
    nqubits: int, zero_positions, nzeros: int, table: ArrayLike
) -> int:
    """Return lexicographic rank from sorted positions of zeros."""
    rank = 0
    start = 0
    remaining = nzeros
    for zpos in zero_positions:
        for q in range(start, zpos):
            rank += table[nqubits - q - 1, remaining - 1]
        start = zpos + 1
        remaining -= 1
    return rank


@lru_cache(maxsize=None)
def _global_fixed_weight_bits(
    nqubits: int, weight: int, backend: Optional[Backend] = None
) -> ArrayLike:
    """Return lexicographically sorted fixed-weight bitstrings."""
    nstates = _global_dims(nqubits, weight)
    if nstates == 0:
        return backend.empty((0, nqubits), dtype=backend.int8)

    if nqubits == 0:  # pragma: no cover
        return backend.zeros((1, 0), dtype=backend.int8)

    table = _global_binom_table(nqubits, weight, backend)
    nzeros = nqubits - weight

    if weight <= nzeros:
        basis = backend.zeros((nstates, nqubits), dtype=backend.int8)
        for one_positions in combinations(range(nqubits), weight):
            rank = _lex_rank_from_one_positions(nqubits, one_positions, weight, table)
            basis[rank, one_positions] = 1
    else:
        basis = backend.ones((nstates, nqubits), dtype=backend.int8)
        for zero_positions in combinations(range(nqubits), nzeros):
            rank = _lex_rank_from_zero_positions(nqubits, zero_positions, nzeros, table)
            basis[rank, zero_positions] = 0

    return basis


@lru_cache(maxsize=None)
def _global_fixed_weight_strings(
    nqubits: int, weight: int, backend: Optional[Backend] = None
) -> Tuple[str, ...]:
    """Return lexicographically sorted fixed-weight bitstrings as strings."""
    return tuple(
        "".join("1" if int(b) else "0" for b in row)
        for row in _global_fixed_weight_bits(nqubits, weight, backend)
    )


@lru_cache(maxsize=None)
def _global_dict_indexes(
    nqubits: int, weight: int, backend: Optional[Backend] = None
) -> Dict[str, Tuple[int, int]]:
    """Return the legacy string-to-condensed-index map."""
    strings = _global_fixed_weight_strings(nqubits, weight, backend)
    return {
        string: (index, int(str(string), 2)) for index, string in enumerate(strings)
    }


@lru_cache(maxsize=None)
def _global_projected_indices(
    nqubits: int,
    weight: int,
    other_qubits: Tuple[int, ...],
    controls: Tuple[int, ...],
    qubits: Tuple[int, ...],
    qubit_values: Tuple[int, ...],
    backend: Optional[Backend] = None,
) -> ArrayLike:
    """Return condensed-state indices for fixed values on selected qubits."""
    other_weight = weight - len(controls) - sum(qubit_values)
    other_bits = _global_fixed_weight_bits(len(other_qubits), other_weight, backend)
    if other_bits.size == 0:
        return backend.empty((0,), dtype=backend.int64)

    bits = backend.zeros((len(other_bits), nqubits), dtype=backend.int8)
    if len(other_qubits) > 0:
        bits[:, list(other_qubits)] = other_bits
    if len(controls) > 0:
        bits[:, list(controls)] = 1
    bits[:, list(qubits)] = backend.cast(qubit_values, dtype=backend.int8)

    return _rank_fixed_weight_bitstrings(bits, nqubits, weight, backend)


@lru_cache(maxsize=None)
def _global_measurement_indices(
    nqubits: int,
    weight: int,
    measured_qubits: Tuple[int, ...],
    backend: Optional[Backend] = None,
) -> ArrayLike:
    """Return measured-bit decimal indices for every basis state."""
    basis = backend.cast(
        _global_fixed_weight_bits(nqubits, weight, backend), dtype=backend.int64
    )
    if len(measured_qubits) == 0:  # pragma: no cover
        return backend.zeros((basis.shape[0],), dtype=backend.int64)
    measured = basis[:, list(measured_qubits)]
    measured_powers = 1 << backend.arange(
        len(measured_qubits) - 1, -1, -1, dtype=backend.int64
    )
    return measured @ measured_powers


@lru_cache(maxsize=None)
def _global_n_qubit_flat_cache(
    nqubits: int,
    weight: int,
    active_qubits: Tuple[int, ...],
    backend: Optional[Backend] = None,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Return host-side flattened transition data for the n-qubit fallback path."""
    basis = _global_fixed_weight_bits(nqubits, weight, backend)
    k = len(active_qubits)
    active_bits = basis[:, list(active_qubits)]

    local_powers = 1 << backend.arange(k - 1, -1, -1, dtype=backend.int64)
    local_indices = active_bits @ local_powers

    patterns = backend.cast(
        (
            backend.arange(2**k, dtype=backend.int64)[:, None]
            >> backend.arange(k - 1, -1, -1, dtype=backend.int64)
        )
        & 1,
        dtype=backend.int8,
    )
    valid_mask = patterns.sum(axis=1)[None, :] == active_bits.sum(axis=1)[:, None]

    rows, cols = backend.where(valid_mask)
    rows = backend.cast(rows, dtype=backend.int64)
    cols = backend.cast(cols, dtype=backend.int64)

    target_indices = backend.empty(rows.shape[0], dtype=backend.int64)
    chunk_rows = max(1, 8_000_000 // max(1, nqubits))

    for start in range(0, rows.shape[0], chunk_rows):
        stop = min(start + chunk_rows, rows.shape[0])
        transformed = basis[rows[start:stop]].copy()
        transformed[:, list(active_qubits)] = patterns[cols[start:stop]]
        target_indices[start:stop] = _rank_fixed_weight_bitstrings(
            transformed, nqubits, weight, backend
        )

    return (
        backend.cast(local_indices, dtype=backend.int64),
        rows,
        cols,
        target_indices,
    )


def _get_basis_strings(self, nqubits: int, weight: int) -> List[str]:
    """Return cached lexicographically sorted basis strings."""
    self._ensure_basis_cache(nqubits, weight)
    key = (nqubits, weight)
    if key not in self._basis_strings_cache:  # pragma: no cover
        self._basis_strings_cache[key] = list(
            self._global_fixed_weight_strings(nqubits, weight, self)
        )
    return self._basis_strings_cache[key]


def _get_backend_index_array(
    self, cache_name: str, key, host_array: ArrayLike
) -> ArrayLike:
    """Return backend-native cached integer indices, or host indices for NumPy."""
    cache = getattr(self, cache_name, None)
    if cache is None:  # pragma: no cover
        cache = {}
        setattr(self, cache_name, cache)
    if key not in cache:
        cache[key] = self.cast(host_array, dtype=self.int64)
    return cache[key]


def _get_projected_indices(
    self,
    nqubits: int,
    weight: int,
    other_qubits: List[int],
    controls: List[int],
    qubits: List[int],
    qubit_values: List[int],
) -> ArrayLike:
    """Return cached condensed-state indices for fixed active-qubit values."""
    key = (
        nqubits,
        weight,
        tuple(other_qubits),
        tuple(controls),
        tuple(qubits),
        tuple(int(v) for v in qubit_values),
    )
    host_indices = _global_projected_indices(
        nqubits,
        weight,
        tuple(other_qubits),
        tuple(controls),
        tuple(qubits),
        tuple(int(v) for v in qubit_values),
        self,
    )
    return self._get_backend_index_array("_projected_index_cache", key, host_indices)


def _ensure_basis_cache(self, nqubits: int, weight: int):
    """Populate cached fixed-weight basis metadata used by hot paths."""

    for cache_name in (
        "_basis_bits_cache",
        "_basis_strings_cache",
        "_dict_indexes_cache",
        "_single_index_cache",
        "_double_index_cache",
        "_phase_index_cache",
        "_measurement_index_cache",
        "_projected_index_cache",
        "_n_qubit_flat_cache",
        "_backend_index_cache",
    ):
        if not hasattr(self, cache_name):
            setattr(self, cache_name, {})

    key = (nqubits, weight)
    if key not in self._basis_bits_cache:
        self._basis_bits_cache[key] = _global_fixed_weight_bits(nqubits, weight, self)


def _build_dict_indexes(self, nqubits: int, weight: int) -> Dict[str, Tuple[int, ...]]:
    """Construct the legacy string-based basis map from cached basis metadata."""
    self._ensure_basis_cache(nqubits, weight)
    key = (nqubits, weight)
    if key not in self._dict_indexes_cache:
        self._basis_strings_cache[key] = list(
            _global_fixed_weight_strings(nqubits, weight, self)
        )
        self._dict_indexes_cache[key] = _global_dict_indexes(nqubits, weight, self)
    return self._dict_indexes_cache[key]


def _get_measurement_indices(
    self, nqubits: int, weight: int, qubits: Union[List[int], Tuple[int, ...], Set[int]]
):
    """Return cached decimal indices for the measured bits of every basis state."""
    self._ensure_basis_cache(nqubits, weight)
    measured_qubits = tuple(qubits)
    key = (nqubits, weight, measured_qubits)
    host_indices = _global_measurement_indices(nqubits, weight, measured_qubits, self)
    return self._get_backend_index_array("_measurement_index_cache", key, host_indices)


def apply_gate(
    self, gate: Gate, state: ArrayLike, nqubits: int, weight: int
) -> ArrayLike:
    """Apply ``gate`` to ``state``.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): gate to apply to ``state``.
        state (ArrayLike): state to apply ``gate`` to.
        nqubits (int): total number of qubits in ``state``.
        weight (int): fixed Hamming weight of ``state``.

    Returns:
        ArrayLike: ``state`` after the action of ``gate``.
    """
    if isinstance(gate, gates.M):
        return gate.apply_hamming_weight(self, state, nqubits, weight)

    if isinstance(gate, gates.CCZ):
        # CCZ has a custom apply method because currently it is the only
        # 3-qubit gate that is also Hamming-weight-preserving
        # and this custom method is faster than the n-qubit method
        return self._apply_gate_ccz(gate, state, nqubits, weight)

    if len(gate.target_qubits) == 1:
        return self._apply_gate_single_qubit(gate, state, nqubits, weight)

    if len(gate.target_qubits) == 2:
        return self._apply_gate_two_qubit(gate, state, nqubits, weight)

    return self._apply_gate_n_qubit(gate, state, nqubits, weight)


def execute_circuit(
    self,
    circuit,
    weight: int,
    initial_state: Optional[ArrayLike] = None,
    nshots: int = 1000,
):
    """Execute ``circuit`` by applying the queue of gates to the ``initial_state``.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): Hamming-weight-preserving circuit
            to be executed.
        weight (int): fixed Hamming weight of the ``initial_state``.
        initial_state (ArrayLike, optional): initial state that ``circuit`` acts on.
            If ``None``, defaults to :math:`\\ket{0^{n-k} \\, 1^{k}}`,
            with :math:`n` being the total number of qubits in the circuit,
            and :math:`k` being the Hamming ``weight``. Defaults to ``None``.
        nshots (int, optional): total number of shots to simulate when ``circuit``
            contains measurement gates (:class:`qibo.gates.M`). Defaults to :math:`1000`.

    Returns:
        :class:`qibo.quantum_info.hamming_weight.HammingWeightResult`: Object
        containing the results of circuit execution of a Hamming-weight-preserving
        circuit.
    """
    from qibo.quantum_info.hamming_weight import (  # pylint: disable=import-outside-toplevel
        HammingWeightResult,
    )

    if circuit.density_matrix:
        raise_error(RuntimeError, "Density matrix simulation is not supported.")

    for gate in circuit.queue:
        if isinstance(gate, gates.Channel):
            raise_error(RuntimeError, "Circuit must not contain channels.")
        if not gate.hamming_weight and not isinstance(gate, gates.M):
            raise_error(
                RuntimeError,
                "Circuit contains non-Hammming weight preserving  gates.",
            )

    nqubits = circuit.nqubits
    self._ensure_basis_cache(nqubits, weight)
    self._dict_indexes = self._dict_indexes_cache.get((nqubits, weight))

    try:
        if initial_state is None:
            n_choose_k = int(binom(nqubits, weight))
            initial_state = self.zeros(n_choose_k)
            initial_state[0] = 1

        state = initial_state
        for gate in circuit.queue:
            state = self.apply_gate(gate, state, nqubits, weight)

        result = HammingWeightResult(
            state,
            weight,
            nqubits,
            measurements=circuit.measurements,
            nshots=nshots,
            platform=self.platform,
        )

        return result

    except self.oom_error:  # pragma: no cover
        raise_error(
            RuntimeError,
            f"State does not fit in {self.device} memory."
            "Please switch the execution device to a "
            "different one using ``qibo.set_device``.",
        )


def _get_lexicographical_order(
    self, nqubits: int, weight: int
) -> Dict[str, Tuple[int, ...]]:
    """Sort bistrings generated from ``self._get_cached_strings`` in lexicographical order.

    Bitstrings are sorted in lexicographical (ascending) order.
    Moreover, they are also converted to integers and both representations are cached,
    creating a map between the full statevector representation and the condensed
    Hamming-weight-preserving representation.

    Args:
        nqubits (int): total number of qubits.
        weight (int): Hamming weight of the state.

    Returns:
        dict: Dictionary with the cached bitstrings and the aforementioned map.
    """
    return self._build_dict_indexes(nqubits, weight)


def _get_single_qubit_matrix(self, gate: Gate) -> Tuple[complex, ...]:
    """Return non-zero elements of the matrix representation of
    Hamming-weight-preserving single-qubit gates.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): Hamming-weight-preserving
            single-qubit gate.

    Returns:
        tuple(complex, complex): Non-zero elements of a Hamming-weight-preserving ``gate``.
    """
    matrix = gate.matrix(backend=self)

    if gate.name in ["cz", "crz", "cu1"]:
        matrix = matrix[2:, 2:]

    return self.diag(matrix)


def _apply_gate_single_qubit(
    self, gate: Gate, state: ArrayLike, nqubits: int, weight: int
) -> ArrayLike:
    """Custom ``apply_gate`` method for Hamming-weight-preserving single-qubit gates.

    Instead of relying on matrix multiplication, this method applies
    Hamming-weight-preserving single-qubit gates by directly multiplying
    the amplitudes of interest elementwise by the necessary phase(s).

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): Hamming-weight-preserving
            single-qubit gate to be applied to ``state``
        state (ArrayLike): state to suffer the action of ``gate``.
        nqubits (int): total number of qubits in the circuit.
        weight (int): Hamming weight of ``state``.

    Returns:
        ArrayLike: ``state`` after the action of ``gate``.
    """
    qubits = list(gate.target_qubits)
    controls = list(gate.control_qubits)
    ncontrols = len(controls)
    other_qubits = sorted(set(range(nqubits)) ^ set(qubits + controls))

    key_0 = (nqubits, weight, tuple(qubits), tuple(controls), 0)
    key_1 = (nqubits, weight, tuple(qubits), tuple(controls), 1)

    if key_0 not in self._single_index_cache:
        self._single_index_cache[key_0] = self._get_projected_indices(
            nqubits, weight, other_qubits, controls, qubits, [0]
        )
    if key_1 not in self._single_index_cache:
        self._single_index_cache[key_1] = self._get_projected_indices(
            nqubits, weight, other_qubits, controls, qubits, [1]
        )

    matrix_00, matrix_11 = self._get_single_qubit_matrix(gate)

    if matrix_00 != 1.0 and nqubits - weight > 0:
        state[self._single_index_cache[key_0]] *= matrix_00

    if matrix_11 != 1.0 and weight - ncontrols > 0:
        state[self._single_index_cache[key_1]] *= matrix_11

    return state


def _update_amplitudes(
    self,
    state: ArrayLike,
    qubits: List[int],
    controls: List[int],
    other_qubits: List[int],
    weight: int,
    matrix_element: Union[complex, float],
    bitlist: List[str],
    shift: int,
) -> ArrayLike:
    """Update in-place the amplitudes changed by a two-qubit Hamming-weight-preserving gate.

    Args:
        state (ArrayLike): state that the two-qubit gate acts on.
        qubits (list): target qubits of the gate.
        controls (list): control qubits of the gate.
        other_qubits (list): remaining qubits in the circuit.
        weight (int): Hamming weight of ``state``.
        matrix_element (complex or float): non-zero element of the gate that
            multiplies the amplitudes of ``state``.
        bitlist (list): If the amplitude being updated is associated to the computational basis
            state :math:`\\ket{\\dots00\\dots}`, then ``bitlist=["0", "0"]``.
            If the amplitude is associated to :math:`\\ket{\\dots11\\dots}`,
            then ``bitlist=["1", "1"]``.
        shift (int): shift in the Hamming weight necessary to generate bitstrings.
            If the amplitude being updated is associated to the computational basis
            state :math:`\\ket{\\dots00\\dots}`, then ``shift`` is :math:`1`.
            If the amplitude is associated to :math:`\\ket{\\dots11\\dots}`,
            then ``shift`` is :math:`-1`.

    Returns:
        ArrayLike: ``state`` after the action of two-qubit Hamming-weight-preserving gate.
    """
    ncontrols = len(controls)
    nqubits = len(qubits) + ncontrols + len(other_qubits)
    phase_key = (
        nqubits,
        weight,
        tuple(qubits),
        tuple(controls),
        tuple(int(b) for b in bitlist),
        shift,
    )

    if phase_key not in self._phase_index_cache:
        self._phase_index_cache[phase_key] = self._get_projected_indices(
            nqubits,
            weight,
            other_qubits,
            controls,
            qubits,
            [int(b) for b in bitlist],
        )

    state[self._phase_index_cache[phase_key]] *= matrix_element
    return state


def _apply_gate_two_qubit(
    self, gate: Gate, state: ArrayLike, nqubits: int, weight: int
) -> ArrayLike:
    """Custom ``apply_gate`` method for Hamming-weight-preserving two-qubit gates.

    Instead of relying on matrix multiplication, this method applies
    Hamming-weight-preserving two-qubit gates by directly multiplying
    the amplitudes of interest elementwise by the necessary phase(s).

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): Hamming-weight-preserving
            two-qubit gate to be applied to ``state``
        state (ArrayLike): state to suffer the action of ``gate``.
        nqubits (int): total number of qubits in the circuit.
        weight (int): Hamming weight of ``state``.

    Returns:
        ArrayLike: ``state`` after the action of ``gate``.
    """
    qubits = list(gate.target_qubits)
    controls = list(gate.control_qubits)
    ncontrols = len(controls)
    other_qubits = sorted(set(range(nqubits)) ^ set(qubits + controls))

    key = (nqubits, weight, tuple(qubits), tuple(controls))
    if key not in self._double_index_cache:
        indexes_in = self._get_projected_indices(
            nqubits, weight, other_qubits, controls, qubits, [1, 0]
        )
        indexes_out = self._get_projected_indices(
            nqubits, weight, other_qubits, controls, qubits, [0, 1]
        )
        self._double_index_cache[key] = (indexes_in, indexes_out)

    indexes_in, indexes_out = self._double_index_cache[key]

    matrix = gate.matrix(backend=self)
    matrix_0000, matrix_1111 = matrix[0, 0], matrix[3, 3]
    matrix_0101, matrix_0110 = matrix[1, 1], matrix[1, 2]
    matrix_1001, matrix_1010 = matrix[2, 1], matrix[2, 2]

    if weight - ncontrols > 0 and weight not in [0, nqubits]:
        old_in, old_out = state[indexes_in], state[indexes_out]
        new_amplitudes_in = matrix_1010 * old_in + matrix_1001 * old_out
        new_amplitudes_out = matrix_0101 * old_out + matrix_0110 * old_in

        state[indexes_in] = new_amplitudes_in
        state[indexes_out] = new_amplitudes_out

    if (
        weight - ncontrols >= 0
        and nqubits - weight > 1
        and (matrix_0000.real != 1 or abs(matrix_0000.imag) > 0)
    ):
        # update the |...00...> amplitudes if necessary
        state = self._update_amplitudes(
            state,
            qubits,
            controls,
            other_qubits,
            weight,
            matrix_0000,
            ["0", "0"],
            shift=1,
        )

    if weight - ncontrols > 1 and (matrix_1111.real != 1 or abs(matrix_1111.imag) > 0):
        # update the |...11...> amplitudes if necessary
        state = self._update_amplitudes(
            state,
            qubits,
            controls,
            other_qubits,
            weight,
            matrix_1111,
            ["1", "1"],
            shift=-1,
        )

    return state


def _apply_gate_ccz(
    self, gate: Gate, state: ArrayLike, nqubits: int, weight: int
) -> ArrayLike:
    """Custom ``apply_gate`` method for the :class:`qibo.gates.CCZ` gate.

    Args:
        gate (:class:`qibo.gates.CCZ`): :math:`2`-controlled :math:`Z` gate
            to be applied to ``state``.
        state (ArrayLike): state to suffer the action of ``gate``.
        nqubits (int): total number of qubits in the circuit.
        weight (int): Hamming-weight of ``state``.

    Returns:
        ArrayLike: ``state`` after the action of the :class:`qibo.gates.CCZ` gate.
    """
    qubits = list(gate.qubits)
    phase_key = (nqubits, weight, tuple(qubits), "ccz")

    self._ensure_basis_cache(nqubits, weight)
    if phase_key not in self._phase_index_cache:
        basis = self._basis_bits_cache[(nqubits, weight)]
        mask = self.all(basis[:, qubits] == 1, axis=1)
        self._phase_index_cache[phase_key] = self._get_backend_index_array(
            "_phase_index_cache", phase_key, self.flatnonzero(mask)
        )

    state[self._phase_index_cache[phase_key]] *= -1
    return state


def _apply_gate_n_qubit(
    self, gate: Gate, state: ArrayLike, nqubits: int, weight: int
) -> ArrayLike:
    """Custom ``apply_gate`` method for Hamming-weight-preserving n-qubit gates.

    This method performs matrix multiplication directly in the subspace with Hamming-weight
    ``weight`` without the need to calculate the full matrix representation of the gate.

    .. note::
        The attribute ``gate.hamming_weight`` must be manually set to ``True``
        for this method to work.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): Hamming-weight-preserving
            :math:`n`-qubit gate to be applied to ``state``
        state (ArrayLike): state to suffer the action of ``gate``.
        nqubits (int): total number of qubits in the circuit.
        weight (int): Hamming weight of ``state``.

    Returns:
        ArrayLike: ``state`` after the action of ``gate``.
    """
    gate_matrix = gate.matrix(backend=self)
    qubits = list(gate.target_qubits)
    gate_qubits = len(qubits)
    if 2**gate_qubits != gate_matrix.shape[0]:
        qubits = list(gate.qubits)
        gate_qubits = len(qubits)
        controls = []
    else:
        controls = list(gate.control_qubits)

    if gate.is_controlled_by and len(controls) > 0:
        ncontrols = len(controls)
        ntargets = gate_qubits
        full_dim = 2 ** (ntargets + ncontrols)
        ctrl_mask = (1 << ncontrols) - 1
        expanded_matrix = self.identity(full_dim, dtype=gate_matrix.dtype)
        target_rows = self.arange(2**ntargets, dtype=self.int64)
        row_index = (target_rows << ncontrols) | ctrl_mask
        expanded_matrix[row_index[:, None], row_index[None, :]] = gate_matrix
        gate_matrix = expanded_matrix

    active_qubits = qubits + controls
    k = len(active_qubits)

    if len(set(active_qubits)) != k:  # pragma: no cover
        raise_error(ValueError, "Duplicate qubit indices in active_qubits")

    self._ensure_basis_cache(nqubits, weight)
    key = (tuple(active_qubits), tuple(controls), nqubits, weight)

    if key not in self._n_qubit_flat_cache:
        host_local_indices, host_rows, host_cols, host_target_indices = (
            _global_n_qubit_flat_cache(nqubits, weight, tuple(active_qubits), self)
        )
        self._n_qubit_flat_cache[key] = (
            self._get_backend_index_array(
                "_backend_index_cache", key + ("local",), host_local_indices
            ),
            self._get_backend_index_array(
                "_backend_index_cache", key + ("rows",), host_rows
            ),
            self._get_backend_index_array(
                "_backend_index_cache", key + ("cols",), host_cols
            ),
            self._get_backend_index_array(
                "_backend_index_cache", key + ("target",), host_target_indices
            ),
        )

    local_indices, rows, cols, target_indices = self._n_qubit_flat_cache[key]

    d = state.shape[0]
    new_state = self.zeros(d, dtype=state.dtype)
    state = self.ascontiguousarray(state)

    coeffs = gate_matrix[cols, local_indices[rows]]
    values = coeffs * state[rows]

    self.add_at(new_state, target_indices, values)
    return new_state


def calculate_symbolic(
    self,
    state: ArrayLike,
    nqubits: int,
    weight: int,
    decimals: int = 5,
    cutoff: float = 1e-10,
    max_terms: int = 20,
) -> str:
    """Dirac notation representation of the state in the computational basis.

    Args:
        state (ArrayLike): state to suffer the action of ``gate``.
        nqubits (int): total number of qubits in the circuit.
        weight (int): Hamming-weight of ``state``.
        decimals (int, optional): Number of decimals for the amplitudes.
            Defaults to :math:`5`.
        cutoff (float, optional): Amplitudes with absolute value smaller than the
            cutoff are ignored from the representation. Defaults to  :math:`1e-10`.
        max_terms (int, optional): Maximum number of terms to print. If the state
            contains more terms they will be ignored. Defaults to :math:`20`.

    Returns:
        str: String representing the state in the computational basis.
    """
    terms = []

    strings = self._get_basis_strings(nqubits, weight)
    for elem in self.nonzero(state)[0]:
        elem = int(elem)
        b = strings[elem]
        if self.abs(state[elem]) >= cutoff:
            x = self.round(state[elem], decimals=decimals)
            terms.append(f"{x}|{b}>")
        if len(terms) >= max_terms:
            terms.append("...")
            return terms
    return terms


def calculate_probabilities(
    self,
    state: ArrayLike,
    qubits: Union[List[int], Tuple[int, ...], Set[int]],
    weight: int,
    nqubits: int,
) -> ArrayLike:
    """Calculate the probabilities of the measured qubits from the statevector.

    Args:
        state (ArrayLike): state to suffer the action of ``gate``.
        qubits (list or tuple or set, optional): qubits that are measured.
        weight (int): Hamming-weight of ``state``.
        nqubits (int): total number of qubits in the circuit.

    Returns:
        ArrayLike: Probabilities over the input qubits.
    """
    rtype = self.real(state).dtype
    measured_indices = self._get_measurement_indices(nqubits, weight, qubits)
    weights_state = self.real(self.abs(state) ** 2)

    probs = self.zeros(2 ** len(qubits), dtype=rtype)
    self.add_at(probs, measured_indices, weights_state)
    return probs


def collapse_state(
    self,
    state: ArrayLike,
    qubits: Union[List[int], Tuple[int, ...], Set[int]],
    shot: List[int],
    weight: int,
    nqubits: int,
    normalize: bool = True,
) -> ArrayLike:
    """Collapse state vector according to measurement shot.

    Args:
        state (ArrayLike): state to suffer the action of ``gate``.
        qubits (list or tuple or set, optional): qubits that are measured.
        shot (list): Decimal value of the bitstring measured.
        weight (int): Hamming-weight of ``state``.
        nqubits (int): total number of qubits in the circuit.

    Returns:
        ArrayLike: collapsed ``state``.
    """
    measured_indices = self._get_measurement_indices(nqubits, weight, qubits)
    shot_bits = self.samples_to_binary(shot, len(qubits))[0]
    shot_index = int("".join(str(s) for s in shot_bits), 2)
    keep_mask = measured_indices == shot_index
    state = self.where(keep_mask, state, self.zeros_like(state))

    if normalize:
        state = state / self.vector_norm(state)

    return state
